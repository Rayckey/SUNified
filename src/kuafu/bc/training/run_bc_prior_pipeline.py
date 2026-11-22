#!/usr/bin/env python3
"""Prepare, train, and test BC prior sweeps for a collected MPC dataset."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from kuafu.paths import DEFAULT_EXTERNAL_ARTIFACT_ROOT, PROJECT_ROOT, resolve_artifact_path
from kuafu.artifact_contracts import file_sha256, learned_program_identity, validate_program_metadata
from kuafu.skills.constraints.tolerance_policy import validate_tolerance_policy_metadata

REPO_ROOT = PROJECT_ROOT
TRAINING_DIR = Path(__file__).resolve().parent
LAUNCH_SWEEP = TRAINING_DIR / "launch_sweep.py"
EVALUATE_SWEEP = TRAINING_DIR / "evaluate_sweep.py"
PUBLISH_PRIORS = TRAINING_DIR / "publish_bc_priors.py"
INITIAL_CONSTRAINT_SWEEP = REPO_ROOT / "src" / "kuafu" / "rl" / "isaac" / "initial_constraint_sweep.py"
INITIAL_WEIGHT_SWEEP = REPO_ROOT / "src" / "kuafu" / "rl" / "isaac" / "initial_weight_sweep.py"
KUAFU_DUAL_GENERATOR = REPO_ROOT / "src" / "kuafu" / "rl" / "launchers" / "prior_conditioned" / "create_residual_hparam_launcher.py"
PREP_DIR = REPO_ROOT / "src" / "kuafu" / "data" / "collection"

if str(PREP_DIR) not in sys.path:
    sys.path.insert(0, str(PREP_DIR))

from prepare_bc_prior_sweep import (  # noqa: E402
    DEFAULT_CHUNK_SIZES,
    DEFAULT_CHUNK_STRIDES,
    DEFAULT_VARIANTS,
    BcPriorPrepConfig,
    infer_task_slug,
    prepare_bc_prior_sweep,
)


ROBOT_ALIASES = {
    "fr3_finray": "fr3_finray",
    "franka": "fr3_finray",
    "panda": "fr3_finray",
    "fr3_finray": "fr3_finray",
    "panda": "franka_panda",
    "franka_panda": "franka_panda",
    "kinova": "kinova_gen3",
    "kinova_gen3": "kinova_gen3",
}

KUAFU_SWEEP_GENERIC_TASK_SUFFIXES = {
    "right_can": "learned_can",
    "can": "learned_can",
    "drawer": "learned_drawer",
    "drawer_stack": "learned_drawer_stack",
    "estop_close": "learned_estop_close",
    "estop_open": "learned_estop_open",
    "hang_cup": "learned_hang_cup",
    "pen_cup": "learned_pen_cup",
    "stack2": "learned_stack_2",
    "stack_2": "learned_stack_2",
    "stack3": "learned_stack_3",
    "stack_3": "learned_stack_3",
}

ROBOT_TASK_PREFIXES = {
    "fr3_finray": "fr3_finray",
    "franka_panda": "panda",
    "kinova_gen3": "kinova",
}


def _float_or_none(value: str | None) -> float | None:
    text = (value or "").strip().rstrip("%")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _default_success_rate_col(robot_id: str, explicit: str | None) -> str:
    if explicit and explicit.strip():
        return explicit.strip()
    if robot_id == "fr3_finray":
        return "success_rate"
    return f"success_rate_{robot_id}"


def _absolute_child_path(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return text
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return str(path.resolve())


DEFAULT_MODEL_CLASS_ORDER = ("bc", "class", "ff3", "ff", "fff3", "fff", "f3", "f")
VARIANT_RE = re.compile(r"(?:^|_)(fff3|ff3|f3|fff|ff|f|class|bc)(?:_stride\d+)?$", re.IGNORECASE)


def _int_or_zero(value: str | None) -> int:
    try:
        return int((value or "").strip())
    except ValueError:
        return 0


def _effective_success_rate(row: dict[str, str], fallback_rate: float) -> float:
    true_success = _int_or_zero(row.get("bc_eval_true_success_count"))
    total_outcomes = _int_or_zero(row.get("bc_eval_total_outcomes"))
    if total_outcomes > 0:
        return 100.0 * float(true_success) / float(total_outcomes)
    return fallback_rate


def _variant_from_row(row: dict[str, str]) -> str:
    explicit = (row.get("variant") or "").strip().lower()
    if explicit:
        return explicit
    for key in ("name", "config", "prior_path", "best_checkpoint"):
        value = (row.get(key) or "").strip()
        if not value:
            continue
        token = Path(value).stem if key != "name" else value
        match = VARIANT_RE.search(token)
        if match:
            return match.group(1).lower()
    return "unknown"


def _model_class_order(raw_order: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    order_items = DEFAULT_MODEL_CLASS_ORDER if raw_order is None else raw_order
    normalized: list[str] = []
    for item in order_items:
        for token in str(item).split(","):
            token = token.strip().lower()
            if token and token not in normalized:
                normalized.append(token)
    return tuple(normalized)


def _candidate_sort_key(item: tuple[float, dict[str, str], str]) -> tuple[float, int, int, str]:
    rate, row, _model_class = item
    return (
        rate,
        -_int_or_zero(row.get("chunk_size")),
        -_int_or_zero(row.get("action_chunk_stride")),
        row.get("name") or "",
    )


def _normalized_path_text(value: str | None) -> str:
    text = (value or "").strip()
    if not text:
        return ""
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return str(path.resolve())


def _select_weight_prior_checkpoint(
    csv_path: Path,
    success_col: str,
    *,
    success_threshold: float,
    model_class_order: list[str] | tuple[str, ...] | None,
    selection_policy: str,
    eval_params_path: str | None = None,
) -> tuple[dict[str, str], float, str]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    expected_params_path = _normalized_path_text(eval_params_path)
    candidates: list[tuple[float, dict[str, str], str]] = []
    for row in rows:
        if expected_params_path:
            row_params_path = _normalized_path_text(row.get("bc_eval_params_path"))
            if row_params_path != expected_params_path:
                continue
        rate = _float_or_none(row.get(success_col))
        if rate is None or rate <= success_threshold:
            continue
        candidates.append((_effective_success_rate(row, float(rate)), row, _variant_from_row(row)))

    if not candidates:
        msg = f"No prior has {success_col} > {success_threshold:g} in {csv_path}"
        if expected_params_path:
            msg += f" with bc_eval_params_path={expected_params_path}"
        raise ValueError(msg)

    if selection_policy == "best":
        best_rate, best_row, _model_class = max(candidates, key=_candidate_sort_key)
        checkpoint = (best_row.get("prior_path") or "").strip() or (best_row.get("best_checkpoint") or "").strip()
        if not checkpoint:
            raise ValueError(f"Selected row has no prior_path or best_checkpoint: {best_row}")
        return best_row, float(best_rate), checkpoint

    if selection_policy != "smallest_class_over_threshold":
        raise ValueError(f"Unknown weight selection policy: {selection_policy!r}")

    ordered_classes = _model_class_order(model_class_order)
    eligible_classes = {model_class for _rate, _row, model_class in candidates}
    selected_class = next((model_class for model_class in ordered_classes if model_class in eligible_classes), None)
    if selected_class is None:
        selected_class = sorted(eligible_classes)[0]

    class_candidates = [item for item in candidates if item[2] == selected_class]
    best_rate, best_row, _model_class = max(class_candidates, key=_candidate_sort_key)
    checkpoint = (best_row.get("prior_path") or "").strip() or (best_row.get("best_checkpoint") or "").strip()
    if not checkpoint:
        raise ValueError(f"Selected row has no prior_path or best_checkpoint: {best_row}")
    return best_row, float(best_rate), checkpoint


def _kuafu_task_slug(task_slug: str | None, task_name: str) -> str:
    slug = (task_slug or infer_task_slug(task_name)).strip().lower()
    aliases = {"right_can": "can", "stack_2": "stack2", "stack_3": "stack3"}
    return aliases.get(slug, slug)


BC_TASK_VARIANTS = {"f", "ff", "fff", "f3", "ff3", "fff3"}


def _task_with_variant(task_name: str, variant: str | None) -> str:
    base = re.sub(r"-(f|ff|fff|f3|ff3|fff3)$", "", task_name.strip(), flags=re.IGNORECASE)
    variant_text = (variant or "").strip().lower()
    if variant_text in BC_TASK_VARIANTS:
        return f"{base}-{variant_text}"
    return task_name.strip()


def _robot_task_prefix(robot_id: str) -> str:
    prefix = ROBOT_TASK_PREFIXES.get(robot_id)
    if prefix is None:
        known = ", ".join(sorted(ROBOT_TASK_PREFIXES))
        raise ValueError(f"No residual-RL task prefix mapping for robot_id={robot_id!r}; known: {known}")
    return prefix


def _kuafu_sweep_root(robot_id: str) -> Path:
    return default_artifact_root(robot_id) / "residual_rl" / "parameter_search"


def _kuafu_dual_root(robot_id: str) -> Path:
    return default_artifact_root(robot_id) / "residual_rl" / "prior_conditioned_training"


def _kuafu_task_dir_name(robot_id: str, task_slug: str) -> str:
    suffix = KUAFU_SWEEP_GENERIC_TASK_SUFFIXES.get(task_slug)
    if suffix is None:
        known = ", ".join(sorted(KUAFU_SWEEP_GENERIC_TASK_SUFFIXES))
        raise ValueError(f"No residual-RL sweep directory mapping for task_slug={task_slug!r}; known: {known}")
    return f"{_robot_task_prefix(robot_id)}_{suffix}"


def _default_kuafu_sweep_dir(robot_id: str, task_slug: str) -> Path:
    return _kuafu_sweep_root(robot_id) / _kuafu_task_dir_name(robot_id, task_slug)


def _path_robot_artifact_id(path: Path) -> str | None:
    parts = path.resolve().parts
    for index, part in enumerate(parts[:-1]):
        if part == "experiment_definitions" and index + 1 < len(parts):
            return parts[index + 1]
    return None


def _validate_kuafu_script_robot(script: Path, robot_id: str) -> None:
    script_robot_id = _path_robot_artifact_id(script)
    if script_robot_id is not None and script_robot_id != robot_id:
        raise ValueError(
            f"residual-RL script {script} belongs to experiment_definitions/{script_robot_id}, "
            f"but --robot_id resolved to {robot_id}. Use a {robot_id} script or omit the override."
        )


def resolve_robot_id(value: str) -> str:
    robot_id = ROBOT_ALIASES.get(value.strip(), value.strip())
    if robot_id not in {"fr3_finray", "franka_panda", "kinova_gen3"}:
        valid = ", ".join(sorted(set(ROBOT_ALIASES.values())))
        raise ValueError(f"Unknown robot id {value!r}; expected one of: {valid}")
    return robot_id


def default_artifact_root(robot_id: str) -> Path:
    runtime_root = os.environ.get("KUAFU_PRIOR_ARTIFACT_ROOT", "").strip()
    if runtime_root:
        return Path(runtime_root).expanduser().resolve()

    task_root = os.environ.get("KUAFU_TASK_ARTIFACT_ROOT", "").strip()
    if task_root:
        return Path(task_root).expanduser().resolve() / "prior_artifacts"

    configured_root = os.environ.get("KUAFU_ARTIFACT_ROOT", "").strip()
    base = (
        Path(configured_root).expanduser().resolve()
        if configured_root
        else DEFAULT_EXTERNAL_ARTIFACT_ROOT.resolve()
    )
    return base / "standalone" / robot_id / "prior_artifacts"


def _require_external_output(path: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if resolved == REPO_ROOT or REPO_ROOT in resolved.parents:
        raise ValueError(
            f"{label} must be outside the source project ({REPO_ROOT}); got {resolved}"
        )
    return resolved


def default_run_name(dataset: Path) -> str:
    return dataset.stem


def _format_env_value_for_log(key: str, value: str) -> str:
    key_upper = key.upper()
    if "KEY" in key_upper or "TOKEN" in key_upper or "SECRET" in key_upper:
        return "***"
    return shlex.quote(value)


def print_command(label: str, cmd: list[str], env_overrides: dict[str, str] | None = None) -> None:
    env_prefix = ""
    if env_overrides:
        env_prefix = " ".join(
            f"{key}={_format_env_value_for_log(key, value)}" for key, value in sorted(env_overrides.items())
        ) + " "
    print(f"[{label}] {env_prefix}{shlex.join(cmd)}")


def run_command(
    label: str,
    cmd: list[str],
    *,
    dry_run: bool,
    env_overrides: dict[str, str] | None = None,
) -> None:
    print_command(label, cmd, env_overrides=env_overrides)
    if dry_run:
        return
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True, env=env)


def build_train_cmd(args: argparse.Namespace, csv_path: Path) -> list[str]:
    cmd = [
        args.python,
        str(LAUNCH_SWEEP),
        "--csv",
        str(csv_path),
        "--gpus",
        *[str(gpu) for gpu in args.gpus],
        "--stagger_sec",
        str(args.stagger_sec),
    ]
    cmd.extend(["--workers_per_gpu", str(args.workers_per_gpu)])
    if args.retrain_existing:
        cmd.append("--overwrite_existing")
    if args.dry_run:
        cmd.append("--dry_run")
    return cmd


def build_test_cmd(args: argparse.Namespace, csv_path: Path, robot_id: str, params_path: str | None = None) -> list[str]:
    cmd = [
        args.python,
        str(EVALUATE_SWEEP),
        "--csv",
        str(csv_path),
        "--gpus",
        *[str(gpu) for gpu in args.gpus],
        "--robot_id",
        robot_id,
        "--task",
        args.task_name,
        "--duration_sec",
        str(args.duration_sec),
        "--num_envs",
        str(args.num_envs),
        "--chunk_exec",
        str(args.chunk_exec),
        "--stagger_sec",
        str(args.stagger_sec),
    ]
    cmd.extend(["--workers_per_gpu", str(args.workers_per_gpu)])
    if args.success_rate_col:
        cmd.extend(["--success_rate_col", args.success_rate_col])
    effective_params_path = params_path or args.params_path
    if args.program_path:
        cmd.extend(["--program_path", args.program_path])
    if effective_params_path:
        cmd.extend(["--params_path", effective_params_path])
    if args.conda_env:
        cmd.extend(["--conda_env", args.conda_env])
    if args.retest_existing:
        cmd.extend(["--skip_tested", "False"])
    if args.dry_run:
        cmd.append("--dry_run")
    return cmd


def build_publish_cmd(args: argparse.Namespace, csv_path: Path, robot_id: str, task_slug: str | None) -> list[str]:
    cmd = [
        args.python,
        str(PUBLISH_PRIORS),
        "--csv",
        str(csv_path),
        "--robot_id",
        robot_id,
        "--priors_root",
        args.priors_root,
    ]
    if task_slug:
        cmd.extend(["--task_slug", task_slug])
    if args.publish_overwrite:
        cmd.append("--overwrite")
    if args.allow_missing_events:
        cmd.append("--allow_missing_events")
    return cmd


def _kuafu_env(args: argparse.Namespace, robot_id: str, *, task_name: str | None = None) -> dict[str, str]:
    env = {
        "KUAFU_TASK": task_name or args.task_name,
        "KUAFU_ARTIFACT_ROBOT_ID": robot_id,
        "KUAFU_EXECUTION_ROBOT_ID": robot_id,
    }
    if args.residual_python:
        env["PYTHON_BIN"] = args.residual_python
    if args.program_path:
        env["KUAFU_PROGRAM_PATH"] = args.program_path
    return env


def _read_exported_env_value(script: Path, key: str) -> str | None:
    if not script.exists():
        return None
    try:
        lines = script.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    prefix = f"{key}="
    for line in lines:
        line = line.strip()
        if not line.startswith("export "):
            continue
        try:
            parts = shlex.split(line, comments=True, posix=True)
        except ValueError:
            continue
        for part in parts[1:]:
            if part.startswith(prefix):
                return part.split("=", 1)[1]
    return None


def _kuafu_credential_env_from_existing_wrappers(robot_id: str) -> dict[str, str]:
    env: dict[str, str] = {}
    credential_scripts: list[Path] = []
    robot_sweep_root = _kuafu_sweep_root(robot_id)
    if robot_sweep_root.exists():
        credential_scripts.extend(sorted(robot_sweep_root.glob("*/acquire_parameters.sh")))

    for key in ("OPENAI_API_KEY", "GEMINI_API_KEY"):
        if os.environ.get(key):
            continue
        for script in credential_scripts:
            value = _read_exported_env_value(script, key)
            if value:
                env[key] = value
                break
    return env


def _read_best_path_file(output_file_path: Path, *, label: str) -> Path:
    try:
        best_json = output_file_path.read_text(encoding="utf-8").strip()
    finally:
        try:
            output_file_path.unlink()
        except OSError:
            pass
    if not best_json:
        raise FileNotFoundError(f"{label} did not write a best.json path.")
    best_path = Path(best_json).expanduser()
    if not best_path.is_absolute():
        best_path = REPO_ROOT / best_path
    if not best_path.exists():
        raise FileNotFoundError(f"{label} returned missing best.json: {best_path}")
    return best_path.resolve()


def _copy_kuafu_best_json_to_robot_task(
    best_path: Path,
    robot_id: str,
    task_slug: str,
    *,
    label: str,
    artifact_name: str = "best.json",
) -> Path:
    target_dir = _default_kuafu_sweep_dir(robot_id, task_slug)
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / artifact_name
    if best_path.resolve() != target_path.resolve():
        shutil.copy2(best_path, target_path)
    print(f"[{label}] generated best.json: {best_path}")
    print(f"[{label}] robot/task {artifact_name}: {target_path}")
    return target_path


def _existing_kuafu_best_json(
    robot_id: str,
    task_slug: str,
    *,
    artifact_name: str,
    label: str,
    expected_task: str | None = None,
    expected_prior: str | None = None,
) -> str | None:
    path = _default_kuafu_sweep_dir(robot_id, task_slug) / artifact_name
    if not path.exists():
        return None
    metadata = _kuafu_best_json_metadata(str(path))
    try:
        validate_program_metadata(metadata)
        validate_tolerance_policy_metadata(metadata)
    except ValueError:
        return None
    actual_task = metadata.get("task")
    if expected_task and actual_task and actual_task != expected_task:
        return None
    actual_prior = metadata.get("prior_checkpoint")
    if expected_prior and actual_prior and not _same_checkpoint_path(actual_prior, expected_prior):
        return None
    if expected_prior:
        from kuafu.bc.training.rl_games_prior import _resolve_prior_checkpoint_path

        try:
            prior_path, _ = _resolve_prior_checkpoint_path(_absolute_child_path(expected_prior) or expected_prior)
        except FileNotFoundError:
            return None
        if metadata.get("prior_checkpoint_sha256") != file_sha256(prior_path):
            return None
    print(f"[{label}] resume: using existing {artifact_name}: {path}")
    return str(path.resolve())


def _run_direct_kuafu_parameter_acquisition(
    args: argparse.Namespace,
    robot_id: str,
    task_slug: str,
    *,
    task_name: str | None = None,
    label: str = "residual-params",
) -> str:
    output_file = tempfile.NamedTemporaryFile(prefix="kuafu_params_", suffix=".txt", delete=False)
    output_file_path = Path(output_file.name)
    output_file.close()
    kuafu_task_name = task_name or args.task_name
    env_overrides = _kuafu_env(args, robot_id, task_name=kuafu_task_name)
    env_overrides.update(_kuafu_credential_env_from_existing_wrappers(robot_id))
    device = f"cuda:{args.gpus[0]}" if args.gpus else "cuda:0"
    cmd = [
        args.residual_python or args.python,
        str(INITIAL_CONSTRAINT_SWEEP),
        "--task",
        kuafu_task_name,
        "--execution_robot_id",
        robot_id,
        "--artifact_robot_id",
        robot_id,
        "--device",
        device,
        "--num_envs",
        str(args.num_envs),
        "--env_seed",
        "42",
        "--gpt_model",
        "gpt-5.5",
        "--temperature",
        "1.0",
        "--launch_script",
        str(Path(__file__).resolve()),
        "--output_best_path_file",
        str(output_file_path),
    ]
    run_command("residual-params", cmd, dry_run=args.dry_run, env_overrides=env_overrides)
    if args.dry_run:
        return f"<direct-residual-best-json:{robot_id}:{task_slug}>"
    best_path = _read_best_path_file(output_file_path, label="Direct residual-RL parameter acquisition")
    target_path = _copy_kuafu_best_json_to_robot_task(
        best_path, robot_id, task_slug, label=label, artifact_name="params_best.json"
    )
    _copy_kuafu_best_json_to_robot_task(best_path, robot_id, task_slug, label=label)
    return str(target_path)


def _resolve_kuafu_script(explicit: str | None, robot_id: str, task_slug: str, script_name: str) -> Path:
    if explicit:
        script = Path(explicit).expanduser().resolve()
    else:
        script = _default_kuafu_sweep_dir(robot_id, task_slug) / script_name
    _validate_kuafu_script_robot(script, robot_id)
    return script


def run_kuafu_parameter_acquisition(args: argparse.Namespace, robot_id: str, task_slug: str) -> str:
    if args.params_path:
        print(f"[residual-params] using provided params_path: {args.params_path}")
        return args.params_path
    if not args.force_residual_params:
        existing = _existing_kuafu_best_json(
            robot_id,
            task_slug,
            artifact_name="params_best.json",
            label="residual-params",
            expected_task=args.task_name,
        )
        if existing:
            return existing
    if not args.residual_params_script:
        print(f"[residual-params] using direct robot-aware acquisition for robot_id={robot_id}, task_slug={task_slug}")
        return _run_direct_kuafu_parameter_acquisition(args, robot_id, task_slug)

    script = _resolve_kuafu_script(args.residual_params_script, robot_id, task_slug, "acquire_parameters.sh")
    if not script.exists():
        raise FileNotFoundError(f"residual-RL parameter script not found: {script}")
    env_overrides = _kuafu_env(args, robot_id)
    run_command("residual-params", ["bash", str(script)], dry_run=args.dry_run, env_overrides=env_overrides)
    best_json = script.parent / "best.json"
    if not args.dry_run and not best_json.exists():
        raise FileNotFoundError(f"residual-RL parameter acquisition did not create {best_json}")
    if not args.dry_run:
        target_path = _copy_kuafu_best_json_to_robot_task(
            best_json.resolve(), robot_id, task_slug, label="residual-params", artifact_name="params_best.json"
        )
        _copy_kuafu_best_json_to_robot_task(best_json.resolve(), robot_id, task_slug, label="residual-params")
        return str(target_path)
    return str(best_json)


def _kuafu_best_json_metadata(path: str | None) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    abs_path = _absolute_child_path(path)
    if not abs_path:
        return metadata
    try:
        with open(abs_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return metadata
    if not isinstance(data, dict):
        return metadata
    for key in ("task", "prior_checkpoint", "prior_checkpoint_sha256", "program_path", "program_sha256"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            metadata[key] = value.strip()
    if "automatic_tolerance_policy" in data:
        # Keep the stamp even if malformed so validation cannot mistake an
        # invalid automatic artifact for an unstamped explicit user file.
        metadata["automatic_tolerance_policy"] = data["automatic_tolerance_policy"]
    return metadata


def _same_checkpoint_path(left: str, right: str) -> bool:
    from kuafu.bc.training.rl_games_prior import _resolve_prior_checkpoint_path

    def canonical(value: str) -> str:
        path = _absolute_child_path(value) or value
        try:
            path, _ = _resolve_prior_checkpoint_path(path)
        except FileNotFoundError:
            pass
        return os.path.realpath(path)

    return canonical(left) == canonical(right)


def _ensure_weight_params_for_task(
    args: argparse.Namespace,
    robot_id: str,
    task_slug: str,
    params_path: str | None,
    kuafu_task_name: str,
    checkpoint: str,
) -> str | None:
    if not params_path or args.dry_run:
        return params_path

    metadata = _kuafu_best_json_metadata(params_path)
    mismatch_reasons: list[str] = []
    try:
        validate_program_metadata(metadata)
        validate_tolerance_policy_metadata(metadata)
    except ValueError as error:
        mismatch_reasons.append(str(error))
    params_task = metadata.get("task")
    if params_task and params_task != kuafu_task_name:
        mismatch_reasons.append(f"task {params_task!r} != {kuafu_task_name!r}")

    params_prior = metadata.get("prior_checkpoint")
    if params_prior and not _same_checkpoint_path(params_prior, checkpoint):
        mismatch_reasons.append("prior_checkpoint does not match selected prior")

    if not mismatch_reasons:
        return params_path

    print("[residual-weights] initial params JSON is stale for the selected prior:")
    for reason in mismatch_reasons:
        print(f"[residual-weights]   {reason}")
    print(f"[residual-weights] acquiring fresh params for active task: {kuafu_task_name}")
    return _run_direct_kuafu_parameter_acquisition(
        args,
        robot_id,
        task_slug,
        task_name=kuafu_task_name,
        label="residual-params-for-weights",
    )


def _run_direct_kuafu_weight_acquisition(
    args: argparse.Namespace,
    robot_id: str,
    task_slug: str,
    kuafu_task_name: str,
    checkpoint: str,
    params_path: str | None,
) -> str:
    output_file = tempfile.NamedTemporaryFile(prefix="kuafu_weights_", suffix=".txt", delete=False)
    output_file_path = Path(output_file.name)
    output_file.close()
    env_overrides = _kuafu_env(args, robot_id, task_name=kuafu_task_name)
    env_overrides.update(_kuafu_credential_env_from_existing_wrappers(robot_id))
    device = f"cuda:{args.gpus[0]}" if args.gpus else "cuda:0"
    prior_checkpoint = _absolute_child_path(checkpoint) or checkpoint
    cmd = [
        args.residual_python or args.python,
        str(INITIAL_WEIGHT_SWEEP),
        "--task",
        kuafu_task_name,
        "--execution_robot_id",
        robot_id,
        "--artifact_robot_id",
        robot_id,
        "--device",
        device,
        "--num_envs",
        str(args.num_envs),
        "--env_seed",
        "42",
        "--prior_checkpoint",
        prior_checkpoint,
        "--prior_obs_stats_count",
        "1000000",
        "--gpt_model",
        "gpt-5.5",
        "--temperature",
        "1.0",
        "--launch_script",
        str(Path(__file__).resolve()),
        "--output_best_path_file",
        str(output_file_path),
    ]
    if params_path:
        cmd.extend(["--initial_weights_params", _absolute_child_path(params_path) or params_path])
    if args.residual_weight_max_eval_steps is not None:
        cmd.extend(["--max_eval_steps", str(args.residual_weight_max_eval_steps)])

    run_command("residual-weights", cmd, dry_run=args.dry_run, env_overrides=env_overrides)
    if args.dry_run:
        return f"<direct-residual-weight-best-json:{robot_id}:{task_slug}>"
    best_path = _read_best_path_file(output_file_path, label="Direct residual-RL weight acquisition")
    target_path = _copy_kuafu_best_json_to_robot_task(
        best_path, robot_id, task_slug, label="residual-weights", artifact_name="weights_best.json"
    )
    _copy_kuafu_best_json_to_robot_task(best_path, robot_id, task_slug, label="residual-weights")
    return str(target_path)


def run_kuafu_weight_acquisition(
    args: argparse.Namespace,
    robot_id: str,
    task_slug: str,
    csv_path: Path,
    params_path: str | None,
) -> str | None:
    success_col = _default_success_rate_col(robot_id, args.success_rate_col)
    checkpoint = ""
    kuafu_task_name = args.task_name
    if args.dry_run:
        try:
            row, _, checkpoint = _select_weight_prior_checkpoint(
                csv_path,
                success_col,
                success_threshold=float(args.residual_success_threshold),
                model_class_order=args.residual_model_class_order,
                selection_policy=args.weight_selection_policy,
                eval_params_path=params_path,
            )
            kuafu_task_name = _task_with_variant(args.task_name, row.get("variant"))
        except ValueError:
            checkpoint = ""
    else:
        row, rate, checkpoint = _select_weight_prior_checkpoint(
            csv_path,
            success_col,
            success_threshold=float(args.residual_success_threshold),
            model_class_order=args.residual_model_class_order,
            selection_policy=args.weight_selection_policy,
            eval_params_path=params_path,
        )
        kuafu_task_name = _task_with_variant(args.task_name, row.get("variant"))
        print(
            f"[residual-weights] selected {args.weight_selection_policy} prior: "
            f"{row.get('name', '<unnamed>')} ({rate:.1f}%) -> {checkpoint}"
        )
        if kuafu_task_name != args.task_name:
            print(f"[residual-weights] using task variant for BC/RL actor match: {kuafu_task_name}")

    if not checkpoint:
        if args.dry_run:
            return f"<direct-residual-weight-best-json:{robot_id}:{task_slug}>"
        raise ValueError("residual-RL weight acquisition requires a selected prior checkpoint.")

    if not args.force_residual_weights:
        existing = _existing_kuafu_best_json(
            robot_id,
            task_slug,
            artifact_name="weights_best.json",
            label="residual-weights",
            expected_task=kuafu_task_name,
            expected_prior=checkpoint,
        )
        if existing:
            return existing

    params_path = _ensure_weight_params_for_task(
        args,
        robot_id,
        task_slug,
        params_path,
        kuafu_task_name,
        checkpoint,
    )

    if not args.residual_weights_script:
        print(f"[residual-weights] using direct robot-aware acquisition for robot_id={robot_id}, task_slug={task_slug}")
        return _run_direct_kuafu_weight_acquisition(
            args,
            robot_id,
            task_slug,
            kuafu_task_name,
            checkpoint,
            params_path,
        )

    script = _resolve_kuafu_script(args.residual_weights_script, robot_id, task_slug, "acquire_weights.sh")
    if not args.dry_run and not script.exists():
        raise FileNotFoundError(f"residual-RL weight script not found: {script}")
    env_overrides = _kuafu_env(args, robot_id, task_name=kuafu_task_name)
    env_overrides["KUAFU_WEIGHT_SWEEP_PRIOR_CHECKPOINT"] = _absolute_child_path(checkpoint) or checkpoint
    if params_path:
        env_overrides["KUAFU_WEIGHT_SWEEP_INITIAL_WEIGHTS_PARAMS"] = _absolute_child_path(params_path) or params_path
    if args.residual_weight_max_eval_steps is not None:
        env_overrides["KUAFU_WEIGHT_SWEEP_MAX_EVAL_STEPS"] = str(args.residual_weight_max_eval_steps)
    run_command("residual-weights", ["bash", str(script)], dry_run=args.dry_run, env_overrides=env_overrides)
    best_json = script.parent / "best.json"
    if not args.dry_run and not best_json.exists():
        raise FileNotFoundError(f"residual-RL weight acquisition did not create {best_json}")
    if not args.dry_run:
        target_path = _copy_kuafu_best_json_to_robot_task(
            best_json.resolve(), robot_id, task_slug, label="residual-weights", artifact_name="weights_best.json"
        )
        _copy_kuafu_best_json_to_robot_task(best_json.resolve(), robot_id, task_slug, label="residual-weights")
        return str(target_path)
    return str(best_json)


def build_residual_launcher_cmd(
    args: argparse.Namespace,
    csv_path: Path,
    robot_id: str,
    task_slug: str,
    initial_weights_params: str | None,
) -> list[str]:
    cmd = [
        args.python,
        str(KUAFU_DUAL_GENERATOR),
        "--csv",
        str(csv_path),
        "--task_name",
        args.task_name,
        "--robot_id",
        robot_id,
        "--task_slug",
        task_slug,
    ]
    if args.success_rate_col:
        cmd.extend(["--success_rate_col", args.success_rate_col])
    if initial_weights_params:
        cmd.extend(["--initial_weights_params", initial_weights_params])
    if args.program_path:
        cmd.extend(["--program_path", args.program_path])
    cmd.extend(["--selection_policy", args.residual_selection_policy])
    cmd.extend(["--success_threshold", str(args.residual_success_threshold)])
    if args.residual_model_class_order:
        cmd.extend(["--model_class_order", *args.residual_model_class_order])
    if args.residual_output_dir:
        cmd.extend(["--output_dir", args.residual_output_dir])
    if args.residual_output_name:
        cmd.extend(["--output_name", args.residual_output_name])
    if args.residual_overwrite:
        cmd.append("--overwrite")
    cmd.extend(["--residual_hparam_iters", str(getattr(args, "residual_hparam_iters", 5))])
    return cmd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate BC prior YAMLs/CSV for a dataset, optionally train the sweep, "
            "then test checkpoints with play_bc_env.py via the existing sweep scripts."
        )
    )
    parser.add_argument("--dataset", required=True, help="Collected HDF5 dataset path.")
    parser.add_argument("--task_name", required=True, help="Exact Isaac Lab task name to evaluate.")
    parser.add_argument("--program_path", default=os.environ.get("KUAFU_PROGRAM_PATH"), help="Explicit learned program shared by evaluation and residual training.")
    parser.add_argument(
        "--robot_id",
        default="fr3_finray",
        help="Robot artifact/eval id: fr3_finray, franka_panda, or kinova_gen3.",
    )
    parser.add_argument(
        "--mode",
        choices=["prepare", "train", "publish", "test", "weights", "residual", "all"],
        default="prepare",
        help="Pipeline phase. all runs train, residual-RL params, test, publish, residual-RL weights, and residual launcher generation.",
    )
    parser.add_argument("--csv", default=None, help="Use an existing sweep CSV instead of generating one.")
    parser.add_argument("--run_name", default=None, help="Name used for generated config/CSV folders.")
    parser.add_argument("--artifact_root", default=None, help="Override generated artifact root.")
    parser.add_argument("--task_slug", default=None, help="Override inferred config task folder.")
    parser.add_argument("--name_prefix", default=None, help="Override generated config/CSV row prefix.")
    parser.add_argument("--obs_key", default="state", help="Observation key for generated BC configs.")
    parser.add_argument("--action_mode", default="absolute_ee", help="Action branch for generated BC configs.")
    parser.add_argument("--time_shift", type=int, default=0, help="BC temporal shift.")
    parser.add_argument("--num_epochs", type=int, default=150, help="BC training epochs.")
    parser.add_argument("--bc_num_workers", type=int, default=0, help="DataLoader worker processes for generated BC configs.")
    parser.add_argument("--bc_pin_memory", action="store_true", help="Enable pinned-memory DataLoader batches in generated BC configs.")
    parser.add_argument(
        "--chunk_sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_CHUNK_SIZES),
        help="BC action chunk sizes. Default: 10 15 30.",
    )
    parser.add_argument(
        "--chunk_strides",
        type=int,
        nargs="+",
        default=list(DEFAULT_CHUNK_STRIDES),
        help="BC action chunk strides. Default: 1 3.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(DEFAULT_VARIANTS),
        help="BC model variants. Default: f ff fff f3 ff3 fff3.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite generated YAML/CSV files.")
    parser.add_argument("--no_sample_weighting", action="store_true", help="Disable stage sample weighting.")
    parser.add_argument(
        "--sample_weighting_no_validation",
        action="store_true",
        help="Do not apply stage weights to validation/best-checkpoint selection.",
    )
    parser.add_argument("--gpus", type=int, nargs="+", default=[0], help="GPUs for train/test sweeps.")
    parser.add_argument("--workers_per_gpu", type=int, default=2, help="Concurrent train/test workers per GPU.")
    parser.add_argument("--stagger_sec", type=float, default=3.0, help="Seconds between worker launches.")
    parser.add_argument("--python", default=sys.executable, help="Python interpreter for child scripts.")
    parser.add_argument("--retrain_existing", action="store_true", help="Retrain rows with best_checkpoint set.")
    parser.add_argument(
        "--priors_root",
        default=None,
        help="Destination for published priors. Defaults below the external artifact root.",
    )
    parser.add_argument("--publish_overwrite", action="store_true", help="Overwrite existing files in Priors/.")
    parser.add_argument("--allow_missing_events", action="store_true", help="Publish even if TensorBoard events are missing.")
    parser.add_argument("--no_publish_after_train", action="store_true", help="Do not publish all trained priors after train/all.")
    parser.add_argument("--duration_sec", type=float, default=180.0, help="play_bc_env duration per checkpoint.")
    parser.add_argument("--num_envs", type=int, default=8192, help="play_bc_env parallel env count.")
    parser.add_argument("--chunk_exec", type=int, default=1, help="Actions executed from each predicted chunk.")
    parser.add_argument("--success_rate_col", default=None, help="CSV success-rate column override.")
    parser.add_argument("--params_path", default=None, help="Optional best.json params path for evaluate_sweep.py.")
    parser.add_argument("--conda_env", default="", help="Optional conda env for evaluate_sweep.py play jobs.")
    parser.add_argument("--retest_existing", action="store_true", help="Retest rows with success_rate set.")
    parser.add_argument("--no_residual_params", dest="no_residual_params", action="store_true", help="Do not acquire residual parameters before BC testing.")
    parser.add_argument("--no_kuafu_params", dest="no_residual_params", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--force_residual_params", dest="force_residual_params", action="store_true", help="Re-run residual parameter acquisition even if params_best.json exists.")
    parser.add_argument("--force_kuafu_params", dest="force_residual_params", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--force_residual_weights", dest="force_residual_weights", action="store_true", help="Re-run residual weight acquisition even if weights_best.json exists.")
    parser.add_argument("--force_kuafu_weights", dest="force_residual_weights", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--residual_params_script", dest="residual_params_script", default=None, help="Override the residual parameter-acquisition script path.")
    parser.add_argument("--kuafu_params_script", dest="residual_params_script", help=argparse.SUPPRESS)
    parser.add_argument("--residual_weights_script", dest="residual_weights_script", default=None, help="Override the residual weight-acquisition script path.")
    parser.add_argument("--kuafu_weights_script", dest="residual_weights_script", help=argparse.SUPPRESS)
    parser.add_argument("--residual_python", dest="residual_python", default="", help="Optional Python interpreter for residual acquisition scripts.")
    parser.add_argument("--kuafu_python", dest="residual_python", help=argparse.SUPPRESS)
    parser.add_argument("--no_residual_weights", dest="no_residual_weights", action="store_true", help="Do not acquire residual weights after BC testing.")
    parser.add_argument("--no_kuafu_weights", dest="no_residual_weights", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--residual_weight_max_eval_steps", dest="residual_weight_max_eval_steps", type=int, default=None, help="Optional BC-prior diagnostic rollout horizon for residual weight acquisition.")
    parser.add_argument("--kuafu_weight_max_eval_steps", dest="residual_weight_max_eval_steps", type=int, help=argparse.SUPPRESS)
    parser.add_argument(
        "--weight_selection_policy",
        choices=["best", "smallest_class_over_threshold"],
        default="best",
        help="BC prior selection policy for residual weight acquisition.",
    )
    parser.add_argument("--no_residual_launcher", action="store_true", help="Do not generate the best-prior residual RL launcher.")
    parser.add_argument("--residual_output_dir", default=None, help="Override generated residual launcher output directory.")
    parser.add_argument("--residual_output_name", default=None, help="Override generated residual launcher filename.")
    parser.add_argument(
        "--residual_hparam_iters", type=int, default=5,
        help="Model tuning iterations after initial residual training; 0 uses supplied hyperparameters only.",
    )
    parser.add_argument(
        "--residual_selection_policy",
        choices=["best", "smallest_class_over_threshold"],
        default="smallest_class_over_threshold",
        help="Prior selection policy for generated residual launchers.",
    )
    parser.add_argument("--residual_success_threshold", type=float, default=20.0, help="Success-rate threshold for residual prior selection.")
    parser.add_argument(
        "--residual_model_class_order",
        nargs="+",
        default=list(DEFAULT_MODEL_CLASS_ORDER),
        help="Preferred model class order for residual prior selection.",
    )
    parser.add_argument("--residuals_root", default="Residuals", help=argparse.SUPPRESS)
    parser.add_argument("--residual_overwrite", action="store_true", help="Overwrite an existing generated residual launcher.")
    parser.add_argument("--dry_run", action="store_true", help="Print training/testing commands without launching.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.program_path:
        if not args.dry_run:
            args.program_path = learned_program_identity(args.program_path)["program_path"]
        os.environ["KUAFU_PROGRAM_PATH"] = args.program_path
    try:
        robot_id = resolve_robot_id(args.robot_id)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
    if args.workers_per_gpu < 1:
        print("[ERROR] --workers_per_gpu must be >= 1", file=sys.stderr)
        sys.exit(1)

    artifact_root = (
        Path(args.artifact_root).expanduser().resolve()
        if args.artifact_root
        else default_artifact_root(robot_id)
    )
    try:
        artifact_root = _require_external_output(artifact_root, label="--artifact_root")
        priors_root = _require_external_output(
            Path(args.priors_root).expanduser()
            if args.priors_root
            else artifact_root / "published_priors",
            label="--priors_root",
        )
        if args.residual_output_dir:
            _require_external_output(
                Path(args.residual_output_dir).expanduser(),
                label="--residual_output_dir",
            )
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
    args.artifact_root = str(artifact_root)
    args.priors_root = str(priors_root)
    os.environ["KUAFU_PRIOR_ARTIFACT_ROOT"] = str(artifact_root)

    dataset = resolve_artifact_path(args.dataset, "BC training trajectory")
    if not dataset.exists():
        print(f"[ERROR] Dataset not found: {dataset}", file=sys.stderr)
        sys.exit(1)

    result = None
    if args.csv:
        csv_path = resolve_artifact_path(args.csv, "BC sweep CSV")
        if not csv_path.exists():
            print(f"[ERROR] CSV not found: {csv_path}", file=sys.stderr)
            sys.exit(1)
        if args.mode in {"train", "test", "publish", "all"}:
            try:
                _require_external_output(csv_path, label="--csv")
            except ValueError as exc:
                print(f"[ERROR] {exc}", file=sys.stderr)
                sys.exit(1)
    else:
        run_name = args.run_name or default_run_name(dataset)

        result = prepare_bc_prior_sweep(
            BcPriorPrepConfig(
                dataset_path=dataset,
                task_name=args.task_name,
                robot_name=robot_id,
                run_name=run_name,
                artifact_root=artifact_root,
                task_slug=args.task_slug,
                name_prefix=args.name_prefix,
                chunk_sizes=tuple(args.chunk_sizes),
                action_chunk_strides=tuple(args.chunk_strides),
                variants=tuple(args.variants),
                obs_key=args.obs_key,
                action_mode=args.action_mode,
                time_shift=args.time_shift,
                num_workers=args.bc_num_workers,
                pin_memory=args.bc_pin_memory,
                num_epochs=args.num_epochs,
                sample_weighting_enabled=not args.no_sample_weighting,
                sample_weighting_validation=not args.sample_weighting_no_validation,
                overwrite=args.overwrite,
            )
        )
        csv_path = result.csv_path

    publish_task_slug = args.task_slug or (result.task_slug if result is not None else None)
    kuafu_task_slug = _kuafu_task_slug(publish_task_slug, args.task_name)
    params_path = args.params_path or None
    weights_path: str | None = None

    print("=== BC Prior Pipeline ===")
    print(f"  Robot:       {robot_id}")
    print(f"  Dataset:     {dataset}")
    print(f"  Task:        {args.task_name}")
    if result is not None:
        print(f"  Config dir:  {result.config_dir}")
        print(f"  YAML files:  {len(result.generated)}")
    print(f"  Sweep CSV:   {csv_path}")
    print(f"  Mode:        {args.mode}")

    published_after_train = False
    if args.mode in {"train", "all"}:
        train_cmd = build_train_cmd(args, csv_path)
        run_command("train", train_cmd, dry_run=args.dry_run)

        if not args.no_publish_after_train:
            publish_cmd = build_publish_cmd(args, csv_path, robot_id, publish_task_slug)
            run_command("publish", publish_cmd, dry_run=args.dry_run)
            published_after_train = True

    if args.mode == "publish":
        publish_cmd = build_publish_cmd(args, csv_path, robot_id, publish_task_slug)
        run_command("publish", publish_cmd, dry_run=args.dry_run)

    if args.mode in {"test", "weights", "all"} and not args.no_residual_params and not params_path:
        params_path = run_kuafu_parameter_acquisition(args, robot_id, kuafu_task_slug)

    if args.mode in {"test", "all"}:
        test_cmd = build_test_cmd(args, csv_path, robot_id, params_path=params_path)
        if args.dry_run and args.mode == "all":
            print("[test] Dry-run note: training has not filled best_checkpoint yet, so this command is for after training completes.")
            print_command("test", test_cmd)
        else:
            run_command("test", test_cmd, dry_run=args.dry_run)

    if args.mode == "all" and not args.no_publish_after_train and not published_after_train:
        publish_cmd = build_publish_cmd(args, csv_path, robot_id, publish_task_slug)
        run_command("publish", publish_cmd, dry_run=args.dry_run)

    if args.mode in {"weights", "all"} and not args.no_residual_weights:
        weights_path = run_kuafu_weight_acquisition(args, robot_id, kuafu_task_slug, csv_path, params_path)

    if args.mode in {"residual", "all"} and not args.no_residual_launcher:
        residual_weights_path = weights_path or params_path
        residual_cmd = build_residual_launcher_cmd(
            args,
            csv_path,
            robot_id,
            kuafu_task_slug,
            residual_weights_path,
        )
        run_command("residual-launcher", residual_cmd, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
