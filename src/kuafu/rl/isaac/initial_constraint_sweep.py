from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


THIS_FILE = Path(__file__).resolve()
SOURCE_ROOT = THIS_FILE.parents[3]
PACKAGE_ROOT = THIS_FILE.parents[2]
for candidate in (str(SOURCE_ROOT), str(PACKAGE_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)



from kuafu.paths import resolve_artifact_path
from kuafu.artifact_contracts import active_program_identity, validate_program_metadata
from kuafu.rl.isaac.env_factory import create_env_and_app
from kuafu.rl.isaac.task_registry import task_names_match_for_metadata
from kuafu.rl.llm.client import OpenAIChatClient
from kuafu.rl.llm.prompts import INIT_SWEEP_INSTRUCTIONS, INIT_SWEEP_PROMPT, SYSTEM_PROMPT
from kuafu.rl.logging.run_dir import (
    append_jsonl,
    append_prompt_transcript,
    make_run_dirs,
    write_json,
)
from kuafu.skills.constraints.tolerance_policy import (
    assign_automatic_tolerances,
    policy_metadata,
    strip_tolerance_annotation,
    summarize_atomic_action,
)

# Human-readable init-sweep LLM log (learning runs use `prompt_transcript.md` only).
INIT_SWEEP_TRANSCRIPT_FILENAME = "init_sweep_prompt_transcript.md"
INIT_STAGE_RESPONSE_KEYS = ("Diagnosis", "Reasoning", "Weights")

def _dump_initial_sweep_yaml_configs(
    *,
    run_dir: str,
    env,
    agent_cfg: dict,
    task: str,
    device: str,
    env_seed: int,
) -> None:
    """Persist the resolved env/agent configs used by the init sweep."""
    from isaaclab.utils.io import dump_yaml

    params_dir = os.path.join(run_dir, "params")
    os.makedirs(params_dir, exist_ok=True)

    env_cfg = getattr(getattr(env, "unwrapped", env), "cfg", None)
    if env_cfg is not None and hasattr(env_cfg, "log_dir"):
        env_cfg.log_dir = run_dir

    if env_cfg is not None:
        dump_yaml(os.path.join(params_dir, "env.yaml"), env_cfg)
    dump_yaml(os.path.join(params_dir, "agent.yaml"), agent_cfg)
    dump_yaml(
        os.path.join(params_dir, "runtime.yaml"),
        {
            "task": str(task),
            "device": str(device),
            "env_seed": int(env_seed),
            "run_dir": run_dir,
            "phase": "initial_constraint_sweep",
            "env_vars": {
                key: os.environ.get(key)
                for key in sorted(os.environ)
                if key.startswith("KUAFU_") or key in {"CUDA_VISIBLE_DEVICES"}
            },
        },
    )
    print(f"[residual_rl] Initial constraint sweep: wrote resolved YAML configs to {params_dir}", flush=True)


def _require_openai_credentials(args: argparse.Namespace) -> None:
    """Fail before Isaac Sim starts if we cannot possibly call the LLM."""
    if args.openai_api_key:
        return
    if os.environ.get("OPENAI_API_KEY") or os.environ.get("AZURE_OPENAI_API_KEY"):
        return
    raise RuntimeError(
        "[residual_rl] Initial constraint sweep requires an OpenAI credential: set OPENAI_API_KEY "
        "(or AZURE_OPENAI_API_KEY), or pass --openai_api_key."
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Assign automatic constraint tolerances and acquire initial residual weights."
    )
    parser.add_argument("--task", type=str, required=True, help="Gym task name.")
    parser.add_argument(
        "--execution_robot_id",
        type=str,
        default=None,
        help="Robot env package to execute. Defaults to inference from --task; legacy tasks use fr3_finray.",
    )
    parser.add_argument(
        "--artifact_robot_id",
        type=str,
        default=None,
        help="Robot id for artifact lineage metadata. Defaults to fr3_finray for existing priors/CSVs.",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Simulation device.")
    parser.add_argument("--num_envs", type=int, default=1, help="Override env count.")
    parser.add_argument("--env_seed", type=int, default=42, help="Environment seed.")
    parser.add_argument("--reasoning_effort", default=None, help="LLM reasoning effort; defaults to KUAFU_LLM_REASONING_EFFORT.")
    parser.add_argument("--gpt_model", type=str, default="gpt-5.5", help="OpenAI model name.")
    parser.add_argument("--temperature", type=float, default=0.2, help="LLM sampling temperature.")
    parser.add_argument(
        "--openai_api_key",
        type=str,
        default=None,
        help="OpenAI API key (optional; otherwise use environment).",
    )
    parser.add_argument(
        "--initial_weights_params",
        type=str,
        default=None,
        help=(
            "Optional JSON file used to seed current stage weights. Its parameter entries are "
            "accepted for artifact compatibility but recalculated automatically in this sweep."
        ),
    )
    parser.add_argument(
        "--launch_script",
        type=str,
        default=None,
        help="Optional launcher script path copied into the run directory.",
    )
    parser.add_argument(
        "--render",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Render the env instead of headless mode.",
    )
    parser.add_argument(
        "--output_best_path_file",
        type=str,
        default=None,
        help="Optional path where the absolute path to generated best.json will be written (single line).",
    )
    return parser


def _canonical_checkpoint_path(path: str) -> str:
    return os.path.realpath(os.path.abspath(os.path.expanduser(path.strip())))


def _validate_initial_stage_overrides_metadata(
    path: str,
    data: dict[str, Any],
    *,
    expected_task: str | None = None,
    expected_prior_checkpoint: str | None = None,
) -> None:
    validate_program_metadata(data)
    metadata_task = data.get("task")
    if isinstance(metadata_task, str) and metadata_task.strip() and expected_task:
        expected_task_clean = str(expected_task).strip()
        if not task_names_match_for_metadata(metadata_task, expected_task_clean):
            raise ValueError(
                "[residual_rl] initial_weights_params task mismatch: "
                f"{path} was generated for task {metadata_task!r}, "
                f"but this sweep is using {expected_task_clean!r}. "
                "Generate a fresh best.json for the active task."
            )

    metadata_prior = data.get("prior_checkpoint")
    if (
        isinstance(metadata_prior, str)
        and metadata_prior.strip()
        and expected_prior_checkpoint
        and str(expected_prior_checkpoint).strip()
    ):
        metadata_prior_abs = _canonical_checkpoint_path(metadata_prior)
        expected_prior_abs = _canonical_checkpoint_path(str(expected_prior_checkpoint))
        if metadata_prior_abs != expected_prior_abs:
            raise ValueError(
                "[residual_rl] initial_weights_params prior mismatch: "
                f"{path} was generated for prior {metadata_prior!r}, "
                f"but this sweep is using {expected_prior_checkpoint!r}. "
                "Generate a fresh best.json for the active prior checkpoint."
            )


def _load_initial_stage_overrides(
    path: str,
    *,
    expected_task: str | None = None,
    expected_prior_checkpoint: str | None = None,
) -> dict[int, dict[str, Any]]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"[residual_rl] initial_weights_params file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(
            f"[residual_rl] initial_weights_params must be a JSON object, got {type(data).__name__}."
        )

    if "stages" in data:
        _validate_initial_stage_overrides_metadata(
            path,
            data,
            expected_task=expected_task,
            expected_prior_checkpoint=expected_prior_checkpoint,
        )
        stages_blob = data["stages"]
    else:
        stages_blob = data
    if not isinstance(stages_blob, dict):
        raise ValueError("[residual_rl] initial_weights_params does not contain a valid stages object.")

    out: dict[int, dict[str, Any]] = {}
    for sid_raw, stage_data in stages_blob.items():
        if not isinstance(stage_data, dict):
            raise ValueError(
                f"[residual_rl] stage entry {sid_raw!r} must be a dict, got {type(stage_data).__name__}."
            )
        sid = int(sid_raw)
        if "weights" not in stage_data:
            raise ValueError(f"[residual_rl] stage {sid} is missing required 'weights'.")
        if "parameters" not in stage_data:
            raise ValueError(f"[residual_rl] stage {sid} is missing required 'parameters'.")
        weights_raw = stage_data["weights"]
        params_raw = stage_data["parameters"]
        if not isinstance(weights_raw, dict):
            raise ValueError(f"[residual_rl] stage {sid} weights must be a dict.")
        if not isinstance(params_raw, dict):
            raise ValueError(f"[residual_rl] stage {sid} parameters must be a dict.")
        if not weights_raw:
            raise ValueError(f"[residual_rl] stage {sid} weights must not be empty.")
        if not params_raw:
            raise ValueError(f"[residual_rl] stage {sid} parameters must not be empty.")
        out[sid] = {
            "weights": {int(k): float(v) for k, v in weights_raw.items()},
            "parameters": {
                int(k): dict(v) for k, v in params_raw.items() if isinstance(v, dict)
            },
        }
        if len(out[sid]["parameters"]) != len(params_raw):
            raise ValueError(
                f"[residual_rl] stage {sid} parameters must map indices to dicts only."
            )
    return out


def _seed_task_info(task_info: dict[str, Any], overrides: dict[int, dict[str, Any]]) -> None:
    if "stages" not in task_info:
        raise RuntimeError("[residual_rl] Task info is missing 'stages'.")
    stages = task_info["stages"]
    if not isinstance(stages, dict):
        raise RuntimeError("[residual_rl] Task info 'stages' must be a dict.")

    for sid, stage_override in overrides.items():
        sid_i = int(sid)
        if sid_i not in stages:
            raise KeyError(f"[residual_rl] initial_weights_params references unknown stage {sid}.")
        stage_meta = stages[sid_i]
        if not isinstance(stage_meta, dict):
            raise TypeError(
                f"[residual_rl] Stage {sid} metadata must be a dict, got {type(stage_meta).__name__}."
            )
        if "weights" in stage_override:
            stage_meta["weights"] = {
                int(k): float(v) for k, v in dict(stage_override["weights"]).items()
            }
        if "parameters" in stage_override:
            stage_meta["parameters"] = {
                int(k): dict(v) for k, v in dict(stage_override["parameters"]).items()
            }


def _collect_task_info(env, *, task: str) -> dict[str, Any]:
    env_u = env.unwrapped
    if not hasattr(env_u, "program_runtime"):
        raise RuntimeError(f"[residual_rl] Env for task {task} has no program_runtime.")
    program_runtime = env_u.program_runtime
    if program_runtime is None:
        raise RuntimeError(f"[residual_rl] Env for task {task} has no program_runtime.")

    info: dict[str, Any] = {}
    if hasattr(env_u, "cfg"):
        cfg = env_u.cfg
    elif hasattr(env, "cfg"):
        cfg = env.cfg
    else:
        raise RuntimeError(f"[residual_rl] Env for task {task} has no cfg.")
    if cfg is None:
        raise RuntimeError(f"[residual_rl] Env for task {task} has no cfg.")

    if not hasattr(cfg, "task_description"):
        raise RuntimeError(f"[residual_rl] Env cfg for task {task} is missing task_description.")
    task_desc = cfg.task_description
    if not isinstance(task_desc, str) or not task_desc.strip():
        raise RuntimeError(f"[residual_rl] Env cfg for task {task} is missing task_description.")
    info["task_description"] = task_desc.strip()

    if hasattr(cfg, "detectable_objects"):
        detectable_objects = cfg.detectable_objects
    else:
        raise RuntimeError(f"[residual_rl] Env cfg for task {task} is missing detectable_objects.")
    if detectable_objects is None:
        raise RuntimeError(f"[residual_rl] Env cfg for task {task} is missing detectable_objects.")
    info["detectable_objects"] = detectable_objects

    stage_count = int(program_runtime.compiled_program.stage_count())
    if stage_count <= 0:
        raise RuntimeError(f"[residual_rl] Task {task} reported invalid stage_count={stage_count}.")

    info["stage_count"] = stage_count
    stages: dict[int, dict[str, Any]] = {}
    for stage_id in range(stage_count):
        atomic_action = program_runtime.get_atomic_action_by_id(stage_id)
        if not isinstance(atomic_action, str) or not atomic_action.strip():
            raise RuntimeError(f"[residual_rl] Stage {stage_id} is missing atomic_action.")
        cdict = program_runtime.get_constraints_by_stage(stage_id)
        if not isinstance(cdict, dict):
            raise TypeError(
                f"[residual_rl] Stage {stage_id} constraints must be a dict, got {type(cdict).__name__}."
            )
        if "Constraints" not in cdict or "Reasoning" not in cdict:
            raise KeyError(f"[residual_rl] Stage {stage_id} constraints dict is missing required keys.")
        constraints = cdict["Constraints"]
        if not isinstance(constraints, list) or not constraints:
            raise RuntimeError(f"[residual_rl] Stage {stage_id} has no constraints.")
        constraints_raw: list[dict[str, Any]] = []
        for constraint_id, constraint in enumerate(constraints):
            if not isinstance(constraint, dict):
                raise TypeError(
                    f"Stage {stage_id} constraint {constraint_id} must be a dict, "
                    f"got {type(constraint).__name__}."
                )
            if "op" not in constraint or "args" not in constraint:
                raise KeyError(
                    f"[residual_rl] Stage {stage_id} constraint {constraint_id} is missing 'op' or 'args'."
                )
            constraints_raw.append(
                {
                    "op": constraint["op"],
                    "args": dict(constraint["args"]),
                    "logic": constraint.get("logic", "and"),
                    "condition": list(constraint.get("condition", []) or []),
                }
            )

        explanations = program_runtime.explain_constraints(stage_id)
        if not isinstance(explanations, list) or not explanations:
            raise RuntimeError(f"[residual_rl] Stage {stage_id} explanations are missing.")
        explanations = [strip_tolerance_annotation(value) for value in explanations]
        weights = program_runtime.get_residual_weights(stage_id)
        if not isinstance(weights, dict) or not weights:
            raise RuntimeError(f"[residual_rl] Stage {stage_id} residual weights are missing.")
        reasoning = cdict["Reasoning"]
        if not isinstance(reasoning, str) or not reasoning.strip():
            raise RuntimeError(f"[residual_rl] Stage {stage_id} reasoning is missing.")

        stages[stage_id] = {
            "explanations": explanations,
            "weights": weights,
            "atomic_action": atomic_action.strip(),
            "constraints_raw": constraints_raw,
            "reasoning": reasoning.strip(),
        }

    info["stages"] = stages
    return info


def _strip_llm_json(raw: str) -> str:
    text = raw.strip()
    match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()
    return text


def _unique_candidate_strings(candidates: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        out.append(candidate)
    return out


def _init_stage_response_candidates(text: str) -> list[str]:
    candidates = [text]

    # Sometimes the model emits a valid object body but drops the leading
    # opening brace, e.g. `"Diagnosis": {...}, "Reasoning": ...`.
    for key in INIT_STAGE_RESPONSE_KEYS:
        for quote in ("\"", "'"):
            token = f"{quote}{key}{quote}"
            idx = text.find(token)
            if 0 <= idx < 200:
                repaired = "{" + text[idx:].strip()
                candidates.append(repaired)
                if not repaired.rstrip().endswith("}"):
                    candidates.append(repaired + "}")
                break

    # Rarer truncation: the answer starts inside the Diagnosis map, e.g.
    # `"0": "... [0,0,0.1] ...", "1": ...}, "Reasoning": ...`.
    if re.match(r"\s*[\"']\d+[\"']\s*:", text):
        repaired = "{\"Diagnosis\":{" + text.strip()
        candidates.append(repaired)
        if not repaired.rstrip().endswith("}"):
            candidates.append(repaired + "}")

    # Last resort for prose wrappers/trailing notes: try every balanced JSON
    # object in the text, but only after full-string repairs above.
    decoder = json.JSONDecoder()
    for idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            _, end = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        candidates.append(text[idx : idx + end])

    return _unique_candidate_strings(candidates)


def _load_json_like(candidate: str) -> Any:
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as json_exc:
        try:
            return ast.literal_eval(candidate)
        except (SyntaxError, ValueError) as ast_exc:
            raise ValueError(
                f"json.loads failed ({json_exc}); ast.literal_eval failed ({ast_exc})"
            ) from ast_exc


def _has_init_stage_response_keys(parsed: Any) -> bool:
    return isinstance(parsed, dict) and all(key in parsed for key in INIT_STAGE_RESPONSE_KEYS)


def _parse_init_stage_response(raw: str) -> dict[str, Any]:
    cleaned = _strip_llm_json(raw)
    first_dict: dict[str, Any] | None = None
    errors: list[str] = []
    for candidate in _init_stage_response_candidates(cleaned):
        try:
            parsed = _load_json_like(candidate)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if not isinstance(parsed, dict):
            errors.append(f"candidate parsed as {type(parsed).__name__}, not object")
            continue
        if _has_init_stage_response_keys(parsed):
            return parsed
        if first_dict is None:
            first_dict = parsed

    if first_dict is not None:
        return first_dict

    preview = cleaned[:240].replace("\n", "\\n")
    detail = "; ".join(errors[-3:])
    raise ValueError(
        f"Could not parse LLM response as a JSON object. Preview: {preview!r}. {detail}"
    )


def _format_indexed_lines(entries: list[Any]) -> str:
    return "\n".join(f"- {entry}" for entry in entries)


def _format_constraint_ops(constraints_raw: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for idx, constraint in enumerate(constraints_raw):
        op = constraint.get("op") if isinstance(constraint, dict) else None
        args = constraint.get("args") if isinstance(constraint, dict) else None
        visible_args = (
            {key: value for key, value in args.items() if key not in {"tolerance_m", "tolerance_rad"}}
            if isinstance(args, dict)
            else args
        )
        args_text = (
            json.dumps(visible_args, sort_keys=True, default=str)
            if isinstance(visible_args, dict)
            else repr(visible_args)
        )
        lines.append(f"- {idx}: op={op!r}, args={args_text}")
    return "\n".join(lines)


def _format_adjacent_stage_context(
    stages: dict[int, dict[str, Any]],
    stage_id: int | None,
    *,
    missing_message: str,
) -> str:
    if stage_id is None:
        return missing_message
    stage_meta = stages.get(int(stage_id))
    if not isinstance(stage_meta, dict):
        return f"Stage {stage_id}: unavailable."
    explanations = stage_meta.get("explanations")
    constraints_raw = stage_meta.get("constraints_raw")
    if not isinstance(explanations, list) or not explanations:
        raise RuntimeError(f"[residual_rl] Stage {stage_id} explanations must be a non-empty list.")
    if not isinstance(constraints_raw, list) or not constraints_raw:
        raise RuntimeError(f"[residual_rl] Stage {stage_id} constraints_raw must be a non-empty list.")
    return (
        f"Stage {stage_id} description: {summarize_atomic_action(stage_meta.get('atomic_action', ''))}\n"
        "Constraint ops/args:\n"
        f"{_format_constraint_ops(constraints_raw)}\n"
        "Index-ordered reward component explanations:\n"
        f"{_format_indexed_lines(explanations)}"
    )


def _require_stage_response_fields(parsed: dict[str, Any], *, stage_id: int) -> None:
    required_top_level = INIT_STAGE_RESPONSE_KEYS
    for key in required_top_level:
        if key not in parsed:
            raise KeyError(f"[residual_rl] Stage {stage_id} LLM response is missing top-level key {key!r}.")
    if parsed["Diagnosis"] is None:
        raise ValueError(f"[residual_rl] Stage {stage_id} Diagnosis must not be null.")
    if parsed["Reasoning"] is None:
        raise ValueError(f"[residual_rl] Stage {stage_id} Reasoning must not be null.")
    if not isinstance(parsed["Weights"], dict):
        raise TypeError(f"[residual_rl] Stage {stage_id} Weights must be a dict.")


def _clip_with_step_and_bounds(old_value: float, value: Any, lower: float, upper: float) -> float:
    try:
        new_value = float(value)
    except Exception as exc:
        raise ValueError(f"[residual_rl] Invalid numeric value {value!r}.") from exc

    old_value = float(old_value)
    if old_value < lower or old_value > upper:
        return min(max(new_value, lower), upper)

    step_lower = 0.5 * old_value
    step_upper = 2.0 * old_value
    lo = max(lower, step_lower)
    hi = min(upper, step_upper)
    if lo > hi:
        lo, hi = lower, upper
    return min(max(new_value, lo), hi)


def _normalize_weights(
    base_weights: dict[int, float], proposed_weights: Any, *, constrain_step: bool = True
) -> dict[int, float]:
    if not isinstance(proposed_weights, dict):
        raise TypeError("[residual_rl] Proposed Weights must be a dict.")
    proposed = proposed_weights
    normalized: dict[int, float] = {}
    expected_keys = {str(idx) for idx in base_weights} | set(base_weights)
    for key in proposed:
        if key not in expected_keys:
            raise KeyError(f"[residual_rl] Unexpected weight index {key!r} in LLM response.")
    for idx, old_value in base_weights.items():
        if str(idx) in proposed:
            candidate = proposed[str(idx)]
        elif idx in proposed:
            candidate = proposed[idx]
        else:
            raise KeyError(f"[residual_rl] Missing weight index {idx} in LLM response.")
        if constrain_step:
            normalized[int(idx)] = _clip_with_step_and_bounds(old_value, candidate, 0.1, 10.0)
        else:
            try:
                normalized[int(idx)] = min(max(float(candidate), 0.1), 10.0)
            except Exception as exc:
                raise ValueError(f"[residual_rl] Invalid weight value {candidate!r}.") from exc
    return normalized


def run_initial_constraint_sweep(args: argparse.Namespace) -> str:
    if args.openai_api_key:
        os.environ["OPENAI_API_KEY"] = str(args.openai_api_key)

    _require_openai_credentials(args)

    launch_script = str(args.launch_script).strip() if args.launch_script else None
    run_dirs = make_run_dirs(
        task=str(args.task),
        argv=sys.argv,
        launch_script_path=launch_script,
        execution_robot_id=getattr(args, "execution_robot_id", None),
        artifact_robot_id=getattr(args, "artifact_robot_id", None),
    )
    init_sweep_transcript_path = os.path.join(run_dirs.root_dir, INIT_SWEEP_TRANSCRIPT_FILENAME)
    print(
        f"[residual_rl] Initial constraint sweep: run_dir={run_dirs.root_dir} "
        f"(best.json when done; init LLM transcript -> {INIT_SWEEP_TRANSCRIPT_FILENAME})",
        flush=True,
    )

    simulation_app = None
    env = None
    try:
        print(
            "[residual_rl] Initial constraint sweep: creating Isaac env (LLM not called yet)…",
            flush=True,
        )
        simulation_app, env, agent_cfg = create_env_and_app(
            task=str(args.task),
            device=str(args.device),
            env_seed=int(args.env_seed),
            num_envs=1,
            headless=not bool(args.render),
            execution_robot_id=getattr(args, "execution_robot_id", None),
            initial_prior_checkpoint=None,
        )
        _dump_initial_sweep_yaml_configs(
            run_dir=run_dirs.root_dir,
            env=env,
            agent_cfg=agent_cfg,
            task=str(args.task),
            device=str(args.device),
            env_seed=int(args.env_seed),
        )
        print(
            "[residual_rl] Initial constraint sweep: collecting stage/constraints from program_runtime…",
            flush=True,
        )
        task_info = _collect_task_info(env, task=str(args.task))

        overrides_path = str(args.initial_weights_params).strip() if args.initial_weights_params else None
        if overrides_path:
            _seed_task_info(
                task_info,
                _load_initial_stage_overrides(overrides_path, expected_task=str(args.task)),
            )
        assign_automatic_tolerances(task_info)

        stage_count = int(task_info["stage_count"])
        stages = task_info["stages"]
        if not isinstance(stages, dict):
            raise TypeError("[residual_rl] task_info['stages'] must be a dict.")
        task_description = task_info["task_description"]
        if not isinstance(task_description, str) or not task_description.strip():
            raise RuntimeError("[residual_rl] task_description must be a non-empty string.")
        detectable_objects = task_info["detectable_objects"]

        system_prompt = SYSTEM_PROMPT + INIT_SWEEP_INSTRUCTIONS
        automatic_tolerance_policy = policy_metadata()
        stage_summaries: dict[str, dict[str, Any]] = {}

        print(
            f"[residual_rl] Initial constraint sweep: will call OpenAI once per stage "
            f"({stage_count} stage(s)); starting stage 0…",
            flush=True,
        )

        for stage_id in range(stage_count):
            if int(stage_id) not in stages:
                raise KeyError(f"[residual_rl] Missing stage metadata for stage {stage_id}.")
            stage_meta = stages[int(stage_id)]
            if not isinstance(stage_meta, dict):
                raise TypeError(f"[residual_rl] Stage metadata {stage_id} must be a dict.")
            explanations = stage_meta["explanations"]
            if not isinstance(explanations, list) or not explanations:
                raise RuntimeError(f"[residual_rl] Stage {stage_id} explanations must be a non-empty list.")
            constraints_raw = stage_meta["constraints_raw"]
            if not isinstance(constraints_raw, list) or not constraints_raw:
                raise RuntimeError(f"[residual_rl] Stage {stage_id} constraints_raw must be a non-empty list.")
            current_weights = {
                int(k): float(v) for k, v in dict(stage_meta["weights"]).items()
            }
            if not current_weights:
                raise RuntimeError(f"[residual_rl] Stage {stage_id} current_weights must not be empty.")

            stage_reasoning = stage_meta["reasoning"]
            if not isinstance(stage_reasoning, str) or not stage_reasoning.strip():
                raise RuntimeError(f"[residual_rl] Stage {stage_id} reasoning must be a non-empty string.")
            automatic_parameters = {
                int(index): dict(values)
                for index, values in dict(stage_meta["parameters"]).items()
            }
            tolerance_decisions = {
                int(index): dict(values)
                for index, values in dict(stage_meta["tolerance_decisions"]).items()
            }
            constraints_explanations = _format_indexed_lines(explanations)
            previous_stage_id = int(stage_id) - 1 if int(stage_id) > 0 else None
            next_stage_id = int(stage_id) + 1 if int(stage_id) + 1 < stage_count else None
            previous_stage_context = _format_adjacent_stage_context(
                stages,
                previous_stage_id,
                missing_message="None; this is the first subtask.",
            )
            next_stage_context = _format_adjacent_stage_context(
                stages,
                next_stage_id,
                missing_message="None; this is the final subtask.",
            )
            user_prompt = INIT_SWEEP_PROMPT.format(
                task_description=task_description,
                detectable_objects=json.dumps(detectable_objects, indent=2, sort_keys=True, default=str),
                stage_id=stage_id,
                stage_count=stage_count,
                atomic_action=summarize_atomic_action(stage_meta["atomic_action"]),
                current_constraint_ops=_format_constraint_ops(constraints_raw),
                previous_stage_context=previous_stage_context,
                next_stage_context=next_stage_context,
                constraints_explanations=constraints_explanations,
                current_weights=json.dumps(current_weights, indent=2, sort_keys=True),
            )

            print(
                f"[residual_rl] Initial constraint sweep: calling OpenAI for stage {stage_id} "
                f"(chat.completions, model={args.gpt_model})…",
                flush=True,
            )
            llm = OpenAIChatClient(
                model=str(args.gpt_model),
                temperature=float(args.temperature),
                reasoning_effort=getattr(args, "reasoning_effort", None),
                num_suggestions=1,
                system_prompt=system_prompt,
            )
            llm_resp = llm.prompt(user_prompt=user_prompt)
            if len(llm_resp.raw_outputs) != 1:
                raise RuntimeError(
                    f"[residual_rl] Expected exactly one LLM response for stage {stage_id}, got {len(llm_resp.raw_outputs)}."
                )
            raw_output = llm_resp.raw_outputs[0]
            timestamp = datetime.now().isoformat()
            append_jsonl(
                run_dirs.openai_payloads_path,
                {
                    "timestamp": timestamp,
                    "kind": "init_sweep_stage",
                    "stage_id": int(stage_id),
                    "messages": llm_resp.messages,
                    "raw_outputs": llm_resp.raw_outputs,
                    "usage": llm_resp.usage,
                    "extra": {
                        "constraints_explanations": explanations,
                        "previous_stage_context": previous_stage_context,
                        "next_stage_context": next_stage_context,
                        "automatic_parameters": automatic_parameters,
                        "tolerance_decisions": tolerance_decisions,
                        "automatic_tolerance_policy": automatic_tolerance_policy,
                    },
                },
            )
            append_prompt_transcript(
                path=init_sweep_transcript_path,
                header={
                    "timestamp": timestamp,
                    "kind": "init_sweep_stage",
                    "stage_id": int(stage_id),
                    "iter": 0,
                },
                messages=llm_resp.messages,
                raw_outputs=llm_resp.raw_outputs,
            )
            parsed = _parse_init_stage_response(raw_output)
            _require_stage_response_fields(parsed, stage_id=stage_id)
            ignored_llm_parameters = "Parameters" in parsed

            normalized_weights = _normalize_weights(current_weights, parsed["Weights"], constrain_step=False)
            normalized_parameters = automatic_parameters
            _diag = parsed["Diagnosis"]
            diagnosis = (_diag if isinstance(_diag, str) else json.dumps(_diag)).strip()
            _reas = parsed["Reasoning"]
            reasoning = (_reas if isinstance(_reas, str) else json.dumps(_reas)).strip()
            append_jsonl(
                run_dirs.iterations_path,
                {
                    "timestamp": timestamp,
                    "kind": "init_sweep_stage",
                    "stage_id": int(stage_id),
                    "iter": 0,
                    "weights": normalized_weights,
                    "parameters": normalized_parameters,
                    "tolerance_source": "automatic",
                    "tolerance_decisions": tolerance_decisions,
                    "ignored_llm_parameters": ignored_llm_parameters,
                    "diagnosis": diagnosis,
                    "reasoning": reasoning,
                },
            )
            append_jsonl(
                run_dirs.trials_path,
                {
                    "timestamp": timestamp,
                    "kind": "init_sweep_stage",
                    "stage_id": int(stage_id),
                    "iter": 0,
                    "weights": normalized_weights,
                    "parameters": normalized_parameters,
                    "tolerance_source": "automatic",
                    "tolerance_decisions": tolerance_decisions,
                    "ignored_llm_parameters": ignored_llm_parameters,
                    "raw_llm_output": raw_output,
                },
            )

            stage_summaries[str(stage_id)] = {
                "iter": 0,
                "weights": normalized_weights,
                "parameters": normalized_parameters,
                "tolerance_source": "automatic",
                "tolerance_decisions": tolerance_decisions,
                "ignored_llm_parameters": ignored_llm_parameters,
                "diagnosis": diagnosis,
                "reasoning": reasoning,
            }

        best_summary = {
            "kind": "initial_constraint_sweep",
            "task": str(args.task),
            **active_program_identity(),
            "automatic_tolerance_policy": automatic_tolerance_policy,
            "stages": stage_summaries,
        }
        write_json(run_dirs.best_path, best_summary)
        output_best_path_file = (
            str(resolve_artifact_path(args.output_best_path_file, "Initial-constraint result pointer"))
            if args.output_best_path_file
            else ""
        )
        if output_best_path_file:
            with open(output_best_path_file, "w", encoding="utf-8") as f:
                f.write(run_dirs.best_path)
            written = Path(output_best_path_file).read_text(encoding="utf-8").strip()
            if written != run_dirs.best_path.strip():
                raise RuntimeError(
                    f"[residual_rl] Output path file mismatch: expected {run_dirs.best_path!r}, got {written!r}."
                )
        print(f"[residual_rl] Initial constraint sweep complete: {run_dirs.root_dir}", flush=True)
        print(f"[residual_rl] best.json: {run_dirs.best_path}", flush=True)
        return run_dirs.root_dir
    finally:
        import traceback as _tb
        _exc_info = sys.exc_info()
        if _exc_info[1] is not None and not isinstance(_exc_info[1], SystemExit):
            print(
                f"[residual_rl] Initial constraint sweep EXCEPTION (before cleanup): "
                f"{type(_exc_info[1]).__name__}: {_exc_info[1]}",
                file=sys.stderr,
                flush=True,
            )
            _tb.print_exc(file=sys.stderr)
        if env is not None:
            env.close()
        if simulation_app is not None:
            simulation_app.close()


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    try:
        run_initial_constraint_sweep(args)
    except BaseException:
        print(
            "[residual_rl] Initial constraint sweep aborted before writing best.json "
            f"(see the external residual_rl artifact root if a run_dir was created).",
            file=sys.stderr,
            flush=True,
        )
        raise


if __name__ == "__main__":
    main()
