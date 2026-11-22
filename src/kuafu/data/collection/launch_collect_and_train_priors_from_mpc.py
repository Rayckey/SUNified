#!/usr/bin/env python3
"""Collect MPC demos, compile BC-prior configs, train, and evaluate the prior sweep."""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
COLLECT_LAUNCHER = Path(__file__).resolve().parent / "launch_collect_from_mpc.py"
PRIOR_PIPELINE = REPO_ROOT / "src" / "kuafu" / "bc" / "training" / "run_bc_prior_pipeline.py"

ROBOT_IDS = {
    "fr3_finray": "fr3_finray",
    "kinova": "kinova_gen3",
    "panda": "franka_panda",
}


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Launch MPC collection, compact worker HDF5s, generate BC-prior YAMLs/CSV, "
            "train, and evaluate the generated prior sweep."
        )
    )
    parser.add_argument(
        "--task_name",
        nargs="+",
        required=True,
        help="One or more exact Isaac Lab task names to collect and train.",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        nargs="+",
        required=True,
        help="GPU IDs used for collection and prior training.",
    )
    parser.add_argument(
        "--workers_per_gpu",
        type=int,
        default=2,
        help="Worker processes per GPU for MPC collection, prior training, and BC eval.",
    )
    parser.add_argument(
        "--num_demos_per_worker",
        type=int,
        default=100,
        help="How many demos each MPC collection worker should collect.",
    )
    parser.add_argument(
        "--robot_name",
        choices=sorted(ROBOT_IDS),
        default=None,
        help="Robot label for compaction/artifacts. Defaults to inferring from task name.",
    )
    parser.add_argument(
        "--run_name",
        default=None,
        help="Run name for outputs. Defaults to the current timestamp for each task.",
    )
    parser.add_argument(
        "--output_root",
        default=None,
        help="Exact collection output directory. Only valid with one task.",
    )
    parser.add_argument(
        "--compact_output_dir",
        default=None,
        help="External directory for compacted HDF5 datasets. Defaults below KUAFU_ARTIFACT_ROOT.",
    )
    parser.add_argument(
        "--compact_skip_bad_files",
        action="store_true",
        help="Forward --skip-bad-files to HDF5 compaction.",
    )
    parser.add_argument(
        "--compact_no_strict_length",
        action="store_true",
        help="Forward --no-strict-length to HDF5 compaction.",
    )
    parser.add_argument(
        "--collect_stagger_sec",
        type=float,
        default=5.0,
        help="Seconds between collection worker launches.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter for collection and training subprocesses.",
    )
    parser.add_argument(
        "--bc_chunk_sizes",
        type=int,
        nargs="+",
        default=[10, 15, 30],
        help="BC action chunk sizes for generated prior configs.",
    )
    parser.add_argument(
        "--bc_chunk_strides",
        type=int,
        nargs="+",
        default=[1, 3],
        help="BC action chunk strides for generated prior configs.",
    )
    parser.add_argument(
        "--bc_variants",
        nargs="+",
        default=["f", "ff", "fff", "f3", "ff3", "fff3"],
        help="BC model variants to generate and train. Defaults to: f ff fff f3 ff3 fff3.",
    )
    parser.add_argument(
        "--bc_obs_key",
        default="state",
        help="Observation key for generated BC configs.",
    )
    parser.add_argument(
        "--bc_action_mode",
        choices=["relative_ee", "relative_joint", "absolute_ee", "absolute_joint"],
        default=None,
        help="Action branch for generated BC configs. Defaults to forwarded --control_output_mode or absolute_ee.",
    )
    parser.add_argument(
        "--bc_time_shift",
        type=int,
        default=0,
        help="BC temporal shift.",
    )
    parser.add_argument(
        "--bc_num_epochs",
        type=int,
        default=150,
        help="BC training epochs.",
    )
    parser.add_argument(
        "--bc_num_workers",
        type=int,
        default=0,
        help="DataLoader worker processes for generated BC configs.",
    )
    parser.add_argument(
        "--bc_pin_memory",
        action="store_true",
        help="Enable pinned-memory DataLoader batches in generated BC configs.",
    )
    parser.add_argument(
        "--bc_artifact_root",
        default=None,
        help="Override generated prior artifact root.",
    )
    parser.add_argument(
        "--bc_task_slug",
        default=None,
        help="Override inferred prior task slug. Only valid with one task.",
    )
    parser.add_argument(
        "--bc_name_prefix",
        default=None,
        help="Override generated prior config/CSV row prefix. Only valid with one task.",
    )
    parser.add_argument(
        "--bc_overwrite",
        action="store_true",
        help="Overwrite generated BC prior YAMLs/CSV.",
    )
    parser.add_argument(
        "--bc_no_sample_weighting",
        action="store_true",
        help="Disable stage-based sample weighting in generated BC configs.",
    )
    parser.add_argument(
        "--bc_sample_weighting_no_validation",
        action="store_true",
        help="Use stage weights for training loss only, not validation/best-checkpoint selection.",
    )
    parser.add_argument(
        "--pipeline_mode",
        choices=["prepare", "train", "test", "all"],
        default="all",
        help="Prior pipeline mode after collection. all runs training, Kuafu params, BC eval, publish, weights, and residual launcher generation.",
    )
    parser.add_argument(
        "--train_stagger_sec",
        type=float,
        default=3.0,
        help="Seconds between prior training launches.",
    )
    parser.add_argument(
        "--eval_duration_sec",
        type=float,
        default=180.0,
        help="BC eval duration per checkpoint.",
    )
    parser.add_argument(
        "--eval_num_envs",
        type=int,
        default=8192,
        help="Parallel env count for BC eval.",
    )
    parser.add_argument(
        "--eval_chunk_exec",
        type=int,
        default=1,
        help="Actions executed from each predicted BC chunk during eval.",
    )
    parser.add_argument(
        "--success_rate_col",
        default=None,
        help="CSV success-rate column override for BC eval.",
    )
    parser.add_argument("--no_residual_launcher", action="store_true", help="Do not generate the residual RL launcher in all mode.")
    parser.add_argument("--residual_output_dir", default=None, help="Override generated residual launcher output directory.")
    parser.add_argument("--residual_output_name", default=None, help="Override generated residual launcher filename.")
    parser.add_argument(
        "--residual_selection_policy",
        choices=["best", "smallest_class_over_threshold"],
        default="smallest_class_over_threshold",
        help="Prior selection policy forwarded to run_bc_prior_pipeline.py.",
    )
    parser.add_argument(
        "--residual_success_threshold",
        type=float,
        default=20.0,
        help="Success-rate threshold forwarded to run_bc_prior_pipeline.py.",
    )
    parser.add_argument(
        "--residual_model_class_order",
        nargs="+",
        default=["bc", "class", "f3", "f", "ff3", "ff", "fff3", "fff"],
        help="Smallest-to-largest model class order forwarded to run_bc_prior_pipeline.py.",
    )
    parser.add_argument("--residual_overwrite", action="store_true", help="Overwrite an existing generated residual launcher.")
    parser.add_argument(
        "--conda_env",
        default="",
        help="Optional conda env for BC eval play jobs.",
    )
    parser.add_argument(
        "--retest_existing",
        action="store_true",
        help="Retest rows that already have a success rate.",
    )
    parser.add_argument(
        "--retrain_existing",
        action="store_true",
        help="Retrain prior CSV rows that already have best_checkpoint set.",
    )
    parser.add_argument(
        "--force_collect",
        action="store_true",
        help="Run collection even if the compact HDF5 output already exists.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print collection and training commands without running them.",
    )
    return parser.parse_known_args()


def external_artifact_root() -> Path:
    configured = os.environ.get("KUAFU_ARTIFACT_ROOT", "").strip()
    return (
        Path(configured).expanduser().resolve()
        if configured
        else REPO_ROOT.parent / "KuafuArtifacts"
    )


def require_external_output(path: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if resolved == REPO_ROOT or REPO_ROOT in resolved.parents:
        raise ValueError(f"{label} must be outside the source project ({REPO_ROOT}); got {resolved}")
    return resolved


def resolve_external_output(value: str | Path, *, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = external_artifact_root() / path
    return require_external_output(path, label=label)


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-") or "run"


def infer_robot_name(task_name: str) -> str:
    tokens = [token.lower() for token in re.split(r"[-_]+", task_name) if token]
    if tokens and tokens[0] == "panda":
        return "panda"
    if "kinova" in tokens:
        return "kinova"
    if tokens and tokens[0] == "fr3_finray":
        return "fr3_finray"
    raise ValueError(
        f"Could not infer robot name from task_name={task_name!r}. "
        "Pass --robot_name explicitly."
    )


def extract_forwarded_option(worker_args: list[str], flag: str) -> str | None:
    i = 0
    while i < len(worker_args):
        arg = worker_args[i]
        if arg == flag:
            if i + 1 < len(worker_args):
                return worker_args[i + 1]
            return None
        if arg.startswith(flag + "="):
            return arg.split("=", 1)[1]
        i += 1
    return None


def print_command(label: str, cmd: list[str]) -> None:
    print(f"[{label}] {shlex.join(cmd)}")


def run_command(label: str, cmd: list[str], *, dry_run: bool) -> None:
    print_command(label, cmd)
    if dry_run:
        return
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def build_paths(args: argparse.Namespace, task_name: str, robot_name: str, task_index: int) -> tuple[str, Path, Path]:
    if args.run_name:
        run_name = safe_filename(args.run_name)
        if len(args.task_name) > 1:
            run_name = f"{run_name}_{safe_filename(task_name)}"
    else:
        run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if task_index:
            run_name = f"{run_name}_{task_index:02d}"

    if args.output_root:
        if len(args.task_name) > 1:
            raise ValueError("--output_root is only valid when exactly one --task_name is provided.")
        output_root = resolve_external_output(args.output_root, label="--output_root")
    else:
        output_root = require_external_output(
            external_artifact_root() / "standalone" / "collection" / safe_filename(task_name) / run_name,
            label="default collection output",
        )

    compact_dir = (
        resolve_external_output(args.compact_output_dir, label="--compact_output_dir")
        if args.compact_output_dir
        else require_external_output(
            external_artifact_root() / "standalone" / "datasets",
            label="default compact output directory",
        )
    )
    compact_output = compact_dir / f"{safe_filename(robot_name)}_{safe_filename(task_name)}_{safe_filename(run_name)}.hdf5"
    return run_name, output_root.resolve(), require_external_output(compact_output, label="compact output")


def build_collect_cmd(
    args: argparse.Namespace,
    *,
    task_name: str,
    robot_name: str,
    output_root: Path,
    compact_output: Path,
    worker_args: list[str],
) -> list[str]:
    cmd = [
        args.python,
        str(COLLECT_LAUNCHER),
        "--gpus",
        *[str(gpu) for gpu in args.gpus],
        "--workers_per_gpu",
        str(args.workers_per_gpu),
        "--num_demos_per_worker",
        str(args.num_demos_per_worker),
        "--task_name",
        task_name,
        "--robot_name",
        robot_name,
        "--output_root",
        str(output_root),
        "--compact_output",
        str(compact_output),
        "--stagger_sec",
        str(args.collect_stagger_sec),
        "--python",
        args.python,
        "--no_bc_prep",
    ]
    if args.compact_skip_bad_files:
        cmd.append("--compact_skip_bad_files")
    if args.compact_no_strict_length:
        cmd.append("--compact_no_strict_length")
    cmd.extend(worker_args)
    return cmd


def build_train_cmd(
    args: argparse.Namespace,
    *,
    task_name: str,
    robot_name: str,
    run_name: str,
    compact_output: Path,
    action_mode: str,
) -> list[str]:
    cmd = [
        args.python,
        str(PRIOR_PIPELINE),
        "--robot_id",
        ROBOT_IDS[robot_name],
        "--mode",
        args.pipeline_mode,
        "--dataset",
        str(compact_output),
        "--task_name",
        task_name,
        "--run_name",
        run_name,
        "--obs_key",
        args.bc_obs_key,
        "--action_mode",
        action_mode,
        "--time_shift",
        str(args.bc_time_shift),
        "--num_epochs",
        str(args.bc_num_epochs),
        "--bc_num_workers",
        str(args.bc_num_workers),
        "--chunk_sizes",
        *[str(size) for size in args.bc_chunk_sizes],
        "--chunk_strides",
        *[str(stride) for stride in args.bc_chunk_strides],
        "--variants",
        *args.bc_variants,
        "--gpus",
        *[str(gpu) for gpu in args.gpus],
        "--duration_sec",
        str(args.eval_duration_sec),
        "--num_envs",
        str(args.eval_num_envs),
        "--chunk_exec",
        str(args.eval_chunk_exec),
        "--workers_per_gpu",
        str(args.workers_per_gpu),
    ]
    if args.success_rate_col:
        cmd.extend(["--success_rate_col", args.success_rate_col])
    if args.conda_env:
        cmd.extend(["--conda_env", args.conda_env])
    if args.retest_existing:
        cmd.append("--retest_existing")
    prior_artifact_root = (
        resolve_external_output(args.bc_artifact_root, label="--bc_artifact_root")
        if args.bc_artifact_root
        else require_external_output(
            compact_output.parent / "prior_artifacts" / safe_filename(run_name),
            label="default BC artifact root",
        )
    )
    cmd.extend(["--artifact_root", str(prior_artifact_root)])
    cmd.extend(["--priors_root", str(prior_artifact_root / "published_priors")])
    if args.bc_task_slug:
        cmd.extend(["--task_slug", args.bc_task_slug])
    if args.bc_name_prefix:
        cmd.extend(["--name_prefix", args.bc_name_prefix])
    if args.bc_overwrite:
        cmd.append("--overwrite")
    if args.bc_pin_memory:
        cmd.append("--bc_pin_memory")
    if args.bc_no_sample_weighting:
        cmd.append("--no_sample_weighting")
    if args.bc_sample_weighting_no_validation:
        cmd.append("--sample_weighting_no_validation")
    if args.no_residual_launcher:
        cmd.append("--no_residual_launcher")
    cmd.extend(["--residual_selection_policy", args.residual_selection_policy])
    cmd.extend(["--residual_success_threshold", str(args.residual_success_threshold)])
    if args.residual_model_class_order:
        cmd.extend(["--residual_model_class_order", *args.residual_model_class_order])
    if args.residual_output_dir:
        cmd.extend(["--residual_output_dir", args.residual_output_dir])
    if args.residual_output_name:
        cmd.extend(["--residual_output_name", args.residual_output_name])
    if args.residual_overwrite:
        cmd.append("--residual_overwrite")
    if args.retrain_existing:
        cmd.append("--retrain_existing")
    return cmd


def main() -> None:
    args, worker_args = parse_args()
    removed_concurrency_args = ("--num_workers", "--max_concurrent", "--eval_workers_per_gpu")
    for flag in removed_concurrency_args:
        if any(arg == flag or arg.startswith(flag + "=") for arg in worker_args):
            raise ValueError(f"{flag} was removed; use --workers_per_gpu as the only concurrency knob.")
    if not COLLECT_LAUNCHER.exists():
        raise FileNotFoundError(f"Collection launcher not found: {COLLECT_LAUNCHER}")
    if not PRIOR_PIPELINE.exists():
        raise FileNotFoundError(f"Prior pipeline not found: {PRIOR_PIPELINE}")
    if len(args.task_name) > 1 and (args.bc_task_slug or args.bc_name_prefix):
        raise ValueError("--bc_task_slug and --bc_name_prefix are only valid with one task.")

    action_mode = (
        args.bc_action_mode
        or extract_forwarded_option(worker_args, "--control_output_mode")
        or "absolute_ee"
    )

    print("=== MPC Collection + BC Prior Pipeline ===")
    print(f"  Repo:              {REPO_ROOT}")
    print(f"  Tasks:             {args.task_name}")
    print(f"  GPUs:              {args.gpus}")
    print(f"  Action mode:       {action_mode}")
    print(f"  Worker args:       {' '.join(worker_args) if worker_args else '(none)'}")
    print()

    for task_index, task_name in enumerate(args.task_name):
        robot_name = args.robot_name or infer_robot_name(task_name)
        run_name, output_root, compact_output = build_paths(args, task_name, robot_name, task_index)

        print(f"=== Task: {task_name} ===")
        print(f"  Robot:             {robot_name}")
        print(f"  Run name:          {run_name}")
        print(f"  Output root:       {output_root}")
        print(f"  Dataset:           {compact_output}")

        collect_cmd = build_collect_cmd(
            args,
            task_name=task_name,
            robot_name=robot_name,
            output_root=output_root,
            compact_output=compact_output,
            worker_args=worker_args,
        )
        train_cmd = build_train_cmd(
            args,
            task_name=task_name,
            robot_name=robot_name,
            run_name=run_name,
            compact_output=compact_output,
            action_mode=action_mode,
        )

        if compact_output.exists() and compact_output.stat().st_size > 0 and not args.force_collect:
            print(f"[collect] resume: using existing dataset {compact_output}")
        else:
            run_command("collect", collect_cmd, dry_run=args.dry_run)
        run_command("prior", train_cmd, dry_run=args.dry_run)
        print()


if __name__ == "__main__":
    main()
