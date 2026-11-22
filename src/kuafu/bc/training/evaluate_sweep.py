#!/usr/bin/env python3
"""Run play_bc_env.py for each row in a sweep CSV using best_checkpoint, log success rate.

Runs are headless (--headless + HEADLESS=1). Run this script with the isaac conda env
active (conda activate isaac) so the play subprocess uses the correct Python. Optionally
use --conda_env isaac to run via `conda run -n isaac` (can fail on some setups). Reads a
CSV with columns name, config, best_checkpoint (and optionally success_rate). Skips rows
with empty best_checkpoint. Use --skip_tested to skip rows that already have success_rate.
Distributes jobs across GPUs round-robin and runs up to N
concurrent play jobs. When a job finishes, parses the play log for success rate and
updates the CSV with success_rate as the last column.

Usage:
    conda activate isaac
    python src/kuafu/bc/training/evaluate_sweep.py --csv src/kuafu/bc/training/sweep.csv --gpus 0 1

    # Choose a robot/task family without hand-editing task names
    python src/kuafu/bc/training/evaluate_sweep.py --csv sweep.csv --gpus 0 --robot_id franka_panda --task_suffix Drawer --duration_sec 120

    # Or use the robot-specific wrapper
    bash src/kuafu/bc/training/robots/franka_panda/test_sweep.sh --csv sweep.csv --gpus 0 --task_suffix Drawer

    # Dry-run
    python src/kuafu/bc/training/evaluate_sweep.py --csv sweep.csv --gpus 0 --dry_run

    # Only run specific rows by name
    python src/kuafu/bc/training/evaluate_sweep.py --csv sweep.csv --gpus 0 1 --only pen_cup_chunking_0,pen_cup_chunking_15
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from kuafu.artifact_contracts import evaluation_identity

PLAY_SCRIPT = Path(__file__).resolve().parent / "play_bc_env.py"
PROJECT_ROOT = Path(__file__).resolve().parents[4]


def _artifact_log_root() -> Path:
    prior_root = os.environ.get("KUAFU_PRIOR_ARTIFACT_ROOT", "").strip()
    if prior_root:
        return Path(prior_root).expanduser().resolve() / "logs"
    task_root = os.environ.get("KUAFU_TASK_ARTIFACT_ROOT", "").strip()
    if task_root:
        return Path(task_root).expanduser().resolve() / "prior_artifacts" / "logs"
    configured = os.environ.get("KUAFU_ARTIFACT_ROOT", "").strip()
    base = Path(configured).expanduser().resolve() if configured else PROJECT_ROOT.parent / "KuafuArtifacts"
    return base / "standalone" / "prior_sweeps" / "logs"


def _require_external_mutable_path(path: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if resolved == PROJECT_ROOT or PROJECT_ROOT in resolved.parents:
        raise ValueError(f"{label} must be outside the source project ({PROJECT_ROOT}); got {resolved}")
    return resolved


BEST_CKPT_COL = "best_checkpoint"
SUCCESS_RATE_COL = "success_rate"
DEFAULT_TASK_SUFFIX = "Drawer"
ROBOT_TASK_PREFIXES = {
    "fr3_finray": "FR3-FinRay-Table",
    "franka_panda": "Panda-Table",
    "kinova_gen3": "Kinova-Table-Gen3",
}
ROBOT_ALIASES = {
    "fr3_finray": "fr3_finray",
    "franka": "fr3_finray",
    "panda": "fr3_finray",
    "panda": "franka_panda",
    "kinova": "kinova_gen3",
}


def _resolve_robot_id(robot_id: str | None) -> str:
    text = (robot_id or "fr3_finray").strip()
    resolved = ROBOT_ALIASES.get(text, text)
    if resolved not in ROBOT_TASK_PREFIXES:
        valid = ", ".join(sorted(ROBOT_TASK_PREFIXES))
        raise ValueError(f"Unknown robot id {robot_id!r}; expected one of: {valid}")
    return resolved


def _compose_task_name(robot_id: str, task_suffix: str) -> str:
    suffix = task_suffix.strip()
    if not suffix:
        raise ValueError("task suffix must be non-empty")
    for prefix in ROBOT_TASK_PREFIXES.values():
        if suffix.startswith(prefix + "-"):
            return suffix
    return f"{ROBOT_TASK_PREFIXES[robot_id]}-{suffix}"


def _resolve_task_name(task: str | None, task_suffix: str, robot_id: str) -> str:
    if task and task.strip():
        text = task.strip()
        for prefix in ROBOT_TASK_PREFIXES.values():
            if text.startswith(prefix + "-"):
                return text
        return _compose_task_name(robot_id, text)
    return _compose_task_name(robot_id, task_suffix)


def _default_success_rate_col(robot_id: str, explicit: str | None) -> str:
    if explicit and explicit.strip():
        return explicit.strip()
    if robot_id == "fr3_finray":
        return SUCCESS_RATE_COL
    return f"{SUCCESS_RATE_COL}_{robot_id}"


def _row_to_cmd(
    row: dict[str, str],
    python: str,
    task: str,
    execution_robot_id: str,
    duration_sec: float,
    num_envs: int,
    chunk_exec: int,
    extra_args: list[str],
    conda_env: str | None,
    params_path: str | None = None,
    program_path: str | None = None,
) -> list[str]:
    ckpt = (row.get(BEST_CKPT_COL) or "").strip()
    if not ckpt:
        raise ValueError(f"Row has no '{BEST_CKPT_COL}'; cannot build play command.")
    play_args = [
        str(PLAY_SCRIPT),
        "--headless",
        "--checkpoint",
        ckpt,
        "--task",
        task,
        "--execution_robot_id",
        execution_robot_id,
        "--duration_sec",
        str(duration_sec),
        "--num_envs",
        str(num_envs),
        "--chunk_exec",
        str(chunk_exec),
    ]
    if params_path:
        play_args.extend(["--params_path", params_path])
    if program_path:
        play_args.extend(["--program_path", program_path])
    play_args.extend(extra_args)
    if conda_env:
        cmd = ["conda", "run", "-n", conda_env, "python"] + play_args
    else:
        cmd = [python] + play_args
    return cmd


def _extract_success_rate(log_path: Path) -> str | None:
    """Parse the play log for machine-parseable 'SUCCESS_RATE:42.5' from play_bc_env.py."""
    try:
        text = log_path.read_text()
    except OSError:
        return None
    # play_bc_env.py prints this line at the end (flush=True) for automation
    match = re.search(r"SUCCESS_RATE:([\d.]+)", text)
    if not match:
        return None
    return match.group(1).strip()


def _extract_bc_eval_stats(log_path: Path, stats_path: Path | None = None) -> dict | None:
    """Load BC rollout stats from the JSON sidecar, falling back to the play log line."""
    if stats_path is not None and stats_path.exists():
        try:
            with stats_path.open(encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except (OSError, json.JSONDecodeError):
            pass
    try:
        text = log_path.read_text(encoding="utf-8")
    except OSError:
        return None
    prefix = "BC_EVAL_STATS_JSON:"
    for line in reversed(text.splitlines()):
        if not line.startswith(prefix):
            continue
        try:
            data = json.loads(line[len(prefix):].strip())
        except json.JSONDecodeError:
            return None
        return data if isinstance(data, dict) else None
    return None


def _compact_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def _constraint_unsatisfied_stats(constraint_stats: dict | None) -> dict:
    if not isinstance(constraint_stats, dict):
        return {}
    out = {}
    for stage_id, constraints in constraint_stats.items():
        if not isinstance(constraints, dict):
            continue
        stage_out = {}
        for constraint_id, stats in constraints.items():
            if not isinstance(stats, dict):
                continue
            total = int(stats.get("total") or 0)
            met = int(stats.get("met") or 0)
            if total <= 0:
                continue
            not_met = max(total - met, 0)
            stage_out[str(constraint_id)] = {
                "rate": float(not_met) / float(total),
                "not_met": int(not_met),
                "total": int(total),
            }
        if stage_out:
            out[str(stage_id)] = stage_out
    return out


def _stats_to_csv_values(stats: dict, stats_path: Path | None) -> dict[str, str]:
    constraint_stats = stats.get("constraint_met_rate_stats")
    return {
        "bc_eval_stats_path": str(stats_path) if stats_path is not None else "",
        "bc_eval_true_success_count": str(int(stats.get("true_success_count") or 0)),
        "bc_eval_false_complete_count": str(int(stats.get("false_complete_count") or 0)),
        "bc_eval_timeout_count": str(int(stats.get("timeout_count") or 0)),
        "bc_eval_total_outcomes": str(int(stats.get("total_outcomes") or stats.get("total_episodes") or 0)),
        "bc_eval_stage_pass_rates": _compact_json(stats.get("stage_pass_rates") or {}),
        "bc_eval_stage_transition_stats": _compact_json(stats.get("stage_transition_stats") or {}),
        "bc_eval_constraint_satisfied_on_timeout": _compact_json(constraint_stats or {}),
        "bc_eval_constraint_unsatisfied_on_timeout": _compact_json(_constraint_unsatisfied_stats(constraint_stats)),
        "bc_eval_params_path": str(stats.get("params_path") or ""),
    }


def _update_csv_row_values(csv_path: Path, row_idx: int, updates: dict[str, str]) -> None:
    """Update one row in the CSV, adding columns as needed."""
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    for column in updates:
        if column not in fieldnames:
            fieldnames.append(column)

    if 0 <= row_idx < len(rows):
        rows[row_idx].update(updates)

    # Strip any None-keyed columns that csv.DictReader creates for trailing commas
    rows = [{k: v for k, v in row.items() if k is not None} for row in rows]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _update_csv_success_rate(csv_path: Path, row_idx: int, success_rate: str, column: str) -> None:
    _update_csv_row_values(csv_path, row_idx, {column: success_rate})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run play_bc_env.py for each sweep row (best_checkpoint), log success rate."
    )
    p.add_argument("--csv", type=str, required=True, help="Path to CSV config file.")
    p.add_argument("--gpus", type=int, nargs="+", default=[0], help="GPU IDs to distribute across.")
    p.add_argument(
        "--workers_per_gpu",
        type=int,
        default=2,
        help="Number of concurrent eval jobs allowed per GPU.",
    )
    p.add_argument("--dry_run", action="store_true", help="Print commands without running them.")
    p.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma-separated list of run names to include (skip the rest).",
    )
    p.add_argument(
        "--skip_tested",
        type=_str_to_bool,
        default=True,
        help="Skip rows that already have a non-empty success_rate in the CSV.",
    )
    p.add_argument(
        "--stagger_sec",
        type=float,
        default=3.0,
        help="Seconds to wait between launching consecutive jobs.",
    )
    p.add_argument(
        "--conda_env",
        type=str,
        default="",
        help="If set, run play via `conda run -n <name> python ...`. Default: use current interpreter (run with isaac env active).",
    )
    p.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python interpreter when not using --conda_env (default: current interpreter).",
    )
    p.add_argument(
        "--robot_id",
        "--execution_robot_id",
        dest="execution_robot_id",
        type=str,
        default="fr3_finray",
        help="Robot task package for evaluation: fr3_finray, franka_panda, or kinova_gen3.",
    )
    p.add_argument(
        "--task_suffix",
        type=str,
        default=DEFAULT_TASK_SUFFIX,
        help="Task suffix composed with --robot_id when --task is not provided, e.g. Drawer, Can-ff, Hang-Cup-fff.",
    )
    p.add_argument(
        "--task",
        type=str,
        default=None,
        help="Optional exact Isaac Lab task name. If this is only a suffix, it is composed with --robot_id.",
    )
    p.add_argument(
        "--success_rate_col",
        type=str,
        default=None,
        help="CSV column to update. Defaults to success_rate for fr3_finray and success_rate_<robot_id> otherwise.",
    )
    p.add_argument(
        "--duration_sec",
        type=float,
        default=180.0,
        help="Simulation duration per run in seconds.",
    )
    p.add_argument(
        "--num_envs",
        type=int,
        default=8192,
        help="Number of parallel environments.",
    )
    p.add_argument(
        "--chunk_exec",
        type=int,
        default=1,
        help="How many actions from each predicted chunk to execute. 0 uses the full chunk.",
    )
    p.add_argument(
        "--params_path",
        type=str,
        default=None,
        help="Path to a best.json file. Passed to play_bc_env.py to overwrite per-stage "
             "constraint parameters and residual weights for all jobs in this sweep.",
    )
    p.add_argument("--program_path", default=os.environ.get("KUAFU_PROGRAM_PATH"), help="Collected learned program to evaluate.")
    args, extra = p.parse_known_args()
    args.extra = extra  # Pass through to play_bc_env.py (e.g. --dataset path.hdf5 --seed 0)
    return args


def main() -> None:
    args = parse_args()
    try:
        execution_robot_id = _resolve_robot_id(args.execution_robot_id)
        task = _resolve_task_name(args.task, args.task_suffix, execution_robot_id)
        success_rate_col = _default_success_rate_col(execution_robot_id, args.success_rate_col)
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)
    if args.workers_per_gpu < 1:
        print("[ERROR] --workers_per_gpu must be >= 1")
        sys.exit(1)
    worker_slots = len(args.gpus) * args.workers_per_gpu

    csv_path = _require_external_mutable_path(Path(args.csv), label="--csv")
    if not csv_path.exists():
        print(f"[ERROR] CSV file not found: {csv_path}")
        sys.exit(1)

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("[ERROR] CSV has no data rows.")
        sys.exit(1)

    only_set: set[str] | None = None
    if args.only:
        only_set = {n.strip() for n in args.only.split(",")}

    jobs: list[dict] = []
    for i, row in enumerate(rows):
        name = row.get("name", f"run_{i}").strip() or f"run_{i}"
        if only_set and name not in only_set:
            continue
        ckpt = (row.get(BEST_CKPT_COL) or "").strip()
        if not ckpt or not Path(ckpt).expanduser().is_file():
            raise FileNotFoundError(f"{name}: missing best_checkpoint: {ckpt!r}; run train-prior first")
        identity = evaluation_identity(
            checkpoint=ckpt, task=task, program_path=args.program_path, params_path=args.params_path,
            duration_sec=args.duration_sec, num_envs=args.num_envs, chunk_exec=args.chunk_exec, extra=args.extra,
        ) if not args.dry_run else "dry-run"
        stats_file = Path(row.get("bc_eval_stats_path") or "")
        if (args.skip_tested and args.program_path and row.get(success_rate_col)
                and row.get("bc_eval_identity") == identity and stats_file.is_file()):
            try:
                cached = json.loads(stats_file.read_text())
            except (OSError, ValueError):
                cached = {}
            if cached.get("evaluation_identity") == identity:
                print(f"  [SKIP] {name}: verified evaluation inputs and statistics")
                continue
        try:
            conda_env = (args.conda_env or "").strip() or None
            cmd = _row_to_cmd(
                row,
                args.python,
                task,
                execution_robot_id,
                args.duration_sec,
                args.num_envs,
                args.chunk_exec,
                args.extra or [],
                conda_env,
                params_path=args.params_path or None,
                program_path=args.program_path,
            )
        except ValueError as e:
            print(f"  [SKIP] {name}: {e}")
            continue
        jobs.append({"name": name, "cmd": cmd, "row_idx": i, "identity": identity})

    if not jobs:
        print("[WARN] No matching jobs with best_checkpoint. Check --only filter or CSV content.")
        sys.exit(0)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    sweep_log_dir = _artifact_log_root() / "evaluation" / execution_robot_id / task / timestamp
    if not args.dry_run:
        sweep_log_dir.mkdir(parents=True, exist_ok=True)

    print("=== BC Play Test Sweep ===")
    print(f"  CSV:            {csv_path}")
    print(f"  Total jobs:     {len(jobs)}")
    print(f"  GPUs:           {args.gpus}")
    print(f"  Workers / GPU:  {args.workers_per_gpu}")
    print(f"  Worker slots:   {worker_slots}")
    print(f"  Robot:          {execution_robot_id}")
    print(f"  Task:           {task}")
    print(f"  Success column: {success_rate_col}")
    print(f"  Duration (s):   {args.duration_sec}")
    print(f"  Conda env:     {args.conda_env or '(none; use current interpreter)'}")
    print(f"  Sweep logs:    {sweep_log_dir}")
    print()

    for job in jobs:
        stats_path = sweep_log_dir / f"{job['name']}.stats.json"
        job["stats_path"] = stats_path
        job["cmd"].extend(["--stats_path", str(stats_path)])

    if args.dry_run:
        for i, j in enumerate(jobs):
            gpu = args.gpus[i % len(args.gpus)]
            print(f"  [{j['name']}] GPU={gpu}")
            print(f"    CUDA_VISIBLE_DEVICES={gpu} {' '.join(j['cmd'])}")
            print()
        print("[DRY RUN] No jobs were launched.")
        return

    active: list[dict] = []
    pending = list(jobs)
    finished: list[dict] = []
    interrupted = False

    def _handle_interrupt(signum, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, _handle_interrupt)
    signal.signal(signal.SIGTERM, _handle_interrupt)

    def _finish_job(a: dict) -> None:
        a["log_file"].close()
        rc = a["returncode"]
        elapsed = time.time() - a["start_time"]
        status = "OK" if rc == 0 else f"FAILED (rc={rc})"
        print(f"  [{a['name']}] finished — {status}  ({elapsed:.0f}s)")

        if rc == 0:
            stats_path = a.get("stats_path")
            stats = _extract_bc_eval_stats(a["log_path"], stats_path if isinstance(stats_path, Path) else None)
            rate = _extract_success_rate(a["log_path"])
            if rate is None and stats is not None and stats.get("success_rate_pct") is not None:
                rate = f"{float(stats['success_rate_pct']):.1f}"

            updates: dict[str, str] = {}
            if rate is not None:
                updates[success_rate_col] = rate
            if stats is not None:
                stats["evaluation_identity"] = a["identity"]
                if isinstance(stats_path, Path):
                    stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")
                updates["bc_eval_identity"] = a["identity"]
                updates.update(_stats_to_csv_values(stats, stats_path if isinstance(stats_path, Path) else None))

            if updates:
                _update_csv_row_values(csv_path, a["row_idx"], updates)
            if rate is not None:
                print(f"  [{a['name']}] {success_rate_col} -> {rate}%")
            else:
                print(f"  [{a['name']}] WARNING: could not find success rate in log")
            if stats is not None:
                print(f"  [{a['name']}] BC eval stats -> {stats_path}")
            else:
                print(f"  [{a['name']}] WARNING: could not find BC eval stats in log")

            if rate is None or stats is None:
                a["returncode"] = 1
                print(f"  [{a['name']}] ERROR: incomplete BC eval output; marking job failed")

        finished.append(a)

    def _poll_active():
        still_running = []
        for a in active:
            rc = a["proc"].poll()
            if rc is not None:
                a["returncode"] = rc
                _finish_job(a)
            else:
                still_running.append(a)
        active.clear()
        active.extend(still_running)

    while pending or active:
        _poll_active()

        if interrupted:
            print("\n[interrupted] Stopping test sweep. CSV already updated for completed jobs.")
            break

        while pending and len(active) < worker_slots:
            j = pending.pop(0)
            # Assign a GPU with an available worker slot.
            active_per_gpu = {gpu: 0 for gpu in args.gpus}
            for a in active:
                active_per_gpu[a["gpu_id"]] = active_per_gpu.get(a["gpu_id"], 0) + 1
            available_gpus = sorted(g for g in args.gpus if active_per_gpu.get(g, 0) < args.workers_per_gpu)
            if not available_gpus:
                pending.insert(0, j)
                break
            gpu_id = available_gpus[0]
            j["gpu_id"] = gpu_id

            log_path = sweep_log_dir / f"{j['name']}.txt"

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            env["PYTHONUNBUFFERED"] = "1"
            env["HEADLESS"] = "1"

            # Prepend env vars to the command so the process that runs python/conda
            # definitely sees them (avoids conda run or other wrappers dropping CUDA_VISIBLE_DEVICES)
            if os.name != "nt":
                launch_cmd = [
                    "env",
                    f"CUDA_VISIBLE_DEVICES={gpu_id}",
                    "HEADLESS=1",
                    "PYTHONUNBUFFERED=1",
                ] + j["cmd"]
            else:
                launch_cmd = j["cmd"]

            log_f = open(log_path, "w")
            print(f"  [{j['name']}] launching on GPU {gpu_id}  ->  {log_path}")
            proc = subprocess.Popen(
                launch_cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                env=env,
            )
            j["proc"] = proc
            j["log_file"] = log_f
            j["log_path"] = log_path
            j["start_time"] = time.time()
            active.append(j)

            if pending:
                time.sleep(args.stagger_sec)

        if active:
            time.sleep(5)

    ok = [f for f in finished if f["returncode"] == 0]
    fail = [f for f in finished if f["returncode"] != 0]
    still_running = [a["name"] for a in active] if interrupted else []

    print()
    print(f"=== Test Sweep {'Interrupted' if interrupted else 'Complete'} ===")
    print(f"  Succeeded: {len(ok)}/{len(finished)}")
    if fail:
        print(f"  Failed:    {', '.join(f['name'] for f in fail)}")
    if still_running:
        print(f"  Still running (not updated in CSV): {', '.join(still_running)}")
    print(f"  Logs:      {sweep_log_dir}")
    print(f"  CSV updated: {csv_path} ({success_rate_col} column)")

    if interrupted:
        sys.exit(130)
    if fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
