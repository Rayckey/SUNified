#!/usr/bin/env python3
"""Run RL training from a CSV of BC priors and write learned RL checkpoints back.

Each CSV row is treated as one RL training job. The BC prior checkpoint is read from a
CSV column (default: ``best_checkpoint``) and passed to the selected train script via
``--prior_checkpoint``. The train script path and Isaac task name are CLI arguments;
everything else can be left at the train script defaults, with optional extra args
passed through unchanged.

When a job finishes successfully, this script parses the train log to find the RL run
directory, locates the learned checkpoint inside it, and writes that path back into the
CSV (default output column: ``best_rl_checkpoint``).
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from kuafu.paths import external_artifact_root, require_external_artifact_path

PRIOR_CKPT_COL = "best_checkpoint"
RL_CKPT_COL = "best_rl_checkpoint"


def _resolve_path(val: str, *, csv_path: Path) -> Path:
    p = Path(val).expanduser()
    if p.is_absolute():
        return p.resolve()
    for base in (Path.cwd(), csv_path.parent):
        candidate = (base / p).resolve()
        if candidate.exists():
            return candidate
    return (Path.cwd() / p).resolve()


def _try_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd()))
    except ValueError:
        return str(path.resolve())


def _infer_task_variant_from_config_name(config_name: str) -> str:
    stem = Path(config_name).stem
    if "u_long_long" in stem.lower():
        return "-Long-Long-U"
    if re.search(r"(?:^|_)f(?:_|$)", stem):
        return "-f"
    if re.search(r"(?:^|_)l(?:_|$)", stem):
        return "-l"
    return ""


def _compose_task_name(task_base: str, variant_suffix: str) -> str:
    if not variant_suffix:
        return task_base
    if task_base.endswith("-Slowing"):
        prefix = task_base[: -len("-Slowing")]
        return f"{prefix}{variant_suffix}-Slowing"
    return f"{task_base}{variant_suffix}"


def _read_prior_config_path(prior_ckpt: Path) -> str:
    run_dir = prior_ckpt.parent.parent if prior_ckpt.parent.name == "checkpoints" else prior_ckpt.parent
    primary_cfg = run_dir / "meta" / "config.yaml"
    fallback_cfg = run_dir / "train_config.yaml"

    if primary_cfg.exists():
        selected_cfg = primary_cfg
    elif fallback_cfg.exists():
        selected_cfg = fallback_cfg
    else:
        raise ValueError(
            "Could not infer RL task variant from prior checkpoint. "
            f"Expected `{primary_cfg}` or fallback `{fallback_cfg}`. Refusing to launch training."
        )

    try:
        text = selected_cfg.read_text()
    except OSError as exc:
        raise ValueError(f"Could not read prior config file: {selected_cfg}") from exc

    match = re.search(r"^\s*config_path:\s*([^\n]+)\s*$", text, re.MULTILINE)
    if not match:
        raise ValueError(
            f"Could not find `config_path` in prior config file: {selected_cfg}. Refusing to launch training."
        )

    return match.group(1).strip().strip("'\"")


def _infer_task_name_from_prior(prior_ckpt: Path, task_base: str) -> str:
    config_name = _read_prior_config_path(prior_ckpt)
    stem_lower = Path(config_name).stem.lower()
    if "long_long" in stem_lower and "u_long_long" not in stem_lower:
        raise ValueError(
            f"Unsupported plain long_long prior for RL task inference: {config_name}. "
            "This repo has registered RL task variants for '-Long-Long-U', '-f', and '-l', "
            "but not for plain 'long_long'."
        )
    variant_suffix = _infer_task_variant_from_config_name(config_name)
    return _compose_task_name(task_base, variant_suffix)


def _row_to_cmd(
    row: dict[str, str],
    python: str,
    *,
    csv_path: Path,
    train_script: str,
    task_base: str,
    prior_col: str,
    extra_args: list[str],
    conda_env: str | None,
    params_path: str | None = None,
) -> tuple[list[str], str]:
    prior_ckpt = (row.get(prior_col) or "").strip()
    if not prior_ckpt:
        raise ValueError(f"Row has no '{prior_col}'; cannot build train command.")

    resolved_train_script = _resolve_path(train_script, csv_path=csv_path)
    if not resolved_train_script.exists():
        raise ValueError(f"Train script not found: {resolved_train_script}")

    resolved_prior_ckpt = _resolve_path(prior_ckpt, csv_path=csv_path)
    if not resolved_prior_ckpt.exists():
        raise ValueError(f"Prior checkpoint not found: {resolved_prior_ckpt}")

    resolved_task = _infer_task_name_from_prior(resolved_prior_ckpt, task_base)
    train_args = [
        str(resolved_train_script),
        "--task",
        resolved_task,
        "--prior_checkpoint",
        str(resolved_prior_ckpt),
    ]
    if params_path:
        train_args.extend(["--params_path", params_path])
    train_args.extend(extra_args)

    if conda_env:
        return ["conda", "run", "-n", conda_env, "python"] + train_args, resolved_task
    return [python] + train_args, resolved_task


def _extract_run_dir(log_path: Path) -> Path | None:
    try:
        text = log_path.read_text()
    except OSError:
        return None
    match = re.search(r"Exact experiment name requested from command line:\s*(.+)", text)
    if not match:
        return None
    return Path(match.group(1).strip()).expanduser().resolve()


def _extract_agent_name(run_dir: Path) -> str | None:
    agent_yaml = run_dir / "params" / "agent.yaml"
    try:
        text = agent_yaml.read_text()
    except OSError:
        return None
    match = re.search(r"^\s*name:\s*([^\n]+)\s*$", text, re.MULTILINE)
    if not match:
        return None
    return match.group(1).strip().strip("'\"")


def _candidate_rl_checkpoints(run_dir: Path) -> list[Path]:
    if not run_dir.exists():
        return []
    candidates = [p.resolve() for p in run_dir.rglob("*.pth") if p.is_file()]
    return sorted(set(candidates), key=lambda p: (p.stat().st_mtime, str(p)), reverse=True)


def _extract_best_rl_checkpoint(log_path: Path) -> str | None:
    run_dir = _extract_run_dir(log_path)
    if run_dir is None or not run_dir.exists():
        return None

    agent_name = _extract_agent_name(run_dir)
    nn_dir = run_dir / "nn"
    preferred: list[Path] = []

    if agent_name:
        preferred.extend(
            [
                nn_dir / f"{agent_name}.pth",
                nn_dir / f"best_{agent_name}.pth",
                nn_dir / f"last_{agent_name}.pth",
            ]
        )
        preferred.extend(sorted(nn_dir.glob(f"{agent_name}*.pth")))
        preferred.extend(sorted(nn_dir.glob(f"*{agent_name}*.pth")))

    preferred.extend([run_dir / "checkpoints" / "best.pth", run_dir / "checkpoints" / "final.pth"])

    for path in preferred:
        if path.is_file():
            return _try_rel(path)

    candidates = _candidate_rl_checkpoints(run_dir)
    if not candidates:
        return None

    nn_candidates = [p for p in candidates if "/nn/" in p.as_posix()]
    if nn_candidates:
        return _try_rel(nn_candidates[0])
    return _try_rel(candidates[0])


def _update_csv(csv_path: Path, row_idx: int, output_col: str, value: str) -> None:
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if output_col not in fieldnames:
        fieldnames.append(output_col)

    if 0 <= row_idx < len(rows):
        rows[row_idx][output_col] = value

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Launch RL training sweep from CSV priors.")
    p.add_argument("--csv", type=str, required=True, help="Path to CSV config file.")
    p.add_argument("--train_script", type=str, required=True, help="Path to RL train script to launch.")
    p.add_argument(
        "--task",
        type=str,
        required=True,
        help="Base Isaac Lab task name. Variant suffix is inferred per prior checkpoint.",
    )
    p.add_argument("--gpus", type=int, nargs="+", default=[0], help="GPU IDs to distribute across.")
    p.add_argument(
        "--max_concurrent",
        type=int,
        default=None,
        help="Max concurrent jobs (default: one per GPU).",
    )
    p.add_argument("--dry_run", action="store_true", help="Print commands without running them.")
    p.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma-separated list of run names to include (skip the rest).",
    )
    p.add_argument(
        "--overwrite_existing",
        action="store_true",
        help="Relaunch rows even if they already have a non-empty learned RL checkpoint column.",
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
        help="If set, run training via `conda run -n <name> python ...`.",
    )
    p.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python interpreter when not using --conda_env (default: current interpreter).",
    )
    p.add_argument(
        "--prior_col",
        type=str,
        default=PRIOR_CKPT_COL,
        help=f"CSV column containing the BC prior checkpoint (default: {PRIOR_CKPT_COL}).",
    )
    p.add_argument(
        "--output_col",
        type=str,
        default=RL_CKPT_COL,
        help=f"CSV column to write the learned RL checkpoint into (default: {RL_CKPT_COL}).",
    )
    p.add_argument(
        "--params_path",
        type=str,
        default=None,
        help="Path to a best.json file. Passed to the train script via --params_path to overwrite "
             "per-stage constraint parameters and residual weights for all jobs in this sweep.",
    )
    args, extra = p.parse_known_args()
    args.extra = [arg for arg in extra if arg != "--"]
    return args


def main() -> None:
    args = parse_args()
    max_concurrent = args.max_concurrent or len(args.gpus)

    csv_path = require_external_artifact_path(args.csv, "RL training sweep CSV")
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
        if not args.overwrite_existing and (row.get(args.output_col) or "").strip():
            print(f"  [SKIP] {name}: already has {args.output_col}")
            continue
        prior_ckpt = (row.get(args.prior_col) or "").strip()
        if not prior_ckpt:
            print(f"  [SKIP] {name}: no {args.prior_col}")
            continue
        try:
            conda_env = (args.conda_env or "").strip() or None
            cmd = _row_to_cmd(
                row,
                args.python,
                csv_path=csv_path,
                train_script=args.train_script,
                task_base=args.task,
                prior_col=args.prior_col,
                extra_args=args.extra or [],
                conda_env=conda_env,
                params_path=args.params_path or None,
            )
            cmd, resolved_task = cmd
        except ValueError as e:
            print(f"  [SKIP] {name}: {e}")
            continue
        jobs.append({"name": name, "cmd": cmd, "row_idx": i, "task": resolved_task})

    if not jobs:
        print(f"[WARN] No matching jobs with {args.prior_col}. Check --only filter or CSV content.")
        sys.exit(0)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    sweep_log_dir = external_artifact_root() / "standalone" / "rl_training_sweeps" / timestamp
    sweep_log_dir.mkdir(parents=True, exist_ok=True)

    print("=== RL Training Sweep ===")
    print(f"  CSV:            {csv_path}")
    print(f"  Train script:   {Path(args.train_script).expanduser()}")
    print(f"  Task base:      {args.task}")
    print(f"  Total jobs:     {len(jobs)}")
    print(f"  GPUs:           {args.gpus}")
    print(f"  Max concurrent: {max_concurrent}")
    print(f"  Prior column:   {args.prior_col}")
    print(f"  Output column:  {args.output_col}")
    print(f"  Conda env:      {args.conda_env or '(none; use current interpreter)'}")
    print(f"  Sweep logs:     {sweep_log_dir}")
    print()

    if args.dry_run:
        for i, j in enumerate(jobs):
            gpu = args.gpus[i % len(args.gpus)]
            print(f"  [{j['name']}] GPU={gpu}")
            print(f"    task={j['task']}")
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
        print(f"  [{a['name']}] finished - {status}  ({elapsed:.0f}s)")

        if rc == 0:
            rl_ckpt = _extract_best_rl_checkpoint(a["log_path"])
            if rl_ckpt is not None:
                _update_csv(csv_path, a["row_idx"], args.output_col, rl_ckpt)
                print(f"  [{a['name']}] {args.output_col} -> {rl_ckpt}")
            else:
                print(f"  [{a['name']}] WARNING: could not find learned RL checkpoint in log/run dir")

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
            print("\n[interrupted] Stopping train sweep. CSV already updated for completed jobs.")
            break

        while pending and len(active) < max_concurrent:
            j = pending.pop(0)
            used_gpus = {a["gpu_id"] for a in active}
            available_gpus = sorted(g for g in args.gpus if g not in used_gpus)
            if not available_gpus:
                pending.insert(0, j)
                break
            gpu_id = available_gpus[0]
            j["gpu_id"] = gpu_id

            log_path = sweep_log_dir / f"{j['name']}.txt"
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            env["PYTHONUNBUFFERED"] = "1"

            if os.name != "nt":
                launch_cmd = [
                    "env",
                    f"CUDA_VISIBLE_DEVICES={gpu_id}",
                    "PYTHONUNBUFFERED=1",
                ] + j["cmd"]
            else:
                launch_cmd = j["cmd"]

            log_f = open(log_path, "w")
            print(f"  [{j['name']}] launching on GPU {gpu_id} task={j['task']}  ->  {log_path}")
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
    print(f"=== Train Sweep {'Interrupted' if interrupted else 'Complete'} ===")
    print(f"  Succeeded: {len(ok)}/{len(finished)}")
    if fail:
        print(f"  Failed:    {', '.join(f['name'] for f in fail)}")
    if still_running:
        print(f"  Still running (not updated in CSV): {', '.join(still_running)}")
    print(f"  Logs:      {sweep_log_dir}")
    print(f"  CSV updated: {csv_path} ({args.output_col} as last column)")


if __name__ == "__main__":
    main()
