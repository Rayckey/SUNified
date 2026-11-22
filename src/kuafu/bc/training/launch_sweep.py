#!/usr/bin/env python3
"""Run a parameter sweep of BC prior training jobs from a CSV config.

Reads a CSV where each row is one training run, assigns each job to the
least-loaded GPU at launch time, and runs up to N concurrent jobs at a time.

Each job's stdout streams to a human-readable .txt log (line-buffered).
When a job finishes, the CSV is updated in-place with the best checkpoint path.

Usage:
    # Run all rows in sweep.csv on GPUs 0 and 1, 2 jobs at a time
    python src/kuafu/bc/training/launch_sweep.py --csv src/kuafu/bc/training/sweep.csv --gpus 0 1

    # 4 concurrent jobs across 4 GPUs
    python src/kuafu/bc/training/launch_sweep.py --csv sweep.csv --gpus 0 1 2 3 --workers_per_gpu 2

    # Dry-run: print commands without executing
    python src/kuafu/bc/training/launch_sweep.py --csv sweep.csv --gpus 0 --dry_run

    # Only run specific rows by name
    python src/kuafu/bc/training/launch_sweep.py --csv sweep.csv --gpus 0 1 --only baseline,chunk4
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

TRAIN_SCRIPT = Path(__file__).resolve().parent / "train_bc_prior_mlp.py"
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


CSV_TO_ARG = {
    "config":               "--config",
}

BEST_CKPT_COL = "best_checkpoint"
PRIOR_PATH_COL = "prior_path"


def _resolve_config_path(val: str, *, csv_path: Path) -> Path:
    """Resolve config YAML path for sweep rows.

    CSV paths are often relative to the repo root (e.g. ``src/kuafu/data/im_yaml/...``),
    while the sweep CSV lives under ``src/kuafu/bc/training/``. Try ``cwd`` first, then the
    CSV's directory, so both conventions work.
    """
    p = Path(val).expanduser()
    if p.is_absolute():
        out = p.resolve()
        if not out.exists():
            raise ValueError(f"Config YAML not found for sweep row: {val}")
        return out
    for base in (Path.cwd(), csv_path.parent):
        candidate = (base / p).resolve()
        if candidate.exists():
            return candidate
    raise ValueError(
        f"Config YAML not found for sweep row: {val} "
        f"(tried cwd={Path.cwd()} and csv_dir={csv_path.parent})"
    )


def _row_to_cmd(row: dict[str, str], python: str, *, csv_path: Path) -> list[str]:
    cmd = [python, str(TRAIN_SCRIPT)]
    for csv_col, cli_arg in CSV_TO_ARG.items():
        if csv_col not in row:
            continue
        val = row[csv_col].strip()
        if val:
            if csv_col == "config":
                val = str(_resolve_config_path(val, csv_path=csv_path))
            cmd.extend([cli_arg, val])
    if "--config" not in cmd:
        raise ValueError("Each sweep row must provide a 'config' column pointing to a YAML file.")
    return cmd


def _extract_best_checkpoint(log_path: Path) -> str | None:
    """Parse the job log to find 'Run dir: <path>' and return the relative best.pth path."""
    try:
        text = log_path.read_text()
    except OSError:
        return None
    match = re.search(r"Run dir:\s*(.+)", text)
    if not match:
        return None
    run_dir = Path(match.group(1).strip())
    best_pth = run_dir / "checkpoints" / "best.pth"
    if best_pth.exists():
        try:
            return str(best_pth.relative_to(Path.cwd()))
        except ValueError:
            return str(best_pth)
    return None


def _update_csv(csv_path: Path, row_idx: int, best_ckpt: str) -> None:
    """Update a single row's best_checkpoint column in the CSV file.

    Writes atomically: serialize to a sibling temp file then ``os.replace`` it
    onto the original. The original CSV is never truncated unless the new file
    has been fully written, so a crash mid-write cannot wipe the sweep config.
    Also defensively strips the ``None`` key that ``csv.DictReader`` produces
    when a row has more columns than the header (e.g. a stray trailing comma),
    which would otherwise make ``DictWriter`` raise.
    """
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header row: {csv_path}")
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    for row in rows:
        row.pop(None, None)

    if BEST_CKPT_COL not in fieldnames:
        fieldnames.append(BEST_CKPT_COL)
    if PRIOR_PATH_COL not in fieldnames:
        fieldnames.append(PRIOR_PATH_COL)

    if 0 <= row_idx < len(rows):
        rows[row_idx][BEST_CKPT_COL] = best_ckpt
        rows[row_idx][PRIOR_PATH_COL] = best_ckpt

    tmp_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
    try:
        with open(tmp_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp_path, csv_path)
    except Exception:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Launch BC prior training sweep from CSV.")
    p.add_argument("--csv", type=str, required=True, help="Path to CSV config file.")
    p.add_argument("--gpus", type=int, nargs="+", default=[0], help="GPU IDs to distribute across.")
    p.add_argument(
        "--workers_per_gpu",
        type=int,
        default=2,
        help="Number of concurrent training jobs allowed per GPU.",
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
        help="Relaunch rows even if they already have a non-empty best_checkpoint in the CSV.",
    )
    p.add_argument(
        "--stagger_sec",
        type=float,
        default=3.0,
        help="Seconds to wait between launching consecutive jobs.",
    )
    p.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python interpreter to use (default: current interpreter).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
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
        if reader.fieldnames is None:
            print(f"[ERROR] CSV has no header row: {csv_path}")
            sys.exit(1)
        rows = list(reader)

    if not rows:
        print("[ERROR] CSV has no data rows.")
        sys.exit(1)

    only_set: set[str] | None = None
    if args.only:
        only_set = {n.strip() for n in args.only.split(",")}

    jobs: list[dict] = []
    for i, row in enumerate(rows):
        raw_name = row["name"] if "name" in row and row["name"] is not None else ""
        name = raw_name.strip() if isinstance(raw_name, str) else str(raw_name).strip()
        if not name:
            name = f"run_{i}"
        if only_set and name not in only_set:
            continue
        has_best_ckpt = False
        if BEST_CKPT_COL in row and row[BEST_CKPT_COL] is not None:
            checkpoint = str(row[BEST_CKPT_COL]).strip()
            has_best_ckpt = bool(checkpoint) and Path(checkpoint).expanduser().is_file()
        if not args.overwrite_existing and has_best_ckpt:
            print(f"  [SKIP] {name}: already has best_checkpoint")
            continue
        cmd = _row_to_cmd(row, args.python, csv_path=csv_path)
        jobs.append({"name": name, "cmd": cmd, "row_idx": i})

    if not jobs:
        print("[WARN] No matching jobs found. Check --only filter or CSV content.")
        sys.exit(0)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    sweep_log_dir = _artifact_log_root() / "train" / timestamp
    if not args.dry_run:
        sweep_log_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== BC Prior Training Sweep ===")
    print(f"  CSV:            {csv_path}")
    print(f"  Total jobs:     {len(jobs)}")
    print(f"  GPUs:           {args.gpus}")
    print(f"  Workers / GPU:  {args.workers_per_gpu}")
    print(f"  Worker slots:   {worker_slots}")
    print(f"  Sweep logs:     {sweep_log_dir}")
    print()

    if args.dry_run:
        preview_counts = {gpu: 0 for gpu in args.gpus}
        for j in jobs:
            gpu = min(args.gpus, key=lambda g: (preview_counts[g], args.gpus.index(g)))
            preview_counts[gpu] += 1
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
            best = _extract_best_checkpoint(a["log_path"])
            if best:
                _update_csv(csv_path, a["row_idx"], best)
                print(f"  [{a['name']}] best checkpoint -> {best}")
            else:
                a["returncode"] = 1
                print(f"  [{a['name']}] ERROR: could not find best checkpoint in log; marking job failed")

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

    def _select_gpu() -> int:
        active_counts = {gpu: 0 for gpu in args.gpus}
        for a in active:
            active_counts[a["gpu_id"]] = active_counts.get(a["gpu_id"], 0) + 1
        available = [gpu for gpu in args.gpus if active_counts.get(gpu, 0) < args.workers_per_gpu]
        if not available:
            raise RuntimeError("No GPU worker slot available despite pending launch.")
        return min(available, key=lambda g: (active_counts[g], args.gpus.index(g)))

    while pending or active:
        _poll_active()

        if interrupted:
            print("\n[interrupted] Stopping sweep. CSV already updated for completed jobs.")
            break

        while pending and len(active) < worker_slots:
            j = pending.pop(0)
            gpu_id = _select_gpu()
            j["gpu_id"] = gpu_id
            log_path = sweep_log_dir / f"{j['name']}.txt"

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            env["PYTHONUNBUFFERED"] = "1"

            log_f = open(log_path, "w")
            print(f"  [{j['name']}] launching on GPU {gpu_id}  ->  {log_path}")
            proc = subprocess.Popen(
                j["cmd"],
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
    print(f"=== Sweep {'Interrupted' if interrupted else 'Complete'} ===")
    print(f"  Succeeded: {len(ok)}/{len(finished)}")
    if fail:
        print(f"  Failed:    {', '.join(f['name'] for f in fail)}")
    if still_running:
        print(f"  Still running (not updated in CSV): {', '.join(still_running)}")
    print(f"  Logs:      {sweep_log_dir}")
    print(f"  CSV updated: {csv_path}")

    if interrupted:
        sys.exit(130)
    if fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
