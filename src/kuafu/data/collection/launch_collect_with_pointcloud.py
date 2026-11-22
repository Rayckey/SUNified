#!/usr/bin/env python3
"""Launch multiple `collect_with_pointcloud.py` workers across one or more GPUs."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

COLLECT_SCRIPT = Path(__file__).resolve().parent / "collect_with_pointcloud.py"


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Launch parallel collect_with_pointcloud workers across GPUs."
    )
    parser.add_argument(
        "--gpus",
        type=int,
        nargs="+",
        required=True,
        help="GPU IDs to distribute workers across.",
    )
    parser.add_argument(
        "--workers_per_gpu",
        type=int,
        default=1,
        help="How many worker processes to launch on each GPU.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Optional total worker count override (default: len(gpus) * workers_per_gpu).",
    )
    parser.add_argument(
        "--num_demos_per_worker",
        type=int,
        default=200,
        help="How many demos each worker should collect.",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        required=True,
        help="Task name forwarded to each worker.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Only store every N-th simulation step in each worker (default: 1).",
    )
    parser.add_argument(
        "--stagger_sec",
        type=float,
        default=10.0,
        help="Seconds to wait between launching consecutive workers.",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python interpreter to use for worker processes.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="External shared output directory. Defaults below KUAFU_ARTIFACT_ROOT.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print worker commands without launching them.",
    )
    return parser.parse_known_args()


def build_output_root(args: argparse.Namespace) -> Path:
    repo_root = Path(__file__).resolve().parents[4]
    configured = os.environ.get("KUAFU_ARTIFACT_ROOT", "").strip()
    base = Path(configured).expanduser().resolve() if configured else repo_root.parent / "KuafuArtifacts"
    if args.output_root:
        output = Path(args.output_root).expanduser()
        if not output.is_absolute():
            output = base / output
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output = base / "standalone" / "pointcloud_collection" / args.task_name / timestamp
    resolved = output.resolve()
    if resolved == repo_root or repo_root in resolved.parents:
        raise ValueError(f"Collection output must be outside the source project ({repo_root}); got {resolved}")
    return resolved


def sanitize_worker_args(worker_args: list[str]) -> list[str]:
    # The launcher owns these arguments and injects worker-specific values itself.
    owned_flags = {
        "--worker_id",
        "--num_demos",
        "--device",
        "--task_name",
        "--stride",
        "--output_dir",
    }
    cleaned: list[str] = []
    i = 0
    while i < len(worker_args):
        arg = worker_args[i]
        key = arg.split("=", 1)[0]
        if key in owned_flags:
            if "=" not in arg and i + 1 < len(worker_args) and not worker_args[i + 1].startswith("--"):
                i += 2
            else:
                i += 1
            continue
        cleaned.append(arg)
        i += 1
    return cleaned


def main() -> None:
    args, worker_args = parse_args()
    worker_args = sanitize_worker_args(worker_args)

    default_num_workers = len(args.gpus) * args.workers_per_gpu
    num_workers = args.num_workers or default_num_workers
    if num_workers <= 0:
        raise ValueError("--num_workers must be positive.")

    output_root = build_output_root(args)
    logs_dir = output_root / "logs"

    jobs: list[dict] = []
    for worker_id in range(num_workers):
        gpu_id = args.gpus[worker_id % len(args.gpus)]
        cmd = [
            args.python,
            str(COLLECT_SCRIPT),
            "--task_name",
            args.task_name,
            "--worker_id",
            str(worker_id),
            "--num_demos",
            str(args.num_demos_per_worker),
            "--stride",
            str(args.stride),
            "--output_dir",
            str(output_root),
            "--device",
            "cuda:0",
            "--headless",
            *worker_args,
        ]
        jobs.append(
            {
                "worker_id": worker_id,
                "gpu_id": gpu_id,
                "cmd": cmd,
                "log_path": logs_dir / f"worker_{worker_id:02d}.txt",
            }
        )

    print("=== Pointcloud Collection Launch ===")
    print(f"  Script:           {COLLECT_SCRIPT}")
    print(f"  Task:             {args.task_name}")
    print(f"  GPUs:             {args.gpus}")
    print(f"  Workers:          {num_workers}")
    print(f"  Demos / worker:   {args.num_demos_per_worker}")
    print(f"  Stride:           {args.stride}")
    print(f"  Output root:      {output_root}")
    print(f"  Forwarded args:   {' '.join(worker_args) if worker_args else '(none)'}")
    print()

    if args.dry_run:
        for job in jobs:
            print(f"  [worker {job['worker_id']}] GPU={job['gpu_id']}")
            print(f"    {' '.join(job['cmd'])}")
            print()
        return

    logs_dir.mkdir(parents=True, exist_ok=True)

    active: list[dict] = []
    interrupted = False

    def _handle_interrupt(signum, frame):
        del signum, frame
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, _handle_interrupt)
    signal.signal(signal.SIGTERM, _handle_interrupt)

    for job in jobs:
        if interrupted:
            break
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["CUDA_VISIBLE_DEVICES"] = str(job["gpu_id"])
        log_file = open(job["log_path"], "w")
        print(f"  [worker {job['worker_id']}] launching on GPU {job['gpu_id']} -> {job['log_path']}")
        proc = subprocess.Popen(
            job["cmd"],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
        )
        job["proc"] = proc
        job["log_file"] = log_file
        job["start_time"] = time.time()
        active.append(job)
        time.sleep(args.stagger_sec)

    while active:
        still_running: list[dict] = []
        for job in active:
            rc = job["proc"].poll()
            if rc is None:
                still_running.append(job)
                continue
            job["log_file"].close()
            elapsed = time.time() - job["start_time"]
            status = "OK" if rc == 0 else f"FAILED (rc={rc})"
            print(f"  [worker {job['worker_id']}] finished - {status} ({elapsed:.0f}s)")
        active = still_running

        if interrupted and active:
            print("\n[interrupted] terminating remaining workers...")
            for job in active:
                job["proc"].terminate()
            break

        if active:
            time.sleep(5)

    for job in jobs:
        log_file = job.get("log_file")
        if log_file is not None and not log_file.closed:
            log_file.close()

    print()
    print(f"Logs:     {logs_dir}")
    if interrupted:
        print("Launch interrupted before all workers completed.")


if __name__ == "__main__":
    main()
