#!/usr/bin/env python3
"""Launch multiple `collect_with_checkpoint.py` workers across one or more GPUs."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

COLLECT_SCRIPT = Path(__file__).resolve().parent / "collect_with_checkpoint.py"

FATAL_LOG_PATTERNS = (
    "Error executing job with overrides:",
    "RuntimeError: CUDA out of memory",
    "torch.OutOfMemoryError",
    "Not enough GPU memory available to create a PhysicsScene",
    "PxgCudaDeviceMemoryAllocator failed",
    "Fatal Python error",
    "Segmentation fault",
)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be numeric, got {raw!r}.") from exc


def terminate_process_tree(proc: subprocess.Popen, grace_sec: float) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except Exception:
        proc.terminate()

    deadline = time.time() + grace_sec
    while time.time() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.25)

    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except Exception:
        proc.kill()


def scan_worker_log(job: dict) -> str | None:
    log_path = job["log_path"]
    try:
        stat = log_path.stat()
    except FileNotFoundError:
        return None

    if stat.st_size <= 0:
        return None

    job["last_log_mtime"] = max(job.get("last_log_mtime", 0.0), stat.st_mtime)
    offset = min(int(job.get("fatal_scan_offset", 0)), stat.st_size)
    if stat.st_size <= offset:
        return None

    try:
        with log_path.open("r", errors="replace") as file_handle:
            file_handle.seek(offset)
            chunk = file_handle.read()
            job["fatal_scan_offset"] = file_handle.tell()
    except OSError:
        return None

    for pattern in FATAL_LOG_PATTERNS:
        if pattern in chunk:
            return pattern
    return None


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Launch parallel collect_with_checkpoint workers across GPUs."
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
        "--num_demos_per_worker",
        type=int,
        default=200,
        help="How many demos each worker should collect.",
    )
    parser.add_argument(
        "--num_envs_per_worker",
        type=int,
        default=1,
        help="How many environments each worker should run.",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task name forwarded to each worker.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the .pth checkpoint file.",
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
    parser.add_argument(
        "--worker_timeout_sec",
        type=float,
        default=_env_float("COLLECT_WORKER_TIMEOUT_SEC", 0.0),
        help="Hard timeout per worker in seconds. Set 0 to disable.",
    )
    parser.add_argument(
        "--worker_log_idle_timeout_sec",
        type=float,
        default=_env_float("COLLECT_WORKER_LOG_IDLE_TIMEOUT_SEC", 0.0),
        help="Fail a worker if its log is unchanged for this many seconds. Set 0 to disable.",
    )
    parser.add_argument(
        "--worker_exit_grace_sec",
        type=float,
        default=_env_float("COLLECT_WORKER_EXIT_GRACE_SEC", 30.0),
        help="Seconds to wait after SIGTERM before SIGKILL when stopping a worker.",
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
        task_name = args.task.split(":")[-1]
        output = base / "standalone" / "checkpoint_collection" / task_name / timestamp
    resolved = output.resolve()
    if resolved == repo_root or repo_root in resolved.parents:
        raise ValueError(f"Collection output must be outside the source project ({repo_root}); got {resolved}")
    return resolved


def sanitize_worker_args(worker_args: list[str]) -> list[str]:
    # The launcher owns these arguments and injects worker-specific values itself.
    owned_flags = {
        "--worker_id",
        "--num_demos",
        "--num_envs",
        "--device",
        "--task",
        "--checkpoint",
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

    num_workers = len(args.gpus) * args.workers_per_gpu
    if num_workers <= 0:
        raise ValueError("Computed worker count must be positive. Check --gpus and --workers_per_gpu.")
    if args.num_envs_per_worker <= 0:
        raise ValueError("--num_envs_per_worker must be positive.")
    if args.worker_timeout_sec < 0:
        raise ValueError("--worker_timeout_sec must be nonnegative.")
    if args.worker_log_idle_timeout_sec < 0:
        raise ValueError("--worker_log_idle_timeout_sec must be nonnegative.")
    if args.worker_exit_grace_sec <= 0:
        raise ValueError("--worker_exit_grace_sec must be positive.")

    output_root = build_output_root(args)
    logs_dir = output_root / "logs"

    jobs: list[dict] = []
    for worker_id in range(num_workers):
        gpu_id = args.gpus[worker_id % len(args.gpus)]
        cmd = [
            args.python,
            str(COLLECT_SCRIPT),
            "--task",
            args.task,
            "--checkpoint",
            args.checkpoint,
            "--worker_id",
            str(worker_id),
            "--num_demos",
            str(args.num_demos_per_worker),
            "--num_envs",
            str(args.num_envs_per_worker),
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

    print("=== Checkpoint Collection Launch ===")
    print(f"  Script:           {COLLECT_SCRIPT}")
    print(f"  Task:             {args.task}")
    print(f"  Checkpoint:       {args.checkpoint}")
    print(f"  GPUs:             {args.gpus}")
    print(f"  Workers:          {num_workers}")
    print(f"  Demos / worker:   {args.num_demos_per_worker}")
    print(f"  Envs / worker:    {args.num_envs_per_worker}")
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
    failures = 0

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
            start_new_session=True,
        )
        job["proc"] = proc
        job["log_file"] = log_file
        job["start_time"] = time.time()
        job["last_log_mtime"] = time.time()
        job["fatal_scan_offset"] = 0
        active.append(job)
        time.sleep(args.stagger_sec)

    while active:
        still_running: list[dict] = []
        now = time.time()
        for job in active:
            proc = job["proc"]
            reason = scan_worker_log(job)
            if reason and "failure_reason" not in job:
                job["failure_reason"] = f"fatal log pattern: {reason}"
                if proc.poll() is None:
                    print(
                        f"  [worker {job['worker_id']}] {job['failure_reason']}; terminating worker",
                        flush=True,
                    )
                    terminate_process_tree(proc, args.worker_exit_grace_sec)

            elapsed = now - job["start_time"]
            idle = now - job.get("last_log_mtime", job["start_time"])
            if (
                proc.poll() is None
                and args.worker_timeout_sec > 0
                and elapsed > args.worker_timeout_sec
            ):
                job["failure_reason"] = f"worker timeout after {elapsed:.0f}s"
                print(
                    f"  [worker {job['worker_id']}] {job['failure_reason']}; terminating worker",
                    flush=True,
                )
                terminate_process_tree(proc, args.worker_exit_grace_sec)

            if (
                proc.poll() is None
                and args.worker_log_idle_timeout_sec > 0
                and idle > args.worker_log_idle_timeout_sec
            ):
                job["failure_reason"] = f"worker log idle for {idle:.0f}s"
                print(
                    f"  [worker {job['worker_id']}] {job['failure_reason']}; terminating worker",
                    flush=True,
                )
                terminate_process_tree(proc, args.worker_exit_grace_sec)

            rc = proc.poll()
            if rc is None:
                still_running.append(job)
                continue

            job["log_file"].close()
            elapsed = time.time() - job["start_time"]
            failure_reason = job.get("failure_reason")
            if rc != 0 or failure_reason:
                failures += 1
                status = f"FAILED (rc={rc})"
                if failure_reason:
                    status = f"{status}; {failure_reason}"
            else:
                status = "OK"
            print(f"  [worker {job['worker_id']}] finished - {status} ({elapsed:.0f}s)")
        active = still_running

        if failures and active:
            print("\n[failure] terminating remaining workers...")
            for job in active:
                terminate_process_tree(job["proc"], args.worker_exit_grace_sec)
            active = []
            break

        if interrupted and active:
            print("\n[interrupted] terminating remaining workers...")
            for job in active:
                terminate_process_tree(job["proc"], args.worker_exit_grace_sec)
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
        raise SystemExit(130)
    if failures:
        print(f"{failures} worker(s) failed.", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
