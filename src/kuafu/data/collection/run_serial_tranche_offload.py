#!/usr/bin/env python3
"""Collect selected 100-demo tranche slots on one GPU and optionally assemble them."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from kuafu.paths import require_external_artifact_path
from typing import Any

import h5py


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
LAUNCHER = SCRIPT_DIR / "launch_collect_from_mpc.py"
COMBINER = SCRIPT_DIR.parent / "datasets" / "combine_hdf5.py"
ISAACLAB_SH = REPO_ROOT / "isaaclab.sh"
DEFAULT_MANIFEST = REPO_ROOT / "assets/configs" / "campaigns" / "mpc_first_two_programs_1600.json"


def checkout_env() -> dict[str, str]:
    """Prefer the Kuafu package belonging to this runner's checkout."""
    env = os.environ.copy()
    source_root = REPO_ROOT / "src"
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(source_root), existing] if existing else [str(source_root)]
    )
    return env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--task", required=True)
    parser.add_argument("--tranche-index", type=int, choices=[1, 2], required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cpu-threads-per-worker", type=int, default=4)
    parser.add_argument("--worker-stagger-sec", type=float, default=0.0)
    parser.add_argument(
        "--program-indices",
        type=int,
        nargs="+",
        choices=[1, 2],
        default=[1, 2],
        help="One-based learned-program indices to collect (default: both).",
    )
    parser.add_argument(
        "--worker-indices",
        type=int,
        nargs="+",
        choices=[0, 1, 2, 3],
        default=[0, 1, 2, 3],
        help="Seed-slot indices to collect (default: all four).",
    )
    parser.add_argument(
        "--slots-only",
        action="store_true",
        help="Collect/reuse the selected slots and exit without tranche assembly.",
    )
    parser.add_argument(
        "--wait-for-path",
        type=Path,
        default=None,
        help="Wait until this file or directory exists before starting collection.",
    )
    parser.add_argument(
        "--wait-poll-sec",
        type=float,
        default=30.0,
        help="Polling interval for --wait-for-path (default: 30 seconds).",
    )
    return parser.parse_args()


def demo_count(path: Path) -> int:
    if not path.is_file():
        return 0
    try:
        with h5py.File(path, "r") as dataset:
            data = dataset.get("data")
            if data is None:
                return 0
            return sum(name.startswith("demo_") for name in data.keys())
    except OSError:
        return 0


def require_count(path: Path, expected: int) -> None:
    actual = demo_count(path)
    if actual != expected:
        raise RuntimeError(f"Expected {expected} demos, got {actual}: {path}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_program(program: dict[str, Any]) -> Path:
    path = (REPO_ROOT / program["path"]).resolve()
    if not path.is_dir() or not (path / "program").is_dir():
        raise FileNotFoundError(f"Invalid learned-program path: {path}")
    if (path / "success.txt").read_text().strip().lower() != "true":
        raise RuntimeError(f"Learned program is not marked successful: {path}")
    if "status: succeeded" not in (path / "stats.txt").read_text():
        raise RuntimeError(f"Learned program did not succeed: {path}")
    return path


def run_logged(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    process: subprocess.Popen[Any] | None = None

    def handle_signal(signum: int, frame: Any) -> None:
        del signum, frame
        if process is not None and process.poll() is None:
            process.terminate()
        raise KeyboardInterrupt

    previous_sigint = signal.signal(signal.SIGINT, handle_signal)
    previous_sigterm = signal.signal(signal.SIGTERM, handle_signal)
    try:
        with log_path.open("w") as log_handle:
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=checkout_env(),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"Command failed ({return_code}); inspect {log_path}")
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)


def combine_slots(
    *,
    slot_paths: list[Path],
    output: Path,
    task: dict[str, Any],
    program_index: int,
    tranche_index: int,
    tranche_seed_base: int,
    log_path: Path,
) -> str:
    for path in slot_paths:
        require_count(path, 100)
    if output.exists():
        require_count(output, 400)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        partial = output.with_name(
            f".{output.stem}.partial.{os.getpid()}{output.suffix}"
        )
        command = [
            sys.executable,
            str(COMBINER),
            *(str(path) for path in slot_paths),
            "-o",
            str(partial),
            "--robot-name",
            task["robot_name"],
        ]
        try:
            run_logged(command, log_path)
            require_count(partial, 400)
            with h5py.File(partial, "a") as dataset:
                dataset.attrs["task_name"] = task["task_name"]
                dataset.attrs["task_slug"] = task["slug"]
                dataset.attrs["robot_name"] = task["robot_name"]
                dataset.attrs["learned_program_path"] = task["programs"][program_index]["path"]
                dataset.attrs["learned_program_index"] = program_index + 1
                dataset.attrs["tranche_index"] = tranche_index
                dataset.attrs["demo_count"] = 400
                dataset.attrs["base_seed"] = tranche_seed_base
                dataset.attrs["provenance_source"] = "serial_single_gpu_offload"
                dataset.attrs["worker_seed_bases_json"] = json.dumps(
                    [tranche_seed_base + worker for worker in range(4)]
                )
                dataset.attrs["source_slot_datasets_json"] = json.dumps(
                    [str(path) for path in slot_paths]
                )
                data = dataset.require_group("data")
                data.attrs["robot_name"] = task["robot_name"]
                data.attrs["total"] = 400
            os.replace(partial, output)
        finally:
            if partial.exists():
                partial.unlink()
    checksum = sha256_file(output)
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{checksum}  {output.name}\n"
    )
    return checksum


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.expanduser().resolve().read_text())
    collection = manifest["collection"]
    if (
        int(collection["successful_trajectories_per_tranche_per_program"]) != 400
        or int(collection["trajectories_per_worker"]) != 100
        or int(collection["seed_stride_per_tranche"]) != 1_000_000
    ):
        raise RuntimeError("Manifest does not describe the expected 4x100 tranche")
    task = next((item for item in manifest["tasks"] if item["slug"] == args.task), None)
    if task is None:
        raise ValueError(f"Unknown task {args.task!r}")
    if args.gpu < 0:
        raise ValueError("--gpu must be non-negative")
    if len(set(args.program_indices)) != len(args.program_indices):
        raise ValueError("--program-indices must not contain duplicates")
    if len(set(args.worker_indices)) != len(args.worker_indices):
        raise ValueError("--worker-indices must not contain duplicates")
    if args.wait_poll_sec <= 0:
        raise ValueError("--wait-poll-sec must be positive")
    if args.wait_for_path is not None:
        wait_path = args.wait_for_path.expanduser().resolve()
        print(f"[wait] waiting for path: {wait_path}", flush=True)
        last_report = 0.0
        while not wait_path.exists():
            now = time.monotonic()
            if now - last_report >= 300.0:
                print(f"[wait] still waiting for path: {wait_path}", flush=True)
                last_report = now
            time.sleep(min(args.wait_poll_sec, 60.0))
        print(f"[wait] path is ready: {wait_path}", flush=True)

    root = (
        require_external_artifact_path(args.output_root, "Serial tranche output root")
        / task["slug"]
        / "tranches"
        / f"tranche_{args.tranche_index:02d}"
    )
    root.mkdir(parents=True, exist_ok=True)
    seed_stride = int(collection["seed_stride_per_tranche"])
    for one_based_program_index in args.program_indices:
        program_index = one_based_program_index - 1
        program = task["programs"][program_index]
        program_path = validate_program(program)
        tranche_seed_base = int(program["base_seed"]) + (
            args.tranche_index - 1
        ) * seed_stride
        slot_paths = [
            root
            / "serial_slots"
            / f"program_{program_index + 1:02d}"
            / f"slot_{worker_index:02d}_100.hdf5"
            for worker_index in range(4)
        ]
        for worker_index in args.worker_indices:
            slot_path = (
                root
                / "serial_slots"
                / f"program_{program_index + 1:02d}"
                / f"slot_{worker_index:02d}_100.hdf5"
            )
            if slot_path.exists():
                require_count(slot_path, 100)
                print(f"[reuse] {slot_path}: 100 demos")
                continue
            attempt = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            run_root = (
                root
                / "serial_runs"
                / f"program_{program_index + 1:02d}_slot_{worker_index:02d}_{attempt}"
            )
            command = [
                str(ISAACLAB_SH),
                "-p",
                str(LAUNCHER),
                "--gpus",
                str(args.gpu),
                "--workers_per_gpu",
                "1",
                "--num_demos_per_worker",
                "100",
                "--cpu_threads_per_worker",
                str(args.cpu_threads_per_worker),
                "--output_root",
                str(run_root),
                "--task_name",
                task["task_name"],
                "--robot_name",
                task["robot_name"],
                "--base_seed",
                str(tranche_seed_base + worker_index),
                "--stagger_sec",
                str(args.worker_stagger_sec),
                "--compact_output",
                str(slot_path),
                "--no_bc_prep",
                "--program_path",
                str(program_path),
                "--control_output_mode",
                str(collection["control_output_mode"]),
                "--action_noise_std",
                str(collection["action_noise_std"]),
            ]
            print(
                f"[serial] {task['slug']} tranche {args.tranche_index} "
                f"program {program_index + 1} slot {worker_index} seed={tranche_seed_base + worker_index}"
            )
            run_logged(
                command,
                root
                / "serial_logs"
                / f"program_{program_index + 1:02d}_slot_{worker_index:02d}_{attempt}.log",
            )
            require_count(slot_path, 100)

        if args.slots_only:
            print(
                f"[slots-complete] {task['slug']} tranche {args.tranche_index} "
                f"program {program_index + 1} slots={args.worker_indices}"
            )
            continue

        program_output = root / f"program_{program_index + 1:02d}_400.hdf5"
        checksum = combine_slots(
            slot_paths=slot_paths,
            output=program_output,
            task=task,
            program_index=program_index,
            tranche_index=args.tranche_index,
            tranche_seed_base=tranche_seed_base,
            log_path=root / "serial_logs" / f"combine_program_{program_index + 1:02d}.log",
        )
        print(f"[program-complete] {program_output}: 400 demos sha256={checksum}")

    if args.slots_only:
        return

    program_outputs = [
        root / "program_01_400.hdf5",
        root / "program_02_400.hdf5",
    ]
    if not all(demo_count(path) == 400 for path in program_outputs):
        print(
            "[partial] Selected programs assembled, but the full 400+400 tranche "
            "is not complete yet."
        )
        return
    program_checksums = [sha256_file(path) for path in program_outputs]
    for path, checksum in zip(program_outputs, program_checksums, strict=True):
        path.with_suffix(path.suffix + ".sha256").write_text(
            f"{checksum}  {path.name}\n"
        )

    completion = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "task_name": task["task_name"],
        "task_slug": task["slug"],
        "robot_name": task["robot_name"],
        "tranche_index": args.tranche_index,
        "collection_mode": "serial_single_gpu_offload",
        "gpu": args.gpu,
        "program_demo_counts": [400, 400],
        "program_datasets": [str(path) for path in program_outputs],
        "program_sha256": program_checksums,
    }
    completion_path = root / "serial_offload_completion.json"
    partial = completion_path.with_name(
        f".{completion_path.name}.partial.{os.getpid()}"
    )
    try:
        partial.write_text(json.dumps(completion, indent=2, sort_keys=True) + "\n")
        os.replace(partial, completion_path)
    finally:
        if partial.exists():
            partial.unlink()
    print(f"[complete] {task['slug']} tranche {args.tranche_index}: 400+400")


if __name__ == "__main__":
    main()
