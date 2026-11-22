#!/usr/bin/env python3
"""Run the first-two-new-program MPC800 campaign one task at a time."""

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

import h5py


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
DEFAULT_MANIFEST = REPO_ROOT / "assets/configs" / "campaigns" / "mpc_first_two_programs_800.json"
LAUNCHER = SCRIPT_DIR / "launch_collect_from_mpc.py"
COMBINER = SCRIPT_DIR.parent / "datasets" / "combine_hdf5.py"
ISAACLAB_SH = REPO_ROOT / "isaaclab.sh"

EXPECTED_TASKS = {
    "drawer": ("Kinova-Table-Gen3-Drawer", "kinova"),
    "drawer_stack": ("Kinova-Table-Gen3-Drawer-Stack", "kinova"),
    "can": ("Kinova-Table-Gen3-Can", "kinova"),
    "hang_cup": ("Kinova-Table-Gen3-Hang-Cup", "kinova"),
    "pen_cup": ("Kinova-Table-Gen3-Pen-Cup", "kinova"),
    "stack2": ("FR3-FinRay-Table-Stack-2", "fr3_finray"),
    "stack3": ("FR3-FinRay-Table-Stack-3", "fr3_finray"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--start-task", type=str, default=None)
    parser.add_argument("--only-task", type=str, default=None)
    parser.add_argument("--worker-stagger-sec", type=float, default=8.0)
    parser.add_argument("--program-start-offset-sec", type=float, default=4.0)
    parser.add_argument("--cpu-threads-per-worker", type=int, default=8)
    parser.add_argument(
        "--gpus",
        type=int,
        nargs="+",
        default=None,
        help="GPU assignment used for both programs. Repeated IDs are allowed.",
    )
    parser.add_argument(
        "--program-1-gpus",
        type=int,
        nargs="+",
        default=None,
        help="Optional GPU assignment override for learned program 1.",
    )
    parser.add_argument(
        "--program-2-gpus",
        type=int,
        nargs="+",
        default=None,
        help="Optional GPU assignment override for learned program 2.",
    )
    parser.add_argument("--keep-intermediates", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def count_demos(path: Path) -> int:
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_program(program: dict) -> Path:
    path = (REPO_ROOT / program["path"]).resolve()
    if not path.is_dir() or not (path / "program").is_dir():
        raise FileNotFoundError(f"Invalid learned-program path: {path}")
    success = (path / "success.txt").read_text().strip().lower()
    if success != "true":
        raise RuntimeError(f"Learned program is not marked successful: {path}")
    stats = (path / "stats.txt").read_text()
    if "status: succeeded" not in stats:
        raise RuntimeError(f"Learned program does not have succeeded status: {path}")
    return path


def validate_task(task: dict) -> None:
    slug = task.get("slug")
    expected = EXPECTED_TASKS.get(slug)
    if expected is None:
        raise ValueError(f"Unexpected campaign task slug: {slug!r}")
    expected_task_name, expected_robot_name = expected
    actual = (task.get("task_name"), task.get("robot_name"))
    if actual != expected:
        raise ValueError(
            f"Wrong robot/task pairing for {slug}: got {actual}, expected {expected}"
        )


def validate_gpu_assignment(gpu_ids: list[int], collection: dict, label: str) -> None:
    if not gpu_ids or any(gpu_id < 0 for gpu_id in gpu_ids):
        raise ValueError(f"{label} must contain non-negative GPU IDs: {gpu_ids}")
    workers_per_gpu = int(collection["workers_per_gpu_per_program"])
    demos_per_worker = int(collection["trajectories_per_worker"])
    expected_demos = int(collection["successful_trajectories_per_program"])
    assigned_demos = len(gpu_ids) * workers_per_gpu * demos_per_worker
    if assigned_demos != expected_demos:
        raise ValueError(
            f"{label} assigns {assigned_demos} successful demos, expected "
            f"{expected_demos}: gpus={gpu_ids}, workers_per_gpu={workers_per_gpu}, "
            f"demos_per_worker={demos_per_worker}"
        )


def build_launch_command(
    *,
    task: dict,
    program: dict,
    program_index: int,
    program_root: Path,
    compact_output: Path,
    gpu_ids: list[int],
    collection: dict,
    worker_stagger_sec: float,
    cpu_threads_per_worker: int,
) -> list[str]:
    program_path = validate_program(program)
    return [
        str(ISAACLAB_SH),
        "-p",
        str(LAUNCHER),
        "--gpus",
        *(str(gpu_id) for gpu_id in gpu_ids),
        "--workers_per_gpu",
        str(collection["workers_per_gpu_per_program"]),
        "--num_demos_per_worker",
        str(collection["trajectories_per_worker"]),
        "--cpu_threads_per_worker",
        str(cpu_threads_per_worker),
        "--output_root",
        str(program_root),
        "--task_name",
        task["task_name"],
        "--robot_name",
        task["robot_name"],
        "--base_seed",
        str(program["base_seed"]),
        "--stagger_sec",
        str(worker_stagger_sec),
        "--compact_output",
        str(compact_output),
        "--no_bc_prep",
        "--program_path",
        str(program_path),
        "--control_output_mode",
        str(collection["control_output_mode"]),
        "--action_noise_std",
        str(collection["action_noise_std"]),
    ]


def terminate_launchers(processes: list[subprocess.Popen]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        if process.poll() is None:
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()


def run_launchers(
    commands: list[tuple[list[str], Path]],
    *,
    program_start_offset_sec: float,
    dry_run: bool,
) -> None:
    if dry_run:
        for command, log_path in commands:
            print(f"[dry-run] log={log_path}")
            print(" ".join(command))
        return

    processes: list[subprocess.Popen] = []
    log_handles = []

    def handle_signal(signum, frame):
        del signum, frame
        terminate_launchers(processes)
        raise KeyboardInterrupt

    previous_sigint = signal.signal(signal.SIGINT, handle_signal)
    previous_sigterm = signal.signal(signal.SIGTERM, handle_signal)
    try:
        for index, (command, log_path) in enumerate(commands):
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = log_path.open("w")
            log_handles.append(log_handle)
            print(f"[launch] {command[command.index('--program_path') + 1]}")
            print(f"[launch] log={log_path}")
            processes.append(
                subprocess.Popen(
                    command,
                    cwd=REPO_ROOT,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                )
            )
            if index + 1 < len(commands):
                time.sleep(program_start_offset_sec)

        return_codes = [process.wait() for process in processes]
        if any(code != 0 for code in return_codes):
            raise RuntimeError(f"MPC launchers failed with exit codes {return_codes}")
    finally:
        terminate_launchers(processes)
        for log_handle in log_handles:
            log_handle.close()
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)


def compact_task(
    *, task: dict, program_outputs: list[Path], final_output: Path, campaign: str
) -> tuple[str, int]:
    counts = [count_demos(path) for path in program_outputs]
    if counts != [400, 400]:
        raise RuntimeError(
            f"Expected two 400-demo program datasets for {task['slug']}, got {counts}"
        )

    if count_demos(final_output) == 800:
        checksum = sha256_file(final_output)
        return checksum, 800

    final_output.parent.mkdir(parents=True, exist_ok=True)
    partial_output = final_output.with_name(
        f".{final_output.stem}.partial.{os.getpid()}{final_output.suffix}"
    )
    command = [
        sys.executable,
        str(COMBINER),
        *(str(path) for path in program_outputs),
        "-o",
        str(partial_output),
        "--robot-name",
        task["robot_name"],
    ]
    try:
        subprocess.run(command, cwd=REPO_ROOT, check=True)
        final_count = count_demos(partial_output)
        if final_count != 800:
            raise RuntimeError(
                f"Combined dataset has {final_count} demos, expected 800: {partial_output}"
            )
        with h5py.File(partial_output, "a") as dataset:
            dataset.attrs["campaign"] = campaign
            dataset.attrs["task_name"] = task["task_name"]
            dataset.attrs["task_slug"] = task["slug"]
            dataset.attrs["robot_name"] = task["robot_name"]
            dataset.attrs["learned_program_paths_json"] = json.dumps(
                [program["path"] for program in task["programs"]]
            )
            dataset.attrs["demos_per_learned_program_json"] = json.dumps([400, 400])
        checksum = sha256_file(partial_output)
        os.replace(partial_output, final_output)
    finally:
        if partial_output.exists():
            partial_output.unlink()

    checksum_path = final_output.with_suffix(final_output.suffix + ".sha256")
    checksum_path.write_text(f"{checksum}  {final_output.name}\n")
    return checksum, 800


def clean_intermediates(program_roots: list[Path], program_outputs: list[Path]) -> list[str]:
    removed: list[str] = []
    for program_root in program_roots:
        for path in sorted((program_root / "datasets").glob("mpc_demos_worker*.hdf5")):
            path.unlink()
            removed.append(str(path))
    for path in program_outputs:
        if path.exists():
            path.unlink()
            removed.append(str(path))
    return removed


def task_paths(output_root: Path, task: dict) -> tuple[Path, list[Path], list[Path], Path]:
    task_root = output_root / task["slug"]
    program_roots = [
        task_root / f"program_{index + 1:02d}_{Path(program['path']).name}"
        for index, program in enumerate(task["programs"])
    ]
    program_outputs = [
        task_root / f"program_{index + 1:02d}_400.hdf5" for index in range(2)
    ]
    final_output = task_root / (
        f"{task['robot_name']}_{task['slug']}_first2_programs_mpc800.hdf5"
    )
    return task_root, program_roots, program_outputs, final_output


def select_tasks(tasks: list[dict], start_task: str | None, only_task: str | None) -> list[dict]:
    slugs = [task["slug"] for task in tasks]
    if only_task is not None:
        if only_task not in slugs:
            raise ValueError(f"Unknown --only-task {only_task!r}; choices: {slugs}")
        return [tasks[slugs.index(only_task)]]
    if start_task is not None:
        if start_task not in slugs:
            raise ValueError(f"Unknown --start-task {start_task!r}; choices: {slugs}")
        return tasks[slugs.index(start_task) :]
    return tasks


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    collection = manifest["collection"]
    output_root = require_external_artifact_path(
        args.output_root if args.output_root is not None else manifest["output_root"],
        "MPC-800 campaign output root",
    )
    tasks = select_tasks(manifest["tasks"], args.start_task, args.only_task)

    common_gpus = list(args.gpus or collection["gpus"])
    program_gpu_ids = [
        list(args.program_1_gpus or common_gpus),
        list(args.program_2_gpus or common_gpus),
    ]
    for index, gpu_ids in enumerate(program_gpu_ids, start=1):
        validate_gpu_assignment(gpu_ids, collection, f"program {index} GPU assignment")

    for task in tasks:
        validate_task(task)
        task_root, program_roots, program_outputs, final_output = task_paths(output_root, task)
        final_count = count_demos(final_output)
        if final_count == 800:
            print(f"[skip] {task['slug']}: verified 800-demo dataset {final_output}")
            continue
        if final_count not in (0, 800):
            raise RuntimeError(
                f"Refusing to overwrite incomplete final dataset ({final_count} demos): {final_output}"
            )

        task_root.mkdir(parents=True, exist_ok=True)
        commands: list[tuple[list[str], Path]] = []
        for index, (program, program_root, program_output) in enumerate(
            zip(task["programs"], program_roots, program_outputs, strict=True)
        ):
            if count_demos(program_output) == 400:
                print(f"[reuse] {program_output}: verified 400 demos")
                continue
            command = build_launch_command(
                task=task,
                program=program,
                program_index=index,
                program_root=program_root,
                compact_output=program_output,
                gpu_ids=program_gpu_ids[index],
                collection=collection,
                worker_stagger_sec=args.worker_stagger_sec,
                cpu_threads_per_worker=args.cpu_threads_per_worker,
            )
            commands.append((command, task_root / f"launcher_program{index + 1:02d}.log"))

        print(f"[task] {task['slug']} ({task['task_name']})")
        run_launchers(
            commands,
            program_start_offset_sec=args.program_start_offset_sec,
            dry_run=args.dry_run,
        )
        if args.dry_run:
            continue

        checksum, final_count = compact_task(
            task=task,
            program_outputs=program_outputs,
            final_output=final_output,
            campaign=manifest["campaign"],
        )
        removed = []
        if not args.keep_intermediates:
            removed = clean_intermediates(program_roots, program_outputs)
        completion = {
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "task_name": task["task_name"],
            "task_slug": task["slug"],
            "robot_name": task["robot_name"],
            "dataset": str(final_output),
            "demo_count": final_count,
            "sha256": checksum,
            "learned_program_paths": [program["path"] for program in task["programs"]],
            "removed_intermediates": removed,
        }
        (task_root / "completion.json").write_text(
            json.dumps(completion, indent=2, sort_keys=True) + "\n"
        )
        print(f"[complete] {task['slug']}: {final_count} demos sha256={checksum}")


if __name__ == "__main__":
    main()
