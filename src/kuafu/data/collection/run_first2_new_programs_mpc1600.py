#!/usr/bin/env python3
"""Collect 800 MPC successes from each of two learned programs per task."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
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
DEFAULT_MANIFEST = REPO_ROOT / "assets/configs" / "campaigns" / "mpc_first_two_programs_1600.json"
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


def checkout_env() -> dict[str, str]:
    """Prefer the Kuafu package belonging to this runner's checkout."""
    env = os.environ.copy()
    source_root = REPO_ROOT / "src"
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(source_root), existing] if existing else [str(source_root)]
    )
    # Detached queue launchers do not inherit an activated Conda shell.  IsaacLab's
    # wrapper selects its Python from CONDA_PREFIX, so bind it explicitly to the
    # interpreter that is already running this audited collector.
    runtime_prefix = Path(sys.prefix)
    runtime_bin = runtime_prefix / "bin"
    runtime_python = runtime_bin / "python"
    if not runtime_python.is_file():
        raise FileNotFoundError(
            f"Collector runtime lacks the expected Python executable: {runtime_python}"
        )
    env["CONDA_PREFIX"] = str(runtime_prefix)
    existing_path = env.get("PATH")
    env["PATH"] = os.pathsep.join(
        [str(runtime_bin), existing_path] if existing_path else [str(runtime_bin)]
    )
    return env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--start-task", type=str, default=None)
    parser.add_argument("--only-task", type=str, default=None)
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Explicit ordered task slugs for a server-specific queue.",
    )
    parser.add_argument(
        "--wait-for-pid",
        type=int,
        default=None,
        help="Wait for an older tranche-1 runner PID before touching task outputs.",
    )
    parser.add_argument(
        "--wait-for-path",
        type=Path,
        action="append",
        default=[],
        help=(
            "After --wait-for-pid, wait for this file or directory before "
            "processing tasks. Repeat the option to require multiple paths."
        ),
    )
    parser.add_argument("--wait-poll-sec", type=float, default=30.0)
    parser.add_argument(
        "--precollect-tranche",
        type=int,
        choices=[1, 2],
        default=None,
        help=(
            "Collect this tranche for the first selected task before waiting for "
            "--wait-for-pid; useful for safely filling GPU headroom."
        ),
    )
    parser.add_argument(
        "--precollect-only",
        action="store_true",
        help=(
            "Collect --precollect-tranche for the first selected task and exit. "
            "This is intended for bounded cross-server tranche offloads."
        ),
    )
    parser.add_argument("--worker-stagger-sec", type=float, default=8.0)
    parser.add_argument("--program-start-offset-sec", type=float, default=4.0)
    parser.add_argument("--cpu-threads-per-worker", type=int, default=8)
    parser.add_argument(
        "--import-legacy-only",
        action="store_true",
        help="Normalize a validated 400+400 legacy tranche and stop before new collection.",
    )
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
    parser.add_argument(
        "--precollect-program-1-gpus",
        type=int,
        nargs="+",
        default=None,
        help="GPU assignment override used only by --precollect-tranche for program 1.",
    )
    parser.add_argument(
        "--precollect-program-2-gpus",
        type=int,
        nargs="+",
        default=None,
        help="GPU assignment override used only by --precollect-tranche for program 2.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_checksum(path: Path) -> str:
    checksum = sha256_file(path)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{checksum}  {path.name}\n"
    )
    return checksum


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial.{os.getpid()}")
    try:
        partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(partial, path)
    finally:
        if partial.exists():
            partial.unlink()


def numbered_demo_names(dataset: h5py.File) -> list[str]:
    data = dataset.get("data")
    if data is None:
        return []
    names: list[tuple[int, str]] = []
    for name in data.keys():
        match = re.fullmatch(r"demo_(\d+)", name)
        if match:
            names.append((int(match.group(1)), name))
    names.sort()
    return [name for _, name in names]


def count_demos(path: Path) -> int:
    if not path.is_file():
        return 0
    try:
        with h5py.File(path, "r") as dataset:
            return len(numbered_demo_names(dataset))
    except OSError:
        return 0


def require_count(path: Path, expected: int, label: str) -> None:
    actual = count_demos(path)
    if actual != expected:
        raise RuntimeError(f"{label} has {actual} demos, expected {expected}: {path}")


def decoded_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass
    return value


def require_scalar_attr(dataset: h5py.File, key: str, expected: Any, path: Path) -> None:
    if key not in dataset.attrs:
        raise RuntimeError(f"Missing required HDF5 attribute {key!r}: {path}")
    actual = decoded_attr(dataset.attrs[key])
    if actual != expected:
        raise RuntimeError(
            f"Wrong HDF5 attribute {key!r} in {path}: got {actual!r}, expected {expected!r}"
        )


def require_json_attr(dataset: h5py.File, key: str, expected: Any, path: Path) -> None:
    if key not in dataset.attrs:
        raise RuntimeError(f"Missing required HDF5 attribute {key!r}: {path}")
    raw = decoded_attr(dataset.attrs[key])
    try:
        actual = json.loads(str(raw))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON HDF5 attribute {key!r} in {path}") from exc
    if actual != expected:
        raise RuntimeError(
            f"Wrong HDF5 attribute {key!r} in {path}: got {actual!r}, expected {expected!r}"
        )


def learned_program_paths(task: dict[str, Any]) -> list[str]:
    return [str(program["path"]) for program in task["programs"]]


def validate_legacy_mixed_tranche(
    path: Path, task: dict[str, Any], per_program_tranche: int
) -> None:
    expected_total = 2 * per_program_tranche
    require_count(path, expected_total, "legacy mixed tranche")
    with h5py.File(path, "r") as dataset:
        require_scalar_attr(dataset, "task_name", task["task_name"], path)
        require_scalar_attr(dataset, "task_slug", task["slug"], path)
        require_scalar_attr(dataset, "robot_name", task["robot_name"], path)
        require_json_attr(dataset, "learned_program_paths_json", learned_program_paths(task), path)
        require_json_attr(
            dataset,
            "demos_per_learned_program_json",
            [per_program_tranche, per_program_tranche],
            path,
        )
        names = numbered_demo_names(dataset)
        numeric_ids = [int(name.removeprefix("demo_")) for name in names]
        if numeric_ids != list(range(expected_total)):
            raise RuntimeError(
                f"Legacy mixed tranche does not have contiguous demo IDs 0..{expected_total - 1}: {path}"
            )


def validate_program(program: dict[str, Any]) -> Path:
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


def validate_task(task: dict[str, Any]) -> None:
    slug = task.get("slug")
    expected = EXPECTED_TASKS.get(slug)
    if expected is None:
        raise ValueError(f"Unexpected campaign task slug: {slug!r}")
    actual = (task.get("task_name"), task.get("robot_name"))
    if actual != expected:
        raise ValueError(
            f"Wrong robot/task pairing for {slug}: got {actual}, expected {expected}"
        )
    if len(task.get("programs", [])) != 2:
        raise ValueError(f"Task {slug} must contain exactly two learned programs")


def validate_manifest(manifest: dict[str, Any]) -> None:
    collection = manifest["collection"]
    per_task = int(collection["successful_trajectories_per_task"])
    per_program = int(collection["successful_trajectories_per_program"])
    tranches = int(collection["tranches_per_program"])
    per_program_tranche = int(
        collection["successful_trajectories_per_tranche_per_program"]
    )
    if (per_task, per_program, tranches, per_program_tranche) != (1600, 800, 2, 400):
        raise ValueError(
            "Campaign must specify exactly 1,600/task = two programs x 800, "
            "collected as two 400-demo tranches per program"
        )
    if per_task != 2 * per_program or per_program != tranches * per_program_tranche:
        raise ValueError("Inconsistent task/program/tranche trajectory totals")
    for task in manifest["tasks"]:
        validate_task(task)


def validate_gpu_assignment(
    gpu_ids: list[int], collection: dict[str, Any], label: str
) -> None:
    if not gpu_ids or any(gpu_id < 0 for gpu_id in gpu_ids):
        raise ValueError(f"{label} must contain non-negative GPU IDs: {gpu_ids}")
    workers_per_gpu = int(collection["workers_per_gpu_per_program"])
    demos_per_worker = int(collection["trajectories_per_worker"])
    expected = int(collection["successful_trajectories_per_tranche_per_program"])
    assigned = len(gpu_ids) * workers_per_gpu * demos_per_worker
    if assigned != expected:
        raise ValueError(
            f"{label} assigns {assigned} demos per tranche, expected {expected}: "
            f"gpus={gpu_ids}, workers_per_gpu={workers_per_gpu}, "
            f"demos_per_worker={demos_per_worker}"
        )


def task_paths(output_root: Path, task: dict[str, Any]) -> dict[str, Path]:
    task_root = output_root / task["slug"]
    return {
        "root": task_root,
        "legacy_mixed": task_root
        / f"{task['robot_name']}_{task['slug']}_first2_programs_mpc800.hdf5",
        "program_1_800": task_root / "program_01_800.hdf5",
        "program_2_800": task_root / "program_02_800.hdf5",
        "final": task_root
        / f"{task['robot_name']}_{task['slug']}_first2_programs_mpc1600.hdf5",
    }


def tranche_root(task_root: Path, tranche_index: int) -> Path:
    return task_root / "tranches" / f"tranche_{tranche_index:02d}"


def tranche_program_paths(task_root: Path, tranche_index: int) -> list[Path]:
    root = tranche_root(task_root, tranche_index)
    return [root / "program_01_400.hdf5", root / "program_02_400.hdf5"]


def seed_base(
    program: dict[str, Any], tranche_index: int, collection: dict[str, Any]
) -> int:
    return int(program["base_seed"]) + (
        tranche_index - 1
    ) * int(collection["seed_stride_per_tranche"])


def ensure_program_metadata(
    path: Path,
    *,
    task: dict[str, Any],
    program_index: int,
    tranche_index: int,
    collection: dict[str, Any],
    source: str,
) -> str:
    expected = int(collection["successful_trajectories_per_tranche_per_program"])
    require_count(path, expected, "learned-program tranche")
    program = task["programs"][program_index]
    with h5py.File(path, "a") as dataset:
        for key, value in (
            ("task_name", task["task_name"]),
            ("task_slug", task["slug"]),
            ("robot_name", task["robot_name"]),
            ("learned_program_path", program["path"]),
        ):
            if key in dataset.attrs and decoded_attr(dataset.attrs[key]) != value:
                raise RuntimeError(
                    f"Refusing conflicting metadata {key!r} in {path}: "
                    f"{decoded_attr(dataset.attrs[key])!r} != {value!r}"
                )
            dataset.attrs[key] = value
        expected_ints = {
            "learned_program_index": program_index + 1,
            "tranche_index": tranche_index,
            "demo_count": expected,
            "base_seed": seed_base(program, tranche_index, collection),
        }
        for key, value in expected_ints.items():
            if key in dataset.attrs and int(dataset.attrs[key]) != value:
                raise RuntimeError(
                    f"Refusing conflicting metadata {key!r} in {path}: "
                    f"{dataset.attrs[key]!r} != {value!r}"
                )
            dataset.attrs[key] = value
        if "provenance_source" not in dataset.attrs:
            dataset.attrs["provenance_source"] = source
        data = dataset.require_group("data")
        data.attrs["robot_name"] = task["robot_name"]
        data.attrs["total"] = expected
    return write_checksum(path)


def split_legacy_mixed_tranche(
    source_path: Path,
    output_paths: list[Path],
    *,
    task: dict[str, Any],
    collection: dict[str, Any],
) -> None:
    per_program = int(collection["successful_trajectories_per_tranche_per_program"])
    validate_legacy_mixed_tranche(source_path, task, per_program)
    source_checksum = sha256_file(source_path)
    with h5py.File(source_path, "r") as source:
        names = numbered_demo_names(source)
        for program_index, output_path in enumerate(output_paths):
            if output_path.exists():
                require_count(output_path, per_program, "existing imported program tranche")
                ensure_program_metadata(
                    output_path,
                    task=task,
                    program_index=program_index,
                    tranche_index=1,
                    collection=collection,
                    source="legacy_mpc800_split",
                )
                continue
            output_path.parent.mkdir(parents=True, exist_ok=True)
            partial = output_path.with_name(
                f".{output_path.stem}.partial.{os.getpid()}{output_path.suffix}"
            )
            selected = names[program_index * per_program : (program_index + 1) * per_program]
            try:
                with h5py.File(partial, "w") as target:
                    for key, value in source.attrs.items():
                        target.attrs[key] = value
                    for stale_key in (
                        "demos_per_learned_program_json",
                        "learned_program_paths_json",
                    ):
                        if stale_key in target.attrs:
                            del target.attrs[stale_key]
                    target.attrs["source_mixed_dataset"] = str(source_path)
                    target.attrs["source_mixed_dataset_sha256"] = source_checksum
                    target.attrs["source_demo_numeric_start"] = program_index * per_program
                    target.attrs["source_demo_numeric_end_inclusive"] = (
                        program_index + 1
                    ) * per_program - 1
                    target_data = target.create_group("data")
                    for key, value in source["data"].attrs.items():
                        target_data.attrs[key] = value
                    target_data.attrs["total"] = per_program
                    for new_index, source_name in enumerate(selected):
                        source.copy(
                            source["data"][source_name],
                            target_data,
                            name=f"demo_{new_index}",
                        )
                require_count(partial, per_program, "split legacy program tranche")
                os.replace(partial, output_path)
            finally:
                if partial.exists():
                    partial.unlink()
            ensure_program_metadata(
                output_path,
                task=task,
                program_index=program_index,
                tranche_index=1,
                collection=collection,
                source="legacy_mpc800_split",
            )
            print(
                f"[import] {task['slug']} program {program_index + 1}: "
                f"legacy demos {program_index * per_program}..{(program_index + 1) * per_program - 1} "
                f"-> {output_path}"
            )


def build_launch_command(
    *,
    task: dict[str, Any],
    program: dict[str, Any],
    tranche_index: int,
    program_root: Path,
    compact_output: Path,
    gpu_ids: list[int],
    collection: dict[str, Any],
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
        str(seed_base(program, tranche_index, collection)),
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


def terminate_launchers(processes: list[subprocess.Popen[Any]]) -> None:
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

    processes: list[subprocess.Popen[Any]] = []
    log_handles = []

    def handle_signal(signum: int, frame: Any) -> None:
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
                    env=checkout_env(),
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


def collect_or_import_tranche(
    *,
    task: dict[str, Any],
    task_root: Path,
    legacy_mixed: Path,
    tranche_index: int,
    gpu_ids_by_program: list[list[int]],
    collection: dict[str, Any],
    args: argparse.Namespace,
) -> list[Path]:
    expected = int(collection["successful_trajectories_per_tranche_per_program"])
    outputs = tranche_program_paths(task_root, tranche_index)
    valid = [count_demos(path) == expected for path in outputs]
    for path, is_valid in zip(outputs, valid, strict=True):
        if path.exists() and not is_valid:
            raise RuntimeError(
                f"Refusing to overwrite incomplete normalized tranche ({count_demos(path)} demos): {path}"
            )

    if tranche_index == 1 and legacy_mixed.exists() and not all(valid):
        if args.dry_run:
            validate_legacy_mixed_tranche(legacy_mixed, task, expected)
            print(f"[dry-run] would import and split validated legacy tranche {legacy_mixed}")
            return outputs
        split_legacy_mixed_tranche(
            legacy_mixed,
            outputs,
            task=task,
            collection=collection,
        )
        valid = [True, True]

    attempt = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    commands: list[tuple[list[str], Path]] = []
    root = tranche_root(task_root, tranche_index)
    for program_index, (program, output, is_valid) in enumerate(
        zip(task["programs"], outputs, valid, strict=True)
    ):
        if is_valid:
            if not args.dry_run:
                ensure_program_metadata(
                    output,
                    task=task,
                    program_index=program_index,
                    tranche_index=tranche_index,
                    collection=collection,
                    source="normalized_tranche_reuse",
                )
            print(f"[reuse] {output}: verified {expected} demos")
            continue
        run_root = root / "runs" / (
            f"program_{program_index + 1:02d}_{Path(program['path']).name}_{attempt}"
        )
        command = build_launch_command(
            task=task,
            program=program,
            tranche_index=tranche_index,
            program_root=run_root,
            compact_output=output,
            gpu_ids=gpu_ids_by_program[program_index],
            collection=collection,
            worker_stagger_sec=args.worker_stagger_sec,
            cpu_threads_per_worker=args.cpu_threads_per_worker,
        )
        commands.append(
            (command, root / "logs" / f"launcher_program{program_index + 1:02d}_{attempt}.log")
        )

    if commands:
        print(f"[tranche] {task['slug']} tranche {tranche_index}")
        run_launchers(
            commands,
            program_start_offset_sec=args.program_start_offset_sec,
            dry_run=args.dry_run,
        )
    if args.dry_run:
        return outputs

    checksums = []
    for program_index, output in enumerate(outputs):
        require_count(output, expected, "completed learned-program tranche")
        checksums.append(
            ensure_program_metadata(
                output,
                task=task,
                program_index=program_index,
                tranche_index=tranche_index,
                collection=collection,
                source="mpc_collection",
            )
        )
    write_json_atomic(
        root / "completion.json",
        {
            "completed_at_utc": utc_now(),
            "task_name": task["task_name"],
            "task_slug": task["slug"],
            "robot_name": task["robot_name"],
            "tranche_index": tranche_index,
            "program_datasets": [str(path) for path in outputs],
            "program_demo_counts": [expected, expected],
            "program_sha256": checksums,
            "learned_program_paths": learned_program_paths(task),
            "base_seeds": [
                seed_base(program, tranche_index, collection) for program in task["programs"]
            ],
        },
    )
    return outputs


def combine_hdf5(
    *,
    inputs: list[Path],
    output: Path,
    expected_count: int,
    robot_name: str,
    attrs: dict[str, Any],
    log_path: Path,
    preserve_demo_content: bool = False,
) -> str:
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite existing aggregate: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(f".{output.stem}.partial.{os.getpid()}{output.suffix}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if preserve_demo_content:
            output_index = 0
            copied_env_args = False
            with h5py.File(partial, "w") as output_dataset:
                output_data = output_dataset.require_group("data")
                for input_path in inputs:
                    with h5py.File(input_path, "r") as input_dataset:
                        input_data = input_dataset.get("data")
                        if input_data is None:
                            raise RuntimeError(
                                f"Content-preserving input lacks /data: {input_path}"
                            )
                        if not copied_env_args and "env_args" in input_data.attrs:
                            output_data.attrs["env_args"] = input_data.attrs["env_args"]
                            copied_env_args = True
                        ids = sorted(
                            int(name[5:])
                            for name in input_data
                            if name.startswith("demo_") and name[5:].isdigit()
                        )
                        if ids != list(range(len(ids))):
                            raise RuntimeError(
                                f"Non-contiguous demo IDs in aggregate input: {input_path}"
                            )
                        for input_index in ids:
                            input_dataset.copy(
                                f"data/demo_{input_index}",
                                output_data,
                                name=f"demo_{output_index}",
                            )
                            output_index += 1
            with log_path.open("w") as log_handle:
                log_handle.write(
                    "[CONTENT_PRESERVING_COMBINE] copied "
                    f"{output_index} demo groups verbatim in declared input order\n"
                )
        else:
            command = [
                sys.executable,
                str(COMBINER),
                *(str(path) for path in inputs),
                "-o",
                str(partial),
                "--robot-name",
                robot_name,
            ]
            with log_path.open("w") as log_handle:
                subprocess.run(
                    command,
                    cwd=REPO_ROOT,
                    env=checkout_env(),
                    check=True,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                )
        require_count(partial, expected_count, "new aggregate")
        with h5py.File(partial, "a") as dataset:
            for key, value in attrs.items():
                dataset.attrs[key] = json.dumps(value) if isinstance(value, (list, dict)) else value
            data = dataset.require_group("data")
            data.attrs["robot_name"] = robot_name
            data.attrs["total"] = expected_count
        checksum = sha256_file(partial)
        os.replace(partial, output)
    finally:
        if partial.exists():
            partial.unlink()
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{checksum}  {output.name}\n"
    )
    return checksum


def validate_program_aggregate(
    path: Path,
    *,
    task: dict[str, Any],
    program_index: int,
    collection: dict[str, Any],
) -> None:
    expected = int(collection["successful_trajectories_per_program"])
    require_count(path, expected, "learned-program aggregate")
    with h5py.File(path, "r") as dataset:
        require_scalar_attr(dataset, "task_name", task["task_name"], path)
        require_scalar_attr(dataset, "task_slug", task["slug"], path)
        require_scalar_attr(dataset, "robot_name", task["robot_name"], path)
        require_scalar_attr(dataset, "learned_program_index", program_index + 1, path)
        require_scalar_attr(
            dataset,
            "learned_program_path",
            task["programs"][program_index]["path"],
            path,
        )
        per_tranche = int(collection["successful_trajectories_per_tranche_per_program"])
        require_json_attr(
            dataset,
            "demos_per_tranche_json",
            [per_tranche, per_tranche],
            path,
        )


def build_program_aggregates(
    *,
    task: dict[str, Any],
    task_root: Path,
    outputs: list[Path],
    collection: dict[str, Any],
    campaign: str,
) -> list[str]:
    per_program = int(collection["successful_trajectories_per_program"])
    per_tranche = int(collection["successful_trajectories_per_tranche_per_program"])
    checksums: list[str] = []
    for program_index, output in enumerate(outputs):
        inputs = [
            tranche_program_paths(task_root, tranche_index)[program_index]
            for tranche_index in (1, 2)
        ]
        for input_path in inputs:
            require_count(input_path, per_tranche, "program tranche input")
        if output.exists():
            validate_program_aggregate(
                output,
                task=task,
                program_index=program_index,
                collection=collection,
            )
            checksum = write_checksum(output)
            print(f"[reuse] {output}: verified {per_program} demos")
        else:
            program = task["programs"][program_index]
            checksum = combine_hdf5(
                inputs=inputs,
                output=output,
                expected_count=per_program,
                robot_name=task["robot_name"],
                attrs={
                    "campaign": campaign,
                    "task_name": task["task_name"],
                    "task_slug": task["slug"],
                    "robot_name": task["robot_name"],
                    "learned_program_index": program_index + 1,
                    "learned_program_path": program["path"],
                    "demo_count": per_program,
                    "demos_per_tranche_json": [per_tranche, per_tranche],
                    "tranche_datasets_json": [str(path) for path in inputs],
                    "tranche_base_seeds_json": [
                        seed_base(program, tranche_index, collection)
                        for tranche_index in (1, 2)
                    ],
                    "content_preservation_mode": (
                        "verbatim_hdf5_demo_group_copy"
                        if task["slug"] == "hang_cup"
                        else "strict_standard_combiner"
                    ),
                },
                log_path=task_root / "logs" / f"combine_program_{program_index + 1:02d}.log",
                preserve_demo_content=task["slug"] == "hang_cup",
            )
            validate_program_aggregate(
                output,
                task=task,
                program_index=program_index,
                collection=collection,
            )
            print(
                f"[aggregate] {task['slug']} program {program_index + 1}: "
                f"{per_tranche}+{per_tranche}={per_program}"
            )
        checksums.append(checksum)
    return checksums


def validate_final(
    path: Path, task: dict[str, Any], collection: dict[str, Any]
) -> None:
    expected = int(collection["successful_trajectories_per_task"])
    per_program = int(collection["successful_trajectories_per_program"])
    require_count(path, expected, "final task dataset")
    with h5py.File(path, "r") as dataset:
        require_scalar_attr(dataset, "task_name", task["task_name"], path)
        require_scalar_attr(dataset, "task_slug", task["slug"], path)
        require_scalar_attr(dataset, "robot_name", task["robot_name"], path)
        require_json_attr(dataset, "learned_program_paths_json", learned_program_paths(task), path)
        require_json_attr(
            dataset,
            "demos_per_learned_program_json",
            [per_program, per_program],
            path,
        )
        require_json_attr(
            dataset,
            "demo_block_layout_json",
            [
                {
                    "demo_numeric_start": 0,
                    "demo_numeric_end_inclusive": per_program - 1,
                    "learned_program_index": 1,
                },
                {
                    "demo_numeric_start": per_program,
                    "demo_numeric_end_inclusive": expected - 1,
                    "learned_program_index": 2,
                },
            ],
            path,
        )


def finalize_task(
    *,
    task: dict[str, Any],
    paths: dict[str, Path],
    collection: dict[str, Any],
    campaign: str,
) -> None:
    task_root = paths["root"]
    program_outputs = [paths["program_1_800"], paths["program_2_800"]]
    program_checksums = build_program_aggregates(
        task=task,
        task_root=task_root,
        outputs=program_outputs,
        collection=collection,
        campaign=campaign,
    )
    final_output = paths["final"]
    expected = int(collection["successful_trajectories_per_task"])
    per_program = int(collection["successful_trajectories_per_program"])
    if final_output.exists():
        validate_final(final_output, task, collection)
        final_checksum = write_checksum(final_output)
        print(f"[reuse] {final_output}: verified {expected} demos")
    else:
        block_layout = [
            {
                "demo_numeric_start": 0,
                "demo_numeric_end_inclusive": per_program - 1,
                "learned_program_index": 1,
            },
            {
                "demo_numeric_start": per_program,
                "demo_numeric_end_inclusive": expected - 1,
                "learned_program_index": 2,
            },
        ]
        final_checksum = combine_hdf5(
            inputs=program_outputs,
            output=final_output,
            expected_count=expected,
            robot_name=task["robot_name"],
            attrs={
                "campaign": campaign,
                "task_name": task["task_name"],
                "task_slug": task["slug"],
                "robot_name": task["robot_name"],
                "demo_count": expected,
                "learned_program_paths_json": learned_program_paths(task),
                "demos_per_learned_program_json": [per_program, per_program],
                "program_datasets_json": [str(path) for path in program_outputs],
                "program_dataset_sha256_json": program_checksums,
                "demo_block_layout_json": block_layout,
                "tranche_seed_bases_json": [
                    [
                        seed_base(program, tranche_index, collection)
                        for tranche_index in (1, 2)
                    ]
                    for program in task["programs"]
                ],
                "content_preservation_mode": (
                    "verbatim_hdf5_demo_group_copy"
                    if task["slug"] == "hang_cup"
                    else "strict_standard_combiner"
                ),
            },
            log_path=task_root / "logs" / "combine_final_mpc1600.log",
            preserve_demo_content=task["slug"] == "hang_cup",
        )
        validate_final(final_output, task, collection)
    completion = {
        "completed_at_utc": utc_now(),
        "campaign": campaign,
        "task_name": task["task_name"],
        "task_slug": task["slug"],
        "robot_name": task["robot_name"],
        "dataset": str(final_output),
        "demo_count": expected,
        "sha256": final_checksum,
        "learned_program_paths": learned_program_paths(task),
        "demos_per_learned_program": [per_program, per_program],
        "program_datasets": [str(path) for path in program_outputs],
        "program_dataset_sha256": program_checksums,
    }
    write_json_atomic(task_root / "completion_mpc1600.json", completion)
    print(
        f"[complete] {task['slug']}: {per_program} program1 + {per_program} program2 "
        f"= {expected} demos sha256={final_checksum}"
    )


def select_tasks(
    tasks: list[dict[str, Any]],
    *,
    start_task: str | None,
    only_task: str | None,
    explicit_tasks: list[str] | None,
) -> list[dict[str, Any]]:
    selectors = sum(value is not None for value in (start_task, only_task, explicit_tasks))
    if selectors > 1:
        raise ValueError("Use only one of --start-task, --only-task, or --tasks")
    by_slug = {task["slug"]: task for task in tasks}
    if explicit_tasks is not None:
        unknown = [slug for slug in explicit_tasks if slug not in by_slug]
        if unknown:
            raise ValueError(f"Unknown --tasks values: {unknown}; choices: {list(by_slug)}")
        if len(set(explicit_tasks)) != len(explicit_tasks):
            raise ValueError(f"Duplicate task in --tasks: {explicit_tasks}")
        return [by_slug[slug] for slug in explicit_tasks]
    slugs = list(by_slug)
    if only_task is not None:
        if only_task not in by_slug:
            raise ValueError(f"Unknown --only-task {only_task!r}; choices: {slugs}")
        return [by_slug[only_task]]
    if start_task is not None:
        if start_task not in by_slug:
            raise ValueError(f"Unknown --start-task {start_task!r}; choices: {slugs}")
        return tasks[slugs.index(start_task) :]
    return tasks


def pid_is_alive(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text().split()[2]
        return state != "Z"
    except FileNotFoundError:
        return False
    except PermissionError:
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False


def wait_for_pid(pid: int, poll_sec: float) -> None:
    if poll_sec <= 0:
        raise ValueError("--wait-poll-sec must be positive")
    print(f"[wait] waiting for predecessor PID {pid}")
    last_report = 0.0
    while pid_is_alive(pid):
        now = time.monotonic()
        if now - last_report >= 300:
            print(f"[wait] predecessor PID {pid} is still active")
            last_report = now
        time.sleep(min(poll_sec, 60.0))
    print(f"[wait] predecessor PID {pid} exited")


def wait_for_paths(paths: list[Path], poll_sec: float) -> None:
    if poll_sec <= 0:
        raise ValueError("--wait-poll-sec must be positive")
    required = [path.expanduser().resolve() for path in paths]
    if not required:
        return
    print("[wait] waiting for required paths:")
    for path in required:
        print(f"[wait]   {path}")
    last_report = 0.0
    while True:
        missing = [path for path in required if not path.exists()]
        if not missing:
            break
        now = time.monotonic()
        if now - last_report >= 300:
            print(
                "[wait] required paths still missing: "
                + ", ".join(str(path) for path in missing)
            )
            last_report = now
        time.sleep(min(poll_sec, 60.0))
    print("[wait] all required paths are ready")


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    validate_manifest(manifest)
    collection = manifest["collection"]
    output_root = require_external_artifact_path(
        args.output_root if args.output_root is not None else manifest["output_root"],
        "MPC-1600 campaign output root",
    )
    tasks = select_tasks(
        manifest["tasks"],
        start_task=args.start_task,
        only_task=args.only_task,
        explicit_tasks=args.tasks,
    )

    common_gpus = list(args.gpus or collection["gpus"])
    gpu_ids_by_program = [
        list(args.program_1_gpus or common_gpus),
        list(args.program_2_gpus or common_gpus),
    ]
    for index, gpu_ids in enumerate(gpu_ids_by_program, start=1):
        validate_gpu_assignment(gpu_ids, collection, f"program {index} GPU assignment")

    precollect_gpu_ids_by_program = [
        list(args.precollect_program_1_gpus or gpu_ids_by_program[0]),
        list(args.precollect_program_2_gpus or gpu_ids_by_program[1]),
    ]
    for index, gpu_ids in enumerate(precollect_gpu_ids_by_program, start=1):
        validate_gpu_assignment(
            gpu_ids, collection, f"precollect program {index} GPU assignment"
        )

    if args.import_legacy_only and args.precollect_tranche is not None:
        raise ValueError("--import-legacy-only cannot be combined with --precollect-tranche")
    if args.precollect_only and args.precollect_tranche is None:
        raise ValueError("--precollect-only requires --precollect-tranche")

    if args.precollect_tranche is not None:
        task = tasks[0]
        paths = task_paths(output_root, task)
        if paths["final"].exists():
            validate_final(paths["final"], task, collection)
            print(
                f"[precollect-skip] {task['slug']}: final 1,600-demo dataset already exists"
            )
        else:
            if not args.dry_run:
                paths["root"].mkdir(parents=True, exist_ok=True)
            collect_or_import_tranche(
                task=task,
                task_root=paths["root"],
                legacy_mixed=paths["legacy_mixed"],
                tranche_index=args.precollect_tranche,
                gpu_ids_by_program=precollect_gpu_ids_by_program,
                collection=collection,
                args=args,
            )

        if args.precollect_only:
            print(
                f"[precollect-only] {task['slug']} tranche "
                f"{args.precollect_tranche} complete"
            )
            return

    if args.wait_for_pid is not None and not args.dry_run:
        wait_for_pid(args.wait_for_pid, args.wait_poll_sec)
    if args.wait_for_path and not args.dry_run:
        wait_for_paths(args.wait_for_path, args.wait_poll_sec)

    for task in tasks:
        validate_task(task)
        paths = task_paths(output_root, task)
        final_output = paths["final"]
        if final_output.exists():
            validate_final(final_output, task, collection)
            print(f"[skip] {task['slug']}: verified 1,600-demo dataset {final_output}")
            continue
        if not args.dry_run:
            paths["root"].mkdir(parents=True, exist_ok=True)
        if args.import_legacy_only:
            if not paths["legacy_mixed"].exists():
                raise FileNotFoundError(
                    f"No legacy mixed tranche to import for {task['slug']}: "
                    f"{paths['legacy_mixed']}"
                )
            collect_or_import_tranche(
                task=task,
                task_root=paths["root"],
                legacy_mixed=paths["legacy_mixed"],
                tranche_index=1,
                gpu_ids_by_program=gpu_ids_by_program,
                collection=collection,
                args=args,
            )
            continue
        for tranche_index in (1, 2):
            collect_or_import_tranche(
                task=task,
                task_root=paths["root"],
                legacy_mixed=paths["legacy_mixed"],
                tranche_index=tranche_index,
                gpu_ids_by_program=gpu_ids_by_program,
                collection=collection,
                args=args,
            )
        if args.dry_run:
            continue
        finalize_task(
            task=task,
            paths=paths,
            collection=collection,
            campaign=manifest["campaign"],
        )


if __name__ == "__main__":
    main()
