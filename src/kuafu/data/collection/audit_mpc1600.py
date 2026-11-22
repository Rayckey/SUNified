#!/usr/bin/env python3
"""Audit exact 800+800 MPC1600 task artifacts and their provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from kuafu.paths import require_external_artifact_path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
EXPECTED_COLLECTION = {
    "successful_trajectories_per_task": 1600,
    "successful_trajectories_per_program": 800,
    "tranches_per_program": 2,
    "successful_trajectories_per_tranche_per_program": 400,
    "seed_stride_per_tranche": 1_000_000,
    "gpus": [0, 1, 2, 3],
    "workers_per_gpu_per_program": 1,
    "simultaneous_workers_per_gpu": 2,
    "trajectories_per_worker": 100,
    "control_output_mode": "absolute_ee",
    "action_noise_std": 0.003,
    "record_failures": False,
    "task_concurrency": 1,
}
EXPECTED_TASKS = {
    "drawer": ("Kinova-Table-Gen3-Drawer", "kinova"),
    "drawer_stack": ("Kinova-Table-Gen3-Drawer-Stack", "kinova"),
    "can": ("Kinova-Table-Gen3-Can", "kinova"),
    "hang_cup": ("Kinova-Table-Gen3-Hang-Cup", "kinova"),
    "pen_cup": ("Kinova-Table-Gen3-Pen-Cup", "kinova"),
    "stack2": ("FR3-FinRay-Table-Stack-2", "fr3_finray"),
    "stack3": ("FR3-FinRay-Table-Stack-3", "fr3_finray"),
    "estop_open": ("Kinova-Table-Gen3-EStop-Open", "kinova"),
    "estop_close": ("Kinova-Table-Gen3-EStop-Close", "kinova"),
}
EXPECTED_SKIPPED_STOP_TASKS = {
    "manipulation-estop-open",
    "manipulation-estop-close",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--provenance-output-root",
        type=Path,
        default=None,
        help=(
            "Original absolute output root recorded inside relocated, "
            "checksum-identical campaign artifacts. Defaults to --output-root."
        ),
    )
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--skip-sha256", action="store_true")
    return parser.parse_args()


def decoded(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass
    return value


def json_attr(dataset: h5py.File, key: str, path: Path) -> Any:
    if key not in dataset.attrs:
        raise RuntimeError(f"Missing attribute {key!r}: {path}")
    try:
        return json.loads(str(decoded(dataset.attrs[key])))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON attribute {key!r}: {path}") from exc


def scalar_attr(dataset: h5py.File, key: str, expected: Any, path: Path) -> None:
    actual = decoded(dataset.attrs.get(key))
    if actual != expected:
        raise RuntimeError(
            f"Wrong attribute {key!r}: got {actual!r}, expected {expected!r}: {path}"
        )


def demo_ids(dataset: h5py.File) -> list[int]:
    data = dataset.get("data")
    if data is None:
        return []
    result = []
    for name in data.keys():
        match = re.fullmatch(r"demo_(\d+)", name)
        if match:
            result.append(int(match.group(1)))
    return sorted(result)


def validate_count(path: Path, expected: int) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as dataset:
        ids = demo_ids(dataset)
        if ids != list(range(expected)):
            raise RuntimeError(
                f"Expected contiguous demo IDs 0..{expected - 1}, got {len(ids)} IDs: {path}"
            )
        data_total = int(dataset["data"].attrs.get("total", -1))
        if data_total != expected:
            raise RuntimeError(
                f"data.total is {data_total}, expected {expected}: {path}"
            )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_checksum(path: Path) -> str:
    checksum = sha256_file(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file():
        raise FileNotFoundError(f"Missing SHA-256 sidecar: {sidecar}")
    fields = sidecar.read_text().strip().split()
    if len(fields) != 2 or fields[0] != checksum or fields[1] != path.name:
        raise RuntimeError(f"SHA-256 sidecar mismatch: {sidecar}")
    return checksum


def validate_manifest(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    collection = manifest.get("collection", {})
    mismatches = {
        key: (collection.get(key), expected)
        for key, expected in EXPECTED_COLLECTION.items()
        if collection.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"Manifest collection settings mismatch: {mismatches}")
    skipped_stop_tasks = manifest.get("skipped_stop_tasks")
    if (
        skipped_stop_tasks is not None
        and set(skipped_stop_tasks) != EXPECTED_SKIPPED_STOP_TASKS
    ):
        raise RuntimeError(
            f"Manifest stop-task exclusions mismatch: {skipped_stop_tasks}"
        )
    tasks = manifest.get("tasks", [])
    if not isinstance(tasks, list) or not tasks:
        raise RuntimeError("Manifest must contain at least one task")
    by_slug = {task.get("slug"): task for task in tasks}
    if len(by_slug) != len(tasks):
        raise RuntimeError("Manifest contains duplicate task slugs")
    unknown = sorted(set(by_slug) - set(EXPECTED_TASKS))
    if unknown:
        raise RuntimeError(f"Manifest contains unknown task slugs: {unknown}")
    all_base_seeds: list[int] = []
    for slug, task in by_slug.items():
        expected_pair = EXPECTED_TASKS[slug]
        actual_pair = (task.get("task_name"), task.get("robot_name"))
        if actual_pair != expected_pair:
            raise RuntimeError(
                f"Manifest task/robot mismatch for {slug}: "
                f"{actual_pair} != {expected_pair}"
            )
        programs = task.get("programs", [])
        if len(programs) != 2:
            raise RuntimeError(f"Manifest task {slug} does not have exactly two programs")
        for program in programs:
            if not isinstance(program.get("path"), str) or not program["path"]:
                raise RuntimeError(f"Manifest task {slug} has invalid learned-program path")
            try:
                base_seed = int(program["base_seed"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"Manifest task {slug} has invalid learned-program base seed"
                ) from exc
            if base_seed < 0:
                raise RuntimeError(f"Manifest task {slug} has negative base seed")
            all_base_seeds.append(base_seed)
    if len(set(all_base_seeds)) != len(all_base_seeds):
        raise RuntimeError(f"Manifest learned-program base seeds are not unique: {all_base_seeds}")
    return by_slug


def validate_learned_program(program: dict[str, Any]) -> None:
    path = (REPO_ROOT / program["path"]).resolve()
    try:
        path.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise RuntimeError(f"Learned-program path escapes project root: {path}") from exc
    if not path.is_dir() or not (path / "program").is_dir():
        raise FileNotFoundError(f"Missing learned-program directory/program: {path}")
    if (path / "success.txt").read_text().strip().lower() != "true":
        raise RuntimeError(f"Learned program is not marked successful: {path}")
    if "status: succeeded" not in (path / "stats.txt").read_text():
        raise RuntimeError(f"Learned program does not have succeeded status: {path}")


def values_equal(left: Any, right: Any) -> bool:
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if left_array.shape != right_array.shape or left_array.dtype != right_array.dtype:
        return False
    try:
        return bool(np.array_equal(left_array, right_array, equal_nan=True))
    except TypeError:
        return bool(np.array_equal(left_array, right_array))


def validate_object_copy(
    source: h5py.Group | h5py.Dataset,
    target: h5py.Group | h5py.Dataset,
    *,
    label: str,
) -> None:
    if type(source) is not type(target):
        raise RuntimeError(
            f"HDF5 object type mismatch for {label}: "
            f"{type(source).__name__} != {type(target).__name__}"
        )
    if set(source.attrs) != set(target.attrs):
        raise RuntimeError(f"HDF5 attribute-name mismatch for {label}")
    for key in source.attrs:
        if not values_equal(source.attrs[key], target.attrs[key]):
            raise RuntimeError(f"HDF5 attribute {key!r} mismatch for {label}")

    if isinstance(source, h5py.Dataset):
        if source.shape != target.shape or source.dtype != target.dtype:
            raise RuntimeError(f"HDF5 dataset shape/dtype mismatch for {label}")
        if not values_equal(source[()], target[()]):
            raise RuntimeError(f"HDF5 dataset values mismatch for {label}")
        return

    if set(source.keys()) != set(target.keys()):
        raise RuntimeError(f"HDF5 child-object mismatch for {label}")
    for name in sorted(source.keys()):
        validate_object_copy(
            source[name],
            target[name],
            label=f"{label}/{name}",
        )


def validate_concatenated_content(
    output_path: Path,
    input_paths: list[Path],
    input_counts: list[int],
) -> int:
    if len(input_paths) != len(input_counts):
        raise ValueError("Content-provenance inputs and counts disagree")
    compared = 0
    with h5py.File(output_path, "r") as output_dataset:
        output_data = output_dataset["data"]
        for input_path, input_count in zip(input_paths, input_counts, strict=True):
            with h5py.File(input_path, "r") as input_dataset:
                input_data = input_dataset["data"]
                for source_index in range(input_count):
                    source_name = f"demo_{source_index}"
                    target_name = f"demo_{compared}"
                    validate_object_copy(
                        input_data[source_name],
                        output_data[target_name],
                        label=(
                            f"{input_path.name}:{source_name} -> "
                            f"{output_path.name}:{target_name}"
                        ),
                    )
                    compared += 1
    return compared


def validate_content_slice(
    output_path: Path,
    input_path: Path,
    *,
    input_start: int,
    output_start: int,
    count: int,
) -> int:
    with (
        h5py.File(input_path, "r") as input_dataset,
        h5py.File(output_path, "r") as output_dataset,
    ):
        input_data = input_dataset["data"]
        output_data = output_dataset["data"]
        for offset in range(count):
            source_name = f"demo_{input_start + offset}"
            target_name = f"demo_{output_start + offset}"
            validate_object_copy(
                input_data[source_name],
                output_data[target_name],
                label=(
                    f"{input_path.name}:{source_name} -> "
                    f"{output_path.name}:{target_name}"
                ),
            )
    return count


def validate_tranche_program(
    path: Path,
    *,
    provenance_root: Path,
    task: dict[str, Any],
    program_index: int,
    tranche_index: int,
    seed_stride: int,
    legacy_checksum_cache: dict[Path, str],
) -> tuple[int, int]:
    validate_count(path, 400)
    program = task["programs"][program_index]
    legacy_content_source: Path | None = None
    fragmented_sources: list[Path] = []
    fragmented_counts: list[int] = []
    fragmented_hashes: list[str] = []
    fragmented_seeds: list[int] = []
    fragmented_servers: list[str] = []
    fragmented_record: Path | None = None
    fragmented_record_checksum: str | None = None
    with h5py.File(path, "r") as dataset:
        scalar_attr(dataset, "task_name", task["task_name"], path)
        scalar_attr(dataset, "task_slug", task["slug"], path)
        scalar_attr(dataset, "robot_name", task["robot_name"], path)
        scalar_attr(dataset, "learned_program_path", program["path"], path)
        scalar_attr(dataset, "learned_program_index", program_index + 1, path)
        scalar_attr(dataset, "tranche_index", tranche_index, path)
        scalar_attr(dataset, "demo_count", 400, path)
        scalar_attr(
            dataset,
            "base_seed",
            int(program["base_seed"]) + (tranche_index - 1) * seed_stride,
            path,
        )
        provenance_source = str(decoded(dataset.attrs.get("provenance_source", "")))
        allowed_sources = {
            "legacy_mpc800_split",
            "mpc_collection",
            "normalized_tranche_reuse",
            "serial_single_gpu_offload",
            "fragmented_single_gpu_offload",
        }
        if provenance_source not in allowed_sources:
            raise RuntimeError(
                f"Unknown or missing provenance_source={provenance_source!r}: {path}"
            )

        if provenance_source == "serial_single_gpu_offload":
            expected_worker_seeds = [
                int(program["base_seed"])
                + (tranche_index - 1) * seed_stride
                + worker_index
                for worker_index in range(4)
            ]
            if json_attr(dataset, "worker_seed_bases_json", path) != expected_worker_seeds:
                raise RuntimeError(f"Wrong serial worker seed provenance: {path}")

        if provenance_source == "fragmented_single_gpu_offload":
            if task["slug"] != "hang_cup" or program_index not in (0, 1) or tranche_index != 2:
                raise RuntimeError(
                    f"Fragmented provenance used outside Hang Cup tranche 2: {path}"
                )
            fragmented_counts = json_attr(
                dataset, "source_fragment_demo_counts_json", path
            )
            fragmented_seeds = json_attr(dataset, "worker_seed_bases_json", path)
            fragmented_hashes = json_attr(
                dataset, "source_fragment_sha256_json", path
            )
            fragmented_servers = json_attr(
                dataset, "source_fragment_servers_json", path
            )
            fragmented_specs = {
                0: {
                    "counts": [
                        66,
                        91,
                        100,
                        63,
                        8,
                        6,
                        9,
                        4,
                        3,
                        5,
                        5,
                        5,
                        4,
                        2,
                        1,
                        3,
                        4,
                        4,
                        4,
                        2,
                        2,
                        2,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                    ],
                    "seeds": [
                        1_410_000,
                        1_410_001,
                        1_410_002,
                        1_410_003,
                        3_410_000,
                        3_410_001,
                        3_410_002,
                        3_410_003,
                        3_410_004,
                        3_410_005,
                        3_430_000,
                        3_430_001,
                        3_430_002,
                        3_430_003,
                        3_430_004,
                        3_430_005,
                        3_430_006,
                        3_430_007,
                        3_430_009,
                        3_430_010,
                        3_430_011,
                        3_430_013,
                        3_430_014,
                        3_430_015,
                        3_430_016,
                        3_430_017,
                        3_430_018,
                        3_430_020,
                        3_430_021,
                    ],
                    "servers": [
                        "hermes",
                        "thor",
                        "hermes",
                        "hermes",
                        "ares",
                        "ares",
                        "thor",
                        "thor",
                        "hermes",
                        "hermes",
                        "hermes",
                        "hermes",
                        "ares",
                        "ares",
                        "ares",
                        "ares",
                        "hermes",
                        "hermes",
                        "thor",
                        "thor",
                        "hermes",
                        "ares",
                        "hermes",
                        "ares",
                        "ares",
                        "ares",
                        "hermes",
                        "hermes",
                        "ares",
                    ],
                    "names": [
                        "original_slot00_seed1410000.hdf5",
                        "original_slot01_seed1410001.hdf5",
                        "original_slot02_seed1410002.hdf5",
                        "original_slot03_seed1410003.hdf5",
                        "frag00_seed3410000_partial_008.hdf5",
                        "frag01_seed3410001_partial_006.hdf5",
                        "frag02_seed3410002_partial_009.hdf5",
                        "frag03_seed3410003_partial_004.hdf5",
                        "frag04_seed3410004_partial_003.hdf5",
                        "frag05_seed3410005_partial_005.hdf5",
                        "frag09_seed3430000_005.hdf5",
                        "frag10_seed3430001_005.hdf5",
                        "frag11_seed3430002_004.hdf5",
                        "frag12_seed3430003_partial_002.hdf5",
                        "frag15_seed3430004_partial_001.hdf5",
                        "frag16_seed3430005_003.hdf5",
                        "frag17_seed3430006_004.hdf5",
                        "frag18_seed3430007_004.hdf5",
                        "frag20_seed3430009_004.hdf5",
                        "frag21_seed3430010_002.hdf5",
                        "frag22_seed3430011_002.hdf5",
                        "frag24_seed3430013_002.hdf5",
                        "frag25_seed3430014_001.hdf5",
                        "frag26_seed3430015_001.hdf5",
                        "frag27_seed3430016_001.hdf5",
                        "frag28_seed3430017_001.hdf5",
                        "frag29_seed3430018_001.hdf5",
                        "frag31_seed3430020_001.hdf5",
                        "frag32_seed3430021_001.hdf5",
                    ],
                },
                1: {
                    "counts": [100, 96, 100, 88, 2, 5, 5, 2, 2],
                    "seeds": [
                        1_420_000,
                        1_420_001,
                        1_420_002,
                        1_420_003,
                        3_420_000,
                        3_420_001,
                        3_420_002,
                        3_431_000,
                        3_431_001,
                    ],
                    "servers": [
                        "thor",
                        "hermes",
                        "hermes",
                        "hermes",
                        "thor",
                        "hermes",
                        "hermes",
                        "thor",
                        "hermes",
                    ],
                    "names": [
                        "original_slot00_seed1420000.hdf5",
                        "original_slot01_seed1420001.hdf5",
                        "original_slot02_seed1420002.hdf5",
                        "original_slot03_seed1420003.hdf5",
                        "frag06_seed3420000_partial_002.hdf5",
                        "frag07_seed3420001_005.hdf5",
                        "frag08_seed3420002_005.hdf5",
                        "frag13_seed3431000_002.hdf5",
                        "frag14_seed3431001_002.hdf5",
                    ],
                },
            }
            fragmented_spec = fragmented_specs[program_index]
            expected_counts = fragmented_spec["counts"]
            expected_seeds = fragmented_spec["seeds"]
            expected_servers = fragmented_spec["servers"]
            expected_names = fragmented_spec["names"]
            expected_source_root = (
                provenance_root
                / "tranches"
                / "tranche_02"
                / f"fragmented_program_{program_index + 1:02d}_sources"
            ).resolve()
            expected_sources = [
                (expected_source_root / name).resolve() for name in expected_names
            ]
            fragmented_sources = [
                Path(item).expanduser().resolve()
                for item in json_attr(
                    dataset, "source_fragment_datasets_json", path
                )
            ]
            if (
                fragmented_counts != expected_counts
                or sum(fragmented_counts) != 400
                or fragmented_seeds != expected_seeds
                or len(set(fragmented_seeds)) != len(fragmented_seeds)
                or fragmented_servers != expected_servers
                or fragmented_sources != expected_sources
                or len(fragmented_hashes) != len(expected_sources)
            ):
                raise RuntimeError(f"Wrong fragmented source provenance: {path}")
            scalar_attr(dataset, "unique_demo_content_fingerprints", 400, path)
            fragmented_record = Path(
                str(decoded(dataset.attrs.get("fragment_provenance_record", "")))
            ).expanduser().resolve()
            expected_record = (
                provenance_root
                / "tranches"
                / "tranche_02"
                / f"program_{program_index + 1:02d}_400.fragmented_provenance.json"
            ).resolve()
            if fragmented_record != expected_record:
                raise RuntimeError(f"Wrong fragmented provenance record path: {path}")
            fragmented_record_checksum = str(
                decoded(dataset.attrs.get("fragment_provenance_record_sha256", ""))
            )
            if not re.fullmatch(r"[0-9a-f]{64}", fragmented_record_checksum):
                raise RuntimeError(f"Invalid fragmented provenance record hash: {path}")

        if provenance_source == "legacy_mpc800_split":
            if tranche_index != 1:
                raise RuntimeError(f"Legacy MPC800 split used outside tranche 1: {path}")
            legacy_path = (
                path.parents[2]
                / f"{task['robot_name']}_{task['slug']}_first2_programs_mpc800.hdf5"
            ).resolve()
            provenance_legacy_path = (
                provenance_root
                / f"{task['robot_name']}_{task['slug']}_first2_programs_mpc800.hdf5"
            ).resolve()
            scalar_attr(
                dataset,
                "source_mixed_dataset",
                str(provenance_legacy_path),
                path,
            )
            scalar_attr(dataset, "source_demo_numeric_start", program_index * 400, path)
            scalar_attr(
                dataset,
                "source_demo_numeric_end_inclusive",
                (program_index + 1) * 400 - 1,
                path,
            )
            if legacy_path not in legacy_checksum_cache:
                with h5py.File(legacy_path, "r") as legacy_dataset:
                    if demo_ids(legacy_dataset) != list(range(800)):
                        raise RuntimeError(
                            f"Legacy source does not contain contiguous demo IDs 0..799: "
                            f"{legacy_path}"
                        )
                    scalar_attr(legacy_dataset, "task_name", task["task_name"], legacy_path)
                    scalar_attr(legacy_dataset, "robot_name", task["robot_name"], legacy_path)
                    if json_attr(
                        legacy_dataset, "learned_program_paths_json", legacy_path
                    ) != [item["path"] for item in task["programs"]]:
                        raise RuntimeError(f"Wrong learned programs in legacy source: {legacy_path}")
                    if json_attr(
                        legacy_dataset, "demos_per_learned_program_json", legacy_path
                    ) != [400, 400]:
                        raise RuntimeError(f"Wrong program split in legacy source: {legacy_path}")
                legacy_checksum_cache[legacy_path] = sha256_file(legacy_path)
            scalar_attr(
                dataset,
                "source_mixed_dataset_sha256",
                legacy_checksum_cache[legacy_path],
                path,
            )
            legacy_content_source = legacy_path

    if legacy_content_source is not None:
        return (
            validate_content_slice(
                path,
                legacy_content_source,
                input_start=program_index * 400,
                output_start=0,
                count=400,
            ),
            0,
        )
    if fragmented_sources:
        compared = 0
        source_records: list[dict[str, Any]] = []
        for source_path, count, checksum, seed, server in zip(
            fragmented_sources,
            fragmented_counts,
            fragmented_hashes,
            fragmented_seeds,
            fragmented_servers,
            strict=True,
        ):
            if source_path.is_symlink() or not source_path.is_file():
                raise FileNotFoundError(f"Missing regular fragment source: {source_path}")
            if sha256_file(source_path) != checksum:
                raise RuntimeError(f"Fragment source SHA-256 mismatch: {source_path}")
            with h5py.File(source_path, "r") as source_dataset:
                if demo_ids(source_dataset) != list(range(count)):
                    raise RuntimeError(
                        f"Fragment source does not contain demos 0..{count - 1}: "
                        f"{source_path}"
                    )
            validate_content_slice(
                path,
                source_path,
                input_start=0,
                output_start=compared,
                count=count,
            )
            compared += count
            source_records.append(
                {
                    "path": str(source_path),
                    "server": server,
                    "seed": seed,
                    "demo_count": count,
                    "sha256": checksum,
                }
            )
        if compared != 400:
            raise RuntimeError(f"Compared {compared} fragmented demos, expected 400")
        if fragmented_record is None or fragmented_record_checksum is None:
            raise RuntimeError(f"Missing fragmented provenance record: {path}")
        if not fragmented_record.is_file():
            raise FileNotFoundError(fragmented_record)
        if sha256_file(fragmented_record) != fragmented_record_checksum:
            raise RuntimeError(
                f"Fragmented provenance record SHA-256 mismatch: {fragmented_record}"
            )
        record = json.loads(fragmented_record.read_text())
        if (
            record.get("task_name") != task["task_name"]
            or record.get("task_slug") != task["slug"]
            or record.get("robot_name") != task["robot_name"]
            or record.get("learned_program_path") != program["path"]
            or record.get("learned_program_index") != program_index + 1
            or record.get("tranche_index") != tranche_index
            or record.get("base_seed")
            != int(program["base_seed"]) + (tranche_index - 1) * seed_stride
            or record.get("demo_count") != 400
            or record.get("provenance_source")
            != "fragmented_single_gpu_offload"
            or record.get("sources") != source_records
            or record.get("source_demo_counts") != fragmented_counts
            or record.get("source_seeds") != fragmented_seeds
            or record.get("source_sha256") != fragmented_hashes
            or record.get("unique_demo_content_fingerprints") != 400
        ):
            raise RuntimeError(
                f"Fragmented provenance record disagrees with tranche: {fragmented_record}"
            )
        return 0, compared
    return 0, 0


def validate_program_aggregate(
    path: Path,
    *,
    provenance_root: Path,
    task: dict[str, Any],
    program_index: int,
    seed_stride: int,
    campaign: str,
) -> None:
    validate_count(path, 800)
    program = task["programs"][program_index]
    with h5py.File(path, "r") as dataset:
        scalar_attr(dataset, "task_name", task["task_name"], path)
        scalar_attr(dataset, "task_slug", task["slug"], path)
        scalar_attr(dataset, "robot_name", task["robot_name"], path)
        scalar_attr(dataset, "campaign", campaign, path)
        scalar_attr(dataset, "learned_program_path", program["path"], path)
        scalar_attr(dataset, "learned_program_index", program_index + 1, path)
        if json_attr(dataset, "demos_per_tranche_json", path) != [400, 400]:
            raise RuntimeError(f"Wrong per-tranche counts: {path}")
        expected_seeds = [int(program["base_seed"]), int(program["base_seed"]) + seed_stride]
        if json_attr(dataset, "tranche_base_seeds_json", path) != expected_seeds:
            raise RuntimeError(f"Wrong tranche seed bases: {path}")
        expected_tranches = [
            str(
                provenance_root
                / "tranches"
                / f"tranche_{tranche_index:02d}"
                / f"program_{program_index + 1:02d}_400.hdf5"
            )
            for tranche_index in (1, 2)
        ]
        if json_attr(dataset, "tranche_datasets_json", path) != expected_tranches:
            raise RuntimeError(f"Wrong tranche dataset provenance: {path}")


def validate_final(
    path: Path,
    *,
    task: dict[str, Any],
    campaign: str,
    program_paths: list[Path],
    provenance_program_paths: list[Path],
    program_checksums: list[str],
    seed_stride: int,
) -> None:
    validate_count(path, 1600)
    with h5py.File(path, "r") as dataset:
        scalar_attr(dataset, "task_name", task["task_name"], path)
        scalar_attr(dataset, "task_slug", task["slug"], path)
        scalar_attr(dataset, "robot_name", task["robot_name"], path)
        scalar_attr(dataset, "campaign", campaign, path)
        expected_programs = [program["path"] for program in task["programs"]]
        if json_attr(dataset, "learned_program_paths_json", path) != expected_programs:
            raise RuntimeError(f"Wrong learned-program paths: {path}")
        if json_attr(dataset, "demos_per_learned_program_json", path) != [800, 800]:
            raise RuntimeError(f"Wrong per-program counts: {path}")
        expected_layout = [
            {
                "demo_numeric_start": 0,
                "demo_numeric_end_inclusive": 799,
                "learned_program_index": 1,
            },
            {
                "demo_numeric_start": 800,
                "demo_numeric_end_inclusive": 1599,
                "learned_program_index": 2,
            },
        ]
        if json_attr(dataset, "demo_block_layout_json", path) != expected_layout:
            raise RuntimeError(f"Wrong final demo block layout: {path}")
        if json_attr(dataset, "program_datasets_json", path) != [
            str(program_path) for program_path in provenance_program_paths
        ]:
            raise RuntimeError(f"Wrong program dataset provenance: {path}")
        if json_attr(dataset, "program_dataset_sha256_json", path) != program_checksums:
            raise RuntimeError(f"Wrong program dataset checksums: {path}")
        expected_seed_bases = [
            [int(program["base_seed"]), int(program["base_seed"]) + seed_stride]
            for program in task["programs"]
        ]
        if json_attr(dataset, "tranche_seed_bases_json", path) != expected_seed_bases:
            raise RuntimeError(f"Wrong final seed provenance: {path}")


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.expanduser().resolve().read_text())
    by_slug = validate_manifest(manifest)
    collection = manifest["collection"]
    campaign = str(manifest["campaign"])
    exact_totals = (
        int(collection["successful_trajectories_per_task"]),
        int(collection["successful_trajectories_per_program"]),
        int(collection["successful_trajectories_per_tranche_per_program"]),
    )
    if exact_totals != (1600, 800, 400):
        raise RuntimeError(f"Manifest has wrong trajectory totals: {exact_totals}")
    seed_stride = int(collection["seed_stride_per_tranche"])
    if seed_stride != 1_000_000:
        raise RuntimeError(f"Unexpected tranche seed stride: {seed_stride}")

    unknown = [slug for slug in args.tasks if slug not in by_slug]
    if unknown:
        raise ValueError(f"Unknown tasks: {unknown}")
    output_root = require_external_artifact_path(args.output_root, "MPC-1600 audit output root")
    provenance_output_root = require_external_artifact_path(
        args.provenance_output_root if args.provenance_output_root is not None else output_root,
        "MPC-1600 provenance output root",
    )
    report: dict[str, Any] = {}
    legacy_checksum_cache: dict[Path, str] = {}

    for slug in args.tasks:
        task = by_slug[slug]
        for program in task["programs"]:
            validate_learned_program(program)
        root = output_root / slug
        provenance_root = provenance_output_root / slug
        artifact_checksums: dict[str, str] = {}
        legacy_split_comparisons = 0
        fragmented_tail_comparisons = 0
        for tranche_index in (1, 2):
            for program_index in (0, 1):
                path = (
                    root
                    / "tranches"
                    / f"tranche_{tranche_index:02d}"
                    / f"program_{program_index + 1:02d}_400.hdf5"
                )
                legacy_compared, fragmented_compared = validate_tranche_program(
                    path,
                    provenance_root=provenance_root,
                    task=task,
                    program_index=program_index,
                    tranche_index=tranche_index,
                    seed_stride=seed_stride,
                    legacy_checksum_cache=legacy_checksum_cache,
                )
                legacy_split_comparisons += legacy_compared
                fragmented_tail_comparisons += fragmented_compared
                if not args.skip_sha256:
                    artifact_checksums[str(path)] = validate_checksum(path)

        program_paths = [root / "program_01_800.hdf5", root / "program_02_800.hdf5"]
        program_checksums: list[str] = []
        for program_index, path in enumerate(program_paths):
            validate_program_aggregate(
                path,
                provenance_root=provenance_root,
                task=task,
                program_index=program_index,
                seed_stride=seed_stride,
                campaign=campaign,
            )
            tranche_paths = [
                root
                / "tranches"
                / f"tranche_{tranche_index:02d}"
                / f"program_{program_index + 1:02d}_400.hdf5"
                for tranche_index in (1, 2)
            ]
            compared = validate_concatenated_content(
                path,
                tranche_paths,
                [400, 400],
            )
            if compared != 800:
                raise RuntimeError(
                    f"Compared {compared} tranche demos for {path}, expected 800"
                )
            checksum = validate_checksum(path) if not args.skip_sha256 else sha256_file(path)
            program_checksums.append(checksum)
            if not args.skip_sha256:
                artifact_checksums[str(path)] = checksum

        final_path = root / f"{task['robot_name']}_{slug}_first2_programs_mpc1600.hdf5"
        provenance_program_paths = [
            provenance_root / "program_01_800.hdf5",
            provenance_root / "program_02_800.hdf5",
        ]
        provenance_final_path = (
            provenance_root
            / f"{task['robot_name']}_{slug}_first2_programs_mpc1600.hdf5"
        )
        validate_final(
            final_path,
            task=task,
            campaign=campaign,
            program_paths=program_paths,
            provenance_program_paths=provenance_program_paths,
            program_checksums=program_checksums,
            seed_stride=seed_stride,
        )
        final_compared = validate_concatenated_content(
            final_path,
            program_paths,
            [800, 800],
        )
        if final_compared != 1600:
            raise RuntimeError(
                f"Compared {final_compared} program demos for {final_path}, expected 1600"
            )
        final_checksum = (
            validate_checksum(final_path) if not args.skip_sha256 else sha256_file(final_path)
        )
        completion_path = root / "completion_mpc1600.json"
        if not completion_path.is_file():
            raise FileNotFoundError(completion_path)
        completion = json.loads(completion_path.read_text())
        if (
            completion.get("campaign") != campaign
            or completion.get("task_name") != task["task_name"]
            or completion.get("task_slug") != task["slug"]
            or completion.get("robot_name") != task["robot_name"]
            or completion.get("demo_count") != 1600
            or completion.get("demos_per_learned_program") != [800, 800]
            or completion.get("learned_program_paths")
            != [program["path"] for program in task["programs"]]
            or completion.get("program_datasets")
            != [str(path) for path in provenance_program_paths]
            or completion.get("program_dataset_sha256") != program_checksums
            or completion.get("sha256") != final_checksum
            or Path(completion.get("dataset", "")).resolve()
            != provenance_final_path.resolve()
        ):
            raise RuntimeError(f"Completion record disagrees with final dataset: {completion_path}")
        report[slug] = {
            "task_name": task["task_name"],
            "robot_name": task["robot_name"],
            "learned_program_paths": [program["path"] for program in task["programs"]],
            "tranche_base_seeds": [
                [
                    int(program["base_seed"]),
                    int(program["base_seed"]) + seed_stride,
                ]
                for program in task["programs"]
            ],
            "demos_per_learned_program": [800, 800],
            "demo_count": 1600,
            "dataset": str(final_path),
            "provenance_dataset": str(provenance_final_path),
            "sha256": final_checksum,
            "completion_record": str(completion_path),
            "completion_record_sha256": sha256_file(completion_path),
            "validated_artifact_count": 7,
            "collection_invariants_verified": True,
            "learned_program_artifacts_verified": 2,
            "content_provenance_verified": True,
            "aggregate_content_provenance_demo_comparisons": 3200,
            "legacy_split_content_provenance_demo_comparisons": (
                legacy_split_comparisons
            ),
            "fragmented_tail_content_provenance_demo_comparisons": (
                fragmented_tail_comparisons
            ),
            "content_provenance_demo_comparisons": (
                3200 + legacy_split_comparisons + fragmented_tail_comparisons
            ),
            "artifact_checksums": artifact_checksums,
        }

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
