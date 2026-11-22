#!/usr/bin/env python3
"""Atomically assemble an exact MPC tranche from explicitly audited fragments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from kuafu.paths import require_external_artifact_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--combiner", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def demo_ids(dataset: h5py.File) -> list[int]:
    data = dataset.get("data")
    if data is None:
        return []
    result: list[int] = []
    for name in data:
        if name.startswith("demo_") and name[5:].isdigit():
            result.append(int(name[5:]))
    return sorted(result)


def value_bytes(value: Any) -> bytes:
    array = np.asarray(value)
    if array.dtype.kind in {"O", "U"}:
        return repr(array.tolist()).encode("utf-8")
    return array.tobytes(order="C")


def demo_fingerprint(group: h5py.Group) -> str:
    digest = hashlib.sha256()

    def visitor(name: str, item: h5py.Group | h5py.Dataset) -> None:
        digest.update(name.encode("utf-8"))
        digest.update(type(item).__name__.encode("ascii"))
        if isinstance(item, h5py.Dataset):
            digest.update(str(item.dtype).encode("ascii"))
            digest.update(repr(item.shape).encode("ascii"))
            digest.update(value_bytes(item[()]))

    group.visititems(visitor)
    return digest.hexdigest()


EXPECTED_TIMESERIES_KEYS = {
    "actions/absolute_ee",
    "actions/absolute_joint",
    "actions/relative_ee",
    "actions/relative_joint",
    "dones",
    "next_obs/full_state",
    "next_obs/state",
    "obs/full_state",
    "obs/state",
    "rewards",
}
LONG_RECORDER_STREAMS = {
    "actions/absolute_ee",
    "actions/absolute_joint",
    "actions/relative_ee",
    "actions/relative_joint",
    "obs/full_state",
    "obs/state",
}


def timeseries_lengths(group: h5py.Group) -> dict[str, int]:
    lengths: dict[str, int] = {}

    def visitor(name: str, item: h5py.Group | h5py.Dataset) -> None:
        if isinstance(item, h5py.Dataset) and name in EXPECTED_TIMESERIES_KEYS:
            if item.ndim < 1:
                raise RuntimeError(f"Timeseries dataset is scalar: {group.name}/{name}")
            lengths[name] = int(item.shape[0])

    group.visititems(visitor)
    if set(lengths) != EXPECTED_TIMESERIES_KEYS:
        missing = sorted(EXPECTED_TIMESERIES_KEYS - set(lengths))
        extra = sorted(set(lengths) - EXPECTED_TIMESERIES_KEYS)
        raise RuntimeError(
            f"Unexpected trajectory timeseries schema for {group.name}: "
            f"missing={missing}, extra={extra}"
        )
    return lengths


def recorder_length_mode(group: h5py.Group) -> str:
    """Accept aligned streams or the recorder's documented off-by-one layout."""
    lengths = timeseries_lengths(group)
    if lengths["obs/state"] <= 0:
        raise RuntimeError(f"Empty BC observation stream: {group.name}")
    if lengths["obs/state"] != lengths["actions/absolute_ee"]:
        raise RuntimeError(f"BC observation/action length mismatch: {group.name}")
    distinct = set(lengths.values())
    if len(distinct) == 1:
        return "aligned"
    minimum = min(distinct)
    expected = {
        name: minimum + (1 if name in LONG_RECORDER_STREAMS else 0)
        for name in EXPECTED_TIMESERIES_KEYS
    }
    if lengths != expected:
        raise RuntimeError(
            f"Trajectory is not aligned or a documented recorder off-by-one: "
            f"{group.name}: {lengths}"
        )
    return "recorder_off_by_one"


def copy_demos_preserving_content(
    input_paths: list[Path],
    counts: list[int],
    output: Path,
) -> None:
    """Concatenate demo groups verbatim in the explicitly audited source order."""
    with h5py.File(output, "w") as output_dataset:
        output_data = output_dataset.require_group("data")
        output_index = 0
        copied_env_args = False
        for input_path, count in zip(input_paths, counts, strict=True):
            with h5py.File(input_path, "r") as input_dataset:
                input_data = input_dataset["data"]
                if not copied_env_args and "env_args" in input_data.attrs:
                    output_data.attrs["env_args"] = input_data.attrs["env_args"]
                    copied_env_args = True
                for input_index in range(count):
                    input_dataset.copy(
                        f"data/demo_{input_index}",
                        output_data,
                        name=f"demo_{output_index}",
                    )
                    output_index += 1
        if output_index != sum(counts):
            raise RuntimeError(
                f"Copied {output_index} demos, expected {sum(counts)}"
            )


def main() -> None:
    args = parse_args()
    plan_path = args.plan.expanduser().resolve()
    output = require_external_artifact_path(
        args.output, "Fragmented tranche output"
    )
    source_root = require_external_artifact_path(
        args.source_root, "Fragment source root"
    )
    combiner = args.combiner.expanduser().resolve()
    plan = json.loads(plan_path.read_text())

    learned_program_index = int(plan.get("learned_program_index", -1))
    plan_specs = {
        1: {
            "learned_program_path": (
                "resources/learned_programs/manipulation-hang-cup/2026-07-29_11-18-23"
            ),
            "base_seed": 1_410_000,
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
            "assembly_scheme": "frozen_320_plus_exact_repartitioned_unique_seed_sources_80",
            "recorder_off_by_one_demo_count": 258,
        },
        2: {
            "learned_program_path": (
                "resources/learned_programs/manipulation-hang-cup/2026-07-29_11-30-12"
            ),
            "base_seed": 1_420_000,
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
            "assembly_scheme": "frozen_384_plus_exact_repartitioned_unique_seed_sources_16",
            "recorder_off_by_one_demo_count": 186,
        },
    }
    if learned_program_index not in plan_specs:
        raise RuntimeError(
            f"Unsupported fragmented learned-program index: {learned_program_index}"
        )
    spec = plan_specs[learned_program_index]
    expected_scalars = {
        "task_name": "Kinova-Table-Gen3-Hang-Cup",
        "task_slug": "hang_cup",
        "robot_name": "kinova",
        "learned_program_path": spec["learned_program_path"],
        "learned_program_index": learned_program_index,
        "tranche_index": 2,
        "base_seed": spec["base_seed"],
        "demo_count": 400,
        "provenance_source": "fragmented_single_gpu_offload",
    }
    mismatches = {
        key: (plan.get(key), expected)
        for key, expected in expected_scalars.items()
        if plan.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"Fragment plan scalar mismatch: {mismatches}")

    sources = plan.get("sources")
    if not isinstance(sources, list) or len(sources) != len(spec["counts"]):
        raise RuntimeError(
            f"Fragment plan must contain exactly {len(spec['counts'])} source records"
        )
    counts = [int(source["demo_count"]) for source in sources]
    seeds = [int(source["seed"]) for source in sources]
    expected_counts = spec["counts"]
    expected_seeds = spec["seeds"]
    if counts != expected_counts or sum(counts) != 400:
        raise RuntimeError(f"Wrong fragment counts: {counts}")
    if seeds != expected_seeds or len(set(seeds)) != len(seeds):
        raise RuntimeError(f"Wrong or duplicate fragment seeds: {seeds}")
    if [str(source["server"]) for source in sources] != spec["servers"]:
        raise RuntimeError("Wrong fragment source-server order")
    if [Path(str(source["path"])).name for source in sources] != spec["names"]:
        raise RuntimeError("Wrong fragment source-file order")

    input_paths: list[Path] = []
    source_hashes: list[str] = []
    all_fingerprints: list[str] = []
    recorder_off_by_one_demo_count = 0
    for source, count in zip(sources, counts, strict=True):
        path = Path(str(source["path"])).expanduser().resolve()
        try:
            path.relative_to(source_root)
        except ValueError as exc:
            raise RuntimeError(f"Fragment source escapes source root: {path}") from exc
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"Missing regular fragment source: {path}")
        checksum = sha256_file(path)
        if checksum != source.get("sha256"):
            raise RuntimeError(f"Fragment SHA-256 mismatch: {path}")
        with h5py.File(path, "r") as dataset:
            ids = demo_ids(dataset)
            if ids != list(range(count)):
                raise RuntimeError(
                    f"Fragment does not contain contiguous demo IDs 0..{count - 1}: {path}"
                )
            fingerprints: list[str] = []
            for index in range(count):
                group = dataset["data"][f"demo_{index}"]
                recorder_off_by_one_demo_count += int(
                    recorder_length_mode(group) == "recorder_off_by_one"
                )
                fingerprints.append(demo_fingerprint(group))
        input_paths.append(path)
        source_hashes.append(checksum)
        all_fingerprints.extend(fingerprints)

    if len(set(all_fingerprints)) != 400:
        raise RuntimeError("Duplicate trajectory content fingerprint across fragments")
    if recorder_off_by_one_demo_count != spec["recorder_off_by_one_demo_count"]:
        raise RuntimeError(
            "Wrong recorder off-by-one demo count: "
            f"{recorder_off_by_one_demo_count} != "
            f"{spec['recorder_off_by_one_demo_count']}"
        )
    if output.exists() or output.with_suffix(output.suffix + ".sha256").exists():
        raise RuntimeError(f"Refusing to overwrite existing tranche output: {output}")
    if not combiner.is_file():
        raise FileNotFoundError(combiner)

    output.parent.mkdir(parents=True, exist_ok=True)
    provenance_path = output.with_name(
        f"{output.stem}.fragmented_provenance.json"
    )
    if provenance_path.exists():
        raise RuntimeError(f"Refusing to overwrite provenance record: {provenance_path}")
    provenance_payload = {
        **expected_scalars,
        "assembly_scheme": spec["assembly_scheme"],
        "source_root": str(source_root),
        "sources": sources,
        "source_demo_counts": counts,
        "source_seeds": seeds,
        "source_sha256": source_hashes,
        "unique_demo_content_fingerprints": len(set(all_fingerprints)),
        "content_preservation_mode": "verbatim_hdf5_demo_group_copy",
        "recorder_off_by_one_demo_count": recorder_off_by_one_demo_count,
        "bc_stream_contract": {
            "obs_key": "state",
            "action_key": "absolute_ee",
            "mismatch_or_empty_demo_count": 0,
        },
    }
    provenance_temporary = provenance_path.with_name(
        f".{provenance_path.name}.partial.{os.getpid()}"
    )
    partial = output.with_name(f".{output.stem}.partial.{os.getpid()}{output.suffix}")
    try:
        provenance_temporary.write_text(
            json.dumps(provenance_payload, indent=2, sort_keys=True) + "\n"
        )
        provenance_checksum = sha256_file(provenance_temporary)
        copy_demos_preserving_content(input_paths, counts, partial)
        with h5py.File(partial, "a") as dataset:
            if demo_ids(dataset) != list(range(400)):
                raise RuntimeError("Combined tranche does not contain demos 0..399")
            for key, value in expected_scalars.items():
                dataset.attrs[key] = value
            dataset.attrs["worker_seed_bases_json"] = json.dumps(seeds)
            dataset.attrs["source_fragment_datasets_json"] = json.dumps(
                [str(path) for path in input_paths]
            )
            dataset.attrs["source_fragment_demo_counts_json"] = json.dumps(counts)
            dataset.attrs["source_fragment_sha256_json"] = json.dumps(source_hashes)
            dataset.attrs["source_fragment_servers_json"] = json.dumps(
                [str(source["server"]) for source in sources]
            )
            dataset.attrs["fragment_provenance_record"] = str(provenance_path)
            dataset.attrs["fragment_provenance_record_sha256"] = provenance_checksum
            dataset.attrs["unique_demo_content_fingerprints"] = 400
            dataset.attrs["content_preservation_mode"] = (
                "verbatim_hdf5_demo_group_copy"
            )
            dataset.attrs["recorder_off_by_one_demo_count"] = (
                recorder_off_by_one_demo_count
            )
            data = dataset.require_group("data")
            data.attrs["robot_name"] = "kinova"
            data.attrs["total"] = 400

        with h5py.File(partial, "r") as dataset:
            output_fingerprints = [
                demo_fingerprint(dataset["data"][f"demo_{index}"])
                for index in range(400)
            ]
        if output_fingerprints != all_fingerprints:
            raise RuntimeError("Combined tranche content does not match source order")
        os.replace(provenance_temporary, provenance_path)
        os.replace(partial, output)
    finally:
        if partial.exists():
            partial.unlink()
        if provenance_temporary.exists():
            provenance_temporary.unlink()

    checksum = sha256_file(output)
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{checksum}  {output.name}\n"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "demo_count": 400,
                "sha256": checksum,
                "provenance_record": str(provenance_path),
                "provenance_record_sha256": provenance_checksum,
                "source_demo_counts": counts,
                "source_seeds": seeds,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
