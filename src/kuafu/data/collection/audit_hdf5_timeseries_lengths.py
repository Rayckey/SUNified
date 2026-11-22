#!/usr/bin/env python3
"""Report per-demo HDF5 timeseries lengths without modifying the dataset."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import h5py


TIMESERIES_PREFIXES = ("actions", "obs/", "next_obs/", "dones", "rewards")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument(
        "--plan",
        action="append",
        default=[],
        type=Path,
        help="Fragment plan whose sources should be inspected (repeatable).",
    )
    parser.add_argument("--obs-key", default="state")
    parser.add_argument("--action-key", default="absolute_ee")
    parser.add_argument("--examples", type=int, default=10)
    return parser.parse_args()


def numeric_demo_names(dataset: h5py.File) -> list[str]:
    data = dataset.get("data")
    if data is None:
        return []
    return [
        f"demo_{index}"
        for index in sorted(
            int(name[5:])
            for name in data
            if name.startswith("demo_") and name[5:].isdigit()
        )
    ]


def timeseries_lengths(group: h5py.Group) -> dict[str, int]:
    lengths: dict[str, int] = {}

    def collect(name: str, item: h5py.Group | h5py.Dataset) -> None:
        if not isinstance(item, h5py.Dataset) or item.ndim < 1:
            return
        if name in {"actions", "dones", "rewards"} or any(
            name.startswith(prefix) for prefix in TIMESERIES_PREFIXES
        ):
            lengths[name] = int(item.shape[0])

    group.visititems(collect)
    return lengths


def inspect_path(
    path: Path,
    *,
    obs_key: str,
    action_key: str,
    example_limit: int,
) -> dict[str, Any]:
    mismatch_patterns: Counter[tuple[tuple[str, int], ...]] = Counter()
    mismatched_examples: list[dict[str, Any]] = []
    bc_stream_examples: list[dict[str, Any]] = []
    demo_count = 0
    mismatched_demo_count = 0
    bc_stream_mismatch_count = 0
    missing_bc_stream_count = 0

    with h5py.File(path, "r") as dataset:
        demo_names = numeric_demo_names(dataset)
        demo_count = len(demo_names)
        for demo_name in demo_names:
            group = dataset["data"][demo_name]
            lengths = timeseries_lengths(group)
            distinct_lengths = sorted(set(lengths.values()))
            if len(distinct_lengths) > 1:
                mismatched_demo_count += 1
                minimum = min(distinct_lengths)
                offsets = tuple(
                    sorted((name, length - minimum) for name, length in lengths.items())
                )
                mismatch_patterns[offsets] += 1
                if len(mismatched_examples) < example_limit:
                    mismatched_examples.append(
                        {"demo": demo_name, "lengths": lengths, "minimum": minimum}
                    )

            obs_path = f"obs/{obs_key}"
            action_path = f"actions/{action_key}"
            if obs_path not in group or action_path not in group:
                missing_bc_stream_count += 1
                if len(bc_stream_examples) < example_limit:
                    bc_stream_examples.append(
                        {
                            "demo": demo_name,
                            "obs_length": lengths.get(obs_path),
                            "action_length": lengths.get(action_path),
                            "reason": "missing",
                        }
                    )
            else:
                obs_length = int(group[obs_path].shape[0])
                action_length = int(group[action_path].shape[0])
                if obs_length != action_length or obs_length == 0:
                    bc_stream_mismatch_count += 1
                    if len(bc_stream_examples) < example_limit:
                        bc_stream_examples.append(
                            {
                                "demo": demo_name,
                                "obs_length": obs_length,
                                "action_length": action_length,
                                "reason": "length_mismatch_or_empty",
                            }
                        )

    rendered_patterns = [
        {
            "count": count,
            "offsets_from_minimum": dict(pattern),
        }
        for pattern, count in mismatch_patterns.most_common()
    ]
    return {
        "path": str(path.resolve()),
        "demo_count": demo_count,
        "timeseries_length_mismatch_demo_count": mismatched_demo_count,
        "bc_stream_length_mismatch_or_empty_demo_count": bc_stream_mismatch_count,
        "missing_bc_stream_demo_count": missing_bc_stream_count,
        "mismatch_patterns": rendered_patterns,
        "mismatched_examples": mismatched_examples,
        "bc_stream_examples": bc_stream_examples,
    }


def main() -> None:
    args = parse_args()
    paths = [path.expanduser().resolve() for path in args.paths]
    for plan_path in args.plan:
        plan = json.loads(plan_path.expanduser().resolve().read_text())
        paths.extend(
            Path(str(source["path"])).expanduser().resolve()
            for source in plan.get("sources", [])
        )
    if not paths:
        raise SystemExit("Provide at least one HDF5 path or --plan")
    if len(set(paths)) != len(paths):
        raise RuntimeError("Duplicate HDF5 path in requested audit")

    records = [
        inspect_path(
            path,
            obs_key=args.obs_key,
            action_key=args.action_key,
            example_limit=args.examples,
        )
        for path in paths
    ]
    totals = {
        "file_count": len(records),
        "demo_count": sum(record["demo_count"] for record in records),
        "timeseries_length_mismatch_demo_count": sum(
            record["timeseries_length_mismatch_demo_count"] for record in records
        ),
        "bc_stream_length_mismatch_or_empty_demo_count": sum(
            record["bc_stream_length_mismatch_or_empty_demo_count"]
            for record in records
        ),
        "missing_bc_stream_demo_count": sum(
            record["missing_bc_stream_demo_count"] for record in records
        ),
    }
    print(json.dumps({"totals": totals, "files": records}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
