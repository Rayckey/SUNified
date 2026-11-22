#!/usr/bin/env python3
"""Build hash-pinned Hang Cup fragment plans from the locked assembler spec."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import h5py

from kuafu.paths import require_external_artifact_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assembler", type=Path, required=True)
    parser.add_argument("--tranche-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def locked_specs(assembler: Path) -> dict[int, dict[str, Any]]:
    tree = ast.parse(assembler.read_text(), filename=str(assembler))
    assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "plan_specs"
            for target in node.targets
        )
    ]
    if len(assignments) != 1:
        raise RuntimeError("Assembler must contain exactly one plan_specs assignment")
    value = ast.literal_eval(assignments[0].value)
    if set(value) != {1, 2}:
        raise RuntimeError(f"Unexpected learned-program specs: {sorted(value)}")
    return value


def demo_ids(dataset: h5py.File) -> list[int]:
    if "data" not in dataset:
        raise RuntimeError("Source dataset has no data group")
    ids: list[int] = []
    for key in dataset["data"].keys():
        if not key.startswith("demo_") or not key.removeprefix("demo_").isdigit():
            raise RuntimeError(f"Unexpected demo key: {key}")
        ids.append(int(key.removeprefix("demo_")))
    return sorted(ids)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite existing plan: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    assembler = args.assembler.expanduser().resolve()
    tranche_root = require_external_artifact_path(
        args.tranche_root, "Fragmented tranche source root"
    )
    output_dir = require_external_artifact_path(
        args.output_dir, "Fragment plan output directory"
    )
    specs = locked_specs(assembler)

    for program_index in (1, 2):
        spec = specs[program_index]
        source_root = tranche_root / f"fragmented_program_{program_index:02d}_sources"
        if source_root.is_symlink() or not source_root.is_dir():
            raise FileNotFoundError(source_root)
        expected_names = list(spec["names"])
        actual_names = sorted(
            path.name for path in source_root.iterdir() if path.is_file()
        )
        if sorted(expected_names) != actual_names:
            raise RuntimeError(
                f"Source-file set mismatch for program {program_index}: "
                f"expected={sorted(expected_names)}, actual={actual_names}"
            )

        sources: list[dict[str, Any]] = []
        for name, count, seed, server in zip(
            spec["names"],
            spec["counts"],
            spec["seeds"],
            spec["servers"],
            strict=True,
        ):
            path = (source_root / name).resolve()
            if path.parent != source_root or path.is_symlink() or not path.is_file():
                raise RuntimeError(f"Unsafe source path: {path}")
            with h5py.File(path, "r") as dataset:
                if demo_ids(dataset) != list(range(count)):
                    raise RuntimeError(
                        f"Wrong demo IDs for {path}; expected 0..{count - 1}"
                    )
            sources.append(
                {
                    "demo_count": count,
                    "path": str(path),
                    "seed": seed,
                    "server": server,
                    "sha256": sha256_file(path),
                }
            )

        if sum(source["demo_count"] for source in sources) != 400:
            raise RuntimeError(f"Program {program_index} plan does not total 400 demos")
        plan = {
            "base_seed": spec["base_seed"],
            "demo_count": 400,
            "learned_program_index": program_index,
            "learned_program_path": spec["learned_program_path"],
            "provenance_source": "fragmented_single_gpu_offload",
            "robot_name": "kinova",
            "sources": sources,
            "task_name": "Kinova-Table-Gen3-Hang-Cup",
            "task_slug": "hang_cup",
            "tranche_index": 2,
        }
        output = output_dir / f"hang_cup_program_{program_index:02d}_fragmented_plan.json"
        write_json_atomic(output, plan)
        print(
            json.dumps(
                {
                    "program_index": program_index,
                    "demo_count": 400,
                    "plan": str(output),
                    "plan_sha256": sha256_file(output),
                    "source_count": len(sources),
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
