#!/usr/bin/env python3
"""Reorganize local pointcloud worker folders into one DemonstrationRecorder dataset.

This is the local-filesystem counterpart of reorganize_pointcloud_s3_workers.py.
It expects worker outputs like:

  SOURCE_ROOT/<timestamp>_workerXX/datasets/episode_N/episode_N.h5
  SOURCE_ROOT/<timestamp>_workerXX/datasets/metadata.json

and creates:

  OUTPUT_ROOT/datasets/episode_M/episode_M.h5
  OUTPUT_ROOT/datasets/metadata.json

Dry run is the default. Use --execute after reviewing the plan.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Literal

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_ROOT = _PROJECT_ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from kuafu.paths import require_external_artifact_path


IGNORED_NESTED_SUFFIXES = (".mp4", ".png")
CopyMethod = Literal["copy", "hardlink", "move"]


def discover_objects(root: Path) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    if not root.exists():
        raise FileNotFoundError(f"source root does not exist: {root}")

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        stat = path.stat()
        objects.append(
            {
                "path": path,
                "rel": path.relative_to(root).as_posix(),
                "size": stat.st_size,
                "mtime": dt.datetime.fromtimestamp(stat.st_mtime, dt.UTC).isoformat(),
            }
        )
    return objects


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def file_uri(path: Path) -> str:
    return path.resolve().as_uri()


def normalize_s3_prefix(prefix: str | None) -> str | None:
    if not prefix:
        return None
    return prefix if prefix.endswith("/") else f"{prefix}/"


def parse_layout(
    source_root: Path,
    output_root: Path,
    objects: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Path], list[Path]]:
    source_episode_re = re.compile(
        r"^(?P<worker>[^/]+)/datasets/episode_(?P<episode>\d+)/episode_(?P=episode)\.h5$"
    )
    source_metadata_re = re.compile(r"^(?P<worker>[^/]+)/datasets/metadata\.json$")
    destination_episode_re = re.compile(
        r"^datasets/episode_(?P<episode>\d+)/episode_(?P=episode)\.h5$"
    )

    episodes: list[dict[str, Any]] = []
    metadata_paths: dict[str, Path] = {}
    unexpected: list[Path] = []

    same_root = source_root.resolve() == output_root.resolve()
    for obj in objects:
        rel = obj["rel"]
        path = obj["path"]
        if rel.endswith(IGNORED_NESTED_SUFFIXES):
            continue

        episode_match = source_episode_re.match(rel)
        if episode_match:
            episodes.append(
                {
                    "source_path": path,
                    "source_rel": rel,
                    "source_worker": episode_match.group("worker"),
                    "source_episode": int(episode_match.group("episode")),
                    "size": int(obj["size"]),
                    "last_modified": obj["mtime"],
                }
            )
            continue

        metadata_match = source_metadata_re.match(rel)
        if metadata_match:
            metadata_paths[metadata_match.group("worker")] = path
            continue

        if same_root and (destination_episode_re.match(rel) or rel == "datasets/metadata.json"):
            continue

        if "/" in rel:
            unexpected.append(path)

    episodes.sort(key=lambda item: (item["source_worker"], item["source_episode"]))
    for new_episode, item in enumerate(episodes):
        item["new_episode"] = new_episode
        item["destination_path"] = output_root / "datasets" / f"episode_{new_episode}" / f"episode_{new_episode}.h5"
        item["metadata_relative_path"] = f"episode_{new_episode}/episode_{new_episode}.h5"

    return episodes, metadata_paths, unexpected


def load_worker_metadata(
    episodes: list[dict[str, Any]],
    metadata_paths: dict[str, Path],
) -> dict[str, dict[str, Any]]:
    workers = sorted({item["source_worker"] for item in episodes})
    missing = [worker for worker in workers if worker not in metadata_paths]
    if missing:
        raise ValueError(f"missing metadata.json for worker(s): {', '.join(missing)}")

    metadata: dict[str, dict[str, Any]] = {}
    for worker in workers:
        metadata[worker] = read_json(metadata_paths[worker])
    return metadata


def build_combined_metadata(
    source_root: Path,
    output_root: Path,
    episodes: list[dict[str, Any]],
    worker_metadata: dict[str, dict[str, Any]],
    metadata_paths: dict[str, Path],
    *,
    s3_prefix: str | None,
) -> dict[str, Any]:
    num_timesteps: list[int] = []
    episode_paths: list[str] = []
    episode_s3_uris: list[str] = []
    episode_file_uris: list[str] = []
    episode_records: list[dict[str, Any]] = []

    for item in episodes:
        worker = item["source_worker"]
        old_episode = item["source_episode"]
        worker_steps = worker_metadata[worker].get("num_timesteps")
        if not isinstance(worker_steps, list):
            raise ValueError(f"{metadata_paths[worker]} has no list field 'num_timesteps'")
        if old_episode >= len(worker_steps):
            raise ValueError(
                f"{metadata_paths[worker]} has {len(worker_steps)} timesteps entries, "
                f"but source episode_{old_episode} exists"
            )

        timesteps = int(worker_steps[old_episode])
        rel_path = item["metadata_relative_path"]
        dst_path = item["destination_path"]
        dst_file_uri = file_uri(dst_path)
        dst_s3_uri = f"{s3_prefix}datasets/{rel_path}" if s3_prefix else None

        num_timesteps.append(timesteps)
        episode_paths.append(rel_path)
        episode_file_uris.append(dst_file_uri)
        if dst_s3_uri:
            episode_s3_uris.append(dst_s3_uri)

        record = {
            "episode": item["new_episode"],
            "path": rel_path,
            "file_uri": dst_file_uri,
            "num_timesteps": timesteps,
            "source_worker": worker,
            "source_episode": old_episode,
            "source_path": str(item["source_path"]),
            "source_file_uri": file_uri(item["source_path"]),
        }
        if dst_s3_uri:
            record["s3_uri"] = dst_s3_uri
        episode_records.append(record)

    worker_summaries: list[dict[str, Any]] = []
    for worker in sorted(worker_metadata):
        metadata = worker_metadata[worker]
        worker_episodes = [item for item in episodes if item["source_worker"] == worker]
        worker_steps = metadata.get("num_timesteps", [])
        declared = metadata.get("num_episodes")
        worker_summaries.append(
            {
                "worker": worker,
                "source_metadata_path": str(metadata_paths[worker]),
                "source_episode_count": len(worker_episodes),
                "metadata_num_episodes": declared,
                "metadata_num_timesteps_entries": len(worker_steps) if isinstance(worker_steps, list) else None,
            }
        )

    payload: dict[str, Any] = {
        "num_timesteps": num_timesteps,
        "num_episodes": len(num_timesteps),
        "episode_paths": episode_paths,
        "episode_file_uris": episode_file_uris,
        "episodes": episode_records,
        "reorganized_from": {
            "source_root": str(source_root),
            "output_root": str(output_root),
            "source_metadata_paths": [str(metadata_paths[worker]) for worker in sorted(metadata_paths)],
            "workers": worker_summaries,
        },
        "generated_at_utc": dt.datetime.now(dt.UTC).isoformat(),
    }
    if s3_prefix:
        payload["episode_s3_uris"] = episode_s3_uris
        payload["reorganized_from"]["s3_prefix"] = s3_prefix
    return payload


def validate_metadata_counts(
    episodes: list[dict[str, Any]],
    worker_metadata: dict[str, dict[str, Any]],
) -> list[str]:
    warnings: list[str] = []
    for worker, metadata in sorted(worker_metadata.items()):
        worker_episode_count = sum(1 for item in episodes if item["source_worker"] == worker)
        metadata_num_episodes = metadata.get("num_episodes")
        num_timesteps = metadata.get("num_timesteps")
        timestep_count = len(num_timesteps) if isinstance(num_timesteps, list) else None

        if metadata_num_episodes != worker_episode_count:
            warnings.append(
                f"{worker}: metadata num_episodes={metadata_num_episodes}, "
                f"found episode files={worker_episode_count}"
            )
        if timestep_count != worker_episode_count:
            warnings.append(
                f"{worker}: metadata num_timesteps entries={timestep_count}, "
                f"found episode files={worker_episode_count}"
            )
    return warnings


def summarize_plan(
    source_root: Path,
    output_root: Path,
    episodes: list[dict[str, Any]],
    metadata_paths: dict[str, Path],
    unexpected: list[Path],
    combined_metadata: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[tuple[Path, int, int]]]:
    destination_metadata_path = output_root / "datasets" / "metadata.json"
    existing_done = 0
    copy_plan: list[dict[str, Any]] = []
    collisions: list[tuple[Path, int, int]] = []

    for item in episodes:
        dst_path: Path = item["destination_path"]
        if not dst_path.exists():
            copy_plan.append(item)
            continue
        existing_size = dst_path.stat().st_size
        if existing_size == item["size"]:
            existing_done += 1
        else:
            collisions.append((dst_path, item["size"], existing_size))

    worker_counts: dict[str, int] = {}
    for item in episodes:
        worker_counts[item["source_worker"]] = worker_counts.get(item["source_worker"], 0) + 1

    print(f"source root:       {source_root}")
    print(f"target dataset:    {output_root / 'datasets'}")
    print(f"workers found:     {len(worker_counts)}")
    print(f"metadata files:    {len(metadata_paths)}")
    print(f"episode files:     {len(episodes)}")
    print(f"episodes to write: {len(copy_plan)}")
    print(f"already in target: {existing_done}")
    print(f"unexpected files:  {len(unexpected)}")
    print(f"target metadata:   {destination_metadata_path}")
    if destination_metadata_path.exists():
        print("target metadata already exists and would be overwritten on --execute")
    print()

    print("worker episode counts:")
    for worker, count in sorted(worker_counts.items()):
        print(f"  {worker}: {count}")
    print()

    sample = episodes[:3] + ([{"source_path": "...", "destination_path": "..."}] if len(episodes) > 6 else []) + episodes[-3:]
    print("sample renumbering:")
    for item in sample:
        print(f"  {item['source_path']} -> {item['destination_path']}")
    print()

    print("combined metadata preview:")
    print(f"  num_episodes: {combined_metadata['num_episodes']}")
    print(f"  num_timesteps entries: {len(combined_metadata['num_timesteps'])}")
    if combined_metadata["episodes"]:
        print(f"  first path: {combined_metadata['episodes'][0]['path']}")
        print(f"  last path:  {combined_metadata['episodes'][-1]['path']}")
    print()

    return copy_plan, collisions


def materialize_episode(src: Path, dst: Path, method: CopyMethod) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if method == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            shutil.copy2(src, dst)
            return
    if method == "copy":
        shutil.copy2(src, dst)
        return
    if method == "move":
        shutil.move(str(src), str(dst))
        return
    raise ValueError(f"unknown method: {method}")


def execute_plan(
    episodes: list[dict[str, Any]],
    copy_plan: list[dict[str, Any]],
    combined_metadata: dict[str, Any],
    output_root: Path,
    *,
    method: CopyMethod,
) -> None:
    if copy_plan:
        print(f"writing {len(copy_plan)} episode files using method={method}")
        for done, item in enumerate(copy_plan, start=1):
            materialize_episode(item["source_path"], item["destination_path"], method)
            if done % 25 == 0 or done == len(copy_plan):
                print(f"  wrote {done}/{len(copy_plan)}")

    mismatches = [
        (item["destination_path"], item["size"], item["destination_path"].stat().st_size if item["destination_path"].exists() else None)
        for item in episodes
        if (not item["destination_path"].exists()) or item["destination_path"].stat().st_size != item["size"]
    ]
    if mismatches:
        print("destination verification failed:", file=sys.stderr)
        for path, expected, actual in mismatches[:20]:
            print(f"  {path}: expected={expected} actual={actual}", file=sys.stderr)
        raise RuntimeError("destination verification failed")
    print(f"verified {len(episodes)} destination episode files")

    metadata_path = output_root / "datasets" / "metadata.json"
    write_json(metadata_path, combined_metadata)
    print(f"wrote {metadata_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Consolidate local pointcloud worker folders into one datasets/ directory."
    )
    parser.add_argument("--source-root", required=True, type=Path, help="directory containing *_workerXX folders")
    parser.add_argument("--output-root", type=Path, help="dataset root to receive datasets/; defaults to source-root")
    parser.add_argument("--s3-prefix", help="optional final s3://.../ prefix used to populate episode_s3_uris")
    parser.add_argument("--method", choices=["copy", "hardlink", "move"], default="hardlink")
    parser.add_argument("--execute", action="store_true", help="write episode files and metadata")
    args = parser.parse_args()

    source_root = require_external_artifact_path(args.source_root, "Point-cloud source root")
    output_root = require_external_artifact_path(
        args.output_root or args.source_root, "Point-cloud output root"
    )
    s3_prefix = normalize_s3_prefix(args.s3_prefix)

    try:
        objects = discover_objects(source_root)
        episodes, metadata_paths, unexpected = parse_layout(source_root, output_root, objects)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 2

    if unexpected:
        print("Unexpected nested files found. Refusing to continue until inspected:", file=sys.stderr)
        for path in unexpected[:50]:
            print(f"  {path}", file=sys.stderr)
        if len(unexpected) > 50:
            print(f"  ... and {len(unexpected) - 50} more", file=sys.stderr)
        return 2

    if not episodes:
        print(f"No source worker episode files found under {source_root}")
        return 0

    try:
        worker_metadata = load_worker_metadata(episodes, metadata_paths)
        metadata_warnings = validate_metadata_counts(episodes, worker_metadata)
        combined_metadata = build_combined_metadata(
            source_root,
            output_root,
            episodes,
            worker_metadata,
            metadata_paths,
            s3_prefix=s3_prefix,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"metadata error: {exc}", file=sys.stderr)
        return 3

    copy_plan, collisions = summarize_plan(
        source_root,
        output_root,
        episodes,
        metadata_paths,
        unexpected,
        combined_metadata,
    )

    if metadata_warnings:
        print("metadata count warnings:")
        for warning in metadata_warnings:
            print(f"  {warning}")
        print()

    if collisions:
        print("Refusing to overwrite destination episode files with mismatched sizes:", file=sys.stderr)
        for path, src_size, dst_size in collisions[:20]:
            print(f"  {path}: source={src_size} destination={dst_size}", file=sys.stderr)
        return 4

    if not args.execute:
        print("DRY RUN ONLY: no local files were written.")
        print("Rerun with --execute after reviewing the plan.")
        return 0

    try:
        execute_plan(episodes, copy_plan, combined_metadata, output_root, method=args.method)
    except RuntimeError as exc:
        print(f"execute failed: {exc}", file=sys.stderr)
        return 5

    print("done: local worker folders have been consolidated into datasets/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
