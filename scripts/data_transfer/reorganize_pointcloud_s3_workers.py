#!/usr/bin/env python3
"""Reorganize uploaded pointcloud worker folders into one DemonstrationRecorder dataset.

The current S3 layout is expected to look like:

  PREFIX/<timestamp>_workerXX/datasets/episode_N/episode_N.h5
  PREFIX/<timestamp>_workerXX/datasets/metadata.json

This script plans or executes a server-side reorganization to:

  PREFIX/datasets/episode_M/episode_M.h5
  PREFIX/datasets/metadata.json

Dry run is the default. Use --execute only after reviewing the plan.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime as dt
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ENDPOINT_URL = "https://s3.wasabisys.com"
PROFILE = "kuafu"
BUCKET = "kuafu"
PREFIX = "pointclouds-match/FR3-FinRay-Table-Drawer-ff-best-ema-downsampled/"
IGNORED_NESTED_SUFFIXES = (".mp4", ".png")


def aws(args: list[str], *, capture: bool = True) -> str:
    cmd = ["aws", "--profile", PROFILE, "--endpoint-url", ENDPOINT_URL, *args]
    if capture:
        proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
        return proc.stdout
    subprocess.run(cmd, check=True)
    return ""


def normalize_prefix(prefix: str) -> str:
    prefix = prefix.removeprefix(f"s3://{BUCKET}/").lstrip("/")
    return prefix if prefix.endswith("/") else f"{prefix}/"


def s3_uri(key: str) -> str:
    return f"s3://{BUCKET}/{key}"


def list_objects(prefix: str) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    token: str | None = None

    while True:
        args = [
            "s3api",
            "list-objects-v2",
            "--bucket",
            BUCKET,
            "--prefix",
            prefix,
            "--output",
            "json",
        ]
        if token:
            args.extend(["--continuation-token", token])

        raw = aws(args)
        payload = json.loads(raw)
        objects.extend(payload.get("Contents", []))

        if not payload.get("IsTruncated"):
            break
        token = payload.get("NextContinuationToken")
        if not token:
            raise RuntimeError("S3 listing was truncated but no continuation token was returned")

    return objects


def read_s3_json(key: str) -> dict[str, Any]:
    raw = aws(["s3", "cp", s3_uri(key), "-", "--only-show-errors"])
    return json.loads(raw)


def copy_object(src_key: str, dst_key: str) -> None:
    aws(
        [
            "s3api",
            "copy-object",
            "--bucket",
            BUCKET,
            "--copy-source",
            f"{BUCKET}/{src_key}",
            "--key",
            dst_key,
        ]
    )


def delete_objects(keys: list[str]) -> None:
    for start in range(0, len(keys), 1000):
        chunk = keys[start : start + 1000]
        payload = {"Objects": [{"Key": key} for key in chunk], "Quiet": True}
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fp:
            json.dump(payload, fp)
            payload_path = Path(fp.name)
        try:
            aws(
                [
                    "s3api",
                    "delete-objects",
                    "--bucket",
                    BUCKET,
                    "--delete",
                    f"file://{payload_path}",
                ]
            )
        finally:
            payload_path.unlink(missing_ok=True)


def upload_json(key: str, payload: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fp:
        json.dump(payload, fp, indent=2)
        fp.write("\n")
        payload_path = Path(fp.name)
    try:
        aws(["s3", "cp", str(payload_path), s3_uri(key), "--only-show-errors"], capture=False)
    finally:
        payload_path.unlink(missing_ok=True)


def parse_layout(prefix: str, objects: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, str], list[str]]:
    source_episode_re = re.compile(
        rf"^{re.escape(prefix)}"
        r"(?P<worker>[^/]+)/datasets/episode_(?P<episode>\d+)/episode_(?P=episode)\.h5$"
    )
    source_metadata_re = re.compile(rf"^{re.escape(prefix)}(?P<worker>[^/]+)/datasets/metadata\.json$")
    destination_episode_re = re.compile(
        rf"^{re.escape(prefix)}datasets/episode_(?P<episode>\d+)/episode_(?P=episode)\.h5$"
    )
    destination_metadata_key = f"{prefix}datasets/metadata.json"

    episodes: list[dict[str, Any]] = []
    metadata_keys: dict[str, str] = {}
    unexpected: list[str] = []

    for obj in objects:
        key = obj["Key"]
        if key.endswith("/"):
            continue
        if key.endswith(IGNORED_NESTED_SUFFIXES):
            continue

        episode_match = source_episode_re.match(key)
        if episode_match:
            episodes.append(
                {
                    "source_key": key,
                    "source_worker": episode_match.group("worker"),
                    "source_episode": int(episode_match.group("episode")),
                    "size": int(obj["Size"]),
                    "last_modified": str(obj.get("LastModified", "")),
                }
            )
            continue

        metadata_match = source_metadata_re.match(key)
        if metadata_match:
            metadata_keys[metadata_match.group("worker")] = key
            continue

        if destination_episode_re.match(key) or key == destination_metadata_key:
            continue

        rel = key.removeprefix(prefix)
        if "/" in rel:
            unexpected.append(key)

    episodes.sort(key=lambda item: (item["source_worker"], item["source_episode"]))
    for new_episode, item in enumerate(episodes):
        item["new_episode"] = new_episode
        item["destination_key"] = f"{prefix}datasets/episode_{new_episode}/episode_{new_episode}.h5"
        item["metadata_relative_path"] = f"episode_{new_episode}/episode_{new_episode}.h5"

    return episodes, metadata_keys, unexpected


def load_worker_metadata(
    episodes: list[dict[str, Any]],
    metadata_keys: dict[str, str],
) -> dict[str, dict[str, Any]]:
    workers = sorted({item["source_worker"] for item in episodes})
    missing = [worker for worker in workers if worker not in metadata_keys]
    if missing:
        raise ValueError(f"missing metadata.json for worker(s): {', '.join(missing)}")

    metadata: dict[str, dict[str, Any]] = {}
    for worker in workers:
        metadata[worker] = read_s3_json(metadata_keys[worker])
    return metadata


def build_combined_metadata(
    prefix: str,
    episodes: list[dict[str, Any]],
    worker_metadata: dict[str, dict[str, Any]],
    metadata_keys: dict[str, str],
) -> dict[str, Any]:
    num_timesteps: list[int] = []
    episode_paths: list[str] = []
    episode_s3_uris: list[str] = []
    episode_records: list[dict[str, Any]] = []

    for item in episodes:
        worker = item["source_worker"]
        old_episode = item["source_episode"]
        worker_steps = worker_metadata[worker].get("num_timesteps")
        if not isinstance(worker_steps, list):
            raise ValueError(f"{metadata_keys[worker]} has no list field 'num_timesteps'")
        if old_episode >= len(worker_steps):
            raise ValueError(
                f"{metadata_keys[worker]} has {len(worker_steps)} timesteps entries, "
                f"but source episode_{old_episode} exists"
            )

        timesteps = int(worker_steps[old_episode])
        rel_path = item["metadata_relative_path"]
        dst_uri = s3_uri(item["destination_key"])
        num_timesteps.append(timesteps)
        episode_paths.append(rel_path)
        episode_s3_uris.append(dst_uri)
        episode_records.append(
            {
                "episode": item["new_episode"],
                "path": rel_path,
                "s3_uri": dst_uri,
                "num_timesteps": timesteps,
                "source_worker": worker,
                "source_episode": old_episode,
                "source_s3_uri": s3_uri(item["source_key"]),
            }
        )

    worker_summaries: list[dict[str, Any]] = []
    for worker in sorted(worker_metadata):
        metadata = worker_metadata[worker]
        worker_episodes = [item for item in episodes if item["source_worker"] == worker]
        worker_steps = metadata.get("num_timesteps", [])
        declared = metadata.get("num_episodes")
        worker_summaries.append(
            {
                "worker": worker,
                "source_metadata_s3_uri": s3_uri(metadata_keys[worker]),
                "source_episode_count": len(worker_episodes),
                "metadata_num_episodes": declared,
                "metadata_num_timesteps_entries": len(worker_steps) if isinstance(worker_steps, list) else None,
            }
        )

    return {
        "num_timesteps": num_timesteps,
        "num_episodes": len(num_timesteps),
        "episode_paths": episode_paths,
        "episode_s3_uris": episode_s3_uris,
        "episodes": episode_records,
        "reorganized_from": {
            "bucket": BUCKET,
            "prefix": prefix,
            "endpoint_url": ENDPOINT_URL,
            "source_metadata_keys": [metadata_keys[worker] for worker in sorted(metadata_keys)],
            "workers": worker_summaries,
        },
        "generated_at_utc": dt.datetime.now(dt.UTC).isoformat(),
    }


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
    prefix: str,
    episodes: list[dict[str, Any]],
    metadata_keys: dict[str, str],
    unexpected: list[str],
    combined_metadata: dict[str, Any],
    object_sizes: dict[str, int],
) -> tuple[list[dict[str, Any]], list[tuple[str, int, int]]]:
    destination_metadata_key = f"{prefix}datasets/metadata.json"
    existing_done = 0
    copy_plan: list[dict[str, Any]] = []
    collisions: list[tuple[str, int, int]] = []

    for item in episodes:
        existing_size = object_sizes.get(item["destination_key"])
        if existing_size is None:
            copy_plan.append(item)
        elif existing_size == item["size"]:
            existing_done += 1
        else:
            collisions.append((item["destination_key"], item["size"], existing_size))

    worker_counts: dict[str, int] = {}
    for item in episodes:
        worker_counts[item["source_worker"]] = worker_counts.get(item["source_worker"], 0) + 1

    print(f"source prefix:      {s3_uri(prefix)}")
    print(f"target dataset:    {s3_uri(prefix + 'datasets/')}")
    print(f"workers found:     {len(worker_counts)}")
    print(f"metadata files:    {len(metadata_keys)}")
    print(f"episode files:     {len(episodes)}")
    print(f"episodes to copy:  {len(copy_plan)}")
    print(f"already in target: {existing_done}")
    print(f"unexpected files:  {len(unexpected)}")
    print(f"target metadata:   {s3_uri(destination_metadata_key)}")
    if destination_metadata_key in object_sizes:
        print("target metadata already exists and would be overwritten on --execute")
    print()

    print("worker episode counts:")
    for worker, count in sorted(worker_counts.items()):
        print(f"  {worker}: {count}")
    print()

    sample = episodes[:3] + ([{"source_key": "...", "destination_key": "..."}] if len(episodes) > 6 else []) + episodes[-3:]
    print("sample renumbering:")
    for item in sample:
        print(f"  {item['source_key']} -> {item['destination_key']}")
    print()

    print("combined metadata preview:")
    print(f"  num_episodes: {combined_metadata['num_episodes']}")
    print(f"  num_timesteps entries: {len(combined_metadata['num_timesteps'])}")
    if combined_metadata["episodes"]:
        print(f"  first path: {combined_metadata['episodes'][0]['path']}")
        print(f"  last path:  {combined_metadata['episodes'][-1]['path']}")
    print()

    return copy_plan, collisions


def execute_plan(
    prefix: str,
    episodes: list[dict[str, Any]],
    copy_plan: list[dict[str, Any]],
    combined_metadata: dict[str, Any],
    metadata_keys: dict[str, str],
    *,
    workers: int,
    keep_source: bool,
) -> None:
    if copy_plan:
        print(f"copying {len(copy_plan)} objects with {workers} workers")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_to_item = {
                pool.submit(copy_object, item["source_key"], item["destination_key"]): item
                for item in copy_plan
            }
            for done, future in enumerate(as_completed(future_to_item), start=1):
                item = future_to_item[future]
                try:
                    future.result()
                except Exception as exc:
                    raise RuntimeError(
                        f"copy failed: {item['source_key']} -> {item['destination_key']}"
                    ) from exc
                if done % 25 == 0 or done == len(copy_plan):
                    print(f"  copied {done}/{len(copy_plan)}")

    post_copy_sizes = {obj["Key"]: int(obj["Size"]) for obj in list_objects(prefix)}
    mismatches = [
        (item["destination_key"], item["size"], post_copy_sizes.get(item["destination_key"]))
        for item in episodes
        if post_copy_sizes.get(item["destination_key"]) != item["size"]
    ]
    if mismatches:
        print("destination verification failed:", file=sys.stderr)
        for key, expected, actual in mismatches[:20]:
            print(f"  {key}: expected={expected} actual={actual}", file=sys.stderr)
        raise RuntimeError("destination verification failed")
    print(f"verified {len(episodes)} destination episode files")

    metadata_key = f"{prefix}datasets/metadata.json"
    upload_json(metadata_key, combined_metadata)
    print(f"uploaded {s3_uri(metadata_key)}")

    if keep_source:
        print("keeping source worker objects because --keep-source was set")
        return

    delete_keys = [item["source_key"] for item in episodes] + list(metadata_keys.values())
    print(f"deleting {len(delete_keys)} old worker objects")
    delete_objects(delete_keys)

    remaining_nested = [
        obj["Key"]
        for obj in list_objects(prefix)
        if obj["Key"] not in {item["destination_key"] for item in episodes}
        and obj["Key"] != metadata_key
        and not obj["Key"].endswith(IGNORED_NESTED_SUFFIXES)
        and "/" in obj["Key"].removeprefix(prefix).strip("/")
    ]
    if remaining_nested:
        print("old nested objects still remain:", file=sys.stderr)
        for key in remaining_nested[:20]:
            print(f"  {key}", file=sys.stderr)
        raise RuntimeError("old nested objects still remain")

    print("done: worker folders have been consolidated into datasets/")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Consolidate S3 pointcloud worker folders into one datasets/ directory."
    )
    parser.add_argument("--prefix", default=PREFIX, help="bucket-relative prefix or s3://kuafu/... URI")
    parser.add_argument("--execute", action="store_true", help="perform S3 copies, metadata upload, and source deletion")
    parser.add_argument("--keep-source", action="store_true", help="on --execute, leave old worker objects in place")
    parser.add_argument("--workers", type=int, default=16, help="parallel S3 copy workers for --execute")
    args = parser.parse_args()

    prefix = normalize_prefix(args.prefix)
    objects = list_objects(prefix)
    object_sizes = {obj["Key"]: int(obj["Size"]) for obj in objects}
    episodes, metadata_keys, unexpected = parse_layout(prefix, objects)

    if unexpected:
        print("Unexpected nested files found. Refusing to continue until inspected:", file=sys.stderr)
        for key in unexpected[:50]:
            print(f"  {key}", file=sys.stderr)
        if len(unexpected) > 50:
            print(f"  ... and {len(unexpected) - 50} more", file=sys.stderr)
        return 2

    if not episodes:
        print(f"No source worker episode files found under {s3_uri(prefix)}")
        return 0

    try:
        worker_metadata = load_worker_metadata(episodes, metadata_keys)
        metadata_warnings = validate_metadata_counts(episodes, worker_metadata)
        combined_metadata = build_combined_metadata(prefix, episodes, worker_metadata, metadata_keys)
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"metadata error: {exc}", file=sys.stderr)
        return 3

    copy_plan, collisions = summarize_plan(
        prefix,
        episodes,
        metadata_keys,
        unexpected,
        combined_metadata,
        object_sizes,
    )

    if metadata_warnings:
        print("metadata count warnings:")
        for warning in metadata_warnings:
            print(f"  {warning}")
        print()

    if collisions:
        print("Refusing to overwrite destination episode files with mismatched sizes:", file=sys.stderr)
        for key, src_size, dst_size in collisions[:20]:
            print(f"  {key}: source={src_size} destination={dst_size}", file=sys.stderr)
        return 4

    if not args.execute:
        print("DRY RUN ONLY: no S3 objects were copied, uploaded, or deleted.")
        print("Rerun with --execute after reviewing the plan.")
        return 0

    try:
        execute_plan(
            prefix,
            episodes,
            copy_plan,
            combined_metadata,
            metadata_keys,
            workers=args.workers,
            keep_source=args.keep_source,
        )
    except RuntimeError as exc:
        print(f"execute failed: {exc}", file=sys.stderr)
        return 5

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
