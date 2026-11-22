#!/usr/bin/env python3
"""Copy selected BC training artifacts into stable, human-named folders.

Given a sweep CSV with columns like:
    name,config,best_checkpoint

This script creates:
    log_filtered/<name>/

and copies into each folder:
    - best.pth
    - TensorBoard event files from the original run directory
    - the row's source config YAML (if available)
    - meta/config.yaml from the run directory (if available)
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

from kuafu.paths import resolve_artifact_path


def resolve_path(value: str, *, csv_path: Path) -> Path:
    """Resolve paths that may be relative to repo root or the CSV directory."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()

    for base in (Path.cwd(), csv_path.parent):
        candidate = (base / path).resolve()
        if candidate.exists():
            return candidate

    return (Path.cwd() / path).resolve()


def copy_file(src: Path, dst: Path, *, overwrite: bool) -> bool:
    """Copy one file, returning whether a copy actually happened."""
    if not src.exists():
        return False
    if dst.exists() and not overwrite:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy best checkpoints and TensorBoard logs from a sweep CSV."
    )
    parser.add_argument("--csv", required=True, help="Path to the sweep CSV.")
    parser.add_argument(
        "--output-dir",
        default="standalone/filtered_bc_runs",
        help="Destination directory for filtered logs.",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma-separated list of run names to include.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite files that already exist in the destination.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned copies without writing files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    csv_path = Path(args.csv).resolve()
    if not csv_path.exists():
        print(f"[ERROR] CSV not found: {csv_path}")
        return 1

    output_dir = resolve_artifact_path(args.output_dir, "Filtered BC artifact directory")
    only_names = None
    if args.only:
        only_names = {name.strip() for name in args.only.split(",") if name.strip()}

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print(f"[ERROR] No rows found in CSV: {csv_path}")
        return 1

    copied_runs = 0
    skipped_runs = 0

    for idx, row in enumerate(rows):
        name = (row.get("name") or f"run_{idx}").strip() or f"run_{idx}"
        if only_names and name not in only_names:
            continue

        best_checkpoint_value = (row.get("best_checkpoint") or "").strip()
        if not best_checkpoint_value:
            print(f"[SKIP] {name}: missing best_checkpoint")
            skipped_runs += 1
            continue

        checkpoint_path = resolve_path(best_checkpoint_value, csv_path=csv_path)
        if not checkpoint_path.exists():
            print(f"[SKIP] {name}: checkpoint not found -> {checkpoint_path}")
            skipped_runs += 1
            continue

        run_dir = checkpoint_path.parent.parent
        dest_dir = output_dir / name

        event_files = sorted(run_dir.glob("events.out.tfevents*"))
        row_config_value = (row.get("config") or "").strip()
        row_config_path = resolve_path(row_config_value, csv_path=csv_path) if row_config_value else None
        meta_config_path = run_dir / "meta" / "config.yaml"

        planned = [
            (checkpoint_path, dest_dir / "best.pth"),
            *((event_file, dest_dir / event_file.name) for event_file in event_files),
        ]
        if row_config_path and row_config_path.exists():
            planned.append((row_config_path, dest_dir / "sweep_config.yaml"))
        if meta_config_path.exists():
            planned.append((meta_config_path, dest_dir / "train_config.yaml"))

        if args.dry_run:
            print(f"[DRY RUN] {name}")
            for src, dst in planned:
                print(f"  {src} -> {dst}")
            copied_runs += 1
            continue

        dest_dir.mkdir(parents=True, exist_ok=True)
        copied_any = False
        for src, dst in planned:
            copied_any = copy_file(src, dst, overwrite=args.overwrite) or copied_any

        if not event_files:
            print(f"[WARN] {name}: no TensorBoard event files found in {run_dir}")

        if copied_any or dest_dir.exists():
            print(f"[OK] {name} -> {dest_dir}")
            copied_runs += 1
        else:
            print(f"[SKIP] {name}: nothing copied")
            skipped_runs += 1

    print()
    print("=== Done ===")
    print(f"CSV: {csv_path}")
    print(f"Output dir: {output_dir}")
    print(f"Processed runs: {copied_runs}")
    print(f"Skipped runs: {skipped_runs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
