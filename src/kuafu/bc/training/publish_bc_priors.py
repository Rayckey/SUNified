#!/usr/bin/env python3
"""Publish trained BC prior sweep outputs into an external artifact tree."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]


def _default_priors_root() -> Path:
    prior_root = os.environ.get("KUAFU_PRIOR_ARTIFACT_ROOT", "").strip()
    if prior_root:
        return Path(prior_root).expanduser().resolve() / "published_priors"
    task_root = os.environ.get("KUAFU_TASK_ARTIFACT_ROOT", "").strip()
    if task_root:
        return Path(task_root).expanduser().resolve() / "prior_artifacts" / "published_priors"
    configured = os.environ.get("KUAFU_ARTIFACT_ROOT", "").strip()
    base = Path(configured).expanduser().resolve() if configured else REPO_ROOT.parent / "KuafuArtifacts"
    return base / "standalone" / "published_priors"


def _require_external_output(path: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if resolved == REPO_ROOT or REPO_ROOT in resolved.parents:
        raise ValueError(f"{label} must be outside the source project ({REPO_ROOT}); got {resolved}")
    return resolved


def _safe_token(value: str) -> str:
    text = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value).strip("._-")
    return text or "prior"


def _relative_to_repo(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _resolve_path(value: str, *, csv_path: Path) -> Path:
    raw = (value or "").strip()
    if not raw:
        raise ValueError("empty path")
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path.resolve()
    for base in (REPO_ROOT, csv_path.parent):
        candidate = (base / path).resolve()
        if candidate.exists():
            return candidate
    return (REPO_ROOT / path).resolve()


def _infer_task_slug_from_csv(csv_path: Path) -> str:
    parts = csv_path.resolve().parts
    if "prior_sweep_csv" in parts:
        index = parts.index("prior_sweep_csv")
        if index + 2 < len(parts) and parts[index + 1] == "generated":
            stem = csv_path.stem
            return stem.rsplit("_", 1)[0] if "_" in stem else stem
    return csv_path.stem


def _infer_task_slug_from_config(config_path: Path) -> str | None:
    parts = config_path.resolve().parts
    if "prior_configs" not in parts:
        return None
    index = parts.index("prior_configs")
    if index + 1 < len(parts):
        return parts[index + 1]
    return None


def _copy_required_file(src: Path, dst: Path, *, overwrite: bool) -> None:
    if src.resolve() == dst.resolve() and src.is_file():
        return
    if dst.exists() and not overwrite:
        return
    if not src.exists():
        if dst.exists():
            return
        raise FileNotFoundError(f"Required source file is missing: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _move_required_file(src: Path, dst: Path, *, overwrite: bool) -> None:
    if not src.exists():
        if dst.exists():
            return
        raise FileNotFoundError(f"Required source file is missing: {src}")
    if src.resolve() == dst.resolve():
        return
    if dst.exists():
        if not overwrite:
            return
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))


def _load_rows(csv_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header row: {csv_path}")
        fieldnames = list(reader.fieldnames)
        rows = list(reader)
    for row in rows:
        row.pop(None, None)
    return fieldnames, rows


def _write_rows(csv_path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    tmp_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
    try:
        with tmp_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp_path, csv_path)
    except Exception:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise


def publish_priors(
    *,
    csv_path: Path,
    robot_id: str,
    task_slug: str | None,
    priors_root: Path,
    overwrite: bool,
    allow_missing_events: bool,
) -> list[Path]:
    fieldnames, rows = _load_rows(csv_path)
    for column in ("prior_path", "published_prior_dir"):
        if column not in fieldnames:
            fieldnames.append(column)

    published: list[Path] = []
    for row in rows:
        name = _safe_token((row.get("name") or "").strip())
        best_raw = (row.get("best_checkpoint") or "").strip()
        if not best_raw:
            print(f"[publish] skip {name}: no best_checkpoint")
            continue

        best_checkpoint = _resolve_path(best_raw, csv_path=csv_path)
        run_dir = best_checkpoint.parent.parent if best_checkpoint.parent.name == "checkpoints" else best_checkpoint.parent
        config_path = _resolve_path(row.get("config") or "", csv_path=csv_path)
        row_task_slug = task_slug or _infer_task_slug_from_config(config_path) or _infer_task_slug_from_csv(csv_path)
        target_dir = priors_root.resolve() / _safe_token(robot_id) / _safe_token(row_task_slug) / name
        target_best = target_dir / "best.pth"

        _move_required_file(best_checkpoint, target_best, overwrite=overwrite)
        _copy_required_file(config_path, target_dir / "sweep_config.yaml", overwrite=overwrite)

        train_config = run_dir / "meta" / "config.yaml"
        if not train_config.exists():
            train_config = run_dir / "train_config.yaml"
        if not train_config.exists() and best_checkpoint == target_best:
            # A resumed already-published run can restore a missing config from
            # its immutable sweep recipe without retraining the checkpoint.
            train_config = config_path
        _copy_required_file(train_config, target_dir / "train_config.yaml", overwrite=overwrite)

        event_files = sorted(run_dir.glob("events.out.tfevents*"))
        if not event_files:
            existing_event_files = sorted(target_dir.glob("events.out.tfevents*"))
            if not existing_event_files and not allow_missing_events:
                raise FileNotFoundError(f"No TensorBoard events.out.tfevents* files found in {run_dir}")
        for event_file in event_files:
            _copy_required_file(event_file, target_dir / event_file.name, overwrite=overwrite)

        row["prior_path"] = _relative_to_repo(target_best)
        row["best_checkpoint"] = _relative_to_repo(target_best)
        row["published_prior_dir"] = _relative_to_repo(target_dir)
        published.append(target_dir)
        print(f"[publish] {name} -> {target_dir}")

    _write_rows(csv_path, fieldnames, rows)
    return published


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish trained BC priors into an external artifact tree.")
    parser.add_argument("--csv", required=True, help="External sweep CSV filled by launch_sweep.py.")
    parser.add_argument("--robot_id", required=True, help="Robot id for the published-prior namespace.")
    parser.add_argument("--task_slug", default=None, help="Optional task slug for the published-prior namespace.")
    parser.add_argument("--priors_root", default=None, help="External destination root for published priors.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing published prior files.")
    parser.add_argument("--allow_missing_events", action="store_true", help="Do not fail if TensorBoard event files are absent.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = _require_external_output(Path(args.csv), label="--csv")
    priors_root = _require_external_output(
        Path(args.priors_root) if args.priors_root else _default_priors_root(),
        label="--priors_root",
    )
    published = publish_priors(
        csv_path=csv_path,
        robot_id=args.robot_id,
        task_slug=args.task_slug,
        priors_root=priors_root,
        overwrite=args.overwrite,
        allow_missing_events=args.allow_missing_events,
    )
    print(f"[publish] Published {len(published)} prior(s).")


if __name__ == "__main__":
    main()
