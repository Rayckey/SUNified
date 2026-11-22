#!/usr/bin/env python3
"""Report best residual checkpoints from TensorBoard success-rate EMA scalars.

This intentionally avoids TensorBoard/TensorFlow dependencies. It reads the
small subset of TFRecord + protobuf needed for scalar summary events emitted by
PyTorch/TensorBoard writers.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_ROOT = _PROJECT_ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from kuafu.paths import require_external_artifact_path


CHECKPOINT_TRUE_SUCCESS_TAGS = (
    "losses/kuafu_success_rate_ema",
    "Episode/Kuafu/success_rate_ema",
    "Kuafu/success_rate_ema",
)
TOP_LEVEL_TRUE_SUCCESS_RE = re.compile(r"^(?:Global|Run_\d+)/success_rate_ema$")
TRUE_SUCCESS_TAG_RE = re.compile(
    r"^(?:"
    r"losses/kuafu_success_rate_ema|"
    r"Episode/Kuafu/success_rate_ema|"
    r"Kuafu/success_rate_ema|"
    r"Global/success_rate_ema|"
    r"Run_\d+/success_rate_ema|"
    r"Kuafu/best_success_rate_ema_so_far"
    r")$"
)
RUN_TRUE_SUCCESS_RE = re.compile(r"^Run_(\d+)/success_rate_ema$")
EP_RE = re.compile(r"(?:^|_)ep_(\d+)(?:_|\.pth$)")


@dataclass(frozen=True)
class ScalarPoint:
    event_file: Path
    run_dir: Path
    tag: str
    scalar_index: int
    step: int
    value: float


@dataclass(frozen=True)
class CheckpointCandidate:
    checkpoint: str
    ep: int | None
    size: int | None


@dataclass(frozen=True)
class CheckpointScore:
    checkpoint: str
    checkpoint_ep: int | None
    checkpoint_size: int | None
    true_success_rate: float
    metric_tag: str
    metric_index: int | None
    metric_step: int | None
    metric_epoch_delta: int | None
    run_dir: str
    event_file: str
    source: str


def read_varint(buf: bytes, pos: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        if pos >= len(buf):
            raise ValueError("truncated varint")
        b = buf[pos]
        pos += 1
        value |= (b & 0x7F) << shift
        if not b & 0x80:
            return value, pos
        shift += 7
        if shift > 70:
            raise ValueError("varint too long")


def skip_field(buf: bytes, pos: int, wire_type: int) -> int:
    if wire_type == 0:
        _, pos = read_varint(buf, pos)
        return pos
    if wire_type == 1:
        return pos + 8
    if wire_type == 2:
        size, pos = read_varint(buf, pos)
        return pos + size
    if wire_type == 5:
        return pos + 4
    raise ValueError(f"unsupported protobuf wire type: {wire_type}")


def iter_fields(buf: bytes) -> Iterator[tuple[int, int, int, int]]:
    pos = 0
    while pos < len(buf):
        key, pos = read_varint(buf, pos)
        field_no = key >> 3
        wire_type = key & 0x07
        value_pos = pos
        pos = skip_field(buf, pos, wire_type)
        if pos > len(buf):
            raise ValueError("protobuf field extends past buffer")
        yield field_no, wire_type, value_pos, pos


def parse_tensor_scalar(buf: bytes) -> float | None:
    dtype = None
    tensor_content = None
    float_values: list[float] = []
    double_values: list[float] = []
    int_values: list[int] = []
    int64_values: list[int] = []
    bool_values: list[int] = []

    for field_no, wire_type, value_pos, end_pos in iter_fields(buf):
        if field_no == 1 and wire_type == 0:
            dtype, _ = read_varint(buf, value_pos)
        elif field_no == 4 and wire_type == 2:
            size, pos = read_varint(buf, value_pos)
            tensor_content = buf[pos : pos + size]
        elif field_no == 5:
            if wire_type == 5:
                float_values.append(struct.unpack("<f", buf[value_pos:end_pos])[0])
            elif wire_type == 2:
                size, pos = read_varint(buf, value_pos)
                payload = buf[pos : pos + size]
                float_values.extend(
                    struct.unpack("<" + "f" * (len(payload) // 4), payload[: len(payload) // 4 * 4])
                )
        elif field_no == 6:
            if wire_type == 1:
                double_values.append(struct.unpack("<d", buf[value_pos:end_pos])[0])
            elif wire_type == 2:
                size, pos = read_varint(buf, value_pos)
                payload = buf[pos : pos + size]
                double_values.extend(
                    struct.unpack("<" + "d" * (len(payload) // 8), payload[: len(payload) // 8 * 8])
                )
        elif field_no == 7:
            if wire_type == 0:
                v, _ = read_varint(buf, value_pos)
                int_values.append(v)
            elif wire_type == 2:
                size, pos = read_varint(buf, value_pos)
                limit = pos + size
                while pos < limit:
                    v, pos = read_varint(buf, pos)
                    int_values.append(v)
        elif field_no == 10:
            if wire_type == 0:
                v, _ = read_varint(buf, value_pos)
                int64_values.append(v)
            elif wire_type == 2:
                size, pos = read_varint(buf, value_pos)
                limit = pos + size
                while pos < limit:
                    v, pos = read_varint(buf, pos)
                    int64_values.append(v)
        elif field_no == 11:
            if wire_type == 0:
                v, _ = read_varint(buf, value_pos)
                bool_values.append(v)
            elif wire_type == 2:
                size, pos = read_varint(buf, value_pos)
                bool_values.extend(buf[pos : pos + size])

    if tensor_content:
        if dtype == 1 and len(tensor_content) >= 4:
            return float(struct.unpack("<f", tensor_content[:4])[0])
        if dtype == 2 and len(tensor_content) >= 8:
            return float(struct.unpack("<d", tensor_content[:8])[0])
        if dtype in {3, 4, 5, 6} and len(tensor_content) >= 4:
            return float(struct.unpack("<i", tensor_content[:4])[0])
        if dtype == 9 and len(tensor_content) >= 8:
            return float(struct.unpack("<q", tensor_content[:8])[0])
        if dtype == 10 and len(tensor_content) >= 1:
            return float(tensor_content[0] != 0)

    if float_values:
        return float(float_values[0])
    if double_values:
        return float(double_values[0])
    if int_values:
        return float(int_values[0])
    if int64_values:
        return float(int64_values[0])
    if bool_values:
        return float(bool_values[0] != 0)
    return None


def parse_summary_values(buf: bytes) -> Iterator[tuple[str, float]]:
    for field_no, wire_type, value_pos, end_pos in iter_fields(buf):
        if field_no != 1 or wire_type != 2:
            continue
        size, pos = read_varint(buf, value_pos)
        value_buf = buf[pos : pos + size]
        tag = None
        scalar = None
        for v_field, v_wire, v_pos, v_end in iter_fields(value_buf):
            if v_field in {1, 2} and v_wire == 2:
                tag_size, tag_pos = read_varint(value_buf, v_pos)
                tag = value_buf[tag_pos : tag_pos + tag_size].decode("utf-8", "replace")
            elif v_field in {2, 4} and v_wire == 5:
                scalar = float(struct.unpack("<f", value_buf[v_pos:v_end])[0])
            elif v_field == 8 and v_wire == 2:
                tensor_size, tensor_pos = read_varint(value_buf, v_pos)
                scalar = parse_tensor_scalar(value_buf[tensor_pos : tensor_pos + tensor_size])
        if tag is not None and scalar is not None and math.isfinite(scalar):
            yield tag, scalar


def iter_event_scalars(path: Path) -> Iterator[tuple[int, str, float]]:
    with path.open("rb") as f:
        while True:
            header = f.read(12)
            if not header:
                break
            if len(header) != 12:
                break
            length = struct.unpack("<Q", header[:8])[0]
            payload = f.read(length)
            crc = f.read(4)
            if len(payload) != length or len(crc) != 4:
                break

            step = 0
            summary_bufs: list[bytes] = []
            try:
                for field_no, wire_type, value_pos, end_pos in iter_fields(payload):
                    if field_no == 2 and wire_type == 0:
                        step, _ = read_varint(payload, value_pos)
                    elif field_no == 5 and wire_type == 2:
                        size, pos = read_varint(payload, value_pos)
                        summary_bufs.append(payload[pos : pos + size])
                for summary in summary_bufs:
                    for tag, value in parse_summary_values(summary):
                        yield step, tag, value
            except Exception:
                continue


def event_run_dir(event_file: Path) -> Path:
    if event_file.parent.name in {"summaries", "tb"}:
        return event_file.parent.parent
    return event_file.parent


def checkpoint_ep(path: str) -> int | None:
    match = EP_RE.search(Path(path).name)
    return int(match.group(1)) if match else None


def parse_s3_listing(path: Path, prefix: str) -> dict[str, int]:
    checkpoints: dict[str, int] = {}
    if not path:
        return checkpoints
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split(maxsplit=3)
        if len(parts) != 4 or not parts[2].isdigit():
            continue
        size = int(parts[2])
        key = parts[3]
        if not key.endswith(".pth"):
            continue
        if prefix and key.startswith(prefix):
            key = key[len(prefix) :]
        checkpoints[key] = size
    return checkpoints


def collect_local_checkpoints(root: Path) -> dict[str, int]:
    checkpoints: dict[str, int] = {}
    for path in root.rglob("*.pth"):
        rel = path.relative_to(root).as_posix()
        try:
            checkpoints[rel] = path.stat().st_size
        except OSError:
            checkpoints[rel] = -1
    return checkpoints


def run_key_from_event(root: Path, event_file: Path) -> str:
    run_dir = event_run_dir(event_file)
    return run_dir.relative_to(root).as_posix()


def find_checkpoint(
    *,
    root: Path,
    point: ScalarPoint,
    checkpoints: dict[str, int],
    strictly_before_metric: bool = False,
) -> CheckpointCandidate:
    run_key = point.run_dir.relative_to(root).as_posix()
    candidates = [
        CheckpointCandidate(path, checkpoint_ep(path), size)
        for path, size in checkpoints.items()
        if path.startswith(run_key + "/") and "/nn/" in path
    ]
    if not candidates:
        candidates = [
            CheckpointCandidate(path, checkpoint_ep(path), size)
            for path, size in checkpoints.items()
            if path.startswith(run_key + "/")
        ]

    ep_candidates = [c for c in candidates if c.ep is not None]
    if ep_candidates:
        if strictly_before_metric:
            before = [c for c in ep_candidates if c.ep is not None and c.ep < point.scalar_index]
            if before:
                return sorted(before, key=lambda c: (c.ep or -1, c.checkpoint))[-1]
            before_or_at = [c for c in ep_candidates if c.ep is not None and c.ep <= point.scalar_index]
            if before_or_at:
                return sorted(before_or_at, key=lambda c: (c.ep or -1, c.checkpoint))[-1]
        exact = [c for c in ep_candidates if c.ep == point.scalar_index]
        if exact:
            return sorted(exact, key=lambda c: ("__" in c.checkpoint, c.checkpoint))[0]
        return sorted(
            ep_candidates,
            key=lambda c: (abs((c.ep or 10**18) - point.scalar_index), c.ep or 10**18, c.checkpoint),
        )[0]

    if candidates:
        preferred = [c for c in candidates if Path(c.checkpoint).name not in {"best.pth"}]
        return sorted(preferred or candidates, key=lambda c: c.checkpoint)[-1]
    return CheckpointCandidate("", None, None)


def collect_points(root: Path, metric_tags: tuple[str, ...] = ()) -> list[ScalarPoint]:
    points: list[ScalarPoint] = []
    exact_metric_tags = set(metric_tags)
    for event_file in sorted(root.rglob("events.out.tfevents*")):
        run_dir = event_run_dir(event_file)
        tag_counts: dict[str, int] = {}
        for step, tag, value in iter_event_scalars(event_file):
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
            if tag in exact_metric_tags or (not exact_metric_tags and TRUE_SUCCESS_TAG_RE.search(tag)):
                points.append(
                    ScalarPoint(
                        event_file=event_file,
                        run_dir=run_dir,
                        tag=tag,
                        scalar_index=tag_counts[tag],
                        step=step,
                        value=value,
                    )
                )
    return points


def series_by_run(points: list[ScalarPoint]) -> dict[str, dict[str, list[ScalarPoint]]]:
    series: dict[str, dict[str, list[ScalarPoint]]] = {}
    for point in points:
        run_key = point.run_dir.as_posix()
        series.setdefault(run_key, {}).setdefault(point.tag, []).append(point)
    return series


def relative_series_by_run(root: Path, points: list[ScalarPoint]) -> dict[str, dict[str, list[ScalarPoint]]]:
    series: dict[str, dict[str, list[ScalarPoint]]] = {}
    for point in points:
        try:
            run_key = point.run_dir.relative_to(root).as_posix()
        except ValueError:
            run_key = point.run_dir.as_posix()
        series.setdefault(run_key, {}).setdefault(point.tag, []).append(point)
    return series


def normalize_checkpoint_path(path: str) -> str:
    p = path.replace("\\", "/")
    for marker in ("/logs/residual_rl/", "logs/residual_rl/", "/Residuals/", "Residuals/", "residuals/"):
        if marker in p:
            return p.split(marker, 1)[1].lstrip("/")
    return p.lstrip("/")


def checkpoint_run_key(path: str) -> str | None:
    if "/nn/" in path:
        return path.split("/nn/", 1)[0]
    return None


def select_tag(run_series: dict[str, list[ScalarPoint]], metric_tags: tuple[str, ...] = ()) -> str | None:
    for tag in metric_tags or CHECKPOINT_TRUE_SUCCESS_TAGS:
        if tag in run_series:
            return tag
    if metric_tags:
        return None
    for tag in sorted(run_series):
        if tag.endswith("/success_rate_ema") and "baseline" not in tag and "best_" not in tag:
            return tag
    return None


def point_for_checkpoint(points: list[ScalarPoint], ep: int) -> ScalarPoint | None:
    if not points:
        return None
    by_index = {point.scalar_index: point for point in points}
    if ep in by_index:
        return by_index[ep]
    by_step = {point.step: point for point in points}
    if ep in by_step:
        return by_step[ep]
    return min(points, key=lambda point: (abs(point.scalar_index - ep), point.scalar_index))


def checkpoint_scores_from_events(
    *,
    root: Path,
    points: list[ScalarPoint],
    checkpoints: dict[str, int],
    metric_tags: tuple[str, ...] = (),
    min_checkpoint_ep: int | None = None,
    checkpoint_before_metric: bool = False,
) -> list[CheckpointScore]:
    series = relative_series_by_run(root, points)
    scores: list[CheckpointScore] = []
    if checkpoint_before_metric:
        for run_key, run_series in sorted(series.items()):
            tag = select_tag(run_series, metric_tags)
            if tag is None:
                continue
            for point in run_series[tag]:
                checkpoint = find_checkpoint(
                    root=root,
                    point=point,
                    checkpoints=checkpoints,
                    strictly_before_metric=True,
                )
                if not checkpoint.checkpoint or checkpoint.ep is None:
                    continue
                if min_checkpoint_ep is not None and checkpoint.ep < min_checkpoint_ep:
                    continue
                try:
                    event_file = point.event_file.relative_to(root).as_posix()
                except ValueError:
                    event_file = point.event_file.as_posix()
                scores.append(
                    CheckpointScore(
                        checkpoint=checkpoint.checkpoint,
                        checkpoint_ep=checkpoint.ep,
                        checkpoint_size=checkpoint.size,
                        true_success_rate=float(point.value),
                        metric_tag=tag,
                        metric_index=point.scalar_index,
                        metric_step=point.step,
                        metric_epoch_delta=checkpoint.ep - point.scalar_index,
                        run_dir=run_key,
                        event_file=event_file,
                        source="event_metric_prior_checkpoint",
                    )
                )
        return scores
    for checkpoint, size in sorted(checkpoints.items()):
        ep = checkpoint_ep(checkpoint)
        run_key = checkpoint_run_key(checkpoint)
        if ep is None or run_key is None:
            continue
        if min_checkpoint_ep is not None and ep < min_checkpoint_ep:
            continue
        run_series = series.get(run_key)
        if not run_series:
            continue
        tag = select_tag(run_series, metric_tags)
        if tag is None:
            continue
        point = point_for_checkpoint(run_series[tag], ep)
        if point is None:
            continue
        try:
            event_file = point.event_file.relative_to(root).as_posix()
        except ValueError:
            event_file = point.event_file.as_posix()
        scores.append(
            CheckpointScore(
                checkpoint=checkpoint,
                checkpoint_ep=ep,
                checkpoint_size=size,
                true_success_rate=float(point.value),
                metric_tag=tag,
                metric_index=point.scalar_index,
                metric_step=point.step,
                metric_epoch_delta=ep - point.scalar_index,
                run_dir=run_key,
                event_file=event_file,
                source="event_checkpoint_epoch",
            )
        )
    return scores


def collect_metadata_scores(
    *,
    metadata_root: Path,
    root: Path,
    points: list[ScalarPoint],
    checkpoints: dict[str, int],
) -> list[CheckpointScore]:
    series = relative_series_by_run(root, points)
    scores: list[CheckpointScore] = []
    for path in sorted(metadata_root.rglob("iterations.jsonl")):
        run_key = path.parent.relative_to(metadata_root).as_posix()
        top_series = series.get(run_key, {})
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line_no, line in enumerate(f, start=1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                checkpoint_raw = record.get("best_checkpoint") or record.get("checkpoint")
                if not checkpoint_raw:
                    continue
                checkpoint = normalize_checkpoint_path(str(checkpoint_raw))
                rate = record.get("true_success_rate")
                metric_tag = "metadata/iterations.true_success_rate"
                metric_index = record.get("iter")
                metric_step = record.get("iter")
                event_file = path.relative_to(metadata_root).as_posix()
                if rate is None and record.get("best_run_idx") is not None:
                    run_tag = f"Run_{int(record['best_run_idx'])}/true_success_rate"
                    pts = top_series.get(run_tag, [])
                    if pts:
                        target_step = record.get("iter")
                        if target_step is not None:
                            point = min(pts, key=lambda p: abs(p.step - int(target_step)))
                        else:
                            point = pts[min(line_no - 1, len(pts) - 1)]
                        rate = point.value
                        metric_tag = run_tag
                        metric_index = point.scalar_index
                        metric_step = point.step
                        try:
                            event_file = point.event_file.relative_to(root).as_posix()
                        except ValueError:
                            event_file = point.event_file.as_posix()
                if rate is None:
                    continue
                ep = checkpoint_ep(checkpoint)
                scores.append(
                    CheckpointScore(
                        checkpoint=checkpoint,
                        checkpoint_ep=ep,
                        checkpoint_size=checkpoints.get(checkpoint),
                        true_success_rate=float(rate),
                        metric_tag=metric_tag,
                        metric_index=int(metric_index) if metric_index is not None else None,
                        metric_step=int(metric_step) if metric_step is not None else None,
                        metric_epoch_delta=None,
                        run_dir=checkpoint_run_key(checkpoint) or run_key,
                        event_file=event_file,
                        source="metadata_iterations",
                    )
                )
    return scores


def score_to_row(score: CheckpointScore) -> dict[str, str | int | float]:
    return {
        "metric_value": f"{score.true_success_rate:.8g}",
        "metric_tag": score.metric_tag,
        "source": score.source,
        "metric_index": "" if score.metric_index is None else score.metric_index,
        "metric_step": "" if score.metric_step is None else score.metric_step,
        "checkpoint": score.checkpoint,
        "checkpoint_ep": "" if score.checkpoint_ep is None else score.checkpoint_ep,
        "metric_epoch_delta": "" if score.metric_epoch_delta is None else score.metric_epoch_delta,
        "checkpoint_size": "" if score.checkpoint_size is None else score.checkpoint_size,
        "run_dir": score.run_dir,
        "event_file": score.event_file,
    }


def score_sort_key(score: CheckpointScore) -> tuple[float, int, int, str]:
    exactness = 0 if score.metric_epoch_delta is None else -abs(score.metric_epoch_delta)
    ep = int(score.checkpoint_ep or -1)
    return (float(score.true_success_rate), exactness, ep, score.checkpoint)


def write_checkpoint_score_report(scores: list[CheckpointScore], output: Path) -> None:
    rows = [score_to_row(score) for score in sorted(scores, key=score_sort_key, reverse=True)]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "metric_value",
                "metric_tag",
                "source",
                "metric_index",
                "metric_step",
                "checkpoint",
                "checkpoint_ep",
                "metric_epoch_delta",
                "checkpoint_size",
                "run_dir",
                "event_file",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def best_scores_by_run(scores: list[CheckpointScore]) -> list[CheckpointScore]:
    best: dict[str, CheckpointScore] = {}
    for score in scores:
        old = best.get(score.run_dir)
        if old is None or score_sort_key(score) > score_sort_key(old):
            best[score.run_dir] = score
    return sorted(best.values(), key=score_sort_key, reverse=True)


def write_tag_inventory(tags_json: Path, output: Path) -> None:
    data = json.loads(tags_json.read_text(encoding="utf-8"))
    counts: dict[str, tuple[int, int]] = {}
    for _event_file, tags in data.items():
        for tag, count in tags.items():
            total, files = counts.get(tag, (0, 0))
            counts[tag] = (total + int(count), files + 1)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["tag", "count", "event_files"])
        writer.writeheader()
        for tag, (count, event_files) in sorted(counts.items(), key=lambda item: (-item[1][0], item[0])):
            writer.writerow({"tag": tag, "count": count, "event_files": event_files})


def best_points(points: list[ScalarPoint]) -> list[ScalarPoint]:
    best: dict[tuple[Path, str], ScalarPoint] = {}
    for point in points:
        key = (point.run_dir, point.tag)
        old = best.get(key)
        if old is None or (point.value, point.scalar_index) > (old.value, old.scalar_index):
            best[key] = point
    return sorted(best.values(), key=lambda p: (-p.value, p.run_dir.as_posix(), p.tag))


def write_report(
    *,
    root: Path,
    points: list[ScalarPoint],
    checkpoints: dict[str, int],
    output: Path,
) -> None:
    rows = []
    for point in best_points(points):
        checkpoint = find_checkpoint(root=root, point=point, checkpoints=checkpoints)
        rows.append(
            {
                "true_success_rate": f"{point.value:.8g}",
                "scalar_index": point.scalar_index,
                "step": point.step,
                "tag": point.tag,
                "run_dir": point.run_dir.relative_to(root).as_posix(),
                "event_file": point.event_file.relative_to(root).as_posix(),
                "checkpoint": checkpoint.checkpoint,
                "checkpoint_ep": "" if checkpoint.ep is None else checkpoint.ep,
                "checkpoint_epoch_delta": "" if checkpoint.ep is None else checkpoint.ep - point.scalar_index,
                "checkpoint_size": "" if checkpoint.size is None else checkpoint.size,
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "true_success_rate",
                "scalar_index",
                "step",
                "tag",
                "run_dir",
                "event_file",
                "checkpoint",
                "checkpoint_ep",
                "checkpoint_epoch_delta",
                "checkpoint_size",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_tags(root: Path, output: Path) -> None:
    tags: dict[str, dict[str, int]] = {}
    for event_file in sorted(root.rglob("events.out.tfevents*")):
        rel = event_file.relative_to(root).as_posix()
        counts: dict[str, int] = {}
        for _step, tag, _value in iter_event_scalars(event_file):
            counts[tag] = counts.get(tag, 0) + 1
        if counts:
            tags[rel] = counts
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(tags, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--s3-listing", type=Path)
    parser.add_argument("--s3-prefix", default="residuals/")
    parser.add_argument("--metadata-root", type=Path)
    parser.add_argument(
        "--metric-tag",
        action="append",
        default=[],
        help="Exact scalar tag to score. Can be repeated. Defaults to true-success tags.",
    )
    parser.add_argument(
        "--min-checkpoint-ep",
        type=int,
        help="Ignore checkpoints with ep smaller than this value.",
    )
    parser.add_argument(
        "--checkpoint-before-metric",
        action="store_true",
        help="Score metric points by the latest checkpoint strictly before each metric point.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--all-checkpoints-output", type=Path)
    parser.add_argument("--tags-output", type=Path)
    parser.add_argument("--tag-inventory-output", type=Path)
    args = parser.parse_args()

    root = require_external_artifact_path(args.root, "Residual checkpoint root")
    args.output = require_external_artifact_path(args.output, "Checkpoint report")
    if args.all_checkpoints_output:
        args.all_checkpoints_output = require_external_artifact_path(
            args.all_checkpoints_output, "All-checkpoints report"
        )
    if args.tags_output:
        args.tags_output = require_external_artifact_path(args.tags_output, "TensorBoard tag report")
    if args.tag_inventory_output:
        args.tag_inventory_output = require_external_artifact_path(
            args.tag_inventory_output, "TensorBoard tag inventory"
        )
    metric_tags = tuple(args.metric_tag)
    points = collect_points(root, metric_tags)
    checkpoints = collect_local_checkpoints(root)
    if args.s3_listing:
        checkpoints.update(parse_s3_listing(args.s3_listing, args.s3_prefix))
    scores = checkpoint_scores_from_events(
        root=root,
        points=points,
        checkpoints=checkpoints,
        metric_tags=metric_tags,
        min_checkpoint_ep=args.min_checkpoint_ep,
        checkpoint_before_metric=bool(args.checkpoint_before_metric),
    )
    if args.metadata_root and not metric_tags:
        scores.extend(
            collect_metadata_scores(
                metadata_root=args.metadata_root.resolve(),
                root=root,
                points=points,
                checkpoints=checkpoints,
            )
        )
    write_checkpoint_score_report(best_scores_by_run(scores), args.output)
    if args.all_checkpoints_output:
        write_checkpoint_score_report(scores, args.all_checkpoints_output)
    if args.tags_output:
        write_tags(root, args.tags_output)
    if args.tag_inventory_output:
        if args.tags_output:
            tag_json = args.tags_output
        else:
            tag_json = args.tag_inventory_output.with_suffix(".json")
            write_tags(root, tag_json)
        write_tag_inventory(tag_json, args.tag_inventory_output)
    print(f"parsed_metric_points={len(points)}")
    print(f"scored_checkpoints={len(scores)}")
    print(f"report={args.output}")


if __name__ == "__main__":
    main()
