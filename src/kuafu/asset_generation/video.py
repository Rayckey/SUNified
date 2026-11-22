"""Bounded temporal sampling: prefer sharp frames within each time window."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from .io import digest, fresh_directory, resolve_artifact_path, write_json
from .schema import Frame, PreparedInput, ReferenceVideo, Scale


def contact_sheet(paths: list[Path], labels: list[str], destination: Path, columns: int = 4) -> None:
    width, height = 320, 272
    sheet = Image.new("RGB", (columns * width, ((len(paths) + columns - 1) // columns) * height), "#eef1f5")
    draw = ImageDraw.Draw(sheet)
    for i, (path, label) in enumerate(zip(paths, labels, strict=True)):
        x, y = (i % columns) * width, (i // columns) * height
        with Image.open(path) as source:
            thumb = ImageOps.contain(source.convert("RGB"), (width - 12, height - 32))
        sheet.paste(thumb, (x + (width - thumb.width) // 2, y + 6))
        draw.text((x + 8, y + height - 22), label, fill="#172033")
    sheet.save(destination)


def _prepare_video(
    video: Path,
    description: str,
    out: Path,
    frame_limit: int = 12,
    max_side: int = 1024,
    crop: list[float] | None = None,
    scale: Scale | None = None,
) -> PreparedInput:
    video = video.expanduser().resolve()
    if not video.is_file():
        raise ValueError(f"Video does not exist: {video}")
    if not description.strip() or len(description) > 16000:
        raise ValueError("Description must contain 1-16000 characters")
    if not 1 <= frame_limit <= 32 or not 128 <= max_side <= 2048:
        raise ValueError("Use 1-32 frames and a maximum image side of 128-2048 pixels")
    if crop is not None:
        if len(crop) != 4 or not all(np.isfinite(crop)):
            raise ValueError("Crop must be four finite normalized coordinates: x0 y0 x1 y1")
        if not (0 <= crop[0] < crop[2] <= 1 and 0 <= crop[1] < crop[3] <= 1):
            raise ValueError("Crop must satisfy 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1")

    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            raise ValueError(f"OpenCV could not decode the video: {video}")
        fps, count = capture.get(cv2.CAP_PROP_FPS), capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if not np.isfinite([fps, count]).all() or fps <= 0 or count < 1:
            raise ValueError("Video has no usable frame count/FPS; remux it to a standard MP4 first")
        count = int(count)
        out = fresh_directory(out)
        (out / "frames").mkdir()
        edges = np.linspace(0, count, min(frame_limit, count) + 1, dtype=int)
        selected: list[Frame] = []
        warnings = []
        fingerprints = []
        for low, high in zip(edges[:-1], edges[1:], strict=True):
            best = None
            # At most five decoded candidates per window; no full-video RAM buffer.
            for index in np.unique(np.linspace(low, high - 1, min(5, high - low), dtype=int)):
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                ok, frame = capture.read()
                if not ok:
                    continue
                h, w = frame.shape[:2]
                if crop is not None:
                    x0, y0, x1, y1 = crop
                    frame = frame[int(y0 * h) : int(y1 * h), int(x0 * w) : int(x1 * w)]
                    h, w = frame.shape[:2]
                    if min(h, w) < 2:
                        raise ValueError("Crop is less than two pixels wide or high")
                factor = min(1.0, max_side / max(h, w))
                frame = cv2.resize(frame, (max(1, round(w * factor)), max(1, round(h * factor))))
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                if best is None or sharpness > best[0]:
                    best = (sharpness, int(index), frame, gray)
            if best is None:
                warnings.append(f"No decodable frame in interval {low}-{high}")
                continue
            sharpness, index, frame, gray = best
            relative = f"frames/frame_{len(selected):03d}.jpg"
            path = out / relative
            if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                raise OSError(f"Could not write {path}")
            selected.append(
                Frame(
                    path=relative, index=index, time_s=index / fps, sharpness=sharpness, sha256=digest(path)
                )
            )
            fingerprints.append(cv2.resize(gray, (32, 32)).astype(float))
        if not selected:
            raise ValueError("No frames could be decoded from the video")
        if len(selected) < 4:
            warnings.append("Fewer than four views; hidden geometry will be especially uncertain")
        if len(fingerprints) > 1:
            changes = [np.mean(np.abs(a - b)) for a, b in zip(fingerprints[:-1], fingerprints[1:])]
            if float(np.median(changes)) < 2:
                warnings.append("Frames look very similar; the clip may reveal little additional geometry")
        if np.median([frame.sharpness for frame in selected]) < 20:
            warnings.append("Selected frames may be blurry or have little texture")
        manifest = PreparedInput(
            version=1,
            source_video=str(video),
            source_sha256=digest(video),
            description=description.strip(),
            fps=fps,
            duration_s=count / fps,
            frame_count=count,
            crop=crop,
            scale=scale,
            frames=selected,
            warnings=warnings,
        )
        write_json(out / "input.json", manifest)
        contact_sheet(
            [out / frame.path for frame in selected],
            [f"frame {f.index} | {f.time_s:.2f}s" for f in selected],
            out / "frames.jpg",
        )
        return manifest
    finally:
        capture.release()


def prepare(
    video: Path,
    description: str,
    out: Path,
    frame_limit: int = 12,
    max_side: int = 1024,
    crop: list[float] | None = None,
    scale: Scale | None = None,
    reference_videos: list[Path] | None = None,
    motion: str = "auto",
) -> PreparedInput:
    """Sample a primary clip and optional views of the same assembly/removed parts."""
    out = resolve_artifact_path(out, "Prepared asset workspace")
    references = reference_videos or []
    if len(references) > 3 or motion not in ("auto", "rigid", "articulated"):
        raise ValueError("Use at most three reference videos and motion auto, rigid or articulated")
    if not len(references) + 1 <= frame_limit <= 32:
        raise ValueError("Use 1-32 total frames, with at least one per video")
    paths = [Path(path).expanduser().resolve() for path in [video, *references]]
    if len(set(paths)) != len(paths) or not all(path.is_file() for path in paths):
        raise ValueError("Videos must be distinct existing files")
    counts = [frame_limit // len(paths) + (i < frame_limit % len(paths)) for i in range(len(paths))]
    manifest = _prepare_video(paths[0], description, out, counts[0], max_side, crop, scale)
    manifest.motion = motion
    for index, (path, count) in enumerate(zip(paths[1:], counts[1:], strict=True), start=1):
        folder = f"reference_{index:02d}"
        reference = _prepare_video(path, description, out / folder, count, max_side, crop, scale)
        manifest.reference_videos.append(
            ReferenceVideo(**{key: getattr(reference, key) for key in ReferenceVideo.model_fields})
        )
        manifest.frames.extend(
            [
                frame.model_copy(update={"path": f"{folder}/{frame.path}", "source_index": index})
                for frame in reference.frames
            ]
        )
        manifest.warnings.extend(f"Video {index}: {warning}" for warning in reference.warnings)
    write_json(out / "input.json", manifest)
    contact_sheet(
        [out / frame.path for frame in manifest.frames],
        [f"video {f.source_index} | {f.time_s:.2f}s" for f in manifest.frames],
        out / "frames.jpg",
    )
    return manifest
