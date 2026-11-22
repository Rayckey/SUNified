"""Render a textured GLB into a looping MP4 using the existing offline preview renderer."""

import argparse
import math
import multiprocessing
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from .io import digest, fresh_directory, write_json
from .render import preview_mesh, render_mesh
from .textured import load_textured_glb

_mesh = None


def load_preview(path: Path) -> trimesh.Trimesh:
    """Apply glTF node transforms and convert Y-up to the renderer's Z-up convention."""
    meshes = load_textured_glb(path)
    to_z = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
    for mesh in meshes:
        mesh.apply_transform(to_z)
    return preview_mesh(meshes)


def _initialize(path: str):
    global _mesh
    _mesh = load_preview(Path(path))


def _frame(job):
    azimuth, elevation, size = job
    return np.asarray(render_mesh(_mesh, azimuth, elevation, size))


def orbit_angles(count: int, start: float = -45) -> list[float]:
    if count < 2 or not math.isfinite(start):
        raise ValueError("An orbit needs at least two frames and a finite starting angle")
    return [start + 360 * index / count for index in range(count)]


def render_turntable(
    visual: Path,
    out: Path,
    *,
    size: int = 720,
    fps: int = 30,
    seconds: float = 8,
    elevation: float = 25,
    start: float = -45,
    workers: int = 4,
    max_bytes: int = 20_000_000,
) -> dict:
    if not isinstance(size, int) or not 32 <= size <= 1440 or size % 2:
        raise ValueError("Size must be an even integer between 32 and 1440")
    if not isinstance(fps, int) or not 1 <= fps <= 60:
        raise ValueError("FPS must be an integer between 1 and 60")
    if not math.isfinite(seconds) or not 0 < seconds <= 30 or fps * seconds < 2:
        raise ValueError("Duration must be at most 30 seconds and contain at least two frames")
    if not math.isfinite(elevation) or not -85 <= elevation <= 85:
        raise ValueError("Elevation must be finite and between -85 and 85 degrees")
    if not isinstance(workers, int) or not 1 <= workers <= 16 or max_bytes <= 0:
        raise ValueError("Use 1..16 workers and a positive file size limit")
    angles = orbit_angles(round(fps * seconds), start)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("Install FFmpeg with the libx264 encoder to create MP4 turntables")
    visual = visual.resolve(strict=True)
    source_hash = digest(visual)
    mesh = load_preview(visual)
    out = fresh_directory(out)
    jobs = [(azimuth, elevation, size) for azimuth in angles]
    # Only publish complete, size-checked outputs; failed encodes leave no partial MP4.
    with tempfile.TemporaryDirectory(prefix=".render-", dir=out) as temporary:
        stage = Path(temporary)
        video = stage / "turntable.mp4"
        command = [
            ffmpeg,
            "-hide_banner",
            "-nostdin",
            "-loglevel",
            "error",
            "-n",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{size}x{size}",
            "-framerate",
            str(fps),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-threads",
            "2",
            "-movflags",
            "+faststart",
            str(video),
        ]
        samples = Image.new("RGB", (size * 2, size * 2), "#eff2f7")
        sample_indices = [index * len(angles) // 4 for index in range(4)]
        pool = None
        with tempfile.TemporaryFile() as errors:
            process = subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errors
            )
            try:
                if workers == 1:
                    frames = (np.asarray(render_mesh(mesh, *job)) for job in jobs)
                else:
                    # Forked workers would retain FFmpeg's stdin and prevent encoder EOF.
                    pool = ProcessPoolExecutor(
                        max_workers=workers,
                        mp_context=multiprocessing.get_context("spawn"),
                        initializer=_initialize,
                        initargs=(str(visual),),
                    )
                    frames = pool.map(_frame, jobs, chunksize=1)
                for index, frame in enumerate(frames):
                    if index == 0:
                        Image.fromarray(frame).save(stage / "poster.jpg", quality=95)
                    for tile, sample_index in enumerate(sample_indices):
                        if index == sample_index:
                            samples.paste(Image.fromarray(frame), (tile % 2 * size, tile // 2 * size))
                    process.stdin.write(frame.tobytes())
                    if (index + 1) % fps == 0:
                        print(f"{out.name}: rendered {index + 1}/{len(angles)} frames", flush=True)
                process.stdin.close()
                if process.wait(timeout=60) != 0:
                    errors.seek(0)
                    raise RuntimeError(f"FFmpeg failed: {errors.read(4096).decode(errors='replace')}")
            except BaseException:
                process.kill()
                process.wait()
                raise
            finally:
                if process.stdin is not None and not process.stdin.closed:
                    process.stdin.close()
                if pool is not None:
                    pool.shutdown(wait=True, cancel_futures=True)
        if video.stat().st_size > max_bytes:
            raise ValueError("Encoded video exceeds the file size limit; reduce size or duration")
        if digest(visual) != source_hash:
            raise ValueError("Source GLB changed during rendering; refusing to publish")
        samples.save(stage / "samples.jpg", quality=93)
        report = {
            "source_glb": os.path.relpath(visual, out),
            "source_sha256": source_hash,
            "renderer": "kuafu.asset_generation CPU UV-textured preview with simple diffuse lighting",
            "view": "one full 360-degree camera orbit around Z at a fixed elevation",
            "source_up_axis": "Y",
            "render_up_axis": "Z",
            "visual_triangles": len(mesh.faces),
            "resolution": [size, size],
            "fps": fps,
            "frame_count": len(angles),
            "duration_s": len(angles) / fps,
            "start_azimuth_deg": start,
            "elevation_deg": elevation,
            "duplicate_endpoint": False,
            "codec": "H.264 / yuv420p, CRF 18, MP4 faststart",
            "max_bytes": max_bytes,
            "video_bytes": video.stat().st_size,
            "files_sha256": {p.name: digest(p) for p in stage.iterdir() if p.is_file()},
            "limitations": (
                "Preview of the existing approximate asset, not a new reconstruction or physics test."
            ),
        }
        write_json(stage / "render.json", report)
        for file in stage.iterdir():
            file.rename(out / file.name)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("visual", type=Path, help="Self-contained textured GLB (glTF Y-up)")
    parser.add_argument("--out", type=Path, required=True, help="New or empty output directory")
    parser.add_argument("--size", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seconds", type=float, default=8)
    parser.add_argument("--elevation", type=float, default=25)
    parser.add_argument("--start", type=float, default=-45)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-bytes", type=int, default=20_000_000)
    args = parser.parse_args(argv)
    from kuafu.paths import resolve_artifact_path

    args.out = resolve_artifact_path(args.out, "Asset turntable output")
    report = render_turntable(**vars(args))
    print(f"Saved {args.out / 'turntable.mp4'} ({report['video_bytes'] / 1_000_000:.2f} MB)")
    return 0


if __name__ == "__main__":
    main()
