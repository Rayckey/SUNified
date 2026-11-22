"""Render a FR3FinRay episode HDF5 point cloud into an MP4 video.

This handles both point cloud layouts used in this repo:

* Newer FR3FinRay files:
    point_cloud/points      vlen float32, each frame flattened as xyzxyz...
    point_cloud/colors      vlen float32/uint8, each frame flattened as rgbrgb...
    point_cloud/num_points  int32
* Older mimic files:
    point_cloud             dense dataset shaped (T, P, 3+) where RGB may be
                            stored in columns 3:6.

Usage:
    python src/kuafu/isaac/learned/visualize_episode_pointcloud.py _test/episode_0.h5
    python src/kuafu/isaac/learned/visualize_episode_pointcloud.py _test/episode_0.h5 --stride 2 --n_points 8192
"""

from __future__ import annotations

import argparse
import os

from kuafu.paths import resolve_artifact_path
from dataclasses import dataclass

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


@dataclass
class FrameData:
    points: np.ndarray
    colors: np.ndarray | None
    ee_pos: np.ndarray | None


class EpisodePointCloud:
    """Small reader that streams one point cloud frame at a time."""

    def __init__(self, path: str):
        self.path = path
        self.file = h5py.File(path, "r")
        self.layout = self._detect_layout()
        self.length = self._infer_length()

    def close(self) -> None:
        self.file.close()

    def __enter__(self) -> "EpisodePointCloud":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _detect_layout(self) -> str:
        pc = self.file.get("point_cloud")
        if isinstance(pc, h5py.Group) and "points" in pc:
            return "group"
        if isinstance(pc, h5py.Dataset):
            return "dense"
        raise KeyError(
            "Could not find a supported point cloud layout. Expected either "
            "'point_cloud/points' or a top-level 'point_cloud' dataset."
        )

    def _infer_length(self) -> int:
        if self.layout == "group":
            return int(self.file["point_cloud/points"].shape[0])
        return int(self.file["point_cloud"].shape[0])

    def frame(self, t: int) -> FrameData:
        if self.layout == "group":
            num_points = None
            if "point_cloud/num_points" in self.file:
                num_points = int(self.file["point_cloud/num_points"][t])
            points = _coerce_xyz(self.file["point_cloud/points"][t], num_points=num_points)
            colors = None
            if "point_cloud/colors" in self.file:
                colors = _coerce_rgb(self.file["point_cloud/colors"][t], num_points=points.shape[0])
        else:
            frame = np.asarray(self.file["point_cloud"][t])
            points = _coerce_xyz(frame)
            colors = _coerce_rgb(frame[:, 3:6], num_points=points.shape[0]) if frame.ndim == 2 and frame.shape[1] >= 6 else None

        points, colors = _filter_finite(points, colors)
        return FrameData(points=points, colors=colors, ee_pos=self.ee_position(t))

    def ee_position(self, t: int) -> np.ndarray | None:
        if "ee_position" in self.file:
            ee = np.asarray(self.file["ee_position"][t], dtype=np.float32).reshape(-1)
            return ee[:3] if ee.size >= 3 and np.isfinite(ee[:3]).all() else None
        if "arm_ee_pose" in self.file:
            ee = np.asarray(self.file["arm_ee_pose"][t], dtype=np.float32).reshape(-1)
            return ee[:3] if ee.size >= 3 and np.isfinite(ee[:3]).all() else None
        if "eef_pose" in self.file:
            eef = np.asarray(self.file["eef_pose"][t], dtype=np.float32).reshape(-1)
            if eef.size >= 16:
                mat = eef[:16].reshape(4, 4)
                ee = mat[:3, 3]
                return ee if np.isfinite(ee).all() else None
        return None


def _coerce_xyz(raw: np.ndarray, num_points: int | None = None) -> np.ndarray:
    arr = np.asarray(raw)
    if arr.ndim == 1:
        if num_points is not None and num_points > 0:
            arr = arr[: num_points * 3]
        trim = arr.size - (arr.size % 3)
        arr = arr[:trim].reshape(-1, 3)
    elif arr.ndim == 2:
        if arr.shape[1] < 3:
            raise ValueError(f"Point cloud frame has shape {arr.shape}; expected at least 3 columns.")
        arr = arr[:, :3]
    elif arr.ndim >= 3 and arr.shape[-1] >= 3:
        arr = arr.reshape(-1, arr.shape[-1])[:, :3]
    else:
        raise ValueError(f"Point cloud frame has unsupported shape {arr.shape}.")
    return arr.astype(np.float32, copy=False)


def _coerce_rgb(raw: np.ndarray, num_points: int | None = None) -> np.ndarray:
    arr = np.asarray(raw)
    if arr.ndim == 1:
        if num_points is not None and num_points > 0:
            arr = arr[: num_points * 3]
        trim = arr.size - (arr.size % 3)
        arr = arr[:trim].reshape(-1, 3)
    elif arr.ndim == 2:
        if arr.shape[1] < 3:
            raise ValueError(f"Color frame has shape {arr.shape}; expected at least 3 columns.")
        arr = arr[:, :3]
    elif arr.ndim >= 3 and arr.shape[-1] >= 3:
        arr = arr.reshape(-1, arr.shape[-1])[:, :3]
    else:
        raise ValueError(f"Color frame has unsupported shape {arr.shape}.")

    arr = arr.astype(np.float32, copy=False)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return arr
    max_value = float(finite.max())
    if max_value > 1.0:
        arr = arr / (255.0 if max_value <= 255.0 else max_value)
    return np.clip(arr, 0.0, 1.0)


def _filter_finite(points: np.ndarray, colors: np.ndarray | None) -> tuple[np.ndarray, np.ndarray | None]:
    valid = np.isfinite(points).all(axis=1)
    if colors is not None:
        if colors.shape[0] != points.shape[0]:
            n = min(colors.shape[0], points.shape[0])
            points = points[:n]
            colors = colors[:n]
            valid = valid[:n]
        valid = valid & np.isfinite(colors).all(axis=1)
    points = points[valid]
    colors = colors[valid] if colors is not None else None
    return points, colors


def _subsample(
    points: np.ndarray,
    colors: np.ndarray | None,
    n_points: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray | None]:
    if n_points <= 0 or points.shape[0] <= n_points:
        return points, colors
    idx = rng.choice(points.shape[0], size=n_points, replace=False)
    return points[idx], colors[idx] if colors is not None else None


def _iter_for_bounds(timesteps: list[int], max_frames: int) -> list[int]:
    if max_frames <= 0 or len(timesteps) <= max_frames:
        return timesteps
    idx = np.linspace(0, len(timesteps) - 1, max_frames).round().astype(int)
    return [timesteps[i] for i in np.unique(idx)]


def compute_bounds(
    episode: EpisodePointCloud,
    timesteps: list[int],
    rng: np.random.Generator,
    bounds_frames: int,
    bounds_points: int,
    percentile: float,
    margin: float,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    samples = []
    ee_samples = []
    for t in _iter_for_bounds(timesteps, bounds_frames):
        frame = episode.frame(t)
        if frame.points.shape[0] == 0:
            continue
        pts, _ = _subsample(frame.points, None, bounds_points, rng)
        samples.append(pts)
        if frame.ee_pos is not None:
            ee_samples.append(frame.ee_pos.reshape(1, 3))

    if not samples:
        raise ValueError("No finite point cloud samples were found for bounds.")

    all_points = np.concatenate(samples, axis=0)
    if percentile > 0.0:
        lo = np.percentile(all_points, percentile, axis=0)
        hi = np.percentile(all_points, 100.0 - percentile, axis=0)
    else:
        lo = all_points.min(axis=0)
        hi = all_points.max(axis=0)

    if ee_samples:
        ee_points = np.concatenate(ee_samples, axis=0)
        lo = np.minimum(lo, ee_points.min(axis=0))
        hi = np.maximum(hi, ee_points.max(axis=0))

    span = np.maximum(hi - lo, 1e-3)
    lo = lo - margin * span
    hi = hi + margin * span
    return (float(lo[0]), float(hi[0])), (float(lo[1]), float(hi[1])), (float(lo[2]), float(hi[2]))


def render_frame(
    ax: plt.Axes,
    points: np.ndarray,
    colors: np.ndarray | None,
    ee_pos: np.ndarray | None,
    azim: float,
    elev: float,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    zlim: tuple[float, float],
    t: int,
    total: int,
    point_size: float,
) -> None:
    ax.cla()
    if colors is None:
        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            s=point_size,
            c=points[:, 2],
            cmap="viridis",
            alpha=0.65,
            linewidths=0,
        )
    else:
        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            s=point_size,
            c=colors,
            alpha=0.75,
            linewidths=0,
        )

    if ee_pos is not None:
        ax.scatter(
            [ee_pos[0]],
            [ee_pos[1]],
            [ee_pos[2]],
            s=80,
            c="red",
            marker="*",
            zorder=10,
            label="EE",
        )
        ax.legend(loc="upper right", fontsize=7, markerscale=0.7)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    ax.set_xlabel("X", fontsize=7)
    ax.set_ylabel("Y", fontsize=7)
    ax.set_zlabel("Z", fontsize=7)
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(f"t = {t} / {total - 1}", fontsize=9)
    ax.tick_params(labelsize=6)
    try:
        ax.set_box_aspect((xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]))
    except Exception:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a FR3FinRay HDF5 point cloud episode as MP4.")
    parser.add_argument("h5_path", help="Path to episode HDF5 file.")
    parser.add_argument("--out", default=None, help="Output MP4 path. Defaults to '<h5>_pointcloud.mp4'.")
    parser.add_argument("--fps", type=int, default=30, help="Output video frame rate.")
    parser.add_argument("--stride", type=int, default=1, help="Render every Nth timestep.")
    parser.add_argument("--start", type=int, default=0, help="First timestep to render.")
    parser.add_argument("--max_frames", type=int, default=0, help="Maximum rendered frames; 0 renders all selected frames.")
    parser.add_argument("--n_points", type=int, default=16384, help="Subsample to this many points per frame; <=0 keeps all.")
    parser.add_argument("--azim", type=float, default=45.0, help="Azimuth angle for the 3D view.")
    parser.add_argument("--elev", type=float, default=25.0, help="Elevation angle for the 3D view.")
    parser.add_argument("--width", type=int, default=640, help="Output video width in pixels.")
    parser.add_argument("--height", type=int, default=480, help="Output video height in pixels.")
    parser.add_argument("--point_size", type=float, default=0.5, help="Matplotlib scatter point size.")
    parser.add_argument("--bounds_frames", type=int, default=80, help="Frames sampled to estimate stable axis bounds.")
    parser.add_argument("--bounds_points", type=int, default=4096, help="Points sampled per bounds frame.")
    parser.add_argument("--bounds_percentile", type=float, default=0.5, help="Percentile clipped from each bounds side; 0 uses exact bounds.")
    parser.add_argument("--margin", type=float, default=0.05, help="Fractional axis margin around computed bounds.")
    parser.add_argument("--crf", type=int, default=20, help="H.264 CRF passed to ffmpeg.")
    parser.add_argument("--preset", default="fast", help="H.264 preset passed to ffmpeg.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for deterministic subsampling.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.h5_path = str(resolve_artifact_path(args.h5_path, "Point-cloud trajectory"))
    if not os.path.exists(args.h5_path):
        raise FileNotFoundError(f"HDF5 file not found: {args.h5_path}")
    if args.stride <= 0:
        raise ValueError("--stride must be positive.")
    if args.fps <= 0:
        raise ValueError("--fps must be positive.")
    if args.start < 0:
        raise ValueError("--start must be non-negative.")

    out_path = args.out
    if out_path is None:
        out_path = os.path.splitext(args.h5_path)[0] + "_pointcloud.mp4"
    out_path = str(resolve_artifact_path(out_path, "Point-cloud video"))
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    dpi = 100
    fig = plt.figure(figsize=(args.width / dpi, args.height / dpi), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    rng = np.random.default_rng(args.seed)

    with EpisodePointCloud(args.h5_path) as episode:
        stop = episode.length
        timesteps = list(range(args.start, stop, args.stride))
        if args.max_frames > 0:
            timesteps = timesteps[: args.max_frames]
        if not timesteps:
            raise ValueError("No timesteps selected for rendering.")

        print(f"Loading: {args.h5_path}")
        print(f"  layout: {episode.layout}")
        print(f"  frames: {episode.length}")
        print(f"  selected frames: {len(timesteps)}")
        print("Computing bounds...")
        xlim, ylim, zlim = compute_bounds(
            episode=episode,
            timesteps=timesteps,
            rng=rng,
            bounds_frames=args.bounds_frames,
            bounds_points=args.bounds_points,
            percentile=args.bounds_percentile,
            margin=args.margin,
        )

        writer = FFMpegWriter(
            fps=args.fps,
            codec="libx264",
            bitrate=-1,
            extra_args=["-pix_fmt", "yuv420p", "-crf", str(args.crf), "-preset", args.preset],
            metadata={"title": os.path.basename(args.h5_path)},
        )

        print(f"Rendering {len(timesteps)} frames -> {out_path}")
        with writer.saving(fig, out_path, dpi=dpi):
            for i, t in enumerate(timesteps):
                frame = episode.frame(t)
                if frame.points.shape[0] == 0:
                    continue
                points, colors = _subsample(frame.points, frame.colors, args.n_points, rng)
                render_frame(
                    ax=ax,
                    points=points,
                    colors=colors,
                    ee_pos=frame.ee_pos,
                    azim=args.azim,
                    elev=args.elev,
                    xlim=xlim,
                    ylim=ylim,
                    zlim=zlim,
                    t=t,
                    total=episode.length,
                    point_size=args.point_size,
                )
                writer.grab_frame()
                if i % 50 == 0:
                    print(f"  {i}/{len(timesteps)} frames done")

    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
