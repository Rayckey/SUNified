"""Render a collected episode HDF5 point cloud into an MP4 video.

Usage:
    python visualize_pointcloud.py <path/to/episode.h5> [--out output.mp4] [--fps 30]
                                   [--stride 1] [--n_points 2048]
                                   [--azim 45] [--elev 25]

    python visualize_pointcloud.py <path/to/episode.h5> --presentation \
                                   --camera_bank best --out output.mp4
"""

import argparse
import os
import shutil
import subprocess
from pathlib import Path

from kuafu.paths import resolve_artifact_path
from dataclasses import dataclass

import h5py
import matplotlib
matplotlib.use("Agg")  # offscreen, no display required
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


@dataclass(frozen=True)
class CameraSpec:
    name: str
    offset_xyz: tuple[float, float, float]
    target_bias_xyz: tuple[float, float, float] = (0.0, 0.0, 0.08)
    focal_length: float = 18.0


# Mirrored from src/kuafu/data/collection/capture_drawer_stack_best_ema_media.py.
# That capture script launches Isaac at import time, so these camera definitions
# stay local for offline point-cloud rendering.
CAMERA_BANKS: dict[str, list[CameraSpec]] = {
    "best": [
        CameraSpec("between_pillars_close_action", (0.00, -0.62, 0.56), (-0.10, 0.22, -0.03), 19.0),
    ],
    "drawer_close": [
        CameraSpec("drawer_close_front", (0.95, -0.05, 0.42), (-0.06, 0.02, 0.05), 22.0),
        CameraSpec("drawer_close_right", (0.80, -0.45, 0.40), (-0.05, 0.03, 0.05), 20.0),
        CameraSpec("gripper_drawer_side", (0.55, 0.45, 0.38), (-0.04, 0.00, 0.05), 22.0),
    ],
    "between_pillars": [
        CameraSpec("between_pillars_center", (0.00, -1.10, 0.48), (-0.02, 0.28, 0.05), 18.0),
        CameraSpec("between_pillars_low", (0.00, -0.95, 0.38), (-0.02, 0.24, 0.03), 20.0),
        CameraSpec("between_pillars_high", (0.02, -1.18, 0.58), (-0.02, 0.32, 0.05), 20.0),
    ],
    "between_pillars_refined": [
        CameraSpec("between_pillars_left_high_soft", (0.00, -1.10, 0.62), (-0.12, 0.28, -0.02), 18.0),
        CameraSpec("between_pillars_left_high_mid", (0.00, -1.10, 0.68), (-0.18, 0.28, -0.03), 19.0),
        CameraSpec("between_pillars_left_high_strong", (0.00, -1.08, 0.68), (-0.25, 0.27, -0.03), 20.0),
    ],
    "between_pillars_close": [
        CameraSpec("between_pillars_close_context", (0.00, -0.78, 0.60), (-0.10, 0.25, -0.03), 18.0),
        CameraSpec("between_pillars_close_action", (0.00, -0.62, 0.56), (-0.10, 0.22, -0.03), 19.0),
        CameraSpec("between_pillars_close_tight", (0.00, -0.48, 0.52), (-0.08, 0.18, -0.03), 20.0),
    ],
    "sweep_a": [
        CameraSpec("hero_front_right", (1.60, -1.45, 0.82), (0.00, 0.00, 0.02), 12.0),
        CameraSpec("robot_strawberry_side", (0.95, 1.55, 0.74), (0.00, 0.00, 0.02), 14.0),
        CameraSpec("drawer_side_low", (-1.25, -1.35, 0.70), (0.00, 0.00, 0.02), 16.0),
    ],
    "sweep_b": [
        CameraSpec("front_low_wide", (1.70, -0.20, 0.58), (0.00, 0.00, 0.02), 12.0),
        CameraSpec("over_robot_shoulder", (0.45, 1.65, 0.95), (0.00, 0.00, 0.02), 16.0),
        CameraSpec("top_oblique_context", (1.05, -0.95, 1.22), (0.00, 0.00, 0.00), 18.0),
    ],
    "hero": [
        CameraSpec("hero_front_right_tight", (1.35, -1.22, 0.72), (0.00, 0.00, 0.02), 18.0),
        CameraSpec("hero_front_right_wide", (1.60, -1.45, 0.82), (0.00, 0.00, 0.02), 12.0),
        CameraSpec("over_robot_shoulder", (0.35, 1.28, 0.92), (-0.02, 0.02, 0.06), 20.0),
    ],
}


def _equal_xyz_limits(xyz: np.ndarray, margin: float) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    center = (mins + maxs) * 0.5
    radius = float(np.max(maxs - mins) * 0.5 + margin)
    return tuple((float(c - radius), float(c + radius)) for c in center)  # type: ignore[return-value]


def _parse_color(raw: str) -> np.ndarray:
    aliases = {
        "black": "#000000",
        "white": "#ffffff",
        "dark": "#101216",
        "charcoal": "#101216",
    }
    value = aliases.get(raw.lower(), raw)
    if value.startswith("#") and len(value) == 7:
        return np.array([int(value[i:i + 2], 16) for i in (1, 3, 5)], dtype=np.uint8)
    parts = [p.strip() for p in value.split(",")]
    if len(parts) == 3:
        return np.array([max(0, min(255, int(float(p)))) for p in parts], dtype=np.uint8)
    raise ValueError(f"Expected color name, #RRGGBB, or R,G,B triplet, got: {raw}")


def _select_camera_spec(bank_name: str, view_name: str | None) -> CameraSpec:
    specs = CAMERA_BANKS[bank_name]
    if view_name is None:
        return specs[0]
    for spec in specs:
        if spec.name == view_name:
            return spec
    valid = ", ".join(spec.name for spec in specs)
    raise ValueError(f"Unknown camera view '{view_name}' for bank '{bank_name}'. Valid views: {valid}")


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm <= 1e-8:
        raise ValueError("Cannot normalize a near-zero vector")
    return v / norm


def _camera_basis(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    forward = _normalize(target - eye)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if abs(float(np.dot(forward, world_up))) > 0.98:
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    right = _normalize(np.cross(forward, world_up))
    up = _normalize(np.cross(right, forward))
    return np.stack([right, up, forward], axis=1).astype(np.float32)


def _finite_xyz(xyz: np.ndarray) -> np.ndarray:
    flat = xyz.reshape(-1, 3)
    return flat[np.isfinite(flat).all(axis=1)]


def _infer_focus(f: h5py.File, xyz: np.ndarray) -> np.ndarray:
    focus_points: list[np.ndarray] = []
    for group_name in ("articulated_root_pos", "obj_positions"):
        obj = f.get(group_name)
        if not isinstance(obj, h5py.Group):
            continue
        for name, ds in obj.items():
            if name in {"floor", "table"} or not isinstance(ds, h5py.Dataset):
                continue
            arr = np.asarray(ds[:], dtype=np.float32)
            if arr.ndim >= 2 and arr.shape[-1] >= 3:
                pts = arr.reshape(-1, arr.shape[-1])[:, :3]
                pts = pts[np.isfinite(pts).all(axis=1)]
                if pts.size:
                    focus_points.append(np.median(pts, axis=0))

    if focus_points:
        focus = np.mean(np.stack(focus_points, axis=0), axis=0).astype(np.float32)
    else:
        flat = _finite_xyz(xyz)
        if flat.shape[0] == 0:
            raise ValueError("point_cloud contains no finite points to infer focus")
        lo = np.percentile(flat, 5, axis=0)
        hi = np.percentile(flat, 95, axis=0)
        focus = ((lo + hi) * 0.5).astype(np.float32)

    # Match the media capture script: keep the look-at center low enough to
    # include table, drawer, end effector, and manipulated objects.
    focus[2] = float(np.clip(focus[2], 0.08, 0.22))
    return focus


def _fit_projection(
    xyz: np.ndarray,
    eye: np.ndarray,
    basis: np.ndarray,
    width: int,
    height: int,
    margin: float,
    quantile: float,
    zoom: float,
    near: float,
) -> tuple[float, float, float]:
    flat = _finite_xyz(xyz)
    cam = (flat - eye) @ basis
    cam = cam[cam[:, 2] > near]
    if cam.shape[0] == 0:
        raise ValueError("No points are in front of the selected camera")
    xn = cam[:, 0] / cam[:, 2]
    yn = cam[:, 1] / cam[:, 2]
    q = float(np.clip(quantile, 0.0, 0.2)) * 100.0
    lo_x, hi_x = np.percentile(xn, [q, 100.0 - q])
    lo_y, hi_y = np.percentile(yn, [q, 100.0 - q])
    span_x = max(float(hi_x - lo_x), 1e-6)
    span_y = max(float(hi_y - lo_y), 1e-6)
    usable_w = max(1.0, width * (1.0 - 2.0 * margin))
    usable_h = max(1.0, height * (1.0 - 2.0 * margin))
    focal_px = min(usable_w / span_x, usable_h / span_y) * float(zoom)
    cx = width * 0.5 - focal_px * float((lo_x + hi_x) * 0.5)
    cy = height * 0.5 + focal_px * float((lo_y + hi_y) * 0.5)
    return float(focal_px), float(cx), float(cy)


class FfmpegRgbWriter:
    """Stream RGB numpy frames to H.264 MP4."""

    def __init__(self, path: str, fps: int, width: int, height: int, crf: int):
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg not found on PATH; MP4 recording requires ffmpeg.")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        self._proc = subprocess.Popen(
            [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s:v",
                f"{width}x{height}",
                "-r",
                str(fps),
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                str(crf),
                "-preset",
                "fast",
                "-movflags",
                "+faststart",
                path,
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def write(self, frame_rgb: np.ndarray) -> None:
        if self._proc.stdin is None:
            raise RuntimeError(f"ffmpeg stdin closed for {self.path}")
        self._proc.stdin.write(np.ascontiguousarray(frame_rgb).tobytes())

    def close(self) -> None:
        if self._proc.stdin is not None:
            self._proc.stdin.close()
        rc = self._proc.wait()
        if rc != 0:
            err = b""
            if self._proc.stderr is not None:
                err = self._proc.stderr.read()
            raise RuntimeError(f"ffmpeg failed for {self.path} (rc={rc}):\n{err.decode(errors='replace')[-2000:]}")


def render_presentation_frame(
    points: np.ndarray,
    colors: np.ndarray | None,
    width: int,
    height: int,
    bg_color: np.ndarray,
    eye: np.ndarray,
    basis: np.ndarray,
    focal_px: float,
    cx: float,
    cy: float,
    point_radius: int,
    near: float,
) -> np.ndarray:
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[:, :] = bg_color

    valid = np.isfinite(points).all(axis=1)
    if colors is not None:
        valid = valid & np.isfinite(colors).all(axis=1)
    pts = points[valid]
    if pts.shape[0] == 0:
        return image
    if colors is None:
        cols = np.full((pts.shape[0], 3), 210, dtype=np.uint8)
    else:
        cols = (np.clip(colors[valid], 0.0, 1.0) * 255.0).astype(np.uint8)

    cam = (pts - eye) @ basis
    in_front = cam[:, 2] > near
    if not np.any(in_front):
        return image
    cam = cam[in_front]
    cols = cols[in_front]
    u = np.rint(cx + focal_px * (cam[:, 0] / cam[:, 2])).astype(np.int32)
    v = np.rint(cy - focal_px * (cam[:, 1] / cam[:, 2])).astype(np.int32)

    pad = max(0, int(point_radius))
    inside = (u >= -pad) & (u < width + pad) & (v >= -pad) & (v < height + pad)
    if not np.any(inside):
        return image
    u = u[inside]
    v = v[inside]
    z = cam[inside, 2]
    cols = cols[inside]

    order = np.argsort(z)[::-1]  # far-to-near painter's order
    u = u[order]
    v = v[order]
    cols = cols[order]

    offsets = [(0, 0)]
    if point_radius > 0:
        offsets = [
            (dx, dy)
            for dy in range(-point_radius, point_radius + 1)
            for dx in range(-point_radius, point_radius + 1)
            if dx * dx + dy * dy <= point_radius * point_radius
        ]
    for dx, dy in offsets:
        uu = u + dx
        vv = v + dy
        mask = (uu >= 0) & (uu < width) & (vv >= 0) & (vv < height)
        image[vv[mask], uu[mask]] = cols[mask]
    return image


def render_presentation_video(
    args: argparse.Namespace,
    xyz: np.ndarray,
    point_rgb: np.ndarray | None,
    focus: np.ndarray,
    out_path: str,
) -> None:
    spec = _select_camera_spec(args.camera_bank, args.camera_view)
    eye = focus + np.asarray(spec.offset_xyz, dtype=np.float32)
    target = focus + np.asarray(spec.target_bias_xyz, dtype=np.float32)
    basis = _camera_basis(eye, target)
    focal_px, cx, cy = _fit_projection(
        xyz,
        eye,
        basis,
        args.width,
        args.height,
        args.fit_margin,
        args.fit_quantile,
        args.zoom,
        args.near,
    )
    bg_color = _parse_color(args.background)

    T, P, _ = xyz.shape
    rng = np.random.default_rng(0)
    timesteps = range(0, T, args.stride)
    writer = FfmpegRgbWriter(out_path, args.fps, args.width, args.height, args.crf)

    print(f"Presentation camera: bank={args.camera_bank}, view={spec.name}")
    print(f"  focus: {focus.tolist()}")
    print(f"  eye: {eye.tolist()}")
    print(f"  target: {target.tolist()}")
    print(f"  auto-fit focal_px={focal_px:.2f}, principal=({cx:.2f}, {cy:.2f})")
    print(f"Rendering {len(timesteps)} frames -> {out_path}")

    try:
        for i, t in enumerate(timesteps):
            pts = xyz[t]
            cols = point_rgb[t] if point_rgb is not None else None
            if args.n_points < P:
                finite = np.isfinite(pts).all(axis=1)
                if cols is not None:
                    finite = finite & np.isfinite(cols).all(axis=1)
                finite_idx = np.flatnonzero(finite)
                if finite_idx.size > args.n_points:
                    idx = rng.choice(finite_idx, size=args.n_points, replace=False)
                    pts = pts[idx]
                    if cols is not None:
                        cols = cols[idx]
            frame = render_presentation_frame(
                pts,
                cols,
                args.width,
                args.height,
                bg_color,
                eye,
                basis,
                focal_px,
                cx,
                cy,
                args.point_radius,
                args.near,
            )
            writer.write(frame)
            if i % 50 == 0:
                print(f"  {i}/{len(timesteps)} frames done")
    finally:
        writer.close()


def render_frame(
    ax: plt.Axes,
    points: np.ndarray,       # (P, 3)
    colors: np.ndarray | None,  # (P, 3) or None
    ee_pos: np.ndarray | None,  # (3,) or None
    azim: float,
    elev: float,
    xlim: tuple,
    ylim: tuple,
    zlim: tuple,
    t: int,
    total: int,
) -> None:
    ax.cla()
    scatter_colors = points[:, 2] if colors is None else colors
    ax.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        s=0.5, c=scatter_colors, cmap=None if colors is not None else "viridis", alpha=0.6,
    )
    if ee_pos is not None:
        ax.scatter(
            [ee_pos[0]], [ee_pos[1]], [ee_pos[2]],
            s=80, c="red", marker="*", zorder=10, label="EE",
        )
        ax.legend(loc="upper right", fontsize=7, markerscale=0.7)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    try:
        ax.set_box_aspect((1.0, 1.0, 1.0))
    except AttributeError:
        pass
    ax.set_xlabel("X", fontsize=7)
    ax.set_ylabel("Y", fontsize=7)
    ax.set_zlabel("Z", fontsize=7)
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(f"t = {t} / {total - 1}", fontsize=9)
    ax.tick_params(labelsize=6)


def main():
    parser = argparse.ArgumentParser(description="Visualize point cloud episode as MP4.")
    parser.add_argument("h5_path", help="Path to episode HDF5 file.")
    parser.add_argument("--out", default=None, help="Output MP4 path (default: alongside h5).")
    parser.add_argument("--fps", type=int, default=30, help="Video frame rate.")
    parser.add_argument("--stride", type=int, default=1, help="Render every Nth timestep.")
    parser.add_argument("--n_points", type=int, default=16384,
                        help="Subsample point cloud to this many points per frame.")
    parser.add_argument("--azim", type=float, default=45, help="Azimuth angle for 3D view.")
    parser.add_argument("--elev", type=float, default=25, help="Elevation angle for 3D view.")
    parser.add_argument("--width", type=int, default=640, help="Output video width in pixels.")
    parser.add_argument("--height", type=int, default=480, help="Output video height in pixels.")
    parser.add_argument(
        "--free_aspect",
        action="store_true",
        help="Do not equalize XYZ metric scale; preserves the old stretched Matplotlib 3D view.",
    )
    parser.add_argument(
        "--presentation",
        action="store_true",
        help="Render a clean full-frame point-cloud video without plot axes using capture-script camera views.",
    )
    parser.add_argument(
        "--camera_bank",
        choices=sorted(CAMERA_BANKS),
        default="best",
        help="Camera bank mirrored from capture_drawer_stack_best_ema_media.py for --presentation.",
    )
    parser.add_argument(
        "--camera_view",
        default=None,
        help="Specific camera view name inside --camera_bank. Defaults to the bank's first camera.",
    )
    parser.add_argument("--background", default="dark", help="Presentation background: name, #RRGGBB, or R,G,B.")
    parser.add_argument("--point_radius", type=int, default=2, help="Presentation point splat radius in pixels.")
    parser.add_argument("--fit_margin", type=float, default=0.025, help="Fractional frame margin for presentation auto-fit.")
    parser.add_argument("--fit_quantile", type=float, default=0.01, help="Outlier quantile ignored for presentation auto-fit.")
    parser.add_argument("--zoom", type=float, default=1.02, help="Scale presentation auto-fit zoom.")
    parser.add_argument("--near", type=float, default=0.01, help="Near clipping distance for presentation projection.")
    parser.add_argument("--crf", type=int, default=18, help="Presentation H.264 CRF; lower is higher quality.")
    args = parser.parse_args()

    args.h5_path = str(resolve_artifact_path(args.h5_path, "Point-cloud trajectory"))
    if not os.path.exists(args.h5_path):
        raise FileNotFoundError(f"HDF5 file not found: {args.h5_path}")

    out_path = args.out
    if out_path is None:
        base = os.path.splitext(args.h5_path)[0]
        suffix = "_pointcloud_presentable.mp4" if args.presentation else "_pointcloud.mp4"
        out_path = base + suffix
    out_path = str(resolve_artifact_path(out_path, "Point-cloud video"))

    print(f"Loading: {args.h5_path}")
    with h5py.File(args.h5_path, "r") as f:
        if "point_cloud" in f:
            point_cloud_ds = f["point_cloud"]
            if not isinstance(point_cloud_ds, h5py.Dataset):
                raise TypeError("point_cloud must be an HDF5 dataset")
            pc = np.asarray(point_cloud_ds[:])
        elif "point_cloud_flat" in f and "point_cloud_count" in f:
            flat_ds = f["point_cloud_flat"]
            count_ds = f["point_cloud_count"]
            if not isinstance(flat_ds, h5py.Dataset):
                raise TypeError("point_cloud_flat must be an HDF5 dataset")
            if not isinstance(count_ds, h5py.Dataset):
                raise TypeError("point_cloud_count must be an HDF5 dataset")
            flat = np.asarray(flat_ds[:])
            counts = np.asarray(count_ds[:], dtype=np.int64)
            if flat.ndim != 2 or flat.shape[-1] < 3:
                raise ValueError("point_cloud_flat must have shape (N, C>=3)")
            if counts.ndim != 1:
                raise ValueError("point_cloud_count must have shape (T,)")
            max_count = int(counts.max(initial=0))
            pc = np.full((counts.shape[0], max_count, flat.shape[-1]), np.nan, dtype=flat.dtype)
            start_idx = 0
            for t, count in enumerate(counts):
                end_idx = start_idx + int(count)
                if end_idx > flat.shape[0]:
                    raise ValueError("point_cloud_count exceeds point_cloud_flat length")
                pc[t, : int(count)] = flat[start_idx:end_idx]
                start_idx = end_idx
            if start_idx != flat.shape[0]:
                raise ValueError("point_cloud_count does not consume point_cloud_flat exactly")
        else:
            raise KeyError("Expected point_cloud or point_cloud_flat + point_cloud_count in HDF5 file")

        focus = _infer_focus(f, pc[..., :3])
        ee = None
        if "ee_position" in f:
            ee_ds = f["ee_position"]
            if not isinstance(ee_ds, h5py.Dataset):
                raise TypeError("ee_position must be an HDF5 dataset")
            ee = np.asarray(ee_ds[:])  # (T, 3)

    xyz = pc[..., :3]
    point_rgb = None
    if pc.shape[-1] >= 6:
        point_rgb = pc[..., 3:6].astype(np.float32)
        finite_rgb = point_rgb[np.isfinite(point_rgb)]
        if finite_rgb.size == 0:
            point_rgb = None
        else:
            max_rgb = float(finite_rgb.max())
            # Handle both uint8-style [0,255] and arbitrary positive ranges.
            if max_rgb > 1.0:
                scale = 255.0 if max_rgb <= 255.0 else max_rgb
                point_rgb = point_rgb / scale
            point_rgb = np.clip(point_rgb, 0.0, 1.0)

    T, P, _ = xyz.shape
    print(f"  point_cloud shape: {pc.shape}")
    if ee is not None:
        print(f"  ee_position shape: {ee.shape}")

    if args.presentation:
        render_presentation_video(args, xyz, point_rgb, focus, out_path)
        print(f"Saved: {out_path}")
        return

    # Compute scene bounds from full trajectory for a stable view
    flat = xyz.reshape(-1, 3)
    flat = flat[np.isfinite(flat).all(axis=1)]
    if flat.shape[0] == 0:
        raise ValueError("point_cloud contains no finite points to render")
    margin = 0.05
    if args.free_aspect:
        xlim = (flat[:, 0].min() - margin, flat[:, 0].max() + margin)
        ylim = (flat[:, 1].min() - margin, flat[:, 1].max() + margin)
        zlim = (flat[:, 2].min() - margin, flat[:, 2].max() + margin)
    else:
        xlim, ylim, zlim = _equal_xyz_limits(flat, margin)

    # Subsample point indices (fixed random seed for consistency)
    rng = np.random.default_rng(0)
    n_pts = min(args.n_points, P)

    # Set up matplotlib figure
    dpi = 100
    fig_w = args.width / dpi
    fig_h = args.height / dpi
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")

    writer = FFMpegWriter(
        fps=args.fps,
        codec="libx264",
        bitrate=-1,
        extra_args=["-pix_fmt", "yuv420p", "-crf", "20", "-preset", "fast"],
        metadata={"title": os.path.basename(args.h5_path)},
    )

    timesteps = range(0, T, args.stride)
    print(f"Rendering {len(timesteps)} frames -> {out_path}")

    with writer.saving(fig, out_path, dpi=dpi):
        for i, t in enumerate(timesteps):
            pts = xyz[t]  # (P, 3)
            cols = point_rgb[t] if point_rgb is not None else None
            valid = np.isfinite(pts).all(axis=1)
            if cols is not None:
                valid = valid & np.isfinite(cols).all(axis=1)
            pts = pts[valid]
            if cols is not None:
                cols = cols[valid]
            if pts.shape[0] == 0:
                continue
            # Subsample
            idx = rng.choice(pts.shape[0], size=min(args.n_points, pts.shape[0]), replace=False)
            pts = pts[idx]
            if cols is not None:
                cols = cols[idx]
            ee_pos = ee[t] if ee is not None else None

            render_frame(ax, pts, cols, ee_pos, args.azim, args.elev, xlim, ylim, zlim, t, T)
            writer.grab_frame()

            if i % 50 == 0:
                print(f"  {i}/{len(timesteps)} frames done")

    plt.close(fig)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
