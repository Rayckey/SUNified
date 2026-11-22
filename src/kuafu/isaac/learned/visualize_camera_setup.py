"""Visualize the FR3FinRay learned-env table camera setup.

The script reads the ``table_cam_*`` definitions from
``src/kuafu/isaac/learned/fr3_finray_env_learned.py`` and plots how their image
frustums intersect the z=0 tabletop/floor plane.

It intentionally avoids importing Isaac Lab so it can run in a lightweight
Python environment with only numpy/matplotlib.

Usage:
    python src/kuafu/isaac/learned/visualize_camera_setup.py
    python src/kuafu/isaac/learned/visualize_camera_setup.py --env-file src/kuafu/isaac/learned/fr3_finray_env_learned.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass, replace
from pathlib import Path

from kuafu.paths import resolve_artifact_path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d.art3d import Line3DCollection


@dataclass(frozen=True)
class CameraModel:
    name: str
    label: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    pos: np.ndarray
    quat_wxyz: np.ndarray
    near: float = 0.01
    far: float = 3.0

    @property
    def intrinsic_matrix(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]], dtype=float)


def _parse_float_list(text: str) -> list[float]:
    return [float(x) for x in re.findall(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?", text)]


def parse_camera_models(env_file: str) -> list[CameraModel]:
    with open(env_file, "r", encoding="utf-8") as f:
        source = f.read()

    starts = list(re.finditer(r"table_cam_(\d+)\s*=\s*TiledCameraCfg\(", source))
    cameras: list[CameraModel] = []
    for i, match in enumerate(starts):
        cam_idx = int(match.group(1))
        block_start = match.start()
        block_end = starts[i + 1].start() if i + 1 < len(starts) else source.find("action_scale", match.end())
        if block_end == -1:
            block_end = len(source)
        block = source[block_start:block_end]
        prefix = source[max(0, block_start - 250) : block_start]

        label_match = re.search(r"#\s*camera\d+\s*[—-]\s*([^\n]+)", prefix)
        label = label_match.group(1).strip() if label_match else f"table_cam_{cam_idx}"

        width_match = re.search(r"\bwidth\s*=\s*(\d+)", block)
        height_match = re.search(r"\bheight\s*=\s*(\d+)", block)
        intr_match = re.search(r"intrinsic_matrix\s*=\s*\[([^\]]+)\]", block, flags=re.S)
        clip_match = re.search(r"clipping_range\s*=\s*\(([^)]+)\)", block, flags=re.S)
        pos_match = re.search(r"pos\s*=\s*\(([^)]+)\)", block, flags=re.S)
        rot_match = re.search(r"rot\s*=\s*\(([^)]+)\)", block, flags=re.S)
        if not all([width_match, height_match, intr_match, pos_match, rot_match]):
            raise ValueError(f"Could not parse table_cam_{cam_idx} from {env_file}")

        width = int(width_match.group(1))
        height = int(height_match.group(1))
        intr = _parse_float_list(intr_match.group(1))
        pos = np.asarray(_parse_float_list(pos_match.group(1)), dtype=float)
        quat = np.asarray(_parse_float_list(rot_match.group(1)), dtype=float)
        near, far = 0.01, 3.0
        if clip_match:
            clip = _parse_float_list(clip_match.group(1))
            if len(clip) >= 2:
                near, far = float(clip[0]), float(clip[1])
        if len(intr) != 9 or pos.shape != (3,) or quat.shape != (4,):
            raise ValueError(f"Malformed camera values for table_cam_{cam_idx}")

        cameras.append(
            CameraModel(
                name=f"table_cam_{cam_idx}",
                label=label,
                width=width,
                height=height,
                fx=float(intr[0]),
                fy=float(intr[4]),
                cx=float(intr[2]),
                cy=float(intr[5]),
                pos=pos,
                quat_wxyz=quat / np.linalg.norm(quat),
                near=near,
                far=far,
            )
        )
    if not cameras:
        raise ValueError(f"No table_cam_* definitions found in {env_file}")
    return cameras


def quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def pixel_ray_local_world(cam: CameraModel, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Return rays in Isaac's camera "world" convention: +X forward, +Y left, +Z up."""
    x_ros = (u - cam.cx) / cam.fx
    y_ros = (v - cam.cy) / cam.fy
    rays = np.stack([np.ones_like(x_ros), -x_ros, -y_ros], axis=-1)
    return rays / np.linalg.norm(rays, axis=-1, keepdims=True)


def camera_to_world_rays(cam: CameraModel, rays_local: np.ndarray) -> np.ndarray:
    rot = quat_wxyz_to_matrix(cam.quat_wxyz)
    return rays_local @ rot.T


def project_floor_to_image(cam: CameraModel, points_w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rot = quat_wxyz_to_matrix(cam.quat_wxyz)
    local = (points_w - cam.pos) @ rot
    x_ros = -local[:, 1]
    y_ros = -local[:, 2]
    z_ros = local[:, 0]
    u = cam.fx * x_ros / z_ros + cam.cx
    v = cam.fy * y_ros / z_ros + cam.cy
    valid = (
        (z_ros >= cam.near)
        & (z_ros <= cam.far)
        & (u >= 0.0)
        & (u < cam.width)
        & (v >= 0.0)
        & (v < cam.height)
    )
    return u, v, valid


def intersect_rays_with_z(cam: CameraModel, u: np.ndarray, v: np.ndarray, z_plane: float) -> np.ndarray:
    rays_local = pixel_ray_local_world(cam, np.asarray(u, dtype=float), np.asarray(v, dtype=float))
    dirs = camera_to_world_rays(cam, rays_local)
    t = (z_plane - cam.pos[2]) / dirs[..., 2]
    points = cam.pos + dirs * t[..., None]
    points[(t <= 0.0) | ~np.isfinite(t)] = np.nan
    return points


def frustum_segments(cam: CameraModel, depths: tuple[float, float] = (0.25, 1.0)) -> np.ndarray:
    corners_u = np.array([0.0, cam.width - 1.0, cam.width - 1.0, 0.0])
    corners_v = np.array([0.0, 0.0, cam.height - 1.0, cam.height - 1.0])
    rays_local = pixel_ray_local_world(cam, corners_u, corners_v)
    dirs = camera_to_world_rays(cam, rays_local)
    segments = []
    for depth in depths:
        scale = depth / np.maximum(1e-9, rays_local[:, 0])
        pts = cam.pos + dirs * scale[:, None]
        for a, b in [(0, 1), (1, 2), (2, 3), (3, 0)]:
            segments.append([pts[a], pts[b]])
        if depth == depths[-1]:
            for p in pts:
                segments.append([cam.pos, p])
    return np.asarray(segments)


def make_variants(cameras: list[CameraModel]) -> list[tuple[str, list[CameraModel], str]]:
    current = cameras
    intended_640 = [replace(cam, width=640, height=480) for cam in cameras]
    scaled_320 = [
        replace(cam, width=320, height=240, fx=cam.fx * 0.5, fy=cam.fy * 0.5, cx=cam.cx * 0.5, cy=cam.cy * 0.5)
        for cam in cameras
    ]
    current_sizes = ", ".join(f"{cam.name}: {cam.width}x{cam.height}" for cam in cameras)
    return [
        ("current config", current, current_sizes),
        ("intended 640x480", intended_640, "same K values with 640x480 image size"),
        ("scaled 320x240", scaled_320, "K divided by 2 for true 320x240 downsample"),
    ]


def coverage_grid(cameras: list[CameraModel], extent: float, resolution: int, z_plane: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.linspace(-extent, extent, resolution)
    ys = np.linspace(-extent, extent, resolution)
    xx, yy = np.meshgrid(xs, ys)
    pts = np.stack([xx.reshape(-1), yy.reshape(-1), np.full(xx.size, z_plane)], axis=-1)
    coverage = np.zeros(pts.shape[0], dtype=np.int32)
    for cam in cameras:
        _, _, valid = project_floor_to_image(cam, pts)
        coverage += valid.astype(np.int32)
    return xx, yy, coverage.reshape(resolution, resolution)


def camera_floor_footprint(cam: CameraModel, z_plane: float) -> np.ndarray:
    corners_u = np.array([0.0, cam.width - 1.0, cam.width - 1.0, 0.0, 0.0])
    corners_v = np.array([0.0, 0.0, cam.height - 1.0, cam.height - 1.0, 0.0])
    return intersect_rays_with_z(cam, corners_u, corners_v, z_plane)


def optical_axis_hit(cam: CameraModel, z_plane: float) -> np.ndarray:
    return intersect_rays_with_z(cam, np.array([cam.cx]), np.array([cam.cy]), z_plane)[0]


def summarize_variant(label: str, cameras: list[CameraModel], extent: float, resolution: int, z_plane: float) -> dict:
    xx, yy, cov = coverage_grid(cameras, extent, resolution, z_plane)
    covered = cov > 0
    return {
        "label": label,
        "floor_coverage_pct": 100.0 * float(covered.mean()),
        "max_overlap": int(cov.max()),
        "xx": xx,
        "yy": yy,
        "coverage": cov,
    }


def plot_topdown(
    variants: list[tuple[str, list[CameraModel], str]],
    out_path: str,
    extent: float,
    resolution: int,
    z_plane: float,
) -> list[dict]:
    summaries = [summarize_variant(name, cams, extent, resolution, z_plane) for name, cams, _ in variants]
    cmap = ListedColormap(["#f8f8f8", "#9ecae1", "#4292c6", "#084594"])
    fig, axes = plt.subplots(1, len(variants), figsize=(6.4 * len(variants), 6.2), constrained_layout=True)
    if len(variants) == 1:
        axes = [axes]

    colors = ["#d62728", "#2ca02c", "#9467bd"]
    for ax, (name, cameras, note), summary in zip(axes, variants, summaries):
        im = ax.imshow(
            summary["coverage"],
            origin="lower",
            extent=(-extent, extent, -extent, extent),
            vmin=0,
            vmax=3,
            cmap=cmap,
            alpha=0.95,
            interpolation="nearest",
        )
        ax.plot([-extent, extent, extent, -extent, -extent], [-extent, -extent, extent, extent, -extent], "k--", lw=1.0)
        ax.scatter([0.0], [0.0], marker="+", s=80, c="black", label="robot base")

        for cam_idx, cam in enumerate(cameras):
            color = colors[cam_idx % len(colors)]
            ax.scatter([cam.pos[0]], [cam.pos[1]], marker="^", s=90, c=color, edgecolors="black", linewidths=0.5)
            ax.text(cam.pos[0], cam.pos[1], f" {cam.name}", fontsize=8, color=color, weight="bold")
            footprint = camera_floor_footprint(cam, z_plane)
            if np.isfinite(footprint[:, :2]).all():
                ax.plot(footprint[:, 0], footprint[:, 1], color=color, lw=1.5)
            hit = optical_axis_hit(cam, z_plane)
            if np.isfinite(hit[:2]).all():
                ax.scatter([hit[0]], [hit[1]], marker="x", s=50, c=color)

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-extent, extent)
        ax.set_ylim(-extent, extent)
        ax.set_xlabel("X on z=0 plane (m)")
        ax.set_ylabel("Y on z=0 plane (m)")
        ax.set_title(f"{name}\ncovered: {summary['floor_coverage_pct']:.1f}% | max overlap: {summary['max_overlap']}")
        ax.text(
            0.02,
            0.02,
            note,
            transform=ax.transAxes,
            fontsize=8,
            va="bottom",
            bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.85},
        )
        ax.grid(color="#dddddd", linewidth=0.5)

    cbar = fig.colorbar(im, ax=axes, shrink=0.82, ticks=[0, 1, 2, 3])
    cbar.ax.set_ylabel("number of cameras seeing z=0")
    fig.suptitle("FR3FinRay Learned Env Camera Coverage on z=0 Table/Floor Plane", fontsize=14)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return summaries


def plot_3d(cameras: list[CameraModel], out_path: str, extent: float, z_plane: float) -> None:
    fig = plt.figure(figsize=(9, 7), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    colors = ["#d62728", "#2ca02c", "#9467bd"]

    xs = [-extent, extent, extent, -extent, -extent]
    ys = [-extent, -extent, extent, extent, -extent]
    zs = [z_plane] * 5
    ax.plot(xs, ys, zs, "k--", lw=1.0, label="4m x 4m floor collider")
    ax.scatter([0.0], [0.0], [0.0], marker="+", s=80, c="black", label="robot base")

    for cam_idx, cam in enumerate(cameras):
        color = colors[cam_idx % len(colors)]
        ax.scatter([cam.pos[0]], [cam.pos[1]], [cam.pos[2]], marker="^", s=80, c=color)
        ax.text(cam.pos[0], cam.pos[1], cam.pos[2] + 0.04, cam.name, color=color, fontsize=8)
        segments = frustum_segments(cam, depths=(0.25, 1.0))
        ax.add_collection3d(Line3DCollection(segments, colors=color, linewidths=1.4, alpha=0.85))
        hit = optical_axis_hit(cam, z_plane)
        if np.isfinite(hit).all():
            ax.plot([cam.pos[0], hit[0]], [cam.pos[1], hit[1]], [cam.pos[2], hit[2]], color=color, lw=1.0, ls=":")

    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_zlim(-0.1, 1.1)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Current Camera Frustums from fr3_finray_env_learned.py")
    ax.view_init(elev=28, azim=-55)
    try:
        ax.set_box_aspect((2 * extent, 2 * extent, 1.2))
    except Exception:
        pass
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _finite_point(point: np.ndarray) -> list[float] | None:
    point = np.asarray(point, dtype=float).reshape(3)
    if not np.isfinite(point).all():
        return None
    return [float(point[0]), float(point[1]), float(point[2])]


def _add_line(
    lines: list[dict],
    a: np.ndarray | list[float],
    b: np.ndarray | list[float],
    color: str,
    width: float = 1.4,
    dash: bool = False,
    alpha: float = 1.0,
) -> None:
    aa = _finite_point(np.asarray(a, dtype=float))
    bb = _finite_point(np.asarray(b, dtype=float))
    if aa is None or bb is None:
        return
    lines.append({"a": aa, "b": bb, "color": color, "width": width, "dash": dash, "alpha": alpha})


def _build_interactive_scene(cameras: list[CameraModel], extent: float, z_plane: float) -> dict:
    colors = ["#d62728", "#2ca02c", "#9467bd"]
    lines: list[dict] = []
    points: list[dict] = []
    labels: list[dict] = []

    # Floor boundary and light grid.
    floor = [
        np.array([-extent, -extent, z_plane]),
        np.array([extent, -extent, z_plane]),
        np.array([extent, extent, z_plane]),
        np.array([-extent, extent, z_plane]),
    ]
    for i in range(4):
        _add_line(lines, floor[i], floor[(i + 1) % 4], "#111111", width=1.8, dash=True)

    grid_step = 0.5
    grid_vals = np.arange(-extent, extent + 1e-9, grid_step)
    for v in grid_vals:
        _add_line(lines, [v, -extent, z_plane], [v, extent, z_plane], "#d0d0d0", width=0.7, alpha=0.75)
        _add_line(lines, [-extent, v, z_plane], [extent, v, z_plane], "#d0d0d0", width=0.7, alpha=0.75)

    # World axes.
    _add_line(lines, [0, 0, z_plane], [0.7, 0, z_plane], "#e41a1c", width=2.4)
    _add_line(lines, [0, 0, z_plane], [0, 0.7, z_plane], "#377eb8", width=2.4)
    _add_line(lines, [0, 0, z_plane], [0, 0, z_plane + 0.7], "#4daf4a", width=2.4)
    labels.extend(
        [
            {"p": [0.75, 0, z_plane], "text": "X", "color": "#e41a1c"},
            {"p": [0, 0.75, z_plane], "text": "Y", "color": "#377eb8"},
            {"p": [0, 0, z_plane + 0.75], "text": "Z", "color": "#4daf4a"},
        ]
    )

    points.append({"p": [0.0, 0.0, z_plane], "text": "robot base", "color": "#000000", "radius": 5})

    for cam_idx, cam in enumerate(cameras):
        color = colors[cam_idx % len(colors)]
        points.append({"p": _finite_point(cam.pos), "text": cam.name, "color": color, "radius": 7})
        labels.append({"p": _finite_point(cam.pos + np.array([0.03, 0.03, 0.06])), "text": f"{cam.name} ({cam.label})", "color": color})

        for segment in frustum_segments(cam, depths=(0.25, 1.0)):
            _add_line(lines, segment[0], segment[1], color, width=1.7, alpha=0.95)

        footprint = camera_floor_footprint(cam, z_plane)
        for i in range(footprint.shape[0] - 1):
            _add_line(lines, footprint[i], footprint[i + 1], color, width=1.5, alpha=0.8)

        hit = optical_axis_hit(cam, z_plane)
        _add_line(lines, cam.pos, hit, color, width=1.3, dash=True, alpha=0.9)
        hit_list = _finite_point(hit)
        if hit_list is not None:
            points.append({"p": hit_list, "text": f"{cam.name} optical axis", "color": color, "radius": 4})

    return {
        "lines": lines,
        "points": [p for p in points if p["p"] is not None],
        "labels": [l for l in labels if l["p"] is not None],
        "center": [0.0, 0.0, 0.35],
        "extent": extent,
        "zPlane": z_plane,
    }


def write_interactive_html(cameras: list[CameraModel], out_path: str, extent: float, z_plane: float) -> None:
    scene_json = json.dumps(_build_interactive_scene(cameras, extent, z_plane))
    html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>FR3FinRay Camera Setup Interactive 3D</title>
  <style>
    html, body { margin: 0; height: 100%; background: #f6f7f9; color: #151515; font-family: system-ui, -apple-system, Segoe UI, sans-serif; }
    #wrap { height: 100%; display: grid; grid-template-rows: auto 1fr; }
    header { padding: 10px 14px; background: white; border-bottom: 1px solid #d8dce2; display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
    h1 { margin: 0; font-size: 16px; font-weight: 650; }
    .hint { color: #5b6470; font-size: 13px; }
    button { border: 1px solid #b8c0cc; background: #fff; border-radius: 6px; padding: 6px 10px; cursor: pointer; }
    button:hover { background: #eef2f7; }
    #canvas { width: 100%; height: 100%; display: block; background: #fbfbfc; cursor: grab; }
    #canvas.dragging { cursor: grabbing; }
  </style>
</head>
<body>
  <div id="wrap">
    <header>
      <h1>FR3FinRay Camera Setup Interactive 3D</h1>
      <button id="reset">Reset View</button>
      <button id="top">Top View</button>
      <button id="iso">Iso View</button>
      <span class="hint">Drag to rotate. Wheel to zoom. Double-click to reset.</span>
    </header>
    <canvas id="canvas"></canvas>
  </div>
  <script>
    const scene = __SCENE_JSON__;
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");
    let yaw = -0.72;
    let pitch = 0.58;
    let zoom = 1.0;
    let dragging = false;
    let lastX = 0;
    let lastY = 0;

    function resize() {
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.floor(rect.width * dpr));
      canvas.height = Math.max(1, Math.floor(rect.height * dpr));
      draw();
    }

    function project(p) {
      const scale = Math.min(canvas.width, canvas.height) / 4.9 * zoom;
      let x = p[0] - scene.center[0];
      let y = p[1] - scene.center[1];
      let z = p[2] - scene.center[2];

      const cy = Math.cos(yaw), sy = Math.sin(yaw);
      const x1 = cy * x - sy * y;
      const y1 = sy * x + cy * y;
      const z1 = z;

      const cp = Math.cos(pitch), sp = Math.sin(pitch);
      const y2 = cp * y1 - sp * z1;
      const z2 = sp * y1 + cp * z1;

      return {
        x: canvas.width / 2 + x1 * scale,
        y: canvas.height / 2 - z2 * scale,
        depth: y2
      };
    }

    function lineDepth(line) {
      return (project(line.a).depth + project(line.b).depth) * 0.5;
    }

    function draw() {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = "#fbfbfc";
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      const lines = [...scene.lines].sort((a, b) => lineDepth(a) - lineDepth(b));
      for (const line of lines) {
        const a = project(line.a);
        const b = project(line.b);
        ctx.save();
        ctx.globalAlpha = line.alpha ?? 1.0;
        ctx.strokeStyle = line.color;
        ctx.lineWidth = (line.width ?? 1.2) * (window.devicePixelRatio || 1);
        ctx.setLineDash(line.dash ? [8, 5] : []);
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
        ctx.restore();
      }

      const points = [...scene.points].sort((a, b) => project(a.p).depth - project(b.p).depth);
      for (const point of points) {
        const p = project(point.p);
        ctx.save();
        ctx.fillStyle = point.color;
        ctx.strokeStyle = "#111";
        ctx.lineWidth = 1.2 * (window.devicePixelRatio || 1);
        ctx.beginPath();
        ctx.arc(p.x, p.y, (point.radius ?? 5) * (window.devicePixelRatio || 1), 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
        ctx.restore();
      }

      ctx.save();
      ctx.font = `${13 * (window.devicePixelRatio || 1)}px system-ui, sans-serif`;
      ctx.textBaseline = "middle";
      for (const label of scene.labels) {
        const p = project(label.p);
        ctx.fillStyle = "rgba(255, 255, 255, 0.82)";
        const metrics = ctx.measureText(label.text);
        ctx.fillRect(p.x + 6, p.y - 10, metrics.width + 8, 20);
        ctx.fillStyle = label.color;
        ctx.fillText(label.text, p.x + 10, p.y);
      }
      ctx.restore();
    }

    function resetView() {
      yaw = -0.72;
      pitch = 0.58;
      zoom = 1.0;
      draw();
    }

    canvas.addEventListener("mousedown", (event) => {
      dragging = true;
      lastX = event.clientX;
      lastY = event.clientY;
      canvas.classList.add("dragging");
    });
    window.addEventListener("mouseup", () => {
      dragging = false;
      canvas.classList.remove("dragging");
    });
    window.addEventListener("mousemove", (event) => {
      if (!dragging) return;
      const dx = event.clientX - lastX;
      const dy = event.clientY - lastY;
      lastX = event.clientX;
      lastY = event.clientY;
      yaw += dx * 0.008;
      pitch = Math.max(-1.45, Math.min(1.45, pitch + dy * 0.008));
      draw();
    });
    canvas.addEventListener("wheel", (event) => {
      event.preventDefault();
      zoom = Math.max(0.2, Math.min(6.0, zoom * Math.exp(-event.deltaY * 0.001)));
      draw();
    }, { passive: false });
    canvas.addEventListener("dblclick", resetView);
    document.getElementById("reset").addEventListener("click", resetView);
    document.getElementById("top").addEventListener("click", () => { yaw = 0; pitch = 1.5707; zoom = 1.0; draw(); });
    document.getElementById("iso").addEventListener("click", resetView);
    window.addEventListener("resize", resize);
    resize();
  </script>
</body>
</html>
"""
    html = html.replace("__SCENE_JSON__", scene_json)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


def print_summary(cameras: list[CameraModel], summaries: list[dict], variants: list[tuple[str, list[CameraModel], str]]) -> None:
    print("Parsed cameras:")
    for cam in cameras:
        principal_ok = 0 <= cam.cx < cam.width and 0 <= cam.cy < cam.height
        status = "inside" if principal_ok else "OUTSIDE"
        print(
            f"  {cam.name} ({cam.label}): image={cam.width}x{cam.height}, "
            f"fx={cam.fx:.1f}, fy={cam.fy:.1f}, cx={cam.cx:.1f}, cy={cam.cy:.1f} ({status}), "
            f"pos=({cam.pos[0]:.3f}, {cam.pos[1]:.3f}, {cam.pos[2]:.3f})"
        )

    print("\nCoverage on z=0 over 4m x 4m floor collider:")
    for summary, (_, _, note) in zip(summaries, variants):
        print(
            f"  {summary['label']}: {summary['floor_coverage_pct']:.1f}% covered, "
            f"max overlap={summary['max_overlap']} ({note})"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize current FR3FinRay learned-env table cameras.")
    parser.add_argument(
        "--env-file",
        default="src/kuafu/isaac/learned/fr3_finray_env_learned.py",
        help="Path to fr3_finray_env_learned.py.",
    )
    parser.add_argument("--out", default=str(resolve_artifact_path(Path("diagnostics") / "camera_setup" / "fr3_finray_camera_setup_coverage.png")), help="Top-down coverage output PNG.")
    parser.add_argument("--out-3d", default=str(resolve_artifact_path(Path("diagnostics") / "camera_setup" / "fr3_finray_camera_setup_3d.png")), help="3D frustum output PNG.")
    parser.add_argument("--out-html", default=str(resolve_artifact_path(Path("diagnostics") / "camera_setup" / "fr3_finray_camera_setup_interactive.html")), help="Interactive rotatable 3D HTML output.")
    parser.add_argument("--extent", type=float, default=2.0, help="Half-width of the floor/table plane to plot.")
    parser.add_argument("--resolution", type=int, default=700, help="Coverage grid resolution per axis.")
    parser.add_argument("--z-plane", type=float, default=0.0, help="Plane height used for camera coverage.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out = str(resolve_artifact_path(args.out, "Camera coverage image"))
    args.out_3d = str(resolve_artifact_path(args.out_3d, "Camera frustum image"))
    args.out_html = str(resolve_artifact_path(args.out_html, "Camera visualization HTML"))
    cameras = parse_camera_models(args.env_file)
    variants = make_variants(cameras)

    for path in [args.out, args.out_3d, args.out_html]:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

    summaries = plot_topdown(variants, args.out, args.extent, args.resolution, args.z_plane)
    plot_3d(cameras, args.out_3d, args.extent, args.z_plane)
    write_interactive_html(cameras, args.out_html, args.extent, args.z_plane)
    print_summary(cameras, summaries, variants)
    print(f"\nSaved top-down coverage: {os.path.abspath(args.out)}")
    print(f"Saved 3D frustums: {os.path.abspath(args.out_3d)}")
    print(f"Saved interactive 3D view: {os.path.abspath(args.out_html)}")


if __name__ == "__main__":
    main()
