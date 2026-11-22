#!/usr/bin/env python3
"""Capture Kinova reset screenshots and joint values for a live env."""

from __future__ import annotations

import argparse
import asyncio
import ctypes
import datetime as _datetime
import json
import time
from pathlib import Path

from kuafu.paths import resolve_artifact_path

from isaaclab.app import AppLauncher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Kinova-Table-Gen3-Hang-Cup")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--settle_frames", type=int, default=16)
    parser.add_argument("--debug_frames", action=argparse.BooleanOptionalAction, default=False)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.num_envs = 1
    args.enable_cameras = True
    args.video = True
    return args


args_cli = parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import UsdGeom  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
import kuafu.isaac.learned.kinova  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab.utils.math import quat_apply  # noqa: E402
from isaacsim.core.utils.stage import get_current_stage  # noqa: E402

from omni.kit.viewport.utility import capture_viewport_to_buffer, get_active_viewport  # noqa: E402


class ViewportCapturer:
    def __init__(self, width: int, height: int):
        self.width = int(width)
        self.height = int(height)
        self.viewport = get_active_viewport()
        if self.viewport is None:
            raise RuntimeError("No active viewport is available.")
        self.viewport.resolution = (self.width, self.height)

    @staticmethod
    def _run(coro):
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        if loop.is_running():
            raise RuntimeError("Viewport capture cannot run inside an active asyncio loop.")
        return loop.run_until_complete(coro)

    def capture_rgb(self) -> np.ndarray:
        captured: dict[str, np.ndarray] = {}

        def _on_capture(buffer, buffer_size, width, height, _format):
            ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.POINTER(ctypes.c_byte * buffer_size)
            ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
            content = ctypes.pythonapi.PyCapsule_GetPointer(buffer, None)
            image = Image.frombytes("RGBA", (width, height), content.contents)
            if (width, height) != (self.width, self.height):
                image = image.resize((self.width, self.height), Image.Resampling.LANCZOS)
            captured["frame"] = np.asarray(image, dtype=np.uint8)[:, :, :3].copy()

        capture_viewport_to_buffer(self.viewport, _on_capture)
        deadline = time.time() + 5.0
        while "frame" not in captured and time.time() < deadline:
            simulation_app.update()
            time.sleep(0.001)
        if "frame" not in captured:
            raise RuntimeError("Viewport buffer capture timed out.")
        return captured["frame"]


def _as_list(tensor: torch.Tensor | np.ndarray) -> list[float]:
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    return [float(x) for x in np.asarray(tensor).reshape(-1)]


def _set_frame_debug_cfg(env_cfg, enabled: bool) -> None:
    for attr_name in ("ee_frame", "l7_frame", "l4_frame"):
        cfg = getattr(env_cfg, attr_name, None)
        if cfg is not None and hasattr(cfg, "debug_vis"):
            cfg.debug_vis = bool(enabled)


def _set_runtime_frame_debug(raw_env, enabled: bool) -> None:
    for sensor_name in ("panda_fingertip", "panda_link7", "panda_link4"):
        sensor = raw_env.scene.sensors.get(sensor_name)
        if sensor is not None and hasattr(sensor, "set_debug_vis"):
            sensor.set_debug_vis(bool(enabled))


def _capture_view(raw_env, capturer: ViewportCapturer, eye: np.ndarray, target: np.ndarray, path: Path) -> None:
    raw_env.sim.set_camera_view(
        eye=tuple(float(v) for v in eye),
        target=tuple(float(v) for v in target),
        camera_prim_path="/OmniverseKit_Persp",
    )
    cam_prim = get_current_stage().GetPrimAtPath("/OmniverseKit_Persp")
    if cam_prim.IsValid():
        UsdGeom.Camera(cam_prim).GetFocalLengthAttr().Set(20.0)
    for _ in range(3):
        raw_env.sim.render()
        simulation_app.update()
    Image.fromarray(capturer.capture_rgb()).save(path)


def main() -> None:
    timestamp = _datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = resolve_artifact_path(
        args_cli.output_dir
        or Path("diagnostics") / args_cli.task / f"kinova_initial_pose_{timestamp}",
        "Initial-pose diagnostic output",
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.enable_cameras = True
    _set_frame_debug_cfg(env_cfg, args_cli.debug_frames)

    env = gym.make(args_cli.task, cfg=env_cfg)
    raw_env = env.unwrapped
    try:
        env.reset()
        _set_runtime_frame_debug(raw_env, args_cli.debug_frames)
        for _ in range(max(1, int(args_cli.settle_frames))):
            raw_env.scene.update(raw_env.physics_dt)
            raw_env.sim.render()
            simulation_app.update()

        robot = raw_env.scene.articulations["robot"]
        joint_names = list(robot.joint_names)
        joint_pos = robot.data.joint_pos[0]
        joint_default = robot.data.default_joint_pos[0]
        robot_root_pos_w = robot.data.root_pos_w[0]
        support_cube_pos_w = raw_env.floor.data.root_pos_w[0] if hasattr(raw_env, "floor") else None
        robot_support_base_pos_w = (
            raw_env.robot_support_base.data.root_pos_w[0] if hasattr(raw_env, "robot_support_base") else None
        )
        arm_names = [f"joint_{i}" for i in range(1, 8)]
        arm_indices = [joint_names.index(name) for name in arm_names]
        arm_pos = joint_pos[arm_indices]
        arm_default = joint_default[arm_indices]

        ee_sensor = raw_env.scene.sensors.get("panda_fingertip")
        fingertip_pos_w = None
        fingertip_quat_w = None
        fingertip_plus_z_w = None
        if ee_sensor is not None:
            fingertip_pos_w = ee_sensor.data.target_pos_w[0, 0]
            fingertip_quat_w = ee_sensor.data.target_quat_w[0, 0]
            fingertip_plus_z_w = quat_apply(
                fingertip_quat_w,
                torch.tensor([0.0, 0.0, 1.0], device=raw_env.device),
            )

        origin = raw_env.scene.env_origins[0].detach().cpu().numpy()
        robot_root = robot_root_pos_w.detach().cpu().numpy()
        if fingertip_pos_w is not None:
            fingertip_focus = fingertip_pos_w.detach().cpu().numpy()
        else:
            fingertip_focus = origin + np.array([0.35, -0.25, 0.45], dtype=float)
        focus = 0.5 * np.asarray(robot_root, dtype=float) + 0.5 * np.asarray(fingertip_focus, dtype=float)
        focus[2] = max(float(focus[2]), 0.25)
        target = focus + np.array([0.0, 0.0, -0.02], dtype=float)

        capturer = ViewportCapturer(args_cli.width, args_cli.height)
        views = {
            "front_oblique.png": np.array([1.35, -1.25, 0.72], dtype=float),
            "side_oblique.png": np.array([1.45, 0.36, 0.68], dtype=float),
            "top_oblique.png": np.array([0.45, -0.55, 1.45], dtype=float),
        }
        screenshot_paths: dict[str, str] = {}
        for name, offset in views.items():
            path = output_dir / name
            _capture_view(raw_env, capturer, focus + offset, target, path)
            screenshot_paths[name] = str(path)

        metadata = {
            "task": args_cli.task,
            "joint_names": arm_names,
            "arm_default": _as_list(arm_default),
            "arm_after_reset": _as_list(arm_pos),
            "joint_1_default": float(arm_default[0].detach().cpu()),
            "joint_1_after_reset": float(arm_pos[0].detach().cpu()),
            "robot_root_pos_w": _as_list(robot_root_pos_w),
            "support_cube_pos_w": _as_list(support_cube_pos_w) if support_cube_pos_w is not None else None,
            "support_cube_visible": not bool(getattr(raw_env.cfg, "hide_support_cube_visuals", False)),
            "robot_support_base_pos_w": (
                _as_list(robot_support_base_pos_w) if robot_support_base_pos_w is not None else None
            ),
            "fingertip_pos_w": _as_list(fingertip_pos_w) if fingertip_pos_w is not None else None,
            "fingertip_quat_wxyz": _as_list(fingertip_quat_w) if fingertip_quat_w is not None else None,
            "fingertip_plus_z_w": _as_list(fingertip_plus_z_w) if fingertip_plus_z_w is not None else None,
            "screenshots": screenshot_paths,
        }
        metadata_path = output_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(metadata, indent=2))
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
