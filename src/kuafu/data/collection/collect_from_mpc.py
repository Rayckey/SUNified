import argparse
from contextlib import ExitStack
import ctypes
import json
import numpy as np
import shutil
import subprocess
import time
import traceback
from pathlib import Path
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Batched multi-stage agent for environment.")
parser.add_argument(
    "--disable_fabric", type=bool, default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to simulate."
)
parser.add_argument("--task_name", type=str, default="FR3-FinRay-Table-Hang-Cup", help="Name of the task.")
parser.add_argument("--worker_id", type=int, default=0, help="Worker ID for parallel collection (controls output subdir).")
parser.add_argument("--num_demos", type=int, default=200, help="Number of demos to collect per worker.")
parser.add_argument("--use_recorder", type=bool, default=True, help="Use env RecorderManager")
parser.add_argument("--record_failures", action="store_true", default=False, help="Also record failed episodes (default: only successful)")
parser.add_argument(
    "--control_output_mode", type=str, default=None,
    choices=["relative_ee", "relative_joint", "absolute_ee", "absolute_joint"],
    help="Defaults to absolute_joint for UR5 and absolute_ee for other robots.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help="Optional shared output directory for all workers in a launch.",
)
parser.add_argument(
    "--program_path",
    type=str,
    default=None,
    help=(
        "Optional SUNProgramEvaluator learned-program run directory. Overrides the task's "
        "mpc_config load_program path before environment creation."
    ),
)
parser.add_argument(
    "--action_noise_std",
    type=float,
    default=0.003,
    help="Std dev of Gaussian noise added to each action dim before env.step (0 disables).",
)
parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help="Optional deterministic seed for the environment and collector RNGs.",
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=0,
    help="Stop after this many total env steps across episode resets (0 disables).",
)
parser.add_argument(
    "--capture_false_finish_dir",
    type=str,
    default=None,
    help="Optional directory for false-finished terminal screenshots and metrics.",
)
parser.add_argument(
    "--capture_false_finish_count",
    type=int,
    default=0,
    help="Stop after capturing this many false-finished terminal states (0 disables).",
)
parser.add_argument(
    "--capture_false_finish_metrics_only",
    action="store_true",
    default=False,
    help=(
        "Write false-finish metrics without importing or using the Kit viewport. "
        "Useful for headless diagnostics where viewport utilities are unavailable."
    ),
)
parser.add_argument(
    "--capture_false_finish_width",
    type=int,
    default=1280,
    help="Viewport screenshot width for false-finished diagnostics.",
)
parser.add_argument(
    "--capture_false_finish_height",
    type=int,
    default=720,
    help="Viewport screenshot height for false-finished diagnostics.",
)
parser.add_argument(
    "--capture_false_finish_settle_frames",
    type=int,
    default=4,
    help="Render frames to settle the review camera before each false-finish capture.",
)
parser.add_argument(
    "--record_video_path",
    type=str,
    default=None,
    help="Optional MP4 path for recording the active viewport during collection.",
)
parser.add_argument(
    "--record_video_width",
    type=int,
    default=960,
    help="Viewport video width.",
)
parser.add_argument(
    "--record_video_height",
    type=int,
    default=540,
    help="Viewport video height.",
)
parser.add_argument(
    "--record_video_fps",
    type=int,
    default=10,
    help="Output video frame rate.",
)
parser.add_argument(
    "--record_video_every",
    type=int,
    default=5,
    help="Record every Nth env step.",
)
parser.add_argument(
    "--record_video_settle_frames",
    type=int,
    default=0,
    help="Render frames to settle the review camera before each recorded video frame.",
)
parser.add_argument(
    "--debug_messages",
    "--debug-messages",
    dest="debug_messages",
    action="store_true",
    default=False,
    help="Enable collector and environment debug diagnostic prints.",
)
parser.add_argument(
    "--gripper_force_print_every",
    type=int,
    default=None,
    help=(
        "Override the environment gripper-force diagnostic interval in steps. "
        "The default leaves the environment configuration unchanged."
    ),
)


import gymnasium as gym

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
DEBUG_MESSAGES = bool(args_cli.debug_messages)

# Keep camera sensors disabled unless explicit viewport video recording is requested.
CAPTURE_FALSE_FINISH = (
    args_cli.capture_false_finish_dir is not None
    and args_cli.capture_false_finish_count > 0
)
RECORD_VIDEO = args_cli.record_video_path is not None
CAPTURE_OUTCOME_VIDEO = False
OUTCOME_VIDEO_TARGETS = {}
if RECORD_VIDEO and args_cli.record_video_every <= 0:
    raise ValueError("--record_video_every must be positive.")
args_cli.enable_cameras = bool(RECORD_VIDEO)
if RECORD_VIDEO:
    # AppLauncher disables the active viewport in headless mode unless it sees a video request.
    args_cli.video = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import os
from isaaclab_tasks.utils import parse_env_cfg
import datetime
from omegaconf import OmegaConf
from kuafu.mpc.simulation.sun_program_executor import SUNProgramExecutor
from kuafu.common.simulation_utils import (
    update_sun_program_executor_data_for_env,
    create_differential_ik_pose_function,
)
from kuafu.paths import resolve_artifact_path
import copy
from PIL import Image
from pxr import UsdGeom
from isaacsim.core.utils.stage import get_current_stage
from isaaclab.utils import math as math_utils
capture_viewport_to_buffer = None
get_active_viewport = None
if (CAPTURE_FALSE_FINISH and not args_cli.capture_false_finish_metrics_only) or RECORD_VIDEO:
    from omni.kit.viewport.utility import capture_viewport_to_buffer, get_active_viewport

from kuafu.isaac.tasks import pen_cup, open_drawer, stack_two
import kuafu.isaac.learned  # noqa: F401
from kuafu.isaac.learned import kinova, panda, ur5  # noqa: F401
from kuafu.data.collection.state_recorder import StateTransitionRecorder


# SETTINGS GO HERE ============================================================
script_dir = str(Path(__file__).resolve().parents[4])
ENV_NAME = args_cli.task_name
WORKER_ID = args_cli.worker_id
NUM_DEMOS = args_cli.num_demos
# ===============================================================================

FALSE_FINISH_CAMERA_SPECS = [
    {
        "name": "front_close",
        "offset_xyz": (0.08, -0.92, 0.58),
        "target_bias_xyz": (0.00, 0.10, 0.05),
        "focal_length": 18.0,
    },
    {
        "name": "right_side",
        "offset_xyz": (0.72, -0.22, 0.52),
        "target_bias_xyz": (-0.03, 0.02, 0.05),
        "focal_length": 22.0,
    },
    {
        "name": "top_oblique",
        "offset_xyz": (0.58, -0.60, 1.00),
        "target_bias_xyz": (0.00, 0.02, 0.03),
        "focal_length": 18.0,
    },
]


class ViewportCapturer:
    """Synchronous helper around Kit viewport buffer captures."""

    def __init__(self, width: int, height: int):
        self.width = int(width)
        self.height = int(height)
        self.viewport = get_active_viewport()
        if not self.viewport:
            raise RuntimeError("No active viewport is available. Try running without --headless.")
        self.viewport.resolution = (self.width, self.height)

    def capture_rgb(self) -> np.ndarray:
        captured = {}

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
            raise RuntimeError("Viewport buffer capture did not return a frame within 5 seconds.")
        return captured["frame"]


class FfmpegRgbWriter:
    def __init__(self, path: Path, fps: int, width: int, height: int):
        self.path = path
        self.width = int(width)
        self.height = int(height)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{self.width}x{self.height}",
            "-r",
            str(int(fps)),
            "-i",
            "-",
            "-an",
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(self.path),
        ]
        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )

    def write(self, frame_rgb: np.ndarray) -> None:
        if frame_rgb.shape[:2] != (self.height, self.width):
            image = Image.fromarray(frame_rgb)
            image = image.resize((self.width, self.height), Image.Resampling.LANCZOS)
            frame_rgb = np.asarray(image, dtype=np.uint8)
        if self._proc.stdin is None:
            raise RuntimeError("ffmpeg stdin is closed.")
        self._proc.stdin.write(np.ascontiguousarray(frame_rgb).tobytes())

    def close(self) -> None:
        if self._proc.stdin is not None:
            self._proc.stdin.close()
            self._proc.stdin = None
        _, stderr = self._proc.communicate()
        if self._proc.returncode != 0:
            raise RuntimeError(stderr.decode("utf-8", errors="replace"))


def _torch_vector(value) -> list[float]:
    if torch.is_tensor(value):
        value = value.detach().cpu().flatten().tolist()
    elif isinstance(value, np.ndarray):
        value = value.reshape(-1).tolist()
    elif not isinstance(value, (list, tuple)):
        value = [value]
    return [float(x) for x in value]


def _first_float(value) -> float:
    return _torch_vector(value)[0]


def _first_bool(value) -> bool:
    if torch.is_tensor(value):
        return bool(value.detach().cpu().flatten()[0].item())
    if isinstance(value, np.ndarray):
        return bool(value.reshape(-1)[0].item())
    if isinstance(value, (list, tuple)):
        return bool(value[0])
    return bool(value)


def _pen_cup_focus(raw_env) -> torch.Tensor:
    points = []
    for name in ("cube_1", "cube_2", "marker", "pen_holder"):
        obj = getattr(raw_env, name, None)
        if obj is None:
            obj = raw_env.scene.rigid_objects.get(name)
        if obj is not None:
            points.append(obj.data.root_state_w[0, :3])
    for name in ("mug_handle_hole_frame", "mug_center_frame", "cup_hanger_branch_1_tip_frame"):
        sensor = getattr(raw_env, name, None)
        if sensor is None:
            sensor = raw_env.scene.sensors.get(name)
        if sensor is not None and hasattr(sensor.data, "target_pos_w"):
            points.append(sensor.data.target_pos_w.squeeze(1)[0, :3])
    if not points:
        return torch.tensor([0.42, -0.35, 0.12], device=raw_env.device)
    focus = torch.stack(points, dim=0).mean(dim=0)
    focus[2] = torch.clamp(focus[2], min=0.07, max=0.28)
    return focus


def _set_false_finish_camera(raw_env, spec: dict[str, object]) -> dict[str, object]:
    focus = _pen_cup_focus(raw_env)
    offset = torch.tensor(spec["offset_xyz"], device=raw_env.device, dtype=torch.float32)
    bias = torch.tensor(spec["target_bias_xyz"], device=raw_env.device, dtype=torch.float32)
    eye = focus + offset
    target = focus + bias
    eye_tuple = tuple(float(x) for x in eye.detach().cpu().tolist())
    target_tuple = tuple(float(x) for x in target.detach().cpu().tolist())
    raw_env.sim.set_camera_view(eye=eye_tuple, target=target_tuple, camera_prim_path="/OmniverseKit_Persp")
    cam_prim = get_current_stage().GetPrimAtPath("/OmniverseKit_Persp")
    if cam_prim.IsValid():
        UsdGeom.Camera(cam_prim).GetFocalLengthAttr().Set(float(spec["focal_length"]))
    return {
        "view": str(spec["name"]),
        "eye": eye_tuple,
        "target": target_tuple,
        "focal_length": float(spec["focal_length"]),
    }


def _write_png(path: Path, frame_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame_rgb).save(path)


def _write_mp4(path: Path, frames: list[np.ndarray], fps: int) -> None:
    if not frames:
        raise RuntimeError(f"No frames available for {path}")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required for --capture_outcome_video_dir")

    first = np.asarray(frames[0], dtype=np.uint8)
    height, width = first.shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(int(fps)),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        assert proc.stdin is not None
        for frame in frames:
            frame = np.asarray(frame, dtype=np.uint8)
            if frame.shape[:2] != (height, width):
                image = Image.fromarray(frame)
                frame = np.asarray(image.resize((width, height), Image.Resampling.LANCZOS), dtype=np.uint8)
            proc.stdin.write(np.ascontiguousarray(frame[:, :, :3]).tobytes())
    finally:
        if proc.stdin is not None:
            proc.stdin.close()
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {ret} while writing {path}")


def _hang_cup_metrics(raw_env, collector_stage_idx: int, step_count: int) -> dict[str, object]:
    mug_center_pos = raw_env.mug_center_frame.data.target_pos_w.squeeze(1)[0]
    mug_handle_hole_pos = raw_env.mug_handle_hole_frame.data.target_pos_w.squeeze(1)[0]
    fingertip_pos = raw_env.ee_frame.data.target_pos_w.squeeze(1)[0]
    branch_tip_pos = raw_env.cup_hanger_branch_1_tip_frame.data.target_pos_w.squeeze(1)[0]
    branch_tip_quat = raw_env.cup_hanger_branch_1_tip_frame.data.target_quat_w.squeeze(1)[0]

    local_y = torch.zeros((1, 3), device=raw_env.device, dtype=branch_tip_pos.dtype)
    local_y[:, 1] = 1.0
    branch_y_axis = math_utils.quat_apply(branch_tip_quat.view(1, 4), local_y)[0]
    branch_base_pos = branch_tip_pos - raw_env.cfg.success_branch_length * branch_y_axis

    handle_base_distance = torch.linalg.norm(mug_handle_hole_pos - branch_base_pos)
    handle_tip_distance = torch.linalg.norm(mug_handle_hole_pos - branch_tip_pos)
    fingertip_center_distance = torch.linalg.norm(fingertip_pos - mug_center_pos)
    handle_near_branch_base = bool(handle_base_distance < raw_env.cfg.success_handle_base_threshold)
    fingertip_clear = bool(fingertip_center_distance > raw_env.cfg.success_fingertip_center_min_distance)

    gripper_openness = None
    gripper_open = None
    try:
        gripper_openness = _first_float(raw_env.program_runtime.controller_api_wrapper.gripper_openness())
        gripper_open = bool(gripper_openness > raw_env.cfg.success_gripper_open_threshold)
    except Exception as exc:
        gripper_openness = f"unavailable: {type(exc).__name__}: {exc}"

    pose_ok = bool(handle_near_branch_base and fingertip_clear and gripper_open)
    hold_frames = None
    if hasattr(raw_env, "_mug_done_hold_frames"):
        try:
            hold_frames = int(raw_env._mug_done_hold_frames.flatten()[0].detach().cpu().item())
        except Exception:
            hold_frames = None

    current_stage = None
    if hasattr(raw_env, "current_stage"):
        try:
            current_stage = int(raw_env.current_stage.flatten()[0].detach().cpu().item())
        except Exception:
            current_stage = None

    return {
        "collector_stage_idx": int(collector_stage_idx),
        "env_current_stage": current_stage,
        "collector_step_count": int(step_count),
        "task_metrics_kind": "hang_cup",
        "thresholds": {
            "handle_base_distance_m": float(raw_env.cfg.success_handle_base_threshold),
            "branch_length_m": float(raw_env.cfg.success_branch_length),
            "fingertip_center_min_distance_m": float(raw_env.cfg.success_fingertip_center_min_distance),
            "gripper_openness": float(raw_env.cfg.success_gripper_open_threshold),
            "success_hold_frames": int(raw_env.cfg.success_hold_frames),
        },
        "handle_base_distance_m": float(handle_base_distance.detach().cpu()),
        "handle_tip_distance_m": float(handle_tip_distance.detach().cpu()),
        "fingertip_center_distance_m": float(fingertip_center_distance.detach().cpu()),
        "handle_near_branch_base": handle_near_branch_base,
        "fingertip_clear": fingertip_clear,
        "gripper_openness": gripper_openness,
        "gripper_open": gripper_open,
        "pose_ok": pose_ok,
        "mug_done_hold_frames": hold_frames,
        "sun_program_evaluator_stage_override": bool(getattr(raw_env.cfg, "sun_program_evaluator_stage_override", False)),
        "mug_center_pos_w": _torch_vector(mug_center_pos),
        "mug_handle_hole_pos_w": _torch_vector(mug_handle_hole_pos),
        "cup_hanger_branch_tip_pos_w": _torch_vector(branch_tip_pos),
        "cup_hanger_branch_base_pos_w": _torch_vector(branch_base_pos),
        "cup_hanger_branch_y_axis_w": _torch_vector(branch_y_axis),
        "fingertip_pos_w": _torch_vector(fingertip_pos),
    }


def _stack3_metrics(raw_env, collector_stage_idx: int, step_count: int) -> dict[str, object]:
    xy_threshold = 0.03
    height_threshold = 0.01
    height_diff = 0.0468
    min_gripper_cube_dist = 0.05

    cube1_pos = raw_env.cube_1.data.root_pos_w[0]
    cube2_pos = raw_env.cube_2.data.root_pos_w[0]
    cube3_pos = raw_env.cube_3.data.root_pos_w[0]

    xy_c12 = torch.linalg.norm(cube1_pos[:2] - cube2_pos[:2])
    xy_c23 = torch.linalg.norm(cube2_pos[:2] - cube3_pos[:2])
    xy_c13 = torch.linalg.norm(cube1_pos[:2] - cube3_pos[:2])
    z_error_c2 = torch.abs(cube2_pos[2] - (cube1_pos[2] + height_diff))
    z_error_c3 = torch.abs(cube3_pos[2] - (cube1_pos[2] + 2.0 * height_diff))

    api = raw_env.program_runtime.controller_api_wrapper
    gripper_openness = _first_float(api.gripper_openness())
    fingertip_pos = api.lookup_robot_frame_position("panda_fingertip")[0]
    fingertip_dists = [
        torch.linalg.norm(fingertip_pos - cube_pos)
        for cube_pos in (cube1_pos, cube2_pos, cube3_pos)
    ]

    xy_pass = bool(
        (xy_c12 < xy_threshold)
        and (xy_c23 < xy_threshold)
        and (xy_c13 < xy_threshold)
    )
    z_pass = bool(
        (z_error_c2 < height_threshold) and (z_error_c3 < height_threshold)
    )
    gripper_open = bool(gripper_openness > 0.9)
    gripper_clear = bool(
        all(distance >= min_gripper_cube_dist for distance in fingertip_dists)
    )

    # This is exactly the environment's physical Stack-3 completion test, but
    # computed from the diagnostic values so no task state is advanced.
    external_get_dones_now = xy_pass and z_pass and gripper_open and gripper_clear

    current_stage = None
    if hasattr(raw_env, "current_stage"):
        try:
            current_stage = int(raw_env.current_stage.flatten()[0].detach().cpu().item())
        except Exception:
            current_stage = None

    return {
        "collector_stage_idx": int(collector_stage_idx),
        "env_current_stage": current_stage,
        "collector_step_count": int(step_count),
        "task_metrics_kind": "stack3",
        "thresholds": {
            "xy_m": xy_threshold,
            "z_m": height_threshold,
            "height_diff_m": height_diff,
            "gripper_openness": 0.9,
            "fingertip_cube_min_distance_m": min_gripper_cube_dist,
        },
        "xy_distance_c12_m": float(xy_c12.detach().cpu()),
        "xy_distance_c23_m": float(xy_c23.detach().cpu()),
        "xy_distance_c13_m": float(xy_c13.detach().cpu()),
        "z_error_cube2_on_cube1_m": float(z_error_c2.detach().cpu()),
        "z_error_cube3_on_cube2_m": float(z_error_c3.detach().cpu()),
        "xy_pass": xy_pass,
        "z_pass": z_pass,
        "gripper_openness": gripper_openness,
        "gripper_open": gripper_open,
        "fingertip_cube_distances_m": [
            float(distance.detach().cpu()) for distance in fingertip_dists
        ],
        "gripper_clear": gripper_clear,
        "external_get_dones_now": external_get_dones_now,
        "cube1_pos_w": _torch_vector(cube1_pos),
        "cube2_pos_w": _torch_vector(cube2_pos),
        "cube3_pos_w": _torch_vector(cube3_pos),
        "fingertip_pos_w": _torch_vector(fingertip_pos),
    }


def _estop_open_metrics(raw_env, collector_stage_idx: int, step_count: int) -> dict[str, object]:
    estop = raw_env.estop
    sem = raw_env.cfg.semantic_cfg
    press_pos = _first_float(estop.data.joint_pos[:, raw_env.press_joint_id])
    twist_pos = _first_float(estop.data.joint_pos[:, raw_env.twist_joint_id])

    # Preserve the most-negative twist seen during this collector episode.  The
    # terminal release stage can let the cap spring back before metrics are
    # written, which would otherwise hide whether the unlock threshold crossed.
    previous_step = getattr(raw_env, "_estop_diag_previous_step", None)
    new_episode = previous_step is None or int(step_count) < int(previous_step)
    if new_episode:
        raw_env._estop_diag_min_twist_pos = twist_pos
        raw_env._estop_diag_stage_history = {}
    else:
        raw_env._estop_diag_min_twist_pos = min(
            float(raw_env._estop_diag_min_twist_pos), twist_pos
        )
    raw_env._estop_diag_previous_step = int(step_count)

    api = raw_env.program_runtime.controller_api_wrapper
    cap_tip_pos = api.get_object_xyz_position("estop_cap_tip")
    cap_center_pos = api.get_object_xyz_position("estop_cap_center")
    fingertip_pos = api.lookup_robot_frame_position("panda_fingertip")
    ee_cap_distance = _first_float(torch.linalg.norm(fingertip_pos - cap_tip_pos, dim=-1))
    ee_cap_center_distance = _first_float(
        torch.linalg.norm(fingertip_pos - cap_center_pos, dim=-1)
    )

    twist_effort = None
    try:
        effort = estop._twist_effort()
        if effort is not None:
            twist_effort = _first_float(effort)
    except Exception:
        pass

    gripper_openness = None
    arm_effort = None
    gripper_joint_effort = None
    try:
        gripper_openness = _first_float(api.gripper_openness())
        joint_effort = api.joint_dict[api.reference_agent]["joint_effort"]
        arm_effort = _first_float(torch.sum(torch.abs(joint_effort[:, :7]), dim=-1))
        gripper_joint_effort = _torch_vector(joint_effort[0, -2:])
    except Exception:
        pass

    history = raw_env._estop_diag_stage_history
    stage_key = str(int(collector_stage_idx))
    stage_row = history.setdefault(
        stage_key,
        {
            "min_twist_pos_rad": twist_pos,
            "min_cap_center_distance_m": ee_cap_center_distance,
            "max_arm_effort": arm_effort,
            "max_abs_twist_effort": None if twist_effort is None else abs(twist_effort),
            "min_gripper_openness": gripper_openness,
            "max_gripper_openness": gripper_openness,
        },
    )
    stage_row["min_twist_pos_rad"] = min(stage_row["min_twist_pos_rad"], twist_pos)
    stage_row["min_cap_center_distance_m"] = min(
        stage_row["min_cap_center_distance_m"], ee_cap_center_distance
    )
    if arm_effort is not None:
        previous = stage_row["max_arm_effort"]
        stage_row["max_arm_effort"] = arm_effort if previous is None else max(previous, arm_effort)
    if twist_effort is not None:
        previous = stage_row["max_abs_twist_effort"]
        value = abs(twist_effort)
        stage_row["max_abs_twist_effort"] = value if previous is None else max(previous, value)
    if gripper_openness is not None:
        minimum = stage_row["min_gripper_openness"]
        maximum = stage_row["max_gripper_openness"]
        stage_row["min_gripper_openness"] = (
            gripper_openness if minimum is None else min(minimum, gripper_openness)
        )
        stage_row["max_gripper_openness"] = (
            gripper_openness if maximum is None else max(maximum, gripper_openness)
        )

    current_stage = None
    if hasattr(raw_env, "current_stage"):
        current_stage = int(raw_env.current_stage.flatten()[0].detach().cpu().item())

    # Read the latched result without invoking _external_get_dones() again:
    # that method advances torque persistence and the E-stop state machine.
    external_get_dones_now = _first_bool(raw_env._success)
    if not (
        bool(getattr(raw_env.cfg, "sun_program_evaluator_stage_override", False))
        or bool(getattr(raw_env.cfg, "ignore_stage_gate_for_success", False))
    ):
        stage_count = int(raw_env.program_runtime.compiled_program.stage_count())
        external_get_dones_now = external_get_dones_now and (
            current_stage is not None and current_stage > int(stage_count / 2)
        )

    threshold = float(sem.unlock_twist_angle_rad - sem.unlock_tolerance_rad)
    return {
        "task_metrics_kind": "estop_open",
        "collector_stage_idx": int(collector_stage_idx),
        "env_current_stage": current_stage,
        "collector_step_count": int(step_count),
        "press_pos_m": press_pos,
        "twist_pos_rad": twist_pos,
        "twist_pos_deg": float(np.degrees(twist_pos)),
        "min_twist_pos_rad": float(raw_env._estop_diag_min_twist_pos),
        "min_twist_pos_deg": float(np.degrees(raw_env._estop_diag_min_twist_pos)),
        "unlock_threshold_rad": threshold,
        "unlock_threshold_deg": float(np.degrees(threshold)),
        "estop_is_open": _first_bool(estop.is_open),
        "env_unlocked_once": _first_bool(raw_env._unlocked_once),
        "env_success_latched": _first_bool(raw_env._success),
        "ee_cap_distance_m": ee_cap_distance,
        "ee_cap_center_distance_m": ee_cap_center_distance,
        "ee_far_enough": bool(ee_cap_distance >= 0.05),
        "twist_joint_effort": twist_effort,
        "gripper_openness": gripper_openness,
        "arm_effort": arm_effort,
        "gripper_joint_effort": gripper_joint_effort,
        "stage_history": history,
        "external_get_dones_now": external_get_dones_now,
        "cap_tip_pos_w": _torch_vector(cap_tip_pos[0]),
        "fingertip_pos_w": _torch_vector(fingertip_pos[0]),
    }


def _false_finish_metrics(raw_env, collector_stage_idx: int, step_count: int) -> dict[str, object]:
    if hasattr(raw_env, "_door_open_amounts"):
        angles = raw_env._door_open_amounts()[0].detach().cpu().tolist()
        threshold = float(raw_env.cfg.door_open_threshold)
        return {
            "task_metrics_kind": "cabinet_door", "collector_stage": collector_stage_idx,
            "step": step_count, "door_open_amounts_rad": angles,
            "door_open_threshold_rad": threshold,
            "external_get_dones_now": max(angles) > threshold,
        }
    if hasattr(raw_env, "estop") and hasattr(raw_env, "twist_joint_id"):
        return _estop_open_metrics(raw_env, collector_stage_idx, step_count)
    if hasattr(raw_env, "mug_handle_hole_frame") and hasattr(raw_env, "cup_hanger_branch_1_tip_frame"):
        return _hang_cup_metrics(raw_env, collector_stage_idx, step_count)
    if all(hasattr(raw_env, name) for name in ("cube_1", "cube_2", "cube_3")):
        return _stack3_metrics(raw_env, collector_stage_idx, step_count)

    marker_state = raw_env.marker.data.root_state_w[0]
    holder_state = raw_env.pen_holder.data.root_state_w[0]
    marker_pos = marker_state[:3]
    holder_pos = holder_state[:3]
    marker_quat = marker_state[3:7]
    holder_quat = holder_state[3:7]

    local_y = torch.zeros((1, 3), device=raw_env.device, dtype=marker_quat.dtype)
    local_z = torch.zeros((1, 3), device=raw_env.device, dtype=holder_quat.dtype)
    local_y[:, 1] = 1.0
    local_z[:, 2] = 1.0
    marker_y_world = math_utils.quat_apply(marker_quat.view(1, 4), local_y)[0]
    holder_z_world = math_utils.quat_apply(holder_quat.view(1, 4), local_z)[0]

    xy_dist = torch.linalg.norm(marker_pos[:2] - holder_pos[:2])
    z_dist = torch.abs(marker_pos[2] - holder_pos[2])
    alignment = torch.abs(torch.sum(marker_y_world * holder_z_world)).clamp(0.0, 1.0)
    geometry_pass = bool(
        (xy_dist < raw_env.cfg.success_xy_tolerance)
        and (z_dist < raw_env.cfg.success_z_tolerance)
        and (alignment >= raw_env.cfg.success_upright_alignment)
    )

    gripper_openness = None
    gripper_opened = None
    try:
        gripper_openness = _first_float(raw_env.program_runtime.controller_api_wrapper.gripper_openness())
        gripper_opened = bool(gripper_openness > 0.5)
    except Exception as exc:
        gripper_openness = f"unavailable: {type(exc).__name__}: {exc}"

    external_get_dones_now = None
    try:
        external_get_dones_now = _first_bool(raw_env._external_get_dones())
    except Exception as exc:
        external_get_dones_now = f"unavailable: {type(exc).__name__}: {exc}"

    current_stage = None
    if hasattr(raw_env, "current_stage"):
        try:
            current_stage = int(raw_env.current_stage.flatten()[0].detach().cpu().item())
        except Exception:
            current_stage = None

    return {
        "collector_stage_idx": int(collector_stage_idx),
        "env_current_stage": current_stage,
        "collector_step_count": int(step_count),
        "thresholds": {
            "xy_m": float(raw_env.cfg.success_xy_tolerance),
            "z_m": float(raw_env.cfg.success_z_tolerance),
            "upright_alignment": float(raw_env.cfg.success_upright_alignment),
            "gripper_openness": 0.5,
        },
        "xy_dist_m": float(xy_dist.detach().cpu()),
        "z_dist_m": float(z_dist.detach().cpu()),
        "upright_alignment": float(alignment.detach().cpu()),
        "geometry_pass_xy_z_orientation": geometry_pass,
        "gripper_openness": gripper_openness,
        "gripper_opened": gripper_opened,
        "external_get_dones_now": external_get_dones_now,
        "sun_program_evaluator_stage_override": bool(getattr(raw_env.cfg, "sun_program_evaluator_stage_override", False)),
        "marker_pos_w": _torch_vector(marker_pos),
        "holder_pos_w": _torch_vector(holder_pos),
        "marker_quat_wxyz": _torch_vector(marker_quat),
        "holder_quat_wxyz": _torch_vector(holder_quat),
        "marker_y_axis_w": _torch_vector(marker_y_world),
        "holder_z_axis_w": _torch_vector(holder_z_world),
        "marker_lin_speed_mps": float(torch.linalg.norm(marker_state[7:10]).detach().cpu()),
        "marker_ang_speed_radps": float(torch.linalg.norm(marker_state[10:13]).detach().cpu()),
    }


def main(cleanup):
    """Random actions agent with Isaac Lab environment."""
    # ================================
    # Set the agent program path
    # Cannot be changed once the environment is created
    env_cfg = parse_env_cfg(
        ENV_NAME,
        num_envs=args_cli.num_envs,
    )
    if args_cli.program_path is not None:
        program_path = Path(args_cli.program_path).expanduser().resolve()
        if not program_path.is_dir():
            raise FileNotFoundError(
                f"SUNProgramEvaluator program path is not a directory: {program_path}"
            )
        if not (program_path / "program").is_dir():
            raise FileNotFoundError(
                f"SUNProgramEvaluator run has no program/ directory: {program_path}"
            )
        env_cfg.mpc_config["load_program"] = str(program_path)
        print(f"[worker {WORKER_ID}] learned-program path: {program_path}")
    if args_cli.seed is not None:
        seed = int(args_cli.seed)
        env_cfg.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    default_output_mode = (
        "absolute_joint" if env_cfg.mpc_config.get("controller_name") == "UR5IKController"
        else "absolute_ee"
    )
    env_cfg.mpc_config["control_output_mode"] = args_cli.control_output_mode or default_output_mode
    env_cfg.mpc_config['use_per_env_weight'] = False
    env_cfg.enable_cameras = bool(RECORD_VIDEO)
    env_cfg.debug_rewards = DEBUG_MESSAGES
    if not DEBUG_MESSAGES:
        for debug_print_attr in (
            "reward_print_every",
            "gripper_force_print_every",
            "episode_outcome_print_every",
            "manipulator_effort_print_every",
        ):
            if hasattr(env_cfg, debug_print_attr):
                setattr(env_cfg, debug_print_attr, 0)
    if args_cli.gripper_force_print_every is not None:
        if not hasattr(env_cfg, "gripper_force_print_every"):
            raise AttributeError(
                f"{type(env_cfg).__name__} has no gripper_force_print_every setting"
            )
        env_cfg.gripper_force_print_every = max(
            0, int(args_cli.gripper_force_print_every)
        )
    env_cfg.sun_program_evaluator_stage_override = True
    control_output_mode = env_cfg.mpc_config["control_output_mode"]
    env_cfg.action_mode = control_output_mode
    project_root = Path(script_dir).resolve()
    configured = os.environ.get("KUAFU_ARTIFACT_ROOT", "").strip()
    external_root = (
        Path(configured).expanduser().resolve()
        if configured
        else project_root.parent / "KuafuArtifacts"
    )
    if args_cli.output_dir is not None:
        output_path = Path(args_cli.output_dir).expanduser()
        if not output_path.is_absolute():
            output_path = external_root / output_path
        output_path = output_path.resolve()
    else:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_path = (
            external_root
            / "standalone"
            / "mpc_collection"
            / ENV_NAME
            / timestamp
        )
    if output_path == project_root or project_root in output_path.parents:
        raise ValueError(
            f"Trajectory output must be outside the source project ({project_root}); got {output_path}"
        )
    GLOBAL_OUTPUT_DIR = str(output_path)
    dataset_dir = str(output_path / "datasets")
    os.makedirs(dataset_dir, exist_ok=True)
    print(f"[worker {WORKER_ID}] output dir: {GLOBAL_OUTPUT_DIR}")
    print(f"[worker {WORKER_ID}] sim device: {env_cfg.sim.device}")
    env_cfg.recorders = None
    env = gym.make(ENV_NAME, cfg=env_cfg)
    cleanup.callback(env.close)
    env.reset()
    # Ensure PhysX → Python caches (scene.state) are initialized.
    # Without this, early reads of env.unwrapped.scene.state[...] can be NaN.
    env.unwrapped.scene.update(env.unwrapped.physics_dt)
    sim_dt = env.unwrapped.dt
    if args_cli.use_recorder:
        env.unwrapped.recorder_manager = StateTransitionRecorder(
            env.unwrapped, Path(dataset_dir) / f"mpc_demos_worker{WORKER_ID}.hdf5", NUM_DEMOS,
            metadata={"env_name": ENV_NAME, "seed": args_cli.seed,
                      "program_path": env_cfg.mpc_config.get("load_program"),
                      "control_dt": float(sim_dt), "action_mode": env_cfg.action_mode,
                      "controller": "mpc"},
            record_failures=args_cli.record_failures, record_scene=True,
        )
        cleanup.callback(env.unwrapped.recorder_manager.close)

    capture_dir = None
    false_finish_capturer = None
    captured_false_finish_count = 0
    if CAPTURE_FALSE_FINISH:
        capture_dir = resolve_artifact_path(
            args_cli.capture_false_finish_dir,
            "False-finish diagnostic directory",
        )
        capture_dir.mkdir(parents=True, exist_ok=True)
        if not args_cli.capture_false_finish_metrics_only:
            false_finish_capturer = ViewportCapturer(
                args_cli.capture_false_finish_width,
                args_cli.capture_false_finish_height,
            )
        capture_kind = "metrics" if false_finish_capturer is None else "screenshots and metrics"
        print(f"[worker {WORKER_ID}] false-finish {capture_kind}: {capture_dir}")

    video_writer = None
    video_capturer = None
    recorded_video_frames = 0
    if RECORD_VIDEO:
        video_path = resolve_artifact_path(
            args_cli.record_video_path,
            "MPC collection video",
        )
        video_capturer = ViewportCapturer(
            args_cli.record_video_width,
            args_cli.record_video_height,
        )
        video_writer = FfmpegRgbWriter(
            video_path,
            args_cli.record_video_fps,
            args_cli.record_video_width,
            args_cli.record_video_height,
        )
        print(f"[worker {WORKER_ID}] recording viewport video: {video_path}")

    estop = env.unwrapped.scene.articulations.get("estop")
    if DEBUG_MESSAGES and estop is not None:
        try:
            twist_joint_id = estop.find_joints(env.unwrapped.cfg.semantic_cfg.twist_joint_name)[0][0]
            hard = estop.data.joint_pos_limits[0, twist_joint_id].detach().cpu()
            soft = estop.data.soft_joint_pos_limits[0, twist_joint_id].detach().cpu()
            pos = estop.data.joint_pos[0, twist_joint_id].detach().cpu()
            print(
                "[estop debug] twist hard limits rad/deg:",
                hard.tolist(),
                torch.rad2deg(hard).tolist(),
            )
            print(
                "[estop debug] twist soft limits rad/deg:",
                soft.tolist(),
                torch.rad2deg(soft).tolist(),
                "current rad/deg:",
                float(pos),
                float(torch.rad2deg(pos)),
            )
        except Exception as exc:
            print(f"[estop debug] failed to read twist limits: {exc}")

    # Extract per-stage constraints directly from the env's SUNProgramEvaluator program_runtime
    program_evaluator = getattr(env.unwrapped, "program_runtime")
    num_stages = program_evaluator.compiled_program.stage_count()
    CONSTRAINTS = [
        copy.deepcopy(program_evaluator.get_constraints_by_stage(stage_id))
        for stage_id in range(num_stages)
    ]


    # ------------------------------------------------------------------
    # set up a single SUNProgramExecutor controller for one environment
    # ------------------------------------------------------------------

    if env_cfg.mpc_config.get("controller_name") == "UR5IKController":
        robot_name = "ur5"
    elif "panda_shoulder" in env_cfg.robot.actuators:
        robot_name = "panda"
    elif "fr3_shoulder" in env_cfg.robot.actuators:
        robot_name = "fr3"
    elif "kinova_shoulder" in env_cfg.robot.actuators:
        robot_name = "kinova"
    else:
        raise ValueError("Invalid robot name; expected panda, fr3, kinova, or ur5")

    if robot_name == "panda":
        env_mpc_config = env_cfg.mpc_config
        panda_debug = ENV_NAME.startswith("Panda-Table-")
        # MPC / Agent configuration for SUNProgramExecutor (mirrors cabinet_test_constraint_torch_stage)
        MPC_CONFIG = {
            "controller_name": "FrankaIKController",
            "device": env_cfg.sim.device,
            "gripper_opened": env_mpc_config.get("gripper_opened", [0.04, 0.04]),
            "gripper_closed": env_mpc_config.get("gripper_closed", [0.0, 0.0]),
            "gripper_max_force": env_mpc_config.get("gripper_max_force", 40.0),
            "manipulator_high_effort_threshold": env_mpc_config.get(
                "manipulator_high_effort_threshold", 8.0
            ),
            "has_finger_contact_sensors": env_mpc_config.get(
                "has_finger_contact_sensors", not panda_debug
            ),
            "object_grasp_debug": DEBUG_MESSAGES and env_mpc_config.get("object_grasp_debug", False),
            "object_grasp_debug_every": env_mpc_config.get("object_grasp_debug_every", 25),
            "robot_gripper_name": env_mpc_config.get(
                "robot_gripper_name",
                "franka_panda" if panda_debug else "panda",
            ),
            "controller": {
                "urdf_path": f"{script_dir}/resources/robots/franka_description/robots/panda_arm_hand_fixed.urdf",
                "ee_frame": env_mpc_config.get("ee_frame", "panda_fingertip"),
                "horizon_length": 4,
                "time_step": sim_dt,
            },
            "load_program": None,
            "control_output_mode": control_output_mode,
            "agent": {
                "id": "agent",
                "device": env_cfg.sim.device,
                "task": "SUNProgramExecutor control for pen_cup_recall_batched",
                "prompt_dir": f"{script_dir}/assets/prompts/Manipulation",
                "initial_residual": f"{script_dir}/src/kuafu/skills/constraints/residuals.py",
                "initial_predicate": f"{script_dir}/src/kuafu/skills/constraints/predicates.py",
                "model_version": "offline",
                "load_path": None,
                "reuse_program": True,
                "max_correction_steps": 5,
            },
        }
    elif robot_name == "fr3":
        env_mpc_config = env_cfg.mpc_config
        MPC_CONFIG = {
            "controller_name": "FR3FinRayIKController",
            "device": env_cfg.sim.device,
            "gripper_opened": [0.0, 0.0],
            "gripper_closed": [0.8, -0.8],
            "gripper_max_force": 1.0,
            "finger_contact_force_mode": env_mpc_config.get(
                "finger_contact_force_mode", "closing_axis_projected"
            ),
            "object_grasp_debug": DEBUG_MESSAGES and env_mpc_config.get("object_grasp_debug", False),
            "object_grasp_debug_every": env_mpc_config.get("object_grasp_debug_every", 100),
            "robot_gripper_name": env_mpc_config.get("robot_gripper_name", "fr3_finray"),
            "controller": {
                "urdf_path": f"{script_dir}/resources/robots/franka_description/robots/panda_arm_hand_fr3_finray.urdf",
                "ee_frame": "panda_fingertip",
                "horizon_length": 4,
                "time_step": sim_dt,
            },
            "load_program": None,
            "control_output_mode": control_output_mode,
            "agent": {
                "id": "agent",
                "device": env_cfg.sim.device,
                "task": "lalalal",
                "prompt_dir": f"{script_dir}/assets/prompts/Manipulation",
                "initial_residual": f"{script_dir}/src/kuafu/skills/constraints/residuals.py",
                "initial_predicate": f"{script_dir}/src/kuafu/skills/constraints/predicates.py",
                "model_version": "offline",
                "load_path": None,
                "reuse_program": True,
                "max_correction_steps": 10,
            },
        }
    elif robot_name in ("kinova", "ur5"):
        env_mpc_config = env_cfg.mpc_config
        MPC_CONFIG = {
            "controller_name": env_mpc_config["controller_name"],
            "device": env_cfg.sim.device,
            "gripper_opened": env_mpc_config.get("gripper_opened", [0.0, 0.0]),
            "gripper_closed": env_mpc_config.get("gripper_closed", [0.8, 0.8]),
            "gripper_max_force": env_mpc_config.get("gripper_max_force", 1.0),
            "gripper_effort_contact_velocity_threshold": env_mpc_config.get(
                "gripper_effort_contact_velocity_threshold", 0.15
            ),
            "kinova_gripper_effort_contact_openness_slope": env_mpc_config.get(
                "kinova_gripper_effort_contact_openness_slope", 12.0
            ),
            "kinova_gripper_effort_contact_openness_intercept": env_mpc_config.get(
                "kinova_gripper_effort_contact_openness_intercept", 0.0
            ),
            "manipulator_high_effort_threshold": env_mpc_config.get(
                "manipulator_high_effort_threshold", 8.0
            ),
            "effort_lpf_alpha": env_mpc_config.get("effort_lpf_alpha", 0.2),
            "has_finger_contact_sensors": False,
            "finger_contact_force_mode": env_mpc_config.get(
                "finger_contact_force_mode", "closing_axis_projected"
            ),
            "object_grasp_debug": DEBUG_MESSAGES and env_mpc_config.get("object_grasp_debug", False),
            "object_grasp_debug_every": env_mpc_config.get("object_grasp_debug_every", 100),
            "robot_gripper_name": env_mpc_config.get("robot_gripper_name", "kinova_robotiq"),
            "controller": {
                "urdf_path": env_mpc_config["urdf_path"],
                "ee_frame": env_mpc_config.get("ee_frame", "panda_fingertip"),
                "horizon_length": 4,
                "time_step": sim_dt,
            },
            "load_program": None,
            "control_output_mode": control_output_mode,
            "agent": {
                "id": "agent",
                "device": env_cfg.sim.device,
                "task": "SUNProgramExecutor control for Kinova FR3FinRay manipulation",
                "prompt_dir": f"{script_dir}/assets/prompts/Manipulation",
                "initial_residual": f"{script_dir}/src/kuafu/skills/constraints/residuals.py",
                "initial_predicate": f"{script_dir}/src/kuafu/skills/constraints/predicates.py",
                "model_version": "offline",
                "load_path": None,
                "reuse_program": True,
                "max_correction_steps": 10,
            },
        }
    cfg_sun_program_executor = OmegaConf.create(MPC_CONFIG)

    # Optional: Differential IK hookup for better kinematics
    inverse_kinematics = create_differential_ik_pose_function(env, env_cfg, robot_name)
    program_executor = SUNProgramExecutor(cfg_sun_program_executor)
    program_executor.active_controller._ik_fun = inverse_kinematics
    if hasattr(program_executor, "test_controller"):
        program_executor.test_controller._ik_fun = inverse_kinematics

    # Get the total number of stages/subtasks (for information only)
    print(f"\nTotal number of stages in program_runtime (SUNProgramEvaluator): {program_evaluator.compiled_program.stage_count()}")

    # Simulation start here
    RECOMPUTE_CALL_INTERVAL = 3
    RECOMPILE_CALL_INTERVAL = RECOMPUTE_CALL_INTERVAL * 10
    STALL_THRESHOLD = 60
    PREDICATE_DEBUG_INTERVAL = max(1, int(os.getenv("KUAFU_PREDICATE_DEBUG_INTERVAL", "100")))
    GRASP_DEBUG = DEBUG_MESSAGES and os.getenv("KUAFU_GRASP_DEBUG", "").strip().lower() in ("1", "true", "yes")
    GRASP_DEBUG_EVERY = max(1, int(os.getenv("KUAFU_GRASP_DEBUG_EVERY", "25")))
    EFFORT_MONITOR = DEBUG_MESSAGES and (
        bool(env_mpc_config.get("effort_monitor", False))
        or os.getenv("KUAFU_EFFORT_MONITOR", "").strip().lower() in ("1", "true", "yes")
    )
    EFFORT_MONITOR_EVERY = max(
        1,
        int(os.getenv("KUAFU_EFFORT_MONITOR_EVERY", env_mpc_config.get("effort_monitor_every", 25))),
    )
    EFFORT_MONITOR_STAGES = {
        int(stage) for stage in env_mpc_config.get("effort_monitor_stages", [])
    }
    MOVE_DEBUG = DEBUG_MESSAGES and os.getenv("KUAFU_MOVE_DEBUG", "").strip().lower() in ("1", "true", "yes")
    POSE_DEBUG = DEBUG_MESSAGES and os.getenv("KUAFU_POSE_DEBUG", "").strip().lower() in ("1", "true", "yes")
    GRASP_POSE_DEBUG = DEBUG_MESSAGES and os.getenv(
        "KUAFU_GRASP_POSE_DEBUG", ""
    ).strip().lower() in ("1", "true", "yes")
    GRASP_POSE_DEBUG_EVERY = max(
        1, int(os.getenv("KUAFU_GRASP_POSE_DEBUG_EVERY", "5"))
    )
    CONTROL_FRAME_DEBUG = DEBUG_MESSAGES and os.getenv(
        "KUAFU_CONTROL_FRAME_DEBUG", ""
    ).strip().lower() in ("1", "true", "yes")
    STACK2_CAPTURE_DEBUG = DEBUG_MESSAGES and os.getenv("KUAFU_STACK2_CAPTURE_DEBUG", "").strip().lower() in ("1", "true", "yes")
    pose_debug_printed_names = False
    stack2_debug_captured_stages = set()
    count = 0
    total_steps = 0
    RECOMPUTE_CALL_COUNTER = 0
    RECOMPILE_CALL_COUNTER = 0
    stage_idx = 0
    dummy_actions = torch.zeros(
        (args_cli.num_envs, env_cfg.action_space),
        device=env_cfg.sim.device,
        dtype=torch.float32,
    )
    env.unwrapped.scene.update(env.unwrapped.physics_dt)
    print("\nStarting main simulation loop...\n")

    # let simulation stablize, not moving
    update_sun_program_executor_data_for_env(env, program_executor, robot_name)
    program_executor.controller_api_wrapper.updatePOVState()
    program_executor.program_learner._constraint_generator._constraints = CONSTRAINTS[0]
    program_executor.program_learner.apply_constraints()

    def outcome_video_done() -> bool:
        if not CAPTURE_OUTCOME_VIDEO:
            return False
        return all(
            outcome_video_counts[name] >= target
            for name, target in OUTCOME_VIDEO_TARGETS.items()
            if target > 0
        )

    def capture_outcome_video_frame(*, force: bool = False) -> None:
        if not CAPTURE_OUTCOME_VIDEO or outcome_video_capturer is None:
            return
        every = max(1, int(args_cli.capture_outcome_video_every))
        if not force and count % every != 0:
            return
        raw_env = env.unwrapped
        try:
            _set_false_finish_camera(raw_env, FALSE_FINISH_CAMERA_SPECS[0])
            raw_env.sim.render()
            simulation_app.update()
            frame = outcome_video_capturer.capture_rgb()
        except Exception as exc:
            print(f"[outcome-video] frame capture failed: {type(exc).__name__}: {exc}")
            return
        outcome_video_frames.append(frame)
        max_frames = max(1, int(args_cli.capture_outcome_video_max_frames))
        if len(outcome_video_frames) > max_frames:
            del outcome_video_frames[: len(outcome_video_frames) - max_frames]

    def record_outcome_video(
        outcome: str,
        collector_stage_idx: int,
        step_count: int,
        pred_text: str,
        *,
        terminal_source: str,
        extra: dict[str, object] | None = None,
        metrics_override: dict[str, object] | None = None,
    ) -> bool:
        if not CAPTURE_OUTCOME_VIDEO or outcome_video_dir is None:
            return outcome_video_done()
        if outcome not in OUTCOME_VIDEO_TARGETS:
            raise ValueError(f"Unknown outcome video class: {outcome}")
        target = OUTCOME_VIDEO_TARGETS[outcome]
        if target <= 0 or outcome_video_counts[outcome] >= target:
            return outcome_video_done()

        capture_outcome_video_frame(force=True)
        if not outcome_video_frames:
            print(f"[outcome-video] no frames available for {outcome}; skipping save")
            return outcome_video_done()

        idx = outcome_video_counts[outcome] + 1
        event_dir = outcome_video_dir / outcome
        video_path = event_dir / f"{outcome}_{idx:03d}_stage{collector_stage_idx}_step{step_count}.mp4"
        metadata_path = event_dir / f"{outcome}_{idx:03d}_stage{collector_stage_idx}_step{step_count}.json"
        pred_path = event_dir / f"{outcome}_{idx:03d}_stage{collector_stage_idx}_step{step_count}_sun_program_executor_predicate.txt"

        frames_to_write = list(outcome_video_frames)
        _write_mp4(video_path, frames_to_write, args_cli.capture_outcome_video_fps)
        outcome_video_counts[outcome] = idx

        metadata = {
            "outcome": outcome,
            "terminal_source": terminal_source,
            "collector_stage_idx": int(collector_stage_idx),
            "collector_step_count": int(step_count),
            "frame_count": len(frames_to_write),
            "fps": int(args_cli.capture_outcome_video_fps),
            "video_path": str(video_path),
            "counts_after": dict(outcome_video_counts),
            "target_counts": dict(OUTCOME_VIDEO_TARGETS),
        }
        if extra is not None:
            metadata["extra"] = extra
        if metrics_override is not None:
            metadata["terminal_metrics"] = metrics_override
        else:
            try:
                metadata["terminal_metrics"] = _false_finish_metrics(env.unwrapped, collector_stage_idx, step_count)
            except Exception as exc:
                metadata["terminal_metrics_error"] = f"{type(exc).__name__}: {exc}"

        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
        pred_path.write_text(pred_text)
        print(
            f"[outcome-video] saved {outcome} {idx}/{target}: {video_path} "
            f"source={terminal_source} counts={outcome_video_counts}"
        )
        return outcome_video_done()

    def _is_stopped():
        if args_cli.use_recorder:
            rm = getattr(env.unwrapped, "recorder_manager")
            total = rm.exported_successful_episode_count
            if args_cli.record_failures:
                total += rm.exported_failed_episode_count
            return total >= NUM_DEMOS
        return False

    def reset_sun_program_executor(idx=0):
        stage_idx = idx
        update_sun_program_executor_data_for_env(env, program_executor, robot_name)
        program_executor.controller_api_wrapper.updatePOVState()
        program_executor.controller_api_wrapper.stall_joint_counter = None
        program_executor.controller_api_wrapper.stall_cost_counter = None
        program_executor.program_learner._constraint_generator._constraints = CONSTRAINTS[stage_idx]
        program_executor.program_learner.apply_constraints()

    def reset_episode_and_sun_program_executor(idx=0, *, suppress_recorder_export=False):
        # Refresh controller state after reset; an extra autoreset must not export twice.
        rm = getattr(env.unwrapped, "recorder_manager", None) if args_cli.use_recorder else None
        prev_export_in_record_pre_reset = None
        if suppress_recorder_export and rm is not None:
            prev_export_in_record_pre_reset = rm.cfg.export_in_record_pre_reset
            rm.cfg.export_in_record_pre_reset = False
        try:
            env.reset()
        finally:
            if prev_export_in_record_pre_reset is not None:
                rm.cfg.export_in_record_pre_reset = prev_export_in_record_pre_reset
        if CAPTURE_OUTCOME_VIDEO:
            outcome_video_frames.clear()
        env.unwrapped.scene.update(env.unwrapped.physics_dt)
        reset_sun_program_executor(idx)

    def capture_false_finish_event(collector_stage_idx: int, step_count: int, pred_text: str) -> bool:
        nonlocal captured_false_finish_count
        if capture_dir is None:
            return False
        captured_false_finish_count += 1
        event_dir = capture_dir / f"false_finish_{captured_false_finish_count:03d}"
        event_dir.mkdir(parents=True, exist_ok=True)

        raw_env = env.unwrapped
        metrics = _false_finish_metrics(raw_env, collector_stage_idx, step_count)
        camera_rows = []
        images = []
        if false_finish_capturer is not None:
            for spec in FALSE_FINISH_CAMERA_SPECS:
                row = _set_false_finish_camera(raw_env, spec)
                for _ in range(max(0, int(args_cli.capture_false_finish_settle_frames))):
                    raw_env.sim.render()
                    simulation_app.update()
                frame = false_finish_capturer.capture_rgb()
                image_path = event_dir / f"{spec['name']}.png"
                _write_png(image_path, frame)
                camera_rows.append(row)
                images.append(str(image_path))

        metrics["camera_views"] = camera_rows
        metrics["images"] = images
        (event_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
        (event_dir / "sun_program_executor_predicate.txt").write_text(pred_text)
        if metrics.get("task_metrics_kind") == "stack3":
            metric_summary = (
                f"xy_pass={metrics['xy_pass']} z_pass={metrics['z_pass']} "
                f"grip={metrics['gripper_open']} clear={metrics['gripper_clear']} "
                f"env_done={metrics['external_get_dones_now']}"
            )
        elif metrics.get("task_metrics_kind") == "estop_open":
            metric_summary = (
                f"open={metrics['estop_is_open']} "
                f"twist_deg={metrics['twist_pos_deg']:.2f} "
                f"torque={metrics['twist_joint_effort']} "
                f"env_done={metrics['external_get_dones_now']}"
            )
        else:
            metric_summary = (
                f"pose_ok={metrics.get('pose_ok', metrics.get('geometry_pass_xy_z_orientation'))} "
                f"grip={metrics.get('gripper_open', metrics.get('gripper_opened'))} "
                f"env_done={metrics.get('external_get_dones_now')}"
            )
        print(
            f"Captured false-finish #{captured_false_finish_count}: {event_dir} "
            f"{metric_summary}"
        )
        return captured_false_finish_count >= args_cli.capture_false_finish_count

    last_pre_step_metrics = None
    while simulation_app.is_running() and not _is_stopped() and not outcome_video_done():
        with torch.inference_mode():

            update_sun_program_executor_data_for_env(env, program_executor, robot_name)

            sun_program_executor_pred, pred_text = program_executor.quantifyPredicate(text=True)
            is_success = sun_program_executor_pred[0].item()
            try:
                last_pre_step_metrics = _false_finish_metrics(env.unwrapped, stage_idx, count)
            except Exception as exc:
                last_pre_step_metrics = {"metrics_error": f"{type(exc).__name__}: {exc}"}

            if DEBUG_MESSAGES and count % 100 == 0 and last_pre_step_metrics.get("task_metrics_kind") == "cabinet_door":
                print(f"[cabinet physical] {json.dumps(last_pre_step_metrics)}", flush=True)

            if DEBUG_MESSAGES and not is_success and (count % PREDICATE_DEBUG_INTERVAL == 0):
                print(
                    f"[stage {stage_idx} | step {count}] predicates not yet satisfied:\n"
                    f"{pred_text}"
                )

            if GRASP_DEBUG and stage_idx == 2 and (count % GRASP_DEBUG_EVERY == 0):
                api = program_executor.controller_api_wrapper
                openness = api.gripper_openness().flatten()
                left_force, right_force = api.finger_contact_force()
                openness_clamped = torch.clamp(openness, min=0.03, max=0.8)
                openness_alpha = (openness_clamped - 0.03) / (0.8 - 0.03)
                required_force = api._gripper_max_force * (0.1 + 0.6 * openness_alpha)
                gripper_effort = api.joint_dict[api.reference_agent]["joint_effort"][:, -2:]
                gripper_pos = api.joint_dict[api.reference_agent]["joint_position"][:, -2:]
                print(
                    "[grasp_debug] "
                    f"stage={stage_idx} step={count} open={openness.detach().cpu().tolist()} "
                    f"effort={gripper_effort.detach().cpu().tolist()} "
                    f"finger_force=({left_force.detach().cpu().tolist()}, {right_force.detach().cpu().tolist()}) "
                    f"required={required_force.detach().cpu().tolist()} "
                    f"gripper_pos={gripper_pos.detach().cpu().tolist()}"
                )

            if (
                EFFORT_MONITOR
                and (not EFFORT_MONITOR_STAGES or stage_idx in EFFORT_MONITOR_STAGES)
                and (count % EFFORT_MONITOR_EVERY == 0)
            ):
                api = program_executor.controller_api_wrapper
                joint_effort = api.joint_dict[api.reference_agent]["joint_effort"]
                arm_effort = torch.sum(torch.abs(joint_effort[:, :7]), dim=-1)
                gripper_effort = joint_effort[:, -2:]
                threshold = float(
                    getattr(
                        api,
                        "_manipulator_high_effort_threshold",
                        getattr(program_executor, "_manipulator_high_effort_threshold", 8.0),
                    )
                )
                openness = api.gripper_openness().flatten()
                openness_clamped = torch.clamp(openness, min=0.03, max=0.8)
                openness_alpha = (openness_clamped - 0.03) / (0.8 - 0.03)
                required_force = api._gripper_max_force * (0.1 + 0.6 * openness_alpha)
                if getattr(api, "has_finger_contact_sensors", True):
                    left_force, right_force = api.finger_contact_force()
                else:
                    left_force, right_force = gripper_effort[:, 0], gripper_effort[:, 1]
                print(
                    "[effort_monitor] "
                    f"stage={stage_idx} step={count} "
                    f"manipulator_sum_abs={arm_effort.detach().cpu().tolist()} "
                    f"threshold={threshold:.3f} "
                    f"high={(arm_effort > threshold).detach().cpu().tolist()} "
                    f"gripper_effort={gripper_effort.detach().cpu().tolist()} "
                    f"finger_force=({left_force.detach().cpu().tolist()}, {right_force.detach().cpu().tolist()}) "
                    f"required_grasp_force={required_force.detach().cpu().tolist()} "
                    f"gripper_max_force={float(api._gripper_max_force):.3f}"
                )

            grasp_frame_name = None
            if ENV_NAME == "Panda-Table-Hang-Cup" and stage_idx == 2:
                grasp_frame_name = "mug_rim_frame_0"
            elif ENV_NAME == "Panda-Table-Drawer-Stack":
                if stage_idx == 2:
                    grasp_frame_name = "top_drawer_handle_frame"
                elif stage_idx == 8:
                    grasp_frame_name = "strawberry_center_frame"
            if (
                GRASP_POSE_DEBUG
                and grasp_frame_name is not None
                and count % GRASP_POSE_DEBUG_EVERY == 0
            ):
                api = program_executor.controller_api_wrapper
                fingertip_pos = api.lookup_robot_frame_position(
                    "panda_fingertip", "real time"
                )
                grasp_frame_pos = api.get_object_xyz_position(
                    grasp_frame_name, "real time"
                )
                position_error = fingertip_pos - grasp_frame_pos
                position_error_norm = torch.linalg.norm(position_error, dim=-1)
                print(
                    "[grasp_pose_debug] "
                    f"stage={stage_idx} step={count} frame={grasp_frame_name} "
                    f"fingertip={fingertip_pos.detach().cpu().tolist()} "
                    f"grasp_frame={grasp_frame_pos.detach().cpu().tolist()} "
                    f"error_world={position_error.detach().cpu().tolist()} "
                    f"error_norm={position_error_norm.detach().cpu().tolist()}"
                )

            if MOVE_DEBUG and stage_idx == 3 and (count % 25 == 0):
                api = program_executor.controller_api_wrapper
                pos_ref = api.get_object_xyz_position("cube_1", "current")
                pos_now = api.get_object_xyz_position("cube_1", "real time")
                disp = pos_now - pos_ref
                axial = disp[:, 2]
                lateral = torch.linalg.norm(disp[:, :2], dim=-1)
                dist = torch.sqrt((axial - 0.15) ** 2 + (lateral ** 2) / 2.0)
                ee_ref = api.lookup_robot_frame_position("panda_fingertip", "current")
                ee_now = api.lookup_robot_frame_position("panda_fingertip", "real time")
                ee_disp = ee_now - ee_ref
                print(
                    "[move_debug] "
                    f"stage={stage_idx} step={count} pos_ref={pos_ref.detach().cpu().tolist()} "
                    f"pos_now={pos_now.detach().cpu().tolist()} disp={disp.detach().cpu().tolist()} "
                    f"axial={axial.detach().cpu().tolist()} lateral={lateral.detach().cpu().tolist()} "
                    f"dist={dist.detach().cpu().tolist()} "
                    f"ee_disp={ee_disp.detach().cpu().tolist()}"
                )

            if POSE_DEBUG and stage_idx in (1, 2, 3) and (count % 25 == 0):
                api = program_executor.controller_api_wrapper
                raw_env = env.unwrapped
                robot_data = raw_env.scene.articulations["robot"].data
                body_names = list(getattr(robot_data, "body_names", []))
                if not pose_debug_printed_names:
                    print(f"[pose_debug] body_names={body_names}")
                    pose_debug_printed_names = True
                wanted = ("bracelet", "finger", "knuckle")
                body_rows = []
                for body_id, body_name in enumerate(body_names):
                    if any(token in body_name for token in wanted):
                        pos = robot_data.body_pos_w[0, body_id].detach().cpu().tolist()
                        body_rows.append((body_name, pos))
                cube1 = raw_env.cube_1.data.root_pos_w[0].detach().cpu().tolist() if hasattr(raw_env, "cube_1") else None
                cube2 = raw_env.cube_2.data.root_pos_w[0].detach().cpu().tolist() if hasattr(raw_env, "cube_2") else None
                ft = api.lookup_robot_frame_position("panda_fingertip", "real time").detach().cpu().tolist()
                fq = api.lookup_robot_frame_rotation("panda_fingertip", "real time").detach().cpu().tolist()
                print(
                    "[pose_debug] "
                    f"stage={stage_idx} step={count} cube1={cube1} cube2={cube2} "
                    f"fingertip={ft} fingertip_quat={fq} bodies={body_rows}"
                )

            if (
                STACK2_CAPTURE_DEBUG
                and false_finish_capturer is not None
                and capture_dir is not None
                and ENV_NAME == "Kinova-Table-Gen3-Stack-2"
                and stage_idx in (2, 3)
                and stage_idx not in stack2_debug_captured_stages
            ):
                raw_env = env.unwrapped
                focus = raw_env.cube_1.data.root_pos_w[0].detach()
                eye = (
                    float(focus[0].item() + 0.18),
                    float(focus[1].item() - 0.42),
                    float(focus[2].item() + 0.26),
                )
                target = (
                    float(focus[0].item()),
                    float(focus[1].item()),
                    float(focus[2].item() + 0.05),
                )
                raw_env.sim.set_camera_view(eye=eye, target=target, camera_prim_path="/OmniverseKit_Persp")
                for _ in range(max(1, int(args_cli.capture_false_finish_settle_frames))):
                    raw_env.sim.render()
                    simulation_app.update()
                frame = false_finish_capturer.capture_rgb()
                image_path = capture_dir / f"stack2_stage_{stage_idx}_step_{count}.png"
                _write_png(image_path, frame)
                stack2_debug_captured_stages.add(stage_idx)
                print(f"[stack2_capture_debug] wrote {image_path}")


            if count == RECOMPUTE_CALL_COUNTER:
                try:
                    u_seq = program_executor.computeControl()
                except Exception as exc:  # noqa: BLE001 - reset this episode on MPC solver failures.
                    print(
                        f"MPC computeControl failed at stage={stage_idx} step={count}: "
                        f"{type(exc).__name__}: {exc}. Resetting episode and continuing."
                    )
                    traceback.print_exc()
                    video_done = record_outcome_video(
                        "timeout",
                        stage_idx,
                        count,
                        pred_text,
                        terminal_source="mpc_compute_failure",
                        extra={"exception": f"{type(exc).__name__}: {exc}"},
                        metrics_override=last_pre_step_metrics,
                    )
                    stage_idx = 0
                    count = 0
                    RECOMPUTE_CALL_COUNTER = 0
                    RECOMPILE_CALL_COUNTER = 0
                    if video_done:
                        break
                    reset_episode_and_sun_program_executor()
                    continue
                last_computed_count = count
                RECOMPUTE_CALL_COUNTER += RECOMPUTE_CALL_INTERVAL

            u_tensor = torch.from_numpy(
                u_seq[(count - last_computed_count+1)]
            ).to(env_cfg.sim.device).to(dtype=torch.float32)
            if control_output_mode == "relative_ee":
                # MPC already outputs EE velocities; the env converts normalized
                # actions back to per-step deltas by multiplying by dt.
                u_tensor[:6] = u_tensor[:6] / env_cfg.ee_action_scale
            action = u_tensor.unsqueeze(0)
            if args_cli.action_noise_std > 0:
                action = action + torch.randn_like(action) * args_cli.action_noise_std
            if CONTROL_FRAME_DEBUG and stage_idx in (1, 2, 7, 8):
                api = program_executor.controller_api_wrapper
                frame_name = "panda_fingertip"
                actual_pos = api.lookup_robot_frame_position(
                    frame_name, "real time"
                )
                snapshot_pos = api.lookup_robot_frame_position(
                    frame_name, "current"
                )
                pin_frame_id = program_executor.active_controller.model.getFrameId(
                    frame_name
                )
                pin_pos = (
                    program_executor.active_controller.data.oMf[
                        pin_frame_id
                    ].translation.copy()
                )
                print(
                    "[control_frame_debug] "
                    f"stage={stage_idx} step={count} "
                    f"actual_pos={actual_pos.detach().cpu().tolist()} "
                    f"snapshot_pos={snapshot_pos.detach().cpu().tolist()} "
                    f"pin_fk_pos={pin_pos.tolist()} "
                    f"horizon_pos={u_seq[:, :3].tolist()} "
                    f"selected_raw_pos={u_tensor[:3].detach().cpu().tolist()} "
                    f"selected_noisy_pos={action[0, :3].detach().cpu().tolist()}"
                )
            capture_outcome_video_frame()
            count += 1
            total_steps += 1
            _, _, terminated, truncated, _ = env.step(action)
            if video_writer is not None and count % args_cli.record_video_every == 0:
                _set_false_finish_camera(env.unwrapped, FALSE_FINISH_CAMERA_SPECS[0])
                for _ in range(max(0, int(args_cli.record_video_settle_frames))):
                    env.unwrapped.sim.render()
                    simulation_app.update()
                video_writer.write(video_capturer.capture_rgb())
                recorded_video_frames += 1
            if args_cli.max_steps > 0 and total_steps >= args_cli.max_steps:
                print(f"Reached max_steps={args_cli.max_steps}; stopping collector smoke run.")
                record_outcome_video(
                    "timeout",
                    stage_idx,
                    count,
                    pred_text,
                    terminal_source="max_steps",
                    extra={"max_steps": int(args_cli.max_steps)},
                    metrics_override=last_pre_step_metrics,
                )
                break

            if count >= STALL_THRESHOLD:
                stalled_envs = (
                    program_executor.controller_api_wrapper.countStall(threshold=1e-4)
                    >= STALL_THRESHOLD
                )
                if stalled_envs.any():
                    stalled_ids = torch.nonzero(stalled_envs, as_tuple=False).flatten().tolist()
                    print(
                        f"Controller stalled in env(s) "
                        f"{stalled_ids}, "
                        f"resetting episode."
                    )
                    video_done = record_outcome_video(
                        "timeout",
                        stage_idx,
                        count,
                        pred_text,
                        terminal_source="controller_stall",
                        extra={"stalled_envs": [int(x) for x in stalled_ids]},
                        metrics_override=last_pre_step_metrics,
                    )
                    stage_idx = 0
                    count = 0
                    RECOMPUTE_CALL_COUNTER = 0
                    RECOMPILE_CALL_COUNTER = 0
                    if video_done:
                        break
                    reset_episode_and_sun_program_executor()
                    continue

            if (terminated | truncated).any():
                cached_success = getattr(env.unwrapped, "_cached_success", terminated)
                cached_failure = getattr(env.unwrapped, "_cached_failure", torch.zeros_like(terminated))
                if _first_bool(cached_success):
                    outcome = "true_success"
                    terminal_source = "env_cached_success"
                elif _first_bool(cached_failure):
                    outcome = "false_success"
                    terminal_source = "env_cached_failure"
                elif _first_bool(truncated):
                    outcome = "timeout"
                    terminal_source = "env_truncated"
                elif _first_bool(terminated):
                    outcome = "true_success"
                    terminal_source = "env_terminated_no_cached_flags"
                else:
                    outcome = "timeout"
                    terminal_source = "env_done_unknown"

                if outcome == "timeout":
                    print("ENV TIMED OUT")
                else:
                    print(f"ENV finished with outcome={outcome} source={terminal_source}")
                video_done = record_outcome_video(
                    outcome,
                    stage_idx,
                    count,
                    pred_text,
                    terminal_source=terminal_source,
                    extra={
                        "terminated": _first_bool(terminated),
                        "truncated": _first_bool(truncated),
                        "cached_success": _first_bool(cached_success),
                        "cached_failure": _first_bool(cached_failure),
                    },
                    metrics_override=last_pre_step_metrics,
                )
                print(f"Total success trajectory saved: {getattr(env.unwrapped, 'recorder_manager').exported_successful_episode_count}")
                stage_idx = 0
                count = 0
                RECOMPUTE_CALL_COUNTER = 0
                RECOMPILE_CALL_COUNTER = 0
                if video_done:
                    break
                # SUNProgramExecutor tracks the env correctly after an explicit reset, but
                # env.step() has already exported the finished episode. Suppress
                # recorder export for this extra reset so no zero-step shell demo
                # is written.
                reset_episode_and_sun_program_executor(suppress_recorder_export=True)
                continue

            if is_success:
                print(f"Stage {stage_idx} passed!")
                if stage_idx < len(CONSTRAINTS) - 1:
                    stage_idx += 1
                    reset_sun_program_executor(stage_idx)
                    RECOMPUTE_CALL_COUNTER = count
                    RECOMPILE_CALL_COUNTER = count + RECOMPILE_CALL_INTERVAL
                else:
                    print("All stages completed across the batch, if you are seeing this message, it means the task has false finished.")
                    video_done = record_outcome_video(
                        "false_success",
                        stage_idx,
                        count,
                        pred_text,
                        terminal_source="sun_program_executor_all_stages_without_env_done",
                    )
                    diagnostic_done = capture_false_finish_event(stage_idx, count, pred_text)
                    print(f"Total success trajectory saved: {getattr(env.unwrapped, 'recorder_manager').exported_successful_episode_count}")
                    if diagnostic_done:
                        print(f"Reached capture_false_finish_count={args_cli.capture_false_finish_count}; stopping diagnostic collector.")
                    if video_done or outcome_video_done():
                        print(f"Reached outcome video targets: {outcome_video_counts}")
                    if diagnostic_done or video_done or outcome_video_done():
                        break
                    stage_idx = 0
                    count = 0
                    RECOMPUTE_CALL_COUNTER = 0
                    RECOMPILE_CALL_COUNTER = RECOMPILE_CALL_INTERVAL
                    reset_episode_and_sun_program_executor()

    if video_writer is not None:
        video_writer.close()
        print(f"[worker {WORKER_ID}] wrote {recorded_video_frames} video frames")

    recorder = env.unwrapped.recorder_manager
    report = {
        "task": ENV_NAME, "seed": args_cli.seed, "total_steps": total_steps,
        "program_path": env_cfg.mpc_config.get("load_program"),
        "requested_demonstrations": NUM_DEMOS, "record_failures": args_cli.record_failures,
        "successful_episodes": recorder.exported_successful_episode_count if recorder else 0,
        "failed_episodes": recorder.exported_failed_episode_count if recorder else 0,
        "attempts": recorder.attempts if recorder else 0,
        "target_met": _is_stopped() if recorder else None,
    }
    (output_path / "collection_result.json").write_text(json.dumps(report, indent=2) + "\n")
    return 2 if recorder is not None and not report["target_met"] else 0

if __name__ == "__main__":
    exit_code = 1
    try:
        with ExitStack() as cleanup:
            exit_code = main(cleanup)
    except Exception:
        traceback.print_exc()
    finally:
        import omni.kit.app

        omni.kit.app.get_app().post_quit(exit_code)
        simulation_app.close(wait_for_replicator=False)
    raise SystemExit(exit_code)
