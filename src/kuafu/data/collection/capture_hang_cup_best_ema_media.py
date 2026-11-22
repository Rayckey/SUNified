#!/usr/bin/env python3
"""Capture review screenshots and MP4s for the Hang-Cup residual policy.

This script is intentionally separate from the point-cloud collection loop.  It
uses the same policy/prior/weights inputs, but records clean RGB media from
compositional camera views for human review.
"""

from __future__ import annotations

import argparse
import asyncio
import ctypes
import csv
import datetime as _datetime
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from kuafu.paths import external_artifact_root, resolve_artifact_path

from isaaclab.app import AppLauncher


DEFAULT_TASK = "FR3-FinRay-Table-Hang-Cup-fff"
DEFAULT_CHECKPOINT = (
    "Residuals/FR3-FinRay-Table-Hang-Cup-fff/2026-05-22_17-09-20/"
    "rl_runs/2026-05-22_17-13-28_Run-3/nn/"
    "last_fr3-finray-hang-cup_ep_2750_rew_15378.236.pth"
)
DEFAULT_PRIOR_CHECKPOINT = "Priors/herm/hang_cup/hang_cup_chunking_10_state_fff/best.pth"
DEFAULT_WEIGHTS_PARAMS_JSON = "Residuals/FR3-FinRay-Table-Hang-Cup-fff/2026-05-22_17-07-48/best.json"
DEFAULT_SEED = 20260516


@dataclass(frozen=True)
class CameraSpec:
    name: str
    offset_xyz: tuple[float, float, float]
    target_bias_xyz: tuple[float, float, float] = (0.0, 0.0, 0.08)
    focal_length: float = 18.0


CAMERA_BANKS: dict[str, list[CameraSpec]] = {
    # Best reviewed Hang-Cup view: a close front angle that keeps the hanger branch, handle hole, and gripper release readable.
    "best": [
        CameraSpec("front_close_hang_cup_success", (0.12, -0.78, 0.56), (0.00, 0.06, 0.05), 24.0),
    ],
    # Close front candidates for the lift, hook alignment, and open-gripper success state.
    "front_hang_cup_close": [
        CameraSpec("front_hang_cup_context", (0.10, -1.02, 0.66), (0.00, 0.10, 0.05), 18.0),
        CameraSpec("front_close_hang_cup_success", (0.12, -0.78, 0.56), (0.00, 0.06, 0.05), 24.0),
        CameraSpec("front_close_hang_cup_tight", (0.10, -0.62, 0.50), (0.00, 0.05, 0.045), 30.0),
    ],
    # Side obliques make the handle-through-branch relationship easier to verify.
    "side_hang_cup": [
        CameraSpec("right_side_hang_cup", (0.76, -0.22, 0.56), (-0.03, 0.02, 0.05), 22.0),
        CameraSpec("left_side_hang_cup", (-0.62, -0.24, 0.56), (0.03, 0.02, 0.05), 22.0),
        CameraSpec("over_robot_hang_cup", (0.08, 0.86, 0.82), (0.00, -0.04, 0.04), 18.0),
    ],
    # First pass task-centric views.
    "sweep_a": [
        CameraSpec("hero_front_right_hang_cup", (0.94, -1.00, 0.74), (0.00, 0.03, 0.05), 16.0),
        CameraSpec("front_low_hang_cup", (0.22, -0.84, 0.44), (0.00, 0.07, 0.05), 22.0),
        CameraSpec("over_robot_shoulder_hang_cup", (0.06, 0.96, 0.86), (0.00, -0.05, 0.04), 18.0),
    ],
    # Alternate candidates if the wrist hides the mug handle or hanger branch.
    "sweep_b": [
        CameraSpec("right_side_low_hang_cup", (0.82, -0.18, 0.46), (-0.03, 0.00, 0.05), 22.0),
        CameraSpec("left_front_oblique_hang_cup", (-0.44, -0.72, 0.58), (0.03, 0.06, 0.05), 22.0),
        CameraSpec("top_oblique_hang_cup", (0.58, -0.60, 1.02), (0.00, 0.02, 0.03), 18.0),
    ],
    # Slightly wider/tighter versions for final comparison.
    "hero": [
        CameraSpec("hero_front_right_tight_hang_cup", (0.78, -0.78, 0.60), (0.00, 0.05, 0.05), 22.0),
        CameraSpec("hero_front_right_wide_hang_cup", (0.96, -1.02, 0.76), (0.00, 0.03, 0.05), 16.0),
        CameraSpec("over_robot_shoulder_hang_cup", (0.06, 0.96, 0.86), (0.00, -0.05, 0.04), 18.0),
    ],
}


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Record deterministic Hang-Cup policy media from candidate camera views."
    )
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--prior_checkpoint", default=DEFAULT_PRIOR_CHECKPOINT)
    parser.add_argument(
        "--prior_mode",
        choices=["full_actor", "residual_rl"],
        default="residual_rl",
        help="Policy playback mode. Hang-Cup best EMA uses residual_rl.",
    )
    parser.add_argument("--weights_params_json", default=DEFAULT_WEIGHTS_PARAMS_JSON)
    parser.add_argument("--agent", default="rl_games_cfg_entry_point")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--camera_bank", choices=sorted(CAMERA_BANKS), default="best")
    parser.add_argument(
        "--capture_backend",
        choices=["viewport", "sensor"],
        default="viewport",
        help="Use the active viewport for review media, or the env camera sensors for debugging.",
    )
    parser.add_argument("--viewport_settle_frames", type=int, default=2)
    parser.add_argument(
        "--debug_frames",
        action="store_true",
        help="Show frame-transformer debug markers for fingertip and link frames.",
    )
    parser.add_argument("--max_steps", type=int, default=1100)
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=1,
        help="Run up to this many deterministic episodes, stopping early on true success.",
    )
    parser.add_argument("--video_every", type=int, default=1, help="Record every Nth policy step.")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--warmup_frames", type=int, default=8)
    parser.add_argument(
        "--paper_background",
        choices=["none", "custom_usd"],
        default="none",
        help="Visual-only USD background for paper screenshots/media.",
    )
    parser.add_argument(
        "--paper_background_usd",
        default="",
        help="Local USD/USDA file to load when --paper_background custom_usd is selected.",
    )
    parser.add_argument(
        "--paper_background_pos",
        default="0.0,0.9,0.0",
        help="XYZ offset relative to the recorded env origin.",
    )
    parser.add_argument("--paper_background_rot", default="0.0,0.0,180.0")
    parser.add_argument("--paper_background_scale", default="1.0,1.0,1.0")
    parser.add_argument(
        "--screenshot_steps",
        default="final",
        help="Comma-separated policy steps to save as PNG stills, or 'final' to capture only after done.",
    )
    parser.add_argument(
        "--output_dir",
        default=str(external_artifact_root() / "media" / "hang_cup_best_ema"),
        help="Directory where a timestamped media run will be written.",
    )
    parser.add_argument("--no_video", action="store_true")
    parser.add_argument("--no_screenshots", action="store_true")
    parser.add_argument("--stochastic_policy", action="store_true")
    parser.add_argument(
        "--probe_success_only",
        action="store_true",
        help="Run a vectorized no-media success probe. Allows --num_envs > 1 and stops after the configured horizon.",
    )
    parser.add_argument(
        "--probe_record_env_id",
        type=int,
        default=-1,
        help="In probe mode, record viewport media focused on this env id while still running all envs.",
    )
    parser.add_argument(
        "--probe_stop_on_success",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="In probe mode, stop as soon as any env records true success.",
    )
    parser.add_argument(
        "--probe_print_every",
        type=int,
        default=10,
        help="In probe mode, print whenever this many additional episodes complete.",
    )
    parser.add_argument(
        "--probe_env_spacing",
        type=float,
        default=0.0,
        help="Optional env spacing override for probe/record runs, useful for isolating the recorded env visually.",
    )
    AppLauncher.add_app_launcher_args(parser)
    args, hydra_args = parser.parse_known_args()
    if args.num_envs != 1 and not args.probe_success_only:
        raise ValueError("This media script expects --num_envs 1 unless --probe_success_only is enabled.")
    if args.max_steps <= 0:
        raise ValueError("--max_steps must be positive.")
    if args.max_episodes <= 0:
        raise ValueError("--max_episodes must be positive.")
    if args.video_every <= 0:
        raise ValueError("--video_every must be positive.")
    if args.probe_record_env_id >= 0 and not args.probe_success_only:
        raise ValueError("--probe_record_env_id requires --probe_success_only.")
    if args.fps <= 0:
        raise ValueError("--fps must be positive.")
    if args.paper_background == "custom_usd" and not args.paper_background_usd:
        raise ValueError("--paper_background_usd is required when --paper_background custom_usd.")
    args.enable_cameras = (not bool(args.probe_success_only)) or args.probe_record_env_id >= 0
    if args.probe_success_only and args.probe_record_env_id < 0:
        args.no_video = True
        args.no_screenshots = True
    if args.capture_backend == "viewport" and (not args.probe_success_only or args.probe_record_env_id >= 0):
        # AppLauncher disables the active viewport in headless mode unless it sees a video request.
        args.video = True
    _validate_hang_cup_pairing(args)
    return args, hydra_args


def _validate_hang_cup_pairing(args: argparse.Namespace) -> None:
    task_match = re.search(r"FR3-FinRay-Table-Hang-Cup-([A-Za-z0-9]+)", str(args.task))
    if task_match is None:
        return
    task_suffix = task_match.group(1)

    checkpoint_match = re.search(
        r"FR3-FinRay-Table-Hang-Cup-([A-Za-z0-9]+)",
        str(args.checkpoint),
    )
    if checkpoint_match is not None and checkpoint_match.group(1) != task_suffix:
        raise ValueError(
            "Hang-Cup checkpoint/task mismatch: "
            f"task suffix {task_suffix} but checkpoint suffix {checkpoint_match.group(1)}."
        )

    prior_match = re.search(r"hang_cup_chunking_[0-9]+_state_([A-Za-z0-9]+)", str(args.prior_checkpoint))
    if prior_match is not None and prior_match.group(1) != task_suffix:
        raise ValueError(
            "Hang-Cup prior/task mismatch: "
            f"task suffix {task_suffix} but prior state suffix {prior_match.group(1)}."
        )


args_cli, hydra_args = parse_args()
for _artifact_attr, _artifact_label in (
    ("checkpoint", "Policy checkpoint"),
    ("prior_checkpoint", "BC prior checkpoint"),
    ("weights_params_json", "Residual parameter JSON"),
    ("output_dir", "Media output directory"),
    ("probe_save_path", "Probe trajectory output"),
    ("replay_actions_file", "Replay trajectory input"),
):
    _artifact_value = getattr(args_cli, _artifact_attr, "")
    if _artifact_value:
        setattr(
            args_cli,
            _artifact_attr,
            str(resolve_artifact_path(_artifact_value, _artifact_label)),
        )
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import UsdGeom  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
import kuafu.isaac.learned  # noqa: F401,E402
import kuafu.isaac.learned.panda  # noqa: F401,E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent  # noqa: E402
from isaaclab.sensors import TiledCameraCfg  # noqa: E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab.utils.math import create_rotation_matrix_from_view, quat_from_matrix  # noqa: E402
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402
from isaacsim.core.utils.stage import get_current_stage  # noqa: E402
try:
    from omni.kit.viewport.utility import capture_viewport_to_buffer, get_active_viewport  # noqa: E402
except ModuleNotFoundError:  # No-camera probe mode uses the lighter non-rendering headless kit.
    capture_viewport_to_buffer = None
    get_active_viewport = None
from rl_games.common import env_configurations, vecenv  # noqa: E402
from rl_games.common.player import BasePlayer  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402

from paper_background_utils import add_configured_paper_background  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[4]
from kuafu.rl.isaac.update_utilities import update_all_parameters, update_weights  # noqa: E402


def _set_env_obs_mode(env_cfg, obs_mode: str) -> str:
    set_obs_mode = getattr(env_cfg, "set_obs_mode", None)
    if callable(set_obs_mode):
        set_obs_mode(obs_mode)
    else:
        setattr(env_cfg, "obs_mode", obs_mode)
        obs_dims = getattr(env_cfg, "obs_dims", None)
        if isinstance(obs_dims, dict) and obs_mode in obs_dims and hasattr(env_cfg, "observation_space"):
            setattr(env_cfg, "observation_space", int(obs_dims[obs_mode]))
    return obs_mode


def _apply_checkpoint_obs_mode(env_cfg, checkpoint_path: str) -> str | None:
    env_yaml_path = Path(checkpoint_path).resolve().parent.parent / "params" / "env.yaml"
    try:
        text = env_yaml_path.read_text()
    except OSError:
        return None
    match = re.search(r"(?m)^obs_mode:\s*([A-Za-z0-9_]+)\s*$", text)
    if match is None:
        return None
    return _set_env_obs_mode(env_cfg, match.group(1))


def _load_all_stages_weights_params(path: str):
    resolved_path = retrieve_file_path(path)
    with open(resolved_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    stages_data = data.get("stages") if isinstance(data.get("stages"), dict) else data
    if not isinstance(stages_data, dict):
        raise ValueError(f"Expected a stages dict in {resolved_path}")

    all_stages_weights = {}
    all_stages_parameters = {}
    for raw_stage_id, stage_data in stages_data.items():
        if not isinstance(stage_data, dict):
            continue
        try:
            stage_id = int(raw_stage_id)
        except (TypeError, ValueError):
            continue
        weights = stage_data.get("weights") or {}
        params = stage_data.get("parameters") or {}
        if isinstance(weights, dict) and weights:
            all_stages_weights[stage_id] = {int(k): float(v) for k, v in dict(weights).items()}
            all_stages_parameters[stage_id] = {int(k): dict(v) for k, v in dict(params).items()}
    if not all_stages_weights:
        raise ValueError(f"No valid stage weights found in {resolved_path}.")
    return all_stages_weights, all_stages_parameters


def _resolve_prior_mode(agent_cfg: dict, explicit_mode: str) -> str:
    mode = str(explicit_mode or "full_actor").strip() or "full_actor"
    residual_cfg = agent_cfg.get("params", {}).get("config", {}).get("residual_rl", {})
    if mode == "full_actor" and isinstance(residual_cfg, dict) and residual_cfg.get("enabled", False):
        return "residual_rl"
    return mode


class FfmpegRgbWriter:
    """Stream RGB numpy frames to H.264 MP4."""

    def __init__(self, path: Path, fps: int, width: int, height: int):
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg not found on PATH; MP4 recording requires ffmpeg.")
        path.parent.mkdir(parents=True, exist_ok=True)
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
                "18",
                "-preset",
                "veryfast",
                "-movflags",
                "+faststart",
                str(path),
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


class ViewportCapturer:
    """Synchronous helper around Kit viewport buffer captures."""

    def __init__(self, width: int, height: int):
        self.width = int(width)
        self.height = int(height)
        self.viewport = get_active_viewport()
        if not self.viewport:
            raise RuntimeError("No active viewport is available. Try running without --headless or keep capture_backend=sensor.")
        self.viewport.resolution = (self.width, self.height)

    @staticmethod
    def _run(coro):
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        if loop.is_running():
            raise RuntimeError("Viewport capture cannot wait inside an already-running asyncio loop.")
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
            raise RuntimeError("Viewport buffer capture did not return a frame within 5 seconds.")
        return captured["frame"]


def _set_viewport_camera(raw_env, row: dict[str, object]) -> None:
    eye = (float(row["eye_x"]), float(row["eye_y"]), float(row["eye_z"]))
    target = (float(row["target_x"]), float(row["target_y"]), float(row["target_z"]))
    raw_env.sim.set_camera_view(eye=eye, target=target, camera_prim_path="/OmniverseKit_Persp")
    cam_prim = get_current_stage().GetPrimAtPath("/OmniverseKit_Persp")
    if cam_prim.IsValid():
        UsdGeom.Camera(cam_prim).GetFocalLengthAttr().Set(float(row["focal_length"]))


def _viewport_frame(raw_env, row: dict[str, object], capturer: ViewportCapturer) -> np.ndarray:
    _set_viewport_camera(raw_env, row)
    for _ in range(max(0, int(args_cli.viewport_settle_frames))):
        raw_env.sim.render()
        simulation_app.update()
    return capturer.capture_rgb()


def _parse_screenshot_steps(raw: str, max_steps: int) -> set[int]:
    raw = str(raw or "").strip()
    if raw.lower() in {"", "final", "none"}:
        return set()
    steps = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        step = int(part)
        if step < 0:
            continue
        steps.add(min(step, max_steps - 1))
    return steps


def _camera_names() -> list[str]:
    return [f"table_cam_{idx}" for idx in range(3)]


def _override_camera_cfg(env_cfg, specs: list[CameraSpec]) -> None:
    for idx, spec in enumerate(specs):
        cam_cfg = getattr(env_cfg, f"table_cam_{idx}")
        cam_cfg.width = int(args_cli.width)
        cam_cfg.height = int(args_cli.height)
        cam_cfg.data_types = ["rgb"]
        cam_cfg.update_period = 0.0
        cam_cfg.spawn = sim_utils.PinholeCameraCfg(
            focal_length=float(spec.focal_length),
            clipping_range=(0.01, 5.0),
            focus_distance=1.0,
            f_stop=0.0,
        )
        cam_cfg.offset = TiledCameraCfg.OffsetCfg(
            pos=spec.offset_xyz,
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="world",
        )


FRAME_DEBUG_CFG_ATTRS = (
    "ee_frame",
    "l4_frame",
    "mug_handle_hole_frame",
    "mug_center_frame",
    "mug_rim_frame_0",
    "mug_rim_frame_1",
    "cup_hanger_branch_1_tip_frame",
)
FRAME_DEBUG_SENSOR_NAMES = (
    "panda_fingertip",
    "panda_link4",
    "mug_handle_hole_frame",
    "mug_center_frame",
    "mug_rim_frame_0",
    "mug_rim_frame_1",
    "cup_hanger_branch_1_tip_frame",
)

def _set_frame_debug_cfg(env_cfg, enabled: bool) -> None:
    for attr_name in FRAME_DEBUG_CFG_ATTRS:
        cfg = getattr(env_cfg, attr_name, None)
        if cfg is not None and hasattr(cfg, "debug_vis"):
            cfg.debug_vis = bool(enabled)


def _set_runtime_frame_debug_vis(raw_env, enabled: bool) -> None:
    for sensor_name in FRAME_DEBUG_SENSOR_NAMES:
        sensor = raw_env.scene.sensors.get(sensor_name)
        if sensor is not None and hasattr(sensor, "set_debug_vis"):
            sensor.set_debug_vis(bool(enabled))


def _get_frame_pos(raw_env, sensor_name: str, env_index: int = 0) -> torch.Tensor | None:
    sensor = raw_env.scene.sensors.get(sensor_name)
    if sensor is None:
        sensor = getattr(raw_env, sensor_name, None)
    if sensor is None:
        return None
    try:
        return sensor.data.target_pos_w[int(env_index), 0]
    except Exception:
        return None


def _get_scene_focus(raw_env, specs: list[CameraSpec], env_index: int = 0) -> torch.Tensor:
    device = raw_env.device
    task_points: list[torch.Tensor] = []

    # Bias the look-at point toward the destination branch while keeping the mug handle in frame.
    for sensor_name in (
        "cup_hanger_branch_1_tip_frame",
        "cup_hanger_branch_1_tip_frame",
        "mug_handle_hole_frame",
        "mug_center_frame",
    ):
        pos = _get_frame_pos(raw_env, sensor_name, env_index)
        if pos is not None:
            task_points.append(pos)

    if not task_points:
        try:
            hanger = raw_env.scene.rigid_objects.get("cup_hanger")
        except Exception:
            hanger = getattr(raw_env, "cup_hanger", None)
        if hanger is not None:
            try:
                task_points.append(hanger.data.root_pos_w[int(env_index)])
            except Exception:
                pass

    if not task_points:
        origins = getattr(raw_env.scene, "env_origins", None)
        if origins is not None and int(env_index) < int(origins.shape[0]):
            return origins[int(env_index)] + torch.tensor([0.42, -0.35, 0.28], device=device)
        return torch.tensor([0.42, -0.35, 0.28], device=device)

    focus = torch.stack(task_points, dim=0).mean(dim=0)
    # The success evidence lives around the elevated branch and handle hole, not the tabletop.
    focus[2] = torch.clamp(focus[2], min=0.16, max=0.42)
    return focus


def _set_camera_views(
    raw_env,
    specs: list[CameraSpec],
    env_id: int = 0,
    set_sensor_poses: bool = True,
) -> list[dict[str, object]]:
    stage = get_current_stage()
    up_axis = UsdGeom.GetStageUpAxis(stage)
    env_index = int(env_id)
    focus = _get_scene_focus(raw_env, specs, env_index)
    rows: list[dict[str, object]] = []

    for idx, spec in enumerate(specs):
        offset = torch.tensor(spec.offset_xyz, device=raw_env.device, dtype=torch.float32)
        bias = torch.tensor(spec.target_bias_xyz, device=raw_env.device, dtype=torch.float32)
        eye = focus + offset
        target = focus + bias
        eyes = eye.view(1, 3)
        targets = target.view(1, 3)
        view_rot = create_rotation_matrix_from_view(eyes, targets, up_axis, device=str(raw_env.device))
        quat = quat_from_matrix(view_rot.transpose(1, 2))
        if set_sensor_poses:
            cam = raw_env.scene.sensors[f"table_cam_{idx}"]
            env_ids = torch.tensor([env_index], device=raw_env.device, dtype=torch.long)
            cam._view.set_world_poses(eyes, quat, env_ids)  # TiledCamera has no public look-at helper.
            if getattr(cam.cfg, "update_latest_camera_pose", False):
                cam._update_poses(env_ids)
        rows.append(
            {
                "sensor": f"table_cam_{idx}" if set_sensor_poses else f"viewport_{idx}",
                "view": spec.name,
                "env_id": env_index,
                "eye_x": float(eye[0].detach().cpu()),
                "eye_y": float(eye[1].detach().cpu()),
                "eye_z": float(eye[2].detach().cpu()),
                "target_x": float(target[0].detach().cpu()),
                "target_y": float(target[1].detach().cpu()),
                "target_z": float(target[2].detach().cpu()),
                "focal_length": spec.focal_length,
            }
        )
    return rows

def _refresh_cameras(raw_env, frames: int) -> None:
    for _ in range(max(0, int(frames))):
        raw_env.sim.render()
        raw_env.scene.update(raw_env.physics_dt)


def _rgb_frame(raw_env, camera_name: str) -> np.ndarray:
    cam = raw_env.scene.sensors[camera_name]
    frame = cam.data.output["rgb"][0].detach().cpu().numpy()
    if frame.dtype != np.uint8:
        frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
    return frame[:, :, :3]


def _write_png(path: Path, frame_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame_rgb).save(path)


def _read_success_flag(raw_env) -> bool:
    cached_success = getattr(raw_env, "_cached_success", None)
    if cached_success is None:
        return False
    try:
        return bool(cached_success[0].item())
    except Exception:
        return False


def _read_outcome_count(raw_env, method_name: str) -> int:
    method = getattr(raw_env, method_name, None)
    if not callable(method):
        return 0
    try:
        return int(method())
    except Exception:
        return 0


def _print_outcome_summary(raw_env) -> None:
    true_success = _read_outcome_count(raw_env, "get_true_success_count")
    false_complete = _read_outcome_count(raw_env, "get_false_complete_count")
    timeout = _read_outcome_count(raw_env, "get_timeout_count")
    print(
        "[INFO] Outcome counts: "
        f"true_success={true_success} false_complete={false_complete} timeout={timeout}",
        flush=True,
    )


def _probe_success_env_ids(raw_env) -> list[int]:
    cached_success = getattr(raw_env, "_cached_success", None)
    if cached_success is None:
        return []
    try:
        ids = cached_success.nonzero(as_tuple=False).flatten().detach().cpu().tolist()
    except Exception:
        return []
    return [int(idx) for idx in ids]


def _run_success_probe(
    env,
    raw_env,
    agent: BasePlayer,
    obs,
    agent_is_rnn: bool,
    run_dir: Path | None = None,
    specs: list[CameraSpec] | None = None,
) -> None:
    deterministic_actions = not args_cli.stochastic_policy
    if hasattr(raw_env, "clear_end_to_end_rate_stats"):
        raw_env.clear_end_to_end_rate_stats()

    record_env_id = int(getattr(args_cli, "probe_record_env_id", -1))
    record_media = record_env_id >= 0
    rows: list[dict[str, object]] = []
    writers: dict[str, FfmpegRgbWriter] = {}
    viewport_capturer = None
    last_reported_total = 0
    last_reported_success = 0
    last_step = 0
    final_recorded = False

    if record_media:
        if run_dir is None:
            raise RuntimeError("Probe media recording requires a run directory.")
        if args_cli.capture_backend != "viewport":
            raise RuntimeError("Probe media recording currently requires --capture_backend viewport.")
        if capture_viewport_to_buffer is None or get_active_viewport is None:
            raise RuntimeError("Viewport capture is unavailable in this Isaac app.")
        if record_env_id >= int(raw_env.num_envs):
            raise ValueError(f"--probe_record_env_id {record_env_id} is outside num_envs={raw_env.num_envs}.")
        specs = list(specs or [])
        if not specs:
            raise RuntimeError("Probe media recording requires at least one camera spec.")
        rows = _set_camera_views(raw_env, specs, env_id=record_env_id, set_sensor_poses=False)
        _write_manifest(run_dir, rows, args_cli)
        viewport_capturer = ViewportCapturer(args_cli.width, args_cli.height)
        if not args_cli.no_video:
            videos_dir = run_dir / "videos"
            videos_dir.mkdir(parents=True, exist_ok=True)
            for idx, spec in enumerate(specs):
                writers[spec.name] = FfmpegRgbWriter(
                    videos_dir / f"{idx:02d}_{spec.name}.mp4",
                    args_cli.fps,
                    args_cli.width,
                    args_cli.height,
                )

    screenshot_steps = _parse_screenshot_steps(args_cli.screenshot_steps, args_cli.max_steps)

    def _record_probe_frames(step: int, final: bool = False) -> None:
        if not record_media or viewport_capturer is None:
            return
        write_screenshot = (not args_cli.no_screenshots) and (final or step in screenshot_steps)
        write_video = bool(writers) and (final or step % args_cli.video_every == 0)
        if not write_screenshot and not write_video:
            return
        for idx, spec in enumerate(specs or []):
            frame = _viewport_frame(raw_env, rows[idx], viewport_capturer)
            if write_video:
                writers[spec.name].write(frame)
            if write_screenshot:
                name = "final.png" if final else f"step_{step:04d}.png"
                _write_png(run_dir / "screenshots" / spec.name / name, frame)

    print("=== Hang-Cup Success Probe ===", flush=True)
    print(f"  seed          : {args_cli.seed}", flush=True)
    print(f"  num envs      : {args_cli.num_envs}", flush=True)
    print(f"  max steps     : {args_cli.max_steps}", flush=True)
    print(f"  action mode   : {'stochastic' if args_cli.stochastic_policy else 'deterministic'}", flush=True)
    if args_cli.paper_background != "none":
        print(f"  paper bg      : {args_cli.paper_background}", flush=True)
        print(f"  bg usd        : {args_cli.paper_background_usd}", flush=True)
        print(f"  bg pos offset : {args_cli.paper_background_pos}", flush=True)
    if record_media:
        print(f"  recording env : {record_env_id}", flush=True)
        print(f"  run dir       : {run_dir}", flush=True)
        for row in rows:
            print(
                "  view          : "
                f"{row['sensor']} -> {row['view']} env={row['env_id']} "
                f"eye=({row['eye_x']:.3f}, {row['eye_y']:.3f}, {row['eye_z']:.3f}) "
                f"target=({row['target_x']:.3f}, {row['target_y']:.3f}, {row['target_z']:.3f})",
                flush=True,
            )

    try:
        for step in range(args_cli.max_steps):
            last_step = step
            _record_probe_frames(step)
            with torch.inference_mode():
                obs_t = agent.obs_to_torch(obs)
                actions = agent.get_action(obs_t, is_deterministic=deterministic_actions)
                obs, _, dones, _ = env.step(actions)
                if isinstance(obs, dict):
                    obs = obs["obs"]
                if isinstance(dones, torch.Tensor):
                    done_count = int(dones.sum().item())
                else:
                    done_count = int(sum(bool(d) for d in dones)) if len(dones) > 0 else 0
                if done_count > 0 and agent_is_rnn and agent.states is not None:
                    for state in agent.states:
                        state[:, dones, :] = 0.0

            true_success = _read_outcome_count(raw_env, "get_true_success_count")
            false_complete = _read_outcome_count(raw_env, "get_false_complete_count")
            timeout = _read_outcome_count(raw_env, "get_timeout_count")
            total = true_success + false_complete + timeout
            success_ids = _probe_success_env_ids(raw_env)
            target_success = record_env_id >= 0 and record_env_id in success_ids
            should_print = (
                total >= last_reported_total + max(1, int(args_cli.probe_print_every))
                or true_success > last_reported_success
                or target_success
                or step == args_cli.max_steps - 1
            )
            if should_print:
                print(
                    "[PROBE] "
                    f"step={step} completed={total} true_success={true_success} "
                    f"false_complete={false_complete} timeout={timeout} "
                    f"current_success_envs={success_ids}",
                    flush=True,
                )
                last_reported_total = total
                last_reported_success = true_success
            if target_success:
                _record_probe_frames(step, final=True)
                final_recorded = True
                print(f"[INFO] Recorded target env {record_env_id} through success at policy step {step}.", flush=True)
                break
            if true_success > 0 and args_cli.probe_stop_on_success and not record_media:
                break
    finally:
        if record_media and not final_recorded:
            _record_probe_frames(last_step, final=True)
        for writer in writers.values():
            writer.close()

    _print_outcome_summary(raw_env)


def _write_manifest(run_dir: Path, rows: list[dict[str, object]], args: argparse.Namespace) -> None:
    with open(run_dir / "camera_manifest.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sensor",
                "view",
                "env_id",
                "eye_x",
                "eye_y",
                "eye_z",
                "target_x",
                "target_y",
                "target_z",
                "focal_length",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    with open(run_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "task": args.task,
                "checkpoint": args.checkpoint,
                "prior_checkpoint": args.prior_checkpoint,
                "prior_mode": args.prior_mode,
                "weights_params_json": args.weights_params_json,
                "seed": args.seed,
                "camera_bank": args.camera_bank,
                "capture_backend": args.capture_backend,
                "viewport_settle_frames": args.viewport_settle_frames,
                "debug_frames": bool(args.debug_frames),
                "max_steps": args.max_steps,
                "max_episodes": args.max_episodes,
                "fps": args.fps,
                "width": args.width,
                "height": args.height,
                "video_every": args.video_every,
                "stochastic_policy": bool(args.stochastic_policy),
                "screenshot_steps": args.screenshot_steps,
                "probe_success_only": bool(getattr(args, "probe_success_only", False)),
                "probe_record_env_id": int(getattr(args, "probe_record_env_id", -1)),
                "probe_env_spacing": float(getattr(args, "probe_env_spacing", 0.0)),
                "paper_background": getattr(args, "paper_background", "none"),
                "paper_background_usd": getattr(args, "paper_background_usd", ""),
                "paper_background_pos": getattr(args, "paper_background_pos", ""),
                "paper_background_rot": getattr(args, "paper_background_rot", ""),
                "paper_background_scale": getattr(args, "paper_background_scale", ""),
            },
            f,
            indent=2,
        )


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    probe_record_media = bool(args_cli.probe_success_only and args_cli.probe_record_env_id >= 0)
    specs = CAMERA_BANKS[args_cli.camera_bank] if (not args_cli.probe_success_only or probe_record_media) else []
    task_name = args_cli.task.split(":")[-1]
    timestamp = _datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(args_cli.output_dir).expanduser()
    if not run_dir.is_absolute():
        run_dir = REPO_ROOT / run_dir
    frame_variant = f"probe_env{args_cli.probe_record_env_id}" if probe_record_media else ("debugframes" if args_cli.debug_frames else "clean")
    run_dir = run_dir / f"{timestamp}_{args_cli.camera_bank}_{frame_variant}_seed{args_cli.seed}"
    screenshots_dir = run_dir / "screenshots"
    videos_dir = run_dir / "videos"
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args_cli.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args_cli.seed)

    env_cfg.scene.num_envs = int(args_cli.num_envs) if args_cli.probe_success_only else 1
    if args_cli.probe_success_only and float(args_cli.probe_env_spacing) > 0.0:
        env_cfg.scene.env_spacing = float(args_cli.probe_env_spacing)
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.mpc_config["num_envs"] = env_cfg.scene.num_envs
    env_cfg.mpc_config["use_per_env_weight"] = False
    env_cfg.enable_cameras = not bool(args_cli.probe_success_only)
    env_cfg.seed = args_cli.seed
    # FrameTransformer debug markers are enabled after reset, once sensor data exists.
    _set_frame_debug_cfg(env_cfg, False)
    if not args_cli.probe_success_only:
        _override_camera_cfg(env_cfg, specs)

    if "mpc" in agent_cfg:
        env_cfg.mpc_config["use_per_env_weight"] = agent_cfg["mpc"]["use_per_env_weight"]
        env_cfg.mpc_config["use_absolute_progress_rewards"] = agent_cfg["mpc"]["use_absolute_progress_rewards"]
        env_cfg.mpc_config["use_relative_progress_rewards"] = agent_cfg["mpc"]["use_relative_progress_rewards"]
        env_cfg.mpc_config["use_velocity_penalty"] = agent_cfg["mpc"]["use_velocity_penalty"]
        env_cfg.mpc_config["regularize_velocity_scale"] = agent_cfg["mpc"]["regularize_velocity_scale"]
        env_cfg.mpc_config["regularize_time_scale"] = agent_cfg["mpc"]["regularize_time_scale"]
        env_cfg.mpc_config["progress_rewards_scale"] = agent_cfg["mpc"]["progress_rewards_scale"]

    if args_cli.device is not None:
        agent_cfg["params"]["config"]["device"] = args_cli.device
        agent_cfg["params"]["config"]["device_name"] = args_cli.device
    agent_cfg["params"]["seed"] = args_cli.seed

    resume_path = retrieve_file_path(args_cli.checkpoint)
    prior_path = retrieve_file_path(args_cli.prior_checkpoint) if args_cli.prior_checkpoint else None
    env_cfg.log_dir = os.path.dirname(os.path.dirname(resume_path))

    checkpoint_obs_mode = _apply_checkpoint_obs_mode(env_cfg, resume_path)
    if checkpoint_obs_mode is not None:
        print(f"[INFO] Set env obs_mode from checkpoint metadata: {checkpoint_obs_mode}")

    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    raw_env = env.unwrapped

    if args_cli.weights_params_json:
        all_stages_weights, all_stages_parameters = _load_all_stages_weights_params(args_cli.weights_params_json)
        update_all_parameters(raw_env, all_stages_parameters=all_stages_parameters)
        update_weights(raw_env, stage_id=0, weights=None, all_stages_weights=all_stages_weights)
        print(f"[INFO] Applied weights/parameters from: {args_cli.weights_params_json}")

    prior_mode = _resolve_prior_mode(agent_cfg, args_cli.prior_mode)
    residual_controller = None
    if prior_mode == "residual_rl":
        if prior_path is None:
            raise ValueError("Residual RL playback requires --prior_checkpoint.")
        from kuafu.rl.isaac.train_rl_games import (  # noqa: WPS433
            _configure_residual_rl_agent_cfg,
            _make_kuafu_residual_rl_games_wrapper,
            _make_residual_action_controller,
            _resolve_env_num_stages,
        )
        from kuafu.bc.training.residual_rl import ResidualRlConfig  # noqa: WPS433

        env_num_stages = _resolve_env_num_stages(env)
        residual_mapping = dict(agent_cfg.get("params", {}).get("config", {}).get("residual_rl", {}) or {})
        residual_mapping["enabled"] = True
        action_dim = int(getattr(env.unwrapped.single_action_space, "shape")[0])
        residual_cfg = ResidualRlConfig.from_mapping(
            residual_mapping,
            action_dim=action_dim,
            env_num_stages=env_num_stages,
        )
        _configure_residual_rl_agent_cfg(agent_cfg, residual_cfg)
        env = _make_kuafu_residual_rl_games_wrapper(
            env,
            rl_device,
            clip_obs,
            clip_actions,
            residual_cfg,
            obs_groups,
            concate_obs_groups,
        )
        residual_controller = _make_residual_action_controller(
            prior_checkpoint=prior_path,
            residual_cfg=residual_cfg,
            rl_env=env,
        )
        playback_epoch = int(residual_cfg.critic_warmup_epochs) + int(residual_cfg.alpha_warmup_epochs)
        residual_controller.set_epoch(playback_epoch)
        print(f"[INFO] Residual RL playback armed with prior: {prior_path}")
        print(f"[INFO] Residual alpha: {residual_controller.alpha:.3f}")
    else:
        env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups)

    vecenv.register(
        "IsaacRlgWrapper",
        lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs

    for _ in range(5):
        simulation_app.update()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    runner = Runner()
    runner.load(agent_cfg)
    agent: BasePlayer = runner.create_player()
    agent.restore(resume_path)
    agent.reset()

    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    _ = agent.get_batch_size(obs, 1)
    agent_is_rnn = bool(getattr(agent, "is_rnn", False))
    if agent_is_rnn:
        agent.init_rnn()

    should_add_background = args_cli.paper_background != "none" and (
        not args_cli.probe_success_only or args_cli.probe_record_env_id >= 0
    )
    if should_add_background:
        add_configured_paper_background(args_cli, raw_env)
        raw_env.sim.render()
        simulation_app.update()

    if args_cli.probe_success_only:
        _run_success_probe(env, raw_env, agent, obs, agent_is_rnn, run_dir if probe_record_media else None, specs)
        env.close()
        return

    manifest_rows = _set_camera_views(raw_env, specs)
    _refresh_cameras(raw_env, args_cli.warmup_frames)
    if args_cli.debug_frames:
        _set_runtime_frame_debug_vis(raw_env, True)
        _refresh_cameras(raw_env, 2)
    viewport_capturer = None
    if args_cli.capture_backend == "viewport":
        viewport_capturer = ViewportCapturer(args_cli.width, args_cli.height)
    _write_manifest(run_dir, manifest_rows, args_cli)

    print("=== Hang-Cup Media Capture ===")
    print(f"  task          : {task_name}")
    print(f"  seed          : {args_cli.seed}")
    print(f"  camera bank   : {args_cli.camera_bank}")
    print(f"  backend       : {args_cli.capture_backend}")
    print(f"  debug frames  : {args_cli.debug_frames}")
    print(f"  max episodes  : {args_cli.max_episodes}")
    print(f"  action mode   : {'stochastic' if args_cli.stochastic_policy else 'deterministic'}")
    if args_cli.paper_background != "none":
        print(f"  paper bg      : {args_cli.paper_background}")
        print(f"  bg usd        : {args_cli.paper_background_usd}")
        print(f"  bg pos offset : {args_cli.paper_background_pos}")
    print(f"  run dir       : {run_dir}")
    for row in manifest_rows:
        print(
            "  view          : "
            f"{row['sensor']} -> {row['view']} "
            f"eye=({row['eye_x']:.3f}, {row['eye_y']:.3f}, {row['eye_z']:.3f}) "
            f"target=({row['target_x']:.3f}, {row['target_y']:.3f}, {row['target_z']:.3f})"
        )

    screenshot_steps = _parse_screenshot_steps(args_cli.screenshot_steps, args_cli.max_steps)
    writers: dict[str, FfmpegRgbWriter] = {}
    last_frames: dict[str, np.ndarray] = {}
    if not args_cli.no_video:
        videos_dir.mkdir(parents=True, exist_ok=True)
        for idx, spec in enumerate(specs):
            writers[spec.name] = FfmpegRgbWriter(
                videos_dir / f"{idx:02d}_{spec.name}.mp4",
                args_cli.fps,
                args_cli.width,
                args_cli.height,
            )

    deterministic_actions = not args_cli.stochastic_policy
    final_done = False
    final_success = False
    episode_idx = 1
    step = 0
    try:
        while simulation_app.is_running() and episode_idx <= args_cli.max_episodes and step < args_cli.max_steps:
            if step in screenshot_steps and not args_cli.no_screenshots:
                if args_cli.capture_backend == "sensor":
                    _refresh_cameras(raw_env, 1)
                for idx, spec in enumerate(specs):
                    if args_cli.capture_backend == "viewport":
                        frame = _viewport_frame(raw_env, manifest_rows[idx], viewport_capturer)
                    else:
                        frame = _rgb_frame(raw_env, f"table_cam_{idx}")
                    last_frames[spec.name] = frame.copy()
                    _write_png(screenshots_dir / spec.name / f"step_{step:04d}.png", frame)

            if writers and step % args_cli.video_every == 0:
                if args_cli.capture_backend == "sensor":
                    _refresh_cameras(raw_env, 1)
                for idx, spec in enumerate(specs):
                    if args_cli.capture_backend == "viewport":
                        frame = _viewport_frame(raw_env, manifest_rows[idx], viewport_capturer)
                    else:
                        frame = _rgb_frame(raw_env, f"table_cam_{idx}")
                    last_frames[spec.name] = frame.copy()
                    writers[spec.name].write(frame)

            with torch.inference_mode():
                obs_t = agent.obs_to_torch(obs)
                actions = agent.get_action(obs_t, is_deterministic=deterministic_actions)
                obs, _, dones, _ = env.step(actions)
                if isinstance(obs, dict):
                    obs = obs["obs"]
                if len(dones) > 0:
                    if isinstance(dones, torch.Tensor):
                        final_done = bool(dones[0].item())
                    else:
                        final_done = bool(dones[0])
                    if final_done:
                        final_success = _read_success_flag(raw_env)
                        outcome = "true_success" if final_success else "non_success_done"
                        print(f"[INFO] Env done at episode {episode_idx} policy step {step} ({outcome}).")
                        if agent_is_rnn and agent.states is not None:
                            for s in agent.states:
                                s[:, dones, :] = 0.0
                        if final_success or episode_idx >= args_cli.max_episodes:
                            break
                        episode_idx += 1
                        step = 0
                        final_done = False
                        last_frames.clear()
                        continue
            step += 1
    finally:
        for writer in writers.values():
            writer.close()

    # Save a final still. If the env auto-reset after success, reuse the last pre-step media frame.
    if not args_cli.no_screenshots:
        if args_cli.capture_backend == "sensor" and not final_done:
            _refresh_cameras(raw_env, max(1, args_cli.warmup_frames // 2))
        for idx, spec in enumerate(specs):
            if final_done and spec.name in last_frames:
                frame = last_frames[spec.name]
            elif args_cli.capture_backend == "viewport":
                frame = _viewport_frame(raw_env, manifest_rows[idx], viewport_capturer)
            else:
                frame = _rgb_frame(raw_env, f"table_cam_{idx}")
            _write_png(screenshots_dir / spec.name / "final.png", frame)

    _print_outcome_summary(raw_env)
    print(f"[INFO] Saved media to: {run_dir}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
