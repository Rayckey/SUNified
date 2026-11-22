#!/usr/bin/env python3
"""Capture review screenshots and MP4s for the Stack-2 best EMA policy.

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


DEFAULT_TASK = "FR3-FinRay-Table-Stack-2-ff"
DEFAULT_CHECKPOINT = (
    "Residuals/FR3-FinRay-Table-Stack-2-ff/2026-04-21_19-57-10/"
    "rl_runs/2026-04-22_05-06-01_Run-2/nn/"
    "last_FR3-FinRay-Table-Stack-2_ep_900_rew__14910.703_.pth"
)
DEFAULT_PRIOR_CHECKPOINT = (
    "Priors/potential_useful/stack_2_damped_higher_noise_state/"
    "stack2_chunking_15_ff_damped_higher/best.pth"
)
DEFAULT_WEIGHTS_PARAMS_JSON = "Residuals/FR3-FinRay-Table-Stack-2-ff/2026-04-21_19-57-10/best.json"
DEFAULT_SEED = 20260516


@dataclass(frozen=True)
class CameraSpec:
    name: str
    offset_xyz: tuple[float, float, float]
    target_bias_xyz: tuple[float, float, float] = (0.0, 0.0, 0.08)
    focal_length: float = 18.0


CAMERA_BANKS: dict[str, list[CameraSpec]] = {
    # Best reviewed Stack-2 view: a close negative-Y angle that keeps both cubes and the gripper readable.
    "best": [
        CameraSpec("front_close_stack_action", (0.08, -0.70, 0.52), (0.00, 0.08, 0.04), 22.0),
    ],
    # Close review candidates for comparing cube visibility during grasp, carry, and release.
    "front_stack_close": [
        CameraSpec("front_close_stack_context", (0.10, -0.88, 0.60), (0.00, 0.10, 0.03), 18.0),
        CameraSpec("front_close_stack_action", (0.08, -0.70, 0.52), (0.00, 0.08, 0.04), 22.0),
        CameraSpec("front_close_stack_tight", (0.08, -0.58, 0.48), (0.00, 0.06, 0.04), 26.0),
    ],
    # Slightly wider front views: good first-pass camera search with table context.
    "front_stack": [
        CameraSpec("front_stack_center", (0.00, -1.05, 0.58), (0.00, 0.12, 0.04), 18.0),
        CameraSpec("front_stack_left", (-0.28, -0.92, 0.56), (0.02, 0.08, 0.04), 18.0),
        CameraSpec("front_stack_right", (0.32, -0.90, 0.56), (-0.02, 0.08, 0.04), 18.0),
    ],
    # Side obliques can make the vertical cube-on-cube relation easier to read.
    "side_stack": [
        CameraSpec("right_side_stack", (0.72, -0.24, 0.52), (-0.02, 0.02, 0.04), 22.0),
        CameraSpec("left_side_stack", (-0.55, -0.28, 0.52), (0.02, 0.02, 0.04), 22.0),
        CameraSpec("over_robot_stack", (0.10, 0.78, 0.76), (0.00, -0.02, 0.03), 20.0),
    ],
    # First pass candidates: all are task-centric, not fixed real-sensor poses.
    "sweep_a": [
        CameraSpec("hero_front_right", (1.10, -1.05, 0.72), (0.00, 0.00, 0.03), 16.0),
        CameraSpec("front_low_stack", (0.22, -0.92, 0.42), (0.00, 0.08, 0.04), 20.0),
        CameraSpec("over_robot_shoulder", (0.05, 0.95, 0.82), (0.00, -0.04, 0.04), 18.0),
    ],
    # Alternate candidates if the first bank occludes either cube or the gripper.
    "sweep_b": [
        CameraSpec("right_side_low", (0.82, -0.20, 0.42), (-0.02, 0.00, 0.04), 22.0),
        CameraSpec("left_front_oblique", (-0.42, -0.70, 0.56), (0.02, 0.06, 0.04), 20.0),
        CameraSpec("top_oblique_context", (0.62, -0.60, 1.02), (0.00, 0.02, 0.02), 18.0),
    ],
    # Slightly tighter versions for the final hero video once an angle is chosen.
    "hero": [
        CameraSpec("hero_front_right_tight", (0.82, -0.78, 0.58), (0.00, 0.04, 0.04), 22.0),
        CameraSpec("hero_front_right_wide", (1.10, -1.05, 0.72), (0.00, 0.00, 0.03), 16.0),
        CameraSpec("over_robot_shoulder", (0.05, 0.95, 0.82), (0.00, -0.04, 0.04), 18.0),
    ],
}



def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Record deterministic Stack-2 policy media from candidate camera views."
    )
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--prior_checkpoint", default=DEFAULT_PRIOR_CHECKPOINT)
    parser.add_argument(
        "--prior_mode",
        choices=["full_actor", "residual_rl"],
        default="residual_rl",
        help="Policy playback mode. Stack-2 best EMA uses residual_rl.",
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
    parser.add_argument("--viewport_settle_frames", type=int, default=0)
    parser.add_argument(
        "--debug_frames",
        action="store_true",
        help="Show frame-transformer debug markers for fingertip and link frames.",
    )
    parser.add_argument("--max_steps", type=int, default=700)
    parser.add_argument("--video_every", type=int, default=1, help="Record every Nth policy step.")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--warmup_frames", type=int, default=8)
    parser.add_argument(
        "--screenshot_steps",
        default="0,90,180,220,225,226,227,228,229",
        help="Comma-separated policy steps to save as PNG stills.",
    )
    parser.add_argument(
        "--output_dir",
        default=str(external_artifact_root() / "media" / "stack2_best_ema"),
        help="Directory where a timestamped media run will be written.",
    )
    parser.add_argument("--no_video", action="store_true")
    parser.add_argument("--no_screenshots", action="store_true")
    parser.add_argument("--stochastic_policy", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    args, hydra_args = parser.parse_known_args()
    if args.num_envs != 1:
        raise ValueError("This media script expects --num_envs 1 so review media is not tiled across randomized envs.")
    if args.max_steps <= 0:
        raise ValueError("--max_steps must be positive.")
    if args.video_every <= 0:
        raise ValueError("--video_every must be positive.")
    if args.fps <= 0:
        raise ValueError("--fps must be positive.")
    args.enable_cameras = True
    if args.capture_backend == "viewport":
        # AppLauncher disables the active viewport in headless mode unless it sees a video request.
        args.video = True
    return args, hydra_args


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
from omni.kit.viewport.utility import capture_viewport_to_buffer, get_active_viewport  # noqa: E402
from rl_games.common import env_configurations, vecenv  # noqa: E402
from rl_games.common.player import BasePlayer  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402


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
    steps = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        step = int(part)
        if step < 0:
            continue
        steps.add(min(step, max_steps - 1))
    steps.add(0)
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
)
FRAME_DEBUG_SENSOR_NAMES = (
    "panda_fingertip",
    "panda_link4",
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


def _get_scene_focus(raw_env, specs: list[CameraSpec]) -> torch.Tensor:
    device = raw_env.device
    task_points: list[torch.Tensor] = []

    for object_name in ("cube_1", "cube_2"):
        try:
            obj = raw_env.scene.rigid_objects.get(object_name)
        except Exception:
            obj = getattr(raw_env, object_name, None)
        if obj is None:
            continue
        try:
            task_points.append(obj.data.root_pos_w[0])
        except Exception:
            pass

    if not task_points:
        return torch.tensor([0.45, -0.35, 0.10], device=device)

    focus = torch.stack(task_points, dim=0).mean(dim=0)
    # Keep the look-at point high enough to show stacking clearance, but low enough to include the table plane.
    focus[2] = torch.clamp(focus[2], min=0.06, max=0.16)
    return focus


def _set_camera_views(raw_env, specs: list[CameraSpec]) -> list[dict[str, object]]:
    stage = get_current_stage()
    up_axis = UsdGeom.GetStageUpAxis(stage)
    focus = _get_scene_focus(raw_env, specs)
    rows: list[dict[str, object]] = []

    for idx, spec in enumerate(specs):
        cam = raw_env.scene.sensors[f"table_cam_{idx}"]
        offset = torch.tensor(spec.offset_xyz, device=raw_env.device, dtype=torch.float32)
        bias = torch.tensor(spec.target_bias_xyz, device=raw_env.device, dtype=torch.float32)
        eye = focus + offset
        target = focus + bias
        eyes = eye.view(1, 3)
        targets = target.view(1, 3)
        view_rot = create_rotation_matrix_from_view(eyes, targets, up_axis, device=str(raw_env.device))
        quat = quat_from_matrix(view_rot.transpose(1, 2))
        env_ids = torch.tensor([0], device=raw_env.device, dtype=torch.long)
        cam._view.set_world_poses(eyes, quat, env_ids)  # TiledCamera has no public look-at helper.
        if getattr(cam.cfg, "update_latest_camera_pose", False):
            cam._update_poses(env_ids)
        rows.append(
            {
                "sensor": f"table_cam_{idx}",
                "view": spec.name,
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


def _write_manifest(run_dir: Path, rows: list[dict[str, object]], args: argparse.Namespace) -> None:
    with open(run_dir / "camera_manifest.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sensor",
                "view",
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
                "fps": args.fps,
                "width": args.width,
                "height": args.height,
                "video_every": args.video_every,
                "stochastic_policy": bool(args.stochastic_policy),
                "screenshot_steps": args.screenshot_steps,
            },
            f,
            indent=2,
        )


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    specs = CAMERA_BANKS[args_cli.camera_bank]
    task_name = args_cli.task.split(":")[-1]
    timestamp = _datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(args_cli.output_dir).expanduser()
    if not run_dir.is_absolute():
        run_dir = REPO_ROOT / run_dir
    frame_variant = "debugframes" if args_cli.debug_frames else "clean"
    run_dir = run_dir / f"{timestamp}_{args_cli.camera_bank}_{frame_variant}_seed{args_cli.seed}"
    screenshots_dir = run_dir / "screenshots"
    videos_dir = run_dir / "videos"
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args_cli.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args_cli.seed)

    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.mpc_config["num_envs"] = 1
    env_cfg.mpc_config["use_per_env_weight"] = False
    env_cfg.enable_cameras = True
    env_cfg.seed = args_cli.seed
    # FrameTransformer debug markers are enabled after reset, once sensor data exists.
    _set_frame_debug_cfg(env_cfg, False)
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

    manifest_rows = _set_camera_views(raw_env, specs)
    _refresh_cameras(raw_env, args_cli.warmup_frames)
    if args_cli.debug_frames:
        _set_runtime_frame_debug_vis(raw_env, True)
        _refresh_cameras(raw_env, 2)
    viewport_capturer = None
    if args_cli.capture_backend == "viewport":
        viewport_capturer = ViewportCapturer(args_cli.width, args_cli.height)
    _write_manifest(run_dir, manifest_rows, args_cli)

    print("=== Stack-2 Media Capture ===")
    print(f"  task          : {task_name}")
    print(f"  seed          : {args_cli.seed}")
    print(f"  camera bank   : {args_cli.camera_bank}")
    print(f"  backend       : {args_cli.capture_backend}")
    print(f"  debug frames  : {args_cli.debug_frames}")
    print(f"  action mode   : {'stochastic' if args_cli.stochastic_policy else 'deterministic'}")
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
    step = 0
    try:
        while simulation_app.is_running() and step < args_cli.max_steps:
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
                        print(f"[INFO] Env done at policy step {step}.")
                        break
                if agent_is_rnn and final_done and agent.states is not None:
                    for s in agent.states:
                        s[:, dones, :] = 0.0
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

    print(f"[INFO] Saved media to: {run_dir}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
