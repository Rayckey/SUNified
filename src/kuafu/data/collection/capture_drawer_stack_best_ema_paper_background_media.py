#!/usr/bin/env python3
"""Capture review screenshots and MP4s for the drawer-stack best EMA policy with optional paper backgrounds.

This script is intentionally separate from the point-cloud collection loop.  It
uses the same policy/prior/weights inputs, but records clean RGB media from
compositional camera views for human review.  The optional paper background is
visual-only and scoped to this capture script.
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


DEFAULT_TASK = "FR3-FinRay-Table-Drawer-Stack-fff"
DEFAULT_CHECKPOINT = (
    "Residuals/FR3-FinRay-Table-Drawer-Stack-fff/2026-05-14_11-09-17/"
    "rl_runs/2026-05-14_11-12-18_Run-2/nn/last_drawer-stack_ep_1050_rew_3812.3352.pth"
)
DEFAULT_PRIOR_CHECKPOINT = "Priors/herm/drawer_stack/drawer_stack_chunking_30_fff/best.pth"
DEFAULT_WEIGHTS_PARAMS_JSON = "Residuals/FR3-FinRay-Table-Drawer-Stack-fff/2026-05-14_11-04-55/best.json"
DEFAULT_SEED = 20260516


@dataclass(frozen=True)
class CameraSpec:
    name: str
    offset_xyz: tuple[float, float, float]
    target_bias_xyz: tuple[float, float, float] = (0.0, 0.0, 0.08)
    focal_length: float = 18.0


CAMERA_BANKS: dict[str, list[CameraSpec]] = {
    # Best reviewed close -Y view: move forward to remove pillar occlusion while keeping drawer, strawberry, and gripper visible.
    "best": [
        CameraSpec("between_pillars_close_action", (0.00, -0.62, 0.56), (-0.10, 0.22, -0.03), 19.0),
    ],
    # Close review candidates: prioritize drawer, strawberry, and gripper while cropping out table pillars.
    "drawer_close": [
        CameraSpec("drawer_close_front", (0.95, -0.05, 0.42), (-0.06, 0.02, 0.05), 22.0),
        CameraSpec("drawer_close_right", (0.80, -0.45, 0.40), (-0.05, 0.03, 0.05), 20.0),
        CameraSpec("gripper_drawer_side", (0.55, 0.45, 0.38), (-0.04, 0.00, 0.05), 22.0),
    ],
    # Requested -Y view: look through the table pillars toward +Y.
    "between_pillars": [
        CameraSpec("between_pillars_center", (0.00, -1.10, 0.48), (-0.02, 0.28, 0.05), 18.0),
        CameraSpec("between_pillars_low", (0.00, -0.95, 0.38), (-0.02, 0.24, 0.03), 20.0),
        CameraSpec("between_pillars_high", (0.02, -1.18, 0.58), (-0.02, 0.32, 0.05), 20.0),
    ],
    # Refined -Y view: higher camera, slight left yaw, and downward look to reduce the right pillar.
    "between_pillars_refined": [
        CameraSpec("between_pillars_left_high_soft", (0.00, -1.10, 0.62), (-0.12, 0.28, -0.02), 18.0),
        CameraSpec("between_pillars_left_high_mid", (0.00, -1.10, 0.68), (-0.18, 0.28, -0.03), 19.0),
        CameraSpec("between_pillars_left_high_strong", (0.00, -1.08, 0.68), (-0.25, 0.27, -0.03), 20.0),
    ],
    # Closer -Y view: move forward past the pillar framing while keeping gripper/drawer/strawberry readable.
    "between_pillars_close": [
        CameraSpec("between_pillars_close_context", (0.00, -0.78, 0.60), (-0.10, 0.25, -0.03), 18.0),
        CameraSpec("between_pillars_close_action", (0.00, -0.62, 0.56), (-0.10, 0.22, -0.03), 19.0),
        CameraSpec("between_pillars_close_tight", (0.00, -0.48, 0.52), (-0.08, 0.18, -0.03), 20.0),
    ],
    # First pass candidates: all are task-centric, not fixed real-sensor poses.
    "sweep_a": [
        CameraSpec("hero_front_right", (1.60, -1.45, 0.82), (0.00, 0.00, 0.02), 12.0),
        CameraSpec("robot_strawberry_side", (0.95, 1.55, 0.74), (0.00, 0.00, 0.02), 14.0),
        CameraSpec("drawer_side_low", (-1.25, -1.35, 0.70), (0.00, 0.00, 0.02), 16.0),
    ],
    # Alternate candidates if the first bank occludes the strawberry or gripper.
    "sweep_b": [
        CameraSpec("front_low_wide", (1.70, -0.20, 0.58), (0.00, 0.00, 0.02), 12.0),
        CameraSpec("over_robot_shoulder", (0.45, 1.65, 0.95), (0.00, 0.00, 0.02), 16.0),
        CameraSpec("top_oblique_context", (1.05, -0.95, 1.22), (0.00, 0.00, 0.00), 18.0),
    ],
    # Slightly tighter versions for the final hero video once an angle is chosen.
    "hero": [
        CameraSpec("hero_front_right_tight", (1.35, -1.22, 0.72), (0.00, 0.00, 0.02), 18.0),
        CameraSpec("hero_front_right_wide", (1.60, -1.45, 0.82), (0.00, 0.00, 0.02), 12.0),
        CameraSpec("over_robot_shoulder", (0.35, 1.28, 0.92), (-0.02, 0.02, 0.06), 20.0),
    ],
}


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Record deterministic drawer-stack policy media from candidate camera views."
    )
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--prior_checkpoint", default=DEFAULT_PRIOR_CHECKPOINT)
    parser.add_argument(
        "--prior_mode",
        choices=["full_actor", "residual_rl"],
        default="residual_rl",
        help="Policy playback mode. Drawer-stack best EMA uses residual_rl.",
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
        "--paper_background",
        type=str,
        default="none",
        choices=[
            "none",
            "custom_usd",
            "simple_room",
            "warehouse_small",
            "warehouse_full",
            "warehouse_shelves",
            "warehouse_forklifts",
            "digital_twin_warehouse",
            "hospital",
            "office",
            "jetracer_track",
            "grid",
            "grid_black",
            "grid_curved",
            "modular_warehouse_h450_center",
            "modular_warehouse_h450_corner_in",
            "modular_warehouse_h450_straight",
            "modular_warehouse_h10_center",
            "modular_warehouse_h10_corner_in",
            "modular_warehouse_h10_straight",
            "outdoor_rivermark",
            "terrain_flat",
            "terrain_rough",
            "terrain_slope",
            "terrain_stairs",
            "procedural_room",
        ],
        help="Visual-only built-in or procedural background for paper screenshots/media.",
    )
    parser.add_argument(
        "--paper_background_usd",
        type=str,
        default="",
        help="Local USD/USDA file to load when --paper_background custom_usd is selected.",
    )
    parser.add_argument("--paper_background_pos", type=str, default="0.0,0.9,0.0")
    parser.add_argument("--paper_background_rot", type=str, default="0.0,0.0,180.0")
    parser.add_argument("--paper_background_scale", type=str, default="1.0,1.0,1.0")
    parser.add_argument(
        "--paper_table_visual",
        type=str,
        default="none",
        choices=[
            "none",
            "hide_original",
            "procedural_slate",
            "procedural_wood",
            "procedural_lab",
            "seattle_lab_table",
            "thorlabs_table",
            "mounts_table",
            "textured_table",
            "thor_table",
        ],
        help="Screenshot-only table visual replacement. Non-none hides the task table render mesh but keeps physics unchanged.",
    )
    parser.add_argument("--paper_table_visual_pos", type=str, default="0.0,0.0,0.0")
    parser.add_argument("--paper_table_visual_rot", type=str, default="0.0,0.0,0.0")
    parser.add_argument("--paper_table_visual_scale", type=str, default="1.0,1.0,1.0")
    parser.add_argument(
        "--debug_frames",
        action="store_true",
        help="Show frame-transformer debug markers for drawer, strawberry, fingertip, and link frames.",
    )
    parser.add_argument("--max_steps", type=int, default=900)
    parser.add_argument("--video_every", type=int, default=1, help="Record every Nth policy step.")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--warmup_frames", type=int, default=8)
    parser.add_argument(
        "--screenshot_steps",
        default="0,90,180,270,360,540,720,899",
        help="Comma-separated policy steps to save as PNG stills.",
    )
    parser.add_argument(
        "--output_dir",
        default=str(external_artifact_root() / "media" / "drawer_stack_best_ema"),
        help="Directory where a timestamped media run will be written.",
    )
    parser.add_argument("--no_video", action="store_true")
    parser.add_argument("--no_screenshots", action="store_true")
    parser.add_argument("--stochastic_policy", action="store_true")
    parser.add_argument(
        "--default_viewport_snapshot",
        action="store_true",
        help="Run policy steps, then save one PNG from the active default viewport without moving the camera.",
    )
    parser.add_argument(
        "--snapshot_name",
        default="",
        help="Optional PNG filename for --default_viewport_snapshot.",
    )
    AppLauncher.add_app_launcher_args(parser)
    args, hydra_args = parser.parse_known_args()
    if args.num_envs != 1 and not args.default_viewport_snapshot:
        raise ValueError("This media script expects --num_envs 1 unless --default_viewport_snapshot is enabled.")
    if args.max_steps <= 0:
        raise ValueError("--max_steps must be positive.")
    if args.video_every <= 0:
        raise ValueError("--video_every must be positive.")
    if args.fps <= 0:
        raise ValueError("--fps must be positive.")
    args.enable_cameras = True
    if args.default_viewport_snapshot:
        args.no_video = True
        args.no_screenshots = True
    if args.capture_backend == "viewport" or args.default_viewport_snapshot:
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
import omni.usd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
import kuafu.isaac.learned  # noqa: F401,E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent  # noqa: E402
from isaaclab.sensors import TiledCameraCfg  # noqa: E402
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, retrieve_file_path  # noqa: E402
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


BACKGROUND_USDS = {
    "simple_room": f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Room/simple_room.usd",
    "warehouse_small": f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse.usd",
    "warehouse_full": f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/full_warehouse.usd",
    "warehouse_shelves": f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd",
    "warehouse_forklifts": f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse_with_forklifts.usd",
    "digital_twin_warehouse": f"{ISAAC_NUCLEUS_DIR}/Environments/Digital_Twin_Warehouse/small_warehouse_digital_twin.usd",
    "hospital": f"{ISAAC_NUCLEUS_DIR}/Environments/Hospital/hospital.usd",
    "office": f"{ISAAC_NUCLEUS_DIR}/Environments/Office/office.usd",
    "jetracer_track": f"{ISAAC_NUCLEUS_DIR}/Environments/Jetracer/jetracer_track_solid.usd",
    "grid": f"{ISAAC_NUCLEUS_DIR}/Environments/Grid/default_environment.usd",
    "grid_black": f"{ISAAC_NUCLEUS_DIR}/Environments/Grid/gridroom_black.usd",
    "grid_curved": f"{ISAAC_NUCLEUS_DIR}/Environments/Grid/gridroom_curved.usd",
    "modular_warehouse_h450_center": f"{ISAAC_NUCLEUS_DIR}/Environments/Modular_Warehouse/Props/warehouse_h450cm_center.usd",
    "modular_warehouse_h450_corner_in": f"{ISAAC_NUCLEUS_DIR}/Environments/Modular_Warehouse/Props/warehouse_h450cm_corner_in.usd",
    "modular_warehouse_h450_straight": f"{ISAAC_NUCLEUS_DIR}/Environments/Modular_Warehouse/Props/warehouse_h450cm_straight.usd",
    "modular_warehouse_h10_center": f"{ISAAC_NUCLEUS_DIR}/Environments/Modular_Warehouse/Props/warehouse_h10m_center.usd",
    "modular_warehouse_h10_corner_in": f"{ISAAC_NUCLEUS_DIR}/Environments/Modular_Warehouse/Props/warehouse_h10m_corner_in.usd",
    "modular_warehouse_h10_straight": f"{ISAAC_NUCLEUS_DIR}/Environments/Modular_Warehouse/Props/warehouse_h10m_straight.usd",
    "outdoor_rivermark": f"{ISAAC_NUCLEUS_DIR}/Environments/Outdoor/Rivermark/rivermark.usd",
    "terrain_flat": f"{ISAAC_NUCLEUS_DIR}/Environments/Terrains/flat_plane.usd",
    "terrain_rough": f"{ISAAC_NUCLEUS_DIR}/Environments/Terrains/rough_plane.usd",
    "terrain_slope": f"{ISAAC_NUCLEUS_DIR}/Environments/Terrains/slope.usd",
    "terrain_stairs": f"{ISAAC_NUCLEUS_DIR}/Environments/Terrains/stairs.usd",
}


TABLE_VISUAL_USDS = {
    "seattle_lab_table": f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
    "thorlabs_table": f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/ThorlabsTable/table_instanceable.usd",
    "mounts_table": f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/table.usd",
    "textured_table": f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/textured_table.usd",
    "thor_table": f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/thor_table.usd",
}


def parse_vec3(text: str) -> tuple[float, float, float]:
    values = [float(x.strip()) for x in text.split(",")]
    assert len(values) == 3, f"Expected comma-separated vec3, got: {text}"
    return tuple(values)


def disable_physics_under(stage: Usd.Stage, root_path: str) -> None:
    root = stage.GetPrimAtPath(root_path)
    assert root.IsValid(), f"Invalid background root path: {root_path}"

    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)

        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)


def _current_stage() -> Usd.Stage:
    stage = omni.usd.get_context().get_stage()
    assert stage is not None, "USD stage does not exist yet."
    return stage


def _set_xform(
    prim,
    pos: tuple[float, float, float],
    rot_xyz_deg: tuple[float, float, float],
    scale: tuple[float, float, float],
) -> None:
    xform_api = UsdGeom.XformCommonAPI(prim)
    xform_api.SetTranslate(Gf.Vec3d(*pos))
    xform_api.SetRotate(
        Gf.Vec3f(*rot_xyz_deg),
        UsdGeom.XformCommonAPI.RotationOrderXYZ,
    )
    xform_api.SetScale(Gf.Vec3f(*scale))


def add_custom_usd_paper_background(
    usd_path: str,
    pos=(0.0, 0.9, 0.0),
    rot_xyz_deg=(0.0, 0.0, 180.0),
    scale=(1.0, 1.0, 1.0),
    prim_path="/World/PaperBackground",
) -> None:
    assert usd_path, "--paper_background_usd is required when --paper_background custom_usd"
    usd_file = Path(usd_path).expanduser()
    assert usd_file.exists(), f"Custom background USD does not exist: {usd_file}"

    stage = _current_stage()

    if stage.GetPrimAtPath(prim_path).IsValid():
        stage.RemovePrim(prim_path)

    xform = UsdGeom.Xform.Define(stage, prim_path)
    prim = xform.GetPrim()
    _set_xform(prim, pos, rot_xyz_deg, scale)

    asset_prim = UsdGeom.Xform.Define(stage, f"{prim_path}/Asset").GetPrim()
    asset_prim.GetReferences().AddReference(str(usd_file.resolve()))

    disable_physics_under(stage, prim_path)
    print(
        f"[paper background] Added custom USD {usd_file} at {prim_path}, "
        f"pos={pos}, rot={rot_xyz_deg}, scale={scale}"
    )


def add_usd_paper_background(
    background_name: str,
    pos=(0.0, 0.9, 0.0),
    rot_xyz_deg=(0.0, 0.0, 180.0),
    scale=(1.0, 1.0, 1.0),
    prim_path="/World/PaperBackground",
) -> None:
    assert background_name in BACKGROUND_USDS, (
        f"Unknown paper background: {background_name}. "
        f"Available: {list(BACKGROUND_USDS.keys())}"
    )

    stage = _current_stage()

    if stage.GetPrimAtPath(prim_path).IsValid():
        stage.RemovePrim(prim_path)

    xform = UsdGeom.Xform.Define(stage, prim_path)
    prim = xform.GetPrim()
    _set_xform(prim, pos, rot_xyz_deg, scale)

    asset_prim = UsdGeom.Xform.Define(stage, f"{prim_path}/Asset").GetPrim()
    asset_prim.GetReferences().AddReference(BACKGROUND_USDS[background_name])

    disable_physics_under(stage, prim_path)
    print(
        f"[paper background] Added {background_name} at {prim_path}, "
        f"pos={pos}, rot={rot_xyz_deg}, scale={scale}"
    )


def _define_room_box(
    stage: Usd.Stage,
    prim_path: str,
    size: tuple[float, float, float],
    pos: tuple[float, float, float],
    color: tuple[float, float, float],
) -> None:
    cube = UsdGeom.Cube.Define(stage, prim_path)
    cube.CreateSizeAttr(1.0)
    cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdGeom.XformCommonAPI(cube.GetPrim()).SetTranslate(Gf.Vec3d(*pos))
    UsdGeom.XformCommonAPI(cube.GetPrim()).SetScale(Gf.Vec3f(*size))


def add_procedural_paper_room(
    pos=(0.0, 0.9, 0.0),
    rot_xyz_deg=(0.0, 0.0, 180.0),
    scale=(1.0, 1.0, 1.0),
    prim_path="/World/PaperBackground",
) -> None:
    stage = _current_stage()

    if stage.GetPrimAtPath(prim_path).IsValid():
        stage.RemovePrim(prim_path)

    xform = UsdGeom.Xform.Define(stage, prim_path)
    _set_xform(xform.GetPrim(), pos, rot_xyz_deg, scale)

    room_parts = (
        ("floor", (4.0, 3.0, 0.03), (0.0, 0.0, -0.03), (0.56, 0.58, 0.56)),
        ("back_wall", (3.5, 0.04, 1.6), (0.0, 1.15, 0.75), (0.70, 0.72, 0.69)),
        ("side_wall", (0.04, 2.2, 1.6), (-1.35, 0.25, 0.75), (0.63, 0.66, 0.64)),
        ("shelf", (1.2, 0.08, 0.04), (0.55, 1.08, 0.8), (0.45, 0.39, 0.32)),
    )
    for name, part_size, part_pos, color in room_parts:
        _define_room_box(stage, f"{prim_path}/{name}", part_size, part_pos, color)

    disable_physics_under(stage, prim_path)
    print(
        f"[paper background] Added procedural_room at {prim_path}, "
        f"pos={pos}, rot={rot_xyz_deg}, scale={scale}"
    )


def add_paper_background(
    background_name: str,
    pos=(0.0, 0.9, 0.0),
    rot_xyz_deg=(0.0, 0.0, 180.0),
    scale=(1.0, 1.0, 1.0),
    prim_path="/World/PaperBackground",
) -> None:
    if background_name == "none":
        return
    if background_name == "procedural_room":
        add_procedural_paper_room(pos=pos, rot_xyz_deg=rot_xyz_deg, scale=scale, prim_path=prim_path)
    else:
        add_usd_paper_background(
            background_name,
            pos=pos,
            rot_xyz_deg=rot_xyz_deg,
            scale=scale,
            prim_path=prim_path,
        )


def hide_existing_table_visuals(table_prim_pattern="/World/envs/env_.*/Table") -> int:
    hidden_count = 0
    for root_prim in sim_utils.find_matching_prims(table_prim_pattern):
        for prim in Usd.PrimRange(root_prim):
            if prim.IsA(UsdGeom.Imageable):
                UsdGeom.Imageable(prim).MakeInvisible()
                hidden_count += 1
    print(f"[paper table] Hid {hidden_count} existing table visual prims under {table_prim_pattern}")
    return hidden_count


def add_usd_paper_table_visual(
    table_name: str,
    pos=(0.0, 0.0, 0.0),
    rot_xyz_deg=(0.0, 0.0, 0.0),
    scale=(1.0, 1.0, 1.0),
    prim_path="/World/PaperTableVisual",
) -> None:
    assert table_name in TABLE_VISUAL_USDS, (
        f"Unknown paper table visual: {table_name}. "
        f"Available: {list(TABLE_VISUAL_USDS.keys())}"
    )

    stage = _current_stage()
    if stage.GetPrimAtPath(prim_path).IsValid():
        stage.RemovePrim(prim_path)

    xform = UsdGeom.Xform.Define(stage, prim_path)
    prim = xform.GetPrim()
    _set_xform(prim, pos, rot_xyz_deg, scale)

    asset_prim = UsdGeom.Xform.Define(stage, f"{prim_path}/Asset").GetPrim()
    asset_prim.GetReferences().AddReference(TABLE_VISUAL_USDS[table_name])

    disable_physics_under(stage, prim_path)
    print(
        f"[paper table] Added {table_name} at {prim_path}, "
        f"pos={pos}, rot={rot_xyz_deg}, scale={scale}"
    )


def add_procedural_paper_table_visual(
    table_name: str,
    pos=(0.0, 0.0, 0.0),
    rot_xyz_deg=(0.0, 0.0, 0.0),
    scale=(1.0, 1.0, 1.0),
    prim_path="/World/PaperTableVisual",
) -> None:
    stage = _current_stage()
    if stage.GetPrimAtPath(prim_path).IsValid():
        stage.RemovePrim(prim_path)

    xform = UsdGeom.Xform.Define(stage, prim_path)
    _set_xform(xform.GetPrim(), pos, rot_xyz_deg, scale)

    palettes = {
        "procedural_slate": {
            "top": (0.09, 0.12, 0.14),
            "edge": (0.18, 0.20, 0.22),
            "leg": (0.42, 0.43, 0.42),
        },
        "procedural_wood": {
            "top": (0.46, 0.30, 0.16),
            "edge": (0.25, 0.17, 0.10),
            "leg": (0.28, 0.20, 0.13),
        },
        "procedural_lab": {
            "top": (0.16, 0.19, 0.22),
            "edge": (0.72, 0.73, 0.70),
            "leg": (0.62, 0.64, 0.62),
        },
    }
    assert table_name in palettes, f"Unknown procedural table visual: {table_name}"
    colors = palettes[table_name]

    parts = [
        ("top", (1.65, 1.10, 0.045), (0.20, -0.05, -0.035), colors["top"]),
        ("front_edge", (1.68, 0.035, 0.055), (0.20, -0.61, -0.045), colors["edge"]),
        ("back_edge", (1.68, 0.035, 0.055), (0.20, 0.51, -0.045), colors["edge"]),
        ("left_edge", (0.035, 1.10, 0.055), (-0.64, -0.05, -0.045), colors["edge"]),
        ("right_edge", (0.035, 1.10, 0.055), (1.04, -0.05, -0.045), colors["edge"]),
        ("front_left_leg", (0.055, 0.055, 0.62), (-0.54, -0.50, -0.36), colors["leg"]),
        ("front_right_leg", (0.055, 0.055, 0.62), (0.94, -0.50, -0.36), colors["leg"]),
        ("back_left_leg", (0.055, 0.055, 0.62), (-0.54, 0.40, -0.36), colors["leg"]),
        ("back_right_leg", (0.055, 0.055, 0.62), (0.94, 0.40, -0.36), colors["leg"]),
    ]
    if table_name == "procedural_lab":
        parts.extend(
            [
                ("left_crossbar", (0.055, 0.80, 0.035), (-0.54, -0.05, -0.32), colors["edge"]),
                ("right_crossbar", (0.055, 0.80, 0.035), (0.94, -0.05, -0.32), colors["edge"]),
            ]
        )

    for name, part_size, part_pos, color in parts:
        _define_room_box(stage, f"{prim_path}/{name}", part_size, part_pos, color)

    disable_physics_under(stage, prim_path)
    print(
        f"[paper table] Added {table_name} at {prim_path}, "
        f"pos={pos}, rot={rot_xyz_deg}, scale={scale}"
    )


def add_paper_table_visual(
    table_name: str,
    pos=(0.0, 0.0, 0.0),
    rot_xyz_deg=(0.0, 0.0, 0.0),
    scale=(1.0, 1.0, 1.0),
    prim_path="/World/PaperTableVisual",
) -> None:
    if table_name in ("none", "hide_original"):
        return
    if table_name.startswith("procedural_"):
        add_procedural_paper_table_visual(table_name, pos=pos, rot_xyz_deg=rot_xyz_deg, scale=scale, prim_path=prim_path)
    else:
        add_usd_paper_table_visual(table_name, pos=pos, rot_xyz_deg=rot_xyz_deg, scale=scale, prim_path=prim_path)


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
    "top_drawer_handle_frame",
    "top_drawer_center_frame",
    "strawberry_center_frame",
)
FRAME_DEBUG_SENSOR_NAMES = (
    "panda_fingertip",
    "panda_link4",
    "top_drawer_handle_frame",
    "top_drawer_center_frame",
    "strawberry_center_frame",
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

    try:
        task_points.append(raw_env.strawberry.data.root_pos_w[0])
    except Exception:
        pass
    try:
        task_points.append(raw_env.top_drawer_center_frame.data.target_pos_w[0, 0])
    except Exception:
        pass
    if not task_points:
        return torch.tensor([0.35, -0.25, 0.16], device=device)

    focus = torch.stack(task_points, dim=0).mean(dim=0)
    # Keep the look-at point low enough to include the table, drawer, gripper, and strawberry.
    focus[2] = torch.clamp(focus[2], min=0.08, max=0.22)
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
                "default_viewport_snapshot": bool(getattr(args, "default_viewport_snapshot", False)),
                "snapshot_name": getattr(args, "snapshot_name", ""),
                "num_envs": int(getattr(args, "num_envs", 1)),
                "screenshot_steps": args.screenshot_steps,
                "paper_background": args.paper_background,
                "paper_background_usd": args.paper_background_usd,
                "paper_background_pos": args.paper_background_pos,
                "paper_background_rot": args.paper_background_rot,
                "paper_background_scale": args.paper_background_scale,
                "paper_table_visual": args.paper_table_visual,
                "paper_table_visual_pos": args.paper_table_visual_pos,
                "paper_table_visual_rot": args.paper_table_visual_rot,
                "paper_table_visual_scale": args.paper_table_visual_scale,
            },
            f,
            indent=2,
        )


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    specs = [] if args_cli.default_viewport_snapshot else CAMERA_BANKS[args_cli.camera_bank]
    task_name = args_cli.task.split(":")[-1]
    timestamp = _datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(args_cli.output_dir).expanduser()
    if not run_dir.is_absolute():
        run_dir = REPO_ROOT / run_dir
    frame_variant = "default_viewport" if args_cli.default_viewport_snapshot else ("debugframes" if args_cli.debug_frames else "clean")
    frame_variant = f"{frame_variant}_bg{args_cli.paper_background}_table{args_cli.paper_table_visual}"
    run_dir = run_dir / f"{timestamp}_{args_cli.camera_bank}_{frame_variant}_seed{args_cli.seed}"
    screenshots_dir = run_dir / "screenshots"
    videos_dir = run_dir / "videos"
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args_cli.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args_cli.seed)

    env_cfg.scene.num_envs = int(args_cli.num_envs) if args_cli.default_viewport_snapshot else 1
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.mpc_config["num_envs"] = env_cfg.scene.num_envs
    env_cfg.mpc_config["use_per_env_weight"] = False
    env_cfg.enable_cameras = not bool(args_cli.default_viewport_snapshot)
    env_cfg.seed = args_cli.seed
    # FrameTransformer debug markers are enabled after reset, once sensor data exists.
    _set_frame_debug_cfg(env_cfg, False)
    if not args_cli.default_viewport_snapshot:
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

    if args_cli.paper_table_visual != "none" and hasattr(env_cfg, "hide_table_visuals"):
        env_cfg.hide_table_visuals = True

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

    if args_cli.paper_table_visual != "none":
        hide_existing_table_visuals()
        pos = parse_vec3(args_cli.paper_table_visual_pos)
        rot = parse_vec3(args_cli.paper_table_visual_rot)
        scale = parse_vec3(args_cli.paper_table_visual_scale)
        add_paper_table_visual(
            args_cli.paper_table_visual,
            pos=pos,
            rot_xyz_deg=rot,
            scale=scale,
        )
        raw_env.sim.render()
        simulation_app.update()

    if args_cli.paper_background != "none":
        pos = parse_vec3(args_cli.paper_background_pos)
        rot = parse_vec3(args_cli.paper_background_rot)
        scale = parse_vec3(args_cli.paper_background_scale)
        if args_cli.paper_background == "custom_usd":
            add_custom_usd_paper_background(
                args_cli.paper_background_usd,
                pos=pos,
                rot_xyz_deg=rot,
                scale=scale,
            )
        else:
            add_paper_background(
                args_cli.paper_background,
                pos=pos,
                rot_xyz_deg=rot,
                scale=scale,
            )
        raw_env.sim.render()
        simulation_app.update()

    if args_cli.default_viewport_snapshot:
        viewport_capturer = ViewportCapturer(args_cli.width, args_cli.height)
        print("=== Drawer Stack Default Viewport Snapshot ===")
        print(f"  task          : {task_name}")
        print(f"  seed          : {args_cli.seed}")
        print(f"  num envs      : {raw_env.num_envs}")
        print(f"  steps         : {args_cli.max_steps}")
        print(f"  action mode   : {'stochastic' if args_cli.stochastic_policy else 'deterministic'}")
        print(f"  paper bg      : {args_cli.paper_background}")
        print(f"  paper table   : {args_cli.paper_table_visual}")
        print(f"  run dir       : {run_dir}")

        deterministic_actions = not args_cli.stochastic_policy
        for step in range(args_cli.max_steps):
            with torch.inference_mode():
                obs_t = agent.obs_to_torch(obs)
                actions = agent.get_action(obs_t, is_deterministic=deterministic_actions)
                obs, _, dones, _ = env.step(actions)
                if isinstance(obs, dict):
                    obs = obs["obs"]
                if agent_is_rnn and agent.states is not None:
                    if isinstance(dones, torch.Tensor) and dones.numel() > 0 and dones.any():
                        for state in agent.states:
                            state[:, dones, :] = 0.0
            if (step + 1) % 50 == 0 or step == args_cli.max_steps - 1:
                print(f"[INFO] Ran {step + 1}/{args_cli.max_steps} policy steps.")

        for _ in range(max(1, int(args_cli.viewport_settle_frames), int(args_cli.warmup_frames))):
            raw_env.sim.render()
            simulation_app.update()
        frame = viewport_capturer.capture_rgb()
        snapshots_dir = run_dir / "snapshots"
        snapshot_name = args_cli.snapshot_name.strip() or f"default_viewport_envs{raw_env.num_envs}_step{args_cli.max_steps:04d}.png"
        if not snapshot_name.lower().endswith(".png"):
            snapshot_name += ".png"
        snapshot_path = snapshots_dir / snapshot_name
        _write_png(snapshot_path, frame)
        with open(run_dir / "run_config.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "task": args_cli.task,
                    "checkpoint": args_cli.checkpoint,
                    "prior_checkpoint": args_cli.prior_checkpoint,
                    "prior_mode": args_cli.prior_mode,
                    "weights_params_json": args_cli.weights_params_json,
                    "seed": args_cli.seed,
                    "num_envs": int(raw_env.num_envs),
                    "max_steps": args_cli.max_steps,
                    "width": args_cli.width,
                    "height": args_cli.height,
                    "stochastic_policy": bool(args_cli.stochastic_policy),
                    "default_viewport_snapshot": True,
                    "paper_background": args_cli.paper_background,
                    "paper_background_usd": args_cli.paper_background_usd,
                    "paper_background_pos": args_cli.paper_background_pos,
                    "paper_background_rot": args_cli.paper_background_rot,
                    "paper_background_scale": args_cli.paper_background_scale,
                    "paper_table_visual": args_cli.paper_table_visual,
                    "paper_table_visual_pos": args_cli.paper_table_visual_pos,
                    "paper_table_visual_rot": args_cli.paper_table_visual_rot,
                    "paper_table_visual_scale": args_cli.paper_table_visual_scale,
                    "snapshot_path": str(snapshot_path),
                },
                f,
                indent=2,
            )
        print(f"[INFO] Saved default viewport snapshot to: {snapshot_path}")
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

    print("=== Drawer Stack Media Capture ===")
    print(f"  task          : {task_name}")
    print(f"  seed          : {args_cli.seed}")
    print(f"  camera bank   : {args_cli.camera_bank}")
    print(f"  backend       : {args_cli.capture_backend}")
    print(f"  debug frames  : {args_cli.debug_frames}")
    print(f"  action mode   : {'stochastic' if args_cli.stochastic_policy else 'deterministic'}")
    print(f"  paper bg      : {args_cli.paper_background}")
    print(f"  paper table   : {args_cli.paper_table_visual}")
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
                    _write_png(screenshots_dir / spec.name / f"step_{step:04d}.png", frame)

            if writers and step % args_cli.video_every == 0:
                if args_cli.capture_backend == "sensor":
                    _refresh_cameras(raw_env, 1)
                for idx, spec in enumerate(specs):
                    if args_cli.capture_backend == "viewport":
                        frame = _viewport_frame(raw_env, manifest_rows[idx], viewport_capturer)
                    else:
                        frame = _rgb_frame(raw_env, f"table_cam_{idx}")
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

    # Save a final still after the last simulated step.
    if not args_cli.no_screenshots:
        if args_cli.capture_backend == "sensor":
            _refresh_cameras(raw_env, max(1, args_cli.warmup_frames // 2))
        for idx, spec in enumerate(specs):
            if args_cli.capture_backend == "viewport":
                frame = _viewport_frame(raw_env, manifest_rows[idx], viewport_capturer)
            else:
                frame = _rgb_frame(raw_env, f"table_cam_{idx}")
            _write_png(screenshots_dir / spec.name / "final.png", frame)

    print(f"[INFO] Saved media to: {run_dir}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
