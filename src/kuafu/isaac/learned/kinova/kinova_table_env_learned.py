# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math

import torch
from pxr import Usd, UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from .base_env_learned import BaseLearnedEnv, BaseLearnedEnvCfg

TABLE_HEIGHT = 0.0
FLOOR_THICKNESS = 1.0


@configclass
class KinovaTableEnvLearnedCfg(BaseLearnedEnvCfg):
    observation_space = 23
    # Match the Panda convention: use an env-local support cube instead of
    # the shared terrain/table mesh so the surface height can be randomized per env.
    terrain: TerrainImporterCfg | None = None

    robot_base_height_range: tuple[float, float] = (0.0, 0.03)
    random_object_spawn_height_range: tuple[float, float] = (-0.02, 0.02)
    object_out_of_bounds_z: float = -0.5
    spawn_angle_lower_deg: float = -60.0
    spawn_angle_upper_deg: float = 60.0
    support_cube_xy_offset: tuple[float, float] = (0.3, 0.0)
    hide_support_cube_visuals: bool = True

    # Invisible 1.0 m x 1.0 m x 1.0 m support cube centered below the surface.
    # Acts as the contact surface instead of a table mesh.
    floor = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Floor",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.3, 0.0, TABLE_HEIGHT - FLOOR_THICKNESS / 2.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.CuboidCfg(
            size=(1.4, 1.4, FLOOR_THICKNESS),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.28, 0.30, 0.32),
                roughness=0.75,
            ),
        ),
    )


    robot_support_base = RigidObjectCfg(
        prim_path="/World/envs/env_.*/RobotSupportBase",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, -0.01),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.CylinderCfg(
            radius=0.13,
            height=0.02,
            axis="Z",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=False,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.10, 0.11, 0.12),
                roughness=0.7,
                metallic=0.0,
            ),
        ),
    )


class KinovaTableEnvLearned(BaseLearnedEnv):
    """Kinova environment with a hidden task support cube and visible robot mount."""

    def __init__(self, cfg: KinovaTableEnvLearnedCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

    def _hide_matching_prim_trees(self, prim_path: str, label: str, warn_if_empty: bool = False) -> int:
        hidden_count = 0
        for root_prim in sim_utils.find_matching_prims(prim_path):
            for prim in Usd.PrimRange(root_prim):
                if prim.IsA(UsdGeom.Imageable):
                    UsdGeom.Imageable(prim).MakeInvisible()
                    hidden_count += 1

        if warn_if_empty and hidden_count == 0:
            print(
                f"[KinovaTableEnvLearned] WARNING: requested hiding {label} visuals but no prims matched "
                f"{prim_path}",
                flush=True,
            )
        return hidden_count

    def _hide_static_surface_visual_prims(self) -> None:
        """Optionally hide the support cube while leaving its collision active."""
        if not self.cfg.hide_support_cube_visuals:
            return
        self._hide_matching_prim_trees(self.cfg.floor.prim_path, "floor")

    def _setup_scene(self):
        super()._setup_scene()

        self.floor = RigidObject(self.cfg.floor)
        self.robot_support_base = RigidObject(self.cfg.robot_support_base)
        self.scene.rigid_objects["floor"] = self.floor
        self.scene.rigid_objects["robot_support_base"] = self.robot_support_base
        self._hide_static_surface_visual_prims()

    def _ensure_support_surface_buffers(self) -> None:
        if hasattr(self, "_robot_base_heights") and hasattr(self, "_support_surface_z_offsets"):
            return
        self._robot_base_heights = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self._support_surface_z_offsets = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)

    def _prepare_support_surface_reset(self, env_ids: torch.Tensor, *, resample: bool = True) -> None:
        """Sample and place the compensated support surface for the selected envs."""
        self._ensure_support_surface_buffers()
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        if len(env_ids) == 0:
            return

        if resample:
            min_height, max_height = self.cfg.robot_base_height_range
            min_height = float(min_height)
            max_height = float(max_height)
            if max_height < min_height:
                raise ValueError(
                    "robot_base_height_range must be ordered as (min_height, max_height), "
                    f"got {self.cfg.robot_base_height_range}"
                )
            if max_height == min_height:
                heights = torch.full((len(env_ids),), min_height, device=self.device, dtype=torch.float32)
            else:
                heights = min_height + torch.rand(len(env_ids), device=self.device) * (max_height - min_height)
            self._robot_base_heights[env_ids] = heights
            self._support_surface_z_offsets[env_ids] = -heights

        surface_origins = self._surface_origins(env_ids)
        floor_root = self.floor.data.default_root_state.clone().index_select(0, env_ids)
        cube_xy_offset = torch.tensor(self.cfg.support_cube_xy_offset, device=self.device, dtype=torch.float32)
        floor_root[:, 0:2] = self.scene.env_origins.index_select(0, env_ids)[:, 0:2] + cube_xy_offset
        floor_root[:, 2] = surface_origins[:, 2] + TABLE_HEIGHT - FLOOR_THICKNESS / 2.0
        floor_root[:, 3:7] = torch.tensor(self.cfg.floor.init_state.rot, device=self.device, dtype=torch.float32)
        self.floor.write_root_pose_to_sim(floor_root[:, :7], env_ids=env_ids)
        self.floor.write_root_velocity_to_sim(torch.zeros((len(env_ids), 6), device=self.device), env_ids=env_ids)
        self.floor.reset(env_ids)

    def _spawn_angle_range(self, sector_index: int = 0, num_sectors: int = 1) -> tuple[float, float]:
        """Return an angular spawn sector inside the configured Kinova workspace."""
        if num_sectors <= 0:
            raise ValueError(f"num_sectors must be positive, got {num_sectors}")
        if sector_index < 0 or sector_index >= num_sectors:
            raise ValueError(f"sector_index must be in [0, {num_sectors}), got {sector_index}")

        lower = math.radians(float(self.cfg.spawn_angle_lower_deg))
        upper = math.radians(float(self.cfg.spawn_angle_upper_deg))
        if upper <= lower:
            raise ValueError(
                "spawn_angle_upper_deg must be greater than spawn_angle_lower_deg, "
                f"got ({self.cfg.spawn_angle_lower_deg}, {self.cfg.spawn_angle_upper_deg})"
            )
        step = (upper - lower) / float(num_sectors)
        return lower + sector_index * step, lower + (sector_index + 1) * step

    def _surface_origins(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Return env origins with the randomized support-surface z offset applied."""
        self._ensure_support_surface_buffers()
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        origins = self.scene.env_origins.index_select(0, env_ids).clone()
        origins[:, 2] += self._support_surface_z_offsets.index_select(0, env_ids)
        return origins

    def _spawn_z_above_surface(self, env_ids: torch.Tensor, z_height: float | torch.Tensor) -> torch.Tensor:
        """Return object root z values above the sampled support surface."""
        surface_z = self._surface_origins(env_ids)[:, 2:3] + TABLE_HEIGHT
        if isinstance(z_height, torch.Tensor):
            z = z_height.to(device=self.device, dtype=surface_z.dtype)
            if z.ndim == 0:
                z = z.expand(surface_z.shape[0]).unsqueeze(-1)
            elif z.ndim == 1:
                z = z.unsqueeze(-1)
            return surface_z + z
        return surface_z + float(z_height)

    def _sample_object_spawn_height_offset(self, shape: int | tuple[int, ...]) -> torch.Tensor:
        """Sample task-object root z offsets without changing floor height randomization."""
        if isinstance(shape, int):
            sample_shape = (shape,)
        else:
            sample_shape = shape
        lower, upper = self.cfg.random_object_spawn_height_range
        lower = float(lower)
        upper = float(upper)
        if upper < lower:
            raise ValueError(
                "random_object_spawn_height_range must be ordered as (lower, upper), "
                f"got {self.cfg.random_object_spawn_height_range}"
            )
        if upper == lower:
            return torch.full(sample_shape, lower, device=self.device, dtype=torch.float32)
        return lower + torch.rand(sample_shape, device=self.device) * (upper - lower)

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self._prepare_support_surface_reset(env_ids, resample=False)
        super()._reset_idx(env_ids)
