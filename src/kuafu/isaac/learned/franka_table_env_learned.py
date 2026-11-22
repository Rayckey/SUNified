# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
from kuafu.paths import OBJECT_ROOT
from dataclasses import MISSING, field
import os
import torch
from omegaconf import OmegaConf
from isaaclab.sensors.frame_transformer.frame_transformer import FrameTransformer
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.utils.torch.transformations import tf_combine, tf_inverse, tf_vector
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema
import isaaclab.sim as sim_utils
from isaaclab.sim import schemas
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObjectCfg, RigidObject
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import sample_uniform, quat_from_euler_xyz
from isaaclab.sensors import TiledCamera, TiledCameraCfg, CameraCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from kuafu.isaac.learned.fr3_finray_env_learned import BaseLearnedEnvCfg, BaseLearnedEnv

TABLE_HEIGHT = 0.0
FLOOR_THICKNESS = 0.05

@configclass
class FrankaTableEnvLearnedCfg(BaseLearnedEnvCfg):
    observation_space = 23
    table_path = str(OBJECT_ROOT / 'table/table_flattened_instanceable_meshes.usd')
    table = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Table",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0), rot=(1, 0, 0, 0)
        ),
        spawn=UsdFileCfg(
            usd_path=table_path,
            scale=(1, 1, 1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            # Disable collisions on the visual table; we provide a simple
            # cuboid collider as the floor instead.
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=False,
            ),
        ),
    )

    # Simple cuboid floor collider (4 m x 4 m) centered at the origin.
    # Acts as the contact surface instead of the complex table mesh.
    hide_table_visuals: bool = False

    floor = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Floor",
        init_state=RigidObjectCfg.InitialStateCfg(
            # Place so that the top of the floor aligns with TABLE_HEIGHT.
            pos=(0.0, 0.0, TABLE_HEIGHT - FLOOR_THICKNESS / 2.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.CuboidCfg(
            size=(4.0, 4.0, FLOOR_THICKNESS),
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
        ),
    )


class FrankaTableEnvLearned(BaseLearnedEnv):
    """Franka table environment: robot and table only, with placeholder rewards/dones."""

    def __init__(self, cfg: FrankaTableEnvLearnedCfg, render_mode: str | None = None, **kwargs):
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
                f"[FrankaTableEnvLearned] WARNING: requested hiding {label} visuals but no prims matched "
                f"{prim_path}",
                flush=True,
            )
        return hidden_count

    def _hide_static_surface_visual_prims(self) -> None:
        """Hide non-robot static surface visuals while leaving physics objects intact."""
        self._hide_matching_prim_trees(self.cfg.floor.prim_path, "floor")
        if getattr(self.cfg, "hide_table_visuals", False):
            self._hide_matching_prim_trees(self.cfg.table.prim_path, "table", warn_if_empty=True)

    def _setup_scene(self):
        super()._setup_scene()

        table_cfg = self.cfg.table
        # Spawn the visual table USD and ensure it has RigidBodyAPI so that
        # RigidObject can bind to it, even though collisions are disabled.
        if table_cfg.spawn is not None:
            table_cfg.spawn.func(
                table_cfg.prim_path,
                table_cfg.spawn,
                translation=table_cfg.init_state.pos,
                orientation=table_cfg.init_state.rot,
            )
            matching_prims = sim_utils.find_matching_prims(table_cfg.prim_path)
            for prim in matching_prims:
                if hasattr(table_cfg.spawn, "rigid_props") and table_cfg.spawn.rigid_props is not None:
                    schemas.define_rigid_body_properties(prim.GetPath().pathString, table_cfg.spawn.rigid_props)
                if hasattr(table_cfg.spawn, "mass_props") and table_cfg.spawn.mass_props is not None:
                    schemas.define_mass_properties(prim.GetPath().pathString, table_cfg.spawn.mass_props)

        self.table = RigidObject(table_cfg)
        self.scene.rigid_objects["table"] = self.table

        # Create the simple cuboid floor collider (RigidObject will spawn it)
        floor_cfg = self.cfg.floor
        self.floor = RigidObject(floor_cfg)
        self.scene.rigid_objects["floor"] = self.floor

        # Hide render geometry for invisible support surfaces. Reapplied after cloning in child envs.
        self._hide_static_surface_visual_prims()
