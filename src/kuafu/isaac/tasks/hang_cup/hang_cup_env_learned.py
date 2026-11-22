# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
from kuafu.paths import OBJECT_ROOT

import os
import time
from isaaclab.sensors.frame_transformer.frame_transformer import FrameTransformer
import torch
from pxr import UsdPhysics

from isaacsim.core.utils.stage import get_current_stage

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObjectCfg, RigidObject
from isaaclab.sim import schemas
from isaaclab.sensors import TiledCamera, TiledCameraCfg, CameraCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils import math as math_utils
from isaaclab.sim.schemas import schemas_cfg
from kuafu.isaac.tasks.env_learned import BaseLearnedEnvCfg, BaseLearnedEnv
from kuafu.isaac.tasks.collision_utils import (
    replace_mesh_collision_with_cup_hanger_cylinders,
    replace_mesh_collision_with_mug_cylinders,
)
from kuafu.mpc.simulation.spatial_utils import quat_to_rotation_6d

ISAAC_NUCLEUS_DIR = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac"
TABLE_HEIGHT = 0.1

@configclass
class HangCupEnvCfg(BaseLearnedEnvCfg):
    """Configuration for the hang cup environment."""
    # env
    observation_space: int = 31  # default state dim; use set_obs_mode() for full_state
    obs_mode: str = "state"  # "state" (31) or "full_state" (65)
    obs_dims = {"state": 31, "full_state": 65}

    # Rigid body properties for the object to be placed (marker)
    mug_properties = sim_utils.RigidBodyPropertiesCfg(
        solver_position_iteration_count=32,
        solver_velocity_iteration_count=2,
        max_angular_velocity=1000.0,
        max_linear_velocity=1000.0,
        max_depenetration_velocity=100.0,
        disable_gravity=False,
    )
    
    _mug_path = str(OBJECT_ROOT / 'large_mug/large_mug.usdc')

    # High-friction physics material for mug so gripper fingers can hold it.
    mug_physics_material = sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="max",
        restitution_combine_mode="multiply",
        static_friction=3.0,
        dynamic_friction=2.0,
        restitution=0.0,
    )

    mug = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Mug",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.4, 0.2, TABLE_HEIGHT+0.01), rot=(0, 0, 0, 1)
        ),
        spawn=UsdFileCfg(
            usd_path=_mug_path,
            scale=(1,1,1),
            rigid_props=mug_properties,
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.0001,  # Very small offset to allow gripper to get close
                rest_offset=0.0,
            ),
            semantic_tags=[("class", "mug")],
        ),
    )

    # Cup holder - static/kinematic object (will not move during simulation)
    # Note: Path is constructed relative to IsaacLab workspace root
    _cup_hanger_path = str(OBJECT_ROOT / 'cup_hanger/cup_hanger.usdc')

    cup_hanger = RigidObjectCfg(
        prim_path="/World/envs/env_.*/CupHolder",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.4, -0.2, TABLE_HEIGHT), rot=(1, 0, 0, 0)
        ),
        spawn=UsdFileCfg(
            usd_path=_cup_hanger_path,
            scale=(1,1,1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,  # Makes the object static - it will not move during simulation
                disable_gravity=True,  # Additional safeguard: disable gravity (though kinematic objects ignore it anyway)
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.0001,  # Very small offset to allow gripper to get close
                rest_offset=0.0,
            ),
        ),
    )

    # Low-friction physics material for cup_hanger to allow smooth sliding
    cup_hanger_physics_material = sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=0.01,  # Very low friction for smooth sliding
        dynamic_friction=0.01,  # Very low friction for smooth sliding
        restitution=0.0,
    )

    table = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Table",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.0, TABLE_HEIGHT/2), rot=(1.0, 0, 0, 0.0)),
        spawn=sim_utils.CuboidCfg(
            size=(0.4, 0.8, TABLE_HEIGHT),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
        ),
    )

    marker_cfg_mug_handle_hole = FRAME_MARKER_CFG.copy()
    marker_cfg_mug_handle_hole.markers["frame"].scale = (0.05, 0.05, 0.05)
    marker_cfg_mug_handle_hole.prim_path = "/Visuals/mug_handle_hole_frame"

    mug_handle_hole_frame = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/panda_link0",
        debug_vis=False,
        visualizer_cfg=marker_cfg_mug_handle_hole,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Mug",
                name="Mug_handle_hole",
                offset=OffsetCfg(pos=(0.06, 0.00, 0.05)),  # Smaller offset for the handle hole
            )
        ],
    )

    marker_cfg_mug_center = FRAME_MARKER_CFG.copy()
    marker_cfg_mug_center.markers["frame"].scale = (0.05, 0.05, 0.05)
    marker_cfg_mug_center.prim_path = "/Visuals/mug_center_frame"

    mug_center_frame = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/panda_link0",
        debug_vis=False,
        visualizer_cfg=marker_cfg_mug_center,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Mug",
                name="Mug_center",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.04)),  # Center of cup body - for pouring and upright detection
            )
        ],
    )

    marker_cfg_mug_center = FRAME_MARKER_CFG.copy()
    marker_cfg_mug_center.markers["frame"].scale = (0.05, 0.05, 0.05)
    marker_cfg_mug_center.prim_path = "/Visuals/mug_rim_frame_0"

    mug_rim_frame_0 = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/panda_link0",
        debug_vis=False,
        visualizer_cfg=marker_cfg_mug_center,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Mug",
                name="mug_rim_frame_0",
                offset=OffsetCfg(pos=(0.0, 0.055, 0.075)),  # Center of cup body - for pouring and upright detection
            )
        ],
    )

    marker_cfg_mug_rim_frame_1 = FRAME_MARKER_CFG.copy()
    marker_cfg_mug_center.markers["frame"].scale = (0.05, 0.05, 0.05)
    marker_cfg_mug_rim_frame_1.prim_path = "/Visuals/mug_rim_frame_1"

    mug_rim_frame_1 = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/panda_link0",
        debug_vis=False,
        visualizer_cfg=marker_cfg_mug_center,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Mug",
                name="mug_rim_frame_1",
                offset=OffsetCfg(pos=(0.0, -0.055, 0.075)),  # Center of cup body - for pouring and upright detection
            )
        ],
    )

    # Cup holder branch frames - 4 frames to signal the cup holder branches
    marker_cfg_cup_hanger_branch_0 = FRAME_MARKER_CFG.copy()
    marker_cfg_cup_hanger_branch_0.markers["frame"].scale = (0.05, 0.05, 0.05)
    marker_cfg_cup_hanger_branch_0.prim_path = "/Visuals/cup_hanger_branch_0_frame"

    cup_hanger_branch_0_frame = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/panda_link0",
        debug_vis=False,
        visualizer_cfg=marker_cfg_cup_hanger_branch_0,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/CupHolder",
                name="CupHolder_branch_0",
                offset=OffsetCfg(
                    pos=(0.0, -0.1, 0.36),
                    rot=(0, 0, 0.173648178, 0.984807753),
                ),
            )
        ],
    )

    marker_cfg_cup_hanger_branch_1 = FRAME_MARKER_CFG.copy()
    marker_cfg_cup_hanger_branch_1.markers["frame"].scale = (0.05, 0.05, 0.05)
    marker_cfg_cup_hanger_branch_1.prim_path = "/Visuals/cup_hanger_branch_1_tip_frame"

    cup_hanger_branch_1_tip_frame = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/panda_link0",
        debug_vis=False,
        visualizer_cfg=marker_cfg_cup_hanger_branch_1,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/CupHolder",
                name="CupHolder_branch_1",
                offset=OffsetCfg(
                    pos=(0.0, 0.1, 0.36),
                    rot=(0.984807753, 0.173648178, 0, 0),
                ),
            )
        ],
    )

    marker_cfg_cup_hanger_branch_2 = FRAME_MARKER_CFG.copy()
    marker_cfg_cup_hanger_branch_2.markers["frame"].scale = (0.05, 0.05, 0.05)
    marker_cfg_cup_hanger_branch_2.prim_path = "/Visuals/cup_hanger_branch_2_frame"

    cup_hanger_branch_2_frame = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/panda_link0",
        debug_vis=False,
        visualizer_cfg=marker_cfg_cup_hanger_branch_2,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/CupHolder",
                name="CupHolder_branch_2",
                offset=OffsetCfg(
                    pos=(0.1, 0, 0.28),
                    rot=(0.696364240, 0.122787804, -0.122787804, -0.696364240),
                ),
            )
        ],
    )

    marker_cfg_cup_hanger_branch_3 = FRAME_MARKER_CFG.copy()
    marker_cfg_cup_hanger_branch_3.markers["frame"].scale = (0.05, 0.05, 0.05)
    marker_cfg_cup_hanger_branch_3.prim_path = "/Visuals/cup_hanger_branch_3_frame"

    cup_hanger_branch_3_frame = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/panda_link0",
        debug_vis=False,
        visualizer_cfg=marker_cfg_cup_hanger_branch_3,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/CupHolder",
                name="CupHolder_branch_3",
                offset=OffsetCfg(
                    pos=(-0.1, 0, 0.28),
                    rot=(0.696364240, 0.122787804, 0.122787804, 0.696364240),
                ),
            )
        ],
    )

    detectable_objects = {
        # "mug": (
        #     "base frame of mug, size: (0.1, 0.13, 0.1)",
        #     "/World/envs/env_*/Mug",
        # ),
        "mug_center_frame": (
            "center frame of large mug, the opening axis is along object frame + z, width too large for gripper, size: (0.13, 0.1, 0.1)",
            "/World/envs/env_*/mug_center_frame",
        ),
        "mug_rim_frame_0": (
            "rim frame 0 of mug, graspable edge, rim edge is tangent to object frame + x or -x",
            "/World/envs/env_*/mug_rim_frame_0",
        ),
        "mug_rim_frame_1": (
            "rim frame 1 of mug, graspable edge, rim edge is tangent to object frame + x or -x",
            "/World/envs/env_*/mug_rim_frame_1",
        ),
        "cup_hanger": (
            "base of cup holder, hanger with 4 tilted branches, size: (0.3, 0.3, 0.35)",
            "/World/envs/env_*/CupHolder",
        ),
        "table": (
            "center frame of table, size: (0.3, 0.9, 0.4)",
            "/World/envs/env_*/Table",
        ),
        "mug_handle_hole_frame": (
            "center of handle opening of mug; handle axis -x; hole/opening axis is ±y (bidirectional); not a graspable frame",
            "/Visuals/mug_handle_hole_frame",
        ),
        "cup_hanger_branch_1_tip_frame": (
            "tip frame of branch 1; branch axis is +y (base->tip, outward); engagement/hooking moves along -y (tip->base, inward), length is 0.05, thickness is 0.01",
            "/Visuals/cup_hanger_branch_1_tip_frame",
        ),
        "panda_link4": (
            "frame of robot link 4",
            "/Visuals/panda_link4_frame",
        ),
        "panda_fingertip": (
            "frame of robot tool frame",
            "/Visuals/panda_hand_frame",
        ),
    }
    # Override mpc_config with task-specific path
    kuafu_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    script_dir = os.path.dirname(os.path.dirname(os.path.dirname(kuafu_dir)))
    mpc_config = {
        "controller_name": "FrankaIKController",
        "device": "cuda",
        "ee_frame": "panda_fingertip",
        "use_per_env_weight": False,
        "load_program": f"{script_dir}/resources/learned_programs/manipulation-hang-cup/done",
        "initial_residual": f"{script_dir}/src/kuafu/skills/constraints/rewards.py",
        "initial_predicate": f"{script_dir}/src/kuafu/skills/constraints/predicates.py",
    }
    del script_dir

    task_description: str = "hang the cup on the cup hanger"

class HangCupEnv(BaseLearnedEnv):
    """Environment for hanging a cup on a cup hanger."""
    cfg: HangCupEnvCfg

    def _setup_scene(self):
        # Call base class setup first (sets up robot, ee_frame, l4_frame, terrain)
        super()._setup_scene()
        
        # Spawn mug USD file first, then apply RigidBodyAPI and collision approximation before creating RigidObject
        # This fixes the missing RigidBodyAPI error and ensures proper collision detection
        stage = get_current_stage()
        mug_cfg = self.cfg.mug
        # Spawn the USD file if spawn config exists
        if mug_cfg.spawn is not None:
            mug_cfg.spawn.func(
                mug_cfg.prim_path,
                mug_cfg.spawn,
                translation=mug_cfg.init_state.pos,
                orientation=mug_cfg.init_state.rot,
            )
        # Apply RigidBodyAPI and collision approximation to all matching prims
        if mug_cfg.spawn is not None:
            matching_prims = sim_utils.find_matching_prims(mug_cfg.prim_path)
            for prim in matching_prims:
                # Always apply RigidBodyAPI (it's safe to apply even if already present)
                if hasattr(mug_cfg.spawn, 'rigid_props') and mug_cfg.spawn.rigid_props is not None:
                    schemas.define_rigid_body_properties(prim.GetPath().pathString, mug_cfg.spawn.rigid_props)
                
                # Apply mass properties if specified
                if hasattr(mug_cfg.spawn, 'mass_props') and mug_cfg.spawn.mass_props is not None:
                    schemas.define_mass_properties(prim.GetPath().pathString, mug_cfg.spawn.mass_props)
                
                # Create physics material for this mug instance
                material_path = f"{prim.GetPath().pathString}/physicsMaterial"
                self.cfg.mug_physics_material.func(material_path, self.cfg.mug_physics_material)

                replace_mesh_collision_with_mug_cylinders(
                    prim,
                    stage=stage,
                    collision_props=getattr(mug_cfg.spawn, "collision_props", None),
                    material_path=material_path,
                )
        
        # Spawn cup hanger USD file, then apply RigidBodyAPI and collision approximation before creating RigidObject
        cup_hanger_cfg = self.cfg.cup_hanger
        if cup_hanger_cfg.spawn is not None:
            cup_hanger_cfg.spawn.func(
                cup_hanger_cfg.prim_path,
                cup_hanger_cfg.spawn,
                translation=cup_hanger_cfg.init_state.pos,
                orientation=cup_hanger_cfg.init_state.rot,
            )
        # Apply RigidBodyAPI and collision approximation to all matching prims
        if cup_hanger_cfg.spawn is not None:
            matching_prims = sim_utils.find_matching_prims(cup_hanger_cfg.prim_path)
            for prim in matching_prims:
                # Apply RigidBodyAPI if not already present
                if not UsdPhysics.RigidBodyAPI(prim):
                    if hasattr(cup_hanger_cfg.spawn, 'rigid_props') and cup_hanger_cfg.spawn.rigid_props is not None:
                        schemas.define_rigid_body_properties(prim.GetPath().pathString, cup_hanger_cfg.spawn.rigid_props)
                
                # Apply mass properties if specified
                if hasattr(cup_hanger_cfg.spawn, 'mass_props') and cup_hanger_cfg.spawn.mass_props is not None:
                    schemas.define_mass_properties(prim.GetPath().pathString, cup_hanger_cfg.spawn.mass_props)
                
                # Create physics material for this cup_hanger instance
                material_path = f"{prim.GetPath().pathString}/physicsMaterial"
                self.cfg.cup_hanger_physics_material.func(material_path, self.cfg.cup_hanger_physics_material)

                replace_mesh_collision_with_cup_hanger_cylinders(
                    prim,
                    stage=stage,
                    collision_props=getattr(cup_hanger_cfg.spawn, "collision_props", None),
                    material_path=material_path,
                )
                
        
        # Now create RigidObjects (they will find the RigidBodyAPI we just applied)
        self.mug = RigidObject(self.cfg.mug)
        self.cup_hanger = RigidObject(self.cfg.cup_hanger)

        # Create the table as a RigidObject (static/kinematic)
        self.table = RigidObject(self.cfg.table)

        # Add to scene
        self.scene.rigid_objects["mug"] = self.mug
        self.scene.rigid_objects["cup_hanger"] = self.cup_hanger
        self.scene.rigid_objects["table"] = self.table

        # Create and register task-specific frame transformers
        self.mug_handle_hole_frame = FrameTransformer(self.cfg.mug_handle_hole_frame)
        self.mug_center_frame = FrameTransformer(self.cfg.mug_center_frame)
        self.mug_rim_frame_0 = FrameTransformer(self.cfg.mug_rim_frame_0)
        self.mug_rim_frame_1 = FrameTransformer(self.cfg.mug_rim_frame_1)
        self.cup_hanger_branch_0_frame = FrameTransformer(self.cfg.cup_hanger_branch_0_frame)
        self.cup_hanger_branch_1_tip_frame = FrameTransformer(self.cfg.cup_hanger_branch_1_tip_frame)
        self.cup_hanger_branch_2_frame = FrameTransformer(self.cfg.cup_hanger_branch_2_frame)
        self.cup_hanger_branch_3_frame = FrameTransformer(self.cfg.cup_hanger_branch_3_frame)
        
        self.scene.sensors["mug_handle_hole_frame"] = self.mug_handle_hole_frame
        self.scene.sensors["mug_center_frame"] = self.mug_center_frame
        self.scene.sensors["cup_hanger_branch_0_frame"] = self.cup_hanger_branch_0_frame
        self.scene.sensors["cup_hanger_branch_1_tip_frame"] = self.cup_hanger_branch_1_tip_frame
        self.scene.sensors["cup_hanger_branch_2_frame"] = self.cup_hanger_branch_2_frame
        self.scene.sensors["cup_hanger_branch_3_frame"] = self.cup_hanger_branch_3_frame
        self.scene.sensors["mug_rim_frame_0"] = self.mug_rim_frame_0
        self.scene.sensors["mug_rim_frame_1"] = self.mug_rim_frame_1

        # Clone and replicate (base class already did this, but we need to do it again after adding new objects)
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def close(self):
        super().close()
        
    # pre-physics step calls

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        if self.recorder_manager is not None and len(env_ids) > 0:
            self.recorder_manager.record_pre_reset(env_ids)

        # Reset hanger and mug positions
        num_envs = len(env_ids)
        cup_offset = torch.rand(num_envs, 3, device=self.device) * torch.tensor(
            [0.1, 0.1, 0.0], device=self.device
        )
        env_ids_tensor = env_ids.to(device=self.device, dtype=torch.long)
        env_origins = self.scene.env_origins.index_select(0, env_ids_tensor)

        root = self.mug.data.default_root_state.clone()
        root = root.index_select(0, env_ids_tensor)
        root[:, :3] += env_origins + cup_offset

        hanger_root = self.cup_hanger.data.default_root_state.clone()
        hanger_root = hanger_root.index_select(0, env_ids_tensor)
        hanger_root[:, :3] += env_origins

        zeros = torch.zeros(num_envs, device=self.device)
        mug_delta_xy = root[:, :2] - hanger_root[:, :2]
        mug_facing_yaw = torch.atan2(mug_delta_xy[:, 1], mug_delta_xy[:, 0])
        yaw_offset = (
            (torch.rand(num_envs, device=self.device) * 2.0 - 1.0)
            * (torch.pi / 2.0)
        )
        hanger_yaw = mug_facing_yaw - torch.pi / 2.0 + yaw_offset
        hanger_root[:, 3:7] = math_utils.quat_from_euler_xyz(zeros, zeros, hanger_yaw)
        self.cup_hanger.write_root_pose_to_sim(hanger_root[:, :7], env_ids=env_ids)
        self.cup_hanger.write_root_velocity_to_sim(
            torch.zeros((num_envs, 6), device=self.device), env_ids=env_ids
        )
        self.cup_hanger.reset(env_ids)

        # write pose and velocities to the sim, then reset buffers
        eulers = torch.zeros(
            (num_envs, 3), device=self.device
        )  # or your real angles
        roll = eulers[:, 0]
        pitch = eulers[:, 1]
        yaw = eulers[:, 2] + torch.rand(num_envs, device=self.device) * torch.pi

        quat = math_utils.quat_from_euler_xyz(roll, pitch, yaw)
        root[:, 3:7] = quat  # quaternions are WXYZ in Isaac Lab

        # write pose and velocities to the sim, then reset buffers
        self.mug.write_root_pose_to_sim(root[:, :7], env_ids=env_ids)
        self.mug.write_root_velocity_to_sim(
            torch.zeros((num_envs, 6), device=self.device), env_ids=env_ids
        )
        self.mug.reset(env_ids)
        super()._reset_idx(env_ids)

    def _get_observations(self) -> dict:
        self.update_data_in_controller()

        api = self.program_runtime.controller_api_wrapper
        agent = api.reference_agent
        if agent is None:
            raise RuntimeError("controller_api_wrapper.reference_agent is not set")

        mug_pos = api.get_object_xyz_position("mug_center_frame")  # (N, 3)
        hanger_pos = api.get_object_xyz_position("cup_hanger_branch_1_tip_frame")  # (N, 3)
        fingertip_pos = api.lookup_robot_frame_position("panda_fingertip")  # (N, 3)
        link4_pos = api.lookup_robot_frame_position("panda_link4")  # (N, 3)

        mug_rot6d = quat_to_rotation_6d(api.get_object_wxyz_quaternion("mug_center_frame"))  # (N, 6)
        hanger_rot6d = quat_to_rotation_6d(api.get_object_wxyz_quaternion("cup_hanger_branch_1_tip_frame"))  # (N, 6)
        fingertip_rot6d = quat_to_rotation_6d(api.lookup_robot_frame_rotation("panda_fingertip"))  # (N, 6)
        link4_rot6d = quat_to_rotation_6d(api.lookup_robot_frame_rotation("panda_link4"))  # (N, 6)

        joint_pos = api.joint_dict[agent]["joint_position"]  # (N, 9)
        joint_vel = api.joint_dict[agent]["joint_velocity"]  # (N, 9)
        joint_effort = self._get_normalized_joint_effort(api, agent)  # (N, 9)
        gripper_effort = joint_effort[:, 7:]  # (N, 2)
        gripper_open = api.gripper_openness().unsqueeze(-1)  # (N, 1)
        stage = self.current_stage  # (N, 1)

        def _f(*tensors: torch.Tensor) -> torch.Tensor:
            return torch.cat(
                [t.to(device=self.device, dtype=torch.float32) for t in tensors],
                dim=-1,
            )

        policy_obs = _f(
            mug_pos,
            hanger_pos,
            fingertip_pos,  # 9
            mug_rot6d,
            hanger_rot6d,
            fingertip_rot6d,  # 18
            gripper_open,
            gripper_effort,  # 3
            stage,  # 1
        )  # total: 31

        full_state_obs = _f(
            mug_pos,
            hanger_pos,
            fingertip_pos,
            link4_pos,  # 12
            mug_rot6d,
            hanger_rot6d,
            fingertip_rot6d,
            link4_rot6d,  # 24
            joint_pos,
            joint_vel,
            joint_effort,  # 27
            gripper_open,  # 1
            stage,  # 1
        )  # total: 65

        obs_dim = int(self.cfg.observation_space)
        policy_obs_to_return = policy_obs if self.cfg.obs_mode == "state" else full_state_obs
        if policy_obs_to_return.shape[-1] != obs_dim:
            raise RuntimeError(
                f"Observation dim mismatch: got {policy_obs_to_return.shape[-1]}, expected {obs_dim}"
            )

        # Convert stage to float for calculation and log to extras
        # "episode/" prefix groups it nicely in WandB
        self.extras["episode/mean_stage"] = torch.mean(self.current_stage.float())
        self.extras["episode/max_stage"] = torch.max(self.current_stage.float())

        return {
            "policy": policy_obs_to_return,
            "full_state": full_state_obs,
            "state": policy_obs,
        }


    def _external_get_dones(self):
        mug_pos = self.mug.data.root_state_w[:, :3]
        cup_hanger_pos = self.cup_hanger.data.root_state_w[:, :3]

        # Time out
        # Goal reached: cup is hanging on the cup holder
        # Calculate distance on xy plane (ignoring z)
        distance_to_holder_xy = torch.norm(mug_pos[:, :2] - cup_hanger_pos[:, :2], dim=1)
        mug_pos_z = mug_pos[:, 2]
        
        # Check if gripper is open (finger joints are the last two joints)
        # For Panda gripper, open position is around 0.04, closed is 0.0
        finger_joint_1 = self._robot.data.joint_pos[:, -2]  # Second to last joint
        finger_joint_2 = self._robot.data.joint_pos[:, -1]  # Last joint
        gripper_open = torch.logical_and(
            finger_joint_1 > 0.03,  # Threshold for open gripper
            finger_joint_2 > 0.03   # Threshold for open gripper
        )
        
        # Check if cup is stable (very low velocity)
        # root_state_w format: [pos(3), quat(4), lin_vel(3), ang_vel(3)]
        mug_lin_vel = self.mug.data.root_state_w[:, 7:10]  # Linear velocity
        mug_ang_vel = self.mug.data.root_state_w[:, 10:13]  # Angular velocity
        mug_lin_vel_norm = torch.norm(mug_lin_vel, dim=1)
        mug_ang_vel_norm = torch.norm(mug_ang_vel, dim=1)
        cup_stable = torch.logical_and(
            mug_lin_vel_norm < 0.01,  # Very low linear velocity (1 cm/s)
            mug_ang_vel_norm < 0.1   # Very low angular velocity (0.1 rad/s)
        )
        
        # Cup should be close to holder on xy plane (within 5 cm), at appropriate height (higher than 0.1 m),
        # gripper should be open, and cup should be stable
        goal_reached = torch.logical_and(
            distance_to_holder_xy < 0.05,  # Within 5 cm on xy plane
            torch.logical_and(
                mug_pos_z > 0.1,  # Cup center height is higher than 0.1 m
                torch.logical_and(
                    gripper_open,  # Gripper is open
                    cup_stable  # Cup is stable (very low velocity)
                )
            )
        )

        terminated = goal_reached

        return terminated
