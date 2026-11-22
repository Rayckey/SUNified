# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
from kuafu.paths import OBJECT_ROOT

import os
import torch
from collections.abc import Sequence
from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics
from isaaclab.sim import schemas
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.utils.torch.transformations import tf_combine, tf_inverse

from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, RenderCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from kuafu.isaac.base.manipulation.panda_manipulation_env import PandaManipulationEnv, PandaManipulationEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg,  AssetBaseCfg, AssetBase
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sensors import (
    FrameTransformer,
    FrameTransformerCfg,
)
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.math import sample_uniform
from isaaclab.utils import math as math_utils


TABLE_HEIGHT = 0.2
@configclass
class PenCupEnvCfg(PandaManipulationEnvCfg):

    observation_space = 23

    # Scene settings - override env_spacing
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4, env_spacing=3.0, replicate_physics=True
    )

    observation_space = 9
    action_space = 9
    use_debug_action_mode: bool = False

    def __post_init__(self):
        """Post-initialization to set robot initial state and call parent's __post_init__."""
        # Call parent's __post_init__ first to set up robot config
        super().__post_init__()
        # Override only the initial state
        self.robot.init_state = ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 1.0,
                "panda_joint2": -0.569,
                "panda_joint3": 0.0,
                "panda_joint4": -2.810,
                "panda_joint5": 0.0,
                "panda_joint6": 3.037,
                "panda_joint7": 0.741,
                "panda_finger_joint.*": 0.04,
            },
            pos=(0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        )

    # Rigid body properties for the object to be placed (marker)
    mug_properties = sim_utils.RigidBodyPropertiesCfg(
        solver_position_iteration_count=32,
        solver_velocity_iteration_count=2,
        max_angular_velocity=1000.0,
        max_linear_velocity=1000.0,
        max_depenetration_velocity=100.0,
        disable_gravity=False,
    )

    # Paths for marker and pen holder assets
    _marker_path = str(OBJECT_ROOT / 'blue_marker/blue_marker.usdc')

    marker = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Marker",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.4, 0.1, TABLE_HEIGHT+0.01), rot=(1, 0, 0, 0)
        ),
        spawn=UsdFileCfg(
            usd_path=_marker_path,
            scale=(1,1,1),
            rigid_props=mug_properties,
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            semantic_tags=[("class", "marker")],
        ),
    )

    # Pen holder - static/kinematic object (will not move during simulation)
    # Note: Path is constructed relative to IsaacLab workspace root
    _pen_holder_path = str(OBJECT_ROOT / 'pen_holder/pen_holder.usdc')
    pen_holder = RigidObjectCfg(
        prim_path="/World/envs/env_.*/PenHolder",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.4, -0.1, TABLE_HEIGHT+0.06), rot=(1, 0, 0, 0)
        ),
        spawn=UsdFileCfg(
            usd_path=_pen_holder_path,
            scale=(1,1,1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,  # Makes the object static - it will not move during simulation
                disable_gravity=True,  # Additional safeguard: disable gravity (though kinematic objects ignore it anyway)
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
    )

    table = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Table",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, TABLE_HEIGHT/2), rot=(1.0, 0, 0, 0.0)),
        spawn=sim_utils.CuboidCfg(
            size=(0.3, 0.9, TABLE_HEIGHT),
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

    # Listens to the required transforms
    marker_cfg = FRAME_MARKER_CFG.copy()
    marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    marker_cfg.prim_path = "/Visuals/panda_hand_frame"

    marker2_cfg = FRAME_MARKER_CFG.copy()
    marker2_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    marker2_cfg.prim_path = "/Visuals/panda_link4_frame"

    ee_frame = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/panda_link0",
        debug_vis=False,
        visualizer_cfg=marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot/panda_hand",
                name="panda_fingertip",
                offset=OffsetCfg(
                    pos=[0.0, 0.0, 0.1034],
                ),
            )
        ],
    )

    l4_frame = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/panda_link0",
        debug_vis=False,
        visualizer_cfg=marker2_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot/panda_link4",
                name="panda_link4",
            )
        ],
    )


class PenCupEnv(PandaManipulationEnv):
    cfg: PenCupEnvCfg

    def __init__(
        self,
        cfg: PenCupEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ):

        # initialize the base class to setup the scene.
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

        print("[INFO]: Completed setting up the environment...")

    def close(self):
        super().close()

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        super()._reset_idx(env_ids)

        if getattr(self.cfg, "disable_reset_randomization", False):
            cup_offset = torch.zeros((len(env_ids), 3), device=self.device)
        else:
            cup_offset = torch.rand(len(env_ids), 3).to(self.device) * torch.tensor([0.1, 0.1, 0.0]).to(self.device)

        # Reset marker position
        if torch.is_tensor(env_ids):
            env_ids_tensor = env_ids.to(device=self.device, dtype=torch.long)
        else:
            env_ids_tensor = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        root = self.marker.data.default_root_state.clone()
        root = root.index_select(0, env_ids_tensor)

        # write pose and velocities to the sim, then reset buffers
        root[:, :3] += self.scene.env_origins.index_select(0, env_ids_tensor) + cup_offset
        eulers = torch.zeros(
            (len(env_ids), 3), device=self.device
        )  # or your real angles
        roll = eulers[:, 0]
        pitch = eulers[:, 1]
        if getattr(self.cfg, "disable_reset_randomization", False):
            yaw = eulers[:, 2]
        else:
            yaw = eulers[:, 2] + torch.rand(len(env_ids), device=self.device) * torch.pi

        quat = math_utils.quat_from_euler_xyz(roll, pitch, yaw)
        root[:, 3:7] = quat  # quaternions are WXYZ in Isaac Lab

        # write pose and velocities to the sim, then reset buffers
        self.marker.write_root_pose_to_sim(root[:, :7], env_ids=env_ids)
        self.marker.write_root_velocity_to_sim(
            torch.zeros((len(env_ids), 6), device=self.device), env_ids=env_ids
        )
        self.marker.reset(env_ids)

        # Reset robot using base class helper
        self._reset_robot(env_ids_tensor)

    def _setup_scene(self):

        self.robot = Articulation(self.cfg.robot)

        # Spawn marker USD file first, then apply RigidBodyAPI and collision approximation before creating RigidObject
        # This fixes the missing RigidBodyAPI error and ensures proper collision detection
        stage = get_current_stage()
        mug_cfg = self.cfg.marker
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

                # Add collision approximation to all mesh prims inside the mug (recursively)
                for child_prim in Usd.PrimRange(prim):
                    if child_prim.IsA(UsdGeom.Mesh):
                        # Apply collision approximation to each mesh
                        mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(child_prim)
                        mesh_collision_api.GetApproximationAttr().Set(UsdPhysics.Tokens.convexDecomposition)

                        # Fine convex decomposition params (PhysX)
                        cd_api = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(child_prim)
                        cd_api.CreateVoxelResolutionAttr().Set(5_000_000)
                        cd_api.CreateMaxConvexHullsAttr().Set(128)
                        cd_api.CreateErrorPercentageAttr().Set(1.0)
                        cd_api.CreateShrinkWrapAttr().Set(True)
                        cd_api.CreateHullVertexLimitAttr().Set(64)
                        cd_api.CreateMinThicknessAttr().Set(0.0005)
                        # Apply collision properties
                        if hasattr(mug_cfg.spawn, 'collision_props') and mug_cfg.spawn.collision_props is not None:
                            schemas.define_collision_properties(child_prim.GetPath().pathString, mug_cfg.spawn.collision_props, stage)

        # Spawn pen holder USD file, then apply RigidBodyAPI and collision approximation before creating RigidObject
        cup_hanger_cfg = self.cfg.pen_holder
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

                # Add collision approximation to all mesh prims inside the cup holder (recursively)
                for child_prim in Usd.PrimRange(prim):
                    if child_prim.IsA(UsdGeom.Mesh):
                        # Apply collision approximation to each mesh
                        mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(child_prim)
                        mesh_collision_api.GetApproximationAttr().Set(UsdPhysics.Tokens.convexDecomposition)

                        # Fine convex decomposition params (PhysX)
                        cd_api = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(child_prim)
                        cd_api.CreateVoxelResolutionAttr().Set(5_000_000)
                        cd_api.CreateMaxConvexHullsAttr().Set(128)
                        cd_api.CreateErrorPercentageAttr().Set(1.0)
                        cd_api.CreateShrinkWrapAttr().Set(True)
                        cd_api.CreateHullVertexLimitAttr().Set(64)
                        cd_api.CreateMinThicknessAttr().Set(0.0005)
                        # Apply collision properties
                        if hasattr(cup_hanger_cfg.spawn, 'collision_props') and cup_hanger_cfg.spawn.collision_props is not None:
                            schemas.define_collision_properties(child_prim.GetPath().pathString, cup_hanger_cfg.spawn.collision_props, stage)

        # Now create RigidObjects (they will find the RigidBodyAPI we just applied)
        self.marker = RigidObject(self.cfg.marker)
        self.pen_holder = RigidObject(self.cfg.pen_holder)

        # Create the table as a RigidObject (static/kinematic)
        self.table = RigidObject(self.cfg.table)

        self.ee_frame = FrameTransformer(self.cfg.ee_frame)
        self.l4_frame = FrameTransformer(self.cfg.l4_frame)

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # Add objects to scene
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["marker"] = self.marker
        self.scene.rigid_objects["pen_holder"] = self.pen_holder
        self.scene.rigid_objects["table"] = self.table

        self.scene.sensors["panda_fingertip"] = self.ee_frame
        self.scene.sensors["panda_link4"] = self.l4_frame

        self.scene.clone_environments(copy_from_source=False)

        # Note: Cup holder is kinematic (static) - no need to set velocities as kinematic objects
        # are automatically static and don't respond to forces

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=1000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])


    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        if getattr(self.cfg, "disable_terminations", False):
            done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            return done, done

        marker_pos = self.marker.data.root_state_w[:, :3]
        pen_holder_pos = self.pen_holder.data.root_state_w[:, :3]

        # Time out
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # Goal reached: marker is placed in the pen holder
        # Calculate distance on xy plane (ignoring z)
        distance_to_holder_xy = torch.norm(marker_pos[:, :2] - pen_holder_pos[:, :2], dim=1)
        marker_pos_z = marker_pos[:, 2]

        # Check if gripper is open (finger joints are the last two joints)
        # For Panda gripper, open position is around 0.04, closed is 0.0
        finger_joint_1 = self.robot.data.joint_pos[:, -2]  # Second to last joint
        finger_joint_2 = self.robot.data.joint_pos[:, -1]  # Last joint
        gripper_open = torch.logical_and(
            finger_joint_1 > 0.03,  # Threshold for open gripper
            finger_joint_2 > 0.03   # Threshold for open gripper
        )

        # Check if marker is stable (very low velocity)
        # root_state_w format: [pos(3), quat(4), lin_vel(3), ang_vel(3)]
        marker_lin_vel = self.marker.data.root_state_w[:, 7:10]  # Linear velocity
        marker_ang_vel = self.marker.data.root_state_w[:, 10:13]  # Angular velocity
        marker_lin_vel_norm = torch.norm(marker_lin_vel, dim=1)
        marker_ang_vel_norm = torch.norm(marker_ang_vel, dim=1)
        cup_stable = torch.logical_and(
            marker_lin_vel_norm < 0.01,  # Very low linear velocity (1 cm/s)
            marker_ang_vel_norm < 0.1   # Very low angular velocity (0.1 rad/s)
        )

        # Marker should be close to holder on xy plane (within 5 cm), at appropriate height (higher than 0.1 m),
        # gripper should be open, and cup should be stable
        goal_reached = torch.logical_and(
            distance_to_holder_xy < 0.05,  # Within 5 cm on xy plane
            torch.logical_and(
                marker_pos_z > 0.1,  # Marker center height is higher than 0.1 m
                torch.logical_and(
                    gripper_open,  # Gripper is open
                    cup_stable  # Cup is stable (very low velocity)
                )
            )
        )

        terminated = goal_reached

        return terminated, time_out

    def _get_observations(self) -> dict:
        joint_states = self.scene.state["articulation"]["robot"]
        pose = self.scene.state["articulation"]["robot"]["root_pose"]
        marker_pos = self.marker.data.root_state_w[:, :3]
        pen_holder_pos = self.pen_holder.data.root_state_w[:, :3]

        return {
            "joint_states": joint_states,
            "pose": pose,
            "marker_pos": marker_pos,
            "pen_holder_pos": pen_holder_pos,
        }


    def _get_rewards(self) -> torch.Tensor:
        marker_pos = self.marker.data.root_state_w[:, :3]
        pen_holder_pos = self.pen_holder.data.root_state_w[:, :3]

        # Reward for getting the marker close to the pen holder
        distance_to_holder = torch.norm(marker_pos - pen_holder_pos, dim=1)

        # Reward inversely proportional to distance (closer = higher reward)
        # Add small epsilon to avoid division by zero
        distance_reward = 1.0 / (distance_to_holder + 0.1)

        # Additional reward for proper height (marker above holder center)
        marker_pos_z = marker_pos[:, 2]
        holder_pos_z = pen_holder_pos[:, 2]
        height_diff = torch.abs(marker_pos_z - (holder_pos_z + 0.05))  # Target height slightly above holder
        height_reward = torch.exp(-10.0 * height_diff)  # Exponential reward for correct height

        total_reward = self.cfg.dist_reward_scale * distance_reward + self.cfg.rot_reward_scale * height_reward
        return total_reward

    # pre-physics step calls (handled by base class)
