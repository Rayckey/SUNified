# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from pxr import UsdGeom

from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.utils.torch.transformations import tf_combine, tf_inverse

from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, RenderCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from kuafu.isaac.base.manipulation.panda_manipulation_env import PandaManipulationEnv, PandaManipulationEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
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

TABLE_HEIGHT = 0.0
ISAAC_NUCLEUS_DIR = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac"

@configclass
class Stack3(PandaManipulationEnvCfg):
    """Configuration for Stack3 environment."""

    observation_space = 9

    # Rigid body properties of each cube
    cube_properties = sim_utils.RigidBodyPropertiesCfg(
        solver_position_iteration_count=16,
        solver_velocity_iteration_count=1,
        max_angular_velocity=1000.0,
        max_linear_velocity=1000.0,
        max_depenetration_velocity=5.0,
        disable_gravity=False,
    )

    # Set each stacking cube deterministically
    cube_1 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Cube_1",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.42, -0.12, TABLE_HEIGHT + 0.0203], rot=[1, 0, 0, 0]
        ),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/blue_block.usd",
            scale=(1.0, 1.0, 1.0),
            rigid_props=cube_properties,
            semantic_tags=[("class", "cube_1")],
        ),
    )
    cube_2 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Cube_2",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.50, 0.02, TABLE_HEIGHT + 0.0203], rot=[1, 0, 0, 0]
        ),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/red_block.usd",
            scale=(1.0, 1.0, 1.0),
            rigid_props=cube_properties,
            semantic_tags=[("class", "cube_2")],
        ),
    )
    cube_3 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Cube_3",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.56, 0.14, TABLE_HEIGHT + 0.0203], rot=[1, 0, 0, 0]
        ),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/green_block.usd",
            scale=(1.0, 1.0, 1.0),
            rigid_props=cube_properties,
            semantic_tags=[("class", "cube_3")],
        ),
    )

    # [ x: -1.4056495, y: -0.5176849, z: -2.5357052 ]

    # List of image observations in policy observations


class Stack3Env(PandaManipulationEnv):
    cfg: Stack3

    def __init__(
        self,
        cfg: Stack3,
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

        base_cube_pos = torch.tensor(
            [
                [
                    [0.42, -0.12, TABLE_HEIGHT + 0.0203],
                    [0.50, 0.02, TABLE_HEIGHT + 0.0203],
                    [0.56, 0.14, TABLE_HEIGHT + 0.0203],
                ]
            ],
            device=self.device,
        )
        if getattr(self.cfg, "disable_reset_randomization", False):
            cube_pos = base_cube_pos.repeat(len(env_ids), 1, 1)
        else:
            jitter = torch.rand(len(env_ids), 3, 3, device=self.device) * torch.tensor(
                [[[0.1, 0.1, 0.0]]], device=self.device
            )
            cube_pos = base_cube_pos.repeat(len(env_ids), 1, 1) + jitter

        root = self.cube_1.data.default_root_state.clone()
        root = root.index_select(0, env_ids)
        root[:, :3] = (
            self.scene.env_origins.index_select(0, env_ids) + cube_pos[:, 0, :]
        )
        eulers = torch.zeros(
            (len(env_ids), 3), device=self.device
        )  # or your real angles
        roll = eulers[:, 0]
        pitch = eulers[:, 1]
        yaw = eulers[:, 2]

        quat = math_utils.quat_from_euler_xyz(roll, pitch, yaw)
        root[:, 3:7] = quat  # quaternions are WXYZ in Isaac Lab

        # write pose and velocities to the sim, then reset buffers
        self.cube_1.write_root_pose_to_sim(root[:, :7], env_ids=env_ids)
        self.cube_1.write_root_velocity_to_sim(
            torch.zeros((len(env_ids), 6), device=self.device), env_ids=env_ids
        )
        self.cube_1.reset(env_ids)

        root = self.cube_2.data.default_root_state.clone()
        root = root.index_select(0, env_ids)
        root[:, :3] = (
            self.scene.env_origins.index_select(0, env_ids) + cube_pos[:, 1, :]
        )
        eulers = torch.zeros(
            (len(env_ids), 3), device=self.device
        )  # or your real angles
        roll = eulers[:, 0]
        pitch = eulers[:, 1]
        yaw = eulers[:, 2]
        quat = math_utils.quat_from_euler_xyz(roll, pitch, yaw)
        root[:, 3:7] = quat  # quaternions are WXYZ in Isaac Lab

        # write pose and velocities to the sim, then reset buffers
        self.cube_2.write_root_pose_to_sim(root[:, :7], env_ids=env_ids)
        self.cube_2.write_root_velocity_to_sim(
            torch.zeros((len(env_ids), 6), device=self.device), env_ids=env_ids
        )
        self.cube_2.reset(env_ids)

        root = self.cube_3.data.default_root_state.clone()
        root = root.index_select(0, env_ids)
        root[:, :3] = (
            self.scene.env_origins.index_select(0, env_ids) + cube_pos[:, 2, :]
        )
        eulers = torch.zeros(
            (len(env_ids), 3), device=self.device
        )  # or your real angles
        roll = eulers[:, 0]
        pitch = eulers[:, 1]
        yaw = eulers[:, 2]
        quat = math_utils.quat_from_euler_xyz(roll, pitch, yaw)
        root[:, 3:7] = quat  # quaternions are WXYZ in Isaac Lab

        # write pose and velocities to the sim, then reset buffers
        self.cube_3.write_root_pose_to_sim(root[:, :7], env_ids=env_ids)
        self.cube_3.write_root_velocity_to_sim(
            torch.zeros((len(env_ids), 6), device=self.device), env_ids=env_ids
        )
        self.cube_3.reset(env_ids)

        # Reset robot using base class helper
        env_ids_tensor = torch.tensor(env_ids, device=self.device, dtype=torch.long) if isinstance(env_ids, (list, tuple)) else env_ids
        self._reset_robot(env_ids_tensor)

    def _setup_scene(self):

        self._robot = Articulation(self.cfg.robot)
        self.cube_1 = RigidObject(self.cfg.cube_1)
        self.cube_2 = RigidObject(self.cfg.cube_2)
        self.cube_3 = RigidObject(self.cfg.cube_3)
        self.ee_frame = FrameTransformer(self.cfg.ee_frame)
        self.l4_frame = FrameTransformer(self.cfg.l4_frame)

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # Add objects to scene
        self.scene.articulations["robot"] = self._robot
        self.scene.rigid_objects["cube_1"] = self.cube_1
        self.scene.rigid_objects["cube_2"] = self.cube_2
        self.scene.rigid_objects["cube_3"] = self.cube_3
        self.scene.sensors["panda_fingertip"] = self.ee_frame
        self.scene.sensors["panda_link4"] = self.l4_frame

        self.scene.clone_environments(copy_from_source=False)

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=1000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        if getattr(self.cfg, "disable_terminations", False):
            done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            return done, done

        robot_pos = self._robot.data.root_state_w[:, :3]
        cube_1_pos = self.cube_1.data.root_state_w[:, :3]
        cube_2_pos = self.cube_2.data.root_state_w[:, :3]
        cube_3_pos = self.cube_3.data.root_state_w[:, :3]

        # Time out
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # Goal reached
        cube_distance = torch.norm(cube_1_pos - cube_2_pos, dim=1)
        cube_distance += torch.norm(cube_2_pos - cube_3_pos, dim=1)
        cube_distance += torch.norm(cube_1_pos - cube_3_pos, dim=1)
        goal_reached = cube_distance < 0.01

        terminated = goal_reached

        return terminated, time_out

    def _get_observations(self) -> dict:
        joint_states = self.scene.state["articulation"]["robot"]
        pose = self.scene.state["articulation"]["robot"]["root_pose"]
        cube_1_pos = self.cube_1.data.root_state_w[:, :3]
        cube_2_pos = self.cube_2.data.root_state_w[:, :3]
        cube_3_pos = self.cube_3.data.root_state_w[:, :3]

        return {
            "joint_states": joint_states,
            "pose": pose,
            "cube_1_pos": cube_1_pos,
            "cube_2_pos": cube_2_pos,
            "cube_3_pos": cube_3_pos,
        }


    def _get_rewards(self) -> torch.Tensor:
        # justa place holder right now
        robot_pos = self._robot.data.root_state_w[:, :3]
        cube_1_pos = self.cube_1.data.root_state_w[:, :3]
        cube_2_pos = self.cube_2.data.root_state_w[:, :3]
        goal_distance = torch.norm(cube_1_pos - cube_2_pos, dim=1)
        total_reward = 1 / cube_1_pos
        return total_reward

    # pre-physics step calls (handled by base class)
