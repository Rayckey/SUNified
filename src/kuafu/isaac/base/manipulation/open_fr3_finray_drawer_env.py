# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
from kuafu.paths import OBJECT_ROOT

import os
from isaaclab.sensors.frame_transformer.frame_transformer import FrameTransformer
import torch

from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.utils.torch.transformations import tf_combine, tf_inverse, tf_vector
from pxr import UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObjectCfg, RigidObject
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import sample_uniform, quat_from_euler_xyz
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from kuafu.isaac.base.manipulation.panda_manipulation_env import (
    PandaManipulationEnvCfg,
    PandaManipulationEnv,
)

@configclass
class OpenDrawerEnvCfg(PandaManipulationEnvCfg):
    # env
    observation_space = 49
    enable_cameras: bool = False

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=3.0, replicate_physics=True
    )

    # cabinet
    cabinet = ArticulationCfg(
        prim_path="/World/envs/env_.*/Cabinet",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(OBJECT_ROOT / 'two_layer_drawer_iphone_with_link/two_layer_drawer_iphone_with_link.usd'),
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.35, -0.4, 0.2),
            rot=(0.70710678, 0, 0, 0.70710678),
            joint_pos={
                "upper_drawer_joint": 0.0,
            },
        ),
        actuators={
            "drawers": ImplicitActuatorCfg(
                joint_names_expr=["upper_drawer_joint"],
                effort_limit_sim=87.0,
                stiffness=0.5,
                damping=0.2,
            )
        },
    )

    marker_cfg2 = FRAME_MARKER_CFG.copy()
    marker_cfg2.markers["frame"].scale = (0.1, 0.1, 0.1)
    marker_cfg2.prim_path = "/Visuals/top_drawer_handle_frame"
    top_drawer_handle_frame = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/panda_link0",
        debug_vis=False,
        visualizer_cfg=marker_cfg2,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Cabinet/upper_drawer_handle",
                name="top_drawer_handle_frame",
                offset=OffsetCfg(
                    pos=[0.0, 0.0, 0.0],
                ),
            )
        ],
    )

    marker_cfg6_1 = FRAME_MARKER_CFG.copy()
    marker_cfg6_1.markers["frame"].scale = (0.1, 0.1, 0.1)
    marker_cfg6_1.prim_path = "/Visuals/top_drawer_center_frame"
    top_drawer_center_frame = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/panda_link0",
        debug_vis=False,
        visualizer_cfg=marker_cfg6_1,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Cabinet/upper_drawer_handle",
                name="top_drawer_center_frame",
                offset=OffsetCfg(
                    pos=[0.0, 0.12, 0.0],
                ),
            )
        ],
    )

    detectable_objects = {
        "cabinet": (
            "mesh of the cabinet, local frame +z is up direction, local frame -y is the forward direction, estimated size: (0.15, 0.25, 0.2)",
            "/World/envs/env_*/Cabinet",
        ),
        "top_drawer_center_frame": (
            "center frame of top drawer, estimated size: (0.15, 0.25, 0.05)",
            "/Visuals/top_drawer_center_frame",
        ),
        "top_drawer_handle_frame": (
            "frame of handle of the top drawer, handle is a bar along x axis, local frame -y is pulling out (open) direction, local frame +y is pushing in (close) direction size: (0.15, 0.02, 0.02)",
            "/Visuals/top_drawer_handle_frame",
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

class OpenDrawerEnv(PandaManipulationEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: OpenDrawerEnvCfg

    def __init__(self, cfg: OpenDrawerEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._initialize_robot_common()

    def _initialize_robot_common(self):
        """Initialize common robot-related variables.
        
        This is called after _setup_scene to ensure the robot is available.
        Subclasses can override this if they need custom initialization.
        """

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # Get robot reference (handle both self.robot and self._robot naming)
        robot_ref = getattr(self, '_robot', None) or getattr(self, 'robot', None)
        if robot_ref is None:
            raise AttributeError("Robot must be initialized in _setup_scene before _initialize_robot_common is called")

        # create auxiliary variables for computing applied action, observations and rewards
        self.robot_dof_lower_limits = robot_ref.data.soft_joint_pos_limits[
            0, :, 0
        ].to(device=self.device)
        self.robot_dof_upper_limits = robot_ref.data.soft_joint_pos_limits[
            0, :, 1
        ].to(device=self.device)

        self.robot_dof_targets = torch.zeros(
            (self.num_envs, robot_ref.num_joints), device=self.device
        )


    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self._cabinet = Articulation(self.cfg.cabinet)
        self.scene.articulations["robot"] = self._robot
        self.scene.articulations["cabinet"] = self._cabinet

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        self.ee_frame = FrameTransformer(self.cfg.ee_frame)
        self.l4_frame = FrameTransformer(self.cfg.l4_frame)

        self.top_drawer_handle_frame = FrameTransformer(
            self.cfg.top_drawer_handle_frame
        )

        self.top_drawer_center_frame = FrameTransformer(
            self.cfg.top_drawer_center_frame
        )

        self.scene.sensors["panda_fingertip"] = self.ee_frame
        self.scene.sensors["panda_link4"] = self.l4_frame
        self.scene.sensors["top_drawer_handle_frame"] = self.top_drawer_handle_frame
        self.scene.sensors["top_drawer_center_frame"] = self.top_drawer_center_frame

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # pre-physics step calls are inherited from parent

    # post-physics step calls

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        if getattr(self.cfg, "disable_terminations", False):
            done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            return done, done

        opened = self._cabinet.data.joint_pos[:, 0] > 0.1

        finger_joint_1 = self._robot.data.joint_pos[:, -2]
        finger_joint_2 = self._robot.data.joint_pos[:, -1]
        gripper_open = torch.logical_and(finger_joint_1 > 0.03, finger_joint_2 > 0.03)

        handle_w = self.top_drawer_handle_frame.data.target_pos_w[:, 0, :]
        fingertip_w = self.ee_frame.data.target_pos_w[:, 0, :]
        far_from_handle = torch.linalg.norm(fingertip_w - handle_w, dim=-1) >= 0.05

        terminated = torch.logical_and(torch.logical_and(opened, gripper_open), far_from_handle)
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        return  self._cabinet.data.joint_pos[:, 0]

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        super()._reset_idx(env_ids)
        # Reset robot using parent's helper method
        self._reset_robot(env_ids)

        # cabinet state
        zeros = torch.zeros(
            (len(env_ids), self._cabinet.num_joints), device=self.device
        )
        self._cabinet.write_joint_state_to_sim(zeros, zeros, env_ids=env_ids)


    def _get_observations(self) -> dict:
        dof_pos_scaled = (
            2.0
            * (self._robot.data.joint_pos - self.robot_dof_lower_limits)
            / (self.robot_dof_upper_limits - self.robot_dof_lower_limits)
            - 1.0
        )

        obs = torch.cat(
            (
                dof_pos_scaled,
                self._robot.data.joint_vel,
                # to_target,
                self._cabinet.data.joint_pos[:, 0].unsqueeze(-1),
                self._cabinet.data.joint_vel[:, 0].unsqueeze(-1),
            ),
            dim=-1,
        )
        return {"policy": torch.clamp(obs, -5.0, 5.0)}
