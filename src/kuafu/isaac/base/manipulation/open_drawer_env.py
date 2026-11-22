# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.sensors.frame_transformer.frame_transformer import FrameTransformer
import torch

from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.utils.torch.transformations import tf_combine, tf_inverse, tf_vector
from pxr import UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObjectCfg, RigidObject
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from kuafu.isaac.base.manipulation.panda_manipulation_env import PandaManipulationEnv, PandaManipulationEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import sample_uniform
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg

@configclass
class OpenDrawerEnvCfg(PandaManipulationEnvCfg):
    """Configuration for OpenDrawer environment."""

    # Override parent defaults
    episode_length_s = 200000
    observation_space = 23
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=3.0, replicate_physics=True
    )

    # cabinet
    cabinet = ArticulationCfg(
        prim_path="/World/envs/env_.*/Cabinet",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd",
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(1.0, 0, 0.4),
            rot=(0.0, 0.0, 0.0, 0.1),
            joint_pos={
                "door_left_joint": 0.0,
                "door_right_joint": 0.0,
                "drawer_bottom_joint": 0.0,
                "drawer_top_joint": 0.0,
            },
        ),
        actuators={
            "drawers": ImplicitActuatorCfg(
                joint_names_expr=["drawer_top_joint", "drawer_bottom_joint"],
                effort_limit_sim=87.0,
                stiffness=0.5,
                damping=0.2,
            ),
            "doors": ImplicitActuatorCfg(
                joint_names_expr=["door_left_joint", "door_right_joint"],
                effort_limit_sim=87.0,
                stiffness=2.0,
                damping=1.0,
            ),
        },
    )


    marker_cfg1 = FRAME_MARKER_CFG.copy()
    marker_cfg1.markers["frame"].scale = (0.1, 0.1, 0.1)
    marker_cfg1.prim_path = "/Visuals/drawer_handle_bottom_frame"
    bottom_cabinet_handle_frame = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/panda_link0",
        debug_vis=True,
        visualizer_cfg=marker_cfg1,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Cabinet/drawer_bottom",
                name="drawer_bottom",
                offset=OffsetCfg(
                    pos=[0.28, 0.0, 0.0],
                ),
            )
        ],
    )

    marker_cfg2 = FRAME_MARKER_CFG.copy()
    marker_cfg2.markers["frame"].scale = (0.1, 0.1, 0.1)
    marker_cfg2.prim_path = "/Visuals/drawer_handle_top_frame"
    top_cabinet_handle_frame = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/panda_link0",
        debug_vis=True,
        visualizer_cfg=marker_cfg2,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Cabinet/drawer_top",
                name="drawer_top",
                offset=OffsetCfg(
                    pos=[0.28, 0.0, 0.0],
                ),
            )
        ],
    )

    marker_cfg3 = FRAME_MARKER_CFG.copy()
    marker_cfg3.markers["frame"].scale = (0.1, 0.1, 0.1)
    marker_cfg3.prim_path = "/Visuals/door_left_nob_frame"
    left_door_nob_frame = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/panda_link0",
        debug_vis=True,
        visualizer_cfg=marker_cfg3,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Cabinet/door_left_nob_link",
                name="door_left_nob_frame",
                offset=OffsetCfg(
                    pos=[0.0, 0.0, 0.0],
                ),
            )
        ],
    )

    marker_cfg4 = FRAME_MARKER_CFG.copy()
    marker_cfg4.markers["frame"].scale = (0.1, 0.1, 0.1)
    marker_cfg4.prim_path = "/Visuals/door_right_nob_frame"
    right_door_nob_frame = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/panda_link0",
        debug_vis=True,
        visualizer_cfg=marker_cfg4,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Cabinet/door_right_nob_link",
                name="door_right_nob_frame",
                offset=OffsetCfg(
                    pos=[0.0, 0.0, 0.0],
                ),
            )
        ],
    )

    marker_cfg5 = FRAME_MARKER_CFG.copy()
    marker_cfg5.markers["frame"].scale = (0.1, 0.1, 0.1)
    marker_cfg5.prim_path = "/Visuals/drawer_bottom_frame"
    bottom_drawer_frame = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/panda_link0",
        debug_vis=True,
        visualizer_cfg=marker_cfg5,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Cabinet/drawer_bottom",
                name="drawer_bottom_handle_frame",
                offset=OffsetCfg(
                    pos=[0.3, 0.0, 0.0],
                ),
            )
        ],
    )

    marker_cfg5_1 = FRAME_MARKER_CFG.copy()
    marker_cfg5_1.markers["frame"].scale = (0.1, 0.1, 0.1)
    marker_cfg5_1.prim_path = "/Visuals/drawer_bottom_frame_center"
    bottom_drawer_frame_center = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/panda_link0",
        debug_vis=True,
        visualizer_cfg=marker_cfg5_1,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Cabinet/drawer_bottom",
                name="drawer_bottom_handle_frame",
                offset=OffsetCfg(
                    pos=[0.1, 0.0, 0.0],
                ),
            )
        ],
    )

    marker_cfg6 = FRAME_MARKER_CFG.copy()
    marker_cfg6.markers["frame"].scale = (0.1, 0.1, 0.1)
    marker_cfg6.prim_path = "/Visuals/drawer_top_frame"
    top_drawer_frame = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/panda_link0",
        debug_vis=True,
        visualizer_cfg=marker_cfg6,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Cabinet/drawer_top",
                name="drawer_top_handle_frame",
                offset=OffsetCfg(
                    pos=[0.3, 0.0, 0.0],
                ),
            )
        ],
    )

    marker_cfg6_1 = FRAME_MARKER_CFG.copy()
    marker_cfg6_1.markers["frame"].scale = (0.1, 0.1, 0.1)
    marker_cfg6_1.prim_path = "/Visuals/drawer_top_frame_center"
    top_drawer_frame_center = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/panda_link0",
        debug_vis=True,
        visualizer_cfg=marker_cfg6_1,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Cabinet/drawer_top",
                name="drawer_top_handle_frame",
                offset=OffsetCfg(
                    pos=[0.1, 0.0, 0.0],
                ),
            )
        ],
    )

    def __post_init__(self):
        """Post-initialization to set environment-specific parameters."""
        super().__post_init__()
        # Override robot configuration for drawer task
        self.robot.spawn.rigid_props.disable_gravity = False  # Initially False
        self.robot.init_state.joint_pos = {
            "panda_joint1": 1.157,
            "panda_joint2": -1.066,
            "panda_joint3": -0.155,
            "panda_joint4": -2.239,
            "panda_joint5": -1.841,
            "panda_joint6": 1.003,
            "panda_joint7": 0.469,
            "panda_finger_joint.*": 0.035,
        }
        # Override actuator parameters (use effort_limit_sim instead of effort_limit)
        self.robot.actuators["panda_shoulder"].effort_limit_sim = 87.0
        self.robot.actuators["panda_forearm"].effort_limit_sim = 12.0
        self.robot.actuators["panda_hand"].effort_limit_sim = 200.0
        # Set disable_gravity back to True (as in original)
        self.robot.spawn.rigid_props.disable_gravity = True
        # Override ee_frame name
        self.ee_frame.target_frames[0].name = "end_effector"


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

        # Drawer-specific initialization
        drawer_local_grasp_pose = torch.tensor(
            [0.3, 0.01, 0.0, 1.0, 0.0, 0.0, 0.0], device=self.device
        )
        self.drawer_local_grasp_pos = drawer_local_grasp_pose[0:3].repeat(
            (self.num_envs, 1)
        )
        self.drawer_local_grasp_rot = drawer_local_grasp_pose[3:7].repeat(
            (self.num_envs, 1)
        )

        self.drawer_inward_axis = torch.tensor(
            [-1, 0, 0], device=self.device, dtype=torch.float32
        ).repeat((self.num_envs, 1))
        self.drawer_up_axis = torch.tensor(
            [0, 0, 1], device=self.device, dtype=torch.float32
        ).repeat((self.num_envs, 1))

        self.drawer_link_idx = self._cabinet.find_bodies("drawer_top")[0][0]

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
        self.drawer_bottom_handle_frame = FrameTransformer(
            self.cfg.bottom_cabinet_handle_frame
        )
        self.drawer_top_handle_frame = FrameTransformer(
            self.cfg.top_cabinet_handle_frame
        )

        self.top_drawer_frame_center = FrameTransformer(
            self.cfg.top_drawer_frame_center
        )
        self.bottom_drawer_frame_center = FrameTransformer(
            self.cfg.bottom_drawer_frame_center
        )

        self.left_door_nob_frame = FrameTransformer(self.cfg.left_door_nob_frame)
        self.right_door_nob_frame = FrameTransformer(self.cfg.right_door_nob_frame)

        self.scene.sensors["panda_fingertip"] = self.ee_frame
        self.scene.sensors["panda_link4"] = self.l4_frame
        self.scene.sensors["drawer_bottom_handle_frame"] = self.drawer_bottom_handle_frame
        self.scene.sensors["drawer_top_handle_frame"] = self.drawer_top_handle_frame
        self.scene.sensors["left_door_nob_frame"] = self.left_door_nob_frame
        self.scene.sensors["right_door_nob_frame"] = self.right_door_nob_frame

        self.scene.sensors["drawer_bottom_center_frame"] = (
            self.bottom_drawer_frame_center
        )
        self.scene.sensors["drawer_top_center_frame"] = self.top_drawer_frame_center

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # post-physics step calls

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        if getattr(self.cfg, "disable_terminations", False):
            done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            return done, done
        terminated = self._cabinet.data.joint_pos[:, 3] > 1.0
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        robot_left_finger_pos = self._robot.data.body_pos_w[
            :, self.left_finger_link_idx
        ]
        robot_right_finger_pos = self._robot.data.body_pos_w[
            :, self.right_finger_link_idx
        ]

        return self._compute_rewards(
            self.actions,
            self._cabinet.data.joint_pos,
            robot_left_finger_pos,
            robot_right_finger_pos,
            self.gripper_forward_axis,
            self.drawer_inward_axis,
            self.gripper_up_axis,
            self.drawer_up_axis,
            self.num_envs,
            self.cfg.dist_reward_scale,
            self.cfg.rot_reward_scale,
            self.cfg.open_reward_scale,
            self.cfg.action_penalty_scale,
            self.cfg.finger_reward_scale,
            self._robot.data.joint_pos,
        )

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        # Reset robot using base class helper
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
                self._cabinet.data.joint_pos[:, 3].unsqueeze(-1),
                self._cabinet.data.joint_vel[:, 3].unsqueeze(-1),
            ),
            dim=-1,
        )
        return {"policy": torch.clamp(obs, -5.0, 5.0)}


    def _compute_rewards(
        self,
        actions,
        cabinet_dof_pos,
        # franka_grasp_pos,
        # drawer_grasp_pos,
        # franka_grasp_rot,
        # drawer_grasp_rot,
        franka_lfinger_pos,
        franka_rfinger_pos,
        gripper_forward_axis,
        drawer_inward_axis,
        gripper_up_axis,
        drawer_up_axis,
        num_envs,
        dist_reward_scale,
        rot_reward_scale,
        open_reward_scale,
        action_penalty_scale,
        finger_reward_scale,
        joint_positions,
    ):


        # regularization on the actions (summed for each environment)
        action_penalty = torch.sum(actions**2, dim=-1)

        # how far the cabinet has been opened out
        open_reward = cabinet_dof_pos[:, 3]  # drawer_top_joint


        rewards = (
            open_reward_scale * open_reward
            - action_penalty_scale * action_penalty
        )

        self.extras["log"] = {
            "open_reward": (open_reward_scale * open_reward).mean(),
            "action_penalty": (-action_penalty_scale * action_penalty).mean(),
        }

        # bonus for opening drawer properly
        rewards = torch.where(cabinet_dof_pos[:, 3] > 0.01, rewards + 0.25, rewards)
        rewards = torch.where(cabinet_dof_pos[:, 3] > 0.2, rewards + 0.25, rewards)
        rewards = torch.where(cabinet_dof_pos[:, 3] > 0.35, rewards + 0.25, rewards)

        return rewards

    def _compute_grasp_transforms(
        self,
        hand_rot,
        hand_pos,
        franka_local_grasp_rot,
        franka_local_grasp_pos,
        drawer_rot,
        drawer_pos,
        drawer_local_grasp_rot,
        drawer_local_grasp_pos,
    ):
        global_franka_rot, global_franka_pos = tf_combine(
            hand_rot, hand_pos, franka_local_grasp_rot, franka_local_grasp_pos
        )
        global_drawer_rot, global_drawer_pos = tf_combine(
            drawer_rot, drawer_pos, drawer_local_grasp_rot, drawer_local_grasp_pos
        )

        return (
            global_franka_rot,
            global_franka_pos,
            global_drawer_rot,
            global_drawer_pos,
        )
