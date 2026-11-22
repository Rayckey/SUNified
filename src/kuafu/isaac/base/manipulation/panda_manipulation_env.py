# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

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
from isaaclab.sim import SimulationCfg, RenderCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform, quat_from_euler_xyz

ISAAC_NUCLEUS_DIR = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac"
)
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg



@configclass
class PandaManipulationEnvCfg(DirectRLEnvCfg):
    """Base configuration class for Panda manipulation environments.
    
    This class provides common configuration elements shared across all
    Panda manipulation tasks. Subclasses can override or extend these
    settings as needed.
    """
    episode_length_s: float = 800.0
    decimation: int = 4
    action_space: int = 9
    state_space: int = 0
    enable_camera: bool = False
    disable_terminations: bool = False
    disable_rewards: bool = False
    use_debug_action_mode: bool = False

    mpc_config: dict = {
        "gripper_opened": [0.04, 0.04],
        "gripper_closed": [0.0, 0.0],
        "gripper_max_force": 40.0,
        "robot_gripper_name": "franka_panda",
        "has_finger_contact_sensors": False,
        "object_grasp_debug": False,
        "object_grasp_debug_every": 25,
    }

    # Simulation settings
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,  # Will be set to decimation in __post_init__
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # Scene settings
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4, env_spacing=3.0, replicate_physics=True
    )

    # Robot configuration
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Franka/franka_instanceable.usd",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 0.0,
                "panda_joint2": -0.32,
                "panda_joint3": 0.0,
                "panda_joint4": -1.25,
                "panda_joint5": 0.0,
                "panda_joint6": 2.0,
                "panda_joint7": 0.8,
                "panda_finger_joint.*": 0.04,
            },
            pos=(0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                effort_limit=87.0,
                velocity_limit=2.175/4,
                velocity_limit_sim=2.175/4,
                stiffness=80.0,
                damping=4.0,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit=12.0,
                velocity_limit=2.61/4,
                velocity_limit_sim=2.61/4,
                stiffness=80.0,
                damping=4.0,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint.*"],
                effort_limit=200.0,
                velocity_limit=0.2,
                stiffness=2e3,
                damping=1e2,
            ),
        },
    )

    # Set render interval to match decimation
    sim.render_interval = decimation
        
    robot.actuators["panda_shoulder"].stiffness = 400.0
    robot.actuators["panda_shoulder"].damping = 80.0
    robot.actuators["panda_forearm"].stiffness = 400.0
    robot.actuators["panda_forearm"].damping = 80.0
    
    # Render configuration
    render_cfg: RenderCfg = RenderCfg(
        rendering_mode="quality",
        enable_translucency=False,
        enable_reflections=False,
        enable_global_illumination=True,
        enable_direct_lighting=True,
        enable_shadows=True,
        enable_ambient_occlusion=True,
    )

    # Ground plane
    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # Common frame transformers for end-effector and link4
    marker_cfg_h = FRAME_MARKER_CFG.copy()
    marker_cfg_h.markers["frame"].scale = (0.03, 0.03, 0.03)
    marker_cfg_h.prim_path = "/Visuals/panda_hand_frame"
    ee_frame = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/panda_link0",
        debug_vis=False,
        visualizer_cfg=marker_cfg_h,
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

    marker_cfg_l = FRAME_MARKER_CFG.copy()
    marker_cfg_l.markers["frame"].scale = (0.03, 0.03, 0.03)
    marker_cfg_l.prim_path = "/Visuals/panda_link4_frame"
    l4_frame = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/panda_link0",
        debug_vis=False,
        visualizer_cfg=marker_cfg_l,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot/panda_link4",
                name="panda_link4",
            )
        ],
    )

    
    action_scale: float = 3.0




class PandaManipulationEnv(DirectRLEnv):
    """Base class for Panda robot manipulation environments.
    
    This class provides common functionality for all Panda manipulation tasks,
    including robot initialization, action handling, and common utilities.
    """
    
    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        robot_ref = getattr(self, "_robot", None) or getattr(self, "robot", None)
        if robot_ref is None:
            raise AttributeError(
                "Robot must be initialized in _setup_scene before PandaManipulationEnv.__init__ completes"
            )
        self._robot = robot_ref
        self.robot = robot_ref

        if getattr(self.cfg, "disable_rewards", False):
            self._get_rewards = self._zero_rewards

        # Match the learned-env init pattern: build DOF limit tensors before
        # any code derives speed scales from them.
        self._robot_dof_lower_limits = robot_ref.data.soft_joint_pos_limits[
            0, :, 0
        ].to(device=self.device)
        self._robot_dof_upper_limits = robot_ref.data.soft_joint_pos_limits[
            0, :, 1
        ].to(device=self.device)
        self.robot_dof_lower_limits = self._robot_dof_lower_limits
        self.robot_dof_upper_limits = self._robot_dof_upper_limits

        # Initialize common robot-related variables after _setup_scene is called
        self.robot_dof_speed_scales = torch.ones_like(self._robot_dof_lower_limits)
        self.robot_dof_speed_scales[robot_ref.find_joints("panda_finger_joint1")[0]] = 0.1
        self.robot_dof_speed_scales[robot_ref.find_joints("panda_finger_joint2")[0]] = 0.1

        self._initialize_robot_common()

    def _zero_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

    def _initialize_robot_common(self):
        """Initialize common robot-related variables.
        
        This is called after _setup_scene to ensure the robot is available.
        Subclasses can override this if they need custom initialization.
        """
        def get_env_local_pose(
            env_pos: torch.Tensor, xformable: UsdGeom.Xformable, device: torch.device
        ):
            """Compute pose in env-local coordinates"""
            world_transform = xformable.ComputeLocalToWorldTransform(0)
            world_pos = world_transform.ExtractTranslation()
            world_quat = world_transform.ExtractRotationQuat()

            px = world_pos[0] - env_pos[0]
            py = world_pos[1] - env_pos[1]
            pz = world_pos[2] - env_pos[2]
            qx = world_quat.imaginary[0]
            qy = world_quat.imaginary[1]
            qz = world_quat.imaginary[2]
            qw = world_quat.real

            return torch.tensor([px, py, pz, qw, qx, qy, qz], device=device)

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # Get robot reference (handle both self.robot and self._robot naming)
        robot_ref = getattr(self, '_robot', None) or getattr(self, 'robot', None)
        if robot_ref is None:
            raise AttributeError("Robot must be initialized in _setup_scene before _initialize_robot_common is called")

        self.robot_dof_targets = torch.zeros(
            (self.num_envs, robot_ref.num_joints), device=self.device
        )

        stage = get_current_stage()
        hand_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(
                stage.GetPrimAtPath("/World/envs/env_0/Robot/panda_link7")
            ),
            self.device,
        )
        lfinger_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(
                stage.GetPrimAtPath("/World/envs/env_0/Robot/panda_leftfinger")
            ),
            self.device,
        )
        rfinger_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(
                stage.GetPrimAtPath("/World/envs/env_0/Robot/panda_rightfinger")
            ),
            self.device,
        )

        finger_pose = torch.zeros(7, device=self.device)
        finger_pose[0:3] = (lfinger_pose[0:3] + rfinger_pose[0:3]) / 2.0
        finger_pose[3:7] = lfinger_pose[3:7]
        hand_pose_inv_rot, hand_pose_inv_pos = tf_inverse(
            hand_pose[3:7], hand_pose[0:3]
        )

        robot_local_grasp_pose_rot, robot_local_pose_pos = tf_combine(
            hand_pose_inv_rot, hand_pose_inv_pos, finger_pose[3:7], finger_pose[0:3]
        )
        robot_local_pose_pos += torch.tensor([0, 0.04, 0], device=self.device)
        self.robot_local_grasp_pos = robot_local_pose_pos.repeat((self.num_envs, 1))
        self.robot_local_grasp_rot = robot_local_grasp_pose_rot.repeat(
            (self.num_envs, 1)
        )

        self.gripper_forward_axis = torch.tensor(
            [0, 0, 1], device=self.device, dtype=torch.float32
        ).repeat((self.num_envs, 1))

        self.gripper_up_axis = torch.tensor(
            [0, 1, 0], device=self.device, dtype=torch.float32
        ).repeat((self.num_envs, 1))

        self.hand_link_idx = robot_ref.find_bodies("panda_link7")[0][0]
        self.left_finger_link_idx = robot_ref.find_bodies("panda_leftfinger")[0][0]
        self.right_finger_link_idx = robot_ref.find_bodies("panda_rightfinger")[0][0]

    def _pre_physics_step(self, actions: torch.Tensor):
        """Handle actions before physics step.
        
        Supports both debug mode (direct action setting) and learning mode
        (incremental action updates). Can be overridden by subclasses.
        """
        use_debug = getattr(self.cfg, 'use_debug_action_mode', False)
        if use_debug:
            # DEBUGGING: Direct action setting
            self.robot_dof_targets = actions.clone()
        else:
            # LEARNING: Incremental action update
            self.actions = actions.clone().clamp(-1.0, 1.0)
            targets = (
                self.robot_dof_targets
                + self.robot_dof_speed_scales
                * self.dt
                * self.actions
                * self.cfg.action_scale
            )
            self.robot_dof_targets[:] = torch.clamp(
                targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits
            )

    def _apply_action(self):
        robot_ref = getattr(self, "_robot", None) or getattr(self, "robot", None)
        if robot_ref is None:
            raise AttributeError("Robot must be initialized before _apply_action is called")
        robot_ref.set_joint_position_target(self.robot_dof_targets)

    def _reset_robot(self, env_ids: torch.Tensor | None):
        """Reset robot to initial state.
        
        This is a helper method that can be called by subclasses in their
        _reset_idx implementation.
        """
        robot_ref = getattr(self, '_robot', None) or getattr(self, 'robot', None)
        if robot_ref is None:
            raise AttributeError("Robot must be initialized before _reset_robot is called")
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        assert env_ids is not None
        num_envs_to_reset = env_ids.shape[0]
        joint_pos = robot_ref.data.default_joint_pos[env_ids].clone()
        if not getattr(self.cfg, "disable_reset_randomization", False):
            joint_pos += sample_uniform(
                -0.125,
                0.125,
                (num_envs_to_reset, robot_ref.num_joints),
                self.device,
            )
            joint_pos = torch.clamp(
                joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits
            )
        joint_vel = torch.zeros_like(joint_pos)
        robot_ref.set_joint_position_target(joint_pos, env_ids=env_ids)
        robot_ref.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
