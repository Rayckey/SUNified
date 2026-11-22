# SPDX-License-Identifier: Apache-2.0
"""Shared UR5 embodiment and object-free learning observations."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg

from kuafu.paths import SKILLS_ROOT
from ..kinova.base_env_learned import ArmWithGripperArticulationCfg, BaseLearnedEnv, UniformLinkPoseCommandCfg
from .assets import (
    ARM_JOINTS, GRIPPER_JOINTS, GRIPPER_ROOT, HOME_JOINTS, TCP_POSITION,
    TCP_QUATERNION, asset_directory, require_assets,
)


def robot_config():
    """Configure six arm actuators and one physical Robotiq driver with mimics."""
    return ArmWithGripperArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(asset_directory() / "ur5_robotiq.usd"),
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, max_depenetration_velocity=5.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=32,
                solver_velocity_iteration_count=4,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={**dict(zip(ARM_JOINTS, HOME_JOINTS)),
                       "finger_joint|right_outer_knuckle_joint|.*inner_(finger|knuckle).*joint": 0.0},
            joint_vel={".*": 0.0},
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=list(ARM_JOINTS), effort_limit_sim=150.0,
                velocity_limit_sim=1.5, stiffness=400.0, damping=40.0,
            ),
            "hand": ImplicitActuatorCfg(
                joint_names_expr=["finger_joint"], effort_limit_sim=20.0,
                velocity_limit_sim=2.0, stiffness=20.0, damping=0.5, armature=0.001,
            ),
            "hand_mimic_followers": ImplicitActuatorCfg(
                joint_names_expr=["right_outer_knuckle_joint", ".*inner_(finger|knuckle).*joint"],
                effort_limit_sim=0.0, velocity_limit_sim=2.0, stiffness=0.0,
                damping=0.0, armature=0.001,
            ),
        },
        arm_actuated_joints=list(ARM_JOINTS),
        gripper_actuated_joints=list(GRIPPER_JOINTS),
        robotiq_control_mode="single_master", robotiq_coupled_joints=["finger_joint"],
        robotiq_target_signs=[1.0], robotiq_logical_state_signs=[1.0, 1.0],
        gripper_opening_angle=0.0, gripper_closing_angle=0.8,
        ee_command_pose=UniformLinkPoseCommandCfg(
            asset=SceneEntityCfg("robot"), body_name="wrist_3_link",
            offset=OffsetCfg(pos=TCP_POSITION, rot=TCP_QUATERNION),
        ),
        left_finger_link="left_inner_finger", right_finger_link="right_inner_finger",
        base_link_name="base_link", curobo_config_path="", curobo_ee_link_name="panda_fingertip",
        frame_transformer_source_prim_path="/World/envs/env_.*/Robot/base_link",
        frame_transformer_target_prim_path="/World/envs/env_.*/Robot/wrist_3_link",
        frame_transformer_offset_pos=TCP_POSITION, frame_transformer_offset_rot=TCP_QUATERNION,
        contact_sensors_info=[], soft_joint_pos_limit_factor=1.0,
    )


class UR5ConfigMixin:
    """Replace only the embodiment, observations, and default reward/program settings."""

    action_mode: str = "relative_joint"
    obs_mode: str = "robot_state"
    enable_cameras: bool = True

    def __post_init__(self):
        super().__post_init__()
        default_program = self.mpc_config.get("load_program")
        self.robot = robot_config()
        self.action_space = {"relative_ee": 8, "absolute_ee": 11,
                             "relative_joint": 8, "absolute_joint": 8}[self.action_mode]
        self.obs_dims = {"robot_state": 8}
        self.set_obs_mode("robot_state")
        self.gripper_contact_parent_path = GRIPPER_ROOT
        self.gripper_contact_links = ("left_inner_finger", "right_inner_finger")
        self.robot_reset_arm_noise_range = (-0.05, 0.05)
        self.skip_dense_reward_computation = True
        self.use_external_rewards = True
        self.ignore_stage_gate_for_success = True
        self.sun_program_evaluator_stage_override = True
        self.reward_print_every = 0
        self.episode_outcome_print_every = 0
        self.collect_terminal_stuck_constraint_stats = False
        # Object frames retain their owning bodies; only the robot source changes.
        for value in vars(self).values():
            if isinstance(value, FrameTransformerCfg):
                value.prim_path = self.robot.frame_transformer_source_prim_path
                value.debug_vis = False
        for name, link, alias in (("ee_frame", "wrist_3_link", "panda_fingertip"),
                                   ("l7_frame", "wrist_3_link", "panda_link7"),
                                   ("l4_frame", "forearm_link", "panda_link4")):
            frame = getattr(self, name)
            frame.target_frames = [FrameTransformerCfg.FrameCfg(
                prim_path=f"/World/envs/env_.*/Robot/{link}", name=alias,
                offset=OffsetCfg(pos=TCP_POSITION, rot=TCP_QUATERNION) if name == "ee_frame" else OffsetCfg(),
            )]
        # Preserve collection's view order/resolution, with wide simulated views
        # covering the UR5 and randomized tabletop workspace. These are not
        # claims of calibration to a physical camera installation.
        self.official_camera_pair = ("table_cam_0", "table_cam_1")
        viewpoints = (
            ((1.1, -0.85, 0.85), (0.39569154125, -0.24891408115, 0.11232713670, 0.87684240246)),
            ((1.0, 0.85, 0.75), (-0.42864508148, -0.22659031686, -0.11197473080, 0.86739949394)),
        )
        for camera, (position, quaternion) in zip((self.table_cam_0, self.table_cam_1), viewpoints):
            camera.width, camera.height = 128, 72
            camera.data_types = ["rgb"]
            camera.spawn = sim_utils.PinholeCameraCfg(
                focal_length=24.0, horizontal_aperture=30.0, clipping_range=(0.01, 5.0),
            )
            camera.offset = TiledCameraCfg.OffsetCfg(pos=position, rot=quaternion, convention="world")
        self.mpc_config = {
            "controller_name": "UR5IKController", "device": self.sim.device,
            "ee_frame": "panda_fingertip", "control_output_mode": "absolute_joint",
            "urdf_path": str(asset_directory() / "ur5_controller.urdf"),
            "load_program": default_program, "use_per_env_weight": False,
            "initial_residual": str(SKILLS_ROOT / "constraints" / "rewards.py"),
            "initial_predicate": str(SKILLS_ROOT / "constraints" / "predicates.py"),
            "gripper_opened": [0.0, 0.0], "gripper_closed": [0.8, 0.8],
            "gripper_closed_command": [0.8, 0.8], "gripper_max_force": 1.0,
            "robot_gripper_name": "ur5_robotiq", "has_finger_contact_sensors": False,
        }

    def set_obs_mode(self, mode):
        if mode != "robot_state":
            raise ValueError("UR5 observations contain robot_state and RGB; use obs_mode='robot_state'.")
        self.obs_mode, self.observation_space = mode, 8


class UR5EnvMixin:
    """Keep privileged task state inside the evaluator, outside learning observations."""

    def __init__(self, cfg, render_mode=None, **kwargs):
        require_assets()
        cfg.mpc_config["device"] = cfg.sim.device
        # Scripts can change action_mode after constructing a config.
        cfg.action_space = {"relative_ee": 8, "absolute_ee": 11,
                            "relative_joint": 8, "absolute_joint": 8}[cfg.action_mode]
        super().__init__(cfg, render_mode, **kwargs)

    def _reset_idx(self, env_ids):
        self._ur5_reset_in_progress = True
        try:
            super()._reset_idx(env_ids)
        finally:
            self._ur5_reset_in_progress = False
        # Task resets query observations before DirectRLEnv forwards the new
        # joint state. Invalidate those early reads so reset returns fresh poses
        # and images, including when only part of a vector environment resets.
        self.scene.write_data_to_sim()
        self.sim.forward()
        for sensor in self.scene.sensors.values():
            sensor.reset(env_ids)

    def external_rewards(self):
        reward = super().external_rewards
        if reward.__func__ is BaseLearnedEnv.external_rewards:
            return self._cached_success.to(dtype=torch.float32)
        return reward()

    def _get_observations(self):
        self.update_data_in_controller()
        api = self.program_runtime.controller_api_wrapper
        state = api.joint_dict[api.reference_agent]["joint_position"].clone()
        if state.shape != (self.num_envs, 8) or not torch.isfinite(state).all():
            raise RuntimeError(f"Invalid UR5 robot state: {state.shape}")
        observations = {"policy": state, "robot_state": state}
        if self.cfg.enable_cameras and not getattr(self, "_ur5_reset_in_progress", False):
            observations["rgb"] = torch.stack(
                [self.scene.sensors[name].data.output["rgb"][..., :3]
                 for name in ("table_cam_0", "table_cam_1")], dim=1,
            )
        return observations
