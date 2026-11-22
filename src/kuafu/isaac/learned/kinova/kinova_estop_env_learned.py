from __future__ import annotations

import math
import os
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab.sensors import FrameTransformer, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_angle_axis, quat_mul

from kuafu.mpc.simulation.spatial_utils import quat_to_rotation_6d
from kuafu.isaac.objects.estop import EStopArticulation, EStopSemanticCfg, make_estop_cfg
from kuafu.isaac.learned.estop_interaction_gate import DEFAULT_ESTOP_TRIGGER_FINGERTIP_MAX_DISTANCE
from .kinova_table_env_learned import (
    KinovaTableEnvLearned,
    KinovaTableEnvLearnedCfg,
)
from kuafu.isaac.tasks.env_utils import random_pose_radial


@configclass
class KinovaEStopEnvLearnedCfg(KinovaTableEnvLearnedCfg):
    """Kinova Gen3 + Robotiq 2F-85 presses a red e-stop button on the table to latch it closed."""

    observation_space = 31
    obs_mode: str = "state"
    obs_dims = {"state": 31, "full_state": 65}
    episode_length_s = 12

    # ------------------------------------------------------------------
    # E-stop articulation (mounted on the table USD top)
    # ------------------------------------------------------------------
    estop: ArticulationCfg = make_estop_cfg(
        prim_path="/World/envs/env_.*/EStop",
        spawn_pose=(0.5, 0.0, 0.01),
    )

    semantic_cfg: EStopSemanticCfg = EStopSemanticCfg()
    estop_trigger_fingertip_max_distance: float = 0.03

    # ------------------------------------------------------------------
    # Frame transformers for the e-stop (names must match the learned
    # constraints' detectable_objects entries)
    # ------------------------------------------------------------------
    _marker_cap_tip = FRAME_MARKER_CFG.copy()
    _marker_cap_tip.markers["frame"].scale = (0.05, 0.05, 0.05)
    _marker_cap_tip.prim_path = "/Visuals/estop_cap_tip_frame"
    estop_cap_tip_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/Links/base_link",
        debug_vis=False,
        visualizer_cfg=_marker_cap_tip,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/EStop/cap_link",
                name="estop_cap_tip",
                offset=OffsetCfg(pos=[0.0, 0.0, 0.05]),
            )
        ],
    )

    _marker_cap_center = FRAME_MARKER_CFG.copy()
    _marker_cap_center.markers["frame"].scale = (0.05, 0.05, 0.05)
    _marker_cap_center.prim_path = "/Visuals/estop_cap_center_frame"
    estop_cap_center_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/Links/base_link",
        debug_vis=False,
        visualizer_cfg=_marker_cap_center,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/EStop/cap_link",
                name="estop_cap_center",
                offset=OffsetCfg(pos=[0.0, 0.0, 0.025]),
            )
        ],
    )

    _marker_base = FRAME_MARKER_CFG.copy()
    _marker_base.markers["frame"].scale = (0.05, 0.05, 0.05)
    _marker_base.prim_path = "/Visuals/estop_base_frame"
    estop_base_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/Links/base_link",
        debug_vis=False,
        visualizer_cfg=_marker_base,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/EStop/base_link",
                name="estop_base_frame",
                offset=OffsetCfg(pos=[0.0, 0.0, 0.06]),
            )
        ],
    )

    detectable_objects = {
        'estop_cap_tip': (
            'tip frame of mushroom cap of the twist open e-stop button for closing, press against local +z axis for 3 cm to close, size:(0.04 m, 0.04 m, 0.04 m)',
            '/Visuals/estop_cap_tip_frame',
        ),
        'estop_cap_center': (
            'center frame of mushroom cap of the twist open e-stop button for unlocking, twist around +z for openings for 45 degrees to unlock; size:(0.04 m, 0.04 m, 0.04 m)',
            '/Visuals/estop_cap_center_frame',
        ),
        'estop_base_frame': (
            'rectangular housing of the e-stop button; size: (0.078, 0.072, 0.073) m',
            '/Visuals/estop_base_frame',
        ),
        'table': (
            'surface the e-stop is mounted on; size: (0.4, 0.6, 0.2) m',
            '/World/envs/env_*/Table',
        ),
        'panda_link4': (
            'frame of robot link 4',
            '/Visuals/panda_link4_frame',
        ),
        'panda_fingertip': (
            'frame of robot tool / end-effector',
            '/Visuals/panda_hand_frame',
        ),
    }

    _kuafu_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    _repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(_kuafu_dir)))
    mpc_config = {
        "controller_name": "FrankaIKController",
        "device": "cuda",
        "ee_frame": "panda_fingertip",
        "load_program": f"{_repo_dir}/resources/learned_programs/manipulation-estop-close/2026-04-24_22-06-17",
        "initial_residual": f"{_repo_dir}/src/kuafu/skills/constraints/rewards.py",
        "initial_predicate": f"{_repo_dir}/src/kuafu/skills/constraints/predicates.py",
        "use_per_env_weight": False,
        "gripper_opened": [0.0, 0.0],
        "gripper_closed": [0.8, 0.8],
        "gripper_closed_command": [0.8, 0.8],
        "gripper_max_force": 1.0,
    }
    del _kuafu_dir, _repo_dir

    task_description = 'press the e-stop button'
class KinovaEStopEnvLearned(KinovaTableEnvLearned):
    cfg: KinovaEStopEnvLearnedCfg

    def __init__(self, cfg: KinovaEStopEnvLearnedCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.press_joint_id = self.estop.find_joints(self.cfg.semantic_cfg.press_joint_name)[0][0]
        self.twist_joint_id = self.estop.find_joints(self.cfg.semantic_cfg.twist_joint_name)[0][0]

        self._pressed_once = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _setup_scene(self):
        super()._setup_scene()

        self.estop = EStopArticulation(self.cfg.estop, semantic_cfg=self.cfg.semantic_cfg)
        self.scene.articulations["estop"] = self.estop

        self.estop_cap_tip_frame = FrameTransformer(self.cfg.estop_cap_tip_frame)
        self.estop_cap_center_frame = FrameTransformer(self.cfg.estop_cap_center_frame)
        self.estop_base_frame = FrameTransformer(self.cfg.estop_base_frame)
        self.scene.sensors["estop_cap_tip"] = self.estop_cap_tip_frame
        self.scene.sensors["estop_cap_center"] = self.estop_cap_center_frame
        self.scene.sensors["estop_base_frame"] = self.estop_base_frame

        self.scene.clone_environments(copy_from_source=False)
        self._hide_static_surface_visual_prims()

        if self.device == "cpu" and self.cfg.terrain is not None:
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=1000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        if self.recorder_manager is not None and len(env_ids) > 0:
            self.recorder_manager.record_pre_reset(env_ids)

        estop_default = self.estop.data.default_joint_pos[env_ids].clone()
        estop_vel = torch.zeros_like(estop_default)
        self.estop.write_joint_state_to_sim(estop_default, estop_vel, env_ids=env_ids)
        self.estop.set_joint_position_target(estop_default, env_ids=env_ids)
        self.estop.reset_state(env_ids)

        # Randomize estop root pose on the table top: x/y via random_pose_radial
        # (matching the can-spawn pattern), with the local z-axis tilted up to
        # 30 degrees from world up so it isn't always perfectly upright.
        n_envs = len(env_ids)
        self._prepare_support_surface_reset(env_ids)
        origins = self._surface_origins(env_ids)
        angle_lower, angle_upper = self._spawn_angle_range()

        estop_pos_sampled, estop_yaw_quat = random_pose_radial(
            0.0,
            dist_min=0.4,
            dist_max=0.6,
            angle_min_rad=angle_lower,
            angle_max_rad=angle_upper,
            n=n_envs,
            device=self.device,
            origins=origins,
            rotate_randomly=True,
        )

        estop_z = self._sample_object_spawn_height_offset(n_envs)

        max_tilt_rad = math.radians(10.0)
        tilt_angle = torch.rand(n_envs, device=self.device) * max_tilt_rad
        tilt_phi = torch.rand(n_envs, device=self.device) * (2.0 * math.pi)
        tilt_axis = torch.stack(
            [
                torch.cos(tilt_phi),
                torch.sin(tilt_phi),
                torch.zeros(n_envs, device=self.device),
            ],
            dim=-1,
        )
        tilt_quat = quat_from_angle_axis(tilt_angle, tilt_axis)
        estop_quat = quat_mul(estop_yaw_quat, tilt_quat)

        estop_pos = torch.cat(
            [estop_pos_sampled[:, :2], self._spawn_z_above_surface(env_ids, estop_z)], dim=1
        )
        estop_root_pose = torch.cat([estop_pos, estop_quat], dim=-1)
        self.estop.write_root_pose_to_sim(estop_root_pose, env_ids=env_ids)
        self.estop.write_root_velocity_to_sim(
            torch.zeros((n_envs, 6), device=self.device), env_ids=env_ids
        )

        self._pressed_once[env_ids] = False
        self._success[env_ids] = False

        super()._reset_idx(env_ids)

    def _get_observations(self) -> dict:
        self.update_data_in_controller()

        api = self.program_runtime.controller_api_wrapper
        agent = api.reference_agent
        if agent is None:
            raise RuntimeError("controller_api_wrapper.reference_agent is not set")

        cap_tip_pos = api.get_object_xyz_position("estop_cap_tip")       # (N, 3)
        cap_center_pos = api.get_object_xyz_position("estop_cap_center") # (N, 3)
        fingertip_pos = api.lookup_robot_frame_position("panda_fingertip")  # (N, 3)
        link4_pos = api.lookup_robot_frame_position("panda_link4")          # (N, 3)

        cap_tip_rot6d = quat_to_rotation_6d(api.get_object_wxyz_quaternion("estop_cap_tip"))       # (N, 6)
        cap_center_rot6d = quat_to_rotation_6d(api.get_object_wxyz_quaternion("estop_cap_center")) # (N, 6)
        fingertip_rot6d = quat_to_rotation_6d(api.lookup_robot_frame_rotation("panda_fingertip"))  # (N, 6)
        link4_rot6d = quat_to_rotation_6d(api.lookup_robot_frame_rotation("panda_link4"))          # (N, 6)

        joint_pos = api.joint_dict[agent]["joint_position"]                   # (N, 9)
        joint_vel = api.joint_dict[agent]["joint_velocity"]                   # (N, 9)
        joint_effort = self._get_normalized_joint_effort(api, agent)          # (N, 9)
        gripper_contact = self._get_normalized_gripper_contact(api)           # (N, 2)
        joint_effort = torch.cat([joint_effort[:, :7], gripper_contact], dim=-1)  # (N, 9)
        gripper_open = api.gripper_openness().unsqueeze(-1)                    # (N, 1)
        stage = self.current_stage                                            # (N, 1)

        def _f(*tensors):
            return torch.cat(
                [t.to(device=self.device, dtype=torch.float32) for t in tensors], dim=-1
            )

        # Policy obs (31): task-relevant state
        policy_obs = _f(
            cap_tip_pos,
            cap_center_pos,
            fingertip_pos,        # 3+3+3 = 9
            cap_tip_rot6d,
            cap_center_rot6d,
            fingertip_rot6d,      # 6+6+6 = 18
            gripper_open,
            gripper_contact,       # 1+2 = 3
            stage,                 # 1
        )  # total: 31

        # Full state obs (65): adds link4, joint state, effort
        full_state_obs = _f(
            cap_tip_pos,
            cap_center_pos,
            fingertip_pos,
            link4_pos,             # 3*4 = 12
            cap_tip_rot6d,
            cap_center_rot6d,
            fingertip_rot6d,
            link4_rot6d,           # 6*4 = 24
            joint_pos,
            joint_vel,
            joint_effort,          # 9*3 = 27
            gripper_open,          # 1
            stage,                 # 1
        )  # total: 65

        obs_dim = int(self.cfg.observation_space)
        policy_obs_to_return = policy_obs if self.cfg.obs_mode == "state" else full_state_obs
        if policy_obs_to_return.shape[-1] != obs_dim:
            raise RuntimeError(
                f"Observation dim mismatch: got {policy_obs_to_return.shape[-1]}, expected {obs_dim}"
            )

        self.extras["episode/mean_stage"] = torch.mean(self.current_stage.float())
        self.extras["episode/max_stage"] = torch.max(self.current_stage.float())

        return {
            "policy": policy_obs_to_return,
            "full_state": full_state_obs,
            "state": policy_obs,
        }

    def _external_get_dones(self) -> torch.Tensor:
        # Advance the latched-closed state machine; this also re-issues the
        # joint position target at -latched_position for closed envs so the
        # press drive holds the button down once the gripper lifts away.
        self.estop.update_state()

        self._pressed_once |= self.estop.is_closed

        # Button must have been pressed AND end effector must be >= 5 cm from cap tip
        api = self.program_runtime.controller_api_wrapper
        cap_tip_pos = api.get_object_xyz_position("estop_cap_tip")         # (N, 3)
        fingertip_pos = api.lookup_robot_frame_position("panda_fingertip") # (N, 3)
        dist = torch.norm(
            fingertip_pos.to(self.device) - cap_tip_pos.to(self.device), dim=-1
        )  # (N,)
        ee_far_enough = dist >= 0.05

        self._success |= (self._pressed_once & ee_far_enough)

        if self.cfg.sun_program_evaluator_stage_override:
            return self._success.clone()
        else:
            past_stage = self.current_stage.flatten() > int(
                self.program_runtime.compiled_program.stage_count() / 2
            )
            return torch.logical_and(self._success, past_stage)
