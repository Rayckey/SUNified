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
from kuafu.isaac.learned.estop_interaction_gate import (
    DEFAULT_ESTOP_TRIGGER_FINGERTIP_MAX_DISTANCE,
    update_estop_state_with_fingertip_gate,
)
from kuafu.isaac.learned.franka_table_env_learned import (
    FrankaTableEnvLearned,
    FrankaTableEnvLearnedCfg,
)
from kuafu.isaac.tasks.env_utils import random_pose_radial


@configclass
class FR3FinRayEStopClosedEnvLearnedCfg(FrankaTableEnvLearnedCfg):
    """fr3 + Robotiq 2F-85 twists a latched-closed e-stop button to release it open."""

    observation_space = 31
    obs_mode: str = "state"
    obs_dims = {"state": 31, "full_state": 65}
    episode_length_s = 12

    def __post_init__(self):
        super().__post_init__()
        self.robot.init_state = ArticulationCfg.InitialStateCfg(
            joint_pos={
                "fr3_joint1": -1.0,
                "fr3_joint2": -0.569,
                "fr3_joint3": 0.0,
                "fr3_joint4": -2.810,
                "fr3_joint5": 0.0,
                "fr3_joint6": 2.5,
                "fr3_joint7": -2.5,
                "robotiq_85_right_knuckle_joint": 0.0,
                "robotiq_85_left_knuckle_joint": 0.0,
            },
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        )

    # ------------------------------------------------------------------
    # E-stop articulation (mounted on the franka_table USD table top)
    # ------------------------------------------------------------------
    estop: ArticulationCfg = make_estop_cfg(
        prim_path="/World/envs/env_.*/EStop",
        spawn_pose=(0.5, 0.0, 0.01),
    )

    semantic_cfg: EStopSemanticCfg = EStopSemanticCfg()
    estop_trigger_fingertip_max_distance: float = DEFAULT_ESTOP_TRIGGER_FINGERTIP_MAX_DISTANCE

    # ------------------------------------------------------------------
    # Frame transformers for the e-stop (names must match the learned
    # constraints' detectable_objects entries)
    # ------------------------------------------------------------------
    _marker_cap_tip = FRAME_MARKER_CFG.copy()
    _marker_cap_tip.markers["frame"].scale = (0.05, 0.05, 0.05)
    _marker_cap_tip.prim_path = "/Visuals/estop_cap_tip_frame"
    estop_cap_tip_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/fr3/base",
        debug_vis=True,
        visualizer_cfg=_marker_cap_tip,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/EStop/cap_link",
                name="estop_cap_tip",
                offset=OffsetCfg(pos=[0.0, 0.0, 0.045]),
            )
        ],
    )

    _marker_cap_center = FRAME_MARKER_CFG.copy()
    _marker_cap_center.markers["frame"].scale = (0.05, 0.05, 0.05)
    _marker_cap_center.prim_path = "/Visuals/estop_cap_center_frame"
    estop_cap_center_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/fr3/base",
        debug_vis=False,
        visualizer_cfg=_marker_cap_center,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/EStop/cap_link",
                name="estop_cap_center",
                offset=OffsetCfg(pos=[0.0, 0.0, 0.03]),
            )
        ],
    )

    _marker_base = FRAME_MARKER_CFG.copy()
    _marker_base.markers["frame"].scale = (0.05, 0.05, 0.05)
    _marker_base.prim_path = "/Visuals/estop_base_frame"
    estop_base_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/fr3/base",
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
            'center frame of mushroom cap of the twist open e-stop button for unlocking, twist around object +z axis for up to 45 degrees to unlock; size:(0.04 m, 0.04 m, 0.04 m)',
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

    _kuafu_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(_kuafu_dir)))
    mpc_config = {
        "controller_name": "FrankaIKController",
        "device": "cuda",
        "ee_frame": "panda_fingertip",
        "load_program": f"{_repo_dir}/resources/learned_programs/manipulation-estop-open/2026-04-27_15-52-56",
        "initial_residual": f"{_repo_dir}/src/kuafu/skills/constraints/rewards.py",
        "initial_predicate": f"{_repo_dir}/src/kuafu/skills/constraints/predicates.py",
        "use_per_env_weight": False,
        "gripper_opened": [0.0, 0.0],
        "gripper_closed": [0.8, -0.8],
        "gripper_max_force": 1.0,
    }
    del _kuafu_dir, _repo_dir

    task_description = 'unlock the e-stop'
class FR3FinRayEStopClosedEnvLearned(FrankaTableEnvLearned):
    cfg: FR3FinRayEStopClosedEnvLearnedCfg

    def __init__(self, cfg: FR3FinRayEStopClosedEnvLearnedCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.press_joint_id = self.estop.find_joints(self.cfg.semantic_cfg.press_joint_name)[0][0]
        self.twist_joint_id = self.estop.find_joints(self.cfg.semantic_cfg.twist_joint_name)[0][0]

        self._unlocked_once = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
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

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=1000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        if self.recorder_manager is not None and len(env_ids) > 0:
            self.recorder_manager.record_pre_reset(env_ids)

        # Reset twist joint to 0 first, then directly latch the press joint
        # at -latched_position via set_closed (which also marks is_closed=True
        # immediately and pins the drive target down). This avoids the
        # one-step gap where the default press drive target (0) could spring
        # the button back up before update_state() detects the press.
        estop_init = self.estop.data.default_joint_pos[env_ids].clone()
        estop_init[:, self.twist_joint_id] = 0.0
        estop_vel = torch.zeros_like(estop_init)
        self.estop.write_joint_state_to_sim(estop_init, estop_vel, env_ids=env_ids)
        self.estop.set_joint_position_target(estop_init, env_ids=env_ids)
        self.estop.reset_state(env_ids)
        self.estop.set_closed(env_ids)

        # Randomize estop root pose on the table top: x/y via random_pose_radial
        # (matching the can-spawn pattern), with the local z-axis tilted up to
        # 30 degrees from world up so it isn't always perfectly upright.
        n_envs = len(env_ids)
        origins = self.scene.env_origins.index_select(0, env_ids)

        estop_pos_sampled, estop_yaw_quat = random_pose_radial(
            0.0,
            dist_min=0.3,
            dist_max=0.5,
            angle_min_rad=-torch.pi/2 + 0.3,
            angle_max_rad=-0.3,
            n=n_envs,
            device=self.device,
            origins=origins,
            rotate_randomly=True,
        )

        estop_z = -0.03 + torch.rand(n_envs, device=self.device) * 0.1

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
            [estop_pos_sampled[:, :2], origins[:, 2:3] + estop_z.unsqueeze(-1)], dim=1
        )
        estop_root_pose = torch.cat([estop_pos, estop_quat], dim=-1)
        self.estop.write_root_pose_to_sim(estop_root_pose, env_ids=env_ids)
        self.estop.write_root_velocity_to_sim(
            torch.zeros((n_envs, 6), device=self.device), env_ids=env_ids
        )

        self._unlocked_once[env_ids] = False
        self._success[env_ids] = False

        super()._reset_idx(env_ids)

    def _get_observations(self) -> dict:
        self.update_data_in_controller()

        api = self.program_runtime.controller_api_wrapper
        agent = api.reference_agent
        if agent is None:
            raise RuntimeError("controller_api_wrapper.reference_agent is not set")

        cap_tip_pos = api.get_object_xyz_position("estop_cap_tip")       # (N, 3)
        base_pos = api.get_object_xyz_position("estop_base_frame") # (N, 3)
        fingertip_pos = api.lookup_robot_frame_position("panda_fingertip")  # (N, 3)
        link4_pos = api.lookup_robot_frame_position("panda_link4")          # (N, 3)

        cap_tip_rot6d = quat_to_rotation_6d(api.get_object_wxyz_quaternion("estop_cap_tip"))       # (N, 6)
        base_rot6d = quat_to_rotation_6d(api.get_object_wxyz_quaternion("estop_base_frame")) # (N, 6)
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
            base_pos,
            fingertip_pos,        # 3+3+3 = 9
            cap_tip_rot6d,
            base_rot6d,
            fingertip_rot6d,      # 6+6+6 = 18
            gripper_open,
            gripper_contact,       # 1+2 = 3
            stage,                 # 1
        )  # total: 31

        # Full state obs (65): adds link4, joint state, effort
        full_state_obs = _f(
            cap_tip_pos,
            base_pos,
            fingertip_pos,
            link4_pos,             # 3*4 = 12
            cap_tip_rot6d,
            base_rot6d,
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

    def external_rewards(self) -> torch.Tensor:
        self.update_data_in_controller()
        update_estop_state_with_fingertip_gate(
            self, "estop_cap_center", self.cfg.estop_trigger_fingertip_max_distance
        )

        cap_center_w = self.estop_cap_center_frame.data.target_pos_w.squeeze(1)
        ft_w = self.ee_frame.data.target_pos_w.squeeze(1)
        cap_dist = torch.linalg.norm(ft_w - cap_center_w, dim=-1)
        reach_reward = torch.exp(-8.0 * cap_dist)
        interact_reward = torch.exp(-40.0 * cap_dist)

        sem = self.cfg.semantic_cfg
        twist_pos = self.estop.data.joint_pos[:, self.twist_joint_id]
        press_pos = self.estop.data.joint_pos[:, self.press_joint_id]
        unlock_angle = max(sem.unlock_twist_angle_rad - sem.unlock_tolerance_rad, 1e-6)
        twist_progress = torch.clamp(-twist_pos / unlock_angle, 0.0, 1.0)
        release_progress = torch.clamp(
            (press_pos + sem.latched_position) / max(sem.latched_position, 1e-6),
            0.0,
            1.0,
        )
        open_reward = self.estop.is_open.float()

        return (
            0.75 * reach_reward
            + 0.75 * interact_reward
            + 2.0 * twist_progress
            + 4.0 * open_reward
            + open_reward * release_progress
        )

    def _external_get_dones(self) -> torch.Tensor:
        # Advance the latched-closed state machine. The button starts closed
        # at reset; it transitions to open once the cap is twisted past the
        # unlock threshold AND the press joint is allowed to spring back up.
        update_estop_state_with_fingertip_gate(
            self, "estop_cap_center", self.cfg.estop_trigger_fingertip_max_distance
        )

        self._unlocked_once |= self.estop.is_open

        # Button must have been released open AND end effector must be >= 5 cm from cap tip
        api = self.program_runtime.controller_api_wrapper
        cap_tip_pos = api.get_object_xyz_position("estop_cap_tip")         # (N, 3)
        fingertip_pos = api.lookup_robot_frame_position("panda_fingertip") # (N, 3)
        dist = torch.norm(
            fingertip_pos.to(self.device) - cap_tip_pos.to(self.device), dim=-1
        )  # (N,)
        ee_far_enough = dist >= 0.05

        self._success |= (self._unlocked_once & ee_far_enough)

        if self.cfg.sun_program_evaluator_stage_override or bool(
            getattr(self.cfg, "ignore_stage_gate_for_success", False)
        ):
            return self._success.clone()

        past_stage = self.current_stage.flatten() > int(
            self.program_runtime.compiled_program.stage_count() / 2
        )
        return torch.logical_and(self._success, past_stage)
