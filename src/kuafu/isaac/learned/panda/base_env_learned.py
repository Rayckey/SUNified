# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
from dataclasses import MISSING, field
import os
from isaaclab.sensors.frame_transformer.frame_transformer import FrameTransformer
import torch
from omegaconf import OmegaConf
import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, RenderCfg, PhysxCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    combine_frame_transforms,
    quat_apply,
    quat_from_angle_axis,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    axis_angle_from_quat,
    sample_uniform,
    subtract_frame_transforms,
    axis_angle_from_quat,
)
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.sensors import TiledCamera, TiledCameraCfg, CameraCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from kuafu.mpc.simulation.sun_program_evaluator import SUNProgramEvaluator
from kuafu.mpc.simulation.spatial_utils import (
    quat_to_rotation_6d,
    rotation_6d_to_quat,
)
import time
from kuafu.common.simulation_utils import (
    combine_joint_states,
    extract_odom_manipulators,
    extract_frame_sensors,
)
from isaaclab.managers.recorder_manager import (
    RecorderManager,
)

@configclass
class UniformLinkPoseCommandCfg:
    asset: SceneEntityCfg = MISSING
    body_name: str = MISSING
    offset: OffsetCfg = MISSING

@configclass
class ArmWithGripperArticulationCfg(ArticulationCfg):
    gripper_opening_angle: float = MISSING
    gripper_closing_angle: float = MISSING
    gripper_actuated_joints: list = field(default_factory=list)

    arm_actuated_joints: list = field(default_factory=list)
    ee_command_pose: UniformLinkPoseCommandCfg = MISSING

    left_finger_link: str = MISSING
    right_finger_link: str = MISSING

    curobo_config_path: str = MISSING
    curobo_ee_link_name: str = MISSING

    base_link_name: str = MISSING
    frame_transformer_source_prim_path: str = MISSING
    frame_transformer_target_prim_path: str = MISSING
    frame_transformer_offset_pos: tuple[float, float, float] = MISSING
    frame_transformer_offset_rot: tuple[float, float, float, float] = MISSING

@configclass
class BaseLearnedEnvCfg(DirectRLEnvCfg):
    """Base configuration class for Panda manipulation environments.
    
    This class provides common configuration elements shared across all
    Panda manipulation tasks. Subclasses can override or extend these
    settings as needed.
    """
    episode_length_s: float = 10
    decimation: int = 4
    action_space: int = 9  # auto-set to 8 for "relative_ee" mode in __post_init__
    observation_space: int = 0  # Set from obs_mode in child config or via set_obs_mode().
    obs_mode: str = "state"  # "state" or "full_state"
    obs_dims: dict | None = None
    state_space: int = 0
    enable_cameras: bool = False

    def set_obs_mode(self, mode: str) -> None:
        """Set observation mode and matching observation_space."""
        self.obs_mode = mode
        if self.obs_dims is not None and mode in self.obs_dims:
            self.observation_space = int(self.obs_dims[mode])

    # Reward composition (all optional via config)
    use_absolute_progress_rewards: bool = False  # positive delta from best reward in current stage
    use_relative_progress_rewards: bool = True  # delta from previous state reward
    use_time_penalty: bool = True               # linear penalty -0.001 * step
    use_velocity_penalty: bool = False            # subtract joint velocity norm * scale
    regularize_velocity_scale: float = 0.01
    regularize_time_scale: float = 0.000
    progress_rewards_scale: float = 100.0

    # Generic SUNProgramEvaluator stage-transition bonus.  The historical behavior multiplies
    # the transition bonus by the remaining episode horizon; this is configurable
    # because that horizon-scaled proxy can dominate the learned objective on
    # long-horizon tasks.
    stage_transition_bonus_scale: float = 1.0
    stage_transition_bonus_use_remaining_horizon: bool = True
    penalize_false_completions: bool = False

    # Per-step diagnostics. Defaults preserve historical behavior; setting these
    # to 0/False removes stdout and per-timeout constraint reflection overhead
    # without changing reward or done calculations.
    reward_print_every: int = 100
    gripper_force_print_every: int = 0
    episode_outcome_print_every: int = 1
    collect_terminal_stuck_constraint_stats: bool = True

    # DEBUG tool
    debug_rewards = False
    sun_program_evaluator_stage_override: bool = False
    skip_dense_reward_computation: bool = False
    ignore_stage_gate_for_success: bool = False
    use_external_rewards: bool = False
    external_rewards_scale: float = 1.0

    # Simulation settings
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation, 
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

    # Used by child env CPU collision filtering. The actual table support
    # surface is the PandaTableEnvLearned floor collider.
    terrain = TerrainImporterCfg(
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

    # Stock IsaacLab Franka Panda with the built-in Panda hand gripper.
    _franka_panda_cfg = FRANKA_PANDA_HIGH_PD_CFG.copy()
    # Match the FR3/FinRay arm-controller PD gains.  Keep the Panda hand
    # actuator configuration from FRANKA_PANDA_HIGH_PD_CFG unchanged.
    _franka_panda_cfg.actuators["panda_shoulder"].stiffness = 200.0
    _franka_panda_cfg.actuators["panda_shoulder"].damping = 40.0
    _franka_panda_cfg.actuators["panda_forearm"].stiffness = 200.0
    _franka_panda_cfg.actuators["panda_forearm"].damping = 40.0
    # Match the deliberately reduced Kinova arm speed limits used by the
    # transferred Kuafu residual-agent configuration.
    _franka_panda_cfg.actuators["panda_shoulder"].velocity_limit_sim = 1.3963 / 2.0
    _franka_panda_cfg.actuators["panda_forearm"].velocity_limit_sim = 1.2218 / 2.0
    _franka_panda_cfg.prim_path = "/World/envs/env_.*/Robot"
    _franka_panda_cfg.spawn.activate_contact_sensors = False
    _franka_panda_cfg.spawn.articulation_props.enabled_self_collisions = False
    # Match the FR3/FinRay articulation solver settings for contact stability.
    _franka_panda_cfg.spawn.articulation_props.solver_position_iteration_count = 16
    _franka_panda_cfg.spawn.articulation_props.solver_velocity_iteration_count = 1

    FRANKA_PANDA_HAND_CFG = ArmWithGripperArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        base_link_name="panda_link0",
        gripper_opening_angle=0.04,
        gripper_closing_angle=0.0,
        gripper_actuated_joints=[
            "panda_finger_joint1",
            "panda_finger_joint2",
        ],
        arm_actuated_joints=[
            "panda_joint1",
            "panda_joint2",
            "panda_joint3",
            "panda_joint4",
            "panda_joint5",
            "panda_joint6",
            "panda_joint7",
        ],
        ee_command_pose=UniformLinkPoseCommandCfg(
            asset=SceneEntityCfg("robot"),
            body_name="panda_hand",
            offset=OffsetCfg(pos=(0.0, 0.0, 0.1034), rot=(1.0, 0.0, 0.0, 0.0)),
        ),
        left_finger_link="panda_leftfinger",
        right_finger_link="panda_rightfinger",
        curobo_config_path="",
        curobo_ee_link_name="panda_hand",
        frame_transformer_source_prim_path="/World/envs/env_.*/Robot/panda_link0",
        frame_transformer_target_prim_path="/World/envs/env_.*/Robot/panda_hand",
        frame_transformer_offset_pos=(0.0, 0.0, 0.1034),
        frame_transformer_offset_rot=(1.0, 0.0, 0.0, 0.0),
        spawn=_franka_panda_cfg.spawn,
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 1.0,
                "panda_joint2": -0.569,
                "panda_joint3": 0.0,
                "panda_joint4": -2.810,
                "panda_joint5": 0.0,
                "panda_joint6": 3.037,
                "panda_joint7": 0.741,
                "panda_finger_joint1": 0.04,
                "panda_finger_joint2": 0.04,
            },
        ),
        actuators=_franka_panda_cfg.actuators,
        soft_joint_pos_limit_factor=_franka_panda_cfg.soft_joint_pos_limit_factor,
    )

    robot = FRANKA_PANDA_HAND_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                "panda_joint1": 1.0,
                "panda_joint2": -0.569,
                "panda_joint3": 0.0,
                "panda_joint4": -2.810,
                "panda_joint5": 0.0,
                "panda_joint6": 3.037,
                "panda_joint7": 0.741,
                "panda_finger_joint1": 0.04,
                "panda_finger_joint2": 0.04,
            },
        ),
    )

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
                    pos=(0.0, 0.0, 0.1034),
                ),
            )
        ],
    )

    marker_cfg_7 = FRAME_MARKER_CFG.copy()
    marker_cfg_7.markers["frame"].scale = (0.03, 0.03, 0.03)
    marker_cfg_7.prim_path = "/Visuals/panda_link7_frame"
    l7_frame = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/panda_link0",
        debug_vis=False,
        visualizer_cfg=marker_cfg_7,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot/panda_link7",
                name="panda_link7",
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

    # ── Real camera sensors (calibrated extrinsics & intrinsics) ────────────
    # Source: deployment camera calibration from 2026-08-04 22:55:57,
    # using its OpenCV intrinsics and FR3-base extrinsics.
    # The real cameras are 1920x1080; sim renders at 1:3 scale (640x360).
    # The world frame is the Franka robot base, matching env-origin robots.
    # Quaternions are in Isaac's "world" camera convention: +X forward, +Y left, +Z up.
    # Real-sensor point-cloud/depth corrections are deployment-only and must not be
    # folded into these simulator camera poses.

    # camera0 — d415_0, RealSense serial 243722064996
    # Intrinsics: fx=452.706607, fy=453.235123, cx=321.258396, cy=189.568542
    table_cam_0 = TiledCameraCfg(
        prim_path="/World/envs/env_.*/table_cam_0",
        update_period=0.0,
        width=640,
        height=360,
        data_types=[
            "rgb",
            "distance_to_image_plane",
        ],
        spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
            intrinsic_matrix=[
                452.706607486801,
                0.0,
                321.258396020213,
                0.0,
                453.235122826526,
                189.568542320687,
                0.0,
                0.0,
                1.0,
            ],
            width=640,
            height=360,
            clipping_range=(0.01, 3.0),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.7795193592233347, 0.5382921825533387, 0.5418182700252682),
            rot=(
                0.24564771501094418,
                0.3544014180424764,
                0.15176530683209102,
                -0.8893953713852715,
            ),
            convention="world",
        ),
    )


    # camera1 — d415_1, RealSense serial 243722064409
    # Intrinsics: fx=456.671065, fy=459.665663, cx=326.470314, cy=182.771107
    table_cam_1 = TiledCameraCfg(
        prim_path="/World/envs/env_.*/table_cam_1",
        update_period=0.0,
        width=640,
        height=360,
        data_types=[
            "rgb",
            "distance_to_image_plane",
        ],
        spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
            intrinsic_matrix=[
                456.671064922986,
                0.0,
                326.470313780438,
                0.0,
                459.665662998423,
                182.771107200687,
                0.0,
                0.0,
                1.0,
            ],
            width=640,
            height=360,
            clipping_range=(0.01, 3.0),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.9406775187046168, 0.022460518792066342, 0.5980049879600953),
            rot=(
                0.04316511366358702,
                -0.43656027387536456,
                -0.08121159331312214,
                0.8949617742383306,
            ),
            convention="world",
        ),
    )

    # camera2 — d415_2, RealSense serial 323522060125
    # Intrinsics: fx=455.478348, fy=459.148763, cx=304.790332, cy=190.051627
    table_cam_2 = TiledCameraCfg(
        prim_path="/World/envs/env_.*/table_cam_2",
        update_period=0.0,
        width=640,
        height=360,
        data_types=[
            "rgb",
            "distance_to_image_plane",
        ],
        spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
            intrinsic_matrix=[
                455.478347763360,
                0.0,
                304.790332362715,
                0.0,
                459.148763061759,
                190.051627135047,
                0.0,
                0.0,
                1.0,
            ],
            width=640,
            height=360,
            clipping_range=(0.01, 3.0),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.24916075245455194, 0.8696591952145148, 0.5967097434855337),
            rot=(
                0.4876125193518985,
                0.4731155962532279,
                0.12570895350658848,
                -0.7229058877623874,
            ),
            convention="world",
        ),
    )

    action_scale: float = 15.0
    ee_action_scale: float = 3.0  # position scale for EE modes (separate from joint action_scale)

    # action mode: "absolute_joint" (MPC absolute targets), "relative_joint" (incremental, 9D),
    #              "relative_ee" (delta pos + axis-angle rotation, 8D), "absolute_ee" (target EE pose, 11D)
    action_mode: str = "absolute_ee"

    def __post_init__(self):
        # "relative_ee" uses 8D actions: [dx, dy, dz, ax, ay, az, grip1, grip2]
        if self.action_mode == "relative_ee":
            self.action_space = 8
        elif self.action_mode == "absolute_ee":
            self.action_space = 11

    # detectable_objects - must be set by child classes
    detectable_objects: dict = {}

    # mpc_config - must be set by child classes
    kuafu_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    script_dir = os.path.dirname(os.path.dirname(os.path.dirname(kuafu_dir)))
    mpc_config = {
        "controller_name": "FrankaIKController",
        "device": "cuda",
        "use_per_env_weight": False,
        "ee_frame": "panda_fingertip",
        "control_output_mode": "relative_joint",
        "load_program": f"{script_dir}/resources/learned_programs/manipulation-hang-cup/debug",
        "initial_residual": f"{script_dir}/src/kuafu/skills/constraints/rewards.py",
        "initial_predicate": f"{script_dir}/src/kuafu/skills/constraints/predicates.py",
        "gripper_opened": [0.04, 0.04],
        "gripper_closed": [0.0, 0.0],
        "gripper_max_force": 40.0,
        "robot_gripper_name": "franka_panda",
        "has_finger_contact_sensors": False,
        "object_grasp_debug": False,
        "object_grasp_debug_every": 25,
    }
    del script_dir
    # task_description - must be set by child classes
    task_description: str = ""

    # Optional: RecorderManager + HDF5 export (obs, actions, scene state, rewards, dones).
    # When set, dataset_export_dir_path and dataset_filename in the config control output path.
    recorders: Optional[LearnedEnvRecorderManagerCfg] = None


class BaseLearnedEnv(DirectRLEnv):
    cfg: BaseLearnedEnvCfg

    """Base class for Panda robot manipulation environments.
    
    This class provides common functionality for all Panda manipulation tasks,
    including robot initialization, action handling, and common utilities.
    """

    def __init__(self, cfg: BaseLearnedEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Simple timing prints (disabled by default). Initialize early because we call
        # update_data_in_controller() during __init__.
        self._profile = bool(int(os.getenv("KUAFU_PROFILE", "0")))
        self._profile_every = int(os.getenv("KUAFU_PROFILE_EVERY", "200"))
        self._profile_sync_cuda = bool(int(os.getenv("KUAFU_PROFILE_SYNC_CUDA", "0")))

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # Resolve the 9 actuated joint indices (7 arm + 2 gripper knuckle)
        robot_cfg = self.cfg.robot
        self._arm_joint_ids, self._arm_joint_names = self._robot.find_joints(
            list(robot_cfg.arm_actuated_joints)
        )
        self._gripper_joint_ids, self._gripper_joint_names = self._robot.find_joints(
            list(robot_cfg.gripper_actuated_joints)
        )
        self._arm_joint_ids_tensor = torch.tensor(
            self._arm_joint_ids, device=self.device, dtype=torch.long
        )
        self._gripper_joint_ids_tensor = torch.tensor(
            self._gripper_joint_ids, device=self.device, dtype=torch.long
        )

        actuated_joint_names = list(robot_cfg.arm_actuated_joints) + list(robot_cfg.gripper_actuated_joints)
        self._actuated_joint_ids, self._actuated_joint_names = self._robot.find_joints(actuated_joint_names)
        self._actuated_joint_ids_tensor = torch.tensor(self._actuated_joint_ids, device=self.device, dtype=torch.long)
        self._num_actuated_joints = len(self._actuated_joint_ids)

        # create auxiliary variables for computing applied action, observations and rewards
        # Full limits (all joints) - kept for reset logic
        self._robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[
            0, :, 0
        ].to(device=self.device)
        self._robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[
            0, :, 1
        ].to(device=self.device)

        # Actuated-only limits for action clamping
        self._actuated_dof_lower_limits = self._robot_dof_lower_limits[self._actuated_joint_ids_tensor]
        self._actuated_dof_upper_limits = self._robot_dof_upper_limits[self._actuated_joint_ids_tensor]

        # Targets for the 9 actuated joints only
        self._robot_dof_targets = torch.zeros(
            (self.num_envs, self._num_actuated_joints), device=self.device
        )

        # Completion tracker: counts how many times each environment has completed the task (persists across resets)
        self._completion_tracker = torch.zeros(
            (self.num_envs,), dtype=torch.long, device=self.device
        )

        # Cumulative per-stage transition counters (never reset; used for rolling-window stage completion).
        # Stored as device tensors after SUNProgramEvaluator is initialized; _episodes_started_cumul is
        # the stage-0 denominator used by rolling-window stage completion metrics.
        self._episodes_started_cumul: int = 0

        # End-to-end episode outcome stats (per-run; cleared via clear_end_to_end_rate_stats()).
        # Total counts across all envs (not per-env).
        # Stored as device scalars to avoid forcing CPU sync every step.
        self._e2e_true_success_total = torch.zeros(
            (), dtype=torch.long, device=self.device
        )
        self._e2e_false_complete_total = torch.zeros(
            (), dtype=torch.long, device=self.device
        )
        self._e2e_timeout_total = torch.zeros(
            (), dtype=torch.long, device=self.device
        )

        # Minimal configuration required by SUNProgramEvaluator and the MPC controller.
        # Get load_program from kwargs if provided, otherwise use default path
        if "robot_gripper_name" not in self.cfg.mpc_config:
            self.cfg.mpc_config["robot_gripper_name"] = "franka_panda"
        # Use the explicit Panda summed-arm-effort threshold instead of
        # falling through to kuafu's generic default.
        self.cfg.mpc_config.setdefault(
            "manipulator_high_effort_threshold", 10.0
        )
        has_finger_contact_sensors = self.cfg.mpc_config.get(
            "has_finger_contact_sensors", False
        )
        if has_finger_contact_sensors not in (False, 0):
            raise ValueError(
                "Panda learned envs must use joint-effort grasp detection: "
                "has_finger_contact_sensors must be False."
            )
        self.cfg.mpc_config["has_finger_contact_sensors"] = False
        if "object_grasp_debug" not in self.cfg.mpc_config:
            self.cfg.mpc_config["object_grasp_debug"] = False
        if "object_grasp_debug_every" not in self.cfg.mpc_config:
            self.cfg.mpc_config["object_grasp_debug_every"] = 25
        cfg_sun_program_evaluator = OmegaConf.create(self.cfg.mpc_config)
        # Add num_envs to config (required by SUNProgramEvaluator)
        cfg_sun_program_evaluator.num_envs = self.num_envs
        cfg_sun_program_evaluator.print_scope_functions_on_compile = (
            os.getenv("KUAFU_PRINT_PROGRAM_FUNCTIONS", "").strip().lower() in ("1", "true", "yes")
        )

        # Initialize SUNProgramEvaluator without auto-loading program

        program_runtime = SUNProgramEvaluator(cfg_sun_program_evaluator, auto_load_program=False)


        self.program_runtime = program_runtime
        # Now manually load program after all functions are set up
        if cfg_sun_program_evaluator.get("load_program"):
            self.program_runtime.load_program(cfg_sun_program_evaluator.load_program)
            env_name = self.__class__.__name__
            print(f"[{env_name}] Loaded program from: {cfg_sun_program_evaluator.load_program}")

        self._program_runtime_has_stages = int(self.program_runtime.compiled_program.stage_count()) > 0
        if self._program_runtime_has_stages:
            self.program_runtime.initialize_batch(self.num_envs)
        # current_stage must exist before update_data_in_controller (it calls _sync_stage_tensor)
        self.current_stage = torch.zeros(
            (self.num_envs, 1), device=self.device, dtype=torch.float32
        )
        self.update_data_in_controller()
        if self._program_runtime_has_stages:
            self._sync_stage_tensor()

        # Stage transition stats: average steps from one stage to the next, per stage.
        # NOTE: these are transition-event samples (not "env count"). Store on
        # device so rollout does not copy every transition to CPU or loop per env.
        n_stages = self._stage_count_or_default()
        self._stage_stats_sum_steps = torch.zeros(
            n_stages, device=self.device, dtype=torch.float64
        )
        self._stage_stats_n_samples = torch.zeros(
            n_stages, device=self.device, dtype=torch.long
        )
        self._stage_advanced_cumul_tensor = torch.zeros(
            n_stages, device=self.device, dtype=torch.long
        )
        self._prev_stage_indices = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        if self._program_runtime_has_stages:
            stage_indices_0 = self.program_runtime.get_stage_indices()
        else:
            stage_indices_0 = torch.zeros(
                self.num_envs, device=self.device, dtype=torch.long
            )
        self._prev_stage_indices.copy_(
            stage_indices_0.to(device=self.device, dtype=torch.long).view(-1)
        )
        step_val = (
            int(self.common_step_counter.item())
            if torch.is_tensor(self.common_step_counter)
            else int(self.common_step_counter)
        )
        self._stage_entered_step = torch.full(
            (self.num_envs,),
            float(step_val),
            device=self.device,
            dtype=torch.float64,
        )
        # Per-env highest stage reached (used to compute "how many envs moved past stage s").
        self._highest_stage_per_env = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )

        self._highest_stage_per_env.copy_(
            stage_indices_0.to(device=self.device, dtype=torch.long).view(-1)
        )
        # Per-run aggregate of constraint stats for envs that terminated via truncation
        # while still assigned to a stage. This is the signal used by Kuafu prompts.
        self._terminal_stuck_constraint_stats = {}

        self.robot_dof_speed_scales = torch.ones(
            (self._num_actuated_joints,), device=self.device
        )
        for gripper_joint_name in robot_cfg.gripper_actuated_joints:
            if gripper_joint_name in self._actuated_joint_names:
                self.robot_dof_speed_scales[self._actuated_joint_names.index(gripper_joint_name)] = 0.1

        # Max angular velocity for "relative_ee" mode, read from the MPC controller.
        self._max_ee_rotation_rad = float(self.program_runtime.mpc_controller.v_ang_max) * self.dt

        # Effort normalization scales for observations.
        # Arm efforts are on the order of 100N; gripper max force comes from MPC config / API.
        self._arm_effort_scale = float(self.cfg.mpc_config.get("arm_max_effort", 50.0))
        gripper_max_from_cfg = self.cfg.mpc_config.get("gripper_max_force", None)
        if gripper_max_from_cfg is not None:
            self._gripper_effort_scale = float(gripper_max_from_cfg)
        else:
            api = self.program_runtime.controller_api_wrapper
            self._gripper_effort_scale = float(
                getattr(api, "_gripper_max_force", 40.0)
            )
        self._gripper_contact_force_scale = self._gripper_effort_scale
        self._gripper_force_monitor_printed_names = False

        # Track highest reward per environment for progress-based rewards
        self._previous_reward_per_env = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )
        # `_get_dones()` advances stages before `_get_rewards()` consumes the
        # cached reward from the old stage. Defer the per-stage high-water reset
        # until that old-stage reward has been processed.
        self._stage_progress_reset_mask = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )

        self._stage_bonus = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )

        if self.cfg.debug_rewards:
            self._debug_cumulative_rewards = torch.zeros(
                self.num_envs, device=self.device, dtype=torch.float32
            )

        self._current_step_action = torch.zeros(
            (self.num_envs, self.single_action_space.shape[0]),
            device=self.device,
            dtype=torch.float32,
        )
        # Per-mode logged actions populated each step for the recorder.
        self._logged_joint_targets = torch.zeros((self.num_envs, 9), device=self.device)
        self._logged_relative_joint = torch.zeros((self.num_envs, 9), device=self.device)
        # Only populated in EE modes; None in joint modes so recorder terms skip them.
        self._logged_absolute_ee: torch.Tensor | None = None  # (N, 11): [pos(3), rot6d(6), grip(2)]
        self._logged_relative_ee: torch.Tensor | None = None  # (N, 8): [pos_delta(3), aa_delta(3), grip(2)]
        if self.cfg.action_mode in ("absolute_ee", "relative_ee"):
            self._logged_absolute_ee = torch.zeros((self.num_envs, 11), device=self.device)
            self._logged_relative_ee = torch.zeros((self.num_envs, 8), device=self.device)

        self._cached_success = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self._cached_failure = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        # Optional RecorderManager (HDF5 export via Isaac Lab recorder terms)
        self.recorder_manager = None
        if self.cfg.recorders is not None:
            # So RecorderManager can read success in record_pre_reset
            class _SuccessTerm:
                def __init__(self, env): self._env = env
                def __getitem__(self, env_ids): return self._env._cached_success[env_ids]
            class _FakeTerminationManager:
                def __init__(self, env): self._env = env; self._success_term = _SuccessTerm(env)
                @property
                def active_terms(self): return ["success"]
                def get_term(self, name): return self._success_term if name == "success" else None
            self.termination_manager = _FakeTerminationManager(self)
            self.recorder_manager = RecorderManager(self.cfg.recorders, self)
            print("[INFO] Recorder Manager (learned env): ", self.recorder_manager)

        self.reward_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)

        # IK controller for end-effector action modes
        self._ik_controller = None
        if self.cfg.action_mode in ("relative_ee", "absolute_ee"):
            self._setup_ik_controller()


    def _has_program_stages(self) -> bool:
        return bool(getattr(self, "_program_runtime_has_stages", False))

    def _stage_count_or_default(self) -> int:
        if not self._has_program_stages():
            return 1
        return max(1, int(self.program_runtime.compiled_program.stage_count()))

    def update_data_in_controller(self):
        """Update controller with latest odometry and sensor data."""
        sync_cuda = getattr(self, "_profile_sync_cuda", False)

        with torch.no_grad():
            t0 = time.perf_counter()
            if sync_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            odom = extract_odom_manipulators(self, self.cfg.detectable_objects)
            if sync_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            t_odom = time.perf_counter() - t0

            t0 = time.perf_counter()
            if sync_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            sensors = extract_frame_sensors(self, {}, self.cfg.detectable_objects)
            if sync_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            t_sensors = time.perf_counter() - t0

            t0 = time.perf_counter()
            self.program_runtime.callbackOdom(odom, sensors)
            t_cb_odom = time.perf_counter() - t0

            t0 = time.perf_counter()
            self.program_runtime.callbackJointState(combine_joint_states(self, robot="franka"))
            t_cb_js = time.perf_counter() - t0

            # Update POV state for environments that need it (first time in stage 0 or after reset)
            t0 = time.perf_counter()
            if self._has_program_stages():
                self.program_runtime.update_pov_if_needed()
            t_pov = time.perf_counter() - t0

        if self._has_program_stages():
            self._sync_stage_tensor()

    def close(self):
        super().close()

    def _setup_scene(self):
        """Setup the scene. Must be implemented by child classes."""
        self._robot = Articulation(self.cfg.robot)

        # Create cameras only if enabled
        if self.cfg.enable_cameras:
            self.table_cam_0 = TiledCamera(self.cfg.table_cam_0)
            self.table_cam_1 = TiledCamera(self.cfg.table_cam_1)
            self.table_cam_2 = TiledCamera(self.cfg.table_cam_2)

        self.ee_frame = FrameTransformer(self.cfg.ee_frame)
        self.l7_frame = FrameTransformer(self.cfg.l7_frame)
        self.l4_frame = FrameTransformer(self.cfg.l4_frame)

        if getattr(self.cfg, "terrain", None) is not None:
            self.cfg.terrain.num_envs = self.scene.cfg.num_envs
            self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
            self.scene._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.articulations["robot"] = self._robot

        self.scene.sensors["panda_fingertip"] = self.ee_frame
        self.scene.sensors["panda_link7"] = self.l7_frame
        self.scene.sensors["panda_link4"] = self.l4_frame

        # Register cameras only if enabled
        if self.cfg.enable_cameras:
            self.scene.sensors["table_cam_0"] = self.table_cam_0
            self.scene.sensors["table_cam_1"] = self.table_cam_1
            self.scene.sensors["table_cam_2"] = self.table_cam_2

    def _process_arm_actions(
        self,
        arm_cmd: torch.Tensor,
        jp_arm: torch.Tensor,
        debug_mode: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process 7D arm commands into joint targets and logged actions.

        - In debug mode, commands are absolute joint targets from MPC.
          We convert them to incremental actions, clamp to [-1, 1], and
          then re-synthesize rate-limited targets from the clamped deltas.
        - In learning mode, commands are already incremental in [-1, 1].
        """
        n_arm = arm_cmd.shape[-1]
        if debug_mode:
            # Clamp absolute targets to joint limits.
            arm_abs = arm_cmd.clone()

            # Convert to normalized incremental actions in [-1, 1].
            arm_incremental = (
                (arm_abs - jp_arm)
                / self.robot_dof_speed_scales[:n_arm]
                / self.dt
                / self.cfg.action_scale
            )

        else:
            # Learning: actions are already incremental; just clamp and apply.
            arm_incremental = arm_cmd
        arm_targets = (
            jp_arm
            + self.robot_dof_speed_scales[:n_arm]
            * self.dt
            * arm_incremental
            * self.cfg.action_scale
        )
        arm_targets = torch.clamp(
            arm_targets,
            self._robot_dof_lower_limits[:n_arm],
            self._robot_dof_upper_limits[:n_arm],
        )
        logged_arm = arm_incremental
        return arm_targets, logged_arm

    def _process_gripper_actions(
        self,
        grip_cmd: torch.Tensor,
        jp_grip: torch.Tensor,
        debug_mode: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process 2D gripper commands into joint targets and logged actions.

        Binary semantics for logging:
        - log +1 when target is (or was commanded) "open"
        - log -1 when target is (or was commanded) "close"
        - log  0 when holding current gripper joint positions.

        Debug vs learning:
        - In debug_mode, `grip_cmd` is already an absolute target from MPC in joint space.
          We apply it (clamped to limits) and *infer* open/close from proximity to the
          calibrated `_gripper_opened` / `_gripper_closed`.
        - In learning mode, `grip_cmd` is a user / policy command:
          >0 → open, <0 → close, ≈0 → hold.
        """
        api = self.program_runtime.controller_api_wrapper
        open_pos = api._gripper_opened.to(
            device=self.device, dtype=jp_grip.dtype
        ).view(1, -1)
        closed_pos = api._gripper_closed.to(
            device=self.device, dtype=jp_grip.dtype
        ).view(1, -1)

        if debug_mode:
            # Treat incoming command as absolute joint target from MPC.
            targets = grip_cmd.clone()

            # Determine whether this target is closer to "open" or "closed".
            # Compare per-finger distances to the calibrated positions.
            diff_open = (targets - open_pos.expand_as(targets)).abs()
            diff_closed = (targets - closed_pos.expand_as(targets)).abs()
            open_mask = diff_open < diff_closed
            close_mask = diff_closed < diff_open

            logged = torch.zeros_like(targets)
            logged = torch.where(open_mask, torch.ones_like(logged), logged)
            logged = torch.where(close_mask, -torch.ones_like(logged), logged)
        else:
            # Learning mode: interpret sign of command as open / close / hold.
            # Start from current joint positions.
            targets = jp_grip.clone()
            open_mask = (grip_cmd > 0).expand_as(targets)
            close_mask = (grip_cmd < 0).expand_as(targets)
            # Apply open / close commands using the calibrated gripper limits.
            targets = torch.where(open_mask, open_pos.expand_as(targets), targets)
            targets = torch.where(close_mask, closed_pos.expand_as(targets), targets)

            # Logged binary commands: +1 (open), -1 (close), 0 (hold).
            logged = torch.zeros_like(targets)
            logged = torch.where(open_mask, torch.ones_like(logged), logged)
            logged = torch.where(close_mask, -torch.ones_like(logged), logged)

        return targets, logged

    def _get_normalized_joint_effort(self, api, agent: str) -> torch.Tensor:
        """Return smoothly normalized joint efforts [7 arm, 2 gripper] for observations.

        - Arm efforts are scaled by an approximate max of ``arm_max_effort`` (default 100N).
        - Gripper efforts are scaled by ``gripper_max_force`` from mpc_config (or API wrapper).
        A smooth tanh nonlinearity keeps features in (-1, 1) without hard clamping.
        """
        effort = api.joint_dict[agent]["joint_effort"]  # (N, 9) = 7 arm + 2 gripper
        n_arm = 7
        arm_scaled = effort[:, :n_arm] / self._arm_effort_scale
        grip_scaled = effort[:, n_arm:] / self._gripper_effort_scale
        arm_eff = torch.tanh(arm_scaled)
        grip_eff = torch.tanh(grip_scaled)
        return torch.cat([arm_eff, grip_eff], dim=-1)

    def _get_normalized_gripper_contact(self, api) -> torch.Tensor:
        """Return normalized fingertip contact magnitudes [left, right] for observations."""
        left_contact, right_contact = api.finger_contact_force()
        gripper_contact = torch.stack([left_contact, right_contact], dim=-1)
        return gripper_contact / self._gripper_contact_force_scale

    def _print_gripper_force_diagnostics(self, env_name: str) -> None:
        """Print the signals and decisions used by Panda grasp detection."""
        api = self.program_runtime.controller_api_wrapper
        robot = self.scene.articulations["robot"]

        logical_effort = api.joint_dict[api.reference_agent]["joint_effort"][:, -2:].detach()
        logical_pos = robot.data.joint_pos.index_select(
            1, self._gripper_joint_ids_tensor
        ).detach()
        logical_target = self._robot_dof_targets[:, len(self._arm_joint_ids):].detach()
        physical_effort = robot.data.applied_torque.index_select(
            1, self._gripper_joint_ids_tensor
        ).detach()
        api_gripper_pos = api.joint_dict[api.reference_agent]["joint_position"][:, -2:].detach()
        api_gripper_vel = api.joint_dict[api.reference_agent].get("joint_velocity")
        if api_gripper_vel is not None:
            api_gripper_vel = api_gripper_vel[:, -2:].detach()

        predicate_contact = api.get_all_predicate_apis()["finger_contact_force"]
        left_contact, right_contact = predicate_contact()
        predicate_force = torch.stack([left_contact, right_contact], dim=-1).detach()

        openness = api.gripper_openness().detach().flatten()
        max_grasp_openness = float(getattr(api, "_gripper_max_grasp_openness", 0.8))
        gripper_max_force = float(getattr(api, "_gripper_max_force", 40.0))
        required_force = torch.full_like(openness, 5.0)

        within_grasp_position = (
            (openness > 0.1) & (openness <= max_grasp_openness)
        )
        velocity_threshold = float(
            getattr(api, "_gripper_effort_contact_velocity_threshold", 0.0)
        )
        if api_gripper_vel is None:
            max_velocity_abs = torch.zeros_like(openness)
            velocity_ok = torch.ones_like(within_grasp_position, dtype=torch.bool)
        else:
            max_velocity_abs = api_gripper_vel.abs().amax(dim=-1)
            velocity_ok = max_velocity_abs < velocity_threshold

        both_force_ok = (
            (predicate_force[:, 0] >= required_force)
            & (predicate_force[:, 1] >= required_force)
        )
        asymmetric_force_ok = torch.zeros_like(
            within_grasp_position, dtype=torch.bool
        )
        object_grasped = both_force_ok & within_grasp_position

        def _mean_pair(values: torch.Tensor) -> list[float]:
            return [float(v) for v in values.mean(dim=0).detach().cpu().tolist()]

        def _max_abs_pair(values: torch.Tensor) -> list[float]:
            return [
                float(v)
                for v in values.abs().amax(dim=0).detach().cpu().tolist()
            ]

        if not self._gripper_force_monitor_printed_names:
            print(
                f"[{env_name}][GripperForce] logical_joints={self._gripper_joint_names} "
                f"actuator_effort_limit={self.cfg.robot.actuators['panda_hand'].effort_limit_sim} "
                f"predicate_gripper_max_force={gripper_max_force:.6f} "
                f"predicate_panda_min_bilateral_effort=5.000000 "
                f"predicate_force_model=franka_panda_joint_effort"
            )
            self._gripper_force_monitor_printed_names = True

        step = int(self.common_step_counter)
        if self._has_program_stages():
            stage_indices = self.program_runtime.get_stage_indices()
        else:
            stage_indices = self.current_stage.view(-1).to(dtype=torch.long)
        stage_values = stage_indices.detach().cpu().tolist()
        print(
            f"[{env_name}][GripperForce] step={step} stage={stage_values} "
            f"logical_effort_abs_mean={_mean_pair(logical_effort.abs())} "
            f"logical_effort_abs_max={_max_abs_pair(logical_effort)} "
            f"physical_effort_abs_mean={_mean_pair(physical_effort.abs())} "
            f"logical_pos_mean={_mean_pair(logical_pos)} "
            f"api_gripper_pos_mean={_mean_pair(api_gripper_pos)} "
            f"logical_target_mean={_mean_pair(logical_target)}"
        )
        print(
            f"[{env_name}][GripperForce] step={step} stage={stage_values} "
            f"predicate_force_mean={_mean_pair(predicate_force)} "
            f"predicate_force_max={_max_abs_pair(predicate_force)} "
            f"openness_mean={float(openness.mean().detach().cpu()):.6f} "
            f"required_force_mean={float(required_force.mean().detach().cpu()):.6f} "
            f"max_velocity_abs_mean={float(max_velocity_abs.mean().detach().cpu()):.6f} "
            f"velocity_threshold={velocity_threshold:.6f}"
        )
        print(
            f"[{env_name}][GripperForce] step={step} stage={stage_values} "
            f"velocity_ok={velocity_ok.detach().cpu().tolist()} "
            f"position_ok={within_grasp_position.detach().cpu().tolist()} "
            f"both_force_ok={both_force_ok.detach().cpu().tolist()} "
            f"asymmetric_force_ok={asymmetric_force_ok.detach().cpu().tolist()} "
            f"object_grasped={object_grasped.detach().cpu().tolist()}"
        )

    def _setup_ik_controller(self):
        """Initialize the differential IK controller for end-effector action modes."""
        robot_cfg = self.cfg.robot
        joint_names = ["panda_joint.*"]
        body_names = ["panda_hand"]

        diff_ik_cfg = DifferentialIKControllerCfg(
            command_type="pose", use_relative_mode=False, ik_method="svd"
        )

        self._ik_controller = DifferentialIKController(
            diff_ik_cfg, num_envs=self.num_envs, device=self.device
        )
        robot_entity_cfg = SceneEntityCfg("robot", joint_names=joint_names, body_names=body_names)
        robot_entity_cfg.resolve(self.scene)
        self._ik_ee_body_id = robot_entity_cfg.body_ids[0]
        self._ik_ee_jacobi_idx = self._ik_ee_body_id - 1
        self._ik_joint_ids = robot_entity_cfg.joint_ids

        # Cache the inverse pose offset (panda_fingertip -> panda_hand).
        # At runtime, the current fingertip pose is read directly from the program_runtime API.
        # The inverse offset is used to convert the desired fingertip target back to
        # panda_hand pose for the IK controller.
        frame_cfg = self.cfg.ee_frame.target_frames[0].offset
        p_off = torch.tensor(frame_cfg.pos, device=self.device, dtype=torch.float32)  # (3,)
        q_off = torch.tensor(frame_cfg.rot, device=self.device, dtype=torch.float32)  # (4,) wxyz
        q_off_inv = quat_inv(q_off.unsqueeze(0)).squeeze(0)
        p_off_inv = quat_apply(q_off_inv.unsqueeze(0), -p_off.unsqueeze(0)).squeeze(0)
        self._ee_inv_offset_pos = p_off_inv    # (3,) fingertip to IK body in fingertip frame
        self._ee_inv_offset_quat = q_off_inv  # (4,) fingertip to IK body rotation

    def _process_ee_actions(self, ee_cmd: torch.Tensor) -> torch.Tensor:
        """Convert EE command to arm joint targets via IK.

        For "relative_ee": ee_cmd is 6D [dx, dy, dz, ax, ay, az].
            Translational deltas are interpreted in the robot base frame.
            Axis-angle deltas are applied on the current fingertip orientation.
        For "absolute_ee": ee_cmd is 9D [x, y, z, rot6d(6)], the desired robot-base-frame pose.

        Returns:
            joint_pos_des: (num_envs, n_arm_joints) desired joint positions for the arm.
        """
        ee_pose_w = self._robot.data.body_pose_w[:, self._ik_ee_body_id]
        cur_ee_pos_w = ee_pose_w[:, :3]
        cur_ee_quat_w = ee_pose_w[:, 3:7]

        if self.cfg.action_mode == "relative_ee":
            # ee_cmd: [dx, dy, dz, ax, ay, az]
            # Translation is in the robot base frame. Rotation remains a local delta on the fingertip pose.
            api = self.program_runtime.controller_api_wrapper
            cur_ft_pos_b = api.lookup_robot_frame_position("panda_fingertip")   # (N, 3)
            cur_ft_quat_b = api.lookup_robot_frame_rotation("panda_fingertip")  # (N, 4) wxyz

            # Fingertip-frame quantities (what we want to log/return).
            pos_delta_ft = ee_cmd[:, :3] * self.dt * self.cfg.ee_action_scale
            ft_pos_target = cur_ft_pos_b + pos_delta_ft
            aa_delta_ft = ee_cmd[:, 3:6] * self.cfg.ee_action_scale * self.dt
            angle = torch.norm(aa_delta_ft, dim=-1)  # (N, 1)
            axis = aa_delta_ft / (angle + 1e-8)                      # (N, 3)
            delta_quat_ft = quat_from_angle_axis(angle, axis)
            ft_quat_target = quat_mul(cur_ft_quat_b, delta_quat_ft)
            # Target link7 pose from target fingertip pose
            N = ee_cmd.shape[0]
            p_off_inv = self._ee_inv_offset_pos.unsqueeze(0).expand(N, -1)
            q_off_inv = self._ee_inv_offset_quat.unsqueeze(0).expand(N, -1)

            pos_target, quat_target = combine_frame_transforms(
                ft_pos_target, ft_quat_target, p_off_inv, q_off_inv
            )
            # Convert desired fingertip target -> panda_hand target for IK (IK-only).
            N = ee_cmd.shape[0]
            p_off_inv = self._ee_inv_offset_pos.unsqueeze(0).expand(N, -1)
            q_off_inv = self._ee_inv_offset_quat.unsqueeze(0).expand(N, -1)
            pos_target, quat_target = combine_frame_transforms(
                ft_pos_target, ft_quat_target, p_off_inv, q_off_inv
            )

            self._ik_controller.set_command(torch.cat([pos_target, quat_target], dim=-1))

            jacobian = self._robot.root_physx_view.get_jacobians()[
                :, self._ik_ee_jacobi_idx, :, self._ik_joint_ids
            ]
            root_pose_w = self._robot.data.root_pose_w
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7],
                cur_ee_pos_w, cur_ee_quat_w,
            )
            joint_pos_ik = self._robot.data.joint_pos[:, self._ik_joint_ids]
            joint_pos_des = self._ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos_ik)
            return joint_pos_des, ft_pos_target, ft_quat_target, pos_delta_ft, aa_delta_ft

        else:  # "absolute_ee"
            # ee_cmd: [x, y, z, rot6d(6)] — desired panda_fingertip pose in robot base frame.
            api = self.program_runtime.controller_api_wrapper
            cur_ft_pos_b = api.lookup_robot_frame_position("panda_fingertip")   # (N, 3)
            cur_ft_quat_b = api.lookup_robot_frame_rotation("panda_fingertip")  # (N, 4) wxyz
            ft_pos_target = ee_cmd[:, :3]
            ft_quat_target = rotation_6d_to_quat(ee_cmd[:, 3:9])
            # Derive relative action from current fingertip pose for logging.
            pos_delta = (ft_pos_target - cur_ft_pos_b) / self.dt / self.cfg.ee_action_scale
            # Compute local (body-frame) rotation delta such that:
            #   ft_quat_target ≈ cur_ft_quat_b * delta_quat
            quat_rel = quat_mul(quat_inv(cur_ft_quat_b), ft_quat_target)
            aa_raw = axis_angle_from_quat(quat_rel)
            aa_delta = aa_raw / self.dt / self.cfg.ee_action_scale
            # Convert desired fingertip target -> panda_hand target for IK (IK-only).
            N = ee_cmd.shape[0]
            p_off_inv = self._ee_inv_offset_pos.unsqueeze(0).expand(N, -1)
            q_off_inv = self._ee_inv_offset_quat.unsqueeze(0).expand(N, -1)
            pos_target, quat_target = combine_frame_transforms(
                ft_pos_target, ft_quat_target, p_off_inv, q_off_inv
            )

            self._ik_controller.set_command(torch.cat([pos_target, quat_target], dim=-1))

            jacobian = self._robot.root_physx_view.get_jacobians()[
                :, self._ik_ee_jacobi_idx, :, self._ik_joint_ids
            ]
            root_pose_w = self._robot.data.root_pose_w
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7],
                cur_ee_pos_w, cur_ee_quat_w,
            )
            joint_pos_ik = self._robot.data.joint_pos[:, self._ik_joint_ids]
            joint_pos_des = self._ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos_ik)
            return joint_pos_des, ft_pos_target, ft_quat_target, pos_delta, aa_delta

    # pre-physics step calls
    def _pre_physics_step(self, actions: torch.Tensor):
        jp = self.program_runtime.controller_api_wrapper.joint_dict[
            self.program_runtime.controller_api_wrapper.reference_agent
        ]["joint_position"]

        n_arm = 7
        mode = self.cfg.action_mode

        if mode in ("absolute_ee", "relative_ee"):
            # "relative_ee": actions [dx,dy,dz, ax,ay,az, g1,g2] (8D)
            # "absolute_ee": actions [x,y,z, rot6d(6), g1,g2] (11D)
            ee_dims = 6 if mode == "relative_ee" else 9
            arm_targets, pos_target, quat_target, pos_delta, aa_delta = self._process_ee_actions(actions[:, :ee_dims])

            logged_arm = (arm_targets - jp[:, :n_arm]) / self.robot_dof_speed_scales[:n_arm]/ self.dt/ self.cfg.action_scale

            arm_targets = torch.clamp(
                arm_targets,
                self._actuated_dof_lower_limits[:n_arm],
                self._actuated_dof_upper_limits[:n_arm],
            )
            grip_cmd = actions[:, ee_dims:ee_dims + 2]
        else:
            arm_cmd = actions[:, :n_arm]
            grip_cmd = actions[:, n_arm:9]
            arm_targets, logged_arm = self._process_arm_actions(
                arm_cmd, jp[:, :n_arm], debug_mode=("absolute" in mode)
            )

        gripper_targets, logged_grip = self._process_gripper_actions(
            grip_cmd, jp[:, n_arm:9], debug_mode=("absolute" in mode)
        )

        self._robot_dof_targets[:] = torch.cat([arm_targets, gripper_targets], dim=-1)

        if self.recorder_manager is not None:
            self._logged_joint_targets[:] = torch.cat([arm_targets, gripper_targets], dim=-1)
            self._logged_relative_joint[:] = torch.cat([logged_arm, logged_grip], dim=-1)
            if mode in ("absolute_ee", "relative_ee"):
                assert self._logged_absolute_ee is not None and self._logged_relative_ee is not None
                self._logged_absolute_ee[:] = torch.cat(
                    [pos_target, quat_to_rotation_6d(quat_target), gripper_targets], dim=-1
                )
                self._logged_relative_ee[:] = torch.cat([pos_delta, aa_delta, logged_grip], dim=-1)
            # Keep _current_step_action as the raw policy action for backward compat.
            self._current_step_action = torch.cat([logged_arm, logged_grip], dim=-1)
            self.recorder_manager.record_pre_step()

    def step(self, actions: torch.Tensor):
        obs, rew, terminated, truncated, info = super().step(actions)
        # record o_{t+1} (as next_obs)
        if self.recorder_manager is not None:
            self.recorder_manager.record_post_step()
        return obs, rew, terminated, truncated, info

    def reset(self, seed: int | None = None, options: dict | None = None):
        self.obs_buf, x = super().reset(seed=seed, options=options)
        return self.obs_buf, x

    def _apply_action(self):
        self._robot.set_joint_position_target(self._robot_dof_targets, joint_ids=self._actuated_joint_ids)

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset environments. Child classes should call super() and add task-specific resets."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if self.recorder_manager is not None and len(env_ids) > 0:
            self.recorder_manager.record_pre_reset(env_ids)

        super()._reset_idx(env_ids)

        self._reset_robot(env_ids)

        if self.cfg.debug_rewards:
            env_name = self.__class__.__name__
            cumulative = self._debug_cumulative_rewards[env_ids]
            print(f"[{env_name}][DebugReward] average episode reward: {cumulative.mean().item():.4f}")
            self._debug_cumulative_rewards[env_ids] = 0.0

        if self._has_program_stages():
            self.program_runtime.reset_stage_indices(env_ids)
            self._sync_stage_tensor()
        else:
            self.current_stage[env_ids] = 0.0
        step_val = (
            int(self.common_step_counter.item())
            if torch.is_tensor(self.common_step_counter)
            else int(self.common_step_counter)
        )
        self._prev_stage_indices[env_ids] = 0
        self._stage_entered_step[env_ids] = float(step_val)
        self._episodes_started_cumul += int(len(env_ids))
        # Reset highest reward per env for progress-based rewards
        self._previous_reward_per_env[env_ids] = 0.0
        self._stage_progress_reset_mask[env_ids] = False
        # Need to refresh the intermediate values so that _get_observations() can use the latest values

        self._get_observations()

        if self.recorder_manager is not None and len(env_ids) > 0:
            self.recorder_manager.record_post_reset(env_ids)


    def _reset_robot(self, env_ids: torch.Tensor | None):
        """Reset robot to initial state.
        
        This is a helper method that can be called by subclasses in their
        _reset_idx implementation.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        assert env_ids is not None

        # Reset rule: arm joints around default; gripper + passive joints always exactly 0.
        num = int(env_ids.shape[0])
        joint_pos = torch.zeros((num, self._robot.num_joints), device=self.device, dtype=torch.float32)

        arm_default = self._robot.data.default_joint_pos[env_ids][:, self._arm_joint_ids_tensor]
        arm_noise = sample_uniform(-0.2, 0.2, arm_default.shape, self.device)
        arm_pos = arm_default + arm_noise
        arm_lower = self._robot_dof_lower_limits[self._arm_joint_ids_tensor]
        arm_upper = self._robot_dof_upper_limits[self._arm_joint_ids_tensor]
        arm_pos = torch.clamp(arm_pos, arm_lower, arm_upper)
        joint_pos[:, self._arm_joint_ids_tensor] = arm_pos

        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # Reset the actuated-only dof targets to match the reset positions
        self._robot_dof_targets[env_ids] = joint_pos[:, self._actuated_joint_ids_tensor]

    def _sync_stage_tensor(self):
        """Sync the current stage tensor with the program_runtime's stage indices."""
        if not self._has_program_stages():
            self.current_stage.zero_()
            return
        stage_indices = self.program_runtime.get_stage_indices()
        if stage_indices is None:
            return
        stage_indices = stage_indices.to(
            self.device, dtype=self.current_stage.dtype
        ).view(-1, 1)
        self.current_stage.copy_(stage_indices)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminate_external = self._external_get_dones()
        # Time out
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        additional_time_outs = getattr(self, "_get_additional_time_outs", None)
        if additional_time_outs is not None:
            time_out = torch.logical_or(time_out, additional_time_outs())
        use_external_rewards = bool(getattr(self.cfg, "use_external_rewards", False))
        skip_dense_rewards = bool(getattr(self.cfg, "skip_dense_reward_computation", False)) or use_external_rewards
        ############################
        # External/manual rewards should replace only the scalar reward returned
        # to RL. Stage-aware BC priors still need SUNProgramEvaluator predicates to advance
        # current_stage; compute_rewards=False keeps that stage machine alive
        # without paying for dense residual reward computation.
        if self._has_program_stages():
            with torch.no_grad():
                satisfied, past_final_stage, state_costs = self.program_runtime.check_and_advance_stages_and_compute_rewards(
                    compute_rewards=not skip_dense_rewards
                )
                new_stages = self.program_runtime.get_stage_indices()
                new_stages_flat = new_stages.to(device=self.device, dtype=torch.long).view(-1)
                if skip_dense_rewards:
                    self._stage_bonus.zero_()
                else:
                    self._stage_bonus = satisfied.to(dtype=torch.float32) * new_stages_flat.to(dtype=torch.float32)
                self._stage_progress_reset_mask.copy_(satisfied)

            # Cache reward inputs for _get_rewards() (DirectRLEnv calls _get_dones()
            # before _get_rewards()). With rewards skipped, _get_rewards() returns
            # zeros and this cache is only kept for consistency/debugging.
            if skip_dense_rewards:
                self._cached_rewards = torch.zeros(
                    self.num_envs, device=self.device, dtype=self._previous_reward_per_env.dtype
                )
            else:
                self._cached_rewards = 1.0 - state_costs

            # Update stage transition stats: average steps from one stage to the next.
            # Accumulate on device; convert to Python only in getters used by logging.
            transitioned = new_stages_flat > self._prev_stage_indices
            transitioned_long = transitioned.to(dtype=torch.long)
            if torch.is_tensor(self.common_step_counter):
                step_val_t = self.common_step_counter.to(device=self.device, dtype=self._stage_entered_step.dtype)
            else:
                step_val_t = torch.tensor(float(self.common_step_counter), device=self.device, dtype=self._stage_entered_step.dtype)
            steps_in_prev = (step_val_t - self._stage_entered_step) * transitioned.to(dtype=self._stage_entered_step.dtype)
            self._stage_stats_sum_steps.scatter_add_(0, self._prev_stage_indices, steps_in_prev)
            self._stage_stats_n_samples.scatter_add_(0, self._prev_stage_indices, transitioned_long)
            self._stage_advanced_cumul_tensor.scatter_add_(0, self._prev_stage_indices, transitioned_long)
            self._stage_entered_step.copy_(
                torch.where(transitioned, step_val_t.expand_as(self._stage_entered_step), self._stage_entered_step)
            )
            self._prev_stage_indices.copy_(new_stages_flat)
            # Track highest stage each env has reached (for "ratio of envs moved past stage").
            self._highest_stage_per_env = torch.maximum(
                self._highest_stage_per_env, new_stages_flat
            )
        else:
            past_final_stage = torch.zeros_like(terminate_external, dtype=torch.bool)
            self._stage_bonus.zero_()
            self._cached_rewards = torch.zeros(
                self.num_envs, device=self.device, dtype=self._previous_reward_per_env.dtype
            )


        # Track completion count before reset happens (persists across resets, increments on each completion)
        self._completion_tracker = self._completion_tracker + terminate_external.long()

        env_name = self.__class__.__name__
        false_finished = past_final_stage if not self.cfg.sun_program_evaluator_stage_override else torch.zeros_like(past_final_stage)
        outcome_print_every = int(self.cfg.episode_outcome_print_every)
        if outcome_print_every > 0 and int(self.common_step_counter) % outcome_print_every == 0:
            if terminate_external.any().item():
                print(
                    f"[{env_name}] Successfully finished #: {torch.sum(terminate_external).cpu().numpy()}"
                )
            if false_finished.any().item():
                print(f"[{env_name}] False finished #: {torch.sum(false_finished).cpu().numpy()}")


        # End-to-end episode outcome totals (per-run).
        # - true_success: external termination condition
        # - false_complete: program_runtime marked past final stage without true success
        # - timeout: episode hit max length
        self._e2e_true_success_total = self._e2e_true_success_total + terminate_external.to(torch.long).sum()
        self._e2e_false_complete_total = self._e2e_false_complete_total + false_finished.to(torch.long).sum()
        self._e2e_timeout_total = self._e2e_timeout_total + time_out.to(torch.long).sum()

        # Cache terminal flags for reward shaping in _get_rewards()
        # (DirectRLEnv calls _get_dones() before _get_rewards()).
        self._cached_success = terminate_external
        self._cached_failure = false_finished

        # Treat both success and failure as MDP terminations.
        # Only time-outs are truncations (so algorithms can bootstrap correctly).
        terminated = torch.logical_or(terminate_external, false_finished)
        broke_physics = self._get_truncated()
        truncated = torch.logical_or(time_out, broke_physics)
        if self.cfg.collect_terminal_stuck_constraint_stats and not skip_dense_rewards:
            stuck_terminal = torch.logical_and(truncated, torch.logical_not(terminated))
            if stuck_terminal.any():
                self._accumulate_terminal_stuck_constraint_stats(stuck_terminal)

        if self._has_program_stages():
            self._sync_stage_tensor()
        return terminated, truncated

    def _get_truncated(self) -> torch.Tensor:
        api = self.program_runtime.controller_api_wrapper
        try:
            joint_vel = api.joint_dict[api.reference_agent]["joint_velocity"]
        except (KeyError, TypeError):
            return torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        max_vel = torch.max(torch.abs(joint_vel), dim=-1).values
        return max_vel > torch.pi * 2

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards based on cost and stage progression."""
        if bool(getattr(self.cfg, "use_external_rewards", False)):
            raw_external_rewards = self.external_rewards().to(device=self.device, dtype=torch.float32).view(-1)
            if raw_external_rewards.shape[0] != self.num_envs:
                raise RuntimeError(
                    f"external_rewards returned shape {tuple(raw_external_rewards.shape)}, expected ({self.num_envs},)"
                )
            self._last_external_rewards = raw_external_rewards.detach().clone()
            rewards = raw_external_rewards * float(getattr(self.cfg, "external_rewards_scale", 1.0))

            # External rewards replace only the dense per-step reward. A true task
            # completion must still dominate the return, otherwise successful
            # early terminations can score lower than timeouts that collect dense
            # shaping for the full horizon.
            success = getattr(self, "_cached_success")
            failure = getattr(self, "_cached_failure")
            max_stage = self._stage_count_or_default()
            highest_reward_per_frame = max_stage
            remaining_frames = (self.max_episode_length - self.episode_length_buf).to(dtype=rewards.dtype)
            terminal_bonus = remaining_frames * highest_reward_per_frame
            rewards = rewards + success.to(dtype=rewards.dtype) * terminal_bonus
            if self.cfg.penalize_false_completions:
                rewards = rewards - failure.to(dtype=rewards.dtype) * terminal_bonus

            self._last_returned_rewards = rewards.detach().clone()
            if self.cfg.debug_rewards:
                self._debug_cumulative_rewards += rewards
            return rewards

        if bool(getattr(self.cfg, "skip_dense_reward_computation", False)) or not self._has_program_stages():
            return torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)

        # Use cost cached from _get_dones() to avoid a second full residual computation.
        _rewards = getattr(self, "_cached_rewards", None)
        if _rewards is None:
            # Fallback if called out of order
            raise ValueError("Rewards not cached")

        # Compute state reward (per-frame reward, kept as-is)
        state_rewards = _rewards 

        # Absolute progress is the positive delta from the best reward reached
        # in the current stage. Relative progress is the positive one-step delta.
        # Convert to float32 for comparison
        state_rewards_f32 = state_rewards.to(dtype=torch.float32)

        # Default to dense state rewards only when no progress shaping is enabled.
        rewards = torch.zeros_like(state_rewards, device=state_rewards.device, dtype=state_rewards.dtype)
        if self.cfg.use_absolute_progress_rewards or self.cfg.use_relative_progress_rewards:
            progress_rewards = torch.clamp(
                state_rewards_f32 - self._previous_reward_per_env,
                min=0.0,
            )
            rewards = rewards + progress_rewards * self.cfg.progress_rewards_scale
            if self.cfg.use_relative_progress_rewards:
                self._previous_reward_per_env = state_rewards_f32.clone()
            else:
                self._previous_reward_per_env = torch.maximum(self._previous_reward_per_env, state_rewards_f32)
        else:
            rewards = rewards + state_rewards

        # Stage advancement happened in `_get_dones()`, but `state_rewards`
        # above belongs to the stage that just completed. Reset only after
        # consuming it so the next stage starts with a fresh high-water mark.
        self._previous_reward_per_env[self._stage_progress_reset_mask] = 0.0
        self._stage_progress_reset_mask.zero_()

        if self.cfg.use_velocity_penalty:
            vel_penalty = self.regularize_rewards().to(dtype=state_rewards.dtype, device=state_rewards.device)
            rewards = rewards - vel_penalty
        if self.cfg.use_time_penalty:
            time_penalty = self.cfg.regularize_time_scale * self.episode_length_buf.to(dtype=state_rewards.dtype)
            rewards = rewards - time_penalty

        # Terminal shaping: grant the best-case reward for all remaining frames
        # on true task success. Optionally restore the historical false-complete
        # penalty for tasks where passing SUNProgramEvaluator without external success should
        # be strongly discouraged.
        success = getattr(self, "_cached_success")
        failure = getattr(self, "_cached_failure")
        max_stage = self._stage_count_or_default()
        highest_reward_per_frame =  max_stage 
        remaining_frames = (self.max_episode_length - self.episode_length_buf).to(dtype=state_rewards.dtype)
        terminal_bonus = remaining_frames * highest_reward_per_frame
        rewards = rewards + success.to(dtype=state_rewards.dtype) * terminal_bonus
        if self.cfg.penalize_false_completions:
            rewards = rewards - failure.to(dtype=state_rewards.dtype) * terminal_bonus

        # Optional stdout diagnostics. The reductions/.item() calls synchronize CUDA,
        # so training configs can set reward_print_every: 100 without changing rewards.
        reward_print_every = int(self.cfg.reward_print_every)
        if reward_print_every > 0 and int(self.common_step_counter) % reward_print_every == 0:
            average_state_rewards = torch.mean(state_rewards).item()
            highest_state_rewards = torch.max(state_rewards).item()
            env_name = self.__class__.__name__
            print(
                f"[{env_name}] Step {self.common_step_counter}, Average state rewards = {average_state_rewards:.6f}), Highest state rewards = {highest_state_rewards:.6f})"
            )

            if self.cfg.use_relative_progress_rewards:
                avg_progress_rewards = torch.mean(progress_rewards).item()
                highest_progress_rewards = torch.max(progress_rewards).item()
                print(
                    f"[{env_name}] Average progress rewards = {avg_progress_rewards:.6f}), Highest progress rewards = {highest_progress_rewards:.6f})"
                )

            if self.cfg.use_time_penalty:
                avg_time_penalty = torch.mean(time_penalty).item()
                print(
                    f"[{env_name}] Average time penalty = {avg_time_penalty:.6f})"
                )
            if self.cfg.use_velocity_penalty:
                avg_vel_penalty = torch.mean(vel_penalty).item()
                print(
                    f"[{env_name}] Average velocity penalty = {avg_vel_penalty:.6f})"
                )
            if self._has_program_stages():
                stage_numbers = self.program_runtime.get_stage_indices()
            else:
                stage_numbers = self.current_stage.view(-1).to(dtype=torch.long)
            print(
                f"[{env_name}] Average stage = {torch.mean(stage_numbers.to(torch.float32)).item()}, Max stage = {torch.max(stage_numbers).item()}"
            )

        gripper_force_print_every = int(
            getattr(self.cfg, "gripper_force_print_every", 0)
        )
        if (
            gripper_force_print_every > 0
            and int(self.common_step_counter) % gripper_force_print_every == 0
        ):
            self._print_gripper_force_diagnostics(self.__class__.__name__)

        stage_transition_bonus = self._stage_bonus.to(dtype=state_rewards.dtype)
        if self.cfg.stage_transition_bonus_use_remaining_horizon:
            stage_transition_bonus = stage_transition_bonus * remaining_frames
        rewards = rewards + stage_transition_bonus * float(self.cfg.stage_transition_bonus_scale)

        if self.cfg.debug_rewards:
            self._debug_cumulative_rewards += rewards

        return rewards

    def external_rewards(self) -> torch.Tensor:
        """Manual task reward used by no-program ablations."""
        raise NotImplementedError(f"{self.__class__.__name__} must implement external_rewards().")

    def regularize_rewards(self) -> torch.Tensor:
        api = self.program_runtime.controller_api_wrapper
        agent = api.reference_agent
        joint_vel = api.joint_dict[agent]["joint_velocity"]
        penalty = torch.norm(joint_vel, dim=-1) * self.cfg.regularize_velocity_scale
        return penalty

    def get_highest_stage_reached_per_env(self) -> list[int]:
        """Return the highest stage reached per env since the last clear.

        This is the canonical signal used to compute "how many envs have moved past stage s":
        `sum(highest_stage_reached_per_env > s)`.
        """
        return [int(x) for x in self._highest_stage_per_env.detach().cpu().view(-1).tolist()]

    def get_stage_completion_cumulative(self) -> tuple[dict[int, int], int]:
        """Return (stage_advanced_cumulative, episodes_started_cumulative).

        stage_advanced_cumulative[N] = total times any env advanced past stage N (never reset).
        episodes_started_cumulative  = total env resets across all time (denominator for stage 0).
        Used to compute rolling-window per-stage completion rates by differencing these counters.
        """
        advanced = self._stage_advanced_cumul_tensor.detach().cpu().view(-1).tolist()
        return {s: int(v) for s, v in enumerate(advanced) if int(v) != 0}, self._episodes_started_cumul

    def get_stage_transition_stats(self) -> dict[int, dict]:
        """Return per-stage statistics for prompt/debugging.

        NOTE: This intentionally does NOT include a per-stage "count". Use
        `get_highest_stage_reached_per_env()` to compute env-pass ratios.

        Returns:
            Dict mapping stage_index -> {"avg_steps": float | None, "samples": int}.
            - "avg_steps": average steps to complete this stage (from transition events), or None if no samples.
            - "samples": number of transition-event samples used to compute avg_steps.
        """
        n_stages = self._stage_count_or_default()
        sum_steps = self._stage_stats_sum_steps.detach().cpu().view(-1).tolist()
        n_samples = self._stage_stats_n_samples.detach().cpu().view(-1).tolist()
        result = {}
        for s in range(n_stages):
            samples = int(n_samples[s]) if s < len(n_samples) else 0
            if samples > 0:
                result[s] = {
                    "avg_steps": float(sum_steps[s]) / samples,
                    "samples": samples,
                }
            else:
                result[s] = {
                    "avg_steps": None,
                    "samples": 0,
                }
        return result

    def clear_end_to_end_rate_stats(self) -> None:
        """Clear end-to-end episode outcome stats for the current run."""
        self._e2e_true_success_total.zero_()
        self._e2e_false_complete_total.zero_()
        self._e2e_timeout_total.zero_()

    def clear_terminal_stuck_constraint_stats(self) -> None:
        """Clear per-stage constraint stats for envs that terminated while stuck."""
        self._terminal_stuck_constraint_stats.clear()

    def get_true_success_count(self) -> int:
        """Return total number of true successes since last clear."""
        return int(self._e2e_true_success_total.item())

    def get_false_complete_count(self) -> int:
        """Return total number of false completes since last clear."""
        return int(self._e2e_false_complete_total.item())

    def get_timeout_count(self) -> int:
        """Return total number of timeouts since last clear."""
        return int(self._e2e_timeout_total.item())

    def get_last_external_rewards(self) -> torch.Tensor | None:
        """Return raw external rewards from the last reward computation, before terminal resets."""
        return getattr(self, "_last_external_rewards", None)

    def get_true_success_rate(self) -> float | None:
        """Return true-success rate since last clear, or None if no completed episodes exist."""
        true_success_count = int(self._e2e_true_success_total.item())
        false_complete_count = int(self._e2e_false_complete_total.item())
        timeout_count = int(self._e2e_timeout_total.item())
        total_completed_episodes = true_success_count + false_complete_count + timeout_count
        if total_completed_episodes == 0:
            return None
        return true_success_count / total_completed_episodes

    def clear_stage_transition_stats(self) -> None:
        """Clear all stage transition statistics. Use this to restart tracking whenever needed."""
        self._stage_stats_sum_steps.zero_()
        self._stage_stats_n_samples.zero_()
        self._stage_advanced_cumul_tensor.zero_()
        if self._has_program_stages():
            stage_indices = self.program_runtime.get_stage_indices()
        else:
            stage_indices = self.current_stage.view(-1).to(dtype=torch.long)
        self.clear_end_to_end_rate_stats()
        self.clear_terminal_stuck_constraint_stats()
        if stage_indices is not None:
            stage_indices_flat = stage_indices.to(device=self.device, dtype=torch.long).view(-1)
            self._prev_stage_indices.copy_(stage_indices_flat)
            self._highest_stage_per_env.copy_(stage_indices_flat)
            if torch.is_tensor(self.common_step_counter):
                step_val_t = self.common_step_counter.to(device=self.device, dtype=self._stage_entered_step.dtype)
            else:
                step_val_t = torch.tensor(float(self.common_step_counter), device=self.device, dtype=self._stage_entered_step.dtype)
            self._stage_entered_step.fill_(float(step_val_t.item()))

    def _accumulate_terminal_stuck_constraint_stats(self, stuck_mask: torch.Tensor) -> None:
        """Accumulate per-constraint stats from envs that truncated while stuck in a stage."""
        if not self._has_program_stages():
            return
        stuck_mask = stuck_mask.to(device=self.device, dtype=torch.bool).view(-1)
        if not torch.any(stuck_mask):
            return
        with torch.no_grad():
            per_constraint_sat, per_constraint_rewards = (
                self.program_runtime.get_per_constraint_satisfaction_and_rewards()
            )
        for stage_id, constraints_dict in per_constraint_sat.items():
            rewards_dict = per_constraint_rewards[stage_id]
            for constraint_id, (stage_mask, satisfied) in constraints_dict.items():
                stage_stuck_mask = stuck_mask[stage_mask]
                total = int(stage_stuck_mask.sum().item())
                if total <= 0:
                    continue
                met = int(satisfied[stage_stuck_mask].sum().item())
                _, reward_tensor = rewards_dict[constraint_id]
                reward_tensor = reward_tensor.view(-1)
                if reward_tensor.numel() != stage_stuck_mask.numel():
                    raise ValueError(
                        f"Stage {stage_id} constraint {constraint_id}: reward tensor shape "
                        f"{tuple(reward_tensor.shape)} does not match stage mask length "
                        f"{stage_stuck_mask.numel()} while accumulating terminal stuck stats."
                    )
                reward_sum = float(reward_tensor[stage_stuck_mask].sum().item())
                stage_stats = self._terminal_stuck_constraint_stats.setdefault(int(stage_id), {})
                stats = stage_stats.setdefault(
                    int(constraint_id),
                    {"met": 0, "total": 0, "reward_sum": 0.0},
                )
                stats["met"] = int(stats["met"]) + met
                stats["total"] = int(stats["total"]) + total
                stats["reward_sum"] = float(stats["reward_sum"]) + reward_sum

    def get_constraint_stats(
        self,
    ) -> dict[int, dict[int, dict[str, float | int]]]:
        """Return per-stage, per-constraint stats for envs that terminated while stuck.

        Returns:
            Dict[stage_id, Dict[constraint_id, {"rate": float, "met": int, "total": int, "reward": float}]].
            Only (stage, constraint) pairs with at least one stuck terminal sample (total > 0).
            rate = met / total; reward = mean weighted residual (residual × weight) over
            terminal stuck envs for that stage/constraint. Reward can exceed 1 when the
            constraint weight > 1.
        """
        out = {}
        for stage_id, constraints_dict in self._terminal_stuck_constraint_stats.items():
            for constraint_id, stats in constraints_dict.items():
                total = int(stats["total"])
                if total <= 0:
                    continue
                met = int(stats["met"])
                reward_sum = float(stats["reward_sum"])
                if stage_id not in out:
                    out[stage_id] = {}
                out[stage_id][constraint_id] = {
                    "rate": met / total,
                    "met": met,
                    "total": total,
                    "reward": reward_sum / float(total),
                }
        return out

    def _external_get_dones(self):
        """Check external termination conditions. Must be implemented by child classes."""
        raise NotImplementedError("Child classes must implement _external_get_dones()")
