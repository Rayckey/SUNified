# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
from typing import Optional
import os
from isaaclab.sensors.frame_transformer.frame_transformer import FrameTransformer
import torch
import time
from pxr import Usd, UsdGeom, UsdPhysics

from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.utils.torch.transformations import tf_combine, tf_inverse, tf_vector

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObjectCfg, RigidObject
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, RenderCfg, PhysxCfg
from isaaclab.sim import schemas
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.math import (
    axis_angle_from_quat,
    combine_frame_transforms,
    quat_apply,
    quat_from_angle_axis,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    subtract_frame_transforms,
)
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.sensors import TiledCamera, TiledCameraCfg, CameraCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import math as math_utils
from omegaconf import OmegaConf
from kuafu.mpc.simulation.sun_program_evaluator import SUNProgramEvaluator
from kuafu.mpc.simulation.spatial_utils import (
    quat_to_rotation_6d,
    rotation_6d_to_quat,
)
from kuafu.common.simulation_utils import (
    combine_joint_states,
    extract_odom_manipulators,
    extract_frame_sensors,
)
from isaaclab.managers.recorder_manager import (
    RecorderManager,
)
from isaaclab.managers import SceneEntityCfg
from kuafu.isaac.tasks.recorder_learned import LearnedEnvRecorderManagerCfg

ISAAC_NUCLEUS_DIR = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac"
TABLE_HEIGHT = 0.1

@configclass
class BaseLearnedEnvCfg(DirectRLEnvCfg):
    """Base configuration class for learned manipulation environments.
    
    This class contains common configuration elements shared across all learned
    manipulation environments. Child classes should inherit from this and add
    task-specific configurations.
    """
    # env
    episode_length_s = 8
    decimation: int = 4
    action_space: int = 9  # auto-set to 8 for "relative_ee" mode in __post_init__
    observation_space: int = 0  # Set from obs_mode in child __post_init__
    obs_mode: str = "state"  # "state" or "full_state"; observation_space set in child __post_init__ to match
    # Optional: set in subclass to support set_obs_mode() after config is created. e.g. {"state": 31, "full_state": 65}
    obs_dims: dict | None = None
    state_space = 0

    def set_obs_mode(self, mode: str) -> None:
        """Set observation mode and observation_space. Use after parse_env_cfg() when you want a different obs than default."""
        self.obs_mode = mode
        if self.obs_dims is not None and mode in self.obs_dims:
            self.observation_space = self.obs_dims[mode]
    enable_cameras: bool = False

    # Reward composition (all optional via config)
    use_absolute_progress_rewards: bool = False  # positive delta from best reward in current stage
    use_relative_progress_rewards: bool = True  # delta from previous state reward
    use_time_penalty: bool = True               # linear penalty -0.001 * step
    use_velocity_penalty: bool = False            # subtract joint velocity norm * scale
    regularize_velocity_scale: float = 0.01
    regularize_time_scale: float = 0.000
    progress_rewards_scale: float = 100.0

    # DEBUG tool
    debug_rewards = False
    sun_program_evaluator_stage_override: bool = False

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physx=PhysxCfg(
            gpu_collision_stack_size=2**28,
            gpu_max_rigid_patch_count=2**21,  # >= 1176864 for drawer_stack with 8192 envs
            gpu_found_lost_pairs_capacity=2**22,  # >= 2293760 (foundLostPairsCapacity)
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=8192, env_spacing=2.0, replicate_physics=True
    )

    # robot - common Franka configuration
    robot = ArticulationCfg(
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
                "panda_joint2": -0.5,
                "panda_joint3": 0.0,
                "panda_joint4": -2.,
                "panda_joint5": 0.0,
                "panda_joint6": 2.5,
                "panda_joint7": 0.7,
                "panda_finger_joint.*": 0.04,
            },
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                effort_limit_sim=87.0,
                velocity_limit_sim=2.175/4,
                stiffness=80.0,
                damping=4.0,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit_sim=12.0,
                velocity_limit_sim=2.61/4,
                stiffness=80.0,
                damping=4.0,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint.*"],
                effort_limit_sim=200.0,
                velocity_limit_sim=0.2,
                stiffness=2e3,
                damping=1e2,
            ),
        },
    )
    robot.spawn.rigid_props.disable_gravity = True

    # Set render interval to match decimation
    sim.render_interval = decimation

    robot.actuators["panda_shoulder"].stiffness = 400.0
    robot.actuators["panda_shoulder"].damping = 80.0
    robot.actuators["panda_forearm"].stiffness = 400.0
    robot.actuators["panda_forearm"].damping = 80.0

    render_cfg = RenderCfg(
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

    # ground plane
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


    # camera_0
    table_cam_0 = TiledCameraCfg(
        prim_path="/World/envs/env_.*/table_cam_0",
        update_period=0.0,
        height=128,
        width=128,
        data_types=[
            "rgb",
            "distance_to_image_plane",
            "instance_segmentation_fast",
            "instance_id_segmentation_fast",
        ],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 20),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(1.65, 0, 1.0),
            rot=tuple(quat_from_euler_xyz(
        torch.tensor(torch.pi/3), torch.tensor(0.0), torch.tensor(torch.pi/2)
    ).tolist()),
            convention="opengl",
        ),
    )

    # camera_1
    table_cam_1 = TiledCameraCfg(
        prim_path="/World/envs/env_.*/table_cam_1",
        update_period=0.0,
        height=128,
        width=128,
        data_types=[
            "rgb",
            "distance_to_image_plane",
            "instance_segmentation_fast",
            "instance_id_segmentation_fast",
        ],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 20),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.4, 1.4, 1.0),
            rot=tuple(quat_from_euler_xyz(
        torch.tensor(torch.pi/3), torch.tensor(0.0), torch.tensor(torch.pi)
    ).tolist()),
            convention="opengl",
        ),
    )


    # camera_2
    table_cam_2 = TiledCameraCfg(
        prim_path="/World/envs/env_.*/table_cam_2",
        update_period=0.0,
        height=128,
        width=128,
        data_types=[
            "rgb",
            "distance_to_image_plane",
            "instance_segmentation_fast",
            "instance_id_segmentation_fast",
        ],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 20),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.4, -1.4, 1.0),
            rot=tuple(quat_from_euler_xyz(
        torch.tensor(torch.pi/3), torch.tensor(0.0), torch.tensor(0.0)
    ).tolist()),
            convention="opengl",
        ),
    )

    # action mode: True for debugging (direct action), False for learning (incremental)
    use_debug_action_mode: bool = False

    action_scale: float = 15.0
    ee_action_scale: float = 3.0  # position scale for EE modes (separate from joint action_scale)

    # action mode: "absolute_joint" (MPC absolute targets), "relative_joint" (incremental, 9D),
    #              "relative_ee" (delta pos + axis-angle rotation, 8D), "absolute_ee" (target EE pose, 11D)
    action_mode: str = "absolute_ee"

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
        "control_output_mode": "absolute_ee",
        "load_program": f"{script_dir}/resources/learned_programs/manipulation-hang-cup/debug",
        "initial_residual": f"{script_dir}/src/kuafu/skills/constraints/rewards.py",
        "initial_predicate": f"{script_dir}/src/kuafu/skills/constraints/predicates.py",
    }
    del script_dir
    # task_description - must be set by child classes
    task_description: str = ""

    # Optional: RecorderManager + HDF5 export (obs, actions, scene state, rewards, dones).
    # When set, dataset_export_dir_path and dataset_filename in the config control output path.
    recorders: Optional[LearnedEnvRecorderManagerCfg] = None

    def __post_init__(self):
        # "relative_ee" uses 8D actions: [dx, dy, dz, ax, ay, az, grip1, grip2]
        if self.action_mode == "relative_ee":
            self.action_space = 8
        elif self.action_mode == "absolute_ee":
            self.action_space = 11

class BaseLearnedEnv(DirectRLEnv):
    """Base environment class for learned manipulation tasks.
    
    This class contains common functionality shared across all learned
    manipulation environments. Child classes should inherit from this and
    implement task-specific methods like _setup_scene, _get_observations,
    and _external_get_dones.
    """
    cfg: BaseLearnedEnvCfg

    def __init__(self, cfg: BaseLearnedEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Simple timing prints (disabled by default). Initialize early because we call
        # update_data_in_controller() during __init__.
        self._profile = bool(int(os.getenv("KUAFU_PROFILE", "0")))
        self._profile_every = int(os.getenv("KUAFU_PROFILE_EVERY", "200"))
        self._profile_sync_cuda = bool(int(os.getenv("KUAFU_PROFILE_SYNC_CUDA", "0")))

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # create auxiliary variables for computing applied action, observations and rewards
        self._robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[
            0, :, 0
        ].to(device=self.device)
        self._robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[
            0, :, 1
        ].to(device=self.device)

        self._robot_dof_targets = torch.zeros(
            (self.num_envs, self._robot.num_joints), device=self.device
        )

        # Completion tracker: counts how many times each environment has completed the task (persists across resets)
        self._completion_tracker = torch.zeros(
            (self.num_envs,), dtype=torch.long, device=self.device
        )

        # End-to-end termination stats (per-run; cleared via clear_end_to_end_rate_stats()).
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

        self.program_runtime.initialize_batch(self.num_envs)
        # Track current stage index per environment for observations
        self.current_stage = torch.zeros(
            (self.num_envs, 1), device=self.device, dtype=torch.float32
        )
        self.update_data_in_controller()
        self._sync_stage_tensor()

        # Stage transition stats: average steps from one stage to the next, per stage.
        # NOTE: these are transition-event samples (not "env count").
        # stage_idx -> {"sum_steps": float, "n_samples": int}
        self._stage_stats = {}
        self._prev_stage_indices = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        stage_indices_0 = self.program_runtime.get_stage_indices()
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

        self.robot_dof_speed_scales = torch.ones_like(self._robot_dof_lower_limits)
        self.robot_dof_speed_scales[self._robot.find_joints("panda_finger_joint1")[0]] = 0.1
        self.robot_dof_speed_scales[self._robot.find_joints("panda_finger_joint2")[0]] = 0.1

        # Effort normalization scales for observations.
        # Arm joints can see ~100N total; gripper max force comes from MPC config (or defaults).
        self._arm_effort_scale = float(self.cfg.mpc_config.get("arm_max_effort", 50.0))
        gripper_max_from_cfg = self.cfg.mpc_config.get("gripper_max_force", None)
        if gripper_max_from_cfg is not None:
            self._gripper_effort_scale = float(gripper_max_from_cfg)
        else:
            api = self.program_runtime.controller_api_wrapper
            self._gripper_effort_scale = float(
                getattr(api, "_gripper_max_force", 40.0)
            )
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

    def update_data_in_controller(self):
        """Update controller with latest odometry and sensor data."""
        sync_cuda = getattr(self, "_profile_sync_cuda", False)

        with torch.no_grad():
            if sync_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            odom = extract_odom_manipulators(self, self.cfg.detectable_objects)
            if sync_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()

            if sync_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            sensors = extract_frame_sensors(self, {}, self.cfg.detectable_objects)
            if sync_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()

            self.program_runtime.callbackOdom(odom, sensors)

            self.program_runtime.callbackJointState(combine_joint_states(self, robot="franka"))

            # Update POV state for environments that need it (first time in stage 0 or after reset)
            self.program_runtime.update_pov_if_needed()

        self._sync_stage_tensor()

    def _setup_scene(self):
        """Setup the scene. Must be implemented by child classes."""
        self._robot = Articulation(self.cfg.robot)
        self.ee_frame = FrameTransformer(self.cfg.ee_frame)
        self.l4_frame = FrameTransformer(self.cfg.l4_frame)

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        if self.cfg.enable_cameras:
            self.table_cam_0 = TiledCamera(self.cfg.table_cam_0)
            self.table_cam_1 = TiledCamera(self.cfg.table_cam_1)
            self.table_cam_2 = TiledCamera(self.cfg.table_cam_2)

        self.scene.articulations["robot"] = self._robot
        self.scene.sensors["panda_fingertip"] = self.ee_frame
        self.scene.sensors["panda_link4"] = self.l4_frame

        if self.cfg.enable_cameras:
            self.scene.sensors["table_cam_0"] = self.table_cam_0
            self.scene.sensors["table_cam_1"] = self.table_cam_1
            self.scene.sensors["table_cam_2"] = self.table_cam_2

    def close(self):
        super().close()

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

    def _setup_ik_controller(self):
        """Initialize the differential IK controller for end-effector action modes."""
        robot_cfg = self.cfg.robot
        if "fr3_shoulder" in robot_cfg.actuators:
            joint_names = ["fr3.*"]
            body_names = ["fr3_link7"]
        else:
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

        # Cache the inverse pose offset (panda_fingertip → fr3_link7).
        # At runtime, the current fingertip pose is read directly from the program_runtime API.
        # The inverse offset is used to convert the desired fingertip target back to
        # fr3_link7 pose for the IK controller.
        frame_cfg = self.cfg.ee_frame.target_frames[0].offset
        p_off = torch.tensor(frame_cfg.pos, device=self.device, dtype=torch.float32)  # (3,)
        q_off = torch.tensor(frame_cfg.rot, device=self.device, dtype=torch.float32)  # (4,) wxyz
        q_off_inv = quat_inv(q_off.unsqueeze(0)).squeeze(0)
        p_off_inv = quat_apply(q_off_inv.unsqueeze(0), -p_off.unsqueeze(0)).squeeze(0)
        self._ee_inv_offset_pos = p_off_inv    # (3,) fingertip→link7 in fingertip frame
        self._ee_inv_offset_quat = q_off_inv  # (4,) fingertip→link7 rotation

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
            # Convert desired fingertip target → fr3_link7 target for IK (IK-only).
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
            # Convert desired fingertip target → fr3_link7 target for IK (IK-only).
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
                self._robot_dof_lower_limits[:n_arm],
                self._robot_dof_upper_limits[:n_arm],
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
        self._robot.set_joint_position_target(self._robot_dof_targets)

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset environments. Child classes should call super() and add task-specific resets."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if self.recorder_manager is not None and len(env_ids) > 0:
            self.recorder_manager.record_pre_reset(env_ids)

        super()._reset_idx(env_ids)

        # robot state - common reset for all environments
        joint_pos = self._robot.data.default_joint_pos[env_ids] + sample_uniform(
            -0.2,
            0.2,
            (len(env_ids), self._robot.num_joints),
            self.device,
        )
        joint_pos = torch.clamp(
            joint_pos, self._robot_dof_lower_limits, self._robot_dof_upper_limits
        )
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        if self.cfg.debug_rewards:
            env_name = self.__class__.__name__
            cumulative = self._debug_cumulative_rewards[env_ids]
            print(f"[{env_name}][DebugReward] average episode reward: {cumulative.mean().item():.4f}")
            self._debug_cumulative_rewards[env_ids] = 0.0

        self.program_runtime.reset_stage_indices(env_ids)
        self._sync_stage_tensor()
        step_val = (
            int(self.common_step_counter.item())
            if torch.is_tensor(self.common_step_counter)
            else int(self.common_step_counter)
        )
        self._prev_stage_indices[env_ids] = 0
        self._stage_entered_step[env_ids] = float(step_val)
        # Reset highest reward per env for progress-based rewards
        self._previous_reward_per_env[env_ids] = 0.0
        self._stage_progress_reset_mask[env_ids] = False
        # Need to refresh the intermediate values so that _get_observations() can use the latest values
        self._get_observations()

        if self.recorder_manager is not None and len(env_ids) > 0:
            self.recorder_manager.record_post_reset(env_ids)

    def _sync_stage_tensor(self):
        """Sync the current stage tensor with the program_runtime's stage indices."""
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
        ############################
        # Compute predicates + residual together to avoid duplicated stage-loop work.
        t0 = time.perf_counter()
        with torch.no_grad():
            satisfied, past_final_stage, state_costs = self.program_runtime.check_and_advance_stages_and_compute_rewards()
            self._stage_bonus = satisfied.to(dtype=torch.float32) * self.program_runtime.get_stage_indices()
            self._stage_progress_reset_mask.copy_(satisfied)

        # Cache cost for _get_rewards() (DirectRLEnv calls _get_dones() before _get_rewards()).
        self._cached_rewards = 1.0 - state_costs

        # Update stage transition stats: average steps from one stage to the next
        new_stages = self.program_runtime.get_stage_indices()
        if new_stages is not None:
            new_stages_flat = new_stages.to(device=self.device, dtype=torch.long).view(-1)
            transitioned = new_stages_flat > self._prev_stage_indices
            if transitioned.any():
                step_val = (
                    int(self.common_step_counter.item())
                    if torch.is_tensor(self.common_step_counter)
                    else int(self.common_step_counter)
                )
                steps_in_prev = (
                    step_val - self._stage_entered_step[transitioned].cpu().numpy()
                )
                prev_stages_np = self._prev_stage_indices[transitioned].cpu().numpy()
                for prev_s, steps in zip(prev_stages_np, steps_in_prev):
                    prev_s_int = int(prev_s)
                    if prev_s_int not in self._stage_stats:
                        self._stage_stats[prev_s_int] = {"sum_steps": 0.0, "n_samples": 0}
                    self._stage_stats[prev_s_int]["sum_steps"] += float(steps)
                    self._stage_stats[prev_s_int]["n_samples"] += 1
                self._stage_entered_step[transitioned] = float(step_val)
            self._prev_stage_indices.copy_(new_stages_flat)
            # Track highest stage each env has reached (for "ratio of envs moved past stage").
            self._highest_stage_per_env = torch.maximum(
                self._highest_stage_per_env, new_stages_flat
            )


        # Track completion count before reset happens (persists across resets, increments on each completion)
        self._completion_tracker = self._completion_tracker + terminate_external.long()

        env_name = self.__class__.__name__
        if terminate_external.any().item():
            print(
                f"[{env_name}] Successfully finished #: {torch.sum(terminate_external).cpu().numpy()}"
            )

        false_finished = past_final_stage if not self.cfg.sun_program_evaluator_stage_override else torch.zeros_like(past_final_stage)
        if false_finished.any().item():
            print(f"[{env_name}] False finished #: {torch.sum(false_finished).cpu().numpy()}")

        # End-to-end termination totals (per-run).
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
        truncated = torch.logical_and(time_out, torch.logical_not(terminated))
        stuck_terminal = torch.logical_and(truncated, torch.logical_not(terminated))
        if stuck_terminal.any():
            self._accumulate_terminal_stuck_constraint_stats(stuck_terminal)

        self._sync_stage_tensor()
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards based on cost and stage progression."""
        t0 = time.perf_counter()
        stage_numbers = self.program_runtime.get_stage_indices()
        t_stage = time.perf_counter() - t0

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

        # Optional: time penalty
        time_penalty = self.cfg.regularize_time_scale * self.episode_length_buf.to(dtype=state_rewards.dtype)

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
            rewards = rewards - time_penalty

        # Terminal shaping: grant/penalise the best-case reward for all remaining frames.
        success = getattr(self, "_cached_success")
        failure = getattr(self, "_cached_failure")
        max_stage = self.program_runtime.compiled_program.stage_count()
        highest_reward_per_frame =  max_stage 
        remaining_frames = (self.max_episode_length - self.episode_length_buf).to(dtype=state_rewards.dtype)
        terminal_bonus = remaining_frames * highest_reward_per_frame
        rewards = rewards + success.to(dtype=state_rewards.dtype) * terminal_bonus
        rewards = rewards - failure.to(dtype=state_rewards.dtype) * terminal_bonus

        # Print average cost every step
        if int(self.common_step_counter) % 100 == 0:
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
            print(
                f"[{env_name}] Average stage = {torch.mean(stage_numbers.to(torch.float32)).item()}, Max stage = {torch.max(stage_numbers).item()}"
            )

        rewards = rewards + self._stage_bonus * (self.max_episode_length - self.episode_length_buf).to(dtype=state_rewards.dtype)

        if self.cfg.debug_rewards:
            self._debug_cumulative_rewards += rewards

        return rewards

    def regularize_rewards(self) -> torch.Tensor:
        api = self.program_runtime.controller_api_wrapper
        agent = api.reference_agent
        joint_vel = api.joint_dict[agent]["joint_velocity"]
        penalty = torch.norm(joint_vel, dim=-1) * self.cfg.regularize_velocity_scale
        return penalty

    def _get_observations(self) -> dict:
        """Get observations. Must be implemented by child classes."""
        raise NotImplementedError("Child classes must implement _get_observations()")

    def get_highest_stage_reached_per_env(self) -> list[int]:
        """Return the highest stage reached per env since the last clear.

        This is the canonical signal used to compute "how many envs have moved past stage s":
        `sum(highest_stage_reached_per_env > s)`.
        """
        return [int(x) for x in self._highest_stage_per_env.detach().cpu().view(-1).tolist()]

    def get_stage_transition_stats(self) -> dict[int, dict]:
        """Return per-stage statistics for prompt/debugging.

        NOTE: This intentionally does NOT include a per-stage "count". Use
        `get_highest_stage_reached_per_env()` to compute env-pass ratios.

        Returns:
            Dict mapping stage_index -> {"avg_steps": float | None, "samples": int}.
            - "avg_steps": average steps to complete this stage (from transition events), or None if no samples.
            - "samples": number of transition-event samples used to compute avg_steps.
        """
        n_stages = self.program_runtime.compiled_program.stage_count()
        result = {}
        for s in range(n_stages):
            d = self._stage_stats.get(s, {"sum_steps": 0.0, "n_samples": 0})
            if d["n_samples"] > 0:
                result[s] = {
                    "avg_steps": d["sum_steps"] / d["n_samples"],
                    "samples": int(d["n_samples"]),
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

    def clear_stage_transition_stats(self) -> None:
        """Clear all stage transition statistics. Use this to restart tracking whenever needed."""
        self._stage_stats.clear()
        stage_indices = self.program_runtime.get_stage_indices()
        self.clear_end_to_end_rate_stats()
        self.clear_terminal_stuck_constraint_stats()
        if stage_indices is not None:
            self._highest_stage_per_env.copy_(
                stage_indices.to(device=self.device, dtype=torch.long).view(-1)
            )

    def _accumulate_terminal_stuck_constraint_stats(self, stuck_mask: torch.Tensor) -> None:
        """Accumulate per-constraint stats from envs that truncated while stuck in a stage."""
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
