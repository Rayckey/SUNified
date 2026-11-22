# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
from kuafu.paths import OBJECT_ROOT
from dataclasses import MISSING, field
import os
from isaacsim.core.utils.stage import get_current_stage
from isaaclab.sensors.frame_transformer.frame_transformer import FrameTransformer
import torch
from omegaconf import OmegaConf
from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics
import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import schemas
from isaaclab.sim import SimulationCfg, RenderCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform, quat_from_euler_xyz
from isaaclab.sensors import TiledCamera, TiledCameraCfg, CameraCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from .panda_table_env_learned import PandaTableEnvLearnedCfg, PandaTableEnvLearned
from kuafu.isaac.tasks.env_utils import random_pose_radial
from kuafu.mpc.simulation.spatial_utils import quat_to_rotation_6d

@configclass
class PandaDrawerEnvLearnedCfg(PandaTableEnvLearnedCfg):
    """Base configuration class for Panda manipulation environments.
    
    This class provides common configuration elements shared across all
    Panda manipulation tasks. Subclasses can override or extend these
    settings as needed.
    """
    observation_space = 22  # default state dim; use set_obs_mode() for full_state
    obs_mode: str = "state"  # "state" (31) or "full_state" (65)
    obs_dims = {"state": 22, "full_state": 56}
    episode_length_s = 15

    # Override robot init_state only, keep rest from base
    def __post_init__(self):
        """Post-initialization to override robot initial state."""
        super().__post_init__()
        # Override only the initial state, keeping the rest of robot config from parent
        self.robot.init_state = ArticulationCfg.InitialStateCfg(
            joint_pos={
              "panda_joint1": -1.0,
                "panda_joint2": -0.42,
                "panda_joint3": 0.0,
                "panda_joint4": -2.25,
                "panda_joint5": 0.00,
                "panda_joint6": 2.00,
                "panda_joint7": 0.80,
                "panda_finger_joint1": 0.0,
                "panda_finger_joint2": 0.0,
            },
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        )

    _cabinet_path = str(OBJECT_ROOT / 'two_layer_drawer_iphone_with_link/two_layer_drawer_iphone_with_link.usd')

    # cabinet
    cabinet = ArticulationCfg(
        prim_path="/World/envs/env_.*/Cabinet",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_cabinet_path,
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.001, rest_offset=0.001
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.6, -0.5, 0.2),
            rot=(0, 0, 0, 1),
            joint_pos={
                "upper_drawer_joint": 0.0,
            },
        ),
        actuators={
            "drawers": ImplicitActuatorCfg(
                joint_names_expr=["upper_drawer_joint"],
                effort_limit_sim=87.0,
                stiffness=0.0,
                damping=3.0,
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
                    pos=[0.0, 0.00, 0.00],
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
                    pos=[0.0, 0.10, 0.02],
                ),
            )
        ],
    )

    detectable_objects = {
        'cabinet': (
            'mesh of the cabinet, local frame +z is up direction, local frame -y is the forward direction, estimated size: (0.15, 0.25, 0.2)',
            '/World/envs/env_*/Cabinet',
        ),
        'top_drawer_center_frame': (
            'center frame of top drawer, estimated size: (0.15, 0.25, 0.05)',
            '/Visuals/top_drawer_center_frame',
        ),
        'top_drawer_handle_frame': (
            "frame of top drawer's handle, handle is a bar along local frame x axis, local frame -y is pulling out direction",
            '/Visuals/top_drawer_handle_frame',
        ),
        'panda_link4': (
            'frame of robot link 4',
            '/Visuals/panda_link4_frame',
        ),
        'panda_fingertip': (
            'frame of robot tool frame',
            '/Visuals/panda_hand_frame',
        ),
    }

    _kuafu_dir = os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    ))
    _repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(_kuafu_dir)))
    mpc_config = {
        "controller_name": "FrankaIKController",
        "device": "cuda",
        "ee_frame": "panda_fingertip",
        "load_program": f"{_repo_dir}/resources/learned_programs/manipulation-fr3-finray-drawer/2026-04-01_14-56-05",
        "initial_residual": f"{_repo_dir}/src/kuafu/skills/constraints/rewards.py",
        "initial_predicate": f"{_repo_dir}/src/kuafu/skills/constraints/predicates.py",
        "use_per_env_weight": False,
        "gripper_opened": [0.04, 0.04],
        "gripper_closed": [0.0, 0.0],
        "gripper_max_force": 40.0,
        "has_finger_contact_sensors": False,
    }
    del _repo_dir

    task_description = 'open the top drawer by 14 cm'
class PandaDrawerEnvLearned(PandaTableEnvLearned):
    cfg: PandaDrawerEnvLearnedCfg

    def __init__(self, cfg: PandaDrawerEnvLearnedCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

    def _replace_frame_collision_with_cube(self):
        stage = get_current_stage()
        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
            useExtentsHint=False,
            ignoreVisibility=False,
        )

        cabinet_prims = sim_utils.find_matching_prims(self.cfg.cabinet.prim_path)
        collision_props = getattr(self.cfg.cabinet.spawn, "collision_props", None)
        replaced_count = 0
        print(
            "[PandaDrawerEnvLearned] Cabinet prims found for collision replacement: "
            f"{len(cabinet_prims)}",
            flush=True,
        )

        for cabinet_prim in cabinet_prims:
            frame_path = f"{cabinet_prim.GetPath().pathString}/frame"
            frame_prim = stage.GetPrimAtPath(frame_path)
            if not frame_prim.IsValid():
                print(
                    "[PandaDrawerEnvLearned] WARNING: Cabinet frame prim not found: "
                    f"{frame_path}",
                    flush=True,
                )
                continue

            frame_collision_path = f"{frame_path}/collisions"
            frame_collision_prim = stage.GetPrimAtPath(frame_collision_path)
            if frame_collision_prim.IsValid():
                frame_collision_prim.SetActive(False)
                print(
                    "[PandaDrawerEnvLearned] Disabled original frame collision subtree: "
                    f"{frame_collision_path}",
                    flush=True,
                )
            else:
                print(
                    "[PandaDrawerEnvLearned] WARNING: Frame collision subtree not found: "
                    f"{frame_collision_path}",
                    flush=True,
                )
                continue

            size = (0.30, 0.23, 0.23)
            center = (0.0, 0.11, 0.1)
            collider_path = f"{frame_path}/cube_collision"
            if stage.GetPrimAtPath(collider_path).IsValid():
                stage.RemovePrim(collider_path)
            collider = UsdGeom.Cube.Define(stage, collider_path)
            collider.CreateSizeAttr(1.0)
            collider_xform = UsdGeom.XformCommonAPI(collider)
            collider_xform.SetTranslate(center)
            collider_xform.SetScale(size)
            UsdGeom.Imageable(collider).MakeInvisible()

            if collision_props is not None:
                schemas.define_collision_properties(collider_path, collision_props, stage)

            replaced_count += 1
            print(
                "[PandaDrawerEnvLearned] Replaced frame collision with cube: "
                f"frame={frame_path}, collider={collider_path}, "
                f"center={center}, size={size}",
                flush=True,
            )

        if replaced_count == 0:
            print(
                "[PandaDrawerEnvLearned] WARNING: No cabinet frame was updated; "
                "collision replacement was not applied.",
                flush=True,
            )

    def _setup_scene(self):
        super()._setup_scene()

        self._cabinet = Articulation(self.cfg.cabinet)
        self._replace_frame_collision_with_cube()
        self.scene.articulations["cabinet"] = self._cabinet
        self.top_drawer_handle_frame = FrameTransformer(
            self.cfg.top_drawer_handle_frame
        )
        self.top_drawer_center_frame = FrameTransformer(
            self.cfg.top_drawer_center_frame
        )
        self.scene.sensors["top_drawer_handle_frame"] = self.top_drawer_handle_frame
        self.scene.sensors["top_drawer_center_frame"] = self.top_drawer_center_frame

        self.scene.clone_environments(copy_from_source=False)
        self._hide_static_surface_visual_prims()

        light_cfg = sim_utils.DomeLightCfg(intensity=1000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        if self.device == "cpu" and self.cfg.terrain is not None:
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

    def _get_observations(self) -> dict:

        self.update_data_in_controller()

        api = self.program_runtime.controller_api_wrapper
        agent = api.reference_agent
        if agent is None:
            raise RuntimeError("controller_api_wrapper.reference_agent is not set")

        handle_pos      = api.get_object_xyz_position("top_drawer_handle_frame")           # (N, 3)
        fingertip_pos   = api.lookup_robot_frame_position("panda_fingertip")               # (N, 3)
        link4_pos       = api.lookup_robot_frame_position("panda_link4")                   # (N, 3)
        handle_rot6d    = quat_to_rotation_6d(api.get_object_wxyz_quaternion("top_drawer_handle_frame"))  # (N, 6)
        fingertip_rot6d = quat_to_rotation_6d(api.lookup_robot_frame_rotation("panda_fingertip"))        # (N, 6)
        link4_rot6d     = quat_to_rotation_6d(api.lookup_robot_frame_rotation("panda_link4"))            # (N, 6)
        cabinet_pos     = self._cabinet.data.root_pos_w                                    # (N, 3)
        cabinet_rot6d   = quat_to_rotation_6d(self._cabinet.data.root_quat_w)             # (N, 6)
        joint_pos       = api.joint_dict[agent]["joint_position"]                          # (N, 9)
        joint_vel       = api.joint_dict[agent]["joint_velocity"]                          # (N, 9)
        joint_effort    = self._get_normalized_joint_effort(api, agent)                    # (N, 9)
        gripper_contact = self._get_normalized_gripper_contact(api)                         # (N, 2)
        joint_effort    = torch.cat([joint_effort[:, :7], gripper_contact], dim=-1)        # (N, 9)
        gripper_open    = api.gripper_openness().unsqueeze(-1)                              # (N, 1)
        stage           = self.current_stage                                               # (N, 1)

        def _f(*tensors):
            return torch.cat([t.to(device=self.device, dtype=torch.float32) for t in tensors], dim=-1)

        # Policy obs (31): task-relevant state only, no joint pos/vel/effort, no link4
        policy_obs = _f(
            handle_pos, fingertip_pos,          # 3+3 = 6
            handle_rot6d, fingertip_rot6d,      # 6+6 = 12
            # cabinet_pos, cabinet_rot6d,          # 3+6 = 9
            gripper_open, gripper_contact,       # 1+2 = 3
            stage,                               # 1
        )  # total: 31

        # Full state obs (65): everything needed to recreate the env
        full_state_obs = _f(
            handle_pos, fingertip_pos, link4_pos,          # 3+3+3 = 9
            handle_rot6d, fingertip_rot6d, link4_rot6d,    # 6+6+6 = 18
            # cabinet_pos, cabinet_rot6d,                     # 3+6 = 9
            joint_pos, joint_vel, joint_effort,             # 9+9+9 = 27
            gripper_open,                                    # 1
            stage,                                           # 1
        )  # total: 65

        obs_dim = int(self.cfg.observation_space)
        policy_obs_to_return = policy_obs if self.cfg.obs_mode == "state" else full_state_obs
        if policy_obs_to_return.shape[-1] != obs_dim:
            raise RuntimeError(
                f"Observation dim mismatch: got {policy_obs_to_return.shape[-1]}, expected {obs_dim}"
            )

        self.extras["episode/mean_stage"] = torch.mean(self.current_stage.float())
        self.extras["episode/max_stage"] = torch.max(self.current_stage.float())

        if not torch.isfinite(policy_obs_to_return).all():
            print("[PandaDrawerEnvLearned] obs contains NaN/Inf; sanitizing with nan_to_num.", flush=True)
            torch.nan_to_num_(policy_obs_to_return, nan=0.0, posinf=0.0, neginf=0.0)
            torch.nan_to_num_(full_state_obs, nan=0.0, posinf=0.0, neginf=0.0)
            torch.nan_to_num_(policy_obs, nan=0.0, posinf=0.0, neginf=0.0)

        return {
            "policy": policy_obs_to_return,
            "full_state": full_state_obs,
            "state": policy_obs,
        }

    def external_rewards(self) -> torch.Tensor:
        self.update_data_in_controller()
        drawer_open = torch.clamp(self._cabinet.data.joint_pos[:, 0] / 0.12, 0.0, 1.0)
        handle_w = self.top_drawer_handle_frame.data.target_pos_w.squeeze(1)
        ft_w = self.ee_frame.data.target_pos_w.squeeze(1)

        handle_dist = torch.linalg.norm(ft_w - handle_w, dim=-1)
        reach_reward = torch.exp(-8.0 * handle_dist)
        interact_reward = torch.exp(-40.0 * handle_dist)
        open_reward = drawer_open

        return 0.75 * reach_reward + 0.75 * interact_reward + 4.0 * open_reward + 2.0 * open_reward.square()

    def _external_get_dones(self,
    ):
        # DirectRLEnv runs _get_dones before _get_observations; refresh controller joints so
        # gripper_openness matches the physics step (frame sensors are already updated).
        self.update_data_in_controller()

        # Task success: drawer open, gripper mostly open, EE at least 0.07 m from handle (world frame).
        opened = self._cabinet.data.joint_pos[:, 0] > 0.1
        gripper_open = self.program_runtime.controller_api_wrapper.gripper_openness().flatten() > 0.7
        handle_w = self.top_drawer_handle_frame.data.target_pos_w.squeeze(1)
        ft_w = self.ee_frame.data.target_pos_w.squeeze(1)
        far_from_handle = torch.linalg.norm(ft_w - handle_w, dim=-1) >= 0.05
        task_done = torch.logical_and(
            torch.logical_and(opened, gripper_open), far_from_handle
        )

        if self.cfg.sun_program_evaluator_stage_override or bool(
            getattr(self.cfg, "ignore_stage_gate_for_success", False)
        ):
            return task_done

        past_stage = self.current_stage.flatten() > int(self.program_runtime.compiled_program.stage_count()/2)
        return torch.logical_and(task_done, past_stage)


    def _reset_idx(self, env_ids: torch.Tensor | None):


        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        if self.recorder_manager is not None and len(env_ids) > 0:
            self.recorder_manager.record_pre_reset(env_ids)

        self._prepare_support_surface_reset(env_ids)

        zeros = torch.zeros((len(env_ids), self._cabinet.num_joints), device=self.device)
        self._cabinet.write_joint_state_to_sim(zeros, zeros, env_ids=env_ids)  

        num_resets = len(env_ids)

        # Randomize cabinet pose
        is_left = torch.rand(num_resets, device=self.device) < 0.5
        dist_left  = torch.rand(num_resets, device=self.device) * 0.2 + 0.4
        dist_right = torch.rand(num_resets, device=self.device) * 0.2 + 0.5
        dist = torch.where(is_left, dist_left, dist_right)
        right_angle_min, right_angle_max = self._spawn_angle_range(0, 2)
        left_angle_min, left_angle_max = self._spawn_angle_range(1, 2)
        azimuth_left = left_angle_min + torch.rand(num_resets, device=self.device) * (left_angle_max - left_angle_min)
        azimuth_right = right_angle_min + torch.rand(num_resets, device=self.device) * (right_angle_max - right_angle_min)
        azimuth = torch.where(is_left, azimuth_left, azimuth_right)
        height = self._sample_object_spawn_height_offset(num_resets)

        facing_yaw = torch.where(is_left, azimuth - torch.pi / 2, azimuth + torch.pi / 2)
        # raw offset in [-π/3, π/6]: negative = toward origin (60°), positive = away (30°)
        # for right the geometry is mirrored, so flip the sign
        raw_offset = -torch.rand(num_resets, device=self.device) * (torch.pi / 3)
        offset_rad = torch.where(is_left, raw_offset, -raw_offset)
        yaw = facing_yaw + offset_rad

        origins = self._surface_origins(env_ids)
        root = self._cabinet.data.default_root_state.clone().index_select(0, env_ids)
        root[:, 0] = origins[:, 0] + dist * torch.cos(azimuth)
        root[:, 1] = origins[:, 1] + dist * torch.sin(azimuth)
        root[:, 2:3] = self._spawn_z_above_surface(env_ids, height)

        z = torch.zeros_like(yaw)
        root[:, 3:7] = quat_from_euler_xyz(z, z, yaw)

        self._cabinet.write_root_pose_to_sim(root[:, :7], env_ids=env_ids)
        self._cabinet.write_root_velocity_to_sim(
            torch.zeros((num_resets, 6), device=self.device), env_ids=env_ids
        )
        self._cabinet.reset(env_ids)

        super()._reset_idx(env_ids)
