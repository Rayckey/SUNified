# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import time
import torch

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import math as math_utils
from kuafu.isaac.tasks.env_learned import BaseLearnedEnvCfg, BaseLearnedEnv
from kuafu.isaac.tasks.env_utils import random_pose_radial
from kuafu.mpc.simulation.spatial_utils import quat_to_rotation_6d

ISAAC_NUCLEUS_DIR = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac"

@configclass
class Stack2(BaseLearnedEnvCfg):

    observation_space = 31  # default state dim; use set_obs_mode() for full_state
    obs_mode: str = "state"  # "state" (31) or "full_state" (65)
    obs_dims = {"state": 31, "full_state": 65}

        # Override robot init_state only, keep rest from base
    def __post_init__(self):
        """Post-initialization to override robot initial state."""
        super().__post_init__()
        # Override only the initial state, keeping the rest of robot config from parent
        self.robot.init_state = ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 0.0,
                "panda_joint2": -0.569,
                "panda_joint3": 0.0,
                "panda_joint4": -2.810,
                "panda_joint5": 0.0,
                "panda_joint6": 3.037,
                "panda_joint7": 0.741,
                "panda_finger_joint.*": 0.04,
            },
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        )

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
            pos=[0.4, 0.0, 0.0203], rot=[1, 0, 0, 0]
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
            pos=[0.55, 0.05, 0.0203], rot=[1, 0, 0, 0]
        ),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/red_block.usd",
            scale=(1.0, 1.0, 1.0),
            rigid_props=cube_properties,
            semantic_tags=[("class", "cube_2")],
        ),
    )

    detectable_objects = {
        "cube_2": (
            "mesh of a green cube, same length on each side, symmetric on all axes, size: (0.04, 0.04, 0.04)",
            "/World/envs/env_*/Cube_2",
        ),
        "cube_1": (
            "mesh of a red cube, same length on each side, symmetric on all axes, size: (0.04, 0.04, 0.04)",
            "/World/envs/env_*/Cube_1",
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

    script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

    mpc_config = {
            "controller_name": "FrankaIKController",
            "device": "cuda",
            "ee_frame": "panda_fingertip",
            "use_per_env_weight": False,
            "load_program": f"{script_dir}/resources/learned_programs/manipulation-stack2/2026-01-09_07-09-06",
            "initial_residual": f"{script_dir}/src/kuafu/skills/constraints/rewards.py",
            "initial_predicate": f"{script_dir}/src/kuafu/skills/constraints/predicates.py",
        }
    del script_dir

    task_description = "stack the cubes"


class Stack2Env(BaseLearnedEnv):
    cfg: Stack2

    def _setup_scene(self):
        super()._setup_scene()

        self.cube_1 = RigidObject(self.cfg.cube_1)
        self.cube_2 = RigidObject(self.cfg.cube_2)

        self.scene.rigid_objects["cube_1"] = self.cube_1
        self.scene.rigid_objects["cube_2"] = self.cube_2

        self.scene.clone_environments(copy_from_source=False)

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=1000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        assert env_ids is not None

        if self.recorder_manager is not None and len(env_ids) > 0:
            self.recorder_manager.record_pre_reset(env_ids)

        n = len(env_ids)
        env_ids_tensor = env_ids
        origins = self.scene.env_origins.index_select(0, env_ids_tensor)

        # Right half (0° to +90°) and left half (0° to -90°); randomly swap which object is on which side
        pos_right, quat_right = random_pose_radial(
            0.01,
            dist_min=0.45,
            dist_max=0.6,
            angle_min_rad=0.0,
            angle_max_rad=torch.pi / 3,
            n=n,
            device=self.device,
            origins=origins,
        )
        pos_left, quat_left = random_pose_radial(
            0.06,
            dist_min=0.45,
            dist_max=0.6,
            angle_min_rad=-torch.pi / 3,
            angle_max_rad=0.0,
            n=n,
            device=self.device,
            origins=origins,
        )

        swap = torch.rand(n, device=self.device) > 0.5
        swap_2 = swap.unsqueeze(1).expand(-1, 2)
        swap_4 = swap.unsqueeze(1).expand(-1, 4)

        marker_xy = torch.where(swap_2, pos_left[:, :2], pos_right[:, :2])
        holder_xy = torch.where(swap_2, pos_right[:, :2], pos_left[:, :2])
        cube1_pos = torch.cat(
            [marker_xy, (origins[:, 2:3] + 0.0203)], dim=1
        )
        cube2_pos = torch.cat(
            [holder_xy, (origins[:, 2:3] + 0.0203)], dim=1
        )
        cube1_quat = torch.where(swap_4, quat_left, quat_right)
        cube2_quat = torch.where(swap_4, quat_right, quat_left)

        root_m = self.cube_1.data.default_root_state.clone().index_select(
            0, env_ids_tensor
        )
        root_m[:, :3] = cube1_pos
        root_m[:, 3:7] = cube1_quat
        self.cube_1.write_root_pose_to_sim(root_m[:, :7], env_ids=env_ids)  # type: ignore[arg-type]
        self.cube_1.write_root_velocity_to_sim(
            torch.zeros((len(env_ids), 6), device=self.device), env_ids=env_ids  # type: ignore[arg-type]
        )
        self.cube_1.reset(env_ids)  # type: ignore[arg-type]

        root_m = self.cube_2.data.default_root_state.clone().index_select(
            0, env_ids_tensor
        )
        root_m[:, :3] = cube2_pos
        root_m[:, 3:7] = cube2_quat
        self.cube_2.write_root_pose_to_sim(root_m[:, :7], env_ids=env_ids)  # type: ignore[arg-type]

        self.cube_2.write_root_velocity_to_sim(
            torch.zeros((len(env_ids), 6), device=self.device), env_ids=env_ids  # type: ignore[arg-type]
        )
        self.cube_2.reset(env_ids)  # type: ignore[arg-type]

        super()._reset_idx(env_ids)

    def _get_observations(self) -> dict:
        self.update_data_in_controller()

        api = self.program_runtime.controller_api_wrapper
        agent = api.reference_agent
        if agent is None:
            raise RuntimeError("controller_api_wrapper.reference_agent is not set")

        cube1_pos = api.get_object_xyz_position("cube_1")  # (N, 3)
        cube2_pos = api.get_object_xyz_position("cube_2")  # (N, 3)
        fingertip_pos = api.lookup_robot_frame_position("panda_fingertip")  # (N, 3)
        link4_pos = api.lookup_robot_frame_position("panda_link4")  # (N, 3)

        cube1_rot6d = quat_to_rotation_6d(api.get_object_wxyz_quaternion("cube_1"))  # (N, 6)
        cube2_rot6d = quat_to_rotation_6d(api.get_object_wxyz_quaternion("cube_2"))  # (N, 6)
        fingertip_rot6d = quat_to_rotation_6d(api.lookup_robot_frame_rotation("panda_fingertip"))  # (N, 6)
        link4_rot6d = quat_to_rotation_6d(api.lookup_robot_frame_rotation("panda_link4"))  # (N, 6)

        joint_pos = api.joint_dict[agent]["joint_position"]  # (N, 9)
        joint_vel = api.joint_dict[agent]["joint_velocity"]  # (N, 9)
        joint_effort = self._get_normalized_joint_effort(api, agent)  # (N, 9)
        gripper_effort = joint_effort[:, 7:]  # (N, 2)
        gripper_open = api.gripper_openness().unsqueeze(-1)  # (N, 1)
        stage = self.current_stage  # (N, 1)

        def _f(*tensors: torch.Tensor) -> torch.Tensor:
            return torch.cat(
                [t.to(device=self.device, dtype=torch.float32) for t in tensors],
                dim=-1,
            )

        policy_obs = _f(
            cube1_pos,
            cube2_pos,
            fingertip_pos,  # 9
            cube1_rot6d,
            cube2_rot6d,
            fingertip_rot6d,  # 18
            gripper_open,
            gripper_effort,  # 3
            stage,  # 1
        )  # total: 31

        full_state_obs = _f(
            cube1_pos,
            cube2_pos,
            fingertip_pos,
            link4_pos,  # 12
            cube1_rot6d,
            cube2_rot6d,
            fingertip_rot6d,
            link4_rot6d,  # 24
            joint_pos,
            joint_vel,
            joint_effort,  # 27
            gripper_open,  # 1
            stage,  # 1
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

    def _external_get_dones(
        self,
        xy_threshold: float = 0.05,
        height_threshold: float = 0.005,
        height_diff: float = 0.0468,
    ):
        cube_1: RigidObject = self.cube_1
        cube_2: RigidObject = self.cube_2

        pos_diff_c12 = cube_1.data.root_pos_w - cube_2.data.root_pos_w

        xy_dist_c12 = torch.norm(pos_diff_c12[:, :2], dim=1)
        h_dist_c12 = torch.abs(pos_diff_c12[:, 2])

        xy_stacked = xy_dist_c12 < xy_threshold
        z_stacked = h_dist_c12 - height_diff < height_threshold

        gripper_stacked = torch.logical_and(xy_stacked, z_stacked)

        return gripper_stacked
