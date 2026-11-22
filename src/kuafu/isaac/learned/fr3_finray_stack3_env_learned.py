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
from isaaclab.sim import SimulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sensors import TiledCamera, TiledCameraCfg, CameraCfg
from kuafu.isaac.tasks.env_learned import BaseLearnedEnvCfg, BaseLearnedEnv, ISAAC_NUCLEUS_DIR
from kuafu.isaac.tasks.env_utils import random_pose_radial
from kuafu.mpc.simulation.spatial_utils import quat_to_rotation_6d
from kuafu.isaac.learned.franka_table_env_learned import FrankaTableEnvLearnedCfg, FrankaTableEnvLearned

@configclass
class FR3FinRayStack3EnvLearnedCfg(FrankaTableEnvLearnedCfg):

    observation_space = 40  # default state dim; use set_obs_mode() for full_state
    episode_length_s = 40.0
    obs_mode: str = "state"  # "state" (40) or "full_state" (74)
    obs_dims = {"state": 40, "full_state": 74}

        # Override robot init_state only, keep rest from base
    def __post_init__(self):
        """Post-initialization to override robot initial state."""
        super().__post_init__()
        # Override only the initial state, keeping the rest of robot config from parent
        self.robot.init_state = ArticulationCfg.InitialStateCfg(
            joint_pos={
                "fr3_joint1": -1.0,
                "fr3_joint2": -0.569,
                "fr3_joint3": 0.0,
                "fr3_joint4": -2.810,
                "fr3_joint5": 0.0,
                "fr3_joint6": 3.037,
                "fr3_joint7": 0.741,
                "robotiq_85_right_knuckle_joint": 0,
                "robotiq_85_left_knuckle_joint": 0,
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
            pos=[0.4, 0.0,  0.0203], rot=[1, 0, 0, 0]
        ),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/blue_block.usd",
            scale=(1.0, 1.0, 1.0),
            rigid_props=cube_properties,
            mass_props=sim_utils.MassPropertiesCfg(mass=0.021),
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
            mass_props=sim_utils.MassPropertiesCfg(mass=0.021),
            semantic_tags=[("class", "cube_2")],
        ),
    )
    cube_3 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Cube_3",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.60, -0.10, 0.0203], rot=[1, 0, 0, 0]
        ),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/green_block.usd",
            scale=(1.0, 1.0, 1.0),
            rigid_props=cube_properties,
            mass_props=sim_utils.MassPropertiesCfg(mass=0.021),
            semantic_tags=[("class", "cube_3")],
        ),
    )
    
    detectable_objects = {
        'cube_2': (
            'mesh of a green cube, symmetric on all axes, size: (0.05, 0.05, 0.05)',
            '/World/envs/env_*/Cube_2',
        ),
        'cube_1': (
            'mesh of a red cube, symmetric on all axes, size: (0.05, 0.05, 0.05)',
            '/World/envs/env_*/Cube_1',
        ),
        'cube_3': (
            'mesh of a green cube, symmetric on all axes, size: (0.05, 0.05, 0.05)',
            '/World/envs/env_*/Cube_3',
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

    _kuafu_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(_kuafu_dir)))
    mpc_config = {
            "controller_name": "FrankaIKController",
            "device": "cuda",
            "ee_frame": "panda_fingertip",
            "use_per_env_weight": False,
            "load_program": f"{_repo_dir}/resources/learned_programs/manipulation-stack3/2026-03-25_19-53-05",
            "initial_residual": f"{_repo_dir}/src/kuafu/skills/constraints/rewards.py",
            "initial_predicate": f"{_repo_dir}/src/kuafu/skills/constraints/predicates.py",
            "gripper_opened": [0.0, 0.0],
            "gripper_closed": [0.8, -0.8],
            "gripper_max_force": 1.0,
        }
    del _kuafu_dir, _repo_dir

    task_description = 'stack the cubes'
class FR3FinRayStack3EnvLearned(FrankaTableEnvLearned):
    """Stack three cubes environment using learned program functions.
    
    Inherits common manipulation logic from BaseLearnedEnv.
    Only implements task-specific scene setup, reset, observations, and termination.
    """
    cfg: FR3FinRayStack3EnvLearnedCfg

    def _setup_scene(self):
        super()._setup_scene()

        # Task-specific objects
        self.cube_1 = RigidObject(self.cfg.cube_1)
        self.cube_2 = RigidObject(self.cfg.cube_2)
        self.cube_3 = RigidObject(self.cfg.cube_3)

        # Register task objects in scene
        self.scene.rigid_objects["cube_1"] = self.cube_1
        self.scene.rigid_objects["cube_2"] = self.cube_2
        self.scene.rigid_objects["cube_3"] = self.cube_3

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        self.scene.clone_environments(copy_from_source=False)
        self._hide_static_surface_visual_prims()

        # Add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=1000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        if self.recorder_manager is not None and len(env_ids) > 0:
            self.recorder_manager.record_pre_reset(env_ids)

        n = len(env_ids)
        env_ids_tensor = env_ids
        origins = self.scene.env_origins.index_select(0, env_ids_tensor)
        z_top = origins[:, 2:3]  + 0.0203

        angle_lower = -torch.pi / 2 + 0.2
        angle_upper = -0.2
        angle_step = (angle_upper - angle_lower) / 3

        # Three sectors spanning the shrunken spawnable region.
        pos_left, quat_left = random_pose_radial(
             0.0203,
            dist_min=0.4,
            dist_max=0.6,
            angle_min_rad=angle_lower,
            angle_max_rad=angle_lower + angle_step,
            n=n,
            device=self.device,
            origins=origins,
        )
        pos_center, quat_center = random_pose_radial(
            0.0203,
            dist_min=0.4,
            dist_max=0.6,
            angle_min_rad=angle_lower + angle_step,
            angle_max_rad=angle_lower + 2 * angle_step,
            n=n,
            device=self.device,
            origins=origins,
        )
        pos_right, quat_right = random_pose_radial(
            0.0203,
            dist_min=0.4,
            dist_max=0.6,
            angle_min_rad=angle_lower + 2 * angle_step,
            angle_max_rad=angle_upper,
            n=n,
            device=self.device,
            origins=origins,
        )

        # Randomly permute which cube goes in which sector
        perm = torch.argsort(torch.rand(n, 3, device=self.device), dim=1)
        pos_all = torch.stack([pos_left[:, :2], pos_center[:, :2], pos_right[:, :2]], dim=1)
        quat_all = torch.stack([quat_left, quat_center, quat_right], dim=1)

        cube1_xy = pos_all[torch.arange(n, device=self.device), perm[:, 0], :]
        cube2_xy = pos_all[torch.arange(n, device=self.device), perm[:, 1], :]
        cube3_xy = pos_all[torch.arange(n, device=self.device), perm[:, 2], :]
        cube1_quat = quat_all[torch.arange(n, device=self.device), perm[:, 0], :]
        cube2_quat = quat_all[torch.arange(n, device=self.device), perm[:, 1], :]
        cube3_quat = quat_all[torch.arange(n, device=self.device), perm[:, 2], :]

        cube1_pos = torch.cat([cube1_xy, z_top], dim=1)
        cube2_pos = torch.cat([cube2_xy, z_top], dim=1)
        cube3_pos = torch.cat([cube3_xy, z_top], dim=1)

        for cube, pos, quat in [
            (self.cube_1, cube1_pos, cube1_quat),
            (self.cube_2, cube2_pos, cube2_quat),
            (self.cube_3, cube3_pos, cube3_quat),
        ]:
            root_m = cube.data.default_root_state.clone().index_select(
                0, env_ids_tensor
            )
            root_m[:, :3] = pos
            root_m[:, 3:7] = quat
            cube.write_root_pose_to_sim(root_m[:, :7], env_ids=env_ids)
            cube.write_root_velocity_to_sim(
                torch.zeros((len(env_ids), 6), device=self.device), env_ids=env_ids
            )
            cube.reset(env_ids)

        super()._reset_idx(env_ids)

    def _external_get_dones(
        self,
        xy_threshold: float = 0.03,
        height_threshold: float = 0.01,
        height_diff: float = 0.0468,
        min_gripper_cube_dist: float = 0.05,
    ):
        cube_1: RigidObject = self.cube_1
        cube_2: RigidObject = self.cube_2
        cube_3: RigidObject = self.cube_3

        pos_diff_c12 = cube_1.data.root_pos_w - cube_2.data.root_pos_w
        pos_diff_c23 = cube_2.data.root_pos_w - cube_3.data.root_pos_w
        pos_diff_c13 = cube_1.data.root_pos_w - cube_3.data.root_pos_w

        # Learned-program order: cube_2 on cube_1, then cube_3 on cube_2.
        xy_dist_c12 = torch.norm(pos_diff_c12[:, :2], dim=1)
        xy_dist_c23 = torch.norm(pos_diff_c23[:, :2], dim=1)
        xy_dist_c13 = torch.norm(pos_diff_c13[:, :2], dim=1)

        xy_stacked = torch.logical_and(
            torch.logical_and(xy_dist_c12 < xy_threshold, xy_dist_c23 < xy_threshold),
            xy_dist_c13 < xy_threshold,
        )

        cube2_target_z = cube_1.data.root_pos_w[:, 2] + height_diff
        cube3_target_z = cube2_target_z + height_diff
        z_stacked = torch.logical_and(
            torch.abs(cube_2.data.root_pos_w[:, 2] - cube2_target_z) < height_threshold,
            torch.abs(cube_3.data.root_pos_w[:, 2] - cube3_target_z) < height_threshold,
        )

        api = self.program_runtime.controller_api_wrapper
        gripper_openness = api.gripper_openness().flatten()
        fingertip_pos = api.lookup_robot_frame_position("panda_fingertip")

        dist_g1 = torch.norm(fingertip_pos - cube_1.data.root_pos_w, dim=1)
        dist_g2 = torch.norm(fingertip_pos - cube_2.data.root_pos_w, dim=1)
        dist_g3 = torch.norm(fingertip_pos - cube_3.data.root_pos_w, dim=1)
        gripper_far_from_all_cubes = torch.logical_and(
            dist_g1 >= min_gripper_cube_dist,
            torch.logical_and(dist_g2 >= min_gripper_cube_dist, dist_g3 >= min_gripper_cube_dist),
        )
        gripper_stacked = torch.logical_and(
            gripper_openness > 0.9,
            torch.logical_and(torch.logical_and(xy_stacked, z_stacked), gripper_far_from_all_cubes),
        )

        return gripper_stacked

    def _get_observations(self) -> dict:
        self.update_data_in_controller()

        api = self.program_runtime.controller_api_wrapper
        agent = api.reference_agent
        if agent is None:
            raise RuntimeError("controller_api_wrapper.reference_agent is not set")

        cube1_pos = api.get_object_xyz_position("cube_1")  # (N, 3)
        cube2_pos = api.get_object_xyz_position("cube_2")  # (N, 3)
        cube3_pos = api.get_object_xyz_position("cube_3")  # (N, 3)
        fingertip_pos = api.lookup_robot_frame_position("panda_fingertip")  # (N, 3)
        link4_pos = api.lookup_robot_frame_position("panda_link4")  # (N, 3)

        cube1_rot6d = quat_to_rotation_6d(api.get_object_wxyz_quaternion("cube_1"))  # (N, 6)
        cube2_rot6d = quat_to_rotation_6d(api.get_object_wxyz_quaternion("cube_2"))  # (N, 6)
        cube3_rot6d = quat_to_rotation_6d(api.get_object_wxyz_quaternion("cube_3"))  # (N, 6)
        fingertip_rot6d = quat_to_rotation_6d(api.lookup_robot_frame_rotation("panda_fingertip"))  # (N, 6)
        link4_rot6d = quat_to_rotation_6d(api.lookup_robot_frame_rotation("panda_link4"))  # (N, 6)

        joint_pos = api.joint_dict[agent]["joint_position"]  # (N, 9)
        joint_vel = api.joint_dict[agent]["joint_velocity"]  # (N, 9)
        joint_effort = self._get_normalized_joint_effort(api, agent)  # (N, 9)
        gripper_contact = self._get_normalized_gripper_contact(api)  # (N, 2)
        joint_effort = torch.cat([joint_effort[:, :7], gripper_contact], dim=-1)  # (N, 9)
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
            cube3_pos,
            fingertip_pos,  # 12
            cube1_rot6d,
            cube2_rot6d,
            cube3_rot6d,
            fingertip_rot6d,  # 24
            gripper_open,
            gripper_contact,  # 3
            stage,  # 1
        )  # total: 40

        full_state_obs = _f(
            cube1_pos,
            cube2_pos,
            cube3_pos,
            fingertip_pos,
            link4_pos,  # 15
            cube1_rot6d,
            cube2_rot6d,
            cube3_rot6d,
            fingertip_rot6d,
            link4_rot6d,  # 30
            joint_pos,
            joint_vel,
            joint_effort,  # 27
            gripper_open,  # 1
            stage,  # 1
        )  # total: 74

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

        cube_height = 0.0468
        cube_pos = [
            self.cube_1.data.root_pos_w,
            self.cube_2.data.root_pos_w,
            self.cube_3.data.root_pos_w,
        ]
        height_offset = torch.zeros_like(cube_pos[0])
        height_offset[:, 2] = cube_height

        pair_rewards = []
        for top_idx, base_idx in ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)):
            target_pos = cube_pos[base_idx] + height_offset
            pair_dist = torch.linalg.norm(cube_pos[top_idx] - target_pos, dim=-1)
            pair_rewards.append(torch.exp(-25.0 * pair_dist))
        best_pair_reward = torch.stack(pair_rewards, dim=0).amax(dim=0)

        stack_rewards = []
        for top_idx, mid_idx, base_idx in (
            (0, 1, 2),
            (0, 2, 1),
            (1, 0, 2),
            (1, 2, 0),
            (2, 0, 1),
            (2, 1, 0),
        ):
            mid_target = cube_pos[base_idx] + height_offset
            top_target = cube_pos[base_idx] + 2.0 * height_offset
            stack_dist = torch.linalg.norm(cube_pos[mid_idx] - mid_target, dim=-1)
            stack_dist = stack_dist + torch.linalg.norm(
                cube_pos[top_idx] - top_target, dim=-1
            )
            stack_rewards.append(torch.exp(-15.0 * stack_dist))
        best_stack_reward = torch.stack(stack_rewards, dim=0).amax(dim=0)

        return 1.5 * best_pair_reward + 4.0 * best_stack_reward
