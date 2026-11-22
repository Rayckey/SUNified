# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
from kuafu.paths import OBJECT_ROOT

import os
from isaaclab.sensors.frame_transformer.frame_transformer import FrameTransformer
import torch
import math
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics, Vt

from isaacsim.core.utils.stage import get_current_stage

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg, RigidObject
from isaaclab.sim import schemas
from isaaclab.sim.utils import bind_physics_material
from isaaclab.utils import configclass
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import math as math_utils
from .panda_table_env_learned import PandaTableEnvLearnedCfg, PandaTableEnvLearned
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from kuafu.isaac.tasks.env_utils import random_pose_radial
from kuafu.mpc.simulation.spatial_utils import quat_to_rotation_6d
from isaaclab.markers.config import FRAME_MARKER_CFG
ISAAC_NUCLEUS_DIR = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac"
@configclass
class PandaCanEnvLearnedCfg(PandaTableEnvLearnedCfg):
    """Configuration for the pen cup manipulation environment."""
    observation_space = 22  # default state dim; use set_obs_mode() for full_state
    obs_mode: str = "state"  # "state" (22) or "full_state" (56)
    obs_dims = {"state": 22, "full_state": 56}
    episode_length_s = 25

    # Rigid body properties for the object to be placed (can)
    can_properties = sim_utils.RigidBodyPropertiesCfg(
        solver_position_iteration_count=32,
        solver_velocity_iteration_count=2,
        max_angular_velocity=1000.0,
        max_linear_velocity=1000.0,
        max_depenetration_velocity=100.0,
        disable_gravity=False,
    )
    can_box_collision_props = sim_utils.CollisionPropertiesCfg(
        collision_enabled=True,
        contact_offset=0.001,
        rest_offset=0.001,
    )
    can_collision_cylinder_radius = 0.02
    can_collision_cylinder_height = 0.1

    # Paths for can and pen holder assets

    can = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Can",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.05, 0.45, 0.05), rot=(1.0, 0.0, 0.0, 0.0)
        ),
        spawn=UsdFileCfg(
            usd_path=str(OBJECT_ROOT / 'pepper/pepper.usdc'),
            scale=(1.0,1.0,1.0),
            rigid_props=can_properties,
            mass_props=sim_utils.MassPropertiesCfg(mass=0.021),
            collision_props=None,
            semantic_tags=[("class", "can")],
        ),
    )


    # Frame at can body origin, offset +0.05 m along local +z (up axis of YCB can).
    marker_cfg_can = FRAME_MARKER_CFG.copy()
    marker_cfg_can.markers["frame"].scale = (0.1, 0.1, 0.1)
    marker_cfg_can.prim_path = "/Visuals/can_center_frame"
    can_center_frame = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/panda_link0",
        debug_vis=True,
        visualizer_cfg=marker_cfg_can,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Can",
                name="can_center",
                offset=OffsetCfg(pos=(0.0,  0.0, 0.0)),
            )
        ],
    )

    detectable_objects = {
        'can': (
            'cylindrical coke can rigid body, local -y is the upright axis and long axis, size (0.04, 0.1, 0.04)',
            '/World/envs/env_*/Can',
        ),
        'can_center_frame': (
            'cylindrical coke can, local -y is the upright axis and long axis, size (0.04, 0.1, 0.04)',
            '/Visuals/can_center_frame',
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
        "load_program": f"{_repo_dir}/resources/learned_programs/manipulation-can/2026-03-21_08-45-33",
        "initial_residual": f"{_repo_dir}/src/kuafu/skills/constraints/rewards.py",
        "initial_predicate": f"{_repo_dir}/src/kuafu/skills/constraints/predicates.py",
        "use_per_env_weight": False,
        "gripper_opened": [0.04, 0.04],
        "gripper_closed": [0.0, 0.0],
        "gripper_max_force": 40.0,
        "has_finger_contact_sensors": False,
    }
    del _repo_dir

    task_description = 'Pick up the fallen over can and place it back down in upright position'
class PandaCanEnvLearned(PandaTableEnvLearned):
    cfg: PandaCanEnvLearnedCfg

    def _setup_scene(self):
        # Call parent setup first (sets up robot, frames, terrain)
        super()._setup_scene()

        # Spawn can USD first, then replace mesh collisions with a simple box collider.
        stage = get_current_stage()
        can_cfg = self.cfg.can
        self.can_center_frame = FrameTransformer(self.cfg.can_center_frame)

        # High-friction physics material for the can (applied only to can meshes)
        can_material_cfg = sim_utils.materials.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
            friction_combine_mode="max",
        )
        can_material_path = "/World/Physics/canMaterial"
        can_material_cfg.func(can_material_path, can_material_cfg)
        # Spawn the USD file if spawn config exists
        if can_cfg.spawn is not None:
            can_cfg.spawn.func(
                can_cfg.prim_path,
                can_cfg.spawn,
                translation=can_cfg.init_state.pos,
                orientation=can_cfg.init_state.rot,
            )
        # Apply rigid-body APIs and replace mesh collisions with a box collider.
        if can_cfg.spawn is not None:
            matching_prims = sim_utils.find_matching_prims(can_cfg.prim_path)
            for prim in matching_prims:
                # Always apply RigidBodyAPI (it's safe to apply even if already present)
                if hasattr(can_cfg.spawn, 'rigid_props') and can_cfg.spawn.rigid_props is not None:
                    schemas.define_rigid_body_properties(prim.GetPath().pathString, can_cfg.spawn.rigid_props)

                # Apply mass properties if specified
                if hasattr(can_cfg.spawn, 'mass_props') and can_cfg.spawn.mass_props is not None:
                    schemas.define_mass_properties(prim.GetPath().pathString, can_cfg.spawn.mass_props)

                # Disable collision on the render meshes so only the box collider participates in contacts.
                for child_prim in Usd.PrimRange(prim):
                    if child_prim.IsA(UsdGeom.Mesh):
                        if child_prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                            child_prim.RemoveAPI(UsdPhysics.MeshCollisionAPI)
                        if child_prim.HasAPI(UsdPhysics.CollisionAPI):
                            child_prim.RemoveAPI(UsdPhysics.CollisionAPI)
                        if child_prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
                            child_prim.RemoveAPI(PhysxSchema.PhysxCollisionAPI)

                center_frame_path = f"{prim.GetPath().pathString}/can_center_frame"
                if not stage.GetPrimAtPath(center_frame_path).IsValid():
                    UsdGeom.Xform.Define(stage, center_frame_path)

                collider_path = f"{center_frame_path}/collision_cylinder"
                if not stage.GetPrimAtPath(collider_path).IsValid():
                    # Octagonal prism: close to cylindrical for grasping but flat faces
                    # prevent rolling on surfaces.
                    n = 8
                    r = self.cfg.can_collision_cylinder_radius
                    h = self.cfg.can_collision_cylinder_height
                    pts, fvc, fvi = [], [], []
                    for ring_y in (-h / 2.0, h / 2.0):
                        for i in range(n):
                            a = 2.0 * math.pi * i / n
                            pts.append(Gf.Vec3f(r * math.cos(a), ring_y, r * math.sin(a)))
                    fvc.append(n); fvi.extend(range(n - 1, -1, -1))       # bottom cap
                    fvc.append(n); fvi.extend(range(n, 2 * n))            # top cap
                    for i in range(n):                                     # side quads
                        j = (i + 1) % n
                        fvc.append(4); fvi.extend([i, j, j + n, i + n])
                    collider = UsdGeom.Mesh.Define(stage, collider_path)
                    collider.CreatePointsAttr(Vt.Vec3fArray(pts))
                    collider.CreateFaceVertexCountsAttr(Vt.IntArray(fvc))
                    collider.CreateFaceVertexIndicesAttr(Vt.IntArray(fvi))
                    mesh_api = UsdPhysics.MeshCollisionAPI.Apply(collider.GetPrim())
                    mesh_api.CreateApproximationAttr("convexHull")
                    UsdGeom.Imageable(collider).MakeInvisible()

                schemas.define_collision_properties(collider_path, self.cfg.can_box_collision_props, stage)
                bind_physics_material(collider_path, can_material_path, stage=stage)

        # Now create RigidObjects (they will find the RigidBodyAPI we just applied)
        self.can = RigidObject(self.cfg.can)

        self.scene.rigid_objects["can"] = self.can
        self.scene.sensors["can_center_frame"] = self.can_center_frame

        # clone and replicate
        self.scene.clone_environments(copy_from_source=True)
        self._hide_static_surface_visual_prims()
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu" and self.cfg.terrain is not None:
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        if self.recorder_manager is not None and len(env_ids) > 0:
            self.recorder_manager.record_pre_reset(env_ids)

        n = len(env_ids)
        env_ids_tensor = env_ids
        self._prepare_support_surface_reset(env_ids_tensor)
        origins = self._surface_origins(env_ids_tensor)
        angle_min, angle_max = self._spawn_angle_range()

        can_height = 0.02
        # Sample a single random pose for the can in front of the robot
        can_pos_sampled, can_quat = random_pose_radial(
            can_height,
            dist_min=0.35, 
            dist_max=0.55, 
            angle_min_rad=angle_min,
            angle_max_rad=angle_max,
            n=n,
            device=self.device,
            origins=origins,
            rotate_randomly=True
        )

        # Keep can at fixed height above the table (z = origin_z + 0.01)
        can_pos = torch.cat(
            [can_pos_sampled[:, :2], self._spawn_z_above_surface(env_ids_tensor, can_height)], dim=1
        )

        # can (dynamic)
        root_m = self.can.data.default_root_state.clone().index_select(
            0, env_ids_tensor
        )
        root_m[:, :3] = can_pos
        root_m[:, 3:7] = can_quat
        self.can.write_root_pose_to_sim(root_m[:, :7], env_ids=env_ids)  # type: ignore[arg-type]
        self.can.write_root_velocity_to_sim(
            torch.zeros((n, 6), device=self.device), env_ids=env_ids  # type: ignore[arg-type]
        )
        self.can.reset(env_ids)  # type: ignore[arg-type]

        super()._reset_idx(env_ids)

    def _get_observations(self) -> dict:
        self.update_data_in_controller()

        api = self.program_runtime.controller_api_wrapper
        agent = api.reference_agent
        if agent is None:
            raise RuntimeError("controller_api_wrapper.reference_agent is not set")

        # --- compute all components once ---
        can_pos      = api.get_object_xyz_position("can_center_frame")                          # (N, 3)
        fingertip_pos   = api.lookup_robot_frame_position("panda_fingertip")             # (N, 3)
        link4_pos       = api.lookup_robot_frame_position("panda_link4")                 # (N, 3)
        can_rot6d    = quat_to_rotation_6d(api.get_object_wxyz_quaternion("can_center_frame"))  # (N, 6)
        fingertip_rot6d = quat_to_rotation_6d(api.lookup_robot_frame_rotation("panda_fingertip"))  # (N, 6)
        link4_rot6d     = quat_to_rotation_6d(api.lookup_robot_frame_rotation("panda_link4"))      # (N, 6)
        joint_pos       = api.joint_dict[agent]["joint_position"]                        # (N, 9)
        joint_vel       = api.joint_dict[agent]["joint_velocity"]                        # (N, 9)
        joint_effort    = self._get_normalized_joint_effort(api, agent)                  # (N, 9)
        gripper_contact = self._get_normalized_gripper_contact(api)                      # (N, 2)
        joint_effort    = torch.cat([joint_effort[:, :7], gripper_contact], dim=-1)     # (N, 9)
        gripper_open    = api.gripper_openness().unsqueeze(-1)                            # (N, 1)
        stage           = self.current_stage                                             # (N, 1)

        def _f(*tensors):
            return torch.cat([t.to(device=self.device, dtype=torch.float32) for t in tensors], dim=-1)

        # Policy obs (31): no joint pos/vel/effort, no link4 — only task-relevant state
        policy_obs = _f(
            can_pos, fingertip_pos,          # 3+3 = 9
            can_rot6d, fingertip_rot6d,    # 6+6 = 18
            gripper_open, gripper_contact,                  # 1+2   = 3
            stage,                                          # 1
        )  # total: 31

        # Full state obs (65): everything needed to recreate the env
        full_state_obs = _f(
            can_pos, fingertip_pos, link4_pos,          # 3+3+3 = 9
            can_rot6d, fingertip_rot6d, link4_rot6d,  # 6+6+6 = 18
            joint_pos, joint_vel, joint_effort,                         # 9+9+9   = 27
            gripper_open,                                               # 1
            stage,                                                      # 1
        )  # total: 56

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

        can_pos = self.can_center_frame.data.target_pos_w.squeeze(1)
        fingertip_pos = self.ee_frame.data.target_pos_w.squeeze(1)
        can_dist = torch.linalg.norm(fingertip_pos - can_pos, dim=-1)
        gripper_openness = self.program_runtime.controller_api_wrapper.gripper_openness()
        reach_reward = torch.exp(-8.0 * can_dist)
        grasp_reward = torch.exp(-40.0 * can_dist) * torch.clamp(
            1.0 - gripper_openness, 0.0, 1.0
        )

        can_quat = self.can.data.root_state_w[:, 3:7]
        local_neg_y = torch.zeros(
            can_quat.shape[0], 3, device=can_quat.device, dtype=can_quat.dtype
        )
        local_neg_y[:, 1] = -1.0
        can_neg_y_world = math_utils.quat_apply(can_quat, local_neg_y)
        upright_reward = torch.clamp(can_neg_y_world[:, 2], 0.0, 1.0).square()

        can_lin_vel = self.can.data.root_state_w[:, 7:10]
        can_ang_vel = self.can.data.root_state_w[:, 10:13]
        static_reward = torch.exp(
            -2.0
            * (
                torch.linalg.norm(can_lin_vel, dim=-1)
                + torch.linalg.norm(can_ang_vel, dim=-1)
            )
        )
        gripper_open = torch.clamp(gripper_openness / 0.5, 0.0, 1.0)
        fingertip_clear = torch.clamp((can_dist - 0.05) / 0.10, 0.0, 1.0)
        placed_reward = upright_reward * static_reward * gripper_open * fingertip_clear

        return (
            0.5 * reach_reward
            + 0.5 * grasp_reward
            + 3.0 * upright_reward
            + 1.5 * placed_reward
        )

    def _external_get_dones(self, ee_can_min_dist_m: float = 0.05):
        # Termination: can is nearly static, gripper is open, can -y points upward,
        # and panda_fingertip is at least ee_can_min_dist_m from the can frame.
        api = self.program_runtime.controller_api_wrapper
        can_pos = api.get_object_xyz_position("can_center_frame")
        fingertip_pos = api.lookup_robot_frame_position("panda_fingertip")
        ee_can_dist = torch.norm(fingertip_pos - can_pos, dim=1)
        fingertip_far_from_can = ee_can_dist >= ee_can_min_dist_m

        can_lin_vel = self.can.data.root_state_w[:, 7:10]
        can_ang_vel = self.can.data.root_state_w[:, 10:13]
        can_lin_vel_norm = torch.norm(can_lin_vel, dim=1)
        can_ang_vel_norm = torch.norm(can_ang_vel, dim=1)
        can_almost_static = torch.logical_and(can_lin_vel_norm < 0.05, can_ang_vel_norm < 0.05)

        can_quat = self.can.data.root_state_w[:, 3:7]
        local_neg_y = torch.zeros(can_quat.shape[0], 3, device=can_quat.device, dtype=can_quat.dtype)
        local_neg_y[:, 1] = -1.0
        can_neg_y_world = math_utils.quat_apply(can_quat, local_neg_y)
        can_neg_y_up = can_neg_y_world[:, 2] > 0.9

        gripper_opened = self.program_runtime.controller_api_wrapper.gripper_openness() > 0.5
        task_done = torch.logical_and(
            torch.logical_and(
                torch.logical_and(can_almost_static, can_neg_y_up),
                gripper_opened,
            ),
            fingertip_far_from_can,
        )
        if self.cfg.sun_program_evaluator_stage_override or bool(
            getattr(self.cfg, "ignore_stage_gate_for_success", False)
        ):
            return task_done

        past_stage = self.current_stage.flatten() > int(self.program_runtime.compiled_program.stage_count() / 2)
        return torch.logical_and(task_done, past_stage)
