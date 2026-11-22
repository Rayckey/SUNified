# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
from kuafu.paths import OBJECT_ROOT

import os
from isaaclab.sensors.frame_transformer.frame_transformer import FrameTransformer
import torch
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema

from isaacsim.core.utils.stage import get_current_stage

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg, RigidObject
from isaaclab.sim import schemas
from isaaclab.sim.utils import bind_physics_material
from isaaclab.utils import configclass
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import math as math_utils
from .kinova_table_env_learned import KinovaTableEnvLearnedCfg, KinovaTableEnvLearned
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from kuafu.isaac.tasks.env_utils import random_pose_radial
from kuafu.mpc.simulation.spatial_utils import quat_to_rotation_6d

@configclass
class KinovaPenCupEnvLearnedCfg(KinovaTableEnvLearnedCfg):
    """Configuration for the pen cup manipulation environment."""
    observation_space = 25  # default state dim; use set_obs_mode() for full_state
    obs_mode: str = "state"  # "state" (31) or "full_state" (65)
    obs_dims = {"state": 25, "full_state": 59}
    episode_length_s = 20
    manipulator_effort_print_every: int = 25
    manipulator_effort_monitor_stages: tuple[int, ...] = (5, 6)
    success_xy_tolerance = 0.03
    success_z_tolerance = 0.06
    success_upright_alignment = 0.7071067811865476  # cos(45 deg)
    success_gripper_marker_distance = 0.05
    pen_holder_height_m: float = 0.11

    # Rigid body properties for the object to be placed (marker)
    marker_properties = sim_utils.RigidBodyPropertiesCfg(
        solver_position_iteration_count=32,
        solver_velocity_iteration_count=2,
        max_angular_velocity=1000.0,
        max_linear_velocity=1000.0,
        max_depenetration_velocity=100.0,
        disable_gravity=False,
    )
    marker_box_collision_props = sim_utils.CollisionPropertiesCfg(
        collision_enabled=True,
        contact_offset=0.001,
        rest_offset=0.001,
    )
    marker_collision_box_size = (0.02, 0.137, 0.02)


    # Paths for marker and pen holder assets
    _marker_path = str(OBJECT_ROOT / 'boardmaster/boardmaster.usdc')

    marker = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Marker",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.5, 0.0, 0.012), rot=(1, 0, 0, 0)
        ),
        spawn=UsdFileCfg(
            usd_path=_marker_path,
            scale=(1,1,1),
            rigid_props=marker_properties,
            mass_props=sim_utils.MassPropertiesCfg(mass=0.021),
            collision_props=None,
            semantic_tags=[("class", "marker")],
        ),
    )

    # Pen holder - static/kinematic object (will not move during simulation)
    _pen_holder_path = str(OBJECT_ROOT / 'pen_holder/pen_holder.usdc')
    pen_holder = RigidObjectCfg(
        prim_path="/World/envs/env_.*/PenHolder",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.4, -0.15, 0.05), rot=(1, 0, 0, 0)
        ),
        spawn=UsdFileCfg(
            usd_path=_pen_holder_path,
            scale=(1,1,1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,  # Makes the object static - it will not move during simulation
                disable_gravity=True,  # Additional safeguard: disable gravity (though kinematic objects ignore it anyway)
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.001,  # Very small offset to allow gripper to get close
                rest_offset=0.001,
            ),
        ),
    )


    detectable_objects = {
        'marker': (
            'center frame of cyclindrical marker/pen lying on its side, local y axis is the length axis, size: (0.02, 0.137, 0.02)',
            '/World/envs/env_*/Marker',
        ),
        'pen_holder': (
            'center frame of pen holder, cyclindrical object with openings along local +z axis for pens, size: (0.08, 0.08, 0.11)',
            '/World/envs/env_*/PenHolder',
        ),
        'table': (
            'center frame of table, size: (0.3, 0.9, 0.4)',
            '/World/envs/env_*/Table',
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
        "load_program": f"{_repo_dir}/resources/learned_programs/manipulation-pen-cup/2026-01-22_01-52-30",
        "initial_residual": f"{_repo_dir}/src/kuafu/skills/constraints/rewards.py",
        "initial_predicate": f"{_repo_dir}/src/kuafu/skills/constraints/predicates.py",
        "use_per_env_weight": False,
        "gripper_opened": [0.0, 0.0],
        "gripper_closed": [0.8, 0.8],
        "gripper_closed_command": [0.8, 0.8],
        "gripper_max_force": 1.0,
        "manipulator_high_effort_threshold": 8.0,
    }
    del _repo_dir

    task_description = 'pick up the fallen over marker, stick it upright, and insert it into the pen holder'
class KinovaPenCupEnvLearned(KinovaTableEnvLearned):
    cfg: KinovaPenCupEnvLearnedCfg

    def _setup_scene(self):
        # Call parent setup first (sets up robot, frames, terrain)
        super()._setup_scene()

        # Spawn marker USD first, then replace mesh collisions with a simple box collider.
        stage = get_current_stage()
        marker_cfg = self.cfg.marker

        # High-friction physics material for the marker box collider.
        marker_material_cfg = sim_utils.materials.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
            friction_combine_mode="max",
        )
        marker_material_path = "/World/Physics/markerMaterial"
        marker_material_cfg.func(marker_material_path, marker_material_cfg)
        # Spawn the USD file if spawn config exists
        if marker_cfg.spawn is not None:
            marker_cfg.spawn.func(
                marker_cfg.prim_path,
                marker_cfg.spawn,
                translation=marker_cfg.init_state.pos,
                orientation=marker_cfg.init_state.rot,
            )
        # Apply rigid-body APIs and replace mesh collisions with a box collider.
        if marker_cfg.spawn is not None:
            matching_prims = sim_utils.find_matching_prims(marker_cfg.prim_path)
            for prim in matching_prims:
                # Always apply RigidBodyAPI (it's safe to apply even if already present)
                if hasattr(marker_cfg.spawn, 'rigid_props') and marker_cfg.spawn.rigid_props is not None:
                    schemas.define_rigid_body_properties(prim.GetPath().pathString, marker_cfg.spawn.rigid_props)

                # Apply mass properties if specified
                if hasattr(marker_cfg.spawn, 'mass_props') and marker_cfg.spawn.mass_props is not None:
                    schemas.define_mass_properties(prim.GetPath().pathString, marker_cfg.spawn.mass_props)

                # Remove collisions from render meshes so only the box collider participates in contacts.
                for child_prim in Usd.PrimRange(prim):
                    if child_prim.IsA(UsdGeom.Mesh):
                        if child_prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                            child_prim.RemoveAPI(UsdPhysics.MeshCollisionAPI)
                        if child_prim.HasAPI(UsdPhysics.CollisionAPI):
                            child_prim.RemoveAPI(UsdPhysics.CollisionAPI)
                        if child_prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
                            child_prim.RemoveAPI(PhysxSchema.PhysxCollisionAPI)

                collider_path = f"{prim.GetPath().pathString}/marker_collision_box"
                if not stage.GetPrimAtPath(collider_path).IsValid():
                    collider = UsdGeom.Cube.Define(stage, collider_path)
                    collider.CreateSizeAttr(1.0)
                    collider_xform = UsdGeom.Xformable(collider)
                    collider_xform.AddScaleOp().Set(self.cfg.marker_collision_box_size)
                    UsdGeom.Imageable(collider).MakeInvisible()

                schemas.define_collision_properties(collider_path, self.cfg.marker_box_collision_props, stage)
                bind_physics_material(collider_path, marker_material_path, stage=stage)

        # Spawn pen holder USD file, then apply RigidBodyAPI and collision approximation before creating RigidObject
        pen_holder_cfg = self.cfg.pen_holder
        if pen_holder_cfg.spawn is not None:
            pen_holder_cfg.spawn.func(
                pen_holder_cfg.prim_path,
                pen_holder_cfg.spawn,
                translation=pen_holder_cfg.init_state.pos,
                orientation=pen_holder_cfg.init_state.rot,
            )
        # Use the collision boxes authored in the pen_holder USD. Keep the visual mesh render-only.
        if pen_holder_cfg.spawn is not None:
            matching_prims = sim_utils.find_matching_prims(pen_holder_cfg.prim_path)
            for prim in matching_prims:
                # Apply RigidBodyAPI if not already present
                if not UsdPhysics.RigidBodyAPI(prim):
                    if hasattr(pen_holder_cfg.spawn, 'rigid_props') and pen_holder_cfg.spawn.rigid_props is not None:
                        schemas.define_rigid_body_properties(prim.GetPath().pathString, pen_holder_cfg.spawn.rigid_props)

                # Apply mass properties if specified
                if hasattr(pen_holder_cfg.spawn, 'mass_props') and pen_holder_cfg.spawn.mass_props is not None:
                    schemas.define_mass_properties(prim.GetPath().pathString, pen_holder_cfg.spawn.mass_props)

                for child_prim in Usd.PrimRange(prim):
                    child_path = child_prim.GetPath().pathString
                    is_holder_collision = "/collisions/" in child_path

                    if child_prim.IsA(UsdGeom.Mesh) and not is_holder_collision:
                        if child_prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                            child_prim.RemoveAPI(UsdPhysics.MeshCollisionAPI)
                        if child_prim.HasAPI(UsdPhysics.CollisionAPI):
                            child_prim.RemoveAPI(UsdPhysics.CollisionAPI)
                        if child_prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
                            child_prim.RemoveAPI(PhysxSchema.PhysxCollisionAPI)
                        if child_prim.HasAPI(PhysxSchema.PhysxConvexDecompositionCollisionAPI):
                            child_prim.RemoveAPI(PhysxSchema.PhysxConvexDecompositionCollisionAPI)
                        continue

                    if is_holder_collision and child_prim.IsA(UsdGeom.Boundable):
                        UsdPhysics.CollisionAPI.Apply(child_prim)
                        if hasattr(pen_holder_cfg.spawn, 'collision_props') and pen_holder_cfg.spawn.collision_props is not None:
                            schemas.define_collision_properties(child_path, pen_holder_cfg.spawn.collision_props, stage)
        # Now create RigidObjects (they will find the RigidBodyAPI we just applied)
        self.marker = RigidObject(self.cfg.marker)
        self.pen_holder = RigidObject(self.cfg.pen_holder)

        self.scene.rigid_objects["marker"] = self.marker
        self.scene.rigid_objects["pen_holder"] = self.pen_holder

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

        if hasattr(self, "_pen_cup_manipulator_effort_counter"):
            self._pen_cup_manipulator_effort_counter[env_ids] = 0

        n = len(env_ids)
        env_ids_tensor = env_ids
        self._prepare_support_surface_reset(env_ids_tensor)
        origins = self._surface_origins(env_ids_tensor)
        right_angle_min, right_angle_max = self._spawn_angle_range(0, 2)
        left_angle_min, left_angle_max = self._spawn_angle_range(1, 2)


        # Split the configured Kinova workspace into two sectors. Decide the side assignment first
        # so each object can keep its own radial distance range.
        marker_on_left = torch.rand(n, device=self.device) > 0.5
        marker_pos_right, marker_quat_right = random_pose_radial(
            0.012,
            dist_min=0.4,
            dist_max=0.6,
            angle_min_rad=right_angle_min,
            angle_max_rad=right_angle_max,
            n=n,
            device=self.device,
            origins=origins,
            rotate_randomly=True,
        )
        marker_pos_left, marker_quat_left = random_pose_radial(
            0.012,
            dist_min=0.4,
            dist_max=0.6,
            angle_min_rad=left_angle_min,
            angle_max_rad=left_angle_max,
            n=n,
            device=self.device,
            origins=origins,
            rotate_randomly=True,
        )
        holder_pos_right, holder_quat_right = random_pose_radial(
            0.05,
            dist_min=0.3,
            dist_max=0.5,
            angle_min_rad=right_angle_min,
            angle_max_rad=right_angle_max,
            n=n,
            device=self.device,
            origins=origins,
            rotate_randomly=True,
        )
        holder_pos_left, holder_quat_left = random_pose_radial(
            0.05,
            dist_min=0.3,
            dist_max=0.5,
            angle_min_rad=left_angle_min,
            angle_max_rad=left_angle_max,
            n=n,
            device=self.device,
            origins=origins,
            rotate_randomly=True,
        )

        marker_on_left_2 = marker_on_left.unsqueeze(1).expand(-1, 2)
        marker_on_left_4 = marker_on_left.unsqueeze(1).expand(-1, 4)
        marker_xy = torch.where(marker_on_left_2, marker_pos_left[:, :2], marker_pos_right[:, :2])
        holder_xy = torch.where(marker_on_left_2, holder_pos_right[:, :2], holder_pos_left[:, :2])
        holder_height = self._sample_object_spawn_height_offset(n) + 0.5 * float(self.cfg.pen_holder_height_m)
        marker_pos = torch.cat(
            [marker_xy, self._spawn_z_above_surface(env_ids_tensor, 0.012)], dim=1
        )
        holder_pos = torch.cat(
            [holder_xy, self._spawn_z_above_surface(env_ids_tensor, holder_height)], dim=1
        )
        marker_quat = torch.where(marker_on_left_4, marker_quat_left, marker_quat_right)
        holder_quat = torch.where(marker_on_left_4, holder_quat_right, holder_quat_left)

        # Wiggle the holder's z-axis: tilt up to 20 degrees in a random direction.
        max_tilt_rad = 5.0 * torch.pi / 180.0
        tilt_angle = torch.rand(n, device=self.device) * max_tilt_rad
        tilt_azimuth = torch.rand(n, device=self.device) * 2.0 * torch.pi
        tilt_axis = torch.stack(
            [
                torch.cos(tilt_azimuth),
                torch.sin(tilt_azimuth),
                torch.zeros(n, device=self.device),
            ],
            dim=1,
        )
        tilt_quat = math_utils.quat_from_angle_axis(tilt_angle, tilt_axis)
        holder_quat = math_utils.quat_mul(tilt_quat, holder_quat)

        # Marker (dynamic)
        root_m = self.marker.data.default_root_state.clone().index_select(
            0, env_ids_tensor
        )
        root_m[:, :3] = marker_pos
        root_m[:, 3:7] = marker_quat
        self.marker.write_root_pose_to_sim(root_m[:, :7], env_ids=env_ids)  # type: ignore[arg-type]
        self.marker.write_root_velocity_to_sim(
            torch.zeros((n, 6), device=self.device), env_ids=env_ids  # type: ignore[arg-type]
        )
        self.marker.reset(env_ids)  # type: ignore[arg-type]

        # Pen holder (kinematic) — same pose randomization
        root_h = self.pen_holder.data.default_root_state.clone().index_select(
            0, env_ids_tensor
        )
        root_h[:, :3] = holder_pos
        root_h[:, 3:7] = holder_quat
        self.pen_holder.write_root_pose_to_sim(root_h[:, :7], env_ids=env_ids)  # type: ignore[arg-type]
        self.pen_holder.write_root_velocity_to_sim(
            torch.zeros((n, 6), device=self.device), env_ids=env_ids
        )
        self.pen_holder.reset(env_ids)  # type: ignore[arg-type]

        super()._reset_idx(env_ids)

    def _get_observations(self) -> dict:
        self.update_data_in_controller()

        api = self.program_runtime.controller_api_wrapper
        agent = api.reference_agent
        if agent is None:
            raise RuntimeError("controller_api_wrapper.reference_agent is not set")

        # --- compute all components once ---
        marker_pos      = api.get_object_xyz_position("marker")                          # (N, 3)
        holder_pos      = api.get_object_xyz_position("pen_holder")                      # (N, 3)
        fingertip_pos   = api.lookup_robot_frame_position("panda_fingertip")             # (N, 3)
        link4_pos       = api.lookup_robot_frame_position("panda_link4")                 # (N, 3)
        marker_rot6d    = quat_to_rotation_6d(api.get_object_wxyz_quaternion("marker"))  # (N, 6)
        fingertip_rot6d = quat_to_rotation_6d(api.lookup_robot_frame_rotation("panda_fingertip"))  # (N, 6)
        link4_rot6d     = quat_to_rotation_6d(api.lookup_robot_frame_rotation("panda_link4"))      # (N, 6)
        joint_pos       = api.joint_dict[agent]["joint_position"]                        # (N, 9)
        joint_vel       = api.joint_dict[agent]["joint_velocity"]                        # (N, 9)
        joint_effort    = self._get_normalized_joint_effort(api, agent)                  # (N, 9)
        gripper_contact = self._get_normalized_gripper_contact(api)                      # (N, 2)
        joint_effort    = torch.cat([joint_effort[:, :7], gripper_contact], dim=-1)     # (N, 9)
        gripper_open    = api.gripper_openness().unsqueeze(-1)                            # (N, 1)
        stage           = self.current_stage                                             # (N, 1)
        self._print_manipulator_effort_diagnostics(api, agent)

        def _f(*tensors):
            return torch.cat([t.to(device=self.device, dtype=torch.float32) for t in tensors], dim=-1)

        # Policy obs (31): no joint pos/vel/effort, no link4 — only task-relevant state
        policy_obs = _f(
            marker_pos, holder_pos, fingertip_pos,          # 3+3+3 = 9
            # marker_rot6d, holder_rot6d, fingertip_rot6d,    # 6+6+6 = 18
            marker_rot6d, fingertip_rot6d,    # 6+6 = 12
            gripper_open, gripper_contact,                  # 1+2   = 3
            stage,                                          # 1
        )  # total: 31

        # Full state obs (65): everything needed to recreate the env
        full_state_obs = _f(
            marker_pos, holder_pos, fingertip_pos, link4_pos,          # 3+3+3+3 = 12
            # marker_rot6d, holder_rot6d, fingertip_rot6d, link4_rot6d,  # 6+6+6+6 = 24
            marker_rot6d, fingertip_rot6d, link4_rot6d,  # 6+6+6 = 18
            joint_pos, joint_vel, joint_effort,                         # 9+9+9   = 27
            gripper_open,                                               # 1
            stage,                                                      # 1
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

    def _print_manipulator_effort_diagnostics(self, api, agent: str) -> None:
        print_every = int(getattr(self.cfg, "manipulator_effort_print_every", 0))
        if print_every <= 0:
            return

        stage_values = self.current_stage.flatten().to(dtype=torch.long)
        monitor_stages = tuple(int(s) for s in getattr(self.cfg, "manipulator_effort_monitor_stages", (5, 6)))
        stage_mask = torch.zeros_like(stage_values, dtype=torch.bool)
        for stage_id in monitor_stages:
            stage_mask |= stage_values == stage_id

        arm_effort = api.joint_dict[agent]["joint_effort"][:, :7].detach()
        arm_effort_abs = torch.abs(arm_effort)
        manipulator_effort = torch.sum(arm_effort_abs, dim=-1)
        threshold = float(
            getattr(
                api,
                "_manipulator_high_effort_threshold",
                self.cfg.mpc_config.get("manipulator_high_effort_threshold", 8.0),
            )
        )

        counter = getattr(self, "_pen_cup_manipulator_effort_counter", None)
        if counter is None or counter.shape[0] != self.num_envs:
            counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.int64)
        above_threshold = manipulator_effort > threshold
        current_step = int(self.common_step_counter)
        if getattr(self, "_pen_cup_manipulator_effort_last_step", None) != current_step:
            counter = torch.where(stage_mask & above_threshold, counter + 1, torch.zeros_like(counter))
            counter = torch.clamp(counter, 0, 5)
            self._pen_cup_manipulator_effort_counter = counter
            self._pen_cup_manipulator_effort_last_step = current_step

        if not torch.any(stage_mask) or current_step % print_every != 0:
            return

        masked_effort = manipulator_effort[stage_mask]
        masked_abs = arm_effort_abs[stage_mask]
        masked_counter = counter[stage_mask]
        masked_trigger = masked_counter >= 5

        unique_stages, stage_counts = torch.unique(stage_values[stage_mask], return_counts=True)
        stage_summary = {
            int(k.item()): int(v.item()) for k, v in zip(unique_stages.detach().cpu(), stage_counts.detach().cpu())
        }

        def _fmt(values: torch.Tensor) -> list[float]:
            return [float(v) for v in values.detach().cpu().tolist()]

        print(
            f"[KinovaPenCupEnvLearned][ManipulatorEffort] step={int(self.common_step_counter)} "
            f"monitored_stages={monitor_stages} active_stage_counts={stage_summary} "
            f"threshold={threshold:.6f} min_consecutive_steps=5 "
            f"sum_abs_mean={float(masked_effort.mean().item()):.6f} "
            f"sum_abs_max={float(masked_effort.max().item()):.6f} "
            f"per_joint_abs_mean={_fmt(masked_abs.mean(dim=0))} "
            f"per_joint_abs_max={_fmt(masked_abs.max(dim=0).values)} "
            f"above_threshold_count={int(above_threshold[stage_mask].sum().item())}/{int(stage_mask.sum().item())} "
            f"counter_max={int(masked_counter.max().item())} "
            f"would_high_effort_count={int(masked_trigger.sum().item())}/{int(stage_mask.sum().item())}",
            flush=True,
        )

    def _pen_cup_geometry_terms(self):
        marker_pos = self.marker.data.root_state_w[:, :3]
        holder_pos = self.pen_holder.data.root_state_w[:, :3]
        marker_quat = self.marker.data.root_state_w[:, 3:7]
        holder_quat = self.pen_holder.data.root_state_w[:, 3:7]

        local_y = torch.zeros_like(marker_pos)
        local_y[:, 1] = 1.0
        local_z = torch.zeros_like(holder_pos)
        local_z[:, 2] = 1.0
        marker_y_world = math_utils.quat_apply(marker_quat, local_y)
        holder_z_world = math_utils.quat_apply(holder_quat, local_z)
        alignment = torch.abs((marker_y_world * holder_z_world).sum(dim=-1)).clamp(0.0, 1.0)

        diff = marker_pos - holder_pos
        xy_dist = torch.linalg.norm(diff[:, :2], dim=-1)
        z_dist = torch.abs(diff[:, 2])
        finished_geometry = torch.logical_and(
            torch.logical_and(
                xy_dist < self.cfg.success_xy_tolerance,
                z_dist < self.cfg.success_z_tolerance,
            ),
            alignment >= self.cfg.success_upright_alignment,
        )
        return xy_dist, z_dist, alignment, finished_geometry

    def external_rewards(self) -> torch.Tensor:
        self.update_data_in_controller()
        xy_dist, _, alignment, _ = self._pen_cup_geometry_terms()

        xy_reward = torch.exp(-20.0 * xy_dist)
        return 8.0 * xy_reward * (0.5 + 0.5 * alignment)

    def _external_get_dones(self):
        # Termination: marker is placed in the pen holder
        marker_pos = self.marker.data.root_state_w[:, :3]
        pen_holder_pos = self.pen_holder.data.root_state_w[:, :3]

        marker_pos_z = marker_pos[:, 2]
        pen_holder_pos_z = pen_holder_pos[:, 2]


        # Marker's y-axis should be within 45 deg of pen holder's z-axis (aligned with holder opening; both +z and -z count)
        marker_quat = self.marker.data.root_state_w[:, 3:7]
        holder_quat = self.pen_holder.data.root_state_w[:, 3:7]
        local_y = torch.zeros(marker_quat.shape[0], 3, device=marker_quat.device, dtype=marker_quat.dtype)
        local_y[:, 1] = 1.0
        local_z = torch.zeros(holder_quat.shape[0], 3, device=holder_quat.device, dtype=holder_quat.dtype)
        local_z[:, 2] = 1.0
        marker_y_world = math_utils.quat_apply(marker_quat, local_y)
        holder_z_world = math_utils.quat_apply(holder_quat, local_z)
        marker_upright = (
            torch.abs((marker_y_world * holder_z_world).sum(dim=-1))
            >= self.cfg.success_upright_alignment
        )

        # Placement succeeds when the marker is in place and the fingertip is at least 10 cm away.
        marker_pos_xy = marker_pos[:, :2]
        pen_holder_pos_xy = pen_holder_pos[:, :2]
        xy_dist = torch.norm(marker_pos_xy - pen_holder_pos_xy, dim=1)

        placed_geometry = (
            (xy_dist < self.cfg.success_xy_tolerance)
            & (torch.abs(marker_pos_z - pen_holder_pos_z) < self.cfg.success_z_tolerance)
            & marker_upright
        )
        fingertip_pos = self.program_runtime.controller_api_wrapper.lookup_robot_frame_position(
            "panda_fingertip"
        )
        gripper_marker_dist = torch.norm(fingertip_pos - marker_pos, dim=1)
        gripper_away = gripper_marker_dist >= self.cfg.success_gripper_marker_distance
        terminated = placed_geometry & gripper_away

        if self.cfg.sun_program_evaluator_stage_override or bool(
            getattr(self.cfg, "ignore_stage_gate_for_success", False)
        ):
            return terminated

        past_stage = self.current_stage.flatten() > int(self.program_runtime.compiled_program.stage_count()/2)
        return torch.logical_and(terminated, past_stage)
