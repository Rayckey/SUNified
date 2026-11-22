# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
from kuafu.paths import OBJECT_ROOT

import os
import time

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab.sensors import CameraCfg, TiledCamera, TiledCameraCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer import FrameTransformer
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema
from isaacsim.core.utils.stage import get_current_stage
from isaaclab.sim import SimulationCfg, schemas
from kuafu.isaac.tasks.env_learned import BaseLearnedEnv, BaseLearnedEnvCfg
from kuafu.isaac.tasks.env_utils import random_pose_radial
from kuafu.mpc.simulation.spatial_utils import quat_to_rotation_6d

ISAAC_NUCLEUS_DIR = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac"
)


@configclass
class DrawerStackEnvCfg(BaseLearnedEnvCfg):
    """Learned env config for the FR3FinRay drawer + cube scene."""

    # Override robot init_state only, keep rest from base
    def __post_init__(self):
        """Post-initialization to override robot initial state."""
        super().__post_init__()
        # Override only the initial state, keeping the rest of robot config from parent
        self.robot.init_state = ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 0.0,
                "panda_joint2": -0.32,
                "panda_joint3": 0.0,
                "panda_joint4": -2.25,
                "panda_joint5": 0.00,
                "panda_joint6": 2.00,
                "panda_joint7": 0.80,
                "panda_finger_joint.*": 0.04,
            },
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        )

    # env
    observation_space = 31  # default state dim; use set_obs_mode() for full_state
    obs_mode: str = "state"  # "state" (31) or "full_state" (65)
    obs_dims = {"state": 31, "full_state": 65}
    enable_cameras: bool = False

    _cabinet_path = str(OBJECT_ROOT / 'two_layer_drawer_iphone_with_link/two_layer_drawer_iphone_with_link.usd')

    cabinet = ArticulationCfg(
        prim_path="/World/envs/env_.*/Cabinet",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_cabinet_path,
            activate_contact_sensors=False,
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.001,
                rest_offset=0.0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.35, -0.4, 0.1),
            rot=(0.70710678, 0, 0, 0.70710678),
            joint_pos={"upper_drawer_joint": 0.0},
        ),
        actuators={
            "drawers": ImplicitActuatorCfg(
                joint_names_expr=["upper_drawer_joint"],
                effort_limit_sim=87.0,
                stiffness=0.0,
                damping=0.2,
            )
        },
    )

    # Rigid body properties and asset path for the strawberry object
    strawberry_properties = sim_utils.RigidBodyPropertiesCfg(
        solver_position_iteration_count=32,
        solver_velocity_iteration_count=2,
        max_angular_velocity=1000.0,
        max_linear_velocity=1000.0,
        max_depenetration_velocity=100.0,
        disable_gravity=False,
    )

    _strawberry_path = str(OBJECT_ROOT / 'Berries/stawberry.usd')

    strawberry = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Strawberry",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.3, 0.2,  0.01),
            rot=(1, 0, 0, 0),
        ),
        spawn=UsdFileCfg(
            usd_path=_strawberry_path,
            scale=(0.01, 0.01, 0.01),
            rigid_props=strawberry_properties,
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            semantic_tags=[("class", "strawberry")],
        ),
    )
    marker_cfg2 = FRAME_MARKER_CFG.copy()  # type: ignore[attr-defined]
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
                offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
            )
        ],
    )

    marker_cfg6 = FRAME_MARKER_CFG.copy()  # type: ignore[attr-defined]
    marker_cfg6.markers["frame"].scale = (0.1, 0.1, 0.1)
    marker_cfg6.prim_path = "/Visuals/top_drawer_center_frame"
    top_drawer_center_frame = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/panda_link0",
        debug_vis=False,
        visualizer_cfg=marker_cfg6,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Cabinet/upper_drawer_handle",
                name="top_drawer_center_frame",
                offset=OffsetCfg(pos=(0.0, 0.13, 0.0)),
            )
        ],
    )

    detectable_objects = {
        "cabinet": (
            "local frame +z is up direction, local frame -y is the forward direction; estimated size (approx.): (0.15, 0.25, 0.2).",
            "/World/envs/env_*/Cabinet",
        ),
        "top_drawer_center_frame": (
            "center frame of top drawer's interior, estimated size (approx.): (0.15, 0.25, 0.05)",
            "/Visuals/top_drawer_center_frame",
        ),
        "top_drawer_handle_frame": (
            "frame of handle of the top drawer, handle is a bar along x axis, local frame -y is pulling out (open) direction, local frame +y is pushing in (close) direction",
            "/Visuals/top_drawer_handle_frame",
        ),
        "strawberry": (
            "center frame of a small strawberry, size: (0.03, 0.03, 0.03)",
            "/World/envs/env_*/Strawberry",
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

    _repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))
    mpc_config = {
        "controller_name": "FrankaIKController",
        "device": "cuda",
        "ee_frame": "panda_fingertip",
        "use_per_env_weight": False,
        "load_program": f"{_repo_dir}/resources/learned_programs/manipulation-fr3-finray-drawer/2026-01-06_21-21-38",
        "initial_residual": f"{_repo_dir}/src/kuafu/skills/constraints/rewards.py",
        "initial_predicate": f"{_repo_dir}/src/kuafu/skills/constraints/predicates.py",
    }
    del _repo_dir

    task_description = "open the drawer until it stops moving and put the strawberry in there"


class DrawerStackEnv(BaseLearnedEnv):
    cfg: DrawerStackEnvCfg

    def _setup_scene(self):
        super()._setup_scene()

        self._cabinet = Articulation(self.cfg.cabinet)

        stage = get_current_stage()
        strawberry_cfg = self.cfg.strawberry
        # Spawn the USD file if spawn config exists
        if strawberry_cfg.spawn is not None:
            strawberry_cfg.spawn.func(
                strawberry_cfg.prim_path,
                strawberry_cfg.spawn,
                translation=strawberry_cfg.init_state.pos,
                orientation=strawberry_cfg.init_state.rot,
            )
        # Apply RigidBodyAPI and collision approximation to all matching prims
        if strawberry_cfg.spawn is not None:
            matching_prims = sim_utils.find_matching_prims(strawberry_cfg.prim_path)
            for prim in matching_prims:
                # Always apply RigidBodyAPI (it's safe to apply even if already present)
                if hasattr(strawberry_cfg.spawn, 'rigid_props') and strawberry_cfg.spawn.rigid_props is not None:
                    schemas.define_rigid_body_properties(prim.GetPath().pathString, strawberry_cfg.spawn.rigid_props)

                # Apply mass properties if specified
                if hasattr(strawberry_cfg.spawn, 'mass_props') and strawberry_cfg.spawn.mass_props is not None:
                    schemas.define_mass_properties(prim.GetPath().pathString, strawberry_cfg.spawn.mass_props)

                # Add collision approximation to all mesh prims inside the mug (recursively)
                for child_prim in Usd.PrimRange(prim):
                    if child_prim.IsA(UsdGeom.Mesh):
                        # Apply collision approximation to each mesh
                        mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(child_prim)
                        mesh_collision_api.GetApproximationAttr().Set(UsdPhysics.Tokens.convexDecomposition)

                        # Fine convex decomposition params (PhysX)
                        cd_api = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(child_prim)
                        cd_api.CreateVoxelResolutionAttr().Set(5_000_000)
                        cd_api.CreateMaxConvexHullsAttr().Set(128)
                        cd_api.CreateErrorPercentageAttr().Set(1.0)
                        cd_api.CreateShrinkWrapAttr().Set(True)
                        cd_api.CreateHullVertexLimitAttr().Set(64)
                        cd_api.CreateMinThicknessAttr().Set(0.0005)
                        # Apply collision properties
                        if hasattr(strawberry_cfg.spawn, 'collision_props') and strawberry_cfg.spawn.collision_props is not None:
                            schemas.define_collision_properties(child_prim.GetPath().pathString, strawberry_cfg.spawn.collision_props, stage)

        # Now create RigidObjects (they will find the RigidBodyAPI we just applied)
        self.strawberry = RigidObject(self.cfg.strawberry)

        self.scene.articulations["cabinet"] = self._cabinet
        self.scene.rigid_objects["strawberry"] = self.strawberry

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        self.top_drawer_handle_frame = FrameTransformer(self.cfg.top_drawer_handle_frame)
        self.top_drawer_center_frame = FrameTransformer(self.cfg.top_drawer_center_frame)

        self.scene.sensors["top_drawer_handle_frame"] = self.top_drawer_handle_frame
        self.scene.sensors["top_drawer_center_frame"] = self.top_drawer_center_frame

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        assert env_ids is not None

        if self.recorder_manager is not None and len(env_ids) > 0:
            self.recorder_manager.record_pre_reset(env_ids)
        num_resets = len(env_ids)

        zeros = torch.zeros((len(env_ids), self._cabinet.num_joints), device=self.device)
        self._cabinet.write_joint_state_to_sim(zeros, zeros, env_ids=env_ids)  

        num_resets = len(env_ids)

        # Randomize cabinet pose in the front-right quadrant, slightly farther from the robot.
        dist = torch.rand(num_resets, device=self.device) * 0.1 + 0.5
        azimuth = -torch.rand(num_resets, device=self.device) * (torch.pi / 2)
        height = torch.rand(num_resets, device=self.device) * 0.1 + 0.1     # 0.15–0.3 m

        # Cabinet yaw: face the robot (azimuth + pi) with a smaller tilt to either side.
        facing_yaw = azimuth + torch.pi
        offset_sign = torch.randint(0, 2, (num_resets,), device=self.device).float() * 2.0 - 1.0
        offset_rad = torch.rand(num_resets, device=self.device) * (torch.pi / 4)
        yaw = facing_yaw + offset_sign * offset_rad


        root = self._cabinet.data.default_root_state.clone().index_select(0, env_ids)
        root[:, 0] = dist * torch.cos(azimuth)
        root[:, 1] = dist * torch.sin(azimuth)
        root[:, 2] = height
        root[:, :3] += self.scene.env_origins.index_select(0, env_ids)

        z = torch.zeros_like(yaw)
        root[:, 3:7] = quat_from_euler_xyz(z, z, yaw)

        self._cabinet.write_root_pose_to_sim(root[:, :7], env_ids=env_ids)
        self._cabinet.write_root_velocity_to_sim(
            torch.zeros((num_resets, 6), device=self.device), env_ids=env_ids
        )
        self._cabinet.reset(env_ids)

        env_ids_tensor = env_ids
        origins = self.scene.env_origins.index_select(0, env_ids_tensor)
        strawberry_pos, strawberry_quat = random_pose_radial(
            0.01,
            dist_min=0.35,
            dist_max=0.45,
            angle_min_rad=-torch.pi / 2,
            angle_max_rad=0.0,
            n=num_resets,
            device=self.device,
            origins=origins,
        )

        root_m = self.strawberry.data.default_root_state.clone().index_select(
            0, env_ids
        )
        root_m[:, :3] = strawberry_pos
        root_m[:, 3:7] = strawberry_quat
        self.strawberry.write_root_pose_to_sim(root_m[:, :7], env_ids=env_ids)  # type: ignore[arg-type]
        self.strawberry.write_root_velocity_to_sim(
            torch.zeros((len(env_ids), 6), device=self.device), env_ids=env_ids  # type: ignore[arg-type]
        )
        self.strawberry.reset(env_ids)  # type: ignore[arg-type]

        super()._reset_idx(env_ids)

    def _get_observations(self) -> dict:
        self.update_data_in_controller()
        api = self.program_runtime.controller_api_wrapper
        agent = api.reference_agent
        if agent is None:
            raise RuntimeError("controller_api_wrapper.reference_agent is not set")

        handle_pos = api.get_object_xyz_position("top_drawer_handle_frame")  # (N, 3)
        strawberry_pos = api.get_object_xyz_position("strawberry")  # (N, 3)
        fingertip_pos = api.lookup_robot_frame_position("panda_fingertip")  # (N, 3)
        link4_pos = api.lookup_robot_frame_position("panda_link4")  # (N, 3)

        handle_rot6d = quat_to_rotation_6d(
            api.get_object_wxyz_quaternion("top_drawer_handle_frame")
        )  # (N, 6)
        strawberry_rot6d = quat_to_rotation_6d(api.get_object_wxyz_quaternion("strawberry"))  # (N, 6)
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
            handle_pos,
            strawberry_pos,
            fingertip_pos,  # 9
            handle_rot6d,
            strawberry_rot6d,
            fingertip_rot6d,  # 18
            gripper_open,
            gripper_effort,  # 3
            stage,  # 1
        )  # total: 31

        full_state_obs = _f(
            handle_pos,
            strawberry_pos,
            fingertip_pos,
            link4_pos,  # 12
            handle_rot6d,
            strawberry_rot6d,
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

    def _external_get_dones(self) -> torch.Tensor:
        strawberry_pos = self.strawberry.data.root_pos_w  # (num_envs, 3)
        drawer_center_pos = self.top_drawer_center_frame.data.target_pos_w.squeeze(1)  # (num_envs, 3)
        diff_xy = strawberry_pos[:, :2] - drawer_center_pos[:, :2]
        diff_z = strawberry_pos[:, 2] - drawer_center_pos[:, 2]
        xy_close = torch.linalg.norm(diff_xy, dim=1) <= 0.05
        z_close = torch.abs(diff_z) <= 0.02
        gripper_openness = self.program_runtime.controller_api_wrapper.gripper_openness().flatten()
        grippers_open = gripper_openness > 0.9
        return torch.logical_and(torch.logical_and(xy_close, z_close), grippers_open)
