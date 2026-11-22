# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
from kuafu.paths import OBJECT_ROOT

import os

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
from pxr import Usd, UsdGeom, UsdPhysics
from isaacsim.core.utils.stage import get_current_stage
from isaaclab.sim import SimulationCfg, schemas
from kuafu.isaac.learned.franka_table_env_learned import FrankaTableEnvLearnedCfg, FrankaTableEnvLearned
from kuafu.mpc.simulation.spatial_utils import quat_to_rotation_6d


DRAWER_COLLISION_CENTER_LOCAL = (0.0, 0.11, 0.1)
DRAWER_COLLISION_SIZE = (0.30, 0.23, 0.23)
STRAWBERRY_CENTER_ANGLE_RAD = -torch.pi / 4
STRAWBERRY_MIN_DRAWER_DISTANCE = 0.25
STRAWBERRY_SPAWN_RADIUS_MIN = 0.33
STRAWBERRY_SPAWN_RADIUS_MAX = 0.55
ISAAC_NUCLEUS_DIR = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac"
)


@configclass
class FR3FinRayTableDrawerStackEnvCfg(FrankaTableEnvLearnedCfg):
    """Learned env config for the FR3FinRay drawer + cube scene."""

    # env
    observation_space = 31  # default state dim; use set_obs_mode() for full_state
    obs_mode: str = "state"  # "state" (31) or "full_state" (65)
    obs_dims = {"state": 31, "full_state": 65}
    episode_length_s = 30


    # Override robot init_state only, keep rest from base
    def __post_init__(self):
        """Post-initialization to override robot initial state."""
        super().__post_init__()
        # Override only the initial state, keeping the rest of robot config from parent
        self.robot.init_state = ArticulationCfg.InitialStateCfg(
            joint_pos={
              "fr3_joint1": -1.0,
                "fr3_joint2": -0.42,
                "fr3_joint3": 0.0,
                "fr3_joint4": -2.25,
                "fr3_joint5": 0.00,
                "fr3_joint6": 2.00,
                "fr3_joint7": 0.80,
                "robotiq_85_left_knuckle_joint": 0.0,
                "robotiq_85_right_knuckle_joint": 0.0,
                "robotiq_85_left_finger_joint": 0.0,
                "robotiq_85_right_finger_joint": 0.0,
                "robotiq_85_left_finger_tip_joint": 0.0,
                "robotiq_85_right_finger_tip_joint": 0.0,
                "robotiq_85_left_finger_knuckle": 0.0,
                "robotiq_85_right_finger_knuckle": 0.0,
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
            pos=(0.3, 0.2,  0.02),
            rot=(1, 0, 0, 0),
        ),
        spawn=UsdFileCfg(
            usd_path=_strawberry_path,
            scale=(0.01, 0.01, 0.01),
            rigid_props=strawberry_properties,
            mass_props=sim_utils.MassPropertiesCfg(mass=0.02),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            semantic_tags=[("class", "strawberry")],
        ),
    )
    
    marker_cfg2 = FRAME_MARKER_CFG.copy()
    marker_cfg2.markers["frame"].scale = (0.1, 0.1, 0.1)
    marker_cfg2.prim_path = "/Visuals/top_drawer_handle_frame"
    top_drawer_handle_frame = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/fr3/base",
        debug_vis=False,
        visualizer_cfg=marker_cfg2,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Cabinet/upper_drawer_handle",
                name="top_drawer_handle_frame",
                offset=OffsetCfg(
                    pos=[0.0, 0.0, 0.0],
                ),
            )
        ],
    )

    marker_cfg6_1 = FRAME_MARKER_CFG.copy()
    marker_cfg6_1.markers["frame"].scale = (0.1, 0.1, 0.1)
    marker_cfg6_1.prim_path = "/Visuals/top_drawer_center_frame"
    top_drawer_center_frame = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/fr3/base",
        debug_vis=False,
        visualizer_cfg=marker_cfg6_1,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Cabinet/upper_drawer_handle",
                name="top_drawer_center_frame",
                offset=OffsetCfg(
                    pos=[0.0, 0.10, 0.0],
                ),
            )
        ],
    )

    marker_cfg6_2 = FRAME_MARKER_CFG.copy()
    marker_cfg6_2.markers["frame"].scale = (0.1, 0.1, 0.1)
    marker_cfg6_2.prim_path = "/Visuals/strawberry_center_frame"
    strawberry_center_frame = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/fr3/base",
        debug_vis=False,
        visualizer_cfg=marker_cfg6_2,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Strawberry",
                name="strawberry_center_frame",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.015),
                ),
            )
        ],
    )

    detectable_objects = {
        'cabinet': (
            'local frame +z is up direction, local frame -y is the forward direction; drawers open approximately 0.14 m; estimated size (approx.): (0.15, 0.25, 0.2).',
            '/World/envs/env_*/Cabinet',
        ),
        'top_drawer_center_frame': (
            'placement target frame at the center of the top drawer interior, estimated size (approx.): (0.15, 0.25, 0.05)',
            '/Visuals/top_drawer_center_frame',
        ),
        'top_drawer_handle_frame': (
            'grasp/contact frame for opening the top drawer; handle is a bar along local x axis; local frame -y is pulling out/open direction for approximately 0.14 m, local frame +y is pushing in/close direction',
            '/Visuals/top_drawer_handle_frame',
        ),
        'strawberry_center_frame': (
            'center frame of a cube-like symmetric strawberry object; treat it as a 0.04 m cube for grasping and placement',
            '/Visuals/strawberry_center_frame',
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
        "load_program": f"{_repo_dir}/resources/learned_programs/manipulation-fr3-finray-drawer-stack/2026-04-01_14-56-05",
        "initial_residual": f"{_repo_dir}/src/kuafu/skills/constraints/rewards.py",
        "initial_predicate": f"{_repo_dir}/src/kuafu/skills/constraints/predicates.py",
        "use_per_env_weight": False,
        "gripper_opened": [0.0, 0.0],
        "gripper_closed": [0.8, -0.8],
        "gripper_max_force": 1.0,
    }
    del _repo_dir

    task_description = 'open the top drawer fully and put the strawberry in top_drawer_center_frame'
class FR3FinRayTableDrawerStackEnv(FrankaTableEnvLearned):
    cfg: FR3FinRayTableDrawerStackEnvCfg

    def __init__(self, cfg: FR3FinRayTableDrawerStackEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

    def _replace_frame_collision_with_cube(self):
        stage = get_current_stage()
        cabinet_prims = sim_utils.find_matching_prims(self.cfg.cabinet.prim_path)
        collision_props = getattr(self.cfg.cabinet.spawn, "collision_props", None)
        replaced_count = 0
        print(
            "[FR3FinRayTableDrawerStackEnv] Cabinet prims found for collision replacement: "
            f"{len(cabinet_prims)}",
            flush=True,
        )

        for cabinet_prim in cabinet_prims:
            frame_path = f"{cabinet_prim.GetPath().pathString}/frame"
            frame_prim = stage.GetPrimAtPath(frame_path)
            if not frame_prim.IsValid():
                print(
                    "[FR3FinRayTableDrawerStackEnv] WARNING: Cabinet frame prim not found: "
                    f"{frame_path}",
                    flush=True,
                )
                continue

            frame_collision_path = f"{frame_path}/collisions"
            frame_collision_prim = stage.GetPrimAtPath(frame_collision_path)
            if frame_collision_prim.IsValid():
                frame_collision_prim.SetActive(False)
                print(
                    "[FR3FinRayTableDrawerStackEnv] Disabled original frame collision subtree: "
                    f"{frame_collision_path}",
                    flush=True,
                )
            else:
                print(
                    "[FR3FinRayTableDrawerStackEnv] WARNING: Frame collision subtree not found: "
                    f"{frame_collision_path}",
                    flush=True,
                )
                continue

            size = DRAWER_COLLISION_SIZE
            center = DRAWER_COLLISION_CENTER_LOCAL
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
                "[FR3FinRayTableDrawerStackEnv] Replaced frame collision with cube: "
                f"frame={frame_path}, collider={collider_path}, "
                f"center={center}, size={size}",
                flush=True,
            )

        if replaced_count == 0:
            print(
                "[FR3FinRayTableDrawerStackEnv] WARNING: No cabinet frame was updated; "
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

        self.strawberry_center_frame = FrameTransformer(self.cfg.strawberry_center_frame)
        self.scene.sensors["top_drawer_handle_frame"] = self.top_drawer_handle_frame
        self.scene.sensors["top_drawer_center_frame"] = self.top_drawer_center_frame
        self.scene.sensors["strawberry_center_frame"] = self.strawberry_center_frame

        stage = get_current_stage()
        strawberry_cfg = self.cfg.strawberry
        if strawberry_cfg.spawn is not None:
            strawberry_cfg.spawn.func(
                strawberry_cfg.prim_path,
                strawberry_cfg.spawn,
                translation=strawberry_cfg.init_state.pos,
                orientation=strawberry_cfg.init_state.rot,
            )

        if strawberry_cfg.spawn is not None:
            matching_prims = sim_utils.find_matching_prims(strawberry_cfg.prim_path)
            for prim in matching_prims:
                if hasattr(strawberry_cfg.spawn, 'rigid_props') and strawberry_cfg.spawn.rigid_props is not None:
                    schemas.define_rigid_body_properties(prim.GetPath().pathString, strawberry_cfg.spawn.rigid_props)

                if hasattr(strawberry_cfg.spawn, 'mass_props') and strawberry_cfg.spawn.mass_props is not None:
                    schemas.define_mass_properties(prim.GetPath().pathString, strawberry_cfg.spawn.mass_props)

                for child_prim in Usd.PrimRange(prim):
                    if child_prim.IsA(UsdGeom.Mesh):
                        mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(child_prim)
                        # Keep strawberry collision coarse and cheap; it only needs
                        # a stable dynamic approximation, not detailed contact geometry.
                        mesh_collision_api.GetApproximationAttr().Set(UsdPhysics.Tokens.convexHull)
                        if hasattr(strawberry_cfg.spawn, 'collision_props') and strawberry_cfg.spawn.collision_props is not None:
                            schemas.define_collision_properties(child_prim.GetPath().pathString, strawberry_cfg.spawn.collision_props, stage)

        self.strawberry = RigidObject(self.cfg.strawberry)
    
        self.scene.rigid_objects["strawberry"] = self.strawberry

        self.scene.clone_environments(copy_from_source=False)
        self._hide_static_surface_visual_prims()

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)


    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        env_ids_tensor = env_ids
        if len(env_ids_tensor) == 0:
            return

        if self.recorder_manager is not None and len(env_ids_tensor) > 0:
            self.recorder_manager.record_pre_reset(env_ids_tensor)

        n = len(env_ids_tensor)
        origins = self.scene.env_origins.index_select(0, env_ids_tensor)

        zeros = torch.zeros((n, self._cabinet.num_joints), device=self.device)
        self._cabinet.write_joint_state_to_sim(zeros, zeros, env_ids=env_ids_tensor)  # type: ignore[arg-type]

        # Match FR3FinRayDrawerEnvLearned cabinet reset exactly.
        is_left = torch.rand(n, device=self.device) < 0.5
        dist_left = torch.rand(n, device=self.device) * 0.1 + 0.45
        dist_right = torch.rand(n, device=self.device) * 0.1 + 0.55
        dist = torch.where(is_left, dist_left, dist_right)
        azimuth_left = -(torch.rand(n, device=self.device)) * 0.5 + 0.1
        azimuth_right = (torch.rand(n, device=self.device)) * 0.5 - torch.pi / 2 - 0.1
        azimuth = torch.where(is_left, azimuth_left, azimuth_right)
        cabinet_height = torch.rand(n, device=self.device) * 0.15 - 0.05

        facing_yaw = torch.where(is_left, azimuth - torch.pi / 2, azimuth + torch.pi / 2)
        raw_offset = -torch.rand(n, device=self.device) * (torch.pi / 3)
        offset_rad = torch.where(is_left, raw_offset, -raw_offset)
        cabinet_yaw = facing_yaw + offset_rad

        cabinet_xy = torch.stack([dist * torch.cos(azimuth), dist * torch.sin(azimuth)], dim=1)
        cabinet_pos = torch.cat(
            [cabinet_xy + origins[:, :2], origins[:, 2:3] + cabinet_height.unsqueeze(-1)],
            dim=1,
        )

        # The cabinet root is not centered on the physical drawer; use the collision box center.
        local_x = DRAWER_COLLISION_CENTER_LOCAL[0]
        local_y = DRAWER_COLLISION_CENTER_LOCAL[1]
        drawer_collision_offset_xy = torch.stack(
            [
                local_x * torch.cos(cabinet_yaw) - local_y * torch.sin(cabinet_yaw),
                local_x * torch.sin(cabinet_yaw) + local_y * torch.cos(cabinet_yaw),
            ],
            dim=1,
        )
        drawer_collision_xy = cabinet_pos[:, :2] + drawer_collision_offset_xy
        drawer_collision_xy_local = drawer_collision_xy - origins[:, :2]

        center_angle = torch.full((n,), STRAWBERRY_CENTER_ANGLE_RAD, device=self.device)
        drawer_angle = torch.atan2(drawer_collision_xy_local[:, 1], drawer_collision_xy_local[:, 0])
        strawberry_radius = (
            torch.rand(n, device=self.device)
            * (STRAWBERRY_SPAWN_RADIUS_MAX - STRAWBERRY_SPAWN_RADIUS_MIN)
            + STRAWBERRY_SPAWN_RADIUS_MIN
        )
        drawer_radius = torch.linalg.norm(drawer_collision_xy_local, dim=1)
        center_angle_delta = torch.atan2(
            torch.sin(center_angle - drawer_angle),
            torch.cos(center_angle - drawer_angle),
        )
        # Prefer the center spawn azimuth; if it is too close to the drawer,
        # rotate just enough around the env origin to satisfy the drawer clearance.
        min_clearance_angle = torch.acos(
            torch.clamp(
                (
                    strawberry_radius.square()
                    + drawer_radius.square()
                    - STRAWBERRY_MIN_DRAWER_DISTANCE**2
                )
                / (2.0 * strawberry_radius * torch.clamp(drawer_radius, min=1e-6)),
                min=-1.0,
                max=1.0,
            )
        )
        strawberry_side = torch.where(
            center_angle_delta >= 0.0,
            torch.ones_like(center_angle_delta),
            -torch.ones_like(center_angle_delta),
        )
        strawberry_angle = torch.where(
            torch.abs(center_angle_delta) >= min_clearance_angle,
            center_angle,
            drawer_angle + strawberry_side * min_clearance_angle,
        )
        strawberry_xy_local = torch.stack(
            [
                strawberry_radius * torch.cos(strawberry_angle),
                strawberry_radius * torch.sin(strawberry_angle),
            ],
            dim=1,
        )

        strawberry_height = float(self.cfg.strawberry.init_state.pos[2])
        strawberry_xy = origins[:, :2] + strawberry_xy_local
        strawberry_yaw = 2 * torch.pi * torch.rand(n, device=self.device)
        strawberry_quat = quat_from_euler_xyz(
            torch.zeros_like(strawberry_yaw),
            torch.zeros_like(strawberry_yaw),
            strawberry_yaw,
        )

        cabinet_root = self._cabinet.data.default_root_state.clone().index_select(0, env_ids_tensor)
        cabinet_root[:, :3] = cabinet_pos
        cabinet_root[:, 3:7] = quat_from_euler_xyz(
            torch.zeros_like(cabinet_yaw),
            torch.zeros_like(cabinet_yaw),
            cabinet_yaw,
        )
        self._cabinet.write_root_pose_to_sim(cabinet_root[:, :7], env_ids=env_ids_tensor)
        self._cabinet.write_root_velocity_to_sim(
            torch.zeros((n, 6), device=self.device), env_ids=env_ids_tensor
        )
        self._cabinet.reset(env_ids_tensor)

        strawberry_root = self.strawberry.data.default_root_state.clone().index_select(0, env_ids_tensor)
        strawberry_root[:, :3] = torch.cat(
            [strawberry_xy, origins[:, 2:3] + strawberry_height],
            dim=1,
        )
        strawberry_root[:, 3:7] = strawberry_quat
        self.strawberry.write_root_pose_to_sim(strawberry_root[:, :7], env_ids=env_ids_tensor)  # type: ignore[arg-type]
        self.strawberry.write_root_velocity_to_sim(
            torch.zeros((n, 6), device=self.device), env_ids=env_ids_tensor  # type: ignore[arg-type]
        )
        self.strawberry.reset(env_ids_tensor)  # type: ignore[arg-type]

        super()._reset_idx(env_ids_tensor)

    def _get_observations(self) -> dict:
        self.update_data_in_controller()

        api = self.program_runtime.controller_api_wrapper
        agent = api.reference_agent
        if agent is None:
            raise RuntimeError("controller_api_wrapper.reference_agent is not set")

        handle_pos       = api.get_object_xyz_position("top_drawer_handle_frame")           # (N, 3)
        strawberry_pos   = api.get_object_xyz_position("strawberry_center_frame")                        # (N, 3)
        fingertip_pos    = api.lookup_robot_frame_position("panda_fingertip")               # (N, 3)
        link4_pos        = api.lookup_robot_frame_position("panda_link4")                   # (N, 3)
        handle_rot6d     = quat_to_rotation_6d(api.get_object_wxyz_quaternion("top_drawer_handle_frame"))  # (N, 6)
        strawberry_rot6d = quat_to_rotation_6d(api.get_object_wxyz_quaternion("strawberry_center_frame"))              # (N, 6)
        fingertip_rot6d  = quat_to_rotation_6d(api.lookup_robot_frame_rotation("panda_fingertip"))        # (N, 6)
        link4_rot6d      = quat_to_rotation_6d(api.lookup_robot_frame_rotation("panda_link4"))            # (N, 6)
        joint_pos        = api.joint_dict[agent]["joint_position"]                          # (N, 9)
        joint_vel        = api.joint_dict[agent]["joint_velocity"]                          # (N, 9)
        joint_effort     = self._get_normalized_joint_effort(api, agent)                    # (N, 9)
        gripper_contact  = self._get_normalized_gripper_contact(api)                         # (N, 2)
        joint_effort     = torch.cat([joint_effort[:, :7], gripper_contact], dim=-1)        # (N, 9)
        gripper_open     = api.gripper_openness().unsqueeze(-1)                              # (N, 1)
        stage            = self.current_stage                                               # (N, 1)

        def _f(*tensors):
            return torch.cat([t.to(device=self.device, dtype=torch.float32) for t in tensors], dim=-1)

        # Policy obs (31): task-relevant state only, no joint pos/vel/effort, no link4
        policy_obs = _f(
            handle_pos, strawberry_pos, fingertip_pos,          # 3+3+3 = 9
            handle_rot6d, strawberry_rot6d, fingertip_rot6d,    # 6+6+6 = 18
            gripper_open, gripper_contact,                       # 1+2   = 3
            stage,                                               # 1
        )  # total: 31

        # Full state obs (65): everything needed to recreate the env
        full_state_obs = _f(
            handle_pos, strawberry_pos, fingertip_pos, link4_pos,          # 3+3+3+3 = 12
            handle_rot6d, strawberry_rot6d, fingertip_rot6d, link4_rot6d,  # 6+6+6+6 = 24
            joint_pos, joint_vel, joint_effort,                             # 9+9+9   = 27
            gripper_open,                                                    # 1
            stage,                                                           # 1
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
        strawberry_pos = self.strawberry.data.root_pos_w
        drawer_center_pos = self.top_drawer_center_frame.data.target_pos_w.squeeze(1)
        drawer_open = torch.clamp(self._cabinet.data.joint_pos[:, 0] / 0.12, 0.0, 1.0)

        diff = strawberry_pos - drawer_center_pos
        xy_dist = torch.linalg.norm(diff[:, :2], dim=-1)
        z_dist = torch.abs(diff[:, 2])
        placement = torch.exp(-20.0 * xy_dist) * torch.exp(-20.0 * z_dist)

        return 2.0 * drawer_open + 4.0 * placement

    def _external_get_dones(self) -> torch.Tensor:
        strawberry_pos = self.strawberry.data.root_pos_w  # (num_envs, 3)
        drawer_center_pos = self.top_drawer_center_frame.data.target_pos_w.squeeze(1)  # (num_envs, 3)
        diff_xy = strawberry_pos[:, :2] - drawer_center_pos[:, :2]
        diff_z = strawberry_pos[:, 2] - drawer_center_pos[:, 2]
        xy_close = torch.linalg.norm(diff_xy, dim=1) <= 0.05
        z_close = torch.logical_and(diff_z <= 0.04, diff_z >= -0.05)
        gripper_openness = self.program_runtime.controller_api_wrapper.gripper_openness().flatten()
        grippers_open = gripper_openness > 0.5
        task_done = torch.logical_and(torch.logical_and(xy_close, z_close), grippers_open)
        if self.cfg.sun_program_evaluator_stage_override or bool(
            getattr(self.cfg, "ignore_stage_gate_for_success", False)
        ):
            return task_done

        final_stage = self.current_stage.flatten() >= (self.program_runtime.compiled_program.stage_count() - 1)
        return torch.logical_and(task_done, final_stage)
