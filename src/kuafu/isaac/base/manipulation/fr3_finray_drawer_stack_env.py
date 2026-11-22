# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
from kuafu.paths import OBJECT_ROOT

import os
from isaaclab.sensors.frame_transformer.frame_transformer import FrameTransformer
import torch

from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.utils.torch.transformations import tf_combine, tf_inverse, tf_vector
from pxr import Usd, UsdGeom, UsdPhysics
import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObjectCfg, RigidObject
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, schemas
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform, quat_from_euler_xyz
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg

from .panda_manipulation_env import PandaManipulationEnv, PandaManipulationEnvCfg


@configclass
class DrawerStackEnvCfg(PandaManipulationEnvCfg):
    """Configuration for the drawer stack environment.
    
    Inherits from PandaManipulationEnvCfg and adds drawer-specific configurations.
    """
    # env
    observation_space = 49
    enable_cameras: bool = False

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=3.0, replicate_physics=True
    )

    # cabinet
    cabinet = ArticulationCfg(
        prim_path="/World/envs/env_.*/Cabinet",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(OBJECT_ROOT / 'two_layer_drawer_iphone_with_link/two_layer_drawer_iphone_with_link.usd'),
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.35, -0.4, 0.2),
            rot=(0.70710678, 0, 0, 0.70710678),
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
            pos=(0.4, 0.1,  0.01),
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
        prim_path="/World/envs/env_.*/Robot/panda_link0",
        debug_vis=False,
        visualizer_cfg=marker_cfg2,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Cabinet/upper_drawer_handle",
                name="top_drawer_handle_frame",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.01),
                    rot=(1, 0, 0, 0),
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
                    pos=(0.0, 0.1, 0.02),
                ),
            )
        ],
    )

    marker_cfg6_2 = FRAME_MARKER_CFG.copy()
    marker_cfg6_2.markers["frame"].scale = (0.1, 0.1, 0.1)
    marker_cfg6_2.prim_path = "/Visuals/strawberry_center_frame"
    strawberry_center_frame = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/panda_link0",
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


    # reward scales (inherited from parent, but keeping for clarity)
    # dist_reward_scale, rot_reward_scale, open_reward_scale, action_penalty_scale, finger_reward_scale
    # are inherited from PandaManipulationEnvCfg


class DrawerStackEnv(PandaManipulationEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: DrawerStackEnvCfg

    def __init__(self, cfg: DrawerStackEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Initialize drawer-specific variables
        # Robot initialization is handled by parent class _initialize_robot_common

        drawer_local_grasp_pose = torch.tensor(
            [0.3, 0.01, 0.0, 1.0, 0.0, 0.0, 0.0], device=self.device
        )
        self.drawer_local_grasp_pos = drawer_local_grasp_pose[0:3].repeat(
            (self.num_envs, 1)
        )
        self.drawer_local_grasp_rot = drawer_local_grasp_pose[3:7].repeat(
            (self.num_envs, 1)
        )

        self.drawer_inward_axis = torch.tensor(
            [-1, 0, 0], device=self.device, dtype=torch.float32
        ).repeat((self.num_envs, 1))
        self.drawer_up_axis = torch.tensor(
            [0, 0, 1], device=self.device, dtype=torch.float32
        ).repeat((self.num_envs, 1))

        self.drawer_link_idx = self._cabinet.find_bodies("upper_drawer")[0][0]


    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
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

                # Keep strawberry contact simple like the learned drawer-stack env: coarse convex hull, not detailed berry geometry.
                collision_mesh_count = 0
                for child_prim in Usd.PrimRange(prim):
                    if child_prim.IsA(UsdGeom.Mesh):
                        mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(child_prim)
                        mesh_collision_api.GetApproximationAttr().Set(UsdPhysics.Tokens.convexHull)
                        if hasattr(strawberry_cfg.spawn, 'collision_props') and strawberry_cfg.spawn.collision_props is not None:
                            schemas.define_collision_properties(child_prim.GetPath().pathString, strawberry_cfg.spawn.collision_props, stage)
                        collision_mesh_count += 1
                print(
                    "[DrawerStackEnv] Strawberry collision uses coarse convexHull approximation "
                    f"for {collision_mesh_count} mesh prim(s), treating it as a cube-like object.",
                    flush=True,
                )

        # Now create RigidObjects (they will find the RigidBodyAPI we just applied)
        self.strawberry = RigidObject(self.cfg.strawberry)

        self.scene.articulations["robot"] = self._robot
        self.scene.articulations["cabinet"] = self._cabinet
        self.scene.rigid_objects["strawberry"] = self.strawberry
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        self.ee_frame = FrameTransformer(self.cfg.ee_frame)
        self.l4_frame = FrameTransformer(self.cfg.l4_frame)
        self.strawberry_center_frame = FrameTransformer(self.cfg.strawberry_center_frame)

        self.top_drawer_handle_frame = FrameTransformer(
            self.cfg.top_drawer_handle_frame
        )

        self.top_drawer_center_frame = FrameTransformer(
            self.cfg.top_drawer_center_frame
        )

        self.scene.sensors["panda_fingertip"] = self.ee_frame
        self.scene.sensors["panda_link4"] = self.l4_frame
        self.scene.sensors["top_drawer_handle_frame"] = self.top_drawer_handle_frame
        self.scene.sensors["top_drawer_center_frame"] = self.top_drawer_center_frame
        self.scene.sensors["strawberry_center_frame"] = self.strawberry_center_frame

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # pre-physics step calls
    # _pre_physics_step and _apply_action are inherited from parent class

    # post-physics step calls

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        if getattr(self.cfg, "disable_terminations", False):
            done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            return done, done

        terminated = self._cabinet.data.joint_pos[:, 0] > 1.5
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        robot_left_finger_pos = self._robot.data.body_pos_w[
            :, self.left_finger_link_idx
        ]
        robot_right_finger_pos = self._robot.data.body_pos_w[
            :, self.right_finger_link_idx
        ]

        return self._compute_rewards(
            self.actions,
            self._cabinet.data.joint_pos
        )

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        if env_ids is None or len(env_ids) == 0:
            return

        # robot state - use parent's helper method
        self._reset_robot(env_ids)

        # cabinet state
        zeros = torch.zeros(
            (len(env_ids), self._cabinet.num_joints), device=self.device
        )
        self._cabinet.write_joint_state_to_sim(zeros, zeros, env_ids=env_ids)

        # Randomize strawberry position on the ground
        if getattr(self.cfg, "disable_reset_randomization", False):
            strawberry_offset = torch.zeros((len(env_ids), 1, 3), device=self.device)
        else:
            strawberry_offset = torch.rand(
                (len(env_ids), 1, 3), device=self.device
            ) * torch.tensor(
                [[[0.2, 0.2, 0.0]]], device=self.device
            )

        root = self.strawberry.data.default_root_state.clone()
        root = root.index_select(0, env_ids)
        root[:, :3] += (
            self.scene.env_origins.index_select(0, env_ids) + strawberry_offset[:, 0, :]
        )
        eulers = torch.zeros(
            (len(env_ids), 3), device=self.device
        )  # or your real angles
        roll = eulers[:, 0]
        pitch = eulers[:, 1]
        yaw = eulers[:, 2]

        quat = quat_from_euler_xyz(roll, pitch, yaw)
        root[:, 3:7] = quat  # quaternions are WXYZ in Isaac Lab

        # write pose and velocities to the sim, then reset buffers
        self.strawberry.write_root_pose_to_sim(root[:, :7], env_ids=env_ids)
        self.strawberry.write_root_velocity_to_sim(
            torch.zeros((len(env_ids), 6), device=self.device), env_ids=env_ids
        )
        self.strawberry.reset(env_ids)


    def _get_observations(self) -> dict:
        dof_pos_scaled = (
            2.0
            * (self._robot.data.joint_pos - self.robot_dof_lower_limits)
            / (self.robot_dof_upper_limits - self.robot_dof_lower_limits)
            - 1.0
        )

        obs = torch.cat(
            (
                dof_pos_scaled,
                self._robot.data.joint_vel,
                # to_target,
                self._cabinet.data.joint_pos[:, 0].unsqueeze(-1),
                self._cabinet.data.joint_vel[:, 0].unsqueeze(-1),
            ),
            dim=-1,
        )
        return {"policy": torch.clamp(obs, -5.0, 5.0)}


    def _compute_rewards(
        self,
        actions,
        cabinet_dof_pos,
    ):
        # how far the cabinet has been opened out
        open_reward = cabinet_dof_pos[:, 0] 

        return open_reward
