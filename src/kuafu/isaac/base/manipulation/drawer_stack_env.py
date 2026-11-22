# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.sensors.frame_transformer.frame_transformer import FrameTransformer
import torch
from collections.abc import Sequence
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.utils.torch.transformations import tf_combine, tf_inverse, tf_vector
from pxr import UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.sim import UsdFileCfg
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObjectCfg, RigidObject
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from kuafu.isaac.base.manipulation.panda_manipulation_env import PandaManipulationEnv, PandaManipulationEnvCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import sample_uniform, quat_from_euler_xyz
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg

ISAAC_NUCLEUS_DIR = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac"

@configclass
class DrawerStackEnvCfg(PandaManipulationEnvCfg):
    # env
    episode_length_s: float = 800.0
    decimation = 4
    action_space = 9
    observation_space = 55
    state_space = 0
    enable_camera: bool = False
    disable_terminations: bool = False
    use_debug_action_mode: bool = False

    # simulation
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

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4, env_spacing=6.0, replicate_physics=True
    )

    # robot (Franka)
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
                "panda_joint2": -0.32,
                "panda_joint3": 0.0,
                "panda_joint4": -1.25,
                "panda_joint5": 0.00,
                "panda_joint6": 2.00,
                "panda_joint7": 0.80,
                "panda_finger_joint.*": 0.04,


            },
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                effort_limit=87.0,
                velocity_limit=2.175,
                stiffness=80.0,
                damping=4.0,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit=12.0,
                velocity_limit=2.61,
                stiffness=80.0,
                damping=4.0,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint.*"],
                effort_limit=200.0,
                velocity_limit=0.2,
                stiffness=2e3,
                damping=1e2,
            ),
        },
    )

    robot.spawn.rigid_props.disable_gravity = True
    robot.actuators["panda_shoulder"].stiffness = 400.0
    robot.actuators["panda_shoulder"].damping = 80.0
    robot.actuators["panda_forearm"].stiffness = 400.0
    robot.actuators["panda_forearm"].damping = 80.0

    # cabinet
    cabinet = ArticulationCfg(
        prim_path="/World/envs/env_.*/Cabinet",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd",
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(1.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 0.1),
            joint_pos={
                "door_left_joint": 0.0,
                "door_right_joint": 0.0,
                "drawer_bottom_joint": 0.0,
                "drawer_top_joint": 0.0,
            },
        ),
        actuators={
            "drawers": ImplicitActuatorCfg(
                joint_names_expr=["drawer_top_joint", "drawer_bottom_joint"],
                effort_limit=87.0,
                stiffness=0.5,
                damping=0.2,
            ),
            "doors": ImplicitActuatorCfg(
                joint_names_expr=["door_left_joint", "door_right_joint"],
                effort_limit=87.0,
                stiffness=2.0,
                damping=1.0,
            ),
        },
    )

    # cubes
    cube_properties = sim_utils.RigidBodyPropertiesCfg(
        solver_position_iteration_count=16,
        solver_velocity_iteration_count=1,
        max_angular_velocity=1000.0,
        max_linear_velocity=1000.0,
        max_depenetration_velocity=5.0,
        disable_gravity=False,
    )

    cube_1 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Cube_1",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.4, 0.0, 0.0203), rot=(1.0, 0.0, 0.0, 0.0)
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
            pos=(0.55, 0.05, 0.0203), rot=(1.0, 0.0, 0.0, 0.0)
        ),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/red_block.usd",
            scale=(1.0, 1.0, 1.0),
            rigid_props=cube_properties,
            semantic_tags=[("class", "cube_2")],
        ),
    )

    marker_cfg0 = FRAME_MARKER_CFG.copy()
    marker_cfg0.markers["frame"].scale = (0.1,0.1,0.1)
    marker_cfg0.prim_path = "/Visuals/panda_hand_frame"
    ee_frame = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/panda_link0",
        debug_vis=False,
        visualizer_cfg=marker_cfg0,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot/panda_hand",
                name="end_effector",
                offset=OffsetCfg(
                    pos=[0.0, 0.0, 0.1034],
                ),
            )
        ],
    )


    marker_cfg1 = FRAME_MARKER_CFG.copy()
    marker_cfg1.markers["frame"].scale = (0.1, 0.1, 0.1)
    marker_cfg1.prim_path = "/Visuals/drawer_handle_bottom_frame"
    bottom_cabinet_handle_frame = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/panda_link0",
        debug_vis=False,
        visualizer_cfg=marker_cfg1,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Cabinet/drawer_bottom",
                name="drawer_bottom",
                offset=OffsetCfg(
                    pos=[0.28, 0.0, 0.0],
                ),
            )
        ],
    )

    marker_cfg2 = FRAME_MARKER_CFG.copy()
    marker_cfg2.markers["frame"].scale = (0.1, 0.1, 0.1)
    marker_cfg2.prim_path = ("/Visuals/drawer_handle_top_frame")
    top_cabinet_handle_frame = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/panda_link0",
        debug_vis=False,
        visualizer_cfg=marker_cfg2,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Cabinet/drawer_top",
                name="drawer_top",
                offset=OffsetCfg(
                    pos=[0.28, 0.0, 0.0],
                ),
            )
        ],
    )

    marker_cfg5 = FRAME_MARKER_CFG.copy()
    marker_cfg5.markers["frame"].scale = (0.1, 0.1, 0.1)
    marker_cfg5.prim_path = ("/Visuals/drawer_bottom_frame")
    bottom_drawer_frame = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/panda_link0",
        debug_vis=False,
        visualizer_cfg=marker_cfg5,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Cabinet/drawer_bottom",
                name="drawer_bottom_handle_frame",
                offset=OffsetCfg(
                    pos=[0.3, 0.0, 0.0],
                ),
            )
        ],
    )

    marker_cfg6 = FRAME_MARKER_CFG.copy()
    marker_cfg6.markers["frame"].scale = (0.1, 0.1, 0.1)
    marker_cfg6.prim_path = ("/Visuals/drawer_top_frame")
    top_drawer_frame = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/panda_link0",
        debug_vis=False,
        visualizer_cfg=marker_cfg6,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Cabinet/drawer_top",
                name="drawer_top_handle_frame",
                offset=OffsetCfg(
                    pos=[0.3, 0.0, 0.0],
                ),
            )
        ],
    )

    marker_cfg7 = FRAME_MARKER_CFG.copy()
    marker_cfg7.markers["frame"].scale = (0.1, 0.1, 0.1)
    marker_cfg7.prim_path = ("/Visuals/panda_link4_frame")
    l4_frame = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/panda_link0",
        debug_vis=False,
        visualizer_cfg=marker_cfg7,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot/panda_link4",
                name="panda_link4",
            )
        ],
    )

    # terrain
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


class DrawerStackEnv(PandaManipulationEnv):
    cfg: DrawerStackEnvCfg

    def __init__(self, cfg: DrawerStackEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # drawer link indices (resolved after _setup_scene)
        # resolve drawer body indices
        self.drawer_top_idx = self.cabinet.find_bodies("drawer_top")[0][0]
        self.drawer_bottom_idx = self.cabinet.find_bodies("drawer_bottom")[0][0]

        # local offsets into drawers (inside the drawer volume)
        self.drawer_local_offset = torch.tensor([-0.20, 0.0, 0.0], device=self.device)

    def _setup_scene(self):
        # spawn assets
        self.robot = Articulation(self.cfg.robot)
        self.cabinet = Articulation(self.cfg.cabinet)
        self.cube_1 = RigidObject(self.cfg.cube_1)
        self.cube_2 = RigidObject(self.cfg.cube_2)

        self.scene.articulations["robot"] = self.robot
        self.scene.articulations["cabinet"] = self.cabinet
        self.scene.rigid_objects["cube_1"] = self.cube_1
        self.scene.rigid_objects["cube_2"] = self.cube_2

        # terrain per envs
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.ee_frame = FrameTransformer(self.cfg.ee_frame)
        self.l4_frame = FrameTransformer(self.cfg.l4_frame)
        self.drawer_bottom_handle_frame = FrameTransformer(self.cfg.bottom_cabinet_handle_frame)
        self.drawer_top_handle_frame = FrameTransformer(self.cfg.top_cabinet_handle_frame)
        self.scene.sensors["ee_frame"] = self.ee_frame
        self.scene.sensors["l4_frame"] = self.l4_frame
        self.scene.sensors["drawer_bottom_handle_frame"] = self.drawer_bottom_handle_frame
        self.scene.sensors["drawer_top_handle_frame"] = self.drawer_top_handle_frame

        # replicate
        self.scene.clone_environments(copy_from_source=False)

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        super()._pre_physics_step(actions)

    def _apply_action(self) -> None:
        super()._apply_action()

    # episode interfaces
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        if getattr(self.cfg, "disable_terminations", False):
            done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            return done, done

        cube_1_pos = self.cube_1.data.root_state_w[:, :3]
        cube_2_pos = self.cube_2.data.root_state_w[:, :3]

        # target world positions inside drawers
        drawer_top_pos = self.cabinet.data.body_pos_w[:, self.drawer_top_idx]
        drawer_top_quat = self.cabinet.data.body_quat_w[:, self.drawer_top_idx]
        drawer_bottom_pos = self.cabinet.data.body_pos_w[:, self.drawer_bottom_idx]
        drawer_bottom_quat = self.cabinet.data.body_quat_w[:, self.drawer_bottom_idx]

        top_offset_world = tf_vector(drawer_top_quat, self.drawer_local_offset.repeat((self.num_envs, 1)))
        bottom_offset_world = tf_vector(drawer_bottom_quat, self.drawer_local_offset.repeat((self.num_envs, 1)))
        target_top = drawer_top_pos + top_offset_world
        target_bottom = drawer_bottom_pos + bottom_offset_world

        d1_top = torch.norm(cube_1_pos - target_top, dim=1)
        d2_bottom = torch.norm(cube_2_pos - target_bottom, dim=1)

        success_1 = d1_top < 0.05
        success_2 = d2_bottom < 0.05
        terminated = torch.logical_and(success_1, success_2)

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _get_rewards(self) -> torch.Tensor:
        # gonna need to change this reward to account for both cubes
        cube_1_pos = self.cube_1.data.root_state_w[:, :3]
        cube_2_pos = self.cube_2.data.root_state_w[:, :3]

        drawer_top_pos = self.cabinet.data.body_pos_w[:, self.drawer_top_idx]
        drawer_top_quat = self.cabinet.data.body_quat_w[:, self.drawer_top_idx]
        drawer_bottom_pos = self.cabinet.data.body_pos_w[:, self.drawer_bottom_idx]
        drawer_bottom_quat = self.cabinet.data.body_quat_w[:, self.drawer_bottom_idx]

        top_offset_world = tf_vector(drawer_top_quat, self.drawer_local_offset.repeat((self.num_envs, 1)))
        bottom_offset_world = tf_vector(drawer_bottom_quat, self.drawer_local_offset.repeat((self.num_envs, 1)))
        target_top = drawer_top_pos + top_offset_world
        target_bottom = drawer_bottom_pos + bottom_offset_world

        d1_top = torch.norm(cube_1_pos - target_top, dim=1)
        d2_bottom = torch.norm(cube_2_pos - target_bottom, dim=1)

        dist_reward =  (1.0 / (1.0 + d1_top) + 1.0 / (1.0 + d2_bottom))
        action_penalty =  torch.sum(self.actions ** 2, dim=1)

        rewards = dist_reward - action_penalty

        # bonus stages as cubes get close
        rewards = torch.where(d1_top < 0.10, rewards + 0.25, rewards)
        rewards = torch.where(d2_bottom < 0.10, rewards + 0.25, rewards)
        rewards = torch.where(d1_top < 0.05, rewards + 0.50, rewards)
        rewards = torch.where(d2_bottom < 0.05, rewards + 0.50, rewards)

        return rewards

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        super()._reset_idx(env_ids)

        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self._reset_robot(env_ids_t)

        # reset cabinet drawers closed
        zeros = torch.zeros((len(env_ids), self.cabinet.num_joints), device=self.device)
        self.cabinet.write_joint_state_to_sim(zeros, zeros, env_ids=env_ids)

        # rest cubes
        if getattr(self.cfg, "disable_reset_randomization", False):
            cube_pos = torch.tensor(
                [[[0.0, 0.0, 0.0], [0.08, 0.0, 0.0]]],
                device=self.device,
            ).repeat(len(env_ids), 1, 1)
        else:
            cube_pos = torch.rand(len(env_ids), 2, 3) * torch.tensor(
                [[[0.1, 0.1, 0.0]]]
            ).to(self.device)

        root = self.cube_1.data.default_root_state.clone()
        root = root.index_select(0, env_ids)
        root[:, :3] += (
            self.scene.env_origins.index_select(0, env_ids) + cube_pos[:, 0, :]
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
        self.cube_1.write_root_pose_to_sim(root[:, :7], env_ids=env_ids)
        self.cube_1.write_root_velocity_to_sim(
            torch.zeros((len(env_ids), 6), device=self.device), env_ids=env_ids
        )
        self.cube_1.reset(env_ids)

        root = self.cube_2.data.default_root_state.clone()
        root = root.index_select(0, env_ids)
        root[:, :3] += (
            self.scene.env_origins.index_select(0, env_ids) + cube_pos[:, 1, :]
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
        self.cube_2.write_root_pose_to_sim(root[:, :7], env_ids=env_ids)
        self.cube_2.write_root_velocity_to_sim(
            torch.zeros((len(env_ids), 6), device=self.device), env_ids=env_ids
        )
        self.cube_2.reset(env_ids)

    def _get_observations(self) -> dict:
        # pack a compact observation similar to prior envs
        dof_pos = self.robot.data.joint_pos
        dof_vel = self.robot.data.joint_vel
        cube_1_pos = self.cube_1.data.root_state_w[:, :3]
        cube_2_pos = self.cube_2.data.root_state_w[:, :3]
        drawer_q = self.cabinet.data.joint_pos[:, :]

        obs = torch.cat((dof_pos, dof_vel, cube_1_pos, cube_2_pos, drawer_q), dim=-1)
        return {"policy": obs}
