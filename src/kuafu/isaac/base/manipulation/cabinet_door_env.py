# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.utils import bind_physics_material
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer import FrameTransformer
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz
from isaacsim.core.utils.stage import get_current_stage

from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip

from .panda_manipulation_env import PandaManipulationEnv, PandaManipulationEnvCfg


ISAAC_NUCLEUS_DIR = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac"
)


def _frame_marker(prim_path: str):
    marker_cfg = FRAME_MARKER_CFG.copy()
    marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
    marker_cfg.prim_path = prim_path
    return marker_cfg


@configclass
class CabinetDoorEnvCfg(PandaManipulationEnvCfg):
    """Base configuration for opening doors on the Sektion cabinet asset."""

    observation_space = 22
    enable_cameras: bool = False
    use_debug_action_mode: bool = False
    disable_terminations: bool = False
    target_mode: str = "either"
    door_open_threshold: float = 1.2
    door_link_physics_material = sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="max",
        restitution_combine_mode="multiply",
        static_friction=2.0,
        dynamic_friction=2.0,
        restitution=0.0,
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=3.0, replicate_physics=True
    )

    robot: ArticulationCfg = PandaManipulationEnvCfg().robot.replace(
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 0.05,
                "panda_joint2": -0.35,
                "panda_joint3": 0.1,
                "panda_joint4": -2.0,
                "panda_joint5": -2.4,
                "panda_joint6": 2.03,
                "panda_joint7": 0.0,
                "panda_finger_joint.*": 0.04,
            },
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        )
    )

    cabinet = ArticulationCfg(
        prim_path="/World/envs/env_.*/Cabinet",
        spawn=sim_utils.UsdFileCfg(
            usd_path=(
                f"{ISAAC_NUCLEUS_DIR}/Props/Sektion_Cabinet/"
                "sektion_cabinet_instanceable.usd"
            ),
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(1.0, 0.0, 0.4),
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
                stiffness=0.0,
                damping=3.0,
            ),
        },
    )

    left_hinge_marker_cfg = _frame_marker("/Visuals/left_door_hinge_frame")
    left_door_hinge_frame = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/panda_link0",
        debug_vis=True,
        visualizer_cfg=left_hinge_marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Cabinet/door_left_link",
                name="left_door_hinge_frame",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
            )
        ],
    )

    right_hinge_marker_cfg = _frame_marker("/Visuals/right_door_hinge_frame")
    right_door_hinge_frame = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/panda_link0",
        debug_vis=True,
        visualizer_cfg=right_hinge_marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Cabinet/door_right_link",
                name="right_door_hinge_frame",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
            )
        ],
    )

    left_door_nob_marker_cfg = _frame_marker("/Visuals/door_left_nob_frame")
    left_door_nob_frame = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/panda_link0",
        debug_vis=True,
        visualizer_cfg=left_door_nob_marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Cabinet/door_left_link",
                name="left_door_nob_frame",
                offset=OffsetCfg(pos=(0.025, 0.35, 0.185)),
            )
        ],
    )

    right_door_nob_marker_cfg = _frame_marker("/Visuals/door_right_nob_frame")
    right_door_nob_frame = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/panda_link0",
        debug_vis=True,
        visualizer_cfg=right_door_nob_marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Cabinet/door_right_link",
                name="right_door_nob_frame",
                offset=OffsetCfg(pos=(0.025, -0.35, 0.185)),
            )
        ],
    )

    rot = tuple(
        quat_from_euler_xyz(
            torch.tensor(1.32645), torch.tensor(0.0), torch.tensor(-2.35)
        ).tolist()
    )

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


@configclass
class CabinetDoorEitherEnvCfg(CabinetDoorEnvCfg):
    """Open either the left or right cabinet door."""

    target_mode: str = "either"


@configclass
class CabinetDoorBothEnvCfg(CabinetDoorEnvCfg):
    """Open both cabinet doors."""

    target_mode: str = "both"


class CabinetDoorEnv(PandaManipulationEnv):
    cfg: CabinetDoorEnvCfg

    def __init__(self, cfg: CabinetDoorEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._resolve_door_joint_indices()

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self._cabinet = Articulation(self.cfg.cabinet)
        self.scene.articulations["robot"] = self._robot
        self.scene.articulations["cabinet"] = self._cabinet
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        self._apply_door_link_friction()

        self.ee_frame = FrameTransformer(self.cfg.ee_frame)
        self.l4_frame = FrameTransformer(self.cfg.l4_frame)
        self.left_door_hinge_frame = FrameTransformer(self.cfg.left_door_hinge_frame)
        self.right_door_hinge_frame = FrameTransformer(self.cfg.right_door_hinge_frame)
        self.left_door_nob_frame = FrameTransformer(self.cfg.left_door_nob_frame)
        self.right_door_nob_frame = FrameTransformer(self.cfg.right_door_nob_frame)
        self.scene.sensors["panda_fingertip"] = self.ee_frame
        self.scene.sensors["panda_link4"] = self.l4_frame
        self.scene.sensors["left_door_hinge_frame"] = self.left_door_hinge_frame
        self.scene.sensors["right_door_hinge_frame"] = self.right_door_hinge_frame
        self.scene.sensors["left_door_nob_frame"] = self.left_door_nob_frame
        self.scene.sensors["right_door_nob_frame"] = self.right_door_nob_frame

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _apply_door_link_friction(self):
        """Bind high-friction material only to the cabinet door link colliders."""
        stage = get_current_stage()
        material_path = "/World/Physics/cabinetDoorHighFrictionMaterial"
        self.cfg.door_link_physics_material.func(
            material_path, self.cfg.door_link_physics_material
        )

        for prim_path in (
            "/World/envs/env_.*/Cabinet/door_left_link",
            "/World/envs/env_.*/Cabinet/door_right_link",
        ):
            for prim in sim_utils.find_matching_prims(prim_path):
                bind_physics_material(
                    prim.GetPath().pathString,
                    material_path,
                    stage=stage,
                )

    def _resolve_door_joint_indices(self):
        if hasattr(self, "left_door_joint_idx") and hasattr(
            self, "right_door_joint_idx"
        ):
            return
        self.left_door_joint_idx = self._cabinet.find_joints("door_left_joint")[0][0]
        self.right_door_joint_idx = self._cabinet.find_joints("door_right_joint")[0][0]

    def _door_open_amounts(self) -> torch.Tensor:
        self._resolve_door_joint_indices()
        joint_pos = self._cabinet.data.joint_pos
        return torch.abs(
            torch.stack(
                (
                    joint_pos[:, self.left_door_joint_idx],
                    joint_pos[:, self.right_door_joint_idx],
                ),
                dim=-1,
            )
        )

    def _target_open_amount(self) -> torch.Tensor:
        door_open = self._door_open_amounts()
        if self.cfg.target_mode == "both":
            return torch.min(door_open, dim=-1).values
        return torch.max(door_open, dim=-1).values

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cfg.disable_terminations:
            done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            return done, done

        door_open = self._door_open_amounts()
        threshold = self.cfg.door_open_threshold
        if self.cfg.target_mode == "both":
            terminated = torch.all(door_open > threshold, dim=-1)
        else:
            terminated = torch.any(door_open > threshold, dim=-1)
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        return self._target_open_amount()

    def _reset_robot(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        super()._reset_idx(env_ids)

        self._reset_robot(env_ids)

        zeros = torch.zeros((len(env_ids), self._cabinet.num_joints), device=self.device)
        self._cabinet.write_joint_state_to_sim(zeros, zeros, env_ids=env_ids)

    def _get_observations(self) -> dict:
        self._resolve_door_joint_indices()
        dof_pos_scaled = (
            2.0
            * (self._robot.data.joint_pos - self.robot_dof_lower_limits)
            / (self.robot_dof_upper_limits - self.robot_dof_lower_limits)
            - 1.0
        )
        door_pos = self._cabinet.data.joint_pos[
            :, [self.left_door_joint_idx, self.right_door_joint_idx]
        ]
        door_vel = self._cabinet.data.joint_vel[
            :, [self.left_door_joint_idx, self.right_door_joint_idx]
        ]

        obs = torch.cat(
            (
                dof_pos_scaled,
                self._robot.data.joint_vel,
                door_pos,
                door_vel,
            ),
            dim=-1,
        )
        return {"policy": torch.clamp(obs, -5.0, 5.0)}


class CabinetDoorEitherEnv(CabinetDoorEnv):
    cfg: CabinetDoorEitherEnvCfg


class CabinetDoorBothEnv(CabinetDoorEnv):
    cfg: CabinetDoorBothEnvCfg
