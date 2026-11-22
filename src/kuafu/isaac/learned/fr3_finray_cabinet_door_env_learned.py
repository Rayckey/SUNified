# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os

import torch

import isaaclab.sim as sim_utils
from kuafu.mpc.simulation.spatial_utils import quat_to_rotation_6d
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer import FrameTransformer
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.utils import bind_physics_material
from isaaclab.utils import configclass
from isaacsim.core.utils.stage import get_current_stage

from kuafu.isaac.learned.fr3_finray_env_learned import (
    BaseLearnedEnv,
    BaseLearnedEnvCfg,
)


ISAAC_NUCLEUS_DIR = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac"
)
FLOOR_THICKNESS = 0.05


def _frame_marker(prim_path: str, scale: float = 0.05):
    marker_cfg = FRAME_MARKER_CFG.copy()
    marker_cfg.markers["frame"].scale = (scale, scale, scale)
    marker_cfg.prim_path = prim_path
    return marker_cfg


@configclass
class FR3FinRayCabinetDoorEnvLearnedCfg(BaseLearnedEnvCfg):
    """FR3/FinRay learned task for opening the Sektion cabinet door."""

    observation_space = 22
    obs_mode: str = "state"
    obs_dims = {"state": 22, "full_state": 56}
    episode_length_s = 15
    door_open_threshold: float = 1.2

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4, env_spacing=3.0, replicate_physics=True
    )

    door_link_physics_material = sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="max",
        restitution_combine_mode="multiply",
        static_friction=2.0,
        dynamic_friction=2.0,
        restitution=0.0,
    )

    floor = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Floor",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, -FLOOR_THICKNESS / 2.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.CuboidCfg(
            size=(4.0, 4.0, FLOOR_THICKNESS),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.28, 0.30, 0.32),
                roughness=0.75,
            ),
        ),
    )

    def __post_init__(self):
        super().__post_init__()
        self.robot.init_state = ArticulationCfg.InitialStateCfg(
            joint_pos={
                "fr3_joint1": -0.0,
                "fr3_joint2": -0.45,
                "fr3_joint3": 0.1,
                "fr3_joint4": -2.25,
                "fr3_joint5": -2.4,
                "fr3_joint6": 2.03,
                "fr3_joint7": 0.0,
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
            pos=(1.12, 0.0, 0.7),
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
                effort_limit_sim=87.0,
                stiffness=0.5,
                damping=0.2,
            ),
            "doors": ImplicitActuatorCfg(
                joint_names_expr=["door_left_joint", "door_right_joint"],
                effort_limit_sim=87.0,
                stiffness=0.0,
                damping=3.0,
            ),
        },
    )

    left_door_hinge_frame = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/fr3/base",
        debug_vis=False,
        visualizer_cfg=_frame_marker("/Visuals/left_door_hinge_frame"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Cabinet/door_left_link",
                name="left_door_hinge_frame",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
            )
        ],
    )

    right_door_hinge_frame = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/fr3/base",
        debug_vis=False,
        visualizer_cfg=_frame_marker("/Visuals/right_door_hinge_frame"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Cabinet/door_right_link",
                name="right_door_hinge_frame",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
            )
        ],
    )

    left_door_nob_frame = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/fr3/base",
        debug_vis=False,
        visualizer_cfg=_frame_marker("/Visuals/door_left_nob_frame"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Cabinet/door_left_link",
                name="left_door_nob_frame",
                offset=OffsetCfg(pos=(0.025, 0.35, 0.185)),
            )
        ],
    )

    right_door_nob_frame = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/fr3/base",
        debug_vis=False,
        visualizer_cfg=_frame_marker("/Visuals/door_right_nob_frame"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Cabinet/door_right_link",
                name="right_door_nob_frame",
                offset=OffsetCfg(pos=(0.025, -0.35, 0.185)),
            )
        ],
    )

    detectable_objects = {
        "cabinet": (
            "Sektion cabinet, local frame +z is up direction and local frame +x is the forward direction",
            "/World/envs/env_*/Cabinet",
        ),
        "door_left_link": (
            "left cabinet door, hinged on its outer vertical edge and opened by rotating away from the cabinet up to approximately 90 degrees; estimated size (approx.): (0.04, 0.34, 0.45)",
            "/World/envs/env_*/Cabinet/door_left_link",
        ),
        "door_right_link": (
            "right cabinet door, hinged on its outer vertical edge and opened by rotating away from the cabinet up to approximately 90 degrees; estimated size (approx.): (0.04, 0.34, 0.45)",
            "/World/envs/env_*/Cabinet/door_right_link",
        ),
        "left_door_hinge_frame": (
            "hinge frame of the left cabinet door; local frame +z is the hinge rotation axis; rotation around local frame -z to open left door, rotation around local frame +z to close left door",
            "/Visuals/left_door_hinge_frame",
        ),
        "right_door_hinge_frame": (
            "hinge frame of the right cabinet door; local frame +z is the hinge rotation axis; rotation around local frame +z to open right door, rotation around local frame -z to close right door",
            "/Visuals/right_door_hinge_frame",
        ),
        "left_door_nob_frame": (
            "frame of a circular left door knob, size (0.02, 0.02, 0.02); grasp against local frame +x",
            "/Visuals/door_left_nob_frame",
        ),
        "right_door_nob_frame": (
            "frame of a circular right door knob, size (0.02, 0.02, 0.02); grasp against local frame +x",
            "/Visuals/door_right_nob_frame",
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

    _kuafu_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(_kuafu_dir)))
    mpc_config = {
        "controller_name": "FrankaIKController",
        "device": "cuda",
        "ee_frame": "panda_fingertip",
        "load_program": (
            f"{_repo_dir}/resources/learned_programs/manipulation-door-either/"
            "2026-05-11_23-08-46"
        ),
        "initial_residual": f"{_repo_dir}/src/kuafu/skills/constraints/rewards.py",
        "initial_predicate": f"{_repo_dir}/src/kuafu/skills/constraints/predicates.py",
        "use_per_env_weight": False,
        "gripper_opened": [0.0, 0.0],
        "gripper_closed": [0.8, -0.8],
        "gripper_max_force": 1.0,
    }
    del _repo_dir

    task_description = "open the right cabinet door fully"


class FR3FinRayCabinetDoorEnvLearned(BaseLearnedEnv):
    cfg: FR3FinRayCabinetDoorEnvLearnedCfg

    def __init__(
        self,
        cfg: FR3FinRayCabinetDoorEnvLearnedCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        super().__init__(cfg, render_mode, **kwargs)
        self._resolve_door_joint_indices()

    def _setup_scene(self):
        super()._setup_scene()

        self._cabinet = Articulation(self.cfg.cabinet)
        self.scene.articulations["cabinet"] = self._cabinet
        self.floor = RigidObject(self.cfg.floor)
        self.scene.rigid_objects["floor"] = self.floor

        self.left_door_hinge_frame = FrameTransformer(self.cfg.left_door_hinge_frame)
        self.right_door_hinge_frame = FrameTransformer(self.cfg.right_door_hinge_frame)
        self.left_door_nob_frame = FrameTransformer(self.cfg.left_door_nob_frame)
        self.right_door_nob_frame = FrameTransformer(self.cfg.right_door_nob_frame)

        self.scene.sensors["left_door_hinge_frame"] = self.left_door_hinge_frame
        self.scene.sensors["right_door_hinge_frame"] = self.right_door_hinge_frame
        self.scene.sensors["left_door_nob_frame"] = self.left_door_nob_frame
        self.scene.sensors["right_door_nob_frame"] = self.right_door_nob_frame

        self.scene.clone_environments(copy_from_source=False)

        self._apply_door_link_friction()

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _apply_door_link_friction(self):
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

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        if self.recorder_manager is not None and len(env_ids) > 0:
            self.recorder_manager.record_pre_reset(env_ids)

        num_resets = len(env_ids)
        root = self._cabinet.data.default_root_state.clone().index_select(0, env_ids)
        root[:, :3] += self.scene.env_origins.index_select(0, env_ids)
        self._cabinet.write_root_pose_to_sim(root[:, :7], env_ids=env_ids)
        self._cabinet.write_root_velocity_to_sim(
            torch.zeros((num_resets, 6), device=self.device), env_ids=env_ids
        )

        zeros = torch.zeros((num_resets, self._cabinet.num_joints), device=self.device)
        self._cabinet.write_joint_state_to_sim(zeros, zeros, env_ids=env_ids)
        self._cabinet.reset(env_ids)

        super()._reset_idx(env_ids)

    def _get_observations(self) -> dict:
        self.update_data_in_controller()

        api = self.program_runtime.controller_api_wrapper
        agent = api.reference_agent
        if agent is None:
            raise RuntimeError("controller_api_wrapper.reference_agent is not set")

        knob_pos = api.get_object_xyz_position("right_door_nob_frame")
        fingertip_pos = api.lookup_robot_frame_position("panda_fingertip")
        link4_pos = api.lookup_robot_frame_position("panda_link4")

        knob_rot6d = quat_to_rotation_6d(
            api.get_object_wxyz_quaternion("right_door_nob_frame")
        )
        fingertip_rot6d = quat_to_rotation_6d(
            api.lookup_robot_frame_rotation("panda_fingertip")
        )
        link4_rot6d = quat_to_rotation_6d(
            api.lookup_robot_frame_rotation("panda_link4")
        )

        joint_pos = api.joint_dict[agent]["joint_position"]
        joint_vel = api.joint_dict[agent]["joint_velocity"]
        joint_effort = self._get_normalized_joint_effort(api, agent)
        gripper_contact = self._get_normalized_gripper_contact(api)
        joint_effort = torch.cat([joint_effort[:, :7], gripper_contact], dim=-1)
        gripper_open = api.gripper_openness().unsqueeze(-1)
        stage = self.current_stage

        def _f(*tensors: torch.Tensor) -> torch.Tensor:
            return torch.cat(
                [t.to(device=self.device, dtype=torch.float32) for t in tensors],
                dim=-1,
            )

        policy_obs = _f(
            knob_pos,
            fingertip_pos,
            knob_rot6d,
            fingertip_rot6d,
            gripper_open,
            gripper_contact,
            stage,
        )

        full_state_obs = _f(
            knob_pos,
            fingertip_pos,
            link4_pos,
            knob_rot6d,
            fingertip_rot6d,
            link4_rot6d,
            joint_pos,
            joint_vel,
            joint_effort,
            gripper_open,
            stage,
        )

        obs = policy_obs if self.cfg.obs_mode == "state" else full_state_obs
        obs_dim = int(self.cfg.observation_space)
        if obs.shape[-1] != obs_dim:
            raise RuntimeError(
                f"Observation dim mismatch: got {obs.shape[-1]}, expected {obs_dim}"
            )

        self.extras["episode/mean_stage"] = torch.mean(self.current_stage.float())
        self.extras["episode/max_stage"] = torch.max(self.current_stage.float())

        if not torch.isfinite(obs).all():
            print(
                "[FR3FinRayCabinetDoorEnvLearned] obs contains NaN/Inf; sanitizing with nan_to_num.",
                flush=True,
            )
            torch.nan_to_num_(obs, nan=0.0, posinf=0.0, neginf=0.0)
            torch.nan_to_num_(full_state_obs, nan=0.0, posinf=0.0, neginf=0.0)
            torch.nan_to_num_(policy_obs, nan=0.0, posinf=0.0, neginf=0.0)

        return {
            "policy": obs,
            "full_state": full_state_obs,
            "state": policy_obs,
        }

    def _external_get_dones(self):
        self.update_data_in_controller()

        opened = torch.any(
            self._door_open_amounts() > float(self.cfg.door_open_threshold), dim=-1
        )
        if self.cfg.sun_program_evaluator_stage_override:
            return opened

        past_stage = self.current_stage.flatten() > int(
            self.program_runtime.compiled_program.stage_count() / 2
        )
        return torch.logical_and(opened, past_stage)
