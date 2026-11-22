# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import datetime

from isaaclab.sensors.frame_transformer.frame_transformer import FrameTransformer
import torch
import os
import time

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObjectCfg, RigidObject
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import sample_uniform, quat_from_euler_xyz
from isaaclab.sensors import TiledCamera, TiledCameraCfg, CameraCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from kuafu.isaac.tasks.env_learned import BaseLearnedEnvCfg, BaseLearnedEnv
from kuafu.mpc.simulation.spatial_utils import quat_to_rotation_6d
TABLE_HEIGHT = 0
ISAAC_NUCLEUS_DIR = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac"

@configclass
class OpenDrawerEnvCfg(BaseLearnedEnvCfg):

    observation_space = 22  # default state dim; use set_obs_mode() for full_state
    obs_mode: str = "state"  # "state" (22) or "full_state" (56)
    obs_dims = {"state": 22, "full_state": 56}
    episode_length_s = 8

    # cabinet
    cabinet = ArticulationCfg(
        prim_path="/World/envs/env_.*/Cabinet",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd",
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(1.0, 0, 0.4),
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
                stiffness=0.0,
                damping=0.2,
            ),
            "doors": ImplicitActuatorCfg(
                joint_names_expr=["door_left_joint", "door_right_joint"],
                effort_limit_sim=87.0,
                stiffness=2.0,
                damping=1.0,
            ),
        },
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
    marker_cfg2.prim_path = "/Visuals/drawer_handle_top_frame"
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

    marker_cfg3 = FRAME_MARKER_CFG.copy()
    marker_cfg3.markers["frame"].scale = (0.1, 0.1, 0.1)
    marker_cfg3.prim_path = "/Visuals/door_left_nob_frame"
    left_door_nob_frame = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/panda_link0",
        debug_vis=False,
        visualizer_cfg=marker_cfg3,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Cabinet/door_left_nob_link",
                name="door_left_nob_frame",
                offset=OffsetCfg(
                    pos=[0.0, 0.0, 0.0],
                ),
            )
        ],
    )

    marker_cfg4 = FRAME_MARKER_CFG.copy()
    marker_cfg4.markers["frame"].scale = (0.1, 0.1, 0.1)
    marker_cfg4.prim_path = "/Visuals/door_right_nob_frame"
    right_door_nob_frame = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/panda_link0",
        debug_vis=False,
        visualizer_cfg=marker_cfg4,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Cabinet/door_right_nob_link",
                name="door_right_nob_frame",
                offset=OffsetCfg(
                    pos=[0.0, 0.0, 0.0],
                ),
            )
        ],
    )

    marker_cfg5 = FRAME_MARKER_CFG.copy()
    marker_cfg5.markers["frame"].scale = (0.1, 0.1, 0.1)
    marker_cfg5.prim_path = "/Visuals/drawer_bottom_frame"
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
    marker_cfg6.prim_path = "/Visuals/drawer_top_frame"
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


    rot = tuple(quat_from_euler_xyz(
        torch.tensor(1.32645), torch.tensor(0.0), torch.tensor(-2.35)
    ).tolist())

    table_cam = TiledCameraCfg(
        prim_path="/World/envs/env_.*/table_cam",
        update_period=0.0,
        height=256,
        width=256,
        data_types=[
            "rgb",
            "distance_to_image_plane",
            "instance_segmentation_fast",
            "instance_id_segmentation_fast",
        ],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 20),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(-1.65, 2.2, 1.18),
            rot=tuple(rot),
            convention="opengl",
        ),
    )

    # action mode: True for debugging (direct action), False for learning (incremental)
    use_debug_action_mode: bool = False
    detectable_objects = {
        "cabinet": (
            "mesh of the cabinet, local frame +z is up direction, local frame +x is the forward direction",
            "/World/envs/env_*/Cabinet",
        ),
        "door_left_link": (
            "mesh of left door, local frame +z is up direction, local frame +x is the pulling out direction",
            "/World/envs/env_*/Cabinet/door_left_link",
        ),
        "door_right_link": (
            "mesh of right door, local frame +z is up direction, local frame +x is the pulling out direction",
            "/World/envs/env_*/Cabinet/door_right_link",
        ),
        "drawer_bottom": (
            "mesh of bottom drawer, local frame +z is up direction, local frame +x is pulling out direction",
            "/World/envs/env_*/Cabinet/drawer_bottom",
        ),
        "drawer_top": (
            "mesh of top drawer, local frame +z is up direction, local frame +x is pulling out direction",
            "/World/envs/env_*/Cabinet/drawer_top",
        ),
        "drawer_bottom_handle_frame": (
            "frame of handle of the bottom drawer, handle is a horizontal bar, z is up direction, x is pulling out direction",
            "/Visuals/drawer_bottom_frame",
        ),
        "drawer_top_handle_frame": (
            "frame of handle of the top drawer, handle is a horizontal bar, local frame +z is up direction, local frame +x is pulling out direction",
            "/Visuals/drawer_top_frame",
        ),
        "left_door_nob_frame": (
            "frame of left door knob, z is up direction, x is protruding axis",
            "/Visuals/door_left_nob_frame",
        ),
        "right_door_nob_frame": (
            "frame of right door knob, z is up direction, x is protruding axis",
            "/Visuals/door_right_nob_frame",
        ),
        "panda_fingertip": ("End-effector frame", "/Visuals/panda_hand_frame"),
        "panda_link4": (
            "Elbow frame at middle of the arm",
            "/Visuals/panda_link4_frame",
        ),
    }

    kuafu_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )

    script_dir = os.path.dirname(os.path.dirname(os.path.dirname(kuafu_dir)))

    mpc_config = {
            "controller_name": "FrankaIKController",
            "device": "cuda",
            "ee_frame": "panda_fingertip",
            "use_per_env_weight": False,
            "load_program": f"{script_dir}/resources/learned_programs/manipulation-cabinet/opened_one_drawer_constraint_2",
            "initial_residual": f"{script_dir}/src/kuafu/skills/constraints/rewards.py",
            "initial_predicate": f"{script_dir}/src/kuafu/skills/constraints/predicates.py",
        }
    del script_dir

    task_description =  f"open the top drawer fully"

class OpenDrawerEnv(BaseLearnedEnv):
    cfg: OpenDrawerEnvCfg

    def _setup_scene(self):
        super()._setup_scene()
        self._cabinet = Articulation(self.cfg.cabinet)
        self.scene.articulations["cabinet"] = self._cabinet

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        self.drawer_bottom_handle_frame = FrameTransformer(
            self.cfg.bottom_cabinet_handle_frame
        )
        self.drawer_top_handle_frame = FrameTransformer(
            self.cfg.top_cabinet_handle_frame
        )

        self.left_door_nob_frame = FrameTransformer(self.cfg.left_door_nob_frame)
        self.right_door_nob_frame = FrameTransformer(self.cfg.right_door_nob_frame)


        self.scene.sensors["drawer_bottom_handle_frame"] = (
            self.drawer_bottom_handle_frame
        )
        self.scene.sensors["drawer_top_handle_frame"] = self.drawer_top_handle_frame
        self.scene.sensors["left_door_nob_frame"] = self.left_door_nob_frame
        self.scene.sensors["right_door_nob_frame"] = self.right_door_nob_frame

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        if self.recorder_manager is not None and len(env_ids) > 0:
            self.recorder_manager.record_pre_reset(env_ids)
        num_resets = len(env_ids)

        # Reset cabinet joint positions to zero
        zeros = torch.zeros(
            (num_resets, self._cabinet.num_joints), device=self.device
        )
        self._cabinet.write_joint_state_to_sim(zeros, zeros, env_ids=env_ids)


        self._cabinet.reset(env_ids)
        super()._reset_idx(env_ids)

    def _get_observations(self) -> dict:
        self.update_data_in_controller()

        api = self.program_runtime.controller_api_wrapper
        agent = api.reference_agent
        if agent is None:
            raise RuntimeError("controller_api_wrapper.reference_agent is not set")

        handle_pos = api.get_object_xyz_position("drawer_top_handle_frame")  # (N, 3)
        fingertip_pos = api.lookup_robot_frame_position("panda_fingertip")  # (N, 3)
        link4_pos = api.lookup_robot_frame_position("panda_link4")  # (N, 3)

        handle_rot6d = quat_to_rotation_6d(
            api.get_object_wxyz_quaternion("drawer_top_handle_frame")
        )  # (N, 6)
        fingertip_rot6d = quat_to_rotation_6d(
            api.lookup_robot_frame_rotation("panda_fingertip")
        )  # (N, 6)
        link4_rot6d = quat_to_rotation_6d(
            api.lookup_robot_frame_rotation("panda_link4")
        )  # (N, 6)

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
            fingertip_pos,  # 6
            handle_rot6d,
            fingertip_rot6d,  # 12
            gripper_open,
            gripper_effort,  # 3
            stage,  # 1
        )  # total: 22

        full_state_obs = _f(
            handle_pos,
            fingertip_pos,
            link4_pos,  # 9
            handle_rot6d,
            fingertip_rot6d,
            link4_rot6d,  # 18
            joint_pos,
            joint_vel,
            joint_effort,  # 27
            gripper_open,  # 1
            stage,  # 1
        )  # total: 56

        obs_dim = int(self.cfg.observation_space)
        policy_obs_to_return = policy_obs if self.cfg.obs_mode == "state" else full_state_obs
        if policy_obs_to_return.shape[-1] != obs_dim:
            raise RuntimeError(
                f"Observation dim mismatch: got {policy_obs_to_return.shape[-1]}, expected {obs_dim}"
            )

        # Convert stage to float for calculation and log to extras
        # "episode/" prefix groups it nicely in WandB
        self.extras["episode/mean_stage"] = torch.mean(self.current_stage.float())
        self.extras["episode/max_stage"] = torch.max(self.current_stage.float())

        if torch.isnan(policy_obs_to_return).any():
            print("obs is nan")
            torch.nan_to_num_(policy_obs_to_return, nan=0.0, posinf=0.0, neginf=0.0)
            torch.nan_to_num_(full_state_obs, nan=0.0, posinf=0.0, neginf=0.0)

        return {
            "policy": policy_obs_to_return,
            "full_state": full_state_obs,
            "state": policy_obs,
        }


    def _external_get_dones(self,
    ):

        # Debug termination condition (can override for testing)
        terminated = self._cabinet.data.joint_pos[:, 3] > 0.2
        gripper_openness = self.program_runtime.controller_api_wrapper.gripper_openness().flatten()

        return torch.logical_and(self.current_stage.flatten() > int(self.program_runtime.compiled_program.stage_count()/2), torch.logical_and(gripper_openness > 0.7, terminated))
