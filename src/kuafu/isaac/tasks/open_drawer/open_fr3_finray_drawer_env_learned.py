# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
from kuafu.paths import OBJECT_ROOT

import os
from isaaclab.sensors.frame_transformer.frame_transformer import FrameTransformer
import torch
import time
from pxr import UsdGeom

from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.utils.torch.transformations import tf_combine, tf_inverse, tf_vector

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObjectCfg, RigidObject
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, RenderCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import sample_uniform, quat_from_euler_xyz
from isaaclab.sensors import TiledCamera, TiledCameraCfg, CameraCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from omegaconf import OmegaConf
from kuafu.mpc.simulation.sun_program_evaluator import SUNProgramEvaluator

from kuafu.isaac.tasks.env_learned import BaseLearnedEnvCfg, BaseLearnedEnv
from kuafu.mpc.simulation.spatial_utils import quat_to_rotation_6d

TABLE_HEIGHT = 0
ISAAC_NUCLEUS_DIR = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac"

@configclass
class OpenDrawerEnvCfg(BaseLearnedEnvCfg):
    # env
    observation_space = 22  # default state dim; use set_obs_mode() for full_state
    obs_mode: str = "state"  # "state" (22) or "full_state" (56)
    obs_dims = {"state": 22, "full_state": 56}

    _cabinet_path = str(OBJECT_ROOT / 'two_layer_drawer_iphone_with_link/two_layer_drawer_iphone_with_link.usd')

    # cabinet
    cabinet = ArticulationCfg(
        prim_path="/World/envs/env_.*/Cabinet",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_cabinet_path,
            activate_contact_sensors=False,
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.001,  # 1mm contact offset for better precision
                rest_offset=0.001,
            ),
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
                damping=0.2,
            )
        },
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
                    pos=[0.0, 0.0, 0.0],
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
                    pos=[0.0, 0.10, 0.0],
                ),
            )
        ],
    )


    detectable_objects = {
            "cabinet": (
                "mesh of the cabinet, local frame +z is up direction, local frame +x is the forward direction",
                "/World/envs/env_*/Cabinet",
            ),
            "upper_drawer": (
                "mesh of upper drawer, local frame +z is up direction, local frame +x is pulling out direction",
                "/World/envs/env_*/Cabinet/upper_drawer",
            ),
            "top_drawer_handle_frame": (
                "frame of handle of the upper drawer, handle is a horizontal bar, z is up direction, x is pulling out direction",
                "/Visuals/top_drawer_handle_frame",
            ),
            "top_drawer_center_frame": (
                "frame of center of the upper drawer, z is up direction, x is pulling out direction",
                "/Visuals/top_drawer_center_frame",
            ),
            "panda_fingertip": ("End-effector frame", "/Visuals/panda_hand_frame"),
            "panda_link4": (
                "Elbow frame at middle of the arm",
                "/Visuals/panda_link4_frame",
            )
        }

    kuafu_dir =  os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    script_dir = os.path.dirname(os.path.dirname(os.path.dirname(kuafu_dir)))

    mpc_config = {
            "controller_name": "FrankaIKController",
            "device": "cuda",
            "ee_frame": "panda_fingertip",
            "use_per_env_weight": False,
            "load_program": f"{script_dir}/resources/learned_programs/manipulation-fr3-finray-drawer/2026-02-26_09-33-35",
            "initial_residual": f"{script_dir}/src/kuafu/skills/constraints/rewards.py",
            "initial_predicate": f"{script_dir}/src/kuafu/skills/constraints/predicates.py",
        }
    del script_dir

    task_description = f"open the top drawer fully"

class OpenDrawerEnv(BaseLearnedEnv):
    cfg: OpenDrawerEnvCfg


    def _setup_scene(self):
        super()._setup_scene()
        self._cabinet = Articulation(self.cfg.cabinet)
        self.scene.articulations["cabinet"] = self._cabinet


        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        self.top_drawer_handle_frame = FrameTransformer(
            self.cfg.top_drawer_handle_frame
        )

        self.top_drawer_center_frame = FrameTransformer(
            self.cfg.top_drawer_center_frame
        )

        # Register task-specific sensors
        self.scene.sensors["top_drawer_handle_frame"] = self.top_drawer_handle_frame
        self.scene.sensors["top_drawer_center_frame"] = self.top_drawer_center_frame

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
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

        # Randomize cabinet pose
        dist = torch.rand(num_resets, device=self.device) * 0.15 + 0.4        # 0.4–0.55 m from robot root
        azimuth = (torch.rand(num_resets, device=self.device) - 0.5) * torch.pi  # placement angle: front hemisphere
        height = torch.rand(num_resets, device=self.device) * 0.15 + 0.15     # 0.15–0.3 m

        # Cabinet yaw: face the robot (azimuth + π) with a 30–90° offset to either side
        facing_yaw = azimuth + torch.pi
        offset_sign = torch.randint(0, 2, (num_resets,), device=self.device).float() * 2.0 - 1.0
        offset_rad = torch.rand(num_resets, device=self.device) * (torch.pi / 3.0) + (torch.pi / 6.0)
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
        super()._reset_idx(env_ids)

    def _get_observations(self) -> dict:
        self.update_data_in_controller()

        api = self.program_runtime.controller_api_wrapper
        agent = api.reference_agent
        if agent is None:
            raise RuntimeError("controller_api_wrapper.reference_agent is not set")

        handle_pos = api.get_object_xyz_position("top_drawer_handle_frame")  # (N, 3)
        fingertip_pos = api.lookup_robot_frame_position("panda_fingertip")  # (N, 3)
        link4_pos = api.lookup_robot_frame_position("panda_link4")  # (N, 3)

        handle_rot6d = quat_to_rotation_6d(
            api.get_object_wxyz_quaternion("top_drawer_handle_frame")
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

        return {
            "policy": policy_obs_to_return,
            "full_state": full_state_obs,
            "state": policy_obs,
        }

    def _external_get_dones(self,
    ):

        # Debug termination condition (can override for testing)
        opened = self._cabinet.data.joint_pos[:, 0] > 0.1

        return torch.logical_and(self.current_stage.flatten() > int(self.program_runtime.compiled_program.stage_count()-2), opened)
