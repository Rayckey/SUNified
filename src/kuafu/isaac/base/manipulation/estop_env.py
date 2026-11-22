from __future__ import annotations

import os
from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformer, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass

from kuafu.isaac.objects.estop import EStopArticulation, EStopSemanticCfg, make_estop_cfg
from kuafu.isaac.base.manipulation.panda_manipulation_env import (
    PandaManipulationEnvCfg,
    PandaManipulationEnv,
)

TABLE_HEIGHT = 0.2


@configclass
class EStopEnvCfg(PandaManipulationEnvCfg):
    """Configuration for the E-Stop environment.

    A Panda robot sits at a table and must press, twist, and release
    the articulated e-stop button mounted on the table surface.
    """

    episode_length_s: float = 8.0
    decimation: int = 2
    action_space: int = 9   # 7 arm + 2 finger joints
    # robot state (9 pos + 9 vel) + ee pos (3) + estop body pos (3)
    # + estop joint state (4) + progress flags (2) = 30
    observation_space: int = 30
    state_space: int = 0

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=8, env_spacing=1.5, replicate_physics=True
    )

    def __post_init__(self):
        super().__post_init__()
        self.sim.render_interval = self.decimation

    # ------------------------------------------------------------------
    # E-stop articulation
    # ------------------------------------------------------------------
    estop: ArticulationCfg = make_estop_cfg(
        prim_path="/World/envs/env_.*/EStop",
        spawn_pose=(0.5, 0.0, TABLE_HEIGHT + 0.01),
    )

    semantic_cfg: EStopSemanticCfg = EStopSemanticCfg()

    success_bonus: float = 5.0
    release_tolerance: float = 0.002

    # ------------------------------------------------------------------
    # Table
    # ------------------------------------------------------------------
    table: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Table",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.5, 0.0, TABLE_HEIGHT / 2), rot=(1.0, 0.0, 0.0, 0.0)
        ),
        spawn=sim_utils.CuboidCfg(
            size=(0.4, 0.6, TABLE_HEIGHT),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
        ),
    )

    # ------------------------------------------------------------------
    # Frame transformers (override parent defaults so prim paths match)
    # ------------------------------------------------------------------
    _marker_h = FRAME_MARKER_CFG.copy()
    _marker_h.markers["frame"].scale = (0.05, 0.05, 0.05)
    _marker_h.prim_path = "/Visuals/panda_hand_frame"

    _marker_l = FRAME_MARKER_CFG.copy()
    _marker_l.markers["frame"].scale = (0.05, 0.05, 0.05)
    _marker_l.prim_path = "/Visuals/panda_link4_frame"

    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/panda_link0",
        debug_vis=False,
        visualizer_cfg=_marker_h,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot/panda_hand",
                name="panda_fingertip",
                offset=OffsetCfg(pos=[0.0, 0.0, 0.1034]),
            )
        ],
    )

    l4_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/panda_link0",
        debug_vis=False,
        visualizer_cfg=_marker_l,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot/panda_link4",
                name="panda_link4",
            )
        ],
    )


class EStopEnv(PandaManipulationEnv):
    """Panda robot presses, twists, and releases an e-stop on a table."""

    cfg: EStopEnvCfg

    def __init__(self, cfg: EStopEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        self.estop_joint_lower_limits = self.estop.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.estop_joint_upper_limits = self.estop.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)

        self.press_joint_id = self.estop.find_joints(self.cfg.semantic_cfg.press_joint_name)[0][0]
        self.twist_joint_id = self.estop.find_joints(self.cfg.semantic_cfg.twist_joint_name)[0][0]

        self._pressed_once = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._unlocked_once = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        print("[INFO]: Completed setting up EStopEnv.")

    def _setup_scene(self):
        # Robot (required before PandaManipulationEnv.__init__ completes)
        self.robot = Articulation(self.cfg.robot)
        self._robot = self.robot

        # E-stop
        self.estop = EStopArticulation(self.cfg.estop, semantic_cfg=self.cfg.semantic_cfg)

        # Table
        self.table = RigidObject(self.cfg.table)

        # Frame transformers
        self.ee_frame = FrameTransformer(self.cfg.ee_frame)
        self.l4_frame = FrameTransformer(self.cfg.l4_frame)

        # Terrain
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # Register with scene
        self.scene.articulations["robot"] = self.robot
        self.scene.articulations["estop"] = self.estop
        self.scene.rigid_objects["table"] = self.table
        self.scene.sensors["panda_fingertip"] = self.ee_frame
        self.scene.sensors["panda_link4"] = self.l4_frame
        self.scene.clone_environments(copy_from_source=False)

        light_cfg = sim_utils.DomeLightCfg(intensity=1000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = list(range(self.num_envs))

        env_ids_tensor = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        super()._reset_idx(env_ids)

        # Reset estop joints
        joint_pos = self.estop.data.default_joint_pos[env_ids_tensor].clone()
        joint_vel = torch.zeros_like(joint_pos)
        self.estop.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids_tensor)
        self.estop.set_joint_position_target(joint_pos, env_ids=env_ids_tensor)

        self._pressed_once[env_ids_tensor] = False
        self._unlocked_once[env_ids_tensor] = False
        self._success[env_ids_tensor] = False

        self._reset_robot(env_ids_tensor)

    def _update_progress_flags(self):
        press_pos = self.estop.data.joint_pos[:, self.press_joint_id]
        twist_pos = self.estop.data.joint_pos[:, self.twist_joint_id]

        pressed_now = press_pos <= -(0.75 * self.cfg.semantic_cfg.latched_position)
        unlocked_now = twist_pos >= (
            self.cfg.semantic_cfg.unlock_twist_angle_rad - self.cfg.semantic_cfg.unlock_tolerance_rad
        )
        released_now = press_pos >= -self.cfg.release_tolerance

        self._pressed_once |= pressed_now
        self._unlocked_once |= self._pressed_once & unlocked_now
        self._success |= self._unlocked_once & released_now

        return press_pos, twist_pos

    def _get_observations(self) -> dict:
        press_pos, twist_pos = self._update_progress_flags()

        robot_joint_pos = self.robot.data.joint_pos          # (N, 9)
        robot_joint_vel = self.robot.data.joint_vel          # (N, 9)
        ee_pos_w = self.ee_frame.data.target_pos_w[:, 0, :]  # (N, 3)
        estop_pos_w = self.estop.data.root_pos_w             # (N, 3)

        obs = torch.cat(
            [
                robot_joint_pos,                             # 9
                robot_joint_vel,                             # 9
                ee_pos_w - self.scene.env_origins,           # 3  (env-local)
                estop_pos_w - self.scene.env_origins,        # 3  (env-local)
                press_pos.unsqueeze(-1),                     # 1
                twist_pos.unsqueeze(-1),                     # 1
                self.estop.data.joint_vel[:, self.press_joint_id].unsqueeze(-1),  # 1
                self.estop.data.joint_vel[:, self.twist_joint_id].unsqueeze(-1),  # 1
                self._pressed_once.float().unsqueeze(-1),    # 1
                self._unlocked_once.float().unsqueeze(-1),   # 1
            ],
            dim=-1,
        )  # total = 30

        return {"policy": torch.clamp(obs, -5.0, 5.0)}

    def _get_rewards(self) -> torch.Tensor:
        press_pos, twist_pos = self._update_progress_flags()

        # Approach reward: guide EE toward estop cap
        ee_pos_w = self.ee_frame.data.target_pos_w[:, 0, :]
        estop_pos_w = self.estop.data.root_pos_w
        dist_to_estop = torch.norm(ee_pos_w - estop_pos_w, dim=-1)
        approach_reward = 1.0 / (dist_to_estop + 0.05)

        # Sequential progress rewards
        press_progress = torch.clamp(-press_pos / max(self.cfg.semantic_cfg.latched_position, 1e-6), 0.0, 1.0)
        twist_progress = torch.clamp(
            twist_pos / max(self.cfg.semantic_cfg.unlock_twist_angle_rad, 1e-6), 0.0, 1.0
        )
        release_progress = torch.clamp(1.0 + press_pos / max(self.cfg.release_tolerance, 1e-6), 0.0, 1.0)

        reward = 0.2 * approach_reward
        reward += 0.5 * press_progress
        reward += 0.5 * twist_progress * self._pressed_once.float()
        reward += release_progress * self._unlocked_once.float()
        reward += self.cfg.success_bonus * self._success.float()
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        if getattr(self.cfg, "disable_terminations", False):
            done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            return done, done

        self._update_progress_flags()
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return self._success.clone(), time_out
