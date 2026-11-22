# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
from kuafu.paths import OBJECT_ROOT

import copy
import math
import os

import torch
from collections.abc import Sequence

from isaacsim.core.utils.stage import get_current_stage
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics, Vt

from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from kuafu.isaac.base.manipulation.panda_manipulation_env import PandaManipulationEnv, PandaManipulationEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.utils import bind_physics_material
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import (
    FrameTransformer,
    FrameTransformerCfg,
)
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
ISAAC_NUCLEUS_DIR = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac"
from isaaclab.utils.math import sample_uniform
from isaaclab.utils import math as math_utils

@configclass
class CanCfg(PandaManipulationEnvCfg):
    """Configuration for a simple single-can manipulation environment.

    Robot: Franka Panda
    Object: One Panda pepper/coke can with random position & orientation per reset.
    """

    observation_space = 55

    # Scene settings
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4, env_spacing=5.0, replicate_physics=True
    )

    def __post_init__(self):
        super().__post_init__()
        self.sim.render_interval = self.decimation
        # Keep the shared PandaManipulationEnvCfg robot joint pose, matching Stack2.
        self.robot.init_state.pos = (0, 0.0, 0.0)


    # Rigid body properties for the Panda pepper can.
    can_properties = sim_utils.RigidBodyPropertiesCfg(
        solver_position_iteration_count=16,
        solver_velocity_iteration_count=1,
        max_angular_velocity=1000.0,
        max_linear_velocity=1000.0,
        max_depenetration_velocity=5.0,
        disable_gravity=False,
    )
    can_box_collision_props = sim_utils.CollisionPropertiesCfg(
        collision_enabled=True,
        contact_offset=0.001,
        rest_offset=0.001,
    )
    can_collision_cylinder_radius = 0.02
    can_collision_cylinder_height = 0.1

    can = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Can",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.1, 0.45, 0.05), rot=(0.0, 1.0, 0.0, 0.0)
        ),
        spawn=UsdFileCfg(
            usd_path=(
                str(OBJECT_ROOT / 'pepper/pepper.usdc')
            ),
            scale=(1.0, 1.0, 1.0),
            rigid_props=can_properties,
            mass_props=sim_utils.MassPropertiesCfg(mass=0.021),
            collision_props=None,
            semantic_tags=[("class", "can")],
        ),
    )

    # Frame at can body origin, matching the learned Panda can frame.
    marker_cfg_can = FRAME_MARKER_CFG.copy()
    marker_cfg_can.markers["frame"].scale = (0.1, 0.1, 0.1)
    marker_cfg_can.prim_path = "/Visuals/can_center_frame"
    can_center_frame = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/panda_link0",
        debug_vis=True,
        visualizer_cfg=marker_cfg_can,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Can",
                name="can_center",
                offset=OffsetCfg(pos=(0.0,  0.0, 0.0)),
            )
        ],
    )


class CanEnv(PandaManipulationEnv):
    cfg: CanCfg

    def __init__(
        self,
        cfg: CanCfg,
        render_mode: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

        # Task-specific tracking
        self.success_threshold = 0.95  # dot product threshold for uprightness
        self.stability_steps_required = 10  # consecutive steps needed for success
        self.consecutive_success_steps = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        print("[INFO]: CanEnv setup complete.")

    # ---------------------------------------------------------------------
    # Scene / reset
    # ---------------------------------------------------------------------
    def _setup_scene(self):  # noqa: D401
        from isaaclab.sim import schemas

        self.robot = Articulation(self.cfg.robot)

        self.ee_frame = FrameTransformer(self.cfg.ee_frame)
        self.l4_frame = FrameTransformer(self.cfg.l4_frame)
        self.can_center_frame = FrameTransformer(self.cfg.can_center_frame)

        # Spawn the Panda pepper can, then replace render-mesh collisions with
        # the same hidden octagonal cylinder collider used by the learned env.
        stage = get_current_stage()
        can_cfg = self.cfg.can
        can_material_cfg = sim_utils.materials.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
            friction_combine_mode="max",
        )
        can_material_path = "/World/Physics/canMaterial"
        can_material_cfg.func(can_material_path, can_material_cfg)

        if can_cfg.spawn is not None:
            spawn_cfg = copy.deepcopy(can_cfg.spawn)
            rigid_props = getattr(spawn_cfg, "rigid_props", None)
            mass_props = getattr(spawn_cfg, "mass_props", None)
            # The pepper USD does not expose the physics APIs expected by the
            # generic file spawner, so apply those properties manually after
            # the prim exists. This also keeps RigidObject from spawning twice.
            spawn_cfg.rigid_props = None
            spawn_cfg.mass_props = None
            spawn_cfg.collision_props = None
            spawn_cfg.func(
                can_cfg.prim_path,
                spawn_cfg,
                translation=can_cfg.init_state.pos,
                orientation=can_cfg.init_state.rot,
            )
            matching_prims = sim_utils.find_matching_prims(can_cfg.prim_path)
            for prim in matching_prims:
                if rigid_props is not None:
                    schemas.define_rigid_body_properties(
                        prim.GetPath().pathString, rigid_props
                    )
                if mass_props is not None:
                    schemas.define_mass_properties(
                        prim.GetPath().pathString, mass_props
                    )

                for child_prim in Usd.PrimRange(prim):
                    if child_prim.IsA(UsdGeom.Mesh):
                        if child_prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                            child_prim.RemoveAPI(UsdPhysics.MeshCollisionAPI)
                        if child_prim.HasAPI(UsdPhysics.CollisionAPI):
                            child_prim.RemoveAPI(UsdPhysics.CollisionAPI)
                        if child_prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
                            child_prim.RemoveAPI(PhysxSchema.PhysxCollisionAPI)

                center_frame_path = f"{prim.GetPath().pathString}/can_center_frame"
                if not stage.GetPrimAtPath(center_frame_path).IsValid():
                    UsdGeom.Xform.Define(stage, center_frame_path)

                collider_path = f"{center_frame_path}/collision_cylinder"
                if not stage.GetPrimAtPath(collider_path).IsValid():
                    n = 8
                    r = self.cfg.can_collision_cylinder_radius
                    h = self.cfg.can_collision_cylinder_height
                    pts, fvc, fvi = [], [], []
                    for ring_y in (-h / 2.0, h / 2.0):
                        for i in range(n):
                            a = 2.0 * math.pi * i / n
                            pts.append(Gf.Vec3f(r * math.cos(a), ring_y, r * math.sin(a)))
                    fvc.append(n)
                    fvi.extend(range(n - 1, -1, -1))
                    fvc.append(n)
                    fvi.extend(range(n, 2 * n))
                    for i in range(n):
                        j = (i + 1) % n
                        fvc.append(4)
                        fvi.extend([i, j, j + n, i + n])
                    collider = UsdGeom.Mesh.Define(stage, collider_path)
                    collider.CreatePointsAttr(Vt.Vec3fArray(pts))
                    collider.CreateFaceVertexCountsAttr(Vt.IntArray(fvc))
                    collider.CreateFaceVertexIndicesAttr(Vt.IntArray(fvi))
                    mesh_api = UsdPhysics.MeshCollisionAPI.Apply(collider.GetPrim())
                    mesh_api.CreateApproximationAttr("convexHull")
                    UsdGeom.Imageable(collider).MakeInvisible()

                schemas.define_collision_properties(
                    collider_path, self.cfg.can_box_collision_props, stage
                )
                bind_physics_material(collider_path, can_material_path, stage=stage)

        can_asset_cfg = self.cfg.can.copy()
        can_asset_cfg.spawn = None
        self.can = RigidObject(can_asset_cfg)

        # add to scene registries
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["can"] = self.can
        # Register camera only if enabled
        self.scene.sensors["panda_fingertip"] = self.ee_frame
        self.scene.sensors["panda_link4"] = self.l4_frame
        self.scene.sensors["can_center_frame"] = self.can_center_frame

        # terrain per envs
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=True)
        
        # Filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # Lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=1000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        super()._reset_idx(env_ids)

        # Random can position (XY within workspace) & TOPPLED orientation
        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        can_root = self.can.data.default_root_state.clone().index_select(0, env_ids_t)

        # Workspace sampling: x in [0.4, 0.7], y in [-0.2, 0.2]
        if getattr(self.cfg, "disable_reset_randomization", False):
            rand_xy = torch.tensor([[0.5, 0.1]], device=self.device).repeat(len(env_ids), 1)
        else:
            rand_xy = sample_uniform(
                torch.tensor([0.5, -0.05], device=self.device),
                torch.tensor([0.05, 0.05], device=self.device),
                (len(env_ids), 2),
                device=self.device,
            )
        env_origins = self.scene.env_origins.index_select(0, env_ids_t)
        can_root[:, 0] = env_origins[:, 0] + rand_xy[:, 0]
        can_root[:, 1] = env_origins[:, 1] + rand_xy[:, 1]
        can_root[:, 2] = env_origins[:, 2] + 0.0203
        # Keep initial orientation from config (default_root_state), only randomize position.

        self.can.write_root_pose_to_sim(can_root[:, :7], env_ids=env_ids)
        self.can.write_root_velocity_to_sim(torch.zeros((len(env_ids), 6), device=self.device), env_ids=env_ids)
        self.can.reset(env_ids)

        self._reset_robot(env_ids_t)

    # ---------------------------------------------------------------------
    # RL API
    # ---------------------------------------------------------------------
    def _get_observations(self) -> dict:
        joint_pos = self.robot.data.joint_pos
        joint_vel = self.robot.data.joint_vel
        can_pose = self.can.data.root_state_w[:, :7]  # (pos xyz, quat wxyz)
        can_vel = self.can.data.root_state_w[:, 7:]  # (lin vel xyz, ang vel xyz)
        
        # Compute uprightness: can's y-axis alignment with world Z-axis
        # Manually extract y-axis from quaternion (w, x, y, z)
        can_quat = can_pose[:, 3:7]  # (w, x, y, z)
        w, x, y, z = can_quat[:, 0], can_quat[:, 1], can_quat[:, 2], can_quat[:, 3]
        
        # Rotated y-axis = [2*(xy - wz), 1 - 2*(x^2 + z^2), 2*(yz + wx)]
        can_y_axis_z = 2.0 * (y * z + w * x)  # z-component of rotated y-axis
        uprightness = can_y_axis_z  # dot product with world Z = [0, 0, 1]
        
        return {
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "can_pose": can_pose,
            "can_velocity": can_vel
        }

    def _apply_action(self) -> None:
        super()._apply_action()

    def _pre_physics_step(self, actions: torch.Tensor):
        super()._pre_physics_step(actions)

    def _get_rewards(self) -> torch.Tensor:
        # Multi-component reward for righting the can
        can_pose = self.can.data.root_state_w[:, :7]
        can_vel = self.can.data.root_state_w[:, 7:]
        hand_pos = self.robot.data.body_pos_w[:, self.hand_link_idx]
        can_pos = can_pose[:, :3]
        can_quat = can_pose[:, 3:7]
        
        # 1. Uprightness reward: maximize alignment of can's y-axis with world Z
        # Manually extract y-axis z-component from quaternion
        w, x, y, z = can_quat[:, 0], can_quat[:, 1], can_quat[:, 2], can_quat[:, 3]
        uprightness = 2.0 * (y * z + w * x)  # z-component of rotated y-axis
        uprightness_reward = torch.where(
            uprightness > 0,
            uprightness ** 2,  # quadratic for positive values
            torch.zeros_like(uprightness)  # zero for pointing downward
        )
        
        # 2. Stability reward: penalize movement when can is upright
        lin_vel = can_vel[:, :3]
        ang_vel = can_vel[:, 3:]
        lin_speed = torch.norm(lin_vel, dim=1)
        ang_speed = torch.norm(ang_vel, dim=1)
        stability_bonus = torch.where(
            uprightness > 0.9,
            torch.exp(-2.0 * (lin_speed + ang_speed)),  # high reward for being still when upright
            torch.zeros_like(uprightness)
        )
        
        # 3. Success bonus: large bonus for being stably upright
        success_bonus = torch.where(
            (uprightness > self.success_threshold) & (lin_speed < 0.05) & (ang_speed < 0.1),
            10.0,
            0.0
        )
        
        # 4. Proximity reward (smaller weight): encourage manipulation
        dist = torch.norm(hand_pos - can_pos, dim=1)
        proximity_reward = 0.1 / (1.0 + dist)
        
        # Total reward
        reward = uprightness_reward * 2.0 + stability_bonus * 3.0 + success_bonus + proximity_reward
        
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        if getattr(self.cfg, "disable_terminations", False):
            done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            return done, done

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        # Success detection: can is upright and stable
        can_pose = self.can.data.root_state_w[:, :7]
        can_vel = self.can.data.root_state_w[:, 7:]
        can_quat = can_pose[:, 3:7]
        
        # Compute uprightness: manually extract y-axis z-component from quaternion
        w, x, y, z = can_quat[:, 0], can_quat[:, 1], can_quat[:, 2], can_quat[:, 3]
        uprightness = 2.0 * (y * z + w * x)  # z-component of rotated y-axis
        
        # Check stability
        lin_speed = torch.norm(can_vel[:, :3], dim=1)
        ang_speed = torch.norm(can_vel[:, 3:], dim=1)
        
        is_stable_upright = (
            (uprightness > self.success_threshold) &
            (lin_speed < 0.05) &
            (ang_speed < 0.1)
        )
        
        # Update consecutive success counter
        self.consecutive_success_steps = torch.where(
            is_stable_upright,
            self.consecutive_success_steps + 1,
            torch.zeros_like(self.consecutive_success_steps)
        )
        
        # Terminate successfully if stable for required steps
        terminated = self.consecutive_success_steps >= self.stability_steps_required
        
        return terminated, time_out

    def close(self):  # noqa: D401
        super().close()
