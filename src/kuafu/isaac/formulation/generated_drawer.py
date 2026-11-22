"""Program-free Franka scene using an exported articulated cabinet and its semantics.

The legacy drawer scene replaces frame collisions and assumes a separate handle
body. Generated assets instead retain their own collision geometry and register
their portable body-offset sensors. No learned program is loaded by this scene.
"""

from __future__ import annotations

import json
import math
import runpy
from pathlib import Path

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.utils.math import create_rotation_matrix_from_view, quat_from_matrix

from kuafu.asset_generation.semantic_runtime import attach_semantics
from kuafu.isaac.formulation.panda import PandaDrawerFormulationEnv
from kuafu.isaac.learned.panda.panda_drawer_env_learned import PandaDrawerEnvLearnedCfg
from kuafu.isaac.learned.panda.panda_table_env_learned import PandaTableEnvLearned


def generated_drawer_cfg(asset_dir, device="cuda:0", *, position=(0.6, -0.5, 0.2), yaw_deg=180.0, cameras=False):
    """Use either pipeline's exported ArticulationCfg at a reproducible reachable pose."""
    asset_dir = Path(asset_dir).expanduser().resolve()
    semantics = json.loads((asset_dir / "semantics.json").read_text())
    required = {"top_drawer_handle_frame", "top_drawer_center_frame"}
    if semantics.get("status") != "annotated" or not required <= semantics.get("frames", {}).keys():
        raise ValueError("Drawer scene requires annotated top-drawer handle and center frames")
    api = runpy.run_path(str(asset_dir / "isaaclab_snippet.py"))
    if "ARTICULATED_OBJECT" not in api:
        raise ValueError("Drawer scene requires the exported articulated configuration")
    cfg = PandaDrawerEnvLearnedCfg()
    cfg.generated_asset_dir = str(asset_dir)
    cfg.sim.device = device
    cfg.scene.num_envs = 1
    cfg.seed = 0
    cfg.enable_cameras = cameras
    if cameras:
        size = semantics["object"]["size_m"]
        target = torch.tensor([[position[0], position[1], position[2] + size[2] / 2]])
        yaw = math.radians(yaw_deg)
        # Orbit the cabinet's front so its handles remain visible at other poses.
        for index, (dx, dy, dz) in enumerate(((-0.65, -0.85, 0.65), (0.65, -0.85, 0.65), (-0.15, -0.65, 0.45))):
            eye = (position[0] + math.cos(yaw) * dx - math.sin(yaw) * dy,
                   position[1] + math.sin(yaw) * dx + math.cos(yaw) * dy,
                   position[2] + dz)
            rotation = create_rotation_matrix_from_view(torch.tensor([eye]), target, "Z", device="cpu")
            camera = getattr(cfg, f"table_cam_{index}")
            camera.offset.pos = eye
            camera.offset.rot = tuple(quat_from_matrix(rotation)[0].tolist())
            camera.offset.convention = "opengl"
    cfg.robot_base_height_range = (0.0, 0.0)
    cfg.random_object_spawn_height_range = (0.0, 0.0)
    cfg.cabinet = api["ARTICULATED_OBJECT"].copy()
    cfg.cabinet.prim_path = "/World/envs/env_.*/Cabinet"
    cfg.cabinet.init_state.pos = tuple(position)
    yaw = math.radians(yaw_deg)
    cfg.cabinet.init_state.rot = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
    cfg.detectable_objects = {
        name: description for name, description in cfg.detectable_objects.items() if name.startswith("panda_")
    }
    cfg.task_description = "open the top drawer by 10 cm"
    return cfg


class GeneratedDrawerFormulationEnv(PandaDrawerFormulationEnv):
    """Retain robot/observation plumbing while using the actual exported cabinet."""

    def _setup_scene(self):
        # Bypass the old cabinet's collision-replacement and handle-body assumptions.
        PandaTableEnvLearned._setup_scene(self)
        x, y, height = self.cfg.cabinet.init_state.pos
        if height > 0.0:
            size = json.loads((Path(self.cfg.generated_asset_dir) / "semantics.json").read_text())["object"]["size_m"]
            support = RigidObjectCfg(
                prim_path="/World/envs/env_.*/CabinetSupport",
                spawn=sim_utils.CuboidCfg(
                    size=(size[0], size[1], height),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.33, 0.36)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(x, y, height / 2),
                    rot=self.cfg.cabinet.init_state.rot,
                ),
            )
            self.scene.rigid_objects["cabinet_support"] = RigidObject(support)
        self._cabinet = Articulation(self.cfg.cabinet)
        self.scene.articulations["cabinet"] = self._cabinet
        sensors = attach_semantics(
            self,
            "cabinet",
            self.cfg.ee_frame.prim_path,
            asset_dir=self.cfg.generated_asset_dir,
        )
        for name, sensor in sensors.items():
            setattr(self, name, sensor)
        self.scene.clone_environments(copy_from_source=False)
        self._hide_static_surface_visual_prims()
        light = sim_utils.DomeLightCfg(intensity=1200.0, color=(0.75, 0.75, 0.75))
        light.func("/World/Light", light)

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self._prepare_support_surface_reset(env_ids)
        root = self._cabinet.data.default_root_state[env_ids].clone()
        root[:, :3] += self.scene.env_origins[env_ids]
        self._cabinet.write_root_pose_to_sim(root[:, :7], env_ids=env_ids)
        self._cabinet.write_root_velocity_to_sim(torch.zeros_like(root[:, 7:]), env_ids=env_ids)
        zeros = torch.zeros((len(env_ids), self._cabinet.num_joints), device=self.device)
        self._cabinet.write_joint_state_to_sim(zeros, zeros, env_ids=env_ids)
        self._cabinet.reset(env_ids)
        PandaTableEnvLearned._reset_idx(self, env_ids)
