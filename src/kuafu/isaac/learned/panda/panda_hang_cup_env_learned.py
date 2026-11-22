from __future__ import annotations
from kuafu.paths import OBJECT_ROOT

import os

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer import FrameTransformer
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim import schemas
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils import math as math_utils
from isaaclab.utils.math import quat_from_euler_xyz
from isaacsim.core.utils.stage import get_current_stage
from pxr import UsdPhysics

from kuafu.mpc.simulation.spatial_utils import quat_to_rotation_6d
from .panda_table_env_learned import (
    PandaTableEnvLearned,
    PandaTableEnvLearnedCfg,
)
from kuafu.isaac.tasks.collision_utils import (
    replace_mesh_collision_with_mug_cylinders,
    replace_mesh_collision_with_simple_hanger_cylinders,
)
from kuafu.isaac.tasks.env_utils import random_pose_radial


@configclass
class PandaHangCupEnvLearnedCfg(PandaTableEnvLearnedCfg):
    observation_space = 31
    obs_mode: str = "state"
    obs_dims = {"state": 31, "full_state": 65}
    episode_length_s = 20
    success_hold_frames: int = 15
    success_handle_checkpoint_threshold: float = 0.03
    success_hanger_checkpoint_offset = (0.0, -0.0393969331, 0.1584201825)
    success_fingertip_center_min_distance: float = 0.1
    success_gripper_open_threshold: float = 0.75
    random_object_spawn_height_range: tuple[float, float] = (-0.02, 0.02)
    spawn_angle_lower_deg: float = 0.0
    spawn_angle_upper_deg: float = 90.0
    randomize_left_right: bool = True
    fixed_left_right_swap: bool = False

        # Override robot init_state only, keep rest from base
    def __post_init__(self):
        """Post-initialization to override robot initial state."""
        super().__post_init__()
        self.sim.physx.gpu_collision_stack_size = 2**29
        self.sim.physx.gpu_max_rigid_contact_count = 2**24
        self.sim.physx.gpu_max_rigid_patch_count = 2**22
        self.sim.physx.gpu_found_lost_pairs_capacity = 2**24
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**27
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2**23
        self.sim.physx.gpu_heap_capacity = 2**28
        self.sim.physx.gpu_temp_buffer_capacity = 2**26
        # Override only the initial state, keeping the rest of robot config from parent
        self.robot.init_state = ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 1.0,
                "panda_joint2": -0.569,
                "panda_joint3": 0.0,
                "panda_joint4": -2.810,
                "panda_joint5": 0.0,
                "panda_joint6": 3.037,
                "panda_joint7": 0.741,
                "panda_finger_joint2": 0,
                "panda_finger_joint1": 0,
            },
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        )

    mug_properties = sim_utils.RigidBodyPropertiesCfg(
        solver_position_iteration_count=32,
        solver_velocity_iteration_count=2,
        max_angular_velocity=1000.0,
        max_linear_velocity=1000.0,
        max_depenetration_velocity=100.0,
        disable_gravity=False,
    )

    _mug_path = str(OBJECT_ROOT / 'large_mug/large_mug.usdc')

    mug_physics_material = sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=0.5,
        dynamic_friction=0.3,
        restitution=0.0,
    )

    mug = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Mug",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.4, 0.2, 0.01), rot=(0, 0, 0, 1)
        ),
        spawn=UsdFileCfg(
            usd_path=_mug_path,
            scale=(1, 1, 1),
            rigid_props=mug_properties,
            mass_props=sim_utils.MassPropertiesCfg(mass=0.01),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.005,
                rest_offset=0.003,
            ),
            semantic_tags=[("class", "mug")],
        ),
    )

    _cup_hanger_path = str(OBJECT_ROOT / 'simple_hanger/assembly1_white_plastic_cyl_collision_static.usda')

    cup_hanger = RigidObjectCfg(
        prim_path="/World/envs/env_.*/CupHolder",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.4, -0.2, 0.0), rot=(1, 0, 0, 0)
        ),
        spawn=UsdFileCfg(
            usd_path=_cup_hanger_path,
            scale=(1, 1, 1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.0001,
                rest_offset=0.0,
            ),
        ),
    )

    cup_hanger_physics_material = sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=0.005,
        dynamic_friction=0.005,
        restitution=0.0,
    )

    marker_cfg_mug_handle_hole = FRAME_MARKER_CFG.copy()
    marker_cfg_mug_handle_hole.markers["frame"].scale = (0.05, 0.05, 0.05)
    marker_cfg_mug_handle_hole.prim_path = "/Visuals/mug_handle_hole_frame"
    mug_handle_hole_frame = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/panda_link0",
        debug_vis=False,
        visualizer_cfg=marker_cfg_mug_handle_hole,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Mug",
                name="Mug_handle_hole",
                offset=OffsetCfg(pos=(0.06, 0.00, 0.05)),
            )
        ],
    )

    marker_cfg_mug_center = FRAME_MARKER_CFG.copy()
    marker_cfg_mug_center.markers["frame"].scale = (0.05, 0.05, 0.05)
    marker_cfg_mug_center.prim_path = "/Visuals/mug_center_frame"
    mug_center_frame = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/panda_link0",
        debug_vis=False,
        visualizer_cfg=marker_cfg_mug_center,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Mug",
                name="Mug_center",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.04)),
            )
        ],
    )

    marker_cfg_mug_rim_frame_0 = FRAME_MARKER_CFG.copy()
    marker_cfg_mug_rim_frame_0.markers["frame"].scale = (0.05, 0.05, 0.05)
    marker_cfg_mug_rim_frame_0.prim_path = "/Visuals/mug_rim_frame_0"
    mug_rim_frame_0 = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/panda_link0",
        debug_vis=False,
        visualizer_cfg=marker_cfg_mug_rim_frame_0,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Mug",
                name="mug_rim_frame_0",
                offset=OffsetCfg(pos=(0.0, 0.04, 0.06)),
            )
        ],
    )

    marker_cfg_mug_rim_frame_1 = FRAME_MARKER_CFG.copy()
    marker_cfg_mug_rim_frame_1.markers["frame"].scale = (0.05, 0.05, 0.05)
    marker_cfg_mug_rim_frame_1.prim_path = "/Visuals/mug_rim_frame_1"
    mug_rim_frame_1 = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/panda_link0",
        debug_vis=False,
        visualizer_cfg=marker_cfg_mug_rim_frame_1,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Mug",
                name="mug_rim_frame_1",
                offset=OffsetCfg(pos=(0.0, -0.04, 0.06)),
            )
        ],
    )

    marker_cfg_cup_hanger_branch_1 = FRAME_MARKER_CFG.copy()
    marker_cfg_cup_hanger_branch_1.markers["frame"].scale = (0.05, 0.05, 0.05)
    marker_cfg_cup_hanger_branch_1.prim_path = "/Visuals/cup_hanger_branch_1_tip_frame"
    cup_hanger_branch_1_tip_frame = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/panda_link0",
        debug_vis=False,
        visualizer_cfg=marker_cfg_cup_hanger_branch_1,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/CupHolder",
                name="CupHolder_branch_1",
                offset=OffsetCfg(
                    # Keep the public branch_1 target name for existing program.
                    # The simple hanger has one branch: local +Y is base->tip.
                    pos=(0.0, -0.11276295, 0.2060421),
                    rot=(0.173648178, 0.984807753, 0.0, 0.0),
                ),
            )
        ],
    )

    marker_cfg_cup_hanger_checkpoint = FRAME_MARKER_CFG.copy()
    marker_cfg_cup_hanger_checkpoint.markers["frame"].scale = (0.06, 0.06, 0.06)
    marker_cfg_cup_hanger_checkpoint.prim_path = "/Visuals/cup_hanger_checkpoint_frame"
    cup_hanger_checkpoint_frame = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/panda_link0",
        debug_vis=True,
        visualizer_cfg=marker_cfg_cup_hanger_checkpoint,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/CupHolder",
                name="CupHolder_checkpoint",
                offset=OffsetCfg(pos=(0.0, -0.0393969331, 0.1584201825)),
            )
        ],
    )

    detectable_objects = {
        'mug_center_frame': (
            'center frame of mug; mug body is too wide for the gripper; use virtual handles instead; size: (0.13, 0.1, 0.1)',
            '/World/envs/env_*/mug_center_frame',
        ),
        'mug_rim_frame_0': (
            'rim frame 0 of mug, virtual handle frame, virtual handle axis is long local + x or -x, virtual approach is local + z, graspable depth is 5 cm',
            '/World/envs/env_*/mug_rim_frame_0',
        ),
        'mug_rim_frame_1': (
            'rim frame 1 of mug, virtual handle frame, virtual handle axis is long local + x or -x, virtual approach is local + z, graspable depth is 5 cm',
            '/World/envs/env_*/mug_rim_frame_1',
        ),
        'mug_handle_hole_frame': (
            'center of handle opening of mug; handle axis -x; hole/opening axis is ±y (bidirectional); too wide for the gripper; not a graspable frame',
            '/Visuals/mug_handle_hole_frame',
        ),
        'cup_hanger': (
            'base of cup holder, static single-branch hanger; base diameter 0.16 m, pole height 0.185 m, one tilted branch only',
            '/World/envs/env_*/CupHolder',
        ),
        'cup_hanger_branch_1_tip_frame': (
            'tip frame of the only hanger branch; branch axis is +y (base->tip, outward toward local -Y and upward 20 deg); engagement/hooking moves along -y (tip->base, inward), exposed length is 0.110 m, diameter is 0.012 m',
            '/Visuals/cup_hanger_branch_1_tip_frame',
        ),
        'table': (
            'center frame of table, size: (0.3, 0.9, 0.4)',
            '/World/envs/env_*/Table',
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

    _kuafu_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    _repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(_kuafu_dir)))
    mpc_config = {
        "controller_name": "FrankaIKController",
        "device": "cuda",
        "ee_frame": "panda_fingertip",
        "use_per_env_weight": False,
        "load_program": f"{_repo_dir}/resources/learned_programs/manipulation-hang-cup/2026-04-07_21-16-24",
        "initial_residual": f"{_repo_dir}/src/kuafu/skills/constraints/rewards.py",
        "initial_predicate": f"{_repo_dir}/src/kuafu/skills/constraints/predicates.py",
        "gripper_opened": [0.04, 0.04],
        "gripper_closed": [0.0, 0.0],
        "gripper_max_force": 40.0,
        "has_finger_contact_sensors": False,
    }
    del _repo_dir

    task_description = 'hook the mug handle through the single cup hanger branch'
class PandaHangCupEnvLearned(PandaTableEnvLearned):
    cfg: PandaHangCupEnvLearnedCfg

    def __init__(
        self,
        cfg: PandaHangCupEnvLearnedCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        super().__init__(cfg, render_mode, **kwargs)
        self._mug_done_hold_frames = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.int32
        )

    def _setup_scene(self):
        super()._setup_scene()

        stage = get_current_stage()
        mug_cfg = self.cfg.mug
        if mug_cfg.spawn is not None:
            mug_cfg.spawn.func(
                mug_cfg.prim_path,
                mug_cfg.spawn,
                translation=mug_cfg.init_state.pos,
                orientation=mug_cfg.init_state.rot,
            )
        if mug_cfg.spawn is not None:
            matching_prims = sim_utils.find_matching_prims(mug_cfg.prim_path)
            for prim in matching_prims:
                if hasattr(mug_cfg.spawn, "rigid_props") and mug_cfg.spawn.rigid_props is not None:
                    schemas.define_rigid_body_properties(
                        prim.GetPath().pathString, mug_cfg.spawn.rigid_props
                    )
                if hasattr(mug_cfg.spawn, "mass_props") and mug_cfg.spawn.mass_props is not None:
                    schemas.define_mass_properties(
                        prim.GetPath().pathString, mug_cfg.spawn.mass_props
                    )

                material_path = f"{prim.GetPath().pathString}/physicsMaterial"
                self.cfg.mug_physics_material.func(material_path, self.cfg.mug_physics_material)

                replace_mesh_collision_with_mug_cylinders(
                    prim,
                    stage=stage,
                    collision_props=getattr(mug_cfg.spawn, "collision_props", None),
                    material_path=material_path,
                )

        cup_hanger_cfg = self.cfg.cup_hanger
        if cup_hanger_cfg.spawn is not None:
            cup_hanger_cfg.spawn.func(
                cup_hanger_cfg.prim_path,
                cup_hanger_cfg.spawn,
                translation=cup_hanger_cfg.init_state.pos,
                orientation=cup_hanger_cfg.init_state.rot,
            )
        if cup_hanger_cfg.spawn is not None:
            matching_prims = sim_utils.find_matching_prims(cup_hanger_cfg.prim_path)
            for prim in matching_prims:
                if not UsdPhysics.RigidBodyAPI(prim):
                    if (
                        hasattr(cup_hanger_cfg.spawn, "rigid_props")
                        and cup_hanger_cfg.spawn.rigid_props is not None
                    ):
                        schemas.define_rigid_body_properties(
                            prim.GetPath().pathString, cup_hanger_cfg.spawn.rigid_props
                        )
                if (
                    hasattr(cup_hanger_cfg.spawn, "mass_props")
                    and cup_hanger_cfg.spawn.mass_props is not None
                ):
                    schemas.define_mass_properties(
                        prim.GetPath().pathString, cup_hanger_cfg.spawn.mass_props
                    )

                material_path = f"{prim.GetPath().pathString}/physicsMaterial"
                self.cfg.cup_hanger_physics_material.func(
                    material_path, self.cfg.cup_hanger_physics_material
                )

                replace_mesh_collision_with_simple_hanger_cylinders(
                    prim,
                    stage=stage,
                    collision_props=getattr(cup_hanger_cfg.spawn, "collision_props", None),
                    material_path=material_path,
                )

        self.mug = RigidObject(self.cfg.mug)
        self.cup_hanger = RigidObject(self.cfg.cup_hanger)

        self.mug_handle_hole_frame = FrameTransformer(self.cfg.mug_handle_hole_frame)
        self.mug_center_frame = FrameTransformer(self.cfg.mug_center_frame)
        self.mug_rim_frame_0 = FrameTransformer(self.cfg.mug_rim_frame_0)
        self.mug_rim_frame_1 = FrameTransformer(self.cfg.mug_rim_frame_1)
        self.cup_hanger_branch_1_tip_frame = FrameTransformer(
            self.cfg.cup_hanger_branch_1_tip_frame
        )
        self.cup_hanger_checkpoint_frame = FrameTransformer(
            self.cfg.cup_hanger_checkpoint_frame
        )

        self.scene.rigid_objects["mug"] = self.mug
        self.scene.rigid_objects["cup_hanger"] = self.cup_hanger
        self.scene.sensors["mug_handle_hole_frame"] = self.mug_handle_hole_frame
        self.scene.sensors["mug_center_frame"] = self.mug_center_frame
        self.scene.sensors["cup_hanger_branch_1_tip_frame"] = self.cup_hanger_branch_1_tip_frame
        self.scene.sensors["cup_hanger_checkpoint_frame"] = self.cup_hanger_checkpoint_frame
        self.scene.sensors["mug_rim_frame_0"] = self.mug_rim_frame_0
        self.scene.sensors["mug_rim_frame_1"] = self.mug_rim_frame_1

        self.scene.clone_environments(copy_from_source=True)
        self._hide_static_surface_visual_prims()
        if self.device == "cpu" and self.cfg.terrain is not None:
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        n = len(env_ids)
        env_ids_tensor = env_ids.to(device=self.device, dtype=torch.long)
        self._mug_done_hold_frames[env_ids_tensor] = 0
        self._prepare_support_surface_reset(env_ids_tensor)
        origins = self._surface_origins(env_ids_tensor)

        right_angle_min, right_angle_max = self._spawn_angle_range(0, 2)
        left_angle_min, left_angle_max = self._spawn_angle_range(1, 2)
        middle_gap_half = 5.0 * torch.pi / 180.0
        right_angle_max -= middle_gap_half
        left_angle_min += middle_gap_half

        mug_pos_right, _ = random_pose_radial(
            0.01,
            dist_min=0.3,
            dist_max=0.5,
            angle_min_rad=right_angle_min,
            angle_max_rad=right_angle_max,
            n=n,
            device=self.device,
            origins=origins,
            rotate_randomly=True,
        )
        mug_pos_left, _ = random_pose_radial(
            0.01,
            dist_min=0.3,
            dist_max=0.5,
            angle_min_rad=left_angle_min,
            angle_max_rad=left_angle_max,
            n=n,
            device=self.device,
            origins=origins,
            rotate_randomly=True,
        )
        hanger_pos_right, _ = random_pose_radial(
            0.0,
            dist_min=0.5,
            dist_max=0.7,
            angle_min_rad=right_angle_min,
            angle_max_rad=right_angle_max,
            n=n,
            device=self.device,
            origins=origins,
            rotate_randomly=True,
        )
        hanger_pos_left, _ = random_pose_radial(
            0.0,
            dist_min=0.5,
            dist_max=0.7,
            angle_min_rad=left_angle_min,
            angle_max_rad=left_angle_max,
            n=n,
            device=self.device,
            origins=origins,
            rotate_randomly=True,
        )

        if self.cfg.randomize_left_right:
            swap = torch.rand(n, device=self.device) > 0.5
        else:
            swap = torch.full(
                (n,), bool(self.cfg.fixed_left_right_swap), dtype=torch.bool, device=self.device
            )
        swap_2 = swap.unsqueeze(1).expand(-1, 2)

        mug_xy = torch.where(swap_2, mug_pos_left[:, :2], mug_pos_right[:, :2])
        hanger_xy = torch.where(
            swap_2, hanger_pos_right[:, :2], hanger_pos_left[:, :2]
        )
        hanger_xy = hanger_xy + (torch.rand((n, 2), device=self.device) * 0.1 - 0.05)
        mug_pos = torch.cat([mug_xy, self._spawn_z_above_surface(env_ids_tensor, 0.01)], dim=1)
        hanger_height = self._sample_object_spawn_height_offset((n, 1))
        hanger_pos = torch.cat([hanger_xy, self._spawn_z_above_surface(env_ids_tensor, hanger_height)], dim=1)

        zeros = torch.zeros(n, device=self.device)
        mug_yaw = torch.rand(n, device=self.device) * 2.0 * torch.pi
        mug_quat = quat_from_euler_xyz(zeros, zeros, mug_yaw)

        to_mug_xy = mug_pos[:, :2] - hanger_pos[:, :2]
        facing_yaw = torch.atan2(to_mug_xy[:, 1], to_mug_xy[:, 0])
        yaw_wiggle = (
            (torch.rand(n, device=self.device) * 2.0 - 1.0) * (torch.pi / 6)
        )
        # The single branch is authored at local azimuth -90 deg.
        hanger_yaw = facing_yaw + torch.pi / 2.0 + yaw_wiggle
        hanger_quat = quat_from_euler_xyz(zeros, zeros, hanger_yaw)

        max_tilt_rad = 10.0 * torch.pi / 180.0
        tilt_angle = torch.rand(n, device=self.device) * max_tilt_rad
        tilt_azimuth = torch.rand(n, device=self.device) * 2.0 * torch.pi
        tilt_axis = torch.stack(
            [
                torch.cos(tilt_azimuth),
                torch.sin(tilt_azimuth),
                zeros,
            ],
            dim=1,
        )
        tilt_quat = math_utils.quat_from_angle_axis(tilt_angle, tilt_axis)
        hanger_quat = math_utils.quat_mul(tilt_quat, hanger_quat)

        if os.getenv("PANDA_RESET_DEBUG", "").strip().lower() in (
            "1",
            "true",
            "yes",
        ):
            print(
                "[PandaHangReset] "
                f"env_ids={env_ids_tensor.detach().cpu().tolist()} "
                f"swap_left_mug={swap.detach().cpu().tolist()} "
                f"mug_pos={mug_pos.detach().cpu().tolist()} "
                f"mug_yaw={mug_yaw.detach().cpu().tolist()} "
                f"hanger_pos={hanger_pos.detach().cpu().tolist()} "
                f"hanger_yaw={hanger_yaw.detach().cpu().tolist()} "
                f"hanger_tilt={tilt_angle.detach().cpu().tolist()}"
            )

        root_m = self.mug.data.default_root_state.clone().index_select(0, env_ids_tensor)
        root_m[:, :3] = mug_pos
        root_m[:, 3:7] = mug_quat

        self.mug.write_root_pose_to_sim(root_m[:, :7], env_ids=env_ids)
        self.mug.write_root_velocity_to_sim(
            torch.zeros((n, 6), device=self.device), env_ids=env_ids
        )
        self.mug.reset(env_ids)

        root_h = self.cup_hanger.data.default_root_state.clone().index_select(
            0, env_ids_tensor
        )
        root_h[:, :3] = hanger_pos
        root_h[:, 3:7] = hanger_quat
        self.cup_hanger.write_root_pose_to_sim(root_h[:, :7], env_ids=env_ids)
        self.cup_hanger.write_root_velocity_to_sim(
            torch.zeros((n, 6), device=self.device), env_ids=env_ids
        )
        self.cup_hanger.reset(env_ids)

        super()._reset_idx(env_ids)

    def _get_observations(self) -> dict:
        self.update_data_in_controller()

        api = self.program_runtime.controller_api_wrapper
        agent = api.reference_agent
        if agent is None:
            raise RuntimeError("controller_api_wrapper.reference_agent is not set")

        mug_pos = api.get_object_xyz_position("mug_center_frame")
        hanger_pos = api.get_object_xyz_position("cup_hanger_branch_1_tip_frame")
        fingertip_pos = api.lookup_robot_frame_position("panda_fingertip")
        link4_pos = api.lookup_robot_frame_position("panda_link4")

        mug_rot6d = quat_to_rotation_6d(api.get_object_wxyz_quaternion("mug_center_frame"))
        hanger_rot6d = quat_to_rotation_6d(
            api.get_object_wxyz_quaternion("cup_hanger_branch_1_tip_frame")
        )
        fingertip_rot6d = quat_to_rotation_6d(api.lookup_robot_frame_rotation("panda_fingertip"))
        link4_rot6d = quat_to_rotation_6d(api.lookup_robot_frame_rotation("panda_link4"))

        joint_pos = api.joint_dict[agent]["joint_position"]
        joint_vel = api.joint_dict[agent]["joint_velocity"]
        joint_effort = self._get_normalized_joint_effort(api, agent)
        gripper_contact = self._get_normalized_gripper_contact(api)
        joint_effort = torch.cat([joint_effort[:, :7], gripper_contact], dim=-1)
        gripper_open = api.gripper_openness().unsqueeze(-1)
        stage = self.current_stage

        def _f(*tensors: torch.Tensor) -> torch.Tensor:
            return torch.cat(
                [t.to(device=self.device, dtype=torch.float32) for t in tensors], dim=-1
            )

        policy_obs = _f(
            mug_pos,
            hanger_pos,
            fingertip_pos,
            mug_rot6d,
            hanger_rot6d,
            fingertip_rot6d,
            gripper_open,
            gripper_contact,
            stage,
        )

        full_state_obs = _f(
            mug_pos,
            hanger_pos,
            fingertip_pos,
            link4_pos,
            mug_rot6d,
            hanger_rot6d,
            fingertip_rot6d,
            link4_rot6d,
            joint_pos,
            joint_vel,
            joint_effort,
            gripper_open,
            stage,
        )

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

        mug_center_pos = self.mug_center_frame.data.target_pos_w.squeeze(1)
        mug_handle_hole_pos = self.mug_handle_hole_frame.data.target_pos_w.squeeze(1)
        fingertip_pos = self.ee_frame.data.target_pos_w.squeeze(1)
        rim_0_pos = self.mug_rim_frame_0.data.target_pos_w.squeeze(1)
        rim_1_pos = self.mug_rim_frame_1.data.target_pos_w.squeeze(1)
        hanger_checkpoint_pos = self.cup_hanger_checkpoint_frame.data.target_pos_w.squeeze(1)

        handle_dist = torch.linalg.norm(mug_handle_hole_pos - hanger_checkpoint_pos, dim=-1)
        hook_reward = torch.exp(-25.0 * handle_dist)

        rim_dist = torch.minimum(
            torch.linalg.norm(fingertip_pos - rim_0_pos, dim=-1),
            torch.linalg.norm(fingertip_pos - rim_1_pos, dim=-1),
        )
        gripper_openness = self.program_runtime.controller_api_wrapper.gripper_openness()
        reach_reward = torch.exp(-8.0 * rim_dist)
        grasp_reward = torch.exp(-40.0 * rim_dist) * torch.clamp(
            1.0 - gripper_openness, 0.0, 1.0
        )

        fingertip_center_dist = torch.linalg.norm(fingertip_pos - mug_center_pos, dim=-1)
        fingertip_clear = torch.clamp(
            fingertip_center_dist / self.cfg.success_fingertip_center_min_distance,
            0.0,
            1.0,
        )
        gripper_open = torch.clamp(
            gripper_openness / self.cfg.success_gripper_open_threshold,
            0.0,
            1.0,
        )

        return (
            0.5 * reach_reward
            + 0.5 * grasp_reward
            + 4.0 * hook_reward
            + hook_reward * fingertip_clear
            + hook_reward * gripper_open
        )

    def _external_get_dones(self):
        mug_center_pos = self.mug_center_frame.data.target_pos_w.squeeze(1)
        mug_handle_hole_pos = self.mug_handle_hole_frame.data.target_pos_w.squeeze(1)
        fingertip_pos = self.ee_frame.data.target_pos_w.squeeze(1)
        hanger_checkpoint_pos = self.cup_hanger_checkpoint_frame.data.target_pos_w.squeeze(1)

        handle_checkpoint_distance = torch.norm(
            mug_handle_hole_pos - hanger_checkpoint_pos, dim=1
        )
        fingertip_center_distance = torch.norm(fingertip_pos - mug_center_pos, dim=1)

        handle_near_checkpoint = handle_checkpoint_distance < self.cfg.success_handle_checkpoint_threshold
        fingertip_clear = (
            fingertip_center_distance > self.cfg.success_fingertip_center_min_distance
        )
        gripper_open = (
            self.program_runtime.controller_api_wrapper.gripper_openness()
            > self.cfg.success_gripper_open_threshold
        )
        pose_ok = torch.logical_and(
            handle_near_checkpoint,
            torch.logical_and(fingertip_clear, gripper_open),
        )
        self._mug_done_hold_frames = torch.where(
            pose_ok,
            self._mug_done_hold_frames + 1,
            torch.zeros_like(self._mug_done_hold_frames),
        )
        terminated = self._mug_done_hold_frames >= self.cfg.success_hold_frames

        if self.cfg.sun_program_evaluator_stage_override or bool(
            getattr(self.cfg, "ignore_stage_gate_for_success", False)
        ):
            return terminated

        past_stage = self.current_stage.flatten() > int(
            self.program_runtime.compiled_program.stage_count() / 2
        )
        return torch.logical_and(terminated, past_stage)
