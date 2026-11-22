# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
from kuafu.paths import OBJECT_ROOT

import os
import torch
from collections.abc import Sequence
from pxr import UsdPhysics
from isaaclab.sim import schemas
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.utils.torch.transformations import tf_combine, tf_inverse

from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, RenderCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg,  AssetBaseCfg, AssetBase
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sensors import (
    FrameTransformer,
    FrameTransformerCfg,
)
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.math import sample_uniform
from isaaclab.utils import math as math_utils
from kuafu.isaac.base.manipulation.panda_manipulation_env import (
    PandaManipulationEnvCfg,
    PandaManipulationEnv,
)
from kuafu.isaac.tasks.collision_utils import (
    replace_mesh_collision_with_mug_cylinders,
    replace_mesh_collision_with_simple_hanger_cylinders,
)

ISAAC_NUCLEUS_DIR = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac"
TABLE_HEIGHT = 0.1
@configclass
class HangCupEnvCfg(PandaManipulationEnvCfg):
    """Configuration for the Hang Cup environment.
    
    Inherits from PandaManipulationEnvCfg and overrides the robot's initial state
    while keeping the same ArticulationCfg structure.
    """

    def __post_init__(self):
        """Post-initialization to set robot initial state and call parent's __post_init__."""
        # Call parent's __post_init__ first to set up robot config
        super().__post_init__()
        # Override only the initial state
        self.robot.init_state = ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 1.0,
                "panda_joint2": -0.569,
                "panda_joint3": 0.0,
                "panda_joint4": -2.810,
                "panda_joint5": 0.0,
                "panda_joint6": 3.037,
                "panda_joint7": 0.741,
                "panda_finger_joint.*": 0.04,
            },
            pos=(0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        )

    observation_space: int = 9
    action_space: int = 9
    use_debug_action_mode: bool = False

    # Rigid body properties for the mug
    mug_properties = sim_utils.RigidBodyPropertiesCfg(
        solver_position_iteration_count=32,
        solver_velocity_iteration_count=2,
        max_angular_velocity=1000.0,
        max_linear_velocity=1000.0,
        max_depenetration_velocity=100.0,
        disable_gravity=False,
    )

    _mug_path = str(OBJECT_ROOT / 'large_mug/large_mug.usdc')

    # High-friction physics material for mug so gripper fingers can hold it
    mug_physics_material = sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="max",
        restitution_combine_mode="multiply",
        static_friction=3.0,
        dynamic_friction=2.0,
        restitution=0.0,
    )

    mug = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Mug",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.4, 0.2, TABLE_HEIGHT+0.01), rot=(0, 0, 0, 1)
        ),
        spawn=UsdFileCfg(
            usd_path=_mug_path,
            scale=(1,1,1),
            rigid_props=mug_properties,
            mass_props=sim_utils.MassPropertiesCfg(mass=0.011),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.002,
                rest_offset=0.0,
            ),
            semantic_tags=[("class", "mug")],
        ),
    )

    # Cup holder - static/kinematic object (will not move during simulation)
    # Note: Path is constructed relative to IsaacLab workspace root
    _cup_hanger_path = str(OBJECT_ROOT / 'simple_hanger/assembly1_white_plastic_cyl_collision_static.usda')
    cup_hanger = RigidObjectCfg(
        prim_path="/World/envs/env_.*/CupHolder",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.4, -0.2, TABLE_HEIGHT), rot=(0, 0, 0, 1)
        ),
        spawn=UsdFileCfg(
            usd_path=_cup_hanger_path,
            scale=(1,1,1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,  # Makes the object static - it will not move during simulation
                disable_gravity=True,  # Additional safeguard: disable gravity (though kinematic objects ignore it anyway)
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.0005,
                rest_offset=0.0,
            ),
        ),
    )
    # Low-friction physics material for cup_hanger to allow smooth sliding
    cup_hanger_physics_material = sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=0.01,  # Very low friction for smooth sliding
        dynamic_friction=0.01,  # Very low friction for smooth sliding
        restitution=0.0,
    )

    table = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Table",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, TABLE_HEIGHT/2), rot=(1.0, 0, 0, 0.0)),
        spawn=sim_utils.CuboidCfg(
            size=(0.3, 0.9, TABLE_HEIGHT),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
            ),
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

    # Frame transformers for ee_frame and l4_frame are inherited from parent


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
                offset=OffsetCfg(pos=(0.06, 0.00, 0.05)),  # Smaller offset for the handle hole
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
                offset=OffsetCfg(pos=(0.0, 0.0, 0.04)),  # Center of cup body - for pouring and upright detection
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
                offset=OffsetCfg(pos=(0.0, 0.0, 0.04)),  # Center of cup body - for pouring and upright detection
            )
        ],
    )

    marker_cfg_mug_center = FRAME_MARKER_CFG.copy()
    marker_cfg_mug_center.markers["frame"].scale = (0.05, 0.05, 0.05)
    marker_cfg_mug_center.prim_path = "/Visuals/mug_rim_frame_0"

    mug_rim_frame_0 = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/panda_link0",
        debug_vis=False,
        visualizer_cfg=marker_cfg_mug_center,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Mug",
                name="mug_rim_frame_0",
                offset=OffsetCfg(pos=(0.0, 0.04, 0.075)),  # Center of cup body - for pouring and upright detection
            )
        ],
    )

    marker_cfg_mug_rim_frame_1 = FRAME_MARKER_CFG.copy()
    marker_cfg_mug_center.markers["frame"].scale = (0.05, 0.05, 0.05)
    marker_cfg_mug_rim_frame_1.prim_path = "/Visuals/mug_rim_frame_1"

    mug_rim_frame_1 = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/panda_link0",
        debug_vis=False,
        visualizer_cfg=marker_cfg_mug_center,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Mug",
                name="mug_rim_frame_1",
                offset=OffsetCfg(pos=(0.0, -0.04, 0.075)),  # Center of cup body - for pouring and upright detection
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


    def __post_init__(self):
        """Post-initialization to set render interval and robot actuator parameters."""
        super().__post_init__()  # Call parent's __post_init__ first
        # Override render interval to match decimation
        self.sim.render_interval = self.decimation


class HangCupEnv(PandaManipulationEnv):
    """Environment for hanging a cup on a cup holder using a Panda robot.
    
    Inherits from PandaManipulationEnv and adds cup-specific objects and logic.
    """
    cfg: HangCupEnvCfg

    def __init__(
        self,
        cfg: HangCupEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        # Initialize the base class (which calls _setup_scene and _initialize_robot_common)
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

        # Simple timing prints (disabled by default). Initialize early because we call
        # update_data_in_controller() during __init__.
        self._profile = bool(int(os.getenv("KUAFU_PROFILE", "0")))
        self._profile_every = int(os.getenv("KUAFU_PROFILE_EVERY", "200"))
        self._profile_sync_cuda = bool(int(os.getenv("KUAFU_PROFILE_SYNC_CUDA", "0")))

        print("[INFO]: Completed setting up the environment...")

    def close(self):
        super().close()

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        super()._reset_idx(env_ids)

        if getattr(self.cfg, "disable_reset_randomization", False):
            cup_offset = torch.zeros((len(env_ids), 3), device=self.device)
        else:
            cup_offset = torch.rand(len(env_ids), 3).to(self.device) * torch.tensor([0.1, 0.1, 0.0]).to(self.device)

        # Reset mug position
        env_ids_tensor = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        root = self.mug.data.default_root_state.clone()
        root = root.index_select(0, env_ids_tensor)

        # write pose and velocities to the sim, then reset buffers
        root[:, :3] += self.scene.env_origins.index_select(0, env_ids_tensor) + cup_offset
        eulers = torch.zeros(
            (len(env_ids), 3), device=self.device
        )  # or your real angles
        roll = eulers[:, 0]
        pitch = eulers[:, 1]
        if getattr(self.cfg, "disable_reset_randomization", False):
            yaw = eulers[:, 2] + torch.pi
        else:
            yaw = eulers[:, 2] + torch.rand(len(env_ids), device=self.device) * torch.pi /2 + torch.pi

        quat = math_utils.quat_from_euler_xyz(roll, pitch, yaw)
        root[:, 3:7] = quat  # quaternions are WXYZ in Isaac Lab

        # write pose and velocities to the sim, then reset buffers
        self.mug.write_root_pose_to_sim(root[:, :7], env_ids=env_ids)
        self.mug.write_root_velocity_to_sim(
            torch.zeros((len(env_ids), 6), device=self.device), env_ids=env_ids
        )
        self.mug.reset(env_ids)

        # Use parent's _reset_robot method
        env_ids_tensor = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        self._reset_robot(env_ids_tensor)

    def _setup_scene(self):

        self._robot = Articulation(self.cfg.robot)

        # Spawn mug USD file first, then apply RigidBodyAPI and collision approximation before creating RigidObject
        # This fixes the missing RigidBodyAPI error and ensures proper collision detection
        stage = get_current_stage()
        mug_cfg = self.cfg.mug
        # Spawn the USD file if spawn config exists
        if mug_cfg.spawn is not None:
            mug_cfg.spawn.func(
                mug_cfg.prim_path,
                mug_cfg.spawn,
                translation=mug_cfg.init_state.pos,
                orientation=mug_cfg.init_state.rot,
            )
        # Apply RigidBodyAPI and collision approximation to all matching prims
        if mug_cfg.spawn is not None:
            matching_prims = sim_utils.find_matching_prims(mug_cfg.prim_path)
            for prim in matching_prims:
                # Always apply RigidBodyAPI (it's safe to apply even if already present)
                if hasattr(mug_cfg.spawn, 'rigid_props') and mug_cfg.spawn.rigid_props is not None:
                    schemas.define_rigid_body_properties(prim.GetPath().pathString, mug_cfg.spawn.rigid_props)

                # Apply mass properties if specified
                if hasattr(mug_cfg.spawn, 'mass_props') and mug_cfg.spawn.mass_props is not None:
                    schemas.define_mass_properties(prim.GetPath().pathString, mug_cfg.spawn.mass_props)

                # Create low-friction physics material for this cup_hanger instance
                material_path = f"{prim.GetPath().pathString}/physicsMaterial"
                self.cfg.mug_physics_material.func(material_path, self.cfg.mug_physics_material)

                replace_mesh_collision_with_mug_cylinders(
                    prim,
                    stage=stage,
                    collision_props=getattr(mug_cfg.spawn, "collision_props", None),
                    material_path=material_path,
                )

        # Spawn cup holder USD file, then apply RigidBodyAPI and collision approximation before creating RigidObject
        cup_hanger_cfg = self.cfg.cup_hanger
        if cup_hanger_cfg.spawn is not None:
            cup_hanger_cfg.spawn.func(
                cup_hanger_cfg.prim_path,
                cup_hanger_cfg.spawn,
                translation=cup_hanger_cfg.init_state.pos,
                orientation=cup_hanger_cfg.init_state.rot,
            )
        # Apply RigidBodyAPI and collision approximation to all matching prims
        if cup_hanger_cfg.spawn is not None:
            matching_prims = sim_utils.find_matching_prims(cup_hanger_cfg.prim_path)
            for prim in matching_prims:
                # Apply RigidBodyAPI if not already present
                if not UsdPhysics.RigidBodyAPI(prim):
                    if hasattr(cup_hanger_cfg.spawn, 'rigid_props') and cup_hanger_cfg.spawn.rigid_props is not None:
                        schemas.define_rigid_body_properties(prim.GetPath().pathString, cup_hanger_cfg.spawn.rigid_props)

                # Apply mass properties if specified
                if hasattr(cup_hanger_cfg.spawn, 'mass_props') and cup_hanger_cfg.spawn.mass_props is not None:
                    schemas.define_mass_properties(prim.GetPath().pathString, cup_hanger_cfg.spawn.mass_props)

                # Create low-friction physics material for this cup_hanger instance
                material_path = f"{prim.GetPath().pathString}/physicsMaterial"
                self.cfg.cup_hanger_physics_material.func(material_path, self.cfg.cup_hanger_physics_material)

                replace_mesh_collision_with_simple_hanger_cylinders(
                    prim,
                    stage=stage,
                    collision_props=getattr(cup_hanger_cfg.spawn, "collision_props", None),
                    material_path=material_path,
                )

        # Now create RigidObjects (they will find the RigidBodyAPI we just applied)
        self.mug = RigidObject(self.cfg.mug)
        self.cup_hanger = RigidObject(self.cfg.cup_hanger)

        # Create the table as a RigidObject (static/kinematic)
        self.table = RigidObject(self.cfg.table)


        self.mug_handle_hole_frame = FrameTransformer(self.cfg.mug_handle_hole_frame)
        self.mug_center_frame = FrameTransformer(self.cfg.mug_center_frame)
        self.cup_hanger_branch_1_tip_frame = FrameTransformer(self.cfg.cup_hanger_branch_1_tip_frame)
        self.mug_rim_frame_0 = FrameTransformer(self.cfg.mug_rim_frame_0)
        self.mug_rim_frame_1 = FrameTransformer(self.cfg.mug_rim_frame_1)
        self.ee_frame = FrameTransformer(self.cfg.ee_frame)
        self.l4_frame = FrameTransformer(self.cfg.l4_frame)

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)


        # Add objects to scene
        self.scene.articulations["robot"] = self._robot
        self.scene.rigid_objects["mug"] = self.mug
        self.scene.rigid_objects["cup_hanger"] = self.cup_hanger
        self.scene.rigid_objects["table"] = self.table

        self.scene.sensors["panda_fingertip"] = self.ee_frame
        self.scene.sensors["panda_link4"] = self.l4_frame
        self.scene.sensors["mug_handle_hole_frame"] = self.mug_handle_hole_frame
        self.scene.sensors["mug_center_frame"] = self.mug_center_frame
        self.scene.sensors["mug_rim_frame_0"] = self.mug_rim_frame_0
        self.scene.sensors["mug_rim_frame_1"] = self.mug_rim_frame_1
        self.scene.sensors["cup_hanger_branch_1_tip_frame"] = self.cup_hanger_branch_1_tip_frame

        self.scene.clone_environments(copy_from_source=False)

        # Note: Cup holder is kinematic (static) - no need to set velocities as kinematic objects
        # are automatically static and don't respond to forces

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=1000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])


    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        if getattr(self.cfg, "disable_terminations", False):
            done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            return done, done

        mug_pos = self.mug.data.root_state_w[:, :3]
        cup_hanger_pos = self.cup_hanger.data.root_state_w[:, :3]

        # Time out
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # Goal reached: cup is hanging on the cup holder
        # Calculate distance on xy plane (ignoring z)
        distance_to_holder_xy = torch.norm(mug_pos[:, :2] - cup_hanger_pos[:, :2], dim=1)
        mug_pos_z = mug_pos[:, 2]

        # Check if gripper is open (finger joints are the last two joints)
        # For Panda gripper, open position is around 0.04, closed is 0.0
        finger_joint_1 = self._robot.data.joint_pos[:, -2]  # Second to last joint
        finger_joint_2 = self._robot.data.joint_pos[:, -1]  # Last joint
        gripper_open = torch.logical_and(
            finger_joint_1 > 0.03,  # Threshold for open gripper
            finger_joint_2 > 0.03   # Threshold for open gripper
        )

        # Check if cup is stable (very low velocity)
        # root_state_w format: [pos(3), quat(4), lin_vel(3), ang_vel(3)]
        mug_lin_vel = self.mug.data.root_state_w[:, 7:10]  # Linear velocity
        mug_ang_vel = self.mug.data.root_state_w[:, 10:13]  # Angular velocity
        mug_lin_vel_norm = torch.norm(mug_lin_vel, dim=1)
        mug_ang_vel_norm = torch.norm(mug_ang_vel, dim=1)
        cup_stable = torch.logical_and(
            mug_lin_vel_norm < 0.01,  # Very low linear velocity (1 cm/s)
            mug_ang_vel_norm < 0.1   # Very low angular velocity (0.1 rad/s)
        )

        # Cup should be close to holder on xy plane (within 5 cm), at appropriate height (higher than 0.1 m),
        # gripper should be open, and cup should be stable
        goal_reached = torch.logical_and(
            distance_to_holder_xy < 0.05,  # Within 5 cm on xy plane
            torch.logical_and(
                mug_pos_z > 0.1,  # Cup center height is higher than 0.1 m
                torch.logical_and(
                    gripper_open,  # Gripper is open
                    cup_stable  # Cup is stable (very low velocity)
                )
            )
        )

        terminated = goal_reached

        return terminated, time_out

    def _get_observations(self) -> dict:
        joint_states = self.scene.state["articulation"]["robot"]
        pose = self.scene.state["articulation"]["robot"]["root_pose"]
        mug_pos = self.mug.data.root_state_w[:, :3]
        cup_hanger_pos = self.cup_hanger.data.root_state_w[:, :3]

        return {
            "joint_states": joint_states,
            "pose": pose,
            "mug_pos": mug_pos,
            "cup_hanger_pos": cup_hanger_pos,
        }


    def _get_rewards(self) -> torch.Tensor:
        mug_pos = self.mug.data.root_state_w[:, 0]
        return mug_pos
