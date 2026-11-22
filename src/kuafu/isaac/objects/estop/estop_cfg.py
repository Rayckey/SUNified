from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from .estop import EStopBuilderCfg, ensure_estop_usd


ESTOP_BUILDER_CFG = EStopBuilderCfg()
ESTOP_USD_PATH = ensure_estop_usd(ESTOP_BUILDER_CFG)


def make_estop_cfg(
    prim_path: str = "/World/envs/env_.*/EStop",
    *,
    builder_cfg: EStopBuilderCfg | None = None,
    spawn_pose: tuple[float, float, float] = (0.0, 0.0, 0.0),
    spawn_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
) -> ArticulationCfg:
    builder_cfg = builder_cfg or ESTOP_BUILDER_CFG
    usd_path = ensure_estop_usd(builder_cfg)
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            # The composed E-stop USD has an articulation-root Xform at the top level.
            # Enabling Isaac Lab's automatic contact-sensor injection on that prim path
            # fails because the root itself is not a rigid body container.
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                max_linear_velocity=10.0,
                max_angular_velocity=20.0,
                max_depenetration_velocity=5.0,
                enable_gyroscopic_forces=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
                sleep_threshold=0.005,
                stabilization_threshold=0.001,
                fix_root_link=True,
            ),
            joint_drive_props=sim_utils.JointDrivePropertiesCfg(max_effort=200.0, max_velocity=4.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=spawn_pose,
            rot=spawn_rot,
            joint_pos={
                "estop_press_joint": 0.0,
                "estop_twist_joint": 0.0,
            },
        ),
        actuators={
            "press": ImplicitActuatorCfg(
                joint_names_expr=["estop_press_joint"],
                effort_limit_sim=10.0,
                velocity_limit_sim=0.5,
                stiffness=300.0,
                damping=30.0,
            ),
            "twist": ImplicitActuatorCfg(
                joint_names_expr=["estop_twist_joint"],
                effort_limit_sim=4.0,
                velocity_limit_sim=4.0,
                stiffness=0.01,
                damping=0.02,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )


ESTOP_CFG = make_estop_cfg()
