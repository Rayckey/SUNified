#!/usr/bin/env python3
"""Suspended standalone Robotiq 2F-85 close diagnostic.

This test does not spawn the Kinova arm or any task objects. It references only
/World/Robotiq_2F_85 from the current Robotiq USD, fixes the gripper base to the
world, commands the current two-driver env gripper target to close, and logs
every gripper joint.
"""

from __future__ import annotations

import argparse
import csv
import math
from functools import partial
from pathlib import Path

from kuafu.paths import resolve_artifact_path

from isaaclab.app import AppLauncher

print = partial(print, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=240, help="Number of close-command physics steps.")
    parser.add_argument("--settle_steps", type=int, default=60, help="Open-gripper settle steps before closing.")
    parser.add_argument(
        "--close_target",
        type=float,
        default=0.8,
        help="Gripper close amount in radians. Defaults to the USD 0.8 rad limit.",
    )
    parser.add_argument("--open_target", type=float, default=0.0, help="Gripper open amount in radians.")
    parser.add_argument("--root_height", type=float, default=1.0, help="Suspended gripper root height in meters.")
    parser.add_argument("--print_every", type=int, default=20, help="Print one measured row every N steps.")
    parser.add_argument(
        "--csv",
        type=str,
        default="diagnostics/kinova_suspended_gripper_close_diag.csv",
        help="CSV output path for measured joint values.",
    )
    parser.add_argument(
        "--wrapper_usd",
        type=str,
        default="diagnostics/robotiq_2f85_suspended_wrapper.usda",
        help="Generated wrapper USD path. The source Robotiq USD is not modified.",
    )
    parser.add_argument(
        "--dummy_parent_attachment",
        action="store_true",
        help="Wrap the Robotiq as a child of a dummy rigid parent through a fixed joint.",
    )
    parser.add_argument(
        "--combined_root_xform",
        action="store_true",
        help="Author the same Robotiq_2F_85 root transform used in the combined Kinova USD wrapper.",
    )
    parser.add_argument(
        "--drive_mode",
        choices=("env", "finger_only", "all_zero_passive"),
        default="env",
        help=(
            "Actuator setup. 'env' matches the current Kinova env gripper actuator groups; "
            "'finger_only' configures only finger_joint; 'all_zero_passive' adds zero-gain passive "
            "actuators for every other gripper joint so IsaacLab reports all DOFs as configured."
        ),
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = False
    return args


args_cli = parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402


GRIPPER_JOINT_NAMES = [
    "finger_joint",
    "right_outer_knuckle_joint",
    "left_outer_finger_joint",
    "right_outer_finger_joint",
    "left_inner_finger_joint",
    "right_inner_finger_joint",
    "left_inner_finger_knuckle_joint",
    "right_inner_finger_knuckle_joint",
]

ENV_DRIVER_GRIPPER_JOINT_NAMES = [
    "finger_joint",
    "right_outer_knuckle_joint",
]

ENV_STABILIZER_GRIPPER_JOINT_NAMES = [
    "left_inner_finger_joint",
    "right_inner_finger_joint",
]


def _source_gripper_usd() -> Path:
    learned_dir = Path(__file__).resolve().parents[1]
    return learned_dir / "assets" / "kinova" / "usd" / "gen3_7dof_vision_robotiq_2f_85" / "Robotiq" / "2F-85" / "Robotiq_2F_85_edit.usd"


def _ensure_wrapper_usd(wrapper_path: Path, source_usd: Path) -> Path:
    """Create a small wrapper that references only the gripper articulation root."""
    wrapper_path = wrapper_path.resolve()
    source_usd = source_usd.resolve()
    wrapper_path.parent.mkdir(parents=True, exist_ok=True)
    if wrapper_path.exists():
        wrapper_path.unlink()

    stage = Usd.Stage.CreateNew(str(wrapper_path))
    if args_cli.dummy_parent_attachment:
        root = stage.DefinePrim("/Robot", "Xform")
        stage.SetDefaultPrim(root)
        UsdPhysics.ArticulationRootAPI.Apply(root)
        PhysxSchema.PhysxArticulationAPI.Apply(root)

        parent = stage.DefinePrim("/Robot/base_link", "Xform")
        if args_cli.combined_root_xform:
            parent_xform = UsdGeom.Xformable(parent)
            parent_xform.ClearXformOpOrder()
            parent_xform.AddTranslateOp().Set(Gf.Vec3d(-1.7965151912234983e-16, -0.02485908567905426, 1.1258597373962402))
            parent_xform.AddOrientOp().Set(Gf.Quatf(-3.635883331298828e-06, 1.0, 4.16336005926686e-15, 1.0510493029299711e-16))
            parent_xform.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 1.0))
        UsdPhysics.RigidBodyAPI.Apply(parent)

        gripper = stage.DefinePrim("/Robot/Robotiq_2F_85", "Xform")
        if args_cli.combined_root_xform:
            gripper_xform = UsdGeom.Xformable(gripper)
            gripper_xform.ClearXformOpOrder()
            gripper_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, -0.024859533074486507, 1.1873847357912486))
            gripper_xform.AddOrientOp().Set(Gf.Quatf(6.123234262925839e-17, -0.7071067690849304, -4.666511963321569e-17, -0.7071067690849304))
            gripper_xform.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 1.0))
        gripper.GetReferences().AddReference(str(source_usd), Sdf.Path("/World/Robotiq_2F_85"))
        gripper.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        gripper.RemoveAPI(PhysxSchema.PhysxArticulationAPI)

        root_fixed = UsdPhysics.FixedJoint.Define(stage, "/Robot/root_joint")
        root_fixed.CreateBody1Rel().SetTargets([Sdf.Path("/Robot/base_link")])
        root_fixed.CreateJointEnabledAttr(True)

        gripper_fixed = UsdPhysics.FixedJoint.Define(stage, "/Robot/gripper_fixed_joint")
        gripper_fixed.CreateBody0Rel().SetTargets([Sdf.Path("/Robot/base_link")])
        gripper_fixed.CreateBody1Rel().SetTargets([Sdf.Path("/Robot/Robotiq_2F_85/base_link")])
        if args_cli.combined_root_xform:
            gripper_fixed.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, -0.061525))
            gripper_fixed.CreateLocalRot0Attr(Gf.Quatf(0.0, 1.0, 0.0, 0.0))
            gripper_fixed.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            gripper_fixed.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        gripper_fixed.CreateJointEnabledAttr(True)
    else:
        root = stage.DefinePrim("/Robotiq_2F_85", "Xform")
        stage.SetDefaultPrim(root)
        if args_cli.combined_root_xform:
            xform = UsdGeom.Xformable(root)
            xform.ClearXformOpOrder()
            xform.AddTranslateOp().Set(Gf.Vec3d(0.0, -0.024859533074486507, 1.1873847357912486))
            xform.AddOrientOp().Set(Gf.Quatf(6.123234262925839e-17, -0.7071067690849304, -4.666511963321569e-17, -0.7071067690849304))
            xform.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 1.0))
        root.GetReferences().AddReference(str(source_usd), Sdf.Path("/World/Robotiq_2F_85"))

        # Add a world fixed joint in the wrapper so the gripper is actually suspended.
        # The native Robotiq USD is untouched.
        fixed = UsdPhysics.FixedJoint.Define(stage, "/Robotiq_2F_85/root_joint")
        fixed.CreateBody1Rel().SetTargets([Sdf.Path("/Robotiq_2F_85/base_link")])
        fixed.CreateJointEnabledAttr(True)

    stage.GetRootLayer().Save()
    return wrapper_path


def _make_actuators(mode: str) -> dict[str, ImplicitActuatorCfg]:
    actuators = {
        "gripper_master": ImplicitActuatorCfg(
            joint_names_expr=ENV_DRIVER_GRIPPER_JOINT_NAMES if mode == "env" else ["finger_joint"],
            effort_limit_sim=16.5 if mode == "env" else 8.0,
            velocity_limit_sim=2.0 if mode == "env" else 1.0,
            stiffness=40.0 if mode == "env" else 11.25,
            damping=1.0 if mode == "env" else 0.1,
            friction=0.0,
            armature=0.0,
        )
    }
    if mode == "all_zero_passive":
        actuators["gripper_all_other_passive"] = ImplicitActuatorCfg(
            joint_names_expr=[
                "right_outer_knuckle_joint",
                ".*_outer_finger_joint",
                ".*_inner_finger_joint",
                ".*_inner_finger_knuckle_joint",
            ],
            effort_limit_sim=1.0,
            velocity_limit_sim=1.0,
            stiffness=0.0,
            damping=0.0,
            friction=0.0,
            armature=0.0,
        )
    return actuators


def _make_gripper_cfg(wrapper_usd: Path) -> ArticulationCfg:
    return ArticulationCfg(
        prim_path="/World/Gripper",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(wrapper_usd),
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                fix_root_link=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.001),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, float(args_cli.root_height)),
            joint_pos={name: 0.0 for name in GRIPPER_JOINT_NAMES},
        ),
        actuators=_make_actuators(args_cli.drive_mode),
        soft_joint_pos_limit_factor=1.0,
    )


def _resolve_joint_ids(robot: Articulation, names: list[str]) -> tuple[list[int], list[str]]:
    ids: list[int] = []
    resolved_names: list[str] = []
    joint_name_to_id = {joint_name: i for i, joint_name in enumerate(robot.joint_names)}
    for name in names:
        if name not in joint_name_to_id:
            print(f"[SuspendedGripperDiag] missing joint: {name}")
            continue
        ids.append(int(joint_name_to_id[name]))
        resolved_names.append(name)
    return ids, resolved_names


def _as_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def _env_driver_targets(reference: torch.Tensor, close_amount: float) -> torch.Tensor:
    signs = torch.tensor([1.0, 1.0], device=reference.device, dtype=reference.dtype).view(1, -1)
    return signs * float(close_amount)


def _env_stabilizer_targets(driver_targets: torch.Tensor) -> torch.Tensor:
    close_amount = 0.5 * (driver_targets[:, 0:1] + driver_targets[:, 1:2])
    close_amount = torch.clamp(close_amount, min=0.0, max=float(args_cli.close_target))
    return torch.cat([-close_amount, -close_amount], dim=-1)


def _record_row(
    robot: Articulation,
    step: int,
    sim_time: float,
    phase: str,
    command_target: float,
    gripper_ids: list[int],
    gripper_names: list[str],
) -> dict[str, float | int | str]:
    row: dict[str, float | int | str] = {
        "step": int(step),
        "time_s": float(sim_time),
        "phase": phase,
        "commanded_finger_target": float(command_target),
        "root_x": _as_float(robot.data.root_pos_w[0, 0]),
        "root_y": _as_float(robot.data.root_pos_w[0, 1]),
        "root_z": _as_float(robot.data.root_pos_w[0, 2]),
    }
    for joint_id, name in zip(gripper_ids, gripper_names):
        row[f"{name}.pos"] = _as_float(robot.data.joint_pos[0, joint_id])
        row[f"{name}.vel"] = _as_float(robot.data.joint_vel[0, joint_id])
        row[f"{name}.target"] = _as_float(robot.data.joint_pos_target[0, joint_id])
        row[f"{name}.computed_torque"] = _as_float(robot.data.computed_torque[0, joint_id])
        row[f"{name}.applied_torque"] = _as_float(robot.data.applied_torque[0, joint_id])
    return row


def _print_row(row: dict[str, float | int | str], names: list[str]) -> None:
    parts = [
        f"step={row['step']}",
        f"t={float(row['time_s']):.3f}",
        f"phase={row['phase']}",
        f"cmd={float(row['commanded_finger_target']):.6f}",
        f"root_z={float(row['root_z']):.6f}",
    ]
    for name in names:
        parts.append(f"{name}.pos={float(row[f'{name}.pos']): .6f}")
    print("[SuspendedGripperDiag] " + " ".join(parts))


def main() -> None:
    source_usd = _source_gripper_usd()
    if not source_usd.exists():
        raise FileNotFoundError(f"Robotiq USD not found: {source_usd}")
    wrapper_usd = _ensure_wrapper_usd(resolve_artifact_path(args_cli.wrapper_usd, "Generated wrapper USD"), source_usd)
    print(f"[SuspendedGripperDiag] source USD: {source_usd}")
    print(f"[SuspendedGripperDiag] wrapper USD: {wrapper_usd}")
    print(f"[SuspendedGripperDiag] drive mode: {args_cli.drive_mode}")

    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[0.55, -0.75, 1.25], target=[0.0, 0.0, float(args_cli.root_height)])

    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.8, 0.8, 0.8))
    light_cfg.func("/World/Light", light_cfg)

    robot = Articulation(cfg=_make_gripper_cfg(wrapper_usd))
    sim.reset()

    root_state = robot.data.default_root_state.clone()
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(torch.zeros_like(root_state[:, 7:]))
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = torch.zeros_like(robot.data.default_joint_vel)
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.reset()
    robot.update(sim.get_physics_dt())

    if args_cli.drive_mode == "env":
        controlled_ids, controlled_names = _resolve_joint_ids(robot, ENV_DRIVER_GRIPPER_JOINT_NAMES)
        if controlled_names != ENV_DRIVER_GRIPPER_JOINT_NAMES:
            missing = [name for name in ENV_DRIVER_GRIPPER_JOINT_NAMES if name not in controlled_names]
            raise RuntimeError(f"Missing env gripper driver joints: {missing}")
        stabilizer_ids = []
        stabilizer_names = []
    else:
        finger_ids, finger_names = robot.find_joints(["finger_joint"])
        if len(finger_ids) != 1:
            raise RuntimeError(f"Expected exactly one finger_joint, got {finger_names}")
        controlled_ids = [int(finger_ids[0])]
        controlled_names = list(finger_names)
        stabilizer_ids = []
        stabilizer_names = []
    hold_targets = robot.data.default_joint_pos[:, controlled_ids].clone()
    if args_cli.drive_mode == "env":
        hold_targets[:, :] = _env_driver_targets(hold_targets, float(args_cli.open_target))
    else:
        hold_targets[:, 0] = float(args_cli.open_target)

    gripper_ids, gripper_names = _resolve_joint_ids(robot, GRIPPER_JOINT_NAMES)

    print("[SuspendedGripperDiag] robot prim: /World/Gripper")
    print(f"[SuspendedGripperDiag] all joints: {robot.joint_names}")
    print(f"[SuspendedGripperDiag] controlled joints: {list(controlled_names)}")
    if stabilizer_names:
        print(f"[SuspendedGripperDiag] stabilizer joints: {stabilizer_names}")
    print(f"[SuspendedGripperDiag] logged gripper joints: {gripper_names}")
    print("[SuspendedGripperDiag] joint limits:")
    for joint_id, name in zip(gripper_ids, gripper_names):
        lo = _as_float(robot.data.joint_pos_limits[0, joint_id, 0])
        hi = _as_float(robot.data.joint_pos_limits[0, joint_id, 1])
        print(f"  {name}: [{lo:.6f}, {hi:.6f}]")

    rows: list[dict[str, float | int | str]] = []
    sim_dt = sim.get_physics_dt()
    total_steps = int(args_cli.settle_steps) + int(args_cli.steps)
    print_every = max(1, int(args_cli.print_every))

    for step in range(total_steps + 1):
        if step < int(args_cli.settle_steps):
            phase = "settle_open"
            command_target = float(args_cli.open_target)
        else:
            phase = "close"
            command_target = float(args_cli.close_target)

        targets = hold_targets.clone()
        if args_cli.drive_mode == "env":
            targets[:, :] = _env_driver_targets(targets, command_target)
        else:
            targets[:, 0] = command_target
        robot.set_joint_position_target(targets, joint_ids=controlled_ids)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim_dt)

        row = _record_row(robot, step, step * sim_dt, phase, command_target, gripper_ids, gripper_names)
        rows.append(row)
        if step % print_every == 0 or step == total_steps:
            _print_row(row, gripper_names)

    csv_path = resolve_artifact_path(args_cli.csv, "Diagnostic CSV")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    final = rows[-1]
    print(f"[SuspendedGripperDiag] wrote CSV: {csv_path}")
    print("[SuspendedGripperDiag] final measured positions:")
    for name in gripper_names:
        print(
            f"  {name}: pos={float(final[f'{name}.pos']):.9f}, "
            f"vel={float(final[f'{name}.vel']):.9f}, "
            f"target={float(final[f'{name}.target']):.9f}, "
            f"applied_torque={float(final[f'{name}.applied_torque']):.9f}"
        )


if __name__ == "__main__":
    main()
    print("[SuspendedGripperDiag] done")
