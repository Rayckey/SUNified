#!/usr/bin/env python3
"""Isolated Kinova Robotiq gripper close diagnostic.

This script spawns only the Kinova+Robotiq articulation, holds the arm at its
initial pose, commands the current two-driver env gripper target to close,
and logs the actual gripper joint positions/velocities/torques.
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
    parser.add_argument("--dt", type=float, default=0.005, help="Physics timestep for the diagnostic.")
    parser.add_argument("--steps", type=int, default=240, help="Number of close-command physics steps.")
    parser.add_argument("--settle_steps", type=int, default=60, help="Open-gripper settle steps before closing.")
    parser.add_argument(
        "--close_target",
        type=float,
        default=0.8,
        help="Gripper close amount in radians. Defaults to the USD 0.8 rad limit.",
    )
    parser.add_argument("--open_target", type=float, default=0.0, help="Gripper open amount in radians.")
    parser.add_argument("--usd_override", type=str, default="", help="Optional robot USD path override for diagnostic asset-copy tests.")
    parser.add_argument("--no_ground", action="store_true", help="Do not spawn the ground plane.")
    parser.add_argument("--root_z_offset", type=float, default=0.0, help="Add this z offset to the robot root pose.")
    parser.add_argument("--zero_arm_pose", action="store_true", help="Set arm joints to zero in the diagnostic before closing the gripper.")
    parser.add_argument("--strong_gripper_gains", action="store_true", help="Use IsaacLab task-style stronger Robotiq gripper gains for this diagnostic.")
    parser.add_argument("--disable_all_collisions", action="store_true", help="Disable all PhysicsCollisionAPI prims under /World/Robot for contact isolation.")
    parser.add_argument("--control_finger_only", action="store_true", help="Only send position target for finger_joint after reset.")
    parser.add_argument(
        "--original_coupled_targets",
        choices=("none", "usd_drive", "measured"),
        default="none",
        help=(
            "For the original IsaacLab Robotiq USD, send targets to the coupled gripper joints. "
            "usd_drive follows the USD-authored inner-finger drive signs; "
            "measured follows the observed passive closing directions."
        ),
    )
    parser.add_argument(
        "--original_active_targets",
        choices=("none", "usd_inner", "measured_inner"),
        default="none",
        help="Diagnostic variant: drive only finger_joint and the two inner_finger joints.",
    )
    parser.add_argument(
        "--urdf_mimic_asset",
        action="store_true",
        help="Use compatibility settings for the URDF-derived Robotiq asset: fixed outer-finger joints are not articulation DOFs and mimic joints are not separately actuated.",
    )
    parser.add_argument(
        "--urdf_independent_asset",
        action="store_true",
        help="Use compatibility settings for the URDF-derived Robotiq asset without PhysX mimic: fixed outer-finger joints are omitted and the six revolute Robotiq joints are driven with URDF mimic signs.",
    )
    parser.add_argument("--print_every", type=int, default=20, help="Print one measured row every N steps.")
    parser.add_argument(
        "--include_outer_finger_passive",
        action="store_true",
        help="Diagnostic variant: include .*_outer_finger_joint in the zero-gain passive actuator.",
    )
    parser.add_argument(
        "--active_passive_gains",
        action="store_true",
        help="Diagnostic variant: give the Robotiq passive actuator group small nonzero PD gains.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="diagnostics/kinova_gripper_close_diag.csv",
        help="CSV output path for measured joint values.",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = False
    return args


args_cli = parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402
from pxr import UsdPhysics  # noqa: E402
from isaaclab.sim.utils import get_current_stage  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402

from kuafu.isaac.learned.kinova.kinova_table_env_learned import KinovaTableEnvLearnedCfg  # noqa: E402


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

URDF_INDEPENDENT_GRIPPER_JOINT_NAMES = [
    "finger_joint",
    "right_outer_knuckle_joint",
    "left_inner_finger_joint",
    "right_inner_finger_joint",
    "left_inner_finger_knuckle_joint",
    "right_inner_finger_knuckle_joint",
]

URDF_INDEPENDENT_GRIPPER_JOINT_SIGNS = [1.0, 1.0, -1.0, -1.0, -1.0, -1.0]

ORIGINAL_COUPLED_GRIPPER_JOINT_NAMES = [
    "finger_joint",
    "right_outer_knuckle_joint",
    "left_inner_finger_joint",
    "right_inner_finger_joint",
    "right_inner_finger_knuckle_joint",
    "left_inner_finger_knuckle_joint",
]

ORIGINAL_COUPLED_GRIPPER_SIGNS = {
    "usd_drive": [1.0, 1.0, -1.0, -1.0, -1.0, -1.0],
    "measured": [1.0, 1.0, -1.0, -1.0, -1.0, -1.0],
}

ORIGINAL_ACTIVE_GRIPPER_JOINT_NAMES = [
    "finger_joint",
    "left_inner_finger_joint",
    "right_inner_finger_joint",
]

ORIGINAL_ACTIVE_GRIPPER_SIGNS = {
    "usd_inner": [1.0, -1.0, -1.0],
    "measured_inner": [1.0, -1.0, -1.0],
}


def _resolve_joint_ids(robot: Articulation, names: list[str]) -> tuple[list[int], list[str]]:
    ids: list[int] = []
    resolved_names: list[str] = []
    joint_name_to_id = {joint_name: i for i, joint_name in enumerate(robot.joint_names)}
    for name in names:
        if name not in joint_name_to_id:
            print(f"[KinovaGripperDiag] missing joint: {name}")
            continue
        ids.append(int(joint_name_to_id[name]))
        resolved_names.append(name)
    return ids, resolved_names


def _as_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def _env_driver_gripper_targets(reference: torch.Tensor, finger_target: float) -> torch.Tensor:
    signs = torch.tensor([1.0, 1.0], device=reference.device, dtype=reference.dtype).view(1, -1)
    return signs * float(finger_target)


def _env_stabilizer_gripper_targets(driver_targets: torch.Tensor, close_limit: float) -> torch.Tensor:
    close_amount = 0.5 * (driver_targets[:, 0:1] + driver_targets[:, 1:2])
    close_amount = torch.clamp(close_amount, min=0.0, max=float(close_limit))
    return torch.cat([-close_amount, -close_amount], dim=-1)


def _urdf_independent_gripper_targets(reference: torch.Tensor, finger_target: float) -> torch.Tensor:
    signs = torch.tensor(
        URDF_INDEPENDENT_GRIPPER_JOINT_SIGNS,
        device=reference.device,
        dtype=reference.dtype,
    ).view(1, -1)
    return signs * float(finger_target)


def _original_coupled_gripper_targets(reference: torch.Tensor, finger_target: float, mode: str) -> torch.Tensor:
    signs = torch.tensor(
        ORIGINAL_COUPLED_GRIPPER_SIGNS[mode],
        device=reference.device,
        dtype=reference.dtype,
    ).view(1, -1)
    return signs * float(finger_target)


def _original_active_gripper_targets(reference: torch.Tensor, finger_target: float, mode: str) -> torch.Tensor:
    signs = torch.tensor(
        ORIGINAL_ACTIVE_GRIPPER_SIGNS[mode],
        device=reference.device,
        dtype=reference.dtype,
    ).view(1, -1)
    return signs * float(finger_target)


def _record_row(robot: Articulation, step: int, sim_time: float, phase: str, command_target: float,
                gripper_ids: list[int], gripper_names: list[str]) -> dict[str, float | int | str]:
    row: dict[str, float | int | str] = {
        "step": int(step),
        "time_s": float(sim_time),
        "phase": phase,
        "commanded_finger_target": float(command_target),
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
    ]
    for name in names:
        parts.append(f"{name}.pos={float(row[f'{name}.pos']): .6f}")
    print("[KinovaGripperDiag] " + " ".join(parts))


def main() -> None:
    print("[KinovaGripperDiag] creating SimulationContext")
    sim_cfg = sim_utils.SimulationCfg(dt=float(args_cli.dt), device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[1.5, -1.5, 1.2], target=[0.0, 0.0, 0.4])

    if not args_cli.no_ground:
        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/defaultGroundPlane", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    env_cfg = KinovaTableEnvLearnedCfg()
    robot_cfg = env_cfg.robot.copy().replace(prim_path="/World/Robot")
    if args_cli.usd_override:
        robot_cfg.spawn.usd_path = args_cli.usd_override
        print(f"[KinovaGripperDiag] variant: robot USD override {args_cli.usd_override}")
    if args_cli.urdf_mimic_asset or args_cli.urdf_independent_asset:
        init_joint_pos = dict(robot_cfg.init_state.joint_pos)
        for fixed_joint_name in ("left_outer_finger_joint", "right_outer_finger_joint"):
            init_joint_pos.pop(fixed_joint_name, None)
        robot_cfg.init_state.joint_pos = init_joint_pos

    if args_cli.urdf_mimic_asset:
        # The URDF-derived USD authors PhysX mimic constraints for these joints.
        # Do not also drive them with IsaacLab actuators; only finger_joint should be driven.
        for actuator_name in ("kinova_gripper_finger", "kinova_gripper_passive"):
            actuator = robot_cfg.actuators.get(actuator_name)
            if actuator is not None:
                actuator.stiffness = 0.0
                actuator.damping = 0.0
                actuator.effort_limit_sim = 0.0
        print("[KinovaGripperDiag] variant: using URDF mimic asset compatibility settings")
    if args_cli.urdf_independent_asset:
        hand = robot_cfg.actuators.get("kinova_hand")
        if hand is None:
            raise RuntimeError("kinova_hand actuator is missing")
        hand.joint_names_expr = list(URDF_INDEPENDENT_GRIPPER_JOINT_NAMES)
        hand.effort_limit_sim = 8.0
        hand.velocity_limit_sim = 1.0
        hand.stiffness = 11.25
        hand.damping = 0.1
        robot_cfg.actuators.pop("kinova_gripper_finger", None)
        robot_cfg.actuators.pop("kinova_gripper_passive", None)
        print("[KinovaGripperDiag] variant: using URDF independent gripper joint targets")
    if args_cli.strong_gripper_gains:
        hand = robot_cfg.actuators.get("kinova_hand")
        if hand is not None:
            hand.stiffness = 40.0
            hand.damping = 1.0
            hand.effort_limit_sim = 8.0
            hand.velocity_limit_sim = 1.0
        finger = robot_cfg.actuators.get("kinova_gripper_finger")
        if finger is not None:
            finger.stiffness = 10.0
            finger.damping = 0.05
            finger.effort_limit_sim = 8.0
            finger.velocity_limit_sim = 10.0
        print("[KinovaGripperDiag] variant: using stronger IsaacLab task-style Robotiq gains")
    if args_cli.include_outer_finger_passive:
        passive = robot_cfg.actuators.get("kinova_gripper_passive")
        if passive is None:
            raise RuntimeError("kinova_gripper_passive actuator is missing")
        names = list(passive.joint_names_expr)
        if ".*_outer_finger_joint" not in names:
            names.append(".*_outer_finger_joint")
        passive.joint_names_expr = names
        print("[KinovaGripperDiag] variant: added .*_outer_finger_joint to passive actuator")
    if args_cli.active_passive_gains:
        passive = robot_cfg.actuators.get("kinova_gripper_passive")
        if passive is None:
            raise RuntimeError("kinova_gripper_passive actuator is missing")
        passive.stiffness = 0.2
        passive.damping = 0.001
        passive.effort_limit_sim = 1.0
        passive.velocity_limit_sim = 1.0
        print("[KinovaGripperDiag] variant: using small nonzero PD gains on passive Robotiq joints")
    print("[KinovaGripperDiag] creating Articulation")
    robot = Articulation(cfg=robot_cfg)

    if args_cli.disable_all_collisions:
        stage = get_current_stage()
        disabled_count = 0
        for prim in stage.Traverse():
            if str(prim.GetPath()).startswith("/World/Robot") and prim.HasAPI(UsdPhysics.CollisionAPI):
                collision = UsdPhysics.CollisionAPI(prim)
                collision.CreateCollisionEnabledAttr(False).Set(False)
                disabled_count += 1
        print(f"[KinovaGripperDiag] variant: disabled collision API on {disabled_count} prims")

    print("[KinovaGripperDiag] sim.reset start")
    sim.reset()
    print("[KinovaGripperDiag] sim.reset done")

    root_state = robot.data.default_root_state.clone()
    root_state[:, 2] += float(args_cli.root_z_offset)
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    joint_pos = robot.data.default_joint_pos.clone()
    if args_cli.zero_arm_pose:
        arm_ids_for_zero, _ = robot.find_joints(list(robot_cfg.arm_actuated_joints))
        joint_pos[:, [int(i) for i in arm_ids_for_zero]] = 0.0
    joint_vel = torch.zeros_like(robot.data.default_joint_vel)
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    print("[KinovaGripperDiag] initial joint state written")
    robot.reset()
    print("[KinovaGripperDiag] robot.reset done")
    robot.update(sim.get_physics_dt())
    print("[KinovaGripperDiag] first robot.update done")

    arm_ids, arm_names = robot.find_joints(list(robot_cfg.arm_actuated_joints))
    finger_ids, finger_names = robot.find_joints(["finger_joint"])
    if len(finger_ids) != 1:
        raise RuntimeError(f"Expected exactly one finger_joint, got {finger_names}")
    env_driver_ids: list[int] = []
    env_driver_names: list[str] = []
    env_driver_target_start = -1
    env_stabilizer_ids: list[int] = []
    env_stabilizer_names: list[str] = []
    urdf_independent_ids: list[int] = []
    urdf_independent_names: list[str] = []
    urdf_independent_target_start = -1
    original_coupled_ids: list[int] = []
    original_coupled_names: list[str] = []
    original_coupled_target_start = -1
    original_active_ids: list[int] = []
    original_active_names: list[str] = []
    original_active_target_start = -1
    if args_cli.urdf_independent_asset:
        urdf_independent_ids, urdf_independent_names = _resolve_joint_ids(robot, URDF_INDEPENDENT_GRIPPER_JOINT_NAMES)
        if urdf_independent_names != URDF_INDEPENDENT_GRIPPER_JOINT_NAMES:
            missing = [name for name in URDF_INDEPENDENT_GRIPPER_JOINT_NAMES if name not in urdf_independent_names]
            raise RuntimeError(f"Missing URDF independent gripper joints: {missing}")
        if args_cli.control_finger_only:
            controlled_ids = urdf_independent_ids
            controlled_names = urdf_independent_names
        else:
            controlled_ids = [int(i) for i in arm_ids] + urdf_independent_ids
            controlled_names = list(arm_names) + urdf_independent_names
        hold_targets = robot.data.default_joint_pos[:, controlled_ids].clone()
        urdf_independent_target_start = len(controlled_ids) - len(urdf_independent_ids)
        hold_targets[:, urdf_independent_target_start:] = _urdf_independent_gripper_targets(
            hold_targets, float(args_cli.open_target)
        )
    elif args_cli.original_coupled_targets != "none":
        original_coupled_ids, original_coupled_names = _resolve_joint_ids(
            robot, ORIGINAL_COUPLED_GRIPPER_JOINT_NAMES
        )
        if original_coupled_names != ORIGINAL_COUPLED_GRIPPER_JOINT_NAMES:
            missing = [
                name
                for name in ORIGINAL_COUPLED_GRIPPER_JOINT_NAMES
                if name not in original_coupled_names
            ]
            raise RuntimeError(f"Missing original USD coupled gripper joints: {missing}")
        controlled_ids = [int(i) for i in arm_ids] + original_coupled_ids
        controlled_names = list(arm_names) + original_coupled_names
        hold_targets = robot.data.default_joint_pos[:, controlled_ids].clone()
        original_coupled_target_start = len(controlled_ids) - len(original_coupled_ids)
        hold_targets[:, original_coupled_target_start:] = _original_coupled_gripper_targets(
            hold_targets, float(args_cli.open_target), args_cli.original_coupled_targets
        )
    elif args_cli.original_active_targets != "none":
        original_active_ids, original_active_names = _resolve_joint_ids(
            robot, ORIGINAL_ACTIVE_GRIPPER_JOINT_NAMES
        )
        if original_active_names != ORIGINAL_ACTIVE_GRIPPER_JOINT_NAMES:
            missing = [
                name
                for name in ORIGINAL_ACTIVE_GRIPPER_JOINT_NAMES
                if name not in original_active_names
            ]
            raise RuntimeError(f"Missing original USD active gripper joints: {missing}")
        controlled_ids = [int(i) for i in arm_ids] + original_active_ids
        controlled_names = list(arm_names) + original_active_names
        hold_targets = robot.data.default_joint_pos[:, controlled_ids].clone()
        original_active_target_start = len(controlled_ids) - len(original_active_ids)
        hold_targets[:, original_active_target_start:] = _original_active_gripper_targets(
            hold_targets, float(args_cli.open_target), args_cli.original_active_targets
        )
    elif args_cli.control_finger_only:
        controlled_ids = [int(finger_ids[0])]
        controlled_names = list(finger_names)
        hold_targets = robot.data.default_joint_pos[:, controlled_ids].clone()
        hold_targets[:, 0] = float(args_cli.open_target)
    else:
        env_driver_ids, env_driver_names = _resolve_joint_ids(robot, ENV_DRIVER_GRIPPER_JOINT_NAMES)
        if env_driver_names != ENV_DRIVER_GRIPPER_JOINT_NAMES:
            missing = [name for name in ENV_DRIVER_GRIPPER_JOINT_NAMES if name not in env_driver_names]
            raise RuntimeError(f"Missing env gripper driver joints: {missing}")
        controlled_ids = [int(i) for i in arm_ids] + env_driver_ids
        controlled_names = list(arm_names) + env_driver_names
        hold_targets = robot.data.default_joint_pos[:, controlled_ids].clone()
        env_driver_target_start = len(controlled_ids) - len(env_driver_ids)
        hold_targets[:, env_driver_target_start:] = _env_driver_gripper_targets(
            hold_targets, float(args_cli.open_target)
        )

    gripper_ids, gripper_names = _resolve_joint_ids(robot, GRIPPER_JOINT_NAMES)

    print("[KinovaGripperDiag] robot prim: /World/Robot")
    print(f"[KinovaGripperDiag] controlled joints: {controlled_names}")
    if env_stabilizer_names:
        print(f"[KinovaGripperDiag] stabilizer joints: {env_stabilizer_names}")
    print(f"[KinovaGripperDiag] gripper joints: {gripper_names}")
    print("[KinovaGripperDiag] joint limits:")
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
        if args_cli.urdf_independent_asset:
            targets[:, urdf_independent_target_start:] = _urdf_independent_gripper_targets(targets, command_target)
        elif args_cli.original_coupled_targets != "none":
            targets[:, original_coupled_target_start:] = _original_coupled_gripper_targets(
                targets, command_target, args_cli.original_coupled_targets
            )
        elif args_cli.original_active_targets != "none":
            targets[:, original_active_target_start:] = _original_active_gripper_targets(
                targets, command_target, args_cli.original_active_targets
            )
        else:
            targets[:, env_driver_target_start:] = _env_driver_gripper_targets(targets, command_target)
        if step < 3:
            print(f"[KinovaGripperDiag] step {step} set target start")
        robot.set_joint_position_target(targets, joint_ids=controlled_ids)
        if step < 3:
            print(f"[KinovaGripperDiag] step {step} write_data_to_sim start")
        robot.write_data_to_sim()
        if step < 3:
            print(f"[KinovaGripperDiag] step {step} sim.step start")
        sim.step()
        if step < 3:
            print(f"[KinovaGripperDiag] step {step} robot.update start")
        robot.update(sim_dt)
        if step < 3:
            print(f"[KinovaGripperDiag] step {step} update done")

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
    print(f"[KinovaGripperDiag] wrote CSV: {csv_path}")
    print("[KinovaGripperDiag] final measured positions:")
    for name in gripper_names:
        print(
            f"  {name}: pos={float(final[f'{name}.pos']):.9f}, "
            f"vel={float(final[f'{name}.vel']):.9f}, "
            f"target={float(final[f'{name}.target']):.9f}, "
            f"applied_torque={float(final[f'{name}.applied_torque']):.9f}"
        )


if __name__ == "__main__":
    main()
    print("[KinovaGripperDiag] done")
