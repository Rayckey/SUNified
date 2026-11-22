#!/usr/bin/env python3
"""Isolated Franka Panda Panda gripper close and force diagnostic.

The script spawns the same stock Franka articulation used by the Panda
learned environments, holds the arm at its initial pose, closes both Panda
fingers, and records the position, velocity, target, and joint effort signals
used by the current ``object_grasped`` predicate.

This is the empty-close baseline. Use the in-environment
``gripper_force_print_every`` monitor for samples with a real task object
between the fingers.
"""

from __future__ import annotations

import argparse
import csv
from functools import partial
from pathlib import Path

from kuafu.paths import resolve_artifact_path

from isaaclab.app import AppLauncher

print = partial(print, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dt",
        type=float,
        default=1.0 / 120.0,
        help="Physics timestep. Defaults to the Panda environment timestep.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=240,
        help="Number of close-command physics steps.",
    )
    parser.add_argument(
        "--settle_steps",
        type=int,
        default=60,
        help="Open-gripper settle steps before closing.",
    )
    parser.add_argument(
        "--close_target",
        type=float,
        default=0.0,
        help="Per-finger closed target in meters.",
    )
    parser.add_argument(
        "--open_target",
        type=float,
        default=0.04,
        help="Per-finger open target in meters.",
    )
    parser.add_argument(
        "--max_grasp_openness",
        type=float,
        default=0.8,
        help="Maximum normalized openness accepted by object_grasped.",
    )
    parser.add_argument(
        "--gripper_max_force",
        type=float,
        default=40.0,
        help="Current generic predicate force scale.",
    )
    parser.add_argument(
        "--panda_bilateral_effort_threshold",
        type=float,
        default=5.0,
        help="Calibrated minimum effort required on each Panda finger.",
    )
    parser.add_argument(
        "--velocity_threshold",
        type=float,
        default=0.2,
        help="Maximum absolute finger speed for joint effort to count as contact.",
    )
    parser.add_argument(
        "--candidate_slope",
        type=float,
        default=0.0,
        help="Optional candidate effort threshold slope versus normalized openness.",
    )
    parser.add_argument(
        "--candidate_intercept",
        type=float,
        default=0.0,
        help="Optional candidate effort threshold intercept.",
    )
    parser.add_argument(
        "--root_z_offset",
        type=float,
        default=0.0,
        help="Add this z offset to the robot root pose.",
    )
    parser.add_argument(
        "--zero_arm_pose",
        action="store_true",
        help="Set the seven arm joints to zero before closing the gripper.",
    )
    parser.add_argument(
        "--no_ground",
        action="store_true",
        help="Do not spawn the ground plane.",
    )
    parser.add_argument(
        "--print_every",
        type=int,
        default=20,
        help="Print one measured row every N steps.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="diagnostics/panda_gripper_close_diag.csv",
        help="CSV output path.",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = False
    return args


args_cli = parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402

from kuafu.isaac.learned.panda.panda_table_env_learned import (  # noqa: E402
    PandaTableEnvLearnedCfg,
)


GRIPPER_JOINT_NAMES = [
    "panda_finger_joint1",
    "panda_finger_joint2",
]


def _as_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def _as_bool(value: torch.Tensor) -> bool:
    return bool(value.detach().cpu().item())


def _record_row(
    robot: Articulation,
    step: int,
    sim_time: float,
    phase: str,
    command_target: float,
    gripper_ids: list[int],
) -> dict[str, float | int | str | bool]:
    positions = robot.data.joint_pos[0, gripper_ids]
    velocities = robot.data.joint_vel[0, gripper_ids]
    computed_effort = robot.data.computed_torque[0, gripper_ids].abs()
    applied_effort = robot.data.applied_torque[0, gripper_ids].abs()

    gripper_range = float(args_cli.open_target) - float(args_cli.close_target)
    if gripper_range <= 0.0:
        raise ValueError("--open_target must be greater than --close_target")
    openness = (
        (positions - float(args_cli.close_target)) / gripper_range
    ).mean()
    max_velocity_abs = velocities.abs().amax()
    min_computed_effort_abs = computed_effort.amin()
    min_applied_effort_abs = applied_effort.amin()

    max_grasp_openness = float(args_cli.max_grasp_openness)
    within_grasp_position = (openness > 0.1) & (
        openness <= max_grasp_openness
    )
    velocity_ok = max_velocity_abs < float(args_cli.velocity_threshold)
    contact_like_effort = torch.where(
        within_grasp_position & velocity_ok,
        computed_effort,
        torch.zeros_like(computed_effort),
    )

    openness_clamped = torch.clamp(
        openness, min=0.03, max=max_grasp_openness
    )
    openness_alpha = (
        (openness_clamped - 0.03) / (max_grasp_openness - 0.03)
    )
    generic_required_force = float(args_cli.gripper_max_force) * (
        0.1 + 0.6 * openness_alpha
    )
    generic_both_force_ok = torch.all(
        contact_like_effort >= generic_required_force
    )

    high_force_threshold = 10.0
    high_contacts = contact_like_effort >= high_force_threshold
    generic_asymmetric_force_ok = (
        (high_contacts[0] ^ high_contacts[1])
        & (openness < 0.3)
        & within_grasp_position
    )
    generic_object_grasped = (
        (generic_both_force_ok & within_grasp_position)
        | generic_asymmetric_force_ok
    )

    panda_required_effort = torch.as_tensor(
        float(args_cli.panda_bilateral_effort_threshold),
        device=contact_like_effort.device,
        dtype=contact_like_effort.dtype,
    )
    both_force_ok = torch.all(
        contact_like_effort >= panda_required_effort
    )
    current_object_grasped = both_force_ok & within_grasp_position

    candidate_required_effort = torch.clamp(
        float(args_cli.candidate_slope) * openness
        + float(args_cli.candidate_intercept),
        min=0.0,
    )
    candidate_contact = (
        within_grasp_position
        & velocity_ok
        & (min_computed_effort_abs >= candidate_required_effort)
    )

    row: dict[str, float | int | str | bool] = {
        "step": int(step),
        "time_s": float(sim_time),
        "phase": phase,
        "commanded_finger_target": float(command_target),
        "openness": _as_float(openness),
        "max_velocity_abs": _as_float(max_velocity_abs),
        "min_computed_effort_abs": _as_float(
            min_computed_effort_abs
        ),
        "min_applied_effort_abs": _as_float(
            min_applied_effort_abs
        ),
        "generic_required_force": _as_float(generic_required_force),
        "panda_required_effort": _as_float(
            panda_required_effort
        ),
        "candidate_required_effort": _as_float(candidate_required_effort),
        "position_ok": _as_bool(within_grasp_position),
        "velocity_ok": _as_bool(velocity_ok),
        "both_force_ok": _as_bool(both_force_ok),
        "generic_both_force_ok": _as_bool(generic_both_force_ok),
        "generic_asymmetric_force_ok": _as_bool(
            generic_asymmetric_force_ok
        ),
        "generic_object_grasped": _as_bool(generic_object_grasped),
        "current_object_grasped": _as_bool(current_object_grasped),
        "candidate_contact": _as_bool(candidate_contact),
    }
    for joint_id, name in zip(gripper_ids, GRIPPER_JOINT_NAMES):
        row[f"{name}.pos"] = _as_float(robot.data.joint_pos[0, joint_id])
        row[f"{name}.vel"] = _as_float(robot.data.joint_vel[0, joint_id])
        row[f"{name}.target"] = _as_float(
            robot.data.joint_pos_target[0, joint_id]
        )
        row[f"{name}.computed_torque"] = _as_float(
            robot.data.computed_torque[0, joint_id]
        )
        row[f"{name}.applied_torque"] = _as_float(
            robot.data.applied_torque[0, joint_id]
        )
    return row


def _print_row(row: dict[str, float | int | str | bool]) -> None:
    print(
        "[PandaGripperDiag] "
        f"step={row['step']} "
        f"t={float(row['time_s']):.3f} "
        f"phase={row['phase']} "
        f"cmd={float(row['commanded_finger_target']):.5f} "
        f"openness={float(row['openness']):.4f} "
        f"min_computed_effort="
        f"{float(row['min_computed_effort_abs']):.4f} "
        f"max_vel={float(row['max_velocity_abs']):.4f} "
        f"generic_required={float(row['generic_required_force']):.4f} "
        f"panda_required={float(row['panda_required_effort']):.4f} "
        f"generic_grasped={row['generic_object_grasped']} "
        f"current_grasped={row['current_object_grasped']} "
        f"candidate_contact={row['candidate_contact']}"
    )


def main() -> None:
    print("[PandaGripperDiag] creating SimulationContext")
    sim_cfg = sim_utils.SimulationCfg(
        dt=float(args_cli.dt),
        device=args_cli.device,
    )
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(
        eye=[1.5, -1.5, 1.2],
        target=[0.0, 0.0, 0.4],
    )

    if not args_cli.no_ground:
        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/defaultGroundPlane", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(
        intensity=3000.0,
        color=(0.75, 0.75, 0.75),
    )
    light_cfg.func("/World/Light", light_cfg)

    env_cfg = PandaTableEnvLearnedCfg()
    robot_cfg = env_cfg.robot.copy().replace(prim_path="/World/Robot")
    print("[PandaGripperDiag] creating Articulation")
    robot = Articulation(cfg=robot_cfg)

    print("[PandaGripperDiag] sim.reset start")
    sim.reset()
    print("[PandaGripperDiag] sim.reset done")

    root_state = robot.data.default_root_state.clone()
    root_state[:, 2] += float(args_cli.root_z_offset)
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])

    joint_pos = robot.data.default_joint_pos.clone()
    arm_ids, arm_names = robot.find_joints(
        list(robot_cfg.arm_actuated_joints)
    )
    if args_cli.zero_arm_pose:
        joint_pos[:, [int(i) for i in arm_ids]] = 0.0
    joint_vel = torch.zeros_like(robot.data.default_joint_vel)
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.reset()
    robot.update(sim.get_physics_dt())

    gripper_ids_raw, gripper_names = robot.find_joints(GRIPPER_JOINT_NAMES)
    gripper_ids = [int(i) for i in gripper_ids_raw]
    if list(gripper_names) != GRIPPER_JOINT_NAMES:
        raise RuntimeError(
            "Expected Panda finger joints "
            f"{GRIPPER_JOINT_NAMES}, got {list(gripper_names)}"
        )

    controlled_ids = [int(i) for i in arm_ids] + gripper_ids
    controlled_names = list(arm_names) + list(gripper_names)
    hold_targets = robot.data.default_joint_pos[:, controlled_ids].clone()
    gripper_target_start = len(controlled_ids) - len(gripper_ids)
    hold_targets[:, gripper_target_start:] = float(args_cli.open_target)

    print(f"[PandaGripperDiag] controlled joints: {controlled_names}")
    print(
        "[PandaGripperDiag] detector settings: "
        f"max_force={float(args_cli.gripper_max_force):.4f} "
        f"max_openness={float(args_cli.max_grasp_openness):.4f} "
        f"velocity_threshold={float(args_cli.velocity_threshold):.4f} "
        f"panda_bilateral_effort_threshold="
        f"{float(args_cli.panda_bilateral_effort_threshold):.4f} "
        f"candidate={float(args_cli.candidate_slope):.4f}*openness"
        f"+{float(args_cli.candidate_intercept):.4f}"
    )
    print("[PandaGripperDiag] gripper joint limits:")
    for joint_id, name in zip(gripper_ids, gripper_names):
        lo = _as_float(robot.data.joint_pos_limits[0, joint_id, 0])
        hi = _as_float(robot.data.joint_pos_limits[0, joint_id, 1])
        print(f"  {name}: [{lo:.6f}, {hi:.6f}]")

    rows: list[dict[str, float | int | str | bool]] = []
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
        targets[:, gripper_target_start:] = command_target
        robot.set_joint_position_target(targets, joint_ids=controlled_ids)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim_dt)

        row = _record_row(
            robot,
            step,
            step * sim_dt,
            phase,
            command_target,
            gripper_ids,
        )
        rows.append(row)
        if step % print_every == 0 or step == total_steps:
            _print_row(row)

    csv_path = resolve_artifact_path(args_cli.csv, "Diagnostic CSV")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    stable_close_rows = [
        row
        for row in rows
        if row["phase"] == "close"
        and bool(row["velocity_ok"])
        and bool(row["position_ok"])
    ]
    print(f"[PandaGripperDiag] wrote CSV: {csv_path}")
    if stable_close_rows:
        efforts = [
            float(row["min_computed_effort_abs"])
            for row in stable_close_rows
        ]
        print(
            "[PandaGripperDiag] stable empty-close "
            "min-computed-effort range: "
            f"[{min(efforts):.6f}, {max(efforts):.6f}] "
            f"over {len(efforts)} samples"
        )
    else:
        print(
            "[PandaGripperDiag] no close samples passed both the "
            "position and velocity gates"
        )


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
    print("[PandaGripperDiag] done")
