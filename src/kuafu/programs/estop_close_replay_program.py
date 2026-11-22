"""Replay saved E-stop-close program and diagnose arm-effort detection."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="Replay saved E-stop-close stages without LLM calls."
)
parser.add_argument(
    "program_run",
    nargs="?",
    default="outputs/learn-programs-estop-close/2026-07-29_06-20-49",
    help="Run directory containing program/subtask_* folders, or the program directory itself.",
)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--max_steps_per_stage", type=int, default=1000)
parser.add_argument("--recompute_interval", type=int, default=5)
parser.add_argument("--settle_steps", type=int, default=100)
parser.add_argument("--stop_stage_index", type=int, default=2)
parser.add_argument("--diagnostic_stage_index", type=int, default=2)
parser.add_argument("--effort_print_interval", type=int, default=25)
parser.add_argument("--print_function", action="store_true")
parser.add_argument("--keep_running", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from omegaconf import OmegaConf

import isaaclab_tasks  # noqa: F401
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab_tasks.utils import parse_env_cfg

from kuafu.mpc.simulation.sun_program_executor import SUNProgramExecutor
from kuafu.isaac.formulation.panda import assert_program_free_panda_env
from kuafu.common.simulation_utils import (
    apply_franka_panda_gripper_config,
    combine_joint_states,
    extract_frame_sensors,
    extract_odom_manipulators,
)


ENV_NAME = "Formulate-Panda-EStop-Close-v0"
OUTPUT_ENV_NAME = "manipulation-estop-close"
DEVICE = args_cli.device

DETECTABLE_OBJECTS = {
    "estop_cap_tip": (
        "tip frame of mushroom cap of the twist open e-stop button for closing, "
        "press against local +z axis for 3 cm to close, size:(0.04 m, 0.04 m, 0.04 m)",
        "/Visuals/estop_cap_tip_frame",
    ),
    "estop_cap_center": (
        "center frame of mushroom cap of the twist open e-stop button for unlocking, "
        "twist around +z for openings for 45 degrees to unlock; size:(0.04 m, 0.04 m, 0.04 m)",
        "/Visuals/estop_cap_center_frame",
    ),
    "estop_base_frame": (
        "rectangular housing of the e-stop button; size: (0.078, 0.072, 0.073) m",
        "/Visuals/estop_base_frame",
    ),
    "table": (
        "surface the e-stop is mounted on; size: (0.4, 0.6, 0.2) m",
        "/World/envs/env_*/Table",
    ),
    "panda_link4": ("frame of robot link 4", "/Visuals/panda_link4_frame"),
    "panda_fingertip": (
        "frame of robot tool / end-effector",
        "/Visuals/panda_hand_frame",
    ),
}


def _stage_index(stage_dir: Path) -> int:
    match = re.fullmatch(r"subtask_(\d+)", stage_dir.name)
    return int(match.group(1)) if match is not None else 10**9


def resolve_stage_dirs(path_arg: str) -> list[Path]:
    root = Path(path_arg).expanduser()
    if not root.is_absolute():
        root = Path.cwd() / root
    program_dir = root if root.name == "program" else root / "program"
    if not program_dir.exists():
        raise FileNotFoundError(f"Program directory does not exist: {program_dir}")
    stage_dirs = sorted(
        (p for p in program_dir.iterdir() if p.is_dir() and p.name.startswith("subtask_")),
        key=_stage_index,
    )
    stage_dirs = [
        stage_dir
        for stage_dir in stage_dirs
        if _stage_index(stage_dir) <= args_cli.stop_stage_index
    ]
    if not stage_dirs:
        raise RuntimeError(f"No subtask_* directories found in {program_dir}")
    for stage_dir in stage_dirs:
        for filename in ("task.txt", "atomic.txt", "constraints.json"):
            if not (stage_dir / filename).exists():
                raise FileNotFoundError(f"Missing {filename} in {stage_dir}")
    return stage_dirs


def reset_effort_and_stall_counters(program_runtime: SUNProgramExecutor) -> None:
    wrapper = program_runtime.controller_api_wrapper
    wrapper.stall_joint_counter = None
    wrapper.stall_cost_counter = None
    if hasattr(wrapper, "gripper_stall_counter"):
        wrapper.gripper_stall_counter = None
    if hasattr(wrapper, "_gripper_stall_state"):
        wrapper._gripper_stall_state = None


def action_to_tensor(action, device: str):
    if torch.is_tensor(action):
        return action.to(dtype=torch.float32, device=device)
    return torch.as_tensor(action, dtype=torch.float32, device=device)


def arm_actuator_effort(robot, attribute: str) -> torch.Tensor:
    return torch.cat(
        (
            getattr(robot.actuators["panda_shoulder"], attribute),
            getattr(robot.actuators["panda_forearm"], attribute),
        ),
        dim=1,
    )


def scalar(value) -> float:
    return float(value.detach().cpu().item())


def vector_text(value: torch.Tensor) -> str:
    values = value.detach().cpu().flatten().tolist()
    return "[" + ", ".join(f"{item:.3f}" for item in values) + "]"


def predicate_counter(program_runtime: SUNProgramExecutor) -> int:
    counter = program_runtime.controller_api_wrapper.stall_cost_counter
    if counter is None:
        return 0
    return int(counter.detach().cpu().flatten()[0].item())


def print_predicate_breakdown(program_runtime: SUNProgramExecutor, stage: dict, reason: str) -> None:
    satisfied, predicate_text = program_runtime.quantifyPredicate(text=True, max_envs=1)
    print("=" * 80, flush=True)
    print(f"[replay] Constraint failure predicate breakdown: {reason}", flush=True)
    print(f"[replay] Combined predicate result: {satisfied}", flush=True)
    if predicate_text:
        print(predicate_text, flush=True)
    constraints = stage.get("constraints", {}).get("Constraints", [])
    for idx, constraint in enumerate(constraints):
        print(
            f"  - term {idx}: op={constraint.get('op')!r} "
            f"condition={constraint.get('condition')!r} "
            f"logic={constraint.get('logic', 'and')!r} "
            f"args={constraint.get('args', {})!r}",
            flush=True,
        )
    print("=" * 80, flush=True)


def main() -> bool:
    stage_dirs = resolve_stage_dirs(args_cli.program_run)
    print("Offline replay stages:", flush=True)
    for stage_dir in stage_dirs:
        task = (stage_dir / "task.txt").read_text(encoding="utf-8").strip()
        print(f"  {_stage_index(stage_dir)}: {stage_dir.name} {task}", flush=True)

    env_cfg = parse_env_cfg(ENV_NAME, device=DEVICE, num_envs=args_cli.num_envs)
    env = gym.make(ENV_NAME, cfg=env_cfg)
    assert_program_free_panda_env(env)
    env.reset()
    env.unwrapped.scene.update(env.unwrapped.physics_dt)

    script_dir = Path(__file__).resolve().parents[3]
    sim_dt = env.unwrapped.dt
    robot = env.unwrapped.scene["robot"]

    diff_ik_cfg = DifferentialIKControllerCfg(
        command_type="pose", use_relative_mode=False, ik_method="dls"
    )
    diff_ik_controller = DifferentialIKController(
        diff_ik_cfg,
        num_envs=env.unwrapped.scene.num_envs,
        device=env_cfg.sim.device,
    )
    robot_entity_cfg = SceneEntityCfg(
        "robot", joint_names=["panda_joint.*"], body_names=["panda_hand"]
    )
    robot_entity_cfg.resolve(env.unwrapped.scene)
    ee_body_id = robot_entity_cfg.body_ids[0]
    ee_jacobi_idx = ee_body_id - 1

    def inverse_kinematics(p_des, quat_des, q_init=None, frame_name="tool"):
        diff_ik_controller.set_command(
            torch.concatenate(
                (
                    torch.as_tensor(p_des, dtype=torch.float32, device=env_cfg.sim.device),
                    torch.as_tensor(quat_des, dtype=torch.float32, device=env_cfg.sim.device),
                ),
                dim=-1,
            )
        )
        jacobian = robot.root_physx_view.get_jacobians()[
            :, ee_jacobi_idx, :, robot_entity_cfg.joint_ids
        ]
        ee_pose_w = robot.data.body_pose_w[:, ee_body_id]
        root_pose_w = robot.data.root_pose_w
        joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3],
            root_pose_w[:, 3:7],
            ee_pose_w[:, 0:3],
            ee_pose_w[:, 3:7],
        )
        joint_pos_des = diff_ik_controller.compute(
            ee_pos_b, ee_quat_b, jacobian, joint_pos
        )
        return joint_pos_des.detach().cpu()

    mpc_config = {
        "controller_name": "FrankaIKController",
        "device": DEVICE,
        "controller": {
            "urdf_path": f"{script_dir}/resources/robots/franka_description/robots/panda_arm_hand_fixed.urdf",
            "ee_frame": "panda_fingertip",
            "horizon_length": 6,
            "time_step": sim_dt,
        },
        "load_program": None,
        "manipulator_high_effort_threshold": 40.0,
        "save_dir": str(script_dir / "outputs" / OUTPUT_ENV_NAME / "offline_replay"),
        "agent": {
            "id": "agent",
            "device": DEVICE,
            "task": "Offline replay of saved E-stop-close learn-programs program",
            "prompt_dir": f"{script_dir}/assets/prompts/Manipulation",
            "initial_residual": f"{script_dir}/src/kuafu/skills/constraints/residuals.py",
            "initial_predicate": f"{script_dir}/src/kuafu/skills/constraints/predicates.py",
            "model_version": "offline",
            "load_path": None,
            "reuse_program": True,
            "max_correction_steps": 0,
        },
    }
    apply_franka_panda_gripper_config(mpc_config)
    program_runtime = SUNProgramExecutor(OmegaConf.create(mpc_config))
    program_runtime.active_controller._ik_fun = inverse_kinematics
    program_runtime.test_controller._ik_fun = inverse_kinematics

    def update_data():
        odom = extract_odom_manipulators(env, DETECTABLE_OBJECTS)
        sensors = extract_frame_sensors(env, {}, DETECTABLE_OBJECTS)
        program_runtime.callbackOdom(odom, sensors)
        program_runtime.callbackJointState(combine_joint_states(env, robot="franka"))

    if args_cli.settle_steps > 0:
        hold_actions = env.unwrapped.scene.state["articulation"]["robot"][
            "joint_position"
        ]
        for _ in range(args_cli.settle_steps):
            env.step(hold_actions)
        env.unwrapped.scene.update(env.unwrapped.physics_dt)
        print(f"Settled for {args_cli.settle_steps} step(s).", flush=True)

    update_data()
    program_runtime.controller_api_wrapper.updatePOVState()
    effort_threshold = program_runtime.controller_api_wrapper._manipulator_high_effort_threshold
    total_steps = 0
    replay_success = True

    for replay_pos, stage_dir in enumerate(stage_dirs, start=1):
        stage_idx = _stage_index(stage_dir)
        diagnostic_stage = stage_idx == args_cli.diagnostic_stage_index
        update_data()
        program_runtime.controller_api_wrapper.updatePOVState()
        reset_effort_and_stall_counters(program_runtime)
        stage = program_runtime.load_program_stage(stage_dir)
        program_runtime.applyConstraint(print_function=args_cli.print_function)
        reset_effort_and_stall_counters(program_runtime)
        print(
            f"[replay] Stage subtask_{stage_idx} ({replay_pos}/{len(stage_dirs)}): "
            f"{stage.get('task', '')}",
            flush=True,
        )
        if diagnostic_stage:
            print(
                f"[effort] diagnostic stage={stage_idx} threshold={effort_threshold:.3f} "
                "required_consecutive_steps=5",
                flush=True,
            )

        u_following = None
        last_computed_step = 0
        recompute_at = 0
        stage_passed = False
        max_filtered = -1.0
        max_raw = -1.0
        max_applied = -1.0
        max_filtered_step = -1
        max_filtered_vector = None
        raw_at_filtered_peak = -1.0
        applied_at_filtered_peak = -1.0
        consecutive_above = 0
        longest_above = 0
        max_predicate_counter = 0

        for stage_step in range(args_cli.max_steps_per_stage):
            if not simulation_app.is_running():
                replay_success = False
                break

            update_data()
            if diagnostic_stage:
                wrapper_effort = program_runtime.controller_api_wrapper.joint_dict["agent"][
                    "joint_effort"
                ][0, :7]
                raw_effort = arm_actuator_effort(robot, "computed_effort")[0]
                applied_effort = arm_actuator_effort(robot, "applied_effort")[0]
                filtered_sum = scalar(torch.sum(torch.abs(wrapper_effort)))
                raw_sum = scalar(torch.sum(torch.abs(raw_effort)))
                applied_sum = scalar(torch.sum(torch.abs(applied_effort)))
                above = filtered_sum > effort_threshold
                consecutive_above = consecutive_above + 1 if above else 0
                longest_above = max(longest_above, consecutive_above)
                max_raw = max(max_raw, raw_sum)
                max_applied = max(max_applied, applied_sum)
                if filtered_sum > max_filtered:
                    max_filtered = filtered_sum
                    max_filtered_step = stage_step
                    max_filtered_vector = wrapper_effort.detach().cpu().clone()
                    raw_at_filtered_peak = raw_sum
                    applied_at_filtered_peak = applied_sum

            satisfied = program_runtime.quantifyPredicate()

            if diagnostic_stage:
                current_predicate_counter = predicate_counter(program_runtime)
                max_predicate_counter = max(max_predicate_counter, current_predicate_counter)
                should_print = (
                    stage_step % max(args_cli.effort_print_interval, 1) == 0
                    or current_predicate_counter >= 5
                )
                if should_print:
                    print(
                        f"[effort] step={stage_step:04d} filtered_sum={filtered_sum:.3f} "
                        f"raw_computed_sum={raw_sum:.3f} applied_sum={applied_sum:.3f} "
                        f"above_threshold={above} consecutive_above={consecutive_above} "
                        f"predicate_counter={current_predicate_counter}",
                        flush=True,
                    )

            if torch.all(satisfied):
                print(
                    f"[replay] Stage subtask_{stage_idx} passed at stage_step={stage_step}, "
                    f"total_steps={total_steps}.",
                    flush=True,
                )
                stage_passed = True
                break

            if u_following is None or stage_step >= recompute_at:
                u_following = program_runtime.computeControl()
                last_computed_step = stage_step
                recompute_at = stage_step + args_cli.recompute_interval

            action_index = stage_step - last_computed_step + 1
            if action_index >= len(u_following):
                action_index = len(u_following) - 1
            if action_index < 0:
                raise RuntimeError("Controller returned no usable actions.")
            actions = action_to_tensor(u_following[action_index], env_cfg.sim.device)
            env.step(actions)
            total_steps += 1

        if diagnostic_stage:
            print(
                f"[effort-summary] stage={stage_idx} threshold={effort_threshold:.3f} "
                f"max_filtered_sum={max_filtered:.3f} max_filtered_step={max_filtered_step} "
                f"raw_sum_at_filtered_peak={raw_at_filtered_peak:.3f} "
                f"applied_sum_at_filtered_peak={applied_at_filtered_peak:.3f} "
                f"max_raw_computed_sum={max_raw:.3f} max_applied_sum={max_applied:.3f} "
                f"longest_consecutive_above={longest_above} "
                f"max_predicate_counter={max_predicate_counter} "
                f"high_effort_triggered={max_predicate_counter >= 5}",
                flush=True,
            )
            if max_filtered_vector is not None:
                print(
                    "[effort-summary] filtered_per_joint_at_peak="
                    f"{vector_text(max_filtered_vector)}",
                    flush=True,
                )

        if not stage_passed:
            if replay_success:
                reason = f"timed out after {args_cli.max_steps_per_stage} step(s)"
                print(f"[replay] Stage subtask_{stage_idx} {reason}.", flush=True)
                print_predicate_breakdown(program_runtime, stage, reason)
            replay_success = False
            break

    print(f"[replay] finished success={replay_success} total_steps={total_steps}", flush=True)

    if args_cli.keep_running and simulation_app.is_running():
        print("[replay] keep_running enabled; close Isaac Sim to exit.", flush=True)
        hold_actions = env.unwrapped.scene.state["articulation"]["robot"][
            "joint_position"
        ].clone()
        while simulation_app.is_running():
            env.step(hold_actions)

    env.close()
    return replay_success


if __name__ == "__main__":
    success = main()
    simulation_app.close()
    raise SystemExit(0 if success else 1)
