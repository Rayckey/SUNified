"""Replay saved FR3FinRay drawer-stack learn-programs program without any LLM calls."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Replay saved FR3FinRay drawer-stack learn-programs stages.")
parser.add_argument(
    "program_run",
    nargs="?",
    default="outputs/manipulation-fr3-finray-drawer-stack/2026-07-07_21-32-09",
    help="Run directory containing program/subtask_* folders, or the program directory itself.",
)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--max_steps_per_stage", type=int, default=1400)
parser.add_argument("--recompute_interval", type=int, default=10)
parser.add_argument("--settle_steps", type=int, default=100)
parser.add_argument("--stop_stage_index", type=int, default=8)
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
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.math import subtract_frame_transforms

from kuafu.mpc.simulation.sun_program_executor import SUNProgramExecutor
from kuafu.isaac.formulation.panda import assert_program_free_panda_env
from kuafu.common.simulation_utils import (
    apply_franka_panda_gripper_config,
    combine_joint_states,
    extract_frame_sensors,
    extract_odom_manipulators,
)


ENV_NAME = "Formulate-Panda-Drawer-Stack-v0"
OUTPUT_ENV_NAME = "manipulation-fr3-finray-drawer-stack"
DEVICE = args_cli.device

DETECTABLE_OBJECTS = {
    "cabinet": (
        "local frame +z is up direction, local frame -y is the forward direction; drawers open approximately 0.14 m; estimated size (approx.): (0.15, 0.25, 0.2).",
        "/World/envs/env_*/Cabinet",
    ),
    "top_drawer_center_frame": (
        "placement target frame at the center of the top drawer interior, estimated size (approx.): (0.15, 0.25, 0.05)",
        "/Visuals/top_drawer_center_frame",
    ),
    "top_drawer_handle_frame": (
        "grasp/contact frame for opening the top drawer; handle is a bar along local x axis; local frame -y is pulling out/open direction for approximately 0.14 m, local frame +y is pushing in/close direction",
        "/Visuals/top_drawer_handle_frame",
    ),
    "strawberry_center_frame": (
        "center frame of a cube-like symmetric strawberry object; treat it as a 0.04 m cube for grasping and placement",
        "/Visuals/strawberry_center_frame",
    ),
    "panda_link4": ("frame of robot link 4", "/Visuals/panda_link4_frame"),
    "panda_fingertip": ("frame of robot tool frame", "/Visuals/panda_hand_frame"),
}


def _stage_index(stage_dir: Path) -> int:
    match = re.fullmatch(r"subtask_(\d+)", stage_dir.name)
    if match is None:
        return 10**9
    return int(match.group(1))


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
    if args_cli.stop_stage_index is not None:
        stage_dirs = [p for p in stage_dirs if _stage_index(p) <= args_cli.stop_stage_index]
    if not stage_dirs:
        raise RuntimeError(f"No subtask_* directories found in {program_dir}")
    for stage_dir in stage_dirs:
        for filename in ("task.txt", "atomic.txt", "constraints.json"):
            if not (stage_dir / filename).exists():
                raise FileNotFoundError(f"Missing {filename} in {stage_dir}")
    return stage_dirs


def reset_stall_counters(program_runtime: SUNProgramExecutor) -> None:
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


def print_predicate_breakdown(program_runtime: SUNProgramExecutor, stage: dict, reason: str) -> None:
    satisfied, predicate_text = program_runtime.quantifyPredicate(text=True, max_envs=1)
    print("=" * 80, flush=True)
    print(f"[replay] Constraint failure predicate breakdown: {reason}", flush=True)
    print(f"[replay] Combined predicate result: {satisfied}", flush=True)
    if predicate_text:
        print(predicate_text, flush=True)
    constraints = stage.get("constraints", {}).get("Constraints", [])
    if constraints:
        print("[replay] Constraint definitions by term index:", flush=True)
        for idx, constraint in enumerate(constraints):
            print(
                f"  - term {idx}: "
                f"op={constraint.get('op')!r} "
                f"condition={constraint.get('condition')!r} "
                f"logic={constraint.get('logic', 'and')!r} "
                f"args={constraint.get('args', {})!r}",
                flush=True,
            )
    else:
        print("[replay] No active constraint definitions available.", flush=True)
    print("=" * 80, flush=True)


def main() -> bool:
    stage_dirs = resolve_stage_dirs(args_cli.program_run)
    print("Offline replay stages:", flush=True)
    for stage_dir in stage_dirs:
        task = (stage_dir / "task.txt").read_text(encoding="utf-8").strip()
        status_path = stage_dir / "status.txt"
        status = status_path.read_text(encoding="utf-8").splitlines()[0] if status_path.exists() else "status: unknown"
        print(f"  {_stage_index(stage_dir)}: {stage_dir.name} {status} {task}", flush=True)

    env_cfg = parse_env_cfg(ENV_NAME, device=DEVICE, num_envs=args_cli.num_envs)
    env = gym.make(ENV_NAME, cfg=env_cfg)
    assert_program_free_panda_env(env)
    env.reset()
    env.unwrapped.scene.update(env.unwrapped.physics_dt)

    script_dir = Path(__file__).resolve().parents[3]
    sim_dt = env_cfg.sim.dt

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
            "horizon_length": 15,
            "time_step": sim_dt,
        },
        "load_program": None,
        "manipulator_high_effort_threshold": 100.0,
        "save_dir": str(script_dir / "outputs" / OUTPUT_ENV_NAME / "offline_replay"),
        "agent": {
            "id": "agent",
            "device": DEVICE,
            "task": "Offline replay of saved FR3FinRay drawer-stack learn-programs program",
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

    def settle():
        if args_cli.settle_steps <= 0:
            return
        hold_actions = env.unwrapped.scene.state["articulation"]["robot"]["joint_position"]
        for _ in range(args_cli.settle_steps):
            env.step(hold_actions)
        env.unwrapped.scene.update(env.unwrapped.physics_dt)
        print(f"Settled for {args_cli.settle_steps} step(s).", flush=True)

    settle()
    update_data()
    program_runtime.controller_api_wrapper.updatePOVState()

    total_steps = 0
    replay_success = True

    for replay_pos, stage_dir in enumerate(stage_dirs, start=1):
        stage_idx = _stage_index(stage_dir)
        update_data()
        program_runtime.controller_api_wrapper.updatePOVState()
        reset_stall_counters(program_runtime)
        stage = program_runtime.load_program_stage(stage_dir)
        program_runtime.applyConstraint(print_function=args_cli.print_function)
        reset_stall_counters(program_runtime)
        task_text = stage.get("task", "")
        print(
            f"[replay] Stage subtask_{stage_idx} ({replay_pos}/{len(stage_dirs)}): {task_text}",
            flush=True,
        )

        u_following = None
        last_computed_step = 0
        recompute_at = 0
        stage_passed = False

        for stage_step in range(args_cli.max_steps_per_stage):
            if not simulation_app.is_running():
                print("[replay] Simulation app stopped.", flush=True)
                replay_success = False
                break

            update_data()
            satisfied = program_runtime.quantifyPredicate()
            if torch.all(satisfied):
                print(
                    f"[replay] Stage subtask_{stage_idx} passed at stage_step={stage_step}, total_steps={total_steps}.",
                    flush=True,
                )
                stage_passed = True
                break

            if u_following is None or stage_step >= recompute_at:
                u_following = program_runtime.computeControl()
                last_computed_step = stage_step
                recompute_at = stage_step + args_cli.recompute_interval
                if program_runtime.hasStalled():
                    reason = program_runtime.stallFailureReason()
                    print(f"[replay] Stage subtask_{stage_idx} failed: {reason}", flush=True)
                    print_predicate_breakdown(program_runtime, stage, reason)
                    replay_success = False
                    break

            action_index = stage_step - last_computed_step + 1
            if action_index >= len(u_following):
                action_index = len(u_following) - 1
            if action_index < 0:
                raise RuntimeError("Controller returned no usable actions.")
            actions = action_to_tensor(u_following[action_index], env_cfg.sim.device)
            env.step(actions)
            total_steps += 1

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
        hold_actions = env.unwrapped.scene.state["articulation"]["robot"]["joint_position"].clone()
        while simulation_app.is_running():
            env.step(hold_actions)

    env.close()
    return replay_success


if __name__ == "__main__":
    success = main()
    simulation_app.close()
    raise SystemExit(0 if success else 1)
