from kuafu.paths import program_output_root
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to an environment with random action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
import numpy as np
from pathlib import Path
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric",
    type=bool,
    default=True,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to simulate."
)
parser.add_argument(
    "--max_atomic_corrections",
    type=int,
    default=5,
    help="Maximum automatic restart/replay atomic-action correction attempts per failed stage.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
from isaaclab_tasks.utils import parse_env_cfg
from kuafu.isaac.formulation.panda import assert_program_free_panda_env


ENV_NAME = "Formulate-Panda-Drawer-v0"
OUTPUT_ENV_NAME = "manipulation-fr3-finray-drawer"


import os
import isaaclab_tasks
import datetime
from kuafu.mpc.simulation.sun_program_executor import SUNProgramExecutor
from omegaconf import OmegaConf
from isaaclab.managers import SceneEntityCfg
from kuafu.common.simulation_utils import (
    apply_franka_panda_gripper_config,
    compile_odom_manipulators,
    update_meta_in_controller,
    write_txt,
    combine_joint_states,
    compile_frame_sensors,
    extract_odom_manipulators,
    extract_frame_sensors,
)
from kuafu.common.simulation_utils import (
    update_sun_program_executor_data_for_env,
    create_differential_ik_pose_function,
)

from isaaclab.utils.math import subtract_frame_transforms
from omni.usd import get_prim_at_path
import time
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg

detectable_objects = {
    "cabinet": (
        "mesh of the cabinet, local frame +z is up direction, local frame -y is the forward direction, estimated size: (0.15, 0.25, 0.2)",
        "/World/envs/env_*/Cabinet",
    ),
    "top_drawer_center_frame": (
        "center frame of top drawer, estimated size: (0.15, 0.25, 0.05)",
        "/Visuals/top_drawer_center_frame",
    ),
    "top_drawer_handle_frame": (
            "frame of top drawer's handle, handle is a bar along local frame x axis, local frame -y is pulling out direction",
            "/Visuals/top_drawer_handle_frame",
    ),
    "panda_link4": (
        "frame of robot link 4",
        "/Visuals/panda_link4_frame",
    ),
    "panda_fingertip": (
        "frame of robot tool frame",
        "/Visuals/panda_hand_frame",
    ),
}
DEVICE = args_cli.device


def main():
    """Random actions agent with Isaac Lab environment."""
    script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    GLOBAL_OUTPUT_DIR = os.path.join(
        str(program_output_root()), f"{OUTPUT_ENV_NAME}", f"{timestamp}"
    )
    os.makedirs(GLOBAL_OUTPUT_DIR, exist_ok=True)

    rgb_save_dir = os.path.join(GLOBAL_OUTPUT_DIR, "images")
    os.makedirs(rgb_save_dir, exist_ok=True)

    def make_program_runtime_for_env(env, env_cfg):
        sim_dt = env_cfg.sim.dt
        inverse_kinematics = create_differential_ik_pose_function(
            env, env_cfg, "panda"
        )

        mpc_config = {
            "controller_name": "FrankaIKController",
            "device": DEVICE,
            "controller": {
                "urdf_path": (
                    f"{script_dir}/resources/robots/franka_description/robots/"
                    "panda_arm_hand_fixed.urdf"
                ),
                "ee_frame": "panda_fingertip",
                "horizon_length": 50,
                "time_step": sim_dt,
            },
            "load_program": None,
        "manipulator_high_effort_threshold": 100.0,
            "agent": {
                "id": "agent",
                "device": DEVICE,
                "task": "open the top drawer by 14 cm",
                "prompt_dir": f"{script_dir}/assets/prompts/Manipulation",
                "initial_residual": (
                    f"{script_dir}/src/kuafu/skills/constraints/residuals.py"
                ),
                "initial_predicate": (
                    f"{script_dir}/src/kuafu/skills/constraints/predicates.py"
                ),
                "model_version": "gpt-5.5",
                "load_path": None,
                "reuse_program": True,
                "max_correction_steps": 5,
            },
        }
        mpc_config["save_dir"] = GLOBAL_OUTPUT_DIR
        apply_franka_panda_gripper_config(mpc_config)
        cfg = OmegaConf.create(mpc_config)
        program_runtime = SUNProgramExecutor(cfg)
        program_runtime.active_controller._ik_fun = inverse_kinematics
        program_runtime.test_controller._ik_fun = inverse_kinematics
        update_sun_program_executor_data_for_env(env, program_runtime, "panda")
        return program_runtime

    RESET_SETTLE_STEPS = 100

    def settle_after_reset(env_to_settle):
        hold_actions = env_to_settle.unwrapped.scene.state["articulation"]["robot"][
            "joint_position"
        ]
        for _ in range(RESET_SETTLE_STEPS):
            env_to_settle.step(hold_actions)
        print(f"Settled reset for {RESET_SETTLE_STEPS} step(s).", flush=True)

    def make_env_and_program_runtime():
        env_cfg = parse_env_cfg(
            ENV_NAME,
            device=DEVICE,
            num_envs=args_cli.num_envs,
        )

        env = gym.make(ENV_NAME, cfg=env_cfg)

        assert_program_free_panda_env(env)
        env.reset()
        settle_after_reset(env)
        program_runtime = make_program_runtime_for_env(env, env_cfg)
        return env, env_cfg, program_runtime

    env, env_cfg, program_runtime = make_env_and_program_runtime()

    count = 0
    total_count = 0
    last_computed_count = 0
    subtasks_count = 0
    task_completed = False
    CONSTRAINT_CALL_INTERVAL = 700
    RECOMPUTE_CALL_INTERVAL = 10
    MAX_COUNTER = None
    MAX_FAIL_FOR_TASK = args_cli.max_atomic_corrections
    print(f"Max atomic-action correction attempts per failed stage: {MAX_FAIL_FOR_TASK}", flush=True)

    recompute_call_counter = 0
    current_stage_start_count = 0
    u_following = None

    need_task_plan = True
    need_atomic_actions = False
    need_constraint = False
    stage_running = False
    active_stage_recorded = False
    active_stage_dir = None

    working_stage_dirs = []
    successful_constraint_counts = []
    replay_stage_dirs = []
    replay_index = 0
    replaying = False
    failed_stage_snapshot = None
    failed_error_prompt = None
    failed_retry_count = 0
    correction_iterations = 0
    exit_reason = "Simulation stopped"

    start = time.time()

    def reset_control_schedule():
        nonlocal last_computed_count, recompute_call_counter, u_following
        last_computed_count = count
        recompute_call_counter = count
        u_following = None

    def episode_lengths():
        if not hasattr(env.unwrapped, "episode_length_buf"):
            return None
        return env.unwrapped.episode_length_buf.clone()

    def _truthy_tensor(value) -> bool:
        if value is None:
            return False
        if torch.is_tensor(value):
            return bool(torch.any(value).item())
        return bool(value)

    def env_success_reason(step_result, before_episode_lengths):
        env_success = False
        if isinstance(step_result, tuple) and len(step_result) >= 5:
            env_success = (
                _truthy_tensor(step_result[2])
                or _truthy_tensor(step_result[3])
            )

        after_episode_lengths = episode_lengths()
        episode_counter_reset = (
            before_episode_lengths is not None
            and after_episode_lengths is not None
            and torch.any(after_episode_lengths < before_episode_lengths)
        )
        if env_success or episode_counter_reset:
            return (
                "Environment reported task success; learn-programs run finished "
                "successfully."
            )
        return None

    def current_llm_usage():
        learner = program_runtime.program_learner._constraint_generator
        prompt_manager = learner.prompt_manager
        if hasattr(prompt_manager, "usage_summary"):
            return prompt_manager.usage_summary()
        return {
            "calls": getattr(prompt_manager, "total_api_calls", 0),
            "prompt_tokens": getattr(prompt_manager, "prompt_tokens", 0),
            "completion_tokens": getattr(prompt_manager, "completion_tokens", 0),
            "total_tokens": getattr(prompt_manager, "total_tokens", 0),
        }

    llm_vlm_calls = 0
    llm_prompt_tokens = 0
    llm_completion_tokens = 0
    llm_total_tokens = 0

    def accumulate_llm_usage(before_usage):
        nonlocal llm_vlm_calls, llm_prompt_tokens
        nonlocal llm_completion_tokens, llm_total_tokens
        after_usage = current_llm_usage()
        llm_vlm_calls += max(after_usage["calls"] - before_usage["calls"], 0)
        llm_prompt_tokens += max(
            after_usage["prompt_tokens"] - before_usage["prompt_tokens"], 0
        )
        llm_completion_tokens += max(
            after_usage["completion_tokens"] - before_usage["completion_tokens"], 0
        )
        llm_total_tokens += max(
            after_usage["total_tokens"] - before_usage["total_tokens"], 0
        )


    def count_constraint_terms(constraints) -> int:
        if isinstance(constraints, dict):
            terms = constraints.get("Constraints", [])
        elif isinstance(constraints, list):
            terms = constraints
        else:
            return 0
        return len(terms) if isinstance(terms, list) else 0

    def current_constraint_count() -> int:
        learner = program_runtime.program_learner._constraint_generator
        return count_constraint_terms(getattr(learner, "_constraints", {}))

    def total_successful_constraints() -> int:
        return sum(
            count for count in successful_constraint_counts if count is not None
        )

    def stats_lines(status: str, reason: str):
        return [
            f"status: {status}",
            f"reason: {reason}",
            f"iterations: {correction_iterations}",
            f"correction_iterations: {correction_iterations}",
            f"control_steps: {total_count}",
            f"current_count: {count}",
            f"completed_subtasks: {subtasks_count}",
            f"total_constraints: {total_successful_constraints()}",
            f"llm_vlm_calls: {llm_vlm_calls}",
            f"prompt_tokens: {llm_prompt_tokens}",
            f"completion_tokens: {llm_completion_tokens}",
            f"total_tokens: {llm_total_tokens}",
        ]

    def write_run_stats(status: str, reason: str):
        lines = stats_lines(status, reason)
        write_txt(os.path.join(GLOBAL_OUTPUT_DIR, "stats.txt"), "\n".join(lines) + "\n")
        print("Learn-programs run stats:", flush=True)
        for line in lines:
            print(f"  {line}", flush=True)

    def abort_loop(message: str):
        print(message, flush=True)
        write_txt(os.path.join(GLOBAL_OUTPUT_DIR, "success.txt"), "False")
        write_run_stats("failed", message)
        raise RuntimeError(message)

    def write_stage_status(stage_dir, status: str, reason: str | None = None):
        lines = [
            f"status: {status}",
            f"subtask_index: {subtasks_count}",
            f"count: {count}",
            f"total_count: {total_count}",
        ]
        if reason is not None:
            lines.append(f"reason: {reason}")
        write_txt(stage_dir / "status.txt", "\n".join(lines) + "\n")

    def record_current_stage(status: str, reason: str | None = None):
        nonlocal active_stage_recorded, active_stage_dir
        program_save_dir = Path(
            os.path.join(
                GLOBAL_OUTPUT_DIR,
                "program",
                "subtask_" + str(subtasks_count),
            )
        )
        os.makedirs(program_save_dir, exist_ok=True)
        program_runtime.record_program(
            program_save_dir / "task.txt",
            program_save_dir / "atomic.txt",
            program_save_dir / "constraints.json",
        )
        write_stage_status(program_save_dir, status, reason)
        active_stage_dir = program_save_dir
        active_stage_recorded = True
        print(
            f"Recorded {status} stage program: {program_save_dir}",
            flush=True,
        )
        return program_save_dir

    def mark_current_stage_failed(reason: str):
        if active_stage_recorded and active_stage_dir is not None:
            write_stage_status(active_stage_dir, "failed", reason)
            print(
                f"Marked failed stage program: {active_stage_dir}",
                flush=True,
            )
        else:
            record_current_stage("failed", reason)

    def save_current_stage_as_working():
        stage_dir = active_stage_dir
        if not active_stage_recorded or stage_dir is None:
            stage_dir = record_current_stage("succeeded")
        else:
            write_stage_status(stage_dir, "succeeded")
            print(
                f"Marked successful stage program: {stage_dir}",
                flush=True,
            )
        while len(working_stage_dirs) <= subtasks_count:
            working_stage_dirs.append(None)
        working_stage_dirs[subtasks_count] = stage_dir
        while len(successful_constraint_counts) <= subtasks_count:
            successful_constraint_counts.append(None)
        successful_constraint_counts[subtasks_count] = current_constraint_count()

    def print_failure_predicate_breakdown(reason: str):
        satisfied, predicate_text = program_runtime.quantifyPredicate(text=True, max_envs=1)
        snapshot = program_runtime.snapshot_program_stage()
        constraints = snapshot.get("constraints", {}).get("Constraints", [])
        print("=" * 80, flush=True)
        print(f"Constraint failure predicate breakdown: {reason}", flush=True)
        print(f"Combined predicate result: {satisfied}", flush=True)
        if predicate_text:
            print(predicate_text, flush=True)
        else:
            print("No per-term predicate text available.", flush=True)
        if constraints:
            print("Constraint definitions by term index:", flush=True)
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
            print("No active constraint definitions available.", flush=True)
        print("=" * 80, flush=True)

    def build_failure_prompt(reason: str, snapshot: dict) -> str:
        lines = [
            "The previous grounded atomic action failed during execution.",
            f"Failure reason: {reason}",
            "Keep the existing high-level subtask. Correct the atomic action natural-language grounding; constraints will be regenerated afterward.",
            f"Current subtask: {snapshot.get('task', '')}",
            "Current atomic actions:",
        ]
        atomic_actions = snapshot.get("atomic_actions", [])
        if atomic_actions:
            lines.extend(f"- {action}" for action in atomic_actions)
        else:
            lines.append("- None recorded")

        stalled = program_runtime.hasStalled()
        if hasattr(stalled, "detach"):
            stalled = torch.any(stalled).item()
        elif hasattr(stalled, "item"):
            stalled = stalled.item()
        stalled = bool(stalled)
        lines.append(f"Controller stalled: {stalled}")
        if stalled:
            lines.append(f"Stall failure reason: {program_runtime.stallFailureReason()}")

        collision = program_runtime.hasCollision()
        if hasattr(collision, "item"):
            collision = collision.item()
        lines.append(f"High manipulator effort/contact: {bool(collision)}")

        satisfied, predicate_text = program_runtime.quantifyPredicate(text=True, max_envs=1)
        lines.append(f"Combined predicate result: {satisfied}")
        if predicate_text:
            lines.append(predicate_text)

        lines.append("Previous failed grounding:")
        lines.append(str(snapshot.get("constraints", {})))
        return "\n".join(lines)

    def load_replay_stage():
        nonlocal stage_running, need_task_plan, need_atomic_actions
        nonlocal need_constraint, active_stage_recorded, active_stage_dir
        nonlocal current_stage_start_count
        path = replay_stage_dirs[replay_index]

        # Match collect_from_mpc.reset_sun_program_executor(): refresh SUNProgramExecutor from the
        # current physical state, snapshot POV, then assign exactly one stage.
        update_sun_program_executor_data_for_env(env, program_runtime, "panda")
        program_runtime.controller_api_wrapper.updatePOVState()
        program_runtime.controller_api_wrapper.stall_joint_counter = None
        program_runtime.controller_api_wrapper.stall_cost_counter = None
        program_runtime.controller_api_wrapper.gripper_stall_counter = None
        program_runtime.controller_api_wrapper._gripper_stall_state = None

        stage = program_runtime.load_program_stage(path)
        program_runtime.applyConstraint()

        stage_running = True
        need_task_plan = False
        need_atomic_actions = False
        need_constraint = False
        active_stage_recorded = True
        active_stage_dir = path
        current_stage_start_count = count
        reset_control_schedule()
        task_text = stage.get("task", "")
        print(
            f"Replaying working constraint {replay_index + 1}/"
            f"{len(replay_stage_dirs)}: {task_text}",
            flush=True,
        )

    def restore_failed_stage_for_correction():
        nonlocal stage_running, need_task_plan, need_atomic_actions
        nonlocal need_constraint, replaying, current_stage_start_count
        if failed_stage_snapshot is None:
            abort_loop("No failed stage snapshot is available for correction")
        update_sun_program_executor_data_for_env(env, program_runtime, "panda")
        program_runtime.controller_api_wrapper.updatePOVState()
        program_runtime.set_program_stage(
            failed_stage_snapshot["task"],
            failed_stage_snapshot["atomic_actions"],
            failed_stage_snapshot["constraints"],
            pending_tasks=failed_stage_snapshot.get("pending_tasks", []),
            completed_tasks=failed_stage_snapshot.get("completed_tasks", []),
            completed_task_atomic_actions=failed_stage_snapshot.get(
                "completed_task_atomic_actions", []
            ),
        )
        stage_running = False
        need_task_plan = False
        need_atomic_actions = True
        need_constraint = False
        replaying = False
        current_stage_start_count = count
        reset_control_schedule()
        print(
            "Reached the failed stage after replay; sending runtime failure "
            "prompt to correct the atomic action."
        )

    def restart_from_failure(reason: str):
        nonlocal env, env_cfg, program_runtime, count, replay_stage_dirs, replay_index
        nonlocal subtasks_count, replaying, failed_stage_snapshot, failed_error_prompt
        nonlocal failed_retry_count, stage_running, active_stage_recorded
        nonlocal active_stage_dir
        nonlocal need_task_plan, need_atomic_actions, need_constraint
        if replaying:
            print(
                "A previously working replay stage failed before reaching the "
                "original failed constraint; treating that replay stage as the "
                "new atomic-action correction target.",
                flush=True,
            )
            subtasks_count = replay_index
            replaying = False
            reason = f"previously working replay stage {replay_index} failed: {reason}"

        print_failure_predicate_breakdown(reason)
        mark_current_stage_failed(reason)
        if failed_retry_count >= MAX_FAIL_FOR_TASK:
            message = "Maximum automatic restarts reached for the current atomic action"
            abort_loop(message)

        failed_stage_snapshot = program_runtime.snapshot_program_stage()
        failed_error_prompt = build_failure_prompt(reason, failed_stage_snapshot)
        failed_retry_count += 1
        replay_stage_dirs = [
            stage_dir
            for stage_dir in working_stage_dirs[:subtasks_count]
            if stage_dir is not None
        ]
        replay_index = 0
        print(
            f"Constraint failed ({reason}). Restarting env and replaying "
            f"{len(replay_stage_dirs)} working constraint(s)."
        )

        env.reset()
        settle_after_reset(env)
        program_runtime = make_program_runtime_for_env(env, env_cfg)
        update_sun_program_executor_data_for_env(env, program_runtime, "panda")
        program_runtime.controller_api_wrapper.updatePOVState()
        print("Environment reset and SUNProgramExecutor controller state rebuilt.", flush=True)
        count = 0
        stage_running = False
        active_stage_recorded = False
        active_stage_dir = None
        need_task_plan = False
        need_atomic_actions = False
        need_constraint = False
        reset_control_schedule()

        if replay_stage_dirs:
            replaying = True
            load_replay_stage()
        else:
            replaying = False
            restore_failed_stage_for_correction()

    while simulation_app.is_running():
        with torch.inference_mode():

            update_sun_program_executor_data_for_env(env, program_runtime, "panda")

            if need_task_plan:
                if count > 0:
                    print(
                        f"Calling for pre-task (high-level planning) at count {count}"
                    )
                update_sun_program_executor_data_for_env(env, program_runtime, "panda")
                before_usage = current_llm_usage()
                success = program_runtime.callbackTask()
                accumulate_llm_usage(before_usage)
                if success:
                    print("VLM called for pre-task (high-level subtasks) successful")
                    need_task_plan = False
                    need_atomic_actions = True
                else:
                    abort_loop("VLM call failed for pre-task")

            if need_atomic_actions:
                runtime_error_prompt = failed_error_prompt
                failed_error_prompt = None
                if runtime_error_prompt is None:
                    print(f"Calling for post-task (grounding subtask) at count {count}")
                else:
                    correction_iterations += 1
                    print(
                        f"Calling for post-task atomic correction at count {count}"
                    )
                before_usage = current_llm_usage()
                success = program_runtime.callbackAtomicActions(error=runtime_error_prompt)
                accumulate_llm_usage(before_usage)
                if success:
                    print("VLM called for post-task (atomic actions) successful")
                    need_atomic_actions = False
                    need_constraint = True
                else:
                    abort_loop("VLM call failed for post-task (atomic actions)")

            if need_constraint:
                before_usage = current_llm_usage()
                success = program_runtime.callbackConstraint()
                accumulate_llm_usage(before_usage)
                if success:
                    program_runtime.applyConstraint()
                    print("VLM called for constraint successful")
                    need_constraint = False
                    stage_running = True
                    active_stage_recorded = False
                    active_stage_dir = None
                    current_stage_start_count = count
                    record_current_stage("running")
                    reset_control_schedule()
                else:
                    abort_loop("VLM call failed for predicate")

            if not stage_running:
                if not (need_task_plan or need_atomic_actions or need_constraint):
                    message = "Learn loop is idle with no pending LLM or control stage."
                    abort_loop(message)
                continue

            if (
                count > current_stage_start_count
                and count - current_stage_start_count >= CONSTRAINT_CALL_INTERVAL
            ):
                restart_from_failure("failed to meet constraints within time limit")
                continue

            if u_following is None or count >= recompute_call_counter:
                u_following = program_runtime.computeControl()
                last_computed_count = count
                recompute_call_counter = count + RECOMPUTE_CALL_INTERVAL

                if program_runtime.hasStalled():
                    restart_from_failure(program_runtime.stallFailureReason())
                    continue

            satisfied = program_runtime.quantifyPredicate()
            if torch.all(satisfied):
                print("Predicate satisfied, moving on to the next step")

                if replaying:
                    replay_index += 1
                    if replay_index < len(replay_stage_dirs):
                        load_replay_stage()
                    else:
                        restore_failed_stage_for_correction()
                    continue

                save_current_stage_as_working()
                program_runtime.terminateCurrentSubtask()
                failed_retry_count = 0
                failed_stage_snapshot = None

                if program_runtime.hasMoreTasks():
                    need_atomic_actions = True
                    stage_running = False
                    active_stage_recorded = False
                    active_stage_dir = None
                    subtasks_count += 1
                    reset_control_schedule()
                    continue
                else:
                    print("All tasks completed, exiting...")
                    task_completed = True
                    exit_reason = "All tasks completed"
                    break

            action_index = (count - last_computed_count) * 2
            if action_index >= len(u_following):
                if len(u_following) == 0:
                    message = "Controller returned an empty action trajectory."
                    abort_loop(message)
                recompute_call_counter = count
                continue
            actions = (
                torch.from_numpy(u_following[action_index])
                .to(dtype=torch.float32)
                .to(env_cfg.sim.device)
            )

            before_episode_lengths = episode_lengths()
            step_result = env.step(actions)
            count += 1
            total_count += 1
            done_reason = env_success_reason(step_result, before_episode_lengths)
            if done_reason is not None:
                print(done_reason, flush=True)
                task_completed = True
                exit_reason = done_reason
                break

    env.close()
    end = time.time()
    print(f"Total time used: {end - start}")

    if task_completed:
        print(f"Task completed in {total_count} steps")

    write_txt(os.path.join(GLOBAL_OUTPUT_DIR, "success.txt"), f"{task_completed}")
    write_run_stats("succeeded" if task_completed else "stopped", exit_reason)


if __name__ == "__main__":
    main()
    simulation_app.close()
