"""Shared robust learn-programs loop for SUNProgramExecutor atomic-action/constraint runners."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable

import torch

from kuafu.programs.stall_policy import evaluate_controller_stall
from kuafu.programs.llm.io_util import write_txt


def run_robust_program_learning_loop(
    *,
    env,
    env_cfg,
    program_runtime,
    make_program_runtime: Callable[[], object],
    update_data_in_controller: Callable,
    simulation_app,
    output_dir: str,
    max_counter: int | None,
    enable_stall_detection: bool,
    max_atomic_corrections: int = 5,
    constraint_call_interval: int = 700,
    recompute_call_interval: int = 10,
    env_number: int = 0,
    task_update_uses_camera: bool = True,
    apply_constraint_kwargs: dict | None = None,
    action_index_offset: int = 1,
    restart_settle_steps: int = 0,
    keep_running: bool = False,
):
    """Run a learn-programs loop that restarts/replays before atomic correction.

    The loop records every stage as soon as constraints start running. Only stages
    that satisfy their predicate are eligible for replay after a later failure.
    Stall detection is an explicit per-task policy because several original
    learn-programs tasks intentionally relied on their constraint timeout instead.
    """

    apply_constraint_kwargs = apply_constraint_kwargs or {}
    if not isinstance(enable_stall_detection, bool):
        raise TypeError("enable_stall_detection must be a bool")
    if max_counter is not None and max_counter <= 0:
        raise ValueError("max_counter must be positive or None")

    count = 0
    total_count = 0
    last_computed_count = 0
    subtasks_count = 0
    task_completed = False
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
    aborted = False
    exit_reason = "Simulation stopped"

    start = time.time()

    print(
        "Max atomic-action correction attempts per failed stage: "
        f"{max_atomic_corrections}",
        flush=True,
    )
    print(
        "Controller stall detection: "
        f"{'enabled' if enable_stall_detection else 'disabled (timeout only)'}",
        flush=True,
    )

    def ensure_api_wrapper_metadata(active_program_runtime):
        wrapper = getattr(active_program_runtime, "controller_api_wrapper", None)
        if wrapper is None or all(hasattr(wrapper, name) for name in ("device", "batch", "dtype")):
            return
        try:
            joint_position = env.unwrapped.scene.state["articulation"]["robot"]["joint_position"]
        except KeyError as exc:
            print(f"Failed to initialize APIWrapper metadata from env state: {exc}", flush=True)
            raise
        wrapper.device = joint_position.device
        wrapper.batch = joint_position.shape[0]
        wrapper.dtype = joint_position.dtype

    def update_data(
        active_program_runtime, *, use_camera: bool = False, includes_meta: bool = False
    ):
        update_data_in_controller(
            env_number,
            active_program_runtime,
            use_camera=use_camera,
            includes_meta=includes_meta,
        )
        ensure_api_wrapper_metadata(active_program_runtime)

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
            f"completed_subtasks: {subtasks_count + int(task_completed)}",
            f"total_constraints: {total_successful_constraints()}",
            f"stall_detection_enabled: {enable_stall_detection}",
            f"llm_vlm_calls: {llm_vlm_calls}",
            f"prompt_tokens: {llm_prompt_tokens}",
            f"completion_tokens: {llm_completion_tokens}",
            f"total_tokens: {llm_total_tokens}",
        ]

    def write_run_stats(status: str, reason: str):
        lines = stats_lines(status, reason)
        write_txt(os.path.join(output_dir, "stats.txt"), "\n".join(lines) + "\n")
        print("Learn-programs run stats:", flush=True)
        for line in lines:
            print(f"  {line}", flush=True)

    def settle_after_reset():
        if restart_settle_steps <= 0:
            return
        hold_actions = env.unwrapped.scene.state["articulation"]["robot"][
            "joint_position"
        ]
        for _ in range(restart_settle_steps):
            env.step(hold_actions)
        print(f"Settled reset for {restart_settle_steps} step(s).", flush=True)

    def abort_loop(message: str):
        nonlocal aborted, exit_reason
        print(message, flush=True)
        write_txt(os.path.join(output_dir, "success.txt"), "False")
        write_run_stats("failed", message)
        exit_reason = message
        aborted = True
        if not keep_running:
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
            os.path.join(output_dir, "program", "subtask_" + str(subtasks_count))
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
            print(f"Marked failed stage program: {active_stage_dir}", flush=True)
        else:
            record_current_stage("failed", reason)

    def save_current_stage_as_working():
        stage_dir = active_stage_dir
        if not active_stage_recorded or stage_dir is None:
            stage_dir = record_current_stage("succeeded")
        else:
            write_stage_status(stage_dir, "succeeded")
            print(f"Marked successful stage program: {stage_dir}", flush=True)
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

        lines.append(
            f"Controller stall detection enabled: {enable_stall_detection}"
        )
        if enable_stall_detection:
            stalled, stall_reason = evaluate_controller_stall(
                program_runtime, enabled=True
            )
            lines.append(f"Controller stalled: {stalled}")
            if stalled:
                lines.append(f"Stall failure reason: {stall_reason}")
        else:
            lines.append("Controller stalled: not evaluated (disabled)")

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

        # Refresh SUNProgramExecutor from the current physical state and execute one saved
        # stage at a time. Keep its full task context available for recovery.
        update_data(program_runtime)
        program_runtime.controller_api_wrapper.updatePOVState()
        program_runtime.controller_api_wrapper.stall_joint_counter = None
        program_runtime.controller_api_wrapper.stall_cost_counter = None
        program_runtime.controller_api_wrapper.gripper_stall_counter = None
        program_runtime.controller_api_wrapper._gripper_stall_state = None

        # Rewind the original failure's history to this replay position. Stages
        # after this one (including the original failed stage) are pending again.
        # If replay itself fails, its snapshot must retain that suffix; loading
        # a standalone stage here would make its correction look like the end
        # of the entire plan.
        completed_tasks = failed_stage_snapshot["completed_tasks"]
        stage = program_runtime.load_program_stage(
            path,
            pending_tasks=(
                completed_tasks[replay_index + 1:]
                + [failed_stage_snapshot["task"]]
                + failed_stage_snapshot["pending_tasks"]
            ),
            completed_tasks=completed_tasks[:replay_index],
            completed_task_atomic_actions=failed_stage_snapshot[
                "completed_task_atomic_actions"
            ][:replay_index],
        )
        program_runtime.applyConstraint(**apply_constraint_kwargs)

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
            return
        update_data(program_runtime)
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
        nonlocal program_runtime, count, replay_stage_dirs, replay_index, subtasks_count
        nonlocal replaying, failed_stage_snapshot, failed_error_prompt
        nonlocal failed_retry_count, stage_running, active_stage_recorded
        nonlocal active_stage_dir, need_task_plan, need_atomic_actions, need_constraint
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
        if failed_retry_count >= max_atomic_corrections:
            abort_loop("Maximum automatic restarts reached for the current atomic action")
            return

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
        settle_after_reset()
        program_runtime = make_program_runtime()
        update_data(program_runtime)
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

    update_data(program_runtime, includes_meta=True)

    while simulation_app.is_running():
        with torch.inference_mode():
            if aborted:
                break

            update_data(program_runtime)

            # Allow the last executed action to satisfy its predicate, but do not
            # ground another stage after exhausting the total execution budget.
            if max_counter is not None and total_count >= max_counter and not stage_running:
                abort_loop(f"Maximum control steps reached ({max_counter})")
                continue

            if need_task_plan:
                if count > 0:
                    print(f"Calling for pre-task (high-level planning) at count {count}")
                update_data(
                    program_runtime,
                    use_camera=task_update_uses_camera,
                    includes_meta=True,
                )
                before_usage = current_llm_usage()
                success = program_runtime.callbackTask()
                accumulate_llm_usage(before_usage)
                if success:
                    print("VLM called for pre-task (high-level subtasks) successful")
                    need_task_plan = False
                    need_atomic_actions = True
                else:
                    abort_loop("VLM call failed for pre-task")
                    continue

            if need_atomic_actions:
                runtime_error_prompt = failed_error_prompt
                failed_error_prompt = None
                if runtime_error_prompt is None:
                    print(f"Calling for post-task (grounding subtask) at count {count}")
                else:
                    correction_iterations += 1
                    print(f"Calling for post-task atomic correction at count {count}")
                before_usage = current_llm_usage()
                success = program_runtime.callbackAtomicActions(error=runtime_error_prompt)
                accumulate_llm_usage(before_usage)
                if success:
                    print("VLM called for post-task (atomic actions) successful")
                    need_atomic_actions = False
                    need_constraint = True
                else:
                    abort_loop("VLM call failed for post-task (atomic actions)")
                    continue

            if need_constraint:
                before_usage = current_llm_usage()
                success = program_runtime.callbackConstraint()
                accumulate_llm_usage(before_usage)
                if success:
                    program_runtime.applyConstraint(**apply_constraint_kwargs)
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
                    continue

            if not stage_running:
                if not (need_task_plan or need_atomic_actions or need_constraint):
                    abort_loop("Learn loop is idle with no pending LLM or control stage.")
                continue

            if (
                count > current_stage_start_count
                and count - current_stage_start_count >= constraint_call_interval
            ):
                restart_from_failure("failed to meet constraints within time limit")
                continue

            if u_following is None or count >= recompute_call_counter:
                u_following = program_runtime.computeControl()
                last_computed_count = count
                recompute_call_counter = count + recompute_call_interval

                stalled, stall_reason = evaluate_controller_stall(
                    program_runtime, enabled=enable_stall_detection
                )
                if stalled:
                    restart_from_failure(stall_reason or "controller stalled")
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

                print("All tasks completed, exiting...")
                task_completed = True
                exit_reason = "All tasks completed"
                break

            if max_counter is not None and total_count >= max_counter:
                abort_loop(f"Maximum control steps reached ({max_counter})")
                continue

            action_index = count - last_computed_count + action_index_offset
            if action_index >= len(u_following):
                if len(u_following) == 0:
                    abort_loop("Controller returned an empty action trajectory.")
                    continue
                print(
                    "Controller returned a short action trajectory "
                    f"({len(u_following)} action(s)); using the final action.",
                    flush=True,
                )
                action_index = len(u_following) - 1

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

    write_txt(os.path.join(output_dir, "success.txt"), f"{task_completed}")
    write_run_stats("succeeded" if task_completed else "stopped", exit_reason)

    if keep_running and simulation_app.is_running():
        print(f"{exit_reason}; keeping the environment running. Close Isaac Sim to exit.")
        hold_actions = env.unwrapped.scene.state["articulation"]["robot"][
            "joint_position"
        ].clone()
        while simulation_app.is_running():
            env.step(hold_actions)

    env.close()
    end = time.time()
    print(f"Total time used: {end - start}")

    if task_completed:
        print(f"Task completed in {total_count} steps")

    return task_completed
