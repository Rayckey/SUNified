"""Generate and physically verify a program on either pipeline's cabinet export.

Run with the Isaac Lab Python environment. Inspect/replay make no model calls;
generate uses the normal task -> atomic actions -> constraint learner and MPC.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import traceback
from pathlib import Path


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


class PreparedRequest(BaseException):
    """Stop inspection at the model boundary, including retrying learner code."""


class RequestBudget:
    """Persist request attempts across controller rebuilds; disable SDK retries."""

    def __init__(self, directory, maximum):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.maximum = maximum
        self.used = len(list(self.directory.glob("request_*.json")))

    def bind(self, manager, *, inspect=False, preview_model=None):
        original = manager._call_model
        if manager.gpt_client is not None:
            manager.gpt_client = manager.gpt_client.with_options(max_retries=0, timeout=180.0)

        def call(response_format):
            payload = {
                "model": preview_model if inspect and preview_model else manager.model_version,
                "messages": manager.prompts,
                "response_format": response_format,
            }
            if inspect:
                write_json(self.directory / "prepared_request.json", payload)
                raise PreparedRequest()
            if self.used >= self.maximum:
                raise RuntimeError(f"Model request budget exhausted ({self.maximum})")
            self.used += 1
            stem = f"request_{self.used:03d}"
            write_json(self.directory / f"{stem}.json", payload)
            print(f"Model request {self.used}/{self.maximum}", flush=True)
            start = time.monotonic()
            try:
                response, content = original(response_format)
                write_json(self.directory / f"response_{self.used:03d}.json", response.model_dump())
                return response, content
            finally:
                print(f"Model request {self.used} finished in {time.monotonic() - start:.1f}s", flush=True)

        manager._call_model = call


def physical_result(initial, final, trajectory, target=0.10, tolerance=0.01):
    """Independent joint-space check; never infer physical success from predicates."""
    import numpy as np

    names = initial["joint_names"]
    target_index = names.index("upper_drawer_joint")
    start = np.asarray(initial["cabinet_joints"])
    end = np.asarray(final["cabinet_joints"])
    trace = np.asarray([row["cabinet_joints"] for row in trajectory] or [end])
    others = [i for i in range(len(names)) if i != target_index]
    travel = float(end[target_index] - start[target_index])
    other_motion = float(np.max(np.abs(trace[:, others] - start[others]))) if others else 0.0
    finite = bool(np.isfinite(trace).all()) and all(
        np.isfinite(row.get("robot_joints", [])).all()
        and all(np.isfinite(value).all() for frame in row.get("frames", {}).values() for value in frame.values())
        for row in [initial, *trajectory, final]
    )
    return {
        "passed": bool(finite and abs(travel - target) <= tolerance and other_motion <= 0.005),
        "target_travel_m": target,
        "tolerance_m": tolerance,
        "actual_travel_m": travel,
        "other_drawer_max_motion_m": other_motion,
        "finite_states": finite,
    }


def run(args, app):
    import numpy as np
    import torch
    from omegaconf import OmegaConf

    from kuafu.common.simulation_utils import (
        apply_franka_panda_gripper_config,
        update_sun_program_executor_data_for_env,
    )
    from kuafu.mpc.controller.bounded_ik import BoundedPoseIK
    from kuafu.mpc.simulation.sun_program_executor import SUNProgramExecutor
    from kuafu.isaac.formulation.generated_drawer import (
        GeneratedDrawerFormulationEnv,
        generated_drawer_cfg,
    )
    from kuafu.isaac.formulation.panda import assert_program_free_panda_env
    from kuafu.programs.robust_loop import run_robust_program_learning_loop
    from kuafu.paths import SKILLS_ROOT, PROMPT_ROOT, ROBOT_RESOURCE_ROOT, resolve_artifact_path

    output = resolve_artifact_path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    os.chdir(output)  # Legacy prompt transcripts must also stay outside source.
    asset = Path(args.asset).resolve()
    budget = RequestBudget(output / "model_requests", args.max_requests)
    cfg = generated_drawer_cfg(
        asset,
        args.device,
        position=args.position,
        yaw_deg=args.yaw,
        cameras=args.enable_cameras,
    )
    cfg.task_description = args.task
    cfg.seed = getattr(args, "seed", 0)
    env = GeneratedDrawerFormulationEnv(cfg)
    assert_program_free_panda_env(env)
    closed = False
    trajectory, solves, replay_stages = [], [], []
    initial, final = {}, {}
    program_passed = False
    error = None
    active_stage = None
    stage_index = 0
    observations, recorded_actions, next_observations = [], [], []
    record_demo = getattr(args, "record_demo", False)
    policy_path = getattr(args, "policy", None)

    def progress(phase, **details):
        write_json(output / "progress.json", {
            "phase": phase, "stage": active_stage, "control_steps": len(trajectory),
            "mpc_solves": len(solves), "updated_at_unix": time.time(), **details,
        })

    def snapshot():
        return {
            "step": len(trajectory),
            "joint_names": list(env._cabinet.joint_names),
            "cabinet_joints": env._cabinet.data.joint_pos[0].detach().cpu().tolist(),
            "robot_joints": env._robot.data.joint_pos[0].detach().cpu().tolist(),
            "robot_joint_velocities": env._robot.data.joint_vel[0].detach().cpu().tolist(),
            "frames": {
                name: {
                    "position": sensor.data.target_pos_w[0, 0].detach().cpu().tolist(),
                    "quaternion_wxyz": sensor.data.target_quat_w[0, 0].detach().cpu().tolist(),
                }
                for name, sensor in env.scene.sensors.items()
                if name.endswith("_frame") or name.startswith("panda_")
            },
        }

    def save_images(label):
        if not args.enable_cameras:
            return
        from PIL import Image

        for name, sensor in env.scene.sensors.items():
            if name.startswith("table_cam"):
                pixels = sensor.data.output["rgb"][0, :, :, :3].detach().cpu().numpy()
                Image.fromarray(pixels).save(output / f"{label}_{name}.png")

    original_step, original_close = env.step, env.close

    def observed_step(action):
        if record_demo:
            from kuafu.data.generated_drawer import policy_observation

            observation = policy_observation(snapshot(), stage_index)
            applied = action.detach().cpu().numpy().reshape(-1)
        result = original_step(action)
        trajectory.append(snapshot())
        if record_demo:
            observations.append(observation)
            recorded_actions.append(applied)
            next_observations.append(policy_observation(trajectory[-1], stage_index))
        if len(trajectory) % 100 == 0:
            print(f"Control step {len(trajectory)}: drawer joints={trajectory[-1]['cabinet_joints']}", flush=True)
            write_json(output / "trajectory.json", trajectory)
        if args.enable_cameras and args.capture_every > 0 and len(trajectory) % args.capture_every == 0:
            save_images(f"step_{len(trajectory):05d}")
        return result

    def observed_close():
        nonlocal final, closed
        if not closed:
            final = snapshot()
            save_images("final")
            original_close()
            closed = True

    env.step, env.close = observed_step, observed_close
    try:
        env.reset()
        hold = env._robot.data.joint_pos.clone()
        for _ in range(30):
            original_step(hold)
        initial = snapshot()
        save_images("initial")
        write_json(
            output / "scene.json",
            {
                "asset": str(asset),
                "asset_hashes": {
                    name: hashlib.sha256((asset / name).read_bytes()).hexdigest()
                    for name in ("asset_dynamic.usda", "asset_geometry.usda", "semantics.json")
                },
                "task": args.task,
                "mode": args.mode,
                "model": args.model,
                "max_requests": args.max_requests,
                "position": args.position,
                "yaw_deg": args.yaw,
                "control_dt": env.step_dt,
                "descriptions": cfg.detectable_objects,
                "initial": initial,
            },
        )
        controller_number = 0

        def make_program_runtime():
            nonlocal controller_number
            controller_number += 1
            config = {
                "controller_name": "FrankaIKController",
                "device": args.device,
                "controller": {
                    "urdf_path": str(ROBOT_RESOURCE_ROOT / "franka_description/robots/panda_arm_hand_fixed.urdf"),
                    "ee_frame": "panda_fingertip",
                    "horizon_length": 15,
                    "time_step": env.step_dt,
                },
                "load_program": None,
                "manipulator_high_effort_threshold": 100.0,
                "save_dir": str(output),
                "agent": {
                    "id": "agent",
                    "device": args.device,
                    "task": args.task,
                    "prompt_dir": str(PROMPT_ROOT / "Manipulation"),
                    "initial_residual": str(SKILLS_ROOT / "constraints/residuals.py"),
                    "initial_predicate": str(SKILLS_ROOT / "constraints/predicates.py"),
                    "model_version": args.model if args.mode == "generate" else "offline",
                    "load_path": None,
                    "reuse_program": True,
                    "max_correction_steps": 1,
                },
            }
            apply_franka_panda_gripper_config(config)
            program_runtime = SUNProgramExecutor(OmegaConf.create(config))
            ik = BoundedPoseIK(program_runtime.active_controller.model, time_step=env.step_dt)
            program_runtime.active_controller._ik_fun = ik
            program_runtime.test_controller._ik_fun = ik
            manager = program_runtime.program_learner._constraint_generator.prompt_manager
            manager.transcript_path = str(output / f"transcript_{controller_number}.md")
            budget.bind(manager, inspect=args.mode == "inspect", preview_model=args.model)
            original_compute = program_runtime.computeControl

            def compute():
                ik.diagnostics.clear()
                progress("solving_mpc")
                actions = original_compute()
                opti = getattr(program_runtime.active_controller, "opti", None)
                stats = opti.stats() if opti is not None else {}
                solves.append(
                    {
                        "step": len(trajectory),
                        "finite_actions": bool(np.isfinite(actions).all()),
                        "success": stats.get("success"),
                        "return_status": stats.get("return_status"),
                        "ik_max_position_error_m": max(d["position_error_m"] for d in ik.diagnostics),
                        "ik_max_orientation_error_rad": max(d["orientation_error_rad"] for d in ik.diagnostics),
                    }
                )
                if not np.isfinite(actions).all():
                    raise RuntimeError("MPC returned non-finite actions")
                write_json(output / "mpc_solves.json", solves)
                progress("executing_controls")
                return actions

            program_runtime.computeControl = compute
            return program_runtime

        program_runtime = make_program_runtime()

        def update(_number=0, active=None, **_kwargs):
            update_sun_program_executor_data_for_env(env, active or program_runtime, "panda")

        update()
        if args.mode == "inspect":
            try:
                program_runtime.callbackTask()
            except PreparedRequest:
                print("Prepared the real task-generation request without sending it.", flush=True)
        elif args.mode == "generate":
            program_passed = run_robust_program_learning_loop(
                env=env,
                env_cfg=cfg,
                program_runtime=program_runtime,
                make_program_runtime=make_program_runtime,
                update_data_in_controller=update,
                simulation_app=app,
                output_dir=str(output),
                max_counter=args.max_steps,
                enable_stall_detection=True,
                max_atomic_corrections=1,
                constraint_call_interval=args.stage_steps,
                recompute_call_interval=5,
                task_update_uses_camera=False,
                restart_settle_steps=30,
            )
        else:
            stage_dirs = sorted(Path(args.program).glob("subtask_*"), key=lambda p: int(p.name.split("_")[-1]))
            if not stage_dirs:
                raise ValueError("No generated subtask_* programs found")
            policy = None
            if policy_path:
                from kuafu.agents.jobs import asset_hashes, tree_hashes
                from kuafu.bc.generated_drawer import GeneratedDrawerPolicy

                policy = GeneratedDrawerPolicy(
                    policy_path, device=args.device, joint_names=env._robot.joint_names,
                    cabinet_joint_names=env._cabinet.joint_names,
                    asset_hashes=asset_hashes(asset), program_hashes=tree_hashes(args.program),
                )
            for stage_index, directory in enumerate(stage_dirs):
                active_stage = directory.name
                progress("compiling_stage")
                update()
                program_runtime.controller_api_wrapper.updatePOVState()
                stage = program_runtime.load_program_stage(directory)
                program_runtime.applyConstraint()
                passed, actions, last_compute = False, None, 0
                for step in range(args.stage_steps + 1):
                    update()
                    if torch.all(program_runtime.quantifyPredicate()):
                        passed = True
                        break
                    if step >= args.stage_steps or len(trajectory) >= args.max_steps or not app.is_running():
                        break
                    if policy is not None:
                        action = policy.action(snapshot(), stage_index, env._robot.data.soft_joint_pos_limits[0])
                        env.step(action)
                    else:
                        if actions is None or step - last_compute >= 5:
                            actions, last_compute = program_runtime.computeControl(), step
                        index = min(step - last_compute + 1, len(actions) - 1)
                        env.step(torch.as_tensor(actions[index], dtype=torch.float32, device=args.device))
                update()
                satisfied, breakdown = program_runtime.quantifyPredicate(text=True, max_envs=1)
                replay_stages.append(
                    {
                        "directory": str(directory),
                        "task": stage["task"],
                        "passed": passed,
                        "predicate": satisfied.detach().cpu().tolist(),
                        "breakdown": breakdown,
                        "state": snapshot(),
                    }
                )
                print(f"Replay {directory.name}: passed={passed}", flush=True)
                if not passed:
                    print(breakdown, flush=True)
                    break
            program_passed = len(replay_stages) == len(stage_dirs) and all(s["passed"] for s in replay_stages)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    finally:
        robot_joint_names = list(env._robot.joint_names)
        cabinet_joint_names = list(env._cabinet.joint_names)
        env.close()
        write_json(output / "trajectory.json", trajectory)
        write_json(output / "mpc_solves.json", solves)
        physical = physical_result(initial, final, trajectory, args.target_travel) if initial else {"passed": False}
        solver_passed = bool(solves) and all(s["success"] and s["finite_actions"] for s in solves)
        report = {
            "mode": args.mode,
            "program_passed": bool(program_passed),
            "physical": physical,
            "passed": bool(program_passed and physical["passed"] and (bool(policy_path) or solver_passed)),
            "controller": "bc_policy" if policy_path else "mpc",
            "solver_passed": None if policy_path else solver_passed,
            "control_steps": len(trajectory),
            "mpc_solves": len(solves),
            "model_requests": budget.used,
            "initial": initial,
            "final": final,
            "replay_stages": replay_stages,
            "error": error,
        }
        write_json(output / "verification.json", report)
        if record_demo and report["passed"]:
            from kuafu.agents.jobs import asset_hashes, tree_hashes
            from kuafu.data.generated_drawer import OBSERVATION_LAYOUT, write_demonstration

            write_demonstration(output / "demonstrations.hdf5", observations, recorded_actions, next_observations, {
                "observation_layout": OBSERVATION_LAYOUT, "joint_names": robot_joint_names,
                "cabinet_joint_names": cabinet_joint_names, "asset_hashes": asset_hashes(asset),
                "program_hashes": tree_hashes(args.program), "target_travel_m": args.target_travel,
                "seed": getattr(args, "seed", 0), "position": list(args.position), "yaw_deg": args.yaw,
                "verification": physical,
            })
        progress("complete" if report["passed"] or args.mode == "inspect" and error is None else "failed", error=error)
        summary = {k: v for k, v in report.items() if k not in ("initial", "final", "replay_stages")}
        print(json.dumps(summary), flush=True)
    return error is None if args.mode == "inspect" else report["passed"]


def main(argv=None):
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=("inspect", "generate", "replay"), default="inspect")
    parser.add_argument("--program", help="Generated program directory for offline replay")
    parser.add_argument("--record-demo", action="store_true", help="Save pre-action observations and MPC actions after a passing replay")
    parser.add_argument("--policy", help="Evaluate a BC checkpoint with the retained SUN Program predicates")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task", default="Open the top drawer by 10 cm.")
    parser.add_argument(
        "--target-travel", type=float, default=0.10, help="Independent physical acceptance target in meters"
    )
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--max-requests", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--stage-steps", type=int, default=700)
    parser.add_argument(
        "--capture-every",
        type=int,
        default=100,
        help="Save local camera images every N control steps; zero disables periodic images",
    )
    parser.add_argument("--position", type=float, nargs=3, default=(0.6, -0.5, 0.2))
    parser.add_argument("--yaw", type=float, default=180.0)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)
    if args.mode == "replay" and not args.program:
        parser.error("--program is required for replay")
    if (args.record_demo or args.policy) and args.mode != "replay":
        parser.error("--record-demo and --policy require --mode replay")
    if args.record_demo and args.policy:
        parser.error("Record teacher demonstrations and evaluate a policy in separate runs")
    # Resolve before changing into the external output directory.
    args.asset = str(Path(args.asset).expanduser().resolve())
    if args.program:
        args.program = str(Path(args.program).expanduser().resolve())
    if args.policy:
        args.policy = str(Path(args.policy).expanduser().resolve())
    launcher = AppLauncher(args)
    status = 1
    try:
        status = 0 if run(args, launcher.app) else 1
    finally:
        import omni.kit.app

        omni.kit.app.get_app().post_quit(status)
        launcher.app.close(wait_for_replicator=False)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
