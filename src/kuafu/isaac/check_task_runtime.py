"""Reset and step a catalog task to check scene inputs, physics, and robot control."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import traceback

from kuafu.paths import resolve_artifact_path
from kuafu.pipeline.task_catalog import get_task


def main(argv=None):
    import pinocchio  # Resolve native libraries before Kit loads USD plugins.
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, help="Catalog task ID")
    parser.add_argument("--robot", required=True, help="Catalog robot ID")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)
    task = get_task(args.task, args.robot)
    entries = task["environments"]
    if len(entries) != 1 or task["status"] != "registered":
        parser.error("Choose a registered task and one robot from kuafu tasks list")
    if args.steps < 2:
        parser.error("--steps must be at least 2")
    output = resolve_artifact_path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    result = {"task": args.task, "robot": args.robot, "gym_id": entries[0]["gym_id"],
              "seed": args.seed, "passed": False, "task_success_tested": False,
              "scope": "Scene, configured program loading, reset, finite observations/rewards, and actuator response",
              "action_mode": "relative_joint", "cameras": args.enable_cameras, "steps": 0}

    def save(phase):
        result["phase"] = phase
        (output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    save("launching")
    launcher = AppLauncher(args)
    env = None
    try:
        import gymnasium as gym
        import torch
        from isaaclab_tasks.utils import parse_env_cfg
        from kuafu.isaac.check_task_configs import configuration_inputs

        def finite(value):
            if torch.is_tensor(value):
                return bool(torch.isfinite(value).all())
            if isinstance(value, dict):
                return all(finite(item) for item in value.values())
            if isinstance(value, (list, tuple)):
                return all(finite(item) for item in value)
            return True

        entry = entries[0]
        importlib.import_module(entry["registration_module"])
        cfg = parse_env_cfg(entry["gym_id"], device=args.device, num_envs=1)
        cfg.seed = args.seed
        cfg.enable_cameras = args.enable_cameras
        cfg.action_mode = "relative_joint"
        cfg.action_space = 8 if args.robot == "ur5_robotiq" else 9
        cfg.mpc_config["control_output_mode"] = cfg.action_mode
        cfg.log_dir = str(output / "runtime")
        for name in ("reward_print_every", "episode_outcome_print_every", "gripper_force_print_every"):
            if hasattr(cfg, name):
                setattr(cfg, name, 0)
        inputs, missing = configuration_inputs(cfg)
        if missing:
            raise FileNotFoundError(f"Missing task inputs: {missing}")
        result["inputs"] = inputs
        save("creating_scene")
        env = gym.make(entry["gym_id"], cfg=cfg).unwrapped
        observations, _ = env.reset(seed=args.seed)
        if not finite(observations):
            raise ValueError("Reset returned non-finite observations")
        result["stages"] = env.program_runtime.compiled_program.stage_count()
        result["program_configured"] = bool(cfg.mpc_config.get("load_program"))
        if result["program_configured"] and result["stages"] <= 0:
            raise ValueError("The configured SUN Program did not load")
        result["observation_shapes"] = {name: list(value.shape) for name, value in observations.items()
                                         if torch.is_tensor(value)}
        initial = env._robot.data.joint_pos.clone()
        max_motion = 0.0
        done_count = 0
        for step in range(args.steps):
            action = torch.zeros((1, env.cfg.action_space), device=env.device)
            action[:, 0] = .02 if step < args.steps // 2 else -.02
            observations, reward, terminated, truncated, _ = env.step(action)
            if not finite((observations, reward, env._robot.data.joint_pos, env._robot.data.joint_vel)):
                raise ValueError(f"Non-finite observations or physics at step {step}")
            max_motion = max(max_motion, float((env._robot.data.joint_pos - initial).abs().max()))
            done_count += int((terminated | truncated).sum())
            result.update(steps=step + 1, max_joint_motion=max_motion, resets=done_count)
            if step % 8 == 0:
                save("stepping")
        if max_motion < 1e-5:
            raise ValueError("Robot did not respond to joint commands")
        result["passed"] = True
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    finally:
        save("complete" if result["passed"] else "failed")
        if env is not None:
            env.close()
        import omni.kit.app

        omni.kit.app.get_app().post_quit(0 if result["passed"] else 1)
        launcher.app.close(wait_for_replicator=False)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
