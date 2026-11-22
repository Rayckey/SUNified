# SPDX-License-Identifier: Apache-2.0
"""Reset and exercise UR5 tasks, checking cameras, actuation, and USD/URDF frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from . import TASKS
from .assets import asset_directory
from kuafu.paths import resolve_artifact_path


def main():
    # Resolve Pinocchio's native libraries before Kit loads its USD plugins.
    import pinocchio as pin
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="UR5-Table-Stack-2", help="Registered UR5 task ID or 'all'")
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--steps", type=int, default=90)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--move-arm", action="store_true", help="Exercise a small reversible arm motion")
    parser.add_argument("--check-mpc", action="store_true", help="Also solve a Cartesian MPC horizon")
    parser.add_argument("--action-mode", choices=("relative_joint", "absolute_joint", "relative_ee", "absolute_ee"),
                        default="relative_joint")
    parser.add_argument("--output", type=Path, default=Path("validation/ur5/smoke.json"))
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.output = resolve_artifact_path(args.output)
    if args.task == "all":
        # Isaac camera annotators have process-global names. Isolate scenes so
        # destruction of an old task cannot detach a new task's render products.
        results = []
        for name in TASKS:
            task = f"UR5-Table-{name}"
            report = args.output.with_name(f"{args.output.stem}-{name}.json")
            report.unlink(missing_ok=True)
            command = [sys.executable, "-m", __package__ + ".smoke", "--task", task,
                       "--num-envs", str(args.num_envs), "--steps", str(args.steps),
                       "--seed", str(args.seed), "--action-mode", args.action_mode,
                       "--device", args.device, "--output", str(report)]
            if args.headless:
                command.append("--headless")
            if args.enable_cameras:
                command.append("--enable_cameras")
            if args.move_arm:
                command.append("--move-arm")
            if args.check_mpc:
                command.append("--check-mpc")
            subprocess.run(command, check=True)
            if not report.is_file():
                raise RuntimeError(f"{task} exited without a passing smoke report")
            results.extend(json.loads(report.read_text())["results"])
            args.output.write_text(json.dumps({"asset_manifest": str(asset_directory() / "manifest.json"),
                                               "results": results}, indent=2) + "\n")
        return
    launcher = AppLauncher(args)
    app = launcher.app
    results = []
    try:
        import gymnasium as gym
        import numpy as np
        import torch
        from isaaclab.utils.math import subtract_frame_transforms
        from isaacsim.core.utils.stage import create_new_stage
        from isaaclab_tasks.utils import parse_env_cfg
        from kuafu.mpc.simulation.spatial_utils import quat_to_rotation_6d

        model = pin.buildModelFromUrdf(str(asset_directory() / "ur5_controller.urdf"))
        data = model.createData()
        tasks = [f"UR5-Table-{name}" for name in TASKS] if args.task == "all" else [args.task]
        for task in tasks:
            create_new_stage()
            cfg = parse_env_cfg(task, device=args.device, num_envs=args.num_envs)
            cfg.seed = args.seed
            cfg.enable_cameras = args.enable_cameras
            cfg.action_mode = args.action_mode
            env = gym.make(task, cfg=cfg).unwrapped
            try:
                obs, _ = env.reset(seed=args.seed)
                assert env._gripper_contact_material_bound, "Fingertip contact material was not bound"
                assert set(obs) == ({"policy", "robot_state", "rgb"} if cfg.enable_cameras else {"policy", "robot_state"})
                assert obs["robot_state"].shape == (args.num_envs, 8)
                initial = obs["robot_state"].clone()
                initial_pos = env.ee_frame.data.target_pos_w[:, 0].clone()
                initial_rot = env.ee_frame.data.target_quat_w[:, 0].clone()
                root = env._robot.data.root_state_w
                target_pos, target_rot = subtract_frame_transforms(root[:, :3], root[:, 3:7], initial_pos, initial_rot)
                maximum_error = 0.0
                for index in range(args.num_envs):
                    pin.forwardKinematics(model, data, initial[index, :6].cpu().numpy().astype(float))
                    pin.updateFramePlacements(model, data)
                    error = np.linalg.norm(data.oMf[model.getFrameId("panda_fingertip")].translation
                                           - target_pos[index].cpu().numpy())
                    if error >= 1e-4:
                        print("UR5_FK_DIAGNOSTIC", {"q": initial[index].tolist(),
                              "pin_tcp": data.oMf[model.getFrameId("panda_fingertip")].translation.tolist(),
                              "sim_tcp": target_pos[index].tolist(), "root": root[index, :7].tolist(),
                              "body_names": env._robot.body_names,
                              "body_positions": env._robot.data.body_pos_w[index].tolist()}, flush=True)
                    maximum_error = max(maximum_error, float(error))
                assert maximum_error < 1e-4, f"USD/URDF fingertip error: {maximum_error} m"
                closed = None
                max_excursion = 0.0
                for step in range(args.steps):
                    action = torch.zeros((args.num_envs, env.cfg.action_space), device=env.device)
                    close = step < args.steps // 2
                    if args.action_mode == "absolute_joint":
                        action[:, :6] = initial[:, :6]
                    elif args.action_mode == "absolute_ee":
                        action[:, :3] = target_pos
                        action[:, 3:9] = quat_to_rotation_6d(target_rot)
                    if args.move_arm:
                        if args.action_mode == "relative_joint":
                            action[:, 5] = 0.01 if close else -0.01
                        elif args.action_mode == "relative_ee":
                            action[:, 0] = 0.01 if close else -0.01
                        elif args.action_mode == "absolute_joint":
                            action[:, 5] += 0.15 if close else 0.0
                        else:
                            action[:, 0] += 0.03 if close else 0.0
                    if args.action_mode.startswith("absolute"):
                        action[:, -2:] = 0.8 if close else 0.0
                    else:
                        action[:, -2:] = -1.0 if close else 1.0
                    obs, reward, terminated, truncated, _ = env.step(action)
                    assert torch.isfinite(reward).all() and torch.isfinite(obs["robot_state"]).all()
                    assert not terminated.any() and not truncated.any(), "Unexpected early termination during smoke"
                    max_excursion = max(max_excursion, float((obs["robot_state"][:, :6] - initial[:, :6]).abs().max()))
                    if step == args.steps // 2 - 1:
                        closed = obs["robot_state"][:, -2:].clone()
                result = {"task": task, "seed": args.seed, "steps": args.steps,
                          "action_mode": args.action_mode, "robot_state_shape": list(obs["robot_state"].shape),
                          "gripper_contact_material_bound": env._gripper_contact_material_bound,
                          "initial_joint_position": initial.cpu().tolist(),
                          "initial_tcp_position_base_m": target_pos.cpu().tolist(),
                          "initial_tcp_quaternion_base_wxyz": target_rot.cpu().tolist(),
                          "max_fk_position_error_m": maximum_error,
                          "closed_gripper_position": closed.cpu().tolist(),
                          "final_gripper_position": obs["robot_state"][:, -2:].cpu().tolist(),
                          "max_arm_drift_rad": float((obs["robot_state"][:, :6] - initial[:, :6]).abs().max())}
                result["max_arm_excursion_rad"] = max_excursion
                if args.move_arm:
                    assert max_excursion > 0.02, "Arm did not respond to motion commands"
                assert (closed - initial[:, -2:]).min() > 0.2, "Gripper did not close"
                assert (closed - obs["robot_state"][:, -2:]).min() > 0.2, "Gripper did not reopen"
                if args.check_mpc:
                    import casadi as ca
                    from kuafu.mpc.controller.mpc import UR5IKController
                    controller = UR5IKController(str(asset_directory() / "ur5_controller.urdf"),
                                                 horizon_length=4, time_step=env.step_dt)
                    q = obs["robot_state"][0].cpu().numpy().astype(float)
                    pose = controller.calculate_frame_transformation("panda_fingertip", q)
                    position = pose[:3, 3] + np.array([0.003, 0, 0])
                    quaternion = pin.Quaternion(pose[:3, :3]).coeffs()[[3, 0, 1, 2]]
                    controller.update_cost_function({"residual_function": lambda x:
                        ca.sumsqr(x[:3] - position) + 0.1 * ca.sumsqr(x[3:7] - quaternion)
                        + 0.01 * ca.sumsqr(x[7:] - q[-2:])})
                    horizon = controller.compute_control(q, np.zeros(8))
                    assert horizon.shape == (4, 8) and np.isfinite(horizon).all()
                    assert np.all(horizon[:, :6] >= model.lowerPositionLimit - 1e-5)
                    assert np.all(horizon[:, :6] <= model.upperPositionLimit + 1e-5)
                    result["mpc"] = {"solver_status": controller.opti.stats()["return_status"],
                                     "horizon_shape": list(horizon.shape),
                                     "ik_diagnostics": controller._ik_fun.diagnostics}
                if cfg.enable_cameras:
                    rgb = obs["rgb"]
                    assert tuple(rgb.shape) == (args.num_envs, 2, 72, 128, 3)
                    assert rgb.dtype == torch.uint8 and rgb.float().std() > 1.0
                    result["rgb_shape"] = list(rgb.shape)
                    from PIL import Image
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    for view in range(2):
                        Image.fromarray(rgb[0, view].cpu().numpy()).save(args.output.parent / f"{task}-view{view}.png")
                results.append(result)
                # A partial vector reset must leave the other robot untouched,
                # and must refresh sensor poses before the next action.
                before_reset = obs["robot_state"].clone()
                env._reset_idx(torch.tensor([0], device=env.device))
                if cfg.enable_cameras:
                    env.sim.render()
                obs = env._get_observations()
                if args.num_envs > 1:
                    torch.testing.assert_close(obs["robot_state"][1:], before_reset[1:], atol=1e-6, rtol=0)
                pin.forwardKinematics(model, data, obs["robot_state"][0, :6].cpu().numpy().astype(float))
                pin.updateFramePlacements(model, data)
                root = env._robot.data.root_state_w
                reset_pos, _ = subtract_frame_transforms(
                    root[:, :3], root[:, 3:7], env.ee_frame.data.target_pos_w[:, 0],
                    env.ee_frame.data.target_quat_w[:, 0],
                )
                reset_error = float(np.linalg.norm(data.oMf[model.getFrameId("panda_fingertip")].translation
                                                   - reset_pos[0].cpu().numpy()))
                assert reset_error < 1e-4, f"Partial reset left stale tool frames: {reset_error} m"
                result["partial_reset_fk_error_m"] = reset_error
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps({"asset_manifest": str(asset_directory() / "manifest.json"),
                                                   "results": results}, indent=2) + "\n")
                print(f"UR5_SMOKE_PASS {task} FK error={maximum_error:.3g} m", flush=True)
            finally:
                env.close()
    except Exception:
        import traceback
        traceback.print_exc()
        raise
    finally:
        app.close()


if __name__ == "__main__":
    main()
