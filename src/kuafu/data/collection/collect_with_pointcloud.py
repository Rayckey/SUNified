import argparse
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Batched multi-stage agent for Stack2 environment.")
parser.add_argument(
    "--disable_fabric", type=bool, default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to simulate."
)
parser.add_argument("--task_name", type=str, default="FR3-FinRay-Table-Drawer", help="Name of the task.")
parser.add_argument("--worker_id", type=int, default=0, help="Worker ID for parallel collection (controls output subdir).")
parser.add_argument("--num_demos", type=int, default=10, help="Number of demos to collect per worker.")
parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help="External shared output root supplied by the collection launcher.",
)
parser.add_argument(
    "--stride",
    type=int,
    default=1,
    help="Only store every N-th simulation step (default: 1, store every step).",
)
parser.add_argument(
    "--dry_run",
    action="store_true",
    default=False,
    help="Print the resolved collector setup and exit before launching Isaac.",
)


import gymnasium as gym

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.dry_run:
    print(
        "[dry-run] pointcloud collector "
        f"task={args_cli.task_name} num_envs={args_cli.num_envs} "
        f"num_demos={args_cli.num_demos} device={args_cli.device} stride={args_cli.stride}"
    )
    raise SystemExit(0)

# Enable cameras for point cloud collection
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import os
from isaaclab_tasks.utils import parse_env_cfg
import datetime
from omegaconf import OmegaConf
from kuafu.mpc.simulation.sun_program_executor import SUNProgramExecutor
from kuafu.common.simulation_utils import (
    update_sun_program_executor_data_for_env,
    create_differential_ik_pose_function,
)
import copy

import kuafu.isaac.learned
from kuafu.isaac.learned import kinova, panda
from kuafu.isaac.learned.saver import DemonstrationRecorder


PANDA_POINT_CLOUD_WORKSPACE_MIN = (0.15, -0.6, -0.05)
PANDA_POINT_CLOUD_WORKSPACE_MAX = (0.8, 0.2, 0.5)
PANDA_POINT_CLOUD_TABLE_NORMAL = (
    -0.014077403631341417,
    0.0046943986465211585,
    0.999889888601914,
)
PANDA_POINT_CLOUD_TABLE_D = 0.040399580987183284
PANDA_POINT_CLOUD_TABLE_CLEARANCE_M = 0.015


def _is_panda_task(task_name: str) -> bool:
    return task_name.startswith("Panda-")


# SETTINGS GO HERE ============================================================
script_dir = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
))
ENV_NAME = args_cli.task_name
DEVICE = "cuda:0"
WORKER_ID = args_cli.worker_id
NUM_DEMOS = args_cli.num_demos
# ==============================================================================

def main():
    """Random actions agent with Isaac Lab environment."""
    # ================================
    # Set the agent program path
    # Cannot be changed once the environment is created
    env_cfg = parse_env_cfg(
        ENV_NAME,
        device=DEVICE,
        num_envs=args_cli.num_envs,
    )
    env_cfg.mpc_config["control_output_mode"] = "absolute_ee"
    env_cfg.mpc_config['use_per_env_weight'] = False
    env_cfg.enable_cameras = True
    env_cfg.debug_rewards = True
    env_cfg.sun_program_evaluator_stage_override = True
    control_output_mode = env_cfg.mpc_config["control_output_mode"]
    env_cfg.action_mode = control_output_mode
    project_root = Path(script_dir).resolve()
    configured = os.environ.get("KUAFU_ARTIFACT_ROOT", "").strip()
    external_root = (
        Path(configured).expanduser().resolve()
        if configured
        else project_root.parent / "KuafuArtifacts"
    )
    if args_cli.output_dir:
        output_root = Path(args_cli.output_dir).expanduser()
        if not output_root.is_absolute():
            output_root = external_root / output_root
        output_root = output_root.resolve()
        dataset_dir = output_root / "workers" / f"worker_{WORKER_ID:02d}" / "datasets"
    else:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        dataset_dir = (
            external_root
            / "standalone"
            / "pointcloud_collection"
            / ENV_NAME
            / f"{timestamp}_worker{WORKER_ID:02d}"
            / "datasets"
        )
    dataset_dir = dataset_dir.resolve()
    if dataset_dir == project_root or project_root in dataset_dir.parents:
        raise ValueError(
            f"Trajectory output must be outside the source project ({project_root}); got {dataset_dir}"
        )
    GLOBAL_OUTPUT_DIR = str(dataset_dir.parent)
    os.makedirs(dataset_dir, exist_ok=True)
    # ================================

    env = gym.make(ENV_NAME, cfg=env_cfg)
    env.reset()
    # Ensure PhysX → Python caches (scene.state) are initialized.
    # Without this, early reads of env.unwrapped.scene.state[...] can be NaN.
    env.unwrapped.scene.update(env.unwrapped.physics_dt)
    sim_dt = env.unwrapped.dt

    # DemonstrationRecorder: HDF5 + point cloud recording
    point_cloud_workspace_min = None
    point_cloud_workspace_max = None
    point_cloud_table_plane_normal = None
    point_cloud_table_plane_d = None
    point_cloud_table_clearance_m = 0.0
    if _is_panda_task(ENV_NAME):
        point_cloud_workspace_min = PANDA_POINT_CLOUD_WORKSPACE_MIN
        point_cloud_workspace_max = PANDA_POINT_CLOUD_WORKSPACE_MAX
        point_cloud_table_plane_normal = PANDA_POINT_CLOUD_TABLE_NORMAL
        point_cloud_table_plane_d = PANDA_POINT_CLOUD_TABLE_D
        point_cloud_table_clearance_m = PANDA_POINT_CLOUD_TABLE_CLEARANCE_M
        print(
            "[INFO] Panda point-cloud crop: "
            f"workspace_min={point_cloud_workspace_min}, "
            f"workspace_max={point_cloud_workspace_max}, "
            f"table_plane_filter={point_cloud_table_plane_normal is not None}"
        )

    saver = DemonstrationRecorder(
        base_dir=dataset_dir,
        fps=1.0 / (env_cfg.sim.dt * env_cfg.decimation * max(1, args_cli.stride)),
        record_stride=max(1, args_cli.stride),
        save_states_only=False,
        point_cloud_workspace_min=point_cloud_workspace_min,
        point_cloud_workspace_max=point_cloud_workspace_max,
        point_cloud_table_plane_normal=point_cloud_table_plane_normal,
        point_cloud_table_plane_d=point_cloud_table_plane_d,
        point_cloud_table_object_clearance_m=point_cloud_table_clearance_m,
    )
    saver.task_description = getattr(env_cfg, "task_description", "")

    # Extract per-stage constraints directly from the env's SUNProgramEvaluator program_runtime
    program_evaluator = getattr(env.unwrapped, "program_runtime")
    num_stages = program_evaluator.compiled_program.stage_count()
    CONSTRAINTS = [
        copy.deepcopy(program_evaluator.get_constraints_by_stage(stage_id))
        for stage_id in range(num_stages)
    ]

    # ------------------------------------------------------------------
    # set up a single SUNProgramExecutor controller for one environment
    # (match robot detection logic with collect_from_mpc.py)
    # ------------------------------------------------------------------

    if "panda_shoulder" in env_cfg.robot.actuators:
        robot_name = "panda"
    elif "fr3_shoulder" in env_cfg.robot.actuators:
        robot_name = "fr3"
    elif "kinova_shoulder" in env_cfg.robot.actuators:
        robot_name = "kinova"
    else:
        raise ValueError("Invalid robot name, only panda, fr3, and kinova are supported")

    env_mpc_config = getattr(env_cfg, "mpc_config", {}) or {}
    if robot_name == "panda":
    # MPC / Agent configuration for SUNProgramExecutor (mirrors cabinet_test_constraint_torch_stage)
        MPC_CONFIG = {
            "controller_name": "FrankaIKController",
            "device": env_cfg.sim.device,
            "gripper_opened": env_mpc_config.get("gripper_opened", [0.04, 0.04]),
            "gripper_closed": env_mpc_config.get("gripper_closed", [0.0, 0.0]),
            "gripper_max_force": env_mpc_config.get("gripper_max_force", 40.0),
            "has_finger_contact_sensors": env_mpc_config.get(
                "has_finger_contact_sensors",
                not ENV_NAME.startswith("Panda-Table-"),
            ),
            "robot_gripper_name": env_mpc_config.get(
                "robot_gripper_name",
                "franka_panda" if ENV_NAME.startswith("Panda-Table-") else "panda",
            ),
            "controller": {
                "urdf_path": f"{script_dir}/resources/robots/franka_description/robots/panda_arm_hand_fixed.urdf",
                "ee_frame": env_mpc_config.get("ee_frame", "panda_fingertip"),
                "horizon_length": 4,
                "time_step": sim_dt,
            },
            "load_program": None,
            "control_output_mode": control_output_mode,
            "agent": {
                "id": "agent",
                "device": env_cfg.sim.device,
                "task": "SUNProgramExecutor control for pen_cup_recall_batched",
                "prompt_dir": f"{script_dir}/assets/prompts/Manipulation",
                "initial_residual": f"{script_dir}/src/kuafu/skills/constraints/residuals.py",
                "initial_predicate": f"{script_dir}/src/kuafu/skills/constraints/predicates.py",
                "model_version": "offline",
                "load_path": None,
                "reuse_program": True,
                "max_correction_steps": 5,
            },
        }
    elif robot_name == "fr3":
        MPC_CONFIG = {
            "controller_name": "FR3FinRayIKController",
            "device": env_cfg.sim.device,
            "gripper_opened": env_mpc_config.get("gripper_opened", [0.0, 0.0]),
            "gripper_closed": env_mpc_config.get("gripper_closed", [0.8, -0.8]),
            "gripper_max_force": env_mpc_config.get("gripper_max_force", 1.0),
            "robot_gripper_name": env_mpc_config.get("robot_gripper_name", "fr3_finray"),
            "controller": {
                "urdf_path": f"{script_dir}/resources/robots/franka_description/robots/panda_arm_hand_fr3_finray.urdf",
                "ee_frame": env_mpc_config.get("ee_frame", "panda_fingertip"),
                "horizon_length": 4,
                "time_step": sim_dt,
            },
            "load_program": None,
            "control_output_mode": control_output_mode,
            "agent": {
                "id": "agent",
                "device": env_cfg.sim.device,
                "task": "lalalal",
                "prompt_dir": f"{script_dir}/assets/prompts/Manipulation",
                "initial_residual": f"{script_dir}/src/kuafu/skills/constraints/residuals.py",
                "initial_predicate": f"{script_dir}/src/kuafu/skills/constraints/predicates.py",
                "model_version": "offline",
                "load_path": None,
                "reuse_program": True,
                "max_correction_steps": 5,
            },
        }
    elif robot_name == "kinova":
        MPC_CONFIG = {
            "controller_name": "KinovaIKController",
            "device": env_cfg.sim.device,
            "gripper_opened": env_mpc_config.get("gripper_opened", [0.0, 0.0]),
            "gripper_closed": env_mpc_config.get("gripper_closed", [0.8, 0.8]),
            "gripper_max_force": env_mpc_config.get("gripper_max_force", 1.0),
            "gripper_effort_contact_velocity_threshold": env_mpc_config.get(
                "gripper_effort_contact_velocity_threshold", 0.15
            ),
            "kinova_gripper_effort_contact_openness_slope": env_mpc_config.get(
                "kinova_gripper_effort_contact_openness_slope", 12.0
            ),
            "kinova_gripper_effort_contact_openness_intercept": env_mpc_config.get(
                "kinova_gripper_effort_contact_openness_intercept", 0.0
            ),
            "robot_gripper_name": env_mpc_config.get("robot_gripper_name", "kinova_robotiq"),
            "controller": {
                "urdf_path": f"{script_dir}/resources/robots/kinova_description/robots/gen3_7dof_vision_controller.urdf",
                "ee_frame": env_mpc_config.get("ee_frame", "panda_fingertip"),
                "horizon_length": 4,
                "time_step": sim_dt,
            },
            "load_program": None,
            "control_output_mode": control_output_mode,
            "agent": {
                "id": "agent",
                "device": env_cfg.sim.device,
                "task": "SUNProgramExecutor control for Kinova manipulation",
                "prompt_dir": f"{script_dir}/assets/prompts/Manipulation",
                "initial_residual": f"{script_dir}/src/kuafu/skills/constraints/residuals.py",
                "initial_predicate": f"{script_dir}/src/kuafu/skills/constraints/predicates.py",
                "model_version": "offline",
                "load_path": None,
                "reuse_program": True,
                "max_correction_steps": 5,
            },
        }
    cfg_sun_program_executor = OmegaConf.create(MPC_CONFIG)

    # Optional: Differential IK hookup for better kinematics
    inverse_kinematics = create_differential_ik_pose_function(env, env_cfg, robot_name)
    program_executor = SUNProgramExecutor(cfg_sun_program_executor)
    program_executor.active_controller._ik_fun = inverse_kinematics
    if hasattr(program_executor, "test_controller"):
        program_executor.test_controller._ik_fun = inverse_kinematics

    # Get the total number of stages/subtasks (for information only)
    print(f"\nTotal number of stages in program_runtime (SUNProgramEvaluator): {program_evaluator.compiled_program.stage_count()}")

    # Simulation start here
    RECOMPUTE_CALL_INTERVAL = 3
    RECOMPILE_CALL_INTERVAL = RECOMPUTE_CALL_INTERVAL * 10
    count = 0
    RECOMPUTE_CALL_COUNTER = 0
    RECOMPILE_CALL_COUNTER = 0
    stage_idx = 0
    stride = max(1, int(args_cli.stride))
    if stride > 1:
        print(f"[INFO] Storing every {stride} simulation steps.")
    step_count = 0
    print("\nStarting main simulation loop...\n")

    # let simulation stablize, not moving
    update_sun_program_executor_data_for_env(env, program_executor, robot_name)
    program_executor.controller_api_wrapper.updatePOVState()
    program_executor.program_learner._constraint_generator._constraints = CONSTRAINTS[0]
    program_executor.program_learner.apply_constraints()

    def _is_stopped():
        return saver.metadata_info["num_episodes"] >= NUM_DEMOS

    def reset_sun_program_executor(idx=0):
        stage_idx = idx
        update_sun_program_executor_data_for_env(env, program_executor, robot_name)
        program_executor.controller_api_wrapper.updatePOVState()
        program_executor.program_learner._constraint_generator._constraints = CONSTRAINTS[stage_idx]
        program_executor.program_learner.apply_constraints()

    def reset_episode_and_sun_program_executor(idx=0):
        # Avoid relying on env.step() auto-reset timing; do an explicit reset so
        # sun_program_executor always reads a fully refreshed episode state.
        env.reset()
        env.unwrapped.scene.update(env.unwrapped.physics_dt)
        reset_sun_program_executor(idx)

    def discard_recorded_attempt(reason: str):
        if len(saver.ee_positions) == 0 and not getattr(saver, "_file_handles", {}):
            return
        num_envs = int(getattr(env.unwrapped, "num_envs", args_cli.num_envs))
        dones = torch.ones(num_envs, device=env_cfg.sim.device, dtype=torch.bool)
        successes = torch.zeros(num_envs, device=env_cfg.sim.device, dtype=torch.bool)
        saver.write(dones, successes)
        print(f"[INFO] Discarded buffered pointcloud attempt after {reason}.")

    while simulation_app.is_running() and not _is_stopped():
        with torch.inference_mode():

            update_sun_program_executor_data_for_env(env, program_executor, robot_name)

            sun_program_executor_pred = program_executor.quantifyPredicate()
            is_success = sun_program_executor_pred[0].item()

            if count == RECOMPUTE_CALL_COUNTER:
                u_seq = program_executor.computeControl()
                last_computed_count = count
                RECOMPUTE_CALL_COUNTER += RECOMPUTE_CALL_INTERVAL

            u_tensor = torch.from_numpy(u_seq[(count - last_computed_count) +1]).to(env_cfg.sim.device).to(dtype=torch.float32)
            action = u_tensor.unsqueeze(0)
            if step_count % stride == 0:
                saver.store(action, env, decision_stage=stage_idx)
            count += 1
            _, _, terminated, truncated, _ = env.step(action)
            step_count += 1

            dones = terminated | truncated
            if dones.any():
                # Use _cached_success so we only save true success (terminate_external),
                # not false_finished; fallback to terminated for envs that don't set it.
                successes = getattr(env.unwrapped, "_cached_success", terminated)
                if successes is None:
                    successes = terminated
                num_saved = saver.write(dones, successes)
                if successes.any():
                    print("ENV finished, if there was no prior message, then the task worked!")
                else:
                    print("ENV TIMED OUT")
                print(f"Total episodes saved: {num_saved}")
                stage_idx = 0
                count = 0
                RECOMPUTE_CALL_COUNTER = 0
                RECOMPILE_CALL_COUNTER = 0
                reset_episode_and_sun_program_executor()
                continue

            if is_success:
                print(f"Stage {stage_idx} passed!")
                if stage_idx < len(CONSTRAINTS) - 1:
                    stage_idx += 1
                    reset_sun_program_executor(stage_idx)
                    RECOMPUTE_CALL_COUNTER = count
                    RECOMPILE_CALL_COUNTER = count + RECOMPILE_CALL_INTERVAL
                else:
                    print("All stages completed across the batch, if you are seeing this message, it means the task has false finished.")
                    print(f"Total episodes saved: {saver.metadata_info['num_episodes']}")
                    discard_recorded_attempt("false finish")
                    stage_idx = 0
                    count = 0
                    RECOMPUTE_CALL_COUNTER = 0
                    RECOMPILE_CALL_COUNTER = RECOMPILE_CALL_INTERVAL
                    reset_episode_and_sun_program_executor()

    saver.stop()
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
