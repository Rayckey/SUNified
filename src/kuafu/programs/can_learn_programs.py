from kuafu.paths import program_output_root
"""Launch Isaac Sim Simulator first for single-can manipulation environment."""

import argparse
from pathlib import Path
from isaaclab.app import AppLauncher

# -----------------------------------------------------------------------------
# CLI Args
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Franka manipulation with single can")
parser.add_argument("--disable_fabric", type=bool, default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--enable_camera", type=bool, default=False, help="Ignored for learn-programs; cameras stay disabled.")

# Append AppLauncher args & parse
parser.add_argument(
    "--max_atomic_corrections",
    type=int,
    default=5,
    help="Maximum automatic atomic-action correction attempts per failed stage.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app  # expose simulation_app for Isaac Lab lifecycle control

import gymnasium as gym
import torch
import os
import datetime
import time
import copy
import numpy as np
from omegaconf import OmegaConf

import isaaclab_tasks  # noqa: F401 (keeps registration utilities available)
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import subtract_frame_transforms

# Project utilities
from kuafu.programs.robust_loop import run_robust_program_learning_loop
from kuafu.common.simulation_utils import (
    apply_franka_panda_gripper_config,
    update_meta_in_controller,
    write_txt,
    combine_joint_states,
    extract_odom_manipulators,
    extract_frame_sensors,
)
from kuafu.mpc.simulation.sun_program_executor import SUNProgramExecutor
from kuafu.isaac.formulation.panda import assert_program_free_panda_env
from isaacsim.core.utils.stage import get_current_stage
# -----------------------------------------------------------------------------
# Environment registration
# -----------------------------------------------------------------------------
ENV_NAME = "Formulate-Panda-Can-v0"
OUTPUT_ENV_NAME = "manipulation-can"

# -----------------------------------------------------------------------------
# Name map for rigid objects + frame sensors (keys must match scene registry names)
# -----------------------------------------------------------------------------
detectable_objects = {
    "can": (
        "cylindrical coke can rigid body, local -y is the upright axis and long axis, size (0.04, 0.1, 0.04)",
        "/World/envs/env_*/Can",
    ),
    "can_center_frame": (
        "cylindrical coke can, local -y is the upright axis and long axis, size (0.04, 0.1, 0.04)",
        "/Visuals/can_center_frame",
    ),

    "panda_link4": ("frame of robot link 4", "/Visuals/panda_link4_frame"),
    "panda_fingertip": ("frame of robot tool frame", "/Visuals/panda_hand_frame"),
}

DEVICE = args_cli.device
VLM_CALL_INTERVAL = 700
RECOMPUTE_CALL_INTERVAL = 10
MAX_COUNTER = None
MAX_FAIL_FOR_TASK = 5


def main():

    # Parse environment configuration
    env_cfg = parse_env_cfg(ENV_NAME, device=DEVICE, num_envs=args_cli.num_envs)
    env = gym.make(ENV_NAME, cfg=env_cfg)
    assert_program_free_panda_env(env)
    env.reset()

    sim_dt = env_cfg.sim.dt
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    GLOBAL_OUTPUT_DIR = os.path.join(
        str(program_output_root()), f"{OUTPUT_ENV_NAME}", f"{timestamp}"
    )
    os.makedirs(GLOBAL_OUTPUT_DIR, exist_ok=True)

    # ----------------------------------------------------------------------------
    # Controller setup (Differential IK)
    # ----------------------------------------------------------------------------
    robot = env.unwrapped.scene["robot"]  # type: ignore[attr-defined]
    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    diff_ik_controller = DifferentialIKController(
        diff_ik_cfg, num_envs=env.unwrapped.scene.num_envs, device=env_cfg.sim.device  # type: ignore[attr-defined]
    )
    robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
    robot_entity_cfg.resolve(env.unwrapped.scene)  # type: ignore[attr-defined]
    ee_body_id = robot_entity_cfg.body_ids[0]
    ee_jacobi_idx = ee_body_id - 1

    def inverse_kinematics(p_des, quat_des, q_init=None, frame_name="tool"):
        """Compute joint positions to reach desired end-effector pose in world coordinates."""
        # set desired pose
        diff_ik_controller.set_command(torch.concatenate((torch.tensor(p_des), torch.tensor(quat_des))))
        jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
        ee_pose_w = robot.data.body_pose_w[:, ee_body_id]
        root_pose_w = robot.data.root_pose_w
        joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
        return joint_pos_des.detach().cpu()

    # ----------------------------------------------------------------------------
    # MPC / Agent configuration (kept structure for compatibility)
    # ----------------------------------------------------------------------------
    MPC_CONFIG = {
        "controller_name": "FrankaIKController",
        "device": DEVICE,
        "controller": {
            "urdf_path": f"{script_dir}/resources/robots/franka_description/robots/panda_arm_hand_fixed.urdf",
            "ee_frame": "panda_fingertip",
            "horizon_length": 50,
            "time_step": sim_dt,
        },
        "load_program": None,
        "manipulator_high_effort_threshold": 20.0,
        "agent": {
            "id": "agent",
            "device": DEVICE,
            "task": "Pick up the fallen over can and place it back down in upright position",
            "prompt_dir": f"{script_dir}/assets/prompts/Manipulation",
            "initial_residual": f"{script_dir}/src/kuafu/skills/constraints/residuals.py",
            "initial_predicate": f"{script_dir}/src/kuafu/skills/constraints/predicates.py",
            "model_version": "gpt-5.5",
            "load_path": None,
            "reuse_program": True,
            "max_correction_steps": 5,
        },
    }

    MPC_CONFIG["save_dir"] = GLOBAL_OUTPUT_DIR
    apply_franka_panda_gripper_config(MPC_CONFIG)
    def make_program_runtime():
        cfg = OmegaConf.create(MPC_CONFIG)
        program_runtime = SUNProgramExecutor(cfg)
        program_runtime.active_controller._ik_fun = inverse_kinematics
        program_runtime.test_controller._ik_fun = inverse_kinematics
        return program_runtime

    program_runtime = make_program_runtime()


    # ----------------------------------------------------------------------------
    # Data update helpers
    # ----------------------------------------------------------------------------
    def update_data_in_controller(
        env_number, program_runtime, use_camera=False, includes_meta=False
    ):
        odom = extract_odom_manipulators(env, detectable_objects)
        sensors = extract_frame_sensors(env, {}, detectable_objects)
        program_runtime.callbackOdom(odom, sensors)
        program_runtime.callbackJointState(combine_joint_states(env, robot="franka"))

    # Simulation start
    # ----------------------------------------------------------------------------
    # Robust control loop
    # ----------------------------------------------------------------------------
    CONSTRAINT_CALL_INTERVAL = 700
    RECOMPUTE_CALL_INTERVAL = 10
    MAX_COUNTER = None

    # Let simulation settle before the controller starts planning.
    dummy_actions = env.unwrapped.scene.state["articulation"]["robot"]["joint_position"]
    for _ in range(5):
        env.step(dummy_actions)

    run_robust_program_learning_loop(
        env=env,
        env_cfg=env_cfg,
        program_runtime=program_runtime,
        make_program_runtime=make_program_runtime,
        update_data_in_controller=update_data_in_controller,
        simulation_app=simulation_app,
        output_dir=GLOBAL_OUTPUT_DIR,
        max_counter=None,
        enable_stall_detection=True,
        max_atomic_corrections=args_cli.max_atomic_corrections,
        constraint_call_interval=CONSTRAINT_CALL_INTERVAL,
        recompute_call_interval=RECOMPUTE_CALL_INTERVAL,
        task_update_uses_camera=False,
        apply_constraint_kwargs={"print_function": True},
        restart_settle_steps=5,
    )

if __name__ == "__main__":
    main()
    simulation_app.close()
