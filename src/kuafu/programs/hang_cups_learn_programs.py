from kuafu.paths import program_output_root
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
    help="Maximum automatic atomic-action correction attempts per failed stage.",
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

ENV_NAME = "Formulate-Panda-Hang-Cup-v0"
OUTPUT_ENV_NAME = "manipulation-hang-cup"

import os
import isaaclab_tasks
import datetime
from kuafu.mpc.simulation.sun_program_executor import SUNProgramExecutor
from omegaconf import OmegaConf
from isaaclab.managers import SceneEntityCfg
from kuafu.programs.robust_loop import run_robust_program_learning_loop
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
from isaaclab.utils.math import subtract_frame_transforms
from omni.usd import get_prim_at_path
import time
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg

detectable_objects = {
    "mug_center_frame": (
        "center frame of mug; mug body is too wide for the gripper; use virtual handles instead; size: (0.13, 0.1, 0.1)",
        "/World/envs/env_*/mug_center_frame",
    ),
    "mug_rim_frame_0": (
        "rim frame 0 of mug, virtual handle frame, virtual handle axis is long local + x or -x, virtual approach is local + z",
        "/World/envs/env_*/mug_rim_frame_0",
    ),

    "mug_rim_frame_1": (
        "rim frame 0 of mug, virtual handle frame, virtual handle axis is long local + x or -x, virtual approach is local + z",
        "/World/envs/env_*/mug_rim_frame_1",
    ),

    "mug_handle_hole_frame": (
        "center of handle opening of mug; handle axis -x; hole/opening axis is ±y (bidirectional); too wide for the gripper; not a graspable frame",
        "/Visuals/mug_handle_hole_frame",
    ),
    "cup_hanger": (
        "base of cup holder, static single-branch hanger; base diameter 0.16 m, pole height 0.185 m, one tilted branch only",
        "/World/envs/env_*/CupHolder",
    ),
    "cup_hanger_branch_1_tip_frame": (
        "tip frame of the only hanger branch; branch axis is +y (base->tip, outward toward local -Y and upward 20 deg); engagement/hooking moves along -y (tip->base, inward), exposed length is 0.110 m, diameter is 0.012 m",
        "/Visuals/cup_hanger_branch_1_tip_frame",
    ),
    "table": (
        "center frame of table, size: (0.3, 0.9, 0.4)",
        "/World/envs/env_*/Table",
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
    env_cfg = parse_env_cfg(
        ENV_NAME,
        device=DEVICE,
        num_envs=args_cli.num_envs,
    )

    env = gym.make(ENV_NAME, cfg=env_cfg)
    assert_program_free_panda_env(env)
    env.reset()

    sim_dt = env.unwrapped.dt
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    GLOBAL_OUTPUT_DIR = os.path.join(
        str(program_output_root()), f"{OUTPUT_ENV_NAME}", f"{timestamp}"
    )
    os.makedirs(GLOBAL_OUTPUT_DIR, exist_ok=True)

    # Camera output writer (optional)
    rgb_save_dir = os.path.join(GLOBAL_OUTPUT_DIR, "images")
    os.makedirs(rgb_save_dir, exist_ok=True)

    # ----------------------------------------------------------------------------
    # Controller setup (Differential IK)
    # ----------------------------------------------------------------------------
    robot = env.unwrapped.scene["robot"]  # type: ignore[attr-defined]
    diff_ik_cfg = DifferentialIKControllerCfg(
        command_type="pose", use_relative_mode=False, ik_method="dls"
    )
    diff_ik_controller = DifferentialIKController(
        diff_ik_cfg,
        num_envs=env.unwrapped.scene.num_envs,  # type: ignore[attr-defined]
        device=env_cfg.sim.device,
    )
    robot_entity_cfg = SceneEntityCfg(
        "robot", joint_names=["panda_joint.*"], body_names=["panda_hand"]
    )
    robot_entity_cfg.resolve(env.unwrapped.scene)  # type: ignore[attr-defined]
    ee_body_id = robot_entity_cfg.body_ids[0]
    ee_jacobi_idx = ee_body_id - 1

    def inverse_kinematics(p_des, quat_des, q_init=None, frame_name="tool"):
        """Compute joint positions to reach desired end-effector pose in world coordinates."""
        diff_ik_controller.set_command(
            torch.concatenate((torch.Tensor(p_des), torch.Tensor(quat_des)), dim=-1)
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

    MPC_CONFIG = {
        "controller_name": "FrankaIKController",
        "device": DEVICE,
        "controller": {
            "urdf_path": f"{script_dir}/resources/robots/franka_description/robots/panda_arm_hand_fixed.urdf",
            "ee_frame": "panda_fingertip",
            "horizon_length": 6,
            "time_step": sim_dt,
        },
        "load_program": None,
        "manipulator_high_effort_threshold": 100.0,
        "agent": {
            "id": "agent",
            "device": DEVICE,
            "task": f"hook the mug handle through the single cup hanger branch",
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
    RECOMPUTE_CALL_INTERVAL = 5
    MAX_COUNTER = None
    MAX_STEPS_PER_CONSTRAINT = 500

    # Let simulation settle before the controller starts planning.
    dummy_actions = env.unwrapped.scene.state["articulation"]["robot"]["joint_position"]
    for _ in range(100):
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
        enable_stall_detection=False,
        max_atomic_corrections=args_cli.max_atomic_corrections,
        constraint_call_interval=MAX_STEPS_PER_CONSTRAINT,
        recompute_call_interval=RECOMPUTE_CALL_INTERVAL,
        task_update_uses_camera=False,
        restart_settle_steps=100,
        apply_constraint_kwargs={},
    )

if __name__ == "__main__":
    main()
    simulation_app.close()
