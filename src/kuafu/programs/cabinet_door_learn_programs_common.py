# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared learn-programs runner for the Sektion cabinet door-opening tasks."""

from __future__ import annotations
from kuafu.paths import program_output_root

import datetime
import os
import time
from pathlib import Path

import gymnasium as gym
import torch
from kuafu.mpc.simulation.sun_program_executor import SUNProgramExecutor
from kuafu.isaac.formulation.panda import assert_program_free_panda_env
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab_tasks.utils import parse_env_cfg
from isaacsim.core.utils.stage import get_current_stage
from omegaconf import OmegaConf

from kuafu.programs.robust_loop import run_robust_program_learning_loop
from kuafu.common.simulation_utils import (
    apply_franka_panda_gripper_config,
    combine_joint_states,
    extract_frame_sensors,
    extract_odom_manipulators,
    update_meta_in_controller,
    write_txt,
)


def run_cabinet_door_learn_programs(
    env_name: str,
    output_env_name: str,
    task_description: str,
    detectable_objects: dict,
    args_cli,
    simulation_app,
    *,
    device: str = "cuda:0",
    max_counter: int | None = None,
    max_atomic_corrections: int = 5,
    manipulator_high_effort_threshold: float | None = None,
    has_finger_contact_sensors: bool = False,
    keep_running: bool = False,
):
    """Run the SUNProgramExecutor learn-programs loop for a cabinet door task."""

    env_cfg = parse_env_cfg(
        env_name,
        device=device,
        num_envs=args_cli.num_envs,
    )

    if hasattr(env_cfg, "enable_cameras"):
        env_cfg.enable_cameras = False
    if hasattr(env_cfg, "enable_camera"):
        env_cfg.enable_camera = False
    env = gym.make(env_name, cfg=env_cfg)
    assert_program_free_panda_env(env)
    env.reset()

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    global_output_dir = os.path.join(
        str(program_output_root()), output_env_name, f"{timestamp}"
    )
    os.makedirs(global_output_dir, exist_ok=True)

    rgb_save_dir = os.path.join(global_output_dir, "images")
    os.makedirs(rgb_save_dir, exist_ok=True)

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

    mpc_config = {
        "controller_name": "FrankaIKController",
        "device": device,
        "controller": {
            "urdf_path": (
                f"{script_dir}/resources/robots/franka_description/robots/"
                "panda_arm_hand_fixed.urdf"
            ),
            "ee_frame": "panda_fingertip",
            "horizon_length": 15,
            "time_step": 1 / 30,
        },
        "load_program": None,
        "manipulator_high_effort_threshold": 100.0,
        "agent": {
            "id": "agent",
            "device": device,
            "task": task_description,
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
        "save_dir": global_output_dir,
    }
    if manipulator_high_effort_threshold is not None:
        mpc_config["manipulator_high_effort_threshold"] = float(
            manipulator_high_effort_threshold
        )
        print(
            "Manipulator high-effort threshold set to "
            f"{mpc_config['manipulator_high_effort_threshold']}"
        )
    if has_finger_contact_sensors:
        print(
            "Ignoring has_finger_contact_sensors=True: raw Franka learn_programs "
            "envs use Panda joint-effort grasp detection like learned Panda."
        )
    apply_franka_panda_gripper_config(mpc_config)
    print(
        "Object-grasped predicate using Panda gripper joint effort fallback "
        "instead of fingertip contact sensors"
    )
    def make_program_runtime():
        cfg = OmegaConf.create(mpc_config)
        program_runtime = SUNProgramExecutor(cfg)
        program_runtime.active_controller._ik_fun = inverse_kinematics
        program_runtime.test_controller._ik_fun = inverse_kinematics
        return program_runtime

    program_runtime = make_program_runtime()

    def update_data_in_controller(
        env_number, program_runtime, use_camera=False, includes_meta=False
    ):
        odom = extract_odom_manipulators(env, detectable_objects)
        sensors = extract_frame_sensors(env, {}, detectable_objects)
        program_runtime.callbackOdom(odom, sensors)
        program_runtime.callbackJointState(combine_joint_states(env, robot="franka"))

    dummy_actions = env.unwrapped.scene.state["articulation"]["robot"][
        "joint_position"
    ]
    for _ in range(100):
        env.step(dummy_actions)

    run_robust_program_learning_loop(
        env=env,
        env_cfg=env_cfg,
        program_runtime=program_runtime,
        make_program_runtime=make_program_runtime,
        update_data_in_controller=update_data_in_controller,
        simulation_app=simulation_app,
        output_dir=global_output_dir,
        max_counter=max_counter,
        enable_stall_detection=True,
        max_atomic_corrections=max_atomic_corrections,
        constraint_call_interval=700,
        recompute_call_interval=10,
        task_update_uses_camera=False,
        apply_constraint_kwargs={"print_function": True},
        restart_settle_steps=100,
        keep_running=keep_running,
    )
