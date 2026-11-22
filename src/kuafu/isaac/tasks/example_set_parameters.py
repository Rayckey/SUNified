# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to an environment with random action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric",
    type=bool,
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--num_envs", type=int, default=12, help="Number of environments to simulate."
)

import gymnasium as gym

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import os
from isaaclab_tasks.utils import parse_env_cfg
import copy


ENV_NAME = "example-manipulation-fr3-finray-drawer"
gym.register(
    id=ENV_NAME,  # keep this example-only registration separate from learn_programs tasks
    entry_point="kuafu.isaac.tasks.open_drawer.open_fr3_finray_drawer_env_learned:OpenDrawerEnv",
    disable_env_checker=True,
    kwargs={
        # <----- This is the only Isaac-Lab–specific bit
        "env_cfg_entry_point": (
            "kuafu.isaac.tasks.open_drawer.open_fr3_finray_drawer_env_learned:OpenDrawerEnvCfg"
        )
    },
)

DEVICE = "cuda:0"

def main():
    """Random actions agent with Isaac Lab environment."""
    env_cfg = parse_env_cfg(
        ENV_NAME,
        device=DEVICE,
        num_envs=args_cli.num_envs,
    )


    # LILILI ==========================
    # Set the agent program path
    # Cannot be changed once the environment is created
    script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))))
    env_cfg.mpc_config['load_program'] = f"{script_dir}/resources/learned_programs/manipulation-fr3-finray-drawer/2026-01-06_21-21-38"
    env_cfg.mpc_config["num_envs"] = env_cfg.scene.num_envs
    env_cfg.use_debug_action_mode = False
    # ================================


    env = gym.make(ENV_NAME, cfg=env_cfg)
    env.reset()

    sim_dt = env_cfg.sim.dt
    script_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))))
    )
    # let simulation stablize
    dummy_actions = torch.zeros_like(
        env.unwrapped.scene.state["articulation"]["robot"]["joint_position"],
        device=env_cfg.sim.device
    )

    # not moving
    env.step(dummy_actions)


    # LILILI ==========================
    # Example: Using the Residual Weights API

    program_runtime = env.unwrapped.program_runtime

    stage_idx = 0
    # Get the text descriptions of the constraints
    print("\n" + "="*60)
    print("The explanations of the constraints")
    print("="*60)
    explanations = program_runtime.explain_constraints(stage_idx)
    for explanation in explanations:
        print(explanation)

    # get a list of weights
    print("\n" + "="*60)
    print("The weights of the constraints")
    print("="*60)
    weights = program_runtime.get_residual_weights(stage_idx)

    # change the weights
    print(f"Resetting the weights on the run: {weights}")
    weights = {i: weights[i]* 0.5 for i in range(len(explanations))}
    program_runtime.set_residual_weights(stage_idx, weights)

    env.step(dummy_actions)

    # LILILI ==========================
    # Example: Get constraints, modify parameters, and reload them
    print("\n" + "="*60)
    print(f"The Overall task is {env_cfg.task_description}")
    print("Example: Modifying constraint parameters and reloading")
    print("="*60)

    # Get the full constraints dictionary for a stage
    modified_constraints = copy.deepcopy(program_runtime.get_constraints_by_stage(stage_idx))
    atomic_action = program_runtime.get_atomic_action_by_id(stage_idx)
    print(f"\nAtomic action for stage {stage_idx}: {atomic_action}")

    print(f"\nOriginal constraints for stage {stage_idx}:")
    for i, constraint in enumerate(modified_constraints.get("Constraints", [])):
        print(f"  Constraint {i}: op={constraint.get('op')}, args={constraint.get('args', {})}")

    # Modify parameters in the constraints
    # Example: Modify tolerance_m for reach_distance constraints
    for constraint in modified_constraints["Constraints"]:
        if constraint.get("op") == "reach_distance":
            args = constraint.get("args", {})
            if "tolerance_m" in args:
                original_tolerance = args["tolerance_m"]
                args["tolerance_m"] = original_tolerance * 1.5  # Increase tolerance by 50%
                print(f"\nModified reach_distance tolerance: {original_tolerance} -> {args['tolerance_m']}")
        elif constraint.get("op") == "reach_point":
            args = constraint.get("args", {})
            if "tolerance_m" in args:
                original_tolerance = args["tolerance_m"]
                args["tolerance_m"] = original_tolerance * 1.2  # Increase tolerance by 20%
                print(f"Modified reach_point tolerance: {original_tolerance} -> {args['tolerance_m']}")
        elif constraint.get("op") == "axis_align":
            args = constraint.get("args", {})
            if "tolerance_rad" in args:
                original_tolerance = args["tolerance_rad"]
                args["tolerance_rad"] = original_tolerance * 1.3  # Increase angular tolerance by 30%
                print(f"Modified axis_align tolerance_rad: {original_tolerance} -> {args['tolerance_rad']}")

    # Reload the modified constraints
    # Note: constraints_list should be a list of constraint dictionaries (one per stage)
    # stage_indices specifies which stages to update
    print(f"\nReloading constraints for stage {stage_idx}...")
    program_runtime.reload_constraints(
        constraints_list=[modified_constraints],  # List of constraint dicts (one per stage)
        stage_indices=[stage_idx]  # List of stage indices to update
    )
    env.step(dummy_actions)
    print("Constraints reloaded successfully!")
    print("="*60)

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
