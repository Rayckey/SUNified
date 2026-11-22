# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test script: run env like Kuafu train, build prompts for the first VLM call, print them and stop.

Runs some fake env steps (random actions) before collecting task_info and building prompts, then
matches the Kuafu train flow up to (but not including) the VLM call. Use --fake_steps to control
how many steps run before prompts are built.

Example (from repo root):
  python src/kuafu/isaac/tasks/preview_program_prompts.py --task open-fr3-finray-drawer --device cuda:0 --num_envs 128
  python src/kuafu/isaac/tasks/preview_program_prompts.py --task open-fr3-finray-drawer --fake_steps 100
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, cast

import torch

# Ensure Kuafu is importable (for prompt templates)
_repo_root = Path(__file__).resolve().parent.parent.parent

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Test script: env + Kuafu prompts (no VLM). Same args as Kuafu train."
)
parser.add_argument("--task", type=str, default="open-fr3-finray-drawer", help="Gym task name (e.g. open-fr3-finray-drawer).")
parser.add_argument("--num_envs", type=int, default=128, help="Number of environments.")
parser.add_argument(
    "--disable_fabric",
    type=bool,
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--fake_steps",
    type=int,
    default=50,
    help="Number of env steps to run with random actions before building prompts (default: 50).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
from isaaclab_tasks.utils import parse_env_cfg
import copy

# Register open-fr3-finray-drawer (same as scripts/residual_rl/train.py)
import kuafu.isaac.tasks.open_drawer  # noqa: F401


def _collect_task_info(env, *, task: str) -> dict[str, Any] | None:
    """Collect task metadata from env (same logic as Kuafu worker_pool._collect_task_info)."""
    env_u = env.unwrapped
    program_runtime = getattr(env_u, "program_runtime", None)
    if program_runtime is None:
        return None
    info: dict[str, Any] = {}
    try:
        cfg = getattr(env_u, "cfg", None) or getattr(env, "cfg", None)
        task_desc = getattr(cfg, "task_description", None)
        if isinstance(task_desc, str) and task_desc.strip():
            info["task_description"] = task_desc.strip()
    except Exception:
        pass
    try:
        stage_count = int(program_runtime.compiled_program.stage_count())
    except Exception:
        return None
    info["stage_count"] = stage_count
    stages: dict[int, dict[str, Any]] = {}
    for stage_id in range(stage_count):
        try:
            atomic_action = None
            try:
                atomic_action = program_runtime.get_atomic_action_by_id(stage_id)
            except Exception:
                atomic_action = None
            constraints_raw = None
            try:
                cdict = program_runtime.get_constraints_by_stage(stage_id)
                clist = []
                for c in cdict["Constraints"]:
                    if isinstance(c, dict):
                        clist.append({"op": c["op"], "args": c["args"]})
                constraints_raw = clist
            except Exception:
                constraints_raw = None
            stages[stage_id] = {
                "explanations": program_runtime.explain_constraints(stage_id),
                "weights": program_runtime.get_residual_weights(stage_id),
                "atomic_action": atomic_action,
                "constraints_raw": constraints_raw,
            }
        except Exception:
            stages[stage_id] = {"explanations": None, "weights": None}
    info["stages"] = stages
    try:
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

        agent_cfg = load_cfg_from_registry(task, "rl_games_cfg_entry_point")
        agent_dict = cast(dict[str, Any], agent_cfg)
        params = agent_dict["params"]
        cfg = params["config"]
        info["rl_games"] = {
            "save_frequency": int(cfg["save_frequency"]),
            "save_best_after": int(cfg["save_best_after"]),
        }
    except Exception:
        pass
    return info


def main():
    """Create env (Kuafu-style), collect task_info, build prompts for stage 0, print and exit (no VLM)."""
    task = args_cli.task
    device = args_cli.device
    num_envs = args_cli.num_envs

    # Same env creation as Kuafu env_factory
    env_cfg = parse_env_cfg(task, device=device, num_envs=num_envs)
    # Optional: set mpc_config if your env requires it (e.g. load_program)
    mpc_config = getattr(env_cfg, "mpc_config")
    mpc_config["num_envs"] = env_cfg.scene.num_envs
    mpc_config["use_per_env_weight"] = False
    setattr(env_cfg, "use_debug_action_mode", False)
    env = gym.make(task, cfg=env_cfg)
    env.reset()

    # Run fake actions so simulation advances before we build prompts
    fake_steps = getattr(args_cli, "fake_steps", 50)
    if fake_steps > 0:
        scene = getattr(env.unwrapped, "scene", None)
        joint_pos = scene.state["articulation"]["robot"]["joint_position"] if scene else None
        if joint_pos is not None:
            action_shape = joint_pos.shape
        else:
            action_shape = (env_cfg.scene.num_envs, 9)
        device = env_cfg.sim.device
        for _ in range(fake_steps):
            random_actions = (torch.rand(*action_shape, device=device) * 2.0 - 1.0)
            env.step(random_actions)
        print(f"[preview_program_prompts] Ran {fake_steps} fake steps (random actions).", flush=True)

    # Collect task_info (same as Kuafu worker after env creation)
    raw_task_info = _collect_task_info(env, task=task)
    if not raw_task_info or not isinstance(raw_task_info, dict):
        print("[preview_program_prompts] ERROR: No task_info (env has no program_runtime or stage_count).", flush=True)
        env.close()
        return
    task_info = cast(dict[str, Any], raw_task_info)

    stage_count = int(task_info["stage_count"])
    if stage_count <= 0:
        print("[preview_program_prompts] ERROR: stage_count <= 0.", flush=True)
        env.close()
        return

    # Build the exact prompts that Kuafu would send for stage 0, first evo turn (no prior run)
    from kuafu.rl.llm.prompts import (
        SINGLE_WEIGHT_PROMPT,
        SINGLE_WEIGHT_TUNING_FORMATTING_INSTRUCTIONS,
        SYSTEM_PROMPT,
    )
    from kuafu.common.env_prompt_helpers import (
        format_constraint_satisfaction_rates,
        format_reward_values,
        format_stage_satisfaction_times,
    )

    stage_id = 0
    stages = task_info["stages"]
    stage_meta = cast(dict[str, Any], stages[stage_id])
    explanations = stage_meta["explanations"]
    current_weights = stage_meta["weights"]
    atomic_action = stage_meta["atomic_action"]
    current_weights_jsonable = {int(k): float(v) for k, v in dict(current_weights).items()}

    task_description = str(task_info["task_description"]).strip()
    constraints_explanations_str = "\n".join(f"- {e}" for e in explanations)
    current_weights_str = json.dumps(current_weights_jsonable, indent=2, sort_keys=True)

    # Get real stats from env after fake steps (same APIs as Kuafu worker_pool)
    env_u = env.unwrapped
    get_stage_stats = getattr(env_u, "get_stage_transition_stats")
    get_constraint_stats_fn = getattr(env_u, "get_constraint_stats") or getattr(
        env_u, "get_constraint_met_rate_stats"
    )
    stage_transition_stats = get_stage_stats() if get_stage_stats is not None else None
    highest_stage_reached_per_env = getattr(env_u, "get_highest_stage_reached_per_env")()
    constraint_met_rate_stats = get_constraint_stats_fn() if get_constraint_stats_fn is not None else None

    single_weight_system = SYSTEM_PROMPT + SINGLE_WEIGHT_TUNING_FORMATTING_INSTRUCTIONS
    stage_satisfaction_times = format_stage_satisfaction_times(
        stage_transition_stats,
        highest_stage_reached_per_env=highest_stage_reached_per_env,
    )
    constraint_satisfaction_rates = format_constraint_satisfaction_rates(
        constraint_met_rate_stats, stage_id=stage_id
    )
    reward_values = format_reward_values(constraint_met_rate_stats, stage_id=stage_id)
    stage_user_prompt = SINGLE_WEIGHT_PROMPT.format(
        task_description=task_description,
        stage_id=stage_id,
        stage_count=stage_count,
        atomic_action=str(atomic_action),
        constraints_explanations=constraints_explanations_str,
        current_weights=current_weights_str,
        stage_satisfaction_times=stage_satisfaction_times,
        constraint_satisfaction_rates=constraint_satisfaction_rates,
        reward_values=reward_values,
    )

    # Print task_info summary and full prompts (would be sent to VLM)
    print("\n" + "=" * 70, flush=True)
    print("TASK INFO (from env program_runtime)", flush=True)
    print("=" * 70, flush=True)
    print(f"task_description: {task_description}", flush=True)
    print(f"stage_count: {stage_count}", flush=True)
    print(f"stage_meta[0] keys: {list(stage_meta.keys())}", flush=True)
    print("=" * 70, flush=True)
    print("SYSTEM MESSAGE (would be sent to VLM)", flush=True)
    print("=" * 70, flush=True)
    print(single_weight_system, flush=True)
    print("=" * 70, flush=True)
    print("USER MESSAGE (stage 0, first turn – would be sent to VLM)", flush=True)
    print("=" * 70, flush=True)
    print(stage_user_prompt, flush=True)
    print("=" * 70, flush=True)
    print("Stopping before VLM call (test script).", flush=True)
    print("=" * 70 + "\n", flush=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
