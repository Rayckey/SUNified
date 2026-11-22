#!/usr/bin/env python3
"""Evaluate a native robomimic checkpoint in a batched unified environment.

This adapter supports robomimic checkpoints only. residual-RL BC-prior checkpoints
use a different schema and should be evaluated with
``src/kuafu/bc/training/play_bc_env.py``.
"""

from __future__ import annotations

import argparse
import faulthandler
import os
from pathlib import Path

from kuafu.paths import resolve_artifact_path
import sys
import traceback

_WORKSPACE_ROOT = Path(__file__).resolve().parents[4]

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default=None)
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--params-path", type=Path, required=True)
parser.add_argument("--program-path", type=Path, default=None)
parser.add_argument("--num-envs", type=int, default=8192)
parser.add_argument("--duration-sec", type=float, default=180.0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output", type=Path, default=None)
parser.add_argument("--obs-key", default="state", choices=("state", "full_state"))
parser.add_argument(
    "--action-mode",
    default="absolute_ee",
    choices=("absolute_ee", "relative_ee", "absolute_joint", "relative_joint"),
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()


def _recipe_or_artifact_path(value: str | Path, *, allow_learned_program: bool = False) -> Path:
    path = Path(value).expanduser()
    source_prefixes = {"resources", "src"}
    if allow_learned_program:
        source_prefixes.add("outputs")
    if not path.is_absolute() and path.parts and path.parts[0] in source_prefixes:
        return (_WORKSPACE_ROOT / path).resolve()
    return resolve_artifact_path(path, "Robomimic evaluation input")


if args_cli.checkpoint:
    args_cli.checkpoint = str(resolve_artifact_path(args_cli.checkpoint, "Robomimic checkpoint"))
args_cli.params_path = _recipe_or_artifact_path(args_cli.params_path)
if args_cli.program_path is not None:
    args_cli.program_path = _recipe_or_artifact_path(
        args_cli.program_path, allow_learned_program=True
    )

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json
import math
import random

import gymnasium as gym
import numpy as np
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils
import torch

from kuafu.rl.isaac.update_utilities import update_parameters, update_weights
from isaaclab_tasks.utils import parse_env_cfg


def _register_task(task: str) -> None:
    if task.startswith("Kinova-"):
        from kuafu.isaac.learned import kinova  # noqa: F401
    elif task.startswith("Panda-"):
        from kuafu.isaac.learned import panda  # noqa: F401
    else:
        from kuafu.isaac import learned  # noqa: F401


def _enable_legacy_prior_program_compatibility(task: str) -> None:
    """Preserve the validation semantics used when these prior sweeps were evaluated."""

    def _allow_legacy_constraint(_constraints):
        return []

    import kuafu.programs.compiler.constraint_validation as validation
    import kuafu.programs.compiler.predicate_generator as predicate_generator
    import kuafu.programs.compiler.predicate_residual_generator as predicate_residual_generator
    import kuafu.programs.compiler.residual_generator as residual_generator

    modules = (
        validation,
        predicate_generator,
        predicate_residual_generator,
        residual_generator,
    )
    for module in modules:
        module.find_manipulator_effort_condition_conflicts = _allow_legacy_constraint
    print(
        "[ROBOMIMIC EVAL] Enabled legacy prior compatibility for mixed "
        "high/low-effort condition alternatives.",
        flush=True,
    )

    if "EStop-Open" in task:
        for module in modules:
            module.find_directional_rotate_axis_align_conflicts = _allow_legacy_constraint
        print(
            "[ROBOMIMIC EVAL] Enabled legacy EStop-Open compatibility for "
            "directional-rotate/axis-align alternatives.",
            flush=True,
        )


def _batched_policy_action(policy, policy_obs, obs_key: str, device: torch.device) -> torch.Tensor:
    """Run one robomimic forward pass over the complete environment batch."""
    if isinstance(policy_obs, torch.Tensor):
        obs = {obs_key: policy_obs}
    else:
        if obs_key not in policy_obs:
            raise KeyError(
                f"Environment policy observations do not contain {obs_key!r}: {tuple(policy_obs)}"
            )
        obs = {obs_key: policy_obs[obs_key]}

    obs = TensorUtils.to_float(
        TensorUtils.to_device(TensorUtils.to_tensor(obs), policy.policy.device)
    )
    if policy.obs_normalization_stats is not None:
        stats = TensorUtils.to_float(
            TensorUtils.to_device(
                TensorUtils.to_tensor(policy.obs_normalization_stats),
                policy.policy.device,
            )
        )
        obs = {key: obs[key] for key in policy.policy.global_config.all_obs_keys}
        obs = ObsUtils.normalize_obs(obs, obs_normalization_stats=stats)

    actions = policy.policy.get_action(obs_dict=obs)
    if not isinstance(actions, torch.Tensor):
        actions = torch.as_tensor(actions)
    return actions.to(device=device, dtype=torch.float32)


def _assert_stage_observation(
    policy_obs, obs_key: str, raw_env, expected_shape
) -> None:
    if isinstance(policy_obs, torch.Tensor):
        observation = policy_obs
    else:
        if obs_key not in policy_obs:
            raise KeyError(f"Missing {obs_key!r} while validating stage observation")
        observation = policy_obs[obs_key]

    if tuple(observation.shape[1:]) != tuple(expected_shape):
        raise ValueError(
            f"Checkpoint expects {obs_key} shape {tuple(expected_shape)}, but the environment "
            f"returned {tuple(observation.shape[1:])}"
        )
    if not hasattr(raw_env, "current_stage"):
        raise RuntimeError("Program-enabled environment does not expose current_stage")
    expected_stage = raw_env.current_stage.reshape(-1).to(
        device=observation.device, dtype=observation.dtype
    )
    observed_stage = observation[:, -1]
    if observed_stage.shape != expected_stage.shape or not torch.allclose(
        observed_stage, expected_stage
    ):
        raise RuntimeError(
            "The final robomimic observation scalar does not match current_stage"
        )
    print(
        f"[ROBOMIMIC EVAL] Verified live stage in {obs_key} observation "
        f"(shape={tuple(observation.shape[1:])}, "
        f"stage_range={float(observed_stage.min()):.0f}..{float(observed_stage.max()):.0f}).",
        flush=True,
    )


def _assert_robomimic_checkpoint(path: Path) -> None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    required = {"model", "config", "shape_metadata"}
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(
            f"{path} is not a native robomimic checkpoint (missing {sorted(missing)}). "
            "residual-RL BC-prior checkpoints are not supported by this adapter."
        )


def _apply_sweep_parameters(env, params_path: Path) -> int:
    raw = json.loads(params_path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a JSON object in {params_path}")
    stages = raw.get("stages")
    if not isinstance(stages, dict) or not stages:
        raise ValueError(f"Missing non-empty 'stages' object in {params_path}")

    all_stage_weights = {}
    for stage_id_text, items in stages.items():
        if not isinstance(items, dict):
            raise ValueError(f"Invalid stage {stage_id_text} in {params_path}")
        parameters = items.get("parameters", {})
        weights = items.get("weights")
        if not isinstance(parameters, dict) or not weights:
            raise ValueError(
                f"Stage {stage_id_text} needs parameters and non-empty weights in {params_path}"
            )
        stage_id = int(stage_id_text)
        update_parameters(env, stage_id=stage_id, parameters=parameters)
        all_stage_weights[stage_id] = weights
    update_weights(env, stage_id=None, weights=None, all_stages_weights=all_stage_weights)

    stage_count = int(env.program_runtime.compiled_program.stage_count())
    expected_stages = set(range(stage_count))
    actual_stages = {int(stage_id) for stage_id in stages}
    if actual_stages != expected_stages:
        raise ValueError(
            f"Parameter stages {sorted(actual_stages)} do not match loaded program "
            f"stages {sorted(expected_stages)} for {params_path}"
        )
    return stage_count


def main() -> None:
    if not args_cli.task or not args_cli.checkpoint:
        parser.error("--task and --checkpoint are required")
    checkpoint_path = Path(args_cli.checkpoint)
    params_path = args_cli.params_path
    if not params_path.is_file():
        raise FileNotFoundError(f"Sweep parameter JSON not found: {params_path}")
    _assert_robomimic_checkpoint(checkpoint_path)
    _enable_legacy_prior_program_compatibility(args_cli.task)
    _register_task(args_cli.task)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    if hasattr(env_cfg, "set_obs_mode"):
        env_cfg.set_obs_mode(args_cli.obs_key)
    if hasattr(env_cfg, "action_mode"):
        env_cfg.action_mode = args_cli.action_mode
    for key, value in (
        ("enable_cameras", False),
        ("debug_rewards", False),
        ("use_debug_action_mode", False),
        # Match prior evaluation: stage predicates advance current_stage and a
        # completed final stage remains a termination outcome.
        ("sun_program_evaluator_stage_override", False),
        ("use_external_rewards", False),
        ("skip_dense_reward_computation", True),
    ):
        if hasattr(env_cfg, key):
            setattr(env_cfg, key, value)

    if not hasattr(env_cfg, "mpc_config"):
        raise RuntimeError(f"{args_cli.task} does not expose mpc_config program")
    program_value = (
        str(args_cli.program_path)
        if args_cli.program_path is not None
        else env_cfg.mpc_config.get("load_program")
    )
    if not program_value:
        raise RuntimeError(f"{args_cli.task} has no configured SUNProgramEvaluator program path")
    program_path = Path(program_value).expanduser().resolve()
    if not program_path.exists():
        raise FileNotFoundError(f"SUNProgramEvaluator program path not found: {program_path}")
    env_cfg.mpc_config["load_program"] = str(program_path)

    traceback_interval = float(os.environ.get("ROBOMIMIC_TRACEBACK_INTERVAL_SEC", "0"))
    if traceback_interval > 0:
        faulthandler.dump_traceback_later(traceback_interval, repeat=True)
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    if traceback_interval > 0:
        faulthandler.cancel_dump_traceback_later()
    raw_env = env.unwrapped
    stage_count = _apply_sweep_parameters(raw_env, params_path)
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    random.seed(args_cli.seed)
    env.seed(args_cli.seed)
    if hasattr(raw_env, "clear_end_to_end_rate_stats"):
        raw_env.clear_end_to_end_rate_stats()
    if hasattr(raw_env, "clear_stage_transition_stats"):
        raw_env.clear_stage_transition_stats()

    # Keep policy and simulator tensors on the same explicitly selected device.
    device = torch.device(args_cli.device)
    policy, checkpoint = FileUtils.policy_from_checkpoint(
        ckpt_path=str(checkpoint_path), device=device, verbose=True
    )
    expected_action_dim = int(checkpoint["shape_metadata"]["ac_dim"])
    env_action_dim = int(env.action_space.shape[-1])
    if expected_action_dim != env_action_dim:
        raise ValueError(
            f"Checkpoint action dim {expected_action_dim} does not match {args_cli.action_mode} "
            f"environment action dim {env_action_dim}."
        )

    policy.start_episode()
    obs_dict, _ = env.reset()
    expected_obs_shape = checkpoint["shape_metadata"]["all_shapes"].get(args_cli.obs_key)
    if expected_obs_shape is None:
        raise KeyError(f"Checkpoint has no shape metadata for {args_cli.obs_key!r}")
    _assert_stage_observation(
        obs_dict["policy"], args_cli.obs_key, raw_env, expected_obs_shape
    )
    stage_observation_verified = True
    step_dt = float(
        getattr(raw_env, "step_dt", None)
        or getattr(raw_env, "dt", None)
        or (env_cfg.sim.dt * getattr(env_cfg, "decimation", 1))
    )
    target_steps = int(math.ceil(args_cli.duration_sec / step_dt))
    total_episodes = 0
    step_count = 0
    simulated_sec = 0.0
    while step_count < target_steps and simulation_app.is_running():
        with torch.inference_mode():
            actions = _batched_policy_action(
                policy, obs_dict["policy"], args_cli.obs_key, device
            )
        obs_dict, _, terminated, truncated, _ = env.step(actions)
        done = terminated | truncated
        step_count += 1
        simulated_sec = min(step_count * step_dt, args_cli.duration_sec)
        n_done = int(done.sum().item())
        if n_done > 0:
            total_episodes += n_done
            successful = (
                int(raw_env.get_true_success_count())
                if hasattr(raw_env, "get_true_success_count")
                else 0
            )
            print(
                f"[ROBOMIMIC EVAL] sim_sec={simulated_sec:.3f} "
                f"episodes={total_episodes} successful={successful} "
                f"success_rate={successful / max(total_episodes, 1):.4f}",
                flush=True,
            )

    successful = (
        int(raw_env.get_true_success_count())
        if hasattr(raw_env, "get_true_success_count")
        else 0
    )
    false_completions = (
        int(raw_env.get_false_complete_count())
        if hasattr(raw_env, "get_false_complete_count")
        else 0
    )
    timeouts = (
        int(raw_env.get_timeout_count())
        if hasattr(raw_env, "get_timeout_count")
        else 0
    )
    result = {
        "robomimic_only": True,
        "task": args_cli.task,
        "checkpoint": str(checkpoint_path),
        "params_path": str(params_path),
        "program_path": str(program_path),
        "stage_count": stage_count,
        "stage_observation_verified": stage_observation_verified,
        "obs_key": args_cli.obs_key,
        "action_mode": args_cli.action_mode,
        "num_envs": args_cli.num_envs,
        "duration_sec": args_cli.duration_sec,
        "simulated_sec": simulated_sec,
        "steps": step_count,
        "episodes": total_episodes,
        "successful": successful,
        "false_completions": false_completions,
        "timeouts": timeouts,
        "success_rate": successful / max(total_episodes, 1),
    }
    if args_cli.output is not None:
        output_path = resolve_artifact_path(args_cli.output, "Robomimic evaluation result")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
