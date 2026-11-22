"""Environment + SimulationApp creation utilities.

IMPORTANT (Isaac Sim):
- `AppLauncher` must be constructed before importing most Isaac Lab modules.
"""

from __future__ import annotations

import inspect
import os


def _infer_initial_obs_mode_from_prior(prior_path: str | None) -> str | None:
    if not prior_path:
        return None
    from kuafu.bc.training.rl_games_prior import infer_prior_obs_key

    return infer_prior_obs_key(prior_path)


def _load_agent_cfg_from_yaml(path: str) -> dict:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise TypeError(f"agent cfg YAML must load to a dict: {path}")
    return cfg


def create_env_and_app(
    *,
    task: str,
    device: str,
    env_seed: int,
    num_envs: int | None,
    headless: bool,
    execution_robot_id: str | None = None,
    initial_prior_checkpoint: str | None = None,
    agent_cfg_path: str | None = None,
):
    # Isaac Sim must be launched first.
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=headless, device=device)
    simulation_app = app_launcher.app

    # Only after SimulationApp exists:
    import gymnasium as gym
    from kuafu.rl.isaac.task_registry import register_task_modules
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    resolved_robot_id = register_task_modules(task=task, execution_robot_id=execution_robot_id)
    print(f"[residual_rl] Registered task modules for execution_robot_id={resolved_robot_id}.", flush=True)

    # parse_env_cfg may return DirectRLEnvCfg or ManagerBasedRLEnvCfg depending on task registry.
    env_cfg = parse_env_cfg(task)
    env_cfg.sim.device = device
    env_cfg.seed = int(env_seed)
    obs_mode = _infer_initial_obs_mode_from_prior(initial_prior_checkpoint)
    if obs_mode is not None and hasattr(env_cfg, "set_obs_mode"):
        env_cfg.set_obs_mode(obs_mode)
        print(f"[residual_rl] Using obs_mode={obs_mode} inferred from BC prior config.", flush=True)
    if num_envs is not None:
        env_cfg.scene.num_envs = int(num_envs)

    if agent_cfg_path:
        agent_cfg = _load_agent_cfg_from_yaml(agent_cfg_path)
        print(f"[residual_rl] Using agent_cfg_path={agent_cfg_path}", flush=True)
    else:
        agent_cfg = load_cfg_from_registry(task, "rl_games_cfg_entry_point")

    env_cfg.mpc_config["num_envs"] = env_cfg.scene.num_envs
    env_cfg.mpc_config["use_per_env_weight"] = False
    from kuafu.artifact_contracts import apply_learned_program

    apply_learned_program(env_cfg)
    mpc_section = agent_cfg["params"].get("mpc", None)
    if mpc_section:
        for key in (
            "use_per_env_weight",
            "use_absolute_progress_rewards",
            "use_relative_progress_rewards",
            "use_velocity_penalty",
            "use_time_penalty",
            "regularize_velocity_scale",
            "regularize_time_scale",
            "progress_rewards_scale",
            "stage_transition_bonus_scale",
            "stage_transition_bonus_use_remaining_horizon",
            "penalize_false_completions",
            "skip_dense_reward_computation",
            "ignore_stage_gate_for_success",
            "use_external_rewards",
            "external_rewards_scale",
            "reward_print_every",
            "episode_outcome_print_every",
            "collect_terminal_stuck_constraint_stats",
        ):
            if key in mpc_section:
                env_cfg.mpc_config[key] = mpc_section[key]
                setattr(env_cfg, key, mpc_section[key])

    env = gym.make(task, cfg=env_cfg)
    return simulation_app, env, agent_cfg


def get_observations_method_as_string(env) -> str:
    env_u = env.unwrapped
    try:
        return inspect.getsource(env_u._get_observations)
    except Exception:
        return ""

_CONSTRAINT_ARG_NORMALIZATION_REMOVED = True
"""Constraints use canonical frame and offset keys at every pipeline boundary."""
