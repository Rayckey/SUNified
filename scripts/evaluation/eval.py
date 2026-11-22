"""Evaluate the best residual-RL checkpoint with optional GUI and video recording."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, cast

SOURCE_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from kuafu.paths import require_external_artifact_path, resolve_artifact_path
from kuafu.rl.isaac.task_registry import register_task_modules
from isaaclab.app import AppLauncher

def _resolve_run_dir(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def _infer_task_from_run_dir(run_dir: Path) -> str | None:
    run_params_path = run_dir / "run_params.json"
    if run_params_path.is_file():
        data = _load_json(run_params_path)
        argv = data.get("argv")
        if isinstance(argv, list):
            for idx, token in enumerate(argv[:-1]):
                if token == "--task":
                    value = argv[idx + 1]
                    if isinstance(value, str) and value.strip():
                        return value.strip()
    parent_name = run_dir.parent.name.strip()
    return parent_name or None


def _select_checkpoint(run_dir: Path, explicit_checkpoint: str | None, stage: int | None) -> tuple[Path, str]:
    if explicit_checkpoint:
        checkpoint_path = Path(explicit_checkpoint).expanduser()
        if not checkpoint_path.is_absolute():
            checkpoint_path = (Path.cwd() / checkpoint_path).resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        return checkpoint_path, "explicit checkpoint"

    best_json_path = run_dir / "best.json"
    if not best_json_path.is_file():
        raise FileNotFoundError(f"Missing best.json in run directory: {best_json_path}")
    best_data = _load_json(best_json_path)

    if stage is not None:
        stage_entry = (best_data.get("stages") or {}).get(str(stage))
        if not isinstance(stage_entry, dict):
            raise ValueError(f"Stage {stage} not found in {best_json_path}")
        checkpoint = stage_entry.get("checkpoint")
        if not isinstance(checkpoint, str) or not checkpoint.strip():
            raise ValueError(f"Stage {stage} in {best_json_path} has no checkpoint")
        checkpoint_path = Path(checkpoint).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        return checkpoint_path, f"best stage {stage}"

    global_entry = best_data.get("global")
    if isinstance(global_entry, dict):
        checkpoint = global_entry.get("checkpoint")
        if isinstance(checkpoint, str) and checkpoint.strip():
            checkpoint_path = Path(checkpoint).expanduser().resolve()
            if checkpoint_path.is_file():
                return checkpoint_path, "best global checkpoint"

    stages = best_data.get("stages") or {}
    if not isinstance(stages, dict) or not stages:
        raise ValueError(f"No stage checkpoints found in {best_json_path}")

    ranked_stage_ids: list[int] = []
    for key in stages:
        try:
            ranked_stage_ids.append(int(key))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid stage id in {best_json_path}: {key!r}") from exc
    if not ranked_stage_ids:
        raise ValueError(f"No numeric stages found in {best_json_path}")

    best_stage = max(ranked_stage_ids)
    stage_entry = stages[str(best_stage)]
    if not isinstance(stage_entry, dict):
        raise ValueError(f"Invalid entry for stage {best_stage} in {best_json_path}")
    checkpoint = stage_entry.get("checkpoint")
    if not isinstance(checkpoint, str) or not checkpoint.strip():
        raise ValueError(f"Stage {best_stage} in {best_json_path} has no checkpoint")
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return checkpoint_path, f"highest solved stage ({best_stage})"


def _infer_checkpoint_obs_dim(checkpoint_path: Path) -> int | None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_state = checkpoint.get("model")
    if not isinstance(model_state, dict):
        return None

    for key in (
        "running_mean_std.running_mean",
        "a2c_network.actor_mlp.0.weight",
    ):
        value = model_state.get(key)
        if isinstance(value, torch.Tensor):
            if key.endswith(".weight") and value.ndim >= 2:
                return int(value.shape[1])
            if value.ndim >= 1:
                return int(value.shape[0])
    return None


def _maybe_set_obs_mode_from_checkpoint(env_cfg: Any, checkpoint_path: Path) -> str | None:
    if not hasattr(env_cfg, "set_obs_mode") or not hasattr(env_cfg, "obs_dims"):
        return None

    obs_dims = getattr(env_cfg, "obs_dims", None)
    if not isinstance(obs_dims, dict):
        return None

    checkpoint_obs_dim = _infer_checkpoint_obs_dim(checkpoint_path)
    if checkpoint_obs_dim is None:
        return None

    for mode_name, obs_dim in obs_dims.items():
        try:
            if int(obs_dim) == checkpoint_obs_dim:
                env_cfg.set_obs_mode(str(mode_name))
                return str(mode_name)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid obs dim for mode {mode_name!r}: {obs_dim!r}") from exc
    return None


parser = argparse.ArgumentParser(description="Evaluate the best checkpoint from a residual-RL run directory.")
parser.add_argument(
    "--run_dir",
    type=str,
    required=True,
    help="Residual-RL run directory, e.g. logs/residual_rl/<task>/<timestamp>.",
)
parser.add_argument("--task", type=str, default=None, help="Task override. By default this is inferred from run_params.json.")
parser.add_argument(
    "--execution_robot_id",
    type=str,
    default=None,
    help="Robot env package to execute. Defaults to inference from --task; legacy tasks use fr3_finray.",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Checkpoint override. By default this is auto-selected from best.json.",
)
parser.add_argument(
    "--stage",
    type=int,
    default=None,
    help="Evaluate the best checkpoint for a specific stage from best.json.",
)
parser.add_argument(
    "--agent",
    type=str,
    default="rl_games_cfg_entry_point",
    help="Name of the RL-Games config entry point registered for the task.",
)
parser.add_argument("--num_envs", type=int, default=32, help="Number of evaluation environments to run.")
parser.add_argument(
    "--duration",
    type=float,
    default=60.0,
    help="Run evaluation for this many wall-clock seconds. Use 0 or a negative value to run until the window is closed.",
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=None,
    help="Optional hard stop on simulator steps, useful when episodes are long.",
)
parser.add_argument(
    "--count_all_dones",
    action="store_true",
    default=False,
    help="Use every terminated or truncated episode in final rate denominators and report physics breaks.",
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for evaluation. Set -1 to randomize.")
parser.add_argument("--video", action="store_true", default=False, help="Record one evaluation clip while continuing evaluation.")
parser.add_argument("--video_length", type=int, default=400, help="Length of the recorded video in simulator steps.")
parser.add_argument(
    "--video_dir",
    type=str,
    default=None,
    help="Optional output directory for videos. Defaults to <run_dir>/videos/eval/<timestamp>/.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time when possible.")
parser.add_argument(
    "--print_every",
    type=int,
    default=10,
    help="Print a progress summary each time this many more episodes complete.",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
    parser.print_help()
    raise SystemExit(0)

AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()
args_cli.run_dir = _resolve_run_dir(args_cli.run_dir)
args_cli.task = args_cli.task or _infer_task_from_run_dir(args_cli.run_dir)
if not args_cli.task:
    raise ValueError("Could not infer the task. Please pass --task explicitly.")
if args_cli.video:
    args_cli.enable_cameras = True

# Clear out sys.argv for Hydra.
sys.argv = [sys.argv[0]] + hydra_args

# Launch Omniverse app.
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
resolved_robot_id = register_task_modules(task=args_cli.task, execution_robot_id=args_cli.execution_robot_id)
print(f"[residual_rl] Registered task modules for execution_robot_id={resolved_robot_id}.", flush=True)


import gymnasium as gym
import torch
from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config


def _get_count(env: Any, attr_name: str) -> int:
    fn = getattr(env, attr_name, None)
    if callable(fn):
        return int(cast(Any, fn()))
    return 0


def _summarize_outcomes(env: Any) -> tuple[int, int, int, int, float]:
    true_success = _get_count(env, "get_true_success_count")
    false_complete = _get_count(env, "get_false_complete_count")
    timeout = _get_count(env, "get_timeout_count")
    total = true_success + false_complete + timeout
    success_rate = true_success / total if total > 0 else 0.0
    return true_success, false_complete, timeout, total, success_rate


def _make_video_dir(run_dir: Path, override_dir: str | None) -> str:
    if override_dir:
        video_dir = resolve_artifact_path(override_dir, "Evaluation video directory")
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        video_dir = run_dir / "videos" / "eval" / timestamp
    video_dir.mkdir(parents=True, exist_ok=True)
    return str(video_dir.resolve())


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict) -> None:
    checkpoint_path, checkpoint_desc = _select_checkpoint(args_cli.run_dir, args_cli.checkpoint, args_cli.stage)
    resolved_obs_mode = _maybe_set_obs_mode_from_checkpoint(env_cfg, checkpoint_path)

    env_cfg.scene.num_envs = int(args_cli.num_envs)
    if getattr(args_cli, "device", None) is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg["params"]["config"]["device"] = args_cli.device
        agent_cfg["params"]["config"]["device_name"] = args_cli.device

    # Keep the learned env in sync with the evaluation actor count.
    if hasattr(env_cfg, "mpc_config") and isinstance(env_cfg.mpc_config, dict):
        env_cfg.mpc_config["num_envs"] = env_cfg.scene.num_envs
        env_cfg.mpc_config["use_per_env_weight"] = False

    mpc_cfg = agent_cfg.get("mpc")
    if isinstance(mpc_cfg, dict) and hasattr(env_cfg, "mpc_config") and isinstance(env_cfg.mpc_config, dict):
        for key in (
            "use_per_env_weight",
            "use_absolute_progress_rewards",
            "use_relative_progress_rewards",
            "use_velocity_penalty",
            "regularize_velocity_scale",
            "regularize_time_scale",
            "progress_rewards_scale",
        ):
            if key in mpc_cfg:
                env_cfg.mpc_config[key] = mpc_cfg[key]

    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    if args_cli.seed is not None:
        agent_cfg["params"]["seed"] = int(args_cli.seed)
    env_cfg.seed = agent_cfg["params"]["seed"]
    env_cfg.log_dir = str(args_cli.run_dir)

    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)

    print("[INFO] Residual-RL evaluation setup")
    print(f"[INFO] Task           : {args_cli.task}")
    print(f"[INFO] Run dir        : {args_cli.run_dir}")
    print(f"[INFO] Checkpoint     : {checkpoint_path}")
    print(f"[INFO] Selection      : {checkpoint_desc}")
    print(f"[INFO] Num envs       : {env_cfg.scene.num_envs}")
    print(f"[INFO] Duration (s)   : {args_cli.duration}")
    print(f"[INFO] Seed           : {env_cfg.seed}")
    if resolved_obs_mode is not None:
        print(f"[INFO] Obs mode       : {resolved_obs_mode} (matched to checkpoint)")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_folder = _make_video_dir(args_cli.run_dir, args_cli.video_dir)
        video_kwargs = {
            "video_folder": video_folder,
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording evaluation video.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups)

    vecenv.register(
        "IsaacRlgWrapper",
        lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = str(checkpoint_path)
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs

    runner = Runner()
    runner.load(agent_cfg)
    agent: BasePlayer = runner.create_player()
    agent.restore(str(checkpoint_path))
    agent.reset()

    raw_env = env.unwrapped
    if hasattr(raw_env, "clear_end_to_end_rate_stats"):
        raw_env.clear_end_to_end_rate_stats()

    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    _ = agent.get_batch_size(obs, 1)

    dt = raw_env.step_dt
    total_steps = 0
    total_dones = 0
    last_reported_episode = 0
    eval_start_time = time.time()

    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            obs = agent.obs_to_torch(obs)
            actions = agent.get_action(obs, is_deterministic=agent.is_deterministic)
            obs, _, dones, _ = env.step(actions)

            n_done = 0
            if isinstance(dones, torch.Tensor):
                n_done = int(dones.sum().item())
            elif len(dones) > 0:
                n_done = int(sum(dones))
            total_dones += n_done

            if n_done > 0:
                true_success, false_complete, timeout, total_completed, success_rate = _summarize_outcomes(raw_env)
                should_report = (
                    args_cli.print_every <= 0
                    or total_completed >= last_reported_episode + args_cli.print_every
                )
                if should_report:
                    print(
                        f"[EVAL] Episodes={total_completed} | "
                        f"True success={true_success} | "
                        f"False complete={false_complete} | "
                        f"Timeout={timeout} | "
                        f"True success rate={success_rate * 100:.1f}%"
                    )
                    last_reported_episode = total_completed

        total_steps += 1
        elapsed_time = time.time() - eval_start_time
        if args_cli.duration > 0 and elapsed_time >= args_cli.duration:
            break
        if args_cli.max_steps is not None and total_steps >= args_cli.max_steps:
            break

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    true_success, false_complete, timeout, total_completed, success_rate = _summarize_outcomes(raw_env)
    physics_break = 0
    if args_cli.count_all_dones or bool(getattr(raw_env.cfg, "kuafu_external_only", False)):
        counter_total = true_success + false_complete + timeout
        physics_break = max(total_dones - counter_total, 0)
        total_completed = total_dones
        success_rate = true_success / total_completed if total_completed > 0 else 0.0
    false_complete_rate = false_complete / total_completed if total_completed > 0 else 0.0
    timeout_rate = timeout / total_completed if total_completed > 0 else 0.0
    physics_break_rate = physics_break / total_completed if total_completed > 0 else 0.0
    elapsed_time = time.time() - eval_start_time

    print("\n" + "=" * 68)
    print("  RESIDUAL-RL EVAL SUMMARY")
    print(f"  Elapsed time (s)     : {elapsed_time:.2f}")
    print(f"  Completed episodes   : {total_completed}")
    print(f"  True successes       : {true_success}")
    print(f"  False completes      : {false_complete}")
    print(f"  Timeouts             : {timeout}")
    print(f"  Physics breaks       : {physics_break}")
    print(f"  True success rate    : {success_rate * 100:.2f}%")
    print(f"  False complete rate  : {false_complete_rate * 100:.2f}%")
    print(f"  Timeout rate         : {timeout_rate * 100:.2f}%")
    print(f"  Physics break rate   : {physics_break_rate * 100:.2f}%")
    print(f"  Total simulator steps: {total_steps}")
    print("=" * 68 + "\n")

    env.close()


if __name__ == "__main__":
    main()  # type: ignore[call-arg]
    simulation_app.close()
