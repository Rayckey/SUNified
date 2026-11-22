"""Play a trained plain-PyTorch BC policy in an Isaac Lab environment.

Mirrors play_policy_env.py but loads checkpoints from train_bc_plain.py
instead of robomimic. The policy runs batched over all envs simultaneously
(no per-env loop). Supports action chunking: when chunk_size > 1 the policy
outputs chunk_size actions at once and they are executed one-by-one before
re-querying the network (receding-horizon execution).

Args:
    task:          Name of the Isaac Lab task / environment.
    checkpoint:    Path to .pth checkpoint from train_bc_plain.py.
    duration_sec:  Simulation time (seconds) to run.
    num_envs:      Number of parallel environments.
    chunk_exec:    How many actions from each chunk to actually execute
                   before re-querying (0 = use full chunk_size from ckpt).
    seed:          Random seed.
"""

import argparse
import sys
from pathlib import Path

_source_root = Path(__file__).resolve().parents[3]
_script_dir = Path(__file__).resolve().parent
for _path in (_source_root, _script_dir):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from kuafu.paths import resolve_artifact_path
from kuafu.artifact_contracts import apply_learned_program, resolve_parameters_path, validate_program_metadata
from kuafu.rl.isaac.task_registry import register_task_modules, resolve_execution_robot_id  # noqa: E402
from kuafu.rl.isaac.update_utilities import update_weights, update_parameters  # noqa: E402

from isaaclab.app import AppLauncher

# -----------------------------------------------------------------------------
# CLI  (AppLauncher must parse before any Isaac imports)
# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Play a plain-PyTorch BC policy in Isaac Lab.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--task", type=str, default="FR3-FinRay-Table-Can")
parser.add_argument(
    "--execution_robot_id",
    type=str,
    default=None,
    help="Robot task package to register. Defaults to inference from --task, then FR3FinRay-Franka.",
)
parser.add_argument("--dry_run", action="store_true", default=False, help="Resolve task registration and exit before launching Isaac.")
parser.add_argument(
    "--checkpoint",
    type=str,
    default=str(resolve_artifact_path(
        "logs/bc_prior_mlp/2026-03-26_05-42-50/checkpoints/best.pth",
        "BC prior checkpoint",
    )),
    help="Path to .pth checkpoint from train_bc_plain.py",
)
parser.add_argument("--duration_sec", type=float, default=120.0)
parser.add_argument("--program_path", default=None, help="Learned program used during collection; overrides the task default.")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument(
    "--chunk_exec",
    type=int,
    default=1,
    help="Actions from each chunk to execute before re-querying (0 = full chunk)",
)
parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help="Random seed for reproducibility. If omitted, runs are non-deterministic.",
)
parser.add_argument(
    "--dataset",
    type=str,
    default=None,
    help="Path to HDF5 dataset. If provided, each env is initialized from a random episode's "
         "initial_state instead of the default random reset.",
)
parser.add_argument(
    "--demo_index",
    type=int,
    default=None,
    help="If set, only use this 0-based demo index from the HDF5 dataset.",
)
parser.add_argument(
    "--mode",
    type=str,
    default="policy",
    choices=["policy", "replay", "examine"],
    help="'policy': run the BC policy. 'replay': replay recorded actions from --dataset. "
         "'examine': replay dataset actions but print L2 error between dataset action and policy action.",
)
parser.add_argument(
    "--past_begin_start",
    default=False,
    help="If set, initialize each env from a random timestep in the episode (reads obs/full_state[t]) "
         "instead of always starting from the beginning. Requires --dataset.",
)
parser.add_argument(
    "--timeshift",
    type=int,
    default=0,
    help="Action index offset relative to the observation timestep for replay mode. "
         "0 = use actions[t], -1 = use actions[t+1] (1 step future), +1 = use actions[t-1] (1 step past). "
         "If omitted, defaults to the checkpoint's time_shift when available.",
)
parser.add_argument(
    "--action_mode",
    type=str,
    default="absolute_ee",
    choices=["relative_joint", "absolute_joint", "absolute_ee", "relative_ee"],
    help="Override the action mode used for replay/examine/policy. "
         "If not set, auto-detected from the checkpoint or dataset.",
)

parser.add_argument(
    "--params_path",
    type=str,
    default=None,
    help="Path to a best.json file (e.g. logs/residual_rl/<task>/<timestamp>/best.json). "
         "Overwrites per-stage constraint parameters and residual weights on the environment "
         "after it is created, using the same mechanism as kuafu.rl.",
)
parser.add_argument(
    "--stats_path",
    type=str,
    default=None,
    help="Optional JSON sidecar path for BC rollout stats used by sweep automation.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.checkpoint = str(resolve_artifact_path(args_cli.checkpoint, "BC checkpoint"))
if args_cli.dataset:
    args_cli.dataset = str(resolve_artifact_path(args_cli.dataset, "BC replay trajectory"))
if args_cli.stats_path:
    args_cli.stats_path = str(resolve_artifact_path(args_cli.stats_path, "BC evaluation statistics"))
if args_cli.params_path:
    args_cli.params_path = str(resolve_parameters_path(args_cli.params_path))

if args_cli.dry_run:
    resolved_robot_id = resolve_execution_robot_id(args_cli.task, args_cli.execution_robot_id)
    print(
        "[dry-run] play_bc_env.py "
        f"task={args_cli.task} execution_robot_id={resolved_robot_id} "
        f"num_envs={args_cli.num_envs} device={args_cli.device} mode={args_cli.mode}"
    )
    raise SystemExit(0)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

resolved_execution_robot_id = register_task_modules(
    task=args_cli.task,
    execution_robot_id=args_cli.execution_robot_id,
)
print(f"[INFO] Registered Kuafu task package for execution_robot_id={resolved_execution_robot_id}", flush=True)

"""Rest everything follows."""

import json
import os
import random

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from isaaclab.utils.datasets import HDF5DatasetFileHandler
from isaaclab_tasks.utils import parse_env_cfg

from kuafu.data.replay_utils import EpisodeActionStream, get_state_setter
from kuafu.bc.training.dataset import Hdf5TrajectoryDataset, Normalizer
from kuafu.bc.training.model import MLPActorCriticPrior, MLPActorPrior


def _get_episode_actions(episode_data, action_mode: str) -> torch.Tensor | None:
    """Return the action tensor for the given mode from a loaded episode."""
    node = episode_data.data.get("actions")
    if node is None:
        return None
    if isinstance(node, dict):
        if action_mode in node:
            return node.get(action_mode)
        legacy_alias = {"absolute_joint": "joint", "absolute_ee": "ee"}
        alias = legacy_alias.get(action_mode)
        if alias is not None:
            return node.get(alias)
        return None
    # Old flat convention: actions tensor = relative_joint
    return node if action_mode == "relative_joint" else None


def load_checkpoint(ckpt_path: str, device: torch.device):
    """Load BC checkpoint and construct model + normalizers (mirror eval_bc_prior_on_dataset)."""
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(ckpt_path)

    # Direct checkpoint load (no extra helpers).
    ckpt = torch.load(ckpt_path, map_location="cpu")

    obs_norm = Normalizer.from_state_dict(ckpt["obs_normalizer"])
    act_norm = Normalizer.from_state_dict(ckpt["act_normalizer"])

    model_state_dict = ckpt["model_state_dicts"]["model_state_dict_norm"]
    uses_actor_critic_layout = any(
        key.startswith("actor_trunk.") or key.startswith("critic_trunk.") or key.startswith("value_layer.")
        for key in model_state_dict
    )
    model_cls = MLPActorCriticPrior if uses_actor_critic_layout else MLPActorPrior

    model = model_cls(
        obs_dim=ckpt["obs_dim"],
        act_dim=ckpt["act_dim"],
        hidden_dims=tuple(ckpt["hidden_dims"]),
        activation=str(ckpt.get("activation", "leaky_relu")),
        negative_slope=float(ckpt.get("negative_slope", 0.01)),
        dropout=float(ckpt.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(model_state_dict)
    model.eval()

    return model, obs_norm, act_norm, ckpt


# -----------------------------------------------------------------------------
# Shared helper
# -----------------------------------------------------------------------------

def _init_env_from_episode(raw_env, ep: dict, env_id: int,
                            past_begin_start: bool, full_state_setter) -> int:
    """Initialize one env from a dataset episode. Returns the start timestep t.

    If past_begin_start and full_state_setter are both available, picks a random
    timestep t within the episode and sets the env from obs/full_state[t].
    Otherwise falls back to initial_state at t=0.
    """
    ep_data = ep["episode_data"]
    if not past_begin_start:
        t = 0
    full_state_series = ep_data.data.get("obs", {}).get("full_state")
    if past_begin_start:
        T = full_state_series.shape[0]
        t = 60
    obs_t = full_state_series[t]
    if obs_t.dim() > 1:
        obs_t = obs_t[0]
    full_state_setter(raw_env, obs_t, env_id=env_id)
    return t


# -----------------------------------------------------------------------------
# Rollout
# -----------------------------------------------------------------------------

def run_for_duration(
    model,
    obs_norm,
    act_norm,
    action_chunk_size,
    chunk_exec,
    obs_key,
    env,
    duration_sec,
    step_dt,
    device,
    history_len=1,
    episodes=None,
    past_begin_start=False,
    full_state_setter=None,
    action_mode="relative_joint",
):
    """Run the BC policy for a fixed simulation duration.

    When chunk_exec > 1 the policy is queried once and then the predicted
    actions are executed one step at a time (receding-horizon style) before
    re-querying. The environment resets individual envs automatically on done.

    If episodes is provided, all envs are initialized from randomly chosen
    episodes using the same reset mechanics as replay mode (EpisodeActionStream.load_env).
    """
    raw_env_init = env if not hasattr(env, "unwrapped") else env.unwrapped
    num_envs = getattr(raw_env_init, "num_envs", 1)
    episode_per_env: list[dict | None] = [None] * num_envs
    start_t_per_env: list[int] = [0] * num_envs

    # Observation history buffer (needed early when priming from dataset for past_begin_start)
    single_obs_dim = obs_norm.mean.shape[0] // history_len
    obs_history = torch.zeros(num_envs, history_len, single_obs_dim, device=device, dtype=torch.float32)

    sim_time = 0.0
    total_episodes = 0
    raw_env = env.unwrapped

    # Action queue per env for chunk execution: list of (num_envs, act_dim) tensors
    action_queue: list[torch.Tensor] = []
    original_act_dim = act_norm.mean.shape[0] // action_chunk_size \
        if action_chunk_size > 1 else act_norm.mean.shape[0]

    # When dataset is provided, use the same reset mechanics as replay: env.reset() then stream.load_env per env.
    stream = None
    if episodes is not None:
        assert full_state_setter is not None, "full_state_setter is required when using episodes"
        stream = EpisodeActionStream(
            episodes=episodes,
            num_envs=num_envs,
            raw_env=raw_env_init,
            device=device,
            action_mode=action_mode,
            timeshift=0,
            past_begin_start=past_begin_start,
            full_state_setter=full_state_setter,
            init_env_from_episode_fn=_init_env_from_episode,
            get_episode_actions_fn=_get_episode_actions,
        )
        env.reset()
        for env_id in range(num_envs):
            stream.load_env(env_id)
        episode_per_env = list(stream.episode_per_env)
        start_t_per_env = [stream.env_action_idx[env_id] for env_id in range(num_envs)]
        raw_env_init.scene.write_data_to_sim()
        raw_env_init.sim.forward()
        if hasattr(raw_env_init, "update_data_in_controller"):
            raw_env_init.update_data_in_controller()
        obs_dict = raw_env_init._get_observations()
    else:
        obs_dict, _ = env.reset()

    def _update_obs_history(new_obs: torch.Tensor, reset_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Maintain per-env observation history buffer and return flattened obs."""
        nonlocal obs_history

        if reset_mask is not None:
            obs_history[reset_mask] = 0.0

        if history_len <= 1:
            # No history stacking: just mirror new_obs.
            return new_obs

        # Shift left and append newest frame at the end: [t-H+1, ..., t-1, t].
        obs_history = torch.roll(obs_history, shifts=-1, dims=1)
        obs_history[:, -1, :] = new_obs
        return obs_history.reshape(num_envs, history_len * single_obs_dim)

    while sim_time < duration_sec and simulation_app.is_running() and not simulation_app.is_exiting():
        # Get current policy observation: (num_envs, single_obs_dim)
        # obs_key=="full_state" → use full_state obs; otherwise use policy obs (="state")
        env_obs_key = "full_state" if obs_key == "full_state" else "policy"
        policy_obs = obs_dict[env_obs_key]
        if not isinstance(policy_obs, torch.Tensor):
            policy_obs = torch.tensor(policy_obs, dtype=torch.float32, device=device)
        else:
            policy_obs = policy_obs.to(device=device, dtype=torch.float32)

        stacked_obs = _update_obs_history(policy_obs)             # (num_envs, history_len*obs_dim)

        # Query the policy. Model outputs chunked actions when action_chunk_size > 1;
        # execute only the first action in each chunk online.
        with torch.no_grad():
            obs_n = obs_norm.normalize(stacked_obs)
            act_n = model(obs_n)
            act_full = act_norm.denormalize(act_n)

        if action_chunk_size > 1:
            act_full = act_full.view(num_envs, action_chunk_size, original_act_dim)
            action = act_full[:, 0, :]
        else:
            action = act_full

        obs_dict, _, terminated, truncated, _ = env.step(action)

        done = terminated | truncated
        n_done = int(done.sum().item())
        if n_done > 0:
            total_episodes += n_done
            # Clear history for done envs so stale frames don't pollute next episode.
            obs_history[done] = 0.0
            # When some envs reset, clear their queued actions (stale)
            # by flushing the queue — next step will re-query for all envs.
            action_queue.clear()
            # Re-init done envs from dataset (same as replay), not env's default reset.
            if stream is not None:
                for env_id in range(num_envs):
                    if done[env_id]:
                        stream.load_env(env_id)
                raw_env_init.scene.write_data_to_sim()
                raw_env_init.sim.forward()
                if hasattr(raw_env_init, "update_data_in_controller"):
                    raw_env_init.update_data_in_controller()
                obs_dict = raw_env_init._get_observations()
            successful = (
                raw_env.get_true_success_count()
                if hasattr(raw_env, "get_true_success_count")
                else 0
            )
            print(
                f"[PLAY] Episodes: {total_episodes} | "
                f"Successful: {successful} | "
                f"Success rate: {successful / total_episodes * 100:.1f}%"
            )

        sim_time += step_dt

    return total_episodes, sim_time


# -----------------------------------------------------------------------------
# Replay recorded actions
# -----------------------------------------------------------------------------

def run_replay_actions(episodes: list[dict], env, duration_sec: float, step_dt: float, device,
                       past_begin_start: bool = False, full_state_setter=None, timeshift: int = 0,
                       action_mode: str = "relative_joint"):
    """Replay recorded actions from dataset episodes across all envs.

    Each env is assigned a random episode and initialized from its initial_state
    (or a random timestep when past_begin_start=True). When an env's episode is
    exhausted or the env terminates, a new random episode is assigned.
    """
    raw_env = env.unwrapped
    num_envs = getattr(raw_env, "num_envs", 1)

    stream = EpisodeActionStream(
        episodes=episodes,
        num_envs=num_envs,
        raw_env=raw_env,
        device=device,
        action_mode=action_mode,
        timeshift=timeshift,
        past_begin_start=past_begin_start,
        full_state_setter=full_state_setter,
        init_env_from_episode_fn=_init_env_from_episode,
        get_episode_actions_fn=_get_episode_actions,
    )

    env.reset()
    for env_id in range(num_envs):
        stream.load_env(env_id)

    _sample_actions = _get_episode_actions(episodes[0]["episode_data"], action_mode)
    assert _sample_actions is not None, f"action_mode '{action_mode}' not found in dataset"
    act_dim = _sample_actions.shape[-1]
    idle_action = torch.zeros(num_envs, act_dim, device=device, dtype=torch.float32)

    sim_time = 0.0
    total_episodes = 0
    counter = 0
    while sim_time < duration_sec and simulation_app.is_running() and not simulation_app.is_exiting():
        env.sim.render()
        counter += 1
        if counter % 30 == 0:
            for env_id in range(num_envs):
                ep = stream.current_episode(env_id)
                if ep is None:
                    continue
                cached_obs_t = ep["episode_data"].data["obs"]["full_state"][stream.current_step(env_id)]
                if cached_obs_t.dim() > 1:
                    cached_obs_t = cached_obs_t[0]
                cached_obs_t = cached_obs_t.to(device=device, dtype=torch.float32)
                current_obs = env.unwrapped._get_observations()["full_state"][env_id]
                idx = stream.current_step(env_id)
                # full_state layout: pos+rot+joint_pos [0:45] | joint_vel [45:54] | joint_effort [54:63] | gripper+stage [63:65]
                l2_pre_vel = torch.linalg.norm(current_obs[:45] - cached_obs_t[:45]).item()
                l2_full = torch.linalg.norm(current_obs - cached_obs_t).item()
                print(
                    f"[REPLAY] env={env_id} step={idx}: "
                    f"L2_full={l2_full:.6f}  L2_pre_vel(pos+rot+jpos)={l2_pre_vel:.6f}"
                )
                # Time shift +1 and -1 (current_obs vs cached at t-1 and t+1)
                series = ep["episode_data"].data["obs"]["full_state"]
                T = series.shape[0]
                for shift, label in [(-1, "t-1"), (1, "t+1")]:
                    k = idx + shift
                    if k < 0 or k >= T:
                        continue
                    cached = series[k]
                    if cached.dim() > 1:
                        cached = cached[0]
                    cached = cached.to(device=device, dtype=torch.float32)
                    curr = current_obs.to(device=device, dtype=torch.float32)
                    l2_pre_vel_shift = torch.linalg.norm(curr[:45] - cached[:45]).item()
                    l2_full_shift = torch.linalg.norm(curr - cached).item()
                    print(
                        f"[REPLAY] env={env_id} step={idx} shift={label}: "
                        f"L2_full={l2_full_shift:.6f}  L2_pre_vel(pos+rot+jpos)={l2_pre_vel_shift:.6f}"
                    )

        actions = idle_action.clone()
        env_exhausted = [False] * num_envs
        for env_id in range(num_envs):
            a = stream.get_action(env_id)
            if a is None:
                env_exhausted[env_id] = True
            else:
                actions[env_id] = a
                stream.advance(env_id)

        obs_dict, _, terminated, truncated, _ = env.step(actions)
        done = terminated | truncated

        for env_id in range(num_envs):
            if done[env_id] or env_exhausted[env_id]:
                if done[env_id]:
                    total_episodes += 1
                stream.load_env(env_id)

        if done.any():
            successful = (
                raw_env.get_true_success_count()
                if hasattr(raw_env, "get_true_success_count")
                else 0
            )
            print(
                f"[REPLAY] Episodes: {total_episodes} | "
                f"Successful: {successful} | "
                f"Success rate: {successful / max(total_episodes, 1) * 100:.1f}%"
            )

        sim_time += step_dt


    return total_episodes, sim_time


# -----------------------------------------------------------------------------
# Examine: replay dataset actions while comparing against policy predictions
# -----------------------------------------------------------------------------

def _get_dataset_obs_at_step(stream: EpisodeActionStream, env_id: int, obs_key: str,
                             history_len: int, device: torch.device) -> torch.Tensor | None:
    """Return the dataset observation at the current step for this env, with history stacking.
    Returns shape (obs_dim,) or (history_len * single_obs_dim,) or None if not available.
    """
    ep = stream.current_episode(env_id)
    if ep is None:
        return None
    idx = stream.current_step(env_id)
    obs_node = ep["episode_data"].data.get("obs") or {}
    obs_series = obs_node.get(obs_key)
    if obs_series is None:
        return None
    if obs_series.dim() == 3:
        obs_series = obs_series[:, 0, :]
    T, single_obs_dim = obs_series.shape[0], obs_series.shape[-1]
    obs_series = obs_series.to(device=device, dtype=torch.float32)

    if history_len <= 1:
        if idx >= T:
            return None
        return obs_series[idx].clone()
    # Stack last history_len frames; pad from the left by repeating the first frame (match training).
    start = max(0, idx - history_len + 1)
    end = idx + 1
    frames = obs_series[start:end]
    n_frames = frames.shape[0]
    if n_frames == 0:
        return None
    if n_frames < history_len:
        first = frames[0:1].repeat(history_len - n_frames, 1)
        frames = torch.cat([first, frames], dim=0)
    return frames.reshape(-1).clone()


def run_examine_actions(
    episodes: list[dict],
    env,
    model,
    obs_norm,
    act_norm,
    obs_key,
    duration_sec: float,
    step_dt: float,
    device,
    past_begin_start: bool = False,
    full_state_setter=None,
    timeshift: int = 0,
    action_mode: str = "relative_joint",
    history_len: int = 1,
    action_chunk_size: int = 1,
):
    """Replay dataset actions in the sim; policy is fed dataset obs at each step (same as offline eval)."""
    raw_env = env.unwrapped
    num_envs = getattr(raw_env, "num_envs", 1)

    stream = EpisodeActionStream(
        episodes=episodes,
        num_envs=num_envs,
        raw_env=raw_env,
        device=device,
        action_mode=action_mode,
        timeshift=timeshift,
        past_begin_start=past_begin_start,
        full_state_setter=full_state_setter,
        init_env_from_episode_fn=_init_env_from_episode,
        get_episode_actions_fn=_get_episode_actions,
    )

    env.reset()
    for env_id in range(num_envs):
        stream.load_env(env_id)
    raw_env._get_observations()

    _sample_actions = _get_episode_actions(episodes[0]["episode_data"], action_mode)
    assert _sample_actions is not None, f"action_mode '{action_mode}' not found in dataset"
    act_dim = _sample_actions.shape[-1]
    idle_action = torch.zeros(num_envs, act_dim, device=device, dtype=torch.float32)

    history_len = int(history_len)
    single_obs_dim = int(obs_norm.mean.shape[0] // max(history_len, 1))
    obs_dim_total = history_len * single_obs_dim

    sim_time = 0.0
    total_episodes = 0

    while sim_time < duration_sec and simulation_app.is_running() and not simulation_app.is_exiting():
        # Policy input = dataset observation at current step (same as offline eval), not sim obs.
        stacked_obs_list: list[torch.Tensor] = []
        for env_id in range(num_envs):
            o = _get_dataset_obs_at_step(stream, env_id, obs_key, history_len, device)
            if o is None:
                stacked_obs_list.append(torch.zeros(obs_dim_total, device=device, dtype=torch.float32))
            else:
                stacked_obs_list.append(o)
        stacked_obs = torch.stack(stacked_obs_list, dim=0)

        # Calculate policy action for each env given dataset observations.
        with torch.no_grad():
            obs_n = obs_norm.normalize(stacked_obs)
            act_n = model(obs_n)
            act_full = act_norm.denormalize(act_n)

        if action_chunk_size > 1:
            original_act_dim = act_norm.mean.shape[0] // max(int(action_chunk_size), 1)
            act_full = act_full.view(num_envs, int(action_chunk_size), original_act_dim)
            policy_action_full = act_full[:, 0, :]
        else:
            policy_action_full = act_full

        # Build dataset actions and compare.
        actions = idle_action.clone()
        env_exhausted = [False] * num_envs
        for env_id in range(num_envs):
            dataset_action = stream.get_action(env_id)
            idx = stream.current_step(env_id)
            if dataset_action is None:
                env_exhausted[env_id] = True
                continue
            actions[env_id] = dataset_action
            pa = policy_action_full[env_id]

            l2 = torch.linalg.norm(pa - dataset_action).item()
            print(f"[EXAMINE] env={env_id} step={idx}: action L2(policy vs dataset)={l2:.6f}")
            stream.advance(env_id)

        obs_dict, _, terminated, truncated, _ = env.step(actions)
        done = terminated | truncated

        for env_id in range(num_envs):
            if done[env_id] or env_exhausted[env_id]:
                if done[env_id]:
                    total_episodes += 1
                stream.load_env(env_id)
                obs_dict = raw_env._get_observations()

        if done.any():
            successful = (
                raw_env.get_true_success_count()
                if hasattr(raw_env, "get_true_success_count")
                else 0
            )
            print(
                f"[EXAMINE] Episodes: {total_episodes} | "
                f"Successful: {successful} | "
                f"Success rate: {successful / max(total_episodes, 1) * 100:.1f}%"
            )

        sim_time += step_dt

    return total_episodes, sim_time


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def _early_detect_action_mode() -> str:
    """Determine action_mode before env creation.

    Policy/eval must default ONLY to what the checkpoint was trained on.
    Replay can default from the dataset structure.
    """
    if args_cli.mode in ("policy", "examine"):
        if not args_cli.checkpoint:
            raise ValueError(f"--mode {args_cli.mode} requires --checkpoint to infer action_mode")
        ckpt = torch.load(args_cli.checkpoint, map_location="cpu")

        # Preferred: explicit action_mode stored in checkpoint (newer runs).
        if "action_mode" in ckpt and ckpt["action_mode"] is not None:
            return ckpt["action_mode"]

        # Fallback: read from meta/config.yaml (resolved nested format).
        try:
            import yaml
            ckpt_path = Path(args_cli.checkpoint).resolve()
            config_path = ckpt_path.parent.parent / "meta" / "config.yaml"
            if config_path.exists():
                with open(config_path) as f:
                    config = yaml.safe_load(f)
                action_mode = config.get("dataset", {}).get("action_mode")
                if action_mode:
                    return action_mode
        except Exception:
            pass

        raise ValueError(
            "Could not infer action_mode from checkpoint. "
            "Pass --action_mode explicitly or ensure it exists in the checkpoint (recommended) "
            "or meta/config.yaml."
        )

    # replay mode
    if args_cli.dataset:
        import h5py as _h5py
        with _h5py.File(args_cli.dataset, "r") as f:
            demos = sorted(f["data"].keys())  # type: ignore[union-attr]
            if demos:
                node = f["data"][demos[0]].get("actions")  # type: ignore[union-attr]
                if isinstance(node, _h5py.Group):
                    for preferred in ("relative_joint", "absolute_joint", "absolute_ee", "relative_ee", "joint", "ee"):
                        if preferred in node:
                            if preferred == "joint":
                                return "absolute_joint"
                            if preferred == "ee":
                                return "absolute_ee"
                            return preferred
    return "relative_joint"


def _resolve_task_name(task_name: str) -> str:
    """Map local shorthand task names to registered Gym task ids."""
    aliases = {
        "right_can": "FR3-FinRay-Table-Can",
    }
    return aliases.get(task_name, task_name)


def _collect_rollout_stats(env, *, total_episodes: int, total_sim_time: float, success_rate_pct: float) -> dict:
    raw_env = env.unwrapped

    def _call(name: str, default):
        if not hasattr(raw_env, name):
            return default
        return getattr(raw_env, name)()

    true_success_count = int(_call("get_true_success_count", 0))
    false_complete_count = int(_call("get_false_complete_count", 0))
    timeout_count = int(_call("get_timeout_count", 0))
    stage_transition_stats = _call("get_stage_transition_stats", {})
    highest_stage_reached_per_env = _call("get_highest_stage_reached_per_env", [])
    constraint_stats = _call("get_constraint_stats", {})

    stage_count = 0
    program_runtime = getattr(raw_env, "program_runtime", None)
    compiled_program = getattr(program_runtime, "compiled_program", None)
    if compiled_program is not None and hasattr(compiled_program, "stage_count"):
        stage_count = int(compiled_program.stage_count())
    elif isinstance(stage_transition_stats, dict) and stage_transition_stats:
        stage_count = max(int(k) for k in stage_transition_stats.keys()) + 1

    n_envs = len(highest_stage_reached_per_env)
    stage_pass_rates = {}
    if n_envs > 0 and stage_count > 0:
        for stage_id in range(stage_count):
            passed = sum(1 for reached in highest_stage_reached_per_env if int(reached) > stage_id)
            stage_pass_rates[str(stage_id)] = {
                "passed": int(passed),
                "total": int(n_envs),
                "rate": float(passed) / float(n_envs),
            }

    total_outcomes = true_success_count + false_complete_count + timeout_count
    if total_outcomes <= 0:
        total_outcomes = int(total_episodes)

    return {
        "total_episodes": int(total_episodes),
        "total_sim_time": float(total_sim_time),
        "success_rate_pct": float(success_rate_pct),
        "true_success_count": true_success_count,
        "false_complete_count": false_complete_count,
        "timeout_count": timeout_count,
        "total_outcomes": int(total_outcomes),
        "stage_count": int(stage_count),
        "stage_pass_rates": stage_pass_rates,
        "stage_transition_stats": stage_transition_stats,
        "constraint_met_rate_stats": constraint_stats,
        "highest_stage_reached_per_env": highest_stage_reached_per_env,
        "checkpoint": str(args_cli.checkpoint),
        "params_path": str(args_cli.params_path) if args_cli.params_path else "",
        **args_cli.program_identity,
    }


def main():
    if args_cli.mode in ("replay", "examine") and not args_cli.dataset:
        raise ValueError(f"--mode {args_cli.mode} requires --dataset")

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<not set>")
    print(f"CUDA_VISIBLE_DEVICES={cuda_visible}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(
            f"Using GPU: cuda:{torch.cuda.current_device()} ({torch.cuda.get_device_name(0)})",
            flush=True,
        )
    else:
        print("Using device: cpu", flush=True)

    if args_cli.action_mode is not None:
        action_mode = args_cli.action_mode
        print(f"Using action_mode (override): {action_mode}")
    else:
        action_mode = _early_detect_action_mode()
        print(f"Detected action_mode: {action_mode}")

    resolved_task = _resolve_task_name(args_cli.task)
    if resolved_task != args_cli.task:
        print(f"Resolved task alias '{args_cli.task}' -> '{resolved_task}'")

    env_cfg = parse_env_cfg(
        resolved_task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.action_mode = action_mode
    env_cfg.enable_cameras = False
    env_cfg.debug_rewards = False
    env_cfg.sun_program_evaluator_stage_override = False

    env_cfg.seed = args_cli.seed
    args_cli.program_identity = apply_learned_program(env_cfg, args_cli.program_path)

    env = gym.make(resolved_task, cfg=env_cfg).unwrapped

    if args_cli.params_path is not None:
        params_path = os.path.abspath(args_cli.params_path)
        if not os.path.isfile(params_path):
            raise FileNotFoundError(f"[params_path] File not found: {params_path}")
        print(f"[INFO]: Loading params/weights from: {params_path}")
        with open(params_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            raise ValueError(f"[params_path] Expected a JSON object, got {type(raw).__name__}: {params_path}")
        validate_program_metadata(raw, args_cli.program_identity)
        if "stages" not in raw:
            raise ValueError(f"[params_path] Missing 'stages' key in JSON: {params_path}")
        stages_data = raw["stages"]
        if not isinstance(stages_data, dict) or not stages_data:
            raise ValueError(f"[params_path] 'stages' must be a non-empty dict: {params_path}")
        all_stages_weights = {}
        for stage_id_str, items in stages_data.items():
            stage_id = int(stage_id_str)
            if "weights" not in items or not items["weights"]:
                raise ValueError(f"[params_path] Stage {stage_id_str} missing non-empty 'weights': {params_path}")
            all_stages_weights[stage_id] = items["weights"]
            update_parameters(env, stage_id=stage_id, parameters=items.get("parameters", {}))
        update_weights(env, stage_id=None, weights=None, all_stages_weights=all_stages_weights)

    if args_cli.seed is not None:
        torch.manual_seed(args_cli.seed)
        if hasattr(env, "seed"):
            getattr(env, "seed")(args_cli.seed)
        random.seed(args_cli.seed)

    step_dt = getattr(env, "step_dt", None) or getattr(env, "dt", None)
    if step_dt is None:
        step_dt = env_cfg.sim.dt * getattr(env_cfg, "decimation", 1)
    print(f"Running for {args_cli.duration_sec}s simulation time (step_dt={step_dt:.4f}s)")

    if hasattr(env, "clear_stage_transition_stats"):
        env.clear_stage_transition_stats()  # type: ignore[attr-defined]
    elif hasattr(env, "clear_end_to_end_rate_stats"):
        env.clear_end_to_end_rate_stats()  # type: ignore[attr-defined]
    if hasattr(env, "clear_terminal_stuck_constraint_stats"):
        env.clear_terminal_stuck_constraint_stats()  # type: ignore[attr-defined]

    # Load dataset episodes (always needed when --dataset is provided).
    episodes = None
    if args_cli.dataset:
        print(f"Loading dataset from: {args_cli.dataset}")
        handler = HDF5DatasetFileHandler()
        handler.open(args_cli.dataset)
        episode_names = list(handler.get_episode_names())
        if args_cli.demo_index is not None:
            if not episode_names:
                raise ValueError("Dataset contains no episodes but --demo_index was provided.")
            idx = int(args_cli.demo_index)
            if idx < 0:
                idx += len(episode_names)
            if idx < 0 or idx >= len(episode_names):
                raise IndexError(
                    f"--demo_index {args_cli.demo_index} out of range for {len(episode_names)} demos"
                )
            chosen_name = episode_names[idx]
            print(f"Using only demo index {idx} -> '{chosen_name}'")
            episode_names = [chosen_name]
        episodes = []
        for name in episode_names:
            ep_data = handler.load_episode(name, device)
            initial_state = ep_data.data.get("initial_state")
            if initial_state is not None:
                episodes.append({"initial_state": initial_state, "episode_data": ep_data})
        handler.close()
        print(f"Loaded {len(episodes)} episodes.")

    full_state_setter = get_state_setter(resolved_task)

    if args_cli.mode == "replay":
        assert episodes is not None
        timeshift = int(args_cli.timeshift) if args_cli.timeshift is not None else 0
        total_episodes, total_sim_time = run_replay_actions(
            episodes, env, args_cli.duration_sec, step_dt, device,
            past_begin_start=args_cli.past_begin_start,
            full_state_setter=full_state_setter,
            timeshift=timeshift,
            action_mode=action_mode,
        )
    elif args_cli.mode == "examine":
        assert episodes is not None, "--mode examine requires --dataset"
        print(f"Loading checkpoint: {args_cli.checkpoint}")
        model, obs_norm, act_norm, ckpt = load_checkpoint(args_cli.checkpoint, device)
        obs_key = ckpt["obs_key"]
        timeshift = int(args_cli.timeshift) if args_cli.timeshift is not None else int(ckpt.get("time_shift", 0))
        action_chunk_size = int(ckpt.get("action_chunk_size", 1))
        history_len = int(ckpt.get("obs_history", ckpt.get("history_len", 1)))
        print(
            f"Examine: sim replays dataset; policy uses dataset obs (obs_key={obs_key}, "
            f"time_shift={timeshift}, history_len={history_len}, action_chunk_size={action_chunk_size})"
        )
        total_episodes, total_sim_time = run_examine_actions(
            episodes, env, model, obs_norm, act_norm, obs_key,
            args_cli.duration_sec, step_dt, device,
            past_begin_start=args_cli.past_begin_start,
            full_state_setter=full_state_setter,
            timeshift=timeshift,
            action_mode=action_mode,
            history_len=history_len,
            action_chunk_size=action_chunk_size,
        )
    else:
        print(f"Loading checkpoint: {args_cli.checkpoint}")
        model, obs_norm, act_norm, ckpt = load_checkpoint(args_cli.checkpoint, device)
        obs_key = ckpt["obs_key"]
        action_chunk_size = ckpt.get("action_chunk_size", 1)
        history_len = ckpt.get("obs_history", ckpt.get("history_len", 1))
        chunk_exec = args_cli.chunk_exec if args_cli.chunk_exec > 0 else action_chunk_size

        print(
            f"Policy: obs_key=obs/{obs_key}, obs_dim={ckpt['obs_dim']}, "
            f"act_dim={ckpt['act_dim']}, chunk_size={action_chunk_size}, "
            f"history_len={history_len}, chunk_exec={chunk_exec}, epoch={ckpt['epoch']}"
        )

        total_episodes, total_sim_time = run_for_duration(
            model, obs_norm, act_norm, action_chunk_size, chunk_exec,
            obs_key, env, args_cli.duration_sec, step_dt, device,
            history_len=history_len,
            episodes=episodes,
            past_begin_start=args_cli.past_begin_start,
            full_state_setter=full_state_setter,
            action_mode=action_mode,
        )

    successful = env.get_true_success_count() if hasattr(env, "get_true_success_count") else 0  # type: ignore[attr-defined]
    success_rate_pct = successful / max(total_episodes, 1) * 100.0
    print("\n" + "=" * 60)
    print("  PLAY SUMMARY")
    print(f"  Simulation time (s)      : {total_sim_time:.2f}")
    print(f"  Total episodes completed : {total_episodes}")
    print(f"  Successful terminations  : {successful}")
    print(f"  Success rate             : {success_rate_pct:.1f}%")
    print("=" * 60 + "\n")
    # Machine-parseable lines for test_sweep / automation (no spaces in keys)
    print(f"SUCCESS_RATE:{success_rate_pct:.1f}", flush=True)
    rollout_stats = _collect_rollout_stats(
        env,
        total_episodes=total_episodes,
        total_sim_time=total_sim_time,
        success_rate_pct=success_rate_pct,
    )
    stats_json = json.dumps(rollout_stats, sort_keys=True)
    print(f"BC_EVAL_STATS_JSON:{stats_json}", flush=True)
    if args_cli.stats_path:
        stats_path = resolve_artifact_path(args_cli.stats_path, "BC evaluation statistics")
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(rollout_stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"BC_EVAL_STATS_PATH:{stats_path}", flush=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
