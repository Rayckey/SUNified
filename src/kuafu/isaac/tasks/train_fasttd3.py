"""Train a FastTD3 agent on the unified Isaac Lab environments.

This script mirrors the structure of src/kuafu/isaac/tasks/train.py for the environment
setup (AppLauncher, Hydra task config, gym.make, residual-weight utilities) and
replaces the RL-Games PPO runner with FastTD3's off-policy TD3 training loop.

Usage
-----
# Basic run
python src/kuafu/isaac/tasks/train_fasttd3.py --task Isaac-My-Task-v0 --num_envs 4096

# Warm-start actor from a BC prior
python src/kuafu/isaac/tasks/train_fasttd3.py --task Isaac-My-Task-v0 \\
       --prior_checkpoint logs/bc_prior_mlp/2025-01-01_00-00-00

# Resume from a FastTD3 checkpoint
python src/kuafu/isaac/tasks/train_fasttd3.py --task Isaac-My-Task-v0 \\
       --checkpoint models/my_run_5000.pt

FastTD3 actor architecture (actor_hidden_dim = H, default 512):
    net:   Linear(obs, H) -> ReLU -> Linear(H, H//2) -> ReLU -> Linear(H//2, H//4) -> ReLU
    fc_mu: Linear(H//4, act_dim) -> Tanh

For BC prior transfer the prior must have been trained with
    hidden_dims = (H, H//2, H//4)
"""
from __future__ import annotations

# ── env vars that FastTD3 sets before any heavy imports ─────────────────────
import os
os.environ.setdefault("TORCHDYNAMO_INLINE_INBUILT_NN_MODULES", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import math
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from distutils.util import strtobool
from pathlib import Path
from typing import Any

import yaml
from kuafu.paths import PROJECT_ROOT, resolve_artifact_path

# ── path setup ──────────────────────────────────────────────────────────────
_REPO_ROOT = PROJECT_ROOT

# ── residual-RL utilities (must be imported before AppLauncher) ──────────────────
from kuafu.isaac.tasks.task_registry import register_task_modules, resolve_execution_robot_id  # noqa: E402
from kuafu.rl.isaac.update_utilities import update_weights, update_parameters  # noqa: E402

# ── IsaacLab AppLauncher (must happen before torch / IsaacLab imports) ──────
from isaaclab.app import AppLauncher  # noqa: E402

# Unified task packages are registered after Isaac launches so --task can select
# FR3FinRay-Franka, Panda-Franka, or Kinova without changing this script.

# ── config bootstrap ─────────────────────────────────────────────────────────
def _resolve_config_path(path_str: str | None) -> Path | None:
    if not path_str:
        return None
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = _REPO_ROOT / path
    return path.resolve()


def _load_fasttd3_config(path_str: str | None) -> dict[str, Any]:
    cfg_path = _resolve_config_path(path_str)
    if cfg_path is None:
        return {}
    if not cfg_path.is_file():
        raise FileNotFoundError(f"FastTD3 config file not found: {cfg_path}")
    with cfg_path.open(encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"FastTD3 config must be a mapping: {cfg_path}")
    return loaded


def _resolve_runtime_path(path_str: str | None) -> str | None:
    if not path_str:
        return None
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        parts = path.parts
        if len(parts) >= 4 and parts[:4] == ("src", "kuafu", "data", "datasets"):
            path = Path("Trajectories") / path.name
    return str(resolve_artifact_path(path, "FastTD3 demonstration trajectory"))


_bootstrap_parser = argparse.ArgumentParser(add_help=False)
_bootstrap_parser.add_argument("--agent_cfg", type=str, default=None)
_bootstrap_args, _ = _bootstrap_parser.parse_known_args()
_raw_fasttd3_cfg = _load_fasttd3_config(_bootstrap_args.agent_cfg)

dataset_cfg = _raw_fasttd3_cfg.get("dataset", {})
model_cfg = _raw_fasttd3_cfg.get("model", {})
logging_cfg = _raw_fasttd3_cfg.get("logging", {})
replay_cfg = _raw_fasttd3_cfg.get("replay", {})

actor_hidden_defaults = model_cfg.get("hidden_dims", [])
actor_hidden_default = (
    int(actor_hidden_defaults[0]) if isinstance(actor_hidden_defaults, list) and actor_hidden_defaults else 512
)
demo_dataset_default = dataset_cfg.get("dataset")
demo_preload_default = bool(replay_cfg.get("preload", bool(demo_dataset_default)))


# ── CLI arguments ────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Train FastTD3 on an IsaacLab environment.")
parser.add_argument("--agent_cfg", type=str, default=_bootstrap_args.agent_cfg,
                    help="Optional FastTD3 YAML config for replay/dataset defaults.")

# --- environment ---
parser.add_argument("--task",     type=str,   default="stack-2",  help="IsaacLab task name.")
parser.add_argument("--execution_robot_id", type=str, default=None,
                    help="Robot task package to register. Defaults to inference from --task, then FR3FinRay-Franka.")
parser.add_argument("--dry_run", action="store_true", default=False,
                    help="Resolve task registration and exit before launching Isaac.")
parser.add_argument("--num_envs", type=int,   default=4096,       help="Number of parallel envs.")
parser.add_argument("--seed",     type=int,   default=-1,         help="Random seed (-1 = random).")

# --- checkpointing ---
parser.add_argument("--checkpoint",        type=str, default=None,
                    help="Path to a FastTD3 .pt checkpoint to resume training.")
parser.add_argument("--prior_checkpoint",  type=str, default=None,
                    help="Path to a BC prior .pth checkpoint or run directory.")
parser.add_argument("--prior_obs_stats_count", type=int, default=1_000_000,
                    help="Synthetic count used when seeding EmpiricalNormalization from BC stats.")
parser.add_argument("--prior_verbose", action="store_true", default=False,
                    help="Print per-tensor info during BC prior transfer.")
parser.add_argument("--demo_dataset", type=str, default=demo_dataset_default,
                    help="HDF5 dataset used to seed replay before online RL.")
parser.add_argument("--demo_obs_key", type=str, default=str(dataset_cfg.get("obs_key", "state")),
                    help="Observation key to load from the demo dataset.")
parser.add_argument("--demo_action_mode", type=str, default=str(dataset_cfg.get("action_mode", "relative_joint")),
                    help="Action key to load from the demo dataset.")
parser.add_argument("--demo_preload", type=lambda x: bool(strtobool(x)), default=demo_preload_default,
                    nargs="?", const=True,
                    help="Preload replay from the demo dataset before online RL rollouts.")
parser.add_argument("--demo_max_transitions", type=int, default=replay_cfg.get("max_transitions"),
                    help="Optional cap on how many demo transitions to preload.")
parser.add_argument("--demo_use_rewards", type=lambda x: bool(strtobool(x)),
                    default=bool(replay_cfg.get("use_rewards", True)), nargs="?", const=True,
                    help="If true, read rewards from the demo HDF5; if false, zero them out.")
parser.add_argument("--demo_verbose", type=lambda x: bool(strtobool(x)),
                    default=bool(replay_cfg.get("verbose", True)), nargs="?", const=True,
                    help="Print replay preload details.")
parser.add_argument("--demo_preload_fraction", type=float,
                    default=replay_cfg.get("preload_fraction"),
                    help="Fill this fraction of the replay buffer with demos (repeat "
                         "round-robin as needed). Mutually exclusive with "
                         "--demo_preload_slots.")
parser.add_argument("--demo_preload_slots", type=int,
                    default=replay_cfg.get("preload_slots"),
                    help="Explicit number of per-env replay slots to fill with demos "
                         "(repeat round-robin as needed). Mutually exclusive with "
                         "--demo_preload_fraction.")

# --- FastTD3 hyper-parameters ---
parser.add_argument("--total_timesteps",      type=int,   default=150_000)
parser.add_argument("--buffer_size",          type=int,   default=1024 * 50)
parser.add_argument("--batch_size",           type=int,   default=32_768)
parser.add_argument("--num_updates",          type=int,   default=2)
parser.add_argument("--num_steps",            type=int,   default=1,    help="N-step return horizon.")
parser.add_argument("--gamma",                type=float, default=0.99)
parser.add_argument("--tau",                  type=float, default=0.1,  help="Target network soft-update rate.")
parser.add_argument("--actor_lr",             type=float, default=3e-4)
parser.add_argument("--critic_lr",            type=float, default=3e-4)
parser.add_argument("--actor_lr_end",         type=float, default=3e-4)
parser.add_argument("--critic_lr_end",        type=float, default=3e-4)
parser.add_argument("--actor_hidden_dim",     type=int,   default=actor_hidden_default)
parser.add_argument("--critic_hidden_dim",    type=int,   default=1024)
parser.add_argument("--num_atoms",            type=int,   default=101)
parser.add_argument("--v_min",                type=float, default=-250.0)
parser.add_argument("--v_max",                type=float, default=250.0)
parser.add_argument("--std_min",              type=float, default=0.05)
parser.add_argument("--std_max",              type=float, default=0.8)
parser.add_argument("--policy_noise",         type=float, default=0.1)
parser.add_argument("--noise_clip",           type=float, default=0.5)
parser.add_argument("--policy_frequency",     type=int,   default=2)
parser.add_argument("--learning_starts",      type=int,   default=10)
parser.add_argument("--init_scale",           type=float, default=0.01)
parser.add_argument("--action_bounds",        type=float, default=1.0,  help="Actions clamped to ±action_bounds.")
parser.add_argument("--weight_decay",         type=float, default=0.1)
parser.add_argument("--use_cdq",              type=lambda x: bool(strtobool(x)), default=True,
                    help="Use Clipped Double Q-learning.")
parser.add_argument("--obs_normalization",    type=lambda x: bool(strtobool(x)), default=True)
parser.add_argument("--reward_normalization", type=lambda x: bool(strtobool(x)), default=False)
parser.add_argument("--use_amp",              type=lambda x: bool(strtobool(x)), default=True)
parser.add_argument("--amp_dtype",            type=str,   default="bf16", choices=["bf16", "fp16"])
parser.add_argument("--compile",              type=lambda x: bool(strtobool(x)), default=False,
                    help="torch.compile the update functions (faster but may conflict with IsaacLab).")
parser.add_argument("--disable_bootstrap",    action="store_true", default=False)
parser.add_argument("--use_grad_norm_clipping", action="store_true", default=False)
parser.add_argument("--max_grad_norm",        type=float, default=0.0)

# --- logging ---
parser.add_argument("--eval_interval",   type=int,  default=5_000)
parser.add_argument("--save_interval",   type=int,  default=5_000)
parser.add_argument(
    "--log_dir",
    type=str,
    default=str(resolve_artifact_path(logging_cfg.get("log_dir", "standalone/fasttd3"), "FastTD3 log root")),
)
parser.add_argument("--exp_name",        type=str,  default=None,
                    help="Experiment name (default: task__seed__timestamp).")
parser.add_argument("--wandb_project",   type=str,  default="KUAFU-FastTD3")
parser.add_argument("--wandb_entity",    type=str,  default=None)
parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False,
                    nargs="?", const=True, help="Enable Weights & Biases tracking.")

# --- task-specific ---
parser.add_argument("--residual_weights", type=str, default=None,
                    help="Path to JSON file with SUNProgramEvaluator residual weights.")

# --- Hydra agent entry point (used only to resolve the env Hydra config) ---
parser.add_argument("--agent", type=str, default="rl_games_cfg_entry_point",
                    help="Hydra agent entry-point name (used only to load the env config).")

AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)

args_cli, hydra_args = parser.parse_known_args()
for _artifact_attr, _artifact_label in (
    ("checkpoint", "FastTD3 checkpoint"),
    ("prior_checkpoint", "BC prior checkpoint"),
):
    _artifact_value = getattr(args_cli, _artifact_attr, None)
    if _artifact_value:
        setattr(args_cli, _artifact_attr, str(resolve_artifact_path(_artifact_value, _artifact_label)))

if args_cli.dry_run:
    resolved_robot_id = resolve_execution_robot_id(args_cli.task, args_cli.execution_robot_id)
    print(
        "[dry-run] train_fasttd3.py "
        f"task={args_cli.task} execution_robot_id={resolved_robot_id} "
        f"num_envs={args_cli.num_envs} device={args_cli.device} agent={args_cli.agent}"
    )
    raise SystemExit(0)

sys.argv = [sys.argv[0]] + hydra_args   # pass remaining args to Hydra

app_launcher    = AppLauncher(args_cli)
simulation_app  = app_launcher.app

resolved_execution_robot_id = register_task_modules(
    task=args_cli.task,
    execution_robot_id=args_cli.execution_robot_id,
)
print(f"[INFO] Registered task package for execution_robot_id={resolved_execution_robot_id}", flush=True)

# ── post-launch imports ──────────────────────────────────────────────────────
import json
import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler

from tensordict import TensorDict

import gymnasium as gym

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.io import dump_yaml

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

# FastTD3 modules (provided by the active environment)
from fast_td3 import Actor, Critic  # type: ignore[import]
from fast_td3_utils import (  # type: ignore[import]
    EmpiricalNormalization,
    RewardNormalizer,
    SimpleReplayBuffer,
    save_params,
    mark_step,
)


# ── IsaacLab → FastTD3 environment adapter ───────────────────────────────────

class IsaacLabEnvAdapter:
    """Thin wrapper that presents an already-created IsaacLab gym env with the
    same interface that FastTD3's training loop expects.

    The adapter exposes:
        num_envs, num_obs, num_actions, num_privileged_obs, asymmetric_obs,
        max_episode_steps
        reset(random_start_init) -> obs
        reset_with_critic_obs()  -> (obs, critic_obs)
        step(actions)            -> (obs, rew, dones, info)
    """

    def __init__(self, env, *, action_bounds: float = 1.0):
        self._env = env
        self.action_bounds = action_bounds

        uw = env.unwrapped
        self.num_envs         = uw.num_envs
        self.max_episode_steps = uw.max_episode_length

        obs_space = uw.single_observation_space
        self.num_obs       = obs_space["policy"].shape[0]
        self.asymmetric_obs = "critic" in obs_space
        self.num_privileged_obs = (
            obs_space["critic"].shape[0] if self.asymmetric_obs else 0
        )
        self.num_actions = uw.single_action_space.shape[0]

    def reset(self, random_start_init: bool = True) -> torch.Tensor:
        obs_dict, _ = self._env.reset()
        if random_start_init:
            uw = self._env.unwrapped
            uw.episode_length_buf = torch.randint_like(
                uw.episode_length_buf, high=int(self.max_episode_steps)
            )
        return obs_dict["policy"]

    def reset_with_critic_obs(self) -> tuple[torch.Tensor, torch.Tensor]:
        obs_dict, _ = self._env.reset()
        return obs_dict["policy"], obs_dict["critic"]

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        clipped = torch.clamp(actions, -1.0, 1.0)
        if self.action_bounds != 1.0:
            clipped = clipped * self.action_bounds
        obs_dict, rew, terminations, truncations, _infos = self._env.step(clipped)
        dones = (terminations | truncations).to(dtype=torch.long)
        obs         = obs_dict["policy"]
        critic_obs  = obs_dict["critic"] if self.asymmetric_obs else None
        info = {
            "time_outs": truncations,
            "observations": {
                "critic": critic_obs,
                "raw": {"obs": obs, "critic_obs": critic_obs},
            },
        }
        return obs, rew, dones, info

    def close(self) -> None:
        self._env.close()


# ── Training entry-point (wrapped by Hydra task config) ─────────────────────

@hydra_task_config(args_cli.task, args_cli.agent)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: dict,
) -> None:
    """FastTD3 training loop for unified Isaac Lab environments."""

    if args_cli.checkpoint is not None and args_cli.prior_checkpoint is not None:
        raise ValueError("--checkpoint and --prior_checkpoint are mutually exclusive.")

    # ── seed ─────────────────────────────────────────────────────────────────
    seed = args_cli.seed if args_cli.seed >= 0 else random.randint(0, 10_000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ── device ───────────────────────────────────────────────────────────────
    device_str = args_cli.device if args_cli.device is not None else "cuda"
    device = torch.device(device_str)

    # ── experiment name / log dir ─────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_name  = args_cli.exp_name or f"{args_cli.task}__{seed}__{timestamp}"
    log_dir = resolve_artifact_path(args_cli.log_dir, "FastTD3 log root") / exp_name
    log_dir.mkdir(parents=True, exist_ok=True)

    # ── env config overrides ──────────────────────────────────────────────────
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device     = device_str
    env_cfg.seed           = seed
    env_cfg.log_dir        = str(log_dir)

    # task-specific mpc_config block (not present on all envs)
    mpc_cfg = getattr(env_cfg, "mpc_config", None)
    if isinstance(mpc_cfg, dict):
        mpc_cfg["num_envs"]            = args_cli.num_envs
        mpc_cfg["use_per_env_weight"]  = False

    # Apply mpc: reward settings from the agent YAML (mirrors train.py logic).
    # train_fasttd3.py receives agent_cfg but previously ignored the mpc: block,
    # leaving reward type at the env dataclass defaults (relative rewards).
    if "mpc" in agent_cfg:
        _mpc = agent_cfg["mpc"]
        if hasattr(env_cfg, "use_absolute_progress_rewards"):
            env_cfg.use_absolute_progress_rewards = bool(_mpc.get("use_absolute_progress_rewards", env_cfg.use_absolute_progress_rewards))
        if hasattr(env_cfg, "use_relative_progress_rewards"):
            env_cfg.use_relative_progress_rewards = bool(_mpc.get("use_relative_progress_rewards", env_cfg.use_relative_progress_rewards))
        if hasattr(env_cfg, "progress_rewards_scale") and "progress_rewards_scale" in _mpc:
            env_cfg.progress_rewards_scale = float(_mpc["progress_rewards_scale"])
        if hasattr(env_cfg, "use_velocity_penalty") and "use_velocity_penalty" in _mpc:
            env_cfg.use_velocity_penalty = bool(_mpc["use_velocity_penalty"])
        if hasattr(env_cfg, "regularize_velocity_scale") and "regularize_velocity_scale" in _mpc:
            env_cfg.regularize_velocity_scale = float(_mpc["regularize_velocity_scale"])
        if hasattr(env_cfg, "use_time_penalty") and "use_time_penalty" in _mpc:
            env_cfg.use_time_penalty = bool(_mpc["use_time_penalty"])
        if hasattr(env_cfg, "regularize_time_scale") and "regularize_time_scale" in _mpc:
            env_cfg.regularize_time_scale = float(_mpc["regularize_time_scale"])

    # ── create env ───────────────────────────────────────────────────────────
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # Dump configs for reproducibility
    dump_yaml(str(log_dir / "params" / "env.yaml"), env_cfg)
    if _raw_fasttd3_cfg:
        dump_yaml(str(log_dir / "params" / "fasttd3.yaml"), _raw_fasttd3_cfg)

    # ── load task-specific residual weights ────────────────────────
    all_stages_weights: dict = {}
    if args_cli.residual_weights is not None:
        weights_path = Path(args_cli.residual_weights).resolve()
        print(f"[INFO] Loading residual weights from: {weights_path}")
        with open(weights_path, encoding="utf-8") as f:
            raw_ = json.load(f)["stages"]
        for stage_id_str, items in raw_.items():
            stage_id = int(stage_id_str)
            all_stages_weights[stage_id] = items["weights"]
            update_parameters(env.unwrapped, stage_id=stage_id, parameters=items["parameters"])
    update_weights(env.unwrapped, stage_id=None, weights=None, all_stages_weights=all_stages_weights)

    # ── env adapter ───────────────────────────────────────────────────────────
    envs = IsaacLabEnvAdapter(env, action_bounds=args_cli.action_bounds)

    n_obs      = envs.num_obs
    n_act      = envs.num_actions
    n_crit_obs = envs.num_privileged_obs if envs.asymmetric_obs else n_obs

    print(
        f"[INFO] env: obs={n_obs}  act={n_act}  critic_obs={n_crit_obs}  "
        f"asymmetric={envs.asymmetric_obs}  num_envs={envs.num_envs}"
    )

    # ── AMP ──────────────────────────────────────────────────────────────────
    amp_enabled = args_cli.use_amp and device.type == "cuda"
    amp_dtype   = torch.bfloat16 if args_cli.amp_dtype == "bf16" else torch.float16
    scaler      = GradScaler(enabled=amp_enabled and amp_dtype == torch.float16)

    # ── observation / reward normalization ────────────────────────────────────
    if args_cli.obs_normalization:
        obs_normalizer       = EmpiricalNormalization(shape=n_obs,      device=device)
        critic_obs_normalizer = EmpiricalNormalization(shape=n_crit_obs, device=device)
    else:
        obs_normalizer        = nn.Identity()
        critic_obs_normalizer = nn.Identity()

    reward_normalizer: nn.Module
    if args_cli.reward_normalization:
        reward_normalizer = RewardNormalizer(
            gamma=args_cli.gamma,
            device=device,
            g_max=min(abs(args_cli.v_min), abs(args_cli.v_max)),
        )
    else:
        reward_normalizer = nn.Identity()

    # ── actor & critic ────────────────────────────────────────────────────────
    actor = Actor(
        n_obs     = n_obs,
        n_act     = n_act,
        num_envs  = envs.num_envs,
        device    = device,
        init_scale = args_cli.init_scale,
        hidden_dim = args_cli.actor_hidden_dim,
        std_min    = args_cli.std_min,
        std_max    = args_cli.std_max,
    )

    qnet = Critic(
        n_obs      = n_crit_obs,
        n_act      = n_act,
        num_atoms  = args_cli.num_atoms,
        v_min      = args_cli.v_min,
        v_max      = args_cli.v_max,
        hidden_dim = args_cli.critic_hidden_dim,
        device     = device,
    )
    qnet_target = Critic(
        n_obs      = n_crit_obs,
        n_act      = n_act,
        num_atoms  = args_cli.num_atoms,
        v_min      = args_cli.v_min,
        v_max      = args_cli.v_max,
        hidden_dim = args_cli.critic_hidden_dim,
        device     = device,
    )
    qnet_target.load_state_dict(qnet.state_dict())

    # ── BC prior warm-start ───────────────────────────────────────────────────
    if args_cli.prior_checkpoint is not None:
        from kuafu.bc.training.fast_td3_prior import apply_bc_prior_to_fast_td3_actor

        print(f"[INFO] Loading BC prior from: {args_cli.prior_checkpoint}")
        report = apply_bc_prior_to_fast_td3_actor(
            actor,
            obs_normalizer if args_cli.obs_normalization else None,
            args_cli.prior_checkpoint,
            obs_stats_count=args_cli.prior_obs_stats_count,
            verbose=args_cli.prior_verbose,
        )
        print(f"[INFO] BC prior report: {report}")

    # ── separate actor_detach for inference (avoids autograd overhead) ────────
    try:
        from tensordict import from_module
        actor_detach = Actor(
            n_obs      = n_obs,
            n_act      = n_act,
            num_envs   = envs.num_envs,
            device     = device,
            init_scale = args_cli.init_scale,
            hidden_dim = args_cli.actor_hidden_dim,
            std_min    = args_cli.std_min,
            std_max    = args_cli.std_max,
        )
        from_module(actor).data.to_module(actor_detach)
        policy = actor_detach.explore
    except Exception:
        policy = actor.explore

    # ── optimizers & LR schedulers ────────────────────────────────────────────
    q_optimizer = optim.AdamW(
        list(qnet.parameters()),
        lr=torch.tensor(args_cli.critic_lr, device=device),
        weight_decay=args_cli.weight_decay,
    )
    actor_optimizer = optim.AdamW(
        list(actor.parameters()),
        lr=torch.tensor(args_cli.actor_lr, device=device),
        weight_decay=args_cli.weight_decay,
    )
    q_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        q_optimizer, T_max=args_cli.total_timesteps,
        eta_min=torch.tensor(args_cli.critic_lr_end, device=device),
    )
    actor_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        actor_optimizer, T_max=args_cli.total_timesteps,
        eta_min=torch.tensor(args_cli.actor_lr_end, device=device),
    )

    # ── replay buffer ─────────────────────────────────────────────────────────
    rb = SimpleReplayBuffer(
        n_env        = envs.num_envs,
        buffer_size  = args_cli.buffer_size,
        n_obs        = n_obs,
        n_act        = n_act,
        n_critic_obs = n_crit_obs,
        asymmetric_obs = envs.asymmetric_obs,
        playground_mode = False,
        n_steps      = args_cli.num_steps,
        gamma        = args_cli.gamma,
        device       = device,
    )

    if args_cli.demo_preload:
        if args_cli.demo_dataset is None:
            raise ValueError(
                "Replay preload requested but no demo dataset was provided. "
                "Set dataset.dataset in --agent_cfg or pass --demo_dataset."
            )
        from kuafu.bc.training.demo_buffer import populate_replay_buffer_from_hdf5

        demo_dataset_path = _resolve_runtime_path(args_cli.demo_dataset)
        if demo_dataset_path is None:
            raise ValueError("Could not resolve demo dataset path.")
        print(f"[INFO] Preloading replay from dataset: {demo_dataset_path}")
        loaded_transitions = populate_replay_buffer_from_hdf5(
            rb,
            dataset_path=demo_dataset_path,
            obs_key=args_cli.demo_obs_key,
            action_mode=args_cli.demo_action_mode,
            device=device,
            prior_ckpt_path=args_cli.prior_checkpoint,
            max_transitions=args_cli.demo_max_transitions,
            use_rewards=args_cli.demo_use_rewards,
            preload_fraction=args_cli.demo_preload_fraction,
            preload_slots=args_cli.demo_preload_slots,
            verbose=args_cli.demo_verbose,
        )
        print(
            "[INFO] Replay preload complete: "
            f"{loaded_transitions} demo transitions loaded; continuing with online RL rollouts."
        )

    policy_noise = args_cli.policy_noise
    noise_clip   = args_cli.noise_clip

    # ── wandb ─────────────────────────────────────────────────────────────────
    if args_cli.track:
        import wandb
        wandb.init(
            project = args_cli.wandb_project,
            entity  = args_cli.wandb_entity,
            name    = exp_name,
            config  = vars(args_cli),
            save_code = True,
            sync_tensorboard = False,
        )

    # ── compile helper ────────────────────────────────────────────────────────
    def _maybe_compile(fn, mode="reduce-overhead"):
        if args_cli.compile:
            return torch.compile(fn, mode=mode)
        return fn

    # ── training sub-functions ───────────────────────────────────────────────

    def update_critic(data: TensorDict, logs: TensorDict) -> TensorDict:
        with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            observations      = data["observations"]
            next_observations = data["next"]["observations"]
            if envs.asymmetric_obs:
                critic_obs      = data["critic_observations"]
                next_critic_obs = data["next"]["critic_observations"]
            else:
                critic_obs      = observations
                next_critic_obs = next_observations
            actions     = data["actions"]
            rewards     = data["next"]["rewards"]
            dones       = data["next"]["dones"].bool()
            truncations = data["next"]["truncations"].bool()

            bootstrap = (
                (~dones).float() if args_cli.disable_bootstrap
                else (truncations | ~dones).float()
            )
            clipped_noise = (
                torch.randn_like(actions).mul(policy_noise).clamp(-noise_clip, noise_clip)
            )
            next_actions = (actor(next_observations) + clipped_noise).clamp(-1.0, 1.0)
            discount     = args_cli.gamma ** data["next"]["effective_n_steps"]

            with torch.no_grad():
                q1_proj, q2_proj = qnet_target.projection(
                    next_critic_obs, next_actions, rewards, bootstrap, discount
                )
                q1_val = qnet_target.get_value(q1_proj)
                q2_val = qnet_target.get_value(q2_proj)
                if args_cli.use_cdq:
                    target_dist = torch.where(
                        q1_val.unsqueeze(1) < q2_val.unsqueeze(1), q1_proj, q2_proj
                    )
                    q1_target = q2_target = target_dist
                else:
                    q1_target, q2_target = q1_proj, q2_proj

            qf1, qf2 = qnet(critic_obs, actions)
            qf1_loss = -torch.sum(q1_target * F.log_softmax(qf1, dim=1), dim=1).mean()
            qf2_loss = -torch.sum(q2_target * F.log_softmax(qf2, dim=1), dim=1).mean()
            qf_loss  = qf1_loss + qf2_loss

        q_optimizer.zero_grad(set_to_none=True)
        scaler.scale(qf_loss).backward()
        scaler.unscale_(q_optimizer)
        if args_cli.use_grad_norm_clipping:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                qnet.parameters(),
                args_cli.max_grad_norm if args_cli.max_grad_norm > 0 else float("inf"),
            )
        else:
            grad_norm = torch.tensor(0.0, device=device)
        scaler.step(q_optimizer)
        scaler.update()

        logs["critic_grad_norm"] = grad_norm.detach()
        logs["qf_loss"]          = qf_loss.detach()
        logs["qf_max"]           = q1_val.max().detach()
        logs["qf_min"]           = q1_val.min().detach()
        return logs

    def update_actor(data: TensorDict, logs: TensorDict) -> TensorDict:
        with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            critic_obs = (
                data["critic_observations"] if envs.asymmetric_obs
                else data["observations"]
            )
            qf1, qf2  = qnet(critic_obs, actor(data["observations"]))
            q1_val    = qnet.get_value(F.softmax(qf1, dim=1))
            q2_val    = qnet.get_value(F.softmax(qf2, dim=1))
            q_val     = torch.minimum(q1_val, q2_val) if args_cli.use_cdq else (q1_val + q2_val) / 2.0
            actor_loss = -q_val.mean()

        actor_optimizer.zero_grad(set_to_none=True)
        scaler.scale(actor_loss).backward()
        scaler.unscale_(actor_optimizer)
        if args_cli.use_grad_norm_clipping:
            a_grad_norm = torch.nn.utils.clip_grad_norm_(
                actor.parameters(),
                args_cli.max_grad_norm if args_cli.max_grad_norm > 0 else float("inf"),
            )
        else:
            a_grad_norm = torch.tensor(0.0, device=device)
        scaler.step(actor_optimizer)
        scaler.update()

        logs["actor_grad_norm"] = a_grad_norm.detach()
        logs["actor_loss"]      = actor_loss.detach()
        return logs

    @torch.no_grad()
    def soft_update(src: nn.Module, tgt: nn.Module, tau: float) -> None:
        src_ps = [p.data for p in src.parameters()]
        tgt_ps = [p.data for p in tgt.parameters()]
        torch._foreach_mul_(tgt_ps, 1.0 - tau)
        torch._foreach_add_(tgt_ps, src_ps, alpha=tau)

    # Optionally compile the hot paths
    update_critic_fn = _maybe_compile(update_critic)
    update_actor_fn  = _maybe_compile(update_actor)
    policy_fn        = _maybe_compile(policy, mode=None)

    normalize_obs        = _maybe_compile(obs_normalizer.forward, mode=None)
    normalize_critic_obs = _maybe_compile(critic_obs_normalizer.forward, mode=None)
    normalize_reward     = _maybe_compile(reward_normalizer.forward, mode=None)
    if args_cli.reward_normalization:
        update_reward_stats = _maybe_compile(reward_normalizer.update_stats, mode=None)

    # ── evaluation ────────────────────────────────────────────────────────────
    @torch.no_grad()
    def evaluate() -> tuple[float, float]:
        ep_returns = torch.zeros(envs.num_envs, device=device)
        ep_lengths = torch.zeros(envs.num_envs, device=device)
        done_masks = torch.zeros(envs.num_envs, dtype=torch.bool, device=device)
        e_obs = envs.reset(random_start_init=False)
        for _ in range(envs.max_episode_steps):
            with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                norm_e_obs = normalize_obs(e_obs, update=False)
                e_actions  = actor(norm_e_obs)
            e_obs, e_rew, e_dones, _ = envs.step(e_actions.float())
            ep_returns = torch.where(~done_masks, ep_returns + e_rew, ep_returns)
            ep_lengths = torch.where(~done_masks, ep_lengths + 1,     ep_lengths)
            done_masks = done_masks | (e_dones > 0)
            if done_masks.all():
                break
        return ep_returns.mean().item(), ep_lengths.mean().item()

    # ── load checkpoint or prior; initialise global step ──────────────────────
    if args_cli.checkpoint is not None:
        ckpt_path = Path(args_cli.checkpoint).resolve()
        print(f"[INFO] Resuming from FastTD3 checkpoint: {ckpt_path}")
        ckpt_data = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        actor.load_state_dict(ckpt_data["actor_state_dict"])
        qnet.load_state_dict(ckpt_data["qnet_state_dict"])
        qnet_target.load_state_dict(ckpt_data["qnet_target_state_dict"])
        obs_normalizer.load_state_dict(ckpt_data["obs_normalizer_state"])
        critic_obs_normalizer.load_state_dict(ckpt_data["critic_obs_normalizer_state"])
        global_step = int(ckpt_data.get("global_step", 0))
    else:
        global_step = 0

    # ── reset environment ─────────────────────────────────────────────────────
    if envs.asymmetric_obs:
        obs, critic_obs = envs.reset_with_critic_obs()
        critic_obs = critic_obs.to(device=device, dtype=torch.float)
    else:
        obs      = envs.reset()
        critic_obs = None  # type: ignore[assignment]

    obs = obs.to(device=device, dtype=torch.float)

    dones      = None
    start_time = None
    measure_burnin = 0
    pbar = tqdm.tqdm(total=args_cli.total_timesteps, initial=global_step, desc=exp_name)

    # ── main loop ─────────────────────────────────────────────────────────────
    while global_step < args_cli.total_timesteps:
        mark_step()
        logs_dict = TensorDict()

        if (
            start_time is None
            and global_step >= 3 + args_cli.learning_starts  # 3-step burnin
        ):
            start_time    = time.time()
            measure_burnin = global_step

        # ---- collect one step -----------------------------------------------
        with torch.no_grad(), autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            norm_obs = normalize_obs(obs)
            actions  = policy_fn(obs=norm_obs, dones=dones)

        next_obs, rewards, dones, infos = envs.step(actions.float())
        truncations = infos["time_outs"]

        if args_cli.reward_normalization:
            update_reward_stats(rewards, dones.float())

        # "true" next obs: use the reset obs (from info) when done
        true_next_obs = torch.where(
            dones[:, None] > 0, infos["observations"]["raw"]["obs"], next_obs
        )
        if envs.asymmetric_obs:
            next_critic_obs      = infos["observations"]["critic"]
            true_next_critic_obs = torch.where(
                dones[:, None] > 0,
                infos["observations"]["raw"]["critic_obs"],
                next_critic_obs,
            )

        transition = TensorDict(
            {
                "observations": obs,
                "actions": actions.to(device=device, dtype=torch.float),
                "next": {
                    "observations": true_next_obs,
                    "rewards":      rewards.to(device=device, dtype=torch.float),
                    "truncations":  truncations.long(),
                    "dones":        dones.long(),
                },
            },
            batch_size=(envs.num_envs,),
            device=device,
        )
        if envs.asymmetric_obs:
            transition["critic_observations"]          = critic_obs
            transition["next"]["critic_observations"]  = true_next_critic_obs

        rb.extend(transition)
        obs = next_obs.to(device=device, dtype=torch.float)
        if envs.asymmetric_obs:
            critic_obs = next_critic_obs.to(device=device, dtype=torch.float)

        # ---- update step ---------------------------------------------------
        if global_step > args_cli.learning_starts:
            for i in range(args_cli.num_updates):
                data = rb.sample(max(1, args_cli.batch_size // envs.num_envs))

                data["observations"]       = normalize_obs(data["observations"])
                data["next"]["observations"] = normalize_obs(data["next"]["observations"])
                if envs.asymmetric_obs:
                    data["critic_observations"]        = normalize_critic_obs(data["critic_observations"])
                    data["next"]["critic_observations"] = normalize_critic_obs(data["next"]["critic_observations"])

                raw_rewards = data["next"]["rewards"]
                data["next"]["rewards"] = normalize_reward(raw_rewards)

                logs_dict = update_critic_fn(data, logs_dict)

                do_actor_update = (
                    (i % args_cli.policy_frequency == 1) if args_cli.num_updates > 1
                    else (global_step % args_cli.policy_frequency == 0)
                )
                if do_actor_update:
                    logs_dict = update_actor_fn(data, logs_dict)

                    # sync actor_detach after each actor update
                    try:
                        from tensordict import from_module
                        from_module(actor).data.to_module(actor_detach)
                    except Exception:
                        pass

                soft_update(qnet, qnet_target, args_cli.tau)

        # ---- periodic logging ----------------------------------------------
        if global_step > args_cli.learning_starts and global_step % 100 == 0 and start_time is not None:
            speed = (global_step - measure_burnin) / (time.time() - start_time)
            log_payload: dict = {
                "speed_sps": speed,
                "frame": global_step * envs.num_envs,
                "env_rewards": rewards.mean().item(),
                "critic_lr": q_scheduler.get_last_lr()[0],
                "actor_lr":  actor_scheduler.get_last_lr()[0],
            }
            for k in ("actor_loss", "qf_loss", "qf_max", "qf_min",
                      "actor_grad_norm", "critic_grad_norm"):
                if k in logs_dict:
                    log_payload[k] = logs_dict[k].mean().item()

            if args_cli.eval_interval > 0 and global_step % args_cli.eval_interval == 0:
                print(f"[INFO] Evaluating at step {global_step} …")
                avg_ret, avg_len = evaluate()
                log_payload["eval_avg_return"] = avg_ret
                log_payload["eval_avg_length"] = avg_len
                print(
                    f"[EVAL] step={global_step}  avg_return={avg_ret:.2f}  "
                    f"avg_length={avg_len:.1f}"
                )
                # reset env after eval
                obs = envs.reset().to(device=device, dtype=torch.float)

            pbar.set_description(f"{speed:.2f} sps")
            if args_cli.track:
                import wandb
                wandb.log(log_payload, step=global_step)

        # ---- periodic checkpoint -------------------------------------------
        if (
            args_cli.save_interval > 0
            and global_step > 0
            and global_step % args_cli.save_interval == 0
        ):
            ckpt_out = log_dir / f"{exp_name}_{global_step}.pt"
            ckpt_out.parent.mkdir(parents=True, exist_ok=True)
            save_params(
                global_step, actor, qnet, qnet_target,
                obs_normalizer, critic_obs_normalizer,
                args_cli, str(ckpt_out),
            )
            print(f"[INFO] Saved checkpoint: {ckpt_out}")

        global_step  += 1
        actor_scheduler.step()
        q_scheduler.step()
        pbar.update(1)

    # ── final save ────────────────────────────────────────────────────────────
    final_ckpt = log_dir / f"{exp_name}_final.pt"
    save_params(
        global_step, actor, qnet, qnet_target,
        obs_normalizer, critic_obs_normalizer,
        args_cli, str(final_ckpt),
    )
    print(f"[INFO] Training complete.  Final checkpoint: {final_ckpt}")

    if args_cli.track:
        import wandb
        wandb.finish()

    envs.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
