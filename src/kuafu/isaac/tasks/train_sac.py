"""Script to train a SAC agent with RLPD (RL with Prior Data) using SKRL.

The replay buffer is pre-seeded with expert demonstrations from an HDF5 file
before any environment interaction.  Optionally the actor is warm-started from
a robomimic BC checkpoint so exploration begins near the demonstrated behaviour.

Usage (basic):
    python src/kuafu/isaac/tasks/train_sac.py --task open-fr3-finray-drawer --num_envs 64

With BC warm-start:
    python src/kuafu/isaac/tasks/train_sac.py \\
        --task open-fr3-finray-drawer --num_envs 64 \\
        --bc_checkpoint $KUAFU_ARTIFACT_ROOT/standalone/robomimic/open-fr3-finray-drawer/bc/.../model_epoch_2000.pth

Resume training:
    python src/kuafu/isaac/tasks/train_sac.py ... \\
        --checkpoint $KUAFU_ARTIFACT_ROOT/standalone/skrl/Open-FR3FinRay-Drawer-SAC/.../checkpoints/agent_50000.pt
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
from distutils.util import strtobool
from pathlib import Path

from kuafu.paths import external_artifact_root, resolve_artifact_path

from isaaclab.app import AppLauncher

# Unified package imports are supplied by the project launcher.

import kuafu.isaac.tasks.open_drawer  # noqa: F401  registers gym envs

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Train SAC+RLPD agent with SKRL.")
parser.add_argument("--task",      type=str, default="open-fr3-finray-drawer",
                    help="Gym task id (must be registered in kuafu.isaac.tasks).")
parser.add_argument("--num_envs",  type=int, default=64,
                    help="Number of parallel environments.")
parser.add_argument("--agent_cfg", type=str,
                    default="src/kuafu/isaac/tasks/open_drawer/agents/sac_rlpd_cfg.yaml",
                    help="Path to SAC+RLPD YAML config (relative to repo root or absolute).")
parser.add_argument("--demo_dataset", type=str, default=None,
                    help="Override demo HDF5 path from config.")
parser.add_argument("--bc_checkpoint",  type=str, default=None,
                    help="Robomimic .pth checkpoint for actor warm-start (overrides config).")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="SKRL checkpoint to resume training.")
parser.add_argument("--timesteps", type=int, default=None,
                    help="Override total training timesteps from config.")
parser.add_argument("--seed", type=int, default=-1)
parser.add_argument("--video",          action="store_true", default=False)
parser.add_argument("--video_length",   type=int, default=400)
parser.add_argument("--video_interval", type=int, default=10_000)
parser.add_argument("--residual_weights", type=str, default=None,
                    help="Path to JSON file with per-stage SUNProgramEvaluator residual weights.")
parser.add_argument(
    "--track",
    type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
    help="Track experiment with Weights & Biases.",
)
parser.add_argument("--wandb-project-name", type=str, default="KUAFU")
parser.add_argument("--wandb-entity",       type=str, default=None)
parser.add_argument("--wandb-name",         type=str, default=None)

# Add Isaac Sim launcher args (--device, --headless, etc.)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)

args_cli, hydra_args = parser.parse_known_args()
for _artifact_attr, _artifact_label in (
    ("demo_dataset", "Demonstration trajectory"),
    ("bc_checkpoint", "BC checkpoint"),
    ("checkpoint", "SAC checkpoint"),
):
    _artifact_value = getattr(args_cli, _artifact_attr, None)
    if _artifact_value:
        setattr(args_cli, _artifact_attr, str(resolve_artifact_path(_artifact_value, _artifact_label)))
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import importlib
import json
import logging
import os
import random
from datetime import datetime

import h5py
import torch
import torch.nn as nn
import yaml

import skrl
from packaging import version

SKRL_MIN = "1.4.3"
if version.parse(skrl.__version__) < version.parse(SKRL_MIN):
    raise RuntimeError(
        f"skrl>={SKRL_MIN} required, got {skrl.__version__}. "
        "Run: pip install 'skrl>=1.4.3'"
    )

import gymnasium as gym
from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.skrl import SkrlVecEnvWrapper

from kuafu.rl.isaac.update_utilities import update_weights, update_parameters
from kuafu.data.weight_transfer import inspect_checkpoint, transfer_weights

logger = logging.getLogger(__name__)

# ── Task → env-config lookup ──────────────────────────────────────────────────
# Add new tasks here as needed.
_ENV_CFG_MAP: dict[str, tuple[str, str]] = {
    "open-fr3-finray-drawer": (
        "kuafu.isaac.tasks.open_drawer.open_fr3_finray_drawer_env_learned",
        "OpenDrawerEnvCfg",
    ),
    "open-drawer": (
        "kuafu.isaac.tasks.open_drawer.open_drawer_env_learned",
        "OpenDrawerEnvCfg",
    ),
}


def _load_env_cfg(task: str):
    if task not in _ENV_CFG_MAP:
        raise ValueError(
            f"Task '{task}' not in _ENV_CFG_MAP. "
            f"Add it or choose from: {list(_ENV_CFG_MAP.keys())}"
        )
    module_path, class_name = _ENV_CFG_MAP[task]
    return getattr(importlib.import_module(module_path), class_name)()


# ── SKRL models ───────────────────────────────────────────────────────────────

class Actor(GaussianMixin, Model):
    """Stochastic actor: obs → (mean, log_std).

    The trunk architecture [256 → 512 → 64] deliberately mirrors the BC MLP
    defined in bc_mlp.json so that weights can be transferred layer-for-layer
    from a robomimic checkpoint.
    """

    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions=True)  # tanh squashing to [-1, 1]

        self.trunk = nn.Sequential(
            nn.Linear(self.num_observations, 256), nn.ELU(),
            nn.Linear(256, 512),                   nn.ELU(),
            nn.Linear(512, 64),                    nn.ELU(),
        )
        self.mean_layer = nn.Linear(64, self.num_actions)
        # Learnable per-action log standard deviation (initialised to 0 = std 1)
        self.log_std = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        x = self.trunk(inputs["states"])
        return self.mean_layer(x), self.log_std, {}


class Critic(DeterministicMixin, Model):
    """Q-value critic: (obs ∥ action) → Q scalar.

    Separate actor and critic networks (no shared trunk) so BC pre-training
    of the actor never interferes with value learning.
    """

    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations + self.num_actions, 256), nn.ELU(),
            nn.Linear(256, 512),                                       nn.ELU(),
            nn.Linear(512, 64),                                        nn.ELU(),
            nn.Linear(64, 1),
        )

    def compute(self, inputs, role):
        x = torch.cat([inputs["states"], inputs["taken_actions"]], dim=-1)
        return self.net(x), {}


# ── Demo seeding ──────────────────────────────────────────────────────────────

def seed_memory_from_hdf5(
    memory: RandomMemory,
    hdf5_path: str,
    num_envs: int,
    device: str,
    obs_key: str = "state",
) -> int:
    """Pre-fill the replay buffer with all expert demonstration transitions.

    Loads every demo from the HDF5 file, concatenates them into one large
    tensor, then inserts them in chunks of ``num_envs`` — the same batch size
    that SKRL's RandomMemory expects during normal online training.

    Returns the total number of transitions inserted.
    """
    print(f"[RLPD] Seeding replay buffer from: {hdf5_path}")
    all_s, all_ns, all_a, all_r, all_d = [], [], [], [], []

    with h5py.File(hdf5_path, "r") as f:
        demo_keys = sorted(f["data"].keys())
        for dk in demo_keys:
            d = f["data"][dk]
            all_s.append( torch.tensor(d["obs"][obs_key][:],      dtype=torch.float32))
            all_ns.append(torch.tensor(d["next_obs"][obs_key][:], dtype=torch.float32))
            all_a.append( torch.tensor(d["actions"][:],           dtype=torch.float32))
            all_r.append( torch.tensor(d["rewards"][:],           dtype=torch.float32).unsqueeze(-1))
            all_d.append( torch.tensor(d["dones"][:],             dtype=torch.float32).unsqueeze(-1))

    states      = torch.cat(all_s,  dim=0).to(device)
    next_states = torch.cat(all_ns, dim=0).to(device)
    actions     = torch.cat(all_a,  dim=0).to(device)
    rewards     = torch.cat(all_r,  dim=0).to(device)
    terminated  = torch.cat(all_d,  dim=0).to(device)
    truncated   = torch.zeros_like(terminated)

    T, n_added = states.shape[0], 0
    for start in range(0, T - num_envs + 1, num_envs):
        end = start + num_envs
        memory.add_samples(
            states=states[start:end],
            actions=actions[start:end],
            rewards=rewards[start:end],
            next_states=next_states[start:end],
            terminated=terminated[start:end],
            truncated=truncated[start:end],
        )
        n_added += num_envs

    print(f"[RLPD] Seeded {n_added:,} transitions from {len(demo_keys)} demonstrations.")
    return n_added


# ── BC weight warm-start ──────────────────────────────────────────────────────

def maybe_load_bc_weights(actor: Actor, bc_checkpoint: str | None) -> None:
    """Warm-start the actor trunk from a robomimic BC checkpoint (optional).

    Prints all checkpoint keys so mismatches are easy to diagnose.
    Update ``DEFAULT_KEY_MAP`` in weight_transfer.py if your checkpoint uses
    different layer names (visible in the printout).
    """
    if bc_checkpoint is None:
        print("[RLPD] No BC checkpoint provided — actor initialised randomly.")
        return

    print(f"[RLPD] Loading BC weights from: {bc_checkpoint}")
    inspect_checkpoint(bc_checkpoint)  # shows all keys for debugging

    sd, n = transfer_weights(actor.state_dict(), bc_checkpoint)
    actor.load_state_dict(sd)

    if n == 0:
        print(
            "[RLPD] WARNING: 0 weights transferred. "
            "Check the key names printed above and update DEFAULT_KEY_MAP in "
            "src/kuafu/data/weight_transfer.py."
        )
    else:
        print(f"[RLPD] Actor warm-started: {n} weight tensors loaded from BC checkpoint.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    seed = args_cli.seed if args_cli.seed != -1 else random.randint(0, 10_000)
    set_seed(seed)

    # ── Load YAML config ─────────────────────────────────────────────────────
    cfg_path = os.path.abspath(args_cli.agent_cfg)
    with open(cfg_path) as f:
        agent_cfg = yaml.safe_load(f)

    sac_cfg   = agent_cfg["sac"]
    rlpd_cfg  = agent_cfg["rlpd"]
    train_cfg = agent_cfg["train"]
    mpc_cfg   = agent_cfg.get("mpc", {})
    exp_cfg   = agent_cfg["agent"]["experiment"]

    # ── Env config ───────────────────────────────────────────────────────────
    env_cfg            = _load_env_cfg(args_cli.task)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device     = args_cli.device or env_cfg.sim.device
    env_cfg.seed           = seed

    # MPC / reward-shaping overrides (mirrors train.py)
    env_cfg.mpc_config["num_envs"]           = env_cfg.scene.num_envs
    env_cfg.mpc_config["use_per_env_weight"] = False
    for k in (
        "use_absolute_progress_rewards", "use_relative_progress_rewards",
        "use_velocity_penalty",          "use_time_penalty",
        "regularize_velocity_scale",     "regularize_time_scale",
        "progress_rewards_scale",
    ):
        if k in mpc_cfg:
            env_cfg.mpc_config[k] = mpc_cfg[k]

    # ── Logging dirs ─────────────────────────────────────────────────────────
    log_root     = str(external_artifact_root() / "standalone" / "skrl" / exp_cfg["directory"])
    log_dir      = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_sac"
    if exp_cfg.get("experiment_name"):
        log_dir += f'_{exp_cfg["experiment_name"]}'
    full_log_dir = os.path.join(log_root, log_dir)
    print(f"[INFO] Logging experiment in directory: {full_log_dir}")

    env_cfg.log_dir = full_log_dir
    dump_yaml(os.path.join(full_log_dir, "params", "env.yaml"),   env_cfg)
    dump_yaml(os.path.join(full_log_dir, "params", "agent.yaml"), agent_cfg)

    # ── Create & wrap env ────────────────────────────────────────────────────
    env = gym.make(
        args_cli.task, cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )
    if args_cli.video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=os.path.join(full_log_dir, "videos", "train"),
            step_trigger=lambda step: step % args_cli.video_interval == 0,
            video_length=args_cli.video_length,
            disable_logger=True,
        )
    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    device   = env_cfg.sim.device
    num_envs = env.num_envs
    print(f"[INFO] Observation space : {env.observation_space}")
    print(f"[INFO] Action space      : {env.action_space}")

    # ── Residual weights (mirrors train.py) ──────────────────────────────────
    all_stages_weights: dict = {}
    if args_cli.residual_weights:
        rw_path = os.path.abspath(args_cli.residual_weights)
        if os.path.exists(rw_path):
            with open(rw_path, encoding="utf-8") as f:
                raw = json.load(f)["stages"]
            for sid_str, items in raw.items():
                sid = int(sid_str)
                all_stages_weights[sid] = items["weights"]
                update_parameters(env.unwrapped, stage_id=sid,
                                  parameters=items["parameters"])
        else:
            logger.warning(f"residual_weights file not found: {rw_path}")
    update_weights(env.unwrapped, stage_id=None, weights=None,
                   all_stages_weights=all_stages_weights)

    # ── SKRL models ───────────────────────────────────────────────────────────
    obs_space = env.observation_space
    act_space = env.action_space

    actor           = Actor(obs_space, act_space, device)
    critic_1        = Critic(obs_space, act_space, device)
    critic_2        = Critic(obs_space, act_space, device)
    target_critic_1 = Critic(obs_space, act_space, device)
    target_critic_2 = Critic(obs_space, act_space, device)

    # Target critics start as exact copies of the online critics
    target_critic_1.load_state_dict(critic_1.state_dict())
    target_critic_2.load_state_dict(critic_2.state_dict())

    models = {
        "policy":          actor,
        "critic_1":        critic_1,
        "critic_2":        critic_2,
        "target_critic_1": target_critic_1,
        "target_critic_2": target_critic_2,
    }

    # ── Replay buffer ─────────────────────────────────────────────────────────
    # Created before the agent so we can seed it with demo data first.
    memory = RandomMemory(
        memory_size=sac_cfg["memory_size"],
        num_envs=num_envs,
        device=device,
    )

    # ── RLPD: seed buffer with demonstrations ────────────────────────────────
    demo_path = args_cli.demo_dataset or rlpd_cfg.get("demo_dataset")
    if demo_path:
        demo_path = str(resolve_artifact_path(demo_path, "Demonstration trajectory"))
        if os.path.exists(demo_path):
            seed_memory_from_hdf5(
                memory, demo_path, num_envs, device,
                obs_key=rlpd_cfg.get("obs_key", "state"),
            )
        else:
            print(f"[RLPD] WARNING: demo dataset not found at {demo_path}")
    else:
        print("[RLPD] WARNING: no demo_dataset set. Buffer starts empty.")

    # ── BC actor warm-start (optional) ────────────────────────────────────────
    bc_ckpt = args_cli.bc_checkpoint or rlpd_cfg.get("bc_checkpoint")
    if bc_ckpt:
        bc_ckpt = str(resolve_artifact_path(bc_ckpt, "BC checkpoint"))
    maybe_load_bc_weights(actor, bc_ckpt)

    # ── SAC agent config ──────────────────────────────────────────────────────
    cfg = SAC_DEFAULT_CONFIG.copy()
    cfg["gradient_steps"]        = sac_cfg["gradient_steps"]
    cfg["batch_size"]            = sac_cfg["batch_size"]
    cfg["discount_factor"]       = sac_cfg["discount_factor"]
    cfg["polyak"]                = sac_cfg["polyak"]
    cfg["actor_learning_rate"]   = sac_cfg["actor_lr"]
    cfg["critic_learning_rate"]  = sac_cfg["critic_lr"]
    cfg["learn_entropy"]         = sac_cfg["learn_entropy"]
    cfg["entropy_learning_rate"] = sac_cfg["entropy_lr"]
    cfg["initial_entropy_value"] = sac_cfg["initial_entropy"]
    cfg["grad_norm_clip"]        = sac_cfg["grad_norm_clip"]
    cfg["learning_starts"]       = sac_cfg.get("learning_starts", 0)
    cfg["experiment"]["directory"]           = log_root
    cfg["experiment"]["experiment_name"]     = log_dir
    cfg["experiment"]["write_interval"]      = train_cfg.get("log_interval",        1_000)
    cfg["experiment"]["checkpoint_interval"] = train_cfg.get("checkpoint_interval", 50_000)

    agent = SAC(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=obs_space,
        action_space=act_space,
        device=device,
    )

    # ── Resume from SKRL checkpoint ───────────────────────────────────────────
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
        print(f"[INFO] Loading SKRL checkpoint from: {resume_path}")
        agent.load(resume_path)

    # ── W&B (optional) ────────────────────────────────────────────────────────
    if args_cli.track:
        import wandb
        wandb.init(
            project=args_cli.wandb_project_name,
            entity=args_cli.wandb_entity,
            name=args_cli.wandb_name or log_dir,
            sync_tensorboard=True,
            save_code=True,
        )

    # ── Train ─────────────────────────────────────────────────────────────────
    timesteps = args_cli.timesteps or train_cfg["timesteps"]
    trainer = SequentialTrainer(
        env=env,
        agents=agent,
        cfg={"timesteps": timesteps, "headless": True},
    )
    trainer.train()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
