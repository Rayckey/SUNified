# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RL-Games."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
from distutils.util import strtobool

from kuafu.paths import external_artifact_root, resolve_artifact_path

from isaaclab.app import AppLauncher

# Custom task packages are registered after Isaac launches so the task name can
# select FR3FinRay-Franka, Panda-Franka, or Kinova without changing this script.

parser = argparse.ArgumentParser(description="Train an RL agent with RL-Games.")
parser.add_argument("--video", type=bool, default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=400, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=500, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=8192, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="stack-2", help="Name of the task.")
parser.add_argument(
    "--execution_robot_id",
    type=str,
    default=None,
    help="Robot task package to register. Defaults to inference from --task, then FR3FinRay-Franka.",
)
parser.add_argument("--dry_run", action="store_true", default=False, help="Resolve task registration and exit before launching Isaac.")
parser.add_argument("--program_path", type=str, default=None, help="relative path to the program folder")
parser.add_argument(
    "--agent", type=str, default="rl_games_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=-1, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument(
    "--prior_checkpoint",
    type=str,
    default=None,
    help="Path to a BC prior checkpoint or BC run directory.",
)
parser.add_argument(
    "--prior_obs_stats_count",
    type=float,
    default=1e6,
    help="RunningMeanStd count to use when copying BC observation stats.",
)
parser.add_argument(
    "--prior_verbose",
    action="store_true",
    default=False,
    help="Print copied BC prior layer shapes.",
)
parser.add_argument("--residual_weights", type=str, default=None, help="Path to JSON file with residual weights for SUNProgramEvaluator constraints.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--wandb-project-name", type=str, default="KUAFU", help="the wandb's project name")
parser.add_argument("--wandb-entity", type=str, default="milesway", help="the entity (team) of wandb's project")
parser.add_argument("--wandb-name", type=str, default="open-drawer-0", help="the name of wandb's run")
parser.add_argument(
    "--track",
    type=lambda x: bool(strtobool(x)),
    default=False,
    nargs="?",
    const=True,
    help="if toggled, this experiment will be tracked with Weights and Biases",
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
) 

from pathlib import Path
_repo_root = Path(__file__).resolve().parent.parent.parent

from task_registry import register_task_modules, resolve_execution_robot_id
from kuafu.rl.isaac.update_utilities import update_weights, update_parameters

AppLauncher.add_app_launcher_args(parser)
# default to headless mode
parser.set_defaults(headless=True)

args_cli, hydra_args = parser.parse_known_args()
for _artifact_attr, _artifact_label in (
    ("checkpoint", "RL checkpoint"),
    ("prior_checkpoint", "BC prior checkpoint"),
):
    _artifact_value = getattr(args_cli, _artifact_attr, None)
    if _artifact_value:
        setattr(args_cli, _artifact_attr, str(resolve_artifact_path(_artifact_value, _artifact_label)))
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

if args_cli.dry_run:
    resolved_robot_id = resolve_execution_robot_id(args_cli.task, args_cli.execution_robot_id)
    print(
        "[dry-run] train.py "
        f"task={args_cli.task} execution_robot_id={resolved_robot_id} "
        f"num_envs={args_cli.num_envs} device={args_cli.device} agent={args_cli.agent}"
    )
    raise SystemExit(0)

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

resolved_execution_robot_id = register_task_modules(
    task=args_cli.task,
    execution_robot_id=args_cli.execution_robot_id,
)
print(f"[INFO] Registered task package for execution_robot_id={resolved_execution_robot_id}", flush=True)

"""Rest everything follows."""

import gymnasium as gym
import json
import logging
import math
import os
import random
import torch
import torch.nn as nn
from datetime import datetime

# PyTorch 2.6+ compatibility: rl_games internally calls torch.load without
# weights_only=False, which fails on older checkpoints. Monkey-patch torch.load
# to default to weights_only=False so pre-trained weights can be loaded.
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rl_games import MultiObserver, PbtAlgoObserver, RlGamesGpuEnv, RlGamesVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

logger = logging.getLogger(__name__)


def _stabilize_sigma_head_after_prior(agent_cfg: dict, agent) -> None:
    """Reset a learned sigma head after copying BC actor weights."""
    net = agent.model.a2c_network
    fixed_sigma = bool(getattr(net, "fixed_sigma", False))
    if fixed_sigma or not hasattr(net, "sigma"):
        return

    sigma_head = net.sigma
    if not isinstance(sigma_head, nn.Linear):
        return

    sigma_init_val = 0.0
    try:
        sigma_init_val = float(
            agent_cfg["params"]["network"]["space"]["continuous"]["sigma_init"].get("val", 0.0)
        )
    except Exception:
        sigma_init_val = 0.0

    with torch.no_grad():
        sigma_head.weight.zero_()
        if sigma_head.bias is not None:
            sigma_head.bias.fill_(sigma_init_val)

    print(
        f"[INFO] Reset learned sigma head after BC prior load: weight=0 bias={sigma_init_val}",
        flush=True,
    )


def _apply_prior_obs_mode(env_cfg, prior_path: str) -> str | None:
    """Set env observation mode to match BC prior metadata."""
    from kuafu.bc.training.rl_games_prior import infer_prior_obs_key

    obs_key = infer_prior_obs_key(prior_path)

    set_obs_mode = getattr(env_cfg, "set_obs_mode", None)
    if callable(set_obs_mode):
        set_obs_mode(obs_key)
    else:
        setattr(env_cfg, "obs_mode", obs_key)
        obs_dims = getattr(env_cfg, "obs_dims", None)
        if isinstance(obs_dims, dict) and obs_key in obs_dims and hasattr(env_cfg, "observation_space"):
            setattr(env_cfg, "observation_space", int(obs_dims[obs_key]))
    return obs_key


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with RL-Games agent."""
    if args_cli.checkpoint is not None and args_cli.prior_checkpoint is not None:
        raise ValueError("--checkpoint and --prior_checkpoint are mutually exclusive.")

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.mpc_config["num_envs"] = env_cfg.scene.num_envs
    env_cfg.mpc_config["use_per_env_weight"] = False

    if "mpc" in agent_cfg:
        env_cfg.mpc_config["use_per_env_weight"]  = False
        env_cfg.mpc_config["use_absolute_progress_rewards"] = agent_cfg["mpc"]["use_absolute_progress_rewards"]
        env_cfg.mpc_config["use_relative_progress_rewards"] = agent_cfg["mpc"]["use_relative_progress_rewards"]
        env_cfg.mpc_config["use_velocity_penalty"] = agent_cfg["mpc"]["use_velocity_penalty"]
        env_cfg.mpc_config["regularize_velocity_scale"] = agent_cfg["mpc"]["regularize_velocity_scale"]
        env_cfg.mpc_config["regularize_time_scale"] = agent_cfg["mpc"]["regularize_time_scale"]
        env_cfg.mpc_config["progress_rewards_scale"] = agent_cfg["mpc"]["progress_rewards_scale"]

    # check for invalid combination of CPU device with distributed training
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # update agent device to match simulation device
    if args_cli.device is not None:
        agent_cfg["params"]["config"]["device"] = args_cli.device
        agent_cfg["params"]["config"]["device_name"] = args_cli.device

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    agent_cfg["params"]["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["params"]["seed"]
    agent_cfg["params"]["config"]["max_epochs"] = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg["params"]["config"]["max_epochs"]
    )
    resume_path = None
    if args_cli.checkpoint is not None:
        resume_path = retrieve_file_path(args_cli.checkpoint)
        agent_cfg["params"]["load_checkpoint"] = True
        agent_cfg["params"]["load_path"] = resume_path
        print(f"[INFO]: Loading model checkpoint from: {agent_cfg['params']['load_path']}")
    prior_path = None
    if args_cli.prior_checkpoint is not None:
        prior_path = retrieve_file_path(args_cli.prior_checkpoint)
        print(f"[INFO]: Loading BC prior from: {prior_path}")
        prior_obs_key = _apply_prior_obs_mode(env_cfg, prior_path)
        if prior_obs_key is not None:
            print(f"[INFO] Set env obs_mode from BC prior metadata: {prior_obs_key}", flush=True)
    train_sigma = agent_cfg["params"]["config"].get("train_sigma", None)

    # multi-gpu training config
    if args_cli.distributed:
        agent_cfg["params"]["seed"] += app_launcher.global_rank
        agent_cfg["params"]["config"]["device"] = f"cuda:{app_launcher.local_rank}"
        agent_cfg["params"]["config"]["device_name"] = f"cuda:{app_launcher.local_rank}"
        agent_cfg["params"]["config"]["multi_gpu"] = True
        # update env config device
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"

    # set the environment seed (after multi-gpu config for updated rank from agent seed)
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg["params"]["seed"]

    # specify directory for logging experiments
    config_name = agent_cfg["params"]["config"]["name"]
    log_root_path = os.path.join(str(external_artifact_root()), "standalone", "rl_games", config_name)
    if "pbt" in agent_cfg and agent_cfg["pbt"]["directory"] != ".":
        log_root_path = os.path.join(agent_cfg["pbt"]["directory"], log_root_path)
    else:
        log_root_path = os.path.abspath(log_root_path)

    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs
    log_dir = agent_cfg["params"]["config"].get("full_experiment_name", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    # set directory into agent config
    # logging directory path: <train_dir>/<full_experiment_name>
    agent_cfg["params"]["config"]["train_dir"] = log_root_path
    agent_cfg["params"]["config"]["full_experiment_name"] = log_dir
    wandb_project = config_name if args_cli.wandb_project_name is None else args_cli.wandb_project_name
    experiment_name = log_dir if args_cli.wandb_name is None else args_cli.wandb_name

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_root_path, log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_root_path, log_dir, "params", "agent.yaml"), agent_cfg)
    print(f"Exact experiment name requested from command line: {os.path.join(log_root_path, log_dir)}")

    # read configurations about the agent-training
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rl-games
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups)

    # load residual weights from JSON if provided
    all_stages_weights = {}
    if args_cli.residual_weights is not None:
        weights_path = os.path.abspath(args_cli.residual_weights)
        print(f"[INFO]: Loading residual weights from: {weights_path}")
        with open(weights_path, "r", encoding="utf-8") as f:
            raw_ = json.load(f)['stages']
            for stage_id_str, items in raw_.items():
                stage_id = int(stage_id_str)
                all_stages_weights[stage_id] = items['weights']
                update_parameters(env.unwrapped, stage_id= stage_id, parameters=items['parameters'])
        # Convert string keys to int: {stage_id: {constraint_idx: float}}

    update_weights(env.unwrapped, stage_id= None, weights=None, all_stages_weights=all_stages_weights)


    # register the environment to rl-games registry
    # note: in agents configuration: environment name must be "rlgpu"
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    # set number of actors into agent config
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs

    # ensure batch_size (num_envs * horizon_length) is divisible by minibatch_size
    horizon_length = agent_cfg["params"]["config"].get("horizon_length", 16)
    batch_size = env.unwrapped.num_envs * horizon_length
    minibatch_size = agent_cfg["params"]["config"].get("minibatch_size", batch_size)

    # create runner from rl-games

    if "pbt" in agent_cfg and agent_cfg["pbt"]["enabled"]:
        observers = MultiObserver([IsaacAlgoObserver(), PbtAlgoObserver(agent_cfg, args_cli)])
        runner = Runner(observers)
    else:
        runner = Runner(IsaacAlgoObserver())

    runner.load(agent_cfg)

    # reset the agent and env
    runner.reset()
    # train the agent

    global_rank = int(os.getenv("RANK", "0"))
    if args_cli.track and global_rank == 0:
        if args_cli.wandb_entity is None:
            raise ValueError("Weights and Biases entity must be specified for tracking.")
        import wandb

        wandb.init(
            project=wandb_project,
            entity=args_cli.wandb_entity,
            name=experiment_name,
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
        )
        wandb_run = getattr(wandb, "run", None)
        if wandb_run is not None and not wandb_run.resumed:
            wandb.config.update({"env_cfg": env_cfg.to_dict()})
            wandb.config.update({"agent_cfg": agent_cfg})

    run_kwargs = {"train": True, "play": False, "sigma": train_sigma}
    if resume_path is not None:
        runner.run({**run_kwargs, "checkpoint": resume_path})
    elif prior_path is None:
        runner.run(run_kwargs)
    else:
        pbt_cfg = agent_cfg.get("pbt")
        if isinstance(pbt_cfg, dict) and pbt_cfg.get("enabled", False):
            raise ValueError("--prior_checkpoint is not supported with PBT enabled.")

        from kuafu.bc.training.rl_games_prior import apply_bc_prior_to_rl_games_agent

        agent = runner.algo_factory.create(runner.algo_name, base_name="run", params=runner.params)
        report = apply_bc_prior_to_rl_games_agent(
            agent,
            prior_path,
            obs_stats_count=args_cli.prior_obs_stats_count,
            verbose=args_cli.prior_verbose,
            load_critic_prior=True,
            require_critic_prior=True,
        )
        print(f"[INFO] Prior load report: {report}", flush=True)
        _stabilize_sigma_head_after_prior(agent_cfg, agent)
        agent.train()

    # Workaround: wandb's gym integration expects an 'enabled' attribute on the
    # video wrapper which newer gymnasium.RecordVideo does not provide.
    if args_cli.video and not hasattr(env, "enabled"):
        env.enabled = True
    env.close()


if __name__ == "__main__":
    main()  # type: ignore[reportCallIssue]
    simulation_app.close()
