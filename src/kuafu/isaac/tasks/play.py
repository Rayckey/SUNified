# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RL-Games."""

"""Launch Isaac Sim Simulator first."""

import argparse
import json
import sys
from pathlib import Path

from kuafu.paths import external_artifact_root, resolve_artifact_path

from isaaclab.app import AppLauncher

from task_registry import register_task_modules, resolve_execution_robot_id  # noqa: E402

DEFAULT_VIEWER_RESOLUTION = (1920, 1080)
DEFAULT_VIEWER_EYE = (1.2, 1.0, 1.0)
DEFAULT_VIEWER_LOOKAT = (0.3, -0.3, 0.0)

parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from RL-Games.")
parser.add_argument("--video", type=bool, default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=3000, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--max_steps",
    type=int,
    default=None,
    help="Stop after this many environment steps (overrides video_length early exit when set).",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="FR3-FinRay-Table-Can-Long-Long-U", help="Name of the task.")
parser.add_argument(
    "--execution_robot_id",
    type=str,
    default=None,
    help="Robot task package to register. Defaults to inference from --task, then FR3FinRay-Franka.",
)
parser.add_argument("--dry_run", action="store_true", default=False, help="Resolve task registration and exit before launching Isaac.")
parser.add_argument(
    "--agent", type=str, default="rl_games_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--checkpoint", type=str, default=str(resolve_artifact_path("Residuals/FR3-FinRay-Table-Can-Long-Long-U/2026-04-03_01-52-47/rl_runs/2026-04-03_14-41-59_Run-3/nn/last_fr3-finray-can_ep_1000_rew_3398.3447.pth", "RL checkpoint")))
parser.add_argument(
    "--weights_params_json",
    type=str,
    default=str(resolve_artifact_path("Residuals/FR3-FinRay-Table-Can-Long-Long-U/2026-04-03_01-52-47/best.json", "Residual parameters")),
    help="Optional Kuafu weights/parameters JSON (same format as best.json or plain stages dict).",
)
parser.add_argument(
    "--obs_mode",
    type=str,
    choices=("state", "full_state"),
    default="full_state",
    help="Override the environment observation mode before loading the checkpoint.",
)
parser.add_argument("--seed", type=int, default=-1, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--use_last_checkpoint",
    action="store_true",
    help="When no checkpoint provided, use the last saved model. Otherwise use the best saved model.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--viewer_resolution",
    type=int,
    nargs=2,
    default=DEFAULT_VIEWER_RESOLUTION,
    metavar=("WIDTH", "HEIGHT"),
    help="Viewer camera resolution used for rendered video.",
)
parser.add_argument(
    "--viewer_eye",
    type=float,
    nargs=3,
    default=DEFAULT_VIEWER_EYE,
    metavar=("X", "Y", "Z"),
    help="Viewer camera eye position used for rendered video.",
)
parser.add_argument(
    "--viewer_lookat",
    type=float,
    nargs=3,
    default=DEFAULT_VIEWER_LOOKAT,
    metavar=("X", "Y", "Z"),
    help="Viewer camera lookat target used for rendered video.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.checkpoint:
    args_cli.checkpoint = str(resolve_artifact_path(args_cli.checkpoint, "RL checkpoint"))
if args_cli.weights_params_json:
    args_cli.weights_params_json = str(
        resolve_artifact_path(args_cli.weights_params_json, "Residual parameters")
    )
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

if args_cli.dry_run:
    resolved_robot_id = resolve_execution_robot_id(args_cli.task, args_cli.execution_robot_id)
    print(
        "[dry-run] play.py "
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
import math
import os
import random
import re
import time
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
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
try: 
    from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
except ImportError:
    from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from kuafu.rl.isaac.update_utilities import update_all_parameters, update_weights
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config


def _set_env_obs_mode(env_cfg, obs_mode: str) -> str:
    """Set env observation mode and keep observation_space in sync."""
    set_obs_mode = getattr(env_cfg, "set_obs_mode", None)
    if callable(set_obs_mode):
        set_obs_mode(obs_mode)
    else:
        setattr(env_cfg, "obs_mode", obs_mode)
        obs_dims = getattr(env_cfg, "obs_dims", None)
        if isinstance(obs_dims, dict) and obs_mode in obs_dims and hasattr(env_cfg, "observation_space"):
            setattr(env_cfg, "observation_space", int(obs_dims[obs_mode]))
    return obs_mode


def _apply_checkpoint_obs_mode(env_cfg, checkpoint_path: str) -> str | None:
    """Set env observation mode from the checkpoint's saved env.yaml metadata."""
    env_yaml_path = Path(checkpoint_path).resolve().parent.parent / "params" / "env.yaml"
    try:
        text = env_yaml_path.read_text()
    except OSError:
        return None

    match = re.search(r"(?m)^obs_mode:\s*([A-Za-z0-9_]+)\s*$", text)
    if match is None:
        return None

    return _set_env_obs_mode(env_cfg, match.group(1))


def _load_all_stages_weights_params(path: str) -> tuple[dict[int, dict[int, float]], dict[int, dict[int, dict[str, object]]]]:
    """Load Kuafu weights/parameters JSON.

    Accepts either the full best.json wrapper with a top-level `stages` key or a
    plain stages dict mapping stage ids to `weights`/`parameters`.
    """
    resolved_path = retrieve_file_path(path)
    with open(resolved_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {resolved_path}, got {type(data).__name__}")

    stages_data = data.get("stages") if isinstance(data.get("stages"), dict) else data
    if not isinstance(stages_data, dict):
        raise ValueError(f"Expected a stages dict in {resolved_path}")

    valid_stage_ids: list[int] = []
    for k, v in stages_data.items():
        if not isinstance(v, dict):
            continue
        try:
            valid_stage_ids.append(int(k))
        except (TypeError, ValueError):
            continue
    if not valid_stage_ids:
        raise ValueError(
            f"No valid stage entries found in {resolved_path}. "
            "Expected integer stage keys mapping to dicts with 'weights' and 'parameters'."
        )

    all_stages_weights: dict[int, dict[int, float]] = {}
    all_stages_parameters: dict[int, dict[int, dict[str, object]]] = {}
    for sid in sorted(valid_stage_ids):
        stage_data = stages_data[str(sid)]
        w_raw = stage_data.get("weights") or {}
        p_raw = stage_data.get("parameters") or {}
        if not isinstance(w_raw, dict) or not w_raw:
            raise ValueError(f"Stage {sid} in {resolved_path} has no valid 'weights' dict.")
        if not isinstance(p_raw, dict):
            raise ValueError(f"Stage {sid} in {resolved_path} has invalid 'parameters'.")
        all_stages_weights[sid] = {int(k): float(v) for k, v in dict(w_raw).items()}
        all_stages_parameters[sid] = {int(k): dict(v) for k, v in dict(p_raw).items()}
    return all_stages_weights, all_stages_parameters


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Play with RL-Games agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.mpc_config["num_envs"] = env_cfg.scene.num_envs
    env_cfg.mpc_config["use_per_env_weight"] = False
    if args_cli.viewer_resolution is not None:
        env_cfg.viewer.resolution = tuple(args_cli.viewer_resolution)
    if args_cli.viewer_eye is not None:
        env_cfg.viewer.eye = tuple(args_cli.viewer_eye)
    if args_cli.viewer_lookat is not None:
        env_cfg.viewer.lookat = tuple(args_cli.viewer_lookat)

    if "mpc" in agent_cfg:
        env_cfg.mpc_config["use_per_env_weight"]  = agent_cfg["mpc"]["use_per_env_weight"]
        env_cfg.mpc_config["use_absolute_progress_rewards"] = agent_cfg["mpc"]["use_absolute_progress_rewards"]
        env_cfg.mpc_config["use_relative_progress_rewards"] = agent_cfg["mpc"]["use_relative_progress_rewards"]
        env_cfg.mpc_config["use_velocity_penalty"] = agent_cfg["mpc"]["use_velocity_penalty"]
        env_cfg.mpc_config["regularize_velocity_scale"] = agent_cfg["mpc"]["regularize_velocity_scale"]
        env_cfg.mpc_config["regularize_time_scale"] = agent_cfg["mpc"]["regularize_time_scale"]
        env_cfg.mpc_config["progress_rewards_scale"] = agent_cfg["mpc"]["progress_rewards_scale"]

    # update agent device to match simulation device
    if args_cli.device is not None:
        agent_cfg["params"]["config"]["device"] = args_cli.device
        agent_cfg["params"]["config"]["device_name"] = args_cli.device

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    agent_cfg["params"]["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["params"]["seed"]
    # set the environment seed (after multi-gpu config for updated rank from agent seed)
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg["params"]["seed"]

    # specify directory for logging experiments
    log_root_path = os.path.join(str(external_artifact_root()), "standalone", "rl_games", agent_cfg["params"]["config"]["name"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # find checkpoint
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rl_games", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint is None:
        # specify directory for logging runs
        run_dir = agent_cfg["params"]["config"].get("full_experiment_name", ".*")
        # specify name of checkpoint
        if args_cli.use_last_checkpoint:
            checkpoint_file = ".*"
        else:
            # this loads the best checkpoint
            checkpoint_file = f"{agent_cfg['params']['config']['name']}.pth"
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, run_dir, checkpoint_file, other_dirs=["nn"])
    else:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir
    if args_cli.obs_mode is not None:
        selected_obs_mode = _set_env_obs_mode(env_cfg, args_cli.obs_mode)
        print(f"[INFO] Set env obs_mode from CLI override: {selected_obs_mode}")
    else:
        checkpoint_obs_mode = _apply_checkpoint_obs_mode(env_cfg, resume_path)
        if checkpoint_obs_mode is not None:
            print(f"[INFO] Set env obs_mode from checkpoint metadata: {checkpoint_obs_mode}")

    # wrap around environment for rl-games
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.weights_params_json:
        all_stages_weights, all_stages_parameters = _load_all_stages_weights_params(args_cli.weights_params_json)
        print(f"[INFO] Applying weights/parameters from: {args_cli.weights_params_json}")
        update_all_parameters(env.unwrapped, all_stages_parameters=all_stages_parameters)
        update_weights(
            env.unwrapped,
            stage_id=0,
            weights=None,
            all_stages_weights=all_stages_weights,
        )

    # wrap for video recording
    if args_cli.video:
        video_folder = os.path.join(log_root_path, log_dir, "videos", "play")
        video_kwargs = {
            "video_folder": video_folder,
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during play.")
        print(f"[INFO] Videos will be saved to: {os.path.abspath(video_folder)}")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rl-games
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups)

    # register the environment to rl-games registry
    # note: in agents configuration: environment name must be "rlgpu"
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    # load previously trained model
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path
    print(f"[INFO]: Loading model checkpoint from: {agent_cfg['params']['load_path']}")

    # set number of actors into agent config
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    # create runner from rl-games
    runner = Runner()
    runner.load(agent_cfg)
    # obtain the agent from the runner
    agent: BasePlayer = runner.create_player()
    agent.restore(resume_path)
    agent.reset()

    dt = env.unwrapped.step_dt
    raw_env = env.unwrapped

    # reset episode counters
    total_episodes = 0
    if hasattr(raw_env, "clear_end_to_end_rate_stats"):
        raw_env.clear_end_to_end_rate_stats()

    # reset environment
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    timestep = 0
    step_count = 0
    # required: enables the flag for batched observations
    _ = agent.get_batch_size(obs, 1)
    # initialize RNN states if used
    if agent.is_rnn:
        agent.init_rnn()
    # simulate environment
    # note: We simplified the logic in rl-games player.py (:func:`BasePlayer.run()`) function in an
    #   attempt to have complete control over environment stepping. However, this removes other
    #   operations such as masking that is used for multi-agent learning by RL-Games.
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # convert obs to agent format
            obs = agent.obs_to_torch(obs)
            # agent stepping
            actions = agent.get_action(obs, is_deterministic=agent.is_deterministic)
            # env stepping
            obs, _, dones, _ = env.step(actions)

            # perform operations for terminated episodes
            if len(dones) > 0:
                # count completed episodes
                if isinstance(dones, torch.Tensor):
                    n_done = int(dones.sum().item())
                else:
                    n_done = int(sum(dones))
                if n_done > 0:
                    total_episodes += n_done
                    successful = (
                        raw_env.get_true_success_count()
                        if hasattr(raw_env, "get_true_success_count")
                        else 0
                    )
                    print(
                        f"[PLAY] Episodes completed: {total_episodes} | "
                        f"Successful: {successful} | "
                        f"Success rate: {successful / total_episodes * 100:.1f}%"
                    )
                # reset rnn state for terminated episodes
                if agent.is_rnn and agent.states is not None:
                    for s in agent.states:
                        s[:, dones, :] = 0.0
        step_count += 1
        if args_cli.max_steps is not None and step_count >= args_cli.max_steps:
            break
        if args_cli.video:
            timestep += 1
            # exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # print final summary
    successful = (
        raw_env.get_true_success_count() if hasattr(raw_env, "get_true_success_count") else 0
    )
    print("\n" + "=" * 60)
    print(f"  PLAY SUMMARY")
    print(f"  Total episodes completed : {total_episodes}")
    print(f"  Successful terminations  : {successful}")
    rate_pct = successful / max(total_episodes, 1) * 100.0
    print(f"  Success rate             : {rate_pct:.1f}%")
    print("=" * 60 + "\n")
    # Machine-parseable line for sweep scripts (same idea as play_bc_env SUCCESS_RATE)
    print(f"RL_SUCCESS_RATE:{rate_pct:.4f}", flush=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
