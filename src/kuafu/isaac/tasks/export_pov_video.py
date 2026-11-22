# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RL-Games."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from kuafu.paths import external_artifact_root, resolve_artifact_path

from isaaclab.app import AppLauncher

# Import custom tasks to auto-register environments
import kuafu.isaac.tasks.open_drawer  # noqa: F401

parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from RL-Games.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=1200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--num_envs_to_record",
    type=int,
    default=9,
    help="Number of environments to record videos from (default: 9).",
)
parser.add_argument(
    "--camera_names",
    type=str,
    default="table_cam_0,table_cam_1,table_cam_2",
    help="Comma-separated camera sensor names to record from (must exist in env.unwrapped.scene.sensors). "
    "Example: 'table_cam_0,table_cam_1,table_cam_2'",
)
parser.add_argument(
    "--depth_clip_m",
    type=float,
    default=2.0,
    help="Clip depth (meters) when recording depth videos.",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=9, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="open-fr3-finray-drawer", help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rl_games_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--checkpoint", type=str, default=str(resolve_artifact_path("Residuals/open-fr3-finray-drawer/2026-02-20_02-18-19/checkpoints/best_global.pth", "RL checkpoint")), help="Path to model checkpoint.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
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
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.checkpoint:
    args_cli.checkpoint = str(resolve_artifact_path(args_cli.checkpoint, "RL checkpoint"))
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


import gymnasium as gym
import math
import os
import random
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
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import custom tasks to auto-register environments
import kuafu.isaac.tasks.open_drawer
import kuafu.isaac.tasks.stack_two
from kuafu.common.simulation_utils import (
    create_video_writers,
    release_video_writers,
    write_camera_frames_to_video,
)


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

    # Ensure camera sensors are created in the environment config when recording video.
    # (Learned envs only instantiate `table_cam_0/1/2` when cfg.enable_cameras is True.)
    if args_cli.video:
        try:
            if hasattr(env_cfg, "enable_cameras"):
                env_cfg.enable_cameras = True
            elif hasattr(env_cfg, "enable_camera"):
                env_cfg.enable_camera = True
        except (AttributeError, TypeError):
            # Some config types may not support assignment; let validation below catch missing cameras.
            pass

    # wrap around environment for rl-games
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # video recording (camera sensors only; no fallback)
    video_writers = None
    video_save_dir = None
    depth_clip_m = float(args_cli.depth_clip_m)
    camera_names: list[str] = []
    sun_program_executor_envs: list[int] = []
    record_env = env
    if args_cli.video:
        camera_names = [n.strip() for n in args_cli.camera_names.split(",") if n.strip()]
        if not camera_names:
            raise RuntimeError("--camera_names is empty after parsing.")

        if not hasattr(record_env.unwrapped, "scene") or not hasattr(record_env.unwrapped.scene, "sensors"):
            raise RuntimeError(
                "Video recording requires camera sensors at env.unwrapped.scene.sensors, but scene/sensors is missing."
            )

        available_sensors = list(record_env.unwrapped.scene.sensors.keys())
        missing = [n for n in camera_names if n not in record_env.unwrapped.scene.sensors]
        if missing:
            raise RuntimeError(
                "Requested camera sensor(s) not found (no fallback): "
                f"{missing}. Available sensors: {available_sensors}"
            )

        # Validate that requested sensors are image-producing cameras (not frame transformers, etc.)
        invalid: list[tuple[str, str]] = []
        for cam_name in camera_names:
            sensor = record_env.unwrapped.scene.sensors[cam_name]
            if not hasattr(sensor, "image_shape"):
                invalid.append((cam_name, type(sensor).__name__))
                continue
            if not hasattr(sensor, "data") or not hasattr(sensor.data, "output"):
                invalid.append((cam_name, type(sensor).__name__))
                continue
            output = sensor.data.output
            if "rgb" not in output or "distance_to_image_plane" not in output:
                invalid.append((cam_name, type(sensor).__name__))

        if invalid:
            invalid_str = ", ".join([f"{n}({t})" for n, t in invalid])
            raise RuntimeError(
                "Requested sensor(s) are not camera sensors with required outputs (no fallback): "
                f"{invalid_str}. Expected each to have `image_shape` and data.output keys "
                "['rgb', 'distance_to_image_plane']."
            )

        num_envs = int(getattr(record_env.unwrapped, "num_envs", 1))
        if int(args_cli.num_envs_to_record) < 0:
            n_rec = num_envs
        else:
            n_rec = min(int(args_cli.num_envs_to_record), num_envs)
        if n_rec <= 0:
            raise RuntimeError(f"Invalid number of envs to record: {n_rec}.")
        sun_program_executor_envs = list(range(n_rec))

        video_save_dir = os.path.join(log_dir, "videos", "play")
        video_writers, depth_clip_m = create_video_writers(
            record_env,
            env_cfg,
            video_save_dir,
            camera_names,
            sun_program_executor_envs,
            depth_clip_m=depth_clip_m,
        )

    # wrap around environment for rl-games
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)

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

    # reset environment
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    timestep = 0
    # required: enables the flag for batched observations
    _ = agent.get_batch_size(obs, 1)
    # initialize RNN states if used
    if agent.is_rnn:
        agent.init_rnn()
    # simulate environment
    # note: We simplified the logic in rl-games player.py (:func:`BasePlayer.run()`) function in an
    #   attempt to have complete control over environment stepping. However, this removes other
    #   operations such as masking that is used for multi-agent learning by RL-Games.
    try:
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
                    # reset rnn state for terminated episodes
                    if agent.is_rnn and agent.states is not None:
                        for s in agent.states:
                            s[:, dones, :] = 0.0

            if args_cli.video and video_writers is not None:
                # record one frame per sim step (after stepping, so sensors are updated)
                write_camera_frames_to_video(record_env, video_writers, camera_names, sun_program_executor_envs, depth_clip_m)
                timestep += 1
                if timestep >= args_cli.video_length:
                    break

            # time delay for real-time evaluation
            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        if args_cli.video and video_writers is not None and video_save_dir is not None:
            release_video_writers(video_writers, video_save_dir)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
