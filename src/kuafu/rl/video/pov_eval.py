"""In-process RL-Games checkpoint playback + multi-env video recording.

This is Kuafu's in-process POV video exporter:
- no top-level AppLauncher side effects
- callable from a subprocess (spawn) started by Kuafu
"""

from __future__ import annotations

import os
import math
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PovEvalConfig:
    task: str
    checkpoint_path: str
    output_dir: str
    video_length: int = 200
    num_envs_to_record: int = 9
    device: str = "cuda:0"
    headless: bool = True
    seed: int = 0
    camera_names: tuple[str, str, str] = ("table_cam_0", "table_cam_1", "table_cam_2")
    execution_robot_id: str | None = None
    all_stages_weights: dict[int, dict[int, float]] | None = None
    all_stages_parameters: dict[int, dict[int, dict[str, object]]] | None = None


def _infer_checkpoint_obs_dim(checkpoint_path: Path) -> int | None:
    import torch

    # These are trusted local training checkpoints; PyTorch 2.6 defaults to
    # weights_only=True, which can reject older checkpoint metadata such as
    # numpy scalars before we even reach the model state dict.
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = ckpt.get("model") if isinstance(ckpt, dict) else None
    if not isinstance(model, dict):
        return None

    for key in (
        "running_mean_std.running_mean",
        "running_mean_std.running_var",
        "a2c_network.actor_mlp.0.weight",
    ):
        value = model.get(key)
        if value is None:
            continue
        if key.endswith(".weight") and getattr(value, "ndim", 0) >= 2:
            return int(value.shape[1])
        if getattr(value, "ndim", 0) >= 1:
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


def _try_enable_cameras_on_cfg(env_cfg: Any) -> None:
    # Best-effort: different env cfgs use different flags.
    for k in ("enable_cameras", "enable_camera"):
        if hasattr(env_cfg, k):
            try:
                setattr(env_cfg, k, True)
            except Exception as exc:
                raise RuntimeError(f"[residual_rl][video] Failed to enable camera flag {k!r}.") from exc


class _FFmpegWriter:
    """Stream raw BGR frames to ffmpeg -> H.264 MP4 (faststart)."""

    def __init__(self, path: str, fps: int, width: int, height: int):
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("[residual_rl] ffmpeg not found on PATH; required for video export.")
        self.path = path
        self._proc = subprocess.Popen(
            [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s:v",
                f"{width}x{height}",
                "-r",
                str(int(fps)),
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "18",
                "-preset",
                "veryfast",
                "-movflags",
                "+faststart",
                path,
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def write(self, frame_bgr: Any) -> None:
        if self._proc.stdin is None:
            raise RuntimeError("[residual_rl] ffmpeg stdin is closed.")
        self._proc.stdin.write(frame_bgr.tobytes())

    def close(self) -> None:
        if self._proc.stdin is not None:
            try:
                self._proc.stdin.close()
            except Exception as exc:
                raise RuntimeError("[residual_rl][video] Failed to close ffmpeg stdin.") from exc
        rc = self._proc.wait()
        if rc != 0:
            err = b""
            if self._proc.stderr is not None:
                try:
                    err = self._proc.stderr.read()
                except Exception:
                    err = b""
            msg = err.decode("utf-8", errors="replace")[-2000:]
            raise RuntimeError(f"[residual_rl] ffmpeg failed (exit={rc}) while writing {self.path}:\n{msg}")


def _tile_rgb_grid(rgb_nhwc: Any, *, env_ids: list[int], grid: int) -> Any:
    """Tile N env RGB frames (H,W,3) into a (grid*H, grid*W, 3) image, padding missing with black."""
    import numpy as np

    if not isinstance(rgb_nhwc, np.ndarray) or rgb_nhwc.ndim != 4 or rgb_nhwc.shape[-1] != 3:
        raise ValueError(f"[residual_rl] Expected rgb (N,H,W,3) ndarray, got {type(rgb_nhwc)} shape={getattr(rgb_nhwc,'shape',None)}")
    H, W = int(rgb_nhwc.shape[1]), int(rgb_nhwc.shape[2])
    canvas = np.zeros((grid * H, grid * W, 3), dtype=np.uint8)
    for k, eid in enumerate(env_ids):
        if eid < 0 or eid >= int(rgb_nhwc.shape[0]):
            continue
        r = k // grid
        c = k % grid
        if r >= grid:
            break
        canvas[r * H : (r + 1) * H, c * W : (c + 1) * W, :] = rgb_nhwc[eid]
    return canvas


def export_pov_videos(cfg: PovEvalConfig, *, _close_app: bool = True) -> list[str]:
    """Play a checkpoint and save 3 camera grid mp4s. Returns list of mp4 paths."""
    # Isaac Sim must be launched first.
    from isaaclab.app import AppLauncher

    print(
        "[residual_rl][video] export_pov_videos start: "
        f"task={cfg.task} checkpoint={cfg.checkpoint_path} output_dir={cfg.output_dir} "
        f"device={cfg.device} headless={cfg.headless} "
        f"video_length={cfg.video_length} num_envs_to_record={cfg.num_envs_to_record} "
        f"execution_robot_id={cfg.execution_robot_id}",
        flush=True,
    )
    app_launcher = AppLauncher(headless=bool(cfg.headless), device=str(cfg.device), enable_cameras=True)
    simulation_app = app_launcher.app

    try:
        import gymnasium as gym
        import random
        import torch

        from kuafu.rl.isaac.task_registry import register_task_modules

        resolved_robot_id = register_task_modules(
            task=cfg.task,
            execution_robot_id=cfg.execution_robot_id,
        )
        print(
            f"[residual_rl][video] Registered task modules for execution_robot_id={resolved_robot_id}.",
            flush=True,
        )

        from isaaclab_tasks.utils import parse_env_cfg
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
        from kuafu.rl.isaac.update_utilities import update_all_parameters, update_weights

        from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
        from rl_games.common import env_configurations, vecenv
        from rl_games.torch_runner import Runner
        from rl_games.common.player import BasePlayer

        # Env cfg
        env_cfg = parse_env_cfg(cfg.task)
        env_cfg.sim.device = str(cfg.device)
        env_cfg.seed = int(cfg.seed if cfg.seed is not None else random.randint(0, 10000))
        resolved_obs_mode = _maybe_set_obs_mode_from_checkpoint(env_cfg, Path(cfg.checkpoint_path))
        env_cfg.scene.num_envs = int(max(1, int(cfg.num_envs_to_record)))
        _try_enable_cameras_on_cfg(env_cfg)
        print(
            "[residual_rl][video] env cfg prepared: "
            f"sim_device={env_cfg.sim.device} seed={env_cfg.seed} num_envs={env_cfg.scene.num_envs} "
            f"resolved_obs_mode={resolved_obs_mode}",
            flush=True,
        )

        output_dir = Path(cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[residual_rl][video] output dir ready: {output_dir}", flush=True)

        print(f"[residual_rl][video] creating gym env for task={cfg.task}", flush=True)
        record_env = gym.make(cfg.task, cfg=env_cfg, render_mode=None)
        env_u = record_env.unwrapped
        if cfg.all_stages_parameters:
            print(
                f"[residual_rl][video] applying all-stage parameters for playback: stages={list(cfg.all_stages_parameters.keys())}",
                flush=True,
            )
            update_all_parameters(env_u, all_stages_parameters=cfg.all_stages_parameters)
        if cfg.all_stages_weights:
            print(
                f"[residual_rl][video] applying all-stage weights for playback: stages={list(cfg.all_stages_weights.keys())}",
                flush=True,
            )
            update_weights(
                env_u,
                stage_id=0,
                weights=None,
                all_stages_weights=cfg.all_stages_weights,
            )
        print("[residual_rl][video] gym env created; resetting once", flush=True)
        record_env.reset()
        print("[residual_rl][video] initial env reset complete", flush=True)

        # Validate cameras exist and are image-producing.
        cam_names = tuple(str(x) for x in cfg.camera_names)
        if not hasattr(record_env.unwrapped, "scene") or not hasattr(record_env.unwrapped.scene, "sensors"):
            raise RuntimeError("[residual_rl] env.unwrapped.scene.sensors missing; cannot record camera videos.")
        available = list(record_env.unwrapped.scene.sensors.keys())
        print(f"[residual_rl][video] available sensors: {available}", flush=True)
        missing = [n for n in cam_names if n not in record_env.unwrapped.scene.sensors]
        if missing:
            raise RuntimeError(
                f"[residual_rl] Requested cameras not found (no fallback): {missing}. Available sensors: {available}"
            )
        for n in cam_names:
            s = record_env.unwrapped.scene.sensors[n]
            print(f"[residual_rl][video] validating camera: {n} type={type(s).__name__}", flush=True)
            if not hasattr(s, "image_shape") or not hasattr(s, "data") or not hasattr(s.data, "output"):
                raise RuntimeError(f"[residual_rl] Sensor '{n}' is not an image camera (got {type(s).__name__}).")
            if "rgb" not in s.data.output:
                raise RuntimeError(f"[residual_rl] Camera '{n}' missing rgb output.")

        num_envs = int(getattr(record_env.unwrapped, "num_envs", env_cfg.scene.num_envs))
        env_ids = list(range(int(min(int(cfg.num_envs_to_record), num_envs))))
        if not env_ids:
            raise RuntimeError("[residual_rl] No envs selected for recording.")
        grid = int(math.ceil(math.sqrt(len(env_ids))))
        print(
            "[residual_rl][video] recording layout prepared: "
            f"cam_names={cam_names} num_envs={num_envs} env_ids={env_ids} grid={grid}",
            flush=True,
        )

        # Determine fps from env_cfg (matches learned env logic: step_dt = sim.dt * decimation).
        sim_dt = float(getattr(getattr(env_cfg, "sim", None), "dt", 1.0 / 60.0))
        decimation = int(getattr(env_cfg, "decimation", getattr(getattr(env_cfg, "sim", None), "decimation", 1)))
        step_dt = sim_dt * max(decimation, 1)
        fps = max(1, int(round(1.0 / step_dt))) if step_dt > 0.0 else 30

        # Use camera 0 to get resolution.
        cam0 = record_env.unwrapped.scene.sensors[cam_names[0]]
        cam_h, cam_w = cam0.image_shape
        out_h, out_w = int(grid * cam_h), int(grid * cam_w)
        print(
            "[residual_rl][video] camera geometry ready: "
            f"base_resolution={cam_w}x{cam_h} tiled_resolution={out_w}x{out_h} fps={fps}",
            flush=True,
        )

        writers: dict[str, _FFmpegWriter] = {}
        out_paths: dict[str, str] = {}
        for cam in cam_names:
            p = str(output_dir / f"{cam}_grid_{grid}x{grid}.mp4")
            out_paths[cam] = p
            writers[cam] = _FFmpegWriter(p, fps=fps, width=out_w, height=out_h)
            print(f"[residual_rl][video] ffmpeg writer ready: camera={cam} path={p}", flush=True)

        # RL-Games agent cfg (playback)
        agent_cfg = load_cfg_from_registry(cfg.task, "rl_games_cfg_entry_point")
        agent_cfg["params"]["config"]["device"] = str(cfg.device)
        agent_cfg["params"]["config"]["device_name"] = str(cfg.device)
        agent_cfg["params"]["load_checkpoint"] = True
        agent_cfg["params"]["load_path"] = str(cfg.checkpoint_path)
        print("[residual_rl][video] RL-Games config prepared", flush=True)

        clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
        clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
        rl_env = RlGamesVecEnvWrapper(record_env, str(cfg.device), clip_obs, clip_actions)
        print("[residual_rl][video] RL env wrapper created", flush=True)

        # Register env (idempotent-ish per process).
        try:
            vecenv.register(
                "IsaacRlgWrapper",
                lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
            )
        except Exception as exc:
            print(
                "[residual_rl][video] WARNING: vecenv.register('IsaacRlgWrapper') failed; "
                f"assuming it is already registered. Error: {type(exc).__name__}: {exc}",
                flush=True,
            )
        try:
            env_configurations.register(
                "rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: rl_env}
            )
        except Exception as exc:
            print(
                "[residual_rl][video] WARNING: env_configurations.register('rlgpu') failed; "
                f"overwriting configuration directly. Error: {type(exc).__name__}: {exc}",
                flush=True,
            )
            try:
                env_configurations.configurations["rlgpu"] = {
                    "vecenv_type": "IsaacRlgWrapper",
                    "env_creator": lambda **kwargs: rl_env,
                }
            except Exception as inner_exc:
                raise RuntimeError("[residual_rl][video] Failed to register or overwrite RL-Games env configuration 'rlgpu'.") from inner_exc

        agent_cfg["params"]["config"]["num_actors"] = rl_env.unwrapped.num_envs
        runner = Runner()
        print("[residual_rl][video] loading RL-Games runner", flush=True)
        runner.load(agent_cfg)
        print("[residual_rl][video] creating player", flush=True)
        agent: BasePlayer = runner.create_player()
        print(f"[residual_rl][video] restoring checkpoint: {cfg.checkpoint_path}", flush=True)
        agent.restore(str(cfg.checkpoint_path))
        agent.reset()
        print("[residual_rl][video] player restore/reset complete", flush=True)

        obs = rl_env.reset()
        if isinstance(obs, dict):
            obs = obs.get("obs", obs)
        print("[residual_rl][video] first rl_env reset complete", flush=True)

        # Step for fixed horizon (video recorder wrapper saves when it hits video_length).
        if agent.is_rnn:
            agent.init_rnn()
        _ = agent.get_batch_size(obs, 1)
        print("[residual_rl][video] entering rollout loop", flush=True)

        timestep = 0
        try:
            while simulation_app.is_running() and timestep < int(cfg.video_length):
                with torch.inference_mode():
                    obs_t = agent.obs_to_torch(obs)
                    actions = agent.get_action(obs_t, is_deterministic=True)
                    obs, _, dones, _ = rl_env.step(actions)
                    if len(dones) > 0 and agent.is_rnn and agent.states is not None:
                        for s in agent.states:
                            s[:, dones, :] = 0.0

                # After stepping, sensors are updated -> capture grids.
                for cam in cam_names:
                    sensor = record_env.unwrapped.scene.sensors[cam]
                    rgb = sensor.data.output["rgb"]
                    if isinstance(rgb, torch.Tensor):
                        rgb = rgb.detach().cpu().numpy()
                    import numpy as np
                    if rgb.dtype != np.uint8:
                        rgb = (rgb * 255).astype(np.uint8) if float(rgb.max()) <= 1.0 else rgb.astype(np.uint8)
                    grid_rgb = _tile_rgb_grid(rgb, env_ids=env_ids, grid=grid)
                    grid_bgr = np.ascontiguousarray(grid_rgb[..., ::-1])
                    writers[cam].write(grid_bgr)

                if timestep == 0:
                    print("[residual_rl][video] first rollout step and frame capture complete", flush=True)
                timestep += 1
        finally:
            # finalize mp4s (write moov atom, etc.)
            print(f"[residual_rl][video] finalizing writers after {timestep} steps", flush=True)
            for w in writers.values():
                try:
                    w.close()
                except Exception as exc:
                    raise RuntimeError("[residual_rl][video] Failed to finalize ffmpeg writer.") from exc
            rl_env.close()
            print("[residual_rl][video] RL env closed", flush=True)

        print(f"[residual_rl][video] export_pov_videos success: outputs={list(out_paths.values())}", flush=True)
        return [out_paths[c] for c in cam_names]
    finally:
        if _close_app:
            try:
                print("[residual_rl][video] closing simulation app", flush=True)
                simulation_app.close()
            except BaseException as exc:
                print(
                    f"[residual_rl][video] WARNING: failed to close simulation app: {type(exc).__name__}: {exc}",
                    flush=True,
                )
