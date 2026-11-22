"""Record aligned state transitions from learned environments without cameras."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np


def _array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value).copy()


class StateTransitionRecorder:
    """Record aligned transitions, retaining successful episodes by default.

    The environment resets before its post-step hook. Its terminal transition
    therefore has no trustworthy next observation here and is omitted entirely.
    Success is taken from the task's cached physical outcome before that reset.
    """

    def __init__(self, env, path, target, metadata=None, *, record_failures=False, record_scene=False):
        if target < 1:
            raise ValueError("target must be positive")
        self.env = env
        self.target = int(target)
        self.count = 0
        self.attempts = 0
        self.record_failures = record_failures
        self.record_scene = record_scene
        self.exported_successful_episode_count = 0
        self.exported_failed_episode_count = 0
        self.cfg = SimpleNamespace(export_in_record_pre_reset=True)
        self.pending = [None] * env.num_envs
        self.episodes = [[] for _ in range(env.num_envs)]
        self.initial_states = [None] * env.num_envs
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = h5py.File(path, "x")
        self.data = self.file.create_group("data")
        self.data.attrs["total"] = 0
        metadata = dict(metadata or {})
        joint_names = getattr(env, "_actuated_joint_names", None)
        if joint_names is not None:
            metadata["robot_state_joint_names"] = list(joint_names)
            metadata["robot_state_dim"] = len(joint_names)
            metadata["robot_state_representation"] = "joint_position"
        self.data.attrs["env_args"] = json.dumps(metadata)
        self.data.attrs["terminal_transition_omitted"] = True

    def _scene_arrays(self):
        values = {}

        def flatten(prefix, value):
            if isinstance(value, dict):
                for key, child in value.items():
                    flatten(f"{prefix}/{key}" if prefix else key, child)
            else:
                values[prefix] = _array(value)

        flatten("", self.env.scene.get_state(is_relative=True))
        return values

    def record_pre_step(self):
        observations = {f"obs/{key}": _array(self.env.obs_buf[key])
                        for key in ("policy", "full_state", "robot_state") if key in self.env.obs_buf}
        if "obs/full_state" not in observations and "obs/robot_state" not in observations:
            raise ValueError("State collection requires full_state or robot_state observations")
        if "obs/policy" in observations:
            observations["obs/state"] = observations.pop("obs/policy")
        values = dict(observations)
        for key, attr in (("relative_joint", "_logged_relative_joint"),
                          ("absolute_joint", "_logged_joint_targets"),
                          ("relative_ee", "_logged_relative_ee"),
                          ("absolute_ee", "_logged_absolute_ee")):
            value = getattr(self.env, attr, None)
            if value is not None:
                values[f"actions/{key}"] = _array(value)
        if "actions/relative_joint" not in values:
            raise ValueError("State collection requires applied relative-joint commands")
        self.pending = [{key: value[index] for key, value in values.items()}
                        for index in range(self.env.num_envs)]
        if self.record_scene:
            scene = self._scene_arrays()
            for index in range(self.env.num_envs):
                if self.initial_states[index] is None:
                    self.initial_states[index] = {key: value[index] for key, value in scene.items()}

    def record_post_step(self):
        following = {f"next_obs/{'state' if key == 'policy' else key}": _array(self.env.obs_buf[key])
                     for key in ("policy", "full_state", "robot_state") if key in self.env.obs_buf}
        rewards = _array(self.env.reward_buf)
        if self.record_scene:
            following.update({f"states/{key}": value for key, value in self._scene_arrays().items()})
        for index, before in enumerate(self.pending):
            if before is None:
                continue
            row = {**before, **{key: value[index] for key, value in following.items()},
                   "rewards": rewards[index], "dones": np.asarray(False)}
            if any(not np.isfinite(value).all() for value in row.values()):
                raise FloatingPointError("Nonfinite state transition")
            self.episodes[index].append(row)
        self.pending = [None] * self.env.num_envs

    def record_pre_reset(self, env_ids):
        indices = range(self.env.num_envs) if env_ids is None else _array(env_ids).reshape(-1).tolist()
        successes = _array(self.env._cached_success).reshape(-1)
        for index in indices:
            rows = self.episodes[index]
            # pending[index] is the action whose result will be replaced by reset.
            self.pending[index] = None
            if rows and self.cfg.export_in_record_pre_reset:
                self.attempts += 1
            if (rows and self.cfg.export_in_record_pre_reset and self.count < self.target
                    and (successes[index] or self.record_failures)):
                group = self.data.create_group(f"demo_{self.count}")
                group.attrs["success"] = bool(successes[index])
                group.attrs["num_samples"] = len(rows)
                group.attrs["terminal_transition_omitted"] = True
                for key in rows[0]:
                    group.create_dataset(key, data=np.stack([row[key] for row in rows]),
                                         compression="gzip", compression_opts=1)
                if self.record_scene:
                    for key, value in self.initial_states[index].items():
                        group.create_dataset(f"initial_state/{key}", data=value[None],
                                             compression="gzip", compression_opts=1)
                self.count += 1
                if successes[index]:
                    self.exported_successful_episode_count += 1
                else:
                    self.exported_failed_episode_count += 1
                self.data.attrs["total"] += len(rows)
                self.file.flush()
                print(f"[STATE COLLECT] Saved {self.count} / {self.target}; attempts={self.attempts}", flush=True)
            self.episodes[index] = []
            self.initial_states[index] = None

    def record_post_reset(self, env_ids):
        pass

    def close(self):
        self.file.close()


def collect_state_episodes(raw_env, wrapped_env, agent, simulation_app, path, target, obs,
                           *, deterministic=True, metadata=None, recorder_class=StateTransitionRecorder):
    """Run an already-loaded checkpoint through the native action/recording hooks."""
    import torch

    recorder = recorder_class(raw_env, path, target, metadata)
    previous = raw_env.recorder_manager
    if previous is not None:
        recorder.close()
        raise ValueError("State collection requires an environment without another recorder")
    raw_env.recorder_manager = recorder
    try:
        while simulation_app.is_running() and recorder.count < target:
            with torch.no_grad():
                tensor_obs = agent.obs_to_torch(obs)
                action = agent.get_action(tensor_obs, is_deterministic=deterministic)
                obs, _, dones, _ = wrapped_env.step(action)
                if isinstance(obs, dict):
                    obs = obs["obs"]
                if getattr(agent, "is_rnn", False) and agent.states is not None:
                    for state in agent.states:
                        state[:, dones, :] = 0.0
        if recorder.count != target:
            raise RuntimeError(f"Simulation stopped after {recorder.count}/{target} demonstrations")
        return {"demonstrations": recorder.count, "attempts": recorder.attempts, "path": str(path)}
    finally:
        raw_env.recorder_manager = previous
        recorder.close()
