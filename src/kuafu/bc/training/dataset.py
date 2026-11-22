from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Literal

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from kuafu.bc.training.normalization import NORMALIZER_EPS


TimeShift = Literal[-2, -1, 0, 1, 2]
FROM_OBS_ACTION_MODE = "from_obs"


@dataclass(frozen=True)
class NormalizerState:
    mean: torch.Tensor
    std: torch.Tensor


class Normalizer:
    def __init__(self, mean: torch.Tensor, std: torch.Tensor, eps: float = NORMALIZER_EPS):
        self.mean = mean.detach().clone().float()
        self.std = std.detach().clone().float().clamp(min=eps)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std.to(x.device) + self.mean.to(x.device)

    def state_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std}

    @staticmethod
    def from_state_dict(d: dict) -> "Normalizer":
        return Normalizer(mean=d["mean"], std=d["std"])


def _resolve_default_paths(
    dataset_path: str,
    demo: str | None,
    obs_key: str,
    action_subkey: str | None = None,
) -> tuple[str, str]:
    """Resolve default obs/actions dataset paths for IsaacLab/robomimic-like layouts."""
    with h5py.File(dataset_path, "r") as f:
        if "data" not in f:
            raise KeyError("Expected group '/data' in HDF5.")
        demos = sorted(list(f["data"].keys()))
        if not demos:
            raise ValueError("No demos found under /data.")
        demo = demo or demos[0]
        if demo not in f["data"]:
            raise KeyError(f"Demo '{demo}' not found under /data. Available: {demos[:10]} ...")
        obs_path = f"data/{demo}/obs/{obs_key}"
        action_path = _action_path_for_subkey(demo, action_subkey)
        if obs_path not in f:
            raise KeyError(f"Obs dataset not found: '{obs_path}'")
        if action_path not in f:
            raise KeyError(f"Action dataset not found: '{action_path}'")
        return obs_path, action_path


def _action_path_for_subkey(demo: str, action_subkey: str | None) -> str:
    if action_subkey == FROM_OBS_ACTION_MODE:
        action_subkey = "absolute_ee"
    return f"data/{demo}/actions/{action_subkey}" if action_subkey else f"data/{demo}/actions"


def _infer_ee_pose_slices(obs_key: str, obs_dim: int) -> tuple[slice, slice]:
    if obs_key == "full_state":
        trailing_dims = 27 + 1 + 1
        if (obs_dim - trailing_dims) <= 0 or (obs_dim - trailing_dims) % 9 != 0:
            raise ValueError(
                f"Cannot infer fingertip pose from full_state obs_dim={obs_dim}. "
                "Expected layout: positions, rotations, joints, gripper_open, stage."
            )
        n_pose_blocks = (obs_dim - trailing_dims) // 9
        if n_pose_blocks < 2:
            raise ValueError(f"full_state obs_dim={obs_dim} does not contain fingertip+link4 pose blocks.")
        pos_start = 3 * (n_pose_blocks - 2)
        rot_base = 3 * n_pose_blocks
        rot_start = rot_base + 6 * (n_pose_blocks - 2)
    else:
        trailing_dims = 1 + 2 + 1
        if (obs_dim - trailing_dims) <= 0 or (obs_dim - trailing_dims) % 9 != 0:
            raise ValueError(
                f"Cannot infer fingertip pose from state obs_dim={obs_dim}. "
                "Expected layout: positions, rotations, gripper_open, gripper_effort, stage."
            )
        n_pose_blocks = (obs_dim - trailing_dims) // 9
        if n_pose_blocks < 1:
            raise ValueError(f"state obs_dim={obs_dim} does not contain a fingertip pose block.")
        pos_start = 3 * (n_pose_blocks - 1)
        rot_base = 3 * n_pose_blocks
        rot_start = rot_base + 6 * (n_pose_blocks - 1)

    return slice(pos_start, pos_start + 3), slice(rot_start, rot_start + 6)


def _actions_from_next_obs(obs_np: np.ndarray, absolute_ee_np: np.ndarray, obs_key: str) -> np.ndarray:
    if obs_np.ndim != 2 or absolute_ee_np.ndim != 2:
        raise ValueError(f"Expected rank-2 obs/actions, got obs{obs_np.shape} act{absolute_ee_np.shape}")
    if obs_np.shape[0] != absolute_ee_np.shape[0]:
        raise ValueError(
            "from_obs requires obs and absolute_ee to have matching timesteps, "
            f"got obs={obs_np.shape[0]} actions={absolute_ee_np.shape[0]}"
        )
    if absolute_ee_np.shape[1] != 11:
        raise ValueError(
            "from_obs expects absolute_ee actions with shape (*, 11) "
            f"([pos(3), rot6d(6), grip(2)]), got {absolute_ee_np.shape}"
        )

    if obs_np.shape[0] == 0:
        return absolute_ee_np.astype(np.float32, copy=True)

    pos_slice, rot_slice = _infer_ee_pose_slices(obs_key, obs_np.shape[1])
    derived = absolute_ee_np.astype(np.float32, copy=True)
    if obs_np.shape[0] == 1:
        return derived

    current_stage = np.rint(obs_np[:-1, -1]).astype(np.int64)
    next_stage = np.rint(obs_np[1:, -1]).astype(np.int64)
    same_stage = current_stage == next_stage
    if np.any(same_stage):
        next_pose = np.concatenate([obs_np[1:, pos_slice], obs_np[1:, rot_slice]], axis=1)
        derived[:-1, :9][same_stage] = next_pose[same_stage]
    return derived


def _demo_sort_key(name: str):
    m = re.fullmatch(r"demo_(\d+)", name)
    if m:
        return (0, int(m.group(1)))
    return (1, name)


def _list_demo_keys(f: h5py.File) -> list[str]:
    if "data" not in f:
        raise KeyError("Expected group '/data' in HDF5.")
    return sorted(list(f["data"].keys()), key=_demo_sort_key)


def _make_pairs(T_obs: int, T_act: int, time_shift: TimeShift) -> tuple[np.ndarray, np.ndarray]:
    """Pair obs[t] with act[t + time_shift], keeping only valid indices."""
    t = np.arange(T_obs, dtype=np.int64)
    a = t + int(time_shift)
    m = (a >= 0) & (a < T_act)
    return t[m], a[m]


def _discounted_returns(rewards: np.ndarray, dones: np.ndarray | None, gamma: float) -> np.ndarray:
    if rewards.ndim != 1:
        raise ValueError(f"Expected rewards to be rank-1, got {rewards.shape}")
    if dones is not None and dones.shape != rewards.shape:
        raise ValueError(f"Expected dones shape {rewards.shape}, got {dones.shape}")
    returns = np.zeros_like(rewards, dtype=np.float32)
    running = 0.0
    for idx in range(rewards.shape[0] - 1, -1, -1):
        terminal = bool(dones[idx]) if dones is not None else False
        if terminal:
            running = 0.0
        running = float(rewards[idx]) + float(gamma) * running
        returns[idx] = running
    return returns


class Hdf5TrajectoryDataset(Dataset):
    """In-memory (obs, action) dataset from HDF5 demos. """

    def __init__(
        self,
        dataset_path: str,
        split: Literal["train", "val"] = "train",
        val_fraction: float = 0.1,
        seed: int = 42,
        demo: str | None = None,
        obs_key: str = "full_state",
        action_subkey: str | None = None,
        time_shift: TimeShift = 0,
        index_mode: Literal["contiguous", "random"] = "contiguous",
        load_returns: bool = False,
        discount: float = 0.99,
    ):
        super().__init__()

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(dataset_path)
        if not (0.0 <= val_fraction < 1.0):
            raise ValueError("val_fraction must be in [0, 1).")
        if split not in ("train", "val"):
            raise ValueError("split must be 'train' or 'val'.")

        self.dataset_path = dataset_path
        self.demo = demo
        self.obs_key = obs_key
        self.time_shift = int(time_shift)

        _resolve_default_paths(
            dataset_path, demo=demo, obs_key=obs_key, action_subkey=action_subkey
        )
        self.obs_path = f"data/<demo>/obs/{obs_key}"

        with h5py.File(dataset_path, "r") as f:
            all_demo_keys = _list_demo_keys(f)
            if not all_demo_keys:
                raise ValueError("No demos found under /data.")

            if demo is not None:
                if demo not in f["data"]:
                    raise KeyError(f"Demo '{demo}' not found under /data. Available: {all_demo_keys[:10]} ...")
                selected_demo_keys = [demo]
            else:
                selected_demo_keys = all_demo_keys

            # split by whole demos
            if val_fraction == 0.0 or len(selected_demo_keys) <= 1:
                train_demo_keys = selected_demo_keys
                val_demo_keys: list[str] = []
            else:
                n_val = int(round(len(selected_demo_keys) * val_fraction))
                n_val = min(max(n_val, 1), len(selected_demo_keys) - 1)
                n_train = len(selected_demo_keys) - n_val
                if index_mode == "contiguous":
                    train_demo_keys = selected_demo_keys[:n_train]
                    val_demo_keys = selected_demo_keys[n_train:]
                elif index_mode == "random":
                    rng = np.random.default_rng(seed)
                    perm = rng.permutation(len(selected_demo_keys))
                    train_idx = perm[:n_train]
                    val_idx = perm[n_train:]
                    train_demo_keys = [selected_demo_keys[i] for i in train_idx]
                    val_demo_keys = [selected_demo_keys[i] for i in val_idx]
                else:
                    raise ValueError(f"Unknown index_mode: {index_mode}")

            chosen_demo_keys = train_demo_keys if split == "train" else val_demo_keys

            # Infer dimensions from the first selected demo, even if chosen split is empty.
            ref_demo = selected_demo_keys[0]
            ref_obs_path = f"data/{ref_demo}/obs/{obs_key}"
            ref_action_path = _action_path_for_subkey(ref_demo, action_subkey)
            ref_obs = np.array(f[ref_obs_path], dtype=np.float32)
            ref_act = np.array(f[ref_action_path], dtype=np.float32)
            if action_subkey == FROM_OBS_ACTION_MODE:
                ref_act = _actions_from_next_obs(ref_obs, ref_act, obs_key)
            if ref_obs.ndim != 2 or ref_act.ndim != 2:
                raise ValueError(f"Expected obs/actions to be rank-2 arrays. Got obs{ref_obs.shape} act{ref_act.shape}")
            obs_dim = ref_obs.shape[1]
            act_dim = ref_act.shape[1]

            obs_chunks: list[np.ndarray] = []
            act_chunks: list[np.ndarray] = []
            ret_chunks: list[np.ndarray] = []
            total_pairs = 0
            total_obs_steps = 0
            total_act_steps = 0

            for dk in chosen_demo_keys:
                current_obs_path = f"data/{dk}/obs/{obs_key}"
                current_action_path = _action_path_for_subkey(dk, action_subkey)
                obs_np = np.array(f[current_obs_path], dtype=np.float32)
                act_np = np.array(f[current_action_path], dtype=np.float32)
                if action_subkey == FROM_OBS_ACTION_MODE:
                    act_np = _actions_from_next_obs(obs_np, act_np, obs_key)
                if obs_np.ndim != 2 or act_np.ndim != 2:
                    raise ValueError(f"Expected obs/actions to be rank-2 arrays. Got obs{obs_np.shape} act{act_np.shape}")
                if obs_np.shape[1] != obs_dim or act_np.shape[1] != act_dim:
                    raise ValueError(
                        f"Inconsistent dims in demo '{dk}': obs {obs_np.shape[1]} vs {obs_dim}, "
                        f"act {act_np.shape[1]} vs {act_dim}"
                    )

                T_obs, T_act = obs_np.shape[0], act_np.shape[0]
                obs_idx, act_idx = _make_pairs(T_obs, T_act, time_shift=time_shift)
                if len(obs_idx) == 0:
                    continue

                obs_chunks.append(obs_np[obs_idx])
                act_chunks.append(act_np[act_idx])
                if load_returns:
                    reward_path = f"data/{dk}/rewards"
                    if reward_path not in f:
                        raise KeyError(
                            f"Dataset is missing '{reward_path}', but load_returns=True. "
                            "Recorder exports must include per-step rewards."
                        )
                    rewards_np = np.array(f[reward_path], dtype=np.float32)
                    if rewards_np.ndim != 1:
                        raise ValueError(f"Expected rewards to be rank-1, got {rewards_np.shape}")
                    if rewards_np.shape[0] != T_obs:
                        raise ValueError(
                            f"Reward length mismatch in demo '{dk}': rewards={rewards_np.shape[0]} obs={T_obs}"
                        )
                    done_path = f"data/{dk}/dones"
                    dones_np = np.array(f[done_path], dtype=np.bool_) if done_path in f else None
                    returns_np = _discounted_returns(rewards_np, dones_np, discount)
                    ret_chunks.append(returns_np[obs_idx])
                total_pairs += len(obs_idx)
                total_obs_steps += T_obs
                total_act_steps += T_act

        if obs_chunks:
            self.obs = torch.from_numpy(np.concatenate(obs_chunks, axis=0))
            self.actions = torch.from_numpy(np.concatenate(act_chunks, axis=0))
            self.returns = (
                torch.from_numpy(np.concatenate(ret_chunks, axis=0)).unsqueeze(-1) if load_returns else None
            )
        else:
            self.obs = torch.empty((0, obs_dim), dtype=torch.float32)
            self.actions = torch.empty((0, act_dim), dtype=torch.float32)
            self.returns = torch.empty((0, 1), dtype=torch.float32) if load_returns else None
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_pairs_total = total_pairs
        self.num_pairs_split = len(self.obs)
        self.T_obs = total_obs_steps
        self.T_act = total_act_steps
        self.all_demo_keys = selected_demo_keys
        self.train_demo_keys = train_demo_keys
        self.val_demo_keys = val_demo_keys
        self.demo_keys = chosen_demo_keys
        self.num_demos_total = len(selected_demo_keys)
        self.num_demos_split = len(chosen_demo_keys)
        self.discount = float(discount)
        self.load_returns = bool(load_returns)

    def __len__(self) -> int:
        return self.obs.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        if self.returns is not None:
            return self.obs[idx], self.actions[idx], self.returns[idx]
        return self.obs[idx], self.actions[idx]


def fit_normalizer(x: torch.Tensor) -> Normalizer:
    x = x.float()
    return Normalizer(mean=x.mean(dim=0), std=x.std(dim=0))
