"""Episode-safe NumPy sequence batches for recurrent world models."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


@dataclass
class SequenceEpisode:
    demonstration: str
    state: np.ndarray
    prev_action: np.ndarray
    reward: np.ndarray
    reward_mask: np.ndarray


def symlog(value):
    """Signed logarithm, preserving zero and the sign of continuous commands."""
    return np.sign(value) * np.log1p(np.abs(value))


class SequenceDataset:
    """Read explicit successors and pair each state with the action leading to it."""

    def __init__(self, path, *, obs_key="full_state", action_key="relative_joint",
                 exclude_last=1, skip_first=1, seed=42, val_fraction=0.1, test_fraction=0.1):
        if exclude_last < 0 or skip_first < 0:
            raise ValueError("exclude_last and skip_first must be nonnegative")
        if not (0 < val_fraction < 1 and 0 < test_fraction < 1 and val_fraction + test_fraction < 1):
            raise ValueError("Validation/test fractions must be positive and sum to less than one")
        path = Path(path).expanduser().resolve()
        self.episodes = []
        dimensions = None
        total, used, terminal_rows = 0, 0, 0
        action_path = f"actions/{action_key}" if action_key else "actions"
        with h5py.File(path, "r") as source:
            if "data" not in source or not source["data"]:
                raise ValueError("Expected nonempty /data/demo_* HDF5 groups")
            env_args = source["data"].attrs.get("env_args", "{}")
            env_args = json.loads(env_args.decode() if isinstance(env_args, bytes) else env_args)
            for name in sorted(source["data"]):
                group = source[f"data/{name}"]
                arrays = []
                for key in (f"obs/{obs_key}", action_path, f"next_obs/{obs_key}"):
                    if key not in group or not isinstance(group[key], h5py.Dataset):
                        raise ValueError(f"{name}: expected dataset {key}")
                    value = np.asarray(group[key], dtype=np.float32)
                    if value.ndim != 2 or not np.isfinite(value).all():
                        raise ValueError(f"{name}/{key}: expected finite rank-2 data")
                    arrays.append(value)
                state, action, successor = arrays
                count, raw_dim = state.shape
                if state.shape != successor.shape or len(action) != count:
                    raise ValueError(f"{name}: inconsistent transition lengths or dimensions")
                if raw_dim <= exclude_last or action.shape[1] == 0:
                    raise ValueError(f"{name}: empty state/action dimensions")
                dims = (raw_dim, action.shape[1])
                if dimensions is not None and dims != dimensions:
                    raise ValueError(f"{name}: inconsistent dimensions between demonstrations")
                dimensions = dims
                done = np.asarray(group["dones"]).reshape(-1) if "dones" in group else np.zeros(count)
                reward = np.asarray(group["rewards"], dtype=np.float32).reshape(-1) if "rewards" in group else np.zeros(count, np.float32)
                if len(done) != count or len(reward) != count or not np.isfinite(done).all() or not np.isfinite(reward).all():
                    raise ValueError(f"{name}: invalid rewards/dones")
                done = done.astype(bool)
                state, successor = state[:, :raw_dim - exclude_last], successor[:, :raw_dim - exclude_last]
                if count > 1 and not np.allclose(successor[:-1][~done[:-1]], state[1:][~done[:-1]], rtol=1e-5, atol=1e-6):
                    raise ValueError(f"{name}: next_obs[t] does not match obs[t+1]; audit recorder alignment")
                keep = ~done
                keep[:skip_first] = False
                starts = np.flatnonzero(keep & ~np.r_[False, keep[:-1]])
                stops = np.flatnonzero(keep & ~np.r_[keep[1:], False]) + 1
                for start, stop in zip(starts, stops):
                    # State s[t+1] is paired with a[t] and r[t], never a[t+1].
                    states = np.concatenate((state[start:stop], successor[stop - 1:stop]))
                    actions = np.concatenate((np.zeros_like(action[:1]), action[start:stop]))
                    rewards = np.r_[np.float32(0), reward[start:stop]].astype(np.float32)
                    mask = np.ones(len(states), np.float32) if "rewards" in group else np.zeros(len(states), np.float32)
                    mask[0] = 0
                    self.episodes.append(SequenceEpisode(name, states, actions, rewards, mask))
                total += count
                used += int(keep.sum())
                terminal_rows += int(done.sum())
        names = sorted({episode.demonstration for episode in self.episodes})
        names = np.random.default_rng(seed).permutation(names).tolist()
        nval, ntest = max(1, round(len(names) * val_fraction)), max(1, round(len(names) * test_fraction))
        if nval + ntest >= len(names):
            raise ValueError("Need at least three usable demonstrations for train/validation/test splits")
        self.split = {"train": names[nval + ntest:], "validation": names[:nval], "test": names[nval:nval + ntest]}
        train = self.for_split("train")
        actions = np.concatenate([episode.prev_action[1:] for episode in train])
        # Upstream RSSM clips action magnitude at one. A signed-log transform
        # preserves information in large legacy joint commands before that clip.
        self.action_scale = np.maximum(1, np.abs(symlog(actions)).max(0)).astype(np.float32)
        states = np.concatenate([episode.state for episode in train])
        self.state_std = states.std(0, dtype=np.float64).clip(1e-3).astype(np.float32)
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        self.metadata = {
            "path": str(path), "sha256": digest, "env_args": env_args,
            "obs_key": obs_key, "action_key": action_key, "exclude_last": exclude_last,
            "skip_first": skip_first, "state_indices": list(range(states.shape[1])),
            "state_dim": states.shape[1], "action_dim": actions.shape[1],
            "demonstrations": len(names), "segments": len(self.episodes),
            "source_transitions": total, "usable_transitions": used,
            "excluded_terminal_transitions": terminal_rows,
            "termination_supervision": "Only nonterminal transitions retained; continuation targets are one",
            "action_transform": "clip(symlog(raw_action) / action_scale, -1, 1)",
            "action_scale": self.action_scale.tolist(), "metric_state_std": self.state_std.tolist(),
        }
        self._windows = {}

    def for_split(self, split):
        names = set(self.split[split])
        return [episode for episode in self.episodes if episode.demonstration in names]

    def sample(self, split, batch_size, length, rng):
        """Draw uniformly from valid windows, resetting recurrent state at each start."""
        if batch_size < 1 or length < 2:
            raise ValueError("Batch size must be positive and sequence length at least two")
        key = (split, length)
        if key not in self._windows:
            episodes = self.for_split(split)
            counts = np.array([max(0, len(e.state) - length + 1) for e in episodes])
            self._windows[key] = episodes, counts.cumsum()
        episodes, cumulative = self._windows[key]
        if not len(cumulative) or not cumulative[-1]:
            raise ValueError(f"No {split} sequences long enough for length {length}")
        draws = rng.integers(cumulative[-1], size=batch_size)
        selected = np.searchsorted(cumulative, draws, side="right")
        starts = draws - np.r_[0, cumulative[:-1]][selected]
        batch = {key: np.stack([getattr(episodes[i], key)[s:s + length] for i, s in zip(selected, starts)])
                 for key in ("state", "prev_action", "reward", "reward_mask")}
        batch["prev_action"] = np.clip(symlog(batch["prev_action"]) / self.action_scale, -1, 1)
        batch["prev_action"][:, 0] = 0
        batch["reward_mask"][:, 0] = 0
        batch["is_first"] = np.zeros((batch_size, length), bool)
        batch["is_first"][:, 0] = True
        return batch
