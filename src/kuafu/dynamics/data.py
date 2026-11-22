"""Read aligned HDF5 transitions without crossing demonstration boundaries."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch


@dataclass
class Episode:
    name: str
    states: torch.Tensor
    actions: torch.Tensor
    next_states: torch.Tensor
    steps: torch.Tensor


@dataclass
class Transitions:
    episodes: list[Episode]
    metadata: dict

    def tensors(self, names: list[str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Concatenate selected whole episodes in the requested order."""
        lookup = {episode.name: episode for episode in self.episodes}
        return tuple(torch.cat([getattr(lookup[name], field) for name in names])
                     for field in ("states", "actions", "next_states"))


def read_transitions(
    path: str | Path, *, obs_key: str = "full_state", action_key: str = "relative_joint",
    exclude_last: int = 1, skip_first: int = 1,
) -> Transitions:
    """Load explicit next observations; omit reset/terminal rows and selected columns.

    The defaults remove Kuafu's trailing SUN Program stage and first reset step.
    All coordinates and action scaling remain in the source dataset's convention.
    Terminal rows are excluded because Isaac may record an autoreset observation.
    """
    if exclude_last < 0 or skip_first < 0:
        raise ValueError("exclude_last and skip_first must be nonnegative")
    path = Path(path).expanduser().resolve()
    action_path = f"actions/{action_key}" if action_key else "actions"
    episodes, dimensions, dropped, total = [], None, 0, 0
    with h5py.File(path, "r") as source:
        if "data" not in source or not source["data"]:
            raise ValueError("Expected nonempty /data/demo_* HDF5 groups")
        env_args = source["data"].attrs.get("env_args", "{}")
        if isinstance(env_args, bytes):
            env_args = env_args.decode()
        env_args = json.loads(env_args)
        for name in sorted(source["data"]):
            group = source[f"data/{name}"]
            arrays = []
            for key in (f"obs/{obs_key}", action_path, f"next_obs/{obs_key}"):
                if key not in group or not isinstance(group[key], h5py.Dataset):
                    raise ValueError(f"{name}: expected dataset {key}")
                array = np.asarray(group[key], dtype=np.float32)
                if array.ndim != 2 or not np.isfinite(array).all():
                    raise ValueError(f"{name}/{key}: expected a finite rank-2 array")
                arrays.append(array)
            states, actions, next_states = arrays
            count, raw_dim = states.shape
            if states.shape != next_states.shape or len(actions) != count:
                raise ValueError(f"{name}: inconsistent obs/action/next_obs lengths or dimensions")
            if raw_dim <= exclude_last or not actions.shape[1]:
                raise ValueError(f"{name}: empty state/action dimensions")
            current_dims = (raw_dim, actions.shape[1])
            if dimensions is not None and current_dims != dimensions:
                raise ValueError(f"{name}: dimensions differ between demonstrations")
            dimensions = current_dims
            dones = np.asarray(group["dones"]).reshape(-1) if "dones" in group else np.zeros(count)
            if len(dones) != count or not np.isfinite(dones).all():
                raise ValueError(f"{name}: invalid dones")
            dones = dones.astype(bool)
            dim = raw_dim - exclude_last
            states, next_states = states[:, :dim], next_states[:, :dim]
            # Fail on stale next_obs instead of silently shifting actions or leaking targets.
            continuous = ~dones[:-1]
            if count > 1 and not np.allclose(
                next_states[:-1][continuous], states[1:][continuous], rtol=1e-5, atol=1e-6,
            ):
                raise ValueError(f"{name}: next_obs[t] does not match obs[t+1]; audit recorder alignment")
            keep = ~dones
            keep[:skip_first] = False
            steps = np.flatnonzero(keep)
            total += count
            dropped += count - len(steps)
            if not len(steps):
                raise ValueError(f"{name}: no usable transitions after reset/terminal filtering")
            episodes.append(Episode(name, *(torch.from_numpy(a[keep].copy())
                                            for a in (states, actions, next_states)),
                                    torch.from_numpy(steps)))
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return Transitions(episodes, {
        "path": str(path), "sha256": digest, "env_args": env_args,
        "obs_key": obs_key, "action_key": action_key, "exclude_last": exclude_last,
        "skip_first": skip_first, "terminal_rows": "excluded", "state_indices": list(range(dim)),
        "raw_state_dim": raw_dim, "state_dim": dim, "action_dim": actions.shape[1],
        "episodes": len(episodes), "source_transitions": total,
        "transitions": total - dropped, "dropped_transitions": dropped,
    })


def split_episodes(data: Transitions, *, seed: int, val_fraction: float, test_fraction: float) -> dict:
    """Create disjoint train/validation/test splits by whole demonstration."""
    if not (0 < val_fraction < 1 and 0 < test_fraction < 1 and val_fraction + test_fraction < 1):
        raise ValueError("Validation/test fractions must be positive and sum to less than one")
    names = np.random.default_rng(seed).permutation([e.name for e in data.episodes]).tolist()
    nval = max(1, round(len(names) * val_fraction))
    ntest = max(1, round(len(names) * test_fraction))
    if nval + ntest >= len(names):
        raise ValueError("Need enough demonstrations for nonempty train, validation, and test splits")
    return {"train": names[nval + ntest:], "validation": names[:nval], "test": names[nval:nval + ntest]}
