"""Load HDF5 demonstration transitions into a FastTD3 SimpleReplayBuffer.

Action-space normalisation
--------------------------
FastTD3's actor outputs actions in approximately [-1, 1] (tanh-squashed).
Raw demo actions in the HDF5 live in robot space (e.g. absolute EE pose in
metres).  Storing them as-is would confuse the critic, which sees mixed
action scales from RL rollouts vs demos.

We normalise demo actions so they live in the same approximate range as the
actor output.  Two sources are supported, tried in order:

1. BC checkpoint  (``prior_ckpt_path``)
   Load ``ckpt["act_normalizer"]`` (mean, std) fit on the demo dataset during
   BC training.  This is the most accurate normaliser because it was fit on
   exactly the same data.

2. Dataset statistics  (fallback)
   If no prior path is given, fit mean/std on the loaded actions on-the-fly.

After normalisation, demo actions ≈ zero-mean unit-variance ≈ the logit that
the BC-initialised actor would produce before its tanh head.  Since
tanh(x) ≈ x for |x| < 1, the action spaces align well.

Replay buffer layout
--------------------
``SimpleReplayBuffer`` stores data as ``(n_env, buffer_size, dim)``.  Each
call to ``extend`` fills ONE slot across ALL envs with the SAME per-env batch.
For demos we instead write transitions directly into the underlying tensors
so we can pack N demo transitions into ceil(N / n_env) slots, distributing
them round-robin across the env dimension.  This is valid because
``SimpleReplayBuffer.sample`` samples each env independently.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    pass  # avoid circular imports; SimpleReplayBuffer imported at call-time


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _demo_sort_key(name: str):
    m = re.fullmatch(r"demo_(\d+)", name)
    return (0, int(m.group(1))) if m else (1, name)


def _load_act_normalizer_from_prior(prior_ckpt_path: str) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return (mean, std) from a BC checkpoint's act_normalizer, or None."""
    try:
        p = Path(prior_ckpt_path).expanduser()
        # resolve directory -> best.pth
        if p.is_dir():
            for base in (p, p / "checkpoints"):
                for name in ("best.pth", "final.pth"):
                    cand = base / name
                    if cand.is_file():
                        p = cand
                        break
                else:
                    continue
                break
        if not p.is_file():
            return None
        ckpt = torch.load(str(p), map_location="cpu", weights_only=False)
        act_norm = ckpt.get("act_normalizer")
        if act_norm and "mean" in act_norm and "std" in act_norm:
            return (
                act_norm["mean"].float(),
                act_norm["std"].float().clamp(min=1e-6),
            )
    except Exception:
        pass
    return None


def _normalize_actions(
    actions: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    return (actions - mean.to(actions.device)) / std.to(actions.device)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@torch.no_grad()
def populate_replay_buffer_from_hdf5(
    rb,
    dataset_path: str,
    obs_key: str,
    action_mode: str,
    device: torch.device,
    *,
    prior_ckpt_path: str | None = None,
    max_transitions: int | None = None,
    use_rewards: bool = True,
    preload_fraction: float | None = None,
    preload_slots: int | None = None,
    verbose: bool = True,
) -> int:
    """Load demo transitions from an HDF5 file into a FastTD3 replay buffer.

    Parameters
    ----------
    rb:
        A ``SimpleReplayBuffer`` instance (from ``fast_td3/fast_td3_utils.py``).
    dataset_path:
        Path to the HDF5 file (e.g. ``fr3_finray_table_drawer_damped_noise.hdf5``).
    obs_key:
        Observation key to read (``"state"`` or ``"full_state"``).
    action_mode:
        Action sub-key (e.g. ``"absolute_ee"``, ``"relative_joint"``).
    device:
        Torch device.  Buffer tensors are assumed to already be on this device.
    prior_ckpt_path:
        Optional path to a BC prior checkpoint or run directory.  When given,
        the ``act_normalizer`` stored in the checkpoint is used to normalise
        demo actions.  Strongly recommended when ``--prior_checkpoint`` is also
        passed to ``train_fasttd3.py``.
    max_transitions:
        Cap on how many demo transitions to load.  ``None`` = load all.
    use_rewards:
        If ``True`` (default) read ``data/<demo>/rewards`` when present.
        If ``False`` force all demo rewards to zero, which is the right choice
        when the dataset's rewards are uninformative (e.g. MPC logs with reward
        proxies that don't match the RL reward definition).  Critic will learn
        only from the online RL reward signal on demo transitions.
    preload_fraction:
        Optional fraction ``(0, 1]`` of the replay buffer to fill with demos.
        If the demo HDF5 has fewer transitions than ``fraction * n_env *
        buffer_size``, demos are repeated round-robin to fill the requested
        slots.  Mutually exclusive with ``preload_slots``.
    preload_slots:
        Optional explicit number of per-env replay slots to fill with demos.
        Repeats demos round-robin when needed.  Capped at ``buffer_size``.
        Mutually exclusive with ``preload_fraction``.
    verbose:
        Print a summary.

    Returns
    -------
    int
        Number of transitions actually written to the buffer.
    """
    import h5py

    # ---- try to get act normalizer from BC checkpoint -------------------
    act_mean: torch.Tensor | None = None
    act_std:  torch.Tensor | None = None
    if prior_ckpt_path is not None:
        result = _load_act_normalizer_from_prior(prior_ckpt_path)
        if result is not None:
            act_mean, act_std = result
            if verbose:
                print(f"[demo_buffer] Using act_normalizer from BC prior checkpoint.")
        else:
            if verbose:
                print(f"[demo_buffer] WARNING: could not load act_normalizer from prior; "
                      "will fit from dataset.")

    # ---- read all demos from HDF5 ---------------------------------------
    obs_list:      list[np.ndarray] = []
    next_obs_list: list[np.ndarray] = []
    act_list:      list[np.ndarray] = []
    rew_list:      list[np.ndarray] = []
    done_list:     list[np.ndarray] = []
    trunc_list:    list[np.ndarray] = []

    with h5py.File(dataset_path, "r") as f:
        if "data" not in f:
            raise KeyError(f"HDF5 file has no '/data' group: {dataset_path}")
        demo_keys = sorted(f["data"].keys(), key=_demo_sort_key)

        if verbose:
            print(f"[demo_buffer] Reading {len(demo_keys)} demos from {dataset_path}")
            if not use_rewards:
                print("[demo_buffer] use_rewards=False: ignoring dataset rewards (all demo rewards set to 0).")

        for dk in demo_keys:
            obs_path = f"data/{dk}/obs/{obs_key}"
            act_path = f"data/{dk}/actions/{action_mode}"
            if obs_path not in f:
                continue
            if act_path not in f:
                continue

            obs_np = np.array(f[obs_path], dtype=np.float32)   # (T, obs_dim)
            act_np = np.array(f[act_path], dtype=np.float32)   # (T, act_dim)
            T = min(obs_np.shape[0], act_np.shape[0])
            if T < 2:
                continue

            # Rewards (optional; respect use_rewards gate)
            rew_path = f"data/{dk}/rewards"
            if use_rewards and rew_path in f:
                rew_np = np.array(f[rew_path], dtype=np.float32).flatten()[:T - 1]
                if len(rew_np) < T - 1:
                    rew_np = np.concatenate([rew_np, np.zeros(T - 1 - len(rew_np), dtype=np.float32)])
            else:
                rew_np = np.zeros(T - 1, dtype=np.float32)

            # Dones (optional; default: only the last step is done)
            done_path = f"data/{dk}/dones"
            if done_path in f:
                done_np = np.array(f[done_path], dtype=np.float32).flatten()[:T - 1]
                if len(done_np) < T - 1:
                    done_np = np.concatenate([done_np, np.zeros(T - 1 - len(done_np), dtype=np.float32)])
            else:
                done_np = np.zeros(T - 1, dtype=np.float32)
            done_np[-1] = 1.0  # last step of demo is always terminal

            # Truncations: treat demo end as a time-limit truncation so the
            # critic bootstraps beyond the demo horizon.
            trunc_np = np.zeros(T - 1, dtype=np.float32)
            trunc_np[-1] = 1.0

            obs_list.append(obs_np[:T - 1])
            next_obs_list.append(obs_np[1:T])
            act_list.append(act_np[:T - 1])
            rew_list.append(rew_np)
            done_list.append(done_np)
            trunc_list.append(trunc_np)

    if not obs_list:
        if verbose:
            print("[demo_buffer] No valid transitions found in dataset.")
        return 0

    # Keep large tensors on CPU throughout; transfer each to `device` only at
    # the final write step (one tensor at a time) to avoid peak GPU memory
    # pressure from simultaneous copies during repeat indexing / reshape.
    obs_all      = torch.from_numpy(np.concatenate(obs_list,      axis=0))
    next_obs_all = torch.from_numpy(np.concatenate(next_obs_list, axis=0))
    act_all      = torch.from_numpy(np.concatenate(act_list,      axis=0))
    rew_all      = torch.from_numpy(np.concatenate(rew_list,      axis=0))
    done_all     = torch.from_numpy(np.concatenate(done_list,     axis=0))
    trunc_all    = torch.from_numpy(np.concatenate(trunc_list,    axis=0))

    # ---- fit act normalizer from dataset if not loaded from prior -------
    if act_mean is None:
        act_mean = act_all.mean(dim=0)
        act_std  = act_all.std(dim=0).clamp(min=1e-6)
        if verbose:
            print("[demo_buffer] Fitted act_normalizer from dataset statistics.")

    # ---- normalise actions to approx [-1, 1] (on CPU) -------------------
    assert act_mean is not None and act_std is not None
    act_all = _normalize_actions(act_all, act_mean.cpu(), act_std.cpu())

    N_total = obs_all.shape[0]
    if max_transitions is not None:
        N_total = min(N_total, max_transitions)
        obs_all      = obs_all[:N_total]
        next_obs_all = next_obs_all[:N_total]
        act_all      = act_all[:N_total]
        rew_all      = rew_all[:N_total]
        done_all     = done_all[:N_total]
        trunc_all    = trunc_all[:N_total]

    # ---- check obs/act dims match buffer --------------------------------
    if obs_all.shape[1] != rb.n_obs:
        raise ValueError(
            f"Demo obs_dim={obs_all.shape[1]} does not match buffer n_obs={rb.n_obs}. "
            f"Check obs_key='{obs_key}' matches the env observation space."
        )
    if act_all.shape[1] != rb.n_act:
        raise ValueError(
            f"Demo act_dim={act_all.shape[1]} does not match buffer n_act={rb.n_act}. "
            f"Check action_mode='{action_mode}' matches the env action space."
        )

    # ---- write transitions directly into buffer tensors -----------------
    # Buffer shape: (n_env, buffer_size, dim).
    # We distribute N transitions round-robin across envs:
    #   transition i -> env (i % n_env), slot (i // n_env)
    # Default behaviour fills ceil(N / n_env) slots; preload_fraction and
    # preload_slots let callers force a larger demo footprint (demos are
    # repeated round-robin to fill the requested slots).
    n_env       = rb.n_env
    buffer_size = rb.buffer_size

    if preload_fraction is not None and preload_slots is not None:
        raise ValueError("Specify only one of preload_fraction, preload_slots.")

    auto_slots = (N_total + n_env - 1) // n_env

    if preload_slots is not None:
        requested_slots = max(1, int(preload_slots))
    elif preload_fraction is not None:
        if not (0.0 < preload_fraction <= 1.0):
            raise ValueError(f"preload_fraction must be in (0, 1], got {preload_fraction}.")
        requested_slots = max(1, int(round(preload_fraction * buffer_size)))
    else:
        requested_slots = auto_slots

    n_slots   = min(requested_slots, buffer_size)
    N_written = min(N_total, n_slots * n_env)
    if verbose and n_slots > auto_slots:
        print(
            f"[demo_buffer] preload requests {n_slots} slots; demos ({N_total} "
            f"transitions, auto={auto_slots} slots) will be repeated round-robin."
        )

    # Pad so total length is a multiple of n_env (repeat from start).
    # Build the index on CPU so the indexed tensors stay on CPU.
    N_padded = n_slots * n_env
    repeat_idx = (
        torch.arange(N_padded) % N_total if N_padded > N_total else None
    )

    # Reshape to (n_slots, n_env, dim) then transpose to (n_env, n_slots, dim).
    # Kept as a local helper that operates on CPU tensors; each tensor is then
    # transferred to `device` one at a time to minimise peak GPU memory.
    def _prepare_and_write(
        src: torch.Tensor,
        extra_dims: tuple[int, ...],
        dst: torch.Tensor,
        *,
        cast_long: bool = False,
    ) -> None:
        t = src if repeat_idx is None else src[repeat_idx]
        shape = (n_slots, n_env) + extra_dims
        t = t[:N_padded].reshape(shape).permute(1, 0, *range(2, 2 + len(extra_dims)))
        t = t.contiguous()
        if cast_long:
            t = t.long()
        dst[:, :n_slots] = t.to(device, non_blocking=True)

    _prepare_and_write(obs_all,      (rb.n_obs,), rb.observations)
    del obs_all
    _prepare_and_write(next_obs_all, (rb.n_obs,), rb.next_observations)
    del next_obs_all
    _prepare_and_write(act_all,      (rb.n_act,), rb.actions)
    del act_all
    _prepare_and_write(rew_all,      (),          rb.rewards)
    del rew_all
    _prepare_and_write(done_all,     (),          rb.dones,       cast_long=True)
    del done_all
    _prepare_and_write(trunc_all,    (),          rb.truncations, cast_long=True)
    del trunc_all

    # Asymmetric obs: we don't have privileged critic obs in the HDF5,
    # so leave those slots as zero (the critic will down-weight them naturally).

    # Advance the pointer so the sampler knows slots 0..n_slots-1 are valid.
    rb.ptr = n_slots

    if verbose:
        print(
            f"[demo_buffer] Loaded {N_total} demo transitions into replay buffer "
            f"({n_slots} slots x {n_env} envs, {N_written} unique transitions written)."
        )

    return N_written
