#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

SOURCE_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = SOURCE_ROOT.parent
SCRIPT_DIR = Path(__file__).resolve().parent
for candidate in (str(SOURCE_ROOT), str(SCRIPT_DIR)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from kuafu.paths import resolve_artifact_path

def _default_bc_log_root() -> Path:
    prior_root = os.environ.get("KUAFU_PRIOR_ARTIFACT_ROOT", "").strip()
    if prior_root:
        return Path(prior_root).expanduser().resolve() / "training_runs"

    task_root = os.environ.get("KUAFU_TASK_ARTIFACT_ROOT", "").strip()
    if task_root:
        return Path(task_root).expanduser().resolve() / "prior_artifacts" / "training_runs"

    configured = os.environ.get("KUAFU_ARTIFACT_ROOT", "").strip()
    base = (
        Path(configured).expanduser().resolve()
        if configured
        else PROJECT_ROOT.parent / "KuafuArtifacts"
    )
    return base / "standalone" / "prior_training"


def _resolve_bc_log_dir(value: object) -> str:
    path = Path(str(value or "")).expanduser()
    if not path.is_absolute():
        parts = path.parts[1:] if path.parts and path.parts[0] == "logs" else path.parts
        path = _default_bc_log_root().joinpath(*parts)
    resolved = path.resolve()
    if resolved == PROJECT_ROOT or PROJECT_ROOT in resolved.parents:
        raise ValueError(
            f"BC log/checkpoint directory must be outside the source project "
            f"({PROJECT_ROOT}); got {resolved}"
        )
    return str(resolved)


def _resolve_bc_dataset_path(value: object) -> str:
    """Resolve every trajectory input beneath the external artifact store."""

    path = Path(str(value or "")).expanduser()
    if not str(path).strip():
        raise ValueError("dataset.dataset must be a non-empty path")
    if not path.is_absolute():
        parts = path.parts
        if len(parts) >= 4 and parts[:4] == ("src", "kuafu", "data", "datasets"):
            path = Path("Trajectories") / path.name
    return str(resolve_artifact_path(path, "BC trajectory dataset"))


from kuafu.bc.training.normalization import NORMALIZER_EPS

try:
    from kuafu.bc.training.dataset import Hdf5TrajectoryDataset, Normalizer, fit_normalizer
    from kuafu.bc.training.model import MLPActorCriticPrior, MLPActorPrior
except ModuleNotFoundError:
    from dataset import Hdf5TrajectoryDataset, Normalizer, fit_normalizer
    from model import MLPActorCriticPrior, MLPActorPrior


@dataclass(frozen=True)
class RunConfig:
    dataset: str
    demo: str | None
    obs_key: str
    action_mode: str
    time_shift: int
    val_fraction: float
    split_mode: str
    seed: int
    action_chunk_size: int
    action_chunk_stride: int
    obs_history: int
    hidden_dims: tuple[int, ...]
    activation: str
    negative_slope: float
    dropout: float
    lr: float
    weight_decay: float
    batch_size: int
    num_workers: int
    pin_memory: bool
    num_epochs: int
    save_every_n_epochs: int
    grad_clip: float
    log_dir: str
    # Optional: demo to monitor with L2 each epoch (by index into /data group).
    monitor_demo_index: int | None = None
    sample_weighting: dict | None = None
    critic_pretraining: dict | None = None


@dataclass(frozen=True)
class SampleWeightConfig:
    enabled: bool
    obs_stage_index: int
    default_weight: float
    stage_weights: tuple[tuple[int, float], ...]
    normalize: bool
    weight_validation: bool


@dataclass(frozen=True)
class CriticPretrainConfig:
    enabled: bool
    gamma: float
    loss_coef: float
    normalize_targets: bool


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def _grad_norm(parameters) -> float:
    total = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        g = p.grad.detach()
        total += float(g.pow(2).sum().item())
    return float(total ** 0.5)


def _default_config_dict() -> dict:
    return {
        "dataset": {
            "dataset": "src/kuafu/data/datasets/fr3_finray_table_pen_cup_random_pen_cup.hdf5",
            "demo": None,
            "obs_key": "state",
            "action_mode": "relative_joint",
            "time_shift": -1,
            "val_fraction": 0.1,
            "split_mode": "random",
            "seed": 1999,
        },
        "preprocessing": {
            "action_chunk_size": 1,
            "action_chunk_stride": 1,
            "obs_history": 1,
        },
        "model": {
            "hidden_dims": [256, 512, 64],
            "activation": "leaky_relu",
            "negative_slope": 0.01,
            "dropout": 0.1,
        },
        "optimization": {
            "lr": 1e-4,
            "weight_decay": 1e-5,
            "batch_size": 1024,
            "num_workers": 0,
            "pin_memory": False,
            "num_epochs": 1000,
            "save_every_n_epochs": 100,
            "grad_clip": 10.0,
        },
        "logging": {
            "log_dir": str(_default_bc_log_root()),
            "monitor_demo_index": None,
        },
        "auto_shift": {
            "epochs": 50,
            "batch_size": 512,
        },
        "sample_weighting": {
            "enabled": False,
            "obs_stage_index": -1,
            "default_weight": 1.0,
            "normalize": True,
            "weight_validation": False,
            "stage_weights": {},
        },
        "critic_pretraining": {
            "enabled": False,
            "gamma": 0.99,
            "loss_coef": 1.0,
            "normalize_targets": True,
        },
    }


def _deep_update(base: dict, updates: dict) -> dict:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _load_yaml_config(config_path: str | os.PathLike[str]) -> tuple[dict, str]:
    merged = _default_config_dict()
    resolved = Path(config_path).expanduser()
    if not resolved.exists():
        raise FileNotFoundError(f"Config YAML not found: {resolved}")
    with open(resolved, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config YAML must contain a mapping at the top level: {resolved}")
    _deep_update(merged, loaded)
    return merged, str(resolved)


def _sample_weight_config_from_dict(raw_cfg: dict | None) -> SampleWeightConfig:
    if raw_cfg is None:
        raise ValueError("Missing required `sample_weighting` config block.")
    stage_weights_raw = raw_cfg.get("stage_weights", {}) or {}
    if not isinstance(stage_weights_raw, dict):
        raise ValueError("sample_weighting.stage_weights must be a mapping of stage_id -> weight")
    stage_weights = tuple((int(stage), float(weight)) for stage, weight in stage_weights_raw.items())
    return SampleWeightConfig(
        enabled=bool(raw_cfg.get("enabled", False)),
        obs_stage_index=int(raw_cfg.get("obs_stage_index", -1)),
        default_weight=float(raw_cfg.get("default_weight", 1.0)),
        stage_weights=stage_weights,
        normalize=bool(raw_cfg.get("normalize", True)),
        weight_validation=bool(raw_cfg.get("weight_validation", False)),
    )


def _critic_pretrain_config_from_dict(raw_cfg: dict | None) -> CriticPretrainConfig:
    if raw_cfg is None:
        raise ValueError("Missing required `critic_pretraining` config block.")
    return CriticPretrainConfig(
        enabled=bool(raw_cfg.get("enabled", True)),
        gamma=float(raw_cfg.get("gamma", 0.99)),
        loss_coef=float(raw_cfg.get("loss_coef", 1.0)),
        normalize_targets=bool(raw_cfg.get("normalize_targets", True)),
    )


def _sample_weights_from_obs(obs: torch.Tensor, cfg: SampleWeightConfig) -> torch.Tensor | None:
    if not cfg.enabled:
        return None
    if obs.ndim != 2:
        raise ValueError(f"Expected obs to be rank-2 for weighting, got shape={tuple(obs.shape)}")
    stage = obs[:, cfg.obs_stage_index].round().to(dtype=torch.int64)
    weights = torch.full(
        (obs.shape[0],),
        fill_value=cfg.default_weight,
        device=obs.device,
        dtype=obs.dtype,
    )
    for stage_id, weight in cfg.stage_weights:
        stage_value = torch.tensor(stage_id, device=stage.device, dtype=stage.dtype)
        weight_value = torch.tensor(weight, device=obs.device, dtype=obs.dtype)
        weights = torch.where(stage == stage_value, weight_value, weights)
    if cfg.normalize:
        weights = weights / weights.mean().clamp(min=1e-6)
    return weights


def _weighted_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    per_sample = (pred - target).pow(2).mean(dim=1)
    if sample_weights is None:
        return per_sample.mean()
    return (per_sample * sample_weights).mean()


def _build_dataset(
    *,
    dataset_path: str,
    demo: str | None,
    obs_key: str,
    action_mode: str,
    time_shift: int,
    split: str,
    val_fraction: float,
    split_mode: str,
    seed: int,
    load_returns: bool = False,
    discount: float = 0.99,
) -> Hdf5TrajectoryDataset:
    return Hdf5TrajectoryDataset(
        dataset_path=dataset_path,
        demo=demo,
        obs_key=obs_key,
        action_subkey=action_mode,
        time_shift=time_shift,  # type: ignore[arg-type]
        split=split,  # type: ignore[arg-type]
        val_fraction=val_fraction,
        index_mode=split_mode,  # type: ignore[arg-type]
        seed=seed,
        load_returns=load_returns,
        discount=discount,
    )


def _make_loader(
    ds: Hdf5TrajectoryDataset,
    batch_size: int,
    *,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    effective_batch_size = min(batch_size, len(ds)) if len(ds) > 0 else 1
    return DataLoader(
        ds,
        batch_size=effective_batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


def _bool_config_value(value: object, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


def _apply_preprocessing(
    ds: Hdf5TrajectoryDataset,
    obs_history: int,
    action_chunk_size: int,
    action_chunk_stride: int,
) -> None:
    if obs_history > 1 or action_chunk_size > 1:
        _apply_history_and_action_chunking_to_dataset(ds, obs_history, action_chunk_size, action_chunk_stride)


def _sorted_demo_keys(dataset_path: str) -> list[str]:
    import h5py

    with h5py.File(dataset_path, "r") as f:
        raw_keys = list(f["data"].keys())  # type: ignore[union-attr]

    def demo_sort_key(name: str):
        if "_" in name:
            try:
                return int(name.rsplit("_", 1)[1])
            except ValueError:
                pass
        return name

    return sorted(raw_keys, key=demo_sort_key)


def _resolve_monitor_demo(dataset_path: str, monitor_demo_index: int) -> tuple[int, str]:
    demos = _sorted_demo_keys(dataset_path)
    if not demos:
        raise ValueError("Dataset contains no demos but --monitor_demo_index was provided.")
    idx = int(monitor_demo_index)
    if idx < 0:
        idx += len(demos)
    if idx < 0 or idx >= len(demos):
        raise IndexError(f"--monitor_demo_index {monitor_demo_index} out of range for {len(demos)} demos")
    return idx, demos[idx]


def _run_config_from_yaml(raw_cfg: dict, *, config_path: str, time_shift: int | None = None) -> RunConfig:
    dataset_cfg = raw_cfg.get("dataset", {})
    preprocessing_cfg = raw_cfg.get("preprocessing", {})
    model_cfg = raw_cfg.get("model", {})
    optimization_cfg = raw_cfg.get("optimization", {})
    logging_cfg = raw_cfg.get("logging", {})

    hidden_dims = tuple(int(v) for v in model_cfg.get("hidden_dims", [256, 512, 64]))
    if len(hidden_dims) == 0:
        raise ValueError("model.hidden_dims must contain at least one integer.")

    resolved_time_shift = time_shift if time_shift is not None else dataset_cfg.get("time_shift", -1)
    if resolved_time_shift == "auto":
        raise ValueError("time_shift must be resolved before building RunConfig.")

    sample_weight_cfg = _sample_weight_config_from_dict(raw_cfg.get("sample_weighting"))
    critic_pretrain_cfg = _critic_pretrain_config_from_dict(raw_cfg.get("critic_pretraining"))

    return RunConfig(
        dataset=_resolve_bc_dataset_path(dataset_cfg.get("dataset")),
        demo=dataset_cfg.get("demo"),
        obs_key=str(dataset_cfg.get("obs_key", "state")),
        action_mode=str(dataset_cfg.get("action_mode", "relative_joint")),
        time_shift=int(resolved_time_shift),
        val_fraction=float(dataset_cfg.get("val_fraction", 0.1)),
        split_mode=str(dataset_cfg.get("split_mode", "random")),
        seed=int(dataset_cfg.get("seed", 1999)),
        action_chunk_size=int(preprocessing_cfg.get("action_chunk_size", 1)),
        action_chunk_stride=int(preprocessing_cfg.get("action_chunk_stride", preprocessing_cfg.get("skip_step", 1))),
        obs_history=int(preprocessing_cfg.get("obs_history", 1)),
        hidden_dims=hidden_dims,
        activation=str(model_cfg.get("activation", "leaky_relu")),
        negative_slope=float(model_cfg.get("negative_slope", 0.01)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        lr=float(optimization_cfg.get("lr", 1e-4)),
        weight_decay=float(optimization_cfg.get("weight_decay", 1e-5)),
        batch_size=int(optimization_cfg.get("batch_size", 1024)),
        num_workers=int(optimization_cfg.get("num_workers", 0)),
        pin_memory=_bool_config_value(optimization_cfg.get("pin_memory"), False),
        num_epochs=int(optimization_cfg.get("num_epochs", 1000)),
        save_every_n_epochs=int(optimization_cfg.get("save_every_n_epochs", 100)),
        grad_clip=float(optimization_cfg.get("grad_clip", 10.0)),
        log_dir=_resolve_bc_log_dir(
            logging_cfg.get("log_dir", str(_default_bc_log_root()))
        ),
        monitor_demo_index=logging_cfg.get("monitor_demo_index"),
        sample_weighting=asdict(sample_weight_cfg),
        critic_pretraining=asdict(critic_pretrain_cfg),
    )


def _auto_shift_settings(raw_cfg: dict) -> tuple[int, int]:
    auto_shift_cfg = raw_cfg.get("auto_shift", {})
    return int(auto_shift_cfg.get("epochs", 50)), int(auto_shift_cfg.get("batch_size", 512))


def _resolved_raw_config(
    raw_cfg: dict, *, config_path: str, time_shift: int, run_cfg: RunConfig
) -> dict:
    resolved = json.loads(json.dumps(raw_cfg))
    resolved.setdefault("dataset", {})
    resolved["dataset"]["dataset"] = run_cfg.dataset
    resolved["dataset"]["time_shift"] = int(time_shift)
    resolved.setdefault("logging", {})
    resolved["logging"]["log_dir"] = run_cfg.log_dir
    resolved["_meta"] = {
        "config_path": config_path,
    }
    return resolved

def _extract_linear_key_pairs(module: nn.Sequential, prefix: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            pairs.append((f"trunk.{name}.weight", f"trunk.{name}.bias"))
    if len(pairs) == 0:
        raise ValueError(f"Expected at least 1 Linear layer in model.{prefix}, found none")
    return pairs


def _actor_trunk_linear_key_pairs(model: nn.Module) -> list[tuple[str, str]]:
    trunk = model.actor_trunk if hasattr(model, "actor_trunk") else model.trunk
    prefix = "actor_trunk" if hasattr(model, "actor_trunk") else "trunk"
    pairs: list[tuple[str, str]] = []
    for name, child in trunk.named_children():
        if isinstance(child, nn.Linear):
            pairs.append((f"{prefix}.{name}.weight", f"{prefix}.{name}.bias"))
    if len(pairs) == 0:
        raise ValueError(f"Expected at least 1 Linear layer in model.{prefix}, found none")
    return pairs


def _critic_trunk_linear_key_pairs(model: MLPActorCriticPrior) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for name, child in model.critic_trunk.named_children():
        if isinstance(child, nn.Linear):
            pairs.append((f"critic_trunk.{name}.weight", f"critic_trunk.{name}.bias"))
    if len(pairs) == 0:
        raise ValueError("Expected at least 1 Linear layer in model.critic_trunk, found none")
    return pairs

def _attach_obs_normalizer_buffers(sd: dict, obs_mean: torch.Tensor, obs_std: torch.Tensor, first_weight_key: str, first_bias_key: str) -> dict:
    # The raw-input export uses ObservationNormalizedLinear. Keeping (x-mean)
    # explicit avoids catastrophic cancellation for constant observation channels.
    prefix = first_weight_key.rsplit(".", 1)[0]
    sd[f"{prefix}.obs_mean"] = obs_mean.detach().cpu().float().clone()
    sd[f"{prefix}.obs_std"] = obs_std.detach().cpu().float().clamp_min(NORMALIZER_EPS)
    return sd


def _bake_act_denorm_into_last_layer(sd: dict, act_mean: torch.Tensor, act_std: torch.Tensor,last_weight_key: str,last_bias_key: str) -> dict:
    W_key, b_key = last_weight_key, last_bias_key
    if W_key not in sd or b_key not in sd:
        raise KeyError(f"Missing keys for last layer baking: {W_key}, {b_key}")
    W = sd[W_key].float().clone()  # (act_dim, hidden)
    b = sd[b_key].float().clone()  # (act_dim,)
    act_mean = act_mean.to(device=W.device, dtype=torch.float32)
    act_std = act_std.to(device=W.device, dtype=torch.float32).clamp(min=NORMALIZER_EPS)
    W_new = act_std.unsqueeze(1) * W
    b_new = act_std * b + act_mean
    sd[W_key] = W_new.to(sd[W_key].dtype)
    sd[b_key] = b_new.to(sd[b_key].dtype)
    return sd


def _has_critic_head(model: nn.Module) -> bool:
    return hasattr(model, "critic_trunk") and hasattr(model, "value_layer")


def export_baked_state_dicts(
    model: nn.Module,
    obs_norm: Normalizer,
    act_norm: Normalizer,
) -> dict[str, dict]:
    """Return baked state dicts for prior usage.

    - prior_norm_obs: expects normalized obs, outputs raw actions
    - prior_raw_obs: expects raw obs via ObservationNormalizedLinear, outputs raw actions
      (explicit obs_mean/obs_std buffers, not a folded float32 affine transform)
    """
    first_w_key, first_b_key = _actor_trunk_linear_key_pairs(model)[0]
    last_w_key, last_b_key = "mean_layer.weight", "mean_layer.bias"

    # keep baked weights on CPU for portability (and to avoid mixed-device arithmetic)
    base = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
    prior_norm_obs = _bake_act_denorm_into_last_layer({k: v.clone() for k, v in base.items()}, act_norm.mean, act_norm.std,         last_w_key, last_b_key)
    prior_raw_obs = _attach_obs_normalizer_buffers(
        {k: v.clone() for k, v in prior_norm_obs.items()}, obs_norm.mean, obs_norm.std, first_w_key,first_b_key
    )
    baked = {
        "model_state_dict_norm": base,
        "model_state_dict_prior_norm_obs": prior_norm_obs,
        "model_state_dict_prior_raw_obs": prior_raw_obs,
    }
    if _has_critic_head(model):
        critic_first_w_key, critic_first_b_key = _critic_trunk_linear_key_pairs(model)[0]
        baked["value_state_dict_norm_obs"] = base
        baked["value_state_dict_raw_obs"] = _attach_obs_normalizer_buffers(
            {k: v.clone() for k, v in base.items()},
            obs_norm.mean,
            obs_norm.std,
            critic_first_w_key,
            critic_first_b_key,
        )
    return baked


def _apply_history_and_action_chunking_to_dataset(
    ds: Hdf5TrajectoryDataset,
    obs_history: int,
    action_chunk_size: int,
    action_chunk_stride: int,
) -> None:
    """In-place modify dataset to stack observation history and action chunks.

    - obs_history: number of past obs (including current) to stack. 1 = disabled.
    - action_chunk_size: number of future actions (including current) to stack. 1 = disabled.
    - action_chunk_stride: spacing between sampled actions in the chunk. 1 = consecutive actions.
    """
    if len(ds) == 0:
        return
    if obs_history < 1:
        raise ValueError(f"obs_history must be >= 1, got {obs_history}")
    if action_chunk_size < 1:
        raise ValueError(f"action_chunk_size must be >= 1, got {action_chunk_size}")
    if action_chunk_stride < 1:
        raise ValueError(f"action_chunk_stride must be >= 1, got {action_chunk_stride}")

    # Observation history stacking
    if obs_history > 1:
        obs = ds.obs.float()  # (N, obs_dim)
        N, obs_dim = obs.shape
        H = int(obs_history)
        first = obs[0:1].repeat(H - 1, 1)
        padded = torch.cat([first, obs], dim=0)  # (N + H - 1, obs_dim)
        chunks = []
        for t in range(N):
            window = padded[t : t + H].reshape(-1)
            chunks.append(window)
        obs_hist = torch.stack(chunks, dim=0)  # (N, H * obs_dim)
        ds.obs = obs_hist
        ds.obs_dim = obs_hist.shape[1]

    # Action chunking (stage-aware: chunks cannot cross stage boundaries).
    # Stage is always the last element of obs (correct after history stacking too).
    # When the stage would change before the chunk is complete, the last action
    # of the current stage is repeated to fill remaining chunk positions.
    if action_chunk_size > 1:
        acts = ds.actions.float()  # (N, act_dim)
        N, act_dim = acts.shape
        C = int(action_chunk_size)
        stride = int(action_chunk_stride)

        stage = ds.obs.float()[:, -1].round().long()  # (N,)

        # end_idx[t] = last contiguous index sharing the same stage as t
        end_idx = torch.arange(N, dtype=torch.long)
        for t in range(N - 2, -1, -1):
            if stage[t] == stage[t + 1]:
                end_idx[t] = end_idx[t + 1]

        # idx_matrix[t, k] = min(t + k * stride, end_idx[t])
        t_idx = torch.arange(N, dtype=torch.long).unsqueeze(1)  # (N, 1)
        k_idx = torch.arange(C, dtype=torch.long).unsqueeze(0)  # (1, C)
        idx_matrix = torch.minimum(t_idx + k_idx * stride, end_idx.unsqueeze(1))  # (N, C)

        acts_chunked = acts[idx_matrix.reshape(-1)].reshape(N, C * act_dim)
        ds.actions = acts_chunked
        ds.act_dim = acts_chunked.shape[1]


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    obs_norm: Normalizer,
    act_norm: Normalizer,
    device: torch.device,
    sample_weight_cfg: SampleWeightConfig | None = None,
) -> float:
    model.eval()
    total = 0.0
    n_samples = 0
    for batch in loader:
        obs, act = batch[:2]
        obs = obs.to(device)
        act = act.to(device)
        pred = model(obs_norm.normalize(obs))
        per_sample = (pred - act_norm.normalize(act)).pow(2).mean(dim=1)
        sample_weights = _sample_weights_from_obs(obs, sample_weight_cfg) if sample_weight_cfg is not None else None
        if sample_weights is not None:
            per_sample = per_sample * sample_weights
        total += float(per_sample.sum().item())
        n_samples += int(per_sample.numel())
    return total / max(n_samples, 1)


@torch.no_grad()
def compute_l2_error_per_action(
    model: nn.Module,
    loader: DataLoader,
    obs_norm: Normalizer,
    act_norm: Normalizer,
    device: torch.device,
) -> float:
    """Return mean L2 error per action (in raw action units) over the loader."""
    model.eval()
    total_l2 = 0.0
    n_samples = 0
    for batch in loader:
        obs, act = batch[:2]
        obs = obs.to(device)
        act = act.to(device)
        pred_norm = model(obs_norm.normalize(obs))
        pred = act_norm.denormalize(pred_norm)
        diff = pred - act
        batch_l2 = diff.pow(2).sum(dim=1).sqrt()  # (batch,)
        total_l2 += float(batch_l2.sum().item())
        n_samples += int(batch_l2.numel())
    return total_l2 / max(n_samples, 1)


@torch.no_grad()
def evaluate_value_loss(
    model: nn.Module,
    loader: DataLoader,
    obs_norm: Normalizer,
    value_norm: Normalizer | None,
    device: torch.device,
) -> float:
    if not _has_critic_head(model):
        return float("nan")
    model.eval()
    total = 0.0
    n_samples = 0
    for batch in loader:
        if len(batch) < 3:
            raise RuntimeError("Critic evaluation expected return targets in every batch, but they were missing.")
        obs, _act, ret = batch
        obs = obs.to(device)
        ret = ret.to(device)
        target = value_norm.normalize(ret) if value_norm is not None else ret
        pred = model.forward_critic(obs_norm.normalize(obs))
        loss = (pred - target).pow(2)
        total += float(loss.sum().item())
        n_samples += int(loss.numel())
    return total / max(n_samples, 1)


def train_one(run_cfg: RunConfig, resolved_config: dict) -> str:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if run_cfg.critic_pretraining is None:
        raise ValueError("RunConfig is missing `critic_pretraining`.")
    critic_cfg = CriticPretrainConfig(**run_cfg.critic_pretraining)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(run_cfg.log_dir) / timestamp
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    _status_path = run_dir / "training_log.txt"
    _status_f = open(_status_path, "w")

    def log(msg: str) -> None:
        print(msg)
        _status_f.write(msg + "\n")
        _status_f.flush()

    log(f"Device: {device}")

    writer = SummaryWriter(log_dir=str(run_dir))
    (run_dir / "meta").mkdir(exist_ok=True)
    with open(run_dir / "meta" / "config.yaml", "w") as f:
        yaml.dump(resolved_config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    train_ds = _build_dataset(
        dataset_path=run_cfg.dataset,
        demo=run_cfg.demo,
        obs_key=run_cfg.obs_key,
        action_mode=run_cfg.action_mode,
        time_shift=run_cfg.time_shift,
        split="train",
        val_fraction=run_cfg.val_fraction,
        split_mode=run_cfg.split_mode,
        seed=run_cfg.seed,
        load_returns=critic_cfg.enabled,
        discount=critic_cfg.gamma,
    )
    val_ds = _build_dataset(
        dataset_path=run_cfg.dataset,
        demo=run_cfg.demo,
        obs_key=run_cfg.obs_key,
        action_mode=run_cfg.action_mode,
        time_shift=run_cfg.time_shift,
        split="val",
        val_fraction=run_cfg.val_fraction,
        split_mode=run_cfg.split_mode,
        seed=run_cfg.seed,
        load_returns=critic_cfg.enabled,
        discount=critic_cfg.gamma,
    )
    log(
        f"Loaded dataset: obs_key={run_cfg.obs_key} action_mode={run_cfg.action_mode} "
        f"T_obs={train_ds.T_obs} T_act={train_ds.T_act} time_shift={run_cfg.time_shift} "
        f"split_mode={run_cfg.split_mode}"
    )
    log(
        f"Demos total={train_ds.num_demos_total} train={len(train_ds.train_demo_keys)} val={len(train_ds.val_demo_keys)} | "
        f"Pairs train={len(train_ds)} val={len(val_ds)}"
    )

    _apply_preprocessing(train_ds, run_cfg.obs_history, run_cfg.action_chunk_size, run_cfg.action_chunk_stride)
    _apply_preprocessing(val_ds, run_cfg.obs_history, run_cfg.action_chunk_size, run_cfg.action_chunk_stride)

    if run_cfg.obs_history > 1 or run_cfg.action_chunk_size > 1:
        log(
            f"After chunking: obs_history={run_cfg.obs_history} action_chunk_size={run_cfg.action_chunk_size} "
            f"action_chunk_stride={run_cfg.action_chunk_stride} "
            f"obs_dim={train_ds.obs_dim} act_dim={train_ds.act_dim}"
        )
    else:
        log(f"obs_dim={train_ds.obs_dim} act_dim={train_ds.act_dim}")
    if run_cfg.sample_weighting is None:
        raise ValueError("RunConfig is missing `sample_weighting`.")
    sample_weight_cfg = SampleWeightConfig(**run_cfg.sample_weighting)
    val_weight_cfg = sample_weight_cfg if sample_weight_cfg.enabled and sample_weight_cfg.weight_validation else None
    critic_enabled = critic_cfg.enabled
    if sample_weight_cfg.enabled:
        train_stage = train_ds.obs[:, sample_weight_cfg.obs_stage_index].round().to(torch.int64)
        unique_stage, counts = torch.unique(train_stage, return_counts=True)
        stage_summary = ", ".join(
            f"{int(stage.item())}:{int(count.item())}" for stage, count in zip(unique_stage, counts, strict=True)
        )
        stage_weight_summary = {int(stage): float(weight) for stage, weight in sample_weight_cfg.stage_weights}
        log(
            "Stage weighting enabled: "
            f"obs_stage_index={sample_weight_cfg.obs_stage_index} "
            f"default_weight={sample_weight_cfg.default_weight:.4g} "
            f"normalize={sample_weight_cfg.normalize} "
            f"weight_validation={sample_weight_cfg.weight_validation} "
            f"stage_weights={stage_weight_summary} "
            f"train_stage_counts={{{stage_summary}}}"
        )
    if run_cfg.val_fraction > 0.0 and len(train_ds.val_demo_keys) == 0:
        log(
            "[WARN] Validation-by-demo requested, but there are no held-out demos in this split. "
            "Use a multi-demo dataset without --demo, or set val_fraction=0.0."
        )
    if critic_enabled:
        if train_ds.returns is None:
            raise RuntimeError("Critic pretraining is enabled, but the training dataset did not load return targets.")
        if len(train_ds.returns) == 0:
            raise RuntimeError("Critic pretraining is enabled, but the training dataset has zero return targets.")
        if len(val_ds) > 0 and val_ds.returns is None:
            raise RuntimeError("Critic pretraining is enabled, but the validation dataset did not load return targets.")

    # normalizers fitted on train only
    obs_norm = fit_normalizer(train_ds.obs)
    act_norm = fit_normalizer(train_ds.actions)
    value_norm = fit_normalizer(train_ds.returns) if critic_enabled else None
    log(
        f"Obs norm: mean[0]={obs_norm.mean.flatten()[0].item():.4g} std[0]={obs_norm.std.flatten()[0].item():.4g} "
        f"(min_std={float(obs_norm.std.min()):.4g})"
    )
    log(
        f"Act norm: mean[0]={act_norm.mean.flatten()[0].item():.4g} std[0]={act_norm.std.flatten()[0].item():.4g} "
        f"(min_std={float(act_norm.std.min()):.4g})"
    )
    if critic_enabled:
        if value_norm is None:
            raise RuntimeError("Critic pretraining is enabled, but value normalization statistics were not computed.")
        raw_ret = train_ds.returns
        log(
            f"Critic pretraining: gamma={critic_cfg.gamma:.4f} loss_coef={critic_cfg.loss_coef:.4g} "
            f"normalize_targets={critic_cfg.normalize_targets} "
            f"return_mean={raw_ret.mean().item():.4g} return_std={raw_ret.std().item():.4g}"
        )

    log(
        f"DataLoader: batch_size={run_cfg.batch_size} num_workers={run_cfg.num_workers} "
        f"pin_memory={run_cfg.pin_memory}"
    )
    train_loader = _make_loader(
        train_ds,
        run_cfg.batch_size,
        shuffle=True,
        num_workers=run_cfg.num_workers,
        pin_memory=run_cfg.pin_memory,
    )
    train_eval_loader = _make_loader(
        train_ds,
        run_cfg.batch_size,
        shuffle=False,
        num_workers=run_cfg.num_workers,
        pin_memory=run_cfg.pin_memory,
    )
    val_loader = (
        _make_loader(
            val_ds,
            run_cfg.batch_size,
            shuffle=False,
            num_workers=run_cfg.num_workers,
            pin_memory=run_cfg.pin_memory,
        )
        if len(val_ds) > 0
        else None
    )

    if critic_enabled:
        model = MLPActorCriticPrior(
            obs_dim=train_ds.obs_dim,
            act_dim=train_ds.act_dim,
            hidden_dims=run_cfg.hidden_dims,
            activation=run_cfg.activation,
            negative_slope=run_cfg.negative_slope,
            dropout=run_cfg.dropout,
        ).to(device)
    else:
        model = MLPActorPrior(
            obs_dim=train_ds.obs_dim,
            act_dim=train_ds.act_dim,
            hidden_dims=run_cfg.hidden_dims,
            activation=run_cfg.activation,
            negative_slope=run_cfg.negative_slope,
            dropout=run_cfg.dropout,
        ).to(device)
    log(
        f"Model params: {sum(p.numel() for p in model.parameters()):,} | "
        f"activation={run_cfg.activation} dropout={run_cfg.dropout} weight_decay={run_cfg.weight_decay} "
        f"critic_enabled={critic_enabled}"
    )

    opt = torch.optim.Adam(model.parameters(), lr=run_cfg.lr, weight_decay=run_cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=run_cfg.num_epochs, eta_min=run_cfg.lr * 0.01)

    best_score = float("inf")
    grad_norm_hist: list[float] = []
    train_loss_hist: list[float] = []
    val_loss_hist: list[float] = []
    train_actor_loss_hist: list[float] = []
    train_critic_loss_hist: list[float] = []
    val_critic_loss_hist: list[float] = []

    def save_ckpt(path: Path, epoch: int, train_loss: float, val_loss: float):
        if critic_enabled:
            if value_norm is None:
                raise RuntimeError("Refusing to save critic-pretrained checkpoint without `value_normalizer`.")
            if train_ds.returns is None:
                raise RuntimeError("Refusing to save critic-pretrained checkpoint without return targets.")
        baked = export_baked_state_dicts(model, obs_norm, act_norm)
        torch.save(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "obs_dim": train_ds.obs_dim,
                "act_dim": train_ds.act_dim,
                "hidden_dims": run_cfg.hidden_dims,
                "activation": run_cfg.activation,
                "negative_slope": run_cfg.negative_slope,
                "dropout": run_cfg.dropout,
                "obs_history": run_cfg.obs_history,
                "action_chunk_size": run_cfg.action_chunk_size,
                "action_chunk_stride": run_cfg.action_chunk_stride,
                "sample_weighting": run_cfg.sample_weighting,
                "obs_key": run_cfg.obs_key,
                "action_mode": run_cfg.action_mode,
                "time_shift": run_cfg.time_shift,
                "val_fraction": run_cfg.val_fraction,
                "split_mode": run_cfg.split_mode,
                "obs_normalizer": obs_norm.state_dict(),
                "act_normalizer": act_norm.state_dict(),
                "value_normalizer": value_norm.state_dict() if critic_enabled else None,
                "value_normalizer_count": int(len(train_ds.returns)) if critic_enabled else 0,
                "optimizer_state_dict": opt.state_dict(),
                "model_state_dicts": baked,
                "critic_pretraining": run_cfg.critic_pretraining,
            },
            str(path),
        )

    monitor_loader: DataLoader | None = None
    if run_cfg.monitor_demo_index is not None:
        idx, monitor_demo = _resolve_monitor_demo(run_cfg.dataset, run_cfg.monitor_demo_index)
        log(f"[monitor] Using demo index {idx} -> '{monitor_demo}' for per-epoch L2 reporting.")
        monitor_ds = _build_dataset(
            dataset_path=run_cfg.dataset,
            demo=monitor_demo,
            obs_key=run_cfg.obs_key,
            action_mode=run_cfg.action_mode,
            time_shift=run_cfg.time_shift,
            split="train",
            val_fraction=0.0,
            split_mode=run_cfg.split_mode,
            seed=run_cfg.seed,
        )
        _apply_preprocessing(monitor_ds, run_cfg.obs_history, run_cfg.action_chunk_size, run_cfg.action_chunk_stride)
        monitor_loader = _make_loader(
            monitor_ds,
            run_cfg.batch_size,
            shuffle=False,
            num_workers=run_cfg.num_workers,
            pin_memory=run_cfg.pin_memory,
        )

    for epoch in range(1, run_cfg.num_epochs + 1):
        model.train()
        t0 = time.time()
        total_loss = 0.0
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        n_samples = 0
        n_batches = 0
        total_grad = 0.0

        for batch in train_loader:
            obs, act = batch[:2]
            ret = batch[2] if len(batch) > 2 else None
            obs = obs.to(device)
            act = act.to(device)
            obs_n = obs_norm.normalize(obs)
            act_n = act_norm.normalize(act)

            pred = model(obs_n)
            sample_weights = _sample_weights_from_obs(obs, sample_weight_cfg)
            actor_loss = _weighted_mse_loss(pred, act_n, sample_weights=sample_weights)
            critic_loss = torch.zeros((), device=device)
            if critic_enabled:
                if ret is None:
                    raise RuntimeError("Critic pretraining enabled but batch is missing return targets.")
                ret = ret.to(device)
                if value_norm is None:
                    raise RuntimeError("Critic pretraining enabled but `value_norm` is unavailable.")
                value_target = value_norm.normalize(ret) if critic_cfg.normalize_targets else ret
                value_pred = model.forward_critic(obs_n)
                critic_loss = (value_pred - value_target).pow(2).mean()
                loss = actor_loss + critic_cfg.loss_coef * critic_loss
            else:
                loss = actor_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_before_clip = _grad_norm(model.parameters())
            if run_cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=run_cfg.grad_clip)
            opt.step()

            batch_size = int(obs.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_actor_loss += float(actor_loss.item()) * batch_size
            total_critic_loss += float(critic_loss.item()) * batch_size
            n_samples += batch_size
            total_grad += float(grad_before_clip)
            n_batches += 1

        scheduler.step()
        train_loss = total_loss / max(n_samples, 1)
        train_actor_loss = total_actor_loss / max(n_samples, 1)
        train_critic_loss = total_critic_loss / max(n_samples, 1) if critic_enabled else float("nan")
        train_eval_loss = evaluate(model, train_eval_loader, obs_norm, act_norm, device=device)
        train_eval_l2 = compute_l2_error_per_action(model, train_eval_loader, obs_norm, act_norm, device=device)
        train_eval_value_loss = evaluate_value_loss(model, train_eval_loader, obs_norm, value_norm, device=device)
        monitor_l2 = float("nan")
        if monitor_loader is not None:
            monitor_l2 = compute_l2_error_per_action(model, monitor_loader, obs_norm, act_norm, device=device)
        mean_grad = total_grad / max(n_batches, 1)
        if val_loader is not None:
            val_loss = evaluate(model, val_loader, obs_norm, act_norm, device=device, sample_weight_cfg=val_weight_cfg)
            val_value_loss = evaluate_value_loss(model, val_loader, obs_norm, value_norm, device=device)
        else:
            val_loss = float("nan")
            val_value_loss = float("nan")

        lr_now = float(scheduler.get_last_lr()[0])
        dt = time.time() - t0

        grad_norm_hist.append(mean_grad)
        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)
        train_actor_loss_hist.append(train_actor_loss)
        train_critic_loss_hist.append(train_critic_loss)
        val_critic_loss_hist.append(val_value_loss)

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/train_actor", train_actor_loss, epoch)
        writer.add_scalar("Loss/train_eval", train_eval_loss, epoch)
        writer.add_scalar("L2/train_eval", train_eval_l2, epoch)
        if critic_enabled:
            writer.add_scalar("Loss/train_critic", train_critic_loss, epoch)
            writer.add_scalar("Loss/train_eval_critic", train_eval_value_loss, epoch)
        if monitor_loader is not None:
            writer.add_scalar("L2/monitor_demo", monitor_l2, epoch)
        if val_loader is not None:
            writer.add_scalar("Loss/val", val_loss, epoch)
            if critic_enabled:
                writer.add_scalar("Loss/val_critic", val_value_loss, epoch)
        writer.add_scalar("GradNorm", mean_grad, epoch)
        writer.add_scalar("LR", lr_now, epoch)

        log(
            f"Epoch {epoch:4d}/{run_cfg.num_epochs} | "
            f"train={train_loss:.6f} actor={train_actor_loss:.6f} critic={train_critic_loss:.6f} "
            f"train_eval={train_eval_loss:.6f} train_eval_critic={train_eval_value_loss:.6f} "
            f"train_eval_l2={train_eval_l2:.6f} "
            f"monitor_demo_l2={(monitor_l2 if monitor_loader is not None else float('nan')):.6f} "
            f"val={(val_loss if val_loader is not None else float('nan')):.6f} "
            f"val_critic={(val_value_loss if val_loader is not None else float('nan')):.6f} | "
            f"grad_norm={mean_grad:.4g} lr={lr_now:.2e} | {dt:.1f}s"
        )

        score = val_loss if val_loader is not None else train_loss
        if score < best_score:
            best_score = score
            save_ckpt(ckpt_dir / "best.pth", epoch, train_loss, val_loss)
            if val_loader is not None:
                log(f"  -> new best val_loss={score:.6f}")
            else:
                log(f"  -> new best train_loss={score:.6f} (no val split)")

        if run_cfg.save_every_n_epochs > 0 and epoch % run_cfg.save_every_n_epochs == 0:
            save_ckpt(ckpt_dir / f"epoch_{epoch:04d}.pth", epoch, train_loss, val_loss)

    save_ckpt(ckpt_dir / "final.pth", run_cfg.num_epochs, train_loss_hist[-1], val_loss_hist[-1])
    writer.close()

    # gradient norm plot
    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(1, len(grad_norm_hist) + 1), grad_norm_hist, linewidth=2)
    plt.xlabel("epoch")
    plt.ylabel("grad_norm")
    plt.title("Gradient norm (mean per epoch)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_png = run_dir / "grad_norm.png"
    plt.savefig(out_png, dpi=200)
    plt.close()
    log(f"Saved: {out_png}")

    # combined diagnostics plot: losses + grad norm
    epochs = np.arange(1, len(grad_norm_hist) + 1)
    train_loss_arr = np.array(train_loss_hist, dtype=np.float32)
    val_loss_arr = np.array(val_loss_hist, dtype=np.float32)
    grad_norm_arr = np.array(grad_norm_hist, dtype=np.float32)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(epochs, train_loss_arr, label="train_loss", linewidth=2, color="tab:blue")
    if np.isfinite(val_loss_arr).any():
        ax1.plot(epochs, val_loss_arr, label="val_loss", linewidth=2, color="tab:orange")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(epochs, grad_norm_arr, label="grad_norm", linewidth=2, linestyle="--", color="tab:green")
    ax2.set_ylabel("grad_norm")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    plt.title("Training diagnostics")
    plt.tight_layout()

    diagnostics_png = run_dir / "training_diagnostics.png"
    plt.savefig(diagnostics_png, dpi=200)
    plt.close(fig)
    log(f"Saved: {diagnostics_png}")

    # also save curves for quick inspection
    np.savez(
        run_dir / "curves.npz",
        grad_norm=grad_norm_arr,
        train_loss=train_loss_arr,
        val_loss=val_loss_arr,
    )

    log(f"Run dir: {run_dir}")
    _status_f.close()
    return str(run_dir)


def pick_best_shift(
    run_cfg: RunConfig,
    candidate_shifts: list[int],
    warmup_epochs: int,
    warmup_batch_size: int,
) -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[auto_shift] Device: {device}")
    results: dict[int, float] = {}
    if run_cfg.sample_weighting is None:
        raise ValueError("RunConfig is missing `sample_weighting`.")
    sample_weight_cfg = SampleWeightConfig(**run_cfg.sample_weighting)
    val_weight_cfg = sample_weight_cfg if sample_weight_cfg.enabled and sample_weight_cfg.weight_validation else None

    for shift in candidate_shifts:
        _set_seed(run_cfg.seed)
        train_ds = _build_dataset(
            dataset_path=run_cfg.dataset,
            demo=run_cfg.demo,
            obs_key=run_cfg.obs_key,
            action_mode=run_cfg.action_mode,
            time_shift=shift,
            split="train",
            val_fraction=run_cfg.val_fraction,
            split_mode=run_cfg.split_mode,
            seed=run_cfg.seed,
        )
        val_ds = _build_dataset(
            dataset_path=run_cfg.dataset,
            demo=run_cfg.demo,
            obs_key=run_cfg.obs_key,
            action_mode=run_cfg.action_mode,
            time_shift=shift,
            split="val",
            val_fraction=run_cfg.val_fraction,
            split_mode=run_cfg.split_mode,
            seed=run_cfg.seed,
        )
        _apply_preprocessing(train_ds, run_cfg.obs_history, run_cfg.action_chunk_size, run_cfg.action_chunk_stride)
        _apply_preprocessing(val_ds, run_cfg.obs_history, run_cfg.action_chunk_size, run_cfg.action_chunk_stride)
        if len(val_ds) == 0:
            raise ValueError(
                "time_shift=auto requires a non-empty validation split by demo. "
                "Use a multi-demo dataset without --demo and set val_fraction > 0."
            )
        obs_norm = fit_normalizer(train_ds.obs)
        act_norm = fit_normalizer(train_ds.actions)

        train_loader = _make_loader(
            train_ds, warmup_batch_size, shuffle=True, num_workers=0, pin_memory=False
        )
        train_eval_loader = _make_loader(
            train_ds, warmup_batch_size, shuffle=False, num_workers=0, pin_memory=False
        )
        val_loader = _make_loader(val_ds, warmup_batch_size, shuffle=False, num_workers=0, pin_memory=False)

        model = MLPActorPrior(
            obs_dim=train_ds.obs_dim,
            act_dim=train_ds.act_dim,
            hidden_dims=run_cfg.hidden_dims,
            activation=run_cfg.activation,
            negative_slope=run_cfg.negative_slope,
            dropout=run_cfg.dropout,
        ).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=run_cfg.lr, weight_decay=run_cfg.weight_decay)

        total_loss = 0.0
        n_samples = 0
        for _ in range(warmup_epochs):
            model.train()
            for obs, act in train_loader:
                obs = obs.to(device)
                act = act.to(device)
                pred = model(obs_norm.normalize(obs))
                sample_weights = _sample_weights_from_obs(obs, sample_weight_cfg)
                loss = _weighted_mse_loss(pred, act_norm.normalize(act), sample_weights=sample_weights)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                batch_size = int(obs.shape[0])
                total_loss += float(loss.item()) * batch_size
                n_samples += batch_size

        train_loss = total_loss / max(n_samples, 1)
        train_eval_loss = evaluate(model, train_eval_loader, obs_norm, act_norm, device=device)
        val_loss = evaluate(model, val_loader, obs_norm, act_norm, device=device, sample_weight_cfg=val_weight_cfg)
        results[int(shift)] = float(val_loss)
        print(
            f"[auto_shift] shift={shift:+d} warmup_train_loss={train_loss:.6f} "
            f"warmup_train_eval_loss={train_eval_loss:.6f} "
            f"warmup_val_loss={val_loss:.6f} (train={len(train_ds)} val={len(val_ds)})"
        )

    best_shift = min(results.keys(), key=lambda s: results[s])
    print("[auto_shift] results:", {k: f"{v:.6f}" for k, v in results.items()})
    print(f"[auto_shift] selected time_shift={best_shift:+d}")
    return int(best_shift)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        default=False,
        help="Load the YAML config, print resolved training inputs, and exit before reading datasets.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_cfg, resolved_config_path = _load_yaml_config(args.config)
    base_run_cfg = _run_config_from_yaml(raw_cfg, config_path=resolved_config_path)
    if args.dry_run:
        print(
            "[dry-run] train_bc_prior_mlp.py "
            f"config={resolved_config_path} dataset={base_run_cfg.dataset} "
            f"obs_key={base_run_cfg.obs_key} action_mode={base_run_cfg.action_mode} "
            f"log_dir={base_run_cfg.log_dir}"
        )
        return
    _set_seed(base_run_cfg.seed)

    time_shift_cfg = raw_cfg["dataset"]["time_shift"]
    if time_shift_cfg == "auto":
        if base_run_cfg.val_fraction == 0.0:
            raise ValueError("time_shift=auto requires val_fraction > 0 so shifts can be compared on validation.")
        auto_shift_epochs, auto_shift_batch_size = _auto_shift_settings(raw_cfg)
        chosen_shift = pick_best_shift(
            base_run_cfg,
            candidate_shifts=[0, 1, -1, 2, -2],
            warmup_epochs=auto_shift_epochs,
            warmup_batch_size=auto_shift_batch_size,
        )
    else:
        chosen_shift = int(time_shift_cfg)

    run_cfg = _run_config_from_yaml(raw_cfg, config_path=resolved_config_path, time_shift=chosen_shift)
    resolved_config = _resolved_raw_config(
        raw_cfg, config_path=resolved_config_path, time_shift=chosen_shift, run_cfg=run_cfg
    )
    train_one(run_cfg, resolved_config=resolved_config)


if __name__ == "__main__":
    main()
