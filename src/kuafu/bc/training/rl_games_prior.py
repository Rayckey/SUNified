from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from kuafu.bc.training.normalization import NORMALIZER_EPS, install_observation_normalizer, normalized_actor_state
import torch.nn as nn


@dataclass(frozen=True)
class PriorLoadReport:
    source_path: str
    resolved_checkpoint_path: str
    used_state_dict: str
    copied_tensors: int
    critic_used_state_dict: str | None
    critic_copied_tensors: int
    obs_norm_applied_to_running_mean_std: bool
    value_norm_applied_to_running_mean_std: bool
    normalize_input: bool
    input_running_mean_std_frozen: bool = False
    epoch: int | None = None
    train_loss: float | None = None
    val_loss: float | None = None
    obs_key: str | None = None
    action_mode: str | None = None
    time_shift: int | None = None
    obs_history: int | None = None
    action_chunk_size: int | None = None
    sample_weighting: dict[str, Any] | None = None


def _as_float64(x: torch.Tensor) -> torch.Tensor:
    return x.detach().clone().to(dtype=torch.float64)


def _resolve_prior_checkpoint_path(prior_path: str) -> tuple[str, Path | None]:
    path = Path(prior_path).expanduser()
    if path.is_file():
        return str(path), path.parent.parent if path.parent.name == "checkpoints" else path.parent

    if path.is_dir():
        candidate_dirs = [path]
        if path.name != "checkpoints":
            candidate_dirs.append(path / "checkpoints")
        for base in candidate_dirs:
            for name in ("best.pth", "final.pth"):
                ckpt_path = base / name
                if ckpt_path.is_file():
                    run_dir = base.parent if base.name == "checkpoints" else base
                    return str(ckpt_path), run_dir

    raise FileNotFoundError(
        f"Could not resolve BC prior checkpoint from '{prior_path}'. "
        "Pass either a .pth checkpoint path or a BC run directory containing checkpoints/best.pth."
    )


def _resolve_prior_run_config_path(ckpt_path: str) -> Path:
    ckpt = Path(ckpt_path).resolve()
    run_dir = ckpt.parent.parent if ckpt.parent.name == "checkpoints" else ckpt.parent
    primary_cfg = run_dir / "meta" / "config.yaml"
    fallback_cfg = run_dir / "train_config.yaml"
    if primary_cfg.is_file():
        return primary_cfg
    if fallback_cfg.is_file():
        return fallback_cfg
    raise FileNotFoundError(
        "Could not find prior run config next to BC checkpoint. "
        f"Expected `{primary_cfg}` or fallback `{fallback_cfg}`."
    )


def _load_adjacent_run_config(ckpt_path: str) -> dict[str, Any]:
    import yaml

    config_path = _resolve_prior_run_config_path(ckpt_path)
    with open(config_path, encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    if not isinstance(loaded, dict):
        raise ValueError(f"Prior run config must deserialize to a mapping: {config_path}")
    return loaded


def infer_prior_obs_key(prior_path: str) -> str:
    resolved_ckpt_path, _ = _resolve_prior_checkpoint_path(prior_path)
    ckpt = torch.load(resolved_ckpt_path, map_location="cpu")
    obs_key = ckpt.get("obs_key")
    if obs_key not in ("state", "full_state"):
        run_cfg = _load_adjacent_run_config(resolved_ckpt_path)
        dataset_cfg = run_cfg.get("dataset", {})
        obs_key = dataset_cfg.get("obs_key")
    if obs_key not in ("state", "full_state"):
        raise ValueError(
            "Could not infer BC prior observation mode. Expected checkpoint/config metadata "
            f"to specify `state` or `full_state`, got {obs_key!r}."
        )
    return str(obs_key)


def _nested_get(mapping: dict[str, Any] | None, *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _as_optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _as_optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _require_mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"Expected `{context}` to be a mapping, got {type(value)}")
    return value


def _set_running_mean_std(rms, mean: torch.Tensor, std: torch.Tensor, count: float) -> None:
    # rl_games RunningMeanStd uses buffers: running_mean, running_var, count
    mean64 = _as_float64(mean).flatten()
    var64 = _as_float64(std).flatten().clamp(min=NORMALIZER_EPS).pow(2)
    if rms.running_mean.numel() != mean64.numel():
        raise ValueError(f"RunningMeanStd mean size mismatch: {rms.running_mean.shape} vs {mean64.shape}")
    rms.running_mean.copy_(mean64.view_as(rms.running_mean))
    rms.running_var.copy_(var64.view_as(rms.running_var))
    rms.count.copy_(torch.tensor(float(count), dtype=torch.float64, device=rms.count.device))


def freeze_running_mean_std(rms) -> None:
    """Pin a rl_games `RunningMeanStd` module to whatever stats it currently holds.

    rl_games' `RunningMeanStd.forward` only updates `running_mean / running_var / count`
    when `self.training is True`.  PPO calls `agent.model.train()` before every rollout,
    which recurses into child `nn.Module`s and flips RMS back into training mode -- so
    just calling `rms.eval()` once is insufficient; the next `model.train()` will
    re-arm the running stats.

    This helper puts `rms` in eval mode AND overrides its `train()` method so that any
    subsequent `rms.train(True)` call is coerced to `train(False)`.  That makes the
    stats invariant for the rest of the run (until `unfreeze_running_mean_std` is
    called), so a BC prior normalized against expert stats is not progressively
    invalidated by RMS drifting toward the rollout distribution.

    Idempotent: calling twice is safe.
    """
    if getattr(rms, "_bc_prior_frozen", False):
        return

    rms._bc_prior_original_train = rms.train  # type: ignore[attr-defined]
    rms._bc_prior_original_forward = rms.forward

    def _frozen_train(_self, mode: bool = True):
        return type(rms).train(_self, False)

    import types

    def _frozen_forward(_self, inputs, denorm=False, mask=None):
        # BC training has neither RL-Games' extra variance epsilon nor its
        # +/-5 clipping. Preserve that exact transform while the prior is frozen.
        mean = _self.running_mean.to(inputs)
        std = _self.running_var.sqrt().to(inputs).clamp_min(NORMALIZER_EPS)
        return inputs * std + mean if denorm else (inputs - mean) / std

    rms.train = types.MethodType(_frozen_train, rms)
    rms.forward = types.MethodType(_frozen_forward, rms)
    rms.eval()
    rms._bc_prior_frozen = True  # type: ignore[attr-defined]


def unfreeze_running_mean_std(rms) -> None:
    """Undo `freeze_running_mean_std`, restoring normal train/eval cycling."""
    if not getattr(rms, "_bc_prior_frozen", False):
        return
    original_train = getattr(rms, "_bc_prior_original_train", None)
    if original_train is not None:
        rms.train = original_train  # type: ignore[assignment]
        del rms._bc_prior_original_train
    original_forward = getattr(rms, "_bc_prior_original_forward", None)
    if original_forward is not None:
        rms.forward = original_forward
        del rms._bc_prior_original_forward
    rms._bc_prior_frozen = False


def _linear_layers(seq: nn.Module) -> list[nn.Linear]:
    layers: list[nn.Linear] = []
    for m in seq.children():
        if isinstance(m, nn.Linear):
            layers.append(m)
    return layers

def _extract_bc_trunk_linear_keys(bc_sd: dict[str, torch.Tensor], prefixes: tuple[str, ...]) -> list[tuple[str, str]]:
    """Find all Linear layer weight/bias pairs in a BC trunk state_dict.

    Supports both:
    - old actor layouts:  trunk.0, trunk.2, trunk.4
    - new actor layouts:  actor_trunk.0, actor_trunk.2, actor_trunk.4
    - critic layouts:     critic_trunk.0, critic_trunk.2, critic_trunk.4
    """
    for prefix in prefixes:
        indices: dict[int, set[str]] = {}
        pattern = re.compile(rf"^{re.escape(prefix)}\.(\d+)\.(weight|bias)$")
        for key in bc_sd.keys():
            m = pattern.match(key)
            if not m:
                continue
            idx = int(m.group(1))
            suffix = m.group(2)
            indices.setdefault(idx, set()).add(suffix)

        pairs: list[tuple[str, str]] = []
        for idx in sorted(indices):
            if {"weight", "bias"}.issubset(indices[idx]):
                pairs.append((f"{prefix}.{idx}.weight", f"{prefix}.{idx}.bias"))
        if pairs:
            return pairs

    raise KeyError(
        "Could not find any BC trunk Linear layers in checkpoint. "
        f"Expected one of prefixes={prefixes}. Available keys sample: {list(bc_sd.keys())[:20]}"
    )


def apply_bc_prior_to_rl_games_agent(
    agent,
    prior_ckpt_path: str,
    *,
    set_obs_running_stats: bool = True,
    obs_stats_count: float = 1e6,
    verbose: bool = True,
    load_critic_prior: bool = False,
    require_critic_prior: bool = False,
    freeze_input_running_mean_std: bool = False,
) -> PriorLoadReport:
    """Load a BC prior checkpoint into an RL-Games PPO agent (A2C continuous).

    This copies BC MLP weights into:
    - agent.model.a2c_network.actor_mlp (all Linear layers)
    - agent.model.a2c_network.mu        (final Linear layer)

    It also optionally sets rl-games observation RunningMeanStd statistics to the dataset mean/std
    from the BC checkpoint when normalize_input=True.
    """
    resolved_ckpt_path, _run_dir = _resolve_prior_checkpoint_path(prior_ckpt_path)

    ckpt = torch.load(resolved_ckpt_path, map_location="cpu")
    if "model_state_dicts" not in ckpt:
        raise KeyError("BC checkpoint missing 'model_state_dicts' (expected output of datasets/BC/train_bc_prior_mlp.py).")
    model_state_dicts = _require_mapping(ckpt["model_state_dicts"], "ckpt['model_state_dicts']")
    try:
        run_cfg = _load_adjacent_run_config(resolved_ckpt_path)
    except FileNotFoundError:
        required_metadata = ("obs_history", "action_chunk_size", "time_shift", "obs_key", "action_mode")
        missing_metadata = [key for key in required_metadata if key not in ckpt]
        if missing_metadata:
            raise
        run_cfg = {}
        print(f"BC checkpoint embeds metadata; adjacent run config is not required: {resolved_ckpt_path}")
    if require_critic_prior and not load_critic_prior:
        raise ValueError("`require_critic_prior=True` requires `load_critic_prior=True`.")

    obs_norm = _require_mapping(ckpt["obs_normalizer"], "ckpt['obs_normalizer']")
    act_norm = _require_mapping(ckpt["act_normalizer"], "ckpt['act_normalizer']")
    obs_mean = obs_norm["mean"]
    obs_std = obs_norm["std"]
    critic_pretraining_raw = ckpt.get("critic_pretraining")
    critic_pretraining_cfg = None if critic_pretraining_raw is None else _require_mapping(
        critic_pretraining_raw, "ckpt['critic_pretraining']"
    )
    critic_checkpoint_enabled = bool(
        isinstance(critic_pretraining_cfg, dict) and critic_pretraining_cfg.get("enabled", False)
    )
    value_norm_raw = ckpt.get("value_normalizer")
    value_norm = None if value_norm_raw is None else _require_mapping(value_norm_raw, "ckpt['value_normalizer']")
    value_stats_count_raw = ckpt.get("value_normalizer_count")
    value_stats_count = float(value_stats_count_raw) if value_stats_count_raw is not None else 0.0
    obs_history = _as_optional_int(ckpt.get("obs_history", _nested_get(run_cfg, "preprocessing", "obs_history")))
    action_chunk_size = _as_optional_int(
        ckpt.get("action_chunk_size", _nested_get(run_cfg, "preprocessing", "action_chunk_size"))
    )
    time_shift = _as_optional_int(ckpt.get("time_shift", _nested_get(run_cfg, "dataset", "time_shift")))
    obs_key = ckpt.get("obs_key", _nested_get(run_cfg, "dataset", "obs_key"))
    action_mode = ckpt.get("action_mode", _nested_get(run_cfg, "dataset", "action_mode"))
    sample_weighting = ckpt.get("sample_weighting", _nested_get(run_cfg, "sample_weighting"))

    # Determine whether rl-games normalizes inputs before a2c_network forward.
    normalize_input = bool(getattr(agent.model, "normalize_input", False))
    has_rms = hasattr(agent.model, "running_mean_std")
    has_value_rms = hasattr(agent.model, "value_mean_std")
    obs_norm_applied = False
    value_norm_applied = False

    # Pick which BC weights to load based on rl-games normalization.
    if normalize_input:
        used_key = "model_state_dict_prior_norm_obs"
        critic_used_key = "value_state_dict_norm_obs"
    else:
        used_key = "model_state_dict_prior_raw_obs"
        critic_used_key = "value_state_dict_raw_obs"
    checkpoint_has_critic_prior = critic_checkpoint_enabled or any(
        key in model_state_dicts for key in ("value_state_dict_norm_obs", "value_state_dict_raw_obs")
    )

    if used_key not in model_state_dicts:
        raise KeyError(f"BC checkpoint missing baked weights: model_state_dicts['{used_key}']")

    # Load the canonical training-normalized weights even for raw-input agents;
    # the latter normalize explicitly in their first layer for numerical stability.
    bc_sd = normalized_actor_state(ckpt)
    trunk_pairs = _extract_bc_trunk_linear_keys(bc_sd, ("actor_trunk", "trunk"))
    required = [k for pair in trunk_pairs for k in pair] + ["mean_layer.weight", "mean_layer.bias"]
    for k in required:
        if k not in bc_sd:
            raise KeyError(f"BC baked state dict missing key: {k}")

    net = agent.model.a2c_network
    if not hasattr(net, "actor_mlp") or not hasattr(net, "mu"):
        raise AttributeError("agent.model.a2c_network does not have expected attributes 'actor_mlp' and 'mu'.")

    if not normalize_input:
        install_observation_normalizer(net.actor_mlp, obs_mean, obs_std)
    actor_linears = _linear_layers(net.actor_mlp)
    if len(actor_linears) != len(trunk_pairs):
        raise ValueError(
            "BC prior architecture does not match RL-Games actor MLP: "
            f"bc_hidden_dims={ckpt.get('hidden_dims')} bc_linear_layers={len(trunk_pairs)} "
            f"rl_linear_layers={len(actor_linears)} "
            f"(obs_history={obs_history}, action_chunk_size={action_chunk_size}, time_shift={time_shift}). "
            "Train BC with the same hidden_dims and preprocessing layout as the PPO actor config."
        )
    bc_obs_dim = int(ckpt.get("obs_dim", bc_sd[trunk_pairs[0][0]].shape[1]))
    rl_obs_dim = int(actor_linears[0].weight.shape[1])
    if bc_obs_dim != rl_obs_dim:
        raise ValueError(
            "BC prior observation dimension does not match RL-Games policy input: "
            f"bc_obs_dim={bc_obs_dim} (obs_key={obs_key}, obs_history={obs_history}, "
            f"time_shift={time_shift}) vs rl_obs_dim={rl_obs_dim}. "
            "Train BC with the same observation type as the PPO policy, e.g. use `obs_key: state` "
            "for a 31-D pen-cup PPO policy, and ensure BC preprocessing matches PPO inputs."
        )

    copied = 0
    critic_copied = 0
    device = actor_linears[0].weight.device

    def copy_linear(dst: nn.Linear, w: torch.Tensor, b: torch.Tensor, name: str, *, critic: bool = False):
        nonlocal copied, critic_copied
        if dst.weight.shape != w.shape or dst.bias.shape != b.shape:
            raise ValueError(
                f"Shape mismatch for {name}: "
                f"dst_w={tuple(dst.weight.shape)} src_w={tuple(w.shape)} "
                f"dst_b={tuple(dst.bias.shape)} src_b={tuple(b.shape)}"
            )
        dst.weight.data.copy_(w.to(device=device, dtype=dst.weight.dtype))
        dst.bias.data.copy_(b.to(device=device, dtype=dst.bias.dtype))
        if critic:
            critic_copied += 2
        else:
            copied += 2
        if verbose:
            print(f"[prior] copied {name}: W{tuple(w.shape)} b{tuple(b.shape)}")

    for idx, (dst, (w_key, b_key)) in enumerate(zip(actor_linears, trunk_pairs, strict=True)):
        copy_linear(dst, bc_sd[w_key], bc_sd[b_key], f"actor_mlp.linear{idx}")

    if not isinstance(net.mu, nn.Linear):
        raise TypeError(f"Expected net.mu to be nn.Linear, got {type(net.mu)}")
    bc_act_dim = int(ckpt.get("act_dim", bc_sd["mean_layer.weight"].shape[0]))
    rl_act_dim = int(net.mu.weight.shape[0])
    mu_weight = bc_sd["mean_layer.weight"]
    mu_bias = bc_sd["mean_layer.bias"]
    if bc_act_dim == rl_act_dim:
        pass
    elif bc_act_dim > rl_act_dim and bc_act_dim % rl_act_dim == 0:
        mu_weight = mu_weight[:rl_act_dim]
        mu_bias = mu_bias[:rl_act_dim]
    else:
        raise ValueError(
            "BC prior action dimension does not match RL-Games policy output: "
            f"bc_act_dim={bc_act_dim} (action_mode={action_mode}, action_chunk_size={action_chunk_size}) "
            f"vs rl_act_dim={rl_act_dim}. "
            "Train BC with the same action representation and chunking as the PPO policy."
        )
    copy_linear(net.mu, mu_weight, mu_bias, "mu")

    if load_critic_prior and not checkpoint_has_critic_prior:
        if verbose:
            print(
                "[prior] BC checkpoint has no critic prior payload; "
                "loading actor prior only for backward compatibility."
            )
    elif load_critic_prior:
        if not hasattr(net, "critic_mlp") or not hasattr(net, "value"):
            raise AttributeError(
                "BC checkpoint includes critic pretraining, but RL-Games network is missing `critic_mlp` or `value`."
            )
        if critic_used_key not in model_state_dicts:
            raise KeyError(
                f"BC checkpoint includes critic pretraining but is missing model_state_dicts['{critic_used_key}']."
            )
        if value_norm is None:
            raise KeyError("BC checkpoint includes critic pretraining but is missing `value_normalizer`.")
        if value_stats_count <= 0:
            raise ValueError(
                "BC checkpoint includes critic pretraining but `value_normalizer_count` is missing or non-positive."
            )
        if not has_value_rms:
            raise AttributeError(
                "BC checkpoint includes critic pretraining, but RL-Games model has no `value_mean_std` buffer."
            )
        value_sd = model_state_dicts["value_state_dict_norm_obs"]
        critic_pairs = _extract_bc_trunk_linear_keys(value_sd, ("critic_trunk",))
        critic_required = [k for pair in critic_pairs for k in pair] + ["value_layer.weight", "value_layer.bias"]
        for k in critic_required:
            if k not in value_sd:
                raise KeyError(f"BC value state dict missing key: {k}")

        if not normalize_input:
            install_observation_normalizer(net.critic_mlp, obs_mean, obs_std)
        critic_linears = _linear_layers(net.critic_mlp)
        if len(critic_linears) != len(critic_pairs):
            raise ValueError(
                "BC critic architecture does not match RL-Games critic MLP: "
                f"bc_hidden_dims={ckpt.get('hidden_dims')} bc_linear_layers={len(critic_pairs)} "
                f"rl_linear_layers={len(critic_linears)}"
            )
        for idx, (dst, (w_key, b_key)) in enumerate(zip(critic_linears, critic_pairs, strict=True)):
            copy_linear(dst, value_sd[w_key], value_sd[b_key], f"critic_mlp.linear{idx}", critic=True)

        if not isinstance(net.value, nn.Linear):
            raise TypeError(f"Expected net.value to be nn.Linear, got {type(net.value)}")
        copy_linear(net.value, value_sd["value_layer.weight"], value_sd["value_layer.bias"], "value", critic=True)

        _set_running_mean_std(
            agent.model.value_mean_std,
            mean=value_norm["mean"],
            std=value_norm["std"],
            count=value_stats_count,
        )
        value_norm_applied = True
        if verbose:
            print(
                f"[prior] set value_mean_std from BC return stats: "
                f"mean_shape={tuple(value_norm['mean'].shape)} std_shape={tuple(value_norm['std'].shape)} "
                f"count={value_stats_count:g}"
            )

    input_rms_frozen = False
    if normalize_input and set_obs_running_stats and has_rms:
        rms = agent.model.running_mean_std
        _set_running_mean_std(rms, mean=obs_mean, std=obs_std, count=obs_stats_count)
        obs_norm_applied = True
        if verbose:
            print(
                f"[prior] set running_mean_std from BC stats: "
                f"mean_shape={tuple(obs_mean.shape)} std_shape={tuple(obs_std.shape)} count={obs_stats_count:g}"
            )
        if freeze_input_running_mean_std:
            freeze_running_mean_std(rms)
            input_rms_frozen = True
            if verbose:
                print(
                    "[prior] froze running_mean_std (running_mean/var/count will NOT update during rollout). "
                    "This keeps the normalization seen by the BC weights identical to the BC Normalizer used "
                    "during training. Call kuafu.bc.training.rl_games_prior.unfreeze_running_mean_std(rms) to "
                    "re-enable running updates."
                )
    elif freeze_input_running_mean_std and verbose:
        if not normalize_input:
            print(
                "[prior] freeze_input_running_mean_std=True ignored because normalize_input=False "
                "(raw-obs prior variant bypasses running_mean_std entirely; nothing to freeze)."
            )
        elif not has_rms:
            print(
                "[prior] freeze_input_running_mean_std=True ignored because agent.model has no "
                "running_mean_std buffer."
            )
        elif not set_obs_running_stats:
            print(
                "[prior] freeze_input_running_mean_std=True ignored because set_obs_running_stats=False "
                "(we cannot freeze RMS without first seeding it with BC stats)."
            )

    return PriorLoadReport(
        source_path=str(prior_ckpt_path),
        resolved_checkpoint_path=resolved_ckpt_path,
        used_state_dict=used_key,
        copied_tensors=copied,
        critic_used_state_dict=critic_used_key if critic_copied > 0 else None,
        critic_copied_tensors=critic_copied,
        obs_norm_applied_to_running_mean_std=obs_norm_applied,
        value_norm_applied_to_running_mean_std=value_norm_applied,
        normalize_input=normalize_input,
        input_running_mean_std_frozen=input_rms_frozen,
        epoch=_as_optional_int(ckpt.get("epoch")),
        train_loss=_as_optional_float(ckpt.get("train_loss")),
        val_loss=_as_optional_float(ckpt.get("val_loss")),
        obs_key=obs_key,
        action_mode=action_mode,
        time_shift=time_shift,
        obs_history=obs_history,
        action_chunk_size=action_chunk_size,
        sample_weighting=sample_weighting if isinstance(sample_weighting, dict) else None,
    )
