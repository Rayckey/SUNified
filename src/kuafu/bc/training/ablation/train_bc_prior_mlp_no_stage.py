#!/usr/bin/env python3
"""Train BC priors while preventing the actor from using the stage scalar.

This is a thin wrapper around ``kuafu.bc.training.train_bc_prior_mlp``.  It keeps
the original observation dimensionality so the resulting checkpoints remain
compatible with the existing prior-loading path, but the stage number is masked
out during supervised training and the exported first-layer stage weights are
zeroed before checkpoint serialization.
"""

from __future__ import annotations

import torch
import torch.nn as nn


from kuafu.bc.training import train_bc_prior_mlp as bc

_ORIGINAL_APPLY_PREPROCESSING = bc._apply_preprocessing
_ORIGINAL_FIT_NORMALIZER = bc.fit_normalizer
_ORIGINAL_EXPORT_BAKED_STATE_DICTS = bc.export_baked_state_dicts
_ORIGINAL_RESOLVED_RAW_CONFIG = bc._resolved_raw_config

_STAGE_MASK_BY_OBS_DIM: dict[int, tuple[int, ...]] = {}


def _stage_indices(obs_dim: int, obs_history: int) -> tuple[int, ...]:
    if obs_history < 1:
        raise ValueError(f"obs_history must be >= 1, got {obs_history}")
    if obs_dim <= 0:
        raise ValueError(f"obs_dim must be positive, got {obs_dim}")
    if obs_history == 1:
        return (obs_dim - 1,)
    if obs_dim % obs_history != 0:
        raise ValueError(
            f"Cannot infer per-frame stage columns: obs_dim={obs_dim} obs_history={obs_history}"
        )
    single_obs_dim = obs_dim // obs_history
    return tuple((frame + 1) * single_obs_dim - 1 for frame in range(obs_history))


def _looks_like_stage_columns(x: torch.Tensor, indices: tuple[int, ...]) -> bool:
    if x.ndim != 2 or not indices or x.shape[0] == 0:
        return False
    cols = x[:, list(indices)]
    if not torch.isfinite(cols).all():
        return False
    return bool(torch.max(torch.abs(cols - torch.round(cols))).item() <= 1e-3)


class StageMaskedNormalizer(bc.Normalizer):
    """Normalizer that zeros stage columns after normalizing observations."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor, stage_indices: tuple[int, ...]):
        super().__init__(mean=mean, std=std)
        self.stage_indices = tuple(int(i) for i in stage_indices)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        out = super().normalize(x)
        if not self.stage_indices:
            return out
        out = out.clone()
        indices = [i if i >= 0 else out.shape[-1] + i for i in self.stage_indices]
        out[..., indices] = 0.0
        return out


def _apply_preprocessing_stage_blind(
    ds,
    obs_history: int,
    action_chunk_size: int,
    action_chunk_stride: int,
) -> None:
    _ORIGINAL_APPLY_PREPROCESSING(ds, obs_history, action_chunk_size, action_chunk_stride)
    indices = _stage_indices(int(ds.obs_dim), int(obs_history))
    _STAGE_MASK_BY_OBS_DIM[int(ds.obs_dim)] = indices


def _fit_normalizer_stage_blind(x: torch.Tensor):
    norm = _ORIGINAL_FIT_NORMALIZER(x)
    if x.ndim == 2:
        indices = _STAGE_MASK_BY_OBS_DIM.get(int(x.shape[1]))
        if indices is not None and _looks_like_stage_columns(x, indices):
            return StageMaskedNormalizer(norm.mean, norm.std, indices)
    return norm


def _first_linear(module: nn.Module) -> nn.Linear | None:
    for child in module.children():
        if isinstance(child, nn.Linear):
            return child
    return None


@torch.no_grad()
def _zero_stage_weights(model: nn.Module, stage_indices: tuple[int, ...]) -> None:
    trunks: list[nn.Module] = []
    if hasattr(model, "actor_trunk"):
        trunks.append(model.actor_trunk)
    elif hasattr(model, "trunk"):
        trunks.append(model.trunk)
    if hasattr(model, "critic_trunk"):
        trunks.append(model.critic_trunk)

    for trunk in trunks:
        first = _first_linear(trunk)
        if first is None:
            continue
        for idx in stage_indices:
            resolved = idx if idx >= 0 else first.weight.shape[1] + idx
            if resolved < 0 or resolved >= first.weight.shape[1]:
                raise IndexError(
                    f"stage index {idx} resolves outside first-layer width {first.weight.shape[1]}"
                )
            first.weight[:, resolved] = 0.0


def _export_baked_state_dicts_stage_blind(model: nn.Module, obs_norm, act_norm):
    stage_indices = tuple(getattr(obs_norm, "stage_indices", ()))
    if stage_indices:
        _zero_stage_weights(model, stage_indices)
    return _ORIGINAL_EXPORT_BAKED_STATE_DICTS(model, obs_norm, act_norm)


def _resolved_raw_config_stage_blind(raw_cfg: dict, *, config_path: str, time_shift: int) -> dict:
    resolved = _ORIGINAL_RESOLVED_RAW_CONFIG(raw_cfg, config_path=config_path, time_shift=time_shift)
    resolved["abla_no_stage_prior"] = {
        "enabled": True,
        "mode": "mask_normalized_stage_and_zero_first_layer_weights",
        "stage_column": "last scalar of each observation-history frame",
        "keeps_observation_dimensionality": True,
    }
    return resolved


def _install_patches() -> None:
    bc._apply_preprocessing = _apply_preprocessing_stage_blind
    bc.fit_normalizer = _fit_normalizer_stage_blind
    bc.export_baked_state_dicts = _export_baked_state_dicts_stage_blind
    bc._resolved_raw_config = _resolved_raw_config_stage_blind


def main() -> None:
    _install_patches()
    bc.main()


if __name__ == "__main__":
    main()
