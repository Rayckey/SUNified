"""Numerically stable checkpoint normalization shared by prior consumers."""

from __future__ import annotations

import torch
from torch import nn

NORMALIZER_EPS = 1e-8


class ObservationNormalizedLinear(nn.Linear):
    """Raw-input first layer: center before scaling, never fold means into biases.

    Folding a constant channel with std=1e-8 into float32 W/b creates two huge
    numbers whose subtraction destroys precision. Keeping the centering explicit
    matches the dataset Normalizer, including constant/near-constant channels.
    """

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return super().forward((inputs - self.obs_mean) / self.obs_std)


def install_observation_normalizer(trunk: nn.Module, mean: torch.Tensor, std: torch.Tensor) -> None:
    for name, layer in trunk.named_children():
        if isinstance(layer, nn.Linear):
            if not isinstance(layer, ObservationNormalizedLinear):
                replacement = ObservationNormalizedLinear(layer.in_features, layer.out_features, bias=layer.bias is not None)
                # Keep parameter objects so an existing optimizer remains valid.
                replacement.weight = layer.weight
                replacement.bias = layer.bias
                setattr(trunk, name, replacement)
                layer = replacement
            for key, value in (("obs_mean", mean), ("obs_std", std.clamp_min(NORMALIZER_EPS))):
                layer.register_buffer(key, value.detach().clone().to(layer.weight).flatten())
            return
    raise ValueError("Prior trunk has no Linear layer to normalize")


def normalized_actor_state(checkpoint: dict) -> dict[str, torch.Tensor]:
    """Rebuild the action-denormalized actor from the original training weights.

    This also repairs older checkpoints whose baked action std used 1e-6 while
    their training Normalizer used 1e-8. Existing checkpoints are never rewritten.
    """
    states = checkpoint["model_state_dicts"]
    if "model_state_dict_norm" not in states:
        return dict(states["model_state_dict_prior_norm_obs"])
    state = dict(states["model_state_dict_norm"])
    norm = checkpoint["act_normalizer"]
    weight, bias = state["mean_layer.weight"], state["mean_layer.bias"]
    std = norm["std"].to(weight).clamp_min(NORMALIZER_EPS)
    mean = norm["mean"].to(bias)
    state["mean_layer.weight"] = std[:, None] * weight
    state["mean_layer.bias"] = std * bias + mean
    return state


def install_checkpoint_normalizer_loader(model: nn.Module) -> None:
    """Recreate raw-input layers when restoring into a freshly built RL model.

    The checkpoint carries their buffers, but the ordinary RL-Games builder only
    creates Linear layers. A pre-load hook restores the required module types
    before PyTorch checks/loads those buffers, preserving optimizer parameters.
    """
    if getattr(model, "_bc_normalizer_load_hook", None) is not None:
        return

    def before_load(module, state, prefix, _metadata, _strict, _missing, _unexpected, _errors):
        suffix = ".obs_mean"
        for key in tuple(state):
            if not key.startswith(prefix) or not key.endswith(suffix):
                continue
            layer_path = key[len(prefix):-len(suffix)]
            parent_path, _, _layer_name = layer_path.rpartition(".")
            parent = module.get_submodule(parent_path) if parent_path else module
            std_key = key[:-len(suffix)] + ".obs_std"
            if std_key not in state:
                raise ValueError(f"Checkpoint normalization is incomplete: missing {std_key}")
            install_observation_normalizer(parent, state[key], state[std_key])

    model._bc_normalizer_load_hook = model.register_load_state_dict_pre_hook(before_load)
