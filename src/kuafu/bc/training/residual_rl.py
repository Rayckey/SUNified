from __future__ import annotations

from dataclasses import dataclass
import math
import os
from typing import Any, cast

import torch
import torch.nn as nn

from kuafu.bc.training.model import MLPActorPrior
from kuafu.bc.training.normalization import (
    NORMALIZER_EPS, install_observation_normalizer, normalized_actor_state,
)
from kuafu.bc.training.rl_games_prior import (
    _extract_bc_trunk_linear_keys,
    _require_mapping,
    _resolve_prior_checkpoint_path,
)

BC_NORMALIZER_EPS = NORMALIZER_EPS


def _parse_optional_bool(value: Any, *, name: str) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean-like value, got {value!r}.")


def _parse_optional_float(value: Any, *, name: str) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a float-like value, got {value!r}.") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite, got {value!r}.")
    return parsed


@dataclass(frozen=True)
class ResidualRlConfig:
    enabled: bool = False
    freeze_bc: bool = True
    use_bc_rms: bool = True
    use_stage_conditioning: bool = True
    mask_stage_observation: bool = False
    num_stages: int = 7
    alpha_init: float = 0.0
    alpha_final: float = 1.0
    alpha_warmup_epochs: int = 100
    critic_warmup_epochs: int = 10
    fixed_lr: float | None = 3e-5
    use_adaptive_lr: bool = False
    e_clip: float = 0.05
    advantage_clip: float | None = 5.0
    residual_logstd_init: float = -4.0
    residual_logstd_min: float | None = None
    residual_logstd_max: float | None = None
    residual_bounds: tuple[float, ...] = ()
    residual_l2_coef: float = 0.0
    clip_composed_actions: bool = False

    @classmethod
    def from_mapping(
        cls,
        mapping: dict[str, Any] | None,
        *,
        action_dim: int,
        env_num_stages: int,
    ) -> "ResidualRlConfig":
        mapping = mapping or {}
        bounds_raw = mapping.get("residual_bounds", ())
        if isinstance(bounds_raw, (int, float)):
            bounds = tuple(float(bounds_raw) for _ in range(int(action_dim)))
        else:
            bounds = tuple(float(x) for x in bounds_raw)
        if bounds and len(bounds) != int(action_dim):
            raise ValueError(
                "residual_rl.residual_bounds must match the action dimension: "
                f"got {len(bounds)} values for act_dim={action_dim}."
            )
        # num_stages is always sourced from the env's program_runtime.
        # If the yaml still carries a value, only accept it if it matches; mismatches are
        # almost always stale config and would silently corrupt stage one-hot conditioning.
        if int(env_num_stages) <= 0:
            raise ValueError(
                f"residual_rl requires env_num_stages > 0; got {env_num_stages}. "
                "Check program_runtime.compiled_program.stage_count()."
            )
        yaml_num_stages = mapping.get("num_stages", None)
        if yaml_num_stages is not None and int(yaml_num_stages) != int(env_num_stages):
            raise ValueError(
                "residual_rl.num_stages in the yaml disagrees with the env's stage count. "
                f"yaml={int(yaml_num_stages)} env={int(env_num_stages)}. "
                "Remove the yaml entry so num_stages is always taken from the env."
            )
        resolved_num_stages = int(env_num_stages)
        fixed_lr_raw = mapping.get("fixed_lr", None)
        advantage_clip_raw = mapping.get("advantage_clip", None)
        residual_logstd_min = _parse_optional_float(
            mapping.get("residual_logstd_min", None),
            name="residual_rl.residual_logstd_min",
        )
        residual_logstd_max = _parse_optional_float(
            mapping.get("residual_logstd_max", None),
            name="residual_rl.residual_logstd_max",
        )
        if (
            residual_logstd_min is not None
            and residual_logstd_max is not None
            and residual_logstd_min > residual_logstd_max
        ):
            raise ValueError(
                "residual_rl.residual_logstd_min must be <= residual_rl.residual_logstd_max; "
                f"got min={residual_logstd_min} max={residual_logstd_max}."
            )
        clip_composed_actions = _parse_optional_bool(
            mapping.get("clip_composed_actions", None),
            name="residual_rl.clip_composed_actions",
        )
        env_clip_composed_actions = _parse_optional_bool(
            os.getenv("KUAFU_RESIDUAL_CLIP_COMPOSED_ACTIONS"),
            name="KUAFU_RESIDUAL_CLIP_COMPOSED_ACTIONS",
        )
        if env_clip_composed_actions is not None:
            clip_composed_actions = env_clip_composed_actions
        if clip_composed_actions is None:
            clip_composed_actions = False
        mask_stage_observation = _parse_optional_bool(
            mapping.get("mask_stage_observation", None),
            name="residual_rl.mask_stage_observation",
        )
        env_mask_stage_observation = _parse_optional_bool(
            os.getenv("KUAFU_RESIDUAL_MASK_STAGE_OBSERVATION"),
            name="KUAFU_RESIDUAL_MASK_STAGE_OBSERVATION",
        )
        if env_mask_stage_observation is not None:
            mask_stage_observation = env_mask_stage_observation
        if mask_stage_observation is None:
            mask_stage_observation = False
        return cls(
            enabled=bool(mapping.get("enabled", False)),
            freeze_bc=bool(mapping.get("freeze_bc", True)),
            use_bc_rms=bool(mapping.get("use_bc_rms", True)),
            use_stage_conditioning=bool(mapping.get("use_stage_conditioning", True)),
            mask_stage_observation=bool(mask_stage_observation),
            num_stages=resolved_num_stages,
            alpha_init=float(mapping.get("alpha_init", 0.0)),
            alpha_final=float(mapping.get("alpha_final", 1.0)),
            alpha_warmup_epochs=int(mapping.get("alpha_warmup_epochs", 100)),
            critic_warmup_epochs=int(mapping.get("critic_warmup_epochs", 10)),
            fixed_lr=(
                None
                if fixed_lr_raw in (None, "")
                else float(fixed_lr_raw)
            ),
            use_adaptive_lr=bool(mapping.get("use_adaptive_lr", False)),
            e_clip=float(mapping.get("e_clip", 0.05)),
            advantage_clip=(
                None
                if advantage_clip_raw in (None, "")
                else float(advantage_clip_raw)
            ),
            residual_logstd_init=float(mapping.get("residual_logstd_init", -4.0)),
            residual_logstd_min=residual_logstd_min,
            residual_logstd_max=residual_logstd_max,
            residual_bounds=bounds,
            residual_l2_coef=float(mapping.get("residual_l2_coef", 0.0)),
            clip_composed_actions=clip_composed_actions,
        )


def _linear_layers(seq: nn.Module) -> list[nn.Linear]:
    return [m for m in seq.children() if isinstance(m, nn.Linear)]


class FrozenBCPrior(nn.Module):
    """Frozen BC actor with an isolated normalization path."""

    def __init__(
        self,
        *,
        actor: nn.Module,
        obs_mean: torch.Tensor | None,
        obs_std: torch.Tensor | None,
        use_bc_rms: bool,
        action_dim: int,
    ) -> None:
        super().__init__()
        self.actor = actor
        self.use_bc_rms = bool(use_bc_rms)
        self.action_dim = int(action_dim)
        actor_device = next(self.actor.parameters()).device
        if obs_mean is None:
            self.register_buffer("_obs_mean", torch.empty(0, device=actor_device), persistent=False)
            self.register_buffer("_obs_std", torch.empty(0, device=actor_device), persistent=False)
        else:
            if obs_std is None:
                raise ValueError("BC checkpoint is missing obs_std while use_bc_rms=True.")
            self.register_buffer(
                "_obs_mean",
                obs_mean.detach().clone().to(device=actor_device, dtype=torch.float32).flatten(),
                persistent=False,
            )
            self.register_buffer(
                "_obs_std",
                obs_std.detach()
                .clone()
                .to(device=actor_device, dtype=torch.float32)
                .flatten()
                .clamp_min(BC_NORMALIZER_EPS),
                persistent=False,
            )
        for param in self.actor.parameters():
            param.requires_grad_(False)
        self.actor.eval()

    @classmethod
    def from_checkpoint(
        cls,
        prior_ckpt_path: str,
        *,
        action_dim: int,
        use_bc_rms: bool,
        device: torch.device | str,
    ) -> "FrozenBCPrior":
        resolved_ckpt_path, _ = _resolve_prior_checkpoint_path(prior_ckpt_path)
        ckpt = torch.load(resolved_ckpt_path, map_location="cpu", weights_only=False)
        model_state_dicts = _require_mapping(ckpt["model_state_dicts"], "ckpt['model_state_dicts']")
        obs_norm = _require_mapping(ckpt["obs_normalizer"], "ckpt['obs_normalizer']")

        bc_sd = normalized_actor_state(ckpt)

        hidden_dims = tuple(int(v) for v in ckpt.get("hidden_dims", (256, 512, 64)))
        activation = str(ckpt.get("activation", "leaky_relu"))
        negative_slope = float(ckpt.get("negative_slope", 0.01))
        dropout = float(ckpt.get("dropout", 0.0))
        trunk_pairs = _extract_bc_trunk_linear_keys(bc_sd, ("actor_trunk", "trunk"))
        obs_dim = int(ckpt.get("obs_dim", bc_sd[trunk_pairs[0][0]].shape[1]))
        actor = MLPActorPrior(
            obs_dim=obs_dim,
            act_dim=int(action_dim),
            hidden_dims=hidden_dims,
            activation=activation,
            negative_slope=negative_slope,
            dropout=dropout,
        ).to(device)

        actor_linears = _linear_layers(actor.trunk)
        if len(actor_linears) != len(trunk_pairs):
            raise ValueError(
                "BC actor architecture does not match checkpoint trunk layout: "
                f"model_layers={len(actor_linears)} checkpoint_layers={len(trunk_pairs)}."
            )
        for dst, (w_key, b_key) in zip(actor_linears, trunk_pairs, strict=True):
            dst.weight.data.copy_(bc_sd[w_key].to(device=device, dtype=dst.weight.dtype))
            dst.bias.data.copy_(bc_sd[b_key].to(device=device, dtype=dst.bias.dtype))

        mu_weight = bc_sd["mean_layer.weight"]
        mu_bias = bc_sd["mean_layer.bias"]
        bc_act_dim = int(ckpt.get("act_dim", mu_weight.shape[0]))
        if bc_act_dim == int(action_dim):
            pass
        elif bc_act_dim > int(action_dim) and bc_act_dim % int(action_dim) == 0:
            mu_weight = mu_weight[: int(action_dim)]
            mu_bias = mu_bias[: int(action_dim)]
        else:
            raise ValueError(
                "BC prior action dimension does not match env action dimension: "
                f"bc_act_dim={bc_act_dim}, action_dim={action_dim}."
            )
        actor.mean_layer.weight.data.copy_(mu_weight.to(device=device, dtype=actor.mean_layer.weight.dtype))
        actor.mean_layer.bias.data.copy_(mu_bias.to(device=device, dtype=actor.mean_layer.bias.dtype))
        if not use_bc_rms:
            install_observation_normalizer(actor.trunk, obs_norm["mean"], obs_norm["std"])
        return cls(
            actor=actor,
            obs_mean=obs_norm.get("mean"),
            obs_std=obs_norm.get("std"),
            use_bc_rms=use_bc_rms,
            action_dim=int(action_dim),
        )

    @torch.no_grad()
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs_mean = cast(torch.Tensor, self._obs_mean)
        obs_std = cast(torch.Tensor, self._obs_std)
        target_device = obs_mean.device if obs_mean.numel() > 0 else next(self.actor.parameters()).device
        obs = obs.to(device=target_device, dtype=torch.float32)
        if self.use_bc_rms:
            if obs_mean.numel() == 0 or obs_std.numel() == 0:
                raise RuntimeError("BC RMS normalization requested but checkpoint stats are unavailable.")
            obs = (obs - obs_mean) / obs_std
        actions = self.actor(obs)
        return actions[:, : self.action_dim]


class ResidualActionController:
    """Wrapper-side action composition for residual RL."""

    def __init__(
        self,
        *,
        cfg: ResidualRlConfig,
        bc_prior: FrozenBCPrior,
        action_dim: int,
        sim_device: torch.device | str,
    ) -> None:
        self.cfg = cfg
        self.bc_prior = bc_prior
        self.action_dim = int(action_dim)
        if not cfg.residual_bounds:
            raise ValueError("Residual RL requires non-empty per-dimension residual_bounds.")
        bounds = torch.tensor(cfg.residual_bounds, device=sim_device, dtype=torch.float32).view(1, -1)
        self._bounds = bounds
        self._alpha = float(cfg.alpha_init)
        # Stats are kept as GPU tensors (never .item()'d in the rollout hot path) so the
        # per-env-step call to ``compose_env_action`` does not force a GPU->CPU sync.
        self._latest_stats: dict[str, torch.Tensor] = {}

    @property
    def alpha(self) -> float:
        return float(self._alpha)

    @property
    def bounds(self) -> torch.Tensor:
        return self._bounds

    def set_epoch(self, epoch: int) -> float:
        current_epoch = int(epoch)
        if current_epoch <= int(self.cfg.critic_warmup_epochs):
            self._alpha = float(self.cfg.alpha_init)
            return self._alpha
        if int(self.cfg.alpha_warmup_epochs) <= 0:
            self._alpha = float(self.cfg.alpha_final)
            return self._alpha
        progress = (current_epoch - int(self.cfg.critic_warmup_epochs)) / float(self.cfg.alpha_warmup_epochs)
        progress = min(max(progress, 0.0), 1.0)
        self._alpha = float(self.cfg.alpha_init) + progress * (
            float(self.cfg.alpha_final) - float(self.cfg.alpha_init)
        )
        return self._alpha

    def make_stage_one_hot(self, stage: torch.Tensor, *, dtype: torch.dtype, device: torch.device | str) -> torch.Tensor:
        stage_idx = stage.to(device=device, dtype=torch.long).view(-1)
        stage_idx = stage_idx.clamp_(0, int(self.cfg.num_stages) - 1)
        out = torch.zeros((stage_idx.shape[0], int(self.cfg.num_stages)), device=device, dtype=dtype)
        out.scatter_(1, stage_idx.unsqueeze(1), 1.0)
        return out

    def augment_observation(self, obs: torch.Tensor, stage: torch.Tensor) -> torch.Tensor:
        if not self.cfg.use_stage_conditioning:
            return obs
        stage_one_hot = self.make_stage_one_hot(stage, dtype=obs.dtype, device=obs.device)
        return torch.cat([obs, stage_one_hot], dim=-1)

    def bound_residual(self, raw_residual: torch.Tensor) -> torch.Tensor:
        bounds = self._bounds
        if bounds.device != raw_residual.device or bounds.dtype != raw_residual.dtype:
            bounds = bounds.to(device=raw_residual.device, dtype=raw_residual.dtype)
            self._bounds = bounds
        return torch.tanh(raw_residual) * bounds

    def scaled_delta(self, raw_residual: torch.Tensor) -> torch.Tensor:
        return self.bound_residual(raw_residual) * float(self._alpha)

    @torch.no_grad()
    def compose_env_action(
        self,
        *,
        raw_residual_action: torch.Tensor,
        bc_obs: torch.Tensor,
        clip_actions: float,
    ) -> torch.Tensor:
        base_action = self.bc_prior(bc_obs)
        bounded_residual = self.bound_residual(raw_residual_action)
        delta = bounded_residual * float(self._alpha)
        composed = base_action + delta
        clipped = torch.clamp(composed, -float(clip_actions), float(clip_actions))
        would_clip_mask = clipped.ne(composed)
        if self.cfg.clip_composed_actions:
            env_action = clipped
            clip_mask = would_clip_mask
        else:
            env_action = composed
            clip_mask = torch.zeros_like(would_clip_mask, dtype=torch.bool)
        # Stash lightweight GPU-side stats for training-time logging. We deliberately do NOT
        # call ``.item()`` here: this function runs every env step, and each ``.item()`` would
        # force a GPU->CPU sync, which roughly halves rollout FPS. The tensors are consumed
        # once per minibatch inside ``latest_stats`` and logged there.
        denom = self._bounds.clamp_min(1e-6)
        saturation = (bounded_residual.abs() / denom) > 0.95
        self._latest_stats = {
            "delta_from_bc_magnitude": delta.abs().mean().detach(),
            "residual_mean_magnitude": bounded_residual.abs().mean().detach(),
            "residual_saturation_fraction": saturation.to(torch.float32).mean().detach(),
            "action_clipping_rate": clip_mask.to(torch.float32).mean().detach(),
            "action_would_clip_rate": would_clip_mask.to(torch.float32).mean().detach(),
        }
        return env_action

    def latest_stats(self, *, device: torch.device | str) -> dict[str, torch.Tensor]:
        target_device = torch.device(device) if isinstance(device, str) else device
        out: dict[str, torch.Tensor] = {}
        for key, value in self._latest_stats.items():
            if value.device != target_device:
                out[key] = value.to(device=target_device, non_blocking=True)
            else:
                out[key] = value
        return out


def zero_residual_mean_head(agent) -> None:
    net = agent.model.a2c_network
    mu = getattr(net, "mu", None)
    if not isinstance(mu, nn.Linear):
        raise TypeError(f"Residual RL expects nn.Linear mu head, got {type(mu)}")
    with torch.no_grad():
        mu.weight.zero_()
        if mu.bias is not None:
            mu.bias.zero_()


def initialize_residual_logstd(agent, logstd_init: float) -> None:
    net = agent.model.a2c_network
    sigma = getattr(net, "sigma", None)
    with torch.no_grad():
        if isinstance(sigma, nn.Parameter):
            sigma.data.fill_(float(logstd_init))
        elif isinstance(sigma, nn.Linear):
            sigma.weight.zero_()
            if sigma.bias is not None:
                sigma.bias.fill_(float(logstd_init))


def install_residual_logstd_clamp(
    agent,
    *,
    logstd_min: float | None = None,
    logstd_max: float | None = None,
):
    """Clamp residual logstd outputs without changing legacy configs."""
    if logstd_min is None and logstd_max is None:
        return None
    if logstd_min is not None and logstd_max is not None and logstd_min > logstd_max:
        raise ValueError(f"Invalid residual logstd clamp: min={logstd_min} max={logstd_max}.")

    net = agent.model.a2c_network
    sigma = getattr(net, "sigma", None)

    with torch.no_grad():
        if isinstance(sigma, nn.Parameter):
            sigma.data.clamp_(min=logstd_min, max=logstd_max)
        elif isinstance(sigma, nn.Linear) and sigma.bias is not None:
            sigma.bias.data.clamp_(min=logstd_min, max=logstd_max)

    if isinstance(sigma, nn.Module):
        def _clamp_sigma_output(_module, _inputs, output):
            if torch.is_tensor(output):
                return output.clamp(min=logstd_min, max=logstd_max)
            return output

        return sigma.register_forward_hook(_clamp_sigma_output)

    return None


def set_residual_mean_trainable(agent, enabled: bool) -> None:
    net = agent.model.a2c_network
    for module_name in ("actor_mlp", "mu"):
        module = getattr(net, module_name, None)
        if module is None:
            continue
        for param in module.parameters():
            param.requires_grad_(bool(enabled))
