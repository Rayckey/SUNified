# DP3 integration: upstream copyright (c) 2024 Yanjie Ze, MIT (LICENSES/DP3-MIT.txt).
"""DP3 with decision-stage supervision and optional predicted stage conditioning."""

from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F
from omegaconf import OmegaConf

from diffusion_policy_3d.model.diffusion.conditional_unet1d import ConditionalResidualBlock1D
from diffusion_policy_3d.model.diffusion.simple_conditional_unet1d import (
    ConditionalResidualBlock1D as SimpleConditionalResidualBlock1D,
)
from diffusion_policy_3d.policy.dp3 import DP3
from diffusion_policy_3d.policy.simple_dp3 import SimpleDP3


class _StageConditioning:
    """A: baseline, B: auxiliary head, C: detached soft stage context.

    Stage labels belong in the training batch. Deployment accepts only the
    baseline point cloud and robot state histories.
    """

    def __init__(
        self, *, variant: str, num_stages: int, stage_hidden_dim: int = 64,
        stage_embedding_dim: int = 16, stage_loss_weight: float = 0.1, **kwargs,
    ):
        if variant not in {"A", "B", "C"}:
            raise ValueError("variant must be A, B, or C")
        if num_stages < 2 or min(stage_hidden_dim, stage_embedding_dim) < 1:
            raise ValueError("Use at least two stages and positive head/embedding dimensions")
        if not 0 < stage_loss_weight < float("inf"):
            raise ValueError("stage_loss_weight must be finite and positive")
        if not kwargs.get("obs_as_global_cond", True) or kwargs.get("condition_type", "film") != "film":
            raise ValueError("This experiment uses DP3 global FiLM conditioning")
        for key in ("shape_meta", "pointcloud_encoder_cfg"):
            if isinstance(kwargs.get(key), dict):
                kwargs[key] = OmegaConf.create(kwargs[key])
        super().__init__(**kwargs)
        if not 1 <= self.n_obs_steps <= self.horizon or not 1 <= self.n_action_steps <= self.horizon - self.n_obs_steps + 1:
            raise ValueError("Observation and action horizons exceed the diffusion trajectory")
        self.variant = variant
        self.num_stages = int(num_stages)
        self.stage_loss_weight = float(stage_loss_weight)
        self.stage_embedding_dim = int(stage_embedding_dim)
        self._expanded_weights = set()
        if variant != "A":
            self.stage_head = nn.Sequential(
                nn.Linear(self.obs_feature_dim * self.n_obs_steps, stage_hidden_dim),
                nn.Mish(), nn.Linear(stage_hidden_dim, self.num_stages),
            )
        if variant == "C":
            self.stage_embeddings = nn.Parameter(torch.randn(num_stages, stage_embedding_dim) * 0.02)
            for name, block in self.model.named_modules():
                if not isinstance(block, (ConditionalResidualBlock1D, SimpleConditionalResidualBlock1D)):
                    continue
                old = block.cond_encoder[1]
                if not isinstance(old, nn.Linear):
                    raise TypeError("Unsupported upstream DP3 FiLM projection")
                new = nn.Linear(old.in_features + stage_embedding_dim, old.out_features,
                                bias=old.bias is not None, device=old.weight.device, dtype=old.weight.dtype)
                with torch.no_grad():
                    new.weight.zero_()
                    new.weight[:, :old.in_features].copy_(old.weight)
                    if old.bias is not None:
                        new.bias.copy_(old.bias)
                block.cond_encoder[1] = new
                self._expanded_weights.add(f"model.{name}.cond_encoder.1.weight")
            if not self._expanded_weights:
                raise RuntimeError("No upstream DP3 FiLM projections found")

    def _encode(self, obs):
        if set(obs) != {"point_cloud", "agent_pos"}:
            raise ValueError("Policy observations must contain only point_cloud and agent_pos")
        if any(value.shape[1] != self.n_obs_steps for value in obs.values()):
            raise ValueError("Expected exactly n_obs_steps recent observations")
        normalized = self.normalizer.normalize(obs)
        if not self.use_pc_color:
            normalized["point_cloud"] = normalized["point_cloud"][..., :3]
        batch_size = obs["agent_pos"].shape[0]
        history = {key: value[:, :self.n_obs_steps].reshape(-1, *value.shape[2:])
                   for key, value in normalized.items()}
        return self.obs_encoder(history).reshape(batch_size, -1)

    def _condition(self, features):
        logits = None if self.variant == "A" else self.stage_head(features)
        if self.variant != "C":
            return features, logits
        # Action gradients train embeddings, while CE alone trains stage probabilities.
        context = logits.softmax(-1).detach() @ self.stage_embeddings
        return torch.cat((features, context), dim=-1), logits

    def stage_logits(self, obs):
        if self.variant == "A":
            return None
        return self.stage_head(self._encode(obs))

    def predict_action(self, obs_dict):
        condition, logits = self._condition(self._encode(obs_dict))
        data = condition.new_zeros((condition.shape[0], self.horizon, self.action_dim))
        prediction = self.conditional_sample(
            data, torch.zeros_like(data, dtype=torch.bool), global_cond=condition,
        )
        prediction = self.normalizer["action"].unnormalize(prediction)
        start = self.n_obs_steps - 1
        result = {"action": prediction[:, start:start + self.n_action_steps], "action_pred": prediction}
        if logits is not None:
            result.update(stage_logits=logits, stage_probabilities=logits.softmax(-1))
        return result

    def loss_terms(self, batch):
        """Return differentiable action and stage losses for gradient audits."""
        condition, logits = self._condition(self._encode(batch["obs"]))
        action = self.normalizer["action"].normalize(batch["action"])
        if tuple(action.shape[1:]) != (self.horizon, self.action_dim):
            raise ValueError("Action shape differs from the configured horizon/dimension")
        noise = torch.randn_like(action)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps,
                                  (action.shape[0],), device=action.device).long()
        noisy = self.noise_scheduler.add_noise(action, noise, timesteps)
        prediction = self.model(noisy, timesteps, global_cond=condition)
        prediction_type = self.noise_scheduler.config.prediction_type
        if prediction_type == "sample":
            target = action
        elif prediction_type == "epsilon":
            target = noise
        elif prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(action, noise, timesteps)
        else:
            raise ValueError(f"Unsupported prediction_type: {prediction_type}")
        mask = batch.get("action_valid", torch.ones(action.shape[:2], device=action.device, dtype=torch.bool))
        if mask.shape != action.shape[:2] or mask.dtype != torch.bool or not mask.any(dim=1).all():
            raise ValueError("Each sample needs a boolean action mask with at least one valid target")
        errors = F.mse_loss(prediction, target, reduction="none").mean(dim=-1)
        action_loss = ((errors * mask).sum(dim=1) / mask.sum(dim=1)).mean()
        stage_loss = action_loss.new_zeros(())
        if logits is not None:
            labels = batch["stage_id"]
            if labels.shape != (action.shape[0],) or labels.dtype != torch.long:
                raise ValueError("stage_id must be an integer decision label per sample")
            if ((labels < -1) | (labels >= self.num_stages)).any():
                raise ValueError("Stage labels are outside the declared stage vocabulary")
            valid = labels != -1
            stage_loss = F.cross_entropy(logits[valid], labels[valid]) if valid.any() else logits.sum() * 0
        return action_loss, stage_loss

    def compute_loss(self, batch):
        action_loss, stage_loss = self.loss_terms(batch)
        total = action_loss + self.stage_loss_weight * stage_loss
        return total, {"bc_loss": action_loss.detach().item(), "stage_loss": stage_loss.detach().item()}

    def load_baseline(self, state: Mapping[str, torch.Tensor]):
        """Load shared DP3 weights, zeroing the new conditioning columns in C."""
        if any(key.startswith("stage_") for key in state):
            raise ValueError("Warm start requires a baseline DP3 checkpoint")
        own = self.state_dict()
        shared = {key for key in own if not key.startswith(("stage_", "normalizer."))}
        incoming = {key for key in state if not key.startswith("normalizer.")}
        if shared != incoming:
            raise ValueError(f"Baseline keys differ: {sorted(shared ^ incoming)}")
        expanded = dict(state)
        for key in shared:
            value = state[key]
            if key in self._expanded_weights:
                expected = (own[key].shape[0], own[key].shape[1] - self.stage_embedding_dim)
                if value.shape != expected:
                    raise ValueError(f"Baseline shape mismatch: {key}")
                value = torch.cat((value, value.new_zeros(value.shape[0], self.stage_embedding_dim)), dim=1)
            if value.shape != own[key].shape:
                raise ValueError(f"Baseline shape mismatch: {key}")
            expanded[key] = value
        result = self.load_state_dict(expanded, strict=False)
        if result.unexpected_keys or any(not key.startswith("stage_") for key in result.missing_keys):
            raise ValueError(f"Incomplete baseline: {result}")


class StageDP3(_StageConditioning, DP3):
    """Upstream DP3 with SUN stage supervision and optional stage conditioning."""


class StageSimpleDP3(_StageConditioning, SimpleDP3):
    """Upstream SimpleDP3 with the same SUN stage supervision and conditioning."""
