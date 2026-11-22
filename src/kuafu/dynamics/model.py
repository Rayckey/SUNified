"""PETS-style Gaussian ensembles with normalized state-delta predictions."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class EnsembleLinear(nn.Module):
    def __init__(self, members: int, inputs: int, outputs: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(members, inputs, outputs))
        self.bias = nn.Parameter(torch.zeros(members, 1, outputs))
        nn.init.trunc_normal_(self.weight, std=1 / (2 * math.sqrt(inputs)), a=-2 / math.sqrt(inputs), b=2 / math.sqrt(inputs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.baddbmm(self.bias, x, self.weight)


class ProbabilisticEnsemble(nn.Module):
    """Independent diagonal-Gaussian networks; leading output dimension is member."""

    def __init__(self, state_dim: int, action_dim: int, *, members: int = 5,
                 hidden_dim: int = 200, hidden_layers: int = 4):
        super().__init__()
        if min(state_dim, action_dim, members, hidden_dim, hidden_layers) < 1:
            raise ValueError("Model dimensions must be positive")
        self.config = dict(state_dim=state_dim, action_dim=action_dim, members=members,
                           hidden_dim=hidden_dim, hidden_layers=hidden_layers)
        self.members = members
        dims = [state_dim + action_dim] + [hidden_dim] * hidden_layers + [2 * state_dim]
        self.layers = nn.ModuleList(EnsembleLinear(members, a, b) for a, b in zip(dims[:-1], dims[1:]))
        # Fixed soft bounds prevent Gaussian variance collapse on constant state coordinates.
        self.register_buffer("min_logvar", torch.full((state_dim,), -10.0))
        self.register_buffer("max_logvar", torch.full((state_dim,), 0.5))
        for name, size, value in (("input_mean", dims[0], 0), ("input_std", dims[0], 1),
                                  ("delta_mean", state_dim, 0), ("delta_std", state_dim, 1)):
            self.register_buffer(name, torch.full((size,), float(value)))

    @torch.no_grad()
    def fit_normalizers(self, states: torch.Tensor, actions: torch.Tensor, next_states: torch.Tensor) -> None:
        """Fit statistics exclusively on the training split."""
        for prefix, values in (("input", torch.cat((states, actions), -1)), ("delta", next_states - states)):
            getattr(self, f"{prefix}_mean").copy_(values.mean(0))
            getattr(self, f"{prefix}_std").copy_(values.std(0, unbiased=False).clamp_min(1e-4))

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return mean/log-variance of normalized deltas for [B,D] or [E,B,D] inputs."""
        x = (torch.cat((states, actions), -1) - self.input_mean) / self.input_std
        if x.ndim == 2:
            x = x.unsqueeze(0).expand(self.members, -1, -1)
        if x.ndim != 3 or x.shape[0] != self.members:
            raise ValueError("Expected [batch, dim] or [members, batch, dim] tensors")
        for layer in self.layers[:-1]:
            x = F.silu(layer(x))
        mean, raw_logvar = self.layers[-1](x).chunk(2, -1)
        logvar = self.max_logvar - F.softplus(self.max_logvar - raw_logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        return mean, logvar

    def nll(self, states: torch.Tensor, actions: torch.Tensor, next_states: torch.Tensor) -> torch.Tensor:
        """Per-member Gaussian negative log likelihood in normalized delta units."""
        mean, logvar = self(states, actions)
        target = (next_states - states - self.delta_mean) / self.delta_std
        return 0.5 * ((target - mean).square() * (-logvar).exp() + logvar + math.log(2 * math.pi)).mean((-2, -1))

    def predict(self, states: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return each member's next-state mean and variance in source coordinates."""
        mean, logvar = self(states, actions)
        return states + self.delta_mean + mean * self.delta_std, logvar.exp() * self.delta_std.square()

    @torch.no_grad()
    def rollout(self, states: torch.Tensor, actions: torch.Tensor, *, particles: int = 20,
                propagation: str = "tsinf", generator: torch.Generator | None = None) -> torch.Tensor:
        """Sample [batch, particles, horizon+1, state_dim] open-loop trajectories.

        TS-infinity keeps a member per particle; TS1 resamples it at each step.
        Actions have shape [batch, horizon, action_dim]. No reference future states
        or SUN Program stage labels are supplied during propagation.
        """
        if propagation not in ("tsinf", "ts1") or particles < 1:
            raise ValueError("Use positive particles and propagation tsinf or ts1")
        if states.ndim != 2 or actions.ndim != 3 or len(states) != len(actions):
            raise ValueError("Expected states [batch, state_dim] and actions [batch, horizon, action_dim]")
        batch, horizon = actions.shape[:2]
        current = states[:, None].expand(-1, particles, -1).reshape(batch * particles, -1)
        member = torch.randint(self.members, (len(current),), device=states.device, generator=generator)
        rows = torch.arange(len(current), device=states.device)
        trajectory = [current.reshape(batch, particles, -1)]
        for step in range(horizon):
            if propagation == "ts1":
                member = torch.randint(self.members, (len(current),), device=states.device, generator=generator)
            action = actions[:, step, None].expand(-1, particles, -1).reshape(batch * particles, -1)
            means, variances = self.predict(current, action)
            noise = torch.randn(current.shape, device=current.device, generator=generator)
            current = means[member, rows] + variances[member, rows].sqrt() * noise
            trajectory.append(current.reshape(batch, particles, -1))
        return torch.stack(trajectory, dim=2)


def load_model(path: str | Path, device: str = "cpu") -> tuple[ProbabilisticEnsemble, dict]:
    """Load a weights-only checkpoint, including normalization and data provenance."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("format") != "kuafu-pets-v1":
        raise ValueError("Unsupported dynamics checkpoint format")
    model = ProbabilisticEnsemble(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    return model.eval(), checkpoint
