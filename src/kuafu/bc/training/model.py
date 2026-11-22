from __future__ import annotations

import torch
import torch.nn as nn


def _make_activation(activation: str, negative_slope: float) -> callable:
    act = activation.lower()
    if act == "leaky_relu":
        return lambda: nn.LeakyReLU(negative_slope=negative_slope)
    if act == "relu":
        return lambda: nn.ReLU()
    raise ValueError("activation must be one of: leaky_relu, relu")


def _build_trunk(
    input_dim: int,
    hidden_dims: tuple[int, ...],
    *,
    activation: str,
    negative_slope: float,
    dropout: float,
) -> nn.Sequential:
    if dropout < 0.0 or dropout >= 1.0:
        raise ValueError("dropout must be in [0, 1).")
    if len(hidden_dims) == 0:
        raise ValueError("hidden_dims must contain at least one hidden layer.")

    act_layer = _make_activation(activation, negative_slope)
    use_dropout = dropout > 0.0
    layers: list[nn.Module] = []
    dims = [input_dim, *hidden_dims]
    for in_dim, out_dim in zip(dims[:-1], dims[1:], strict=True):
        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(act_layer())
        if use_dropout:
            layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)


class MLPActorPrior(nn.Module):
    """MLP policy prior: obs -> action_mean.

    Hidden sizes and activation are configurable.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims: tuple[int, ...] = (256, 512, 64),
        activation: str = "leaky_relu",
        negative_slope: float = 0.01,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.trunk = _build_trunk(
            obs_dim,
            hidden_dims,
            activation=activation,
            negative_slope=negative_slope,
            dropout=dropout,
        )
        self.mean_layer = nn.Linear(hidden_dims[-1], act_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mean_layer(self.trunk(obs))


class MLPActorCriticPrior(nn.Module):
    """Separate actor/critic priors matching RL-Games `separate: True` layouts."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims: tuple[int, ...] = (256, 512, 64),
        activation: str = "leaky_relu",
        negative_slope: float = 0.01,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.actor_trunk = _build_trunk(
            obs_dim,
            hidden_dims,
            activation=activation,
            negative_slope=negative_slope,
            dropout=dropout,
        )
        self.mean_layer = nn.Linear(hidden_dims[-1], act_dim)
        self.critic_trunk = _build_trunk(
            obs_dim,
            hidden_dims,
            activation=activation,
            negative_slope=negative_slope,
            dropout=dropout,
        )
        self.value_layer = nn.Linear(hidden_dims[-1], 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.forward_actor(obs)

    def forward_actor(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mean_layer(self.actor_trunk(obs))

    def forward_critic(self, obs: torch.Tensor) -> torch.Tensor:
        return self.value_layer(self.critic_trunk(obs))

