"""DreamerV3 RSSM with a permutation-invariant XYZ encoder and set decoder."""

import elements
import embodied.jax
import embodied.jax.nets as nn
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np

from dreamerv3 import rssm
from dreamerv3.agent import Agent


def chamfer_squared(prediction, target):
    """Symmetric mean squared nearest-neighbor distance, invariant to point order."""
    distance = (jnp.square(prediction).sum(-1)[..., :, None]
                + jnp.square(target).sum(-1)[..., None, :]
                - 2 * jnp.einsum("...ik,...jk->...ij", prediction, target))
    distance = jnp.maximum(distance, 0)
    return 0.5 * (distance.min(-1).mean(-1) + distance.min(-2).mean(-1))


class PointEncoder(nj.Module):
    """Shared pointwise features followed by symmetric max and mean pooling."""

    def __call__(self, points, task):
        x = nn.cast(points)
        for i, units in enumerate((64, 128, 256)):
            x = self.sub(f"point{i}", nn.Linear, units, winit="trunc_normal_in")(x)
            x = jax.nn.silu(self.sub(f"norm{i}", nn.Norm, "rms")(x))
        x = jnp.concatenate((x.max(-2), x.mean(-2), task), -1)
        x = self.sub("pool", nn.Linear, 256, winit="trunc_normal_in")(x)
        return jax.nn.silu(self.sub("poolnorm", nn.Norm, "rms")(x))


class PointDecoder(nj.Module):
    num_points: int

    def __call__(self, features):
        x = nn.cast(features)
        for i in range(3):
            x = self.sub(f"hidden{i}", nn.Linear, 256, winit="trunc_normal_in")(x)
            x = jax.nn.silu(self.sub(f"norm{i}", nn.Norm, "rms")(x))
        x = self.sub("points", nn.Linear, self.num_points * 3, winit="trunc_normal_in")(x)
        return jnp.tanh(x).reshape((*x.shape[:-1], self.num_points, 3))


class PointCloudWorldModel(nj.Module):
    """Shared task-conditioned latent dynamics with XYZ-only visual observations.

    The RSSM, balanced KL, optimizer, reward and continuation heads are upstream
    DreamerV3. The PointNet-style encoder, point-set decoder and Chamfer
    reconstruction replace its image modules for this modality.
    """

    def __init__(self, num_points, action_dim, task_dim, config, burnin=8):
        self.config, self.burnin, self.task_dim, self.num_points = config, burnin, task_dim, num_points
        actions = {"action": elements.Space(np.float32, (action_dim,), -1, 1),
                   "task": elements.Space(np.float32, (task_dim,), 0, 1)}
        self.enc = PointEncoder(name="enc")
        self.dyn = rssm.RSSM(actions, **config.dyn.rssm, name="dyn")
        self.dec = PointDecoder(num_points=num_points, name="dec")
        self.rew = embodied.jax.MLPHead(elements.Space(np.float32, ()), **config.rewhead, name="rew")
        self.con = embodied.jax.MLPHead(elements.Space(bool, (), 0, 2), **config.conhead, name="con")
        self.modules = [self.enc, self.dyn, self.dec, self.rew, self.con]
        self.opt = embodied.jax.Optimizer(self.modules, Agent._make_opt(self, **config.opt), name="opt")

    def _features(self, features):
        deter, stoch = nn.cast((features["deter"], features["stoch"]))
        return jnp.concatenate((deter, stoch.reshape((*deter.shape[:-1], -1))), -1)

    def loss(self, batch, training=False):
        reset = batch["is_first"]
        points = jnp.where(batch["time_mask"][..., None, None] > 0, batch["points"], 0)
        tokens = self.enc(points, batch["task"])
        _, _, losses, features, metrics = self.dyn.loss(self.dyn.initial(reset.shape[0]), tokens,
            {"action": batch["prev_action"], "task": batch["task"]}, reset, training)
        features = self._features(features)
        prediction = self.dec(features)
        # Scale a mean set loss to a per-observation sum, like Dreamer vectors.
        losses["rec"] = chamfer_squared(prediction, points) * self.num_points
        losses["rew"] = self.rew(features, 2).loss(batch["reward"]) * batch["reward_mask"]
        losses["con"] = self.con(features, 2).loss(jnp.ones_like(batch["reward"]))
        mask = batch["time_mask"][:, self.burnin:]
        reduced = {key: ((value[:, self.burnin:] * mask).sum(-1) / jnp.maximum(mask.sum(-1), 1)).mean()
                   for key, value in losses.items()}
        loss = sum(value * self.config.loss_scales[key] for key, value in reduced.items())
        metrics.update({"loss/" + key: value for key, value in reduced.items()})
        metrics["loss/total"] = loss
        return loss.astype(jnp.float32), metrics

    def train(self, batch):
        metrics, losses = self.opt(self.loss, batch, training=True, has_aux=True)
        return {**metrics, **losses}

    def forecast(self, context_points, previous_actions, future_actions, task):
        """Predict future point sets without access to future observations."""
        size, length = context_points.shape[:2]
        resets = jnp.zeros((size, length), bool).at[:, 0].set(True)
        task_context = jnp.broadcast_to(task[:, None], (size, length, self.task_dim))
        tokens = self.enc(context_points, task_context)
        carry, _, _ = self.dyn.observe(self.dyn.initial(size), tokens,
            {"action": previous_actions, "task": task_context}, resets, False)
        horizon = future_actions.shape[1]
        _, features, _ = self.dyn.imagine(carry, {"action": future_actions,
            "task": jnp.broadcast_to(task[:, None], (size, horizon, self.task_dim))}, horizon, False)
        return self.dec(self._features(features))
