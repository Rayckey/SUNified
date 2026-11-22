"""DreamerV3 image encoder, RSSM, and image decoder without actor or critic."""

import elements
import embodied.jax
import embodied.jax.nets as nn
import jax.numpy as jnp
import ninjax as nj
import numpy as np

from dreamerv3 import rssm
from dreamerv3.agent import Agent


class RGBWorldModel(nj.Module):
    """Predict both camera views from visual context, actions, and task identity."""

    def __init__(self, image_shape, action_dim, task_dim, config, burnin=8):
        self.config, self.burnin, self.task_dim = config, burnin, task_dim
        rgb = {"rgb": elements.Space(np.uint8, tuple(image_shape), 0, 256)}
        task = {"task": elements.Space(np.float32, (task_dim,), 0, 1)}
        actions = {"action": elements.Space(np.float32, (action_dim,), -1, 1), **task}
        self.enc = rssm.Encoder({**rgb, **task}, **config.enc.simple, name="enc")
        self.dyn = rssm.RSSM(actions, **config.dyn.rssm, name="dyn")
        self.dec = rssm.Decoder(rgb, **config.dec.simple, name="dec")
        self.rew = embodied.jax.MLPHead(elements.Space(np.float32, ()), **config.rewhead, name="rew")
        self.con = embodied.jax.MLPHead(elements.Space(bool, (), 0, 2), **config.conhead, name="con")
        self.modules = [self.enc, self.dyn, self.dec, self.rew, self.con]
        self.opt = embodied.jax.Optimizer(self.modules, Agent._make_opt(self, **config.opt), name="opt")

    def loss(self, batch, training=False):
        reset = batch["is_first"]
        size = reset.shape[0]
        images = jnp.where(batch["time_mask"][..., None, None, None] > 0, batch["rgb"], 0)
        _, _, tokens = self.enc(self.enc.initial(size), {"rgb": images, "task": batch["task"]}, reset, training)
        _, _, losses, features, metrics = self.dyn.loss(self.dyn.initial(size), tokens,
            {"action": batch["prev_action"], "task": batch["task"]}, reset, training)
        _, _, recons = self.dec(self.dec.initial(size), features, reset, training)
        losses["rec"] = recons["rgb"].loss(images.astype(jnp.float32) / 255)
        features = jnp.concatenate((nn.cast(features["deter"]),
                                   nn.cast(features["stoch"]).reshape((*reset.shape, -1))), -1)
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

    def forecast(self, context_rgb, previous_actions, future_actions, task):
        """Decode action-conditioned priors as float RGB in [0,1], without future images."""
        size, length = context_rgb.shape[:2]
        reset = jnp.zeros((size, length), bool).at[:, 0].set(True)
        task_context = jnp.broadcast_to(task[:, None], (size, length, self.task_dim))
        _, _, tokens = self.enc(self.enc.initial(size), {"rgb": context_rgb, "task": task_context}, reset, False)
        carry, _, _ = self.dyn.observe(self.dyn.initial(size), tokens,
                                       {"action": previous_actions, "task": task_context}, reset, False)
        horizon = future_actions.shape[1]
        future = {"action": future_actions,
                  "task": jnp.broadcast_to(task[:, None], (size, horizon, self.task_dim))}
        _, features, _ = self.dyn.imagine(carry, future, horizon, False)
        _, _, recons = self.dec(self.dec.initial(size), features, jnp.zeros((size, horizon), bool), False)
        return recons["rgb"].pred()
