"""World-model-only wrapper around the pinned upstream DreamerV3 components."""

import elements
import embodied.jax
import embodied.jax.nets as nn
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np

from dreamerv3 import rssm
from dreamerv3.agent import Agent


def state_prediction(reconstruction, symlog):
    """Invert the upstream vector head's signed-log output into observation units."""
    prediction = reconstruction.pred()
    return nn.symexp(prediction) if symlog else prediction


class WorldModel(nj.Module):
    """Upstream encoder/RSSM/decoder/reward/continuation heads without actor/critic."""

    def __init__(self, state_dim, action_dim, config, burnin=8, task_dim=0):
        self.config = config
        self.burnin = burnin
        self.task_dim = task_dim
        obs_space = {"state": elements.Space(np.float32, (state_dim,))}
        act_space = {"action": elements.Space(np.float32, (action_dim,), -1, 1)}
        encoder_space = dict(obs_space)
        if task_dim:
            encoder_space["task"] = elements.Space(np.float32, (task_dim,), 0, 1)
            encoder_space["state_mask"] = elements.Space(np.float32, (state_dim,), 0, 1)
            act_space["task"] = elements.Space(np.float32, (task_dim,), 0, 1)
        self.enc = rssm.Encoder(encoder_space, **config.enc.simple, name="enc")
        self.dyn = rssm.RSSM(act_space, **config.dyn.rssm, name="dyn")
        self.dec = rssm.Decoder(obs_space, **config.dec.simple, name="dec")
        self.rew = embodied.jax.MLPHead(elements.Space(np.float32, ()), **config.rewhead, name="rew")
        self.con = embodied.jax.MLPHead(elements.Space(bool, (), 0, 2), **config.conhead, name="con")
        self.modules = [self.enc, self.dyn, self.dec, self.rew, self.con]
        self.opt = embodied.jax.Optimizer(self.modules, Agent._make_opt(self, **config.opt), name="opt")

    def loss(self, batch, training=False):
        """Upstream KL-balanced and symlog reconstruction losses, after burn-in."""
        reset = batch["is_first"]
        size = reset.shape[0]
        observations = {"state": batch["state"]}
        actions = {"action": batch["prev_action"]}
        if self.task_dim:
            observations = {"state": batch["state"] * batch["state_mask"] * batch["time_mask"][..., None],
                            "task": batch["task"], "state_mask": batch["state_mask"]}
            actions["task"] = batch["task"]
        _, _, tokens = self.enc(self.enc.initial(size), observations, reset, training)
        _, _, losses, features, metrics = self.dyn.loss(
            self.dyn.initial(size), tokens, actions, reset, training)
        _, _, recons = self.dec(self.dec.initial(size), features, reset, training)
        features = jnp.concatenate((nn.cast(features["deter"]),
                                   nn.cast(features["stoch"]).reshape((*reset.shape, -1))), -1)
        if self.task_dim:
            # Pinned upstream returns Agg(MSE): access per-coordinate losses
            # before aggregation so absent object slots never become targets.
            element_loss = recons["state"].output.loss(batch["state"])
            mask = batch["state_mask"]
            losses["rec"] = (element_loss * mask).sum(-1) * mask.shape[-1] / mask.sum(-1)
        else:
            losses["rec"] = recons["state"].loss(batch["state"])
        losses["rew"] = self.rew(features, 2).loss(batch["reward"]) * batch["reward_mask"]
        # Dataset terminal/autoreset rows were excluded; do not invent success labels.
        losses["con"] = self.con(features, 2).loss(jnp.ones_like(batch["reward"]))
        if self.task_dim:
            mask = batch["time_mask"][:, self.burnin:]
            # Normalize each sequence independently: short tasks get equal
            # weight instead of being diluted by their masked padding.
            reduced = {key: ((value[:, self.burnin:] * mask).sum(-1)
                             / jnp.maximum(mask.sum(-1), 1)).mean()
                       for key, value in losses.items()}
        else:
            reduced = {key: value[:, self.burnin:].mean() for key, value in losses.items()}
        loss = sum(value * self.config.loss_scales[key] for key, value in reduced.items())
        metrics.update({"loss/" + key: value for key, value in reduced.items()})
        metrics["loss/total"] = loss
        return loss.astype(jnp.float32), metrics

    def train(self, batch):
        metrics, losses = self.opt(self.loss, batch, training=True, has_aux=True)
        return {**metrics, **losses}

    def forecast(self, context, prev_actions, future_actions, task=None, state_mask=None):
        """Observe context once, then decode priors driven solely by future actions.

        Action tensors must use the saved preprocessing transform. Returns
        [batch, horizon, state_dim] predictions in model observation coordinates;
        the public loader restores any dataset normalization.
        """
        size, length = context.shape[:2]
        resets = jnp.zeros((size, length), bool).at[:, 0].set(True)
        observations = {"state": context}
        actions = {"action": prev_actions}
        future = {"action": future_actions}
        if self.task_dim:
            observations = {"state": context * state_mask[:, None],
                            "state_mask": jnp.broadcast_to(state_mask[:, None], context.shape),
                            "task": jnp.broadcast_to(task[:, None], (size, length, self.task_dim))}
            actions["task"] = observations["task"]
            future["task"] = jnp.broadcast_to(task[:, None], (size, future_actions.shape[1], self.task_dim))
        _, _, tokens = self.enc(self.enc.initial(size), observations, resets, False)
        carry, _, _ = self.dyn.observe(self.dyn.initial(size), tokens,
                                       actions, resets, False)
        horizon = future_actions.shape[1]
        _, features, _ = self.dyn.imagine(carry, future, horizon, False)
        resets = jnp.zeros((size, horizon), bool)
        _, _, recons = self.dec(self.dec.initial(size), features, resets, False)
        prediction = state_prediction(recons["state"], self.dec.symlog)
        return prediction * state_mask[:, None] if self.task_dim else prediction
