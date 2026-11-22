"""Fit and run BC policies under the retained SUN stage predicates."""

import json
import numpy as np
import torch

from kuafu.bc.training.dataset import Normalizer
from kuafu.bc.training.model import MLPActorPrior
from kuafu.data.generated_drawer import OBSERVATION_LAYOUT, policy_observation


def fit_nearest_policy(dataset, checkpoint, seed=42):
    """Fit a stage-conditioned nearest-neighbor BC baseline to whole training episodes."""
    import h5py

    with h5py.File(dataset, "r") as source:
        contract = json.loads(source.attrs["metadata"])
        names = sorted(source["data"])
        if len(names) < 2:
            raise ValueError("BC needs separate training and validation episodes")
        order = np.random.default_rng(seed).permutation(len(names))
        count = max(1, int(np.ceil(len(names) * 0.25)))
        validation_names = [names[i] for i in order[:count]]
        train_names = [names[i] for i in order[count:]]

        def transitions(episodes):
            return tuple(np.concatenate([source[f"data/{name}/{key}"][:] for name in episodes])
                         for key in ("obs/state", "actions/absolute_joint"))

        observations, actions = transitions(train_names)
        validation_obs, validation_actions = transitions(validation_names)
    joints = len(contract["joint_names"])
    cabinet = len(contract["cabinet_joint_names"])
    # Compare measured configurations and handle position; velocities are noisy near contact.
    features = np.r_[0:joints, 2*joints:2*joints+cabinet+3]
    floors = np.r_[np.full(joints, .05), np.full(cabinet + 3, .01)]
    scale = np.maximum(observations[:, features].std(axis=0), floors).astype(np.float32)
    saved = {"generated_drawer_contract": contract, "nearest": {
        "observations": torch.from_numpy(observations), "actions": torch.from_numpy(actions),
        "features": torch.from_numpy(features), "scale": torch.from_numpy(scale),
    }}
    torch.save(saved, checkpoint)
    policy = GeneratedDrawerPolicy(checkpoint, device="cpu", **{key: contract[key] for key in
                                    ("joint_names", "cabinet_joint_names", "asset_hashes", "program_hashes")})
    predicted = np.stack([policy.nearest_action(obs, int(obs[-1])).numpy() for obs in validation_obs])
    return {"method": "nearest_neighbor_bc", "train_episodes": train_names,
            "validation_episodes": validation_names, "training_transitions": len(observations),
            "validation_transitions": len(validation_obs),
            "validation_joint_target_mse": float(np.mean((predicted - validation_actions) ** 2)),
            "contract": contract}


class GeneratedDrawerPolicy:
    def __init__(self, checkpoint, *, device, joint_names, cabinet_joint_names, asset_hashes, program_hashes):
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        contract = saved["generated_drawer_contract"]
        actual = {"joint_names": list(joint_names), "cabinet_joint_names": list(cabinet_joint_names),
                  "asset_hashes": asset_hashes, "program_hashes": program_hashes}
        if any(contract[key] != value for key, value in actual.items()):
            raise ValueError("Policy robot, asset or SUN Program differs from its training contract")
        if contract["observation_layout"] != OBSERVATION_LAYOUT:
            raise ValueError("Policy observation layout differs from the runtime contract")
        self.device = device
        self.nearest = saved.get("nearest")
        if self.nearest is not None:
            self.nearest = {key: value.to(device) for key, value in self.nearest.items()}
            return
        self.single_network = "stages" not in saved
        self.stages = []
        for stage in saved.get("stages", [saved]):
            if stage["action_mode"] not in {"absolute_joint", "relative_joint"} or stage["action_chunk_size"] != 1 or stage["obs_history"] != 1:
                raise ValueError("The generated drawer adapter requires single-step joint targets")
            model = MLPActorPrior(stage["obs_dim"], stage["act_dim"],
                                  tuple(stage["hidden_dims"]), activation=stage["activation"],
                                  negative_slope=stage["negative_slope"], dropout=stage["dropout"]).to(device)
            model.load_state_dict(stage["model_state_dicts"]["model_state_dict_norm"])
            model.eval()
            self.stages.append((model, Normalizer.from_state_dict(stage["obs_normalizer"]),
                                Normalizer.from_state_dict(stage["act_normalizer"]), stage["action_mode"]))

    @torch.no_grad()
    def action(self, snapshot, stage_index, limits):
        if self.nearest is not None:
            action = self.nearest_action(policy_observation(snapshot, stage_index), stage_index)
            return torch.clamp(action, limits[:, 0], limits[:, 1])
        if not self.single_network and not 0 <= stage_index < len(self.stages):
            raise ValueError("Policy has no network for this SUN stage")
        observation = torch.as_tensor(policy_observation(snapshot, stage_index), device=self.device).unsqueeze(0)
        model, obs_normalizer, act_normalizer, mode = self.stages[0 if self.single_network else stage_index]
        action = act_normalizer.denormalize(model(obs_normalizer.normalize(observation)))[0]
        if mode == "relative_joint":
            action = action + observation[0, :len(action)]
        if not torch.isfinite(action).all():
            raise ValueError("Learned policy produced non-finite joint targets")
        return torch.clamp(action, limits[:, 0], limits[:, 1])

    @torch.no_grad()
    def nearest_action(self, observation, stage_index):
        data = self.nearest
        mask = data["observations"][:, -1] == stage_index
        if not mask.any():
            raise ValueError("Training data has no decisions for this SUN stage")
        obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        delta = (data["observations"][mask][:, data["features"]] - obs[data["features"]]) / data["scale"]
        index = delta.square().sum(dim=1).argmin()
        return data["actions"][mask][index]
