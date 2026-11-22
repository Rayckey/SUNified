"""Dynamics data, stochastic propagation, and end-to-end fitting regressions."""

import json

import h5py
import numpy as np
import pytest
import torch

from kuafu.dynamics.data import read_transitions, split_episodes
from kuafu.dynamics.model import ProbabilisticEnsemble, load_model
from kuafu.dynamics.train import TrainConfig, run_training


def make_data(path, episodes=12, length=30):
    rng = np.random.default_rng(12)
    with h5py.File(path, "w") as source:
        data = source.create_group("data")
        for index in range(episodes):
            states = np.zeros((length + 1, 3), dtype=np.float32)
            states[0, :2] = rng.normal(size=2)
            states[:, 2] = index  # Nonphysical SUN stage; excluded from the model.
            actions = rng.normal(size=(length, 1)).astype(np.float32)
            for step in range(length):
                states[step + 1, :2] = 0.8 * states[step, :2] + actions[step] * np.array([0.3, -0.2])
            group = data.create_group(f"demo_{index}")
            group["obs/full_state"] = states[:-1]
            group["next_obs/full_state"] = states[1:]
            group["actions/relative_joint"] = actions
            group["dones"] = np.zeros(length, dtype=bool)
    return path


def test_splits_are_disjoint_and_targets_remain_aligned(tmp_path):
    path = make_data(tmp_path / "data.hdf5")
    data = read_transitions(path)
    split = split_episodes(data, seed=7, val_fraction=0.2, test_fraction=0.2)
    assert data.metadata["source_transitions"] == 360
    assert data.metadata["transitions"] == 348
    assert data.metadata["state_indices"] == [0, 1]
    assert len(data.metadata["sha256"]) == 64
    names = [name for values in split.values() for name in values]
    assert len(names) == len(set(names)) == 12
    assert split == split_episodes(data, seed=7, val_fraction=0.2, test_fraction=0.2)
    states, actions, targets = data.tensors(split["train"])
    torch.testing.assert_close(targets, 0.8 * states + actions * torch.tensor([0.3, -0.2]))


@pytest.mark.parametrize("problem", ["stale", "length", "nan"])
def test_rejects_corrupt_transitions(tmp_path, problem):
    path = make_data(tmp_path / "data.hdf5")
    with h5py.File(path, "r+") as source:
        group = source["data/demo_0"]
        if problem == "stale":
            group["next_obs/full_state"][:] = group["obs/full_state"][:]
        elif problem == "length":
            del group["actions/relative_joint"]
            group["actions/relative_joint"] = np.zeros((2, 1), dtype=np.float32)
        else:
            group["actions/relative_joint"][2, 0] = np.nan
    with pytest.raises(ValueError, match={"stale": "alignment", "length": "lengths", "nan": "finite"}[problem]):
        read_transitions(path)


def test_terminal_autoresets_are_not_training_targets(tmp_path):
    path = make_data(tmp_path / "data.hdf5")
    with h5py.File(path, "r+") as source:
        source["data/demo_0/dones"][5] = True
        source["data/demo_0/next_obs/full_state"][5] = 10000
    data = read_transitions(path)
    episode = next(e for e in data.episodes if e.name == "demo_0")
    assert 5 not in episode.steps.tolist()
    assert episode.next_states.abs().max() < 10


def test_normalization_and_uncertainty_are_in_source_units():
    model = ProbabilisticEnsemble(2, 1, members=3, hidden_dim=8, hidden_layers=1)
    states = torch.tensor([[1., 2.], [3., 4.]])
    actions = torch.tensor([[1.], [-1.]])
    targets = states + torch.tensor([[2., 4.], [4., 8.]])
    model.fit_normalizers(states, actions, targets)
    for parameter in model.parameters():
        parameter.data.zero_()
    mean, logvar = model(states, actions)
    prediction, variance = model.predict(states, actions)
    torch.testing.assert_close(prediction, (states + model.delta_mean).expand(3, -1, -1))
    torch.testing.assert_close(variance, logvar.exp() * model.delta_std.square())
    assert torch.isfinite(model.nll(states, actions, targets)).all()
    assert (logvar >= -10).all() and (logvar <= 0.501).all()


def test_tsinf_keeps_member_and_ts1_resamples():
    model = ProbabilisticEnsemble(1, 1, members=2, hidden_dim=4, hidden_layers=1)

    def deterministic_members(states, actions):
        means = states.unsqueeze(0) + torch.tensor([-1., 1.]).reshape(2, 1, 1)
        return means, torch.zeros_like(means)

    model.predict = deterministic_members
    initial, actions = torch.zeros(1, 1), torch.zeros(1, 12, 1)
    rollout = model.rollout(initial, actions, particles=64, generator=torch.Generator().manual_seed(2))
    increments = rollout.diff(dim=2)
    torch.testing.assert_close(increments, increments[:, :, :1].expand_as(increments))
    assert set(rollout[0, :, -1, 0].tolist()) == {-12., 12.}
    resampled = model.rollout(initial, actions, particles=64, propagation="ts1",
                              generator=torch.Generator().manual_seed(2))
    assert (resampled[0, :, -1, 0].abs() < 12).any()


def test_training_learns_and_checkpoint_reproduces_predictions(tmp_path):
    path = make_data(tmp_path / "data.hdf5", episodes=20, length=40)
    output = tmp_path / "run"
    config = TrainConfig(str(path), str(output), members=3, hidden_dim=32, hidden_layers=2,
                         epochs=35, batch_size=64, learning_rate=0.003, patience=35,
                         rollout_horizon=5, rollout_windows=16, particles=10, threads=1)
    report = run_training(config)
    assert report["test"]["normalized_rmse"] < 0.4 * report["test"]["persistence_normalized_rmse"]
    assert report["test_rollout"]["horizons"]["5"]["normalized_rmse"] < report["test_rollout"]["horizons"]["5"]["persistence_normalized_rmse"]
    model, checkpoint = load_model(output / "best.pt")
    data = read_transitions(path)
    train = data.tensors(checkpoint["run"]["split"]["train"])
    torch.testing.assert_close(model.input_mean, torch.cat(train[:2], -1).mean(0))
    copy, _ = load_model(output / "best.pt")
    for first, second in zip(model.predict(train[0][:10], train[1][:10]), copy.predict(train[0][:10], train[1][:10])):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert json.loads((output / "evaluation.json").read_text()) == report
    with pytest.raises(ValueError, match="new output"):
        run_training(config)
