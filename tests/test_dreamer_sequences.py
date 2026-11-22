"""Check recurrent data boundaries, action timing, and training-only preprocessing."""

import h5py
import numpy as np
import pytest

from kuafu.dynamics.sequences import SequenceDataset, symlog


def make_sequences(path):
    with h5py.File(path, "w") as source:
        data = source.create_group("data")
        for demo in range(12):
            group = data.create_group(f"demo_{demo}")
            state = np.column_stack((np.arange(31), np.full(31, demo), np.zeros(31))).astype(np.float32)
            group["obs/full_state"] = state[:-1]
            group["next_obs/full_state"] = state[1:]
            group["actions/relative_joint"] = np.arange(1, 31, dtype=np.float32)[:, None]
            group["rewards"] = np.arange(1, 31, dtype=np.float32)
            group["dones"] = np.zeros(30, bool)
    return path


def test_previous_action_and_reward_lead_to_current_state(tmp_path):
    data = SequenceDataset(make_sequences(tmp_path / "data.hdf5"))
    batch = data.sample("train", 20, 12, np.random.default_rng(3))
    actions = np.expm1(batch["prev_action"][:, 1:, 0] * data.action_scale[0])
    np.testing.assert_allclose(actions, batch["state"][:, 1:, 0], atol=1e-5)
    np.testing.assert_array_equal(batch["reward"][:, 1:], batch["state"][:, 1:, 0])
    assert (batch["prev_action"][:, 0] == 0).all()
    assert batch["is_first"][:, 0].all() and not batch["is_first"][:, 1:].any()
    assert not batch["reward_mask"][:, 0].any()
    np.testing.assert_array_equal(np.diff(batch["state"][:, :, 0], axis=1), 1)
    np.testing.assert_array_equal(np.diff(batch["state"][:, :, 1], axis=1), 0)
    assert data.metadata["state_dim"] == 2


def test_splits_and_action_scale_exclude_validation_and_test(tmp_path):
    path = make_sequences(tmp_path / "data.hdf5")
    data = SequenceDataset(path)
    names = [name for group in data.split.values() for name in group]
    assert len(names) == len(set(names)) == 12
    with h5py.File(path, "r+") as source:
        for name in data.split["validation"] + data.split["test"]:
            source[f"data/{name}/actions/relative_joint"][:] = 1e6
    changed = SequenceDataset(path)
    np.testing.assert_array_equal(data.action_scale, changed.action_scale)
    assert data.split == changed.split


def test_terminal_autoresets_split_sequences(tmp_path):
    path = make_sequences(tmp_path / "data.hdf5")
    with h5py.File(path, "r+") as source:
        source["data/demo_0/dones"][12] = True
        source["data/demo_0/next_obs/full_state"][12] = 10000
    data = SequenceDataset(path)
    segments = [e for e in data.episodes if e.demonstration == "demo_0"]
    assert len(segments) == 2
    assert segments[0].state[-1, 0] == 12
    assert segments[1].state[0, 0] == 13
    assert data.metadata["excluded_terminal_transitions"] == 1
    assert all(e.state.max() < 10000 for e in segments)


def test_stale_next_obs_and_short_windows_fail(tmp_path):
    path = make_sequences(tmp_path / "data.hdf5")
    data = SequenceDataset(path)
    with pytest.raises(ValueError, match="long enough"):
        data.sample("train", 2, 100, np.random.default_rng(1))
    with h5py.File(path, "r+") as source:
        source["data/demo_0/next_obs/full_state"][:] = source["data/demo_0/obs/full_state"][:]
    with pytest.raises(ValueError, match="alignment"):
        SequenceDataset(path)


def test_signed_log_preserves_command_magnitude_inside_training_range():
    commands = np.array([-90., -3., -1., 0., 1., 3., 90.], np.float32)
    scale = np.max(np.abs(symlog(commands)))
    transformed = symlog(commands) / scale
    assert np.abs(transformed).max() <= 1
    assert np.all(np.diff(transformed) > 0)
    recovered = np.sign(transformed) * np.expm1(np.abs(transformed) * scale)
    np.testing.assert_allclose(commands, recovered, rtol=1e-6)
