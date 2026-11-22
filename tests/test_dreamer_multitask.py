"""Shared-state semantics, task balancing, padding, and held-out data isolation."""

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from kuafu.dynamics.multitask import MultiTaskSequenceDataset, canonical_state, state_indices


def make_multitask_index(directory):
    tasks = {}
    for task, count, length in (("drawer", 1, 12), ("stack3", 3, 20)):
        path = directory / f"{task}.hdf5"
        dim = 46 + 9 * count
        env_name = "Kinova-Table-Gen3-" + ("Drawer" if task == "drawer" else "Stack-3")
        with h5py.File(path, "w") as source:
            data = source.create_group("data")
            data.attrs["env_args"] = json.dumps({"env_name": env_name})
            for demo in range(12):
                group = data.create_group(f"demo_{demo}")
                state = np.zeros((length + 1, dim + 1), np.float32)
                state[:, 0] = demo + 100 * count
                state[:, 9 * (count + 2)] = np.arange(length + 1)
                group["obs/full_state"] = state[:-1]
                group["next_obs/full_state"] = state[1:]
                group["actions/relative_joint"] = np.broadcast_to(
                    np.arange(1, length + 1, dtype=np.float32)[:, None], (length, 9))
                group["rewards"] = np.arange(1, length + 1, dtype=np.float32)
                group["dones"] = np.zeros(length, bool)
        tasks[task] = {"dataset": str(path), "task": env_name,
                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    path = directory / "datasets.json"
    path.write_text(json.dumps({"phase": "complete", "tasks": tasks}))
    return path


def test_robot_columns_align_across_different_object_counts():
    robot = np.r_[np.arange(100, 103), np.arange(200, 206),
                  np.arange(110, 113), np.arange(210, 216), np.arange(300, 327), 0.4]
    for task, objects in (("drawer", 1), ("drawer_stack", 2), ("stack3", 3)):
        position = np.r_[np.arange(objects * 3), np.arange(100, 103), np.arange(110, 113)]
        rotation = np.r_[np.arange(objects * 6), np.arange(200, 206), np.arange(210, 216)]
        native = np.r_[position, rotation, np.arange(300, 327), 0.4].astype(np.float32)
        columns = state_indices(task)
        shared = canonical_state(native, columns)
        np.testing.assert_allclose(shared[:46], robot)
        np.testing.assert_array_equal(shared[columns], native)
        assert not shared[46 + 9 * objects:].any()
        assert len(set(columns)) == len(native)
    with pytest.raises(ValueError, match="audited"):
        state_indices("unknown")


def test_balanced_tasks_keep_action_timing_and_mask_short_episodes(tmp_path):
    data = MultiTaskSequenceDataset(make_multitask_index(tmp_path))
    batch = data.sample("train", 8, 32, np.random.default_rng(3))
    identities = batch["task"].argmax(-1)
    assert np.bincount(identities[:, 0]).tolist() == [4, 4]
    assert (identities == identities[:, :1]).all()
    for i, task_id in enumerate(identities[:, 0]):
        task = data.tasks[task_id]
        valid = batch["time_mask"][i].astype(bool)
        assert valid.sum() == (12 if task == "drawer" else 20)
        assert valid[:valid.sum()].all() and not valid[valid.sum():].any()
        assert not batch["state"][i, ~valid].any()
        assert not batch["reward_mask"][i, ~valid].any()
        np.testing.assert_array_equal(batch["state_mask"][i, 0], data.masks[task])
        np.testing.assert_array_equal(np.diff(batch["state"][i, valid, 46]), 0)
        recovered = np.expm1(batch["prev_action"][i, 1:valid.sum(), 0] * data.action_scale[0])
        joint_column = np.flatnonzero(data.indices[task] == 18).item()
        physical = (batch["state"][i, 1:valid.sum(), 18] * data.state_scale[task][joint_column]
                    + data.state_mean[task][joint_column])
        np.testing.assert_allclose(recovered, physical, atol=1e-5)
        np.testing.assert_allclose(batch["reward"][i, 1:valid.sum()], physical, atol=1e-5)
        assert batch["is_first"][i, 0] and not batch["is_first"][i, 1:].any()
        assert not batch["prev_action"][i, 0].any()
    with pytest.raises(ValueError, match="long enough"):
        data.sample_task("drawer", "test", 2, 50, np.random.default_rng(0))


def test_shared_preprocessing_uses_only_each_tasks_training_demos(tmp_path):
    path = make_multitask_index(tmp_path)
    before = MultiTaskSequenceDataset(path)
    index = json.loads(path.read_text())
    for task, row in index["tasks"].items():
        splits = before.split
        all_names = [name for split in splits.values() for name in split[task]]
        assert len(all_names) == len(set(all_names)) == 12
        with h5py.File(row["dataset"], "r+") as source:
            for name in splits["validation"][task] + splits["test"][task]:
                group = source[f"data/{name}"]
                group["actions/relative_joint"][:] = 1e6
                for key in ("obs/full_state", "next_obs/full_state"):
                    group[key][:] = group[key][:] * 100
        row["sha256"] = hashlib.sha256(Path(row["dataset"]).read_bytes()).hexdigest()
    path.write_text(json.dumps(index))
    after = MultiTaskSequenceDataset(path)
    assert after.split == before.split
    np.testing.assert_array_equal(after.action_scale, before.action_scale)
    for task in before.tasks:
        np.testing.assert_array_equal(after.datasets[task].state_std, before.datasets[task].state_std)
        np.testing.assert_array_equal(after.state_mean[task], before.state_mean[task])
        np.testing.assert_array_equal(after.state_scale[task], before.state_scale[task])


def test_changed_dataset_hash_is_rejected(tmp_path):
    path = make_multitask_index(tmp_path)
    index = json.loads(path.read_text())
    index["tasks"]["drawer"]["sha256"] = "not-the-recorded-hash"
    path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="hash changed"):
        MultiTaskSequenceDataset(path)
