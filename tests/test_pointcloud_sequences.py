"""Point clouds remain visual sets with episode-safe action timing."""

import json

import h5py
import numpy as np
import pytest

from kuafu.dynamics.pointcloud_sequences import PointCloudSequenceDataset, prepare_pointcloud_cache
from kuafu.dynamics.rgb_sequences import file_hash
from kuafu.dynamics.sequences import symlog


def make_pointcloud_index(root, tasks=2, length=12):
    root.mkdir(parents=True, exist_ok=True)
    index = {"format": "kuafu-pointcloud-datasets-v1", "phase": "complete", "tasks": {}}
    for task in range(tasks):
        path = root / f"task_{task}.hdf5"
        with h5py.File(path, "w") as source:
            demos = source.create_group("data")
            demos.attrs["env_args"] = json.dumps({"pointcloud_timing": "pre_action"})
            for episode in range(10):
                n = length + episode
                demo = demos.create_group(f"demo_{episode:02d}")
                demo.attrs.update(success=True, terminal_transition_omitted=True)
                points = np.zeros((n, 64, 3), np.float32)
                points[..., 0] = 0.3 + np.arange(n)[:, None] * 0.001
                points[..., 1] = np.linspace(-0.2, 0.2, 64)
                points[..., 2] = 0.2 + task * 0.1
                demo.create_dataset("obs/pointcloud", data=points)
                # These must never enter the visual model or its data checks.
                demo.create_dataset("obs/full_state", data=np.full((n, 55), np.nan))
                demo.create_dataset("actions/relative_joint", data=np.broadcast_to(np.arange(n)[:, None] + 1, (n, 9)))
                demo.create_dataset("rewards", data=np.arange(n, dtype=np.float32) + 1)
                demo.create_dataset("dones", data=np.zeros(n, bool))
        index["tasks"][f"task_{task}"] = {"path": str(path), "sha256": file_hash(path), "demonstrations": 10}
    path = root / "index.json"
    path.write_text(json.dumps(index))
    return path


def test_point_sequences_are_visual_only_balanced_and_aligned(tmp_path):
    index = make_pointcloud_index(tmp_path)
    cache = tmp_path / "cache"
    prepare_pointcloud_cache(index, cache, num_points=32)
    data = PointCloudSequenceDataset(cache)
    batch = data.sample("train", 4, 32, np.random.default_rng(42))
    np.testing.assert_array_equal(batch["task"][:, 0].argmax(-1), [0, 0, 1, 1])
    assert batch["points"].shape == (4, 32, 32, 3)
    assert set(batch) == {"points", "prev_action", "reward", "reward_mask", "time_mask", "is_first", "task"}
    for i in range(4):
        n = int(batch["time_mask"][i].sum())
        x_meters = batch["points"][i, :n, 0, 0] * 0.85 + 0.65
        t = np.rint((x_meters - 0.3) / 0.001)
        np.testing.assert_array_equal(np.diff(t), 1)
        np.testing.assert_array_equal(batch["reward"][i, :n], t)
        np.testing.assert_allclose(batch["prev_action"][i, 1:n, 0], symlog(t[1:]) / data.action_scale[0], rtol=1e-6)
        assert not batch["points"][i, n:].any()
    for task in data.tasks:
        heldout = set(data.split["test"][task] + data.split["validation"][task])
        for spec in data.metadata["tasks"][task]["episodes"]:
            if spec["name"] in heldout:
                path = cache / spec["values"]
                with np.load(path) as old:
                    actions, rewards = old["previous_action"], old["reward"]
                np.savez(path, previous_action=np.full_like(actions, 1e10), reward=rewards)
    np.testing.assert_array_equal(PointCloudSequenceDataset(cache).action_scale, data.action_scale)
    with pytest.raises(ValueError, match="multiple"):
        data.sample("train", 3, 12, np.random.default_rng(0))


def test_pointcloud_source_hash_is_verified(tmp_path):
    index = make_pointcloud_index(tmp_path)
    with h5py.File(tmp_path / "task_0.hdf5", "r+") as source:
        source["data/demo_00/obs/pointcloud"][0] = 0
    with pytest.raises(ValueError, match="checksum"):
        prepare_pointcloud_cache(index, tmp_path / "cache", num_points=32)
