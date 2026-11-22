"""Visual sequence timing, held-out normalization, task balance, and padding."""

import json

import h5py
import numpy as np
import pytest

from kuafu.dynamics.rgb_sequences import RGBSequenceDataset, file_hash, prepare_rgb_cache, resize_rgb
from kuafu.dynamics.sequences import symlog


def make_rgb_index(root, tasks=2, length=12):
    root.mkdir(exist_ok=True, parents=True)
    index = {"format": "kuafu-rgb-datasets-v1", "phase": "complete", "tasks": {}}
    for task in range(tasks):
        path = root / f"task_{task}.hdf5"
        with h5py.File(path, "w") as output:
            data = output.create_group("data")
            data.attrs["env_args"] = json.dumps({"rgb_timing": "pre_action", "control_dt": 1 / 30})
            for episode in range(10):
                n = length + episode
                demo = data.create_group(f"demo_{episode:02d}")
                demo.attrs.update(success=True, terminal_transition_omitted=True)
                values = np.arange(n, dtype=np.uint8) + 20 * task
                rgb = np.broadcast_to(values[:, None, None, None, None], (n, 2, 4, 8, 3)).copy()
                rgb[:, 1] += 100
                demo.create_dataset("obs/rgb", data=rgb)
                demo.create_dataset("actions/relative_joint", data=np.broadcast_to(np.arange(n)[:, None] + 1, (n, 9)))
                demo.create_dataset("rewards", data=np.arange(n, dtype=np.float32) + 1)
                demo.create_dataset("dones", data=np.zeros(n, bool))
        index["tasks"][f"task_{task}"] = {"path": str(path), "sha256": file_hash(path), "demonstrations": 10}
    path = root / "index.json"
    path.write_text(json.dumps(index))
    return path


def test_rgb_cache_and_sampling_preserve_camera_action_timing(tmp_path):
    index = make_rgb_index(tmp_path)
    cache = tmp_path / "cache"
    prepare_rgb_cache(index, cache, height=48, width=48)
    data = RGBSequenceDataset(cache)
    batch = data.sample("train", 4, 32, np.random.default_rng(42))
    np.testing.assert_array_equal(batch["task"][:, 0].argmax(-1), [0, 0, 1, 1])
    assert batch["rgb"].dtype == np.uint8 and batch["rgb"].shape == (4, 32, 48, 48, 6)
    for i in range(4):
        n = int(batch["time_mask"][i].sum())
        task = int(batch["task"][i, 0].argmax())
        time = batch["rgb"][i, :n, 0, 0, 0].astype(float) - 20 * task
        np.testing.assert_array_equal(np.diff(time), 1)
        np.testing.assert_array_equal(batch["rgb"][i, :n, 0, 0, 3] - batch["rgb"][i, :n, 0, 0, 0], 100)
        np.testing.assert_allclose(batch["prev_action"][i, 1:n, 0], symlog(time[1:]) / data.action_scale[0], rtol=1e-6)
        np.testing.assert_array_equal(batch["reward"][i, :n], time)
        assert not batch["rgb"][i, n:].any() and not batch["reward_mask"][i, 0]
    assert all(len(data.split["train"][t]) == 8 for t in data.tasks)
    assert set(batch) == {"rgb", "prev_action", "reward", "reward_mask", "time_mask", "is_first", "task"}
    with pytest.raises(ValueError, match="multiple"):
        data.sample("train", 3, 32, np.random.default_rng(0))
    with pytest.raises(ValueError, match="long enough"):
        data.sample_task("task_0", "test", 1, 100, np.random.default_rng(0), pad=False)


def test_rgb_action_scale_uses_only_training_episodes(tmp_path):
    index = make_rgb_index(tmp_path)
    cache = tmp_path / "cache"
    prepare_rgb_cache(index, cache, height=48, width=48)
    data = RGBSequenceDataset(cache)
    for task in data.tasks:
        heldout = set(data.split["validation"][task] + data.split["test"][task])
        for spec in data.metadata["tasks"][task]["episodes"]:
            if spec["name"] in heldout:
                path = cache / spec["values"]
                with np.load(path) as old:
                    reward = old["reward"]
                    action = np.full_like(old["previous_action"], 1e10)
                np.savez(path, previous_action=action, reward=reward)
    np.testing.assert_array_equal(RGBSequenceDataset(cache).action_scale, data.action_scale)


def test_rgb_source_checksum_and_camera_layout_are_enforced(tmp_path):
    index = make_rgb_index(tmp_path)
    with h5py.File(tmp_path / "task_0.hdf5", "r+") as source:
        source["data/demo_00/obs/rgb"][0] = 200
    with pytest.raises(ValueError, match="checksum"):
        prepare_rgb_cache(index, tmp_path / "cache")
    with pytest.raises(ValueError, match="uint8"):
        resize_rgb(np.zeros((2, 3, 4, 5)))
