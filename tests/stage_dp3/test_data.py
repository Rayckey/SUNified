from conftest import require_stage_runtime

require_stage_runtime()

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch
import zarr

from kuafu.data.stage_labels import snapshot_decision_stage
from kuafu.sim2real.dp3.dataset import StageDataset
from kuafu.sim2real.dp3.export import export_dataset, trajectory_splits


def make_episodes(root, count=6):
    result = []
    for episode in range(count):
        path = root / f"episode_{episode}" / f"episode_{episode}.h5"
        path.parent.mkdir(parents=True)
        length = 9
        with h5py.File(path, "w") as handle:
            handle.attrs.update(decision_stage_timing="pre_action", decision_stage_count=3, episode=episode)
            pose = np.tile(np.eye(4, dtype=np.float32), (length, 1, 1))
            pose[:, 0, 3] = np.arange(length) + episode * 10
            handle["action"] = np.c_[pose.reshape(length, 16), np.ones(length)].astype(np.float32)
            rng = np.random.default_rng(episode)
            handle["point_cloud"] = rng.uniform(0.1, 0.9, (length, 16, 6)).astype(np.float32)
            handle["observed_gripper"] = np.linspace(0.1, 0.9, length, dtype=np.float32)
            handle["decision_stage"] = np.repeat(np.arange(3), 3)
            handle["decision_valid"] = np.ones(length, dtype=bool)
            handle["joint_pos"] = np.zeros((length, 9), dtype=np.float32)
            handle["joint_targets"] = np.zeros((length, 11), dtype=np.float32)
        result.append(path)
    return result


@pytest.fixture
def exported(tmp_path):
    paths = make_episodes(tmp_path / "source")
    output = tmp_path / "data.zarr"
    manifest = export_dataset(paths, output, task="test-task", stage_names=["reach", "grasp", "place"], num_points=12)
    return output, manifest, paths


def test_export_labels_follow_observations_and_chunks_use_current_decision(exported):
    output, manifest, _ = exported
    root = zarr.open_group(str(output), "r")
    assert root["data/source_row"][:8].tolist() == list(range(8))
    assert root["data/stage_id"][:8].tolist() == [0, 0, 0, 1, 1, 1, 2, 2]
    assert root["data/state"][0, 0] == 0
    assert root["data/action"][0, 0] == 1
    assert root["data/state"][0, -1] == np.float32(0.1)
    assert root["data/action"][0, -1] == 1
    data = StageDataset(output, horizon=8, n_obs_steps=2, pad_before=1, pad_after=7,
                        action_stride=2, verify_content=True)
    start, end, current = data.sampler.indices[0]
    sample = data[0]
    assert current == start
    assert sample["stage_id"].item() == 0
    assert sample["action_valid"].tolist() == [False, True, True, True, True, False, False, False]
    torch.testing.assert_close(sample["obs"]["agent_pos"][0], sample["obs"]["agent_pos"][1])
    assert root["data/stage_id"][min(current + 6, end - 1)] == 2
    audit = json.loads((output / "stage_transitions.json").read_text())
    assert audit[0]["transitions"][0] == {
        "observation_row": 3, "stage_before": 0, "stage_at_decision": 1, "target_action_row": 4}


def test_trajectory_partition_and_train_only_normalizer(exported):
    output, manifest, _ = exported
    data = StageDataset(output, horizon=8, n_obs_steps=2, pad_before=1, pad_after=7)
    rows = {part: set(data.for_partition(part).sampler.indices[:, 2]) for part in ("train", "validation", "test")}
    assert not rows["train"] & rows["validation"]
    assert not rows["train"] & rows["test"]
    assert not rows["validation"] & rows["test"]
    stats = data.get_normalizer()["agent_pos"].get_input_stats()
    expected = max(manifest["splits"]["train"]) * 10 + 7
    assert stats["max"][0] == expected
    saved = data.get_normalizer().state_dict()
    for part in ("validation", "test"):
        for key, value in data.for_partition(part).get_normalizer().state_dict().items():
            torch.testing.assert_close(value, saved[key])


@pytest.mark.parametrize("action_stride", [1, 3])
@pytest.mark.parametrize("n_obs_steps", [1, 2, 4])
def test_default_windows_keep_short_final_stages_and_mask_padded_targets(exported, action_stride, n_obs_steps):
    output, manifest, _ = exported
    data = StageDataset(output, horizon=16, n_obs_steps=n_obs_steps, action_stride=action_stride)
    for partition, episode_ids in manifest["splits"].items():
        subset = data.for_partition(partition)
        assert len(subset) == 8 * len(episode_ids)
        assert subset.class_counts() == [count * len(episode_ids) for count in (3, 3, 2)]
        last = subset[-1]
        assert last["stage_id"].item() == 2
        assert last["action_valid"].tolist() == [True] * n_obs_steps + [False] * (16 - n_obs_steps)
        assert subset[0]["action_valid"][:n_obs_steps].tolist() == [False] * (n_obs_steps - 1) + [True]


@pytest.mark.parametrize("fault", ["missing", "timing", "length", "range"])
def test_export_rejects_missing_or_misaligned_labels(tmp_path, fault):
    paths = make_episodes(tmp_path / "source", count=3)
    with h5py.File(paths[0], "a") as handle:
        if fault == "missing":
            del handle["decision_stage"]
        elif fault == "timing":
            handle.attrs["decision_stage_timing"] = "post_action"
        elif fault == "length":
            del handle["decision_valid"]
            handle["decision_valid"] = np.ones(8, dtype=bool)
        else:
            handle["decision_stage"][0] = 3
    output = tmp_path / "data.zarr"
    with pytest.raises(ValueError):
        export_dataset(paths, output, task="test", stage_names=["a", "b", "c"], num_points=12)
    assert not output.exists()


def test_output_is_protected_and_content_change_detected(exported):
    output, manifest, paths = exported
    with pytest.raises(FileExistsError):
        export_dataset(paths, output, task="test", stage_names=["a", "b", "c"])
    zarr.open_group(str(output), "a")["data/action"][0, 0] = 999
    with pytest.raises(ValueError, match="content hash"):
        StageDataset(output, horizon=8, n_obs_steps=2, pad_before=1, pad_after=7, verify_content=True)


def test_stage_snapshot_is_independent_of_monitor_advance_and_marks_terminal():
    raw = SimpleNamespace(num_envs=2, device="cpu", current_stage=torch.tensor([[0.], [2.]]),
                          program_runtime=SimpleNamespace(compiled_program=SimpleNamespace(stage_count=lambda: 3)))
    raw.unwrapped = raw
    stage, valid, count = snapshot_decision_stage(raw)
    raw.current_stage.add_(1)
    assert stage.tolist() == [0, 2] and valid.tolist() == [True, True] and count == 3
    later, valid, _ = snapshot_decision_stage(raw)
    assert later.tolist() == [1, -1] and valid.tolist() == [True, False]
    explicit, _, _ = snapshot_decision_stage(raw, 2)
    assert explicit.tolist() == [2, 2]


def test_pointcloud_collector_records_before_step():
    path = Path(__file__).parents[2] / "src/kuafu/data/collection/collect_with_pointcloud.py"
    tree = ast.parse(path.read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and isinstance(n.func.value, ast.Name)]
    store = next(n for n in calls if n.func.value.id == "saver" and n.func.attr == "store")
    step = next(n for n in calls if n.func.value.id == "env" and n.func.attr == "step")
    assert store.lineno < step.lineno
    assert any(k.arg == "decision_stage" for k in store.keywords)


def test_bc_uses_same_row_actions_and_terminal_decisions_are_masked(tmp_path):
    paths = make_episodes(tmp_path / "source")
    for path in paths:
        with h5py.File(path, "a") as handle:
            handle["decision_stage"][-1] = -1
            handle["decision_valid"][-1] = False
    output = tmp_path / "bc.zarr"
    manifest = export_dataset(paths, output, task="test", stage_names=["a", "b", "c"],
                               action_format="bc", num_points=12)
    assert manifest["export"]["action_shift"] == 0
    data = StageDataset(output, horizon=8, n_obs_steps=2, pad_before=1, pad_after=7)
    assert all(data.replay_buffer["decision_valid"][data.sampler.indices[:, 2]])
    assert data[0]["obs"]["agent_pos"].shape == (2, 8)
    assert data[0]["action"].shape == (8, 10)
    end_index = next(i for i, (start, end, current) in enumerate(data.sampler.indices) if current == end - 2)
    assert data[end_index]["stage_id"] == 2
    assert data[end_index]["action_valid"].tolist() == [True, True, False, False, False, False, False, False]


def test_duplicate_trajectories_are_rejected(tmp_path):
    import shutil
    paths = make_episodes(tmp_path / "source", count=3)
    shutil.copyfile(paths[0], paths[1])
    with pytest.raises(ValueError, match="Duplicate"):
        export_dataset(paths, tmp_path / "data.zarr", task="test", stage_names=["a", "b", "c"], num_points=12)
