from conftest import require_stage_runtime

require_stage_runtime()

from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import torch
import zarr

from kuafu.sim2real.dp3.dataset import StageDataset
from kuafu.sim2real.dp3.export import export_dataset
from kuafu.sim2real.dp3.isaac_adapter import IsaacAdapter
from kuafu.sim2real.dp3.object_frame import ObjectFrame, rotation_from_6d
from kuafu.sim2real.dp3.relative import convert_object_relative
from kuafu.sim2real.dp3.train import load_policy, run_training
from test_data import make_episodes
from test_training import small_experiment


def pose7(rotation, translation):
    return np.r_[translation, np.roll(Rotation.from_matrix(rotation).as_quat(), 1)]


def eff_pose(rotation, translation, gripper=0.4):
    return np.r_[translation, rotation[:, :2].reshape(6), gripper].astype(np.float32)


def absolute_dataset(tmp_path, fault=None):
    paths = make_episodes(tmp_path / "source")
    for i, path in enumerate(paths):
        with h5py.File(path, "a") as handle:
            action = handle["action"][:]
            action[:, 3] *= 0.01
            handle["action"][:] = action
            pose = pose7(Rotation.from_euler("xyz", [0.1, -0.2, i * 0.3]).as_matrix(), [i * 0.1, 0.2, -0.5])
            if fault != "missing":
                handle["articulated_root_pos/cabinet"] = np.tile(pose[:3], (9, 1))
                handle["articulated_root_rot/cabinet"] = np.tile(pose[3:], (9, 1))
                if fault == "moving":
                    handle["articulated_root_pos/cabinet"][-1, 0] += 0.02
                if fault == "quaternion":
                    handle["articulated_root_rot/cabinet"][-1] = 0
    path = tmp_path / "absolute.zarr"
    manifest = export_dataset(paths, path, task="test-task", stage_names=["a", "b", "c"], num_points=12)
    return path, manifest


def test_rotated_frame_has_expected_axes_and_preserves_gripper_and_rgb():
    frame = ObjectFrame([1, 2, 3, np.sqrt(0.5), 0, 0, np.sqrt(0.5)])
    rotation = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    absolute = eff_pose(rotation, [1, 3, 3], 0.23)
    relative = frame.encode_pose(absolute)
    np.testing.assert_allclose(relative, eff_pose(np.eye(3), [1, 0, 0], 0.23), atol=1e-7)
    cloud = np.array([[1, 3, 3, 0.2, 0.3, 0.4]], dtype=np.float32)
    np.testing.assert_allclose(frame.encode_cloud(cloud)[0, :3], [1, 0, 0], atol=1e-7)
    np.testing.assert_array_equal(frame.encode_cloud(cloud)[:, 3:], cloud[:, 3:])
    np.testing.assert_allclose(frame.decode_pose(relative), absolute, atol=1e-7)
    np.testing.assert_allclose(frame.decode_cloud(frame.encode_cloud(cloud)), cloud, atol=1e-7)


def test_global_rigid_motion_does_not_change_relative_policy_inputs_or_targets():
    rng = np.random.default_rng(53)
    obj, ee, global_rotation = Rotation.random(3, random_state=rng).as_matrix()
    obj_pos, ee_pos, shift = rng.normal(size=(3, 3))
    frame = ObjectFrame(pose7(obj, obj_pos))
    changed = ObjectFrame(pose7(global_rotation @ obj, global_rotation @ obj_pos + shift))
    target = eff_pose(ee, ee_pos)
    changed_target = eff_pose(global_rotation @ ee, global_rotation @ ee_pos + shift)
    np.testing.assert_allclose(frame.encode_pose(target), changed.encode_pose(changed_target), atol=3e-7)
    cloud = rng.normal(size=(4, 32, 6)).astype(np.float32)
    changed_cloud = cloud.copy()
    changed_cloud[..., :3] = cloud[..., :3] @ global_rotation.T + shift
    np.testing.assert_allclose(frame.encode_cloud(cloud), changed.encode_cloud(changed_cloud), atol=1e-6)


def test_frame_rejects_moving_roots_invalid_quaternions_and_degenerate_predictions():
    frame = ObjectFrame([0, 0, 0, 1, 0, 0, 0])
    frame.check_stationary([0, 0, 0, -1, 0, 0, 0])
    with pytest.raises(ValueError, match="stationary"):
        frame.check_stationary([0, 0.01, 0, 1, 0, 0, 0])
    with pytest.raises(ValueError, match="stationary"):
        frame.check_stationary(pose7(Rotation.from_euler("z", 0.01).as_matrix(), [0, 0, 0]))
    with pytest.raises(ValueError, match="unit wxyz"):
        ObjectFrame([0, 0, 0, 0, 0, 0, 0])
    with pytest.raises(ValueError, match="Degenerate"):
        frame.decode_pose(np.zeros(10))
    with pytest.raises(ValueError, match="finite"):
        frame.encode_cloud(np.array([[np.nan, 0, 0]]))


def test_conversion_preserves_decision_alignment_and_cross_stage_chunks(tmp_path):
    source, original = absolute_dataset(tmp_path)
    output = tmp_path / "relative.zarr"
    manifest = convert_object_relative(source, output, reference_object="cabinet", chunk_rows=3)
    assert manifest["dataset_id"] != original["dataset_id"]
    assert manifest["conversion"]["source_dataset_id"] == original["dataset_id"]
    assert manifest["splits"] == original["splits"]
    src, dst = (zarr.open_group(str(p), mode="r") for p in (source, output))
    for key in ("source_row", "stage_id", "decision_valid", "img"):
        np.testing.assert_array_equal(src[f"data/{key}"][:], dst[f"data/{key}"][:])
    np.testing.assert_array_equal(src["meta/episode_ends"][:], dst["meta/episode_ends"][:])
    start = 0
    for end in dst["meta/episode_ends"][:]:
        frame = ObjectFrame(dst["data/reference_pose"][start])
        for name in ("state", "agent_pos", "action"):
            np.testing.assert_allclose(frame.decode_pose(dst[f"data/{name}"][start:end]),
                                       src[f"data/{name}"][start:end], atol=2e-6)
        np.testing.assert_allclose(frame.decode_cloud(dst["data/point_cloud"][start:end]),
                                   src["data/point_cloud"][start:end], atol=2e-6)
        start = end
    dataset = StageDataset(output, horizon=8, n_obs_steps=2, action_stride=2, verify_content=True)
    old = StageDataset(source, horizon=8, n_obs_steps=2, action_stride=2)
    for partition in ("train", "validation", "test"):
        data = dataset.for_partition(partition)
        before = old.for_partition(partition)
        assert data.class_counts() == before.class_counts()
        for i in range(len(data)):
            current = data.sampler.indices[i, 2]
            frame = ObjectFrame(dst["data/reference_pose"][current])
            sample = data[i]
            np.testing.assert_allclose(frame.decode_pose(sample["action"]), before[i]["action"], atol=2e-6)
            torch.testing.assert_close(sample["action_valid"], before[i]["action_valid"])
            assert set(sample["obs"]) == {"agent_pos", "point_cloud"}
    with pytest.raises(FileExistsError):
        convert_object_relative(source, output, reference_object="cabinet")
    with pytest.raises(ValueError, match="absolute eff-simple"):
        convert_object_relative(output, tmp_path / "twice.zarr", reference_object="cabinet")


@pytest.mark.parametrize("fault", ["missing", "moving", "quaternion"])
def test_failed_conversion_never_publishes_a_dataset(tmp_path, fault):
    source, _ = absolute_dataset(tmp_path, fault)
    output = tmp_path / "relative.zarr"
    with pytest.raises(ValueError):
        convert_object_relative(source, output, reference_object="cabinet")
    assert not output.exists()


def test_changed_sources_are_rejected(tmp_path):
    source, manifest = absolute_dataset(tmp_path)
    with h5py.File(manifest["sources"][0]["path"], "a") as f:
        f["articulated_root_pos/cabinet"][0, 0] += 1
    with pytest.raises(ValueError, match="trajectory changed"):
        convert_object_relative(source, tmp_path / "relative.zarr", reference_object="cabinet")


def test_live_adapter_uses_environment_local_reference_and_decodes_entire_chunk():
    adapter = object.__new__(IsaacAdapter)
    adapter.relative = {"reference_object": "cabinet"}
    origin = torch.tensor([[12., -8., 0.]])
    pose = np.array([0.3, 0.2, -0.01, np.sqrt(0.5), 0, 0, np.sqrt(0.5)], dtype=np.float32)
    root = torch.tensor(pose).unsqueeze(0)
    root[:, :3] += origin
    class Scene(dict):
        env_origins = origin
    adapter.raw = SimpleNamespace(scene=Scene(cabinet=SimpleNamespace(data=SimpleNamespace(root_state_w=root))))
    adapter.object_frame = ObjectFrame(adapter._reference_pose())
    target = eff_pose(np.eye(3), [0.6, 0.4, 0.1])
    observed = {"point_cloud": np.array([[0.6, 0.4, 0.1]], dtype=np.float32), "agent_pos": target}
    encoded = adapter.encode_observation(observed)
    np.testing.assert_allclose(adapter.decode_action(encoded["agent_pos"]), target, atol=1e-6)
    chunk = np.stack([encoded["agent_pos"]] * 8)
    np.testing.assert_allclose(adapter.decode_action(chunk), np.stack([target] * 8), atol=1e-6)
    root[0, 0] += 0.03
    with pytest.raises(ValueError, match="stationary"):
        adapter.decode_action(chunk)


@pytest.mark.parametrize("simple", [False, True], ids=["DP3", "SimpleDP3"])
def test_relative_dataset_trains_and_restores_its_frame_contract(tmp_path, simple):
    source, _ = absolute_dataset(tmp_path)
    output = tmp_path / "relative.zarr"
    manifest = convert_object_relative(source, output, reference_object="cabinet")
    _, config = small_experiment(tmp_path / "configuration", simple=simple)
    run_training(config, output, tmp_path / "training", variants=["C"])
    policy, saved = load_policy(tmp_path / "training/seed_42/C/best.ckpt")
    assert saved["experiment"]["dataset_manifest"]["export"] == manifest["export"]
    dataset = StageDataset(output, horizon=8, n_obs_steps=2)
    with torch.inference_mode():
        result = policy.predict_action({key: value[None] for key, value in dataset[0]["obs"].items()})
    action = result["action"][0].numpy()
    decoded = ObjectFrame(dataset.replay_buffer["reference_pose"][0]).decode_pose(action)
    assert decoded.shape == (3, 10) and np.isfinite(decoded).all()
    np.testing.assert_allclose(np.linalg.det(rotation_from_6d(decoded[:, 3:9])), 1, atol=1e-6)
