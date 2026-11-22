"""Regression checks for LiLa-WAM timing, split isolation, and control encoding."""

import json

import h5py
import numpy as np
import pytest

from kuafu.dynamics.lilawam_data import (
    KinovaLiLaDataset, MEAN, STATE_DIMS, STD, camera_image, decode_actions,
    encode_actions, file_hash, joint_positions, prepare_data, write_json,
)
from kuafu.dynamics.lilawam_runtime import balanced_indices


@pytest.fixture
def recordings(tmp_path):
    index = {"format": "kuafu-rgb-datasets-v1", "phase": "complete", "tasks": {}}
    splits = {split: {} for split in ("train", "validation", "test")}
    for task, dimension in STATE_DIMS.items():
        source = tmp_path / f"{task}.hdf5"
        with h5py.File(source, "w") as handle:
            data = handle.create_group("data")
            data.attrs["env_args"] = json.dumps({"rgb_timing": "pre_action", "control_dt": 1 / 30,
                                                  "camera_keys": ["table_cam_0", "table_cam_1"]})
            for number, split in enumerate(splits):
                name = f"demo_{number}"
                splits[split][task] = [name]
                demo = data.create_group(name)
                demo.attrs.update(success=True, terminal_transition_omitted=True)
                frames = np.empty((20, 2, 72, 128, 3), np.uint8)
                frames[:, 0] = np.arange(20)[:, None, None, None]
                frames[:, 1] = (100 + np.arange(20))[:, None, None, None]
                demo.create_dataset("obs/rgb", data=frames, compression="gzip")
                state = np.full((20, dimension), -999, np.float32)
                state[:, -29:-20] = np.arange(20)[:, None] + number * 1000
                demo.create_dataset("obs/full_state", data=state)
                actions = np.broadcast_to(np.arange(20)[:, None] + number * 1000, (20, 9))
                demo.create_dataset("actions/relative_joint", data=actions.astype(np.float32))
                demo.create_dataset("dones", data=np.zeros(20, bool))
        index["tasks"][task] = {"path": str(source), "sha256": file_hash(source), "demonstrations": 3}
    index_path, run_path = tmp_path / "index.json", tmp_path / "run.json"
    write_json(index_path, index)
    write_json(run_path, {"dataset": {"identity": {"index_sha256": file_hash(index_path)}}, "split": splits})
    return index_path, run_path, tmp_path / "prepared"


def test_symlog_preserves_extreme_and_binary_commands():
    controls = np.asarray([-720, -2.7, -1, 0, 1, 2.7, 720], np.float32)
    np.testing.assert_allclose(decode_actions(encode_actions(controls)), controls, rtol=1e-6)
    assert np.max(np.abs(encode_actions(controls))) < 7


@pytest.mark.parametrize("task,dimension", STATE_DIMS.items())
def test_proprioception_excludes_object_pose_and_stage(task, dimension):
    state = np.full(dimension, 9876, np.float32)
    state[-29:-20] = np.arange(9)
    np.testing.assert_array_equal(joint_positions(state, task), np.arange(9))
    with pytest.raises(ValueError, match="Unaudited"):
        joint_positions(np.zeros(dimension + 1), task)


def test_camera_order_normalization_and_aspect_ratio():
    rgb = np.zeros((2, 72, 128, 3), np.uint8)
    rgb[0, :, :, 0] = 255
    rgb[1, :, :, 2] = 255
    actual = camera_image(rgb) * STD + MEAN
    assert actual.shape == (3, 144, 512)
    np.testing.assert_allclose(actual[:, 50, 50], [1, 0, 0], atol=1e-7)
    np.testing.assert_allclose(actual[:, 50, 300], [0, 0, 1], atol=1e-7)
    with pytest.raises(ValueError, match="aspect ratio"):
        camera_image(rgb, (320, 240))


def test_real_future_and_train_only_normalization(recordings):
    index, run, output = recordings
    manifest = prepare_data(index, run, output, horizon=16)
    stats = json.loads((output / "normalization.json").read_text())["robotwin2"]
    np.testing.assert_allclose(stats["action"]["max"], encode_actions(np.full(9, 18)))
    np.testing.assert_array_equal(stats["state"]["max"], np.full(9, 3))
    dataset = KinovaLiLaDataset(output / "manifest.json")
    assert len(dataset) == 27  # Three valid anchors in each task; no end padding.
    for task in dataset.tasks:
        lo, hi = dataset.task_ranges[task]
        first, last = dataset[lo], dataset[hi - 1]
        assert first["anchor"] == 1 and last["anchor"] == 3
        assert first["episode"] == last["episode"] == "demo_0"
        np.testing.assert_array_equal(first["state"], np.ones((1, 9)))
        np.testing.assert_allclose(decode_actions(first["action_sequence"])[:, 0], np.arange(1, 17), rtol=1e-6)
        np.testing.assert_array_equal(first["previous_action"], np.zeros(9))
        pixels = last["future_pixel_values"] * STD + MEAN
        np.testing.assert_allclose(pixels[:, 10, 10] * 255, np.full(3, 19), atol=1e-5)
        assert manifest["tasks"][task]["anchors_by_split"] == {"train": 3, "validation": 3, "test": 3}
    with pytest.raises(IndexError):
        dataset[len(dataset)]
    with pytest.raises(IndexError):
        dataset[-1]
    assert dataset.__getstate__()["_handles"] == {}
    validation = KinovaLiLaDataset(output / "manifest.json", "validation")
    assert validation[0]["episode"] == "demo_1"
    np.testing.assert_array_equal(validation[0]["state"], np.full((1, 9), 1001))
    dataset.close(); validation.close()


def test_balanced_reproducible_sampling():
    class Dataset:
        tasks = ["short", "long"]
        task_ranges = {"short": (0, 3), "long": (3, 100)}
    first = balanced_indices(Dataset(), seed=42, samples=1000)
    np.testing.assert_array_equal(first, balanced_indices(Dataset(), seed=42, samples=1000))
    assert (first < 3).sum() == 500
    assert first.min() >= 0 and first.max() < 100


def test_reject_overlap_and_checksum_mismatch(recordings):
    index, run, output = recordings
    previous = json.loads(run.read_text())
    previous["split"]["test"]["cabinet_door"] = ["demo_0"]
    write_json(run, previous)
    with pytest.raises(ValueError, match="overlapping"):
        prepare_data(index, run, output)
    source = json.loads(index.read_text())["tasks"]["cabinet_door"]["path"]
    with h5py.File(source, "a") as handle:
        handle.attrs["changed"] = True
    with pytest.raises(ValueError, match="checksum mismatch"):
        prepare_data(index, run, output)


def test_reject_short_episode_instead_of_padding(recordings):
    with pytest.raises(ValueError, match="too short"):
        prepare_data(*recordings, horizon=19)


def test_optimizer_resume_matches_uninterrupted_run(recordings, monkeypatch):
    torch = pytest.importorskip("torch")
    OmegaConf = pytest.importorskip("omegaconf").OmegaConf
    from kuafu.dynamics import lilawam_runtime as runtime

    index, run, output = recordings
    prepare_data(index, run, output)
    condition_dir = output.parent / "conditions"
    encoder_dir = output.parent / "fixture_encoder"
    encoder_dir.mkdir()
    # This test substitutes a tiny CPU model; these bytes are a provenance fixture.
    weights = encoder_dir / "model.safetensors"
    weights.write_bytes(b"unit-test fixture, not model weights")
    meta = {"fit_split": "train", "manifest_sha256": file_hash(output / "manifest.json"),
            "encoder_sha256": file_hash(weights), "tasks": {}}
    for task in STATE_DIMS:
        path = condition_dir / task / "task_cond.npy"
        path.parent.mkdir(parents=True)
        np.save(path, np.zeros(1024, np.float32))
        meta["tasks"][task] = {"sha256": file_hash(path)}
    write_json(condition_dir / "meta.json", meta)
    config = OmegaConf.create({
        "common": {"action_dim": 9, "state_dim": 9, "proprio_len": 1, "action_chunk_size": 16},
        "dataset": {"task_cond_dir": str(condition_dir), "image_size": [512, 144]},
        "model": {"vision_encoder": {"checkpoint_path": str(encoder_dir)}},
        "kuafu": {"manifest": str(output / "manifest.json"), "normalization": str(output / "normalization.json"),
                  "device": "cpu", "precision": "fp32", "seed": 42, "evaluate_every": 1,
                  "validation_windows": 1, "test_windows": 1},
        "training": {"batch_size": 3, "grad_accum_steps": 2, "epochs": 2, "learning_rate": 1e-3,
                     "lr_min": 1e-4, "betas": [.9, .99], "weight_decay": .01, "grad_clip_norm": 1},
        "system": {"num_workers": 0}})

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.action_model = torch.nn.Linear(9, 9)
            self.vision_encoder = torch.nn.Linear(1, 1)
            self.vision_encoder.requires_grad_(False)

        def forward(self, batch):
            prediction = self.action_model(batch["state"][:, 0])
            target = batch["action_sequence"][:, 0] + torch.randn_like(prediction) * .01
            loss = (prediction - target).square().mean()
            return loss, {"loss_mse": float(loss.detach()), "loss_future_feat": 0.}

    monkeypatch.setattr(runtime, "load_config", lambda path: (config, None))
    monkeypatch.setattr(runtime, "build_model", lambda *args: TinyModel())
    evaluations = []

    def evaluate(model, dataset, *args, **kwargs):
        evaluations.append(dataset.split)
        score = float(model.action_model.weight.detach().square().sum())
        return {"selection_score": score, "macro_average": {"fixture_score": score}}

    monkeypatch.setattr(runtime, "evaluate", evaluate)
    complete, interrupted = output.parent / "continuous", output.parent / "interrupted"
    runtime.train("unused.yaml", complete, max_updates=3)
    runtime.train("unused.yaml", interrupted, max_updates=1)
    runtime.train("unused.yaml", interrupted, resume=interrupted / "last.pt", max_updates=3)
    full = torch.load(complete / "last.pt", weights_only=False)
    resumed = torch.load(interrupted / "last.pt", weights_only=False)
    assert full["step"] == resumed["step"] == 3
    assert full["next_sample"] == resumed["next_sample"] == 18
    assert full["scheduler"] == resumed["scheduler"]
    for key in full["model"]:
        torch.testing.assert_close(full["model"][key], resumed["model"][key], rtol=0, atol=0)
    for key in full["optimizer"]["state"]:
        for field in full["optimizer"]["state"][key]:
            torch.testing.assert_close(full["optimizer"]["state"][key][field],
                                       resumed["optimizer"]["state"][key][field], rtol=0, atol=0)
    # Starting the low-LR stage resets the optimizer/update counter but preserves identity.
    config.training.learning_rate = 1e-4
    stage2 = output.parent / "stage2"
    runtime.train("unused.yaml", stage2, init_from=complete / "best.pt", max_updates=1)
    second = torch.load(stage2 / "last.pt", weights_only=False)
    assert second["step"] == 1 and second["identity"] == full["identity"]
    assert second["optimizer"]["param_groups"][0]["initial_lr"] == 1e-4
    assert "validation" in evaluations and "test" in evaluations
    # Reject silent replacement of a task vector even if its metadata is unchanged.
    np.save(condition_dir / "drawer" / "task_cond.npy", np.ones(1024, np.float32))
    with pytest.raises(ValueError, match="Task condition changed"):
        runtime.validate_conditions(config)


def test_gated_encoder_never_starts_training(tmp_path, monkeypatch):
    OmegaConf = pytest.importorskip("omegaconf").OmegaConf
    from types import SimpleNamespace
    from kuafu.dynamics import lilawam, lilawam_pipeline

    config = OmegaConf.create({"model": {"vision_encoder": {"checkpoint_path": str(tmp_path / "encoder")}}})
    OmegaConf.save(config, tmp_path / "stage1.yaml")
    used_tokens = []

    def gated(output, token_file):
        used_tokens.append(token_file)
        error = PermissionError("test access gate")
        error.response = SimpleNamespace(status_code=403)
        raise error

    def unexpected_child(*args, **kwargs):
        pytest.fail("A gated encoder must not launch GPU preprocessing or training")

    monkeypatch.setattr(lilawam, "download_encoder", gated)
    monkeypatch.setattr(lilawam_pipeline.subprocess, "Popen", unexpected_child)
    lilawam_pipeline.run(tmp_path, token_file="supplied-token.txt", wait_for_access=False)
    status = json.loads((tmp_path / "pipeline_status.json").read_text())
    assert status["phase"] == "waiting_for_encoder_access"
    assert status["http_status"] == 403 and status["production_training_started"] is False
    assert used_tokens == ["supplied-token.txt"]
