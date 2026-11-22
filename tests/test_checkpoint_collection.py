import json
from types import SimpleNamespace

import pytest
import torch

from kuafu.data.collection.checkpoint_options import (
    checkpoint_params_directory,
    find_checkpoint_params,
    collection_camera_keys,
    configure_object_sides,
    limit_completed_successes,
)


@pytest.mark.parametrize("as_tensor", [False, True])
@pytest.mark.parametrize("remaining, expected", [(0, []), (1, [0]), (2, [0, 3]), (10, [0, 3])])
def test_completion_budget_counts_only_finished_successes(as_tensor, remaining, expected):
    dones = [True, False, True, True]
    successes = [True, True, False, True]
    if as_tensor:
        dones, successes = torch.tensor(dones), torch.tensor(successes)
    before = torch.as_tensor(successes).clone()
    accepted, completed = limit_completed_successes(dones, successes, remaining)
    assert accepted.nonzero().flatten().tolist() == expected
    assert completed == 2
    assert torch.equal(torch.as_tensor(successes), before)


def test_completion_budget_rejects_broadcasting_masks():
    with pytest.raises(ValueError, match="matching lengths"):
        limit_completed_successes([True, True], [True], 2)


def test_calibrated_cameras_and_custom_allowlist():
    panda = ("table_cam_0", "table_cam_1", "table_cam_2")
    assert collection_camera_keys(is_panda=True) == panda
    assert collection_camera_keys(panda, is_panda=True) == panda
    assert collection_camera_keys(is_kinova=True) == panda[:2]
    assert collection_camera_keys(["custom_camera"], is_kinova=True) == ("custom_camera",)
    assert collection_camera_keys() is None
    with pytest.raises(ValueError, match="in order"):
        collection_camera_keys(panda[::-1], is_panda=True)
    with pytest.raises(ValueError, match="in order"):
        collection_camera_keys(panda[:2], is_panda=True)
    with pytest.raises(ValueError, match="duplicates"):
        collection_camera_keys(["camera", "camera"])


@pytest.mark.parametrize("force_swap", [False, True])
def test_fixed_object_side_settings(force_swap):
    cfg = SimpleNamespace(randomize_left_right=True, fixed_left_right_swap=False)
    configure_object_sides(cfg)
    assert cfg.randomize_left_right is True
    configure_object_sides(cfg, disable_randomization=not force_swap, force_swap=force_swap)
    assert cfg.randomize_left_right is False
    assert cfg.fixed_left_right_swap is force_swap


def test_invalid_object_side_options_do_not_mutate_config():
    cfg = SimpleNamespace(randomize_left_right=True, fixed_left_right_swap=False)
    with pytest.raises(ValueError, match="mutually exclusive"):
        configure_object_sides(cfg, disable_randomization=True, force_swap=True)
    assert cfg.randomize_left_right is True
    with pytest.raises(ValueError, match="fixed_left_right_swap"):
        configure_object_sides(SimpleNamespace(randomize_left_right=True), force_swap=True)


def test_explicit_playback_parameters_resolve_from_artifact_store(tmp_path, monkeypatch):
    monkeypatch.setenv("KUAFU_ARTIFACT_ROOT", str(tmp_path))
    params = tmp_path / "params"
    params.mkdir()
    (params / "agent.yaml").write_text("{}")
    with pytest.raises(FileNotFoundError, match="env.yaml"):
        checkpoint_params_directory("params")
    (params / "env.yaml").write_text("{}")
    assert checkpoint_params_directory("params") == params


@pytest.mark.parametrize("reference", [
    "Residuals/task/run/rl_runs/worker/nn/selected.pth",
    "/old/checkout/Residuals/task/run/rl_runs/worker/nn/selected.pth",
    "rl_runs/worker/summaries",
])
def test_moved_checkpoint_uses_selected_worker_params(tmp_path, reference):
    archive = tmp_path / "downloaded/task/run"
    checkpoint = archive / "checkpoints/best.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    params = archive / "rl_runs/worker/params"
    params.mkdir(parents=True)
    for filename in ("agent.yaml", "env.yaml"):
        (params / filename).write_text("{}")
    (archive / "best_checkpoint.json").write_text(json.dumps({
        "best_checkpoint": {"selected_worker_checkpoint_archive_path": reference},
    }))
    assert find_checkpoint_params(checkpoint, tmp_path / "unrelated-checkout") == params
    (params / "env.yaml").unlink()
    assert find_checkpoint_params(checkpoint, tmp_path / "unrelated-checkout") is None
