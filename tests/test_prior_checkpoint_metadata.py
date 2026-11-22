from types import SimpleNamespace

import h5py
import pytest
import torch
from torch import nn
from kuafu.bc.training.rl_games_prior import apply_bc_prior_to_rl_games_agent, infer_prior_obs_key
from kuafu.rl.isaac.train_rl_games import _load_bc_supervised_dataset


@pytest.mark.parametrize("location", ["config", "artifact", "working_directory"])
def test_bc_regularization_loads_portable_dataset(tmp_path, monkeypatch, location):
    """A copied prior must load its demonstrations outside the author's checkout."""
    run_dir = tmp_path / "prior"
    artifact_dir = tmp_path / "artifacts"
    working_dir = tmp_path / "unrelated"
    for directory in (run_dir, artifact_dir, working_dir):
        directory.mkdir()
    monkeypatch.chdir(working_dir)
    monkeypatch.setenv("KUAFU_ARTIFACT_ROOT", str(artifact_dir))
    dataset_dir = {"config": run_dir, "artifact": artifact_dir, "working_directory": working_dir}[location]
    dataset_path = dataset_dir / "demonstrations.hdf5"
    observations = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    actions = torch.tensor([[0.1], [0.2], [0.3]])
    with h5py.File(dataset_path, "w") as handle:
        handle.create_dataset("data/demo_0/obs/state", data=observations.numpy())
        handle.create_dataset("data/demo_0/actions/relative_joint", data=actions.numpy())
    (run_dir / "train_config.yaml").write_text("dataset:\n  dataset: demonstrations.hdf5\n")
    checkpoint = run_dir / "best.pth"
    torch.save({"obs_normalizer": {"mean": torch.ones(2), "std": torch.full((2,), 2.0)}}, checkpoint)

    result = _load_bc_supervised_dataset(str(checkpoint), "cpu")

    assert result["num_pairs"] == 3
    torch.testing.assert_close(result["obs_raw"], observations)
    torch.testing.assert_close(result["actions_raw"], actions)
    torch.testing.assert_close(result["obs_mean"], torch.ones(2))
    torch.testing.assert_close(result["obs_std"], torch.full((2,), 2.0))


def test_standalone_checkpoint_metadata_takes_precedence(tmp_path):
    path = tmp_path / "best.pth"
    torch.save({"obs_key": "full_state"}, path)
    assert infer_prior_obs_key(str(path)) == "full_state"
    (tmp_path / "train_config.yaml").write_text("dataset:\n  obs_key: state\n")
    assert infer_prior_obs_key(str(path)) == "full_state"


def test_old_checkpoint_uses_adjacent_config(tmp_path):
    path = tmp_path / "best.pth"
    torch.save({}, path)
    (tmp_path / "train_config.yaml").write_text("dataset:\n  obs_key: state\n")
    assert infer_prior_obs_key(str(path)) == "state"
    (tmp_path / "train_config.yaml").write_text("dataset:\n  obs_key: invalid\n")
    with pytest.raises(ValueError, match="observation mode"):
        infer_prior_obs_key(str(path))


def test_standalone_prior_keeps_unified_observation_normalization(tmp_path):
    state = {
        "actor_trunk.0.weight": torch.eye(2),
        "actor_trunk.0.bias": torch.zeros(2),
        "mean_layer.weight": torch.tensor([[1.0, 2.0]]),
        "mean_layer.bias": torch.tensor([0.5]),
    }
    checkpoint = {
        "obs_key": "full_state", "obs_history": 1, "action_chunk_size": 1,
        "time_shift": 0, "action_mode": "joint_pos",
        "obs_normalizer": {"mean": torch.tensor([10.0, 20.0]), "std": torch.tensor([2.0, 4.0])},
        "act_normalizer": {"mean": torch.zeros(1), "std": torch.ones(1)},
        "model_state_dicts": {
            "model_state_dict_prior_norm_obs": state,
            "model_state_dict_prior_raw_obs": state,
        },
    }
    path = tmp_path / "best.pth"
    torch.save(checkpoint, path)
    network = SimpleNamespace(actor_mlp=nn.Sequential(nn.Linear(2, 2)), mu=nn.Linear(2, 1))
    agent = SimpleNamespace(model=SimpleNamespace(normalize_input=False, a2c_network=network))
    report = apply_bc_prior_to_rl_games_agent(agent, str(path), verbose=False)
    assert report.obs_key == "full_state"
    assert report.copied_tensors == 4
    result = network.mu(network.actor_mlp(torch.tensor([[12.0, 28.0]])))
    torch.testing.assert_close(result, torch.tensor([[5.5]]))
    checkpoint.pop("action_mode")
    torch.save(checkpoint, path)
    with pytest.raises(FileNotFoundError, match="prior run config"):
        apply_bc_prior_to_rl_games_agent(agent, str(path), verbose=False)
