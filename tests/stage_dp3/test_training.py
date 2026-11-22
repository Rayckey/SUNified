from conftest import require_stage_runtime

require_stage_runtime()

import json
import os
from pathlib import Path

import pytest
import torch
import yaml

from kuafu.sim2real.dp3.dataset import StageDataset
from kuafu.sim2real.dp3.export import export_dataset
from kuafu.sim2real.dp3.train import load_checkpoint, load_policy, resolve_experiment, run_training
from test_data import make_episodes


def small_experiment(tmp_path, device="cpu", *, simple=False):
    paths = make_episodes(tmp_path / "source")
    dataset = tmp_path / "data.zarr"
    export_dataset(paths, dataset, task="test-task", stage_names=["reach", "grasp", "place"], num_points=12)
    baseline = "dp3_train_kuafu_eff_simple.yaml" if simple else "dp3_train_kuafu_eff_simple_full.yaml"
    config = {
        "baseline_config": str(Path(os.environ["KUAFU2REAL_ROOT"]) / "configs" / baseline),
        "seeds": [42], "epochs": 2, "batch_size": 16, "num_workers": 0, "device": device, "warmup_steps": 0,
        "stage": {"stage_hidden_dim": 8, "stage_embedding_dim": 4, "stage_loss_weight": 0.1},
        "policy_overrides": {"horizon": 8, "n_obs_steps": 2, "n_action_steps": 3,
                             "diffusion_step_embed_dim": 16, "down_dims": [16, 32], "n_groups": 4,
                             "kernel_size": 3, "encoder_output_dim": 16, "num_inference_steps": 2,
                             "pointcloud_encoder_cfg": {"in_channels": 3, "out_channels": 16,
                                                         "use_layernorm": True, "final_norm": "layernorm"}},
    }
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(config))
    return dataset, path


@pytest.mark.parametrize("simple", [False, True], ids=["DP3", "SimpleDP3"])
def test_train_abc_selects_action_loss_and_loads_deployment_checkpoint(tmp_path, simple):
    dataset, config = small_experiment(tmp_path, simple=simple)
    output = tmp_path / "run"
    result = run_training(config, dataset, output)
    ids, initial_norms = [], []
    for variant in "ABC":
        best = output / "seed_42" / variant / "best.ckpt"
        policy, checkpoint = load_policy(best)
        assert type(policy).__name__ == ("StageSimpleDP3" if simple else "StageDP3")
        assert policy.variant == variant
        metadata = checkpoint["experiment"]
        ids.append(metadata["comparison_id"])
        values = [json.loads(path.read_text())["validation"]["action_loss"]
                  for path in sorted(best.parent.glob("metrics_*.json"))]
        assert metadata["best_validation_action_loss"] == min(values)
        assert metadata["parameters"] > 0
        assert "stage_id" not in checkpoint["cfg"].policy.shape_meta.obs
        initial_norms.append(policy.normalizer.state_dict())
    assert len(set(ids)) == 1
    for key in initial_norms[0]:
        for normalizer in initial_norms[1:]:
            torch.testing.assert_close(initial_norms[0][key], normalizer[key])
    with pytest.raises(ValueError, match="new output"):
        run_training(config, dataset, output)
    assert run_training(config, dataset, output, resume=True) == result


def test_training_windows_preserve_final_stage_when_execution_chunk_is_shorter(tmp_path):
    dataset, config = small_experiment(tmp_path)
    resolved = resolve_experiment(config, dataset)
    data = StageDataset(dataset, **resolved["window"])
    assert resolved["window"]["pad_after"] == 6
    assert all(count > 0 for count in data.class_counts())
    assert data[-1]["stage_id"].item() == 2


@pytest.mark.parametrize("simple", [False, True], ids=["DP3", "SimpleDP3"])
def test_epoch_resume_matches_uninterrupted_training(tmp_path, monkeypatch, simple):
    from kuafu.sim2real.dp3 import train
    dataset, config = small_experiment(tmp_path, simple=simple)
    reference = tmp_path / "reference"
    interrupted = tmp_path / "interrupted"
    run_training(config, dataset, reference, variants=["C"])
    save = train.save_checkpoint
    def interrupt_after_epoch(path, **kwargs):
        save(path, **kwargs)
        if path.name == "latest.ckpt" and kwargs["epoch"] == 0:
            raise InterruptedError("simulated interruption after durable epoch checkpoint")
    monkeypatch.setattr(train, "save_checkpoint", interrupt_after_epoch)
    with pytest.raises(InterruptedError):
        run_training(config, dataset, interrupted, variants=["C"])
    monkeypatch.setattr(train, "save_checkpoint", save)
    run_training(config, dataset, interrupted, variants=["C"], resume=True)
    expected = load_checkpoint(reference / "seed_42/C/latest.ckpt")
    actual = load_checkpoint(interrupted / "seed_42/C/latest.ckpt")
    for model in ("model", "ema_model"):
        for key, value in expected["state_dicts"][model].items():
            torch.testing.assert_close(value, actual["state_dicts"][model][key], rtol=0, atol=0)
