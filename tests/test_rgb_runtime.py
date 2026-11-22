"""Optional actual upstream visual training, padding, and inference regression."""

import argparse
import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jax")
UPSTREAM = os.environ.get("KUAFU_DREAMERV3_ROOT")
if not UPSTREAM or not (Path(UPSTREAM) / "dreamerv3/rssm.py").is_file():
    pytest.skip("Set KUAFU_DREAMERV3_ROOT for visual runtime tests", allow_module_level=True)


def test_rgb_model_trains_masks_padding_and_reloads(tmp_path):
    import jax
    import ninjax as nj
    from test_rgb_sequences import make_rgb_index
    from kuafu.dynamics.dreamerv3 import configure_runtime, load_forecaster
    from kuafu.dynamics.rgb_sequences import prepare_rgb_cache, RGBSequenceDataset
    from kuafu.dynamics.rgb_dreamer import run_rgb_training

    index = make_rgb_index(tmp_path)
    prepare_rgb_cache(index, tmp_path / "cache", height=48, width=48)
    data = RGBSequenceDataset(tmp_path / "cache")
    config, _ = configure_runtime(UPSTREAM, "cpu", "size1m")
    config = config.update({"enc.simple.strided": True, "dec.simple.strided": True})
    from kuafu.dynamics.rgb_model import RGBWorldModel
    model = RGBWorldModel((48, 48, 6), 9, 2, config, 1, name="wm")
    batch = data.sample("train", 2, 24, np.random.default_rng(0))
    params = nj.init(model.loss)({}, batch, seed=0)
    loss = jax.jit(nj.pure(model.loss))
    expected = loss(params, batch, seed=1)[1][0]
    changed = {key: value.copy() for key, value in batch.items()}
    changed["rgb"][changed["time_mask"] == 0] = 255
    actual = loss(params, changed, seed=1)[1][0]
    np.testing.assert_allclose(expected, actual, rtol=1e-6)
    args = argparse.Namespace(rgb_index=str(index), output=str(tmp_path / "run"), seed=42,
        resume_from="", image_height=48, image_width=48, steps=1, batch_size=2, length=4,
        burnin=1, context=2, horizon=2, particles=1, report_every=1, eval_batches=1,
        dreamerv3_root=UPSTREAM, device="cpu", model_size="size1m")
    report = run_rgb_training(args)
    assert report["steps_completed"] == 1
    forecast, run = load_forecaster(tmp_path / "run/best.npz")
    assert run["encoder_inputs"] == ["rgb", "task"] and run["privileged_state_input"] is False
    rgb = batch["rgb"][:1, :2]
    previous = np.zeros((1, 2, 9), np.float32)
    future = np.zeros((1, 2, 9), np.float32)
    prediction = forecast(rgb, previous, future, task="task_0", particles=2, seed=5)
    assert prediction.shape == (1, 2, 2, 48, 48, 6)
    assert np.isfinite(prediction).all() and prediction.min() >= 0 and prediction.max() <= 1
    np.testing.assert_array_equal(prediction, forecast(rgb, previous, future, task="task_0", particles=2, seed=5))
    with pytest.raises(ValueError, match="dtype"):
        forecast(rgb.astype(float), previous, future, task="task_0")

    args.steps = 2
    args.resume_from = str(tmp_path / "run")
    args.output = str(tmp_path / "resumed")
    run_rgb_training(args)
    args.resume_from = ""
    args.output = str(tmp_path / "full")
    run_rgb_training(args)
    with np.load(tmp_path / "resumed/last.npz") as resumed, np.load(tmp_path / "full/last.npz") as full:
        assert set(resumed.files) == set(full.files)
        for key in full.files:
            np.testing.assert_allclose(resumed[key], full[key], rtol=2e-6, atol=2e-7, err_msg=key)
