"""Optional upstream RSSM training and point-set invariance regression."""

import argparse
import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jax")
UPSTREAM = os.environ.get("KUAFU_DREAMERV3_ROOT")
if not UPSTREAM or not (Path(UPSTREAM) / "dreamerv3/rssm.py").is_file():
    pytest.skip("Set KUAFU_DREAMERV3_ROOT for point-cloud runtime tests", allow_module_level=True)


def test_pointcloud_model_is_permutation_invariant_and_resumes(tmp_path):
    from test_pointcloud_sequences import make_pointcloud_index
    from kuafu.dynamics.dreamerv3 import configure_runtime, load_forecaster
    from kuafu.dynamics.pointcloud_sequences import prepare_pointcloud_cache, PointCloudSequenceDataset
    from kuafu.dynamics.pointcloud_dreamer import run_pointcloud_training
    config, _ = configure_runtime(UPSTREAM, "cpu", "size1m")
    from kuafu.dynamics.pointcloud_model import PointEncoder, PointCloudWorldModel, chamfer_squared
    import jax
    import ninjax as nj

    points = np.random.default_rng(0).uniform(-1, 1, (2, 3, 32, 3)).astype(np.float32)
    task = np.tile(np.eye(2, dtype=np.float32)[:, None], (1, 3, 1))
    encoder = PointEncoder(name="enc")
    params = nj.init(encoder)({}, points, task, seed=1)
    expected = nj.pure(encoder)(params, points, task, seed=2)[1]
    actual = nj.pure(encoder)(params, points[..., ::-1, :], task, seed=2)[1]
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(chamfer_squared(points, points[..., ::-1, :]), 0, atol=2e-7)
    translated = points + np.array([0, 0, 10], np.float32)
    assert (np.asarray(chamfer_squared(points, translated)) > 60).all()

    index = make_pointcloud_index(tmp_path)
    prepare_pointcloud_cache(index, tmp_path / "cache", num_points=32)
    data = PointCloudSequenceDataset(tmp_path / "cache")
    model = PointCloudWorldModel(32, 9, 2, config, 1, name="wm")
    batch = data.sample("train", 2, 24, np.random.default_rng(0))
    params = nj.init(model.loss)({}, batch, seed=0)
    loss = jax.jit(nj.pure(model.loss))
    expected = loss(params, batch, seed=1)[1][0]
    changed = {key: value.copy() for key, value in batch.items()}
    changed["points"][changed["time_mask"] == 0] = 999
    np.testing.assert_allclose(loss(params, changed, seed=1)[1][0], expected, rtol=1e-6)
    args = argparse.Namespace(pointcloud_index=str(index), num_points=32, output=str(tmp_path / "first"),
        seed=42, resume_from="", steps=1, batch_size=2, length=4, burnin=1, context=2, horizon=2,
        particles=1, report_every=1, eval_batches=1, dreamerv3_root=UPSTREAM, device="cpu", model_size="size1m")
    run_pointcloud_training(args)
    forecast, metadata = load_forecaster(tmp_path / "first/best.npz")
    assert metadata["encoder_inputs"] == ["points", "task"] and not metadata["privileged_state_input"]
    context = (batch["points"][:1, :2] * np.array(data.metadata["point_scale"])
               + np.array(data.metadata["point_center"]))
    actions = np.zeros((1, 2, 9), np.float32)
    predicted = forecast(context, actions, actions, task="task_0", particles=2, seed=7)
    assert predicted.shape == (1, 2, 2, 32, 3) and np.isfinite(predicted).all()
    np.testing.assert_array_equal(predicted, forecast(context, actions, actions, task="task_0", particles=2, seed=7))
    args.steps, args.resume_from, args.output = 2, str(tmp_path / "first"), str(tmp_path / "resumed")
    run_pointcloud_training(args)
    args.resume_from, args.output = "", str(tmp_path / "full")
    run_pointcloud_training(args)
    with np.load(tmp_path / "resumed/last.npz") as a, np.load(tmp_path / "full/last.npz") as b:
        assert set(a.files) == set(b.files)
        for key in a.files:
            np.testing.assert_allclose(a[key], b[key], rtol=2e-6, atol=2e-7, err_msg=key)
