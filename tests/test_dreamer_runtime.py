"""Optional upstream-runtime regression: optimizer resume and checkpoint inference."""

import argparse
import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jax")
UPSTREAM = os.environ.get("KUAFU_DREAMERV3_ROOT")
if not UPSTREAM or not (Path(UPSTREAM) / "dreamerv3/rssm.py").is_file():
    pytest.skip("Set KUAFU_DREAMERV3_ROOT to run the pinned upstream runtime test", allow_module_level=True)

from kuafu.dynamics.dreamerv3 import load_forecaster, run_training
from test_dreamer_sequences import make_sequences


def test_resume_matches_uninterrupted_optimizer_and_forecast(tmp_path):
    dataset = make_sequences(tmp_path / "data.hdf5")

    def config(output, steps, resume=""):
        return argparse.Namespace(
            dataset=str(dataset), output=str(output), task="test", dreamerv3_root=UPSTREAM,
            resume_from=str(resume), device="cpu", model_size="size1m", obs_key="full_state",
            action_key="relative_joint", exclude_last=1, skip_first=1, seed=42, steps=steps,
            batch_size=2, length=8, burnin=2, report_every=1, eval_batches=1, context=4, horizon=3, particles=1,
        )

    uninterrupted = tmp_path / "uninterrupted"
    first = tmp_path / "first"
    continued = tmp_path / "continued"
    run_training(config(uninterrupted, 3))
    run_training(config(first, 2))
    report = run_training(config(continued, 3, first))
    assert report["steps_completed"] == 3
    with np.load(uninterrupted / "last.npz", allow_pickle=False) as expected, np.load(continued / "last.npz", allow_pickle=False) as actual:
        assert set(expected.files) == set(actual.files)
        for name in expected.files:
            np.testing.assert_allclose(actual[name], expected[name], atol=2e-7, rtol=2e-6, err_msg=name)
    context = np.zeros((2, 4, 2), np.float32)
    previous = np.zeros((2, 4, 1), np.float32)
    future = np.ones((2, 3, 1), np.float32)
    forecast, _ = load_forecaster(continued / "best.npz")
    prediction = forecast(context, previous, future, particles=2, seed=8)
    assert prediction.shape == (2, 2, 3, 2)
    assert np.isfinite(prediction).all()
    np.testing.assert_array_equal(prediction, forecast(context, previous, future, particles=2, seed=8))
    with pytest.raises(ValueError, match="dimensions"):
        forecast(context, previous, future[:, :, :0])


def test_decoder_prediction_restores_physical_units():
    from kuafu.dynamics.dreamerv3 import configure_runtime
    configure_runtime(UPSTREAM, "cpu", "size1m")
    import jax.numpy as jnp
    from embodied.jax import nets, outs
    from kuafu.dynamics.dreamer_model import state_prediction

    physical = jnp.asarray([-20., -2., 0., 0.25, 3., 50.])
    output = outs.Agg(outs.MSE(nets.symlog(physical), nets.symlog), 1, jnp.sum)
    np.testing.assert_allclose(state_prediction(output, True), physical, atol=1e-5)
    np.testing.assert_array_equal(state_prediction(outs.MSE(physical), False), physical)


def test_shared_model_masks_invalid_states_and_reloads_across_tasks(tmp_path):
    from kuafu.dynamics.dreamerv3 import configure_runtime
    from kuafu.dynamics.multitask import MultiTaskSequenceDataset
    from kuafu.dynamics.shared_dreamer import run_shared_training
    from test_dreamer_multitask import make_multitask_index
    import jax
    import ninjax as nj

    path = make_multitask_index(tmp_path)
    data = MultiTaskSequenceDataset(path)
    config, _ = configure_runtime(UPSTREAM, "cpu", "size1m")
    from kuafu.dynamics.dreamer_model import WorldModel
    model = WorldModel(73, 9, config, burnin=2, task_dim=2, name="wm")
    batch = data.sample("train", 2, 24, np.random.default_rng(42))
    parameters = nj.init(model.loss)({}, batch, seed=42)
    loss_fn = jax.jit(nj.pure(model.loss))
    expected = loss_fn(parameters, batch, seed=43)[1][0]
    changed = {key: value.copy() for key, value in batch.items()}
    invalid = (changed["state_mask"] == 0) | (changed["time_mask"][..., None] == 0)
    changed["state"][invalid] = 12345
    actual = loss_fn(parameters, changed, seed=43)[1][0]
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    def arguments(output, steps, resume=""):
        return argparse.Namespace(
            dataset_index=str(path), dataset=None, output=str(output), task="shared",
            dreamerv3_root=UPSTREAM, resume_from=str(resume), device="cpu", model_size="size1m",
            obs_key="full_state", action_key="relative_joint", exclude_last=1, skip_first=1,
            seed=42, steps=steps, batch_size=2, length=8, burnin=2, report_every=1,
            eval_batches=1, context=4, horizon=3, particles=1,
        )
    first, resumed, full = (tmp_path / name for name in ("shared_first", "shared_resumed", "shared_full"))
    run_shared_training(arguments(first, 1))
    run_shared_training(arguments(resumed, 2, first))
    report = run_shared_training(arguments(full, 2))
    assert set(report["test"]["per_task"]) == set(data.tasks)
    with np.load(full / "last.npz", allow_pickle=False) as expected, np.load(resumed / "last.npz", allow_pickle=False) as actual:
        assert expected.files == actual.files
        for key in expected.files:
            np.testing.assert_allclose(actual[key], expected[key], atol=2e-7, rtol=2e-6, err_msg=key)
    forecast, run = load_forecaster(resumed / "best.npz")
    assert run["dataset"]["task_dim"] == 2
    for task in data.tasks:
        dimension = data.datasets[task].metadata["state_dim"]
        prediction = forecast(np.zeros((2, 4, dimension)), np.zeros((2, 4, 9)),
                              np.ones((2, 3, 9)), task=task, particles=2)
        assert prediction.shape == (2, 2, 3, dimension)
        assert np.isfinite(prediction).all()
