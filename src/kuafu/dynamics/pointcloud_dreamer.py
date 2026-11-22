"""Train and evaluate a shared point-cloud adaptation of the DreamerV3 world model."""

import json
from pathlib import Path

import numpy as np

from kuafu.dynamics.dreamerv3 import configure_runtime, load_weights
from kuafu.dynamics.sequences import symlog


FORMAT = "kuafu-dreamerv3-pointcloud-world-model-v1"


def run_pointcloud_training(args):
    from kuafu.dynamics.rgb_dreamer import run_visual_training
    return run_visual_training(args, modality="pointcloud")


def evaluate_pointcloud(parameters, predict, data, split, *, context, horizon, batches, particles,
                        seed, batch_size=4, preview=None):
    """Measure symmetric Chamfer error in meters, averaging errors across particles."""
    import jax
    from kuafu.dynamics.pointcloud_model import chamfer_squared

    distance = jax.jit(chamfer_squared)
    center = np.asarray(data.metadata["point_center"], np.float32)
    scale = np.asarray(data.metadata["point_scale"], np.float32)
    reports = {}
    for task_id, task in enumerate(data.tasks):
        task_seed = seed + 10000 * task_id
        rng = np.random.default_rng(task_seed)
        error, persistence = np.zeros(horizon, np.float64), np.zeros(horizon, np.float64)
        for index in range(batches):
            batch = data.sample_task(task, split, batch_size, context + horizon, rng, pad=False)
            inputs = jax.device_put((batch["points"][:, :context], batch["prev_action"][:, :context],
                                    batch["prev_action"][:, context:], batch["task"][:, 0]))
            target = batch["points"][:, context:] * scale + center
            initial = batch["points"][:, context - 1:context] * scale + center
            for particle in range(particles):
                prediction = np.asarray(predict(parameters, *inputs, seed=task_seed + index * particles + particle)[1])
                prediction = prediction * scale + center
                if not np.isfinite(prediction).all():
                    raise FloatingPointError(f"{task}: nonfinite point-cloud forecast")
                error += np.asarray(distance(prediction, target)).mean(0) / particles
                if preview is not None and index == particle == 0:
                    directory = Path(preview)
                    directory.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(directory / f"{task}.npz", initial=initial[0, 0], target=target[0], prediction=prediction[0])
            persistence += np.asarray(distance(np.broadcast_to(initial, target.shape), target)).mean(0)
        scores = {str(step): {"chamfer_rmse_m": float(np.sqrt(error[step - 1] / batches)),
                             "persistence_chamfer_rmse_m": float(np.sqrt(persistence[step - 1] / batches))}
                  for step in sorted({1, min(5, horizon), min(10, horizon), horizon})}
        reports[task] = {"windows": batches * batch_size, "horizons": scores}
    macro = {step: {key: float(np.mean([r["horizons"][step][key] for r in reports.values()]))
                    for key in ("chamfer_rmse_m", "persistence_chamfer_rmse_m")}
             for step in next(iter(reports.values()))["horizons"]}
    return {"per_task": reports, "macro_average": macro, "context": context, "horizon": horizon,
            "particles": particles, "aggregation": "equal-task mean; mean squared set error across particles",
            "metric": "sqrt(0.5 * (mean nearest squared pred-to-target + mean nearest squared target-to-pred))",
            "units": "meters", "point_order": "ignored; no indexwise point matching"}


def load_pointcloud_forecaster(checkpoint, *, device="cpu", dreamerv3_root=None):
    """Return forecasts [B,particles,horizon,N,3] in environment-local meters."""
    checkpoint = Path(checkpoint).resolve()
    run = json.loads((checkpoint.parent / "run.json").read_text())
    if run.get("format") != FORMAT:
        raise ValueError("Expected a point-cloud DreamerV3 checkpoint")
    configure_runtime(dreamerv3_root or run["upstream"]["root"], device, run["arguments"]["model_size"])
    import elements
    import jax
    import ninjax as nj
    from kuafu.dynamics.pointcloud_model import PointCloudWorldModel

    metadata = run["dataset"]
    tasks = sorted(metadata["tasks"])
    num_points = metadata["num_points"]
    model = PointCloudWorldModel(num_points, 9, len(tasks), elements.Config(run["agent_config"]),
                                 run["arguments"]["burnin"], name="wm")
    parameters = load_weights(checkpoint)
    predict = jax.jit(nj.pure(model.forecast))
    action_scale = np.asarray(metadata["action_scale"], np.float32)
    center, scale = [np.asarray(metadata[key], np.float32) for key in ("point_center", "point_scale")]

    def forecast(context_points, previous_actions, future_actions, *, task, particles=1, seed=0):
        points = np.asarray(context_points, np.float32)
        previous, future = np.asarray(previous_actions, np.float32), np.asarray(future_actions, np.float32)
        if (task not in tasks or points.ndim != 4 or points.shape[-2:] != (num_points, 3)
                or min(points.shape[:2]) < 1 or previous.shape != (*points.shape[:2], 9)
                or future.ndim != 3 or future.shape[0] != len(points) or future.shape[-1] != 9
                or future.shape[1] < 1 or particles < 1
                or not all(np.isfinite(x).all() for x in (points, previous, future))):
            raise ValueError("Invalid point-cloud/action dimensions, task, or particle count")
        normalized = (points - center) / scale
        if np.max(np.abs(normalized)) > 1.001:
            raise ValueError("Point clouds must use the recorded workspace and local meter coordinates")
        onehot = np.eye(len(tasks), dtype=np.float32)[[tasks.index(task)]].repeat(len(points), axis=0)
        inputs = jax.device_put((normalized, np.clip(symlog(previous) / action_scale, -1, 1),
                                np.clip(symlog(future) / action_scale, -1, 1), onehot))
        return np.stack([np.asarray(predict(parameters, *inputs, seed=seed + i)[1]) * scale + center
                         for i in range(particles)], axis=1)
    return forecast, run
