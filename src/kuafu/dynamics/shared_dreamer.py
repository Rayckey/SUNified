"""Train and evaluate one task-conditioned DreamerV3 world model for Kinova."""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
import shutil
import time

import numpy as np

from kuafu.dynamics.dreamerv3 import configure_runtime, load_weights, save_weights, write_json
from kuafu.dynamics.multitask import MultiTaskSequenceDataset, canonical_state
from kuafu.dynamics.sequences import symlog
from kuafu.paths import resolve_artifact_path


FORMAT = "kuafu-dreamerv3-shared-world-model-v1"


def evaluate_shared(parameters, forecast_fn, data, split, *, batch_size, batches,
                    context, horizon, particles, seed):
    """Report physical-unit errors per task and equally weighted task RMSE."""
    import jax

    reports = {}
    for task_id, task in enumerate(data.tasks):
        task_seed = seed + 1000 * task_id
        rng = np.random.default_rng(task_seed)
        columns = data.indices[task]
        mean, scale = data.state_mean[task], data.state_scale[task]
        mse = np.zeros((horizon, len(columns)), np.float64)
        persistence = np.zeros_like(mse)
        for index in range(batches):
            batch = data.sample_task(task, split, batch_size, context + horizon, rng)
            target = batch["state"][:, context:, columns] * scale + mean
            initial = batch["state"][:, context - 1:context, columns] * scale + mean
            inputs = jax.device_put((batch["state"][:, :context],
                                     batch["prev_action"][:, :context],
                                     batch["prev_action"][:, context:],
                                     batch["task"][:, 0], batch["state_mask"][:, 0]))
            predictions = [np.asarray(forecast_fn(parameters, *inputs,
                           seed=task_seed + index * particles + particle)[1])[..., columns]
                           for particle in range(particles)]
            prediction = np.mean(predictions, axis=0) * scale + mean
            if not np.isfinite(prediction).all():
                raise FloatingPointError(f"Nonfinite {task} forecast")
            mse += np.square(prediction.astype(np.float64) - target).sum(0)
            persistence += np.square(initial.astype(np.float64) - target).sum(0)
        mse /= batch_size * batches
        persistence /= batch_size * batches
        standard_deviation = data.datasets[task].state_std
        scores = {}
        for step in sorted({1, min(5, horizon), min(10, horizon), horizon}):
            scores[str(step)] = {
                "state_normalized_rmse": float(np.sqrt(np.mean(mse[step - 1] / standard_deviation ** 2))),
                "persistence_state_normalized_rmse": float(np.sqrt(np.mean(persistence[step - 1] / standard_deviation ** 2))),
                "rmse_per_dim": np.sqrt(mse[step - 1]).tolist(),
                "persistence_rmse_per_dim": np.sqrt(persistence[step - 1]).tolist(),
            }
        reports[task] = {"windows": batch_size * batches, "horizons": scores}
    macro = {step: {key: float(np.mean([report["horizons"][step][key] for report in reports.values()]))
                    for key in ("state_normalized_rmse", "persistence_state_normalized_rmse")}
             for step in next(iter(reports.values()))["horizons"]}
    return {"context": context, "particles": particles, "per_task": reports,
            "macro_average": macro, "aggregation": "arithmetic mean of per-task normalized RMSE"}


def run_shared_training(args):
    """Train shared weights; select using validation tasks, evaluate test once."""
    for key in ("steps", "batch_size", "length", "report_every", "eval_batches", "particles", "context", "horizon"):
        if getattr(args, key) < 1:
            raise ValueError(f"{key} must be positive")
    if not 0 <= args.burnin < args.length:
        raise ValueError("burnin must be nonnegative and shorter than sequence length")
    if (args.obs_key, args.action_key, args.exclude_last) != ("full_state", "relative_joint", 1):
        raise ValueError("Shared Kinova layouts require full_state, relative_joint, and exclude_last=1")
    output = resolve_artifact_path(args.output, "Shared DreamerV3 output")
    if output.exists():
        raise ValueError(f"Use a new output directory; refusing to overwrite {output}")
    data = MultiTaskSequenceDataset(args.dataset_index, seed=args.seed, skip_first=args.skip_first)
    rng = np.random.default_rng(args.seed)
    batch = data.sample("train", args.batch_size, args.length, rng)
    for task, dataset in data.datasets.items():
        if min(len(e.state) for e in dataset.episodes) <= args.burnin:
            raise ValueError(f"{task} has episodes shorter than burnin")
        for split in ("validation", "test"):
            data.sample_task(task, split, args.batch_size, args.context + args.horizon,
                             np.random.default_rng(args.seed))
    config, upstream = configure_runtime(args.dreamerv3_root, args.device, args.model_size)
    import jax
    import ninjax as nj
    from kuafu.dynamics.dreamer_model import WorldModel

    serialized_config = json.loads(json.dumps(dict(config)))
    start_step, source_run = 0, None
    if args.resume_from:
        source_run = Path(args.resume_from).expanduser().resolve()
        previous = json.loads((source_run / "run.json").read_text())
        progress = json.loads((source_run / "progress.json").read_text())
        if previous.get("format") != FORMAT or progress.get("phase") != "complete":
            raise ValueError("Shared resume requires a completed shared-model run")
        if (previous["dataset"] != data.metadata or previous["split"] != data.split
                or previous["agent_config"] != serialized_config):
            raise ValueError("Shared resume requires unchanged data, splits, and model configuration")
        for key in ("seed", "batch_size", "length", "burnin", "context", "horizon", "eval_batches", "particles"):
            if previous["arguments"][key] != getattr(args, key):
                raise ValueError(f"Resume requires unchanged {key}")
        start_step = progress["step"]
        if args.steps <= start_step:
            raise ValueError("steps must exceed the resumed run's update count")
        rng.bit_generator.state = json.loads((source_run / "sampler_state.json").read_text())
    model = WorldModel(data.metadata["state_dim"], 9, config, burnin=args.burnin,
                       task_dim=len(data.tasks), name="wm")
    output.mkdir(parents=True)
    run = {"format": FORMAT, "arguments": vars(args), "upstream": upstream,
           "agent_config": serialized_config, "dataset": data.metadata, "split": data.split,
           "compute_dtype": "float32", "start_step": start_step,
           "selection": "equal-task mean validation final-horizon normalized RMSE",
           "forecast_output_units": "dataset units after inverse signed-log and per-task normalization",
           "versions": {name: importlib.metadata.version(name) for name in
                        ("jax", "jaxlib", "ninjax", "elements", "optax", "numpy", "h5py")}}
    write_json(output / "run.json", run)
    write_json(output / "progress.json", {"phase": "initializing", "step": start_step})
    print(json.dumps({"output": str(output), "tasks": data.tasks, "upstream": upstream,
                      "split_episodes": {key: sum(map(len, rows.values())) for key, rows in data.split.items()},
                      "usable_transitions": data.metadata["usable_transitions"]}), flush=True)
    parameters = nj.init(model.train)({}, jax.device_put(batch), seed=args.seed)
    if source_run:
        restored = load_weights(source_run / "last.npz")
        if parameters.keys() != restored.keys() or any(parameters[k].shape != restored[k].shape for k in parameters):
            raise ValueError("Shared resume checkpoint structure mismatch")
        parameters = restored
    train_fn = jax.jit(nj.pure(model.train))
    forecast_fn = jax.jit(nj.pure(model.forecast))
    loss_fn = jax.jit(nj.pure(model.loss))
    eval_options = dict(batch_size=args.batch_size, batches=args.eval_batches, context=args.context,
                        horizon=args.horizon, particles=args.particles)
    best_score, best_step = float("inf"), 0
    if source_run:
        best = json.loads((source_run / "best.json").read_text())
        best_score = best["validation"]["macro_average"][str(args.horizon)]["state_normalized_rmse"]
        best_step = best["step"]
        shutil.copyfile(source_run / "best.npz", output / "best.npz")
        write_json(output / "best.json", best)
    started = time.monotonic()
    for step in range(start_step + 1, args.steps + 1):
        batch = data.sample("train", args.batch_size, args.length, rng)
        parameters, metrics = train_fn(parameters, jax.device_put(batch), seed=args.seed + step)
        if step == start_step + 1 or step % args.report_every == 0 or step == args.steps:
            metrics = {key: float(value) for key, value in metrics.items()}
            if not all(np.isfinite(value) for value in metrics.values()):
                raise FloatingPointError(f"Nonfinite training metrics at {step}")
            validation = evaluate_shared(parameters, forecast_fn, data, "validation",
                                          seed=args.seed + 100000, **eval_options)
            valbatch = data.sample("validation", args.batch_size, args.length, np.random.default_rng(args.seed + 123))
            _, (_, losses) = loss_fn(parameters, jax.device_put(valbatch), seed=args.seed + 999)
            score = validation["macro_average"][str(args.horizon)]["state_normalized_rmse"]
            if score < best_score:
                best_score, best_step = score, step
                save_weights(output / "best.npz", {k: v for k, v in parameters.items() if not k.startswith("wm/opt/")})
                write_json(output / "best.json", {"step": step, "validation": validation})
            record = {"step": step, "elapsed_seconds": time.monotonic() - started,
                      "train": metrics, "validation_loss": {key: float(value) for key, value in losses.items()},
                      "validation_forecast_rmse": score, "best_step": best_step,
                      "per_task_validation_rmse": {task: report["horizons"][str(args.horizon)]["state_normalized_rmse"]
                                                   for task, report in validation["per_task"].items()}}
            with (output / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
            write_json(output / "progress.json", {"phase": "training", **record})
            print(json.dumps(record, allow_nan=False), flush=True)
    save_weights(output / "last.npz", parameters)
    write_json(output / "sampler_state.json", rng.bit_generator.state)
    best = load_weights(output / "best.npz")
    report = {"steps_completed": args.steps, "best_step": best_step,
              "elapsed_seconds": time.monotonic() - started,
              "validation": json.loads((output / "best.json").read_text())["validation"],
              "test": evaluate_shared(best, forecast_fn, data, "test", seed=args.seed + 200000,
                                        **{**eval_options, "batches": 2 * args.eval_batches})}
    write_json(output / "evaluation.json", report)
    write_json(output / "progress.json", {"phase": "complete", "step": args.steps, "best_step": best_step})
    print(json.dumps({"finished": str(output / "best.npz"), "best_step": best_step,
                      "test_final_horizon": report["test"]["macro_average"][str(args.horizon)]}), flush=True)
    return report


def load_shared_forecaster(checkpoint, *, device="cpu", dreamerv3_root=None):
    """Load shared weights; forecast(..., task=slug) uses that task's native state order."""
    checkpoint = Path(checkpoint).expanduser().resolve()
    run = json.loads((checkpoint.parent / "run.json").read_text())
    if run.get("format") != FORMAT:
        raise ValueError("Expected a shared DreamerV3 world-model checkpoint")
    configure_runtime(dreamerv3_root or run["upstream"]["root"], device, run["arguments"]["model_size"])
    import elements
    import jax
    import ninjax as nj
    from kuafu.dynamics.dreamer_model import WorldModel

    metadata = run["dataset"]
    model = WorldModel(metadata["state_dim"], metadata["action_dim"], elements.Config(run["agent_config"]),
                       burnin=run["arguments"]["burnin"], task_dim=metadata["task_dim"], name="wm")
    parameters = load_weights(checkpoint)
    predict = jax.jit(nj.pure(model.forecast))
    scale = np.asarray(metadata["action_scale"], np.float32)

    def forecast(context_states, previous_actions, future_actions, *, task, particles=4, seed=0):
        if task not in metadata["tasks"]:
            raise ValueError(f"Unknown task {task!r}; choose one of {metadata['tasks']}")
        task_meta = metadata["per_task"][task]
        states, previous, future = [np.asarray(value, np.float32)
                                    for value in (context_states, previous_actions, future_actions)]
        if any(value.ndim != 3 or not np.isfinite(value).all() for value in (states, previous, future)):
            raise ValueError("Context and actions must be finite rank-3 arrays")
        if (states.shape[:2] != previous.shape[:2] or len(states) != len(future)
                or states.shape[-1] != task_meta["state_dim"] or previous.shape[-1] != 9
                or future.shape[-1] != 9 or min(states.shape[:2]) < 1 or future.shape[1] < 1 or particles < 1):
            raise ValueError("Invalid task-specific state/action dimensions or particle count")
        columns = np.asarray(task_meta["shared_indices"], np.int32)
        mean = np.asarray(task_meta.get("state_mean", np.zeros(len(columns))), np.float32)
        state_scale = np.asarray(task_meta.get("state_scale", np.ones(len(columns))), np.float32)
        identity = np.eye(metadata["task_dim"], dtype=np.float32)[metadata["tasks"].index(task)]
        inputs = jax.device_put((canonical_state((states - mean) / state_scale, columns),
                                 np.clip(symlog(previous) / scale, -1, 1),
                                 np.clip(symlog(future) / scale, -1, 1),
                                 np.broadcast_to(identity, (len(states), len(identity))),
                                 np.broadcast_to(np.asarray(task_meta["state_mask"], np.float32),
                                                 (len(states), metadata["state_dim"]))))
        predictions = [np.asarray(predict(parameters, *inputs, seed=seed + i)[1])[..., columns] * state_scale + mean
                       for i in range(particles)]
        return np.stack(predictions, axis=1)

    return forecast, run
