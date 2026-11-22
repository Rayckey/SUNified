"""Train upstream DreamerV3's recurrent world model on offline demonstrations."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from kuafu.dynamics.sequences import SequenceDataset
from kuafu.paths import resolve_artifact_path


UPSTREAM_REVISION = "e01491fad6434b2245a3b8ca201dd7faedcc458c"


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def configure_runtime(root, device, model_size):
    """Load the pinned upstream source and initialize JAX on the requested device."""
    root = Path(root).expanduser().resolve()
    if not (root / "dreamerv3/rssm.py").is_file():
        raise ValueError("Set --dreamerv3-root or KUAFU_DREAMERV3_ROOT to the upstream checkout")
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if revision != UPSTREAM_REVISION:
        raise ValueError(f"Expected upstream revision {UPSTREAM_REVISION}, got {revision}")
    if subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True).strip():
        raise ValueError("Use a clean upstream DreamerV3 checkout for reproducible training")
    if device == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
    elif device.startswith("cuda:") and device[5:].isdigit():
        os.environ["CUDA_VISIBLE_DEVICES"] = device[5:]
        os.environ["JAX_PLATFORMS"] = "cuda"
    else:
        raise ValueError("device must be cpu or cuda:<GPU index>")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    sys.path.insert(0, str(root))
    import elements
    import jax
    import jax.numpy as jnp
    import embodied.jax.nets as nn
    from ruamel.yaml import YAML

    # Quadro RTX 8000 has no native bfloat16 arithmetic.
    nn.COMPUTE_DTYPE = jnp.float32
    # Match upstream's YAML parser: PyYAML interprets several exponent values as strings.
    configurations = YAML(typ="safe").load((root / "dreamerv3/configs.yaml").read_text())
    config = elements.Config(configurations["defaults"]).update(configurations[model_size]).agent
    return config, {"repository": "https://github.com/danijar/dreamerv3", "revision": revision,
                    "root": str(root), "devices": [str(d) for d in jax.devices()]}


def save_weights(path, parameters):
    """Atomically store array-only Ninjax parameters; loading never executes pickle."""
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **{key: np.asarray(value) for key, value in parameters.items()})
    temporary.replace(path)


def load_weights(path):
    """Load an array-only checkpoint onto the active JAX device."""
    import jax
    with np.load(path, allow_pickle=False) as source:
        return jax.device_put({key: source[key] for key in source.files})


def load_forecaster(checkpoint, *, device="cpu", dreamerv3_root=None):
    """Load a callable forecast(context_states, previous_actions, future_actions).

    Inputs use original dataset coordinates and command scaling. Context arrays
    are [B,C,D] and [B,C,A], where previous_actions[:,t] leads to states[:,t].
    Future actions [B,H,A] start at the last context state. The returned callable
    produces [B,particles,H,D] samples without observing future states.
    """
    checkpoint = Path(checkpoint).expanduser().resolve()
    run = json.loads((checkpoint.parent / "run.json").read_text())
    if run.get("format") == "kuafu-dreamerv3-pointcloud-world-model-v1":
        from kuafu.dynamics.pointcloud_dreamer import load_pointcloud_forecaster
        return load_pointcloud_forecaster(checkpoint, device=device, dreamerv3_root=dreamerv3_root)
    if run.get("format") == "kuafu-dreamerv3-rgb-world-model-v1":
        from kuafu.dynamics.rgb_dreamer import load_rgb_forecaster
        return load_rgb_forecaster(checkpoint, device=device, dreamerv3_root=dreamerv3_root)
    if run.get("format") == "kuafu-dreamerv3-shared-world-model-v1":
        from kuafu.dynamics.shared_dreamer import load_shared_forecaster
        return load_shared_forecaster(checkpoint, device=device, dreamerv3_root=dreamerv3_root)
    if run.get("format") != "kuafu-dreamerv3-world-model-v1":
        raise ValueError("Unsupported Dreamer world-model checkpoint format")
    root = dreamerv3_root or run["upstream"]["root"]
    configure_runtime(root, device, run["arguments"]["model_size"])
    import elements
    import jax
    import ninjax as nj
    from kuafu.dynamics.dreamer_model import WorldModel

    metadata = run["dataset"]
    model = WorldModel(metadata["state_dim"], metadata["action_dim"], elements.Config(run["agent_config"]),
                        burnin=run["arguments"]["burnin"], name="wm")
    parameters = load_weights(checkpoint)
    predict = jax.jit(nj.pure(model.forecast))
    scale = np.asarray(metadata["action_scale"], np.float32)

    def forecast(context_states, previous_actions, future_actions, *, particles=4, seed=0):
        from kuafu.dynamics.sequences import symlog
        states, previous, future = [np.asarray(value, np.float32)
                                    for value in (context_states, previous_actions, future_actions)]
        if any(value.ndim != 3 or not np.isfinite(value).all() for value in (states, previous, future)):
            raise ValueError("Context and actions must be finite rank-3 arrays")
        if (states.shape[:2] != previous.shape[:2] or len(states) != len(future)
                or states.shape[-1] != metadata["state_dim"]
                or previous.shape[-1] != metadata["action_dim"] or future.shape[-1] != metadata["action_dim"]
                or states.shape[0] < 1 or states.shape[1] < 1 or future.shape[1] < 1 or particles < 1):
            raise ValueError("Invalid context/action dimensions or particle count")
        inputs = jax.device_put((states, np.clip(symlog(previous) / scale, -1, 1),
                                 np.clip(symlog(future) / scale, -1, 1)))
        predictions = [np.asarray(predict(parameters, *inputs, seed=seed + index)[1])
                       for index in range(particles)]
        return np.stack(predictions, axis=1)

    return forecast, run


def evaluate_forecasts(parameters, forecast_fn, data, split, *, batch_size, batches,
                       context, horizon, particles, seed):
    """Measure open-loop predictions after context assimilation on fixed windows."""
    import jax
    rng = np.random.default_rng(seed)
    mse = np.zeros((horizon, data.metadata["state_dim"]), np.float64)
    persistence = np.zeros_like(mse)
    for index in range(batches):
        batch = data.sample(split, batch_size, context + horizon, rng)
        targets = batch["state"][:, context:]
        initial = batch["state"][:, context - 1:context]
        inputs = jax.device_put((batch["state"][:, :context], batch["prev_action"][:, :context],
                                 batch["prev_action"][:, context:]))
        predictions = []
        for particle in range(particles):
            _, prediction = forecast_fn(parameters, *inputs, seed=seed + index * particles + particle)
            predictions.append(np.asarray(prediction))
        prediction = np.mean(predictions, axis=0)
        if not np.isfinite(prediction).all():
            raise FloatingPointError("Nonfinite Dreamer forecast")
        mse += np.square(prediction.astype(np.float64) - targets).sum(0)
        persistence += np.square(initial.astype(np.float64) - targets).sum(0)
    mse /= batch_size * batches
    persistence /= batch_size * batches
    result = {"windows": batch_size * batches, "context": context, "particles": particles, "horizons": {}}
    for step in sorted({1, min(5, horizon), min(10, horizon), horizon}):
        result["horizons"][str(step)] = {
            "state_normalized_rmse": float(np.sqrt(np.mean(mse[step - 1] / data.state_std ** 2))),
            "persistence_state_normalized_rmse": float(np.sqrt(np.mean(persistence[step - 1] / data.state_std ** 2))),
            "rmse_per_dim": np.sqrt(mse[step - 1]).tolist(),
            "persistence_rmse_per_dim": np.sqrt(persistence[step - 1]).tolist(),
        }
    return result


def run_training(args):
    """Train only Dreamer's world-model modules and select by validation forecast error."""
    for key in ("steps", "batch_size", "length", "report_every", "eval_batches", "particles", "context", "horizon"):
        if getattr(args, key) < 1:
            raise ValueError(f"{key} must be positive")
    if not 0 <= args.burnin < args.length:
        raise ValueError("burnin must be nonnegative and shorter than sequence length")
    output = resolve_artifact_path(args.output, "DreamerV3 output")
    if output.exists():
        raise ValueError(f"Use a new output directory; refusing to overwrite {output}")
    config, upstream = configure_runtime(args.dreamerv3_root, args.device, args.model_size)
    import jax
    import ninjax as nj
    from kuafu.dynamics.dreamer_model import WorldModel

    data = SequenceDataset(args.dataset, obs_key=args.obs_key, action_key=args.action_key,
                           exclude_last=args.exclude_last, skip_first=args.skip_first, seed=args.seed)
    start_step, source_run = 0, None
    if args.resume_from:
        source_run = Path(args.resume_from).expanduser().resolve()
        previous = json.loads((source_run / "run.json").read_text())
        progress = json.loads((source_run / "progress.json").read_text())
        if progress.get("phase") != "complete" or not (source_run / "last.npz").is_file():
            raise ValueError("resume-from must contain a completed run with last.npz")
        serialized_config = json.loads(json.dumps(dict(config)))
        if previous["dataset"] != data.metadata or previous["split"] != data.split or previous["agent_config"] != serialized_config:
            raise ValueError("Resume requires identical dataset, preprocessing, splits, and model configuration")
        for key in ("seed", "batch_size", "length", "burnin", "context", "horizon", "eval_batches", "particles"):
            if previous["arguments"][key] != getattr(args, key):
                raise ValueError(f"Resume requires unchanged {key}")
        start_step = progress["step"]
        if args.steps <= start_step:
            raise ValueError("steps must exceed the resumed run's completed update count")
    model = WorldModel(data.metadata["state_dim"], data.metadata["action_dim"], config,
                        burnin=args.burnin, name="wm")
    rng = np.random.default_rng(args.seed)
    batch = jax.device_put(data.sample("train", args.batch_size, args.length, rng))
    if start_step:
        # Recreate the exact sampler state without loading the old batches.
        _, cumulative = data._windows[("train", args.length)]
        for _ in range(start_step):
            rng.integers(cumulative[-1], size=args.batch_size)
    # Validate evaluation windows before creating a run or compiling expensive kernels.
    for split in ("validation", "test"):
        data.sample(split, args.batch_size, args.context + args.horizon, np.random.default_rng(args.seed))
    output.mkdir(parents=True)
    run = {"format": "kuafu-dreamerv3-world-model-v1", "arguments": vars(args),
           "upstream": upstream, "agent_config": dict(config), "dataset": data.metadata,
           "split": data.split, "compute_dtype": "float32", "start_step": start_step,
           "selection": "validation open-loop final-horizon state-normalized RMSE",
           "forecast_output_units": "dataset units after inverse signed-log transform",
           "versions": {name: importlib.metadata.version(name) for name in
                        ("jax", "jaxlib", "ninjax", "elements", "optax", "chex", "portal", "granular", "numpy", "h5py")}}
    write_json(output / "run.json", run)
    write_json(output / "progress.json", {"phase": "initializing", "step": 0})
    print(json.dumps({"output": str(output), "upstream": upstream,
                      "split_episodes": {key: len(value) for key, value in data.split.items()},
                      "usable_transitions": data.metadata["usable_transitions"]}), flush=True)
    # Ninjax also initializes Python-side optimizer tree definitions. Restore
    # arrays only after constructing that structure, even when weights exist.
    parameters = nj.init(model.train)({}, batch, seed=args.seed)
    if source_run:
        restored = load_weights(source_run / "last.npz")
        if parameters.keys() != restored.keys() or any(parameters[key].shape != restored[key].shape for key in parameters):
            raise ValueError("Resume checkpoint parameter/optimizer structure does not match the model")
        parameters = restored
    train_fn = jax.jit(nj.pure(model.train))
    forecast_fn = jax.jit(nj.pure(model.forecast))
    loss_fn = jax.jit(nj.pure(model.loss))
    eval_options = dict(batch_size=args.batch_size, batches=args.eval_batches, context=args.context,
                        horizon=args.horizon, particles=args.particles)
    best_score, best_step = float("inf"), 0
    if source_run:
        best = json.loads((source_run / "best.json").read_text())
        best_step = best["step"]
        best_score = best["validation"]["horizons"][str(args.horizon)]["state_normalized_rmse"]
        shutil.copyfile(source_run / "best.npz", output / "best.npz")
        write_json(output / "best.json", best)
    started = time.monotonic()
    for step in range(start_step + 1, args.steps + 1):
        batch = jax.device_put(data.sample("train", args.batch_size, args.length, rng))
        parameters, metrics = train_fn(parameters, batch, seed=args.seed + step)
        if step == start_step + 1 or step % args.report_every == 0 or step == args.steps:
            metrics = {key: float(value) for key, value in metrics.items()}
            if not all(np.isfinite(value) for value in metrics.values()):
                raise FloatingPointError(f"Nonfinite Dreamer training metrics at step {step}")
            validation = evaluate_forecasts(parameters, forecast_fn, data, "validation",
                                            seed=args.seed + 100000, **eval_options)
            valbatch = data.sample("validation", args.batch_size, args.length, np.random.default_rng(args.seed + 123))
            _, (_, losses) = loss_fn(parameters, jax.device_put(valbatch), seed=args.seed + 999)
            score = validation["horizons"][str(args.horizon)]["state_normalized_rmse"]
            if score < best_score:
                best_score, best_step = score, step
                save_weights(output / "best.npz", {key: value for key, value in parameters.items() if not key.startswith("wm/opt/")})
                write_json(output / "best.json", {"step": step, "validation": validation})
            record = {"step": step, "elapsed_seconds": time.monotonic() - started,
                      "train": metrics, "validation_loss": {key: float(value) for key, value in losses.items()},
                      "validation_forecast_rmse": score, "best_step": best_step}
            with (output / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
            write_json(output / "progress.json", {"phase": "training", **record})
            print(json.dumps(record, allow_nan=False), flush=True)
    save_weights(output / "last.npz", parameters)
    best = load_weights(output / "best.npz")
    report = {"steps_completed": args.steps, "best_step": best_step,
              "elapsed_seconds": time.monotonic() - started,
              "validation": json.loads((output / "best.json").read_text())["validation"],
              "test": evaluate_forecasts(best, forecast_fn, data, "test", seed=args.seed + 200000,
                                          **{**eval_options, "batches": 2 * args.eval_batches})}
    write_json(output / "evaluation.json", report)
    write_json(output / "progress.json", {"phase": "complete", "step": args.steps, "best_step": best_step})
    print(json.dumps({"finished": str(output / "best.npz"), "best_step": best_step,
                      "test_final_horizon": report["test"]["horizons"][str(args.horizon)]}), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset")
    source.add_argument("--dataset-index", help="Completed Kinova dataset index for one shared world model")
    source.add_argument("--rgb-index", help="Completed synchronized RGB dataset index for a shared visual world model")
    source.add_argument("--pointcloud-index", help="Completed XYZ point-cloud index for a shared visual world model")
    parser.add_argument("--num-points", type=int, default=512)
    parser.add_argument("--image-height", type=int, default=64)
    parser.add_argument("--image-width", type=int, default=112)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task", default="Kinova-Table-Gen3-Pen-Cup")
    parser.add_argument("--resume-from", default="", help="Completed run directory; steps is the total target update count")
    parser.add_argument("--dreamerv3-root", default=os.environ.get("KUAFU_DREAMERV3_ROOT", ""))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model-size", choices=("size1m", "size12m", "size25m", "size50m"), default="size1m")
    parser.add_argument("--obs-key", default="full_state")
    parser.add_argument("--action-key", default="relative_joint")
    for key, default in {"exclude_last": 1, "skip_first": 1, "seed": 42, "steps": 5000,
                         "batch_size": 16, "length": 64, "burnin": 8, "report_every": 250,
                         "eval_batches": 8, "context": 16, "horizon": 20, "particles": 4}.items():
        parser.add_argument("--" + key.replace("_", "-"), type=int, default=default)
    args = parser.parse_args(argv)
    try:
        if args.pointcloud_index:
            from kuafu.dynamics.pointcloud_dreamer import run_pointcloud_training
            run_pointcloud_training(args)
        elif args.rgb_index:
            from kuafu.dynamics.rgb_dreamer import run_rgb_training
            run_rgb_training(args)
        elif args.dataset_index:
            from kuafu.dynamics.shared_dreamer import run_shared_training
            run_shared_training(args)
        else:
            run_training(args)
    except (ValueError, OSError, ImportError, subprocess.CalledProcessError) as error:
        parser.exit(2, f"ERROR: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
