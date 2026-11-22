"""Train and inspect a shared DreamerV3 world model from synchronized RGB demos."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import time

import numpy as np

from kuafu.dynamics.dreamerv3 import configure_runtime, load_weights, save_weights, write_json
from kuafu.dynamics.rgb_sequences import RGBSequenceDataset, file_hash, prepare_rgb_cache
from kuafu.dynamics.sequences import symlog
from kuafu.paths import resolve_artifact_path


FORMAT = "kuafu-dreamerv3-rgb-world-model-v1"


def evaluate_rgb(parameters, predict, data, split, *, context, horizon, batches, particles, seed,
                 batch_size=4, preview=None):
    """Score fixed open-loop RGB predictions against future frames and persistence."""
    import jax

    tasks = {}
    for task_id, task in enumerate(data.tasks):
        task_seed = seed + 10000 * task_id
        rng = np.random.default_rng(task_seed)
        mse, baseline, moving_mse, moving_base, moving_count = [np.zeros(horizon, np.float64) for _ in range(5)]
        for index in range(batches):
            batch = data.sample_task(task, split, batch_size, context + horizon, rng, pad=False)
            inputs = jax.device_put((batch["rgb"][:, :context], batch["prev_action"][:, :context],
                                    batch["prev_action"][:, context:], batch["task"][:, 0]))
            predictions = [np.asarray(predict(parameters, *inputs, seed=task_seed + index * particles + particle)[1])
                           for particle in range(particles)]
            prediction = np.mean(predictions, axis=0)
            if not np.isfinite(prediction).all():
                raise FloatingPointError(f"{task}: nonfinite RGB forecast")
            target = batch["rgb"][:, context:].astype(np.float32) / 255
            initial = batch["rgb"][:, context - 1:context].astype(np.float32) / 255
            error, persist = np.square(prediction - target), np.square(initial - target)
            mse += error.mean((0, 2, 3, 4))
            baseline += persist.mean((0, 2, 3, 4))
            moving = np.repeat((np.abs(target - initial).reshape(*target.shape[:-1], 2, 3).mean(-1) > 0.04), 3, axis=-1)
            moving_mse += (error * moving).sum((0, 2, 3, 4))
            moving_base += (persist * moving).sum((0, 2, 3, 4))
            moving_count += moving.sum((0, 2, 3, 4))
            if preview is not None and index == 0:
                _preview(Path(preview) / f"{task}.png", initial[0, 0], target[0], prediction[0])
        mse, baseline = mse / batches, baseline / batches
        horizons = {}
        for step in sorted({1, min(5, horizon), min(10, horizon), horizon}):
            h = step - 1
            horizons[str(step)] = {
                "rgb_rmse": float(np.sqrt(mse[h])), "persistence_rgb_rmse": float(np.sqrt(baseline[h])),
                "rgb_psnr_db": float(-10 * np.log10(max(mse[h], 1e-12))),
                "moving_region_rgb_rmse": float(np.sqrt(moving_mse[h] / max(moving_count[h], 1))),
                "persistence_moving_region_rgb_rmse": float(np.sqrt(moving_base[h] / max(moving_count[h], 1))),
                "moving_pixels": int(moving_count[h]),
            }
        tasks[task] = {"windows": batches * batch_size, "horizons": horizons}
    macro = {step: {key: float(np.mean([r["horizons"][step][key] for r in tasks.values()]))
                    for key in ("rgb_rmse", "persistence_rgb_rmse", "moving_region_rgb_rmse",
                                "persistence_moving_region_rgb_rmse")}
             for step in next(iter(tasks.values()))["horizons"]}
    return {"per_task": tasks, "macro_average": macro, "context": context, "horizon": horizon,
            "particles": particles, "rgb_units": "[0,1]", "aggregation": "equal task mean",
            "moving_region": "per-camera mean absolute RGB change from last context frame > 0.04"}


def _preview(path, initial, target, prediction):
    from PIL import Image, ImageDraw
    height, width = initial.shape[:2]
    steps = sorted({0, len(target) // 2, len(target) - 1})
    canvas = Image.new("RGB", (width * 4, (height + 22) * 4), "white")
    draw = ImageDraw.Draw(canvas)
    for camera in range(2):
        for kind, values in enumerate((target, prediction)):
            y = (2 * camera + kind) * (height + 22)
            draw.text((4, y + 3), f"Camera {camera}: {'truth' if kind == 0 else 'forecast'}", fill="black")
            for column, frame in enumerate([initial] + [values[t] for t in steps]):
                image = np.clip(frame[..., 3 * camera:3 * camera + 3] * 255, 0, 255).astype(np.uint8)
                canvas.paste(Image.fromarray(image), (column * width, y + 22))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.resize((canvas.width * 2, canvas.height * 2)).save(path)


def run_rgb_training(args):
    return run_visual_training(args, modality="rgb")


def run_visual_training(args, *, modality):
    """Train visual weights from scratch; retain best validation and resumable last weights."""
    if modality not in ("rgb", "pointcloud"):
        raise ValueError("Unsupported visual modality")
    pointcloud = modality == "pointcloud"
    if pointcloud:
        from kuafu.dynamics.pointcloud_dreamer import FORMAT as format_name, evaluate_pointcloud as evaluate
        from kuafu.dynamics.pointcloud_sequences import PointCloudSequenceDataset, prepare_pointcloud_cache
    else:
        format_name, evaluate = FORMAT, evaluate_rgb
    observation_key = "points" if pointcloud else "rgb"
    score_key = "chamfer_rmse_m" if pointcloud else "rgb_rmse"
    for key in ("steps", "batch_size", "length", "context", "horizon", "particles", "report_every", "eval_batches"):
        if getattr(args, key) < 1:
            raise ValueError(f"{key} must be positive")
    if not 0 <= args.burnin < args.length:
        raise ValueError("burnin must be shorter than the training sequence")
    source_run = Path(args.resume_from).resolve() if args.resume_from else None
    output = resolve_artifact_path(args.output, f"{modality} DreamerV3 output")
    if output.exists():
        raise ValueError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True)
    write_json(output / "progress.json", {"phase": f"preparing_{modality}", "step": 0})
    cache = output / f"{modality}_cache"
    if source_run:
        cache.symlink_to((source_run / f"{modality}_cache").resolve(), target_is_directory=True)
    if pointcloud:
        prepare_pointcloud_cache(args.pointcloud_index, cache, num_points=args.num_points)
        data = PointCloudSequenceDataset(cache, seed=args.seed)
    else:
        prepare_rgb_cache(args.rgb_index, cache, height=args.image_height, width=args.image_width)
        data = RGBSequenceDataset(cache, seed=args.seed)
    rng = np.random.default_rng(args.seed)
    batch = data.sample("train", args.batch_size, args.length, rng)
    for task in data.tasks:
        if min(len(e[observation_key]) for e in data.episodes[task]) <= args.burnin:
            raise ValueError(f"{task}: episode shorter than burnin")
        for split in ("validation", "test"):
            data.sample_task(task, split, 1, args.context + args.horizon, np.random.default_rng(0), pad=False)
    config, upstream = configure_runtime(args.dreamerv3_root, args.device, args.model_size)
    # The upstream strided-CNN option reduces full-resolution convolution cost.
    if not pointcloud:
        config = config.update({"enc.simple.strided": True, "dec.simple.strided": True})
    start_step, checkpoint = 0, None
    if source_run:
        previous = json.loads((source_run / "run.json").read_text())
        checkpoint = json.loads((source_run / "checkpoint.json").read_text())
        if (previous.get("format") != format_name or previous["dataset"] != data.metadata
                or previous["split"] != data.split or previous["agent_config"] != json.loads(json.dumps(dict(config)))):
            raise ValueError("Visual resume requires unchanged data, splits, and model configuration")
        for key in ("seed", "batch_size", "length", "burnin", "context", "horizon", "particles", "eval_batches"):
            if previous["arguments"][key] != getattr(args, key):
                raise ValueError(f"Visual resume requires unchanged {key}")
        if file_hash(source_run / "last.npz") != checkpoint["sha256"]:
            raise ValueError("Visual optimizer checkpoint and sampler state are inconsistent")
        start_step = checkpoint["step"]
        if args.steps <= start_step:
            raise ValueError("steps must exceed the saved update count")
    import jax
    import ninjax as nj
    if pointcloud:
        from kuafu.dynamics.pointcloud_model import PointCloudWorldModel
        model = PointCloudWorldModel(args.num_points, 9, len(data.tasks), config, args.burnin, name="wm")
    else:
        from kuafu.dynamics.rgb_model import RGBWorldModel
        model = RGBWorldModel(data.metadata["image_shape"], 9, len(data.tasks), config, args.burnin, name="wm")
    run = {"format": format_name, "arguments": vars(args), "upstream": upstream,
           "agent_config": dict(config), "dataset": data.metadata, "split": data.split,
           "compute_dtype": "float32", "selection": f"equal-task validation final-horizon {score_key}",
           "encoder_inputs": [observation_key, "task"], "dynamics_inputs": ["previous_action", "task"],
           "forecast_output_units": ("XYZ float32 in environment-local meters" if pointcloud
                                     else "RGB float32 in [0,1], cameras concatenated by channel"),
           "privileged_state_input": False, "actor_critic": False}
    if pointcloud:
        run["modality_adaptation"] = "PointNet-style encoder, MLP point-set decoder, symmetric Chamfer reconstruction; upstream DreamerV3 RSSM"
    run["start_step"] = start_step
    write_json(output / "run.json", run)
    write_json(output / "progress.json", {"phase": "initializing", "step": 0})
    parameters = nj.init(model.train)({}, jax.device_put(batch), seed=args.seed)
    if source_run:
        restored = load_weights(source_run / "last.npz")
        if parameters.keys() != restored.keys() or any(parameters[k].shape != restored[k].shape for k in parameters):
            raise ValueError("Visual checkpoint structure mismatch")
        parameters = restored
        rng.bit_generator.state = checkpoint["sampler_state"]
    predict = jax.jit(nj.pure(model.forecast))
    train = jax.jit(nj.pure(model.train))
    options = dict(context=args.context, horizon=args.horizon, batches=args.eval_batches,
                   particles=args.particles, batch_size=2)
    best_score, best_step = float("inf"), 0
    if source_run:
        best = json.loads((source_run / "best.json").read_text())
        best_step = best["step"]
        if best_step > start_step:
            raise ValueError("Best checkpoint is newer than the saved optimizer state")
        best_score = best["validation"]["macro_average"][str(args.horizon)][score_key]
        shutil.copyfile(source_run / "best.npz", output / "best.npz")
        write_json(output / "best.json", best)
    started, recent_time, recent_step = time.monotonic(), time.monotonic(), start_step
    print(json.dumps({f"training_{modality}": str(output), "tasks": data.tasks,
                      "parameters": sum(v.size for k, v in parameters.items() if not k.startswith("wm/opt/"))}), flush=True)
    for step in range(start_step + 1, args.steps + 1):
        parameters, metrics = train(parameters, jax.device_put(data.sample("train", args.batch_size, args.length, rng)),
                                    seed=args.seed + step)
        if step == start_step + 1 or step % 25 == 0:
            jax.block_until_ready(metrics)
            elapsed = time.monotonic() - recent_time
            seconds_per_step = elapsed / (step - recent_step)
            write_json(output / "progress.json", {"phase": "training", "step": step, "target": args.steps,
                       "elapsed_seconds": time.monotonic() - started, "seconds_per_step": seconds_per_step,
                       "eta_training_seconds": (args.steps - step) * seconds_per_step, "best_step": best_step})
            recent_time, recent_step = time.monotonic(), step
        if step == start_step + 1 or step % args.report_every == 0 or step == args.steps:
            metrics = {key: float(value) for key, value in metrics.items()}
            if not all(np.isfinite(value) for value in metrics.values()):
                raise FloatingPointError(f"Nonfinite {modality} training metric at step {step}")
            validation = evaluate(parameters, predict, data, "validation", seed=args.seed + 100000, **options)
            score = validation["macro_average"][str(args.horizon)][score_key]
            if score < best_score:
                best_score, best_step = score, step
                save_weights(output / "best.npz", {k: v for k, v in parameters.items() if not k.startswith("wm/opt/")})
                write_json(output / "best.json", {"step": step, "validation": validation})
            # Periodic optimizer checkpoints protect longer visual runs.
            save_weights(output / "last.npz", parameters)
            write_json(output / "sampler_state.json", rng.bit_generator.state)
            checkpoint = {"step": step, "sha256": file_hash(output / "last.npz"),
                          "sampler_state": rng.bit_generator.state}
            temporary = output / "checkpoint.tmp"
            write_json(temporary, checkpoint)
            temporary.replace(output / "checkpoint.json")
            record = {"step": step, "elapsed_seconds": time.monotonic() - started, "train": metrics,
                      "validation": validation["macro_average"][str(args.horizon)], "best_step": best_step}
            with (output / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
            print(json.dumps(record, allow_nan=False), flush=True)
            recent_time, recent_step = time.monotonic(), step
    best = load_weights(output / "best.npz")
    test = evaluate(best, predict, data, "test", seed=args.seed + 200000,
                        preview=output / "test_previews", **{**options, "batches": 2 * args.eval_batches})
    report = {"steps_completed": args.steps, "best_step": best_step, "elapsed_seconds": time.monotonic() - started,
              "validation": json.loads((output / "best.json").read_text())["validation"], "test": test}
    write_json(output / "evaluation.json", report)
    write_json(output / "progress.json", {"phase": "complete", "step": args.steps, "best_step": best_step})
    print(json.dumps({f"finished_{modality}": str(output / "best.npz"), "test": test["macro_average"][str(args.horizon)]}), flush=True)
    return report


def load_rgb_forecaster(checkpoint, *, device="cpu", dreamerv3_root=None):
    """Load forecast(context_rgb, previous_actions, future_actions, task=...).

    RGB must be uint8 [B,C,H,W,6], with both cameras in recorded order. Actions
    use the original applied relative-joint units. Output: [B,P,Horizon,H,W,6]
    float32 RGB in [0,1]. Only visual context is assimilated.
    """
    checkpoint = Path(checkpoint).resolve()
    run = json.loads((checkpoint.parent / "run.json").read_text())
    if run.get("format") != FORMAT:
        raise ValueError("Expected an RGB DreamerV3 checkpoint")
    configure_runtime(dreamerv3_root or run["upstream"]["root"], device, run["arguments"]["model_size"])
    import elements
    import jax
    import ninjax as nj
    from kuafu.dynamics.rgb_model import RGBWorldModel
    metadata = run["dataset"]
    tasks = sorted(metadata["tasks"])
    model = RGBWorldModel(metadata["image_shape"], 9, len(tasks), elements.Config(run["agent_config"]),
                          run["arguments"]["burnin"], name="wm")
    parameters = load_weights(checkpoint)
    predict = jax.jit(nj.pure(model.forecast))
    scale = np.asarray(metadata["action_scale"], np.float32)

    def forecast(context_rgb, previous_actions, future_actions, *, task, particles=1, seed=0):
        rgb = np.asarray(context_rgb)
        previous, future = np.asarray(previous_actions, np.float32), np.asarray(future_actions, np.float32)
        if (task not in tasks or rgb.dtype != np.uint8 or rgb.ndim != 5
                or tuple(rgb.shape[2:]) != tuple(metadata["image_shape"]) or min(rgb.shape[:2]) < 1
                or previous.shape != (*rgb.shape[:2], 9) or future.ndim != 3
                or future.shape[0] != rgb.shape[0] or future.shape[-1] != 9 or future.shape[1] < 1
                or particles < 1 or not np.isfinite(previous).all() or not np.isfinite(future).all()):
            raise ValueError("Invalid RGB/action dimensions, task, dtype, or particle count")
        onehot = np.eye(len(tasks), dtype=np.float32)[[tasks.index(task)]].repeat(len(rgb), axis=0)
        inputs = jax.device_put((rgb, np.clip(symlog(previous) / scale, -1, 1),
                                np.clip(symlog(future) / scale, -1, 1), onehot))
        return np.stack([np.asarray(predict(parameters, *inputs, seed=seed + i)[1]) for i in range(particles)], axis=1)
    return forecast, run
