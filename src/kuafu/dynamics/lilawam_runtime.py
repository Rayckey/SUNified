"""Train the pinned official LiLa-WAM on Kuafu recordings and score held-out data."""

from __future__ import annotations

from contextlib import nullcontext
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from kuafu.dynamics.lilawam_data import KinovaLiLaDataset, camera_image, file_hash, write_json
from kuafu.paths import resolve_artifact_path


UPSTREAM_REVISION = "339860f65d5cde9e229c037b672889bc4f70497c"


def load_config(path):
    from omegaconf import OmegaConf
    config = OmegaConf.load(path)
    root = Path(config.kuafu.upstream).resolve()
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if revision != UPSTREAM_REVISION:
        raise ValueError(f"Expected LiLa-WAM revision {UPSTREAM_REVISION}, got {revision}")
    # Importing the upstream package creates untracked __pycache__ directories.
    if subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], text=True).strip():
        raise ValueError("Tracked files in the pinned upstream checkout must be unchanged")
    sys.path.insert(0, str(root))
    from models.model_runner import ModelFactory
    import models.model_runner as imported
    if not Path(imported.__file__).resolve().is_relative_to(root):
        raise ValueError("A different 'models' package shadowed the pinned LiLa-WAM checkout")
    return config, ModelFactory


def precision(config, device):
    import torch
    if config.kuafu.precision == "fp16" and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    return nullcontext()


def encoder(config, factory, device):
    import torch
    path = Path(config.model.vision_encoder.checkpoint_path)
    if not (path / "config.json").exists() or not (path / "model.safetensors").exists():
        raise FileNotFoundError(f"Pretrained DINOv3 weights are required at {path}; random weights are not a training fallback")
    meta = json.loads((path / "config.json").read_text())
    if meta.get("hidden_size") != 1024 or meta.get("num_hidden_layers") != 24 or meta.get("patch_size") != 16:
        raise ValueError("Expected the official DINOv3 ViT-L/16 encoder configuration")
    return factory.create_vision_encoder(str(path), torch.float32, device)


def prepare_conditions(config_path):
    """Compute VTTs from training episodes only, using the frozen pretrained encoder."""
    import h5py
    import torch
    config, factory = load_config(config_path)
    device = torch.device(config.kuafu.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.set_num_threads(4)
    manifest = Path(config.kuafu.manifest)
    data = json.loads(manifest.read_text())
    target = resolve_artifact_path(config.dataset.task_cond_dir, "LiLa-WAM task conditions")
    if (target / "meta.json").exists():
        old = json.loads((target / "meta.json").read_text())
        validate_conditions(config)
        return old
    model, width, _, _ = encoder(config, factory, device)
    target.mkdir(parents=True, exist_ok=True)
    meta = {"manifest_sha256": file_hash(manifest), "fit_split": "train", "tasks": {},
            "encoder_sha256": file_hash(Path(config.model.vision_encoder.checkpoint_path) / "model.safetensors"),
            "feature": "mean(last usable CLS - first usable CLS), pretrained frozen DINOv3",
            "image_size": list(config.dataset.image_size)}
    for task, spec in sorted(data["tasks"].items()):
        total, count, pending = np.zeros(width, np.float64), 0, []

        def consume():
            nonlocal total, count, pending
            if not pending:
                return
            batch = torch.from_numpy(np.stack(pending)).to(device)
            with torch.no_grad(), precision(config, device):
                cls = model(pixel_values=batch, return_dict=True).last_hidden_state[:, 0].float().cpu().numpy()
            diffs = cls[1::2] - cls[::2]
            total += diffs.sum(0, dtype=np.float64)
            count += len(diffs)
            pending = []

        with h5py.File(spec["source"]["path"], "r") as handle:
            for episode in spec["episodes"]:
                if episode["split"] != "train":
                    continue
                rgb = handle["data"][episode["name"]]["obs/rgb"]
                pending.extend(camera_image(rgb[t], config.dataset.image_size) for t in (1, len(rgb) - 1))
                if len(pending) >= int(config.kuafu.vtt_batch_size):
                    consume()
            consume()
        vector = (total / count).astype(np.float32)
        if not np.isfinite(vector).all() or count != len(data["splits"]["train"][task]):
            raise ValueError(f"Invalid VTT for {task}")
        directory = target / task
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "task_cond.npy", vector)
        meta["tasks"][task] = {"episodes": count, "dimension": width, "sha256": file_hash(directory / "task_cond.npy")}
        print(json.dumps({"prepared_vtt": task, "training_episodes": count}), flush=True)
    write_json(target / "meta.json", meta)
    return meta


def validate_conditions(config):
    """Reject changed inputs or task vectors before initializing a training run."""
    manifest = json.loads(Path(config.kuafu.manifest).read_text())
    target = Path(config.dataset.task_cond_dir)
    meta = json.loads((target / "meta.json").read_text())
    if (config.common.action_dim != 9 or config.common.state_dim != 9 or config.common.proprio_len != 1
            or config.common.action_chunk_size != manifest["horizon"]
            or list(config.dataset.image_size) != manifest["image_size"]
            or manifest["normalization_sha256"] != file_hash(config.kuafu.normalization)):
        raise ValueError("Configuration or normalization does not match the prepared recordings")
    if (meta["fit_split"] != "train" or meta["manifest_sha256"] != file_hash(config.kuafu.manifest)
            or meta["encoder_sha256"] != file_hash(Path(config.model.vision_encoder.checkpoint_path) / "model.safetensors")):
        raise ValueError("Task conditions must use this encoder and this manifest's training split")
    if set(meta["tasks"]) != set(manifest["tasks"]):
        raise ValueError("Task-condition coverage does not match the recordings")
    for task, record in meta["tasks"].items():
        if record["sha256"] != file_hash(target / task / "task_cond.npy"):
            raise ValueError(f"Task condition changed for {task}")
    return meta


def build_model(config, factory, device):
    import torch
    from models.model_runner import VLAWrapper
    vision, width, registers, patch = encoder(config, factory, device)
    action = factory.create_action_model(config, width, len(config.model.vision_encoder.feat_layers), width, patch)
    model = VLAWrapper(vision, action, config.training.time_sampler,
        list(config.model.vision_encoder.feat_layers), config.model.vision_encoder.include_cls_register,
        registers, device, torch.float32, config.kuafu.normalization,
        train_config={"time_mu": config.training.time_mu, "time_sigma": config.training.time_sigma,
                      "use_vel_weight": False, "vel_weight_alpha": .2, "vel_weight_sigma": .01,
                      "use_future_feat": True, "lambda_future_feat": config.training.lambda_future_feat},
        future_feat_target_layer=config.model.future_feat.target_layer).to(device)
    return model


def balanced_indices(dataset, seed, samples):
    """Sample equal task exposure, uniformly over valid anchors within each task."""
    rng = np.random.default_rng(seed)
    task_ids = np.arange(samples) % len(dataset.tasks)
    rng.shuffle(task_ids)
    result = np.empty(samples, np.int64)
    for i, task in enumerate(dataset.tasks):
        selected = task_ids == i
        lo, hi = dataset.task_ranges[task]
        result[selected] = rng.integers(lo, hi, size=int(selected.sum()))
    return result


def evaluate(model, dataset, config, *, windows_per_task=16):
    """Score action chunks and future features without passing future observations to the policy."""
    import torch
    import torch.nn.functional as F
    from torch.utils.data import default_collate
    device = model.device
    tasks = {}
    model.eval()
    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.no_grad(), torch.random.fork_rng(devices=devices):
        for task_id, task in enumerate(dataset.tasks):
            rng = np.random.default_rng(42000 + task_id)
            lo, hi = dataset.task_ranges[task]
            indices = rng.choice(np.arange(lo, hi), size=windows_per_task, replace=False)
            sums = {key: 0.0 for key in ("action_mse_symlog", "repeat_previous_action_mse_symlog",
                                       "future_cosine_distance", "current_feature_cosine_distance")}
            for begin in range(0, windows_per_task, 2):
                torch.manual_seed(42000 + task_id * 100 + begin)
                samples = [dataset[int(i)] for i in indices[begin:begin + 2]]
                batch = default_collate(samples)
                size = len(samples)
                with precision(config, device):
                    features = model.get_vision_features(batch["pixel_values"])
                    state = model.normalize_state(batch["state"].to(device))
                    condition = batch["task_cond"].to(device)
                    actions = torch.randn((size, config.common.action_chunk_size, 9), device=device)
                    for step in range(config.common.num_inference_steps):
                        t = torch.full((size,), step / config.common.num_inference_steps, device=device)
                        pred = model.action_model(t=t, noisy_actions=actions, dino_features_list=features,
                                                  qpos_history=state, task_cond=condition)
                        actions = actions + pred["final_pred"] / config.common.num_inference_steps
                    final = model.action_model(t=torch.ones(size, device=device), noisy_actions=actions,
                                               dino_features_list=features, qpos_history=state, task_cond=condition)
                    future_prediction = model.action_model.future_feat_decoder(final["cond_tokens"])
                    future_target = model.get_future_target_features(batch["future_pixel_values"])
                    target_layer = list(config.model.vision_encoder.feat_layers).index(config.model.future_feat.target_layer)
                    current_target = features[target_layer][:, 1 + model.num_register_tokens:]
                    prediction = model.denormalize_action(actions).float()
                target = batch["action_sequence"].to(device)
                previous = batch["previous_action"].to(device)[:, None]
                values = {"action_mse_symlog": ((prediction - target) ** 2).mean().item(),
                          "repeat_previous_action_mse_symlog": ((previous - target) ** 2).mean().item(),
                          "future_cosine_distance": (1 - F.cosine_similarity(future_prediction.float(), future_target.float(), dim=-1)).mean().item(),
                          "current_feature_cosine_distance": (1 - F.cosine_similarity(current_target.float(), future_target.float(), dim=-1)).mean().item()}
                if not all(math.isfinite(v) for v in values.values()):
                    raise FloatingPointError(f"Nonfinite held-out metric for {task}")
                for key, value in values.items():
                    sums[key] += size * value
            metrics = {key: value / windows_per_task for key, value in sums.items()}
            metrics["action_rmse_symlog"] = math.sqrt(metrics.pop("action_mse_symlog"))
            metrics["repeat_previous_action_rmse_symlog"] = math.sqrt(metrics.pop("repeat_previous_action_mse_symlog"))
            tasks[task] = {"windows": windows_per_task, **metrics}
    macro = {key: float(np.mean([v[key] for v in tasks.values()])) for key in metrics}
    model.train(); model.vision_encoder.eval()
    return {"split": dataset.split, "per_task": tasks, "macro_average": macro,
            "selection_score": macro["action_rmse_symlog"] + config.training.lambda_future_feat * macro["future_cosine_distance"],
            "horizon_steps": config.common.action_chunk_size, "future_images_are_targets_only": True}


def save_checkpoint(path, value):
    import torch
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def train(config_path, output, *, resume=None, init_from=None, max_updates=None):
    """Run mixed-precision training with validation selection and resumable optimizer state."""
    import torch
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader
    if resume and init_from:
        raise ValueError("Use resume for interruption recovery or init_from for a new LR stage")
    config, factory = load_config(config_path)
    device = torch.device(config.kuafu.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.set_num_threads(4)
    torch.manual_seed(config.kuafu.seed)
    output = resolve_artifact_path(output, "LiLa-WAM training output")
    if output.exists() and not resume:
        raise ValueError(f"Refusing to overwrite training output {output}")
    validate_conditions(config)
    identity = {"manifest_sha256": file_hash(config.kuafu.manifest),
                "normalization_sha256": file_hash(config.kuafu.normalization),
                "conditions_sha256": file_hash(Path(config.dataset.task_cond_dir) / "meta.json"),
                "upstream_revision": UPSTREAM_REVISION}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "progress.json", {"phase": "initializing", "step": 0})
    datasets = {split: KinovaLiLaDataset(config.kuafu.manifest, split, config.dataset.task_cond_dir)
                for split in ("train", "validation", "test")}
    model = build_model(config, factory, device)
    optimizer = torch.optim.AdamW(model.action_model.parameters(), lr=config.training.learning_rate,
        betas=tuple(config.training.betas), weight_decay=config.training.weight_decay)
    batch_size, accumulation = int(config.training.batch_size), int(config.training.grad_accum_steps)
    if min(batch_size, accumulation, int(config.training.epochs), 1 if max_updates is None else max_updates) < 1:
        raise ValueError("Batch size, accumulation, epochs, and max_updates must be positive")
    updates_per_epoch = math.ceil(len(datasets["train"]) / (batch_size * accumulation))
    samples_per_epoch = updates_per_epoch * batch_size * accumulation
    target = updates_per_epoch * int(config.training.epochs)
    if max_updates is not None:
        target = min(target, max_updates)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
        T_max=updates_per_epoch * int(config.training.epochs), eta_min=config.training.lr_min)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and config.kuafu.precision == "fp16", init_scale=1024.)
    start_epoch, start_sample, step, best_score = 0, 0, 0, float("inf")
    if resume or init_from:
        checkpoint = torch.load(resume or init_from, map_location=device, weights_only=False)
        if checkpoint["identity"] != identity:
            raise ValueError("Checkpoint data, normalization, task conditions, or upstream revision changed")
        model.action_model.load_state_dict(checkpoint["model"], strict=True)
        if resume:
            if checkpoint["config"] != OmegaConf.to_container(config, resolve=True):
                raise ValueError("Resume requires an unchanged training configuration")
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            scaler.load_state_dict(checkpoint["scaler"])
            start_epoch, start_sample, step, best_score = [checkpoint[k] for k in ("epoch", "next_sample", "step", "best_score")]
            torch.set_rng_state(checkpoint["cpu_rng"].cpu())
            if device.type == "cuda":
                torch.cuda.set_rng_state(checkpoint["cuda_rng"].cpu(), device)
    write_json(output / "run.json", {"identity": identity, "config": OmegaConf.to_container(config, resolve=True),
        "training_examples": len(datasets["train"]), "updates_per_epoch": updates_per_epoch, "target_updates": target,
        "trainable_parameters": sum(p.numel() for p in model.action_model.parameters()),
        "frozen_parameters": sum(p.numel() for p in model.vision_encoder.parameters()),
        "action_encoding": "symlog(relative_joint)", "RGB_decoder": False})
    started = time.monotonic()
    model.train(); model.vision_encoder.eval()
    optimizer.zero_grad(set_to_none=True)
    epoch, consecutive_overflows = start_epoch, 0
    while step < target:
        offset = start_sample if epoch == start_epoch else 0
        order = balanced_indices(datasets["train"], config.kuafu.seed + epoch, samples_per_epoch)[offset:]
        workers = int(config.system.num_workers)
        loader = DataLoader(datasets["train"], batch_size=batch_size, sampler=order.tolist(), num_workers=workers,
            pin_memory=device.type == "cuda", drop_last=True,
            generator=torch.Generator().manual_seed(config.kuafu.seed + epoch),
            **({"multiprocessing_context": "spawn", "persistent_workers": True} if workers else {}))
        accumulated_loss = 0.0
        for micro, batch in enumerate(loader):
            with precision(config, device):
                loss, info = model(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite LiLa-WAM training loss")
            scaler.scale(loss / accumulation).backward()
            accumulated_loss += float(loss.detach()) / accumulation
            if (micro + 1) % accumulation:
                continue
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.action_model.parameters(), config.training.grad_clip_norm)
            old_scale = scaler.get_scale()
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            if scaler.get_scale() < old_scale:
                consecutive_overflows += 1
                if consecutive_overflows >= 20:
                    raise FloatingPointError("Twenty consecutive fp16 overflows; training needs numerical review")
                print(json.dumps({"fp16_overflow_skipped": True, "step": step}), flush=True)
                accumulated_loss = 0.0
                continue
            consecutive_overflows = 0
            scheduler.step(); step += 1
            elapsed = time.monotonic() - started
            progress = {"phase": "training", "step": step, "target": target, "epoch": epoch + 1,
                        "loss": accumulated_loss, "flow_loss": info["loss_mse"], "future_feature_loss": info["loss_future_feat"],
                        "learning_rate": scheduler.get_last_lr()[0], "elapsed_seconds": elapsed,
                        "gpu": str(device), "gradient_norm": float(norm)}
            write_json(output / "progress.json", progress)
            if step == 1 or step % 20 == 0:
                print(json.dumps(progress), flush=True)
            if step == 1 or step % config.kuafu.evaluate_every == 0 or step >= target:
                validation = evaluate(model, datasets["validation"], config, windows_per_task=config.kuafu.validation_windows)
                score = validation["selection_score"]
                if score < best_score:
                    best_score = score
                    save_checkpoint(output / "best.pt", {"model": model.action_model.state_dict(), "step": step,
                        "identity": identity, "validation": validation, "config": OmegaConf.to_container(config, resolve=True)})
                    write_json(output / "best.json", {"step": step, "validation": validation})
                progress["validation"] = validation["macro_average"]
                with (output / "metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(progress) + "\n")
                next_sample = offset + (micro + 1) * batch_size
                next_epoch = epoch
                if next_sample == samples_per_epoch:
                    next_epoch, next_sample = epoch + 1, 0
                save_checkpoint(output / "last.pt", {"model": model.action_model.state_dict(), "identity": identity,
                    "config": OmegaConf.to_container(config, resolve=True), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "step": step,
                    "epoch": next_epoch, "next_sample": next_sample, "best_score": best_score,
                    "cpu_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None})
            accumulated_loss = 0.0
            if step >= target:
                break
        epoch += 1
    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.action_model.load_state_dict(checkpoint["model"])
    test = evaluate(model, datasets["test"], config, windows_per_task=config.kuafu.test_windows)
    write_json(output / "evaluation.json", {"steps_completed": step, "best_step": checkpoint["step"], "test": test,
                                          "validation": checkpoint["validation"]})
    write_json(output / "progress.json", {"phase": "complete", "step": step, "best_step": checkpoint["step"]})
    for dataset in datasets.values():
        dataset.close()
    return test
