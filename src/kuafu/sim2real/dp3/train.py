"""Matched DP3 training with validation action-loss checkpoint selection."""

import copy
import json
import math
import os
from pathlib import Path
import random

import hydra
import numpy as np
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader
from diffusion_policy_3d.model.diffusion.ema_model import EMAModel

from kuafu.paths import resolve_artifact_path

from .artifacts import file_digest, json_digest, write_json
from .dataset import StageDataset
from .metrics import add_confusion, stage_metrics


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_device(batch, device):
    return {key: to_device(value, device) if isinstance(value, dict) else value.to(device)
            for key, value in batch.items()}


def load_checkpoint(path, device="cpu"):
    # Checkpoints are executable pickle artifacts; load only trusted experiment outputs.
    import dill
    return torch.load(Path(path), map_location=device, pickle_module=dill, weights_only=False)


def load_policy(path, device="cpu"):
    payload = load_checkpoint(path)
    model = hydra.utils.instantiate(payload["cfg"].policy)
    states = payload["state_dicts"]
    model.load_state_dict(states.get("ema_model", states["model"]))
    return model.to(device).eval(), payload


def save_checkpoint(path, *, model, ema_model, optimizer, scheduler, policy_config, experiment, epoch, best):
    import dill
    payload = {
        "cfg": OmegaConf.create({"policy": policy_config, "task": {"shape_meta": policy_config["shape_meta"]}}),
        "state_dicts": {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict()},
        "pickles": {"epoch": dill.dumps(epoch)},
        "experiment": {**experiment, "epoch": epoch, "best_validation_action_loss": best},
    }
    if ema_model is not None:
        payload["state_dicts"]["ema_model"] = ema_model.state_dict()
    temporary = Path(path).with_suffix(".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


@torch.no_grad()
def offline_metrics(model, dataset, *, batch_size, device, seed):
    """Evaluate one declared partition; action loss uses fixed diffusion noise."""
    model.eval()
    confusion = np.zeros((dataset.num_stages, dataset.num_stages), dtype=np.int64)
    action_total, stage_total, count = 0.0, 0.0, 0
    devices = [torch.device(device).index or 0] if str(device).startswith("cuda") else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            batch = to_device(batch, device)
            action, stage = model.loss_terms(batch)
            size = len(batch["stage_id"])
            action_total += action.item() * size
            stage_total += stage.item() * size
            count += size
            logits = model.stage_logits(batch["obs"])
            if logits is not None:
                add_confusion(confusion, batch["stage_id"].cpu().numpy(), logits.argmax(-1).cpu().numpy())
    return {"setting": f"offline_{dataset.partition}", "samples": count,
            "action_loss": action_total / count, "stage_loss": stage_total / count,
            "stage": stage_metrics(confusion) if model.variant != "A" else None}


def resolve_experiment(config_path, dataset_path, *, provenance=None):
    cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    baseline_path = Path(cfg["baseline_config"]).expanduser().resolve()
    base = OmegaConf.load(baseline_path)
    policy = OmegaConf.to_container(base.policy, resolve=True)
    backbones = {
        "diffusion_policy_3d.policy.dp3.DP3": "kuafu.sim2real.dp3.policy.StageDP3",
        "diffusion_policy_3d.policy.simple_dp3.SimpleDP3": "kuafu.sim2real.dp3.policy.StageSimpleDP3",
    }
    baseline_target = policy.pop("_target_")
    if baseline_target not in backbones:
        raise ValueError("Use a Kuafu2Real DP3 or SimpleDP3 baseline configuration")
    policy.update(cfg.get("policy_overrides", {}))
    if "_target_" in policy or any(key.startswith("stage_") or key in {"variant", "num_stages"} for key in policy):
        raise ValueError("Use the shared stage settings instead of overriding experiment variants")
    manifest = json.loads((Path(dataset_path) / "stage_dataset.json").read_text())
    exported = manifest["export"]
    policy["shape_meta"] = {
        "obs": {"point_cloud": {"shape": [exported["num_points"], 3], "type": "point_cloud"},
                "agent_pos": {"shape": [exported["agent_pos_dim"]], "type": "low_dim"}},
        "action": {"shape": [exported["action_dim"]]},
    }
    if policy.get("use_pc_color", False):
        raise ValueError("The first experiment uses the Kuafu2Real XYZ baseline")
    policy["_target_"] = backbones[baseline_target]
    if set(cfg.get("stage", {})) - {"stage_hidden_dim", "stage_embedding_dim", "stage_loss_weight"}:
        raise ValueError("Stage settings accept only head size, embedding size, and CE weight")
    policy.update(cfg.get("stage", {}), num_stages=len(manifest["stage_names"]))
    if "variant" in policy:
        raise ValueError("Variants are selected by the experiment runner")
    optimizer = OmegaConf.to_container(base.optimizer, resolve=True)
    optimizer.update(cfg.get("optimizer_overrides", {}))
    if optimizer.pop("_target_") != "torch.optim.AdamW":
        raise ValueError("This runner uses the baseline AdamW optimizer")
    resolved = {
        "dataset_id": manifest["dataset_id"], "policy": policy, "optimizer": optimizer,
        "window": {"horizon": policy["horizon"], "n_obs_steps": policy["n_obs_steps"],
                   "pad_before": policy["n_obs_steps"] - 1, "pad_after": policy["horizon"] - policy["n_obs_steps"],
                   "action_stride": int(cfg.get("action_stride", 1))},
        "precision": "float32", "lr_scheduler": "cosine",
        "epochs": int(cfg.get("epochs", base.training.num_epochs)),
        "batch_size": int(cfg.get("batch_size", base.dataloader.batch_size)),
        "num_workers": int(cfg.get("num_workers", 0)), "seeds": cfg.get("seeds", [42, 43, 44]),
        "device": str(cfg.get("device", base.training.device)),
        "warmup_steps": int(cfg.get("warmup_steps", base.training.lr_warmup_steps)),
        "ema": OmegaConf.to_container(base.ema, resolve=True) if base.training.use_ema else None,
        "selection": "minimum validation action loss, EMA when enabled; earliest epoch wins ties",
        "baseline_config": {"path": str(baseline_path), "sha256": file_digest(baseline_path)},
        "dependencies": provenance or {}, "hypothesis": cfg.get("hypothesis", ""),
        "failure_evidence": cfg.get("failure_evidence", []),
        "warm_start": cfg.get("warm_start"),
    }
    if min(resolved["epochs"], resolved["batch_size"]) < 1 or resolved["num_workers"] < 0:
        raise ValueError("Training epochs/batch size must be positive and workers nonnegative")
    if not resolved["seeds"] or len(set(resolved["seeds"])) != len(resolved["seeds"]):
        raise ValueError("Supply distinct training seeds")
    if resolved["warm_start"]:
        resolved["warm_start_sha256"] = file_digest(resolved["warm_start"])
    return resolved


def run_training(config_path, dataset_path, output, *, variants=("A", "B", "C"), resume=False, provenance=None):
    resolved = resolve_experiment(config_path, dataset_path, provenance=provenance)
    if not variants or len(set(variants)) != len(variants) or set(variants) - set("ABC"):
        raise ValueError("Select distinct variants from A, B, C")
    output = resolve_artifact_path(output, "Stage experiment output")
    comparison_id = json_digest(resolved)
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "experiment.json"
    if summary_path.exists():
        previous = json.loads(summary_path.read_text())
        if not resume or previous["comparison_id"] != comparison_id:
            raise ValueError("Use a new output directory or resume the identical experiment")
    elif any(output.iterdir()):
        raise FileExistsError("Experiment output is not empty")
    dataset = StageDataset(dataset_path, **resolved["window"], verify_content=True)
    validation = dataset.get_validation_dataset()
    normalizer = dataset.get_normalizer()
    write_json(summary_path, {"comparison_id": comparison_id, "config": resolved,
                             "class_counts": {p: dataset.for_partition(p).class_counts()
                                              for p in ("train", "validation", "test")}})
    device = resolved["device"]
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    for seed in resolved["seeds"]:
        for variant in variants:
            directory = output / f"seed_{seed}" / variant
            directory.mkdir(parents=True, exist_ok=True)
            seed_all(seed)
            policy_cfg = {**resolved["policy"], "variant": variant}
            model = hydra.utils.instantiate(OmegaConf.create(policy_cfg))
            model.set_normalizer(normalizer)
            if resolved["warm_start"]:
                source = load_checkpoint(resolved["warm_start"])
                state = source["state_dicts"].get("ema_model", source["state_dicts"]["model"])
                for key, value in normalizer.state_dict().items():
                    if "normalizer." + key not in state or not torch.equal(state["normalizer." + key], value):
                        raise ValueError("Warm-start normalizer must match this dataset's training partition")
                model.load_baseline(state)
            model.to(device)
            ema_model = copy.deepcopy(model).eval() if resolved["ema"] else None
            ema_config = dict(resolved["ema"] or {})
            ema_config.pop("_target_", None)
            ema = EMAModel(ema_model, **ema_config) if ema_model is not None else None
            optimizer_config = dict(resolved["optimizer"])
            if not str(device).startswith("cuda"):
                optimizer_config["fused"] = False
            optimizer = torch.optim.AdamW(model.parameters(), **optimizer_config)
            steps_per_epoch = math.ceil(len(dataset) / resolved["batch_size"])
            total_steps = steps_per_epoch * resolved["epochs"]
            warmup = resolved["warmup_steps"]
            def schedule(step):
                if step < warmup:
                    return (step + 1) / max(1, warmup)
                progress = (step - warmup) / max(1, total_steps - warmup)
                return 0.5 * (1 + math.cos(math.pi * min(progress, 1)))
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
            start, best = 0, float("inf")
            latest = directory / "latest.ckpt"
            if resume and latest.exists():
                saved = load_checkpoint(latest)
                if saved["experiment"]["comparison_id"] != comparison_id:
                    raise ValueError("Checkpoint belongs to a different experiment")
                for name, item in (("model", model), ("optimizer", optimizer), ("scheduler", scheduler)):
                    item.load_state_dict(saved["state_dicts"][name])
                if ema_model is not None:
                    ema_model.load_state_dict(saved["state_dicts"]["ema_model"])
                start = saved["experiment"]["epoch"] + 1
                best = saved["experiment"]["best_validation_action_loss"]
                if ema is not None:
                    ema.optimization_step = start * steps_per_epoch
            experiment = {"comparison_id": comparison_id, "dataset_id": dataset.manifest["dataset_id"],
                          "variant": variant, "seed": seed, "config": resolved,
                          "dataset_manifest": dataset.manifest,
                          "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}
            for epoch in range(start, resolved["epochs"]):
                # Re-seeding per epoch makes resumes exact and keeps A/B/C noise/order paired.
                seed_all(seed + epoch * 1009)
                generator = torch.Generator().manual_seed(seed + epoch)
                loader = DataLoader(dataset, batch_size=resolved["batch_size"], shuffle=True,
                                    generator=generator, num_workers=resolved["num_workers"], drop_last=False)
                model.train()
                loss_sum, count = 0.0, 0
                for batch in loader:
                    batch = to_device(batch, device)
                    optimizer.zero_grad(set_to_none=True)
                    loss, _ = model.compute_loss(batch)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Nonfinite training loss: {directory}, epoch {epoch}")
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    if ema is not None:
                        ema.step(model)
                    loss_sum += loss.item() * len(batch["stage_id"])
                    count += len(batch["stage_id"])
                metrics = offline_metrics(ema_model or model, validation, batch_size=resolved["batch_size"],
                                          device=device, seed=seed + 917)
                if not math.isfinite(metrics["action_loss"]):
                    raise FloatingPointError("Nonfinite validation action loss")
                improved = metrics["action_loss"] < best
                best = min(best, metrics["action_loss"])
                row = {"epoch": epoch, "training_loss": loss_sum / count, "validation": metrics}
                write_json(directory / f"metrics_{epoch:04d}.json", row)
                options = dict(model=model, ema_model=ema_model, optimizer=optimizer, scheduler=scheduler,
                               policy_config=policy_cfg, experiment=experiment, epoch=epoch, best=best)
                if improved:
                    save_checkpoint(directory / "best.ckpt", **options)
                save_checkpoint(latest, **options)
                print(json.dumps({"variant": variant, "seed": seed, **row}), flush=True)
    return {"comparison_id": comparison_id, "output": str(output)}
