"""Prepare, validate, and train LiLa-WAM using audited Kinova demonstrations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time


def download_encoder(output, token_file=None):
    """Download a pinned encoder through normal Hugging Face authorization."""
    import re
    from huggingface_hub import HfApi, snapshot_download
    from kuafu.dynamics.lilawam_data import file_hash, write_json
    from kuafu.paths import resolve_artifact_path
    token = None
    if token_file:
        matches = re.findall(r"\bhf_[A-Za-z0-9]+\b", Path(token_file).read_text())
        if len(matches) != 1:
            raise ValueError("Expected one Hugging Face token in the supplied file")
        token = matches[0]
    repo = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    revision = "ea8dc2863c51be0a264bab82070e3e8836b02d51"
    output = resolve_artifact_path(output, "DINOv3 checkpoint")
    HfApi(token=token).model_info(repo, revision=revision)
    snapshot_download(repo_id=repo, revision=revision, token=token, local_dir=output,
        allow_patterns=["config.json", "model.safetensors", "preprocessor_config.json", "README.md", "LICENSE*"], max_workers=3)
    report = {"repository": repo, "revision": revision, "path": str(output),
              "weights_sha256": file_hash(output / "model.safetensors")}
    write_json(output / "kuafu_provenance.json", report)
    return report


def smoke(config_path, output, batch_size=8):
    """Exercise the complete model architecture with random encoder weights, never a training fallback."""
    import torch
    import numpy as np
    from transformers import DINOv3ViTConfig, DINOv3ViTModel
    from torch.utils.data import default_collate
    from kuafu.dynamics.lilawam_data import KinovaLiLaDataset, write_json
    from kuafu.dynamics.lilawam_runtime import load_config, precision, evaluate
    from kuafu.paths import resolve_artifact_path
    config, factory = load_config(config_path)
    from models.model_runner import VLAWrapper
    device = torch.device(config.kuafu.device)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    dataset = KinovaLiLaDataset(config.kuafu.manifest, "train")
    indices = [dataset.task_ranges[dataset.tasks[i % len(dataset.tasks)]][0] for i in range(batch_size)]
    batch = default_collate([dataset[i] for i in indices])
    batch["task_cond"] = torch.zeros(batch_size, 1024)
    vision = DINOv3ViTModel(DINOv3ViTConfig(hidden_size=1024, num_hidden_layers=24,
        num_attention_heads=16, intermediate_size=4096, patch_size=16, num_register_tokens=4))
    vision.eval()
    for parameter in vision.parameters():
        parameter.requires_grad = False
    action = factory.create_action_model(config, 1024, 3, 1024, 16)
    model = VLAWrapper(vision, action, "logit_normal", [-12, -8, -4], True, 4, device, torch.float32,
        config.kuafu.normalization, {"time_mu": 0., "time_sigma": 1., "use_vel_weight": False,
            "vel_weight_alpha": .2, "vel_weight_sigma": .01, "use_future_feat": True, "lambda_future_feat": .5}, -4).to(device)
    optimizer = torch.optim.AdamW(action.parameters(), lr=2e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and config.kuafu.precision == "fp16",
                                  init_scale=1024.)
    records = []
    for step in range(3):
        started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        with precision(config, device):
            loss, info = model(batch)
        assert torch.isfinite(loss)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(action.parameters(), 1.)
        assert torch.isfinite(norm) and float(norm) > 0
        observed = next(p for p in action.parameters()
                        if p.grad is not None and torch.count_nonzero(p.grad).item() > 0)
        before = observed.detach().clone()
        scaler.step(optimizer); scaler.update()
        assert not torch.equal(before, observed.detach()), "Optimizer did not change a parameter"
        assert all(p.grad is None for p in vision.parameters())
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        records.append({"step": step + 1, "loss": float(loss.detach()), "flow_loss": info["loss_mse"],
                        "future_feature_loss": info["loss_future_feat"], "gradient_norm": float(norm),
                        "seconds": time.monotonic() - started})
        print(json.dumps({"architecture_smoke": records[-1]}), flush=True)
        del before
    # Exercise held-out action inference and feature scoring on actual recordings.
    validation = KinovaLiLaDataset(config.kuafu.manifest, "validation")
    validation.conditions = {task: np.zeros(1024, np.float32) for task in validation.tasks}
    checked = evaluate(model, validation, config, windows_per_task=2)
    report = {"passed": True, "scope": "architecture and data plumbing only; random frozen visual encoder",
        "pretrained_encoder_loaded": False, "production_training_started": False,
        "batch_size": batch_size, "device": str(device), "steps": records,
        "trainable_parameters": sum(p.numel() for p in action.parameters()),
        "frozen_parameters": sum(p.numel() for p in vision.parameters()),
        "validation_inference_tasks_checked": list(checked["per_task"]),
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None}
    write_json(resolve_artifact_path(output, "LiLa-WAM smoke report"), report)
    dataset.close(); validation.close()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--rgb-index", required=True)
    prepare.add_argument("--split-run", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--horizon", type=int, default=16)
    download = commands.add_parser("download-encoder")
    download.add_argument("--output", required=True)
    download.add_argument("--token-file")
    conditions = commands.add_parser("conditions")
    conditions.add_argument("--config", required=True)
    trainer = commands.add_parser("train")
    trainer.add_argument("--config", required=True)
    trainer.add_argument("--output", required=True)
    trainer.add_argument("--resume")
    trainer.add_argument("--init-from")
    trainer.add_argument("--max-updates", type=int)
    check = commands.add_parser("smoke")
    check.add_argument("--config", required=True)
    check.add_argument("--output", required=True)
    check.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            from kuafu.dynamics.lilawam_data import prepare_data
            result = prepare_data(args.rgb_index, args.split_run, args.output, horizon=args.horizon)
            result = {"tasks": len(result["tasks"]), "output": args.output}
        elif args.command == "download-encoder":
            result = download_encoder(args.output, args.token_file)
        elif args.command == "conditions":
            from kuafu.dynamics.lilawam_runtime import prepare_conditions
            result = prepare_conditions(args.config)
        elif args.command == "train":
            from kuafu.dynamics.lilawam_runtime import train
            result = train(args.config, args.output, resume=args.resume, init_from=args.init_from, max_updates=args.max_updates)
        else:
            result = smoke(args.config, args.output, args.batch_size)
        print(json.dumps(result), flush=True)
    except Exception as error:
        # Never echo credential-bearing request headers or URLs from HF exceptions.
        if args.command == "download-encoder":
            status = getattr(getattr(error, "response", None), "status_code", None)
            print(json.dumps({"error_type": type(error).__name__, "http_status": status,
                              "message": "Encoder download failed; verify account access to the gated DINOv3 repository"}), flush=True)
            return 2
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
