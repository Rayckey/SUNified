#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import yaml
import torch

SOURCE_ROOT = Path(__file__).resolve().parents[3]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from kuafu.paths import resolve_artifact_path
from kuafu.bc.training.dataset import Hdf5TrajectoryDataset, Normalizer
from kuafu.bc.training.model import MLPActorCriticPrior, MLPActorPrior


def _apply_history_and_action_chunking_to_dataset(
    ds: Hdf5TrajectoryDataset,
    obs_history: int,
    action_chunk_size: int,
) -> None:
    """Match the history/chunking preprocessing used during training."""
    if len(ds) == 0:
        return

    if obs_history > 1:
        obs = ds.obs.float()
        N, obs_dim = obs.shape
        H = int(obs_history)
        first = obs[0:1].repeat(H - 1, 1)
        padded = torch.cat([first, obs], dim=0)
        chunks = []
        for t in range(N):
            window = padded[t : t + H].reshape(-1)
            chunks.append(window)
        obs_hist = torch.stack(chunks, dim=0)
        ds.obs = obs_hist
        ds.obs_dim = obs_hist.shape[1]

    if action_chunk_size > 1:
        acts = ds.actions.float()
        N, act_dim = acts.shape
        C = int(action_chunk_size)
        last = acts[-1:].repeat(C - 1, 1)
        padded = torch.cat([acts, last], dim=0)
        chunks = []
        for t in range(N):
            window = padded[t : t + C].reshape(-1)
            chunks.append(window)
        acts_chunked = torch.stack(chunks, dim=0)
        ds.actions = acts_chunked
        ds.act_dim = acts_chunked.shape[1]


@torch.no_grad()
def eval_shift(
    ckpt: dict,
    dataset_path: str,
    demo: str | None,
    obs_key: str,
    action_mode: str | None,
    time_shift: int,
    num_samples: int,
    seed: int,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Some checkpoints may store the literal string "None" for action_mode.
    if isinstance(action_mode, str) and action_mode.lower() == "none":
        action_mode = None

    ds = Hdf5TrajectoryDataset(
        dataset_path=dataset_path,
        demo=demo,
        obs_key=obs_key,
        action_subkey=action_mode,
        time_shift=time_shift,  # type: ignore[arg-type]
        split="train",
        val_fraction=0.0,
        index_mode=ckpt.get("split_mode", "contiguous"),
        seed=seed,
    )

    obs_history = int(ckpt.get("obs_history", 1))
    action_chunk_size = int(ckpt.get("action_chunk_size", 1))
    if obs_history > 1 or action_chunk_size > 1:
        _apply_history_and_action_chunking_to_dataset(ds, obs_history, action_chunk_size)

    obs_norm = Normalizer.from_state_dict(ckpt["obs_normalizer"])
    act_norm = Normalizer.from_state_dict(ckpt["act_normalizer"])

    model_state_dict = ckpt["model_state_dicts"]["model_state_dict_norm"]
    uses_actor_critic_layout = any(
        key.startswith("actor_trunk.") or key.startswith("critic_trunk.") or key.startswith("value_layer.")
        for key in model_state_dict
    )
    model_cls = MLPActorCriticPrior if uses_actor_critic_layout else MLPActorPrior

    model = model_cls(
        obs_dim=ckpt["obs_dim"],
        act_dim=ckpt["act_dim"],
        hidden_dims=tuple(ckpt["hidden_dims"]),
        activation=str(ckpt.get("activation", "leaky_relu")),
        negative_slope=float(ckpt.get("negative_slope", 0.01)),
        dropout=float(ckpt.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(model_state_dict)
    model.eval()

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), size=min(num_samples, len(ds)), replace=False)
    obs = ds.obs[idx].to(device)
    act = ds.actions[idx].to(device)

    pred_n = model(obs_norm.normalize(obs))
    pred = act_norm.denormalize(pred_n)

    mse = float(((pred - act) ** 2).mean().item())
    l2 = torch.linalg.norm(pred - act, dim=-1)
    mean_l2 = float(l2.mean().item())
    p90_l2 = float(torch.quantile(l2, 0.9).item())

    pred_np = pred.detach().cpu().numpy()
    act_np = act.detach().cpu().numpy()
    return {
        "time_shift": int(time_shift),
        "N": int(len(idx)),
        "mse": float(mse),
        "mean_l2": float(mean_l2),
        "p90_l2": float(p90_l2),
        "pred_min": float(pred_np.min()),
        "pred_max": float(pred_np.max()),
        "gt_min": float(act_np.min()),
        "gt_max": float(act_np.max()),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=str(resolve_artifact_path("logs/bc_prior_mlp/2026-03-13_18-15-22/checkpoints/best.pth", "BC checkpoint")))
    p.add_argument(
        "--dataset",
        type=str,
        default=str(resolve_artifact_path("Trajectories/fr3_finray_table_pen_cup_random_pen_cup.hdf5", "BC evaluation dataset")),
        help="If omitted, use the training dataset from meta/config.yaml",
    )
    p.add_argument("--demo", type=str, default=None)
    p.add_argument(
        "--demo_index",
        type=int,
        default=None,
        help="If set, use this 0-based demo index from the HDF5 dataset (overrides --demo).",
    )
    p.add_argument(
        "--obs_key",
        type=str,
        default=None,
        choices=["state", "full_state"],
        help="If omitted, use the obs_key from meta/config.yaml",
    )
    p.add_argument("--num_samples", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval_all_shifts", action="store_true", default=False)
    p.add_argument("--enable_cameras", action="store_true", default=False)
    args = p.parse_args()
    args.checkpoint = str(resolve_artifact_path(args.checkpoint, "BC checkpoint"))
    if args.dataset:
        args.dataset = str(resolve_artifact_path(args.dataset, "BC evaluation trajectory"))

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(args.checkpoint)

    ckpt = torch.load(args.checkpoint, map_location="cpu")

    # Load the original run configuration that produced this checkpoint.
    ckpt_path = Path(args.checkpoint)
    run_dir = ckpt_path.parent.parent
    config_path = run_dir / "meta" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml not found next to checkpoint: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    dataset_cfg = config.get("dataset", {})

    dataset_path = args.dataset or dataset_cfg.get("dataset")
    if dataset_path:
        dataset_path = str(resolve_artifact_path(dataset_path, "BC evaluation trajectory"))
    if not dataset_path:
        raise ValueError("Dataset path not in meta/config.yaml and --dataset not provided.")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(dataset_path)

    # If demo_index is provided, map it to a demo name from the HDF5 file.
    demo = args.demo
    if args.demo_index is not None:
        import h5py as _h5py

        with _h5py.File(dataset_path, "r") as f:
            raw_keys = list(f["data"].keys())  # type: ignore[union-attr]
            def _demo_sort_key(name: str):
                s = str(name)
                if "_" in s:
                    try:
                        return int(s.rsplit("_", 1)[1])
                    except ValueError:
                        pass
                return s
            demos = sorted(raw_keys, key=_demo_sort_key)
            if not demos:
                raise ValueError("Dataset contains no demos but --demo_index was provided.")
            idx = int(args.demo_index)
            if idx < 0:
                idx += len(demos)
            if idx < 0 or idx >= len(demos):
                raise IndexError(f"--demo_index {args.demo_index} out of range for {len(demos)} demos")
            demo = demos[idx]
            print(f"Using only demo index {idx} -> '{demo}'")

    # Use obs_key exactly as in training unless explicitly overridden.
    obs_key = args.obs_key or dataset_cfg["obs_key"]
    print(f"Checkpoint: {args.checkpoint}")
    print(
        f"  epoch={ckpt.get('epoch')} train_loss={ckpt.get('train_loss'):.6f} val_loss={ckpt.get('val_loss'):.6f} "
        f"obs_dim={ckpt['obs_dim']} act_dim={ckpt['act_dim']} time_shift={ckpt.get('time_shift')}"
    )
    print(f"Dataset: {dataset_path}  obs_key={obs_key} demo={demo}")

    # Use the time_shift chosen during training by default.
    shifts = [int(dataset_cfg.get("time_shift", ckpt.get("time_shift", 0)))]
    if args.eval_all_shifts:
        shifts = [0, 1, -1, 2, -2]

    for s in shifts:
        res = eval_shift(
            ckpt=ckpt,
            dataset_path=dataset_path,
            demo=demo,
            obs_key=obs_key,
            action_mode=dataset_cfg.get("action_mode"),
            time_shift=s,
            num_samples=args.num_samples,
            seed=args.seed,
        )
        print(
            f"shift={res['time_shift']:+d} N={res['N']} mse={res['mse']:.6g} "
            f"mean_l2={res['mean_l2']:.6g} p90_l2={res['p90_l2']:.6g} "
            f"pred_range=[{res['pred_min']:.3g},{res['pred_max']:.3g}] "
            f"gt_range=[{res['gt_min']:.3g},{res['gt_max']:.3g}]"
        )


if __name__ == "__main__":
    main()

