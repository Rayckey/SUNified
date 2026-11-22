"""Train a PETS-style dynamics ensemble without launching Isaac Sim."""

from __future__ import annotations

import argparse

from kuafu.dynamics.train import TrainConfig, run_training


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Kuafu HDF5 with data/demo_*/obs, actions, next_obs")
    parser.add_argument("--output", required=True, help="New directory, relative to KUAFU_ARTIFACT_ROOT or absolute")
    parser.add_argument("--task", default="", help="Task identity stored with the checkpoint")
    parser.add_argument("--obs-key", default="full_state")
    parser.add_argument("--action-key", default="relative_joint", help="Subkey under actions; empty for flat actions")
    parser.add_argument("--exclude-last", type=int, default=1, help="Trailing observation columns to omit (1: SUN stage)")
    parser.add_argument("--skip-first", type=int, default=1, help="Initial rows to omit per demonstration (reset artifacts)")
    for name in ("members", "hidden_dim", "hidden_layers", "epochs", "batch_size", "patience", "seed",
                 "threads", "rollout_horizon", "rollout_windows", "particles"):
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=TrainConfig.__dataclass_fields__[name].default)
    for name in ("learning_rate", "weight_decay", "val_fraction", "test_fraction"):
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=TrainConfig.__dataclass_fields__[name].default)
    parser.add_argument("--device", default="cpu", help="cpu or an explicit CUDA device such as cuda:0")
    args = parser.parse_args(argv)
    try:
        run_training(TrainConfig(**vars(args)))
    except (ValueError, OSError) as error:
        parser.exit(2, f"ERROR: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
