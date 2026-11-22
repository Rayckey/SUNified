#!/usr/bin/env python3
"""Generate BC prior training configs and a sweep CSV for a collected dataset."""

from __future__ import annotations

import argparse
import csv
import os
import re
from dataclasses import dataclass
from pathlib import Path

from kuafu.paths import resolve_artifact_path


REPO_ROOT = Path(__file__).resolve().parents[4]

ROBOT_ARTIFACT_DIRS = {
    "fr3_finray": "fr3_finray",
    "franka": "fr3_finray",
    "panda": "fr3_finray",
    "fr3_finray": "fr3_finray",
    "kinova": "kinova_gen3",
    "kinova_gen3": "kinova_gen3",
    "panda": "franka_panda",
    "franka_panda": "franka_panda",
}

TASK_SLUG_ALIASES = {
    "can": "right_can",
    "stack_2": "stack2",
    "stack_two": "stack2",
    "stack_3": "stack3",
    "stack_three": "stack3",
}

VARIANT_HIDDEN_DIMS = {
    "bc": [256, 512, 256],
    "f": [1024, 512, 256, 512, 1024],
    "ff": [2048, 1024, 512, 1024, 2048],
    "fff": [4096, 2048, 1024, 2048, 4096],
    "f3": [1024, 512, 1024],
    "ff3": [2048, 1024, 2048],
    "fff3": [4096, 2048, 4096],
    "class": [1024, 512, 256, 512, 256],
}

DEFAULT_VARIANTS = ("f", "ff", "fff", "f3", "ff3", "fff3")
DEFAULT_CHUNK_SIZES = (10, 15, 30)
DEFAULT_CHUNK_STRIDES = (1, 3)
VARIANT_ALIASES = {
    "c": "class",
}


@dataclass(frozen=True)
class BcPriorPrepConfig:
    dataset_path: Path
    task_name: str
    robot_name: str
    run_name: str
    artifact_root: Path | None = None
    task_slug: str | None = None
    name_prefix: str | None = None
    chunk_sizes: tuple[int, ...] = DEFAULT_CHUNK_SIZES
    action_chunk_strides: tuple[int, ...] = DEFAULT_CHUNK_STRIDES
    variants: tuple[str, ...] = DEFAULT_VARIANTS
    obs_key: str = "state"
    action_mode: str = "absolute_ee"
    time_shift: int = 0
    val_fraction: float = 0.1
    split_mode: str = "random"
    seed: int = 1999
    obs_history: int = 1
    lr: float = 1.0e-5
    weight_decay: float = 1.0e-5
    batch_size: int = 1024
    num_workers: int = 0
    pin_memory: bool = False
    num_epochs: int = 300
    save_every_n_epochs: int = 50
    grad_clip: float = 10.0
    dropout: float = 0.1
    sample_weighting_enabled: bool = True
    sample_weighting_validation: bool = True
    overwrite: bool = False


@dataclass(frozen=True)
class GeneratedBcPrior:
    name: str
    variant: str
    chunk_size: int
    action_chunk_stride: int
    config_path: Path
    prior_path: str


@dataclass(frozen=True)
class BcPriorPrepResult:
    task_slug: str
    config_dir: Path
    csv_path: Path
    generated: tuple[GeneratedBcPrior, ...]


def _safe_token(value: str, *, lowercase: bool = True) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-")
    if lowercase:
        text = text.lower()
    return text or "run"


def _relative_to_repo(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def infer_task_slug(task_name: str) -> str:
    text = task_name.strip()
    for prefix in (
        "FR3-FinRay-Table-",
        "Panda-Table-",
        "Kinova-Table-Gen3-",
        "Kinova-Table-",
    ):
        if text.startswith(prefix):
            text = text[len(prefix) :]
            break
    text = re.sub(r"-(f|ff|fff|f3|ff3|fff3)$", "", text, flags=re.IGNORECASE)
    slug = _safe_token(text.replace("-", "_"))
    return TASK_SLUG_ALIASES.get(slug, slug)


def default_artifact_root(robot_name: str) -> Path:
    runtime_root = os.environ.get("KUAFU_PRIOR_ARTIFACT_ROOT", "").strip()
    if runtime_root:
        return Path(runtime_root).expanduser().resolve()

    task_root = os.environ.get("KUAFU_TASK_ARTIFACT_ROOT", "").strip()
    if task_root:
        return Path(task_root).expanduser().resolve() / "prior_artifacts"

    artifact_dir = ROBOT_ARTIFACT_DIRS.get(robot_name, "fr3_finray")
    configured_root = os.environ.get("KUAFU_ARTIFACT_ROOT", "").strip()
    base = (
        Path(configured_root).expanduser().resolve()
        if configured_root
        else REPO_ROOT.parent / "KuafuArtifacts"
    )
    return base / "standalone" / artifact_dir / "prior_artifacts"


def _yaml_scalar(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_yaml_scalar(item) for item in value) + "]"
    text = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"'


def _yaml_lines(data: dict[str, object], *, indent: int = 0) -> list[str]:
    lines: list[str] = []
    prefix = " " * indent
    for key, value in data.items():
        if isinstance(value, dict):
            if value:
                lines.append(f"{prefix}{key}:")
                lines.extend(_yaml_lines(value, indent=indent + 2))
            else:
                lines.append(f"{prefix}{key}: {{}}")
        else:
            lines.append(f"{prefix}{key}: {_yaml_scalar(value)}")
    return lines


def _write_yaml(path: Path, data: dict[str, object], *, header: str, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {header}", * _yaml_lines(data)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _normalize_variant(raw_variant: str) -> str:
    variant = raw_variant.strip().lower()
    return VARIANT_ALIASES.get(variant, variant)


def _known_variants_text() -> str:
    return ", ".join(sorted([*VARIANT_HIDDEN_DIMS, *VARIANT_ALIASES]))


def _config_dict(
    cfg: BcPriorPrepConfig,
    *,
    dataset_path: str,
    task_slug: str,
    run_name: str,
    artifact_root: Path,
    chunk_size: int,
    action_chunk_stride: int,
    hidden_dims: list[int],
) -> dict[str, object]:
    return {
        "dataset": {
            "dataset": dataset_path,
            "demo": None,
            "obs_key": cfg.obs_key,
            "action_mode": cfg.action_mode,
            "time_shift": cfg.time_shift,
            "val_fraction": cfg.val_fraction,
            "split_mode": cfg.split_mode,
            "seed": cfg.seed,
        },
        "preprocessing": {
            "action_chunk_size": chunk_size,
            "action_chunk_stride": action_chunk_stride,
            "obs_history": cfg.obs_history,
        },
        "model": {
            "hidden_dims": hidden_dims,
            "activation": "relu",
            "dropout": cfg.dropout,
        },
        "optimization": {
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
            "batch_size": cfg.batch_size,
            "num_workers": cfg.num_workers,
            "pin_memory": cfg.pin_memory,
            "num_epochs": cfg.num_epochs,
            "save_every_n_epochs": cfg.save_every_n_epochs,
            "grad_clip": cfg.grad_clip,
        },
        "logging": {
            "log_dir": str(
                artifact_root
                / "training_runs"
                / cfg.robot_name
                / task_slug
                / run_name
            ),
            "monitor_demo_index": None,
        },
        "auto_shift": {
            "epochs": 50,
            "batch_size": 512,
        },
        "sample_weighting": {
            "enabled": cfg.sample_weighting_enabled,
            "obs_stage_index": -1,
            "default_weight": 1.0,
            "normalize": True,
            "weight_validation": cfg.sample_weighting_validation,
            "stage_weights": {},
        },
        "critic_pretraining": {
            "enabled": False,
            "gamma": 0.99,
            "loss_coef": 1.0,
            "normalize_targets": True,
        },
    }


def prepare_bc_prior_sweep(cfg: BcPriorPrepConfig) -> BcPriorPrepResult:
    if not cfg.dataset_path.exists():
        raise FileNotFoundError(f"Collected dataset does not exist: {cfg.dataset_path}")
    if not cfg.chunk_sizes:
        raise ValueError("At least one BC chunk size is required.")
    if not cfg.action_chunk_strides:
        raise ValueError("At least one BC action chunk stride is required.")
    if not cfg.variants:
        raise ValueError("At least one BC variant is required.")

    task_slug = _safe_token(cfg.task_slug) if cfg.task_slug else infer_task_slug(cfg.task_name)
    run_name = _safe_token(cfg.run_name)
    name_prefix = _safe_token(cfg.name_prefix) if cfg.name_prefix else f"{task_slug}_{run_name}"
    artifact_root = (cfg.artifact_root or default_artifact_root(cfg.robot_name)).expanduser().resolve()
    if artifact_root == REPO_ROOT or REPO_ROOT in artifact_root.parents:
        raise ValueError(
            f"BC artifact root must be outside the source project ({REPO_ROOT}); got {artifact_root}"
        )
    config_dir = artifact_root / "prior_configs" / task_slug / "generated" / run_name
    csv_dir = artifact_root / "prior_sweep_csv" / "generated"
    csv_path = csv_dir / f"{name_prefix}.csv"

    dataset_rel = _relative_to_repo(cfg.dataset_path)
    generated: list[GeneratedBcPrior] = []
    for chunk_size in cfg.chunk_sizes:
        if chunk_size <= 0:
            raise ValueError(f"BC chunk sizes must be positive, got {chunk_size}")
        for action_chunk_stride in cfg.action_chunk_strides:
            if action_chunk_stride <= 0:
                raise ValueError(f"BC action chunk strides must be positive, got {action_chunk_stride}")
            stride_suffix = "" if action_chunk_stride == 1 else f"_stride{action_chunk_stride}"
            for raw_variant in cfg.variants:
                variant = _normalize_variant(raw_variant)
                if variant not in VARIANT_HIDDEN_DIMS:
                    raise ValueError(f"Unknown BC variant {raw_variant!r}; expected one of: {_known_variants_text()}")
                name = f"{name_prefix}_chunking_{chunk_size}_{variant}{stride_suffix}"
                config_path = config_dir / f"{name}.yaml"
                data = _config_dict(
                    cfg,
                    dataset_path=dataset_rel,
                    task_slug=task_slug,
                    run_name=run_name,
                    artifact_root=artifact_root,
                    chunk_size=chunk_size,
                    action_chunk_stride=action_chunk_stride,
                    hidden_dims=VARIANT_HIDDEN_DIMS[variant],
                )
                _write_yaml(
                    config_path,
                    data,
                    header=f"Auto-generated BC prior config for {cfg.task_name} from {dataset_rel}.",
                    overwrite=cfg.overwrite,
                )
                generated.append(
                    GeneratedBcPrior(
                        name=name,
                        variant=variant,
                        chunk_size=chunk_size,
                        action_chunk_stride=action_chunk_stride,
                        config_path=config_path,
                        prior_path="",
                    )
                )

    if not csv_path.exists() or cfg.overwrite:
        csv_dir.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "name",
                "config",
                "prior_path",
                "best_checkpoint",
                "success_rate",
                "variant",
                "chunk_size",
                "action_chunk_stride",
                "diagnosis",
                "dataset",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for item in generated:
                writer.writerow(
                    {
                        "name": item.name,
                        "config": _relative_to_repo(item.config_path),
                        "prior_path": item.prior_path,
                        "best_checkpoint": item.prior_path,
                        "success_rate": "",
                        "variant": item.variant,
                        "chunk_size": item.chunk_size,
                        "action_chunk_stride": item.action_chunk_stride,
                        "diagnosis": (
                            f"variant={item.variant}; "
                            f"hidden_dims={VARIANT_HIDDEN_DIMS[item.variant]}; "
                            f"chunk_size={item.chunk_size}; "
                            f"action_chunk_stride={item.action_chunk_stride}; "
                            f"obs_key={cfg.obs_key}; "
                            f"action_mode={cfg.action_mode}; "
                            f"time_shift={cfg.time_shift}; "
                            f"dataset={dataset_rel}"
                        ),
                        "dataset": dataset_rel,
                    }
                )

    return BcPriorPrepResult(
        task_slug=task_slug,
        config_dir=config_dir,
        csv_path=csv_path,
        generated=tuple(generated),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate BC prior YAML configs and a sweep CSV for a collected MPC dataset.",
    )
    parser.add_argument("--dataset", required=True, help="Collected HDF5 dataset path.")
    parser.add_argument("--task_name", required=True, help="Isaac task name used for collection.")
    parser.add_argument("--robot_name", default="fr3_finray", help="Robot label used for artifact ownership.")
    parser.add_argument("--run_name", required=True, help="Collection run name used in generated filenames.")
    parser.add_argument("--artifact_root", default=None, help="Override robot artifact root.")
    parser.add_argument("--task_slug", default=None, help="Override inferred task slug.")
    parser.add_argument("--name_prefix", default=None, help="Override generated row/config name prefix.")
    parser.add_argument("--chunk_sizes", type=int, nargs="+", default=list(DEFAULT_CHUNK_SIZES), help="BC action chunk sizes.")
    parser.add_argument("--chunk_strides", type=int, nargs="+", default=list(DEFAULT_CHUNK_STRIDES), help="BC action chunk strides.")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(DEFAULT_VARIANTS),
        help="BC model variants to generate. Default: f ff fff f3 ff3 fff3. Also supports bc and class.",
    )
    parser.add_argument("--obs_key", default="state", help="HDF5 observation key to train on.")
    parser.add_argument("--action_mode", default="absolute_ee", help="HDF5 action branch to train on.")
    parser.add_argument("--time_shift", type=int, default=0, help="BC temporal shift.")
    parser.add_argument("--num_epochs", type=int, default=150, help="BC training epochs.")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader worker processes for generated BC configs.")
    parser.add_argument("--pin_memory", action="store_true", help="Enable pinned-memory DataLoader batches in generated BC configs.")
    parser.add_argument("--no_sample_weighting", action="store_true", help="Disable stage-based loss weighting.")
    parser.add_argument(
        "--sample_weighting_no_validation",
        action="store_true",
        help="Keep stage weights out of validation loss and best-checkpoint selection.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite generated files if they exist.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = prepare_bc_prior_sweep(
        BcPriorPrepConfig(
            dataset_path=resolve_artifact_path(args.dataset, "BC training trajectory"),
            task_name=args.task_name,
            robot_name=args.robot_name,
            run_name=args.run_name,
            artifact_root=resolve_artifact_path(args.artifact_root, "BC artifact root") if args.artifact_root else None,
            task_slug=args.task_slug,
            name_prefix=args.name_prefix,
            chunk_sizes=tuple(args.chunk_sizes),
            action_chunk_strides=tuple(args.chunk_strides),
            variants=tuple(args.variants),
            obs_key=args.obs_key,
            action_mode=args.action_mode,
            time_shift=args.time_shift,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            num_epochs=args.num_epochs,
            sample_weighting_enabled=not args.no_sample_weighting,
            sample_weighting_validation=not args.sample_weighting_no_validation,
            overwrite=args.overwrite,
        )
    )
    print("=== BC Prior Prep ===")
    print(f"  Task slug:   {result.task_slug}")
    print(f"  Config dir:  {result.config_dir}")
    print(f"  Sweep CSV:   {result.csv_path}")
    print(f"  YAML files:  {len(result.generated)}")


if __name__ == "__main__":
    main()
