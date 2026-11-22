"""Optional experiment command line: export, train, evaluate, and audit."""

import argparse
import json
from pathlib import Path

from kuafu.paths import PROJECT_ROOT

from .artifacts import git_revision
from .runtime import configure


def main(argv=None):
    parser = argparse.ArgumentParser(description="Learned-stage DP3 experiment using Kuafu2Real")
    parser.add_argument("--kuafu2real-root")
    parser.add_argument("--dp3-root")
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="Export stage-labeled HDF5 demonstrations")
    export.add_argument("inputs", nargs="+")
    export.add_argument("--output", required=True)
    export.add_argument("--task", required=True)
    export.add_argument("--stages", required=True, help="JSON array of stage names in teacher stage-ID order")
    export.add_argument("--num-points", type=int, default=4096)
    export.add_argument("--action-format", choices=("eff-simple", "bc"), default="eff-simple")
    export.add_argument("--action-shift", type=int)
    export.add_argument("--clip-zone", help="Kuafu2Real point-cloud crop YAML")
    export.add_argument("--seed", type=int, default=42)
    export.add_argument("--validation-fraction", type=float, default=0.15)
    export.add_argument("--test-fraction", type=float, default=0.15)
    relative = commands.add_parser("relative", help="Convert a stage dataset to a stationary object's root frame")
    relative.add_argument("source")
    relative.add_argument("--output", required=True)
    relative.add_argument("--reference-object", required=True)
    relative.add_argument("--chunk-rows", type=int, default=128)
    train = commands.add_parser("train", help="Train matched A/B/C variants")
    train.add_argument("--config", required=True)
    train.add_argument("--dataset", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--variants", nargs="+", choices=list("ABC"), default=list("ABC"))
    train.add_argument("--resume", action="store_true")
    evaluate = commands.add_parser("evaluate", help="Evaluate paired held-out simulator conditions")
    evaluate.add_argument("--experiment", required=True)
    evaluate.add_argument("--dataset", required=True)
    evaluate.add_argument("--protocol", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--device", default="cuda:0")
    evaluate.add_argument("--adapter", default="kuafu.sim2real.dp3.isaac_adapter:create_adapter")
    args = parser.parse_args(argv)
    roots = configure(args.kuafu2real_root, args.dp3_root)
    # Baseline YAML uses the same environment variable as Kuafu2Real's checkout.
    import os
    os.environ["KUAFU2REAL_ROOT"] = str(roots["kuafu2real"])
    provenance = {name: git_revision(path) for name, path in roots.items()}
    provenance["kuafu"] = git_revision(PROJECT_ROOT)
    import importlib.metadata
    import platform
    provenance["runtime"] = {"python": platform.python_version(), "packages": {
        name: importlib.metadata.version(name)
        for name in ("torch", "diffusers", "zarr", "numpy", "hydra-core", "h5py")
    }}
    if args.command == "export":
        from kuafu2real.clip_zone import load_point_cloud_clip_zone
        from .export import export_dataset
        result = export_dataset(
            args.inputs, args.output, task=args.task, stage_names=json.loads(Path(args.stages).read_text()),
            num_points=args.num_points, action_format=args.action_format, action_shift=args.action_shift,
            seed=args.seed, validation_fraction=args.validation_fraction, test_fraction=args.test_fraction,
            clip_zone=load_point_cloud_clip_zone(args.clip_zone), provenance=provenance,
        )
        print(json.dumps({"dataset_id": result["dataset_id"], "output": args.output, "splits": result["splits"]}, indent=2))
    elif args.command == "relative":
        from kuafu.paths import resolve_artifact_path
        from .relative import convert_object_relative
        output = resolve_artifact_path(args.output, "Object-relative dataset output")
        result = convert_object_relative(args.source, output, reference_object=args.reference_object,
                                         chunk_rows=args.chunk_rows, provenance=provenance)
        print(json.dumps({"dataset_id": result["dataset_id"], "output": str(output),
                          "trajectories": result["export"]["num_episodes"],
                          "decisions": result["export"]["num_steps"]}, indent=2))
    elif args.command == "train":
        from .train import run_training
        print(json.dumps(run_training(args.config, args.dataset, args.output, variants=args.variants,
                                      resume=args.resume, provenance=provenance), indent=2))
    else:
        from .evaluate import evaluate_experiment
        result = evaluate_experiment(args.experiment, args.dataset, args.protocol, args.output,
                                     adapter_factory=args.adapter, device=args.device)
        print(json.dumps(result["aggregate"], indent=2))


if __name__ == "__main__":
    main()
