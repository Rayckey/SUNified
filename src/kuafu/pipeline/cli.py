"""Command-line interface for the Kuafu harness."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .config import ConfigError, PipelineConfig, project_root
from .pipeline import Pipeline, PipelineError, STAGES, stage_slice


DEFAULT_CONFIG = project_root() / "assets/configs" / "fr3_finray_drawer.example.toml"


def _add_range_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--from-stage", choices=STAGES, default=None)
    parser.add_argument("--through-stage", choices=STAGES, default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kuafu",
        description=(
            "Kuafu: generate SUN Programs, collect MPC demonstrations, train BC priors, "
            "evaluate policies, and train residual RL."
        ),
    )
    parser.add_argument(
        "-c",
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Pipeline TOML file (default: assets/configs/fr3_finray_drawer.example.toml).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    handoff = subparsers.add_parser("handoff", help="Create or resume one complete agent run.")
    handoff.add_argument("handoff_args", nargs=argparse.REMAINDER)

    tasks = subparsers.add_parser("tasks", help="Browse all task families, robots and SUN Program examples.")
    tasks.add_argument("task_args", nargs=argparse.REMAINDER)

    agent = subparsers.add_parser("agent", help="Tools and MCP server for your own agent.")
    agent.add_argument("agent_args", nargs=argparse.REMAINDER)

    dynamics = subparsers.add_parser("train-dynamics", help="Train a PETS-style ensemble on recorded transitions.")
    dynamics.add_argument("dynamics_args", nargs=argparse.REMAINDER)

    dreamer = subparsers.add_parser("train-dreamerv3", help="Train DreamerV3's world model on recorded sequences.")
    dreamer.add_argument("dreamer_args", nargs=argparse.REMAINDER)

    assets = subparsers.add_parser(
        "assets", help="Generate, texture, export, and validate simulation assets from video.",
        description="Run 'kuafu assets --help' for the asset-generation commands.",
    )
    assets.add_argument("asset_args", nargs=argparse.REMAINDER)

    drawer = subparsers.add_parser(
        "verify-drawer", help="Inspect, generate, or replay a program on an exported cabinet in Isaac Lab.",
    )
    drawer.add_argument("drawer_args", nargs=argparse.REMAINDER)

    doctor = subparsers.add_parser("doctor", help="Check paths, runtime, and credentials without launching Isaac.")
    doctor.add_argument("--json", action="store_true", dest="as_json")

    status = subparsers.add_parser("status", help="Show persisted stage and artifact state.")
    status.add_argument("--json", action="store_true", dest="as_json")

    plan = subparsers.add_parser("plan", help="Print commands without running or writing artifacts.")
    _add_range_options(plan)
    plan.add_argument("--prepare-only", action="store_true", help="Do not execute the generated residual launcher.")

    run = subparsers.add_parser("run", help="Run a sequential range of pipeline stages.")
    _add_range_options(run)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--force", action="store_true", help="Re-run stages recorded complete and overwrite generated configs.")
    run.add_argument("--prepare-only", action="store_true", help="Generate but do not execute the residual launcher.")

    stage = subparsers.add_parser("stage", help="Run one named pipeline stage.")
    stage.add_argument("name", choices=STAGES)
    stage.add_argument("--dry-run", action="store_true")
    stage.add_argument("--force", action="store_true")
    stage.add_argument("--prepare-only", action="store_true", help="Generate but do not execute the residual launcher.")
    return parser


def _print_status(pipeline: Pipeline, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(pipeline.state.data, indent=2, sort_keys=True))
        return
    print(f"State: {pipeline.state.path}")
    stages = pipeline.state.data.get("stages", {})
    for stage in STAGES:
        status = stages.get(stage, {}).get("status", "pending") if isinstance(stages, dict) else "pending"
        print(f"  {stage:15s} {status}")
    artifacts = pipeline.state.data.get("artifacts", {})
    if artifacts:
        print("Artifacts:")
        for key, value in sorted(artifacts.items()):
            print(f"  {key}: {value}")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "tasks":
        from .task_catalog import main as tasks_main
        return tasks_main(argv[1:])
    if argv and argv[0] == "handoff":
        from .handoff_cli import main as handoff_main
        return handoff_main(argv[1:])
    if argv and argv[0] == "train-dreamerv3":
        from kuafu.dynamics.dreamerv3 import main as dreamer_main
        return dreamer_main(argv[1:])
    if argv and argv[0] == "train-dynamics":
        try:
            from kuafu.dynamics.__main__ import main as dynamics_main
        except ImportError as error:
            print(f"Dynamics training requires: pip install -e '.[dynamics]' ({error})", file=sys.stderr)
            return 2
        return dynamics_main(argv[1:])
    if argv and argv[0] == "agent":
        try:
            from kuafu.agents.cli import main as agent_main
            return agent_main(argv[1:])
        except ImportError as error:
            print(f"Agent tools require: pip install -e '.[agents]' ({error})", file=sys.stderr)
            return 2
    if argv and argv[0] == "verify-drawer":
        from kuafu.programs.generated_drawer import main as drawer_main
        return drawer_main(argv[1:])
    # Asset preparation does not need a manipulation-task manifest or Isaac Sim.
    # Keep its optional geometry dependencies out of the training CLI imports.
    if argv and argv[0] == "assets":
        try:
            from kuafu.asset_generation.cli import main as asset_main
        except ImportError as error:
            print(
                "Asset generation requires the optional dependencies: "
                "python -m pip install -e '.[assets]' "
                f"({error})", file=sys.stderr,
            )
            return 2
        return asset_main(argv[1:])
    args = build_parser().parse_args(argv)
    if args.command == "handoff":
        return main(["handoff", *args.handoff_args])
    if args.command == "agent":
        return main(["agent", *args.agent_args])
    if args.command == "train-dynamics":
        return main(["train-dynamics", *args.dynamics_args])
    if args.command == "train-dreamerv3":
        return main(["train-dreamerv3", *args.dreamer_args])
    if args.command == "assets":
        return main(["assets", *args.asset_args])
    if args.command == "verify-drawer":
        return main(["verify-drawer", *args.drawer_args])
    try:
        config = PipelineConfig.load(args.config)
        if args.command == "doctor":
            pipeline = Pipeline(config, dry_run=True)
            errors, warnings = pipeline.doctor()
            if args.as_json:
                print(json.dumps({"ok": not errors, "errors": errors, "warnings": warnings}, indent=2))
            else:
                print("Kuafu workflow doctor")
                for warning in warnings:
                    print(f"  WARN: {warning}")
                for error in errors:
                    print(f"  ERROR: {error}")
                if not errors:
                    print("  OK: component and runtime paths are present")
            return 1 if errors else 0

        if args.command == "status":
            pipeline = Pipeline(config, dry_run=True)
            _print_status(pipeline, as_json=args.as_json)
            return 0

        if args.command == "plan":
            stages = stage_slice(args.from_stage, args.through_stage)
            print(f"Planning stages: {', '.join(stages)}")
            Pipeline(config, dry_run=True, prepare_only=args.prepare_only).run(stages)
            return 0

        if args.command == "run":
            stages = stage_slice(args.from_stage, args.through_stage)
            Pipeline(
                config,
                dry_run=args.dry_run,
                force=args.force,
                prepare_only=args.prepare_only,
            ).run(stages)
            return 0

        if args.command == "stage":
            Pipeline(
                config,
                dry_run=args.dry_run,
                force=args.force,
                prepare_only=args.prepare_only,
            ).run_stage(args.name)
            return 0
        raise AssertionError(f"Unhandled command: {args.command}")
    except (
        ConfigError,
        PipelineError,
        FileNotFoundError,
        ValueError,
        OSError,
        subprocess.CalledProcessError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
