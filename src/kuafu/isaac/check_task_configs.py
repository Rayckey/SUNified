"""Load task configurations and check their local inputs in the Isaac runtime."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

from kuafu.paths import PROJECT_ROOT, external_artifact_root, resolve_artifact_path
from kuafu.pipeline.task_catalog import get_task, list_tasks
from kuafu.isaac.learned.ur5.assets import asset_directory as ur5_asset_directory


def _reference(path):
    if "://" in path:
        return path
    path = Path(path).resolve()
    ur5_root = ur5_asset_directory().resolve()
    for root, prefix in ((PROJECT_ROOT, ""), (ur5_root, "${KUAFU_UR5_ASSET_DIR}/"),
                         (external_artifact_root(), "${KUAFU_ARTIFACT_ROOT}/")):
        if path.is_relative_to(root):
            return prefix + path.relative_to(root).as_posix()
    return str(path)


def configuration_inputs(cfg):
    """Return portable references and missing local files for one configuration."""
    assets = {}
    for name, value in vars(cfg).items():
        if name.startswith("_") or name.isupper():
            continue
        path = getattr(getattr(value, "spawn", None), "usd_path", None)
        if isinstance(path, str):
            assets[name] = path
    mpc = getattr(cfg, "mpc_config", {})
    controller = {key: mpc[key] for key in
                  ("urdf_path", "initial_residual", "initial_predicate") if mpc.get(key)}
    program = mpc.get("load_program")
    missing = [path for path in (*assets.values(), *controller.values())
               if "://" not in path and not Path(path).is_file()]
    if program and not (Path(program) / "program").is_dir():
        missing.append(program)
    inputs = {
        "assets": {name: _reference(path) for name, path in assets.items()},
        "controller_inputs": {name: _reference(path) for name, path in controller.items()},
        "default_program": _reference(program) if program else None,
        "interaction_frames": {
            name: {"description": value[0], "prim_path": value[1]}
            for name, value in getattr(cfg, "detectable_objects", {}).items()
        },
    }
    return inputs, missing


def main(argv=None):
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", help="Catalog task ID; defaults to all registered tasks")
    parser.add_argument("--robot", help="Catalog robot ID; defaults to all robots")
    parser.add_argument("--variants", action="store_true", help="Also load every registered policy variant")
    parser.add_argument("--output", default="validation/task-configurations.json")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)
    tasks = [row for row in list_tasks(args.robot) if row["status"] == "registered"
             and (args.task is None or row["id"] == args.task)]
    if not tasks:
        parser.error("Choose a registered task and robot from kuafu tasks list")
    output = resolve_artifact_path(args.output)
    launcher = AppLauncher(args)
    rows = []
    passed = False
    try:
        from isaaclab_tasks.utils import parse_env_cfg

        for task in tasks:
            for entry in get_task(task["id"], args.robot)["environments"]:
                ids = [entry["gym_id"], *(entry["variants"] if args.variants else [])]
                for gym_id in ids:
                    row = {"task": task["id"], "robot": entry["robot_id"], "gym_id": gym_id}
                    try:
                        importlib.import_module(entry["registration_module"])
                        cfg = parse_env_cfg(gym_id, device=args.device, num_envs=1)
                        inputs, missing = configuration_inputs(cfg)
                        matches = gym_id != entry["gym_id"] or inputs == entry["configuration"]
                        row.update(inputs=inputs, missing=missing, catalog_matches=matches,
                                   passed=not missing and matches)
                    except Exception as error:
                        row.update(passed=False, error=f"{type(error).__name__}: {error}")
                    rows.append(row)
        passed = bool(rows) and all(row["passed"] for row in rows)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({
            "scope": "Configuration loading and local file checks; remote assets and physics require a scene replay.",
            "passed": passed, "configurations": rows,
        }, indent=2) + "\n")
        print(json.dumps({"passed": passed, "configurations": len(rows), "report": str(output),
                          "failed": [row["gym_id"] for row in rows if not row["passed"]]}), flush=True)
    finally:
        import omni.kit.app

        omni.kit.app.get_app().post_quit(0 if passed else 1)
        launcher.app.close(wait_for_replicator=False)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
