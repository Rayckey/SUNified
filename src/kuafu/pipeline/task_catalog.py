"""Discover bundled tasks without importing a simulator or model provider."""

from __future__ import annotations

import argparse
import json

from kuafu.paths import TASK_ROOT


def list_tasks(robot_id=None):
    """Return task summaries, optionally filtered to a registered robot."""
    catalog = json.loads((TASK_ROOT / "index.json").read_text())
    if robot_id and robot_id not in catalog["robots"]:
        raise ValueError(f"Unknown robot {robot_id!r}; choose from {', '.join(catalog['robots'])}")
    return [task for task in catalog["tasks"] if not robot_id or robot_id in task["robots"]]


def get_task(task_id, robot_id=None):
    """Read a task's instructions, assets, registrations and program examples."""
    item = next((task for task in list_tasks() if task["id"] == task_id), None)
    if item is None:
        raise ValueError(f"Unknown task {task_id!r}; run kuafu tasks list")
    result = json.loads((TASK_ROOT / item["spec"]).read_text())
    if robot_id:
        selected = [env for env in result["environments"] if env["robot_id"] == robot_id]
        if not selected:
            raise ValueError(f"{task_id} has no registered {robot_id} environment; choose from {item['robots']}")
        result["environments"] = selected
        if "configuration" in selected[0]:
            program_path = selected[0]["configuration"]["default_program"]
            result["example_program"] = next(
                (program for program in result["saved_programs"] if program["path"] == program_path), None,
            )
            result["interaction_frames"] = selected[0]["configuration"]["interaction_frames"]
            result["interaction_frames_source"] = selected[0]["config_entry_point"]
            result["scene_source"] = selected[0]["source"]
            result["success_check"]["source"] = selected[0]["source"]
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(prog="kuafu tasks", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    listing = sub.add_parser("list", help="List task families, robots and available interfaces")
    listing.add_argument("--robot", dest="robot_id")
    listing.add_argument("--json", action="store_true", dest="as_json")
    show = sub.add_parser("show", help="Read one task's complete input and output references")
    show.add_argument("task_id")
    show.add_argument("--robot", dest="robot_id")
    args = parser.parse_args(argv)
    try:
        if args.command == "show":
            print(json.dumps(get_task(args.task_id, args.robot_id), indent=2))
        else:
            tasks = list_tasks(args.robot_id)
            if args.as_json:
                print(json.dumps(tasks, indent=2))
            else:
                for task in tasks:
                    print(f"{task['id']:24s} {task['title']} [{task['status']}]" +
                          (f" — {', '.join(task['robots'])}" if task['robots'] else ""))
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
