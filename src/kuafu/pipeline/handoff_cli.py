"""Create and inspect a persistent agent run from the shell."""

import argparse
import json

from kuafu.paths import external_artifact_root
from .handoff import HandoffRun


def main(argv=None):
    parser = argparse.ArgumentParser(prog="kuafu handoff")
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("init", help="Save task inputs and create a run")
    create.add_argument("--task", required=True)
    create.add_argument("--description", default="")
    create.add_argument("--video", action="append", default=[])
    create.add_argument("--run-id")
    status = sub.add_parser("status", help="Show attempts, artifacts and recovery information")
    status.add_argument("run_id")
    status.add_argument("--verify", action="store_true", help="Check recorded output hashes before resuming")
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            run = HandoffRun.create(external_artifact_root(), task=args.task, description=args.description,
                                    videos=args.video, run_id=args.run_id)
        else:
            run = HandoffRun(external_artifact_root(), args.run_id)
        print(json.dumps(run.status(verify_outputs=getattr(args, "verify", False)), indent=2))
        return 0
    except (OSError, ValueError) as error:
        parser.exit(2, f"{error}\n")
