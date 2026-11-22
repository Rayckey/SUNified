"""Run the cabinet video example through MPC collection, BC and policy evaluation."""

import argparse
import json
from pathlib import Path
import runpy

from kuafu.agents import ToolService
from kuafu.agents.examples.custom_agent import wait_for_job
from kuafu.paths import PROJECT_ROOT


def main():
    settings = json.loads((Path(__file__).resolve().parent / "input/task.json").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--episodes", type=int, default=settings["episodes"])
    parser.add_argument("--method", choices=("nearest_neighbor", "mlp"), default=settings["method"])
    parser.add_argument("--epochs", type=int, default=settings["epochs"], help="Epochs per SUN stage for --method mlp")
    parser.add_argument("--seed", type=int, default=settings["seed"])
    args = parser.parse_args()
    formation = runpy.run_path(str(PROJECT_ROOT / "examples/01_video_to_rollout/run.py"))
    formation["main"](["--run-id", args.run_id] + (["--resume"] if args.resume else []))
    service = ToolService(run_id=args.run_id)

    def state():
        response = service.call("kuafu_run_status", {"verify_outputs": True})
        if not response["ok"]:
            raise RuntimeError(response)
        return response["result"]

    def complete(step, tool, arguments):
        prior = state()["steps"].get(step)
        response = None
        if args.resume and prior and prior["status"] == "complete" and not prior.get("stale"):
            directory = Path(prior["directory"])
            if json.loads((directory / "request.json").read_text()) == {"tool": tool, "arguments": arguments}:
                response = json.loads((directory / "response.json").read_text())
        response = response or service.call(tool, arguments)
        if not response["ok"]:
            raise RuntimeError(response)
        result = wait_for_job(service, response["result"]["job_id"])
        print(json.dumps({"step": step, **result}, indent=2), flush=True)
        if result["status"] != "succeeded":
            raise RuntimeError(f"{step} failed; inspect its logs and resume with the same run ID")
        return result

    verification = state()["steps"]["verify-program"]["job_id"]
    collection = complete("collect", "kuafu_collect_drawer",
                          {"verification_job": verification, "episodes": args.episodes, "seed": args.seed})
    training = complete("train-prior", "kuafu_train_drawer_policy",
                        {"collection_job": collection["job_id"], "method": args.method,
                         "epochs": args.epochs, "seed": args.seed})
    evaluation = complete("eval-prior", "kuafu_evaluate_drawer_policy",
                          {"training_job": training["job_id"], "seed": args.seed + args.episodes})
    summary = {"run_id": args.run_id, "collection": collection["result"], "training": training["result"],
               "evaluation": evaluation["result"],
               "artifacts": {**collection["artifacts"], **training["artifacts"], **evaluation["artifacts"]}}
    path = Path(state()["directory"]) / "video_to_policy.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Results: {path}", flush=True)
    return 0 if evaluation["result"]["policy_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
