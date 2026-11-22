"""Replay saved agent proposals through the same tools used by live agents."""

import argparse
import json
from pathlib import Path

from kuafu.agents import ToolService
from kuafu.agents.examples.custom_agent import wait_for_job
from kuafu.paths import PROJECT_ROOT


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    example = Path(__file__).resolve().parent
    task = json.loads((example / "input/task.json").read_text())
    service = ToolService()
    if args.resume:
        if not args.run_id:
            parser.error("--resume requires --run-id")
        response = service.call("kuafu_run_status", {"run_id": args.run_id})
    else:
        request = {"task": task["task"], "description": task["object_description"],
                   "videos": [str(example / f"input/IMG_{number:04d}.mov") for number in (28, 29)]}
        if args.run_id:
            request["run_id"] = args.run_id
        response = service.call("kuafu_create_run", request)
    if not response["ok"]:
        raise RuntimeError(response["error"])
    state = response["result"]
    service.run_id = state["run_id"]
    print(f"Run: {service.run_id}\nArtifacts: {state['directory']}", flush=True)

    def call(step, tool, arguments):
        state = service.call("kuafu_run_status", {"verify_outputs": args.resume})["result"]
        prior = state["steps"].get(step)
        if args.resume and prior and prior["status"] == "complete" and not prior.get("stale"):
            directory = Path(prior["directory"])
            saved = json.loads((directory / "request.json").read_text())
            if saved == {"tool": tool, "arguments": arguments}:
                print(f"Reuse {step}: {directory}", flush=True)
                return json.loads((directory / "response.json").read_text())["result"]
        response = service.call(tool, arguments)
        if not response["ok"]:
            raise RuntimeError(response)
        print(f"{step}: {response['directory']}", flush=True)
        return response["result"]

    videos = [str(Path(state["directory"]) / item["path"]) for item in state["manifest"]["inputs"]]
    call("prepare-video", "kuafu_prepare_video", {"video": videos[0], "reference_videos": videos[1:],
         "description": task["object_description"], "frames": 8, "motion": "articulated"})
    scene = json.loads((PROJECT_ROOT / task["asset_recipe"]).read_text())
    asset = call("build-asset", "kuafu_build_asset", {"scene": scene, "mass_kg": 2.5, "estimated_mass": True})
    inspection = call("inspect", "kuafu_inspect_drawer", {"asset": asset["asset"], "task": task["task"],
                      "target_travel_m": task["target_travel_m"]})
    observed = wait_for_job(service, inspection["job_id"])
    if observed["status"] != "succeeded":
        raise RuntimeError(observed)
    proposal = json.loads((example / "input/program.json").read_text())
    program = call("form-program", "kuafu_submit_program", {"context_job": inspection["job_id"], "program": proposal})
    verification = call("verify-program", "kuafu_verify_program", {"program_id": program["program_id"]})
    result = wait_for_job(service, verification["job_id"])
    print(json.dumps(result, indent=2))
    if result["status"] != "succeeded" or not result.get("accepted_run"):
        raise RuntimeError("Read verification feedback in the run directory and revise the proposal")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
