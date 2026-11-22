"""Durable local jobs; client disconnects do not interrupt a running simulation.

The worker alone owns the child process group. Cancellation writes a request
file, so a reconnecting client never sends signals to a potentially reused PID.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

from kuafu.paths import PROJECT_ROOT, SOURCE_ROOT


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def asset_hashes(asset):
    return {
        name: digest(Path(asset) / name) for name in ("asset_dynamic.usda", "asset_geometry.usda", "semantics.json")
    }


def tree_hashes(path):
    return {str(p.relative_to(path)): digest(p) for p in sorted(Path(path).rglob("*")) if p.is_file()}


def process_token(pid):
    try:
        # /proc start time distinguishes a surviving worker from PID reuse.
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if fields[0] == "Z" else fields[19]
    except (OSError, IndexError):
        return None


def launch(directory, request):
    write_json(directory / "request.json", request)
    write_json(directory / "status.json", {"status": "queued"})
    env = {**os.environ, "PYTHONPATH": str(SOURCE_ROOT), "PYTHONUNBUFFERED": "1"}
    with (directory / "worker.log").open("w") as log:
        worker = subprocess.Popen(
            [sys.executable, "-m", "kuafu.agents.jobs", str(directory)],
            env=env,
            cwd=PROJECT_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    write_json(directory / "worker.json", {"pid": worker.pid, "token": process_token(worker.pid)})


def status(directory):
    result = read_json(directory / "status.json")
    if result["status"] in {"queued", "running"}:
        worker = read_json(directory / "worker.json")
        if worker["token"] is None or process_token(worker["pid"]) != worker["token"]:
            result = {
                "status": "interrupted",
                "error": "Worker stopped before writing a final result; inspect worker.log.",
            }
    progress = directory / "output/progress.json"
    if progress.is_file():
        result["progress"] = read_json(progress)
    return result


def run_process(command, directory, timeout_s, env):
    """Bound runtime and terminate only the process group created for this job."""
    with (directory / "run.log").open("w") as log:
        child = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        start = time.monotonic()
        reason = None
        try:
            while child.poll() is None:
                if (directory / "cancel.request").exists():
                    reason = "cancelled"
                    break
                if time.monotonic() - start > timeout_s:
                    reason = "timed_out"
                    break
                time.sleep(0.2)
        finally:
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                except ProcessLookupError:
                    child.wait()
        return child.returncode, reason


def command_for(directory, request):
    if request["kind"] in {"drawer-collect", "drawer-train", "drawer-evaluate"}:
        return [request["python"], "-m", "kuafu.pipeline.drawer_policy", str(directory)]
    if request["kind"] == "stage":
        command = [request["python"], "-m", "kuafu", "-c", str(directory / "pipeline.toml"), "stage", request["stage"]]
        if request["prepare_only"]:
            command.append("--prepare-only")
        return command
    context = request["context"]
    command = [
        request["python"],
        "-m",
        "kuafu",
        "verify-drawer",
        "--headless",
        "--enable_cameras",
        "--device",
        request["device"],
        "--asset",
        context["asset"],
        "--task",
        context["task"],
        "--target-travel",
        str(context["target_travel_m"]),
        "--position",
        *map(str, context["position"]),
        "--yaw",
        str(context["yaw_deg"]),
        "--max-requests",
        "0",
        "--mode",
        "inspect" if request["kind"] == "inspect" else "replay",
        "--output",
        str(directory / "output"),
    ]
    if request["kind"] == "verify":
        command += [
            "--program",
            str(directory / "input/program"),
            "--max-steps",
            str(request["max_steps"]),
            "--stage-steps",
            str(request["stage_steps"]),
        ]
    return command


def finish(directory, request, returncode, reason):
    result = {"status": reason or ("succeeded" if returncode == 0 else "failed"), "returncode": returncode}
    if request["kind"] in {"drawer-collect", "drawer-train", "drawer-evaluate"}:
        if result["status"] == "succeeded":
            from kuafu.pipeline.drawer_policy import finish as finish_learning

            result.update(finish_learning(directory, request))
        return result
    if request["kind"] == "stage":
        from kuafu.pipeline.config import PipelineConfig

        config = PipelineConfig.load(directory / "pipeline.toml")
        state_path = config.task_artifact_root / "pipeline_state.json"
        result["pipeline_state"] = str(state_path)
        if state_path.is_file():
            state = read_json(state_path)
            stage = state.get("stages", {}).get(request["stage"], {})
            result.update(stage_state=stage, artifacts=stage.get("artifacts", {}))
            if result["status"] == "succeeded":
                if stage.get("status") == "prepared":
                    result["status"] = "prepared"
                elif stage.get("status") != "complete" or not stage.get("files"):
                    result.update(status="failed", error="Stage exited without a completed, verified handoff")
                if config.data.get("handoff") and stage.get("status") in {"complete", "prepared"}:
                    # The standalone pipeline edits its prior CSV during evaluation.
                    # Handoff attempts own separate copies, so record their hashes too.
                    files = stage.setdefault("files", {})
                    for value in stage.get("artifacts", {}).values():
                        if isinstance(value, str) and Path(value).is_file() and value not in files:
                            files[value] = digest(value)
        elif result["status"] == "succeeded":
            result.update(status="failed", error="Stage exited without a persisted pipeline state")
        return result
    report_path = directory / "output/verification.json"
    if report_path.exists():
        report = read_json(report_path)
        result["verification"] = {key: value for key, value in report.items() if key not in {"initial", "final"}}
        # Full states are available in the report; keep tool responses compact.
        for stage in result["verification"].get("replay_stages", []):
            stage.pop("state", None)
    if result["status"] != "succeeded":
        return result
    scene_path = directory / "output/scene.json"
    scene = read_json(scene_path)
    if (
        scene["asset_hashes"] != request["context"]["asset_hashes"]
        or asset_hashes(request["context"]["asset"]) != scene["asset_hashes"]
    ):
        raise ValueError("Asset changed during the job; inspect the new revision before proposing a program.")
    if request["kind"] == "inspect":
        result["scene"] = scene
        return result
    if not report_path.exists():
        raise ValueError("Verification report is missing; exit status alone is not proof of success.")
    if not (
        report.get("passed")
        and report.get("program_passed")
        and report.get("solver_passed")
        and report.get("physical", {}).get("passed")
        and report.get("model_requests") == 0
    ):
        return {
            **result,
            "status": "failed",
            "error": "Exit status alone is not proof: physical/program/solver verification did not pass.",
        }
    if tree_hashes(directory / "input/program") != request["program_hashes"]:
        raise ValueError("Program changed during verification; refusing to publish a mismatched result.")
    # Promote an immutable copy of exactly what was verified. The proposed
    # revision stays unverified, including when another attempt later fails.
    accepted = directory / "accepted"
    shutil.copytree(directory / "input/program", accepted / "program")
    for stage in (accepted / "program").glob("subtask_*"):
        (stage / "status.txt").write_text("status: succeeded\n")
    from kuafu.artifact_contracts import learned_program_identity

    identity = learned_program_identity(accepted)
    write_json(
        accepted / "provenance.json",
        {
            "context": request["context"],
            "program_hashes": request["program_hashes"],
            "accepted_identity": identity,
            "verification_sha256": digest(report_path),
        },
    )
    (accepted / "stats.txt").write_text("status: succeeded\nsource: external_agent\n")
    (accepted / "success.txt").write_text("True\n")
    result["accepted_run"] = str(accepted)
    result["accepted_identity"] = identity
    return result


def worker(directory):
    directory = Path(directory)
    write_json(directory / "worker.json", {"pid": os.getpid(), "token": process_token(os.getpid())})
    request = read_json(directory / "request.json")
    write_json(directory / "status.json", {"status": "running", "started_at": time.time()})
    try:
        if (directory / "cancel.request").exists():
            result = {"status": "cancelled", "returncode": None}
        else:
            env = {
                **os.environ,
                "PYTHONPATH": str(SOURCE_ROOT),
                "PYTHONUNBUFFERED": "1",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
            }
            code, reason = run_process(command_for(directory, request), directory, request["timeout_s"], env)
            result = finish(directory, request, code, reason)
    except Exception as exc:
        result = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    write_json(directory / "status.json", {**result, "finished_at": time.time()})


if __name__ == "__main__":
    worker(sys.argv[1])
