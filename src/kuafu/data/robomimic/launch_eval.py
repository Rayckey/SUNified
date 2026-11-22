#!/usr/bin/env python3
"""Evaluate native robomimic checkpoints across a supervised multi-GPU queue."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[4]
_CONFIGURED_ARTIFACT_ROOT = os.environ.get("KUAFU_ARTIFACT_ROOT", "").strip()
ARTIFACT_ROOT = (
    Path(_CONFIGURED_ARTIFACT_ROOT).expanduser().resolve()
    if _CONFIGURED_ARTIFACT_ROOT
    else ROOT.parent / "KuafuArtifacts"
)
DEFAULT_TRAIN_RUN = (
    ARTIFACT_ROOT / "standalone" / "robomimic" / "training" / "2026-07-11_robomimic_priors"
)


def _require_external_artifact(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if resolved == ROOT or ROOT in resolved.parents:
        raise ValueError(f"{label} must be outside the source project: {resolved}")
    return resolved


def _artifact_reference(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ARTIFACT_ROOT / path
    return _require_external_artifact(path, label)


def _manifest_input(value: str | Path, label: str) -> Path:
    """Resolve immutable source recipes locally and run artifacts externally."""

    path = Path(value).expanduser()
    if not path.is_absolute() and path.parts and path.parts[0] in {"resources", "src"}:
        return (ROOT / path).resolve()
    return _artifact_reference(path, label)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def _checkpoint_for(train_run: Path, name: str) -> Path:
    matches = sorted((train_run / "models" / name).glob("**/models/model_epoch_2000.pth"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one epoch-2000 checkpoint for {name}, found {len(matches)}")
    return matches[0].resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("manifest.json"))
    parser.add_argument("--train-run", type=Path, default=DEFAULT_TRAIN_RUN)
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y-%m-%d_robomimic_eval"))
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--num-envs", type=int, default=8192)
    parser.add_argument("--duration-sec", type=float, default=180.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    gpus = [int(value) for value in args.gpus.split(",") if value.strip()]
    if not gpus:
        parser.error("--gpus must contain at least one GPU")
    manifest = json.loads(args.manifest.expanduser().resolve().read_text())
    jobs = list(manifest["jobs"])
    run_dir = _require_external_artifact(
        ARTIFACT_ROOT / "standalone" / "robomimic" / "evaluation" / args.run_id,
        "Evaluation run",
    )
    process_logs = run_dir / "process_logs"
    results_dir = run_dir / "results"
    process_logs.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    evaluator = Path(__file__).with_name("eval_policy_env.py").resolve()
    train_run = _artifact_reference(args.train_run, "Training run")

    pending = jobs.copy()
    running: dict[int, dict] = {}
    finished: dict[str, dict] = {}
    failed: dict[str, dict] = {}
    stopping = False
    created_at = _utc_now()

    def request_stop(_signum, _frame) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    def save_state() -> None:
        _write_json(
            run_dir / "launch_state.json",
            {
                "robomimic_only": True,
                "success_source": "current _external_get_dones()",
                "program_source": (
                    "task mpc_config load_program or manifest program_path override"
                ),
                "parameter_source": "manifest collect-and-upload sweep JSON",
                "run_id": args.run_id,
                "created_at": created_at,
                "updated_at": _utc_now(),
                "num_envs": args.num_envs,
                "duration_sec": args.duration_sec,
                "gpus": gpus,
                "pending": [job["name"] for job in pending],
                "running": {
                    str(gpu): {
                        "name": entry["job"]["name"],
                        "task": entry["job"]["task"],
                        "pid": entry["process"].pid,
                        "checkpoint": str(entry["checkpoint"]),
                        "params_path": str(entry["params_path"]),
                        "program_path": str(entry["program_path"]) if entry["program_path"] else None,
                    }
                    for gpu, entry in running.items()
                },
                "finished": finished,
                "failed": failed,
            },
        )

    try:
        while pending or running:
            if stopping:
                break
            for gpu in gpus:
                if gpu in running or not pending:
                    continue
                job = pending.pop(0)
                checkpoint = _checkpoint_for(train_run, job["name"])
                params_path = _manifest_input(job["params_path"], "Parameter JSON")
                if not params_path.is_file():
                    raise FileNotFoundError(f"Parameter JSON not found for {job['name']}: {params_path}")
                program_path = None
                if job.get("program_path"):
                    program_path = _manifest_input(job["program_path"], "SUN Program")
                    if not program_path.is_dir():
                        raise FileNotFoundError(
                            f"Program path not found for {job['name']}: {program_path}"
                        )
                result_path = results_dir / f"{job['name']}.json"
                log_path = process_logs / f"{job['name']}.log"
                command = [
                    sys.executable,
                    str(evaluator),
                    "--task",
                    job["task"],
                    "--checkpoint",
                    str(checkpoint),
                    "--params-path",
                    str(params_path),
                    "--num-envs",
                    str(args.num_envs),
                    "--duration-sec",
                    str(args.duration_sec),
                    "--seed",
                    str(args.seed),
                    "--output",
                    str(result_path),
                    "--device",
                    "cuda:0",
                    "--headless",
                ]
                if program_path is not None:
                    command.extend(["--program-path", str(program_path)])
                environment = os.environ.copy()
                environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
                environment["PYTHONUNBUFFERED"] = "1"
                log_handle = log_path.open("w")
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=environment,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                running[gpu] = {
                    "job": job,
                    "checkpoint": checkpoint,
                    "params_path": params_path,
                    "program_path": program_path,
                    "result_path": result_path,
                    "log_path": log_path,
                    "log_handle": log_handle,
                    "process": process,
                    "started_at": _utc_now(),
                }
            save_state()
            time.sleep(2)
            for gpu, entry in list(running.items()):
                returncode = entry["process"].poll()
                if returncode is None:
                    continue
                entry["log_handle"].close()
                job = entry["job"]
                record = {
                    "gpu": gpu,
                    "returncode": returncode,
                    "started_at": entry["started_at"],
                    "finished_at": _utc_now(),
                    "log": str(entry["log_path"]),
                }
                if returncode == 0 and entry["result_path"].is_file():
                    record["result"] = json.loads(entry["result_path"].read_text())
                    finished[job["name"]] = record
                else:
                    failed[job["name"]] = record
                del running[gpu]
            save_state()
    finally:
        if stopping:
            for entry in running.values():
                try:
                    os.killpg(entry["process"].pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            deadline = time.monotonic() + 10
            while running and time.monotonic() < deadline:
                for gpu, entry in list(running.items()):
                    if entry["process"].poll() is not None:
                        entry["log_handle"].close()
                        del running[gpu]
                time.sleep(0.2)
            for entry in running.values():
                try:
                    os.killpg(entry["process"].pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                entry["log_handle"].close()
        save_state()

    return 1 if stopping or failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
