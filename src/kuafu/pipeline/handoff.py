"""Durable, inspectable handoffs for an agent's complete task run."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import uuid

from kuafu.paths import require_external_artifact_path


STEPS = (
    "prepare-video", "build-asset", "inspect", "form-program", "verify-program",
    "collect", "train-prior", "eval-prior", "train-residual",
)
TOOL_STEPS = {
    "kuafu_prepare_video": "prepare-video", "kuafu_build_asset": "build-asset",
    "kuafu_inspect_drawer": "inspect", "kuafu_submit_program": "form-program",
    "kuafu_verify_program": "verify-program",
    "kuafu_collect_drawer": "collect", "kuafu_train_drawer_policy": "train-prior",
    "kuafu_evaluate_drawer_policy": "eval-prior",
}
LEARNING_STEPS = STEPS[5:]


def _now():
    return datetime.now(timezone.utc).isoformat()


def _write(path, data):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _read(path):
    return json.loads(Path(path).read_text())


def _hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class HandoffRun:
    """One run owns inputs, step attempts, jobs, and the artifact index.

    Every retry creates a new attempt. A filesystem lock serializes index updates
    across independent MCP clients; expensive tool calls run outside that lock.
    """

    def __init__(self, artifact_root, run_id):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", run_id):
            raise ValueError("run_id must be 1–64 letters, numbers, hyphens or underscores")
        base = require_external_artifact_path(artifact_root)
        self.root = base / "runs" / run_id
        if not self.root.resolve().is_relative_to(base.resolve()):
            raise ValueError("Run directory must stay inside the artifact root")
        self.run_id = run_id

    @classmethod
    def create(cls, artifact_root, *, task, description="", videos=(), run_id=None):
        run = cls(artifact_root, run_id or uuid.uuid4().hex)
        sources = [Path(p).expanduser().resolve(strict=True) for p in videos]
        if any(not p.is_file() for p in sources):
            raise ValueError("Each video input must be a file")
        run.root.mkdir(parents=True, exist_ok=False)
        (run.root / "inputs").mkdir()
        records = []
        for index, source in enumerate(sources):
            target = run.root / "inputs" / f"{index:02d}_{source.name}"
            shutil.copy2(source, target)
            records.append({"path": str(target.relative_to(run.root)), "sha256": _hash(target),
                            "source": str(source), "bytes": target.stat().st_size})
        _write(run.root / "run.json", {"schema_version": 1, "run_id": run.run_id,
                                       "task": task, "description": description,
                                       "created_at": _now(), "inputs": records})
        _write(run.root / "index.json", {"attempts": [], "ids": {}})
        return run

    @contextmanager
    def _locked(self):
        if not (self.root / "run.json").is_file():
            raise ValueError(f"Unknown run: {self.run_id}")
        with (self.root / ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield

    def begin(self, tool, arguments):
        from kuafu.agents.jobs import process_token

        step = arguments["stage"] if tool == "kuafu_start_stage" else TOOL_STEPS[tool]
        if step not in STEPS:
            raise ValueError(f"Unsupported handoff step: {step}")
        with self._locked():
            index = _read(self.root / "index.json")
            for item in index["attempts"]:
                receipt = self._refresh(self.root / item)
                if receipt["status"] in {"running", "queued"}:
                    if receipt["step"] == step:
                        raise ValueError(f"{step} is already running; inspect its job before retrying")
                    if step in LEARNING_STEPS and receipt["step"] in LEARNING_STEPS:
                        raise ValueError(f"{receipt['step']} is still running; finish its handoff before starting {step}")
            attempt = self.root / "steps" / f"{STEPS.index(step) + 1:02d}-{step}" / uuid.uuid4().hex
            attempt.mkdir(parents=True)
            _write(attempt / "request.json", {"tool": tool, "arguments": arguments})
            _write(attempt / "receipt.json", {
                "step": step, "attempt_id": attempt.name, "status": "running", "started_at": _now(),
                "owner_pid": os.getpid(), "owner_token": process_token(os.getpid()),
                "input_request": "request.json", "output_response": "response.json", "log": "tool.log",
            })
            index["attempts"].append(str(attempt.relative_to(self.root)))
            _write(self.root / "index.json", index)
        return attempt

    def allocate(self, attempt, category):
        path = attempt / category / uuid.uuid4().hex
        path.mkdir(parents=True)
        with self._locked():
            index = _read(self.root / "index.json")
            index["ids"][path.name] = {"category": category, "path": str(path.relative_to(self.root))}
            _write(self.root / "index.json", index)
        return path

    def lookup(self, category, identifier):
        with self._locked():
            item = _read(self.root / "index.json")["ids"].get(identifier)
        if item is None or item["category"] != category:
            raise ValueError(f"Unknown {category} ID in run {self.run_id}: {identifier}")
        path = (self.root / item["path"]).resolve()
        if not path.is_relative_to(self.root.resolve()) or not path.is_dir():
            raise ValueError("Invalid artifact index entry")
        return path

    def finish(self, attempt, response):
        with self._locked():
            _write(attempt / "response.json", response)
            receipt = _read(attempt / "receipt.json")
            result = response.get("result", {})
            receipt.update(status="complete" if response["ok"] else "failed", finished_at=_now())
            if result.get("job_id"):
                receipt.update(status="queued", job_id=result["job_id"], job_directory=result["directory"])
            if not response["ok"]:
                receipt["error"] = response["error"]
            _write(attempt / "receipt.json", receipt)
            if receipt["status"] not in {"queued", "running"}:
                self._inventory(attempt)

    def attach_job(self, attempt, directory):
        """Record the job before spawning it so a disconnect cannot orphan it."""
        with self._locked():
            receipt = _read(attempt / "receipt.json")
            receipt.update(job_id=directory.name, job_directory=str(directory))
            _write(attempt / "receipt.json", receipt)

    def prepare_learning(self, attempt, config):
        """Inherit completed upstream artifacts into a new learning revision."""
        from kuafu.pipeline.state import StateStore

        step = _read(attempt / "receipt.json")["step"]
        root = self.root / "learning" / attempt.name
        root.mkdir(parents=True, exist_ok=False)
        state = StateStore(root / "pipeline_state.json", config.task_slug)
        latest = self.status(verify_outputs=True)["steps"]
        for parent in LEARNING_STEPS[:LEARNING_STEPS.index(step)]:
            record = latest.get(parent)
            if record is None:
                continue
            if record["status"] != "complete" or record.get("stale"):
                raise ValueError(f"Complete the current {parent} revision before starting {step}")
            stage = record.get("job_result", {}).get("stage_state", {})
            if stage.get("status") != "complete":
                raise ValueError(f"{parent} has no completed learning handoff")
            state.data["stages"][parent] = stage
            state.artifacts.update(stage.get("artifacts", {}))

        handoff = {"run_root": str(self.root), "attempt": attempt.name}
        if step in {"eval-prior", "train-residual"}:
            configured = config.section("eval").get("csv", "")
            source = (config.resolve_artifact_path(configured, field="[eval].csv") if configured
                      else state.artifacts.get("prior_csv"))
            if not source:
                raise ValueError("Supply [eval].csv or complete train-prior in this run")
            # Evaluation updates the CSV in place. Copy it so the upstream
            # training/evaluation revision retains its original records.
            target = root / "inputs" / "prior.csv"
            target.parent.mkdir()
            shutil.copy2(source, target)
            state.artifacts["prior_csv"] = str(target)
            handoff["prior_csv"] = str(target)
        state.save()
        return handoff

    def _inventory(self, attempt):
        files = {}
        for path in sorted(attempt.rglob("*")):
            if path.is_file() and path.name not in {"artifacts.json", "receipt.json", "tool.log"}:
                files[str(path.relative_to(attempt))] = {"bytes": path.stat().st_size, "sha256": _hash(path)}
        _write(attempt / "artifacts.json", files)

    def _refresh(self, attempt):
        from kuafu.agents import jobs

        receipt = _read(attempt / "receipt.json")
        if receipt["status"] not in {"queued", "running", "interrupted"}:
            return receipt
        if receipt.get("job_directory"):
            try:
                state = jobs.status(Path(receipt["job_directory"]))
            except FileNotFoundError:
                if jobs.process_token(receipt["owner_pid"]) == receipt["owner_token"]:
                    return receipt
                state = {"status": "interrupted", "error": "Agent ended while launching the job; inspect its directory"}
            status = state["status"]
            receipt["status"] = "complete" if status == "succeeded" else status
            receipt["job_result"] = state
            if status not in {"queued", "running", "interrupted"} and not (attempt / "response.json").exists():
                _write(attempt / "response.json", {
                    "ok": True, "recovered": True,
                    "result": {"job_id": receipt["job_id"], "directory": receipt["job_directory"], "status": status},
                })
        elif jobs.process_token(receipt["owner_pid"]) != receipt["owner_token"]:
            receipt.update(status="interrupted", error="Agent process ended before this step returned")
        if receipt["status"] not in {"queued", "running"}:
            receipt["finished_at"] = _now()
            self._inventory(attempt)
        _write(attempt / "receipt.json", receipt)
        return receipt

    def status(self, *, verify_outputs=False):
        with self._locked():
            manifest = _read(self.root / "run.json")
            index = _read(self.root / "index.json")
            attempts = [{**self._refresh(self.root / item), "directory": str(self.root / item)}
                        for item in index["attempts"]]
        latest = {item["step"]: item for item in attempts}
        positions = {item["step"]: n for n, item in enumerate(attempts)}
        for step in STEPS:
            if step not in latest:
                continue
            item = latest[step]
            if verify_outputs and item["status"] == "complete":
                directory = Path(item["directory"])
                inventory = _read(directory / "artifacts.json")
                changed = [name for name, metadata in inventory.items()
                           if not (directory / name).is_file() or _hash(directory / name) != metadata["sha256"]]
                learning_files = item.get("job_result", {}).get("stage_state", {}).get("files", {})
                changed.extend(name for name, checksum in learning_files.items()
                               if not Path(name).is_file() or _hash(name) != checksum)
                if changed:
                    item.update(stale=True, stale_reason="Recorded output files changed or are missing", changed_files=changed)
            upstream = STEPS[:STEPS.index(step)]
            if any(positions.get(parent, -1) > positions[step] or latest.get(parent, {}).get("stale") for parent in upstream):
                item["stale"] = True
                item["stale_reason"] = "An upstream step has a newer attempt; continue from that revision"
        return {"run_id": self.run_id, "directory": str(self.root), "manifest": manifest,
                "steps": latest, "attempts": attempts,
                "resume": "Pass this run_id to agent tools. Read the latest step's request, response and logs; "
                          "retry with kuafu_retry_step or submit revised inputs. Earlier attempts are retained."}

    def retry_request(self, attempt_id):
        state = self.status()
        item = next((x for x in state["attempts"] if x["attempt_id"] == attempt_id), None)
        if item is None:
            raise ValueError(f"Unknown attempt: {attempt_id}")
        if item["status"] in {"running", "queued"}:
            raise ValueError("Wait for or cancel the running job before retrying")
        request = _read(Path(item["directory"]) / "request.json")
        if request["tool"] == "kuafu_start_stage" and item.get("job_directory"):
            saved = Path(item["job_directory"]) / "pipeline.toml"
            if saved.is_file():
                request["arguments"]["config"] = str(saved)
        return request
