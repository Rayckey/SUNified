"""Atomic state file used to hand artifacts between long-running stages."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class StateStore:
    def __init__(self, path: Path, task_slug: str):
        self.path = path
        self.data: dict[str, Any] = {
            "schema_version": 1,
            "task_slug": task_slug,
            "created_at_utc": utc_now(),
            "updated_at_utc": utc_now(),
            "artifacts": {},
            "stages": {},
        }
        if path.is_file():
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise ValueError(f"Invalid pipeline state: {path}")
            self.data = loaded

    @property
    def artifacts(self) -> dict[str, Any]:
        value = self.data.setdefault("artifacts", {})
        if not isinstance(value, dict):
            raise ValueError(f"Invalid artifacts mapping in {self.path}")
        return value

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data["updated_at_utc"] = utc_now()
        temporary = self.path.with_name(f".{self.path.name}.tmp.{os.getpid()}")
        temporary.write_text(json.dumps(self.data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, self.path)

    def stage_started(self, name: str, command: list[str] | None = None) -> None:
        stages = self.data.setdefault("stages", {})
        stages[name] = {
            "status": "running",
            "started_at_utc": utc_now(),
            "command": command or [],
        }
        self.save()

    def stage_finished(self, name: str, *, artifacts: dict[str, Any] | None = None,
                       status: str = "complete", inputs: str = "", files: dict[str, str] | None = None) -> None:
        stages = self.data.setdefault("stages", {})
        current = stages.setdefault(name, {})
        current.update({"status": status, "updated_at_utc": utc_now(), "inputs": inputs, "files": files or {}})
        if status == "complete":
            current["completed_at_utc"] = utc_now()
        if artifacts:
            self.artifacts.update(artifacts)
            current["artifacts"] = artifacts
        self.save()

    def stage_failed(self, name: str, error: BaseException) -> None:
        stages = self.data.setdefault("stages", {})
        current = stages.setdefault(name, {})
        current.update(
            {
                "status": "failed",
                "failed_at_utc": utc_now(),
                "error": f"{type(error).__name__}: {error}",
            }
        )
        self.save()

    def stage_complete(self, name: str) -> bool:
        stages = self.data.get("stages", {})
        return isinstance(stages, dict) and stages.get(name, {}).get("status") == "complete"
