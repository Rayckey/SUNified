"""TOML configuration and path resolution for the unified workflow."""

from __future__ import annotations

import os
import re
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kuafu.paths import DEFAULT_EXTERNAL_ARTIFACT_ROOT, PROJECT_ROOT


class ConfigError(ValueError):
    """Raised when a pipeline configuration is missing or inconsistent."""


def project_root() -> Path:
    return PROJECT_ROOT


@dataclass(frozen=True)
class PipelineConfig:
    """Thin, validated view over a pipeline TOML document.

    Ordinary source paths resolve from the project root. Relative generated-artifact
    paths resolve beneath ``artifact_root`` so copied task configs cannot write into
    the source tree.
    """

    path: Path
    data: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "PipelineConfig":
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise ConfigError(f"Configuration does not exist: {source}")
        with source.open("rb") as handle:
            data = tomllib.load(handle)
        config = cls(path=source, data=data)
        config.validate()
        return config

    def validate(self) -> None:
        version = self.data.get("schema_version")
        if version != 1:
            raise ConfigError(f"schema_version must be 1, got {version!r}")
        for section in ("workspace", "task", "learn_programs", "collect", "prior", "eval", "residual"):
            if not isinstance(self.data.get(section), dict):
                raise ConfigError(f"Missing [{section}] section")
        for key in ("slug", "task_name", "robot_name", "robot_id"):
            if not str(self.section("task").get(key, "")).strip():
                raise ConfigError(f"[task].{key} is required")
        gpus = self.gpus
        if not gpus or any(gpu < 0 for gpu in gpus):
            raise ConfigError("[workspace].gpus must contain non-negative GPU ids")
        if int(self.section("collect").get("successful_demos_per_program", 0)) <= 0:
            raise ConfigError("[collect].successful_demos_per_program must be positive")
        if int(self.section("collect").get("workers_per_gpu_per_program", 0)) <= 0:
            raise ConfigError("[collect].workers_per_gpu_per_program must be positive")
        _ = self.artifact_root

    def section(self, name: str) -> dict[str, Any]:
        value = self.data.get(name)
        if not isinstance(value, dict):
            raise ConfigError(f"Missing [{name}] section")
        return value

    def resolve_path(self, value: str | Path, *, allow_empty: bool = False) -> Path:
        text = os.path.expandvars(str(value)).strip()
        if not text:
            if allow_empty:
                return Path()
            raise ConfigError("Expected a non-empty path")
        path = Path(text).expanduser()
        if not path.is_absolute():
            path = project_root() / path
        return path.resolve()

    def require_external_artifact_path(self, path: Path, *, field: str) -> Path:
        """Reject generated-output paths inside the source project.

        Learned programs are deliberately stored under resources/learned_programs and
        do not use this method. Every other generated artifact must live outside
        PROJECT_ROOT.
        """

        resolved = path.expanduser().resolve()
        root = project_root().resolve()
        if resolved == root or root in resolved.parents:
            raise ConfigError(
                f"{field} must be outside the source project ({root}); got {resolved}. "
                "Set KUAFU_ARTIFACT_ROOT or use an external absolute path."
            )
        return resolved

    def resolve_artifact_path(self, value: str | Path, *, field: str) -> Path:
        text = os.path.expandvars(str(value)).strip()
        if not text:
            raise ConfigError(f"{field} must be a non-empty artifact path")
        path = Path(text).expanduser()
        if not path.is_absolute():
            path = self.artifact_root / path
        return self.require_external_artifact_path(path, field=field)

    @property
    def task_slug(self) -> str:
        return str(self.section("task")["slug"]).strip()

    @property
    def task_name(self) -> str:
        return str(self.section("task")["task_name"]).strip()

    @property
    def robot_name(self) -> str:
        return str(self.section("task")["robot_name"]).strip()

    @property
    def robot_id(self) -> str:
        return str(self.section("task")["robot_id"]).strip()

    @property
    def prior_task_slug(self) -> str:
        return str(self.section("prior").get("task_slug", self.task_slug)).strip()

    @property
    def gpus(self) -> tuple[int, ...]:
        raw = self.section("workspace").get("gpus", [0])
        if not isinstance(raw, list):
            raise ConfigError("[workspace].gpus must be a TOML array")
        return tuple(int(item) for item in raw)

    @property
    def python(self) -> str:
        configured = str(self.section("workspace").get("python", "")).strip()
        return os.path.expandvars(os.path.expanduser(configured)) if configured else sys.executable

    @property
    def isaaclab_root(self) -> Path:
        override = os.environ.get("ISAACLAB_ROOT", "").strip()
        value = override or str(self.section("workspace").get("isaaclab_root", "..")).strip()
        return self.resolve_path(value)

    @property
    def artifact_root(self) -> Path:
        override = os.environ.get("KUAFU_ARTIFACT_ROOT", "").strip()
        configured = str(
            self.section("workspace").get("artifact_root", str(DEFAULT_EXTERNAL_ARTIFACT_ROOT))
        ).strip()
        value = override or configured or str(DEFAULT_EXTERNAL_ARTIFACT_ROOT)
        return self.require_external_artifact_path(
            self.resolve_path(value), field="[workspace].artifact_root"
        )

    @property
    def task_artifact_root(self) -> Path:
        handoff = self.data.get("handoff", {})
        if handoff.get("run_root"):
            root = self.require_external_artifact_path(Path(handoff["run_root"]), field="[handoff].run_root")
            if not (root / "run.json").is_file():
                raise ConfigError("[handoff].run_root must identify a run created by kuafu_create_run")
            attempt = handoff.get("attempt", "")
            if attempt and not re.fullmatch(r"[0-9a-f]{32}", attempt):
                raise ConfigError("[handoff].attempt must be a tool-created attempt ID")
            output = (root / "learning" / attempt).resolve()
            if not output.is_relative_to(root):
                raise ConfigError("Handoff outputs must remain within their run")
            return output
        return self.artifact_root / self.task_slug

    def string_list(self, section: str, key: str, default: tuple[str, ...] = ()) -> tuple[str, ...]:
        raw = self.section(section).get(key, list(default))
        if not isinstance(raw, list):
            raise ConfigError(f"[{section}].{key} must be a TOML array")
        return tuple(str(item) for item in raw)

    def int_list(self, section: str, key: str, default: tuple[int, ...] = ()) -> tuple[int, ...]:
        return tuple(int(item) for item in self.string_list(section, key, tuple(str(v) for v in default)))
