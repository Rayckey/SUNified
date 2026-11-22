"""Subprocess environment and command execution helpers."""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path
from typing import Mapping, TextIO

from kuafu.paths import SOURCE_ROOT

from .config import PipelineConfig, project_root


def command_text(command: list[str]) -> str:
    return shlex.join(str(part) for part in command)


def pipeline_environment(config: PipelineConfig, extra: Mapping[str, str] | None = None) -> dict[str, str]:
    root = project_root()
    env = os.environ.copy()
    # IsaacLab's shell wrapper initializes tab stops even with redirected output.
    # Its `tabs` call fails under the noninteractive TERM=dumb default.
    if env.get("TERM", "") in {"", "dumb"}:
        env["TERM"] = "xterm"
    python_paths = [SOURCE_ROOT, root]
    inherited = env.get("PYTHONPATH", "").strip()
    if inherited:
        python_paths.append(Path(inherited))
    env["PYTHONPATH"] = os.pathsep.join(str(path) for path in python_paths)
    env["KUAFU_PROJECT_ROOT"] = str(root)
    env["KUAFU_ARTIFACT_ROOT"] = str(config.artifact_root)
    env["KUAFU_TASK_ARTIFACT_ROOT"] = str(config.task_artifact_root)
    env["KUAFU_PROGRAM_OUTPUT_ROOT"] = str(config.task_artifact_root / "programs")
    env["KUAFU_RESIDUAL_LOG_ROOT"] = str(config.task_artifact_root / "residual" / "runs")
    env["ISAACLAB_ROOT"] = str(config.isaaclab_root)
    if extra:
        env.update({str(key): str(value) for key, value in extra.items()})
    return env


class CommandRunner:
    def __init__(self, config: PipelineConfig, *, dry_run: bool = False):
        self.config = config
        self.dry_run = dry_run

    def run(
        self,
        label: str,
        command: list[str],
        *,
        extra_env: Mapping[str, str] | None = None,
        stdout: TextIO | None = None,
    ) -> subprocess.CompletedProcess[str] | None:
        print(f"[{label}] {command_text(command)}", flush=True)
        if self.dry_run:
            return None
        return subprocess.run(
            command,
            cwd=project_root(),
            env=pipeline_environment(self.config, extra_env),
            stdout=stdout,
            stderr=subprocess.STDOUT if stdout is not None else None,
            text=True,
            check=True,
        )
