"""Content identities used at pipeline boundaries (no simulator dependencies)."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from kuafu.paths import PROJECT_ROOT, resolve_artifact_path


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def learned_program_identity(path: str | Path) -> dict[str, str]:
    root = Path(path).expanduser()
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    root = root.resolve()
    program = root / "program"
    files = sorted(p for p in program.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    if not program.is_dir() or not files:
        raise FileNotFoundError(f"Missing or empty SUN Program directory: {program}")
    digest = hashlib.sha256()
    for item in files:
        digest.update(str(item.relative_to(program)).encode() + b"\0")
        digest.update(file_sha256(item).encode() + b"\0")
    return {"program_path": str(root), "program_sha256": digest.hexdigest()}


def active_program_identity() -> dict[str, str]:
    path = os.environ.get("KUAFU_PROGRAM_PATH", "").strip()
    return learned_program_identity(path) if path else {}


def validate_program_metadata(data: dict, expected: dict[str, str] | None = None) -> None:
    identity = active_program_identity() if expected is None else expected
    if identity and any(data.get(key) != value for key, value in identity.items()):
        raise ValueError("Parameters/weights belong to a different or unidentified learned program; regenerate them for the selected program.")


def apply_learned_program(env_cfg: Any, path: str | Path | None = None) -> dict[str, str]:
    """Explicit CLI selection wins over the environment, then the task default."""
    selected = path or os.environ.get("KUAFU_PROGRAM_PATH") or env_cfg.mpc_config.get("load_program")
    if not selected:
        raise ValueError("Evaluation/training requires a learned program; supply --program_path or KUAFU_PROGRAM_PATH.")
    identity = learned_program_identity(selected)
    env_cfg.mpc_config["load_program"] = identity["program_path"]
    print(f"[learned-program] {json.dumps(identity, sort_keys=True)}", flush=True)
    return identity


def resolve_parameters_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute() and candidate.parts and candidate.parts[0] in {"resources", "src"}:
        return (PROJECT_ROOT / candidate).resolve()
    return resolve_artifact_path(candidate, "Residual parameters")


def evaluation_identity(*, checkpoint: str, task: str, program_path: str | None,
                        params_path: str | None, duration_sec: float, num_envs: int,
                        chunk_exec: int, extra: list[str] | None = None) -> str:
    """A cached rate is usable only for these exact inputs and rollout settings."""
    ckpt = resolve_artifact_path(checkpoint, "BC checkpoint")
    params = resolve_parameters_path(params_path) if params_path else None
    payload = {
        "checkpoint": str(ckpt), "checkpoint_sha256": file_sha256(ckpt), "task": task,
        "program": learned_program_identity(program_path) if program_path else {},
        "params": str(params) if params else "", "params_sha256": file_sha256(params) if params else "",
        "duration_sec": float(duration_sec), "num_envs": int(num_envs), "chunk_exec": int(chunk_exec),
        "extra": extra or [],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
