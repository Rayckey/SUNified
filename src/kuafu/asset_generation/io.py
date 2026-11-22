import hashlib
import json
import os
import tempfile
from pathlib import Path

from pydantic import BaseModel

from kuafu.paths import resolve_artifact_path


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path: Path, value) -> None:
    """Atomic checkpoints prevent a stopped run from leaving half a JSON file."""
    path = resolve_artifact_path(path, "Asset-generation output")
    data = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(data, stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def fresh_directory(path: Path) -> Path:
    path = resolve_artifact_path(path, "Asset-generation output")
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f"Output must be a new or empty directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def local_file(workspace: Path, relative: str) -> Path:
    path = (workspace / relative).resolve()
    if not path.is_relative_to(workspace.resolve()) or not path.is_file():
        raise ValueError(f"Missing or unsafe workspace file: {relative}")
    return path
