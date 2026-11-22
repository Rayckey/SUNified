"""Content identities and atomic experiment artifacts."""

import hashlib
import json
import os
from pathlib import Path
import subprocess


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def zarr_digest(path):
    root = Path(path)
    return json_digest({str(file.relative_to(root)): file_digest(file)
                        for group in ("data", "meta") for file in sorted((root / group).rglob("*"))
                        if file.is_file()})


def git_revision(path):
    def git(*args):
        result = subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    untracked = (git("ls-files", "--others", "--exclude-standard") or "").splitlines()
    files = {name: file_digest(Path(path) / name) for name in untracked if (Path(path) / name).is_file()}
    return {"path": str(Path(path).resolve()), "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain")),
            "working_changes_sha256": json_digest({"diff": git("diff", "HEAD", "--binary"), "untracked": files})}
