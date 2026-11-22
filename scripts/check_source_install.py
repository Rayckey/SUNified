"""Install a source archive away from the checkout and verify public resources."""

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
import venv


def check_public_paths(paths):
    """Reject local artifacts even when a developer force-adds them to Git."""
    local_dirs = {
        ".git", ".hydra", ".pytest_cache", ".ruff_cache", ".mypy_cache", "__pycache__",
        "buffer", "outputs", "logs", "reports", "test-results", "validation", "analysis",
        "wandb", "mlruns",
    }
    local_names = {
        ".DS_Store", ".coverage", ".asset_hash", "creds.sh",
        "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY", "GEMINI_API_KEY",
    }
    suffixes = (".pyc", ".pyo", ".orig", ".rej", ".bak", ".tmp", ".swp", ".log", ".metadata.pkl", "~")
    unexpected = []
    for name in paths:
        path = PurePosixPath(name)
        if (local_dirs.intersection(path.parts)
                or path.name in local_names or path.name.endswith(suffixes)
                or (path.name.startswith(".env") and path.name != ".env.example")
                or ("objects" in path.parts and (
                    {"renders", "screenshots"}.intersection(path.parts) or path.name == "visual_check.html"))):
            unexpected.append(name)
    if unexpected:
        raise ValueError("Public source contains local artifacts:\n" + "\n".join(unexpected))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    extracted = output / "source"
    with tarfile.open(args.archive) as archive:
        check_public_paths(archive.getnames())
        archive.extractall(extracted, filter="data")
    roots = list(extracted.glob("*/pyproject.toml"))
    if len(roots) != 1:
        raise ValueError("Expected one Kuafu project in the source archive")
    source = roots[0].parent
    venv.create(output / "venv", with_pip=True)
    python = output / "venv/bin/python"
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("KUAFU_ROOT", None)
    env["KUAFU_ARTIFACT_ROOT"] = str(output / "artifacts")
    commands = [
        [str(python), "-m", "pip", "install", "-e", f"{source}[agents,isaac,tests]"],
        [str(python), "-m", "pip", "check"],
        [str(python), "-m", "kuafu", "agent", "tools"],
        [str(python), "-m", "kuafu", "tasks", "list"],
        [str(python), "-m", "pytest", "-q", str(source / "tests/test_source_integrity.py"),
         str(source / "tests/test_task_catalog.py"), str(source / "tests/test_example_artifacts.py"),
         str(source / "tests/test_ur5_assets.py")],
    ]
    for command in commands:
        subprocess.run(command, cwd=output, env=env, check=True)
    required = ["assets/prompts", "assets/skills", "assets/tasks", "assets/configs", "assets/objects",
                "assets/robots", "resources/learned_programs", "examples/04_video_to_policy/run.py",
                "examples/01_video_to_rollout/input/IMG_0028.mov", "LICENSES", "docs/INSTALL.md",
                "assets/robots/ur5_robotiq/ur5_robotiq.usd",
                "assets/robots/ur5_robotiq/ur5_controller.urdf",
                "assets/robots/ur5_robotiq/source/arm.urdf"]
    for name in required:
        if not (source / name).exists():
            raise FileNotFoundError(f"Source archive omitted {name}")
    report = {"source_install_passed": True, "commands": commands, "bundled_resources": required}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
