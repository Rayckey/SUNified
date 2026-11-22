"""Startup dependencies for the camera-disabled can formulation runner."""

import ast
from pathlib import Path


def test_can_runner_does_not_import_unused_camera_extension():
    runner = (
        Path(__file__).resolve().parents[1]
        / "src/kuafu/programs/can_learn_programs.py"
    )
    tree = ast.parse(runner.read_text(encoding="utf-8"))
    imported_modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.append(node.module)
    assert not any(name.startswith("omni.replicator") for name in imported_modules)
