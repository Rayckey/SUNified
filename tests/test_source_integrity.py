"""Catch migration errors without importing Isaac or launching model requests.

An import can sit behind an optional training branch and evade the normal unit
suite. Resolve local module names statically so those branches remain usable.
"""

import ast
import os
import re
import subprocess
import sys
from pathlib import Path
from runpy import run_path

ROOT = Path(__file__).resolve().parents[1]


def test_tracked_files_exclude_local_artifacts():
    if not (ROOT / ".git").exists():
        return  # Source archives are checked before extraction by the install smoke test.
    paths = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT, text=True).split("\0")
    check = run_path(str(ROOT / "scripts/check_source_install.py"))["check_public_paths"]
    check(path for path in paths if path)


def test_internal_absolute_imports_resolve():
    missing = []
    for directory in (ROOT / "src", ROOT / "scripts"):
        for path in directory.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
                    names = [node.module]
                for name in names:
                    if not name.startswith("kuafu."):
                        continue
                    target = ROOT / "src" / Path(*name.split("."))
                    if not target.with_suffix(".py").is_file() and not target.is_dir():
                        missing.append(f"{path.relative_to(ROOT)}:{node.lineno}: {name}")
    assert not missing, "Missing project modules:\n" + "\n".join(missing)


def test_public_documentation_links_resolve():
    missing = []
    pages = [ROOT / name for name in ("README.md", "CONTRIBUTING.md", "AGENTS.md")]
    pages.extend((ROOT / "docs").glob("*.md"))
    for directory in ("assets", "examples", "src"):
        pages.extend((ROOT / directory).rglob("*.md"))
    for page in pages:
        text = re.sub(r"```.*?```", "", page.read_text(), flags=re.DOTALL)
        for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", text):
            if "://" in target or target.startswith(("#", "mailto:")):
                continue
            target = target.split("#", 1)[0]
            if target:
                resolved = (page.parent / target).resolve()
                if not resolved.is_relative_to(ROOT) or not resolved.exists():
                    missing.append(f"{page.relative_to(ROOT)}: {target}")
    assert not missing, "Broken documentation links:\n" + "\n".join(missing)


def test_cli_help_from_outside_checkout(tmp_path):
    env = {**os.environ, "KUAFU_PYTHON": sys.executable}
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [str(ROOT / "bin/kuafu"), "--help"],
        cwd=tmp_path, env=env, text=True, capture_output=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("usage: kuafu ")
    assert "verify-drawer" in result.stdout
