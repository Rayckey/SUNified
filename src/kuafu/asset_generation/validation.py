"""Run asset physics checks in a separate Isaac Python process."""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from kuafu.paths import PROJECT_ROOT, SOURCE_ROOT, resolve_artifact_path

from .articulation import verify_articulation_report
from .io import digest, write_json


def validate_asset(asset, *, report=None, python=None, steps=240, probe_radius=None, cavity_floor_z=None):
    asset = resolve_artifact_path(asset, "Generated simulation asset")
    report = resolve_artifact_path(report or asset.parent / "physics_smoke.json", "Asset physics report")
    if not asset.is_file():
        raise FileNotFoundError(asset)
    if steps < 240:
        raise ValueError("Asset validation requires at least 240 physics steps")
    if (probe_radius is None) != (cavity_floor_z is None):
        raise ValueError("Supply both probe_radius and cavity_floor_z")
    if probe_radius is not None and (probe_radius <= 0 or cavity_floor_z < 0):
        raise ValueError("Probe radius must be positive and cavity floor nonnegative")
    command = [
        python or os.environ.get("KUAFU_ISAAC_PYTHON") or sys.executable,
        str(PROJECT_ROOT / "scripts/assets/smoke_isaac.py"),
        "--asset",
        str(asset),
        "--steps",
        str(steps),
        "--headless",
    ]
    if probe_radius is not None:
        command += ["--probe-radius", str(probe_radius), "--cavity-floor-z", str(cavity_floor_z)]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SOURCE_ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    report.parent.mkdir(parents=True, exist_ok=True)
    # A fresh report prevents a simulator shutdown from reusing an older pass.
    with tempfile.TemporaryDirectory(prefix=".asset-check-", dir=report.parent) as temporary:
        pending = Path(temporary) / "physics_smoke.json"
        result = subprocess.run([*command, "--report", str(pending)], env=env, check=False)
        if result.returncode:
            if pending.is_file():
                os.replace(pending, report)
            raise RuntimeError(f"Isaac asset validation failed with exit code {result.returncode}: {report}")
        if not pending.is_file():
            raise RuntimeError("Isaac exited without writing a new physics report")
        payload = json.loads(pending.read_text())
        if payload.get("passed") is not True or Path(payload.get("asset", "")).resolve() != asset:
            raise ValueError("Isaac did not report a passing check for the requested asset")
        if payload.get("steps", 0) < steps or abs(payload.get("physics_dt", 0) - 1 / 120) > 1e-9:
            raise ValueError("Physics report does not cover the requested simulation duration")
        layers = payload.get("asset_layers", [])
        if asset not in {Path(layer["path"]).resolve() for layer in layers}:
            raise ValueError("Physics report does not identify the tested USD asset")
        for layer in layers:
            if digest(Path(layer["path"])) != layer["sha256"]:
                raise ValueError("USD asset changed after its physics check")
        verify_articulation_report(asset.parent, payload)
        if probe_radius is not None:
            probe = payload.get("cavity_probe", {})
            if (
                probe.get("passed") is not True
                or probe.get("radius_m") != probe_radius
                or probe.get("expected_floor_z_m") != cavity_floor_z
            ):
                raise ValueError("Physics report does not cover the requested cavity probe")
        os.replace(pending, report)
    metadata_path = asset.parent / "asset.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text())
        metadata["isaac_physics_tested"] = True
        metadata["physics_validation"] = {
            "report": os.path.relpath(report, asset.parent),
            "scope": payload.get("scope", "See the physics report for validation scope"),
            "asset_layers": payload["asset_layers"],
        }
        write_json(metadata_path, metadata)
    return report
