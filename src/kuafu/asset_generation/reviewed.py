"""Replay reviewed scene manifests and publish only hash-matched physics results.

This module rebuilds saved geometry; it does not infer an object from a new video.
"""

import json
import re
import zipfile
from pathlib import Path
from typing import Literal

from pydantic import Field

from .articulation import verify_articulation_report
from .export import export_asset
from .geometry import build_asset
from .io import digest, fresh_directory, local_file, resolve_artifact_path, write_json
from .schema import PhysicsConfig, PreparedInput, SceneSpec, StrictModel


class Probe(StrictModel):
    radius_m: float = Field(gt=0)
    floor_z_m: float = Field(ge=0)


class ReviewedJob(StrictModel):
    workspace: str
    scene: str
    physics: PhysicsConfig
    review: str
    probe: Probe | None = None


class ReviewedBatch(StrictModel):
    version: Literal[1]
    source_mode: Literal["interactive_assistant"]
    jobs: list[ReviewedJob] = Field(min_length=1, max_length=32)


def _jobs(manifest: Path, output_name: str):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", output_name):
        raise ValueError("output_name must be a simple directory name")
    base = manifest.resolve().parent
    batch = ReviewedBatch.model_validate_json(manifest.read_text())
    found, result = set(), []
    for job in batch.jobs:
        workspace = (base / job.workspace).resolve()
        resolve_artifact_path(workspace, "Reviewed asset workspace")
        if not workspace.is_relative_to(base) or not workspace.is_dir() or workspace in found:
            raise ValueError("Workspaces must be unique existing directories within the manifest directory")
        if (workspace / output_name).is_symlink():
            raise ValueError("Output directories must not be symbolic links")
        found.add(workspace)
        result.append((job, workspace, workspace / output_name))
    return result


def build_reviewed(manifest: Path, output_name: str = "final") -> list[Path]:
    jobs = _jobs(manifest, output_name)
    # Check every source and destination before generating any output.
    prepared = []
    for job, workspace, out in jobs:
        if out.exists() and (not out.is_dir() or any(out.iterdir())):
            raise ValueError(f"Output must be new or empty: {out}")
        inputs = PreparedInput.model_validate_json(local_file(workspace, "input.json").read_text())
        for frame in inputs.frames:
            if digest(local_file(workspace, frame.path)) != frame.sha256:
                raise ValueError(f"Prepared reference frame changed: {frame.path}")
        source = local_file(workspace, job.scene)
        scene = SceneSpec.model_validate_json(source.read_text())
        prepared.append((job, workspace, out, inputs, source, scene, build_asset(scene, inputs.scale)))
    for job, workspace, out, inputs, source, scene, asset in prepared:
        metadata = export_asset(asset, scene, fresh_directory(out), job.physics, inputs.scale)
        provenance = {
            "execution": "deterministic_replay_of_reviewed_scene",
            "source_reconstruction_mode": "interactive_assistant",
            "source_video": inputs.source_video,
            "source_sha256": inputs.source_sha256,
            "selected_frames": len(inputs.frames),
            "scene_source": str(source),
            "scene_sha256": digest(source),
            "standalone_model_api_calls": 0,
            "metric_calibration": inputs.scale is not None,
            "review": job.review,
        }
        metadata["reconstruction"] = provenance
        write_json(out / "asset.json", metadata)
        write_json(out / "reconstruction.json", provenance)
        write_json(out / "build_recipe.json", job)
        write_json(out / "run.json", {"status": "exported_pending_physics", **provenance})
    return [out for _, _, out in jobs]


def verify_physics_report(out: Path, probe: Probe | None = None) -> dict:
    out = out.resolve()
    report = json.loads(local_file(out, "physics_smoke.json").read_text())
    if (
        report.get("passed") is not True
        or Path(report.get("asset", "")).resolve() != out / "asset_dynamic.usda"
    ):
        raise ValueError("Physics report must pass for this exact dynamic asset")
    if report.get("steps", 0) < 240 or abs(report.get("physics_dt", 0) - 1 / 120) > 1e-9:
        raise ValueError("Expected at least 240 physics steps at 120 Hz")
    layers = report.get("asset_layers", [])
    checked = set()
    for layer in layers:
        path = Path(layer["path"]).resolve()
        if not path.is_relative_to(out) or not path.is_file() or digest(path) != layer["sha256"]:
            raise ValueError("Physics result is stale or references an unexpected USD layer")
        checked.add(path.name)
    if not {"asset_dynamic.usda", "asset_geometry.usda"}.issubset(checked):
        raise ValueError("Physics report must include hashes of the dynamic and geometry layers")
    verify_articulation_report(out, report)
    if probe is not None:
        measured = report.get("cavity_probe", {})
        if measured.get("passed") is not True:
            raise ValueError("Required cavity/shelf probe did not pass")
        if (
            measured.get("radius_m") != probe.radius_m
            or measured.get("expected_floor_z_m") != probe.floor_z_m
        ):
            raise ValueError("Physics probe does not match the reviewed recipe")
    return report


def package_reviewed(manifest: Path, output_name: str = "final") -> list[Path]:
    prepared = []
    for job, workspace, out in _jobs(manifest, output_name):
        archive_name = (
            f"{workspace.name}_asset.zip"
            if output_name == "final"
            else f"{workspace.name}_{output_name}_asset.zip"
        )
        archive = workspace / archive_name
        if archive.exists():
            raise ValueError(f"Archive already exists; preserve it and use another output name: {archive}")
        report = verify_physics_report(out, job.probe)
        metadata = json.loads(local_file(out, "asset.json").read_text())
        scene = SceneSpec.model_validate_json(local_file(out, "scene.json").read_text())
        if PhysicsConfig.model_validate(metadata["physics"]) != job.physics:
            raise ValueError("Asset physics parameters differ from the reviewed recipe")
        expected = SceneSpec.model_validate_json(local_file(workspace, job.scene).read_text())
        if expected != scene:
            raise ValueError("Asset scene differs from the reviewed recipe")
        prepared.append((job, out, archive, report, metadata, scene))
    for job, out, archive, report, metadata, scene in prepared:
        metadata["isaac_physics_tested"] = True
        metadata["physics_validation"] = {"report": "physics_smoke.json", "scope": report["scope"]}
        write_json(out / "asset.json", metadata)
        state = json.loads((out / "run.json").read_text())
        state.update(status="complete", physics_report="physics_smoke.json", scope=report["scope"])
        write_json(out / "run.json", state)
        dimensions = " x ".join(f"{x * 1000:.1f}" for x in metadata["dimensions_m"])
        notes = "\n".join(f"- {note}" for note in scene.assumptions)
        readme = (
            f"# {scene.name}\n\n{scene.summary}\n\n"
            "Review `viewer.html` and `comparison.jpg` when present. "
            "Load `asset_dynamic.usda` for simulation or `asset_static.usda` for scenery "
            "using `isaaclab_snippet.py`. Keep the asset folder together.\n\n"
            f"Bounds: {dimensions} mm ({metadata['scale_basis']}). "
            f"Mass: {metadata['effective_mass_kg']} kg ({metadata['mass_basis']}). "
            f"{metadata['visual_triangles']} visual triangles; "
            f"{metadata['collision_pieces']} convex colliders.\n\n"
            "USD/OBJ/STL: meters, Z-up, origin at the centered bottom plane. GLB: meters, Y-up. "
            "Use USD scale `(1, 1, 1)`.\n\n"
            f"## Visual review\n\n{job.review}\n\n"
            f"## Verified physics\n\n{report['scope']} See `physics_smoke.json` for tested USD hashes. "
            "Revalidate after changing geometry.\n\n"
            f"## Assumptions\n\n{notes}\n\n"
            "## Reproduce\n\nUse `scene.json` as geometry input and copy the physics/probe "
            "settings from `build_recipe.json` into your batch manifest. Run "
            "`kuafu assets build-reviewed MANIFEST --output-name replay` with a fresh output name.\n"
        )
        (out / "README.md").write_text(readme)
        with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as bundle:
            for path in sorted(out.rglob("*")):
                if path.is_file() and not path.is_symlink():
                    bundle.write(path, path.relative_to(out))
        with zipfile.ZipFile(archive) as bundle:
            if bundle.testzip() is not None:
                raise RuntimeError(f"Archive integrity check failed: {archive}")
    return [archive for _, _, archive, _, _, _ in prepared]
