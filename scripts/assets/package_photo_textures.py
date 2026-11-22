"""Package an existing photo bake only after hash-verified Isaac physics tests.

This records a physics pass, not a certificate of texture accuracy.
"""

import argparse
import json
import zipfile
from pathlib import Path

from kuafu.asset_generation.delighting import ALGORITHM, DelightConfig
from kuafu.asset_generation.io import digest, local_file, resolve_artifact_path, write_json
from kuafu.asset_generation.reviewed import Probe, verify_physics_report


def verify_photo_provenance(bake, provenance):
    if provenance.get("backend") != "video_pixel_projection":
        raise ValueError("Expected a video-pixel texture bake")
    correction = provenance.get("delighting")
    if correction is None:
        if provenance.get("model_calls") != 0:
            raise ValueError("A direct video-pixel bake must record zero model calls")
        return False
    if correction.get("algorithm") != ALGORITHM or provenance.get("bake_model_calls") != 0:
        raise ValueError("Unsupported de-lighting provenance")
    manifest = local_file(bake, "delighting.json")
    data = json.loads(manifest.read_text())
    if (
        digest(manifest) != correction.get("manifest_sha256")
        or data.get("version") != 1
        or data.get("status") != "complete_needs_appearance_review"
        or not 1 <= len(data.get("views", [])) <= 12
        or provenance.get("model_calls") != data.get("model_calls")
        or data.get("model_calls") != len(data["views"])
        or data.get("recipe_sha256") != provenance.get("recipe_sha256")
        or any(
            correction.get(key) != data.get(key)
            for key in ("algorithm", "model", "config", "views", "color_spaces")
        )
    ):
        raise ValueError("Changed or inconsistent de-lighting provenance")
    DelightConfig.model_validate(data["config"])
    return True


def package(bake: Path, archive: Path, probe: Probe | None = None):
    bake = resolve_artifact_path(bake, "Asset texture bake")
    archive = resolve_artifact_path(archive, "Asset archive")
    asset = bake / "asset"
    if archive.exists():
        raise ValueError("Choose a new archive path; refusing to overwrite")
    provenance = json.loads((bake / "provenance.json").read_text())
    metadata = json.loads((asset / "asset.json").read_text())
    corrected = verify_photo_provenance(bake, provenance)
    if digest(local_file(asset, "inference_provenance.json")) != digest(bake / "provenance.json"):
        raise ValueError("Asset no longer matches the parent bake provenance")
    visual_hash = digest(bake / "visual.glb")
    if provenance.get("visual_sha256") != visual_hash or metadata.get("source_visual_sha256") != visual_hash:
        raise ValueError("Photo mesh no longer matches recorded source hashes")
    for relative, expected in metadata["files_sha256"].items():
        path = local_file(asset, relative)
        if digest(path) != expected:
            raise ValueError(f"Changed asset file: {relative}")
    report = verify_physics_report(asset, probe)
    metadata.update(
        isaac_physics_tested=True,
        status="physics_validated_visual_approximation",
        texture_accuracy="approximate; seams, residual lighting and inferred surfaces remain",
        physics_validation={
            "report": "physics_smoke.json",
            "report_sha256": digest(asset / "physics_smoke.json"),
            "scope": "240 steps at 120 Hz, ground contact and the configured cavity/shelf probe",
        },
        warnings=[
            "Photo textures use approximate landmark-fitted cameras; no metric or photometric calibration.",
            "Estimated lighting correction, when used, is not calibrated albedo; "
            "seams, residual lighting and inferred fills can remain.",
            "Physics smoke checks do not certify texture fidelity or task-level grasping performance.",
            "Absolute dimensions and mass remain video-only estimates or placeholders.",
        ],
    )
    lighting_note = (
        "Original video pixels were corrected using a local intrinsic-lighting model before UV baking. "
        "This preserves source texture detail where possible; color and lighting removal are estimates. "
        f"The provenance records {provenance['model_calls']} upstream local model calls; "
        "the bake itself makes no model calls. "
        if corrected
        else "Textures use uncorrected video pixels; the video-pixel bake makes zero model calls. "
    )
    (asset / "README.md").write_text(
        "# Photo-textured simulation variant\n\n"
        + lighting_note
        + "The original approximate rigid geometry is unchanged. "
        "No gated model or cloud image service was used. The original assets are unchanged.\n\n"
        "Load `asset_dynamic.usda` for a rigid object or `asset_static.usda` for a fixture. "
        "Keep `textures/` with the USD/OBJ files. `asset.glb` embeds its textures. "
        "`preview.png` renders the visual; `collisions.png` shows the separate convex proxies.\n\n"
        "Isaac Sim physics smoke test **passed**: 240 steps at 120 Hz. See `physics_smoke.json` "
        "for exact hashed layers and any cavity/shelf probe. This is not a task benchmark.\n\n"
        "Appearance remains approximate: seams, baked lighting and inferred fills are possible. "
        "`inference_provenance.json` records the video-pixel backend, lighting correction if used, cameras "
        "and measured coverage. These are not SAM 3D outputs. "
        "Green coverage pixels in the parent bake folder are observed; orange pixels are inferred. "
        "Scale and mass remain assumptions. `collision_scene.json` rebuilds colliders, not the visual.\n"
    )
    metadata["files_sha256"] = {
        str(path.relative_to(asset)): digest(path)
        for path in sorted(asset.rglob("*"))
        if path.is_file() and path.name != "asset.json"
    }
    write_json(asset / "asset.json", metadata)
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(asset.rglob("*")):
            if path.is_symlink():
                raise ValueError("Symlinks are not allowed in a portable asset archive")
            if path.is_file():
                bundle.write(path, str(path.relative_to(asset)))
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None:
            raise RuntimeError("Archive integrity check failed")
    print(
        json.dumps(
            {
                "archive": str(archive.resolve()),
                "sha256": digest(archive),
                "physics_passed": report["passed"],
                "texture_accuracy": "approximate",
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bake", type=Path)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--probe-radius", type=float)
    parser.add_argument("--floor-z", type=float)
    args = parser.parse_args()
    if (args.probe_radius is None) != (args.floor_z is None):
        parser.error("Supply --probe-radius and --floor-z together")
    probe = Probe(radius_m=args.probe_radius, floor_z_m=args.floor_z) if args.probe_radius else None
    package(args.bake, args.archive, probe)
