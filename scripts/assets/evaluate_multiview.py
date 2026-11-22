"""Verify texture-only exports and complete videos; this is not a physics test.

Requires usd-core and the normal project dependencies. Raises on any failed check.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from pxr import Usd

from kuafu.asset_generation.io import digest, local_file, write_json
from kuafu.asset_generation.textured import load_textured_glb


def evaluate(workspace, baseline, candidate, turntable):
    old, new = workspace / baseline, workspace / candidate
    old_meshes = load_textured_glb(old / "asset/asset.glb")
    new_meshes = load_textured_glb(new / "asset/asset.glb")
    assert len(old_meshes) == len(new_meshes)
    for a, b in zip(old_meshes, new_meshes, strict=True):
        for attribute in ("vertices", "faces"):
            np.testing.assert_array_equal(getattr(a, attribute), getattr(b, attribute))
        np.testing.assert_array_equal(a.visual.uv, b.visual.uv)
        for attribute in ("roughnessFactor", "metallicFactor"):
            assert getattr(a.visual.material, attribute) == getattr(b.visual.material, attribute)
    a = Usd.Stage.Open(str((old / "asset/asset_geometry.usda").resolve()))
    b = Usd.Stage.Open(str((new / "asset/asset_geometry.usda").resolve()))
    old_colliders = {
        str(p.GetPath()): p
        for p in a.Traverse()
        if p.GetTypeName() == "Mesh" and p.GetName().startswith("Collider_")
    }
    new_colliders = {
        str(p.GetPath()): p
        for p in b.Traverse()
        if p.GetTypeName() == "Mesh" and p.GetName().startswith("Collider_")
    }
    assert old_colliders.keys() == new_colliders.keys()
    for path, old_prim in old_colliders.items():
        for attribute in ("points", "faceVertexCounts", "faceVertexIndices"):
            np.testing.assert_array_equal(
                old_prim.GetAttribute(attribute).Get(), new_colliders[path].GetAttribute(attribute).Get()
            )
    textures = 0
    for prim in b.Traverse():
        file = prim.GetAttribute("inputs:file")
        if file and file.Get():
            assert prim.GetAttribute("inputs:sourceColorSpace").Get() == "sRGB"
            assert local_file(new / "asset", file.Get().path).is_file()
            textures += 1
    metadata = json.loads((new / "asset/asset.json").read_text())
    assert metadata.get("isaac_physics_tested") is not True
    assert not (new / "asset/physics_smoke.json").exists()
    for relative, expected in metadata["files_sha256"].items():
        assert digest(local_file(new / "asset", relative)) == expected
    provenance = json.loads((new / "provenance.json").read_text())
    assert digest(new / "visual.glb") == provenance["visual_sha256"]
    assert digest(new / "provenance.json") == digest(new / "asset/inference_provenance.json")
    assert len({v["frame"] for v in provenance["source_views"]}) == len(provenance["source_views"])
    assert provenance["view_fusion"] == "robust" and not provenance["geometry_changed"]
    for view in provenance["source_views"]:
        assert digest(local_file(workspace, view["frame"])) == view["frame_sha256"]
    before = json.loads((old / "provenance.json").read_text())

    def coverage(report):
        parts = report["parts"]
        for part in parts:
            assert part["observed_texels"] == sum(part["source_view_texels"])
        return sum(p["observed_texels"] for p in parts) / sum(p["surface_texels"] for p in parts)

    folder = workspace / turntable
    render = json.loads((folder / "render.json").read_text())
    source = (folder / render["source_glb"]).resolve()
    assert source == (new / "asset/asset.glb").resolve()
    assert digest(source) == render["source_sha256"]
    for name, sha in render["files_sha256"].items():
        assert digest(local_file(folder, name)) == sha
    video = folder / "turntable.mp4"
    assert video.stat().st_size == render["video_bytes"] < 20_000_000
    capture, count = cv2.VideoCapture(str(video)), 0
    try:
        assert capture.isOpened() and capture.get(cv2.CAP_PROP_FPS) == render["fps"]
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            assert list(frame.shape[1::-1]) == render["resolution"]
            count += 1
    finally:
        capture.release()
    assert count == render["frame_count"] == 240
    return dict(
        object=workspace.name,
        baseline_views=len(before["source_views"]),
        registered_views=len(provenance["source_views"]),
        baseline_observed_uv_fraction=coverage(before),
        observed_uv_fraction=coverage(provenance),
        coverage_note="New robust coverage requires two agreeing observations; baseline required one.",
        visual_parts=len(new_meshes),
        collision_parts=len(new_colliders),
        srgb_texture_bindings=textures,
        vertices_faces_uvs_unchanged=True,
        roughness_metallic_unchanged=True,
        collision_points_indices_unchanged=True,
        physics_rerun=False,
        delighting_model_calls=provenance["model_calls"],
        asset_glb_sha256=digest(source),
        asset=f"{workspace.name}/{candidate}/asset/asset.glb",
        comparison=f"{workspace.name}/{candidate}/before_after.jpg",
        appearance_status="review_required_not_calibrated_albedo",
        turntable=dict(
            path=f"{workspace.name}/{turntable}/turntable.mp4",
            sha256=digest(video),
            bytes=video.stat().st_size,
            decoded_frames=count,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspaces", type=Path, nargs="+")
    parser.add_argument("--baseline", default="photo_delighted")
    parser.add_argument("--candidate", default="photo_multiview_detail")
    parser.add_argument("--turntable", default="turntable_multiview")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Choose a new evaluation report path")
    results = [evaluate(p, args.baseline, args.candidate, args.turntable) for p in args.workspaces]
    write_json(
        args.out,
        dict(
            scope="export invariance, provenance and complete-video decoding; "
            "not a texture-accuracy benchmark or a physics rerun",
            objects=results,
        ),
    )
    print(json.dumps(results, indent=2))
