"""Synthetic texture fixtures test formats only; these are NOT SAM 3D reconstructions."""

import json

import numpy as np
import pytest
import trimesh
from PIL import Image

from kuafu.asset_generation.geometry import build_asset
from kuafu.asset_generation.io import write_json
from kuafu.asset_generation.render import render_mesh
from kuafu.asset_generation.schema import PhysicsConfig
from kuafu.asset_generation.textured import align_visuals, build_textured_asset, load_textured_glb


@pytest.fixture
def textured_glb(tmp_path):
    # An asymmetrically colored texture exposes UV flips and accidental flat-color exports.
    pixels = np.zeros((16, 16, 3), np.uint8)
    pixels[:8, :8] = [255, 0, 0]
    pixels[:8, 8:] = [0, 255, 0]
    pixels[8:, :8] = [0, 0, 255]
    pixels[8:, 8:] = [255, 255, 0]
    box = trimesh.creation.box(extents=[2, 3, 1])
    # Explode vertices to give each face independent UV coordinates.
    vertices = box.vertices[box.faces].reshape(-1, 3)
    mesh = trimesh.Trimesh(vertices=vertices, faces=np.arange(len(vertices)).reshape(-1, 3), process=False)
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=np.tile([[0.02, 0.02], [0.98, 0.02], [0.02, 0.98]], (len(mesh.faces), 1)),
        material=trimesh.visual.material.PBRMaterial(
            baseColorTexture=Image.fromarray(pixels),
            baseColorFactor=[1.0, 1.0, 1.0, 1.0],
            roughnessFactor=0.8,
            metallicFactor=0.0,
            alphaMode="OPAQUE",
        ),
    )
    path = tmp_path / "synthetic.glb"
    mesh.export(path)
    return path, pixels


def test_uv_texture_survives_glb_obj_usd_exports(textured_glb, holder, tmp_path):
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

    path, pixels = textured_glb
    recipe = tmp_path / "collisions.json"
    write_json(recipe, holder)
    out = tmp_path / "textured"
    original = load_textured_glb(path)[0]
    metadata = build_textured_asset(path, recipe, out, PhysicsConfig(mass_kg=0.2, mass_is_estimate=True))
    assert metadata["isaac_physics_tested"] is False
    assert metadata["status"] == "needs_visual_alignment_and_physics_review"
    assert metadata["mass_basis"] == "estimated" and metadata["scale_basis"] == "estimated"
    for reloaded in (
        load_textured_glb(out / "asset.glb")[0],
        trimesh.load(out / "asset.obj", force="mesh", process=False),
    ):
        assert reloaded.visual.kind == "texture"
        np.testing.assert_allclose(reloaded.visual.uv, original.visual.uv)
    np.testing.assert_array_equal(Image.open(out / "textures/basecolor_000.png"), pixels)
    np.testing.assert_array_equal(
        load_textured_glb(out / "asset.glb")[0].visual.material.baseColorTexture, pixels
    )
    stage = Usd.Stage.Open(str(out / "asset_dynamic.usda"))
    assert stage.GetDefaultPrim().HasAPI(UsdPhysics.RigidBodyAPI)
    visual = stage.GetPrimAtPath("/Asset/Visuals/Part_000")
    uv = UsdGeom.PrimvarsAPI(visual).GetPrimvar("st")
    assert uv.GetInterpolation() == "vertex"
    np.testing.assert_allclose(np.array(uv.Get()), original.visual.uv)
    material, _ = UsdShade.MaterialBindingAPI(visual).ComputeBoundMaterial()
    surface = UsdShade.Shader(stage.GetPrimAtPath(str(material.GetPath()) + "/Surface"))
    connection = surface.GetInput("diffuseColor").GetConnectedSource()
    assert connection and str(connection[0].GetPath()).endswith("/BaseColor")
    texture = UsdShade.Shader(stage.GetPrimAtPath(str(material.GetPath()) + "/BaseColor"))
    assert texture.GetInput("sourceColorSpace").Get() == "sRGB"
    texture_path = texture.GetInput("file").Get()
    assert texture_path.path == "textures/basecolor_000.png" and texture_path.resolvedPath
    colliders = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]
    assert len(colliders) == len(build_asset(holder).collisions)
    assert not visual.HasAPI(UsdPhysics.CollisionAPI)
    assert (out / "preview.png").is_file() and (out / "collisions.png").is_file()
    assert json.loads((out / "asset.json").read_text())["source_visual_sha256"]
    assert not (out / "scene.json").exists()  # Do not pretend the collision recipe rebuilds the visual.


def test_uniform_alignment_corrects_up_axis_preserves_shape(textured_glb):
    meshes = load_textured_glb(textured_glb[0])
    matrix, dimensions = align_visuals(meshes, [0.2, 0.1, 0.3])
    np.testing.assert_allclose(dimensions, [0.2, 0.1, 0.3], atol=1e-7)
    np.testing.assert_allclose(meshes[0].bounds, [[-0.1, -0.05, 0], [0.1, 0.05, 0.3]], atol=1e-7)
    np.testing.assert_allclose(np.linalg.svd(matrix[:3, :3])[1], [0.1] * 3)


def test_preview_samples_texture_spatially(textured_glb):
    mesh = load_textured_glb(textured_glb[0])[0]
    pixels = np.array(render_mesh(mesh, size=160))
    assert ((pixels[..., 0] > 150) & (pixels[..., 1] < 20) & (pixels[..., 2] < 20)).any()
    assert ((pixels[..., 2] > 150) & (pixels[..., 0] < 20) & (pixels[..., 1] < 20)).any()


def test_no_flat_color_fallback_and_no_overwrite(textured_glb, holder, tmp_path):
    path = tmp_path / "untextured.glb"
    trimesh.creation.box().export(path)
    with pytest.raises(ValueError, match="UV"):
        load_textured_glb(path)
    out = tmp_path / "existing"
    out.mkdir()
    sentinel = out / "user.txt"
    sentinel.write_text("keep")
    recipe = tmp_path / "collisions.json"
    write_json(recipe, holder)
    with pytest.raises(ValueError, match="new or empty"):
        build_textured_asset(textured_glb[0], recipe, out, PhysicsConfig())
    assert sentinel.read_text() == "keep"


def test_mismatched_provenance_rejected(textured_glb, holder, tmp_path):
    write_json(tmp_path / "provenance.json", {"visual_sha256": "wrong"})
    recipe = tmp_path / "collisions.json"
    write_json(recipe, holder)
    with pytest.raises(ValueError, match="provenance"):
        build_textured_asset(textured_glb[0], recipe, tmp_path / "new", PhysicsConfig())
    assert not (tmp_path / "new").exists()
