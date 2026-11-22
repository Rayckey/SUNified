"""Synthetic cameras/images validate projection math, not real-object accuracy."""

import inspect
import json

import cv2
import numpy as np
import pytest
import trimesh

from kuafu.asset_generation.phototexture import (
    DEFAULT_FUSION,
    bake_mesh,
    bake_video_textures,
    depth_buffer,
    fit_camera,
    fuse_view_samples,
    project,
    triangle_pixels,
)
from kuafu.asset_generation.render import preview_mesh


def test_material_guard_rejects_cross_material_color_without_changing_rgb():
    from kuafu.asset_generation.phototexture import sample_color_guard

    pixels = np.array([[220, 35, 40], [245, 235, 25], [180, 185, 180]], np.uint8)
    before = pixels.copy()
    np.testing.assert_array_equal(sample_color_guard(pixels, {"min_red_ratio": 1.4}), [True, False, False])
    np.testing.assert_array_equal(sample_color_guard(pixels, {"max_saturation": 0.25}), [False, False, True])
    np.testing.assert_array_equal(pixels, before)


@pytest.fixture
def camera():
    return {
        "rotation": np.eye(3).tolist(),
        "translation": [0, 0, 3.0],
        "focal_px": 150.0,
        "principal_point": [64.0, 64.0],
        "camera_center": [0, 0, -3.0],
    }


def test_camera_recovers_synthetic_correspondences():
    mesh = trimesh.creation.box(extents=[0.3, 0.2, 0.1])
    rotation, _ = cv2.Rodrigues(np.array([0.3, -0.2, 0.1]))
    truth = {
        "rotation": rotation,
        "translation": [0.1, -0.2, 1.5],
        "focal_px": 1100.0,
        "principal_point": [360, 640],
    }
    pixels, _ = project(mesh.vertices, truth)
    fitted = fit_camera(mesh.vertices, pixels, (720, 1280))
    assert fitted["reprojection_rms_px"] < 1e-5
    assert fitted["focal_px"] == pytest.approx(truth["focal_px"], rel=1e-5)
    np.testing.assert_allclose(fitted["rotation"], rotation, atol=1e-5)


@pytest.mark.parametrize("points", [np.zeros((6, 3)), np.ones((3, 3)), np.full((6, 3), np.nan)])
def test_degenerate_camera_landmarks_rejected(points):
    with pytest.raises(ValueError, match="landmarks"):
        fit_camera(points, np.zeros((len(points), 2)), (128, 128))


def test_depth_buffer_uses_nearest_surface(camera):
    mesh = trimesh.creation.box(extents=[1, 1, 1])
    depth = depth_buffer(mesh, camera, (128, 128))
    assert depth[64, 64] == pytest.approx(2.5)
    assert np.isinf(depth[0, 0])


def test_triangle_rasterizer_degenerate_and_barycentric():
    assert triangle_pixels([[1, 1], [1, 1], [1, 1]], 16, 16) is None
    y, x, weights = triangle_pixels([[0, 0], [10, 0], [0, 10]], 16, 16)
    assert len(y) == len(x) > 0
    np.testing.assert_allclose(weights.sum(axis=1), 1)


def test_bake_preserves_geometry_and_keeps_inferred_coverage_separate(camera):
    mesh = trimesh.creation.box(extents=[1, 1, 1])
    mesh.metadata.update(name="test_cube", roughness=0.5, metallic=0)
    mesh.visual.face_colors = [120, 130, 140, 255]
    image = np.full((128, 128, 3), [200, 60, 20], np.uint8)
    view = {
        "image": image,
        "mask": np.ones((128, 128), bool),
        "camera": camera,
        "depth": depth_buffer(mesh, camera, (128, 128)),
    }
    baked, coverage, report = bake_mesh(mesh, [view], 128, 0.02, fusion="best")
    np.testing.assert_allclose(baked.vertices[baked.faces], mesh.vertices[mesh.faces])
    assert 0 < report["observed_uv_fraction"] < 0.3
    assert report["observed_texels"] == report["source_view_texels"][0]
    assert ((np.array(coverage) == [52, 187, 117]).all(axis=-1)).sum() == report["observed_texels"]
    colors = np.array(baked.visual.material.baseColorTexture)
    assert ((colors == [200, 60, 20]).all(axis=-1)).any()
    assert ((colors == [120, 130, 140]).all(axis=-1)).any()  # Unseen charts keep a prior.


def test_known_occluder_pixels_are_not_baked_on_other_parts(camera):
    mesh = trimesh.creation.box(extents=[1, 1, 1])
    mesh.metadata.update(name="wood", roughness=0.5, metallic=0)
    mesh.visual.face_colors = [120, 130, 140, 255]
    view = {
        "image": np.full((128, 128, 3), 255, np.uint8),
        "mask": np.ones((128, 128), bool),
        "camera": camera,
        "depth": depth_buffer(mesh, camera, (128, 128)),
        "occluders": [{"mask": np.ones((128, 128), bool), "parts": ["handle"]}],
    }
    _, _, report = bake_mesh(mesh, [view], 128, 0.02, fusion="best")
    assert report["observed_texels"] == 0


@pytest.mark.parametrize(
    "source, corrected, accepted",
    [([180, 185, 180], [220, 35, 40], True), ([220, 35, 40], [180, 185, 180], False)],
)
def test_delighting_does_not_change_material_guard_or_coverage(camera, source, corrected, accepted):
    mesh = trimesh.creation.box(extents=[1, 1, 1])
    mesh.metadata.update(name="test_cube", roughness=0.5, metallic=0)
    mesh.visual.face_colors = [120, 130, 140, 255]
    view = {
        "image": np.full((128, 128, 3), source, np.uint8),
        "mask": np.ones((128, 128), bool),
        "camera": camera,
        "depth": depth_buffer(mesh, camera, (128, 128)),
        "color_guards": {"test_cube": {"max_saturation": 0.25}},
    }
    _, _, baseline = bake_mesh(mesh, [view], 128, 0.02, fusion="best")
    view["texture_image"] = np.full((128, 128, 3), corrected, np.uint8)
    baked, _, report = bake_mesh(mesh, [view], 128, 0.02, fusion="best")
    assert report == baseline
    assert bool(report["observed_texels"]) == accepted
    if accepted:
        assert (np.array(baked.visual.material.baseColorTexture) == corrected).all(axis=2).any()
    np.testing.assert_allclose(baked.vertices[baked.faces], mesh.vertices[mesh.faces])


def test_multi_material_preview_does_not_invert_rgb():
    from PIL import Image

    mesh = trimesh.creation.box()
    pixels = np.array([[[255, 0, 0], [0, 0, 255]], [[0, 0, 255], [255, 0, 0]]], np.uint8)
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=np.zeros((len(mesh.vertices), 2)),
        material=trimesh.visual.material.PBRMaterial(
            baseColorTexture=Image.fromarray(pixels),
            baseColorFactor=[1.0, 1.0, 1.0, 1.0],
        ),
    )
    other = mesh.copy()
    other.apply_translation([4, 0, 0])
    material = other.visual.material
    # A distinct material forces the library's texture packing code to run.
    material.roughnessFactor = 0.6
    merged = preview_mesh([mesh, other])
    pixels = np.array(merged.visual.material.baseColorTexture)
    assert ((pixels[:, :, :3] == [255, 0, 0]).all(axis=-1)).any()
    assert ((pixels[:, :, :3] == [0, 0, 255]).all(axis=-1)).any()
    assert (mesh.visual.material.baseColorFactor == [255] * 4).all()


def test_fusion_rejects_disagreeing_color_even_if_its_angle_score_is_best():
    colors = np.array([[[180, 180, 170]], [[185, 180, 170]], [[90, 40, 20]]], np.uint8)
    result, source = fuse_view_samples(colors, colors.astype(float), np.array([[1.0], [2.0], [10.0]]))
    assert source.tolist() == [1]
    assert result[0, 0] > 180 and result[0, 1] == 180 and result[0, 2] == 170


def test_fusion_blends_broad_color_but_keeps_single_view_fine_detail():
    detail = np.tile([-20, 20], 50)
    broad = np.array([100, 110, 115])[:, None, None] * np.ones((3, 100, 3))
    colors = (broad + detail[None, :, None]).astype(np.uint8)
    result, source = fuse_view_samples(colors, broad, np.tile([[3.0], [1.0], [1.0]], (1, 100)))
    assert np.all(source == 0)
    assert result[:, 0].mean() == pytest.approx(105)
    assert result[:, 0].std() == pytest.approx(20)
    np.testing.assert_array_equal(np.diff(result[:, 0].astype(int)), np.diff(detail))


def test_fusion_does_not_erase_real_ink_when_other_views_are_slightly_misaligned():
    colors = np.array([[[15, 15, 15]], [[180, 180, 180]], [[180, 180, 180]]], np.uint8)
    broad = np.full((3, 1, 3), 170, dtype=np.float32)
    result, source = fuse_view_samples(colors, broad, np.array([[4.0], [1.0], [1.0]]))
    assert source.tolist() == [0]
    np.testing.assert_array_equal(result, [[15, 15, 15]])


def test_fusion_keeps_agreeing_dark_print_and_rejects_single_view_or_missing_data():
    colors = np.full((3, 4, 3), 20, np.uint8)
    scores = np.array([[1, 1, 0, 1], [1, 0, 0, 1], [1, 0, 0, 0]], float)
    result, source = fuse_view_samples(colors, colors.astype(float), scores)
    assert source.tolist() == [0, -1, -1, 0]
    np.testing.assert_array_equal(result[[0, 3]], 20)


def test_robust_bake_preserves_geometry_and_marks_unseen_prior(camera):
    mesh = trimesh.creation.box(extents=[1, 1, 1])
    mesh.metadata.update(name="test_cube", roughness=0.5, metallic=0)
    mesh.visual.face_colors = [20, 30, 40, 255]
    view = {
        "image": np.full((128, 128, 3), [180, 160, 140], np.uint8),
        "mask": np.ones((128, 128), bool),
        "camera": camera,
        "depth": depth_buffer(mesh, camera, (128, 128)),
    }
    baked, coverage, report = bake_mesh(mesh, [view, view, view], 128, 0.02)
    np.testing.assert_allclose(baked.vertices[baked.faces], mesh.vertices[mesh.faces])
    assert 0 < report["observed_uv_fraction"] < 0.3
    assert report["minimum_observations"] == 2 and report["fusion"] == "robust"
    assert "part median" in report["unseen_policy"]
    assert np.any((np.array(coverage) == [225, 89, 56]).all(axis=-1))
    assert report["observed_texels"] == sum(report["source_view_texels"])
    with pytest.raises(ValueError, match="at least three"):
        bake_mesh(mesh, [view], 128, 0.02)


def test_all_baking_apis_default_to_robust_fusion():
    assert DEFAULT_FUSION == "robust"
    for function in (bake_mesh, bake_video_textures):
        assert inspect.signature(function).parameters["fusion"].default == DEFAULT_FUSION


@pytest.mark.parametrize("override, expected", [([], "robust"), (["--fusion", "best"], "best")])
def test_cli_uses_robust_unless_legacy_fusion_is_explicit(tmp_path, monkeypatch, override, expected):
    from kuafu.asset_generation import cli

    seen = []

    def fake_bake(*args, **kwargs):
        seen.append(kwargs["fusion"])
        return tmp_path / "out/asset/asset_dynamic.usda"

    monkeypatch.setattr(cli, "bake_video_textures", fake_bake)
    assert (
        cli.main(
            [
                "bake-video-textures",
                str(tmp_path),
                "--recipe",
                str(tmp_path / "recipe.json"),
                "--out",
                str(tmp_path / "out"),
                *override,
            ]
        )
        == 0
    )
    assert seen == [expected]


def test_default_bake_does_not_silently_fall_back_for_two_views(tmp_path):
    recipe = tmp_path / "recipe.json"
    recipe.write_text(json.dumps({"version": 1, "views": [{"frame": "a.jpg"}, {"frame": "b.jpg"}]}))
    out = tmp_path / "new"
    with pytest.raises(ValueError, match="track-photo-views.*--fusion best"):
        bake_video_textures(tmp_path, recipe, out)
    assert not out.exists()
