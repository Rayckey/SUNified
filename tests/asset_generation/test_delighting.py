"""Known-answer fixtures test correction/provenance, not learned-model accuracy.

These tests use a deterministic fake predictor, never download model weights and
do not require torch, CUDA, diffusers, a network connection or a model account.
"""

import json
import runpy
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from kuafu.asset_generation.cli import main
from kuafu.asset_generation.delighting import (
    DelightConfig,
    correct_lighting,
    delight_frames,
    linear_to_srgb,
    load_delighting,
    srgb_to_linear,
)
from kuafu.asset_generation.geometry import build_asset
from kuafu.asset_generation.io import digest, write_json
from kuafu.asset_generation.phototexture import bake_video_textures, project
from kuafu.asset_generation.schema import SceneSpec
from kuafu.asset_generation.video import prepare


def rgb8(linear):
    return np.round(linear_to_srgb(linear) * 255).astype(np.uint8)


def test_linear_color_conversion_not_gamma_baked_twice():
    encoded = np.arange(256, dtype=np.float32) / 255
    np.testing.assert_allclose(linear_to_srgb(srgb_to_linear(encoded)), encoded, atol=2e-7)
    assert srgb_to_linear([0.5])[0] == pytest.approx(0.214041, abs=1e-6)


def test_removes_known_shadow_and_highlight_without_flattening_grain_or_lettering():
    yy, xx = np.mgrid[:128, :192]
    grain = 1 + 0.13 * np.sin(xx * np.pi / 3) + 0.03 * np.sin(yy / 4)
    truth = grain[:, :, None] * np.array([0.42, 0.25, 0.12])
    cv2.putText(truth, "TEST", (18, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0.035, 0.035, 0.035), 3)
    shading = (0.30 + xx / 240)[:, :, None] * np.array([0.90, 1.0, 1.12])
    highlight = 0.15 * np.exp(-((xx - 145) ** 2 + (yy - 32) ** 2) / 600)
    source = rgb8(truth * shading + highlight[:, :, None])
    output, _ = correct_lighting(source, truth, np.ones(xx.shape, bool), DelightConfig())
    recovered = srgb_to_linear(output / 255)
    assert np.mean(np.abs(recovered - truth)) < 0.009
    assert np.mean(np.abs(recovered - truth)) < np.mean(np.abs(srgb_to_linear(source / 255) - truth)) / 5
    ink = truth[:, :, 0] < 0.04
    assert recovered[ink].mean() < 0.05
    assert np.corrcoef(recovered[10, 10:180, 0], truth[10, 10:180, 0])[0, 1] > 0.97


def test_source_fine_detail_survives_a_smooth_model_guide():
    yy, xx = np.mgrid[:128, :128]
    pattern = 1 + 0.15 * ((xx + yy) % 2 * 2 - 1)
    truth = np.repeat((0.3 * pattern)[:, :, None], 3, axis=2)
    source = rgb8(truth * 0.5)
    guide = np.full(truth.shape, 0.3, np.float32)  # Model omitted the actual fine pattern.
    output, _ = correct_lighting(source, guide, np.ones(xx.shape, bool), DelightConfig(smooth_fraction=0.01))
    recovered = srgb_to_linear(output / 255)
    assert recovered[8:-8, 8:-8, 0].std() > 0.04  # Not the flat model image.
    assert np.mean(np.abs(recovered - truth)) < 0.007


def test_material_edges_and_color_ratios_survive_constant_lighting():
    truth = np.full((64, 96, 3), [0.5, 0.12, 0.04], np.float32)
    truth[:, 48:] = [0.06, 0.4, 0.12]
    source = rgb8(truth * [0.4, 0.5, 0.6])
    output, _ = correct_lighting(source, truth, np.ones((64, 96), bool), DelightConfig())
    np.testing.assert_allclose(srgb_to_linear(output / 255), truth, atol=0.003)


def test_correction_never_blurs_original_or_samples_mask_background():
    image = np.full((64, 64, 3), [110, 100, 90], np.uint8)
    mask = np.zeros((64, 64), bool)
    mask[8:56, 8:56] = True
    mask[25:38, 25:38] = False  # A real hole, not a lighting estimate.
    first = srgb_to_linear(image / 255) * 2
    second = first.copy()
    second[~mask] = 0.99
    a, _ = correct_lighting(image, first, mask, DelightConfig())
    b, _ = correct_lighting(image, second, mask, DelightConfig())
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a[~mask], image[~mask])
    assert (a[mask] > image[mask]).all()


def test_gain_limits_black_pixels_and_disabled_correction():
    image = np.full((64, 64, 3), 20, np.uint8)
    image[10:15] = 0
    guide = np.ones(image.shape, np.float32)
    mask = np.ones(image.shape[:2], bool)
    output, report = correct_lighting(image, guide, mask, DelightConfig(max_gain=4))
    assert report["gain_limited_fraction"] == 1
    assert np.isfinite(output).all() and output[10:15].max() == 0
    assert output.max() <= rgb8(srgb_to_linear([20 / 255]) * 4)[0]
    unchanged, _ = correct_lighting(image, guide, mask, DelightConfig(strength=0))
    np.testing.assert_array_equal(unchanged, image)


@pytest.mark.parametrize(
    "change",
    [
        {"steps": 0},
        {"ensemble_size": 0},
        {"max_gain": 0.5},
        {"strength": float("nan")},
        {"smooth_fraction": -1},
    ],
)
def test_invalid_config_rejected(change):
    with pytest.raises(ValueError):
        DelightConfig(**change)


@pytest.mark.parametrize(
    "bad", [np.full((32, 32, 3), np.nan), np.ones((32, 32, 3)) * 2, np.ones((31, 32, 3))]
)
def test_bad_intrinsic_arrays_rejected(bad):
    with pytest.raises(ValueError, match="Albedo"):
        correct_lighting(np.full((32, 32, 3), 100, np.uint8), bad, np.ones((32, 32), bool), DelightConfig())


class FakePredictor:
    info = {"id": "synthetic-test-only-not-marigold", "revision": "fixture"}

    def __init__(self):
        self.seeds = []

    def predict(self, image, seed):
        self.seeds.append(seed)
        albedo = srgb_to_linear(image / 255) * 0.65
        return {"albedo": albedo, "shading": np.ones_like(albedo), "residual": np.zeros_like(albedo)}


@pytest.fixture
def recipe_workspace(video, tmp_path):
    workspace = tmp_path / "prepared"
    prepared = prepare(video, "synthetic rigid block", workspace, 2, 128)
    write_json(workspace / "mask.json", {"include": [[[8, 8], [115, 8], [115, 87], [8, 87]]]})
    recipe = workspace / "photo_recipe.json"
    write_json(
        recipe,
        {
            "version": 1,
            "scene": "scene.json",
            "views": [{"frame": frame.path, "mask": "mask.json"} for frame in prepared.frames],
        },
    )
    return workspace, recipe


@pytest.fixture
def corrected_job(recipe_workspace, tmp_path):
    workspace, recipe = recipe_workspace
    predictor = FakePredictor()
    manifest = delight_frames(workspace, recipe, tmp_path / "corrected", predictor=predictor)
    return workspace, recipe, manifest, predictor


def test_workflow_preserves_input_hashes_dimensions_mask_holes_and_caches_results(corrected_job):
    workspace, recipe, manifest, predictor = corrected_job
    images, info = load_delighting(workspace, recipe, manifest)
    assert predictor.seeds == [42, 43]
    assert info["model_calls"] == 2
    assert info["model"]["id"].startswith("synthetic-test-only")
    for entry in info["views"]:
        original = np.array(Image.open(workspace / entry["frame"]).convert("RGB"))
        assert images[entry["frame"]].shape == original.shape
        np.testing.assert_array_equal(images[entry["frame"]][:8], original[:8])
        assert digest(workspace / entry["frame"]) == entry["frame_sha256"]
        assert (manifest.parent / entry["comparison"]).is_file()
    load_delighting(workspace, recipe, manifest)
    assert predictor.seeds == [42, 43]  # Reuse requires no new inference.


def test_no_overwrite_before_inference(corrected_job):
    workspace, recipe, manifest, predictor = corrected_job
    before = digest(manifest)
    with pytest.raises(ValueError, match="new or empty"):
        delight_frames(workspace, recipe, manifest.parent, predictor=predictor)
    assert predictor.seeds == [42, 43] and digest(manifest) == before


def test_changed_source_fails_before_inference_or_output_creation(recipe_workspace, tmp_path):
    workspace, recipe = recipe_workspace
    frame = json.loads(recipe.read_text())["views"][0]["frame"]
    (workspace / frame).write_bytes(b"changed source")
    predictor = FakePredictor()
    with pytest.raises(ValueError, match="Source frame changed"):
        delight_frames(workspace, recipe, tmp_path / "new", predictor=predictor)
    assert not predictor.seeds and not (tmp_path / "new").exists()


@pytest.mark.parametrize("target", ["recipe", "input", "mask", "corrected", "diagnostic"])
def test_changed_inputs_or_outputs_rejected(corrected_job, target):
    workspace, recipe, manifest, _ = corrected_job
    info = json.loads(manifest.read_text())
    paths = {
        "recipe": recipe,
        "input": workspace / "input.json",
        "mask": workspace / "mask.json",
        "corrected": manifest.parent / info["views"][0]["corrected"],
        "diagnostic": manifest.parent / info["views"][0]["comparison"],
    }
    with paths[target].open("ab") as stream:
        stream.write(b" \n")  # Valid JSON still gets rejected by identity checks.
    with pytest.raises(ValueError, match="changed|Changed"):
        load_delighting(workspace, recipe, manifest)


@pytest.mark.parametrize(
    "change", ["order", "missing", "duplicate", "incomplete", "path_escape", "size", "background"]
)
def test_manifest_rejects_incomplete_misaligned_or_unsafe_frames(corrected_job, tmp_path, change):
    workspace, recipe, manifest, _ = corrected_job
    data = json.loads(manifest.read_text())
    first = data["views"][0]
    if change == "order":
        data["views"].reverse()
    elif change == "missing":
        data["views"].pop()
    elif change == "duplicate":
        data["views"][1] = first.copy()
    elif change == "incomplete":
        data["status"] = "running"
    elif change == "path_escape":
        outside = tmp_path / "outside.png"
        Image.new("RGB", (128, 96)).save(outside)
        first["corrected"] = "../outside.png"
        first["corrected_sha256"] = digest(outside)
        data["files_sha256"]["../outside.png"] = digest(outside)
    else:
        path = manifest.parent / first["corrected"]
        image = np.array(Image.open(path))
        if change == "size":
            image = image[:-1]
        else:
            image[0, 0] = 0
        Image.fromarray(image).save(path)
        first["corrected_sha256"] = digest(path)
        data["files_sha256"][first["corrected"]] = digest(path)
    write_json(manifest, data)
    with pytest.raises(ValueError):
        load_delighting(workspace, recipe, manifest)


def test_prediction_failure_does_not_publish_a_partial_manifest(recipe_workspace, tmp_path):
    class BrokenPredictor(FakePredictor):
        def predict(self, image, seed):
            result = super().predict(image, seed)
            if len(self.seeds) == 2:
                result["shading"][0, 0] = np.nan
            return result

    workspace, recipe = recipe_workspace
    with pytest.raises(RuntimeError, match="finite"):
        delight_frames(workspace, recipe, tmp_path / "failed", predictor=BrokenPredictor())
    assert not (tmp_path / "failed/manifest.json").exists()


def test_source_edit_during_inference_prevents_manifest_publication(recipe_workspace, tmp_path):
    workspace, recipe = recipe_workspace

    class EditingPredictor(FakePredictor):
        def predict(self, image, seed):
            result = super().predict(image, seed)
            with (workspace / "mask.json").open("a") as stream:
                stream.write("\n")
            return result

    with pytest.raises(ValueError, match="Inputs changed during"):
        delight_frames(workspace, recipe, tmp_path / "failed", predictor=EditingPredictor())
    assert not (tmp_path / "failed/manifest.json").exists()


def test_cli_dispatches_options_without_loading_optional_runtime(monkeypatch, recipe_workspace, tmp_path):
    workspace, recipe = recipe_workspace
    seen = []

    def fake(workspace, recipe, out, config, **kwargs):
        seen.append((config, kwargs))
        return out / "manifest.json"

    monkeypatch.setattr("kuafu.asset_generation.cli.delight_frames", fake)
    assert (
        main(
            [
                "delight-frames",
                str(workspace),
                "--recipe",
                str(recipe),
                "--out",
                str(tmp_path / "new"),
                "--strength",
                "0.8",
                "--seed",
                "17",
                "--device",
                "cpu",
                "--local-files-only",
            ]
        )
        == 0
    )
    assert seen[0][0].strength == 0.8 and seen[0][0].seed == 17
    assert seen[0][1]["device"] == "cpu" and seen[0][1]["local_files_only"] is True


def test_end_to_end_bake_changes_maps_not_geometry_cameras_coverage_or_physics_status(
    recipe_workspace, tmp_path, mug
):
    workspace, recipe_path = recipe_workspace
    data = mug.model_dump()
    data["scale_basis"] = "estimated"
    data["parts"] = data["parts"][:1]
    data["parts"][0]["geometry"] = {"kind": "box", "size": {"x": 0.06, "y": 0.05, "z": 0.08}}
    spec = SceneSpec.model_validate(data)
    write_json(workspace / "scene.json", spec)
    asset = build_asset(spec)
    points = asset.mesh().vertices
    rotation, _ = cv2.Rodrigues(np.array([0.2, 0.3, 0.1]))
    pixels, _ = project(
        points,
        {"rotation": rotation, "translation": [0, 0, 0.35], "focal_px": 200, "principal_point": [64, 48]},
    )
    source_points = (points - asset.metadata["translation_m"]) / asset.metadata["scale_factor"]
    recipe = json.loads(recipe_path.read_text())
    for entry in recipe["views"]:
        entry["landmarks"] = [
            {"xyz": xyz.tolist(), "uv": uv.tolist()} for xyz, uv in zip(source_points, pixels, strict=True)
        ]
    write_json(recipe_path, recipe)
    manifest = delight_frames(workspace, recipe_path, tmp_path / "delit", predictor=FakePredictor())
    # Whitespace changes must not make the copied provenance fail packaging.
    manifest.write_text(json.dumps(json.loads(manifest.read_text()), separators=(",", ":")))
    raw, corrected = tmp_path / "raw", tmp_path / "baked"
    bake_video_textures(workspace, recipe_path, raw, 128, fusion="best")
    bake_video_textures(workspace, recipe_path, corrected, 128, delighting_manifest=manifest, fusion="best")
    for filename in (
        "cameras.json",
        "coverage_000.png",
        "asset/asset_dynamic.usda",
        "asset/asset_static.usda",
        "asset/collision_scene.json",
    ):
        assert digest(raw / filename) == digest(corrected / filename)
    assert digest(raw / "texture_000.png") != digest(corrected / "texture_000.png")
    provenance = json.loads((corrected / "provenance.json").read_text())
    assert provenance["model_calls"] == 2 and provenance["bake_model_calls"] == 0
    assert provenance["geometry_changed"] is False and provenance["photometric_calibration"] is False
    assert provenance["delighting"]["manifest_sha256"] == digest(corrected / "delighting.json")
    metadata = json.loads((corrected / "asset/asset.json").read_text())
    assert metadata["isaac_physics_tested"] is False
    assert metadata["status"] == "needs_texture_review_and_physics_validation"
    publisher = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/assets/package_photo_textures.py"))
    assert publisher["verify_photo_provenance"](corrected, provenance) is True
    assert (
        publisher["verify_photo_provenance"](raw, json.loads((raw / "provenance.json").read_text())) is False
    )
    # A corrected candidate must still have its own real physics report before packaging.
    with pytest.raises(ValueError, match="physics_smoke"):
        publisher["package"](corrected, tmp_path / "candidate.zip")
    assert not (tmp_path / "candidate.zip").exists()
    provenance["model_calls"] = 0
    with pytest.raises(ValueError, match="inconsistent"):
        publisher["verify_photo_provenance"](corrected, provenance)
