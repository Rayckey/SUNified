import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from kuafu.asset_generation.cli import main
from kuafu.asset_generation.sam3d import (
    polygon_mask,
    prepare_sam3d,
    run_sam3d,
    validate_job,
    validate_mask,
    worker_environment,
)
from kuafu.asset_generation.sam3d_worker import check_targets
from kuafu.asset_generation.video import prepare


@pytest.fixture
def sam_job(video, tmp_path):
    workspace = tmp_path / "prepared"
    prepare(video, "the moving rigid block", workspace, 4, 128)
    polygon = tmp_path / "polygon.json"
    polygon.write_text(json.dumps({"include": [[[8, 18], [55, 18], [55, 78], [8, 78]]]}))
    return prepare_sam3d(workspace, 0, polygon, tmp_path / "masked")


def test_mask_preparation_preserves_pixels_and_records_no_inference(sam_job):
    job = validate_job(sam_job)
    assert job["status"] == "prepared_not_inferred"
    assert job["backend"] == "facebook/sam-3d-objects"
    assert job["mask_source"] == "manual_polygon"
    assert job["selected_frame_ordinal"] == 0
    assert job["description"] == "the moving rigid block"
    rgb = np.array(Image.open(sam_job.parent / "image.png"))
    cutout = np.array(Image.open(sam_job.parent / "cutout.png"))
    assert np.array_equal(rgb, cutout[:, :, :3])
    assert np.array_equal(cutout[:, :, 3], Image.open(sam_job.parent / "mask.png"))
    assert (sam_job.parent / "mask_review.jpg").is_file()


def test_changed_mask_is_rejected(sam_job):
    Image.new("L", (128, 96), 255).save(sam_job.parent / "mask.png")
    with pytest.raises(ValueError, match="mask changed"):
        validate_job(sam_job)


def test_mask_review_and_weights_are_required(sam_job, tmp_path):
    out = tmp_path / "candidate"
    with pytest.raises(ValueError, match="mask-reviewed"):
        run_sam3d(sam_job, out, tmp_path / "runtime", 0, False)
    with pytest.raises(RuntimeError, match="checkpoints are unavailable"):
        run_sam3d(sam_job, out, tmp_path / "runtime", 0, True)
    assert not out.exists()


def test_polygon_exclusions_preserve_real_holes():
    mask = polygon_mask(
        (64, 64),
        {
            "include": [[[5, 5], [59, 5], [59, 59], [5, 59]]],
            "exclude": [[[20, 20], [40, 20], [40, 40], [20, 40]]],
        },
    )
    validate_mask(mask, (64, 64))
    assert mask[10, 10] == 255 and mask[30, 30] == 0 and mask[0, 0] == 0


@pytest.mark.parametrize(
    "polygon",
    [
        [],
        [[1, 2], [3, 4]],
        [[-1, 2], [3, 4], [5, 6]],
        [[1, 2], [3, 4], [64, 6]],
        [[1, 2], [3, 4], [float("nan"), 6]],
    ],
)
def test_invalid_polygons_rejected(polygon):
    with pytest.raises(ValueError):
        polygon_mask((64, 64), {"include": [polygon]})


@pytest.mark.parametrize(
    "mask",
    [
        np.ones((20, 20), np.uint8),
        np.zeros((20, 20), np.uint8),
        np.full((20, 20), 255, np.uint8),
        np.zeros((20, 20, 3), np.uint8),
    ],
)
def test_invalid_binary_masks_rejected(mask):
    with pytest.raises(ValueError):
        validate_mask(mask, (20, 20))


def test_runtime_isolation_keeps_local_authentication(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/fake/isaac")
    monkeypatch.setenv("VIRTUAL_ENV", "/fake/host")
    monkeypatch.setenv("HF_HOME", "/fake/user/auth")
    monkeypatch.setenv("HF_TOKEN", "unit-test-placeholder-not-a-secret")
    env = worker_environment(Path("/tmp/sam3d_test"), 3)
    assert env["PYTHONPATH"] == "/tmp/sam3d_test/repo"
    assert "VIRTUAL_ENV" not in env
    assert env["CUDA_VISIBLE_DEVICES"] == "3"
    assert env["HF_HOME"] == "/fake/user/auth"
    assert env["HF_TOKEN"] == "unit-test-placeholder-not-a-secret"
    assert env["HF_HUB_CACHE"].startswith("/tmp/sam3d_test/")
    assert env["ATTN_BACKEND"] == "xformers"


@pytest.mark.parametrize("target", ["os.system", "builtins.eval", "torch.__dict__", "subprocess.Popen"])
def test_hydra_targets_restricted(target):
    with pytest.raises(ValueError, match="Unapproved Hydra"):
        check_targets({"children": [{"_target_": target}]})
    check_targets(
        {"_target_": "sam3d_objects.pipeline.inference_pipeline_pointmap.InferencePipelinePointMap"}
    )


def test_cli_missing_weights_is_an_error_not_a_fake_asset(sam_job, tmp_path, capsys):
    out = tmp_path / "candidate"
    assert (
        main(
            [
                "sam3d-run",
                str(sam_job),
                "--mask-reviewed",
                "--root",
                str(tmp_path / "runtime"),
                "--out",
                str(out),
            ]
        )
        == 1
    )
    assert "checkpoints are unavailable" in capsys.readouterr().err
    assert not out.exists()
