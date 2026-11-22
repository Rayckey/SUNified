"""CPU-only tracking tests; numeric acceptance is not real-image accuracy."""

import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import trimesh
from PIL import Image

from kuafu.asset_generation import multiview
from kuafu.asset_generation.geometry import build_asset
from kuafu.asset_generation.io import digest, write_json
from kuafu.asset_generation.multiview import TrackConfig, camera_rejection, expand_photo_views, track_points, warp_masks
from kuafu.asset_generation.phototexture import project, read_mask
from kuafu.asset_generation.video import prepare


def test_bidirectional_flow_tracks_translation_and_never_revives_lost_points():
    rng = np.random.default_rng(19)
    source = rng.integers(0, 256, (160, 160), dtype=np.uint8)
    target = cv2.warpAffine(
        source, np.float32([[1, 0, 3], [0, 1, 2]]), (160, 160), borderMode=cv2.BORDER_REFLECT101
    )
    points = np.array([[64, 64], [96, 64], [64, 96], [96, 96]], np.float32)
    tracked, keep = track_points(source, target, points, [True, True, False, True])
    np.testing.assert_array_equal(keep, [True, True, False, True])
    np.testing.assert_allclose(tracked[keep], points[keep] + [3, 2], atol=0.15)


def test_bad_forward_backward_match_is_rejected(monkeypatch):
    count = 0

    def wrong_flow(previous, current, points, unused, **kwargs):
        nonlocal count
        count += 1
        return points + 10, np.ones((len(points), 1), np.uint8), None

    monkeypatch.setattr(cv2, "calcOpticalFlowPyrLK", wrong_flow)
    _, keep = track_points(np.zeros((100, 100), np.uint8), np.zeros((100, 100), np.uint8), [[40, 40]], [True])
    assert count == 2 and not keep.any()


def test_mask_transport_uses_backward_coordinates_and_preserves_holes():
    mask = np.zeros((48, 64), np.uint8)
    mask[8:36, 8:36] = 255
    mask[18:23, 18:23] = 0
    flow = np.zeros((48, 64, 2), np.float32)
    flow[..., 0] = -5
    result = warp_masks({"object": mask, "region": mask.copy()}, flow)
    np.testing.assert_array_equal(result["object"][:, 5:], mask[:, :-5])
    assert not result["object"][:, :5].any()
    np.testing.assert_array_equal(result["object"], result["region"])
    np.testing.assert_array_equal(np.unique(result["object"]), [0, 255])


def test_camera_quality_gates_both_error_and_focal_drift():
    seed = {"focal_px": 1000}
    config = TrackConfig()
    assert camera_rejection({"focal_px": 1200, "reprojection_rms_px": 3}, seed, 12, config) is None
    assert (
        camera_rejection({"focal_px": 1200, "reprojection_rms_px": 13}, seed, 12, config)
        == "reprojection_error"
    )
    assert camera_rejection({"focal_px": 1800, "reprojection_rms_px": 1}, seed, 12, config) == "focal_drift"
    assert camera_rejection({"focal_px": 500, "reprojection_rms_px": 1}, seed, 12, config) == "focal_drift"


@pytest.mark.parametrize(
    "change",
    [{"max_views": 13}, {"max_fb_error_px": 0}, {"min_mask_iou": float("nan")}, {"max_focal_ratio": 0.5}],
)
def test_invalid_tracking_config(change):
    with pytest.raises(ValueError):
        TrackConfig(**change)


@pytest.fixture
def tracking_workspace(tmp_path, video, holder, monkeypatch):
    root = tmp_path / "job"
    prepared = prepare(video, "synthetic tracking test", root, 3, 128)
    write_json(root / "scene.json", holder)
    mask = np.zeros((96, 128), np.uint8)
    mask[20:76, 25:101] = 255
    Image.fromarray(mask).save(root / "mask.png")
    asset = build_asset(holder)
    points = trimesh.creation.box(extents=[0.04, 0.04, 0.04]).vertices
    truth = {"rotation": np.eye(3), "translation": [0, 0, 0.4], "focal_px": 128, "principal_point": [64, 48]}
    pixels, _ = project(points * asset.metadata["scale_factor"] + asset.metadata["translation_m"], truth)
    part = asset.visuals[0].metadata["name"]
    recipe = root / "seeds.json"
    write_json(
        recipe,
        {
            "version": 1,
            "scene": "scene.json",
            "views": [
                {
                    "frame": prepared.frames[0].path,
                    "mask": "mask.png",
                    "landmarks": [
                        dict(xyz=p.tolist(), uv=q.tolist()) for p, q in zip(points, pixels, strict=True)
                    ],
                    "regions": {part: "mask.png"},
                    "occluders": [{"mask": "mask.png", "parts": [part]}],
                }
            ],
        },
    )
    monkeypatch.setattr(multiview, "track_points", lambda prev, cur, points, valid, error: (points, valid))
    monkeypatch.setattr(multiview, "refine_mask", lambda image, propagated: propagated)
    monkeypatch.setattr(multiview, "depth_buffer", lambda *args: np.where(mask > 0, 1.0, np.inf))
    flow = SimpleNamespace(calc=lambda *args: np.zeros((96, 128, 2), np.float32))
    monkeypatch.setattr(cv2, "DISOpticalFlow_create", lambda *args: flow)
    return root, recipe, prepared, part


def test_expansion_preserves_sources_masks_regions_and_provenance(tracking_workspace):
    root, recipe, prepared, part = tracking_workspace
    before = {f.path: digest(root / f.path) for f in prepared.frames}
    result = expand_photo_views(root, recipe, root / "expanded", TrackConfig(max_views=3))
    expanded = json.loads(result.read_text())
    report = json.loads((result.parent / "report.json").read_text())
    assert len(expanded["views"]) == 3
    assert report["seed_count"] == 1 and report["registered_views"] == 3
    assert report["model_calls"] == 0 and report["status"] == "needs_registration_and_mask_review"
    assert all(digest(root / f) == sha for f, sha in before.items())
    assert report["inputs"][str(recipe)] == digest(recipe)
    for view in expanded["views"][1:]:
        assert view["registration_source"].startswith("forward_backward")
        for annotation in (view["mask"], view["regions"][part], view["occluders"][0]["mask"]):
            np.testing.assert_array_equal(
                read_mask(root, annotation, (128, 96)), Image.open(root / "mask.png")
            )
        assert view["occluders"][0]["parts"] == [part]
    assert (result.parent / "registrations.jpg").is_file()
    with pytest.raises(ValueError, match="new or empty"):
        expand_photo_views(root, recipe, result.parent)


def test_mask_camera_disagreement_rejects_tracked_views(tracking_workspace, monkeypatch):
    root, recipe, _, _ = tracking_workspace
    monkeypatch.setattr(multiview, "depth_buffer", lambda *args: np.full((96, 128), np.inf))
    result = expand_photo_views(root, recipe, root / "rejected")
    assert len(json.loads(result.read_text())["views"]) == 1
    report = json.loads((result.parent / "report.json").read_text())
    assert all(r["reason"] == "mask_camera_disagreement" for r in report["rejected"])


@pytest.mark.parametrize("target", ["video", "frame", "duplicate", "outside", "ram"])
def test_invalid_inputs_fail_without_publishing_output(tracking_workspace, tmp_path, target):
    root, recipe, prepared, _ = tracking_workspace
    kwargs, config = {}, TrackConfig()
    out = root / "bad"
    if target == "video":
        wrong = tmp_path / "wrong.avi"
        wrong.write_bytes(b"not the original video")
        kwargs["video"] = wrong
    elif target == "frame":
        (root / prepared.frames[0].path).write_bytes(b"modified frame")
    elif target == "duplicate":
        kwargs["extra_anchors"] = recipe
    elif target == "outside":
        out = tmp_path / "outside"
    else:
        config = TrackConfig(max_decoded_frames=2)
    with pytest.raises(ValueError):
        expand_photo_views(root, recipe, out, config, **kwargs)
    assert not out.exists()


def test_mask_file_cannot_escape_workspace(tmp_path):
    with pytest.raises(ValueError):
        read_mask(tmp_path, "../outside.png", (32, 32))


def test_refinement_keeps_clear_object_and_excludes_background():
    image = np.full((96, 96, 3), [30, 60, 150], np.uint8)
    image[20:76, 20:76] = [220, 50, 40]
    mask = np.zeros((96, 96), np.uint8)
    mask[18:78, 18:78] = 255
    refined = multiview.refine_mask(image, mask)
    assert refined[30:60, 30:60].min() == 255
    assert not refined[:20].any() and not refined[:, :20].any()
