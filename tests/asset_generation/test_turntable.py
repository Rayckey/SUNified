import json
import shutil

import cv2
import numpy as np
import pytest
import trimesh
from PIL import Image

from kuafu.asset_generation.io import digest
from kuafu.asset_generation.turntable import load_preview, orbit_angles, render_turntable


@pytest.fixture
def visual(tmp_path):
    mesh = trimesh.creation.box(extents=[0.2, 0.1, 0.4])
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=np.zeros((len(mesh.vertices), 2)),
        material=trimesh.visual.material.PBRMaterial(
            baseColorTexture=Image.new("RGB", (4, 4), (220, 60, 30)),
            baseColorFactor=[1.0, 1.0, 1.0, 1.0],
        ),
    )
    mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    path = tmp_path / "source.glb"
    trimesh.Scene(mesh).export(path)
    return path


def test_preview_restores_z_up_and_texture_without_changing_source(visual):
    before = digest(visual)
    mesh = load_preview(visual)
    np.testing.assert_allclose(mesh.extents, [0.2, 0.1, 0.4], atol=1e-7)
    assert mesh.visual.kind == "texture"
    assert mesh.visual.material.baseColorTexture.getpixel((0, 0))[:3] == (220, 60, 30)
    assert digest(visual) == before


def test_orbit_is_one_turn_without_duplicate_endpoint():
    assert orbit_angles(4) == [-45, 45, 135, 225]
    with pytest.raises(ValueError):
        orbit_angles(1)


@pytest.mark.parametrize(
    "options",
    [{"size": 63}, {"seconds": float("nan")}, {"fps": 0}, {"workers": 0}, {"elevation": 90}],
)
def test_invalid_options_do_not_create_outputs(visual, tmp_path, options):
    out = tmp_path / "invalid"
    with pytest.raises(ValueError):
        render_turntable(visual, out, **options)
    assert not out.exists()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is needed for the real encoding test")
@pytest.mark.parametrize("workers", [1, 2])
def test_real_mp4_encode_and_no_overwrite(visual, tmp_path, workers):
    out = tmp_path / "turntable"
    report = render_turntable(visual, out, size=64, fps=4, seconds=1, workers=workers)
    assert report["frame_count"] == 4
    assert report["video_bytes"] < 20_000_000
    assert report["source_sha256"] == digest(visual)
    assert json.loads((out / "render.json").read_text()) == report
    for name, expected in report["files_sha256"].items():
        assert digest(out / name) == expected
    capture = cv2.VideoCapture(str(out / "turntable.mp4"))
    frames = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        capture.release()
    assert len(frames) == 4
    assert all(frame.shape == (64, 64, 3) for frame in frames)
    assert not np.array_equal(frames[0], frames[1])
    with pytest.raises(ValueError, match="new or empty"):
        render_turntable(visual, out, size=64, fps=4, seconds=1, workers=1)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is needed for the real encoding test")
def test_size_limit_does_not_publish_partial_video(visual, tmp_path):
    out = tmp_path / "oversized"
    with pytest.raises(ValueError, match="file size limit"):
        render_turntable(visual, out, size=64, fps=4, seconds=1, workers=1, max_bytes=1)
    assert not list(out.iterdir())
