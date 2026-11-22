import json

import cv2
import pytest

from kuafu.asset_generation.io import local_file
from kuafu.asset_generation.video import prepare


def test_preparation_covers_timeline_records_hashes_and_applies_crop(video, tmp_path):
    out = tmp_path / "job"
    manifest = prepare(video, "the rigid blue block", out, 6, 128, [0.25, 0, 0.75, 1])
    assert len(manifest.frames) == 6
    assert manifest.frames[0].index < 5
    assert manifest.frames[-1].index >= 25
    assert [f.index for f in manifest.frames] == sorted(f.index for f in manifest.frames)
    assert manifest.duration_s == pytest.approx(3)
    assert all(len(f.sha256) == 64 for f in manifest.frames)
    image = cv2.imread(str(out / manifest.frames[0].path))
    assert image.shape[:2] == (96, 64)
    assert json.loads((out / "input.json").read_text())["description"] == "the rigid blue block"
    assert (out / "frames.jpg").is_file()


def test_preparation_never_overwrites_existing_output(video, tmp_path):
    out = tmp_path / "job"
    out.mkdir()
    sentinel = out / "keep.txt"
    sentinel.write_text("user data")
    with pytest.raises(ValueError, match="new or empty"):
        prepare(video, "object", out)
    assert sentinel.read_text() == "user data"


@pytest.mark.parametrize("crop", [[0, 0, 0, 1], [-0.1, 0, 1, 1], [0, 0, float("nan"), 1]])
def test_invalid_crops_fail_before_writing(video, tmp_path, crop):
    out = tmp_path / "job"
    with pytest.raises(ValueError, match="Crop"):
        prepare(video, "object", out, crop=crop)
    assert not out.exists()


def test_workspace_paths_cannot_escape(tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("outside")
    workspace = tmp_path / "job"
    workspace.mkdir()
    with pytest.raises(ValueError, match="unsafe"):
        local_file(workspace, "../outside.txt")
