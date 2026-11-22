import cv2
import numpy as np
import pytest

from kuafu.asset_generation.demo import demo_scene
from kuafu.asset_generation.schema import SceneSpec


@pytest.fixture
def mug():
    return demo_scene()


@pytest.fixture
def holder(mug):
    data = mug.model_dump()
    data["scale_basis"] = "estimated"
    data["parts"] = data["parts"][:1]
    data["parts"][0]["geometry"] = {
        "kind": "rounded_container",
        "size": {"x": 0.08, "y": 0.08, "z": 0.112},
        "corner_radius": 0.0125,
        "wall_thickness": 0.0029,
        "base_thickness": 0.0045,
        "bottom_inset": 0.0012,
        "edge_radius": 0.003,
    }
    data["parts"][0]["position"] = {"x": 0, "y": 0, "z": 0}
    data["parts"][0]["rotation_deg"] = {"x": 0, "y": 0, "z": 0}
    return SceneSpec.model_validate(data)


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "input.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (128, 96))
    assert writer.isOpened(), "MJPG encoding must be available for video integration tests"
    for i in range(30):
        frame = np.full((96, 128, 3), 230, dtype=np.uint8)
        cv2.rectangle(frame, (10 + i, 20), (45 + i, 75), (100, 40, 10), -1)
        cv2.putText(frame, str(i), (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        writer.write(frame)
    writer.release()
    return path
