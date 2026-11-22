"""Prompt cleanup must never turn ordinary model wording into a failed run."""

from copy import deepcopy
import json

from kuafu.common.prompt_context import content_for_model, text_for_model
from kuafu.skills.constraints.tolerance_policy import receive_constraint_geometry


def test_unrecognized_prose_does_not_block_geometry():
    # The first phrase is from the live drawer-stack failure.
    for phrase in (
        "within the commanded geometric tolerance of that axis-offset target",
        "within calibration tolerance",
        "with a completion threshold chosen by the controller",
        "within the commanded torlerance",
    ):
        text = f"Move 0.12 m along the handle's -y axis, {phrase}."
        assert text_for_model(text).startswith("Move 0.12 m along the handle's -y axis")


def test_structured_settings_are_removed_without_changing_geometry():
    geometry = {"offset_from_origin_xyz": [0.0, -0.12, 0.02]}
    payload = {"Constraints": [{"op": "reach_point", "args": {
        **geometry, "tolerance_m": 999.0,
    }}]}
    original = deepcopy(payload)
    cleaned = text_for_model("Program: " + json.dumps(payload))
    assert json.loads(cleaned.removeprefix("Program: ")) == {
        "Constraints": [{"op": "reach_point", "args": geometry}]
    }
    assert payload == original


def test_recognized_annotations_are_still_cleaned():
    assert text_for_model("Move 0.12 m (tolerance 0.01 m).") == "Move 0.12 m."


def test_multimodal_history_is_copied_without_changing_images():
    history = [{"role": "user", "content": [
        {"type": "text", "text": "Move 0.12 m (tolerance 0.01 m)."},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,example"}},
    ]}]
    original = deepcopy(history)
    cleaned = content_for_model(history)
    assert cleaned[0]["content"][0]["text"] == "Move 0.12 m."
    assert cleaned[0]["content"][1] == history[0]["content"][1]
    assert history == original


def test_receiver_ignores_model_authored_numeric_settings():
    payload = {"Constraints": [{
        "op": "freeze_orientation",
        "args": {
            "controlled_entity": "robot_frame",
            "controlled_target": "panda_fingertip",
            "tolerance_rad": 999.0,
        },
    }]}
    original = deepcopy(payload)
    received = receive_constraint_geometry(
        payload, atomic_action="Keep the gripper orientation fixed.",
        detectable_objects={},
    )
    assert received["Constraints"][0]["args"]["tolerance_rad"] == 0.2
    assert received["TolerancePolicy"]["source"] == "receiver"
    assert payload == original
