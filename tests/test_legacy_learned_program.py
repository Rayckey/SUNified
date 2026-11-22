import json

import pytest

from kuafu.paths import LEARNED_PROGRAM_ROOT
from kuafu.pipeline.pipeline import Pipeline


def valid(path, *, allow_legacy=False):
    return Pipeline._learned_program_is_valid(None, path, allow_legacy=allow_legacy)


@pytest.fixture
def legacy(tmp_path):
    stage = tmp_path / "program" / "subtask_0"
    stage.mkdir(parents=True)
    (stage / "task.txt").write_text("Stack the blocks.")
    (stage / "atomic.txt").write_text("Reach the block.")
    (stage / "constraints.json").write_text(json.dumps({"Constraints": [{"op": "reach_point"}]}))
    return tmp_path


def test_legacy_program_requires_explicit_selection(legacy):
    assert valid(legacy, allow_legacy=True)
    assert not valid(legacy)


def test_bundled_stack2_snapshot_is_usable():
    assert valid(LEARNED_PROGRAM_ROOT / "manipulation-stack2" / "2026-03-25_19-53-05", allow_legacy=True)


@pytest.mark.parametrize("file, content", [
    ("success.txt", "False"),
    ("stats.txt", "status: failed"),
    ("program/subtask_0/status.txt", "status: failed"),
    ("program/subtask_0/constraints.json", "{broken"),
    ("program/subtask_0/constraints.json", '{"Constraints": []}'),
    ("program/subtask_0/atomic.txt", ""),
])
def test_rejects_failed_or_incomplete_archive(legacy, file, content):
    (legacy / file).write_text(content)
    assert not valid(legacy, allow_legacy=True)


def test_rejects_missing_stage(legacy):
    (legacy / "program/subtask_0").rename(legacy / "program/subtask_1")
    assert not valid(legacy, allow_legacy=True)


def test_new_completed_formulation_still_accepted(legacy):
    (legacy / "success.txt").write_text("True")
    assert valid(legacy)
