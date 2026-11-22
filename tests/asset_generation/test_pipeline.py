import json

import pytest

from kuafu.asset_generation.demo import DemoModel
from kuafu.asset_generation.pipeline import reconstruct
from kuafu.asset_generation.schema import PhysicsConfig, Review, RunConfig, Scale, SceneSpec
from kuafu.asset_generation.video import prepare


def test_agent_refines_a_solid_cup_into_a_hollow_simulation_asset(video, tmp_path):
    job = tmp_path / "job"
    prepare(video, "blue mug, open interior", job, 4, scale=Scale(size_m=0.1, axis="z"))
    model = DemoModel()
    result = reconstruct(job, RunConfig(iterations=2), PhysicsConfig(mass_kg=0.25), model)
    assert result["status"] == "complete"
    assert result["best_iteration"] == 1
    assert result["api_calls"] == 0
    assert model.proposals == 2
    assert (job / "final/asset_dynamic.usda").is_file()
    assert (job / "final/viewer.html").is_file()
    assert json.loads((job / "final/scene.json").read_text())["parts"][0]["geometry"]["kind"] == "lathe"


class ScriptedModel:
    def __init__(self, scene, scores):
        self.scene, self.scores, self.requests = scene, iter(scores), 0

    def complete(self, schema, instructions, prompt, images):
        self.requests += 1
        assert all(image.is_file() for image in images)
        if schema is SceneSpec:
            return self.scene
        score = next(self.scores)
        if isinstance(score, Exception):
            raise score
        return Review(
            score=score, acceptable=False, issues=["needs work"], suggested_changes=[], uncertainty=[]
        )


def test_budget_exhaustion_keeps_best_candidate_not_last(video, tmp_path, mug):
    job = tmp_path / "job"
    prepare(video, "mug", job, 2)
    model = ScriptedModel(mug, [0.8, 0.4])
    result = reconstruct(job, RunConfig(iterations=2), PhysicsConfig(), model)
    assert result["status"] == "needs_review"
    assert result["best_iteration"] == 0
    assert result["review"]["score"] == 0.8
    assert model.requests == 4


def test_provider_failure_preserves_exportable_candidate(video, tmp_path, mug):
    job = tmp_path / "job"
    prepare(video, "mug", job, 2)
    model = ScriptedModel(mug, [RuntimeError("provider offline")])
    result = reconstruct(job, RunConfig(iterations=2), PhysicsConfig(), model)
    assert result["status"] == "needs_review"
    assert "provider offline" in result["error"]
    assert result["review"] is None
    assert (job / "final/asset_dynamic.usda").is_file()


def test_modified_frames_fail_before_any_api_call(video, tmp_path, mug):
    job = tmp_path / "job"
    prepared = prepare(video, "mug", job, 2)
    (job / prepared.frames[0].path).write_bytes(b"changed")
    model = ScriptedModel(mug, [0.5])
    with pytest.raises(ValueError, match="changed"):
        reconstruct(job, RunConfig(), PhysicsConfig(), model)
    assert model.requests == 0
