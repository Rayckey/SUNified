import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from kuafu.artifact_contracts import file_sha256, learned_program_identity
from kuafu.rl.isaac.evaluation_stats import load_bc_eval_stats


@pytest.fixture
def evaluation(tmp_path, monkeypatch):
    monkeypatch.setenv("KUAFU_ARTIFACT_ROOT", str(tmp_path))
    monkeypatch.delenv("KUAFU_PROGRAM_PATH", raising=False)
    checkpoint = tmp_path / "best.pth"
    checkpoint.write_bytes(b"test checkpoint")
    payload = {
        "checkpoint": str(checkpoint), "checkpoint_sha256": file_sha256(checkpoint),
        "task": "test-task", "total_episodes": 8,
        "program_path": "/test/program", "program_sha256": "program-hash",
        "constraint_met_rate_stats": {"0": {"2": {"met": 3, "total": 8, "rate": 0.375, "reward": 0.5}}},
        "stage_transition_stats": {"0": {"count": 8}},
    }
    return tmp_path / "stats.json", checkpoint, payload


def test_saved_eval_restores_numeric_keys(evaluation):
    path, checkpoint, payload = evaluation
    path.write_text(json.dumps(payload))
    result = load_bc_eval_stats("stats.json", prior_checkpoint=checkpoint, task="test-task")
    assert result["constraint_met_rate_stats"][0][2]["total"] == 8
    assert result["stage_transition_stats"][0]["count"] == 8
    assert result["completed_episodes"] == 8


@pytest.mark.parametrize("changes, message", [
    ({"checkpoint": "/tmp/different.pth"}, "checkpoint"),
    ({"checkpoint": None}, "checkpoint"),
    ({"task": "wrong-task"}, "task"),
    ({"checkpoint_sha256": "old-content"}, "content has changed"),
    ({"total_episodes": 0}, "completed episodes"),
])
def test_rejects_unrelated_or_incomplete_eval(evaluation, changes, message):
    path, checkpoint, payload = evaluation
    path.write_text(json.dumps({**payload, **changes}))
    with pytest.raises(ValueError, match=message):
        load_bc_eval_stats(path, prior_checkpoint=checkpoint, task="test-task")


def test_rejects_changed_learned_program(evaluation):
    path, checkpoint, payload = evaluation
    path.write_text(json.dumps(payload))
    expected = {"program_path": "/test/program", "program_sha256": "new-program"}
    with pytest.raises(ValueError, match="learned program"):
        load_bc_eval_stats(path, prior_checkpoint=checkpoint, task="test-task", expected_program=expected)


def test_weight_sweep_reuses_saved_rollout_and_records_usage(evaluation, monkeypatch):
    from kuafu.rl.isaac import initial_weight_sweep as sweep
    from kuafu.rl.llm.client import LlmResponse

    path, checkpoint, payload = evaluation
    program = path.parent / "program"
    (program / "program").mkdir(parents=True)
    (program / "program/program.json").write_text("{}")
    payload.update(learned_program_identity(program))
    payload.update({"stage_count": 1, "highest_stage_reached_per_env": [0, 1, 1]})
    path.write_text(json.dumps(payload))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    observed = {}
    env = SimpleNamespace(
        unwrapped=SimpleNamespace(cfg=SimpleNamespace(mpc_config={"load_program": str(program)})),
        close=lambda: observed.update(env_closed=True),
    )
    app = SimpleNamespace(close=lambda: observed.update(app_closed=True))
    def create_env(**kwargs):
        observed["num_envs"] = kwargs["num_envs"]
        return app, env, {}
    monkeypatch.setattr(sweep, "create_env_and_app", create_env)
    monkeypatch.setattr(sweep, "evaluate_bc_prior_policy_rollout", lambda **_: pytest.fail("Saved evaluation must avoid a new rollout"))
    monkeypatch.setattr(sweep, "_collect_task_info", lambda *_, **__: {
        "task_description": "Open the gripper", "detectable_objects": {}, "stage_count": 1,
        "stages": {0: {"weights": {0: 1.0}, "parameters": {0: {}},
                       "constraints_raw": [{"op": "open_gripper", "args": {}}],
                       "explanations": ["Open gripper"], "atomic_action": "[Release] Open the gripper"}},
    })
    monkeypatch.setattr(sweep, "_apply_task_info_to_env", lambda *_: None)
    def format_times(*_, **kwargs):
        observed["stats_num_envs"] = kwargs["num_env"]
        return "saved transition statistics"
    monkeypatch.setattr(sweep, "format_stage_satisfaction_times", format_times)
    class FakeClient:
        def __init__(self, **kwargs):
            observed["reasoning_effort"] = kwargs["reasoning_effort"]

        def prompt(self, **kwargs):
            return LlmResponse(messages=[], raw_outputs=[json.dumps({
                "Diagnosis": "test", "Reasoning": "test", "Weights": {"0": 1.0},
            })], usage={"total_tokens": 9})
    monkeypatch.setattr(sweep, "OpenAIChatClient", FakeClient)
    args = sweep._build_arg_parser().parse_args([
        "--task", "test-task", "--prior_checkpoint", str(checkpoint),
        "--bc_eval_stats_path", str(path), "--reasoning_effort", "high",
        "--execution_robot_id", "fr3_finray",
    ])
    result = Path(sweep.run_initial_weight_sweep(args))
    assert observed == {"num_envs": 1, "stats_num_envs": 3, "reasoning_effort": "high", "env_closed": True, "app_closed": True}
    summary = json.loads((result / "best.json").read_text())
    assert summary["program_sha256"] == payload["program_sha256"]
    logged = list(result.rglob("*payload*.jsonl"))
    assert len(logged) == 1
    assert json.loads(logged[0].read_text().splitlines()[0])["usage"] == {"total_tokens": 9}
