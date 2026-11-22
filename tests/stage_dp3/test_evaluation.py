from conftest import require_stage_runtime

require_stage_runtime()

import json

import numpy as np
import pytest

from conftest import make_policy
from kuafu.sim2real.dp3.evaluate import evaluate_experiment, rollout_episode, summarize_episodes, validate_protocol
from kuafu.sim2real.dp3.metrics import stage_metrics
from kuafu.sim2real.dp3.train import load_checkpoint, run_training
from kuafu.sim2real.dp3.isaac_adapter import recorded_tolerance_updates
from test_training import small_experiment


class AccountingAdapter:
    """Deterministic test fixture for episode accounting, not a robotics simulator."""

    def __init__(self, **kwargs):
        self.closed = False

    def reset(self, *, seed, condition):
        self.time = 0
        self.succeed = condition.get("succeed", True)
        return {"obs": self.observation(), "stage_id": 0, "initial_state_id": str(seed)}

    def observation(self):
        return {"point_cloud": np.full((12, 3), self.time / 10, dtype=np.float32),
                "agent_pos": np.full(10, self.time / 10, dtype=np.float32)}

    def step(self, action):
        assert action.shape == (10,)
        self.time += 1
        done = self.time == 3
        return {"obs": self.observation(), "stage_id": min(self.time, 2), "done": done,
                "success": done and self.succeed}

    def close(self):
        self.closed = True


def test_rollout_counts_failures_time_limits_and_separates_monitor_input():
    policy = make_policy("B")
    adapter = AccountingAdapter()
    options = dict(policy=policy, adapter=adapter, seed=10, chunk_exec=1)
    success = rollout_episode(**options, condition={}, max_steps=5)
    failure = rollout_episode(**options, condition={"succeed": False}, max_steps=5)
    timeout = rollout_episode(**options, condition={}, max_steps=1)
    assert [episode["success"] for episode in (success, failure, timeout)] == [True, False, False]
    assert failure["failure_stage"] == 2 and timeout["failure_stage"] == 1
    assert failure["stage"]["support"] == [1, 1, 1]
    summary = summarize_episodes([success, failure, timeout], 3)
    assert summary["successes"] == 1 and summary["attempts"] == 3
    assert summary["transitions"][2]["episode_failed_before_advancing"] == 1
    assert summary["stage"]["support"] == [3, 2, 2]


def test_macro_accuracy_is_not_dominated_by_long_stage():
    metrics = stage_metrics([[90, 10, 0], [0, 1, 1], [0, 0, 0]])
    assert metrics["per_stage_accuracy"] == [0.9, 0.5, None]
    assert metrics["macro_accuracy"] == 0.7
    assert metrics["accuracy"] == 91 / 102


def test_protocol_requires_explicit_nominal_and_varied_conditions():
    protocol = {"task": "test-task", "conditions": {"nominal": {}}, "seeds": [1], "max_steps": 4, "chunk_exec": 1}
    with pytest.raises(ValueError, match="controlled"):
        validate_protocol(protocol)
    protocol["conditions"]["varied"] = {"succeed": False}
    validate_protocol(protocol)
    protocol["seeds"] = [1, 1]
    with pytest.raises(ValueError, match="distinct"):
        validate_protocol(protocol)


def test_full_evaluation_uses_matched_checkpoints_and_conditions(tmp_path):
    dataset, config = small_experiment(tmp_path)
    run = tmp_path / "run"
    run_training(config, dataset, run)
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({"task": "test-task", "conditions": {"nominal": {}, "varied": {"succeed": False}},
                                    "seeds": [400, 401], "max_steps": 4, "chunk_exec": 1}))
    result = evaluate_experiment(run, dataset, protocol, tmp_path / "eval",
                                 adapter_factory="test_evaluation:AccountingAdapter", device="cpu")
    assert len(result["runs"]) == 6
    for row in result["runs"]:
        assert row["rollout"]["attempts"] == 2
        assert row["rollout"]["successes"] == (2 if row["condition"] == "nominal" else 0)
        assert row["offline_heldout"]["setting"] == "offline_test"
        assert row["rollout"]["stage"] is None if row["variant"] == "A" else row["rollout"]["stage"] is not None

    import torch
    latest = run / "seed_42/C/latest.ckpt"
    checkpoint = load_checkpoint(latest)
    checkpoint["experiment"]["epoch"] = 0
    torch.save(checkpoint, latest)
    with pytest.raises(ValueError, match="complete training budget"):
        evaluate_experiment(run, dataset, protocol, tmp_path / "incomplete",
                            adapter_factory="unavailable_adapter:create", device="cpu")
    assert not (tmp_path / "incomplete").exists()


def test_transition_on_terminal_step_survives_simulator_auto_reset():
    class TerminalAdapter(AccountingAdapter):
        def step(self, action):
            return {"obs": None, "stage_id": -1, "done": True, "success": False, "advanced_stages": [0]}
    episode = rollout_episode(make_policy("A"), TerminalAdapter(), seed=1, condition={}, max_steps=2, chunk_exec=1)
    assert episode["failure_stage"] == 1
    assert episode["transitions"][0]["reached_later_stage_or_final_success"] == 1
    assert episode["transitions"][1]["episode_failed_before_advancing"] == 1


def test_recorded_tolerances_restore_teacher_values_without_changing_task():
    import copy
    current = [{"name": "reach", "constraints": [{"op": "reach_point",
        "args": {"reference_origin": "handle", "tolerance_m": 0.01}, "condition": ["geometry"]}]}]
    recorded = copy.deepcopy(current)
    recorded[0]["constraints"][0]["args"]["tolerance_m"] = 0.03
    assert recorded_tolerance_updates(current, recorded) == {0: {0: {"tolerance_m": 0.03}}}
    assert current[0]["constraints"][0]["args"]["tolerance_m"] == 0.01
    for key, value in (("reference_origin", "different_object"), ("travel_distance_m", 0.5)):
        changed = copy.deepcopy(recorded)
        changed[0]["constraints"][0]["args"][key] = value
        with pytest.raises(ValueError, match="beyond recorded tolerances"):
            recorded_tolerance_updates(current, changed)
    recorded[0]["constraints"][0]["args"]["tolerance_m"] = float("nan")
    with pytest.raises(ValueError, match="finite scalars"):
        recorded_tolerance_updates(current, recorded)
