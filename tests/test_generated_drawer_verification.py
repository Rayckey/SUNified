"""The physical acceptance check is independent of model-generated predicates."""

import json
from types import SimpleNamespace

import pytest
from kuafu.programs.generated_drawer import PreparedRequest, RequestBudget, physical_result


def state(top=0.0, left=0.0, right=0.0):
    return {
        "joint_names": ["lower_left_drawer_joint", "upper_drawer_joint", "lower_right_drawer_joint"],
        "cabinet_joints": [left, top, right],
    }


@pytest.mark.parametrize(
    "end,passed",
    [
        (state(top=0.1), True),
        (state(), False),
        (state(top=0.145), False),
        (state(top=0.1, left=0.02), False),
        (state(top=float("nan")), False),
    ],
)
def test_acceptance_requires_actual_requested_travel(end, passed):
    assert physical_result(state(), end, [end])["passed"] is passed


def test_other_drawer_disturbance_is_not_hidden_by_return_to_start():
    end = state(top=0.1)
    assert not physical_result(state(), end, [state(left=0.03), end])["passed"]


def test_request_budget_survives_controller_rebuild_and_process_resume(tmp_path):
    calls = []
    response = SimpleNamespace(model_dump=lambda: {"id": "test-response"})
    content = '{"tasks": ["open drawer"]}'

    def manager():
        return SimpleNamespace(
            _call_model=lambda _: calls.append(True) or (response, content),
            gpt_client=None,
            model_version="test",
            prompts=[],
        )

    budget = RequestBudget(tmp_path, 1)
    first = manager()
    budget.bind(first)
    actual_response, actual_content = first._call_model({})
    assert actual_response is response
    assert actual_content == content
    assert json.loads((tmp_path / "response_001.json").read_text()) == {"id": "test-response"}
    second = manager()
    RequestBudget(tmp_path, 1).bind(second)
    with pytest.raises(RuntimeError, match="budget exhausted"):
        second._call_model({})
    assert len(calls) == 1


def test_inspection_stops_without_spending_a_request(tmp_path):
    manager = SimpleNamespace(
        _call_model=lambda _: pytest.fail("Must not call model"), gpt_client=None, model_version="offline", prompts=[]
    )
    budget = RequestBudget(tmp_path, 1)
    budget.bind(manager, inspect=True)
    with pytest.raises(PreparedRequest):
        manager._call_model({})
    assert budget.used == 0
    assert (tmp_path / "prepared_request.json").exists()
