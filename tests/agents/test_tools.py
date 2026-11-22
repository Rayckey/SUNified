"""External proposals must receive honest feedback before they become runnable program."""

import copy
import os
import shutil
import sys
import time
from pathlib import Path

import pytest
from kuafu.agents import ToolService, jobs


@pytest.fixture
def inspected(tmp_path):
    service = ToolService(tmp_path, isaac_python=sys.executable)
    asset = tmp_path / "asset"
    asset.mkdir()
    for name in ("asset_dynamic.usda", "asset_geometry.usda", "semantics.json"):
        (asset / name).write_text("{}")
    context = {
        "asset": str(asset),
        "asset_hashes": jobs.asset_hashes(asset),
        "task": "Open top drawer 10 cm",
        "target_travel_m": 0.1,
        "position": [0.6, -0.5, 0.2],
        "yaw_deg": 180.0,
    }
    directory = service._new("jobs")
    (directory / "output").mkdir()
    jobs.write_json(directory / "request.json", {"kind": "inspect", "context": context})
    jobs.write_json(directory / "status.json", {"status": "succeeded"})
    jobs.write_json(
        directory / "output/scene.json",
        {
            "initial": {"frames": {"panda_fingertip": {}, "top_drawer_handle_frame": {}}},
            "descriptions": {"top_drawer_handle_frame": "A handle on a sliding drawer"},
            "asset_hashes": context["asset_hashes"],
        },
    )
    proposal = {
        "stages": [
            {
                "task": "Approach the handle",
                "atomic_actions": ["Approach 10 cm before the handle"],
                "constraints": {
                    "Reasoning": "Keep a standoff before contact",
                    "Constraints": [
                        {
                            "logic": "and",
                            "condition": ["geometry"],
                            "op": "reach_point",
                            "args": {
                                "controlled_entity": "robot_frame",
                                "controlled_target": "panda_fingertip",
                                "reference_basis": "object",
                                "reference_origin": "top_drawer_handle_frame",
                                "offset_from_origin_xyz": [0, -0.1, 0],
                            },
                        }
                    ],
                },
            }
        ]
    }
    return service, directory, proposal, context


def test_invalid_frame_feedback_can_be_repaired_without_marking_success(inspected):
    service, context, proposal, _ = inspected
    invalid = copy.deepcopy(proposal)
    invalid["stages"][0]["constraints"]["Constraints"][0]["args"]["reference_origin"] = "imagined_handle"
    failed = service.call("kuafu_submit_program", {"context_job": context.name, "program": invalid})
    assert not failed["ok"] and "unknown reference_origin" in failed["error"]["message"]
    accepted = service.call("kuafu_submit_program", {"context_job": context.name, "program": proposal})
    assert accepted["ok"], accepted
    path = Path(accepted["result"]["directory"])
    assert (path / "success.txt").read_text().strip() == "False"
    normalized = jobs.read_json(path / "program/subtask_0/constraints.json")
    assert normalized["TolerancePolicy"]["source"] == "receiver"
    assert normalized["Constraints"][0]["args"]["tolerance_m"] > 0
    assert "TolerancePolicy" not in proposal["stages"][0]["constraints"]


@pytest.mark.parametrize("change", ["tolerance", "code", "nan", "world_basis", "entity"])
def test_agent_cannot_inject_code_thresholds_or_invalid_geometry(inspected, change):
    service, context, proposal, _ = inspected
    term = proposal["stages"][0]["constraints"]["Constraints"][0]
    if change == "tolerance":
        term["args"]["tolerance_m"] = 100
    elif change == "code":
        term["op"] = "__import__('os').system"
    elif change == "nan":
        term["args"]["offset_from_origin_xyz"][0] = float("nan")
    elif change == "world_basis":
        term["args"]["reference_basis"] = "world"
    else:
        term["args"]["controlled_entity"] = "gripped_object"
    result = service.call("kuafu_submit_program", {"context_job": context.name, "program": proposal})
    assert not result["ok"]
    assert not (service.root / "programs").exists()


def test_changed_asset_requires_a_new_inspection(inspected):
    service, context, proposal, task = inspected
    Path(task["asset"], "semantics.json").write_text('{"changed":true}')
    result = service.call("kuafu_submit_program", {"context_job": context.name, "program": proposal})
    assert not result["ok"] and "changed" in result["error"]["message"]


@pytest.mark.parametrize("outcome", ["physical_failure", "pass", "missing_report", "changed_program", "changed_asset"])
def test_promotion_requires_physical_success_and_unchanged_inputs(inspected, outcome):
    service, context_job, proposal, context = inspected
    submitted = service.call("kuafu_submit_program", {"context_job": context_job.name, "program": proposal})
    program = Path(submitted["result"]["directory"])
    directory = service._new("jobs")
    shutil.copytree(program / "program", directory / "input/program")
    (directory / "output").mkdir()
    jobs.write_json(directory / "output/scene.json", {"asset_hashes": context["asset_hashes"]})
    physical_pass = outcome != "physical_failure"
    jobs.write_json(
        directory / "output/verification.json",
        {
            "passed": physical_pass,
            "program_passed": True,
            "solver_passed": True,
            "physical": {"passed": physical_pass},
            "model_requests": 0,
            "replay_stages": [],
        },
    )
    request = {"kind": "verify", "context": context, "program_hashes": jobs.tree_hashes(directory / "input/program")}
    if outcome in {"missing_report", "changed_program", "changed_asset"}:
        if outcome == "missing_report":
            (directory / "output/verification.json").unlink()
        elif outcome == "changed_program":
            (directory / "input/program/subtask_0/task.txt").write_text("Modified after execution")
        else:
            Path(context["asset"], "semantics.json").write_text('{"changed":true}')
        with pytest.raises(ValueError, match="missing|changed"):
            jobs.finish(directory, request, 0, None)
        assert not (directory / "accepted").exists()
        return
    if not physical_pass:
        result = jobs.finish(directory, request, 0, None)
        assert result["status"] == "failed"
        assert result["verification"]["physical"]["passed"] is False
        assert not (directory / "accepted").exists()
    else:
        result = jobs.finish(directory, request, 0, None)
        accepted = Path(result["accepted_run"])
        assert result["accepted_identity"]["program_path"] == str(accepted)
        assert len(result["accepted_identity"]["program_sha256"]) == 64
        assert (accepted / "success.txt").read_text().strip() == "True"
        assert "succeeded" in (accepted / "program/subtask_0/status.txt").read_text()
        assert (program / "success.txt").read_text().strip() == "False"


@pytest.mark.parametrize("cancel", [False, True])
def test_job_timeout_and_cancellation_stop_owned_process(tmp_path, cancel):
    if cancel:
        (tmp_path / "cancel.request").touch()
    code, reason = jobs.run_process(
        [sys.executable, "-c", "import time; time.sleep(30)"], tmp_path, 0.05, dict(os.environ)
    )
    assert code != 0
    assert reason == ("cancelled" if cancel else "timed_out")


def test_job_failure_is_persistent_after_client_disconnect(inspected, tmp_path):
    service, _, _, context = inspected
    runtime = tmp_path / "test-runtime"
    runtime.write_text("#!/bin/sh\necho 'test runtime unavailable'\nexit 9\n")
    runtime.chmod(0o755)
    service.isaac_python = str(runtime)
    result = service.call(
        "kuafu_inspect_drawer",
        {
            "asset": context["asset"],
            "task": context["task"],
            "target_travel_m": 0.1,
        },
    )
    assert result["ok"], result
    # A different service instance models a new CLI process / MCP connection.
    reader = ToolService(tmp_path)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        state = reader.call("kuafu_job_status", {"job_id": result["result"]["job_id"]})["result"]
        if state["status"] not in {"queued", "running"}:
            break
        time.sleep(0.1)
    assert state["status"] == "failed", state
    assert state["returncode"] == 9
    assert "test runtime unavailable" in Path(state["log"]).read_text()


def test_artifact_reader_rejects_symlink_escape(tmp_path):
    root = tmp_path / "artifacts"
    root.mkdir()
    private = tmp_path / "outside.txt"
    private.write_text("outside")
    (root / "link.txt").symlink_to(private)
    result = ToolService(root).call("kuafu_read_artifact", {"path": str(root / "link.txt")})
    assert not result["ok"]


def test_unknown_tool_is_structured_error(tmp_path):
    result = ToolService(tmp_path).call("run_arbitrary_shell", {"command": "ignored"})
    assert result["protocol_version"] == 1 and not result["ok"]


@pytest.mark.parametrize("setting", ["learn_programs", "acquire_weights", "skip_parameter_search", "hparam_iterations"])
def test_stage_tools_reject_implicit_builtin_model_search(tmp_path, setting):
    root = Path(__file__).resolve().parents[2]
    text = (root / "assets/configs/fr3_finray_drawer.example.toml").read_text()
    stage = "eval-prior"
    if setting == "learn_programs":
        text = text.replace("enabled = false", "enabled = true")
        stage = "collect"
    elif setting == "skip_parameter_search":
        text = text.replace("acquire_weights = true", "acquire_weights = false")
    elif setting == "hparam_iterations":
        stage = "train-residual"
    config = tmp_path / "pipeline.toml"
    config.write_text(text)
    result = ToolService(tmp_path).call("kuafu_start_stage", {"config": str(config), "stage": stage})
    assert not result["ok"] and setting in result["error"]["message"]
    assert not (tmp_path / "agents/jobs").exists()
