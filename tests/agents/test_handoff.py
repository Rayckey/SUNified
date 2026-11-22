"""Exercise failures, client reconnects and revision preservation across handoffs."""

import json
import os
from pathlib import Path

import pytest

from kuafu.agents import ToolService
from kuafu.agents import jobs
from kuafu.pipeline.handoff import HandoffRun


def create(tmp_path):
    video = tmp_path / "cabinet.mp4"
    video.write_bytes(b"original video input")
    return HandoffRun.create(tmp_path / "artifacts", task="Open the top drawer by 10 cm.",
                             description="A cabinet with three sliding drawers.",
                             videos=[video], run_id="drawer")


def test_inputs_survive_original_file_changes(tmp_path):
    run = create(tmp_path)
    (tmp_path / "cabinet.mp4").write_bytes(b"edited")
    state = run.status()
    assert (run.root / state["manifest"]["inputs"][0]["path"]).read_bytes() == b"original video input"
    assert state["attempts"] == []


def test_failed_step_can_be_retried_after_reconnect(tmp_path, monkeypatch):
    run = create(tmp_path)
    service = ToolService(tmp_path / "artifacts", run_id=run.run_id)

    def broken(self, scene):
        directory = self._new("assets")
        (directory / "partial.json").write_text('{"built": false}')
        raise RuntimeError("mesh construction failed")

    monkeypatch.setattr(ToolService, "_kuafu_build_asset", broken)
    response = service.call("kuafu_build_asset", {"scene": {"name": "drawer"}})
    assert not response["ok"]
    failed = run.status()["steps"]["build-asset"]
    assert failed["status"] == "failed"
    assert "mesh construction failed" in (Path(failed["directory"]) / "tool.log").read_text()
    old_files = json.loads((Path(failed["directory"]) / "artifacts.json").read_text())
    assert any(p.endswith("partial.json") for p in old_files)

    def repaired(self, scene):
        assert scene == {"name": "drawer"}
        directory = self._new("assets")
        (directory / "asset.usda").write_text("#usda 1.0\n")
        return {"asset": str(directory)}

    monkeypatch.setattr(ToolService, "_kuafu_build_asset", repaired)
    reconnected = ToolService(tmp_path / "artifacts")
    response = reconnected.call("kuafu_retry_step", {"run_id": run.run_id, "attempt_id": failed["attempt_id"]})
    assert response["ok"], response
    state = run.status()
    assert [a["status"] for a in state["attempts"]] == ["failed", "complete"]
    assert state["attempts"][0]["directory"] != state["attempts"][1]["directory"]
    assert json.loads((Path(failed["directory"]) / "artifacts.json").read_text()) == old_files
    asset = Path(response["result"]["asset"])
    assert run.lookup("assets", asset.name) == asset


def test_live_step_rejects_duplicate_and_detects_interruption(tmp_path, monkeypatch):
    run = create(tmp_path)
    attempt = run.begin("kuafu_build_asset", {"scene": {}})
    service = ToolService(tmp_path / "artifacts", run_id=run.run_id)
    duplicate = service.call("kuafu_build_asset", {"scene": {}})
    assert not duplicate["ok"] and "already running" in duplicate["error"]["message"]
    monkeypatch.setattr(jobs, "process_token", lambda pid: None)
    assert run.status()["steps"]["build-asset"]["status"] == "interrupted"
    assert run.retry_request(attempt.name)["arguments"] == {"scene": {}}


def test_completed_job_is_discoverable_after_client_disconnect(tmp_path):
    run = create(tmp_path)
    attempt = run.begin("kuafu_inspect_drawer", {"asset": "cabinet", "task": "Open", "target_travel_m": .1})
    directory = run.allocate(attempt, "jobs")
    jobs.write_json(directory / "status.json", {"status": "succeeded", "returncode": 0})
    run.finish(attempt, {"ok": True, "result": {"job_id": directory.name, "directory": str(directory)}})
    service = ToolService(tmp_path / "artifacts")
    state = service.call("kuafu_run_status", {"run_id": run.run_id})
    assert state["ok"]
    assert state["result"]["steps"]["inspect"]["status"] == "complete"
    assert service.call("kuafu_job_status", {"run_id": run.run_id, "job_id": directory.name})["result"]["status"] == "succeeded"


def test_upstream_revision_marks_downstream_result_stale(tmp_path):
    run = create(tmp_path)
    for tool in ["kuafu_build_asset", "kuafu_inspect_drawer", "kuafu_build_asset"]:
        attempt = run.begin(tool, {})
        run.finish(attempt, {"ok": True, "result": {}})
    state = run.status()
    assert state["steps"]["inspect"]["stale"]
    assert "stale" not in state["steps"]["build-asset"]


def test_changed_output_invalidates_handoff_and_downstream(tmp_path):
    run = create(tmp_path)
    attempt = run.begin("kuafu_build_asset", {})
    asset = run.allocate(attempt, "assets") / "scene.json"
    asset.write_text('{}')
    run.finish(attempt, {"ok": True, "result": {}})
    inspection = run.begin("kuafu_inspect_drawer", {})
    run.finish(inspection, {"ok": True, "result": {}})
    asset.write_text('{"changed": true}')
    state = run.status(verify_outputs=True)
    assert state["steps"]["build-asset"]["stale"]
    assert state["steps"]["inspect"]["stale"]


def test_job_is_recovered_when_agent_exits_before_returning(tmp_path, monkeypatch):
    run = create(tmp_path)
    attempt = run.begin("kuafu_inspect_drawer", {})
    directory = run.allocate(attempt, "jobs")
    run.attach_job(attempt, directory)
    monkeypatch.setattr(jobs, "process_token", lambda pid: None)
    assert run.status()["steps"]["inspect"]["status"] == "interrupted"
    jobs.write_json(directory / "status.json", {"status": "succeeded", "returncode": 0})
    state = run.status()["steps"]["inspect"]
    assert state["status"] == "complete"
    assert state["job_id"] == directory.name
    response = json.loads((attempt / "response.json").read_text())
    assert response["recovered"] and response["result"]["job_id"] == directory.name


def manifest(tmp_path):
    source = Path(__file__).resolve().parents[2] / "assets/configs/fr3_finray_drawer.example.toml"
    text = source.read_text().replace("acquire_weights = true", "acquire_weights = false")
    text = text.replace("skip_parameter_search = false", "skip_parameter_search = true")
    path = tmp_path / "task.toml"
    path.write_text(text)
    return path


def test_learning_revisions_preserve_upstream_and_previous_outputs(tmp_path, monkeypatch):
    from kuafu.pipeline.config import PipelineConfig
    from kuafu.pipeline.pipeline import Pipeline
    from kuafu.pipeline.state import StateStore

    run = create(tmp_path)
    config = manifest(tmp_path)
    service = ToolService(tmp_path / "artifacts", run_id=run.run_id)
    outputs = []

    def launch(directory, request):
        cfg = PipelineConfig.load(directory / "pipeline.toml")
        root = cfg.task_artifact_root
        state = StateStore(root / "pipeline_state.json", cfg.task_slug)
        step = request["stage"]
        if step == "collect":
            output = root / "demonstrations.hdf5"
            output.write_bytes(b"recorded transitions")
            artifacts = {"dataset": str(output)}
        elif step == "train-prior":
            assert Path(state.artifacts["dataset"]).read_bytes() == b"recorded transitions"
            output = root / "prior.csv"
            output.write_text("checkpoint,success\npolicy.pth,\n")
            artifacts = {"dataset": state.artifacts["dataset"], "prior_csv": str(output)}
        else:
            output = Pipeline(cfg)._prior_csv()
            assert output.is_relative_to(root)
            assert output.read_text() == "checkpoint,success\npolicy.pth,\n"
            output.write_text("checkpoint,success\npolicy.pth,80\n")
            artifacts = {"prior_csv": str(output)}
        state.stage_finished(step, artifacts=artifacts, files={str(output): jobs.digest(output)})
        jobs.write_json(directory / "request.json", request)
        jobs.write_json(directory / "status.json", jobs.finish(directory, request, 0, None))
        outputs.append(output)

    monkeypatch.setattr(jobs, "launch", launch)
    for stage in ("collect", "train-prior", "eval-prior", "train-prior"):
        response = service.call("kuafu_start_stage", {"config": str(config), "stage": stage})
        assert response["ok"], response
        assert run.status()["steps"][stage]["status"] == "complete"
    assert len({output.parent for output in outputs}) == 4
    assert outputs[1].read_text() == "checkpoint,success\npolicy.pth,\n"
    assert outputs[2].read_text() == "checkpoint,success\npolicy.pth,80\n"
    assert run.status()["steps"]["eval-prior"]["stale"]
    outputs[0].write_bytes(b"changed transitions")
    state = run.status(verify_outputs=True)
    assert state["steps"]["collect"]["stale"]
    assert state["steps"]["train-prior"]["stale"]


def test_training_retry_uses_saved_manifest(tmp_path, monkeypatch):
    from kuafu.pipeline.config import PipelineConfig

    run = create(tmp_path)
    config = manifest(tmp_path)
    observed = []

    def launch(directory, request):
        cfg = PipelineConfig.load(directory / "pipeline.toml")
        observed.append((cfg.section("collect")["action_noise_std"], cfg.task_artifact_root))
        jobs.write_json(directory / "request.json", request)
        jobs.write_json(directory / "status.json", {"status": "failed", "returncode": 1})

    monkeypatch.setattr(jobs, "launch", launch)
    service = ToolService(tmp_path / "artifacts", run_id=run.run_id)
    first = service.call("kuafu_start_stage", {"config": str(config), "stage": "collect"})
    assert first["ok"], first
    config.write_text(config.read_text().replace("action_noise_std = 0.003", "action_noise_std = 0.9"))
    retry = service.call("kuafu_retry_step", {"attempt_id": first["attempt_id"]})
    assert retry["ok"], retry
    assert [value for value, _ in observed] == [0.003, 0.003]
    assert observed[0][1] != observed[1][1]


def test_learning_stage_waits_for_active_upstream_job(tmp_path):
    run = create(tmp_path)
    run.begin("kuafu_start_stage", {"stage": "collect"})
    with pytest.raises(ValueError, match="collect is still running"):
        run.begin("kuafu_start_stage", {"stage": "train-prior"})


@pytest.mark.parametrize("record,expected", [
    ({}, "failed"),
    ({"status": "running"}, "failed"),
    ({"status": "complete", "files": {}}, "failed"),
    ({"status": "complete", "files": {"dataset": "sha256"}}, "succeeded"),
    ({"status": "prepared"}, "prepared"),
])
def test_stage_exit_requires_persisted_completion(tmp_path, record, expected):
    run = create(tmp_path)
    directory = tmp_path / "job"
    directory.mkdir()
    config = manifest(tmp_path)
    (directory / "pipeline.toml").write_text(config.read_text() + f'\n[handoff]\nrun_root = "{run.root}"\n')
    learning = run.root / "learning"
    learning.mkdir()
    jobs.write_json(learning / "pipeline_state.json", {"stages": {"collect": record}})
    result = jobs.finish(directory, {"kind": "stage", "stage": "collect"}, 0, None)
    assert result["status"] == expected


def test_learning_receipt_hashes_the_attempts_prior_csv(tmp_path):
    run = create(tmp_path)
    directory = tmp_path / "job"
    directory.mkdir()
    config = manifest(tmp_path)
    (directory / "pipeline.toml").write_text(config.read_text() + f'\n[handoff]\nrun_root = "{run.root}"\n')
    learning = run.root / "learning"
    learning.mkdir()
    csv = learning / "prior.csv"
    csv.write_text("name,checkpoint\nexample,example.pth\n")
    stage = {"status": "complete", "files": {"checkpoint": "previously-recorded"},
             "artifacts": {"prior_csv": str(csv)}}
    jobs.write_json(learning / "pipeline_state.json", {"stages": {"train-prior": stage}})
    result = jobs.finish(directory, {"kind": "stage", "stage": "train-prior"}, 0, None)
    assert result["stage_state"]["files"][str(csv)] == jobs.digest(csv)


def test_running_job_exposes_last_controller_operation(tmp_path):
    from kuafu.programs.generated_drawer import write_json

    (tmp_path / "output").mkdir()
    jobs.write_json(tmp_path / "status.json", {"status": "running"})
    jobs.write_json(tmp_path / "worker.json", {"pid": os.getpid(), "token": jobs.process_token(os.getpid())})
    progress = {"phase": "solving_mpc", "stage": "subtask_2", "control_steps": 100}
    write_json(tmp_path / "output/progress.json", progress)
    observed = jobs.status(tmp_path)
    assert observed["status"] == "running" and observed["progress"] == progress
    assert not list((tmp_path / "output").glob("*.tmp"))
