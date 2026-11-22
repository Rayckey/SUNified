import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from kuafu.asset_generation import cli, validation
from kuafu.asset_generation.export import export_asset
from kuafu.asset_generation.io import digest, fresh_directory
from kuafu.asset_generation.model import OpenAIModel
from kuafu.asset_generation.schema import PhysicsConfig, RunConfig
from kuafu.paths import PROJECT_ROOT
from kuafu.pipeline.cli import main


def test_unified_asset_command_needs_no_task_manifest(capsys):
    assert main(["-c", "/missing/task.toml", "assets", "schema"]) == 0
    assert "SceneSpec" in json.loads(capsys.readouterr().out)["title"]


def test_asset_cli_resolves_relative_outputs(tmp_path, monkeypatch):
    monkeypatch.setenv("KUAFU_ARTIFACT_ROOT", str(tmp_path))
    received = []

    def demo(out):
        received.append(out)
        return {"workspace": str(out / "job")}

    monkeypatch.setattr(cli, "run_demo", demo)
    assert main(["assets", "demo", "--out", "assets/demo"]) == 0
    assert received == [tmp_path / "assets/demo"]


def test_generated_assets_cannot_write_inside_source():
    target = PROJECT_ROOT / "outputs/asset-test-must-not-exist"
    with pytest.raises(ValueError, match="outside the source"):
        fresh_directory(target)
    with pytest.raises(ValueError, match="outside the source"):
        export_asset(None, None, target, PhysicsConfig())
    assert not target.exists()


def test_response_cache_boundary_is_checked_before_model_request():
    with pytest.raises(ValueError, match="outside the source"):
        OpenAIModel(RunConfig(), PROJECT_ROOT / "cache/asset-model")


def test_multi_video_preparation_resolves_all_outputs_outside_source(video, tmp_path, monkeypatch):
    import shutil
    from kuafu.asset_generation.video import prepare

    second = tmp_path / "second.avi"
    shutil.copy2(video, second)
    monkeypatch.setenv("KUAFU_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    prepared = prepare(video, "Assemble the cabinet and removed drawer", Path("cabinet"),
                       frame_limit=4, reference_videos=[second], motion="articulated")
    workspace = tmp_path / "artifacts/cabinet"
    assert (workspace / "input.json").is_file()
    assert (workspace / "reference_01/input.json").is_file()
    assert all((workspace / frame.path).is_file() for frame in prepared.frames)


@pytest.fixture
def asset(tmp_path):
    path = tmp_path / "asset_dynamic.usda"
    path.write_text("#usda 1.0\n")
    return path


def test_asset_validation_preserves_simulator_failure(asset, monkeypatch):
    monkeypatch.setattr(validation.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=1))
    with pytest.raises(RuntimeError, match="exit code 1"):
        validation.validate_asset(asset)


def test_asset_validation_rejects_stale_report(asset, monkeypatch):
    old = asset.parent / "physics_smoke.json"
    old.write_text('{"passed": true}')
    monkeypatch.setattr(validation.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0))
    with pytest.raises(RuntimeError, match="without writing a new"):
        validation.validate_asset(asset)


@pytest.mark.parametrize("changed", [False, True])
def test_asset_validation_checks_actual_usd_hash(asset, monkeypatch, changed):
    original_hash = digest(asset)
    metadata = asset.parent / "asset.json"
    metadata.write_text('{"isaac_physics_tested": false}')

    def run(command, **kwargs):
        assert command[0] == "/chosen/isaac/python"
        pending = Path(command[command.index("--report") + 1])
        pending.write_text(json.dumps({
            "passed": True, "asset": str(asset), "steps": 240, "physics_dt": 1 / 120,
            "asset_layers": [{"path": str(asset), "sha256": original_hash}],
        }))
        if changed:
            asset.write_text("changed")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(validation.subprocess, "run", run)
    if changed:
        with pytest.raises(ValueError, match="changed after"):
            validation.validate_asset(asset, python="/chosen/isaac/python")
    else:
        report = validation.validate_asset(asset, python="/chosen/isaac/python")
        assert json.loads(report.read_text())["passed"] is True
        assert json.loads(metadata.read_text())["isaac_physics_tested"] is True
