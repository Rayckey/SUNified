import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

COLLECTION = Path(__file__).resolve().parents[1] / "src/kuafu/data/collection"


def test_media_capture_uses_checkout_root_and_configured_interpreter(tmp_path):
    """Exercise the launcher from another directory without starting Isaac."""
    interpreter = tmp_path / "capture-python"
    interpreter.write_text(
        f"#!{sys.executable}\n"
        + """
import json, os, pathlib, sys
pathlib.Path(os.environ["CAPTURE_AUDIT"]).write_text(json.dumps({
    "cwd": os.getcwd(), "script": str(pathlib.Path(sys.argv[1]).resolve()),
    "script_exists": pathlib.Path(sys.argv[1]).is_file(), "args": sys.argv[2:],
}))
"""
    )
    interpreter.chmod(0o755)
    fixture = tmp_path / "placeholder.pth"
    fixture.write_bytes(b"test")
    audit = tmp_path / "capture.json"
    env = {
        **os.environ,
        "KUAFU_PYTHON": str(interpreter),
        "CAPTURE_AUDIT": str(audit),
        "CHECKPOINT": str(fixture),
        "PRIOR_CHECKPOINT": str(fixture),
        "WEIGHTS_PARAMS_JSON": str(fixture),
        "KUAFU_ARTIFACT_ROOT": str(tmp_path),
    }
    env.pop("PYTHON_BIN", None)
    result = subprocess.run(
        ["bash", str(COLLECTION / "capture_drawer_stack_best_ema_media.sh")],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    captured = json.loads(audit.read_text())
    assert Path(captured["cwd"]) == COLLECTION.parents[3]
    assert captured["script_exists"]
    assert Path(captured["script"]) == COLLECTION / "capture_drawer_stack_best_ema_media.py"
    assert captured["args"][captured["args"].index("--checkpoint") + 1] == str(fixture)


def test_shared_loop_forwards_options_and_stages_worker_layout(tmp_path):
    """Exercise the real shell loop with local stand-ins for collection and upload."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    interpreter = fake_bin / "collection-python"
    interpreter.write_text(
        f"#!{sys.executable}\n"
        + """
import json, os, pathlib, sys
if sys.argv[1].endswith("launch_collect_with_checkpoint.py"):
    args = sys.argv[2:]
    pathlib.Path(os.environ["AUDIT_ARGS"]).write_text(json.dumps(args))
    root = pathlib.Path(args[args.index("--output_root") + 1])
    dataset = root / "workers/worker_00/datasets"
    dataset.mkdir(parents=True)
    (dataset / "episode_000.h5").write_text("synthetic episode")
    (dataset / "episode_tmp_env0").write_text("unfinished episode")
else:
    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
"""
    )
    interpreter.chmod(0o755)
    # Fail the mock upload to verify that staged data is retained on failure.
    aws = fake_bin / "aws"
    aws.write_text("#!/bin/sh\nexit 93\n")
    aws.chmod(0o755)
    artifact = tmp_path / "checkpoint.pth"
    artifact.write_bytes(b"test")
    params = tmp_path / "params"
    params.mkdir()
    for name in ("agent.yaml", "env.yaml", "runtime.yaml"):
        (params / name).write_text("{}")
    env = {
        **os.environ,
        "PATH": os.pathsep.join([str(fake_bin), os.environ["PATH"], str(Path.home() / ".local/bin")]),
        "KUAFU_PYTHON": str(interpreter),
        "KUAFU_ARTIFACT_ROOT": str(tmp_path),
        "AUDIT_ARGS": str(tmp_path / "args.json"),
        "GPUS": "0",
        "DEMOS_PER_CHUNK": "3",
        "NUM_ENVS_PER_WORKER": "2",
        "NUM_ITERS": "1",
        "TASK": "Panda-Table-Stack-2-fff3",
        "CHECKPOINT": str(artifact),
        "PRIOR_CHECKPOINT": str(artifact),
        "WEIGHTS_PARAMS_JSON": str(artifact),
        "PRIOR_MODE": "residual_rl",
        "CHECKPOINT_PARAMS_DIR": str(params),
        "CHECKPOINT_PARAMS_REQUIRED_FILES": "agent.yaml env.yaml runtime.yaml",
        "CAMERA_KEYS": "table_cam_0 table_cam_1 table_cam_2",
        "EXPECTED_NUM_CAMERAS": "3",
        "FORCE_LEFT_RIGHT_SWAP": "1",
        "DISABLE_LEFT_RIGHT_RANDOMIZATION": "0",
        "USE_TILED_CAMERAS": "1",
        "COLLECT_OUT": str(tmp_path / "collect"),
        "POINTCLOUD_DIR": str(tmp_path / "staged"),
        "LOOP_LOG_DIR": str(tmp_path / "logs"),
        "COLLECT_LOOP_NAME": "test",
        "LOCAL_ORGANIZE_ZARR_BEFORE_UPLOAD": "0",
    }
    result = subprocess.run(
        ["bash", "-c", 'source "$1"', "test-loop", str(COLLECTION / "collect_and_upload_loop_common.sh")],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 93, result.stdout + result.stderr
    args = json.loads((tmp_path / "args.json").read_text())
    for option in ("--force_left_right_swap", "--use_tiled_cameras", "--camera_keys", "--checkpoint_params_dir"):
        assert option in args
    assert args[args.index("--checkpoint_params_dir") + 1] == str(params)
    assert len(list((tmp_path / "staged").rglob("episode_000.h5"))) == 1
    assert not list((tmp_path / "staged").rglob("episode_tmp_env*"))


@pytest.mark.parametrize(
    "script, task, cameras",
    [
        ("collect_and_upload_loop_kinova_drawer_fff3.sh", "Kinova-Table-Gen3-Drawer-fff", ""),
        ("collect_and_upload_loop_kinova_stack3_fff3.sh", "Kinova-Table-Gen3-Stack-3-fff3", ""),
        (
            "collect_and_upload_loop_panda_stack2.sh",
            "Panda-Table-Stack-2-fff3",
            "table_cam_0 table_cam_1 table_cam_2",
        ),
        (
            "collect_and_upload_loop_panda_hang_cup_best_ema.sh",
            "Panda-Table-Hang-Cup-fff3",
            "table_cam_0 table_cam_1 table_cam_2",
        ),
        (
            "collect_and_upload_loop_panda_drawer_stack_best_ema.sh",
            "Panda-Table-Drawer-Stack-fff3",
            "table_cam_0 table_cam_1 table_cam_2",
        ),
    ],
)
def test_accepted_recipe_defaults_from_another_directory(tmp_path, script, task, cameras):
    # Intercept only the shared loop; execute the recipe itself, including aliases.
    shell = """
source() { printf '%s\n' "$TASK" "$CHECKPOINT_PARAMS_DIR" "${CAMERA_KEYS:-}" "$CHECKPOINT"; }
export -f source
bash "$1"
"""
    env = {
        key: value
        for key, value in os.environ.items()
        if key
        not in {
            "TASK",
            "CHECKPOINT",
            "PRIOR_CHECKPOINT",
            "CHECKPOINT_PARAMS_DIR",
            "CAMERA_KEYS",
        }
    }
    env["KUAFU_ARTIFACT_ROOT"] = str(tmp_path)
    result = subprocess.run(
        ["bash", "-c", shell, "test-recipe", str(COLLECTION / script)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[0] == task
    assert lines[1].startswith(str(tmp_path) + "/Residuals/")
    assert lines[2] == cameras
    assert lines[3].startswith(str(tmp_path) + "/Residuals/")
