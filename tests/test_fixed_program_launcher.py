import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from kuafu.rl.isaac import env_factory, train_rl_games, update_utilities
from kuafu.rl.launchers.ablation import train_fixed_program as launcher


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    calls = {"exit_code": 0}

    def close():
        # Isaac Sim's fast shutdown terminates from close(), including when a
        # Python exception is pending. Preserve that behavior in this regression.
        raise SystemExit(calls["exit_code"])

    app = SimpleNamespace(close=close)
    env = SimpleNamespace(unwrapped=SimpleNamespace(), close=lambda: None)
    monkeypatch.setattr(env_factory, "create_env_and_app", lambda **kw: (app, env, {}))
    monkeypatch.setattr(update_utilities, "apply_updates_and_wrap_reset", lambda *a, **kw: None)
    omni = ModuleType("omni")
    omni.kit = ModuleType("omni.kit")
    omni.kit.app = ModuleType("omni.kit.app")
    omni.kit.app.get_app = lambda: SimpleNamespace(post_quit=lambda code: calls.update(exit_code=code))
    for name, module in (("omni", omni), ("omni.kit", omni.kit), ("omni.kit.app", omni.kit.app)):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(sys, "argv", [
        "train_fixed_program.py", "--task", "Panda-Table-Stack-2-f",
        "--num_envs", "1", "--max_train_epoch", "2", "--env_seed", "10", "--worker_idx", "3",
        "--run_root", str(tmp_path),
    ])
    return tmp_path, calls


def test_passes_worker_seed_and_records_real_checkpoint(runtime, monkeypatch):
    path, calls = runtime

    def train(*, seed, **kwargs):
        calls["seed"] = seed
        checkpoint = path / "trained.pth"
        checkpoint.write_bytes(b"saved model")
        return str(path), str(checkpoint)

    monkeypatch.setattr(train_rl_games, "train_rl_games", train)
    with pytest.raises(SystemExit) as exited:
        launcher.main()
    assert exited.value.code == 0
    assert calls["seed"] == 13
    completion = json.loads((path / "completion.json").read_text())
    assert completion["seed"] == 13
    assert completion["checkpoint"] == str(path / "trained.pth")


@pytest.mark.parametrize("missing_checkpoint", [False, True])
def test_failure_survives_simulator_fast_shutdown(runtime, monkeypatch, capsys, missing_checkpoint):
    path, _ = runtime

    def train(**kwargs):
        if missing_checkpoint:
            return str(path), None
        raise ValueError("invalid residual configuration")

    monkeypatch.setattr(train_rl_games, "train_rl_games", train)
    with pytest.raises(SystemExit) as exited:
        launcher.main()
    assert exited.value.code == 1
    assert "Traceback" in capsys.readouterr().err
    assert not (path / "completion.json").exists()
