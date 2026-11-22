"""Keep the native training handoff usable without delegating inference to SUN."""

import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from kuafu.bc.training import run_bc_prior_pipeline as prior
from kuafu.rl.cli import residual_hparams
from kuafu.rl.core import runner_residual_hparams as runner
from kuafu.rl.launchers.prior_conditioned import create_residual_hparam_launcher as launcher
from kuafu.pipeline.config import PipelineConfig
from kuafu.pipeline.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]


def test_external_training_setting_reaches_generated_launcher(monkeypatch):
    cfg = PipelineConfig.load(ROOT / "assets/configs/fr3_finray_drawer.example.toml")
    cfg.data["residual"]["hparam_iterations"] = 0
    pipeline = Pipeline(cfg, dry_run=True, prepare_only=True)
    calls = []
    monkeypatch.setattr(pipeline.runner, "run", lambda label, command: calls.append(command))
    pipeline.train_residual()
    command = calls[0]
    monkeypatch.setattr(sys, "argv", command[1:])
    args = prior.parse_args()
    generated = prior.build_residual_launcher_cmd(args, Path("sweep.csv"), "fr3_finray", "drawer", "weights.json")
    monkeypatch.setattr(sys, "argv", generated[1:])
    settings = launcher.parse_args()
    assert settings.residual_hparam_iters == 0
    assert settings.video_feedback_flag == "--no-video_feedback"


@pytest.fixture
def training_args():
    parser = runpy.run_path(str(ROOT / "scripts/residual_rl/train.py"))["_build_arg_parser"]()
    args = parser.parse_args(
        [
            "--initial_weights_params",
            "/tmp/supplied-weights.json",
            "--initial_prior_checkpoint",
            "/tmp/prior.pth",
            "--initial_agent_hyperparameters",
            "/tmp/supplied-hyperparameters.json",
            "--no-video_feedback",
        ]
    )
    args.residual_hparam_iters = 0
    return args


def test_zero_iterations_runs_supplied_training_trial_without_model_client(training_args, monkeypatch):
    recorded = {}

    class Trial:
        def __init__(self, **kwargs):
            recorded.update(kwargs)

        def run(self):
            recorded["ran"] = True

    actual_runner = runner.ResidualHparamRunner
    monkeypatch.setattr(runner, "ResidualHparamRunner", Trial)
    residual_hparams.run_from_cli(training_args)
    assert recorded["ran"] and recorded["residual_hparam_iters"] == 0
    assert recorded["initial_agent_hyperparameters"] == "/tmp/supplied-hyperparameters.json"
    assert recorded["video_feedback"] is False

    def unexpected_client(**kwargs):
        pytest.fail("A supplied training trial must not construct a model client")

    monkeypatch.setattr(runner, "OpenAIChatClient", unexpected_client)
    actual_runner._residual_hparam_finetune(
        SimpleNamespace(_residual_hparam_iters=0),
        initial_weights={},
        initial_parameters={},
        stage_count=1,
    )


@pytest.mark.parametrize("invalid", ["negative", "video", "evaluation_only"])
def test_zero_iteration_mode_rejects_implicit_inference_or_no_training(training_args, invalid):
    if invalid == "negative":
        training_args.residual_hparam_iters = -1
    elif invalid == "video":
        training_args.video_feedback = True
    else:
        training_args.initial_eval_checkpoint = "/tmp/evaluation-only.pth"
    with pytest.raises(ValueError, match="iterations|iters|training trial"):
        residual_hparams.run_from_cli(training_args)
