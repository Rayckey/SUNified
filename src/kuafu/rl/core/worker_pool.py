"""Multiprocessing worker pool for Isaac Sim training.

Key invariants (Isaac-specific):
- Each worker process must create its own SimulationApp via AppLauncher.
- Do NOT import heavy Isaac modules in the parent process.
"""

from __future__ import annotations

import datetime
import faulthandler
import glob
import json
import math
import multiprocessing
import multiprocessing.connection
import multiprocessing.queues
import os
import queue
import re
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast


@dataclass(frozen=True)
class TrainConfig:
    """Configuration payload sent to workers."""

    stage_id: int | None
    all_stages_weights: dict[int, dict[int, float]] | None
    # Optional constraint arg overrides: constraint_idx -> {arg_name: new_value}
    # Only applied if stage_id is not None.
    progress_metric_expr: str | None
    # Optional: constraint arg overrides for multiple stages (stage_id -> parameters dict).
    all_stages_parameters: dict[int, dict[int, dict[str, object]]] | None = None
    # Optional: agent YAML overrides for residual-RL hyperparameter tuning.
    agent_hyperparameters: dict[str, object] | None = None
    # Optional: override training length for this run (max_epochs).
    max_train_epoch_override: int | None = None
    # Optional: resume training from an RL-Games checkpoint.
    resume_from_checkpoint: str | None = None
    # Optional: skip training and evaluate this checkpoint directly.
    eval_only_checkpoint: str | None = None
    # Optional: initialize training from a BC prior checkpoint.
    prior_checkpoint: str | None = None
    prior_mode: str = "full_actor"
    prior_obs_stats_count: float = 1e6
    prior_verbose: bool = False
    prior_critic_init_mode: str = "actor_only_random_critic"
    prior_freeze_actor_epochs: int = 0
    # While the actor is frozen, linearly ramp sigma from its silenced value back to the
    # saved PPO sigma over the LAST `prior_sigma_ramp_epochs` frozen epochs.  Sigma is
    # therefore already restored before the actor is unfrozen, so PPO never operates on
    # a degenerate near-zero sigma.  See ``_freeze_actor_for_epochs`` for why this matters.
    prior_sigma_ramp_epochs: int = 0
    # Override for the sigma ramp/restore target.  When None (default), the sigma ramp
    # ends at the PPO config's ``sigma_init`` saved from the checkpoint (backward-compat).
    # Setting a lower value (for example -5.5 ⇒ std ≈ 0.004) keeps post-unfreeze action
    # noise below the BC prior's stage-completion tolerance; without this, restoring
    # sigma to the default -3 (std ≈ 0.05, ~5 cm EE noise at clip_actions=1.0) destroys
    # stage completion on tight predicates even though the actor weights are unchanged.
    prior_sigma_final_val: float | None = None
    # After the actor is unfrozen, linearly ramp the actor's learning rate from 0 up to
    # the base adaptive LR over `prior_actor_lr_warmup_epochs` epochs.  Prevents Adam's
    # cold-start first-step (≈ lr · sign(g) per parameter) from blowing the BC-initialized
    # actor out of `[-1,1]`, which triggers bounds loss and the adaptive LR scheduler
    # collapses the LR to ~1e-6 in response.  0 disables the warmup.
    prior_actor_lr_warmup_epochs: int = 0
    # Optional hard cap for the actor optimizer group after BC-prior unfreeze.
    # 0 disables the cap. Recommended when base PPO LR is large (for example 1e-2).
    prior_actor_lr_cap: float = 0.0
    # Keep observation running_mean_std pinned to the BC expert stats that
    # apply_bc_prior_to_rl_games_agent seeds.  Prevents RMS from drifting toward
    # the live rollout distribution during the freeze window (or the whole run
    # if no freeze is configured), which at num_envs=8192 was shown to silently
    # push the frozen BC weights OOD.
    prior_freeze_input_running_mean_std: bool = False
    # If True, unfreeze the observation running_mean_std at the same epoch the
    # actor is unfrozen, so PPO's fine-tuning phase can adapt the normalizer.
    prior_unfreeze_input_running_mean_std_after_freeze: bool = False
    # If True AND actor LR warmup is enabled (>0 epochs), defer the RMS unfreeze
    # from the actor-unfreeze boundary to the end of the actor LR warmup window
    # (unfreeze_epoch + prior_actor_lr_warmup_epochs).  Unfreezing RMS at the
    # actor-unfreeze boundary lets the normalizer shift during PPO minibatch
    # updates, which moves observations under a frozen actor mean head, produces
    # fake KL, and triggers PPO's adaptive LR scheduler to slash the base LR —
    # even when actor_lr warmup correctly held actor gradients at 0.  Deferring
    # keeps the BC-seeded RMS pinned while the actor actually warms up.
    prior_rms_unfreeze_after_actor_lr_warmup: bool = False
    # When True (and prior_actor_lr_warmup_epochs > 0), pin rl_games'
    # agent.scheduler.update(...) to return the pre-unfreeze last_lr for the
    # duration of the actor LR warmup window.  This prevents the default
    # AdaptiveScheduler from reacting to the spurious one-shot KL that PPO
    # reports at the actor-unfreeze boundary (observed: kl ~ 130 causing
    # last_lr to be crushed from 1e-2 to 7e-5 in a single epoch, which then
    # multiplies our warmup factor and delivers effectively zero LR to the
    # actor for the entire warmup window, stalling learning).  The original
    # scheduler is restored unchanged at the end of warmup.
    prior_lr_schedule_freeze_during_warmup: bool = False
    # Optional: pin the PPO optimizer to a fixed learning rate for the whole run
    # and neutralize rl_games' adaptive LR scheduler.  When set (> 0), this
    # disables the entire scheduler-avoidance coordination set (actor_lr_cap,
    # actor_lr_warmup, lr_schedule_freeze_during_warmup, rms_unfreeze_after_
    # actor_lr_warmup) as no-ops, because the fixed LR is THE LR: rl_games'
    # AdaptiveScheduler cannot react to KL transients if we've replaced its
    # update() with a constant return.  Recommended for BC-warm-start regimes
    # where the policy starts at a good operating point and only needs small
    # stable updates, rather than the aggressive-then-tapering profile that
    # AdaptiveScheduler assumes for from-scratch PPO.
    prior_fixed_lr: float | None = None
    # After actor unfreeze, keep the PPO actor close to the BC prior for the first
    # `prior_bc_anchor_epochs` epochs using an MSE anchor on the action mean.
    prior_bc_anchor_epochs: int = 0
    prior_bc_anchor_coef: float = 0.0
    # Supervised BC-on-dataset regularizer.  When prior_bc_dataset_coef > 0, every
    # PPO minibatch step also adds `coef(t) * MSE(policy.mu(obs_expert), action_expert)`,
    # where (obs_expert, action_expert) are sampled from the BC HDF5 dataset.  Obs
    # normalization for this pathway is pinned to the BC checkpoint's saved
    # obs_normalizer stats (not the live running_mean_std), so the regularizer
    # stays correct even after RMS unfreezes.  The coefficient decays linearly
    # from `coef` to `coef * 0.1` over `prior_bc_dataset_decay_epochs` epochs
    # after the actor-unfreeze boundary, then holds at the 0.1*coef floor
    # indefinitely.  Unlike prior_bc_anchor, this loss is *always on* after
    # unfreeze.  prior_bc_dataset_path is optional; when omitted, the dataset
    # path is auto-resolved from the BC run's train_config.yaml.
    prior_bc_dataset_coef: float = 0.0
    prior_bc_dataset_decay_epochs: int = 0
    prior_bc_dataset_batch_size: int = 1024
    prior_bc_dataset_path: str | None = None
    prior_bc_action_reward_coef: float = 0.0
    # Optional: skip this worker (no training; returns a skipped result).
    skip: bool = False
    # for logging/debug
    tag: str = ""


@dataclass(frozen=True)
class TrainResult:
    worker_idx: int
    success: bool
    log_dir: str | None = None
    checkpoint_path: str | None = None
    progress_metric_eval: float | None = None
    # Backward-compatible alias used by older tests/scripts before the metric was renamed.
    success_metric_eval: float | None = None
    reward_metric_eval: float | None = None
    exception: str | None = None
    exception_traceback: str | None = None
    skipped: bool = False
    # Optional env feedback (from env_learned.get_stage_transition_stats / get_constraint_met_rate_stats).
    stage_transition_stats: dict | None = None
    highest_stage_reached_per_env: list[int] | None = None
    constraint_met_rate_stats: dict | None = None
    # Optional end-to-end termination stats from env_learned.
    true_success_count: int | None = None
    false_complete_count: int | None = None
    timeout_count: int | None = None
    # TensorBoard checkpoint selection metadata. Training chooses the checkpoint immediately
    # before the highest post-critic-warmup success_rate_window before running eval.
    selection_success_rate_window: float | None = None
    # Backward-compatible alias for older runner code/logs; active Kuafu stores the window value here too.
    selection_success_rate_ema: float | None = None
    selection_false_complete_rate: float | None = None
    selection_timeout_rate: float | None = None
    selection_metric_tag: str | None = None
    selection_metric_index: int | None = None
    selection_metric_step: int | None = None
    selection_critic_warmup_epochs: int | None = None
    selection_checkpoint_path: str | None = None
    selection_checkpoint_epoch: int | None = None
    selection_checkpoint_policy: str | None = None
    selection_source: str | None = None
    kuafu_training_parameters_path: str | None = None
    # Canonical applied values read back from the worker env (after updates are applied).
    # These are used to seed the next run/prompt so we don't keep propagating sparse LLM patches.
    applied_stage_weights: dict[int, float] | None = None
    applied_stage_parameters: dict[int, dict[str, object]] | None = None
    applied_agent_hyperparameters: dict[str, object] | None = None

    def __post_init__(self) -> None:
        if self.progress_metric_eval is None and self.success_metric_eval is not None:
            object.__setattr__(self, "progress_metric_eval", self.success_metric_eval)
        elif self.success_metric_eval is None and self.progress_metric_eval is not None:
            object.__setattr__(self, "success_metric_eval", self.progress_metric_eval)


def _worker_output_path(kuafu_run_dir: str | None, worker_idx: int) -> str | None:
    if not kuafu_run_dir:
        return None
    return os.path.join(os.path.abspath(kuafu_run_dir), "worker_logs", f"worker_{int(worker_idx)}.log")


def _tail_text(path: str, max_chars: int = 12000) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    if len(text) <= max_chars:
        return text
    return "...<truncated>...\n" + text[-max_chars:]


def _warn_nonfatal(context: str, exc: BaseException) -> None:
    print(
        f"[residual_rl] WARNING: non-fatal error during {context}: {type(exc).__name__}: {exc}",
        file=sys.__stderr__,
        flush=True,
    )



_PEAK_SELECTION_SUCCESS_RATE_WINDOW_TAG = "Episode/Kuafu/success_rate_window"
_PEAK_SELECTION_SUCCESS_RATE_WINDOW_ALT_TAG = "Kuafu/success_rate_window"
_PEAK_SELECTION_FALSE_COMPLETE_TAG = "Episode/Kuafu/false_complete_rate"
_PEAK_SELECTION_TIMEOUT_TAG = "Episode/Kuafu/timeout_rate"
_PEAK_SELECTION_SCALAR_TAGS = (
    _PEAK_SELECTION_SUCCESS_RATE_WINDOW_TAG,
    _PEAK_SELECTION_SUCCESS_RATE_WINDOW_ALT_TAG,
    _PEAK_SELECTION_FALSE_COMPLETE_TAG,
    _PEAK_SELECTION_TIMEOUT_TAG,
)
_EP_RE = re.compile(r"(?:^|_)ep_(\d+)(?:_|\.pth$)")
_TRUE_SUCCESS_REPORT_MODULE: Any | None = None


@dataclass(frozen=True)
class _SelectionScalarPoint:
    metric_index: int
    step: int
    value: float
    event_file: str


def _true_success_report_module() -> Any:
    global _TRUE_SUCCESS_REPORT_MODULE
    if _TRUE_SUCCESS_REPORT_MODULE is not None:
        return _TRUE_SUCCESS_REPORT_MODULE
    import importlib.util

    script_path = Path(__file__).resolve().parents[4] / "scripts" / "evaluation" / "true_success_checkpoint_report.py"
    spec = importlib.util.spec_from_file_location("_residual_true_success_checkpoint_report", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load TensorBoard event helper from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _TRUE_SUCCESS_REPORT_MODULE = module
    return module


def _run_dir_from_log_dir(log_dir: str) -> Path:
    path = Path(log_dir)
    if path.name in {"summaries", "tb", "nn"}:
        return path.parent
    return path


def _checkpoint_epoch_from_path(path: str | None) -> int | None:
    if not path:
        return None
    match = _EP_RE.search(Path(path).name)
    return int(match.group(1)) if match else None


def _extract_critic_warmup_epochs(agent_cfg: Any) -> int:
    try:
        params = agent_cfg.get("params") if isinstance(agent_cfg, dict) else None
        config = params.get("config") if isinstance(params, dict) else None
        residual_rl = config.get("residual_rl") if isinstance(config, dict) else None
        value = residual_rl.get("critic_warmup_epochs") if isinstance(residual_rl, dict) else 0
        epochs = int(float(value or 0))
    except (TypeError, ValueError):
        epochs = 0
    return max(0, epochs)


def _read_selection_scalar_series(log_dir: str) -> dict[str, list[_SelectionScalarPoint]]:
    module = _true_success_report_module()
    root = Path(log_dir)
    series: dict[str, list[_SelectionScalarPoint]] = {tag: [] for tag in _PEAK_SELECTION_SCALAR_TAGS}
    if not root.exists():
        return series

    event_files = sorted(root.rglob("events.out.tfevents*"))
    tag_counts: dict[str, int] = {tag: 0 for tag in _PEAK_SELECTION_SCALAR_TAGS}
    for event_file in event_files:
        for step, tag, value in module.iter_event_scalars(event_file):
            if tag not in series:
                continue
            try:
                scalar = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(scalar):
                continue
            tag_counts[tag] += 1
            series[tag].append(
                _SelectionScalarPoint(
                    metric_index=int(tag_counts[tag]),
                    step=int(step),
                    value=scalar,
                    event_file=str(event_file),
                )
            )
    return series


def _value_at_selection_point(
    points: list[_SelectionScalarPoint],
    *,
    metric_index: int,
    step: int,
) -> float | None:
    for point in points:
        if int(point.metric_index) == int(metric_index):
            return float(point.value)
    for point in points:
        if int(point.step) == int(step):
            return float(point.value)
    return None


def _checkpoint_for_metric_step(
    *,
    log_dir: str,
    fallback_checkpoint: str | None,
    metric_step: int,
    min_epoch_exclusive: int | None = None,
    strictly_before: bool = False,
) -> str | None:
    run_dir = _run_dir_from_log_dir(log_dir)
    candidates: list[str] = []
    nn_dir = run_dir / "nn"
    if nn_dir.is_dir():
        candidates.extend(sorted(glob.glob(str(nn_dir / "*.pth"))))
    if fallback_checkpoint and fallback_checkpoint not in candidates:
        candidates.append(str(fallback_checkpoint))
    if not candidates:
        if min_epoch_exclusive is None:
            return fallback_checkpoint
        fallback_epoch = _checkpoint_epoch_from_path(fallback_checkpoint)
        if fallback_epoch is not None and fallback_epoch > int(min_epoch_exclusive):
            return fallback_checkpoint
        return None

    ep_candidates: list[tuple[str, int]] = []
    for checkpoint in candidates:
        ep = _checkpoint_epoch_from_path(checkpoint)
        if ep is not None:
            ep_candidates.append((checkpoint, ep))
    if ep_candidates:
        min_epoch = int(min_epoch_exclusive) if min_epoch_exclusive is not None else None
        after_warmup = [
            (checkpoint, ep)
            for checkpoint, ep in ep_candidates
            if min_epoch is None or ep > min_epoch
        ]
        if strictly_before:
            before = [(checkpoint, ep) for checkpoint, ep in after_warmup if ep < int(metric_step)]
            if before:
                return max(before, key=lambda item: (item[1], item[0]))[0]
            return None
        before_or_at = [(checkpoint, ep) for checkpoint, ep in after_warmup if ep <= int(metric_step)]
        if before_or_at:
            return max(before_or_at, key=lambda item: (item[1], item[0]))[0]
        if after_warmup:
            return min(after_warmup, key=lambda item: (item[1], item[0]))[0]
        return None

    if fallback_checkpoint:
        if min_epoch_exclusive is None:
            return str(fallback_checkpoint)
        fallback_epoch = _checkpoint_epoch_from_path(fallback_checkpoint)
        if fallback_epoch is not None and fallback_epoch > int(min_epoch_exclusive):
            return str(fallback_checkpoint)
        return None
    try:
        return max(candidates, key=os.path.getmtime)
    except OSError:
        return sorted(candidates)[-1]


def _select_post_critic_peak_checkpoint(
    *,
    log_dir: str,
    fallback_checkpoint: str | None,
    critic_warmup_epochs: int,
) -> tuple[str | None, dict[str, object | None]]:
    warmup = max(0, int(critic_warmup_epochs))
    metadata: dict[str, object | None] = {
        "selection_success_rate_window": None,
        "selection_success_rate_ema": None,
        "selection_false_complete_rate": None,
        "selection_timeout_rate": None,
        "selection_metric_tag": _PEAK_SELECTION_SUCCESS_RATE_WINDOW_TAG,
        "selection_metric_index": None,
        "selection_metric_step": None,
        "selection_critic_warmup_epochs": warmup,
        "selection_checkpoint_path": fallback_checkpoint,
        "selection_checkpoint_epoch": _checkpoint_epoch_from_path(fallback_checkpoint),
        "selection_checkpoint_policy": "strictly_before_window_peak_after_critic_warmup",
        "selection_source": "fallback_latest_checkpoint",
    }
    try:
        series = _read_selection_scalar_series(log_dir)
    except Exception as exc:
        _warn_nonfatal("post-critic-warmup checkpoint selection event read", exc)
        metadata.update(
            {
                "selection_checkpoint_path": None,
                "selection_checkpoint_epoch": None,
                "selection_source": "event_read_failed_for_success_rate_window_selection",
            }
        )
        return None, metadata

    window_points = list(series.get(_PEAK_SELECTION_SUCCESS_RATE_WINDOW_TAG, []))
    if not window_points:
        window_points = list(series.get(_PEAK_SELECTION_SUCCESS_RATE_WINDOW_ALT_TAG, []))
    if not window_points:
        metadata.update(
            {
                "selection_checkpoint_path": None,
                "selection_checkpoint_epoch": None,
                "selection_source": "no_success_rate_window_events",
            }
        )
        return None, metadata

    valid_points = [point for point in window_points if int(point.step) > warmup]
    if not valid_points:
        metadata.update(
            {
                "selection_checkpoint_path": None,
                "selection_checkpoint_epoch": None,
                "selection_source": "no_post_critic_warmup_success_rate_window",
            }
        )
        return None, metadata

    false_points = list(series.get(_PEAK_SELECTION_FALSE_COMPLETE_TAG, []))
    timeout_points = list(series.get(_PEAK_SELECTION_TIMEOUT_TAG, []))

    def sort_key(point: _SelectionScalarPoint) -> tuple[float, float, float, int, int]:
        false_rate = _value_at_selection_point(false_points, metric_index=point.metric_index, step=point.step)
        timeout_rate = _value_at_selection_point(timeout_points, metric_index=point.metric_index, step=point.step)
        return (
            float(point.value),
            -float(false_rate if false_rate is not None else 1.0),
            -float(timeout_rate if timeout_rate is not None else 1.0),
            int(point.step),
            int(point.metric_index),
        )

    peak = max(valid_points, key=sort_key)
    peak_false = _value_at_selection_point(false_points, metric_index=peak.metric_index, step=peak.step)
    peak_timeout = _value_at_selection_point(timeout_points, metric_index=peak.metric_index, step=peak.step)
    selected_checkpoint = _checkpoint_for_metric_step(
        log_dir=log_dir,
        fallback_checkpoint=fallback_checkpoint,
        metric_step=int(peak.step),
        min_epoch_exclusive=warmup,
        strictly_before=True,
    )
    selection_source = "post_critic_warmup_success_rate_window_peak_prior_checkpoint"
    if selected_checkpoint is None:
        selection_source = "no_checkpoint_after_critic_warmup_for_success_rate_window_peak"
    metadata.update(
        {
            "selection_success_rate_window": float(peak.value),
            "selection_success_rate_ema": float(peak.value),
            "selection_false_complete_rate": float(peak_false) if peak_false is not None else None,
            "selection_timeout_rate": float(peak_timeout) if peak_timeout is not None else None,
            "selection_metric_tag": _PEAK_SELECTION_SUCCESS_RATE_WINDOW_TAG,
            "selection_metric_index": int(peak.metric_index),
            "selection_metric_step": int(peak.step),
            "selection_checkpoint_path": selected_checkpoint,
            "selection_checkpoint_epoch": _checkpoint_epoch_from_path(selected_checkpoint),
            "selection_checkpoint_policy": "strictly_before_window_peak_after_critic_warmup",
            "selection_source": selection_source,
        }
    )
    return selected_checkpoint, metadata


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except Exception:
            pass
    return repr(value)


def _write_kuafu_training_parameters(*, log_dir: str, record: dict[str, Any]) -> str | None:
    run_dir = _run_dir_from_log_dir(str(log_dir))
    params_dir = run_dir / "params"
    try:
        params_dir.mkdir(parents=True, exist_ok=True)
        path = params_dir / "kuafu_training_parameters.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(_json_safe(record), f, indent=2, sort_keys=True)
            f.write("\n")
        return str(path)
    except Exception as exc:
        _warn_nonfatal("writing per-run Kuafu training parameters", exc)
        return None


_ALLOWED_AGENT_HYPERPARAMETERS = {
    "clip_actions",
    "residual_bounds",
    "clip_composed_actions",
    "alpha_final",
    "alpha_warmup_epochs",
}


def _action_dimension_labels(action_mode: object, action_dim: object) -> list[str]:
    mode = str(action_mode or "").strip()
    try:
        dim = int(action_dim)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        dim = 0

    if mode == "absolute_ee":
        labels = [
            "0: panda_fingertip target x position in robot base frame",
            "1: panda_fingertip target y position in robot base frame",
            "2: panda_fingertip target z position in robot base frame",
            "3: panda_fingertip target orientation rot6d[0]",
            "4: panda_fingertip target orientation rot6d[1]",
            "5: panda_fingertip target orientation rot6d[2]",
            "6: panda_fingertip target orientation rot6d[3]",
            "7: panda_fingertip target orientation rot6d[4]",
            "8: panda_fingertip target orientation rot6d[5]",
            "9: gripper finger 1 target",
            "10: gripper finger 2 target",
        ]
    elif mode == "relative_ee":
        labels = [
            "0: panda_fingertip delta x position in robot base frame",
            "1: panda_fingertip delta y position in robot base frame",
            "2: panda_fingertip delta z position in robot base frame",
            "3: panda_fingertip axis-angle delta x",
            "4: panda_fingertip axis-angle delta y",
            "5: panda_fingertip axis-angle delta z",
            "6: gripper finger 1 command",
            "7: gripper finger 2 command",
        ]
    elif mode in {"absolute_joint", "relative_joint"} or dim == 9:
        basis = "target" if "absolute" in mode else "delta"
        grip_basis = "target" if "absolute" in mode else "command"
        labels = [f"{i}: arm joint {i + 1} {basis}" for i in range(7)]
        labels.extend([f"7: gripper finger 1 {grip_basis}", f"8: gripper finger 2 {grip_basis}"])
    else:
        labels = [f"{i}: action dimension {i}" for i in range(max(dim, 0))]

    if dim > 0 and len(labels) != dim:
        return [f"{i}: action dimension {i}" for i in range(dim)]
    return labels


def _coerce_bool(value: object, *, name: str) -> bool:
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{name} must be a bool, got {value!r}")


def _coerce_finite_float(value: object, *, name: str) -> float:
    try:
        val = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric, got {value!r}") from exc
    import math

    if not math.isfinite(val):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return float(val)


def _coerce_nonnegative_int(value: object, *, name: str) -> int:
    val = _coerce_finite_float(value, name=name)
    ival = int(val)
    if float(ival) != float(val):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if ival < 0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")
    return ival


def _extract_agent_hyperparameters(agent_cfg: Any) -> dict[str, object]:
    if not isinstance(agent_cfg, dict):
        return {}
    params = agent_cfg.get("params")
    if not isinstance(params, dict):
        return {}
    env_params = params.get("env")
    if not isinstance(env_params, dict):
        env_params = {}
    config = params.get("config")
    if not isinstance(config, dict):
        config = {}
    residual_rl = config.get("residual_rl")
    if not isinstance(residual_rl, dict):
        residual_rl = {}

    out: dict[str, object] = {}
    if "clip_actions" in env_params:
        out["clip_actions"] = _coerce_finite_float(env_params["clip_actions"], name="clip_actions")
    if "alpha_final" in residual_rl:
        out["alpha_final"] = _coerce_finite_float(residual_rl["alpha_final"], name="alpha_final")
    if "alpha_warmup_epochs" in residual_rl:
        out["alpha_warmup_epochs"] = _coerce_nonnegative_int(
            residual_rl["alpha_warmup_epochs"], name="alpha_warmup_epochs"
        )
    if "clip_composed_actions" in residual_rl:
        out["clip_composed_actions"] = _coerce_bool(
            residual_rl["clip_composed_actions"], name="clip_composed_actions"
        )
    elif residual_rl:
        # Match ResidualRlConfig.from_mapping(): omitted means disabled.
        out["clip_composed_actions"] = False
    if "residual_bounds" in residual_rl:
        bounds = residual_rl["residual_bounds"]
        if not isinstance(bounds, (list, tuple)):
            raise ValueError("residual_bounds must be a list")
        out["residual_bounds"] = [
            _coerce_finite_float(v, name=f"residual_bounds[{i}]") for i, v in enumerate(bounds)
        ]
    return out


def _apply_agent_hyperparameters(agent_cfg: Any, overrides: dict[str, object] | None) -> dict[str, object]:
    if not overrides:
        return _extract_agent_hyperparameters(agent_cfg)
    if not isinstance(agent_cfg, dict):
        raise TypeError("agent_cfg must be a dict to apply residual hyperparameters")
    unknown = set(overrides) - _ALLOWED_AGENT_HYPERPARAMETERS
    if unknown:
        raise KeyError(f"Unsupported residual agent hyperparameters: {sorted(unknown)}")

    params = agent_cfg.setdefault("params", {})
    if not isinstance(params, dict):
        raise TypeError("agent_cfg['params'] must be a dict")
    env_params = params.setdefault("env", {})
    if not isinstance(env_params, dict):
        raise TypeError("agent_cfg['params']['env'] must be a dict")
    config = params.setdefault("config", {})
    if not isinstance(config, dict):
        raise TypeError("agent_cfg['params']['config'] must be a dict")
    residual_rl = config.setdefault("residual_rl", {})
    if not isinstance(residual_rl, dict):
        raise TypeError("agent_cfg['params']['config']['residual_rl'] must be a dict")

    current = _extract_agent_hyperparameters(agent_cfg)
    if "clip_actions" in overrides:
        env_params["clip_actions"] = _coerce_finite_float(overrides["clip_actions"], name="clip_actions")
    if "alpha_final" in overrides:
        residual_rl["alpha_final"] = _coerce_finite_float(overrides["alpha_final"], name="alpha_final")
    if "alpha_warmup_epochs" in overrides:
        residual_rl["alpha_warmup_epochs"] = _coerce_nonnegative_int(
            overrides["alpha_warmup_epochs"], name="alpha_warmup_epochs"
        )
    if "clip_composed_actions" in overrides:
        residual_rl["clip_composed_actions"] = _coerce_bool(
            overrides["clip_composed_actions"], name="clip_composed_actions"
        )
    if "residual_bounds" in overrides:
        raw_bounds = overrides["residual_bounds"]
        if not isinstance(raw_bounds, (list, tuple)):
            raise ValueError("residual_bounds must be a list")
        bounds = [
            _coerce_finite_float(v, name=f"residual_bounds[{i}]") for i, v in enumerate(raw_bounds)
        ]
        current_bounds = current.get("residual_bounds")
        if isinstance(current_bounds, list) and len(current_bounds) != len(bounds):
            raise ValueError(
                f"residual_bounds length mismatch: got {len(bounds)}, expected {len(current_bounds)}"
            )
        residual_rl["residual_bounds"] = bounds
    return _extract_agent_hyperparameters(agent_cfg)


def _configure_worker_logging(kuafu_run_dir: str | None, worker_idx: int) -> tuple[Any | None, str | None]:
    worker_output_log_path = _worker_output_path(kuafu_run_dir, worker_idx)
    if not worker_output_log_path:
        return None, None

    os.makedirs(os.path.dirname(worker_output_log_path), exist_ok=True)
    worker_log_file = open(worker_output_log_path, "a", encoding="utf-8", buffering=1)

    # Redirect the child process stdout/stderr file descriptors so Python, faulthandler,
    # and native libraries all write into the same per-worker log.
    try:
        sys.stdout.flush()
    except Exception as exc:
        _warn_nonfatal("worker stdout flush before redirection", exc)
    try:
        sys.stderr.flush()
    except Exception as exc:
        _warn_nonfatal("worker stderr flush before redirection", exc)
    os.dup2(worker_log_file.fileno(), 1)
    os.dup2(worker_log_file.fileno(), 2)

    try:
        faulthandler.enable(file=worker_log_file, all_threads=True)
    except Exception as exc:
        _warn_nonfatal("worker faulthandler setup", exc)

    print(f"[residual_rl] Worker {worker_idx}: redirecting stdout/stderr to {worker_output_log_path}", flush=True)
    return worker_log_file, worker_output_log_path


class WorkerPool:
    def __init__(
        self,
        *,
        task: str,
        device: str,
        env_seed: int,
        num_workers: int,
        num_envs: int | None,
        headless: bool,
        max_train_epoch: int,
        kuafu_run_dir: str | None = None,
        stop_on_error: bool = False,
        initial_prior_checkpoint: str | None = None,
        execution_robot_id: str | None = None,
    ) -> None:
        self._task = task
        self._device = device
        self._env_seed = int(env_seed)
        self._num_workers = int(num_workers)
        self._num_envs = num_envs
        self._headless = bool(headless)
        self._max_train_epoch = int(max_train_epoch)
        self._kuafu_run_dir = kuafu_run_dir
        self._worker_failures_path = (
            os.path.join(os.path.abspath(kuafu_run_dir), "worker_failures.jsonl")
            if kuafu_run_dir
            else None
        )
        self._stop_on_error = bool(stop_on_error)
        self._initial_prior_checkpoint = initial_prior_checkpoint
        self._execution_robot_id = execution_robot_id

        self._termination_event = multiprocessing.Event()
        self._cfg_queues = [multiprocessing.Queue() for _ in range(self._num_workers)]
        # NOTE: `multiprocessing.Queue` is a factory function; use `multiprocessing.queues.Queue` for typing.
        self._results_queue: multiprocessing.queues.Queue[Any] = multiprocessing.Queue()
        self._task_info_queue: multiprocessing.queues.Queue[Any] = multiprocessing.Queue()
        self._obs_method_queue: multiprocessing.queues.Queue[Any] = multiprocessing.Queue()

        self._procs: list[multiprocessing.Process] = []

    def _spawn_worker(self, worker_idx: int, *, publish_bootstrap_metadata: bool) -> None:
        p = multiprocessing.Process(
            target=_worker_entry,
            kwargs=dict(
                worker_idx=worker_idx,
                task=self._task,
                device=self._device,
                env_seed=self._env_seed,
                num_envs=self._num_envs,
                headless=self._headless,
                max_train_epoch=self._max_train_epoch,
                kuafu_run_dir=self._kuafu_run_dir,
                cfg_queue=self._cfg_queues[worker_idx],
                results_queue=self._results_queue,
                task_info_queue=self._task_info_queue,
                obs_method_queue=self._obs_method_queue,
                termination_event=self._termination_event,
                initial_prior_checkpoint=self._initial_prior_checkpoint,
                execution_robot_id=self._execution_robot_id,
                publish_bootstrap_metadata=publish_bootstrap_metadata,
            ),
        )
        # Non-daemon so we can join/terminate cleanly on KeyboardInterrupt.
        p.daemon = False
        p.start()
        if worker_idx < len(self._procs):
            self._procs[worker_idx] = p
        else:
            self._procs.append(p)

    def _respawn_worker(self, worker_idx: int) -> None:
        old_proc = self._procs[worker_idx]
        try:
            old_proc.join(timeout=0)
        except Exception as exc:
            _warn_nonfatal(f"joining old worker {worker_idx} before respawn", exc)

        old_queue = self._cfg_queues[worker_idx]
        self._cfg_queues[worker_idx] = multiprocessing.Queue()
        try:
            old_queue.close()
        except Exception as exc:
            _warn_nonfatal(f"closing old config queue for worker {worker_idx}", exc)
        try:
            old_queue.join_thread()
        except Exception as exc:
            _warn_nonfatal(f"joining old config queue thread for worker {worker_idx}", exc)

        print(f"[residual_rl] Respawning dead worker {worker_idx} with a fresh config queue.", flush=True)
        self._spawn_worker(worker_idx, publish_bootstrap_metadata=False)

    def _ensure_live_workers(self) -> None:
        for worker_idx in range(self._num_workers):
            proc = self._procs[worker_idx]
            if proc.is_alive():
                continue
            self._respawn_worker(worker_idx)

    def start(self) -> None:
        for idx in range(self._num_workers):
            self._spawn_worker(idx, publish_bootstrap_metadata=(idx == 0))

    def close(self) -> None:
        self._termination_event.set()
        for q in self._cfg_queues:
            try:
                q.put_nowait(None)
            except Exception as exc:
                _warn_nonfatal("sending worker shutdown sentinel", exc)
        for p in self._procs:
            try:
                p.join(timeout=5)
            except Exception as exc:
                _warn_nonfatal(f"joining worker pid={getattr(p, 'pid', None)}", exc)
        for p in self._procs:
            try:
                if p.is_alive():
                    p.terminate()
            except Exception as exc:
                _warn_nonfatal(f"terminating worker pid={getattr(p, 'pid', None)}", exc)
        for p in self._procs:
            try:
                p.join(timeout=5)
            except Exception as exc:
                _warn_nonfatal(f"joining terminated worker pid={getattr(p, 'pid', None)}", exc)

    def get_task_info(self, timeout_s: float = 60.0) -> dict | None:
        try:
            return self._task_info_queue.get(timeout=timeout_s)
        except queue.Empty:
            return None

    def get_observations_method(self, timeout_s: float = 60.0) -> str | None:
        try:
            return self._obs_method_queue.get(timeout=timeout_s)
        except queue.Empty:
            return None

    def _log_failures(self, results: list[TrainResult], cfgs: list[TrainConfig]) -> None:
        if not self._worker_failures_path:
            return

        failure_records: list[dict[str, Any]] = []
        for result in results:
            if result.success or result.skipped:
                continue

            cfg = cfgs[result.worker_idx] if 0 <= result.worker_idx < len(cfgs) else None
            failure_records.append(
                {
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "task": self._task,
                    "worker_idx": int(result.worker_idx),
                    "cfg_tag": getattr(cfg, "tag", "") if cfg is not None else "",
                    "cfg_stage_id": getattr(cfg, "stage_id", None) if cfg is not None else None,
                    "cfg_progress_metric_expr": getattr(cfg, "progress_metric_expr", None) if cfg is not None else None,
                    "cfg_max_train_epoch_override": (
                        getattr(cfg, "max_train_epoch_override", None) if cfg is not None else None
                    ),
                    "cfg_resume_from_checkpoint": (
                        getattr(cfg, "resume_from_checkpoint", None) if cfg is not None else None
                    ),
                    "cfg_eval_only_checkpoint": (
                        getattr(cfg, "eval_only_checkpoint", None) if cfg is not None else None
                    ),
                    "cfg_prior_checkpoint": getattr(cfg, "prior_checkpoint", None) if cfg is not None else None,
                    "cfg_agent_hyperparameters": (
                        getattr(cfg, "agent_hyperparameters", None) if cfg is not None else None
                    ),
                    "exception": result.exception,
                    "traceback": result.exception_traceback,
                    "log_dir": result.log_dir,
                    "checkpoint_path": result.checkpoint_path,
                }
            )

        if not failure_records:
            return

        os.makedirs(os.path.dirname(self._worker_failures_path), exist_ok=True)
        with open(self._worker_failures_path, "a", encoding="utf-8") as f:
            for record in failure_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def _train_result_from_payload(worker_idx: int, payload_obj: Any) -> TrainResult:
        payload = cast(dict[str, Any], payload_obj)
        return TrainResult(
            worker_idx=int(worker_idx),
            success=bool(payload.get("success", False)),
            log_dir=payload.get("log_dir"),
            checkpoint_path=payload.get("checkpoint_path"),
            progress_metric_eval=payload.get("progress_metric_eval", payload.get("success_metric_eval")),
            success_metric_eval=payload.get("success_metric_eval", payload.get("progress_metric_eval")),
            reward_metric_eval=payload.get("reward_metric_eval"),
            exception=payload.get("exception"),
            exception_traceback=payload.get("exception_traceback"),
            skipped=bool(payload.get("skipped", False)),
            stage_transition_stats=payload.get("stage_transition_stats"),
            highest_stage_reached_per_env=payload.get("highest_stage_reached_per_env"),
            constraint_met_rate_stats=payload.get("constraint_met_rate_stats"),
            true_success_count=payload.get("true_success_count"),
            false_complete_count=payload.get("false_complete_count"),
            timeout_count=payload.get("timeout_count"),
            selection_success_rate_window=payload.get("selection_success_rate_window"),
            selection_success_rate_ema=payload.get("selection_success_rate_ema"),
            selection_false_complete_rate=payload.get("selection_false_complete_rate"),
            selection_timeout_rate=payload.get("selection_timeout_rate"),
            selection_metric_tag=payload.get("selection_metric_tag"),
            selection_metric_index=payload.get("selection_metric_index"),
            selection_metric_step=payload.get("selection_metric_step"),
            selection_critic_warmup_epochs=payload.get("selection_critic_warmup_epochs"),
            selection_checkpoint_path=payload.get("selection_checkpoint_path"),
            selection_checkpoint_epoch=payload.get("selection_checkpoint_epoch"),
            selection_checkpoint_policy=payload.get("selection_checkpoint_policy"),
            selection_source=payload.get("selection_source"),
            kuafu_training_parameters_path=payload.get("kuafu_training_parameters_path"),
            applied_stage_weights=payload.get("applied_stage_weights"),
            applied_stage_parameters=payload.get("applied_stage_parameters"),
            applied_agent_hyperparameters=payload.get("applied_agent_hyperparameters"),
        )

    def _dead_worker_result(self, worker_idx: int, exit_code: int | None) -> TrainResult:
        worker_output_path = _worker_output_path(self._kuafu_run_dir, worker_idx)
        tail = None
        if worker_output_path and os.path.isfile(worker_output_path):
            try:
                tail = _tail_text(worker_output_path)
            except Exception as exc:
                tail = f"[residual_rl] Failed to read worker log tail from {worker_output_path}: {exc!r}"

        exception = "[residual_rl] Worker process died before reporting a result."
        if exit_code is not None:
            exception += f" exit_code={exit_code}"
        if worker_output_path:
            exception += f" log={worker_output_path}"

        exception_traceback = None
        if tail:
            exception_traceback = f"[residual_rl] Last worker output from {worker_output_path}:\n{tail}"

        return TrainResult(
            worker_idx=int(worker_idx),
            success=False,
            exception=exception,
            exception_traceback=exception_traceback,
            skipped=False,
        )

    def train(self, cfgs: list[TrainConfig]) -> list[TrainResult]:
        # Fan out configs (pad if needed).
        #
        # Some callers (e.g. the optional dry run) intentionally train a single config even when
        # multiple workers are available. In that case, pad remaining workers with skip=True so
        # we still receive exactly one result per worker.
        self._ensure_live_workers()
        cfgs = list(cfgs)
        if len(cfgs) > self._num_workers:
            cfgs = cfgs[: self._num_workers]
        while len(cfgs) < self._num_workers:
            cfgs.append(
                TrainConfig(
                    stage_id=None,
                    all_stages_weights=None,
                    all_stages_parameters=None,
                    progress_metric_expr=None,
                    max_train_epoch_override=None,
                    resume_from_checkpoint=None,
                    skip=True,
                    tag="skip",
                )
            )
        for idx in range(self._num_workers):
            self._cfg_queues[idx].put(cfgs[idx])

        results_by_worker: dict[int, TrainResult] = {}
        pending_workers = set(range(self._num_workers))
        results_reader = getattr(self._results_queue, "_reader", None)
        if results_reader is None:
            raise RuntimeError("[residual_rl] multiprocessing.Queue is missing its reader handle.")

        while pending_workers:
            waitables = [results_reader] + [self._procs[idx].sentinel for idx in sorted(pending_workers)]
            ready = multiprocessing.connection.wait(waitables)

            # Drain any queued results first so a worker that exits right after reporting is
            # recorded from its payload instead of being misclassified as a hard crash.
            if results_reader in ready:
                while True:
                    try:
                        worker_idx, payload_obj = self._results_queue.get_nowait()
                    except queue.Empty:
                        break
                    worker_idx = int(worker_idx)
                    if worker_idx not in pending_workers:
                        continue
                    results_by_worker[worker_idx] = self._train_result_from_payload(worker_idx, payload_obj)
                    pending_workers.remove(worker_idx)

            for worker_idx in list(sorted(pending_workers)):
                proc = self._procs[worker_idx]
                if proc.sentinel not in ready:
                    continue
                if proc.is_alive():
                    continue
                exit_code = proc.exitcode
                print(
                    f"[residual_rl] Worker {worker_idx} exited before posting a result; "
                    f"recording failure (exit_code={exit_code}).",
                    flush=True,
                )
                results_by_worker[worker_idx] = self._dead_worker_result(worker_idx, exit_code)
                pending_workers.remove(worker_idx)

        results = [results_by_worker[idx] for idx in sorted(results_by_worker)]
        # keep stable ordering
        self._log_failures(results, cfgs)
        if self._stop_on_error:
            failures = [r for r in results if not r.success and not r.skipped]
            if failures:
                summary = "; ".join(
                    f"worker {r.worker_idx}: {r.exception or 'unknown error'}" for r in failures
                )
                extra = (
                    f" See {self._worker_failures_path} for tracebacks."
                    if self._worker_failures_path
                    else ""
                )
                raise RuntimeError(f"[residual_rl] stop_on_error triggered: {summary}{extra}")
        return results


def _worker_entry(
    *,
    worker_idx: int,
    task: str,
    device: str,
    env_seed: int,
    num_envs: int | None,
    headless: bool,
    max_train_epoch: int,
    kuafu_run_dir: str | None,
    cfg_queue: Any,
    results_queue: Any,
    task_info_queue: Any,
    obs_method_queue: Any,
    termination_event: Any,
    initial_prior_checkpoint: str | None,
    execution_robot_id: str | None,
    publish_bootstrap_metadata: bool,
) -> None:
    # IMPORTANT: keep imports light here; Isaac Sim is launched later.
    env = None
    simulation_app = None
    agent_cfg = None
    worker_log_file = None
    try:
        worker_log_file, _worker_output_log_path = _configure_worker_logging(kuafu_run_dir, worker_idx)

        from kuafu.rl.isaac.env_factory import create_env_and_app, get_observations_method_as_string
        from kuafu.rl.isaac.update_utilities import apply_updates_and_wrap_reset
        from kuafu.rl.isaac.train_rl_games import evaluate_rl_games, train_rl_games

        def _close_env_and_app() -> None:
            nonlocal env, simulation_app, agent_cfg

            env_to_close = env
            sim_to_close = simulation_app
            env = None
            simulation_app = None
            agent_cfg = None

            try:
                if env_to_close is not None:
                    env_to_close.close()
            except Exception as exc:
                _warn_nonfatal("worker env close", exc)
            try:
                if sim_to_close is not None:
                    sim_to_close.close()
            except Exception as exc:
                _warn_nonfatal("worker SimulationApp close", exc)

            # Isaac/PhysX resources can linger briefly after close(); best-effort cache
            # cleanup makes per-stage simulator recreation more reliable under VRAM pressure.
            try:
                import gc

                gc.collect()
            except Exception as exc:
                _warn_nonfatal("worker Python GC after env close", exc)
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as exc:
                _warn_nonfatal("worker CUDA cache cleanup", exc)

        def _create_env_bundle() -> None:
            nonlocal env, simulation_app, agent_cfg

            simulation_app, env, agent_cfg = create_env_and_app(
                task=task,
                device=device_used,
                env_seed=env_seed + int(worker_idx),
                num_envs=num_envs,
                headless=headless,
                execution_robot_id=execution_robot_id,
                initial_prior_checkpoint=initial_prior_checkpoint,
                agent_cfg_path=os.environ.get("KUAFU_AGENT_CFG_PATH") or None,
            )


        # ---- Distribute workers across GPUs (one GPU per worker, round-robin) ----
        # By default all workers would use the same GPU. Pin each worker to a different
        # GPU so they share the available GPUs (worker 0 -> GPU 0, worker 1 -> GPU 1, ...).
        # - If CUDA_VISIBLE_DEVICES is set, we map workers onto those IDs.
        # - Else we use 0..device_count()-1. After pinning, only that GPU is visible in
        #   this process, so wepen pass device_used="cuda:0" downstream.
        device_in = str(device)
        device_used = device_in
        if device_in.startswith("cuda"):
            cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
            chosen: str | None = None
            if cvd:
                ids = [s.strip() for s in cvd.split(",") if s.strip()]
                if ids:
                    chosen = ids[int(worker_idx) % len(ids)]
            else:
                try:
                    import torch
                    n = int(torch.cuda.device_count())
                except Exception as exc:
                    raise RuntimeError("[residual_rl] Failed to inspect CUDA device count for worker pinning.") from exc
                if n > 0:
                    chosen = str(int(worker_idx) % n)
            if chosen is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = chosen
                device_used = "cuda:0"
            print(
                f"[residual_rl] Worker {worker_idx}: GPU pinning -> CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} device_used={device_used}",
                flush=True,
            )

        # Send required metadata once from the initial bootstrap worker.
        # Keep this initial env alive so that worker can reuse it for its first
        # real cfg instead of tearing down and recreating Isaac immediately.
        if publish_bootstrap_metadata:
            _create_env_bundle()
            obs_method_queue.put(get_observations_method_as_string(env))
            task_info_queue.put(_collect_task_info(env, task=task, agent_cfg=agent_cfg))

        while not termination_event.is_set():
            cfg = cfg_queue.get()
            if cfg is None:
                # Sentinel from close(): exit loop without raising.
                break
            try:
                if isinstance(cfg, TrainConfig) and bool(cfg.skip):
                    print(f"[residual_rl] Worker {worker_idx}: skip=True tag={cfg.tag}", flush=True)
                    results_queue.put((worker_idx, {"success": True, "skipped": True}))
                    continue

                print(
                    "[residual_rl] Worker {wi} (GPU={gpu}): received cfg: tag={tag} stage_id={stage_id} "
                    "has_all_stages_weights={has_all} has_all_stages_parameters={has_all_p} "
                    "max_train_epoch_override={mti} resume_from_checkpoint={resume} "
                    "eval_only_checkpoint={eval_only} prior_checkpoint={prior} prior_mode={prior_mode} "
                    "prior_critic_init_mode={critic_mode} prior_freeze_actor_epochs={freeze} "
                    "prior_sigma_ramp_epochs={sigma_ramp} "
                    "prior_sigma_final_val={sigma_final} "
                    "prior_actor_lr_warmup_epochs={lr_warmup} "
                    "prior_actor_lr_cap={lr_cap} "
                    "prior_freeze_input_rms={freeze_rms} prior_unfreeze_input_rms_after={unfreeze_rms} "
                    "prior_rms_unfreeze_after_actor_lr_warmup={rms_defer} "
                    "prior_lr_schedule_freeze_during_warmup={sched_freeze} "
                    "prior_fixed_lr={fixed_lr} "
                    "prior_bc_dataset_coef={bc_ds_coef} "
                    "prior_bc_dataset_decay_epochs={bc_ds_decay} "
                    "prior_bc_dataset_batch_size={bc_ds_bs} "
                    "agent_hyperparameters={agent_hparams}".format(
                        wi=worker_idx,
                        gpu=os.environ.get("CUDA_VISIBLE_DEVICES", device_used),
                        tag=getattr(cfg, "tag", ""),
                        stage_id=getattr(cfg, "stage_id", None),
                        has_all=bool(getattr(cfg, "all_stages_weights", None)),
                        has_all_p=bool(getattr(cfg, "all_stages_parameters", None)),
                        mti=getattr(cfg, "max_train_epoch_override", None),
                        resume=bool(getattr(cfg, "resume_from_checkpoint", None)),
                        eval_only=bool(getattr(cfg, "eval_only_checkpoint", None)),
                        prior=bool(getattr(cfg, "prior_checkpoint", None)),
                        prior_mode=getattr(cfg, "prior_mode", "full_actor"),
                        critic_mode=getattr(cfg, "prior_critic_init_mode", "actor_only_random_critic"),
                        freeze=getattr(cfg, "prior_freeze_actor_epochs", 0),
                        sigma_ramp=getattr(cfg, "prior_sigma_ramp_epochs", 0),
                        sigma_final=getattr(cfg, "prior_sigma_final_val", None),
                        lr_warmup=getattr(cfg, "prior_actor_lr_warmup_epochs", 0),
                        lr_cap=getattr(cfg, "prior_actor_lr_cap", 0.0),
                        freeze_rms=getattr(cfg, "prior_freeze_input_running_mean_std", False),
                        unfreeze_rms=getattr(
                            cfg, "prior_unfreeze_input_running_mean_std_after_freeze", False
                        ),
                        rms_defer=getattr(
                            cfg, "prior_rms_unfreeze_after_actor_lr_warmup", False
                        ),
                        sched_freeze=getattr(
                            cfg, "prior_lr_schedule_freeze_during_warmup", False
                        ),
                        fixed_lr=getattr(cfg, "prior_fixed_lr", None),
                        bc_ds_coef=getattr(cfg, "prior_bc_dataset_coef", 0.0),
                        bc_ds_decay=getattr(cfg, "prior_bc_dataset_decay_epochs", 0),
                        bc_ds_bs=getattr(cfg, "prior_bc_dataset_batch_size", 1024),
                        agent_hparams=getattr(cfg, "agent_hyperparameters", None),
                    ),
                    flush=True,
                )

                # Recreate Isaac Sim resources for subsequent runs so PhysX/CUDA state
                # from prior stages cannot accumulate inside a long-lived worker process.
                # Worker 0 reuses its bootstrap env for the first real cfg.
                if env is None or simulation_app is None or agent_cfg is None:
                    _create_env_bundle()

                # Apply weights/parameters and install episode-level logging hooks.
                # Catch failures from bad LLM output here so this worker reports failure and others proceed.
                if isinstance(cfg, TrainConfig):
                    try:
                        if cfg.stage_id is None:
                            raise RuntimeError("[residual_rl] Worker config is missing stage_id.")
                        apply_updates_and_wrap_reset(
                            env,
                            stage_id=int(cfg.stage_id),
                            all_stages_weights=cfg.all_stages_weights,
                            all_stages_parameters=cfg.all_stages_parameters,
                            progress_metric_expr=cfg.progress_metric_expr,
                        )
                    except Exception as e:
                        msg = f"{type(e).__name__}: {repr(e)}"
                        tb = traceback.format_exc()
                        print("[residual_rl] Worker {} failed to apply weights/parameters to program_runtime: {}".format(worker_idx, msg), flush=True)
                        print(tb, flush=True)
                        results_queue.put(
                            (
                                worker_idx,
                                {
                                    "success": False,
                                    "exception": msg,
                                    "exception_traceback": tb,
                                    "skipped": False,
                                },
                            )
                        )
                        continue
                else:
                    raise TypeError(f"Unsupported cfg type: {type(cfg)}")

                applied_agent_hyperparameters = _apply_agent_hyperparameters(agent_cfg, cfg.agent_hyperparameters)
                if cfg.agent_hyperparameters:
                    print(
                        f"[residual_rl] Worker {worker_idx}: applied residual agent hyperparameters "
                        f"{json.dumps(applied_agent_hyperparameters, sort_keys=True)}",
                        flush=True,
                    )

                selection_metadata: dict[str, object | None]
                if cfg.eval_only_checkpoint:
                    checkpoint_path = str(cfg.eval_only_checkpoint)
                    log_dir = os.path.dirname(checkpoint_path) or kuafu_run_dir or "."
                    selection_metadata = {
                        "selection_success_rate_window": None,
                        "selection_success_rate_ema": None,
                        "selection_false_complete_rate": None,
                        "selection_timeout_rate": None,
                        "selection_metric_tag": _PEAK_SELECTION_SUCCESS_RATE_WINDOW_TAG,
                        "selection_metric_index": None,
                        "selection_metric_step": None,
                        "selection_critic_warmup_epochs": _extract_critic_warmup_epochs(agent_cfg),
                        "selection_checkpoint_path": checkpoint_path,
                        "selection_checkpoint_epoch": _checkpoint_epoch_from_path(checkpoint_path),
                        "selection_checkpoint_policy": "eval_only_checkpoint",
                        "selection_source": "eval_only_checkpoint",
                    }
                else:
                    log_dir, checkpoint_path = train_rl_games(
                        env=env,
                        task=task,
                        device=device_used,
                        max_train_epoch=int(cfg.max_train_epoch_override) if cfg.max_train_epoch_override is not None else max_train_epoch,
                        worker_idx=worker_idx,
                        seed=int(env_seed) + int(worker_idx),
                        resume_from_checkpoint=cfg.resume_from_checkpoint,
                        prior_checkpoint=cfg.prior_checkpoint,
                        prior_mode=cfg.prior_mode,
                        prior_obs_stats_count=cfg.prior_obs_stats_count,
                        prior_verbose=cfg.prior_verbose,
                        prior_critic_init_mode=cfg.prior_critic_init_mode,
                        prior_freeze_actor_epochs=cfg.prior_freeze_actor_epochs,
                        prior_sigma_ramp_epochs=cfg.prior_sigma_ramp_epochs,
                        prior_sigma_final_val=cfg.prior_sigma_final_val,
                        prior_actor_lr_warmup_epochs=cfg.prior_actor_lr_warmup_epochs,
                        prior_actor_lr_cap=cfg.prior_actor_lr_cap,
                        prior_freeze_input_running_mean_std=cfg.prior_freeze_input_running_mean_std,
                        prior_unfreeze_input_running_mean_std_after_freeze=(
                            cfg.prior_unfreeze_input_running_mean_std_after_freeze
                        ),
                        prior_rms_unfreeze_after_actor_lr_warmup=(
                            cfg.prior_rms_unfreeze_after_actor_lr_warmup
                        ),
                        prior_lr_schedule_freeze_during_warmup=(
                            cfg.prior_lr_schedule_freeze_during_warmup
                        ),
                        prior_fixed_lr=cfg.prior_fixed_lr,
                        prior_bc_anchor_epochs=cfg.prior_bc_anchor_epochs,
                        prior_bc_anchor_coef=cfg.prior_bc_anchor_coef,
                        prior_bc_dataset_coef=cfg.prior_bc_dataset_coef,
                        prior_bc_dataset_decay_epochs=cfg.prior_bc_dataset_decay_epochs,
                        prior_bc_dataset_batch_size=cfg.prior_bc_dataset_batch_size,
                        prior_bc_dataset_path=cfg.prior_bc_dataset_path,
                        prior_bc_action_reward_coef=cfg.prior_bc_action_reward_coef,
                        kuafu_run_dir=kuafu_run_dir,
                        agent_cfg=agent_cfg,
                    )
                    checkpoint_path, selection_metadata = _select_post_critic_peak_checkpoint(
                        log_dir=str(log_dir),
                        fallback_checkpoint=checkpoint_path,
                        critic_warmup_epochs=_extract_critic_warmup_epochs(agent_cfg),
                    )
                env_u = cast(Any, env.unwrapped)
                if not checkpoint_path:
                    raise RuntimeError("[residual_rl] Missing checkpoint for eval.")
                if not cfg.progress_metric_expr:
                    raise RuntimeError("[residual_rl] Worker config is missing progress_metric_expr for eval.")

                eval_payload = evaluate_rl_games(
                    env=env,
                    task=task,
                    device=device_used,
                    checkpoint_path=str(checkpoint_path),
                    prior_checkpoint=cfg.prior_checkpoint,
                    prior_mode=cfg.prior_mode,
                    agent_cfg=agent_cfg,
                    progress_metric_expr=str(cfg.progress_metric_expr),
                    seed=int(env_seed) + int(worker_idx),
                )

                progress_metric_eval = float(cast(Any, eval_payload["progress_metric_eval"]))
                reward_metric_eval = float(cast(Any, eval_payload["average_episodic_return"]))
                stage_transition_stats = cast(dict, eval_payload["stage_transition_stats"])
                highest_stage_reached_per_env = cast(list[int], eval_payload["highest_stage_reached_per_env"])
                constraint_met_rate_stats = cast(dict, eval_payload["constraint_met_rate_stats"])
                true_success_count = int(cast(Any, eval_payload["true_success_count"]))
                false_complete_count = int(cast(Any, eval_payload["false_complete_count"]))
                timeout_count = int(cast(Any, eval_payload["timeout_count"]))

                # Read back the canonical/applied weights + parameters from the env.
                program_runtime = env_u.program_runtime
                sid = int(cfg.stage_id)

                w = program_runtime.get_residual_weights(sid)
                applied_stage_weights = {int(k): float(v) for k, v in dict(w).items()}
                cdef = program_runtime.get_constraints_by_stage(sid)
                terms = cdef["Constraints"]

                allowed = {"tolerance_m", "tolerance_rad"}
                params_out: dict[int, dict[str, object]] = {}
                for idx, term in enumerate(list(terms)):
                    args = term.get("args", {})
                    patch: dict[str, object] = {}
                    for field in allowed:
                        if field in args:
                            patch[field] = float(args[field])
                    params_out[int(idx)] = patch
                applied_stage_parameters = params_out

                if cfg.eval_only_checkpoint:
                    phase = "eval_only"
                else:
                    phase = "global" if "global" in str(getattr(cfg, "tag", "")) else f"stage{getattr(cfg, 'stage_id', '?')}"
                phase_summary = "eval done" if cfg.eval_only_checkpoint else "train+eval done"
                kuafu_training_parameters_path = None
                if not cfg.eval_only_checkpoint:
                    run_dir_for_params = _run_dir_from_log_dir(str(log_dir))
                    kuafu_training_parameters_path = _write_kuafu_training_parameters(
                        log_dir=str(log_dir),
                        record={
                            "schema_version": 1,
                            "timestamp": datetime.datetime.now().isoformat(),
                            "task": task,
                            "phase": phase,
                            "worker_idx": int(worker_idx),
                            "tag": getattr(cfg, "tag", ""),
                            "stage_id": getattr(cfg, "stage_id", None),
                            "device_requested": device,
                            "device_used": device_used,
                            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                            "env_seed": int(env_seed) + int(worker_idx),
                            "num_envs": num_envs,
                            "progress_metric_expr": cfg.progress_metric_expr,
                            "max_train_epoch_default": int(max_train_epoch),
                            "max_train_epoch_override": cfg.max_train_epoch_override,
                            "resume_from_checkpoint": cfg.resume_from_checkpoint,
                            "prior": {
                                "checkpoint": cfg.prior_checkpoint,
                                "mode": cfg.prior_mode,
                                "obs_stats_count": cfg.prior_obs_stats_count,
                                "verbose": cfg.prior_verbose,
                                "critic_init_mode": cfg.prior_critic_init_mode,
                                "freeze_actor_epochs": cfg.prior_freeze_actor_epochs,
                                "sigma_ramp_epochs": cfg.prior_sigma_ramp_epochs,
                                "sigma_final_val": cfg.prior_sigma_final_val,
                                "actor_lr_warmup_epochs": cfg.prior_actor_lr_warmup_epochs,
                                "actor_lr_cap": cfg.prior_actor_lr_cap,
                                "freeze_input_running_mean_std": cfg.prior_freeze_input_running_mean_std,
                                "unfreeze_input_running_mean_std_after_freeze": (
                                    cfg.prior_unfreeze_input_running_mean_std_after_freeze
                                ),
                                "rms_unfreeze_after_actor_lr_warmup": (
                                    cfg.prior_rms_unfreeze_after_actor_lr_warmup
                                ),
                                "lr_schedule_freeze_during_warmup": cfg.prior_lr_schedule_freeze_during_warmup,
                                "fixed_lr": cfg.prior_fixed_lr,
                                "bc_anchor_epochs": cfg.prior_bc_anchor_epochs,
                                "bc_anchor_coef": cfg.prior_bc_anchor_coef,
                                "bc_dataset_coef": cfg.prior_bc_dataset_coef,
                                "bc_dataset_decay_epochs": cfg.prior_bc_dataset_decay_epochs,
                                "bc_dataset_batch_size": cfg.prior_bc_dataset_batch_size,
                                "bc_dataset_path": cfg.prior_bc_dataset_path,
                                "bc_action_reward_coef": cfg.prior_bc_action_reward_coef,
                            },
                            "requested_agent_hyperparameters": cfg.agent_hyperparameters,
                            "applied_agent_hyperparameters": applied_agent_hyperparameters,
                            "all_stages_weights": cfg.all_stages_weights,
                            "all_stages_parameters": cfg.all_stages_parameters,
                            "applied_stage_weights": applied_stage_weights,
                            "applied_stage_parameters": applied_stage_parameters,
                            "checkpoint_selection": selection_metadata,
                            "outputs": {
                                "log_dir": log_dir,
                                "run_dir": str(run_dir_for_params),
                                "checkpoint_path": checkpoint_path,
                                "progress_metric_eval": progress_metric_eval,
                                "average_episodic_return_eval": reward_metric_eval,
                                "true_success_count": true_success_count,
                                "false_complete_count": false_complete_count,
                                "timeout_count": timeout_count,
                            },
                            "config_files": {
                                "agent_yaml": str(run_dir_for_params / "params" / "agent.yaml"),
                                "env_yaml": str(run_dir_for_params / "params" / "env.yaml"),
                                "runtime_yaml": str(run_dir_for_params / "params" / "runtime.yaml"),
                            },
                        },
                    )
                print(
                    f"[residual_rl] Worker {worker_idx} (GPU={os.environ.get('CUDA_VISIBLE_DEVICES', device_used)}, "
                    f"phase={phase}): {phase_summary}. "
                    f"log_dir={log_dir} checkpoint={checkpoint_path} "
                    f"selection_source={selection_metadata.get('selection_source')} "
                    f"selection_success_rate_window={selection_metadata.get('selection_success_rate_window')} "
                    f"selection_metric_index={selection_metadata.get('selection_metric_index')} "
                    f"selection_critic_warmup_epochs={selection_metadata.get('selection_critic_warmup_epochs')} "
                    f"kuafu_training_parameters={kuafu_training_parameters_path} "
                    f"eval_progress_rate={progress_metric_eval} "
                    f"eval_average_episodic_return={reward_metric_eval}",
                    flush=True,
                )

                results_queue.put(
                    (
                        worker_idx,
                        {
                            "success": True,
                            "skipped": False,
                            "log_dir": log_dir,
                            "checkpoint_path": checkpoint_path,
                            "progress_metric_eval": progress_metric_eval,
                            "reward_metric_eval": reward_metric_eval,
                            "stage_transition_stats": stage_transition_stats,
                            "highest_stage_reached_per_env": highest_stage_reached_per_env,
                            "constraint_met_rate_stats": constraint_met_rate_stats,
                            "true_success_count": true_success_count,
                            "false_complete_count": false_complete_count,
                            "timeout_count": timeout_count,
                            "selection_success_rate_window": selection_metadata.get("selection_success_rate_window"),
                            "selection_success_rate_ema": selection_metadata.get("selection_success_rate_ema"),
                            "selection_false_complete_rate": selection_metadata.get("selection_false_complete_rate"),
                            "selection_timeout_rate": selection_metadata.get("selection_timeout_rate"),
                            "selection_metric_tag": selection_metadata.get("selection_metric_tag"),
                            "selection_metric_index": selection_metadata.get("selection_metric_index"),
                            "selection_metric_step": selection_metadata.get("selection_metric_step"),
                            "selection_critic_warmup_epochs": selection_metadata.get("selection_critic_warmup_epochs"),
                            "selection_checkpoint_path": selection_metadata.get("selection_checkpoint_path"),
                            "selection_checkpoint_epoch": selection_metadata.get("selection_checkpoint_epoch"),
                            "selection_checkpoint_policy": selection_metadata.get("selection_checkpoint_policy"),
                            "selection_source": selection_metadata.get("selection_source"),
                            "kuafu_training_parameters_path": kuafu_training_parameters_path,
                            "applied_stage_weights": applied_stage_weights,
                            "applied_stage_parameters": applied_stage_parameters,
                            "applied_agent_hyperparameters": applied_agent_hyperparameters,
                        },
                    )
                )
            except Exception as e:
                msg = f"{type(e).__name__}: {repr(e)}"
                tb = traceback.format_exc()
                print("[residual_rl] Worker exception traceback:", flush=True)
                print(tb, flush=True)
                results_queue.put(
                    (
                        worker_idx,
                        {
                            "success": False,
                            "exception": msg,
                            "exception_traceback": tb,
                            "skipped": False,
                        },
                    )
                )
            finally:
                _close_env_and_app()
    except Exception as e:
        msg = f"{type(e).__name__}: {repr(e)}"
        tb = traceback.format_exc()
        print("[residual_rl] Worker bootstrap exception traceback:", flush=True)
        print(tb, flush=True)
        results_queue.put(
            (
                worker_idx,
                {
                    "success": False,
                    "exception": msg,
                    "exception_traceback": tb,
                    "skipped": False,
                },
            )
        )
    finally:
        try:
            if env is not None:
                env.close()
        except Exception as exc:
            _warn_nonfatal("worker final env close", exc)
        try:
            if simulation_app is not None:
                simulation_app.close()
        except Exception as exc:
            _warn_nonfatal("worker final SimulationApp close", exc)
        try:
            if worker_log_file is not None:
                worker_log_file.flush()
                worker_log_file.close()
        except Exception as exc:
            _warn_nonfatal("worker log file close", exc)


def _collect_task_info(env, *, task: str, agent_cfg: dict[str, Any] | None = None) -> dict[str, Any] | None:
    env_u = env.unwrapped
    program_runtime = getattr(env_u, "program_runtime", None)
    if program_runtime is None:
        raise RuntimeError(f"[residual_rl] Task {task} environment is missing program_runtime.")
    info: dict[str, Any] = {}
    cfg = getattr(env_u, "cfg", None) or getattr(env, "cfg", None)
    if cfg is None:
        raise RuntimeError(f"[residual_rl] Task {task} environment is missing cfg.")
    task_desc = getattr(cfg, "task_description", None)
    if not isinstance(task_desc, str) or not task_desc.strip():
        raise RuntimeError(f"[residual_rl] Env cfg for task {task} is missing task_description.")
    info["task_description"] = task_desc.strip()
    detectable_objects = getattr(cfg, "detectable_objects", None)
    if detectable_objects is None:
        raise RuntimeError(f"[residual_rl] Env cfg for task {task} is missing detectable_objects.")
    info["detectable_objects"] = detectable_objects
    stage_count = int(program_runtime.compiled_program.stage_count())
    if stage_count <= 0:
        raise RuntimeError(f"[residual_rl] Task {task} reported invalid stage_count={stage_count}.")
    info["stage_count"] = stage_count
    action_mode = getattr(cfg, "action_mode", None)
    try:
        action_dim = int(getattr(env_u.single_action_space, "shape")[0])
    except Exception:
        action_dim = int(getattr(cfg, "action_space", 0) or 0)
    info["action_mode"] = str(action_mode or "")
    info["action_dim"] = int(action_dim)
    info["action_dimension_labels"] = _action_dimension_labels(action_mode, action_dim)
    stages: dict[int, dict[str, Any]] = {}
    for stage_id in range(stage_count):
        atomic_action = program_runtime.get_atomic_action_by_id(stage_id)
        if not isinstance(atomic_action, str) or not atomic_action.strip():
            raise RuntimeError(f"[residual_rl] Stage {stage_id} is missing atomic_action.")
        cdict = program_runtime.get_constraints_by_stage(stage_id)
        if not isinstance(cdict, dict):
            raise TypeError(f"[residual_rl] Stage {stage_id} constraints must be a dict.")
        # Keep only what is useful for prompting: op + args for each constraint.
        constraints = cdict.get("Constraints")
        if not isinstance(constraints, list) or not constraints:
            raise RuntimeError(f"[residual_rl] Stage {stage_id} constraints must be a non-empty list.")
        clist = []
        for constraint_id, c in enumerate(constraints):
            if not isinstance(c, dict):
                raise TypeError(
                    f"Stage {stage_id} constraint {constraint_id} must be a dict, got {type(c).__name__}."
                )
            if "op" not in c or "args" not in c:
                raise KeyError(f"[residual_rl] Stage {stage_id} constraint {constraint_id} is missing 'op' or 'args'.")
            clist.append({"op": c["op"], "args": c["args"]})
        constraints_raw = clist
        stage_reasoning_raw = cdict.get("Reasoning")
        if not isinstance(stage_reasoning_raw, str) or not stage_reasoning_raw.strip():
            raise RuntimeError(f"[residual_rl] Stage {stage_id} is missing Reasoning.")
        stage_reasoning = stage_reasoning_raw.strip()
        explanations = program_runtime.explain_constraints(stage_id)
        if not isinstance(explanations, list) or not explanations:
            raise RuntimeError(f"[residual_rl] Stage {stage_id} explanations must be a non-empty list.")
        stages[stage_id] = {
            "explanations": explanations,
            "weights": program_runtime.get_residual_weights(stage_id),
            "atomic_action": atomic_action.strip(),
            "constraints_raw": constraints_raw,
            "reasoning": stage_reasoning,
        }

    info["stages"] = stages
    # RL-Games checkpoint cadence (used for post-solve “golden period” waiting).
    try:
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

        agent_cfg_for_info: Any = agent_cfg
        if agent_cfg_for_info is None:
            agent_cfg_for_info = load_cfg_from_registry(task, "rl_games_cfg_entry_point")
        rl_games_cfg: dict[str, Any] = {}
        if isinstance(agent_cfg_for_info, dict):
            params = agent_cfg_for_info.get("params")
            if isinstance(params, dict):
                config = params.get("config")
                if isinstance(config, dict):
                    rl_games_cfg = config
        residual_rl_cfg = rl_games_cfg.get("residual_rl") if isinstance(rl_games_cfg, dict) else None
        if not isinstance(residual_rl_cfg, dict):
            residual_rl_cfg = {}
        try:
            critic_warmup_epochs = int(float(residual_rl_cfg.get("critic_warmup_epochs") or 0))
        except (TypeError, ValueError):
            critic_warmup_epochs = 0
        info["rl_games"] = {
            "save_frequency": int(rl_games_cfg.get("save_frequency") or 0),
            "save_best_after": int(rl_games_cfg.get("save_best_after") or 0),
            "critic_warmup_epochs": max(0, critic_warmup_epochs),
        }
        info["agent_hyperparameters"] = _extract_agent_hyperparameters(agent_cfg_for_info)
    except Exception as exc:
        raise RuntimeError("[residual_rl] Failed to collect RL-Games task metadata.") from exc
    return info
