"""Residual-RL hyperparameter search runner.

This runner intentionally keeps the old stage/global weight-tuning path out of
the active flow. It loads post-BC weights through --initial_weights_params and
recomputes constraint tolerances deterministically before training.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, cast

from kuafu.rl.core.core_utilities import extract_tolerances_from_constraints_raw
from kuafu.rl.core.runner_base import ResidualBaseRunner
from kuafu.rl.core.worker_pool import TrainConfig, TrainResult
from kuafu.rl.llm.client import OpenAIChatClient
from kuafu.rl.llm.prompts import SYSTEM_PROMPT
from kuafu.rl.llm.prompts_residual_hparams import (
    RESIDUAL_HPARAM_PROMPT,
    RESIDUAL_HPARAM_TUNING_FORMATTING_INSTRUCTIONS,
)
from kuafu.rl.logging.run_dir import append_jsonl, append_prompt_transcript, write_json
from kuafu.skills.constraints.tolerance_policy import (
    assign_automatic_tolerances,
    policy_metadata,
    summarize_atomic_action,
)
from kuafu.common.env_prompt_helpers import format_stage_satisfaction_times


class ResidualHparamRunner(ResidualBaseRunner):
    def __init__(
        self,
        *,
        residual_hparam_iters: int,
        initial_agent_hyperparameters: str | None = None,
        **kwargs: Any,
    ) -> None:
        self._initial_agent_hyperparameters_path = (initial_agent_hyperparameters or "").strip() or None
        super().__init__(**kwargs)
        self._residual_hparam_iters = int(residual_hparam_iters)
        self._last_best_log_dir: str | None = None
        self._last_archive_dir: str | None = None
        self._initial_agent_hyperparameters_artifact_path: str | None = None
        if self._initial_agent_hyperparameters_path is not None:
            self._initial_agent_hyperparameters_artifact_path = self._copy_input_json_artifact(
                source_path=self._initial_agent_hyperparameters_path,
                artifact_name="initial_agent_hyperparameters.json",
                source_label="initial_agent_hyperparameters",
            )

    def run(self) -> None:
        self._banner(
            "RESIDUAL HPARAM RUN START",
            f"task={self._task}  workers={self._num_workers}  device={self._device}",
        )
        print(f"[residual_rl] logs: {self._run_dirs.root_dir}", flush=True)
        self._pool.start()
        try:
            task_info = self._pool.get_task_info(timeout_s=300.0)
            if task_info is None:
                raise RuntimeError("[residual_rl] Failed to collect task info from worker pool.")
            self._task_info = task_info
            self._seed_task_info_from_initial_weights_params()
            assign_automatic_tolerances(self._task_info)
            self._synchronize_automatic_tolerance_summary()
            self._seed_task_info_from_initial_agent_hyperparameters()

            stage_count = int(self._task_info["stage_count"])
            self._run_initial_residual_bootstrap(stage_count=stage_count)
            fixed_weights, fixed_parameters = self._all_stage_config_from_task_info(stage_count=stage_count)

            self._banner(
                "RESIDUAL HPARAM TUNING",
                f"iters={self._residual_hparam_iters}  workers={self._num_workers}",
            )
            self._residual_hparam_finetune(
                initial_weights=fixed_weights,
                initial_parameters=fixed_parameters,
                stage_count=stage_count,
            )
            self._banner("RUN COMPLETE", f"logs={self._last_archive_dir or self._run_dirs.root_dir}")
            self._write_completion_receipt()
        finally:
            self._pool.close()

    def _write_completion_receipt(self) -> None:
        destination = os.environ.get("KUAFU_RESIDUAL_COMPLETION_PATH", "").strip()
        if not destination:
            return
        from kuafu.artifact_contracts import active_program_identity, file_sha256
        from kuafu.paths import PROJECT_ROOT, require_external_artifact_path

        candidates = [self._last_best_checkpoint_path,
                      os.path.join(self._run_dirs.checkpoints_dir, "best_global.pth")]
        if self._last_archive_dir:
            archive_record = Path(self._last_archive_dir) / "best_checkpoint.json"
            if archive_record.is_file():
                archived = json.loads(archive_record.read_text())
                candidates.append(archived.get("best_checkpoint", {}).get("archive_path"))
        checkpoint = None
        for candidate in candidates:
            if candidate:
                path = Path(candidate).expanduser()
                path = path if path.is_absolute() else PROJECT_ROOT / path
                if path.is_file():
                    checkpoint = path.resolve()
                    break
        if checkpoint is None:
            raise RuntimeError("Residual run finished without a readable trained checkpoint")
        target = require_external_artifact_path(destination, "Residual completion receipt")
        write_json(str(target), {
            "task": self._task, "checkpoint": str(checkpoint), "checkpoint_sha256": file_sha256(checkpoint),
            "input_identity": os.environ.get("KUAFU_RESIDUAL_INPUT_IDENTITY", ""),
            "run_dir": self._last_archive_dir or self._run_dirs.root_dir,
            **active_program_identity(),
        })

    def _seed_task_info_from_initial_weights_params(self) -> None:
        if not self._previous_run_best_stages:
            raise RuntimeError(
                "[residual_rl] residual-hparam mode requires --initial_weights_params. "
                "Run the BC/post-BC weight query first and pass its best.json."
            )
        stages = self._task_info.get("stages")
        if not isinstance(stages, dict):
            raise RuntimeError("[residual_rl] Task info is missing stage metadata.")
        for sid_str, stage_data in self._previous_run_best_stages.items():
            if not isinstance(stage_data, dict):
                raise ValueError(f"[residual_rl] Invalid stage entry in initial weights for stage {sid_str!r}.")
            try:
                stage_id = int(sid_str)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"[residual_rl] Invalid stage id in initial weights: {sid_str!r}") from exc
            if stage_id not in stages:
                continue
            if "weights" in stage_data:
                stages[stage_id]["weights"] = {
                    int(k): float(v) for k, v in dict(stage_data["weights"]).items()
                }
            if "parameters" in stage_data:
                stages[stage_id]["parameters"] = dict(stage_data["parameters"])
            seeded_entry = cast(dict[str, Any], self._best_summary["stages"].get(str(stage_id)) or {})
            if "weights" in stage_data:
                seeded_entry["weights"] = {
                    int(k): float(v) for k, v in dict(stage_data["weights"]).items()
                }
            if "parameters" in stage_data:
                seeded_entry["parameters"] = {
                    int(k): dict(v) for k, v in dict(stage_data["parameters"]).items()
                }
            self._best_summary["stages"][str(stage_id)] = seeded_entry
        print(
            f"[residual_rl] Seeded fixed weights from initial best.json "
            f"({len(self._previous_run_best_stages)} stages)",
            flush=True,
        )

    def _synchronize_automatic_tolerance_summary(self) -> None:
        stages = self._task_info.get("stages")
        if not isinstance(stages, dict):
            raise RuntimeError("[residual_rl] Task info is missing stage metadata.")
        self._best_summary["automatic_tolerance_policy"] = policy_metadata()
        for stage_id, stage_meta in stages.items():
            if not isinstance(stage_meta, dict):
                raise TypeError(f"[residual_rl] Stage {stage_id} metadata must be a dict.")
            entry = cast(
                dict[str, Any],
                self._best_summary["stages"].get(str(stage_id)) or {},
            )
            entry["parameters"] = {
                int(index): dict(values)
                for index, values in dict(stage_meta["parameters"]).items()
            }
            entry["tolerance_source"] = "automatic"
            entry["tolerance_decisions"] = stage_meta.get("tolerance_decisions", {})
            self._best_summary["stages"][str(stage_id)] = entry

    def _extract_initial_agent_hyperparameters_payload(self, data: dict[str, Any]) -> dict[str, Any]:
        if isinstance(data.get("global"), dict):
            global_hparams = data["global"].get("agent_hyperparameters")
            if isinstance(global_hparams, dict):
                return cast(dict[str, Any], global_hparams)
        if isinstance(data.get("best_summary_global"), dict):
            archive_hparams = data["best_summary_global"].get("agent_hyperparameters")
            if isinstance(archive_hparams, dict):
                return cast(dict[str, Any], archive_hparams)
        if isinstance(data.get("agent_hyperparameters"), dict):
            return cast(dict[str, Any], data["agent_hyperparameters"])
        if isinstance(data.get("Hyperparameters"), dict):
            return data
        return data

    def _seed_task_info_from_initial_agent_hyperparameters(self) -> None:
        if self._initial_agent_hyperparameters_path is None:
            return
        path = self._initial_agent_hyperparameters_path
        if not os.path.isfile(path):
            raise FileNotFoundError(f"[residual_rl] initial_agent_hyperparameters file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(
                f"[residual_rl] initial_agent_hyperparameters: expected a JSON object, got {type(data).__name__}"
            )
        raw = self._extract_initial_agent_hyperparameters_payload(data)
        current = self._initial_residual_agent_hyperparameters()
        hparams = self._coerce_residual_agent_hyperparameters(raw, current=current)
        self._task_info["agent_hyperparameters"] = hparams
        if isinstance(self._best_summary.get("global"), dict):
            cast(dict[str, Any], self._best_summary["global"])["agent_hyperparameters"] = hparams
        print(
            "[residual_rl] Seeded residual agent hyperparameters from "
            f"{path}: {json.dumps(hparams, sort_keys=True)}",
            flush=True,
        )

    def _run_initial_residual_bootstrap(self, *, stage_count: int) -> None:
        bootstrap_eval_only = self._initial_eval_checkpoint is not None
        bootstrap_hparams = self._initial_residual_agent_hyperparameters()
        if bootstrap_eval_only:
            dry_ckpt = self._initial_eval_checkpoint
            dry_prior = self._initial_prior_checkpoint if self._initial_prior_mode == "residual_rl" else None
        else:
            dry_ckpt = self._initial_checkpoint
            dry_prior = (
                self._initial_prior_checkpoint
                if self._initial_prior_mode == "residual_rl" or dry_ckpt is None
                else None
            )
        if bootstrap_eval_only:
            dry_source = "eval_only_from_checkpoint"
        elif dry_ckpt is not None and dry_prior is not None:
            dry_source = "from_initial_residual_checkpoint"
        elif dry_ckpt is not None:
            dry_source = "from_initial_checkpoint"
        elif dry_prior is not None:
            dry_source = "from_initial_prior"
        else:
            dry_source = "scratch"

        dry_epochs = self._current_progressive_train_epoch()
        dry_progress_metric_expr = f"torch.mean((self._highest_stage_per_env >= {int(stage_count)}).float())"
        self._banner(
            "INITIAL RESIDUAL EVAL" if bootstrap_eval_only else "INITIAL RESIDUAL RUN",
            f"epochs={dry_epochs}  checkpoint={dry_source}",
        )
        dry_cfgs = [
            self._make_residual_train_config(
                stage_count=stage_count,
                max_train_epoch_override=dry_epochs,
                progress_metric_expr=dry_progress_metric_expr,
                resume_from_checkpoint=None if bootstrap_eval_only else dry_ckpt,
                eval_only_checkpoint=dry_ckpt if bootstrap_eval_only else None,
                prior_checkpoint=(
                    dry_prior
                    if self._initial_prior_mode == "residual_rl"
                    else (None if bootstrap_eval_only else dry_prior)
                ),
                agent_hyperparameters=bootstrap_hparams,
                tag="initial_eval_checkpoint" if bootstrap_eval_only else "initial_residual_run",
            )
            for _ in range(int(self._num_workers))
        ]
        dry_results = self._pool.train(dry_cfgs)
        if not bootstrap_eval_only:
            self._iteration_count += 1
        dry_best_run_idx, dry_best_metric, dry_best_secondary, _dry_feedbacks = self._select_best_result(
            dry_results,
            stage_id=None,
        )
        dry_best = dry_results[dry_best_run_idx]
        for run_idx, r in enumerate(dry_results):
            marker = " * BEST" if run_idx == dry_best_run_idx else ""
            status = "skip" if r.skipped else ("ok" if r.success else "FAIL")
            print(
                f"[residual_rl]   Worker {r.worker_idx} (initial residual): "
                f"status={status} eval_progress_rate={r.progress_metric_eval} "
                f"eval_average_episodic_return={r.reward_metric_eval}{marker}",
                flush=True,
            )
        self._last_stage_best_stage_transition_stats = dry_best.stage_transition_stats
        self._last_stage_best_highest_stage_reached_per_env = dry_best.highest_stage_reached_per_env
        self._last_stage_best_constraint_met_rate_stats = dry_best.constraint_met_rate_stats
        self._last_stage_best_true_success_count = dry_best.true_success_count
        self._last_stage_best_false_complete_count = dry_best.false_complete_count
        self._last_stage_best_timeout_count = dry_best.timeout_count
        self._last_best_checkpoint_path = dry_best.checkpoint_path
        self._last_best_log_dir = dry_best.log_dir
        dry_true_stats = self._update_true_success_incumbent(dry_best, set_bc_baseline=True)
        self._save_prompt_context()
        print(
            f"[residual_rl] Initial residual run done: best_progress_rate={dry_best_metric}  "
            f"best_average_episodic_return={dry_best_secondary}  "
            f"true_success_rate={float(dry_true_stats['true_success_rate']):.6f}  "
            f"false_complete_rate={float(dry_true_stats['false_complete_rate']):.6f}  "
            f"best_worker={dry_best.worker_idx}  checkpoint={dry_best.checkpoint_path}",
            flush=True,
        )

    def _current_progressive_train_epoch(self) -> int:
        return int(
            min(
                int(self._init_train_epoch)
                + int(self._epoch_train_increment) * int(self._iteration_count),
                int(self._max_train_epoch),
            )
        )

    def _make_residual_train_config(
        self,
        *,
        stage_count: int,
        max_train_epoch_override: int,
        progress_metric_expr: str,
        resume_from_checkpoint: str | None,
        prior_checkpoint: str | None,
        agent_hyperparameters: dict[str, object] | None,
        eval_only_checkpoint: str | None = None,
        tag: str,
    ) -> TrainConfig:
        fixed_weights, fixed_parameters = self._all_stage_config_from_task_info(stage_count=stage_count)
        return TrainConfig(
            stage_id=0,
            all_stages_weights=fixed_weights,
            all_stages_parameters=fixed_parameters,
            agent_hyperparameters=agent_hyperparameters,
            progress_metric_expr=progress_metric_expr,
            max_train_epoch_override=int(max_train_epoch_override),
            resume_from_checkpoint=resume_from_checkpoint,
            eval_only_checkpoint=eval_only_checkpoint,
            prior_checkpoint=prior_checkpoint,
            prior_mode=self._initial_prior_mode,
            prior_obs_stats_count=self._initial_prior_obs_stats_count,
            prior_verbose=self._initial_prior_verbose,
            prior_critic_init_mode=self._initial_prior_critic_init_mode,
            prior_freeze_actor_epochs=self._initial_prior_freeze_actor_epochs,
            prior_sigma_ramp_epochs=self._initial_prior_sigma_ramp_epochs,
            prior_sigma_final_val=self._initial_prior_sigma_final_val,
            prior_actor_lr_warmup_epochs=self._initial_prior_actor_lr_warmup_epochs,
            prior_actor_lr_cap=self._initial_prior_actor_lr_cap,
            prior_freeze_input_running_mean_std=self._initial_prior_freeze_input_running_mean_std,
            prior_unfreeze_input_running_mean_std_after_freeze=self._initial_prior_unfreeze_input_running_mean_std_after_freeze,
            prior_rms_unfreeze_after_actor_lr_warmup=self._initial_prior_rms_unfreeze_after_actor_lr_warmup,
            prior_lr_schedule_freeze_during_warmup=self._initial_prior_lr_schedule_freeze_during_warmup,
            prior_fixed_lr=self._initial_prior_fixed_lr,
            prior_bc_anchor_epochs=self._initial_prior_bc_anchor_epochs,
            prior_bc_anchor_coef=self._initial_prior_bc_anchor_coef,
            prior_bc_dataset_coef=self._initial_prior_bc_dataset_coef,
            prior_bc_dataset_decay_epochs=self._initial_prior_bc_dataset_decay_epochs,
            prior_bc_dataset_batch_size=self._initial_prior_bc_dataset_batch_size,
            prior_bc_dataset_path=self._initial_prior_bc_dataset_path,
            prior_bc_action_reward_coef=self._initial_prior_bc_action_reward_coef,
            tag=tag,
        )

    def _all_stage_config_from_task_info(
        self,
        *,
        stage_count: int,
    ) -> tuple[dict[int, dict[int, float]], dict[int, dict[int, dict[str, object]]]]:
        stages_all = cast(dict[int, dict[str, Any]], self._task_info.get("stages") or {})
        all_stages_weights: dict[int, dict[int, float]] = {}
        all_stages_parameters: dict[int, dict[int, dict[str, object]]] = {}
        for sid in range(int(stage_count)):
            sm = cast(dict[str, Any], stages_all.get(int(sid)) or {})
            base_weights = sm.get("weights") or {}
            constraints_raw = sm.get("constraints_raw")
            if "parameters" in sm:
                base_parameters = cast(dict[int, dict[str, object]], sm["parameters"])
            else:
                base_parameters = (
                    extract_tolerances_from_constraints_raw(constraints_raw)
                    if constraints_raw is not None
                    else {}
                )
            all_stages_weights[int(sid)] = {
                int(k): float(v) for k, v in dict(base_weights).items()
            }
            all_stages_parameters[int(sid)] = {
                int(k): dict(v) for k, v in dict(base_parameters).items()
            }
        return all_stages_weights, all_stages_parameters

    @staticmethod
    def _clamp_float(value: object, *, name: str, lower: float, upper: float) -> float:
        try:
            val = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be numeric, got {value!r}") from exc
        if not math.isfinite(val):
            raise ValueError(f"{name} must be finite, got {value!r}")
        return min(max(float(val), float(lower)), float(upper))

    @staticmethod
    def _clamp_int(value: object, *, name: str, lower: int, upper: int) -> int:
        val = ResidualHparamRunner._clamp_float(value, name=name, lower=float(lower), upper=float(upper))
        rounded = int(round(val))
        return min(max(rounded, int(lower)), int(upper))

    @staticmethod
    def _coerce_bool(value: object, *, name: str) -> bool:
        if isinstance(value, bool):
            return bool(value)
        if isinstance(value, str):
            v = value.strip().lower()
            if v in {"1", "true", "yes", "on"}:
                return True
            if v in {"0", "false", "no", "off"}:
                return False
        raise ValueError(f"{name} must be bool, got {value!r}")

    def _coerce_residual_agent_hyperparameters(
        self,
        raw: dict[str, Any],
        *,
        current: dict[str, object] | None,
    ) -> dict[str, object]:
        if "Hyperparameters" in raw:
            raw = cast(dict[str, Any], raw["Hyperparameters"])
        if not isinstance(raw, dict):
            raise ValueError("residual hyperparameters must be a JSON object")

        allowed = {"clip_actions", "residual_bounds", "clip_composed_actions", "alpha_final", "alpha_warmup_epochs"}
        unknown = set(raw.keys()) - allowed
        if unknown:
            raise KeyError(f"Unsupported residual hyperparameter keys: {sorted(unknown)}")
        current = dict(current or {})
        current.setdefault("clip_composed_actions", False)
        missing_current = allowed - set(current.keys())
        if missing_current:
            raise ValueError(
                "Current residual agent hyperparameters are missing required keys: "
                f"{sorted(missing_current)}"
            )

        current_bounds = current.get("residual_bounds")
        if not isinstance(current_bounds, list) or not current_bounds:
            raise ValueError("Current residual_bounds must be a non-empty list.")
        raw_bounds = raw.get("residual_bounds", current_bounds)
        if not isinstance(raw_bounds, (list, tuple)):
            raise ValueError("residual_bounds must be a list")
        if len(raw_bounds) != len(current_bounds):
            raise ValueError(
                f"residual_bounds length mismatch: got {len(raw_bounds)}, expected {len(current_bounds)}"
            )

        return {
            "clip_actions": self._clamp_float(
                raw.get("clip_actions", current["clip_actions"]),
                name="clip_actions",
                lower=0.5,
                upper=2.0,
            ),
            "residual_bounds": [
                self._clamp_float(v, name=f"residual_bounds[{i}]", lower=0.02, upper=1.0)
                for i, v in enumerate(raw_bounds)
            ],
            "clip_composed_actions": self._coerce_bool(
                raw.get("clip_composed_actions", current["clip_composed_actions"]),
                name="clip_composed_actions",
            ),
            "alpha_final": self._clamp_float(
                raw.get("alpha_final", current["alpha_final"]),
                name="alpha_final",
                lower=0.05,
                upper=1.5,
            ),
            "alpha_warmup_epochs": self._clamp_int(
                raw.get("alpha_warmup_epochs", current["alpha_warmup_epochs"]),
                name="alpha_warmup_epochs",
                lower=0,
                upper=1000,
            ),
        }

    def _initial_residual_agent_hyperparameters(self) -> dict[str, object]:
        raw = self._task_info.get("agent_hyperparameters")
        if not isinstance(raw, dict):
            raise RuntimeError(
                "[residual_rl] residual-hparam mode requires agent_hyperparameters in task metadata."
            )
        return self._coerce_residual_agent_hyperparameters(raw, current=raw)

    def _parse_residual_hparam_action(
        self,
        raw: str,
        *,
        current: dict[str, object],
    ) -> dict[str, object]:
        cleaned = self._strip_llm_json(raw)
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            import ast

            try:
                parsed = ast.literal_eval(cleaned)
            except Exception as exc:
                raise ValueError(
                    "[residual_rl] Failed to parse residual hparam LLM output as JSON or "
                    f"Python literal. Raw (truncated): {raw[:300]}"
                ) from exc
        if not isinstance(parsed, dict):
            raise ValueError("LLM residual hyperparameter output must be a JSON object")
        return self._coerce_residual_agent_hyperparameters(parsed, current=current)

    @staticmethod
    def _load_scalar_events(log_dir: str, tags: tuple[str, ...]) -> dict[str, list[tuple[int, float]]]:
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        except Exception as exc:
            raise RuntimeError("Missing dependency `tensorboard` for residual event summaries.") from exc

        event_acc = EventAccumulator(log_dir, size_guidance={"scalars": 0})
        event_acc.Reload()
        available = set(event_acc.Tags().get("scalars", []))
        data: dict[str, list[tuple[int, float]]] = {}
        for tag in tags:
            if tag not in available:
                data[tag] = []
                continue
            values: list[tuple[int, float]] = []
            for ev in event_acc.Scalars(tag):
                try:
                    value = float(ev.value)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(value):
                    values.append((int(ev.step), value))
            data[tag] = values
        return data

    @staticmethod
    def _format_scalar_events_for_prompt(tag: str, events: list[tuple[int, float]]) -> str:
        if not events:
            return f"- {tag}: missing"
        values = [v for _step, v in events]
        steps = [s for s, _value in events]
        peak_idx = max(range(len(values)), key=lambda i: values[i])
        peak_value = values[peak_idx]
        peak_step = steps[peak_idx]
        last_value = values[-1]
        tail = values[-min(10, len(values)):]
        recent = ", ".join(f"{v:.6g}" for v in tail)
        return (
            f"- {tag}: n={len(values)} first={values[0]:.6g} last={last_value:.6g} "
            f"min={min(values):.6g} max={max(values):.6g} "
            f"peak_step={peak_step} peak_index={peak_idx} "
            f"post_peak_drop={peak_value - last_value:.6g} "
            f"tail_mean={sum(tail) / float(len(tail)):.6g} recent=[{recent}]"
        )

    @staticmethod
    def _format_success_rate_window_collapse_for_prompt(
        window_events: list[tuple[int, float]],
        total_events: list[tuple[int, float]],
        *,
        critic_warmup_epochs: int,
    ) -> str:
        tag = "Episode/Kuafu/success_rate_window"
        if not window_events:
            return f"- {tag}: missing"
        values = [v for _step, v in window_events]
        steps = [s for s, _value in window_events]
        totals = [v for _step, v in total_events]
        if len(totals) != len(values):
            totals = []
        warmup = max(0, int(critic_warmup_epochs))
        post_warmup_indices = [i for i, step in enumerate(steps) if int(step) > warmup]
        if not post_warmup_indices:
            return (
                f"- {tag}: n={len(values)} no post-critic-warmup points "
                f"critic_warmup_epochs={warmup}"
            )
        final_total = totals[-1] if totals else None
        if final_total and final_total > 0.0:
            min_total = max(1000.0, 0.05 * float(final_total))
            candidate_indices = [
                i for i in post_warmup_indices if totals and totals[i] >= min_total
            ]
        else:
            min_total = 0.0
            candidate_indices = list(post_warmup_indices)
        if not candidate_indices:
            candidate_indices = list(post_warmup_indices)

        peak_idx = max(candidate_indices, key=lambda i: values[i])
        peak_value = values[peak_idx]
        peak_step = steps[peak_idx]
        first_post_idx = post_warmup_indices[0]
        last_post_idx = post_warmup_indices[-1]
        first_post_value = values[first_post_idx]
        latest_value = values[last_post_idx]
        drop = peak_value - latest_value
        drop_fraction = drop / max(abs(peak_value), 1e-8)
        collapse_drop_fraction_threshold = 0.5
        collapse = drop_fraction >= collapse_drop_fraction_threshold and peak_idx < last_post_idx

        collapse_onset_idx: int | None = None
        collapse_onset_value_threshold = peak_value * (1.0 - collapse_drop_fraction_threshold)
        if collapse:
            for idx in post_warmup_indices:
                if idx <= peak_idx:
                    continue
                if values[idx] <= collapse_onset_value_threshold:
                    collapse_onset_idx = idx
                    break
            if collapse_onset_idx is None:
                collapse_onset_idx = last_post_idx

        important_indices = {first_post_idx, peak_idx, last_post_idx}
        if collapse_onset_idx is not None:
            important_indices.add(collapse_onset_idx)
        max_samples = 9
        remaining_slots = max(0, max_samples - len(important_indices))
        if remaining_slots > 0:
            if remaining_slots == 1:
                important_indices.add(post_warmup_indices[len(post_warmup_indices) // 2])
            else:
                for slot in range(remaining_slots):
                    pos = round(slot * (len(post_warmup_indices) - 1) / float(remaining_slots - 1))
                    important_indices.add(post_warmup_indices[int(pos)])
        trend_indices = sorted(important_indices, key=lambda i: (steps[i], i))
        trend = ", ".join(
            f"{idx + 1}@{steps[idx]}={values[idx]:.6g}" for idx in trend_indices
        )

        total_text = f" final_total={int(final_total)}" if final_total is not None else ""
        if collapse_onset_idx is None:
            onset_text = "collapse_onset=none"
        else:
            onset_text = (
                f"collapse_onset_value={values[collapse_onset_idx]:.6g} "
                f"collapse_onset_step={steps[collapse_onset_idx]} "
                f"collapse_onset_metric_index={collapse_onset_idx + 1} "
                f"collapse_onset_drop_from_peak={peak_value - values[collapse_onset_idx]:.6g}"
            )
        collapse_interpretation = (
            "collapse_interpretation=collapse_by_50pct_peak_to_latest_rule"
            if collapse
            else "collapse_interpretation=normal_run_possible_no_50pct_peak_to_latest_dip"
        )
        return (
            f"- {tag}: n={len(values)} post_warmup_n={len(post_warmup_indices)} "
            f"critic_warmup_epochs={warmup} "
            f"first_post_warmup={first_post_value:.6g} first_post_warmup_step={steps[first_post_idx]} "
            f"first_post_warmup_metric_index={first_post_idx + 1} "
            f"stable_peak={peak_value:.6g} stable_peak_step={peak_step} "
            f"stable_peak_metric_index={peak_idx + 1} "
            f"latest_post_warmup={latest_value:.6g} latest_post_warmup_step={steps[last_post_idx]} "
            f"latest_post_warmup_metric_index={last_post_idx + 1} "
            f"peak_to_latest_drop={drop:.6g} peak_to_latest_drop_fraction={drop_fraction:.6g} "
            f"collapse_after_peak={str(collapse).lower()} "
            f"collapse_rule=latest_post_warmup_at_least_50pct_below_previous_peak "
            f"{collapse_interpretation} {onset_text} "
            f"stable_peak_min_terminations={min_total:.0f}{total_text} "
            f"trend_samples_post_warmup(index@step=value)=[{trend}]"
        )

    def _summarize_reward_success_events(self, log_dir: str | None, *, critic_warmup_epochs: int) -> str:
        exact_tags = (
            "Episode/Kuafu/success_rate_window",
            "Episode/Kuafu/termination_total_count",
            "Episode/Kuafu/false_complete_rate",
            "Episode/Kuafu/timeout_rate",
            "Kuafu/rollout_step_reward_mean_window/iter",
            "Kuafu/rollout_epoch_step_reward_mean/iter",
        )
        if not log_dir:
            return "(No TensorBoard event data available: no log_dir recorded.)"
        try:
            data = self._load_scalar_events(log_dir, exact_tags)
        except Exception as exc:
            return f"(Failed to read TensorBoard event data from {log_dir}: {type(exc).__name__}: {exc})"

        rows = [
            self._format_success_rate_window_collapse_for_prompt(
                data.get("Episode/Kuafu/success_rate_window", []),
                data.get("Episode/Kuafu/termination_total_count", []),
                critic_warmup_epochs=int(critic_warmup_epochs),
            ),
            self._format_scalar_events_for_prompt(
                "Episode/Kuafu/false_complete_rate",
                data.get("Episode/Kuafu/false_complete_rate", []),
            ),
            self._format_scalar_events_for_prompt(
                "Episode/Kuafu/timeout_rate",
                data.get("Episode/Kuafu/timeout_rate", []),
            ),
            self._format_scalar_events_for_prompt(
                "Kuafu/rollout_step_reward_mean_window/iter",
                data.get("Kuafu/rollout_step_reward_mean_window/iter", []),
            ),
            self._format_scalar_events_for_prompt(
                "Kuafu/rollout_epoch_step_reward_mean/iter",
                data.get("Kuafu/rollout_epoch_step_reward_mean/iter", []),
            ),
        ]
        return "\n".join(rows)

    def _render_success_rate_window_plot_for_prompt(
        self,
        log_dir: str | None,
        *,
        iter_idx: int,
        critic_warmup_epochs: int,
    ) -> str | None:
        tag = "Episode/Kuafu/success_rate_window"
        if not log_dir:
            return None
        try:
            data = self._load_scalar_events(log_dir, (tag,))
            events = data.get(tag, [])
            if not events:
                return None

            import matplotlib

            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt

            steps = [int(step) for step, _value in events]
            values = [float(value) for _step, value in events]
            warmup = max(0, int(critic_warmup_epochs))
            post_indices = [i for i, step in enumerate(steps) if int(step) > warmup]
            if not post_indices:
                post_indices = list(range(len(values)))
            peak_idx = max(post_indices, key=lambda i: values[i])
            latest_idx = post_indices[-1]

            plot_dir = os.path.join(self._run_dirs.root_dir, "prompt_plots")
            os.makedirs(plot_dir, exist_ok=True)
            plot_path = os.path.join(plot_dir, f"success_rate_window_iter_{int(iter_idx):03d}.png")

            fig, ax = plt.subplots(figsize=(2.4, 2.4), dpi=100)
            ax.plot(steps, values, color="#1f77b4", linewidth=1.4)
            if steps[0] <= warmup <= steps[-1]:
                ax.axvline(warmup, color="#8c8c8c", linestyle="--", linewidth=0.8)
            ax.scatter([steps[peak_idx]], [values[peak_idx]], color="#2ca02c", s=14, zorder=3)
            ax.scatter([steps[latest_idx]], [values[latest_idx]], color="#d62728", s=14, zorder=3)
            ax.set_title("success_rate_window", fontsize=8)
            ax.set_xlabel("epoch", fontsize=7)
            ax.set_ylabel("rate", fontsize=7)
            ax.set_ylim(-0.02, 1.02)
            ax.tick_params(labelsize=6, length=2)
            ax.grid(True, alpha=0.25, linewidth=0.5)
            fig.tight_layout(pad=0.3)
            fig.savefig(plot_path, dpi=100)
            plt.close(fig)
            return plot_path
        except Exception as exc:
            print(
                f"[residual_rl] Failed to render success_rate_window plot for LLM prompt: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            return None

    def _residual_hparam_finetune(
        self,
        *,
        initial_weights: dict[int, dict[int, float]],
        initial_parameters: dict[int, dict[int, dict[str, object]]],
        stage_count: int,
    ) -> None:
        if int(self._residual_hparam_iters) <= 0:
            return

        task_description = (self._task_info.get("task_description") or "").strip() or self._task
        stages_all = cast(dict[int, dict[str, Any]], self._task_info.get("stages") or {})
        current_hparams = self._initial_residual_agent_hyperparameters()

        stage_label_lines: list[str] = []
        for sid in range(int(stage_count)):
            sm = cast(dict[str, Any], stages_all[int(sid)])
            atomic_action = sm.get("atomic_action")
            if not isinstance(atomic_action, str) or not atomic_action.strip():
                raise RuntimeError(f"[residual_rl] residual_hparam: stage {sid} is missing atomic_action.")
            stage_label_lines.append(f"{sid}: {summarize_atomic_action(atomic_action)}")

        stage_labels = "\n".join(stage_label_lines)
        detectable_objects_str = json.dumps(
            self._task_info["detectable_objects"], indent=2, sort_keys=True, default=str
        )
        action_mode = str(self._task_info.get("action_mode") or "(unknown)")
        try:
            action_dim = int(self._task_info.get("action_dim") or 0)
        except (TypeError, ValueError):
            action_dim = 0
        action_labels_raw = self._task_info.get("action_dimension_labels")
        if isinstance(action_labels_raw, list) and action_labels_raw:
            action_labels = [str(v) for v in action_labels_raw]
        else:
            bounds = current_hparams.get("residual_bounds")
            inferred_dim = len(bounds) if isinstance(bounds, list) else max(action_dim, 0)
            action_labels = [f"{i}: action dimension {i}" for i in range(int(inferred_dim))]
        bounds = current_hparams.get("residual_bounds")
        if isinstance(bounds, list) and len(action_labels) != len(bounds):
            action_labels.append(
                f"WARNING: action label count {len(action_labels)} does not match residual_bounds length {len(bounds)}."
            )
        action_dimension_labels = "\n".join(f"- {label}" for label in action_labels)
        action_mode_summary = f"{action_mode} ({action_dim}D action space)" if action_dim > 0 else action_mode

        llm_hparam = OpenAIChatClient(
            model=self._gpt_model,
            temperature=self._temperature,
            num_suggestions=self._num_workers,
            system_prompt=SYSTEM_PROMPT + RESIDUAL_HPARAM_TUNING_FORMATTING_INSTRUCTIONS,
        )

        assistant_prompt: str | None = None
        best_archive_candidate: tuple[int, TrainResult, dict[str, float | int], dict[str, object]] | None = None
        for iter_idx in range(int(self._residual_hparam_iters)):
            highest_stage_reached = self._last_stage_best_highest_stage_reached_per_env
            if highest_stage_reached is None:
                raise RuntimeError("[residual_rl] residual_hparam: missing highest_stage_reached_per_env stats")
            true_success_count = int(self._last_stage_best_true_success_count or 0)
            false_complete_count = int(self._last_stage_best_false_complete_count or 0)
            timeout_count = int(self._last_stage_best_timeout_count or 0)
            total_episodes = int(true_success_count + false_complete_count + timeout_count)
            if total_episodes <= 0:
                total_episodes = int(len(highest_stage_reached))
            denom = float(total_episodes) if total_episodes > 0 else 1.0
            true_success_rate = true_success_count / denom
            false_complete_rate = false_complete_count / denom
            timeout_rate = timeout_count / denom
            outcome_stats = "\n".join(
                [
                    f"- Eval true_success_rate: {true_success_rate:.6f} "
                    f"({true_success_count}/{total_episodes})",
                    f"- Episode/Kuafu/false_complete_rate: {false_complete_rate:.6f} "
                    f"({false_complete_count}/{total_episodes})",
                    f"- Episode/Kuafu/timeout_rate: {timeout_rate:.6f} "
                    f"({timeout_count}/{total_episodes})",
                ]
            )
            stage_satisfaction_summary = format_stage_satisfaction_times(
                self._last_stage_best_stage_transition_stats,
                highest_stage_reached_per_env=highest_stage_reached,
                stage_id=None,
                num_env=self._num_envs,
            )
            current_hparams_str = json.dumps(current_hparams, indent=2, sort_keys=True, default=str)
            rl_games_info = self._task_info.get("rl_games")
            if isinstance(rl_games_info, dict):
                critic_warmup_epochs = int(rl_games_info.get("critic_warmup_epochs") or 0)
            else:
                critic_warmup_epochs = 0
            reward_success_event_summary = self._summarize_reward_success_events(
                self._last_best_log_dir,
                critic_warmup_epochs=critic_warmup_epochs,
            )
            success_rate_window_plot_path = self._render_success_rate_window_plot_for_prompt(
                self._last_best_log_dir,
                iter_idx=int(iter_idx),
                critic_warmup_epochs=critic_warmup_epochs,
            )
            if success_rate_window_plot_path:
                success_rate_plot_summary = (
                    "Attached image: 240x240 plot of Episode/Kuafu/success_rate_window. "
                    "Inspect the visual trend and decide whether it shows true collapse, "
                    "normal run variation, under-correction, or stable improvement."
                )
                prompt_image_paths = [success_rate_window_plot_path]
            else:
                success_rate_plot_summary = "(No success_rate_window plot image could be attached.)"
                prompt_image_paths = None

            user_prompt = RESIDUAL_HPARAM_PROMPT.format(
                task_description=task_description,
                detectable_objects=detectable_objects_str,
                stage_labels=stage_labels,
                action_mode=action_mode_summary,
                action_dimension_labels=action_dimension_labels,
                current_hyperparameters=current_hparams_str,
                outcome_stats=outcome_stats,
                stage_satisfaction_summary=stage_satisfaction_summary,
                reward_success_event_summary=reward_success_event_summary,
                success_rate_plot_summary=success_rate_plot_summary,
            )

            print(
                f"\n[residual_rl] -- RESIDUAL HPARAM iter {iter_idx}/{int(self._residual_hparam_iters)-1} "
                f"({self._num_workers} workers) --",
                flush=True,
            )
            print(user_prompt, flush=True)

            llm_resp = llm_hparam.prompt(
                user_prompt=user_prompt,
                assistant_prompt=assistant_prompt,
                image_paths=prompt_image_paths,
            )
            ts = __import__("datetime").datetime.now().isoformat()
            append_jsonl(
                self._run_dirs.openai_payloads_path,
                {
                    "timestamp": ts,
                    "kind": "residual_hparam_tuning",
                    "iter": int(iter_idx),
                    "messages": llm_resp.messages,
                    "raw_outputs": llm_resp.raw_outputs,
                    "usage": llm_resp.usage,
                    "extra": {
                        "stage_count": int(stage_count),
                        "current_hyperparameters": current_hparams,
                        "success_rate_window_plot_path": success_rate_window_plot_path,
                    },
                },
            )
            append_prompt_transcript(
                path=self._run_dirs.prompt_transcript_path,
                header={"timestamp": ts, "kind": "residual_hparam_tuning", "iter": int(iter_idx)},
                messages=llm_resp.messages,
                raw_outputs=llm_resp.raw_outputs,
            )

            parsed_actions: list[dict[str, object]] = []
            for output_idx, raw in enumerate(llm_resp.raw_outputs):
                try:
                    parsed_actions.append(self._parse_residual_hparam_action(raw, current=current_hparams))
                except Exception as exc:
                    raise RuntimeError(
                        f"[residual_rl] residual_hparam proposal {output_idx} failed to parse "
                        f"({type(exc).__name__}: {exc}). Aborting instead of reusing "
                        f"current hyperparameters. Raw (truncated): {raw[:300]}"
                    ) from exc

            resume_ckpt, prior_ckpt, warm_start_label = self._get_iteration_warm_start()
            progress_metric_expr = f"torch.mean((self._highest_stage_per_env >= {int(stage_count)}).float())"
            current_epochs = self._current_progressive_train_epoch()
            cfgs: list[TrainConfig] = []
            trial_keys: list[str] = []
            for hparams in parsed_actions:
                trial_keys.append(
                    self._make_trial_key(
                        kind="residual_hparam",
                        stage_id=None,
                        weights=None,
                        parameters=None,
                        all_stages_weights=initial_weights,
                        all_stages_parameters=initial_parameters,
                    )
                    + ":"
                    + json.dumps(hparams, sort_keys=True, default=str)
                )
                cfgs.append(
                    self._make_residual_train_config(
                        stage_count=stage_count,
                        max_train_epoch_override=current_epochs,
                        progress_metric_expr=progress_metric_expr,
                        resume_from_checkpoint=resume_ckpt,
                        prior_checkpoint=prior_ckpt,
                        agent_hyperparameters=hparams,
                        tag=f"residual_hparam_iter{iter_idx}",
                    )
                )

            if resume_ckpt or prior_ckpt:
                print(
                    f"[residual_rl] Warm-starting residual hparam iter {iter_idx} via {warm_start_label}: "
                    f"{resume_ckpt or prior_ckpt}",
                    flush=True,
                )
            results = self._pool.train(cfgs)
            self._iteration_count += 1

            best_run_idx, iter_best, iter_best_secondary, run_feedbacks = self._select_best_result(
                results,
                stage_id=None,
            )
            r_best = results[best_run_idx]
            fb_best = run_feedbacks[best_run_idx]
            candidate_hparams = dict(cfgs[best_run_idx].agent_hyperparameters or current_hparams)
            if r_best.applied_agent_hyperparameters is not None:
                candidate_hparams = dict(r_best.applied_agent_hyperparameters)
            candidate_true_stats = self._result_true_success_stats(r_best)
            candidate_success_rate_window_raw = r_best.selection_success_rate_window
            if candidate_success_rate_window_raw is None:
                raise RuntimeError(
                    "[residual_rl] residual_hparam requires selection_success_rate_window for "
                    "selection logging; refusing to fall back to true_success_rate."
                )
            candidate_success_rate_window = float(candidate_success_rate_window_raw)
            accepted_this_iter = True
            acceptance_reason = f"accepted:selection_success_rate_window={candidate_success_rate_window:.6f}"

            for run_idx, r in enumerate(results):
                marker = " * BEST" if run_idx == best_run_idx else ""
                status = "skip" if r.skipped else ("ok" if r.success else "FAIL")
                print(
                    f"[residual_rl]   Worker {r.worker_idx} (residual hparam iter {iter_idx}): "
                    f"status={status} eval_progress_rate={r.progress_metric_eval} "
                    f"eval_average_episodic_return={r.reward_metric_eval}{marker}",
                    flush=True,
                )
            print(
                f"[residual_rl] Residual hparam iter {iter_idx}: eval_best_progress_rate={float(iter_best)} "
                f"eval_best_average_episodic_return={iter_best_secondary} "
                f"best_worker={r_best.worker_idx}",
                flush=True,
            )

            if accepted_this_iter:
                current_hparams = candidate_hparams
                self._last_stage_best_stage_transition_stats = r_best.stage_transition_stats
                self._last_stage_best_highest_stage_reached_per_env = r_best.highest_stage_reached_per_env
                self._last_stage_best_constraint_met_rate_stats = r_best.constraint_met_rate_stats
                self._last_stage_best_true_success_count = r_best.true_success_count
                self._last_stage_best_false_complete_count = r_best.false_complete_count
                self._last_stage_best_timeout_count = r_best.timeout_count
                self._last_best_checkpoint_path = r_best.checkpoint_path
                self._last_best_log_dir = r_best.log_dir
                candidate_true_stats = self._update_true_success_incumbent(r_best)
                self._save_prompt_context()
                self._best_global_progress_rate = float(iter_best)
                self._maybe_store_best_checkpoint(
                    kind="residual_hparam",
                    stage_id=None,
                    iter_idx=int(iter_idx),
                    checkpoint_path=r_best.checkpoint_path,
                    progress_rate=float(iter_best),
                    weights=None,
                    parameters=None,
                    all_stages_weights=initial_weights,
                    all_stages_parameters=initial_parameters,
                    true_success_stats=candidate_true_stats,
                )
                if isinstance(self._best_summary.get("global"), dict):
                    cast(dict[str, Any], self._best_summary["global"])["agent_hyperparameters"] = current_hparams
                    write_json(self._run_dirs.best_path, self._best_summary)
                best_archive_candidate = (
                    int(iter_idx),
                    r_best,
                    dict(candidate_true_stats),
                    dict(current_hparams),
                )
            else:
                print(
                    f"[residual_rl] Residual hparam iter {iter_idx}: candidate rejected by window guard: "
                    f"{acceptance_reason}",
                    flush=True,
                )

            self._record_trial(
                trial_key=trial_keys[best_run_idx],
                kind="residual_hparam",
                stage_id=int(stage_count),
                iter_idx=int(iter_idx),
                weights=initial_weights,
                parameters=initial_parameters,
                all_stages_weights=initial_weights,
                all_stages_parameters=initial_parameters,
                progress_rate=float(iter_best),
                log_dir=r_best.log_dir or "",
                checkpoint_path=r_best.checkpoint_path or "",
                raw_llm_output=llm_resp.raw_outputs[best_run_idx],
                accepted=accepted_this_iter,
                rejection_reason=None if accepted_this_iter else acceptance_reason,
                true_success_stats=candidate_true_stats,
            )

            candidate_true_success_count = int(candidate_true_stats["true_success_count"])
            candidate_false_complete_count = int(candidate_true_stats["false_complete_count"])
            candidate_timeout_count = int(candidate_true_stats["timeout_count"])
            true_success_rate = float(candidate_true_stats["true_success_rate"])
            false_complete_rate = float(candidate_true_stats["false_complete_rate"])
            self._tb_writer.add_text("ResidualHparam/raw_llm_output", llm_resp.raw_outputs[best_run_idx], iter_idx)
            self._tb_writer.add_text("ResidualHparam/run_feedback", fb_best, iter_idx)
            self._tb_writer.add_text(
                "ResidualHparam/agent_hyperparameters",
                json.dumps(candidate_hparams, indent=2, sort_keys=True, default=str),
                iter_idx,
            )
            self._tb_writer.add_scalar("ResidualHparam/progress_rate", float(iter_best), iter_idx)
            self._tb_writer.add_scalar("ResidualHparam/true_success_rate", true_success_rate, iter_idx)
            self._tb_writer.add_scalar("ResidualHparam/false_complete_rate", false_complete_rate, iter_idx)
            if candidate_success_rate_window_raw is not None:
                self._tb_writer.add_scalar(
                    "ResidualHparam/selection_success_rate_window",
                    float(candidate_success_rate_window_raw),
                    iter_idx,
                )
            self._tb_writer.add_scalar("ResidualHparam/true_success_count", candidate_true_success_count, iter_idx)
            self._tb_writer.add_scalar("ResidualHparam/false_complete_count", candidate_false_complete_count, iter_idx)
            self._tb_writer.add_scalar("ResidualHparam/timeout_count", candidate_timeout_count, iter_idx)
            self._tb_writer.add_scalar("ResidualHparam/accepted", 1.0 if accepted_this_iter else 0.0, iter_idx)

            append_jsonl(
                self._run_dirs.iterations_path,
                {
                    "timestamp": __import__("datetime").datetime.now().isoformat(),
                    "kind": "residual_hparam_iter",
                    "iter": int(iter_idx),
                    "best_progress_rate": float(iter_best),
                    "best_run_idx": int(best_run_idx),
                    "best_worker_idx": int(results[best_run_idx].worker_idx),
                    "num_workers": int(len(results)),
                    "best_agent_hyperparameters": current_hparams if accepted_this_iter else candidate_hparams,
                    "best_checkpoint": self._last_best_checkpoint_path,
                    "accepted": bool(accepted_this_iter),
                    "rejection_reason": None if accepted_this_iter else acceptance_reason,
                    "true_success_count": candidate_true_success_count,
                    "false_complete_count": candidate_false_complete_count,
                    "timeout_count": candidate_timeout_count,
                    "true_success_rate": true_success_rate,
                    "false_complete_rate": false_complete_rate,
                    "success_rate_window": candidate_success_rate_window,
                    "selection_score": float(candidate_true_stats["selection_score"]),
                    "checkpoint_selection": {
                        "success_rate_window": r_best.selection_success_rate_window,
                        "success_rate_ema_compat": r_best.selection_success_rate_ema,
                        "false_complete_rate": r_best.selection_false_complete_rate,
                        "timeout_rate": r_best.selection_timeout_rate,
                        "metric_tag": r_best.selection_metric_tag,
                        "metric_index": r_best.selection_metric_index,
                        "metric_step": r_best.selection_metric_step,
                        "critic_warmup_epochs": r_best.selection_critic_warmup_epochs,
                        "checkpoint_epoch": r_best.selection_checkpoint_epoch,
                        "checkpoint_policy": r_best.selection_checkpoint_policy,
                        "source": r_best.selection_source,
                    },
                    "best_true_success_rate_so_far": self._best_true_success_rate_so_far,
                    "bc_baseline_true_success_rate": self._bc_baseline_true_success_rate,
                    "per_worker_progress_rate": [r.progress_metric_eval for r in results],
                    "per_worker_agent_hyperparameters": [
                        r.applied_agent_hyperparameters if r.success else None for r in results
                    ],
                    "per_worker_kuafu_training_parameters_path": [
                        r.kuafu_training_parameters_path if r.success else None for r in results
                    ],
                    "per_worker_true_success_rate": [
                        float(self._result_true_success_stats(r)["true_success_rate"]) if r.success else None
                        for r in results
                    ],
                    "per_worker_selection_success_rate_window": [
                        r.selection_success_rate_window if r.success else None for r in results
                    ],
                    "per_worker_selection_metric_tag": [
                        r.selection_metric_tag if r.success else None for r in results
                    ],
                    "per_worker_selection_metric_index": [
                        r.selection_metric_index if r.success else None for r in results
                    ],
                    "per_worker_selection_checkpoint_epoch": [
                        r.selection_checkpoint_epoch if r.success else None for r in results
                    ],
                    "per_worker_false_complete_rate": [
                        float(self._result_true_success_stats(r)["false_complete_rate"]) if r.success else None
                        for r in results
                    ],
                },
            )

            assistant_prompt = llm_resp.raw_outputs[best_run_idx]

        if best_archive_candidate is not None:
            archive_iter_idx, archive_result, archive_true_stats, archive_hparams = best_archive_candidate
            archive_dir = self._archive_passed_residual_run(
                iter_idx=int(archive_iter_idx),
                r_best=archive_result,
                true_success_stats=archive_true_stats,
                agent_hyperparameters=archive_hparams,
            )
            self._last_archive_dir = archive_dir
            self._banner("RESIDUAL HPARAM ARCHIVED", f"archive={archive_dir}")
        return
