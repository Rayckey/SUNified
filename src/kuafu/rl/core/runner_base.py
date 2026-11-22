"""Shared Kuafu runner infrastructure used by active runner modes.

Old stage/global weight tuning lives in kuafu.rl.core.deprecated.runner_stage_global.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import hashlib
import warnings
from typing import Any, cast

from kuafu.rl.core.worker_pool import TrainConfig, TrainResult, WorkerPool
from kuafu.rl.logging.run_dir import (
    append_jsonl,
    append_prompt_transcript,
    load_tensorboard_scalars,
    make_run_dirs,
    make_tb_writer,
    write_json,
)
from kuafu.rl.isaac.task_registry import (
    resolve_artifact_robot_id,
    resolve_execution_robot_id,
    task_names_match_for_metadata,
)

from kuafu.rl.core.core_utilities import extract_tolerances_from_constraints_raw
from kuafu.common.env_prompt_helpers import (
    format_end_to_end_rate_stats,
    format_constraint_satisfaction_rates,
    format_stage_satisfaction_times,
    format_reward_values,
)

class ResidualBaseRunner:
    """Share rollout, feedback, and artifact handling across residual learners."""

    _TRUE_SUCCESS_ACCEPTANCE_TOLERANCE = 0.01

    def __init__(
        self,
        *,
        task: str,
        device: str,
        num_workers: int,
        num_envs: int | None,
        env_seed: int,
        max_train_epoch: int,
        init_train_epoch: int,
        epoch_train_increment: int,
        feedback_subsampling: int,
        gpt_model: str,
        temperature: float,
        headless: bool,
        video_feedback: bool,
        video_length: int,
        video_num_envs_to_record: int,
        gemini_model: str = "gemini-2.0-flash",
        resume_run_dir: str | None = None,
        initial_eval_checkpoint: str | None = None,
        initial_checkpoint: str | None = None,
        initial_weights_params: str | None = None,
        restart_each_iter_from_initial_warm_start: bool = False,
        initial_prior_checkpoint: str | None = None,
        initial_prior_mode: str = "full_actor",
        initial_prior_obs_stats_count: float = 1e6,
        initial_prior_verbose: bool = False,
        initial_prior_critic_init_mode: str = "actor_only_random_critic",
        initial_prior_freeze_actor_epochs: int = 0,
        initial_prior_sigma_ramp_epochs: int = 0,
        initial_prior_sigma_final_val: float | None = None,
        initial_prior_actor_lr_warmup_epochs: int = 0,
        initial_prior_actor_lr_cap: float = 0.0,
        initial_prior_freeze_input_running_mean_std: bool = False,
        initial_prior_unfreeze_input_running_mean_std_after_freeze: bool = False,
        initial_prior_rms_unfreeze_after_actor_lr_warmup: bool = False,
        initial_prior_lr_schedule_freeze_during_warmup: bool = False,
        initial_prior_fixed_lr: float | None = None,
        initial_prior_bc_anchor_epochs: int = 0,
        initial_prior_bc_anchor_coef: float = 0.0,
        initial_prior_bc_dataset_coef: float = 0.0,
        initial_prior_bc_dataset_decay_epochs: int = 0,
        initial_prior_bc_dataset_batch_size: int = 1024,
        initial_prior_bc_dataset_path: str | None = None,
        initial_prior_bc_action_reward_coef: float = 0.0,
        stop_on_error: bool = False,
        argv: list[str] | None = None,
        launch_script_path: str | None = None,
        run_timestamp: str | None = None,
        execution_robot_id: str | None = None,
        artifact_robot_id: str | None = None,
        # tune_parameters: bool,
    ) -> None:
        self._task = task
        self._execution_robot_id = resolve_execution_robot_id(task, execution_robot_id)
        self._artifact_robot_id = resolve_artifact_robot_id(artifact_robot_id)
        self._device = device
        self._num_workers = int(num_workers)
        self._num_envs = num_envs
        self._env_seed = int(env_seed)
        self._max_train_epoch = int(max_train_epoch)
        self._init_train_epoch = int(init_train_epoch)
        self._epoch_train_increment = int(epoch_train_increment)
        self._feedback_subsampling = int(feedback_subsampling)
        self._gpt_model = gpt_model
        self._temperature = float(temperature)
        self._headless = bool(headless)
        self._initial_eval_checkpoint: str | None = (initial_eval_checkpoint or "").strip() or None
        self._initial_checkpoint: str | None = (initial_checkpoint or "").strip() or None
        self._restart_each_iter_from_initial_warm_start = bool(
            restart_each_iter_from_initial_warm_start
        )
        self._previous_run_dir: str | None = None
        self._resume_run_dir: str | None = (resume_run_dir or "").strip() or None
        self._initial_prior_checkpoint: str | None = (initial_prior_checkpoint or "").strip() or None
        self._initial_prior_mode = str(initial_prior_mode).strip() or "full_actor"
        self._initial_prior_obs_stats_count = float(initial_prior_obs_stats_count)
        self._initial_prior_verbose = bool(initial_prior_verbose)
        self._initial_prior_critic_init_mode = str(initial_prior_critic_init_mode).strip()
        self._initial_prior_freeze_actor_epochs = int(initial_prior_freeze_actor_epochs)
        self._initial_prior_sigma_ramp_epochs = int(
            initial_prior_sigma_ramp_epochs
        )
        if initial_prior_sigma_final_val is None:
            self._initial_prior_sigma_final_val: float | None = None
        else:
            try:
                _sigma_final_val = float(initial_prior_sigma_final_val)
                self._initial_prior_sigma_final_val = (
                    _sigma_final_val if math.isfinite(_sigma_final_val) else None
                )
            except (TypeError, ValueError):
                self._initial_prior_sigma_final_val = None
        self._initial_prior_actor_lr_warmup_epochs = int(initial_prior_actor_lr_warmup_epochs)
        self._initial_prior_actor_lr_cap = float(initial_prior_actor_lr_cap or 0.0)
        self._initial_prior_freeze_input_running_mean_std = bool(
            initial_prior_freeze_input_running_mean_std
        )
        self._initial_prior_unfreeze_input_running_mean_std_after_freeze = bool(
            initial_prior_unfreeze_input_running_mean_std_after_freeze
        )
        self._initial_prior_rms_unfreeze_after_actor_lr_warmup = bool(
            initial_prior_rms_unfreeze_after_actor_lr_warmup
        )
        self._initial_prior_lr_schedule_freeze_during_warmup = bool(
            initial_prior_lr_schedule_freeze_during_warmup
        )
        if initial_prior_fixed_lr is None:
            self._initial_prior_fixed_lr: float | None = None
        else:
            try:
                _fixed_lr_val = float(initial_prior_fixed_lr)
                self._initial_prior_fixed_lr = (
                    _fixed_lr_val
                    if (math.isfinite(_fixed_lr_val) and _fixed_lr_val > 0.0)
                    else None
                )
            except (TypeError, ValueError):
                self._initial_prior_fixed_lr = None
        self._initial_prior_bc_anchor_epochs = int(initial_prior_bc_anchor_epochs)
        self._initial_prior_bc_anchor_coef = float(initial_prior_bc_anchor_coef or 0.0)
        self._initial_prior_bc_dataset_coef = float(initial_prior_bc_dataset_coef or 0.0)
        self._initial_prior_bc_dataset_decay_epochs = int(initial_prior_bc_dataset_decay_epochs or 0)
        self._initial_prior_bc_dataset_batch_size = int(initial_prior_bc_dataset_batch_size or 1024)
        self._initial_prior_bc_dataset_path: str | None = (
            str(initial_prior_bc_dataset_path) if initial_prior_bc_dataset_path else None
        )
        self._initial_prior_bc_action_reward_coef = float(initial_prior_bc_action_reward_coef or 0.0)
        self._stop_on_error = bool(stop_on_error)
        allowed_prior_modes = {
            "actor_only_random_critic",
            "actor_and_critic_prior",
            "actor_only_zero_critic",
        }
        allowed_initial_prior_modes = {"full_actor", "residual_rl"}
        if self._initial_prior_mode not in allowed_initial_prior_modes:
            raise ValueError(
                f"Unsupported initial_prior_mode={self._initial_prior_mode!r}. "
                f"Expected one of {sorted(allowed_initial_prior_modes)}."
            )
        if self._initial_prior_critic_init_mode not in allowed_prior_modes:
            raise ValueError(
                f"Unsupported initial_prior_critic_init_mode={self._initial_prior_critic_init_mode!r}. "
                f"Expected one of {sorted(allowed_prior_modes)}."
            )
        if (
            self._initial_checkpoint is not None
            and self._initial_prior_checkpoint is not None
            and self._initial_prior_mode != "residual_rl"
        ):
            raise ValueError(
                "initial_checkpoint and initial_prior_checkpoint are mutually exclusive "
                "warm-start sources except when initial_prior_mode='residual_rl'."
            )
        if (
            self._restart_each_iter_from_initial_warm_start
            and self._initial_checkpoint is None
            and self._initial_prior_checkpoint is None
        ):
            raise ValueError(
                "restart_each_iter_from_initial_warm_start requires initial_checkpoint "
                "or initial_prior_checkpoint."
            )

        self._video_feedback = bool(video_feedback)
        self._video_length = int(video_length)
        self._video_num_envs_to_record = int(video_num_envs_to_record)
        self._gemini_model = str(gemini_model)

        self._run_dirs = make_run_dirs(
            task=task,
            argv=argv,
            launch_script_path=launch_script_path,
            run_timestamp=run_timestamp,
            execution_robot_id=self._execution_robot_id,
            artifact_robot_id=self._artifact_robot_id,
        )
        self._tb_writer = make_tb_writer(self._run_dirs.tb_dir)

        self._pool = WorkerPool(
            task=task,
            device=device,
            env_seed=env_seed,
            num_workers=num_workers,
            num_envs=num_envs,
            headless=headless,
            max_train_epoch=max_train_epoch,
            kuafu_run_dir=self._run_dirs.root_dir,
            stop_on_error=self._stop_on_error,
            initial_prior_checkpoint=self._initial_prior_checkpoint,
            execution_robot_id=self._execution_robot_id,
        )

        # Filled after pool starts
        self._task_info: dict[str, Any] = {}
        # Cache results to avoid re-training identical proposals.
        self._trial_cache: dict[str, TrainResult] = {}
        # Best-so-far across whole run (stage-wise + global)
        self._best_summary: dict[str, Any] = {"stages": {}, "global": None}
        self._best_stage_progress_rate: dict[int, float] = {}
        self._best_global_progress_rate: float | None = None
        self._bc_baseline_true_success_rate: float | None = None
        self._best_true_success_rate_so_far: float | None = None
        self._incumbent_true_success_rate: float | None = None
        self._incumbent_false_complete_rate: float | None = None
        # Stats from previous stage's best run (from worker_pool results_queue) for initial prompt.
        self._last_stage_best_stage_transition_stats: dict | None = None
        self._last_stage_best_highest_stage_reached_per_env: list[int] | None = None
        self._last_stage_best_constraint_met_rate_stats: dict | None = None
        self._last_stage_best_true_success_count: int | None = None
        self._last_stage_best_false_complete_count: int | None = None
        self._last_stage_best_timeout_count: int | None = None
        # Checkpoint for the “previous learned policy” used to seed prompt stats.
        self._last_best_checkpoint_path: str | None = None
        # Best weights/parameters per stage from previous run (so next run continues from there).
        self._previous_run_best_stages: dict[str, dict[str, Any]] = {}
        self._resume_all_stages_weights: dict[int, dict[int, float]] | None = None
        self._resume_all_stages_parameters: dict[int, dict[int, dict[str, object]]] | None = None

        # Per-stage winner from the most recent tuning *iteration* in this run.
        # IMPORTANT: We do NOT write these back into self._task_info, so each new iteration starts
        # from the task's baseline weights/parameters while still letting later stages re-apply
        # the finished stages' winners.
        self._iter_best_stage_weights: dict[int, dict[int, float]] = {}
        self._iter_best_stage_parameters: dict[int, dict[int, dict[str, object]]] = {}

        # Explicit resume: load prompt context + best stage weights/parameters only when requested.
        if self._resume_run_dir is not None:
            self._load_resume_run(self._resume_run_dir)

        # Load initial weights/parameters used alongside the chosen bootstrap source.
        self._initial_weights_params_path: str | None = (initial_weights_params or "").strip() or None
        self._initial_weights_params_artifact_path: str | None = None
        if self._initial_weights_params_path is not None:
            self._load_initial_weights_params(self._initial_weights_params_path)
            self._initial_weights_params_artifact_path = self._copy_input_json_artifact(
                source_path=self._initial_weights_params_path,
                artifact_name="initial_weights_params.json",
                source_label="initial_weights_params",
            )

        # Progressive epochs: start init_train_epoch, +epoch_train_increment after every train query.
        self._iteration_count = 0

    def _prompt_context_path(self) -> str:
        return os.path.join(self._run_dirs.root_dir, "prompt_context.json")

    def _save_prompt_context(self) -> None:
        """Persist the latest prompt seed stats so restarts can reuse them."""
        payload = {
            "checkpoint_path": self._last_best_checkpoint_path,
            "stage_transition_stats": self._last_stage_best_stage_transition_stats,
            "highest_stage_reached_per_env": self._last_stage_best_highest_stage_reached_per_env,
            "constraint_met_rate_stats": self._last_stage_best_constraint_met_rate_stats,
            "true_success_count": self._last_stage_best_true_success_count,
            "false_complete_count": self._last_stage_best_false_complete_count,
            "timeout_count": self._last_stage_best_timeout_count,
            "bc_baseline_true_success_rate": self._bc_baseline_true_success_rate,
            "best_true_success_rate_so_far": self._best_true_success_rate_so_far,
            "incumbent_true_success_rate": self._incumbent_true_success_rate,
            "incumbent_false_complete_rate": self._incumbent_false_complete_rate,
        }
        try:
            write_json(self._prompt_context_path(), payload)
        except Exception as exc:
            raise RuntimeError(
                f"[residual_rl] Failed to write prompt context: {self._prompt_context_path()}"
            ) from exc

    def _copy_input_json_artifact(self, *, source_path: str, artifact_name: str, source_label: str) -> str:
        """Copy a JSON input artifact into this run so the run is self-contained."""
        if not os.path.isfile(source_path):
            raise FileNotFoundError(f"[residual_rl] input artifact file not found: {source_path}")
        inputs_dir = os.path.join(self._run_dirs.root_dir, "inputs")
        os.makedirs(inputs_dir, exist_ok=True)
        src_abs = os.path.abspath(source_path)
        dst = os.path.join(inputs_dir, artifact_name)
        try:
            if os.path.abspath(dst) != src_abs:
                shutil.copy2(src_abs, dst)
            meta_path = os.path.join(inputs_dir, f"{os.path.splitext(artifact_name)[0]}_source.json")
            write_json(
                meta_path,
                {
                    "source_label": source_label,
                    "source_path": src_abs,
                    "copied_path": dst,
                    "run_root": self._run_dirs.root_dir,
                },
            )
        except Exception as exc:
            raise RuntimeError(
                f"[residual_rl] Failed to copy {source_label} JSON artifact from {src_abs} to {dst}"
            ) from exc
        print(f"[residual_rl] Copied {source_label} JSON artifact to {dst}", flush=True)
        return dst

    def _load_resume_run(self, resume_run_dir: str) -> None:
        """Explicitly load prompt context + best stage weights/parameters from a run dir.

        This is only triggered when `resume_run_dir` (and optionally `resume_checkpoint`)
        is provided; there is no auto-discovery of "latest previous run".
        """
        if not os.path.isdir(resume_run_dir):
            raise FileNotFoundError(f"[residual_rl] resume_run_dir not a directory: {resume_run_dir}")

        self._previous_run_dir = resume_run_dir

        # Load prompt context if present (non-fatal if missing).
        prompt_path = os.path.join(resume_run_dir, "prompt_context.json")
        if os.path.exists(prompt_path):
            with open(prompt_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self._last_best_checkpoint_path = data.get("checkpoint_path")
                self._last_stage_best_stage_transition_stats = data.get("stage_transition_stats")
                self._last_stage_best_highest_stage_reached_per_env = data.get("highest_stage_reached_per_env")
                self._last_stage_best_constraint_met_rate_stats = data.get("constraint_met_rate_stats")
                self._last_stage_best_true_success_count = data.get("true_success_count")
                self._last_stage_best_false_complete_count = data.get("false_complete_count")
                self._last_stage_best_timeout_count = data.get("timeout_count")
                self._bc_baseline_true_success_rate = data.get("bc_baseline_true_success_rate")
                self._best_true_success_rate_so_far = data.get("best_true_success_rate_so_far")
                self._incumbent_true_success_rate = data.get("incumbent_true_success_rate")
                self._incumbent_false_complete_rate = data.get("incumbent_false_complete_rate")

        # Load best.json (required for seeding weights/parameters).
        best_path = os.path.join(resume_run_dir, "best.json")
        if not os.path.exists(best_path):
            raise FileNotFoundError(f"[residual_rl] resume run dir has no best.json: {best_path}")
        self._copy_input_json_artifact(
            source_path=best_path,
            artifact_name="resume_best.json",
            source_label="resume_best",
        )
        with open(best_path, "r", encoding="utf-8") as bf:
            best_data = json.load(bf)
        if not isinstance(best_data, dict):
            raise ValueError(f"[residual_rl] Invalid best.json format: {best_path}")

        # Prefer global all-stage weights/parameters if available; otherwise fall back to per-stage entries.
        stages_seed: dict[str, dict[str, Any]] = {}
        global_entry = best_data.get("global")
        if (
            isinstance(global_entry, dict)
            and global_entry.get("all_stages_weights") is not None
            and global_entry.get("all_stages_parameters") is not None
        ):
            w_raw = cast(dict[str, dict[str, float]], global_entry["all_stages_weights"])
            p_raw = cast(dict[str, dict[str, dict[str, object]]], global_entry["all_stages_parameters"])
            for sid_str, w_stage in dict(w_raw).items():
                sid = str(int(sid_str))
                stages_seed[sid] = {
                    "weights": {int(k): float(v) for k, v in dict(w_stage).items()},
                    "parameters": {
                        int(k): dict(v)
                        for k, v in dict(cast(dict[str, dict[str, object]], p_raw.get(sid_str, {}))).items()
                    },
                }
        else:
            stages_seed = cast(dict[str, dict[str, Any]], best_data.get("stages") or {})

        self._previous_run_best_stages = dict(stages_seed)
        # Also store canonical nested dicts for sending to workers.
        self._resume_all_stages_weights = {}
        self._resume_all_stages_parameters = {}
        for sid_str, stage_data in self._previous_run_best_stages.items():
            if not isinstance(stage_data, dict):
                raise ValueError(f"[residual_rl] Invalid stage entry in resume best.json for stage {sid_str!r}.")
            try:
                sid = int(sid_str)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"[residual_rl] Invalid stage id in resume best.json: {sid_str!r}") from exc
            w_raw2 = stage_data.get("weights") or {}
            p_raw2 = stage_data.get("parameters") or {}
            if isinstance(w_raw2, dict):
                self._resume_all_stages_weights[sid] = {int(k): float(v) for k, v in dict(w_raw2).items()}
            if isinstance(p_raw2, dict):
                # parameters expected: constraint_idx -> {arg_name: value}
                self._resume_all_stages_parameters[sid] = {int(k): dict(v) for k, v in dict(p_raw2).items()}
        print(
            f"[residual_rl] Loaded resume run from {resume_run_dir}: {len(self._previous_run_best_stages)} stages, "
            f"checkpoint={self._last_best_checkpoint_path}",
            flush=True,
        )

    def _validate_initial_weights_params_metadata(self, path: str, data: dict[str, Any]) -> None:
        """Reject full best.json files generated for a different task."""
        from kuafu.artifact_contracts import file_sha256, validate_program_metadata

        validate_program_metadata(data)
        if data.get("prior_checkpoint") and self._initial_prior_checkpoint:
            from kuafu.bc.training.rl_games_prior import _resolve_prior_checkpoint_path

            prior_checkpoint, _ = _resolve_prior_checkpoint_path(self._initial_prior_checkpoint)
            if data.get("prior_checkpoint_sha256") != file_sha256(prior_checkpoint):
                raise ValueError("Residual weights were acquired for a different or unidentified BC checkpoint; rerun weight acquisition.")
        metadata_task = data.get("task")
        if isinstance(metadata_task, str) and metadata_task.strip():
            expected_task = str(self._task).strip()
            if not task_names_match_for_metadata(metadata_task, expected_task):
                raise ValueError(
                    "[residual_rl] initial_weights_params task mismatch: "
                    f"{path} was generated for task {metadata_task!r}, "
                    f"but this run is using {expected_task!r}. "
                    "Generate a fresh best.json for the active task."
                )

    def _load_initial_weights_params(self, path: str) -> None:
        """Load stage weights and compatibility parameter maps from JSON.

        Active runners recompute the parameter maps automatically after loading;
        neither the loaded nor recomputed values are included in an LLM prompt.

        Accepts either the full best.json format::

            {
                "global": ...,
                "stages": {
                    "0": {"weights": {...}, "parameters": {...}, "checkpoint": "...", ...},
                    ...
                }
            }

        or the plain stages-dict format::

            {
                "0": {"weights": {"0": 1.0, "1": 0.5}, "parameters": {"0": {"tolerance_m": 0.05}}},
                "1": {"weights": {"0": 1.0}, "parameters": {}}
            }
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"[residual_rl] initial_weights_params file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError(f"[residual_rl] initial_weights_params: expected a JSON object, got {type(data).__name__}")

        # Detect full best.json wrapper format and unwrap to stages dict.
        if "stages" in data and isinstance(data.get("stages"), dict):
            self._validate_initial_weights_params_metadata(path, data)
            stages_data = data["stages"]
        else:
            stages_data = data

        # Validate that the stages dict contains at least one parseable stage entry.
        valid_stage_ids: list[int] = []
        for k, v in stages_data.items():
            if not isinstance(v, dict):
                raise ValueError(f"[residual_rl] initial_weights_params: stage {k!r} is not a dict in {path}.")
            try:
                valid_stage_ids.append(int(k))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"[residual_rl] initial_weights_params: invalid stage key {k!r} in {path}.") from exc
        if not valid_stage_ids:
            raise ValueError(
                f"[residual_rl] initial_weights_params: no valid stage entries found in {path}. "
                f"Expected integer stage keys (e.g. '0', '1') mapping to dicts with 'weights' and 'parameters'. "
                f"Top-level keys found: {list(stages_data.keys())}"
            )

        # Parse into the same structures used by _load_resume_run.
        self._previous_run_best_stages = {}
        self._resume_all_stages_weights = {}
        self._resume_all_stages_parameters = {}


        for sid in valid_stage_ids:
            stage_data = stages_data[str(sid)]
            self._previous_run_best_stages[str(sid)] = dict(stage_data)
            w_raw = stage_data.get("weights") or {}
            p_raw = stage_data.get("parameters") or {}
            if not isinstance(w_raw, dict) or not w_raw:
                raise ValueError(
                    f"[residual_rl] initial_weights_params: stage {sid} has no valid 'weights' dict in {path}."
                )
            if not isinstance(p_raw, dict):
                raise ValueError(
                    f"[residual_rl] initial_weights_params: stage {sid} 'parameters' is not a dict in {path}."
                )
            self._resume_all_stages_weights[sid] = {int(k): float(v) for k, v in dict(w_raw).items()}
            self._resume_all_stages_parameters[sid] = {int(k): dict(v) for k, v in dict(p_raw).items()}


        print(
            f"[residual_rl] Loaded initial weights and compatibility parameters from {path}: "
            f"{len(self._previous_run_best_stages)} stages",
            flush=True,
        )
        print(
            "[residual_rl] initial_weights_params contents:\n"
            + json.dumps(self._previous_run_best_stages, indent=2, sort_keys=True, default=str),
            flush=True,
        )

    def _banner(self, title: str, detail: str = "") -> None:
        """Print a visible status banner to the terminal."""
        line = f"[residual_rl] ══ {title} ══"
        if detail:
            line += f"  {detail}"
        print(f"\n{line}", flush=True)

    def _get_iteration_warm_start(self) -> tuple[str | None, str | None, str]:
        """Choose how each LLM iteration initializes policy weights."""
        if self._restart_each_iter_from_initial_warm_start:
            if self._initial_checkpoint is not None:
                if self._initial_prior_mode == "residual_rl":
                    return (
                        self._initial_checkpoint,
                        self._initial_prior_checkpoint,
                        "initial_residual_checkpoint_each_iter",
                    )
                return self._initial_checkpoint, None, "initial_checkpoint_each_iter"
            if self._initial_prior_checkpoint is not None:
                return None, self._initial_prior_checkpoint, "initial_prior_each_iter"
            raise RuntimeError(
                "[residual_rl] restart_each_iter_from_initial_warm_start requires "
                "initial_checkpoint or initial_prior_checkpoint."
            )
        return None, None, "scratch"

    @staticmethod
    def _stage_pass_rate_expr(stage_id: int) -> str:
        """Fraction of envs that advanced beyond the current stage."""
        return f"torch.mean((self._highest_stage_per_env > {int(stage_id)}).float())"

    @staticmethod
    def _true_success_stats_from_counts(
        *,
        true_success_count: int | None,
        false_complete_count: int | None,
        timeout_count: int | None,
    ) -> dict[str, float | int]:
        true_count = int(true_success_count or 0)
        false_count = int(false_complete_count or 0)
        timeout = int(timeout_count or 0)
        total = int(true_count + false_count + timeout)
        true_rate = true_count / float(total) if total > 0 else 0.0
        false_rate = false_count / float(total) if total > 0 else 0.0
        timeout_rate = timeout / float(total) if total > 0 else 0.0
        return {
            "true_success_count": true_count,
            "false_complete_count": false_count,
            "timeout_count": timeout,
            "total_episodes": total,
            "true_success_rate": true_rate,
            "false_complete_rate": false_rate,
            "timeout_rate": timeout_rate,
            "selection_score": true_rate - 0.5 * false_rate,
        }

    def _result_true_success_stats(self, r: TrainResult) -> dict[str, float | int]:
        return self._true_success_stats_from_counts(
            true_success_count=r.true_success_count,
            false_complete_count=r.false_complete_count,
            timeout_count=r.timeout_count,
        )

    def _last_true_success_stats(self) -> dict[str, float | int]:
        return self._true_success_stats_from_counts(
            true_success_count=self._last_stage_best_true_success_count,
            false_complete_count=self._last_stage_best_false_complete_count,
            timeout_count=self._last_stage_best_timeout_count,
        )

    @staticmethod
    def _format_rate(value: float | None) -> str:
        return "unknown" if value is None else f"{float(value):.6f}"

    def _true_success_prompt_context(self) -> str:
        stats = self._last_true_success_stats()
        return "\n".join(
            [
                "Strict objective: maximize true_success_rate. False completions are failures.",
                "Do not loosen constraints in ways that increase false completions; stage/proxy completion cannot justify true-success regression.",
                f"true_success_count: {int(stats['true_success_count'])}",
                f"false_complete_count: {int(stats['false_complete_count'])}",
                f"timeout_count: {int(stats['timeout_count'])}",
                f"total_episodes: {int(stats['total_episodes'])}",
                f"true_success_rate: {float(stats['true_success_rate']):.6f}",
                f"false_complete_rate: {float(stats['false_complete_rate']):.6f}",
                f"timeout_rate: {float(stats['timeout_rate']):.6f}",
                f"best_true_success_rate_so_far: {self._format_rate(self._best_true_success_rate_so_far)}",
                f"bc_baseline_true_success_rate: {self._format_rate(self._bc_baseline_true_success_rate)}",
            ]
        )

    def _update_true_success_incumbent(
        self,
        r: TrainResult,
        *,
        set_bc_baseline: bool = False,
    ) -> dict[str, float | int]:
        stats = self._result_true_success_stats(r)
        true_rate = float(stats["true_success_rate"])
        false_rate = float(stats["false_complete_rate"])
        if set_bc_baseline or self._bc_baseline_true_success_rate is None:
            self._bc_baseline_true_success_rate = true_rate
        if self._best_true_success_rate_so_far is None or true_rate > float(self._best_true_success_rate_so_far):
            self._best_true_success_rate_so_far = true_rate
        self._incumbent_true_success_rate = true_rate
        self._incumbent_false_complete_rate = false_rate
        return stats

    def _true_success_acceptance_decision(
        self,
        r: TrainResult,
    ) -> tuple[bool, str, dict[str, float | int]]:
        stats = self._result_true_success_stats(r)
        candidate_true = float(stats["true_success_rate"])
        candidate_false = float(stats["false_complete_rate"])
        reference_true = self._best_true_success_rate_so_far
        if reference_true is None:
            reference_true = self._incumbent_true_success_rate
        if reference_true is None:
            return True, "accepted:no_incumbent", stats

        tolerance = float(self._TRUE_SUCCESS_ACCEPTANCE_TOLERANCE)
        if candidate_true > float(reference_true):
            return True, "accepted:true_success_improved", stats
        if candidate_true < float(reference_true) - tolerance:
            return (
                False,
                (
                    "rejected:true_success_regression "
                    f"candidate={candidate_true:.6f} reference={float(reference_true):.6f} tolerance={tolerance:.6f}"
                ),
                stats,
            )

        incumbent_false = self._incumbent_false_complete_rate
        if incumbent_false is None or candidate_false <= float(incumbent_false):
            return True, "accepted:within_true_success_tolerance_no_false_complete_regression", stats
        return (
            False,
            (
                "rejected:false_complete_regression "
                f"candidate={candidate_false:.6f} incumbent={float(incumbent_false):.6f}"
            ),
            stats,
        )

    def _select_best_result(
        self,
        results: list[TrainResult],
        *,
        stage_id: int | None,
    ) -> tuple[int, float, float, list[str]]:
        """Pick the best successful worker by training selection score, then progress metrics."""
        best_run_idx: int | None = None
        best_progress_rate: float | None = None
        best_reward_metric: float | None = None
        best_sort_key: tuple[float, float, float, float, float] | None = None
        run_feedbacks: list[str] = []

        for run_idx, r in enumerate(results):
            if r.skipped:
                run_feedbacks.append("")
                continue
            if not r.success:
                run_feedbacks.append("")
                continue
            if not r.log_dir:
                raise RuntimeError(f"[residual_rl] Successful worker {r.worker_idx} is missing log_dir.")
            if r.progress_metric_eval is None:
                raise RuntimeError(
                    f"[residual_rl] Successful worker {r.worker_idx} is missing eval progress_metric_eval."
                )
            if r.reward_metric_eval is None:
                raise RuntimeError(
                    f"[residual_rl] Successful worker {r.worker_idx} is missing eval average episodic return."
                )
            fb = self._get_eval_feedback(r, stage_id=stage_id)
            run_feedbacks.append(fb)
            progress_rate = float(r.progress_metric_eval)
            reward_metric = float(r.reward_metric_eval)
            true_stats = self._result_true_success_stats(r)
            if r.selection_success_rate_window is None:
                if r.selection_checkpoint_policy == "eval_only_checkpoint":
                    selection_score = float(true_stats["true_success_rate"])
                else:
                    raise RuntimeError(
                        f"[residual_rl] Successful worker {r.worker_idx} is missing selection_success_rate_window; "
                        "refusing to rank Kuafu training by EMA or eval fallback."
                    )
            else:
                selection_score = float(r.selection_success_rate_window)
            selection_false = (
                float(r.selection_false_complete_rate)
                if r.selection_false_complete_rate is not None
                else float(true_stats["false_complete_rate"])
            )
            selection_timeout = (
                float(r.selection_timeout_rate)
                if r.selection_timeout_rate is not None
                else float(true_stats["timeout_rate"])
            )
            sort_key = (
                selection_score,
                -selection_false,
                -selection_timeout,
                progress_rate,
                reward_metric,
            )
            if best_sort_key is None or sort_key > best_sort_key:
                best_run_idx = run_idx
                best_progress_rate = progress_rate
                best_reward_metric = reward_metric
                best_sort_key = sort_key

        if best_run_idx is None or best_progress_rate is None or best_reward_metric is None:
            scope = "global finetune" if stage_id is None else f"stage {stage_id}"
            failures = [
                f"worker {r.worker_idx}: {r.exception or 'unknown error'}"
                for r in results
                if (not r.success and not r.skipped)
            ]
            failure_suffix = f" Failures: {'; '.join(failures)}." if failures else ""
            failure_log_path = os.path.join(self._run_dirs.root_dir, "worker_failures.jsonl")
            log_suffix = f" See {failure_log_path} for tracebacks." if failures else ""
            raise ValueError(f"No best run found for {scope}.{failure_suffix}{log_suffix}")
        return best_run_idx, best_progress_rate, best_reward_metric, run_feedbacks

    def _get_eval_feedback(self, r: TrainResult, *, stage_id: int | None = None) -> str:
        """Summarize the post-training eval rollout used for worker selection."""
        if r.progress_metric_eval is None or r.reward_metric_eval is None:
            raise RuntimeError("[residual_rl] Eval feedback requested without eval metrics.")

        true_stats = self._result_true_success_stats(r)
        lines = [
            "Strict objective: maximize true_success_rate. False completions are failures.",
            f"Eval progress_rate: {float(r.progress_metric_eval):.6f}",
            f"Eval average episodic return: {float(r.reward_metric_eval):.6f}",
            (
                "Eval termination counts: "
                f"true_success={int(true_stats['true_success_count'])} "
                f"false_complete={int(true_stats['false_complete_count'])} "
                f"timeout={int(true_stats['timeout_count'])} "
                f"total={int(true_stats['total_episodes'])}"
            ),
            f"Eval true_success_rate: {float(true_stats['true_success_rate']):.6f}",
            f"Eval false_complete_rate: {float(true_stats['false_complete_rate']):.6f}",
            f"Eval timeout_rate: {float(true_stats['timeout_rate']):.6f}",
            f"Eval true_success_selection_score: {float(true_stats['selection_score']):.6f}",
        ]
        selection_window_value = r.selection_success_rate_window
        if selection_window_value is not None:
            success_window = f"{float(selection_window_value):.6f}"
            lines.append(
                "Training checkpoint selection: "
                f"source={r.selection_source or 'unknown'} "
                f"metric_tag={r.selection_metric_tag or 'unknown'} "
                f"checkpoint_policy={r.selection_checkpoint_policy or 'unknown'} "
                f"post_critic_warmup_success_rate_window={success_window} "
                f"false_complete_rate={float(r.selection_false_complete_rate or 0.0):.6f} "
                f"timeout_rate={float(r.selection_timeout_rate or 0.0):.6f} "
                f"metric_index={r.selection_metric_index} "
                f"metric_step={r.selection_metric_step} "
                f"critic_warmup_epochs={r.selection_critic_warmup_epochs} "
                f"checkpoint_epoch={r.selection_checkpoint_epoch}"
            )

        true_success_count = int(r.true_success_count or 0)
        false_complete_count = int(r.false_complete_count or 0)
        timeout_count = int(r.timeout_count or 0)
        total_completed = true_success_count + false_complete_count + timeout_count

        if stage_id is None:
            e2e = format_end_to_end_rate_stats(
                true_success_count=true_success_count,
                false_complete_count=false_complete_count,
                total_episodes=total_completed,
            )
            lines.append(f"Eval formatted true_success_rate: {e2e['true_success_rate']}")
            lines.append(f"Eval formatted false_complete_rate: {e2e['false_complete_rate']}")
        else:
            highest = r.highest_stage_reached_per_env
            pass_count = (
                sum(1 for h in highest if int(h) > int(stage_id))
                if highest is not None
                else 0
            )
            total_envs = len(highest) if highest is not None else int(self._num_envs or 0)
            lines.append(f"Eval envs passed stage {stage_id}: {pass_count}/{total_envs}")

        stage_text = format_stage_satisfaction_times(
            r.stage_transition_stats,
            highest_stage_reached_per_env=r.highest_stage_reached_per_env,
            stage_id=stage_id,
            num_env=self._num_envs,
        )
        constraint_text = format_constraint_satisfaction_rates(
            r.constraint_met_rate_stats,
            stage_id=stage_id,
        )
        reward_text = format_reward_values(
            r.constraint_met_rate_stats,
            stage_id=stage_id,
        )
        lines.append(f"Eval stage satisfaction times:\n{stage_text}")
        lines.append(f"Eval constraint satisfaction rates:\n{constraint_text}")
        lines.append(f"Eval reward values:\n{reward_text}")
        return "\n".join(lines) + "\n"
    @staticmethod
    def _strip_llm_json(raw: str) -> str:
        """Strip markdown code fences and surrounding text to extract JSON."""
        import re
        text = raw.strip()
        m = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
        if m:
            text = m.group(1).strip()
        for prefix in ("{", "["):
            idx = text.find(prefix)
            if idx > 0 and idx < 200:
                text = text[idx:]
                break
        return text

    def _parse_stage_action(self, raw: str) -> dict[str, Any]:
        """Parse a stage action dict emitted by the LLM."""
        cleaned = self._strip_llm_json(raw)
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            import ast
            try:
                parsed = ast.literal_eval(cleaned)
            except Exception as exc:
                raise ValueError(
                    "[residual_rl] _parse_stage_action failed to parse LLM output. "
                    f"Raw (truncated): {raw[:300]}"
                ) from exc
        weights = parsed.get('Weights', {})
        # Completion thresholds are deterministic and never accepted from an
        # LLM response, including accidental/hallucinated extra fields.
        return {"Weights": weights, "Parameters": {}}

    def _make_trial_key(
        self,
        *,
        kind: str,
        stage_id: int | None,
        weights: dict[int, float] | None,
        parameters: dict[int, dict[str, object]] | None,
        all_stages_weights: dict[int, dict[int, float]] | None,
        all_stages_parameters: dict[int, dict[int, dict[str, object]]] | None = None,
    ) -> str:
        payload = {
            "kind": kind,
            "stage_id": stage_id,
            "weights": {int(k): float(v) for k, v in (weights or {}).items()} if weights is not None else None,
            "parameters": parameters,
            "all_stages_weights": all_stages_weights,
            "all_stages_parameters": all_stages_parameters,
        }
        s = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(s.encode("utf-8")).hexdigest()

    def _record_trial(
        self,
        *,
        trial_key: str,
        kind: str,
        stage_id: int,
        iter_idx: int,
        weights: dict,
        parameters: dict,
        all_stages_weights: dict,
        all_stages_parameters: dict,
        progress_rate: float,
        log_dir: str,
        checkpoint_path: str,
        raw_llm_output: str,
        accepted: bool = True,
        rejection_reason: str | None = None,
        true_success_stats: dict[str, float | int] | None = None,
    ) -> None:
        rec = {
            "timestamp": __import__("datetime").datetime.now().isoformat(),
            "trial_key": trial_key,
            "kind": kind,
            "stage_id": stage_id,
            "iter": int(iter_idx),
            "weights": weights,
            "parameters": parameters,
            "all_stages_weights": all_stages_weights,
            "all_stages_parameters": all_stages_parameters,
            "progress_rate": float(progress_rate),
            "log_dir": log_dir,
            "checkpoint_path": checkpoint_path,
            "raw_llm_output": raw_llm_output,
            "accepted": bool(accepted),
            "rejection_reason": rejection_reason,
            "best_true_success_rate_so_far": self._best_true_success_rate_so_far,
            "bc_baseline_true_success_rate": self._bc_baseline_true_success_rate,
            "incumbent_true_success_rate": self._incumbent_true_success_rate,
            "incumbent_false_complete_rate": self._incumbent_false_complete_rate,
        }
        if true_success_stats is not None:
            rec.update(
                {
                    "true_success_count": int(true_success_stats["true_success_count"]),
                    "false_complete_count": int(true_success_stats["false_complete_count"]),
                    "timeout_count": int(true_success_stats["timeout_count"]),
                    "total_episodes": int(true_success_stats["total_episodes"]),
                    "true_success_rate": float(true_success_stats["true_success_rate"]),
                    "false_complete_rate": float(true_success_stats["false_complete_rate"]),
                    "timeout_rate": float(true_success_stats["timeout_rate"]),
                    "selection_score": float(true_success_stats["selection_score"]),
                }
            )
        append_jsonl(self._run_dirs.trials_path, rec)

    def _archive_passed_residual_run(
        self,
        *,
        iter_idx: int,
        r_best: TrainResult,
        true_success_stats: dict[str, float | int],
        agent_hyperparameters: dict[str, object] | None,
        false_to_success_ema_ratio: float | None = None,
    ) -> str:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        source_root_abs = os.path.abspath(self._run_dirs.root_dir)
        archive_dir_abs = os.path.join(
            repo_root,
            "Residuals",
            self._task,
            os.path.basename(os.path.normpath(self._run_dirs.root_dir)),
        )

        def repo_rel(path: str | None) -> str | None:
            if not path:
                return None
            return os.path.relpath(os.path.abspath(str(path)), repo_root).replace(os.sep, "/")

        def relativize_json_paths(value: Any) -> Any:
            if isinstance(value, str) and os.path.isabs(value):
                return repo_rel(value)
            if isinstance(value, dict):
                return {str(k): relativize_json_paths(v) for k, v in value.items()}
            if isinstance(value, list):
                return [relativize_json_paths(v) for v in value]
            if isinstance(value, tuple):
                return [relativize_json_paths(v) for v in value]
            return value

        def archive_abs_for(source_path: str | None) -> str | None:
            if not source_path:
                return None
            abs_path = os.path.abspath(str(source_path))
            rel = os.path.relpath(abs_path, source_root_abs)
            if rel == os.pardir or rel.startswith(os.pardir + os.sep):
                return None
            return os.path.join(archive_dir_abs, rel)

        global_summary = self._best_summary.get("global")
        if isinstance(global_summary, dict):
            best_checkpoint_source_abs = os.path.abspath(
                str(global_summary.get("checkpoint") or r_best.checkpoint_path or "")
            )
        elif r_best.checkpoint_path:
            best_checkpoint_source_abs = os.path.abspath(str(r_best.checkpoint_path))
        else:
            best_checkpoint_source_abs = ""

        best_checkpoint_archive_abs = archive_abs_for(best_checkpoint_source_abs)
        selected_checkpoint_archive_abs = archive_abs_for(r_best.checkpoint_path)
        selected_log_dir_archive_abs = archive_abs_for(r_best.log_dir)
        best_summary_archive_abs = os.path.join(archive_dir_abs, "best.json")
        best_checkpoint_json_abs = os.path.join(archive_dir_abs, "best_checkpoint.json")
        marker_path_abs = os.path.join(archive_dir_abs, "residuals_archive.json")

        archive_dir_rel = repo_rel(archive_dir_abs)
        source_root_rel = repo_rel(source_root_abs)
        best_checkpoint_archive_rel = repo_rel(best_checkpoint_archive_abs)
        selected_checkpoint_archive_rel = repo_rel(selected_checkpoint_archive_abs)
        selected_log_dir_archive_rel = repo_rel(selected_log_dir_archive_abs)
        best_checkpoint_json_rel = repo_rel(best_checkpoint_json_abs)
        best_summary_archive_rel = repo_rel(best_summary_archive_abs)
        marker_path_rel = repo_rel(marker_path_abs)

        archive_record: dict[str, Any] = {
            "schema_version": 1,
            "kind": "residual_hparam_completed_archive",
            "task": self._task,
            "execution_robot_id": self._execution_robot_id,
            "artifact_robot_id": self._artifact_robot_id,
            "timestamp": __import__("datetime").datetime.now().isoformat(),
            "iter": int(iter_idx),
            "accepted": True,
            "completed": True,
            "source_run_dir": source_root_rel,
            "archive_run_dir": archive_dir_rel,
            "best_summary_source_path": repo_rel(self._run_dirs.best_path),
            "best_summary_archive_path": best_summary_archive_rel,
            "best_checkpoint_json_path": best_checkpoint_json_rel,
            "archive_marker_path": marker_path_rel,
            "best_checkpoint": {
                "source_path": repo_rel(best_checkpoint_source_abs),
                "archive_path": best_checkpoint_archive_rel,
                "selected_worker_checkpoint_source_path": repo_rel(r_best.checkpoint_path),
                "selected_worker_checkpoint_archive_path": selected_checkpoint_archive_rel,
                "selection_checkpoint_epoch": r_best.selection_checkpoint_epoch,
            },
            "selected_worker": {
                "worker_idx": int(r_best.worker_idx),
                "source_log_dir": repo_rel(r_best.log_dir),
                "archive_log_dir": selected_log_dir_archive_rel,
            },
            "stats": {
                "true_success_count": int(true_success_stats["true_success_count"]),
                "false_complete_count": int(true_success_stats["false_complete_count"]),
                "timeout_count": int(true_success_stats["timeout_count"]),
                "total_episodes": int(true_success_stats["total_episodes"]),
                "true_success_rate": float(true_success_stats["true_success_rate"]),
                "false_complete_rate": float(true_success_stats["false_complete_rate"]),
                "timeout_rate": float(true_success_stats["timeout_rate"]),
                "selection_score": float(true_success_stats["selection_score"]),
            },
            "completion": {
                "mode": "residual_hparam_iters_exhausted",
            },
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
            "agent_hyperparameters": relativize_json_paths(
                json.loads(json.dumps(agent_hyperparameters, default=str))
            )
            if agent_hyperparameters is not None
            else None,
            "best_summary_global": relativize_json_paths(
                json.loads(json.dumps(global_summary, default=str))
            )
            if isinstance(global_summary, dict)
            else None,
        }

        if false_to_success_ema_ratio is not None:
            cast(dict[str, Any], archive_record["stats"])["false_to_success_ema_ratio"] = float(
                false_to_success_ema_ratio
            )

        flush = getattr(self._tb_writer, "flush", None)
        if callable(flush):
            flush()
        close = getattr(self._tb_writer, "close", None)
        if callable(close):
            close()

        try:
            os.makedirs(os.path.dirname(archive_dir_abs), exist_ok=True)
            if os.path.exists(archive_dir_abs):
                raise FileExistsError(f"Archive directory already exists: {archive_dir_abs}")
            shutil.move(source_root_abs, archive_dir_abs)
        except Exception as exc:
            raise RuntimeError(
                f"[residual_rl] Failed to move completed residual run from {source_root_rel} to {archive_dir_rel}"
            ) from exc

        if best_checkpoint_archive_abs and not os.path.exists(best_checkpoint_archive_abs):
            best_checkpoint_archive_abs = selected_checkpoint_archive_abs
            best_checkpoint_archive_rel = selected_checkpoint_archive_rel
            archive_record["best_checkpoint"]["archive_path"] = best_checkpoint_archive_rel
        archive_record["best_checkpoint"]["archive_exists"] = bool(
            best_checkpoint_archive_abs and os.path.exists(best_checkpoint_archive_abs)
        )

        archive_best_summary = relativize_json_paths(
            json.loads(json.dumps(self._best_summary, default=str))
        )
        archive_global = archive_best_summary.get("global")
        if isinstance(archive_global, dict):
            archive_global["checkpoint_source"] = archive_global.get("checkpoint")
            if best_checkpoint_archive_rel:
                archive_global["checkpoint"] = best_checkpoint_archive_rel
            archive_global["residuals_archive_dir"] = archive_dir_rel
            archive_global["source_run_dir"] = source_root_rel
            archive_global["best_checkpoint_json"] = best_checkpoint_json_rel
        archive_record["best_summary_global"] = archive_global if isinstance(archive_global, dict) else None

        write_json(best_summary_archive_abs, archive_best_summary)
        write_json(best_checkpoint_json_abs, archive_record)
        write_json(marker_path_abs, archive_record)
        print(
            f"[residual_rl] Moved completed residual run: archive_dir={archive_dir_rel} "
            f"best_checkpoint={best_checkpoint_archive_rel or repo_rel(best_checkpoint_source_abs)}",
            flush=True,
        )
        return archive_dir_rel

    def _maybe_store_best_checkpoint(
        self,
        *,
        kind: str,
        stage_id: int | None,
        iter_idx: int,
        checkpoint_path: str | None,
        progress_rate: float,
        weights: dict[int, float] | None,
        parameters: dict[int, dict[str, object]] | None,
        all_stages_weights: dict[int, dict[int, float]] | None,
        all_stages_parameters: dict[int, dict[int, dict[str, object]]] | None = None,
        true_success_stats: dict[str, float | int] | None = None,
    ) -> None:
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            return
        if kind == "stage" and stage_id is not None:
            dst = os.path.join(self._run_dirs.checkpoints_dir, f"best_stage_{stage_id}.pth")
            # Persist the merged all-stage configuration so downstream global tuning / resumes
            # can reconstruct "initial sweep + stage-wise winners so far" in the correct order.
            if all_stages_weights is not None:
                for sid, stage_weights in dict(all_stages_weights).items():
                    stage_entry = cast(
                        dict[str, Any],
                        self._best_summary["stages"].get(str(int(sid))) or {},
                    )
                    stage_entry["weights"] = {
                        int(k): float(v) for k, v in dict(stage_weights).items()
                    }
                    if all_stages_parameters is not None and int(sid) in all_stages_parameters:
                        stage_entry["parameters"] = {
                            int(k): dict(v)
                            for k, v in dict(all_stages_parameters[int(sid)]).items()
                        }
                    self._best_summary["stages"][str(int(sid))] = stage_entry
            stage_entry = cast(dict[str, Any], self._best_summary["stages"].get(str(stage_id)) or {})
            stage_entry.update(
                {
                    "checkpoint": dst,
                    "iter": int(iter_idx),
                    "progress_rate": float(progress_rate),
                    "weights": weights,
                    "parameters": parameters,
                    "accepted": True,
                    "rejection_reason": None,
                    "best_true_success_rate_so_far": self._best_true_success_rate_so_far,
                    "bc_baseline_true_success_rate": self._bc_baseline_true_success_rate,
                }
            )
            if true_success_stats is not None:
                stage_entry.update(
                    {
                        "true_success_count": int(true_success_stats["true_success_count"]),
                        "false_complete_count": int(true_success_stats["false_complete_count"]),
                        "timeout_count": int(true_success_stats["timeout_count"]),
                        "total_episodes": int(true_success_stats["total_episodes"]),
                        "true_success_rate": float(true_success_stats["true_success_rate"]),
                        "false_complete_rate": float(true_success_stats["false_complete_rate"]),
                        "timeout_rate": float(true_success_stats["timeout_rate"]),
                        "selection_score": float(true_success_stats["selection_score"]),
                    }
                )
            self._best_summary["stages"][str(stage_id)] = stage_entry
        else:
            dst = os.path.join(self._run_dirs.checkpoints_dir, "best_global.pth")
            self._best_summary["global"] = {
                "checkpoint": dst,
                "iter": int(iter_idx),
                "progress_rate": float(progress_rate),
                "all_stages_weights": all_stages_weights,
                "all_stages_parameters": all_stages_parameters,
                    "accepted": True,
                "rejection_reason": None,
                "best_true_success_rate_so_far": self._best_true_success_rate_so_far,
                "bc_baseline_true_success_rate": self._bc_baseline_true_success_rate,
            }
            if true_success_stats is not None:
                cast(dict[str, Any], self._best_summary["global"]).update(
                    {
                        "true_success_count": int(true_success_stats["true_success_count"]),
                        "false_complete_count": int(true_success_stats["false_complete_count"]),
                        "timeout_count": int(true_success_stats["timeout_count"]),
                        "total_episodes": int(true_success_stats["total_episodes"]),
                        "true_success_rate": float(true_success_stats["true_success_rate"]),
                        "false_complete_rate": float(true_success_stats["false_complete_rate"]),
                        "timeout_rate": float(true_success_stats["timeout_rate"]),
                        "selection_score": float(true_success_stats["selection_score"]),
                    }
                )
        try:
            shutil.copy2(checkpoint_path, dst)
        except Exception as exc:
            raise RuntimeError(
                f"[residual_rl] Failed to copy best checkpoint from {checkpoint_path} to {dst}"
            ) from exc
        write_json(self._run_dirs.best_path, self._best_summary)
