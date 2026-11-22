"""Deprecated stage/global Kuafu runner.

This is the old stage/global weight-tuning pipeline. Active residual-RL
hyperparameter search uses kuafu.rl.core.runner_base.ResidualBaseRunner instead.
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
from kuafu.rl.llm.client import OpenAIChatClient
from kuafu.rl.video.video_prompts import build_gemini_prompt
from kuafu.rl.llm.parsing import parse_nested_stage_dict, parse_python_dict
from kuafu.rl.llm.prompts import (
    FAILURE_FEEDBACK,
    SINGLE_BOTH_PROMPT,
    SINGLE_BOTH_TUNING_FORMATTING_INSTRUCTIONS,
    MULTI_STAGE_END2END_PROMPT,
    MULTI_STAGE_SANITY_TUNING_FORMATTING_INSTRUCTIONS,
    VIDEO_EVIDENCE_PROMPT,
    SYSTEM_PROMPT,
)
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
)

from kuafu.rl.core.core_utilities import extract_tolerances_from_constraints_raw
from kuafu.skills.constraints.tolerance_policy import (
    assign_automatic_tolerances,
    policy_metadata,
    strip_tolerance_annotation,
    summarize_atomic_action,
)
from kuafu.common.env_prompt_helpers import (
    format_end_to_end_rate_stats,
    format_constraint_satisfaction_rates,
    format_stage_satisfaction_times,
    format_reward_values,
)

class ResidualRunner:
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
        stage_success_fraction: float,
        maximun_proposals: int,
        global_finetune_iters: int,
        global_finetune_max_false_to_true_ratio: float,
        stage_advance_completion_rate_cap: float,
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
        self._stage_success_fraction = float(stage_success_fraction)
        self._maximun_proposals = int(maximun_proposals)
        self._global_finetune_iters = int(global_finetune_iters)
        self._global_finetune_max_false_to_true_ratio = float(global_finetune_max_false_to_true_ratio)
        self._stage_advance_completion_rate_cap = float(stage_advance_completion_rate_cap)
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

        self._llm: OpenAIChatClient | None = None

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

    def _load_initial_weights_params(self, path: str) -> None:
        """Load stage weights and compatibility parameter maps from JSON.

        Parameter values are recomputed automatically after loading and are
        never included in an LLM prompt.

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
        """Pick the best successful worker by strict true success, then progress metrics."""
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
            selection_score = (
                float(r.selection_success_rate_ema)
                if r.selection_success_rate_ema is not None
                else float(true_stats["true_success_rate"])
            )
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
        if r.selection_success_rate_ema is not None:
            success_ema = (
                "none"
                if r.selection_success_rate_ema is None
                else f"{float(r.selection_success_rate_ema):.6f}"
            )
            lines.append(
                "Training checkpoint selection: "
                f"source={r.selection_source or 'unknown'} "
                f"metric_tag={r.selection_metric_tag or 'unknown'} "
                f"checkpoint_policy={r.selection_checkpoint_policy or 'unknown'} "
                f"post_critic_warmup_success_rate_ema={success_ema} "
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

    def run(self) -> None:
        self._banner("RUN START", f"task={self._task}  workers={self._num_workers}  device={self._device}")
        print(f"[residual_rl] logs: {self._run_dirs.root_dir}", flush=True)
        self._pool.start()
        try:
            task_info = self._pool.get_task_info(timeout_s=300.0)
            if task_info is None:
                raise RuntimeError("[residual_rl] Failed to collect task info from worker pool.")
            self._task_info = task_info

            # If resuming, seed stage weights/parameters from the provided run dir.
            if self._previous_run_best_stages:
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
                    if "checkpoint" in stage_data and stage_data.get("checkpoint"):
                        seeded_entry["checkpoint"] = stage_data.get("checkpoint")
                    if "iter" in stage_data:
                        seeded_entry["iter"] = stage_data.get("iter")
                    self._best_summary["stages"][str(stage_id)] = seeded_entry
                print(
                    f"[residual_rl] Seeded stage weights from resume run "
                    f"({len(self._previous_run_best_stages)} stages)",
                    flush=True,
                )

            # The LLM never owns completion thresholds. Recompute them even
            # when an older resume artifact contains model-selected values,
            # then apply the automatic map in the very first worker rollout.
            assign_automatic_tolerances(self._task_info)
            stages = cast(dict[int, dict[str, Any]], self._task_info["stages"])
            self._resume_all_stages_weights = {
                int(stage_id): {
                    int(index): float(value)
                    for index, value in dict(stage_meta["weights"]).items()
                }
                for stage_id, stage_meta in stages.items()
            }
            self._resume_all_stages_parameters = {
                int(stage_id): {
                    int(index): dict(values)
                    for index, values in dict(stage_meta["parameters"]).items()
                }
                for stage_id, stage_meta in stages.items()
            }
            self._best_summary["automatic_tolerance_policy"] = policy_metadata()
            for stage_id, stage_meta in stages.items():
                entry = cast(
                    dict[str, Any],
                    self._best_summary["stages"].get(str(stage_id)) or {},
                )
                entry["parameters"] = self._resume_all_stages_parameters[int(stage_id)]
                entry["tolerance_source"] = "automatic"
                entry["tolerance_decisions"] = stage_meta.get("tolerance_decisions", {})
                self._best_summary["stages"][str(stage_id)] = entry

            stage_count = int(self._task_info["stage_count"])
            start_with_individual_tuning = True
            if self._last_stage_best_highest_stage_reached_per_env is not None:
                pre_dry_stage_id = self._compute_initial_stage_to_tune(stage_count=stage_count)
                start_with_individual_tuning = int(pre_dry_stage_id) < int(stage_count)

            run_initial_bootstrap = True
            if run_initial_bootstrap:
                bootstrap_eval_only = self._initial_eval_checkpoint is not None
                if bootstrap_eval_only:
                    dry_ckpt = self._initial_eval_checkpoint
                    dry_prior = (
                        self._initial_prior_checkpoint
                        if self._initial_prior_mode == "residual_rl"
                        else None
                    )
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
                dry_epochs = (
                    int(self._init_train_epoch)
                    if start_with_individual_tuning
                    else int(self._max_train_epoch)
                )
                dry_progress_metric_expr = (
                    self._stage_pass_rate_expr(0)
                    if start_with_individual_tuning
                    else f"torch.mean((self._highest_stage_per_env >= {int(stage_count)}).float())"
                )
                dry_tag = "dry_run" if start_with_individual_tuning else "global_dry_run"
                dry_mode = "individual_stage_tuning" if start_with_individual_tuning else "global_tuning"
                self._banner(
                    "INITIAL EVAL" if bootstrap_eval_only else "DRY RUN",
                    f"mode={dry_mode}  epochs={dry_epochs}  checkpoint={dry_source}",
                )
                stage_id = 0
                dry_cfgs = [
                    TrainConfig(
                        stage_id=stage_id,
                        all_stages_weights=self._resume_all_stages_weights,
                        all_stages_parameters=self._resume_all_stages_parameters,
                        progress_metric_expr=dry_progress_metric_expr,
                        max_train_epoch_override=dry_epochs,
                        resume_from_checkpoint=None if bootstrap_eval_only else dry_ckpt,
                        eval_only_checkpoint=dry_ckpt if bootstrap_eval_only else None,
                        prior_checkpoint=(
                            dry_prior
                            if self._initial_prior_mode == "residual_rl"
                            else (None if bootstrap_eval_only else dry_prior)
                        ),
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
                        tag="initial_eval_checkpoint" if bootstrap_eval_only else dry_tag,
                    )
                    for _ in range(int(self._num_workers))
                ]
                # Run the initial bootstrap (eval-only from checkpoint, or train+eval dry run)
                # on every available worker, then keep only the best worker's stats/checkpoint
                # so prompt context mirrors later learning iterations.
                dry_results = self._pool.train(dry_cfgs)
                dry_best_run_idx, dry_best_metric, dry_best_secondary, _dry_feedbacks = self._select_best_result(
                    dry_results,
                    stage_id=stage_id,
                )
                dry_best = dry_results[dry_best_run_idx]
                for run_idx, r in enumerate(dry_results):
                    marker = " ★ BEST" if run_idx == dry_best_run_idx else ""
                    status = "skip" if r.skipped else ("ok" if r.success else "FAIL")
                    progress = r.progress_metric_eval
                    secondary = r.reward_metric_eval
                    print(
                        f"[residual_rl]   Worker {r.worker_idx} ({'initial eval' if bootstrap_eval_only else 'dry run'}): "
                        f"status={status} eval_progress_rate={progress} eval_average_episodic_return={secondary}{marker}",
                        flush=True,
                    )
                self._last_stage_best_stage_transition_stats = dry_best.stage_transition_stats
                self._last_stage_best_highest_stage_reached_per_env = dry_best.highest_stage_reached_per_env
                self._last_stage_best_constraint_met_rate_stats = dry_best.constraint_met_rate_stats
                self._last_stage_best_true_success_count = dry_best.true_success_count
                self._last_stage_best_false_complete_count = dry_best.false_complete_count
                self._last_stage_best_timeout_count = dry_best.timeout_count
                self._last_best_checkpoint_path = dry_best.checkpoint_path
                dry_true_stats = self._update_true_success_incumbent(dry_best, set_bc_baseline=True)
                self._save_prompt_context()
                print(
                    f"[residual_rl] {'Initial eval' if bootstrap_eval_only else 'Dry run'} done: best_progress_rate={dry_best_metric}  "
                    f"best_average_episodic_return={dry_best_secondary}  "
                    f"true_success_rate={float(dry_true_stats['true_success_rate']):.6f}  "
                    f"false_complete_rate={float(dry_true_stats['false_complete_rate']):.6f}  "
                    f"best_worker={dry_best.worker_idx}  checkpoint={dry_best.checkpoint_path}",
                    flush=True,
                )

            # Stage-wise tuning produces a single joint (all-stages) configuration.
            # We keep it in stage-indexed dicts but treat it as the "global" config thereafter.
            # After the dry run (or when loading previous prompt context), determine whether
            # we should run stage-wise tuning at all, and if so, where to start.
            start_stage_id = 0
            skip_stage_tuning = False
            if self._last_stage_best_highest_stage_reached_per_env is not None:
                start_stage_id = self._compute_initial_stage_to_tune(stage_count=stage_count)
                if int(start_stage_id) >= int(stage_count):
                    # Dry run (or loaded previous context) already reached the end or hit the
                    # completion cap condition; skip per-stage tuning entirely.
                    skip_stage_tuning = True
                    print(
                        "[residual_rl] Dry-run/previous stats indicate all stages passed or completion "
                        "cap hit; skipping individual stage tuning.",
                        flush=True,
                    )
                else:
                    print(
                        f"[residual_rl] Initial stage chosen from stats: start_stage_id={start_stage_id}",
                        flush=True,
                    )

            if skip_stage_tuning:
                print(
                    "\n"
                    "╔══════════════════════════════════════════════════════════════╗\n"
                    "║  STAGE TUNING — SKIPPED (all stages passed or cap hit)      ║\n"
                    "╚══════════════════════════════════════════════════════════════╝",
                    flush=True,
                )
                # No stage-wise tuning: take the per-stage weights/parameters directly from
                # the task info (which may already be seeded from a previous run).
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
                    all_stages_parameters[int(sid)] = dict(base_parameters)
            else:
                print(
                    "\n"
                    "╔══════════════════════════════════════════════════════════════╗\n"
                    f"║  STAGE TUNING — {self._maximun_proposals} proposals, "
                    f"{self._num_workers} workers, start_stage={start_stage_id}"
                    + " " * max(0, 21 - len(str(self._maximun_proposals)) - len(str(self._num_workers)) - len(str(start_stage_id)))
                    + "║\n"
                    "╚══════════════════════════════════════════════════════════════╝",
                    flush=True,
                )
                all_stages_weights, all_stages_parameters = self._tune_individual_stage(
                    start_stage_id=start_stage_id
                )

            # Optional global finetune using the stage-wise result as the seed.
            if self._global_finetune_iters > 0:
                print(
                    "\n"
                    "╔══════════════════════════════════════════════════════════════╗\n"
                    f"║  GLOBAL FINETUNE — {self._global_finetune_iters} iters, "
                    f"{self._num_workers} workers"
                    + " " * max(0, 30 - len(str(self._global_finetune_iters)) - len(str(self._num_workers)))
                    + "║\n"
                    "╚══════════════════════════════════════════════════════════════╝",
                    flush=True,
                )
                self._global_finetune(
                    initial_weights=all_stages_weights,
                    initial_parameters=all_stages_parameters,
                    stage_count=stage_count,
                )
            else:
                print(
                    "\n[residual_rl] Global finetune: SKIPPED (global_finetune_iters=0)",
                    flush=True,
                )

            self._banner("RUN COMPLETE", f"logs={self._run_dirs.root_dir}")
        finally:
            self._pool.close()

    def _compute_initial_stage_to_tune(self, *, stage_count: int) -> int:
        """Decide which stage to tune first based on latest env stats.

        Behaviour:
        - If completion rate (true_success + false_complete) / total_episodes reaches
          `stage_advance_completion_rate_cap`, we return `stage_count` to signal that
          individual-stage tuning can be skipped entirely.
        - Otherwise, we scan stages from 0..stage_count-1 and return the first stage
          whose pass count is below the threshold `num_env * stage_success_fraction`.
        - If all stages appear passed, we again return `stage_count` (skip tuning).
        - If stats are missing or invalid, we fall back to 0 to preserve behaviour.
        """
        highest = self._last_stage_best_highest_stage_reached_per_env
        if highest is None or stage_count <= 0:
            return 0

        total_eps = len(highest)
        if total_eps <= 0:
            return 0

        num_env = self._num_envs
        if num_env is None:
            # Fallback: infer env count from stats length.
            num_env = total_eps
        threshold = float(num_env) * float(self._stage_success_fraction)

        # Check completion cap: if almost all episodes have terminated (true success or
        # false complete), we treat it as the same stopping condition used inside
        # `_tune_individual_stage` and skip stage-wise tuning.
        true_c = int(self._last_stage_best_true_success_count or 0)
        false_c = int(self._last_stage_best_false_complete_count or 0)
        timeout_c = int(self._last_stage_best_timeout_count or 0)
        denom_eps = int(true_c + false_c + timeout_c)
        if denom_eps <= 0:
            # Fallback: older envs may not report termination counts; assume one episode per env.
            denom_eps = int(total_eps)
        completion_rate = (true_c + false_c) / float(denom_eps)
        if completion_rate >= float(self._stage_advance_completion_rate_cap):
            return int(stage_count)

        # Otherwise, find the first stage that is not yet passed by the pass-count threshold.
        for sid in range(int(stage_count)):
            pass_count_sid = sum(1 for h in highest if int(h) > int(sid))
            if float(pass_count_sid) < float(threshold):
                return int(sid)

        # All stages already above threshold; signal that stage-wise tuning can be skipped.
        return int(stage_count)

    def _global_finetune(
        self,
        *,
        initial_weights: dict[int, dict[int, float]],
        initial_parameters: dict[int, dict[int, dict[str, object]]],
        stage_count: int,
    ) -> None:
        """Optional global fine-tuning pass (end-to-end).

        Similar to `_tune_individual_stage`, but:
        - uses `MULTI_STAGE_END2END_PROMPT` for end-to-end reasoning, and
        - proposes weights for *all* stages jointly while parameters stay fixed.
        """
        if int(self._global_finetune_iters) <= 0:
            return

        stages_all = cast(dict[int, dict[str, Any]], self._task_info.get("stages") or {})
        task_description = (self._task_info.get("task_description") or "").strip() or self._task

        # 1) Seed from env defaults (task_info) so every stage has weights/parameters.
        current_weights_by_stage: dict[int, dict[int, float]] = {}
        current_parameters_by_stage: dict[int, dict[int, dict[str, object]]] = {}
        for sid in range(int(stage_count)):
            sm = cast(dict[str, Any], stages_all.get(int(sid)) or {})
            base_weights = sm.get("weights") or {}
            constraints_raw = sm.get("constraints_raw")
            if "parameters" in sm:
                base_parameters = cast(dict[int, dict[str, object]], sm["parameters"])
            else:
                base_parameters = extract_tolerances_from_constraints_raw(constraints_raw) if constraints_raw is not None else {}
            current_weights_by_stage[int(sid)] = {int(k): float(v) for k, v in dict(base_weights).items()}
            current_parameters_by_stage[int(sid)] = dict(base_parameters)

        # 2) Overwrite with previous-run (or stage-wise) weights/parameters where present.
        for sid in range(int(stage_count)):
            if sid in initial_weights:
                current_weights_by_stage[sid] = {int(k): float(v) for k, v in initial_weights[sid].items()}
            else:
                warnings.warn(
                    f"[residual_rl] global_finetune: no weights for stage {sid} in previous run; using env defaults.",
                    stacklevel=2,
                )
            if sid in initial_parameters:
                current_parameters_by_stage[sid] = {int(k): dict(v) for k, v in initial_parameters[sid].items()}
            else:
                warnings.warn(
                    f"[residual_rl] global_finetune: no parameters for stage {sid} in previous run; using env defaults.",
                    stacklevel=2,
                )

        # Build static prompt blocks from task metadata.
        stage_descriptions_lines: list[str] = []
        stage_descriptions_reasoning_lines: list[str] = []
        constraints_expl_lines: list[str] = []
        for sid in range(int(stage_count)):
            sm = cast(dict[str, Any], stages_all[int(sid)])
            atomic_action = sm.get("atomic_action")
            if not isinstance(atomic_action, str) or not atomic_action.strip():
                raise RuntimeError(f"[residual_rl] global_finetune: stage {sid} is missing atomic_action.")
            action_summary = summarize_atomic_action(atomic_action)
            stage_descriptions_lines.append(f"  Stage {sid}: {action_summary}")
            stage_descriptions_reasoning_lines.append(f"Stage {sid}: {action_summary}")
            constraints_expl_lines.append(f"Stage {sid}:")
            for e in cast(list, sm["explanations"]):
                constraints_expl_lines.append(f"  - {strip_tolerance_annotation(e)}")
        stage_descriptions = "\n".join(stage_descriptions_lines)
        stage_descriptions_reasoning = "\n".join(stage_descriptions_reasoning_lines)
        constraints_explanations_by_stage = "\n".join(constraints_expl_lines)
        detectable_objects_str = json.dumps(
            self._task_info["detectable_objects"], indent=2, sort_keys=True, default=str
        )

        def _make_gemini_prompt(
            w_by_stage: dict[int, dict[int, float]] | None = None,
            tsr: str | float | None = None,
            fcr: str | float | None = None,
            iter_ctx: str | None = None,
        ) -> str:
            cstats = self._last_stage_best_constraint_met_rate_stats
            constraint_text = format_constraint_satisfaction_rates(cstats) if cstats else None
            trans_text = format_stage_satisfaction_times(
                self._last_stage_best_stage_transition_stats,
                highest_stage_reached_per_env=self._last_stage_best_highest_stage_reached_per_env,
                num_env=self._num_envs,
            ) if self._last_stage_best_stage_transition_stats else None
            return build_gemini_prompt(
                task_description=task_description,
                stage_count=int(stage_count),
                stage_descriptions=stage_descriptions,
                constraints_explanations_by_stage=constraints_explanations_by_stage,
                current_weights_by_stage=w_by_stage,
                true_success_rate=tsr,
                false_complete_rate=fcr,
                constraint_stats_text=constraint_text,
                stage_transition_text=trans_text,
                num_envs_total=self._num_envs,
                num_envs_in_video=self._video_num_envs_to_record,
                iteration_context=iter_ctx,
            )

        # Multi-stage finetune sends weight-only proposals; parameters remain fixed.
        multi_system = SYSTEM_PROMPT + MULTI_STAGE_SANITY_TUNING_FORMATTING_INSTRUCTIONS
        llm_multi = OpenAIChatClient(
            model=self._gpt_model,
            temperature=self._temperature,
            num_suggestions=self._num_workers,
            system_prompt=multi_system,
        )

        stage_transition_stats = self._last_stage_best_stage_transition_stats
        highest_stage_reached = self._last_stage_best_highest_stage_reached_per_env
        per_stage_satisfaction_time_stats = format_stage_satisfaction_times(
            stage_transition_stats,
            highest_stage_reached_per_env=highest_stage_reached,
            stage_id=None,
            num_env=self._num_envs,
        )

        # End-to-end rates: prefer explicit env termination stats (Kuafu env_learned),
        # otherwise fall back to "stage complete rate" derived from highest_stage_reached.
        if highest_stage_reached is None:
            raise RuntimeError("[residual_rl] global_finetune: missing highest_stage_reached_per_env stats")
        total_eps = int(len(highest_stage_reached))
        # Numeric end-to-end rates for control logic.
        true_success_count = int(self._last_stage_best_true_success_count or 0)
        false_complete_count = int(self._last_stage_best_false_complete_count or 0)
        timeout_count = int(self._last_stage_best_timeout_count or 0)
        denom_eps_int = int(true_success_count + false_complete_count + timeout_count)
        if denom_eps_int <= 0:
            denom_eps_int = int(total_eps)
        denom_eps = float(denom_eps_int)
        true_success_rate_value = true_success_count / denom_eps if denom_eps > 0.0 else 0.0
        false_complete_rate_value = false_complete_count / denom_eps if denom_eps > 0.0 else 0.0
        # Human-readable rates for LLM prompts.
        e2e = format_end_to_end_rate_stats(
            true_success_count=true_success_count,
            false_complete_count=false_complete_count,
            total_episodes=denom_eps_int,
        )
        true_success_rate_text = e2e["true_success_rate"]
        false_complete_rate_text = e2e["false_complete_rate"]


        # Video evidence: use previous-run saved video if available; else generate or omit.
        video_rollouts: str
        prev_run_dir = getattr(self, "_previous_run_dir", None)
        rollouts_file = os.path.join(prev_run_dir or "", "videos", "global_preprompt_rollouts.txt")
        if prev_run_dir and os.path.isfile(rollouts_file):
            with open(rollouts_file, "r", encoding="utf-8") as f:
                video_rollouts = f.read()
            video_rollouts = VIDEO_EVIDENCE_PROMPT.format(video_rollouts=video_rollouts)
            print("[residual_rl] Using video evidence from previous run.", flush=True)
        elif self._video_feedback and self._last_best_checkpoint_path:
            from kuafu.rl.video.export import generate_video_feedback

            video_out_dir = os.path.join(self._run_dirs.root_dir, "videos", "global_preprompt")
            video_rollouts, video_meta = generate_video_feedback(
                task=self._task,
                checkpoint_path=str(self._last_best_checkpoint_path),
                video_length=self._video_length,
                num_envs_to_record=self._video_num_envs_to_record,
                output_dir=video_out_dir,
                device=self._device,
                headless=True,
                seed=int(self._env_seed),
                execution_robot_id=self._execution_robot_id,
                all_stages_weights=current_weights_by_stage,
                all_stages_parameters=current_parameters_by_stage,
                gemini_enabled=bool(self._gemini_model),
                gemini_model=self._gemini_model,
                context_text=_make_gemini_prompt(
                    w_by_stage=current_weights_by_stage,
                    tsr=true_success_rate_text,
                    fcr=false_complete_rate_text,
                    iter_ctx="Pre-loop baseline video (before any global finetune iteration).",
                ),
            )
            rollouts_save_path = os.path.join(self._run_dirs.root_dir, "videos", "global_preprompt_rollouts.txt")
            try:
                os.makedirs(os.path.dirname(rollouts_save_path), exist_ok=True)
                with open(rollouts_save_path, "w", encoding="utf-8") as f:
                    f.write(video_rollouts)
            except Exception as exc:
                raise RuntimeError(
                    f"[residual_rl] Failed to save global preprompt rollout text: {rollouts_save_path}"
                ) from exc
            video_rollouts = VIDEO_EVIDENCE_PROMPT.format(video_rollouts=video_rollouts)
            append_jsonl(
                self._run_dirs.videos_path,
                {
                    "timestamp": __import__("datetime").datetime.now().isoformat(),
                    "kind": "global_preprompt",
                    "iter": None,
                    "checkpoint_path": str(self._last_best_checkpoint_path),
                    "video_out_dir": str(video_out_dir),
                    "meta": dict(video_meta),
                },
            )
            if not bool(video_meta.get("ok")):
                raise RuntimeError(
                    "[residual_rl] Video feedback required for global preprompt but generation failed: "
                    f"{video_meta.get('error') or 'unknown error'}"
                )
        else:
            video_rollouts = "(No video evidence provided.)"
            if self._video_feedback and not self._last_best_checkpoint_path:
                video_rollouts = "\n[residual_rl] Video feedback: no checkpoint available.\n"

        # Strict baseline key sets for proposal validation.
        baseline_stage_keys = set(current_weights_by_stage.keys())
        baseline_weight_keys = {s: set(current_weights_by_stage[s].keys()) for s in baseline_stage_keys}
        assistant_prompt: str | None = None
        for iter_idx in range(int(self._global_finetune_iters)):
            current_weights_str = json.dumps(current_weights_by_stage, indent=2, sort_keys=True)

            stage_user_prompt = MULTI_STAGE_END2END_PROMPT.format(
                task_description=task_description,
                detectable_objects=detectable_objects_str,
                stage_count=int(stage_count),
                stage_descriptions_reasoning=stage_descriptions_reasoning,
                constraints_explanations_by_stage=constraints_explanations_by_stage,
                current_weights_by_stage=current_weights_str,
                true_success_rate=true_success_rate_text,
                false_complete_rate=false_complete_rate_text,
                true_success_context=self._true_success_prompt_context(),
                per_stage_satisfaction_time_stats=per_stage_satisfaction_time_stats,
                video_rollouts=video_rollouts,
            )

            print(
                f"\n[residual_rl] ── GLOBAL FINETUNE iter {iter_idx}/{int(self._global_finetune_iters)-1} "
                f"({self._num_workers} workers) ──",
                flush=True,
            )
            print(stage_user_prompt, flush=True)

            llm_resp = llm_multi.prompt(user_prompt=stage_user_prompt, assistant_prompt=assistant_prompt)
            ts = __import__("datetime").datetime.now().isoformat()
            append_jsonl(
                self._run_dirs.openai_payloads_path,
                {
                    "timestamp": ts,
                    "kind": "global_finetune",
                    "iter": int(iter_idx),
                    "messages": llm_resp.messages,
                    "raw_outputs": llm_resp.raw_outputs,
                    "extra": {"stage_count": int(stage_count)},
                },
            )
            append_prompt_transcript(
                path=self._run_dirs.prompt_transcript_path,
                header={"timestamp": ts, "kind": "global_finetune", "iter": int(iter_idx)},
                messages=llm_resp.messages,
                raw_outputs=llm_resp.raw_outputs,
            )

            # Parse proposals (STRICT). Prompt already demands full stage/index integrity.
            parsed_actions: list[tuple[dict[int, dict[int, float]], dict[int, dict[int, dict[str, object]]]]] = []
            for raw in llm_resp.raw_outputs:
                try:
                    cleaned = self._strip_llm_json(raw)
                    try:
                        parsed = json.loads(cleaned)
                    except json.JSONDecodeError:
                        import ast
                        parsed = ast.literal_eval(cleaned)

                    w_raw = parsed["Weights"]
                    weights_by_stage: dict[int, dict[int, float]] = {
                        int(s): {int(k): float(v) for k, v in dict(w).items()} for s, w in dict(w_raw).items()
                    }
                    params_by_stage = {
                        int(s): {int(i): dict(patch) for i, patch in dict(stage_params).items()}
                        for s, stage_params in current_parameters_by_stage.items()
                    }

                    if set(weights_by_stage.keys()) != baseline_stage_keys:
                        raise KeyError("[residual_rl] global_finetune: proposal stage keys mismatch in Weights")
                    for s in baseline_stage_keys:
                        if set(weights_by_stage[s].keys()) != baseline_weight_keys[s]:
                            raise KeyError(f"[residual_rl] global_finetune: proposal weight keys mismatch at stage {s}")

                    parsed_actions.append((weights_by_stage, params_by_stage))
                except Exception as exc:
                    print(
                        f"[residual_rl] WARNING: global_finetune proposal parse failed ({type(exc).__name__}: {exc}); "
                        f"reusing current best weights for this slot. Raw (truncated): {raw[:300]}",
                        flush=True,
                    )
                    parsed_actions.append((
                        {int(s): dict(w) for s, w in current_weights_by_stage.items()},
                        {int(s): dict(p) for s, p in current_parameters_by_stage.items()},
                    ))

            # Global finetune evaluates end-to-end performance, so always train each
            # proposal to the full budget instead of using the progressive stage-wise schedule.
            current_epochs = int(self._max_train_epoch)
            resume_ckpt, prior_ckpt, warm_start_label = self._get_iteration_warm_start()

            # End-to-end progress rate: fraction of envs that reached stage_count (no fallbacks).
            progress_metric_expr = f"torch.mean((self._highest_stage_per_env >= {int(stage_count)}).float())"

            cfgs: list[TrainConfig] = []
            trial_keys: list[str] = []
            for (w_all, p_all) in parsed_actions:
                trial_keys.append(
                    self._make_trial_key(
                        kind="global",
                        stage_id=None,
                        weights=None,
                        parameters=None,
                        all_stages_weights=w_all,
                        all_stages_parameters=p_all,
                    )
                )
                cfgs.append(
                    TrainConfig(
                        stage_id=0,  # worker requires a stage_id for readback bookkeeping
                        all_stages_weights=w_all,
                        progress_metric_expr=progress_metric_expr,
                        all_stages_parameters=p_all,
                        max_train_epoch_override=int(current_epochs),
                        resume_from_checkpoint=resume_ckpt,
                        prior_checkpoint=prior_ckpt,
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
                        tag=f"global_iter{iter_idx}",
                    )
                )

            if resume_ckpt or prior_ckpt:
                print(
                    f"[residual_rl] Warm-starting global finetune iter {iter_idx} via {warm_start_label}: "
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
            best_progress_rate = float(iter_best)
            candidate_true_stats = self._result_true_success_stats(r_best)
            accepted_this_iter, acceptance_reason, candidate_true_stats = self._true_success_acceptance_decision(r_best)

            # Per-worker result summary.
            for run_idx, r in enumerate(results):
                marker = " ★ BEST" if run_idx == best_run_idx else ""
                status = "skip" if r.skipped else ("ok" if r.success else "FAIL")
                progress = r.progress_metric_eval
                secondary = r.reward_metric_eval
                print(
                    f"[residual_rl]   Worker {r.worker_idx} (global iter {iter_idx}): "
                    f"status={status} eval_progress_rate={progress} eval_average_episodic_return={secondary}{marker}",
                    flush=True,
                )
            print(
                f"[residual_rl] Global finetune iter {iter_idx}: eval_best_progress_rate={best_progress_rate} "
                f"eval_best_average_episodic_return={iter_best_secondary} "
                f"(best_worker={results[best_run_idx].worker_idx})",
                flush=True,
            )

            candidate_all_weights = cfgs[best_run_idx].all_stages_weights
            candidate_all_parameters = cfgs[best_run_idx].all_stages_parameters
            if candidate_all_weights is None or candidate_all_parameters is None:
                raise RuntimeError("[residual_rl] global_finetune: missing candidate all-stages weights/parameters")

            if accepted_this_iter:
                # Roll forward only true-success-safe joint weights/parameters.
                current_weights_by_stage = {int(s): dict(w) for s, w in dict(candidate_all_weights).items()}
                current_parameters_by_stage = {int(s): dict(p) for s, p in dict(candidate_all_parameters).items()}
                for sid in range(int(stage_count)):
                    self._iter_best_stage_weights[int(sid)] = dict(current_weights_by_stage[int(sid)])
                    self._iter_best_stage_parameters[int(sid)] = dict(current_parameters_by_stage[int(sid)])

                # Update evidence for the next prompt.
                self._last_stage_best_stage_transition_stats = r_best.stage_transition_stats
                self._last_stage_best_highest_stage_reached_per_env = r_best.highest_stage_reached_per_env
                self._last_stage_best_constraint_met_rate_stats = r_best.constraint_met_rate_stats
                self._last_stage_best_true_success_count = r_best.true_success_count
                self._last_stage_best_false_complete_count = r_best.false_complete_count
                self._last_stage_best_timeout_count = r_best.timeout_count
                self._last_best_checkpoint_path = r_best.checkpoint_path
                candidate_true_stats = self._update_true_success_incumbent(r_best)
                self._save_prompt_context()
            else:
                print(
                    f"[residual_rl] Global finetune iter {iter_idx}: candidate rejected by true-success guard: "
                    f"{acceptance_reason}",
                    flush=True,
                )

            stage_transition_stats = self._last_stage_best_stage_transition_stats
            highest_stage_reached = self._last_stage_best_highest_stage_reached_per_env

            per_stage_satisfaction_time_stats = format_stage_satisfaction_times(
                stage_transition_stats,
                highest_stage_reached_per_env=highest_stage_reached,
                stage_id=None,
                num_env=self._num_envs,
            )
            if highest_stage_reached is None:
                raise RuntimeError("[residual_rl] global_finetune: missing highest_stage_reached_per_env after run")
            true_success_count = int(self._last_stage_best_true_success_count or 0)
            false_complete_count = int(self._last_stage_best_false_complete_count or 0)
            timeout_count = int(self._last_stage_best_timeout_count or 0)
            if true_success_count == 0:
                print(
                    "[residual_rl] WARNING: true_success_count is 0; continuing true-success-first "
                    "selection with a zero success baseline.",
                    flush=True,
                )
            denom_eps_int = int(true_success_count + false_complete_count + timeout_count)
            if denom_eps_int <= 0:
                # Fallback: treat as one episode per env if no termination counts.
                denom_eps_int = int(len(highest_stage_reached))
            denom_eps = float(denom_eps_int)
            true_success_rate_value = true_success_count / denom_eps if denom_eps > 0.0 else 0.0
            false_complete_rate_value = false_complete_count / denom_eps if denom_eps > 0.0 else 0.0
            e2e = format_end_to_end_rate_stats(
                true_success_count=true_success_count,
                false_complete_count=false_complete_count,
                total_episodes=denom_eps_int,
            )
            true_success_rate_text = e2e["true_success_rate"]
            false_complete_rate_text = e2e["false_complete_rate"]

            # Status summary after each global finetune iteration.
            print(
                f"[residual_rl] ── STATUS: global_iter={iter_idx}/{int(self._global_finetune_iters)-1}  "
                f"best_progress_rate={best_progress_rate:.4f}  "
                f"true_success={true_success_count}/{denom_eps_int} ({true_success_rate_value:.1%})  "
                f"false_complete={false_complete_count}/{denom_eps_int} ({false_complete_rate_value:.1%})",
                flush=True,
            )

            # Early stop: good true success rate with low false/true ratio.
            should_stop_global = False
            if true_success_rate_value >= float(self._stage_success_fraction):
                denom = max(float(true_success_rate_value), 1e-8)
                ratio_false_to_true = (
                    float(false_complete_rate_value) / denom
                )
                if ratio_false_to_true <= float(self._global_finetune_max_false_to_true_ratio):
                    self._banner(
                        "GLOBAL FINETUNE EARLY STOP ✓",
                        f"true_success={true_success_rate_value:.1%}  "
                        f"false/true={ratio_false_to_true:.3f} <= {self._global_finetune_max_false_to_true_ratio}",
                    )
                    should_stop_global = True


            is_last_iter = iter_idx >= int(self._global_finetune_iters) - 1
            if accepted_this_iter and self._video_feedback and r_best.checkpoint_path and not is_last_iter and not should_stop_global:
                from kuafu.rl.video.export import generate_video_feedback

                video_out_dir = os.path.join(
                    self._run_dirs.root_dir, "videos", f"global_iter{int(iter_idx)}_postbest"
                )
                _raw_rollouts, video_meta = generate_video_feedback(
                    task=self._task,
                    checkpoint_path=str(r_best.checkpoint_path),
                    video_length=self._video_length,
                    num_envs_to_record=self._video_num_envs_to_record,
                    output_dir=video_out_dir,
                    device=self._device,
                    headless=True,
                    seed=int(self._env_seed),
                    execution_robot_id=self._execution_robot_id,
                    all_stages_weights=current_weights_by_stage,
                    all_stages_parameters=current_parameters_by_stage,
                    gemini_enabled=bool(self._gemini_model),
                    gemini_model=self._gemini_model,
                    context_text=_make_gemini_prompt(
                        w_by_stage=current_weights_by_stage,
                        tsr=true_success_rate_text,
                        fcr=false_complete_rate_text,
                        iter_ctx=f"Post-best video after global finetune iteration {iter_idx}.",
                    ),
                )
                video_rollouts = VIDEO_EVIDENCE_PROMPT.format(video_rollouts=_raw_rollouts)
                append_jsonl(
                    self._run_dirs.videos_path,
                    {
                        "timestamp": __import__("datetime").datetime.now().isoformat(),
                        "kind": "global_postbest",
                        "iter": int(iter_idx),
                        "checkpoint_path": str(r_best.checkpoint_path),
                        "video_out_dir": str(video_out_dir),
                        "meta": dict(video_meta),
                    },
                )
                if not bool(video_meta.get("ok")):
                    raise RuntimeError(
                        "[residual_rl] Video feedback required for global post-best rollout but generation failed: "
                        f"{video_meta.get('error') or 'unknown error'}"
                    )
            elif is_last_iter:
                print("[residual_rl] Skipping post-best video on final iteration (no further LLM turn to use it).", flush=True)

            # Record/serialize best global result only when accepted by true-success guard.
            if accepted_this_iter:
                if not r_best.checkpoint_path:
                    raise RuntimeError("[residual_rl] global_finetune: best run has no checkpoint")
                self._best_global_progress_rate = float(best_progress_rate)
                self._maybe_store_best_checkpoint(
                    kind="global",
                    stage_id=None,
                    iter_idx=int(iter_idx),
                    checkpoint_path=r_best.checkpoint_path,
                    progress_rate=float(best_progress_rate),
                    weights=None,
                    parameters=None,
                    all_stages_weights=current_weights_by_stage,
                    all_stages_parameters=current_parameters_by_stage,
                    true_success_stats=candidate_true_stats,
                )

            self._record_trial(
                trial_key=trial_keys[best_run_idx],
                kind="global",
                stage_id=int(stage_count),
                iter_idx=int(iter_idx),
                weights=candidate_all_weights,
                parameters=candidate_all_parameters,
                all_stages_weights=candidate_all_weights,
                all_stages_parameters=candidate_all_parameters,
                progress_rate=float(best_progress_rate),
                log_dir=(
                    r_best.log_dir
                    if r_best.log_dir is not None
                    else (_ for _ in ()).throw(RuntimeError("[residual_rl] global_finetune: best run missing log_dir"))
                ),
                checkpoint_path=r_best.checkpoint_path,
                raw_llm_output=llm_resp.raw_outputs[best_run_idx],
                accepted=accepted_this_iter,
                rejection_reason=None if accepted_this_iter else acceptance_reason,
                true_success_stats=candidate_true_stats,
            )

            self._tb_writer.add_text("Global/raw_llm_output", llm_resp.raw_outputs[best_run_idx], iter_idx)
            self._tb_writer.add_text("Global/run_feedback", fb_best, iter_idx)
            self._tb_writer.add_scalar("Global/progress_rate", float(best_progress_rate), iter_idx)
            candidate_true_success_count = int(candidate_true_stats["true_success_count"])
            candidate_false_complete_count = int(candidate_true_stats["false_complete_count"])
            candidate_timeout_count = int(candidate_true_stats["timeout_count"])
            true_success_rate = float(candidate_true_stats["true_success_rate"])
            false_complete_rate = float(candidate_true_stats["false_complete_rate"])
            self._tb_writer.add_scalar("Global/true_success_rate", true_success_rate, iter_idx)
            self._tb_writer.add_scalar("Global/false_complete_rate", false_complete_rate, iter_idx)
            self._tb_writer.add_scalar("Global/true_success_count", candidate_true_success_count, iter_idx)
            self._tb_writer.add_scalar("Global/false_complete_count", candidate_false_complete_count, iter_idx)
            self._tb_writer.add_scalar("Global/timeout_count", candidate_timeout_count, iter_idx)
            self._tb_writer.add_scalar("Global/accepted", 1.0 if accepted_this_iter else 0.0, iter_idx)
            self._tb_writer.add_scalar(
                "Kuafu/best_true_success_rate_so_far",
                float(self._best_true_success_rate_so_far or 0.0),
                iter_idx,
            )
            self._tb_writer.add_scalar(
                "Kuafu/bc_baseline_true_success_rate",
                float(self._bc_baseline_true_success_rate or 0.0),
                iter_idx,
            )

            # Log to iterations.jsonl (mirrors stage_iter logging).
            append_jsonl(
                self._run_dirs.iterations_path,
                {
                    "timestamp": __import__("datetime").datetime.now().isoformat(),
                    "kind": "global_finetune_iter",
                    "iter": int(iter_idx),
                    "best_progress_rate": float(best_progress_rate),
                    "best_run_idx": int(best_run_idx),
                    "best_worker_idx": int(results[best_run_idx].worker_idx),
                    "num_workers": int(len(results)),
                    "best_weights": current_weights_by_stage,
                    "best_parameters": current_parameters_by_stage,
                    "best_checkpoint": self._last_best_checkpoint_path,
                    "accepted": bool(accepted_this_iter),
                    "rejection_reason": None if accepted_this_iter else acceptance_reason,
                    "true_success_count": candidate_true_success_count,
                    "false_complete_count": candidate_false_complete_count,
                    "timeout_count": candidate_timeout_count,
                    "true_success_rate": true_success_rate,
                    "false_complete_rate": false_complete_rate,
                    "selection_score": float(candidate_true_stats["selection_score"]),
                    "best_true_success_rate_so_far": self._best_true_success_rate_so_far,
                    "bc_baseline_true_success_rate": self._bc_baseline_true_success_rate,
                    "per_worker_progress_rate": [
                        r.progress_metric_eval for r in results
                    ],
                    "per_worker_true_success_rate": [
                        float(self._result_true_success_stats(r)["true_success_rate"]) if r.success else None
                        for r in results
                    ],
                    "per_worker_false_complete_rate": [
                        float(self._result_true_success_stats(r)["false_complete_rate"]) if r.success else None
                        for r in results
                    ],
                },
            )

            assistant_prompt = llm_resp.raw_outputs[best_run_idx]
            if should_stop_global:
                break
        return

    def _tune_individual_stage(
        self,
        *,
        start_stage_id: int = 0,
    ) -> tuple[dict[int, dict[int, float]], dict[int, dict[int, dict[str, object]]]]:

        # Retrieve core task info.
        if self._num_envs is None:
            raise RuntimeError("[residual_rl] num_envs must be set before stage tuning.")
        num_env = int(self._num_envs)
        threshold = num_env * self._stage_success_fraction
        stage_success_fraction = self._stage_success_fraction
        stage_count = self._task_info["stage_count"]
        stages = cast(dict[int, dict[str, Any]], self._task_info["stages"])

        # Keep the latest accepted true-success-safe stage result for the return value.
        # Rejected evo_turn winners are logged but do not roll forward.
        final_progress_rate: float | None = None
        final_weights: dict[int, float] | None = None
        final_ckpt: str | None = None
        final_log_dir: str | None = None
        final_worker_idx: int | None = None
        final_parameters: dict[int, dict[str, object]] | None = None
        final_result: TrainResult | None = None
        stage_assistant_prompt: str | None = None
        # Single-stage tuning sends weight-only proposals; parameters remain fixed.
        single_both_system = SYSTEM_PROMPT + SINGLE_BOTH_TUNING_FORMATTING_INSTRUCTIONS

        # Initialize per-stage weights/parameters from task info (possibly already seeded from a previous run).
        stage_weights: dict[int, dict[int, float]] = {}
        stage_parameters: dict[int, dict[int, dict[str, object]]] = {}
        for sid in range(int(stage_count)):
            stage_meta_init = cast(dict[str, Any], stages[int(sid)])
            base_weights = stage_meta_init["weights"]
            constraints_raw = stage_meta_init["constraints_raw"]
            if "parameters" in stage_meta_init:
                base_parameters = cast(dict[int, dict[str, object]], stage_meta_init["parameters"])
            else:
                base_parameters = extract_tolerances_from_constraints_raw(constraints_raw)
            stage_weights[int(sid)] = {int(k): float(v) for k, v in dict(base_weights).items()}
            stage_parameters[int(sid)] = dict(base_parameters)
            self._iter_best_stage_weights[int(sid)] = dict(stage_weights[int(sid)])
            self._iter_best_stage_parameters[int(sid)] = dict(stage_parameters[int(sid)])

        # Optionally skip stages that are already passed according to the latest stats.
        if start_stage_id < 0 or int(start_stage_id) >= int(stage_count):
            stage_id = 0
        else:
            stage_id = int(start_stage_id)
        iteration_count = 0
        while (iteration_count < self._maximun_proposals and stage_id < stage_count):
            print(
                f"\n[residual_rl] ── STAGE TUNING stage {stage_id}/{stage_count - 1} "
                f"(iter {iteration_count}/{self._maximun_proposals}, "
                f"{self._num_workers} workers) "
                f"pass_threshold={threshold} ──",
                flush=True,
            )

            llm_single = OpenAIChatClient(
                model=self._gpt_model,
                temperature=self._temperature,
                num_suggestions=self._num_workers,
                system_prompt=single_both_system,
            )

            stage_meta = cast(dict[str, Any], stages[stage_id])
            explanations = [strip_tolerance_annotation(value) for value in stage_meta["explanations"]]
            current_weights = stage_meta["weights"]
            constraints_raw = stage_meta["constraints_raw"]
            if "parameters" in stage_meta:
                current_parameters = cast(dict[int, dict[str, object]], stage_meta["parameters"])
            else:
                current_parameters = extract_tolerances_from_constraints_raw(constraints_raw)
            atomic_action = summarize_atomic_action(stage_meta["atomic_action"])

            # Use the best from previous iterations (if any); on first iter these are the task defaults.
            current_weights_jsonable = dict(self._iter_best_stage_weights[int(stage_id)])
            current_parameters = dict(self._iter_best_stage_parameters[int(stage_id)])

            raw_desc = self._task_info["task_description"]
            task_description = (raw_desc or "").strip() 
            constraints_explanations_str = "\n".join(f"- {e}" for e in explanations)

            # Use stats from the previous learned policy (dry-run, or last best run, possibly loaded from a prior session).
            stage_transition_stats = self._last_stage_best_stage_transition_stats
            constraint_met_rate_stats = self._last_stage_best_constraint_met_rate_stats
            stage_user_prompt = ""


            # Rebuild the prompt every evo_turn so current weights reflect the last iteration's best.
            current_weights_str = json.dumps(current_weights_jsonable, indent=2, sort_keys=True)
            detectable_objects_str = json.dumps(
                self._task_info["detectable_objects"], indent=2, sort_keys=True, default=str
            )
            stage_satisfaction_times = format_stage_satisfaction_times(
                stage_transition_stats,
                highest_stage_reached_per_env=self._last_stage_best_highest_stage_reached_per_env,
                stage_id=stage_id,
                num_env=self._num_envs,
            )
            constraint_satisfaction_rates = format_constraint_satisfaction_rates(
                constraint_met_rate_stats, stage_id=stage_id
            )
            reward_values = format_reward_values(constraint_met_rate_stats, stage_id=stage_id)
            stage_user_prompt = SINGLE_BOTH_PROMPT.format(
                task_description=task_description,
                detectable_objects=detectable_objects_str,
                stage_id=stage_id,
                stage_count=stage_count,
                atomic_action=str(atomic_action),
                constraints_explanations=constraints_explanations_str,
                current_weights=current_weights_str,
                end_to_end_feedback=self._true_success_prompt_context(),
                stage_satisfaction_times=stage_satisfaction_times,
                constraint_satisfaction_rates=constraint_satisfaction_rates,
                reward_values=reward_values,
            )

            llm_resp = llm_single.prompt(user_prompt=stage_user_prompt, assistant_prompt=stage_assistant_prompt)

            ts = __import__("datetime").datetime.now().isoformat()
            append_jsonl(
                self._run_dirs.openai_payloads_path,
                {
                    "timestamp": ts,
                    "kind": "stage_weight_tuning",
                    "iter": iteration_count,
                    "stage_id": int(stage_id),
                    "messages": llm_resp.messages,
                    "raw_outputs": llm_resp.raw_outputs,
                    "extra": {"constraints_explanations": explanations},
                },
            )
            append_prompt_transcript(
                path=self._run_dirs.prompt_transcript_path,
                header={"timestamp": ts, "kind": "stage_weight_tuning", "stage_id": int(stage_id), "iter": iteration_count},
                messages=llm_resp.messages,
                raw_outputs=llm_resp.raw_outputs,
            )

            # Parse weight-only proposals. Any extra model fields are ignored.
            parsed_actions = [self._parse_stage_action(o) for o in llm_resp.raw_outputs]
            print(
                f"[residual_rl] Parsed {len(parsed_actions)} proposals (num_workers={self._num_workers})",
                flush=True,
            )
            current_epochs = min(
                self._init_train_epoch + self._epoch_train_increment * self._iteration_count,
                self._max_train_epoch,
            ) 
            cfgs: list[TrainConfig] = []
            stage_trial_keys: list[str] = []
            resume_ckpt, prior_ckpt, warm_start_label = self._get_iteration_warm_start()
            for act in parsed_actions:
                # Conventional stage progress: fraction of envs that advanced past this stage.
                progress_metric_expr = self._stage_pass_rate_expr(stage_id)

                # IMPORTANT: when tuning a later stage, explicitly apply the current "best so far"
                # weights/parameters for all prior stages so worker envs don't drift across iterations.

                proposal_all_weights: dict[int, dict[int, float]] = {
                    int(s): {int(k): float(v) for k, v in dict(w).items()}
                    for s, w in stage_weights.items()
                }
                proposal_all_parameters: dict[int, dict[int, dict[str, object]]] = {
                    int(s): dict(p) for s, p in stage_parameters.items()
                }

                proposal_all_weights[int(stage_id)] = {
                    int(k): float(v) for k, v in dict(act["Weights"]).items()
                }
                trial_key = self._make_trial_key(
                    kind="stage",
                    stage_id=stage_id,
                    weights=act["Weights"],
                    parameters=proposal_all_parameters[int(stage_id)],
                    all_stages_weights=proposal_all_weights,
                    all_stages_parameters=proposal_all_parameters,
                )
                stage_trial_keys.append(trial_key)
                cfgs.append(
                    TrainConfig(
                        stage_id=stage_id,
                        all_stages_weights=proposal_all_weights,
                        all_stages_parameters=proposal_all_parameters,
                        progress_metric_expr=progress_metric_expr,
                        max_train_epoch_override=current_epochs,
                        resume_from_checkpoint=resume_ckpt,
                        prior_checkpoint=prior_ckpt,
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
                        tag=f"stage{stage_id}_iter{iteration_count}",
                    )
                )

            if resume_ckpt or prior_ckpt:
                print(
                    f"[residual_rl] Warm-starting stage {stage_id} iter {iteration_count} via "
                    f"{warm_start_label}: {resume_ckpt or prior_ckpt}",
                    flush=True,
                )
            results = self._pool.train(cfgs)
            self._iteration_count += 1

            best_run_idx, iter_best, iter_best_secondary, run_feedbacks = self._select_best_result(
                results,
                stage_id=stage_id,
            )

            r_best = results[best_run_idx]
            fb_best = run_feedbacks[best_run_idx] if best_run_idx < len(run_feedbacks) else ""

            # Per-worker result summary.
            for run_idx, r in enumerate(results):
                marker = " ★ BEST" if run_idx == best_run_idx else ""
                status = "skip" if r.skipped else ("ok" if r.success else "FAIL")
                progress = r.progress_metric_eval
                secondary = r.reward_metric_eval
                print(
                    f"[residual_rl]   Worker {r.worker_idx} (stage {stage_id}, iter {iteration_count}): "
                    f"status={status} eval_progress_rate={progress} eval_average_episodic_return={secondary}{marker}",
                    flush=True,
                )
            print(
                f"[residual_rl] Stage {stage_id} iter {iteration_count}: eval_best_progress_rate={iter_best} "
                f"eval_best_average_episodic_return={iter_best_secondary} "
                f"(best_worker={results[best_run_idx].worker_idx})",
                flush=True,
            )
            progress_best = float(iter_best)

            # STRICT: log canonical env-readback values only.
            if r_best.applied_stage_weights is None:
                raise RuntimeError(
                    f"[residual_rl] Missing applied_stage_weights for best run (stage_id={stage_id}, iter={iteration_count})."
                )
            if r_best.applied_stage_parameters is None:
                raise RuntimeError(
                    f"[residual_rl] Missing applied_stage_parameters for best run (stage_id={stage_id}, iter={iteration_count})."
                )

            trial_weights = r_best.applied_stage_weights
            trial_parameters = r_best.applied_stage_parameters
            candidate_true_stats = self._result_true_success_stats(r_best)
            accepted_this_iter, acceptance_reason, candidate_true_stats = self._true_success_acceptance_decision(r_best)

            # Candidate values read back from the worker env.
            best_progress_rate = progress_best
            best_weights = {int(k): float(v) for k, v in dict(trial_weights).items()}
            best_parameters = trial_parameters
            best_ckpt = r_best.checkpoint_path
            best_log_dir = r_best.log_dir
            best_worker_idx = int(r_best.worker_idx)
            best_result = r_best

            if accepted_this_iter:
                # Use the accepted true-success winner for the next iteration; discard regressions.
                stage_weights[int(stage_id)] = dict(best_weights)
                stage_parameters[int(stage_id)] = dict(best_parameters)
                self._iter_best_stage_weights[int(stage_id)] = dict(best_weights)
                self._iter_best_stage_parameters[int(stage_id)] = dict(best_parameters)

                # Roll forward the env stats shown in the next prompt.
                self._last_stage_best_stage_transition_stats = r_best.stage_transition_stats
                self._last_stage_best_highest_stage_reached_per_env = r_best.highest_stage_reached_per_env
                self._last_stage_best_constraint_met_rate_stats = r_best.constraint_met_rate_stats
                self._last_stage_best_true_success_count = r_best.true_success_count
                self._last_stage_best_false_complete_count = r_best.false_complete_count
                self._last_stage_best_timeout_count = r_best.timeout_count
                self._last_best_checkpoint_path = r_best.checkpoint_path
                candidate_true_stats = self._update_true_success_incumbent(r_best)
                self._save_prompt_context()

                # Overwrite stage "best" bookkeeping only when true-success guard accepts.
                self._best_stage_progress_rate[int(stage_id)] = float(best_progress_rate)
                self._maybe_store_best_checkpoint(
                    kind="stage",
                    stage_id=stage_id,
                    iter_idx=int(iteration_count),
                    checkpoint_path=best_ckpt,
                    progress_rate=float(best_progress_rate),
                    weights=best_weights,
                    parameters=best_parameters,
                    all_stages_weights=stage_weights,
                    all_stages_parameters=stage_parameters,
                    true_success_stats=candidate_true_stats,
                )
            else:
                print(
                    f"[residual_rl] Stage {stage_id} iter {iteration_count}: candidate rejected by true-success guard: "
                    f"{acceptance_reason}",
                    flush=True,
                )

            self._record_trial(
                trial_key=stage_trial_keys[best_run_idx],
                kind="stage",
                stage_id=stage_id,
                iter_idx=int(iteration_count),
                weights=trial_weights,
                parameters=trial_parameters,
                all_stages_weights=cfgs[best_run_idx].all_stages_weights or stage_weights,
                all_stages_parameters=cfgs[best_run_idx].all_stages_parameters or stage_parameters,
                progress_rate=best_progress_rate,
                log_dir=r_best.log_dir or "",
                checkpoint_path=r_best.checkpoint_path or "",
                raw_llm_output=llm_resp.raw_outputs[best_run_idx],
                accepted=accepted_this_iter,
                rejection_reason=None if accepted_this_iter else acceptance_reason,
                true_success_stats=candidate_true_stats,
            )
            # Cache for future de-duplication (only cache successful runs).
            if r_best.success and r_best.log_dir:
                self._trial_cache[stage_trial_keys[best_run_idx]] = r_best

            # TB bookkeeping
            self._tb_writer.add_text(
                f"Run_{best_run_idx}/raw_llm_output",
                llm_resp.raw_outputs[best_run_idx],
                int(iteration_count),
            )
            self._tb_writer.add_text(
                f"Run_{best_run_idx}/run_feedback",
                fb_best if r_best.success else str(r_best.exception),
                int(iteration_count),
            )
            self._tb_writer.add_scalar(
                f"Run_{best_run_idx}/progress_rate",
                float(best_progress_rate),
                int(iteration_count),
            )
            true_success_count = int(candidate_true_stats["true_success_count"])
            false_complete_count = int(candidate_true_stats["false_complete_count"])
            timeout_count = int(candidate_true_stats["timeout_count"])
            true_success_rate = float(candidate_true_stats["true_success_rate"])
            false_complete_rate = float(candidate_true_stats["false_complete_rate"])
            self._tb_writer.add_scalar(
                f"Run_{best_run_idx}/true_success_rate",
                true_success_rate,
                int(iteration_count),
            )
            self._tb_writer.add_scalar(
                f"Run_{best_run_idx}/false_complete_rate",
                false_complete_rate,
                int(iteration_count),
            )
            self._tb_writer.add_scalar(
                f"Run_{best_run_idx}/true_success_count",
                true_success_count,
                int(iteration_count),
            )
            self._tb_writer.add_scalar(
                f"Run_{best_run_idx}/false_complete_count",
                false_complete_count,
                int(iteration_count),
            )
            self._tb_writer.add_scalar(
                f"Run_{best_run_idx}/timeout_count",
                timeout_count,
                int(iteration_count),
            )
            self._tb_writer.add_scalar(
                f"Run_{best_run_idx}/accepted",
                1.0 if accepted_this_iter else 0.0,
                int(iteration_count),
            )
            self._tb_writer.add_scalar(
                "Kuafu/best_true_success_rate_so_far",
                float(self._best_true_success_rate_so_far or 0.0),
                int(iteration_count),
            )
            self._tb_writer.add_scalar(
                "Kuafu/bc_baseline_true_success_rate",
                float(self._bc_baseline_true_success_rate or 0.0),
                int(iteration_count),
            )

            # iteration log
            append_jsonl(
                self._run_dirs.iterations_path,
                {
                    "timestamp": __import__("datetime").datetime.now().isoformat(),
                    "kind": "stage_iter",
                    "stage_id": int(stage_id),
                    "iter": int(iteration_count),
                    "best_progress_rate": float(best_progress_rate),
                    "iter_best_progress_rate": float(best_progress_rate),
                    "best_run_idx": int(best_run_idx),
                    "best_weights": best_weights,
                    "best_checkpoint": best_ckpt,
                    "best_parameters": best_parameters,
                    "accepted": bool(accepted_this_iter),
                    "rejection_reason": None if accepted_this_iter else acceptance_reason,
                    "true_success_count": true_success_count,
                    "false_complete_count": false_complete_count,
                    "timeout_count": timeout_count,
                    "true_success_rate": true_success_rate,
                    "false_complete_rate": false_complete_rate,
                    "selection_score": float(candidate_true_stats["selection_score"]),
                    "best_true_success_rate_so_far": self._best_true_success_rate_so_far,
                    "bc_baseline_true_success_rate": self._bc_baseline_true_success_rate,
                    "per_worker_true_success_rate": [
                        float(self._result_true_success_stats(r)["true_success_rate"]) if r.success else None
                        for r in results
                    ],
                    "per_worker_false_complete_rate": [
                        float(self._result_true_success_stats(r)["false_complete_rate"]) if r.success else None
                        for r in results
                    ],
                },
            )

            stage_assistant_prompt = llm_resp.raw_outputs[best_run_idx]

            if accepted_this_iter:
                # Save accepted per-iteration winner as the final answer so the function
                # returns the latest true-success-safe stage state.
                final_progress_rate = best_progress_rate
                final_weights = best_weights
                final_ckpt = best_ckpt
                final_log_dir = best_log_dir
                final_worker_idx = best_worker_idx
                final_parameters = best_parameters
                final_result = best_result

            # Check if this stage has passed: ratio of envs that have moved past it >= threshold.
            highest = cast(list[int], self._last_stage_best_highest_stage_reached_per_env)
            pass_count = sum(1 for h in highest if int(h) > int(stage_id))
            total_eps = len(highest)
            true_c = int(self._last_stage_best_true_success_count or 0)
            false_c = int(self._last_stage_best_false_complete_count or 0)
            timeout_c = int(self._last_stage_best_timeout_count or 0)
            denom_eps = int(true_c + false_c + timeout_c)
            if denom_eps <= 0:
                # Fallback: assume one episode per env when termination counts aren't available.
                denom_eps = int(total_eps)
            completion_rate = (true_c + false_c) / float(denom_eps) if denom_eps else 0.0
            completion_cap_hit = completion_rate >= float(self._stage_advance_completion_rate_cap)

            # Status summary after each iteration.
            print(
                f"[residual_rl] ── STATUS: stage={stage_id}/{stage_count-1}  "
                f"iter={iteration_count}/{self._maximun_proposals}  "
                f"pass_count={pass_count}/{int(threshold)}  "
                f"completion={completion_rate:.1%}  "
                f"true_success={true_c}  false_complete={false_c}  "
                f"best_progress_rate={best_progress_rate:.2f}",
                flush=True,
            )

            # If completion is too high, stop tuning entirely (even if the stage isn't passed yet).
            if completion_cap_hit:
                self._banner(
                    "STAGE TUNING STOPPED",
                    f"completion_rate={completion_rate:.1%} >= cap={self._stage_advance_completion_rate_cap}  "
                    f"stage={stage_id}  pass_count={pass_count}",
                )
                # “Golden period”: continue training until the next checkpoint boundary so we
                # don’t quit mid-improvement, and always end with a saved checkpoint.
                if final_progress_rate is not None and accepted_this_iter:
                    self._post_solve_train_until_next_checkpoint_all_workers(
                        stage_id=stage_id,
                        best_progress_rate=float(final_progress_rate),
                        cfgs=cfgs,
                        results=results,
                    )
                stage_id = stage_count
                iteration_count += 1
                continue

            # Stage progression: if the current stage is passed, jump to the first stage that is
            # below the pass-count threshold across ALL stages (including earlier ones that may
            # have regressed), or finish if all are passed.
            if pass_count >= threshold:
                next_stage = stage_count
                for sid in range(0, int(stage_count)):
                    pass_count_sid = sum(1 for h in highest if int(h) > int(sid))
                    if pass_count_sid < threshold:
                        next_stage = int(sid)
                        break

                self._banner(
                    f"STAGE {stage_id} PASSED ✓",
                    f"pass_count={pass_count}>={int(threshold)}  → advancing to stage {next_stage}",
                )
                # “Golden period”: continue training until the next checkpoint boundary so we
                # don’t quit mid-improvement, and always end with a saved checkpoint.
                if final_progress_rate is not None and accepted_this_iter:
                    self._post_solve_train_until_next_checkpoint_all_workers(
                        stage_id=stage_id,
                        best_progress_rate=float(final_progress_rate),
                        cfgs=cfgs,
                        results=results,
                    )

                stage_id = int(next_stage)
                if stage_id >= stage_count:
                    self._banner("ALL STAGES PASSED ✓", "proceeding to global finetune (if enabled)")
                else:
                    pass_count_next = sum(1 for h in highest if int(h) > int(stage_id))
                    print(
                        f"[residual_rl] Next stage {stage_id}: pass_count={pass_count_next}/{int(threshold)} — will tune.",
                        flush=True,
                    )
                # Reset carry-over LLM context so each stage starts clean.
                stage_assistant_prompt = None
                iteration_count += 1
                continue

            print(
                f"[residual_rl] Stage {stage_id} not yet passed (count={pass_count} < {threshold}); tuning again.",
                flush=True,
            )
            iteration_count += 1

        # Persist this stage's best result stats for next stage's initial prompt (from worker_pool results_queue).
        if final_result is not None:
            self._last_stage_best_stage_transition_stats = final_result.stage_transition_stats
            self._last_stage_best_highest_stage_reached_per_env = final_result.highest_stage_reached_per_env
            self._last_stage_best_constraint_met_rate_stats = final_result.constraint_met_rate_stats
            self._last_stage_best_true_success_count = final_result.true_success_count
            self._last_stage_best_false_complete_count = final_result.false_complete_count
            self._last_stage_best_timeout_count = final_result.timeout_count
            self._last_best_checkpoint_path = final_result.checkpoint_path
            self._save_prompt_context()
        _ = final_progress_rate, final_ckpt, final_log_dir, final_worker_idx
        return stage_weights, stage_parameters

    def _post_solve_train_until_next_checkpoint_all_workers(
        self,
        *,
        stage_id: int,
        best_progress_rate: float,
        cfgs: list[TrainConfig],
        results: list[TrainResult],
    ) -> None:
        """After we hit the stage progress threshold, keep training for all runs until next ckpt boundary.

        This avoids “over-designing” a winner-only continuation; all proposals get the same
        chance to benefit from the post-threshold “golden period”.
        """
        if not results:
            raise RuntimeError("[residual_rl] post-solve continuation requested with no worker results.")

        # Determine checkpoint cadence from the task's RL-Games config.
        rl_cfg = (self._task_info.get("rl_games") or {}) if isinstance(self._task_info, dict) else {}
        save_freq = int((rl_cfg.get("save_frequency") or 0))
        if save_freq <= 0:
            return

        # Use the best run by progress to decide the checkpoint boundary to wait for.
        best_idx, _best_val, _best_secondary, _best_feedbacks = self._select_best_result(
            results,
            stage_id=stage_id,
        )
        best_log_dir = results[best_idx].log_dir
        if best_idx is None or best_log_dir is None:
            raise RuntimeError("[residual_rl] post-solve continuation has no best worker log_dir.")

        try:
            data = load_tensorboard_scalars(best_log_dir)
        except Exception as exc:
            raise RuntimeError(
                f"[residual_rl] Failed to load TensorBoard scalars for post-solve continuation: {best_log_dir}"
            ) from exc
        progress_series = data.get("Episode/Kuafu/progress_rate", [])
        if not progress_series:
            raise RuntimeError(
                f"[residual_rl] Missing Episode/Kuafu/progress_rate during post-solve continuation: {best_log_dir}"
            )

        meet_epoch = None
        for i, v in enumerate(progress_series):
            try:
                if float(v) >= float(self._stage_success_fraction):
                    meet_epoch = int(i + 1)
                    break
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"[residual_rl] Invalid Episode/Kuafu/progress_rate value at index {i}: {v!r}"
                ) from exc
        if meet_epoch is None:
            raise RuntimeError(
                "[residual_rl] post-solve continuation was requested, but no epoch met the stage success threshold."
            )

        desired_epoch = int((meet_epoch // save_freq) + 1) * int(save_freq)
        current_epoch = int(len(progress_series))
        if current_epoch >= desired_epoch:
            return

        print(
            f"[residual_rl] Post-solve golden period: wait until next checkpoint boundary. "
            f"met_epoch~{meet_epoch}, current_epoch~{current_epoch}, save_frequency={save_freq} -> target_epoch={desired_epoch}.",
            flush=True,
        )

        # Resume every worker that has a checkpoint; others remain as-is.
        resume_cfgs: list[TrainConfig] = []
        for wi in range(self._num_workers):
            r = next((x for x in results if int(getattr(x, 'worker_idx', -1)) == wi), None)
            base_cfg = cfgs[wi] if wi < len(cfgs) else TrainConfig(
                stage_id=int(stage_id),
                all_stages_weights=None,
                progress_metric_expr=self._stage_pass_rate_expr(stage_id),
                all_stages_parameters=None,
            )
            if r is None or not r.success or not r.checkpoint_path:
                # No checkpoint to resume from; don't do extra training for this worker.
                resume_cfgs.append(base_cfg)
                continue
            resume_cfgs.append(
                TrainConfig(
                    stage_id=base_cfg.stage_id,
                    all_stages_weights=base_cfg.all_stages_weights,
                    all_stages_parameters=base_cfg.all_stages_parameters,
                    progress_metric_expr=self._stage_pass_rate_expr(stage_id),
                    max_train_epoch_override=int(desired_epoch),
                    resume_from_checkpoint=str(r.checkpoint_path),
                    prior_checkpoint=base_cfg.prior_checkpoint,
                    prior_mode=base_cfg.prior_mode,
                    prior_obs_stats_count=base_cfg.prior_obs_stats_count,
                    prior_verbose=base_cfg.prior_verbose,
                    prior_critic_init_mode=base_cfg.prior_critic_init_mode,
                    prior_freeze_actor_epochs=base_cfg.prior_freeze_actor_epochs,
                    prior_sigma_ramp_epochs=base_cfg.prior_sigma_ramp_epochs,
                    prior_sigma_final_val=base_cfg.prior_sigma_final_val,
                    prior_actor_lr_warmup_epochs=base_cfg.prior_actor_lr_warmup_epochs,
                    prior_actor_lr_cap=base_cfg.prior_actor_lr_cap,
                    prior_freeze_input_running_mean_std=base_cfg.prior_freeze_input_running_mean_std,
                    prior_unfreeze_input_running_mean_std_after_freeze=(
                        base_cfg.prior_unfreeze_input_running_mean_std_after_freeze
                    ),
                    prior_rms_unfreeze_after_actor_lr_warmup=base_cfg.prior_rms_unfreeze_after_actor_lr_warmup,
                    prior_lr_schedule_freeze_during_warmup=base_cfg.prior_lr_schedule_freeze_during_warmup,
                    prior_fixed_lr=base_cfg.prior_fixed_lr,
                    prior_bc_anchor_epochs=base_cfg.prior_bc_anchor_epochs,
                    prior_bc_anchor_coef=base_cfg.prior_bc_anchor_coef,
                    prior_bc_dataset_coef=base_cfg.prior_bc_dataset_coef,
                    prior_bc_dataset_decay_epochs=base_cfg.prior_bc_dataset_decay_epochs,
                    prior_bc_dataset_batch_size=base_cfg.prior_bc_dataset_batch_size,
                    prior_bc_dataset_path=base_cfg.prior_bc_dataset_path,
                    skip=False,
                    tag=f"stage{stage_id}_postsolve_to_{desired_epoch}",
                )
            )

        resumed = self._pool.train(resume_cfgs)

        # After the golden period, pick the best checkpoint and store it.
        final_best_idx, _final_best, _final_best_secondary, _resumed_feedbacks = self._select_best_result(
            resumed,
            stage_id=stage_id,
        )
        r = resumed[final_best_idx]
        if not r.checkpoint_path:
            return
        # After golden period, roll forward prompt seed stats + checkpoint (so next stage prompt/video uses it).
        self._last_stage_best_stage_transition_stats = r.stage_transition_stats
        self._last_stage_best_highest_stage_reached_per_env = r.highest_stage_reached_per_env
        self._last_stage_best_constraint_met_rate_stats = r.constraint_met_rate_stats
        self._last_stage_best_true_success_count = r.true_success_count
        self._last_stage_best_false_complete_count = r.false_complete_count
        self._last_stage_best_timeout_count = r.timeout_count
        self._last_best_checkpoint_path = r.checkpoint_path
        self._save_prompt_context()
        # Save the (post-solve) best checkpoint as the stage checkpoint.
        final_weights = {
            int(k): float(v)
            for k, v in getattr(r, "applied_stage_weights").items()
        }
        final_parameters = getattr(r, "applied_stage_parameters")
        self._maybe_store_best_checkpoint(
            kind="stage",
            stage_id=int(stage_id),
            iter_idx=-1,
            checkpoint_path=r.checkpoint_path,
            progress_rate=float(_final_best),
            weights=final_weights,
            parameters=final_parameters,
            all_stages_weights=None,
            all_stages_parameters=None,
        )

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
