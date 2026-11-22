"""CLI wiring for the separate residual-RL hyperparameter search entrypoint."""

from __future__ import annotations

import sys

from kuafu.rl.isaac.task_registry import resolve_artifact_robot_id, resolve_execution_robot_id


def _none_if_empty(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    val = float(value)  # type: ignore[arg-type]
    return val if val > 0.0 else None


def run_from_cli(args) -> None:
    num_workers = int(args.num_workers)
    if num_workers <= 0:
        raise ValueError("num_workers must be >= 1.")
    residual_hparam_iters = int(getattr(args, "residual_hparam_iters", 0))
    if residual_hparam_iters < 0:
        raise ValueError("--residual_hparam_iters must be >= 0.")
    # An external agent can run one measured training trial with supplied
    # hyperparameters, then decide its next revision from the local artifacts.
    if residual_hparam_iters == 0 and bool(args.video_feedback):
        raise ValueError("Use --no-video_feedback with zero model iterations; inspect videos in your agent.")

    initial_eval_checkpoint = _none_if_empty(getattr(args, "initial_eval_checkpoint", None))
    initial_checkpoint = _none_if_empty(getattr(args, "initial_checkpoint", None))
    initial_weights_params = _none_if_empty(getattr(args, "initial_weights_params", None))
    if initial_weights_params is None:
        raise ValueError(
            "train_residual_hparams.py requires --initial_weights_params from the post-BC parameter/weight query."
        )
    initial_agent_hyperparameters = _none_if_empty(getattr(args, "initial_agent_hyperparameters", None))
    initial_prior_checkpoint = _none_if_empty(getattr(args, "initial_prior_checkpoint", None))
    if initial_prior_checkpoint is None:
        raise ValueError("train_residual_hparams.py requires --initial_prior_checkpoint for residual_rl mode.")

    legacy_eval_only_flag = bool(getattr(args, "resume_from_prev_iter_checkpoint", False))
    if legacy_eval_only_flag:
        if initial_eval_checkpoint:
            raise ValueError(
                "Use either --initial_eval_checkpoint or the legacy --resume_from_prev_iter_checkpoint flow, not both."
            )
        if initial_checkpoint is None:
            raise ValueError("--resume_from_prev_iter_checkpoint requires --initial_checkpoint.")
        initial_eval_checkpoint = initial_checkpoint
        initial_checkpoint = None

    if residual_hparam_iters == 0 and initial_eval_checkpoint is not None:
        raise ValueError("Zero-iteration training requires a training trial, not an evaluation-only checkpoint.")

    restart_each_iter_from_initial_warm_start = bool(
        getattr(args, "restart_each_iter_from_initial_warm_start", False)
    ) or bool(getattr(args, "restart_each_iter_from_initial_prior_checkpoint", False))
    execution_robot_id = resolve_execution_robot_id(
        str(args.task), getattr(args, "execution_robot_id", None)
    )
    artifact_robot_id = resolve_artifact_robot_id(getattr(args, "artifact_robot_id", None))

    from kuafu.rl.core.runner_residual_hparams import ResidualHparamRunner

    launch_script = _none_if_empty(getattr(args, "launch_script", None))

    runner = ResidualHparamRunner(
        task=str(args.task),
        device=str(args.device),
        num_workers=int(num_workers),
        num_envs=None if args.num_envs is None else int(args.num_envs),
        env_seed=int(args.env_seed),
        max_train_epoch=int(args.max_train_epoch),
        init_train_epoch=int(getattr(args, "init_train_epoch", 50)),
        epoch_train_increment=int(getattr(args, "epoch_train_increment", 5)),
        feedback_subsampling=int(args.feedback_subsampling),
        residual_hparam_iters=residual_hparam_iters,
        gpt_model=str(args.gpt_model),
        temperature=float(args.temperature),
        headless=not bool(args.render),
        video_feedback=bool(args.video_feedback),
        video_length=int(args.video_length),
        video_num_envs_to_record=int(args.video_num_envs_to_record),
        gemini_model=str(getattr(args, "gemini_model", "gemini-2.0-flash")),
        initial_eval_checkpoint=initial_eval_checkpoint,
        initial_checkpoint=initial_checkpoint,
        initial_weights_params=initial_weights_params,
        initial_agent_hyperparameters=initial_agent_hyperparameters,
        restart_each_iter_from_initial_warm_start=restart_each_iter_from_initial_warm_start,
        initial_prior_checkpoint=initial_prior_checkpoint,
        initial_prior_mode=str(getattr(args, "initial_prior_mode", "residual_rl")),
        initial_prior_obs_stats_count=float(getattr(args, "initial_prior_obs_stats_count", 1e6)),
        initial_prior_verbose=bool(getattr(args, "initial_prior_verbose", False)),
        initial_prior_critic_init_mode=str(
            getattr(args, "initial_prior_critic_init_mode", "actor_only_random_critic")
        ),
        initial_prior_freeze_actor_epochs=int(getattr(args, "initial_prior_freeze_actor_epochs", 0)),
        initial_prior_sigma_ramp_epochs=int(getattr(args, "initial_prior_sigma_ramp_epochs", 0)),
        initial_prior_sigma_final_val=(
            None
            if getattr(args, "initial_prior_sigma_final_val", None) in (None, "")
            else float(getattr(args, "initial_prior_sigma_final_val"))
        ),
        initial_prior_actor_lr_warmup_epochs=int(getattr(args, "initial_prior_actor_lr_warmup_epochs", 0)),
        initial_prior_actor_lr_cap=float(getattr(args, "initial_prior_actor_lr_cap", 0.0) or 0.0),
        initial_prior_freeze_input_running_mean_std=bool(
            getattr(args, "initial_prior_freeze_input_running_mean_std", False)
        ),
        initial_prior_unfreeze_input_running_mean_std_after_freeze=bool(
            getattr(args, "initial_prior_unfreeze_input_running_mean_std_after_freeze", False)
        ),
        initial_prior_rms_unfreeze_after_actor_lr_warmup=bool(
            getattr(args, "initial_prior_rms_unfreeze_after_actor_lr_warmup", False)
        ),
        initial_prior_lr_schedule_freeze_during_warmup=bool(
            getattr(args, "initial_prior_lr_schedule_freeze_during_warmup", False)
        ),
        initial_prior_fixed_lr=_optional_float(getattr(args, "initial_prior_fixed_lr", None)),
        initial_prior_bc_anchor_epochs=int(getattr(args, "initial_prior_bc_anchor_epochs", 0)),
        initial_prior_bc_anchor_coef=float(getattr(args, "initial_prior_bc_anchor_coef", 0.0) or 0.0),
        initial_prior_bc_dataset_coef=float(getattr(args, "initial_prior_bc_dataset_coef", 0.0) or 0.0),
        initial_prior_bc_dataset_decay_epochs=int(getattr(args, "initial_prior_bc_dataset_decay_epochs", 0) or 0),
        initial_prior_bc_dataset_batch_size=int(getattr(args, "initial_prior_bc_dataset_batch_size", 1024) or 1024),
        initial_prior_bc_dataset_path=_none_if_empty(getattr(args, "initial_prior_bc_dataset_path", None)),
        initial_prior_bc_action_reward_coef=float(
            getattr(args, "initial_prior_bc_action_reward_coef", 0.0) or 0.0
        ),
        stop_on_error=bool(getattr(args, "stop_on_error", False)),
        argv=sys.argv,
        launch_script_path=launch_script,
        execution_robot_id=execution_robot_id,
        artifact_robot_id=artifact_robot_id,
    )
    runner.run()
