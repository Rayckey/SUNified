"""Kuafu CLI implementation.

This module intentionally keeps imports light; Isaac Sim is launched inside worker processes.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
import math

from kuafu.rl.isaac.task_registry import (
    resolve_artifact_robot_id,
    resolve_execution_robot_id,
)


@dataclass(frozen=True)
class ResidualCliConfig:
    task: str
    device: str
    num_workers: int
    num_envs: int | None
    env_seed: int
    max_train_epoch: int
    init_train_epoch: int
    epoch_train_increment: int
    feedback_subsampling: int
    stage_success_fraction: float
    maximun_proposals: int
    global_finetune_iters: int
    global_finetune_max_false_to_true_ratio: float
    stage_advance_completion_rate_cap: float
    gpt_model: str
    temperature: float
    render: bool
    video_feedback: bool
    video_length: int
    video_num_envs_to_record: int
    gemini_model: str = "gemini-2.0-flash"
    dry_run: bool = True
    resume_run_dir: str | None = None
    resume_checkpoint: str | None = None
    initial_eval_checkpoint: str | None = None
    initial_checkpoint: str | None = None
    initial_weights_params: str | None = None
    restart_each_iter_from_initial_warm_start: bool = False
    initial_prior_checkpoint: str | None = None
    initial_prior_mode: str = "full_actor"
    initial_prior_obs_stats_count: float = 1e6
    initial_prior_verbose: bool = False
    initial_prior_critic_init_mode: str = "actor_only_random_critic"
    initial_prior_freeze_actor_epochs: int = 0
    initial_prior_sigma_ramp_epochs: int = 0
    initial_prior_sigma_final_val: float | None = None
    initial_prior_actor_lr_warmup_epochs: int = 0
    initial_prior_actor_lr_cap: float = 0.0
    initial_prior_freeze_input_running_mean_std: bool = False
    initial_prior_unfreeze_input_running_mean_std_after_freeze: bool = False
    initial_prior_rms_unfreeze_after_actor_lr_warmup: bool = False
    initial_prior_lr_schedule_freeze_during_warmup: bool = False
    initial_prior_fixed_lr: float | None = None
    initial_prior_bc_anchor_epochs: int = 0
    initial_prior_bc_anchor_coef: float = 0.0
    initial_prior_bc_dataset_coef: float = 0.0
    initial_prior_bc_dataset_decay_epochs: int = 0
    initial_prior_bc_dataset_batch_size: int = 1024
    initial_prior_bc_dataset_path: str | None = None
    initial_prior_bc_action_reward_coef: float = 0.0
    stop_on_error: bool = False
    execution_robot_id: str | None = None
    artifact_robot_id: str | None = None


def run_from_cli(args) -> None:
    num_workers = int(args.num_workers)
    if num_workers <= 0:
        raise ValueError("num_workers must be >= 1.")

    # Stage budget is specified as a proposal cap, and converted to stage iterations.
    # total_proposals ~= num_workers * max_stage_iters
    maximun_proposals = int(getattr(args, "maximun_proposals"))


    print(
        f"[residual_rl] Stage budget: maximun_proposals={maximun_proposals}, num_workers={num_workers})",
        flush=True,
    )

    initial_eval_checkpoint = getattr(args, "initial_eval_checkpoint", None) or None
    if initial_eval_checkpoint:
        initial_eval_checkpoint = str(initial_eval_checkpoint).strip() or None
    initial_checkpoint = getattr(args, "initial_checkpoint", None) or None
    if initial_checkpoint:
        initial_checkpoint = str(initial_checkpoint).strip() or None
    initial_weights_params = getattr(args, "initial_weights_params", None) or None
    if initial_weights_params:
        initial_weights_params = str(initial_weights_params).strip() or None
    initial_prior_checkpoint = getattr(args, "initial_prior_checkpoint", None) or None
    if initial_prior_checkpoint:
        initial_prior_checkpoint = str(initial_prior_checkpoint).strip() or None
    legacy_eval_only_flag = bool(getattr(args, "resume_from_prev_iter_checkpoint", False))
    if legacy_eval_only_flag:
        if initial_eval_checkpoint:
            raise ValueError(
                "Use either --initial_eval_checkpoint or the legacy "
                "--resume_from_prev_iter_checkpoint flow, not both."
            )
        if initial_checkpoint is None:
            raise ValueError(
                "--resume_from_prev_iter_checkpoint requires --initial_checkpoint."
            )
        print(
            "[residual_rl] WARNING: --resume_from_prev_iter_checkpoint is deprecated; "
            "use --initial_eval_checkpoint instead.",
            flush=True,
        )
        initial_eval_checkpoint = initial_checkpoint
        initial_checkpoint = None
    restart_each_iter_from_initial_warm_start = bool(
        getattr(args, "restart_each_iter_from_initial_warm_start", False)
    ) or bool(getattr(args, "restart_each_iter_from_initial_prior_checkpoint", False))
    execution_robot_id = resolve_execution_robot_id(
        str(args.task), getattr(args, "execution_robot_id", None)
    )
    artifact_robot_id = resolve_artifact_robot_id(getattr(args, "artifact_robot_id", None))

    cfg = ResidualCliConfig(
        task=str(args.task),
        device=str(args.device),
        num_workers=int(num_workers),
        num_envs=None if args.num_envs is None else int(args.num_envs),
        env_seed=int(args.env_seed),
        max_train_epoch=int(args.max_train_epoch),
        init_train_epoch=int(getattr(args, "init_train_epoch", 50)),
        epoch_train_increment=int(getattr(args, "epoch_train_increment", 5)),
        feedback_subsampling=int(args.feedback_subsampling),
        stage_success_fraction=float(args.stage_success_fraction),
        maximun_proposals=int(maximun_proposals),
        global_finetune_iters=int(args.global_finetune_iters),
        global_finetune_max_false_to_true_ratio=float(
            getattr(args, "global_finetune_max_false_to_true_ratio")
        ),
        stage_advance_completion_rate_cap=float(
            getattr(args, "stage_advance_completion_rate_cap")
        ),
        gpt_model=str(args.gpt_model),
        temperature=float(args.temperature),
        render=bool(args.render),
        video_feedback=bool(args.video_feedback),
        video_length=int(args.video_length),
        video_num_envs_to_record=int(args.video_num_envs_to_record),
        gemini_model=str(getattr(args, "gemini_model", "gemini-2.0-flash")),
        initial_eval_checkpoint=initial_eval_checkpoint,
        initial_checkpoint=initial_checkpoint,
        initial_weights_params=initial_weights_params,
        restart_each_iter_from_initial_warm_start=restart_each_iter_from_initial_warm_start,
        initial_prior_checkpoint=initial_prior_checkpoint,
        initial_prior_mode=str(getattr(args, "initial_prior_mode", "full_actor")),
        initial_prior_obs_stats_count=float(getattr(args, "initial_prior_obs_stats_count", 1e6)),
        initial_prior_verbose=bool(getattr(args, "initial_prior_verbose", False)),
        initial_prior_critic_init_mode=str(
            getattr(args, "initial_prior_critic_init_mode", "actor_only_random_critic")
        ),
        initial_prior_freeze_actor_epochs=int(
            getattr(args, "initial_prior_freeze_actor_epochs", 0)
        ),
        initial_prior_sigma_ramp_epochs=int(
            getattr(args, "initial_prior_sigma_ramp_epochs", 0)
        ),
        initial_prior_sigma_final_val=(
            None
            if getattr(args, "initial_prior_sigma_final_val", None) in (None, "")
            else float(getattr(args, "initial_prior_sigma_final_val"))
        ),
        initial_prior_actor_lr_warmup_epochs=int(
            getattr(args, "initial_prior_actor_lr_warmup_epochs", 0)
        ),
        initial_prior_actor_lr_cap=float(
            getattr(args, "initial_prior_actor_lr_cap", 0.0) or 0.0
        ),
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
        initial_prior_fixed_lr=(
            None
            if getattr(args, "initial_prior_fixed_lr", None) in (None, "")
            else float(getattr(args, "initial_prior_fixed_lr"))
        ),
        initial_prior_bc_anchor_epochs=int(
            getattr(args, "initial_prior_bc_anchor_epochs", 0)
        ),
        initial_prior_bc_anchor_coef=float(
            getattr(args, "initial_prior_bc_anchor_coef", 0.0) or 0.0
        ),
        initial_prior_bc_dataset_coef=float(
            getattr(args, "initial_prior_bc_dataset_coef", 0.0) or 0.0
        ),
        initial_prior_bc_dataset_decay_epochs=int(
            getattr(args, "initial_prior_bc_dataset_decay_epochs", 0) or 0
        ),
        initial_prior_bc_dataset_batch_size=int(
            getattr(args, "initial_prior_bc_dataset_batch_size", 1024) or 1024
        ),
        initial_prior_bc_dataset_path=(
            str(getattr(args, "initial_prior_bc_dataset_path", "") or "") or None
        ),
        initial_prior_bc_action_reward_coef=float(
            getattr(args, "initial_prior_bc_action_reward_coef", 0.0) or 0.0
        ),
        stop_on_error=bool(getattr(args, "stop_on_error", False)),
        execution_robot_id=execution_robot_id,
        artifact_robot_id=artifact_robot_id,
    )

    # Late imports so `Kuafu` can be imported without Isaac Sim.
    from kuafu.rl.core.runner import ResidualRunner

    launch_script = getattr(args, "launch_script", None)
    if launch_script:
        launch_script = str(launch_script).strip() or None

    runner = ResidualRunner(
        task=cfg.task,
        device=cfg.device,
        num_workers=cfg.num_workers,
        num_envs=cfg.num_envs,
        env_seed=cfg.env_seed,
        max_train_epoch=cfg.max_train_epoch,
        init_train_epoch=cfg.init_train_epoch,
        epoch_train_increment=cfg.epoch_train_increment,
        feedback_subsampling=cfg.feedback_subsampling,
        stage_success_fraction=cfg.stage_success_fraction,
        maximun_proposals=cfg.maximun_proposals,
        global_finetune_iters=cfg.global_finetune_iters,
        global_finetune_max_false_to_true_ratio=cfg.global_finetune_max_false_to_true_ratio,
        stage_advance_completion_rate_cap=cfg.stage_advance_completion_rate_cap,
        gpt_model=cfg.gpt_model,
        temperature=cfg.temperature,
        headless=not cfg.render,
        video_feedback=cfg.video_feedback,
        video_length=cfg.video_length,
        video_num_envs_to_record=cfg.video_num_envs_to_record,
        gemini_model=cfg.gemini_model,
        initial_eval_checkpoint=cfg.initial_eval_checkpoint,
        initial_checkpoint=cfg.initial_checkpoint,
        initial_weights_params=cfg.initial_weights_params,
        restart_each_iter_from_initial_warm_start=cfg.restart_each_iter_from_initial_warm_start,
        initial_prior_checkpoint=cfg.initial_prior_checkpoint,
        initial_prior_mode=cfg.initial_prior_mode,
        initial_prior_obs_stats_count=cfg.initial_prior_obs_stats_count,
        initial_prior_verbose=cfg.initial_prior_verbose,
        initial_prior_critic_init_mode=cfg.initial_prior_critic_init_mode,
        initial_prior_freeze_actor_epochs=cfg.initial_prior_freeze_actor_epochs,
        initial_prior_sigma_ramp_epochs=cfg.initial_prior_sigma_ramp_epochs,
        initial_prior_sigma_final_val=cfg.initial_prior_sigma_final_val,
        initial_prior_actor_lr_warmup_epochs=cfg.initial_prior_actor_lr_warmup_epochs,
        initial_prior_actor_lr_cap=cfg.initial_prior_actor_lr_cap,
        initial_prior_freeze_input_running_mean_std=cfg.initial_prior_freeze_input_running_mean_std,
        initial_prior_unfreeze_input_running_mean_std_after_freeze=cfg.initial_prior_unfreeze_input_running_mean_std_after_freeze,
        initial_prior_rms_unfreeze_after_actor_lr_warmup=cfg.initial_prior_rms_unfreeze_after_actor_lr_warmup,
        initial_prior_lr_schedule_freeze_during_warmup=cfg.initial_prior_lr_schedule_freeze_during_warmup,
        initial_prior_fixed_lr=cfg.initial_prior_fixed_lr,
        initial_prior_bc_anchor_epochs=cfg.initial_prior_bc_anchor_epochs,
        initial_prior_bc_anchor_coef=cfg.initial_prior_bc_anchor_coef,
        initial_prior_bc_dataset_coef=cfg.initial_prior_bc_dataset_coef,
        initial_prior_bc_dataset_decay_epochs=cfg.initial_prior_bc_dataset_decay_epochs,
        initial_prior_bc_dataset_batch_size=cfg.initial_prior_bc_dataset_batch_size,
        initial_prior_bc_dataset_path=cfg.initial_prior_bc_dataset_path,
        initial_prior_bc_action_reward_coef=cfg.initial_prior_bc_action_reward_coef,
        stop_on_error=cfg.stop_on_error,
        argv=sys.argv,
        launch_script_path=launch_script,
        execution_robot_id=cfg.execution_robot_id,
        artifact_robot_id=cfg.artifact_robot_id,
    )
    runner.run()
