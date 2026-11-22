#!/usr/bin/env python3
"""Train a fixed Kuafu program or manual reward without LLM/VLM calls.

This is for ablations where the reward program is already chosen:

- vanilla: use the task's default program and a plain dense state reward
- evolved/tuned: load a downloaded best.json and apply its weights/parameters
- no-program: use the task's manually designed external_rewards() function

The script intentionally bypasses scripts/residual_rl/train.py, initial_constraint_sweep.py,
and all prompt/video feedback loops.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[5]

from kuafu.paths import external_artifact_root, resolve_artifact_path
from kuafu.rl.isaac.task_registry import resolve_artifact_robot_id, resolve_execution_robot_id, task_log_root


def _artifact_path(path: str | None, label: str) -> str | None:
    if path is None or not str(path).strip():
        return None
    return str(resolve_artifact_path(str(path).strip(), label))


def _recipe_or_artifact_path(path: str | None, label: str) -> str | None:
    """Allow immutable project recipes; route generated JSON/YAML externally."""

    if path is None or not str(path).strip():
        return None
    candidate = Path(str(path).strip()).expanduser()
    allowed_source_roots = {"assets", "resources", "src"}
    if not candidate.is_absolute():
        if candidate.parts and candidate.parts[0] in allowed_source_roots:
            candidate = (REPO_ROOT / candidate).resolve()
        else:
            return str(resolve_artifact_path(candidate, label))
    else:
        candidate = candidate.resolve()
    try:
        relative = candidate.relative_to(REPO_ROOT)
    except ValueError:
        return str(resolve_artifact_path(candidate, label))
    if relative.parts and relative.parts[0] in allowed_source_roots:
        return str(candidate)
    return str(resolve_artifact_path(candidate, label))


def _load_weights_params(
    path: str | None,
) -> tuple[dict[int, dict[int, float]] | None, dict[int, dict[int, dict[str, object]]] | None]:
    resolved = _recipe_or_artifact_path(path, "Ablation weights/parameters")
    if resolved is None:
        return None, None
    with open(resolved, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"weights params JSON must be an object: {resolved}")
    stages = data.get("stages", data)
    if not isinstance(stages, dict) or not stages:
        raise ValueError(f"weights params JSON has no stages: {resolved}")

    all_weights: dict[int, dict[int, float]] = {}
    all_parameters: dict[int, dict[int, dict[str, object]]] = {}
    for sid_raw, stage_raw in stages.items():
        sid = int(sid_raw)
        if not isinstance(stage_raw, dict):
            raise TypeError(f"stage {sid} must be an object in {resolved}")
        weights_raw = stage_raw.get("weights") or {}
        params_raw = stage_raw.get("parameters") or {}
        if not isinstance(weights_raw, dict) or not weights_raw:
            raise ValueError(f"stage {sid} has no weights in {resolved}")
        if not isinstance(params_raw, dict):
            raise TypeError(f"stage {sid} parameters must be an object in {resolved}")
        all_weights[sid] = {int(k): float(v) for k, v in dict(weights_raw).items()}
        all_parameters[sid] = {
            int(k): dict(v) for k, v in dict(params_raw).items()
        }
    return all_weights, all_parameters


def _set_reward_flag(env_u: Any, agent_cfg: dict[str, Any], key: str, value: Any) -> None:
    cfg = getattr(env_u, "cfg", None)
    if cfg is not None:
        if hasattr(cfg, key):
            setattr(cfg, key, value)
        mpc_config = getattr(cfg, "mpc_config", None)
        if isinstance(mpc_config, dict):
            mpc_config[key] = value
    mpc = agent_cfg.setdefault("params", {}).setdefault("mpc", {})
    if isinstance(mpc, dict):
        mpc[key] = value


def _apply_reward_preset(env_u: Any, agent_cfg: dict[str, Any], preset: str) -> None:
    if preset == "config":
        return

    dense_overrides = {
        "use_absolute_progress_rewards": False,
        "use_relative_progress_rewards": False,
        "use_velocity_penalty": False,
        "use_time_penalty": False,
        "progress_rewards_scale": 1.0,
        "stage_transition_bonus_scale": 0.0,
        "stage_transition_bonus_use_remaining_horizon": False,
        "penalize_false_completions": False,
        "use_external_rewards": False,
        "skip_dense_reward_computation": False,
        "ignore_stage_gate_for_success": False,
        "external_rewards_scale": 1.0,
    }
    if preset == "dense":
        overrides = dense_overrides
    elif preset == "external":
        overrides = {
            **dense_overrides,
            "use_external_rewards": True,
            "skip_dense_reward_computation": True,
            "ignore_stage_gate_for_success": True,
            "collect_terminal_stuck_constraint_stats": False,
        }
    else:
        raise ValueError(f"unsupported reward preset: {preset}")

    for key, value in overrides.items():
        _set_reward_flag(env_u, agent_cfg, key, value)
    if preset == "external":
        setattr(env_u, "_Kuafu_constraint_stats_stride_steps", 0)


def _apply_residual_observation_mode(
    agent_cfg: dict[str, Any],
    *,
    hide_stage_observation: bool,
) -> None:
    if not hide_stage_observation:
        return
    config = agent_cfg.setdefault("params", {}).setdefault("config", {})
    residual_cfg = config.setdefault("residual_rl", {})
    if not isinstance(residual_cfg, dict):
        raise TypeError("params.config.residual_rl must be a mapping to hide stage observations.")
    residual_cfg["use_stage_conditioning"] = False
    residual_cfg["mask_stage_observation"] = True


def _apply_abla_residual_overrides(
    agent_cfg: dict[str, Any],
    *,
    clip_actions: float | None = None,
    residual_clip_composed_actions: bool | None = None,
    residual_alpha_warmup_epochs: int | None = None,
) -> None:
    params = agent_cfg.setdefault("params", {})
    if clip_actions is not None:
        env_cfg = params.setdefault("env", {})
        if not isinstance(env_cfg, dict):
            raise TypeError("params.env must be a mapping to override clip_actions.")
        env_cfg["clip_actions"] = float(clip_actions)

    if residual_clip_composed_actions is None and residual_alpha_warmup_epochs is None:
        return
    config = params.setdefault("config", {})
    residual_cfg = config.setdefault("residual_rl", {})
    if not isinstance(residual_cfg, dict):
        raise TypeError("params.config.residual_rl must be a mapping to override residual settings.")
    if residual_clip_composed_actions is not None:
        residual_cfg["clip_composed_actions"] = bool(residual_clip_composed_actions)
    if residual_alpha_warmup_epochs is not None:
        residual_cfg["alpha_warmup_epochs"] = int(residual_alpha_warmup_epochs)


def _default_progress_expr() -> str:
    return (
        "torch.mean((self._highest_stage_per_env >= "
        "self.program_runtime.compiled_program.stage_count()).float())"
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train fixed Kuafu reward weights/parameters with RL-Games only."
    )
    p.add_argument("--task", required=True)
    p.add_argument(
        "--execution_robot_id",
        default=None,
        help="Robot env package to execute. Defaults to inference from --task; legacy tasks use fr3_finray.",
    )
    p.add_argument(
        "--artifact_robot_id",
        default=None,
        help="Robot id for artifact lineage metadata. Defaults to fr3_finray for existing priors/CSVs.",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num_envs", type=int, default=8192)
    p.add_argument("--env_seed", type=int, default=42)
    p.add_argument("--worker_idx", type=int, default=0)
    p.add_argument("--max_train_epoch", type=int, required=True)
    p.add_argument("--weights_params_json", default="")
    p.add_argument("--agent_cfg", default="")
    p.add_argument("--reward_preset", choices=["config", "dense", "external"], default="config")
    p.add_argument("--tag", default="")
    p.add_argument("--run_root", default="")
    p.add_argument("--render", action="store_true", default=False)
    p.add_argument("--eval_after_train", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--progress_metric_expr", default=_default_progress_expr())

    p.add_argument("--clip_actions", type=float, default=None)
    p.add_argument(
        "--residual_clip_composed_actions",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override residual_rl.clip_composed_actions for this ablation run.",
    )
    p.add_argument("--residual_alpha_warmup_epochs", type=int, default=None)

    p.add_argument("--prior_checkpoint", default="")
    p.add_argument("--prior_mode", choices=["full_actor", "residual_rl"], default="full_actor")
    p.add_argument(
        "--residual_hide_stage_observation",
        action="store_true",
        default=False,
        help="Hide scalar stage observations and disable appended stage one-hot for residual RL.",
    )
    p.add_argument("--prior_obs_stats_count", type=float, default=15000000.0)
    p.add_argument(
        "--prior_critic_init_mode",
        choices=[
            "actor_only_random_critic",
            "actor_and_critic_prior",
            "actor_only_zero_critic",
        ],
        default="actor_only_random_critic",
    )
    p.add_argument("--prior_actor_lr_cap", type=float, default=0.0)
    p.add_argument("--prior_bc_anchor_coef", type=float, default=0.0)
    p.add_argument("--prior_bc_dataset_coef", type=float, default=0.0)
    p.add_argument("--prior_bc_dataset_batch_size", type=int, default=1024)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    execution_robot_id = resolve_execution_robot_id(args.task, args.execution_robot_id)
    artifact_robot_id = resolve_artifact_robot_id(args.artifact_robot_id)
    args.execution_robot_id = execution_robot_id
    args.artifact_robot_id = artifact_robot_id
    weights_params_path = _recipe_or_artifact_path(args.weights_params_json, "Ablation weights/parameters")
    all_weights, all_parameters = _load_weights_params(args.weights_params_json)
    agent_cfg_path = _recipe_or_artifact_path(args.agent_cfg, "Ablation agent configuration")
    prior_checkpoint = _artifact_path(args.prior_checkpoint, "Ablation BC prior checkpoint")

    run_label = args.tag or f"{args.task}_{args.reward_preset}"
    if args.run_root:
        run_root = resolve_artifact_path(args.run_root, "Ablation run")
    else:
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        task_root = Path(
            task_log_root(
                str(external_artifact_root() / "residual_rl" / "ablation"),
                args.task,
                execution_robot_id,
            )
        )
        run_root = task_root / f"{stamp}_{run_label}"
    run_root.mkdir(parents=True, exist_ok=True)

    from kuafu.rl.isaac.env_factory import create_env_and_app
    from kuafu.rl.isaac.train_rl_games import evaluate_rl_games, train_rl_games
    from kuafu.rl.isaac.update_utilities import apply_updates_and_wrap_reset

    simulation_app = None
    env = None
    try:
        env_seed = int(args.env_seed) + int(args.worker_idx)
        simulation_app, env, agent_cfg = create_env_and_app(
            task=args.task,
            device=args.device,
            env_seed=env_seed,
            num_envs=args.num_envs,
            headless=not args.render,
            execution_robot_id=execution_robot_id,
            initial_prior_checkpoint=prior_checkpoint,
            agent_cfg_path=agent_cfg_path,
        )
        env_u = env.unwrapped
        _apply_reward_preset(env_u, agent_cfg, args.reward_preset)
        _apply_residual_observation_mode(
            agent_cfg,
            hide_stage_observation=bool(args.residual_hide_stage_observation),
        )
        _apply_abla_residual_overrides(
            agent_cfg,
            clip_actions=args.clip_actions,
            residual_clip_composed_actions=args.residual_clip_composed_actions,
            residual_alpha_warmup_epochs=args.residual_alpha_warmup_epochs,
        )
        apply_updates_and_wrap_reset(
            env,
            stage_id=0,
            all_stages_weights=all_weights,
            all_stages_parameters=all_parameters,
            progress_metric_expr=str(args.progress_metric_expr),
        )

        print("[Abla] Fixed-program training")
        print(f"[Abla] task={args.task}")
        print(f"[Abla] execution_robot_id={execution_robot_id}")
        print(f"[Abla] artifact_robot_id={artifact_robot_id}")
        print(f"[Abla] reward_preset={args.reward_preset}")
        print(f"[Abla] agent_cfg={agent_cfg_path or '<registry>'}")
        print(f"[Abla] weights_params_json={weights_params_path or '<default program>'}")
        print(f"[Abla] prior_checkpoint={prior_checkpoint or '<none>'}")
        print(f"[Abla] prior_mode={args.prior_mode if prior_checkpoint else '<none>'}")
        print(f"[Abla] residual_hide_stage_observation={bool(args.residual_hide_stage_observation)}")
        print(f"[Abla] clip_actions={args.clip_actions if args.clip_actions is not None else '<config>'}")
        print(
            f"[Abla] residual_clip_composed_actions="
            f"{args.residual_clip_composed_actions if args.residual_clip_composed_actions is not None else '<config>'}"
        )
        print(
            f"[Abla] residual_alpha_warmup_epochs="
            f"{args.residual_alpha_warmup_epochs if args.residual_alpha_warmup_epochs is not None else '<config>'}"
        )
        print(f"[Abla] max_train_epoch={args.max_train_epoch}")
        print(f"[Abla] env_seed={env_seed} worker_idx={args.worker_idx}")
        print(f"[Abla] run_root={run_root}")

        log_dir, checkpoint_path = train_rl_games(
            env=env,
            task=args.task,
            device=args.device,
            max_train_epoch=int(args.max_train_epoch),
            worker_idx=int(args.worker_idx),
            seed=env_seed,
            prior_checkpoint=prior_checkpoint,
            prior_mode=args.prior_mode,
            prior_obs_stats_count=float(args.prior_obs_stats_count),
            prior_critic_init_mode=args.prior_critic_init_mode,
            prior_actor_lr_cap=float(args.prior_actor_lr_cap),
            prior_bc_anchor_coef=float(args.prior_bc_anchor_coef),
            prior_bc_dataset_coef=float(args.prior_bc_dataset_coef),
            prior_bc_dataset_batch_size=int(args.prior_bc_dataset_batch_size),
            kuafu_run_dir=str(run_root),
            agent_cfg=agent_cfg,
        )
        print(f"[Abla] train_log_dir={log_dir}")
        print(f"[Abla] checkpoint={checkpoint_path}")
        if not checkpoint_path or not Path(checkpoint_path).is_file():
            raise RuntimeError("Fixed-program training returned without a saved checkpoint.")

        eval_payload = None
        if args.eval_after_train and checkpoint_path:
            eval_payload = evaluate_rl_games(
                env=env,
                task=args.task,
                device=args.device,
                checkpoint_path=str(checkpoint_path),
                prior_checkpoint=prior_checkpoint,
                prior_mode=args.prior_mode,
                agent_cfg=agent_cfg,
                progress_metric_expr=str(args.progress_metric_expr),
                seed=env_seed,
            )
            print("[Abla] eval=" + json.dumps(eval_payload, sort_keys=True, default=str))
        (run_root / "completion.json").write_text(
            json.dumps({
                "task": args.task,
                "checkpoint": str(checkpoint_path),
                "prior_checkpoint": prior_checkpoint,
                "seed": env_seed,
                "max_train_epoch": int(args.max_train_epoch),
                "evaluation": eval_payload,
            }, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
    except BaseException:
        # Kit's fast shutdown exits the process from close(), before Python
        # can print a pending exception. Report it and set Kit's exit status
        # before the finally block releases the simulator.
        traceback.print_exc()
        sys.stderr.flush()
        if simulation_app is not None:
            import omni.kit.app

            omni.kit.app.get_app().post_quit(1)
        raise
    finally:
        if env is not None:
            env.close()
        if simulation_app is not None:
            simulation_app.close()


if __name__ == "__main__":
    main()
