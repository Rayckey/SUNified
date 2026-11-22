#!/usr/bin/env python3
"""Evaluate a fixed ablation checkpoint without LLM/VLM calls."""

from __future__ import annotations

import argparse
import copy
import glob
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_fixed_program import (
    _apply_reward_preset,
    _apply_abla_residual_overrides,
    _apply_residual_observation_mode,
    _default_progress_expr,
    _artifact_path,
    _load_weights_params,
    _recipe_or_artifact_path,
)
from kuafu.paths import external_artifact_root, resolve_artifact_path
from kuafu.rl.isaac.task_registry import resolve_artifact_robot_id, resolve_execution_robot_id, task_log_root


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except Exception:
            pass
    return str(obj)


def _checkpoint_epoch(path: Path) -> int | None:
    match = re.search(r"_ep_(\d+)", path.name)
    if match:
        return int(match.group(1))
    return None


def _checkpoint_reward(path: Path) -> float | None:
    match = re.search(r"_rew_+(-?\d+(?:\.\d+)?)_?\.pth$", path.name)
    if match:
        return float(match.group(1))
    return None


def _checkpoint_run_dir(path: Path) -> Path:
    if path.parent.name == "nn":
        return path.parent.parent
    return path.parent


def _load_scalar_values_by_step(run_dir: Path, tag: str) -> dict[int, float]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception as exc:
        raise RuntimeError(
            "TensorBoard is required for checkpoint_selector=best_window_success_after_epoch"
        ) from exc

    event_paths = sorted((run_dir / "summaries").glob("events.out.tfevents.*"))
    if not event_paths:
        raise FileNotFoundError(f"no TensorBoard event files found under {run_dir / 'summaries'}")

    values: dict[int, float] = {}
    for event_path in event_paths:
        accumulator = EventAccumulator(str(event_path), size_guidance={"scalars": 0})
        accumulator.Reload()
        if tag not in accumulator.Tags().get("scalars", []):
            continue
        for scalar in accumulator.Scalars(tag):
            values[int(scalar.step)] = float(scalar.value)
    if not values:
        raise FileNotFoundError(f"scalar tag {tag!r} not found under {run_dir / 'summaries'}")
    return values


def _metric_at_or_before_epoch(values_by_step: dict[int, float], epoch: int) -> tuple[int, float] | None:
    if epoch in values_by_step:
        return epoch, values_by_step[epoch]
    earlier_steps = [step for step in values_by_step if step <= epoch]
    if not earlier_steps:
        return None
    step = max(earlier_steps)
    return step, values_by_step[step]


def _resolve_checkpoint(args: argparse.Namespace) -> Path:
    if args.checkpoint:
        checkpoint = resolve_artifact_path(args.checkpoint, "Ablation checkpoint")
        if not checkpoint.is_file():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
        return checkpoint

    search_roots: list[Path] = []
    if args.run_dir:
        search_roots.append(resolve_artifact_path(args.run_dir, "Ablation run"))
    else:
        if not args.checkpoint_tag:
            raise ValueError("--checkpoint_tag is required when --checkpoint is not provided")
        base_search_root = resolve_artifact_path(args.search_root, "Ablation checkpoint search root")
        search_root = Path(
            task_log_root(
                str(base_search_root),
                args.task,
                getattr(args, "execution_robot_id", None),
            )
        )
        search_roots.append(search_root)

    candidates: list[Path] = []
    for root in search_roots:
        if root.is_file() and root.suffix == ".pth":
            candidates.append(root)
            continue
        if not root.exists():
            continue
        if args.checkpoint_tag and root.name == args.checkpoint_tag:
            patterns = [str(root / "rl_runs" / "*" / "nn" / "*.pth")]
        elif args.checkpoint_tag and root.name.endswith("_" + args.checkpoint_tag):
            patterns = [str(root / "rl_runs" / "*" / "nn" / "*.pth")]
        elif args.checkpoint_tag:
            patterns = [
                str(root / f"*_{args.checkpoint_tag}" / "rl_runs" / f"*_Run-{args.worker_idx}" / "nn" / "*.pth"),
                str(root / f"*_{args.checkpoint_tag}" / "rl_runs" / "*" / "nn" / "*.pth"),
            ]
        else:
            patterns = [str(root / "**" / "nn" / "*.pth")]
        for pattern in patterns:
            candidates.extend(Path(p) for p in glob.glob(pattern, recursive=True))

    candidates = sorted({p.resolve() for p in candidates if p.is_file()})
    if not candidates:
        where = ", ".join(str(p) for p in search_roots)
        raise FileNotFoundError(
            f"no checkpoint candidates found for tag={args.checkpoint_tag!r} under {where}"
        )

    selector = str(args.checkpoint_selector)
    if selector == "latest_epoch":
        with_epoch = [(p, _checkpoint_epoch(p)) for p in candidates]
        with_epoch = [(p, e) for p, e in with_epoch if e is not None]
        if with_epoch:
            return max(with_epoch, key=lambda item: (int(item[1]), item[0].stat().st_mtime))[0]
        return max(candidates, key=lambda p: p.stat().st_mtime)
    if selector == "latest_mtime":
        return max(candidates, key=lambda p: p.stat().st_mtime)
    if selector == "named_best":
        named = [p for p in candidates if not p.name.startswith("last_")]
        if named:
            return max(named, key=lambda p: p.stat().st_mtime)
        return max(candidates, key=lambda p: p.stat().st_mtime)
    if selector == "best_after_epoch":
        min_epoch = int(args.min_checkpoint_epoch)
        with_reward = []
        for p in candidates:
            epoch = _checkpoint_epoch(p)
            reward = _checkpoint_reward(p)
            if epoch is not None and epoch > min_epoch and reward is not None:
                with_reward.append((p, epoch, reward))
        if with_reward:
            return max(with_reward, key=lambda item: (float(item[2]), int(item[1]), item[0].stat().st_mtime))[0]
        raise FileNotFoundError(
            f"no reward-bearing checkpoint candidates found after epoch {min_epoch} "
            f"for tag={args.checkpoint_tag!r}; pass CHECKPOINT=/path/to/model.pth to override"
        )
    if selector == "best_window_success_after_epoch":
        min_epoch = int(args.min_checkpoint_epoch)
        metric_tag = str(args.checkpoint_metric_tag)
        metric_cache: dict[Path, dict[int, float]] = {}
        with_metric = []
        for p in candidates:
            epoch = _checkpoint_epoch(p)
            if epoch is None or epoch <= min_epoch:
                continue
            run_dir = _checkpoint_run_dir(p).resolve()
            if run_dir not in metric_cache:
                metric_cache[run_dir] = _load_scalar_values_by_step(run_dir, metric_tag)
            metric = _metric_at_or_before_epoch(metric_cache[run_dir], epoch)
            if metric is None:
                continue
            metric_step, metric_value = metric
            with_metric.append((p, epoch, metric_step, metric_value))
        if with_metric:
            selected = max(
                with_metric,
                key=lambda item: (float(item[3]), int(item[1]), int(item[2]), item[0].stat().st_mtime),
            )
            metric_info = {
                "tag": metric_tag,
                "epoch": int(selected[1]),
                "step": int(selected[2]),
                "value": float(selected[3]),
                "checkpoint": str(selected[0]),
            }
            setattr(args, "selected_checkpoint_metric", metric_info)
            print("[AblaEval] selected_checkpoint_metric=" + json.dumps(metric_info, sort_keys=True))
            return selected[0]
        raise FileNotFoundError(
            f"no checkpoints after epoch {min_epoch} could be matched to scalar tag {metric_tag!r} "
            f"for tag={args.checkpoint_tag!r}; pass CHECKPOINT=/path/to/model.pth to override"
        )
    raise ValueError(f"unsupported checkpoint selector: {selector}")


def _add_rates(payload: dict[str, Any]) -> None:
    true_success = int(payload.get("true_success_count") or 0)
    false_complete = int(payload.get("false_complete_count") or 0)
    timeout = int(payload.get("timeout_count") or 0)
    total = true_success + false_complete + timeout
    payload["total_episodes"] = total
    payload["true_success_rate"] = true_success / float(total) if total else 0.0
    payload["false_complete_rate"] = false_complete / float(total) if total else 0.0
    payload["timeout_rate"] = timeout / float(total) if total else 0.0


def _add_summary_stats(payload: dict[str, Any], *, reward_preset: str) -> None:
    highest = payload.get("highest_stage_reached_per_env")
    if isinstance(highest, list) and highest:
        stages = [int(x) for x in highest]
        payload["max_stage_reached"] = max(stages)
        payload["mean_highest_stage_reached"] = sum(stages) / float(len(stages))

    max_return = payload.get("max_episodic_return")
    if max_return is not None:
        payload["max_reward"] = float(max_return)
        if reward_preset == "external":
            payload["max_external_reward"] = float(max_return)


def _snapshot_program(env_u: Any) -> dict[str, Any]:
    program_runtime = getattr(env_u, "program_runtime", None)
    compiled_program = getattr(program_runtime, "compiled_program", None)
    if program_runtime is None or compiled_program is None:
        raise RuntimeError("environment does not expose program_runtime.compiled_program")
    if not hasattr(program_runtime, "get_constraints_by_stage"):
        raise RuntimeError("program_runtime does not expose get_constraints_by_stage()")

    stage_count = int(compiled_program.stage_count())
    constraints = [
        copy.deepcopy(program_runtime.get_constraints_by_stage(stage_id))
        for stage_id in range(stage_count)
    ]
    weights: dict[int, dict[int, Any]] = {}
    if hasattr(program_runtime, "get_residual_weights"):
        weights = {
            stage_id: copy.deepcopy(program_runtime.get_residual_weights(stage_id))
            for stage_id in range(stage_count)
        }
    return {"constraints": constraints, "weights": weights}


def _restore_program(env_u: Any, snapshot: dict[str, Any]) -> None:
    program_runtime = getattr(env_u, "program_runtime", None)
    if program_runtime is None:
        raise RuntimeError("environment does not expose program_runtime")

    constraints = list(snapshot.get("constraints") or [])
    if constraints:
        program_runtime.reload_constraints(
            constraints_list=copy.deepcopy(constraints),
            stage_indices=list(range(len(constraints))),
        )

    weights = snapshot.get("weights") or {}
    if weights and hasattr(program_runtime, "set_residual_weights"):
        for stage_id, stage_weights in dict(weights).items():
            if stage_weights:
                program_runtime.set_residual_weights(int(stage_id), copy.deepcopy(stage_weights))


def _payload_delta(primary: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in (
        "average_episodic_return",
        "progress_metric_eval",
        "true_success_rate",
        "false_complete_rate",
        "timeout_rate",
    ):
        left = primary.get(key)
        right = baseline.get(key)
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            out[f"{key}_delta"] = float(left) - float(right)
    return out


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a yielded Kuafu ablation checkpoint with the same fixed reward setup."
    )
    parser.add_argument("--task", required=True)
    parser.add_argument(
        "--execution_robot_id",
        default=None,
        help="Robot env package to execute. Defaults to inference from --task; legacy tasks use fr3_finray.",
    )
    parser.add_argument(
        "--artifact_robot_id",
        default=None,
        help="Robot id for artifact lineage metadata. Defaults to fr3_finray for existing priors/CSVs.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_envs", type=int, default=8192)
    parser.add_argument("--env_seed", type=int, default=42)
    parser.add_argument("--worker_idx", type=int, default=0)
    parser.add_argument("--weights_params_json", default="")
    parser.add_argument("--agent_cfg", default="")
    parser.add_argument("--reward_preset", choices=["config", "dense", "external"], default="config")
    parser.add_argument("--prior_checkpoint", default="")
    parser.add_argument("--prior_mode", choices=["full_actor", "residual_rl"], default="full_actor")
    parser.add_argument(
        "--residual_hide_stage_observation",
        action="store_true",
        default=False,
        help="Hide scalar stage observations and disable appended stage one-hot for residual RL.",
    )
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--checkpoint_tag", default="")
    parser.add_argument(
        "--checkpoint_selector",
        choices=[
            "latest_epoch",
            "latest_mtime",
            "named_best",
            "best_after_epoch",
            "best_window_success_after_epoch",
        ],
        default="named_best",
    )
    parser.add_argument("--checkpoint_metric_tag", default="Episode/Kuafu/success_rate_window")
    parser.add_argument("--run_dir", default="")
    parser.add_argument("--min_checkpoint_epoch", type=int, default=200)
    parser.add_argument(
        "--search_root",
        default=str(external_artifact_root() / "residual_rl" / "ablation"),
    )
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--tag", default="")
    parser.add_argument("--render", action="store_true", default=False)
    parser.add_argument("--max_eval_steps", type=int, default=0)
    parser.add_argument("--progress_metric_expr", default=_default_progress_expr())
    parser.add_argument("--clip_actions", type=float, default=None)
    parser.add_argument(
        "--residual_clip_composed_actions",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override residual_rl.clip_composed_actions for this ablation eval.",
    )
    parser.add_argument("--residual_alpha_warmup_epochs", type=int, default=None)
    parser.add_argument(
        "--naive_reward_eval",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also evaluate the checkpoint under a vanilla dense/default-program reward.",
    )
    parser.add_argument("--naive_reward_preset", choices=["config", "dense", "external"], default="dense")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    execution_robot_id = resolve_execution_robot_id(args.task, args.execution_robot_id)
    artifact_robot_id = resolve_artifact_robot_id(args.artifact_robot_id)
    args.execution_robot_id = execution_robot_id
    args.artifact_robot_id = artifact_robot_id
    checkpoint = _resolve_checkpoint(args)
    weights_params_path = _recipe_or_artifact_path(args.weights_params_json, "Ablation weights/parameters")
    all_weights, all_parameters = _load_weights_params(args.weights_params_json)
    agent_cfg_path = _recipe_or_artifact_path(args.agent_cfg, "Ablation agent configuration")
    prior_checkpoint = _artifact_path(args.prior_checkpoint, "Ablation BC prior checkpoint")

    run_label = args.tag or f"eval_{args.checkpoint_tag or checkpoint.stem}"
    if args.output_dir:
        output_root = resolve_artifact_path(args.output_dir, "Ablation evaluation output")
    else:
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        task_root = Path(
            task_log_root(
                str(external_artifact_root() / "residual_rl" / "ablation_evaluation"),
                args.task,
                execution_robot_id,
            )
        )
        output_root = task_root / f"{stamp}_{run_label}"
    output_root.mkdir(parents=True, exist_ok=True)

    from kuafu.rl.isaac.env_factory import create_env_and_app
    from kuafu.rl.isaac.train_rl_games import evaluate_rl_games
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
        env_u = env.unwrapped
        base_agent_cfg = copy.deepcopy(agent_cfg)
        default_program_snapshot = _snapshot_program(env_u)
        eval_agent_cfg = copy.deepcopy(base_agent_cfg)
        _apply_reward_preset(env_u, eval_agent_cfg, args.reward_preset)
        _apply_residual_observation_mode(
            eval_agent_cfg,
            hide_stage_observation=bool(args.residual_hide_stage_observation),
        )
        _apply_abla_residual_overrides(
            eval_agent_cfg,
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

        print("[AblaEval] Evaluating fixed-program checkpoint")
        print(f"[AblaEval] task={args.task}")
        print(f"[AblaEval] execution_robot_id={execution_robot_id}")
        print(f"[AblaEval] artifact_robot_id={artifact_robot_id}")
        print(f"[AblaEval] checkpoint={checkpoint}")
        print(f"[AblaEval] checkpoint_selector={args.checkpoint_selector}")
        selected_checkpoint_metric = getattr(args, "selected_checkpoint_metric", None)
        if selected_checkpoint_metric is not None:
            print("[AblaEval] checkpoint_metric=" + json.dumps(selected_checkpoint_metric, sort_keys=True))
        print(f"[AblaEval] reward_preset={args.reward_preset}")
        print(f"[AblaEval] agent_cfg={agent_cfg_path or '<registry>'}")
        print(f"[AblaEval] weights_params_json={weights_params_path or '<default program>'}")
        print(f"[AblaEval] prior_checkpoint={prior_checkpoint or '<none>'}")
        print(f"[AblaEval] prior_mode={args.prior_mode if prior_checkpoint else '<none>'}")
        print(f"[AblaEval] residual_hide_stage_observation={bool(args.residual_hide_stage_observation)}")
        print(f"[AblaEval] clip_actions={args.clip_actions if args.clip_actions is not None else '<config>'}")
        print(
            f"[AblaEval] residual_clip_composed_actions="
            f"{args.residual_clip_composed_actions if args.residual_clip_composed_actions is not None else '<config>'}"
        )
        print(
            f"[AblaEval] residual_alpha_warmup_epochs="
            f"{args.residual_alpha_warmup_epochs if args.residual_alpha_warmup_epochs is not None else '<config>'}"
        )
        print(f"[AblaEval] env_seed={env_seed} worker_idx={args.worker_idx}")
        print(f"[AblaEval] output_dir={output_root}")

        payload = evaluate_rl_games(
            env=env,
            task=args.task,
            device=args.device,
            checkpoint_path=str(checkpoint),
            prior_checkpoint=prior_checkpoint,
            prior_mode=args.prior_mode,
            agent_cfg=eval_agent_cfg,
            progress_metric_expr=str(args.progress_metric_expr),
            seed=env_seed,
            max_eval_steps=(int(args.max_eval_steps) if int(args.max_eval_steps) > 0 else None),
        )
        _add_rates(payload)
        result: dict[str, Any] = {
            "task": args.task,
            "execution_robot_id": execution_robot_id,
            "artifact_robot_id": artifact_robot_id,
            "checkpoint": str(checkpoint),
            "checkpoint_tag": args.checkpoint_tag,
            "checkpoint_selector": args.checkpoint_selector,
            "selected_checkpoint_metric": getattr(args, "selected_checkpoint_metric", None),
            "reward_preset": args.reward_preset,
            "agent_cfg": agent_cfg_path or "<registry>",
            "weights_params_json": weights_params_path or "<default program>",
            "prior_checkpoint": prior_checkpoint or "<none>",
            "prior_mode": args.prior_mode if prior_checkpoint else "<none>",
            "env_seed": env_seed,
            "worker_idx": int(args.worker_idx),
            "num_envs": int(args.num_envs),
            "payload": payload,
        }
        _add_summary_stats(payload, reward_preset=str(args.reward_preset))
        if args.naive_reward_eval:
            _restore_program(env_u, default_program_snapshot)
            naive_agent_cfg = copy.deepcopy(base_agent_cfg)
            _apply_reward_preset(env_u, naive_agent_cfg, args.naive_reward_preset)
            _apply_residual_observation_mode(
                naive_agent_cfg,
                hide_stage_observation=bool(args.residual_hide_stage_observation),
            )
            apply_updates_and_wrap_reset(
                env,
                stage_id=0,
                all_stages_weights=None,
                all_stages_parameters=None,
                progress_metric_expr=str(args.progress_metric_expr),
            )
            print("[AblaEval] Evaluating naive reward baseline")
            print(f"[AblaEval] naive_reward_preset={args.naive_reward_preset}")
            print("[AblaEval] naive_weights_params_json=<default program>")
            naive_payload = evaluate_rl_games(
                env=env,
                task=args.task,
                device=args.device,
                checkpoint_path=str(checkpoint),
                prior_checkpoint=prior_checkpoint,
                prior_mode=args.prior_mode,
                agent_cfg=naive_agent_cfg,
                progress_metric_expr=str(args.progress_metric_expr),
                seed=env_seed,
                max_eval_steps=(int(args.max_eval_steps) if int(args.max_eval_steps) > 0 else None),
            )
            _add_rates(naive_payload)
            _add_summary_stats(naive_payload, reward_preset=str(args.naive_reward_preset))
            naive_delta = _payload_delta(payload, naive_payload)
            result["naive_reward_eval"] = {
                "reward_preset": args.naive_reward_preset,
                "weights_params_json": "<default program>",
                "payload": naive_payload,
                "primary_minus_naive": naive_delta,
            }
            print("[AblaEval] naive_reward_result=" + json.dumps(naive_payload, sort_keys=True, default=_json_default))
            print("[AblaEval] primary_minus_naive=" + json.dumps(naive_delta, sort_keys=True, default=_json_default))
        result_path = output_root / "eval.json"
        result_path.write_text(json.dumps(result, indent=2, sort_keys=True, default=_json_default) + "\n")
        print("[AblaEval] result=" + json.dumps(payload, sort_keys=True, default=_json_default))
        print(f"[AblaEval] wrote={result_path}")
    finally:
        if env is not None:
            env.close()
        if simulation_app is not None:
            simulation_app.close()


if __name__ == "__main__":
    main()
