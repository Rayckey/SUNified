from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


THIS_FILE = Path(__file__).resolve()
SOURCE_ROOT = THIS_FILE.parents[3]
PACKAGE_ROOT = THIS_FILE.parents[2]
for candidate in (str(SOURCE_ROOT), str(PACKAGE_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)



from kuafu.paths import resolve_artifact_path
from kuafu.artifact_contracts import file_sha256, learned_program_identity, validate_program_metadata
from kuafu.bc.training.rl_games_prior import _resolve_prior_checkpoint_path
from kuafu.rl.core.core_utilities import extract_tolerances_from_constraints_raw
from kuafu.rl.isaac.env_factory import create_env_and_app
from kuafu.rl.isaac.evaluation_stats import load_bc_eval_stats
from kuafu.rl.isaac.initial_constraint_sweep import (
    _collect_task_info,
    _format_indexed_lines,
    _load_initial_stage_overrides,
    _load_json_like,
    _normalize_weights,
    _require_openai_credentials,
    _seed_task_info,
    _strip_llm_json,
)
from kuafu.rl.isaac.train_rl_games import evaluate_bc_prior_policy_rollout
from kuafu.rl.isaac.update_utilities import update_all_parameters, update_weights
from kuafu.rl.llm.client import OpenAIChatClient
from kuafu.rl.llm.prompts import (
    SINGLE_BOTH_PROMPT,
    SINGLE_BOTH_TUNING_FORMATTING_INSTRUCTIONS,
    SYSTEM_PROMPT,
)
from kuafu.rl.logging.run_dir import (
    append_jsonl,
    append_prompt_transcript,
    make_run_dirs,
    write_json,
)
from kuafu.skills.constraints.tolerance_policy import (
    assign_automatic_tolerances,
    policy_metadata,
    summarize_atomic_action,
)
from kuafu.common.env_prompt_helpers import (
    format_constraint_satisfaction_rates,
    format_constraint_unsatisfaction_rates,
    format_reward_values,
    format_stage_satisfaction_times,
)


WEIGHT_SWEEP_TRANSCRIPT_FILENAME = "weight_sweep_prompt_transcript.md"
WEIGHT_STAGE_RESPONSE_KEYS = ("Diagnosis", "Reasoning", "Weights")

BC_PRIOR_WEIGHT_SWEEP_NOTE = """
You are doing an initial weight acquisition pass from a deterministic BC-prior rollout.
The rollout statistics are not RL-training results; they show where the BC prior times out.
The timeout/stuck component rates are the key evidence: high not_satisfied means that
component blocked the BC prior for envs that reached and timed out in that stage.

Adjust ONLY reward-component Weights for the current stage.
"""


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Acquire initial residual stage weights from a BC-prior timeout diagnostic rollout."
    )
    parser.add_argument("--task", type=str, required=True, help="Gym task name.")
    parser.add_argument(
        "--execution_robot_id",
        type=str,
        default=None,
        help="Robot env package to execute. Defaults to inference from --task; legacy tasks use fr3_finray.",
    )
    parser.add_argument(
        "--artifact_robot_id",
        type=str,
        default=None,
        help="Robot id for artifact lineage metadata. Defaults to fr3_finray for existing priors/CSVs.",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Simulation/RL device.")
    parser.add_argument("--num_envs", type=int, default=8192, help="Override env count.")
    parser.add_argument("--env_seed", type=int, default=42, help="Environment seed.")
    parser.add_argument("--prior_checkpoint", type=str, required=True, help="BC prior checkpoint or run-dir path.")
    parser.add_argument(
        "--prior_obs_stats_count",
        type=float,
        default=1e6,
        help="Synthetic sample count used when copying BC observation stats into RL-Games RMS.",
    )
    parser.add_argument(
        "--prior_verbose",
        action="store_true",
        default=False,
        help="Print verbose BC prior weight-copy logging.",
    )
    parser.add_argument(
        "--prior_freeze_input_running_mean_std",
        action="store_true",
        default=False,
        help="Pin rl_games observation RMS to the BC prior stats during the diagnostic rollout.",
    )
    parser.add_argument(
        "--max_eval_steps",
        type=int,
        default=None,
        help="Diagnostic rollout horizon. Defaults to env max_episode_length.",
    )
    parser.add_argument(
        "--bc_eval_stats_path", type=str, default=None,
        help="Completed BC rollout statistics to reuse instead of running another diagnostic rollout.",
    )
    parser.add_argument("--reasoning_effort", default=None, help="LLM reasoning effort; defaults to KUAFU_LLM_REASONING_EFFORT.")
    parser.add_argument("--gpt_model", type=str, default="gpt-5.5", help="OpenAI model name.")
    parser.add_argument("--temperature", type=float, default=0.2, help="LLM sampling temperature.")
    parser.add_argument(
        "--openai_api_key",
        type=str,
        default=None,
        help="OpenAI API key (optional; otherwise use environment).",
    )
    parser.add_argument(
        "--initial_weights_params",
        type=str,
        default=None,
        help=(
            "Optional best.json/stages JSON used to seed starting weights. Parameter entries are "
            "accepted for compatibility and then replaced by the automatic policy."
        ),
    )
    parser.add_argument(
        "--launch_script",
        type=str,
        default=None,
        help="Optional launcher script path copied into the run directory.",
    )
    parser.add_argument(
        "--render",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Render the env instead of headless mode.",
    )
    parser.add_argument(
        "--output_best_path_file",
        type=str,
        default=None,
        help="Optional path where the absolute path to generated best.json will be written.",
    )
    return parser


def _stage_parameters(stage_meta: dict[str, Any]) -> dict[int, dict[str, object]]:
    if "parameters" in stage_meta:
        raw_parameters = stage_meta["parameters"]
        if not isinstance(raw_parameters, dict):
            raise TypeError("[residual_rl] stage parameters must be a dict.")
        return {int(k): dict(v) for k, v in dict(raw_parameters).items()}
    constraints_raw = stage_meta.get("constraints_raw")
    if not isinstance(constraints_raw, list):
        raise TypeError("[residual_rl] stage constraints_raw must be a list.")
    return extract_tolerances_from_constraints_raw(constraints_raw)


def _all_stage_weights_and_parameters(
    task_info: dict[str, Any],
) -> tuple[dict[int, dict[int, float]], dict[int, dict[int, dict[str, object]]]]:
    stages = task_info.get("stages")
    if not isinstance(stages, dict):
        raise RuntimeError("[residual_rl] Task info is missing stage metadata.")
    stage_count = int(task_info["stage_count"])
    all_weights: dict[int, dict[int, float]] = {}
    all_parameters: dict[int, dict[int, dict[str, object]]] = {}
    for sid in range(stage_count):
        stage_meta = stages[int(sid)]
        if not isinstance(stage_meta, dict):
            raise TypeError(f"[residual_rl] Stage {sid} metadata must be a dict.")
        all_weights[int(sid)] = {
            int(k): float(v) for k, v in dict(stage_meta["weights"]).items()
        }
        all_parameters[int(sid)] = _stage_parameters(stage_meta)
    return all_weights, all_parameters


def _apply_task_info_to_env(env, task_info: dict[str, Any]) -> None:
    all_weights, all_parameters = _all_stage_weights_and_parameters(task_info)
    env_u = env.unwrapped
    update_all_parameters(env_u, all_stages_parameters=all_parameters)
    update_weights(env_u, stage_id=None, weights=None, all_stages_weights=all_weights)


def _enable_timeout_constraint_stats(env) -> None:
    env_u = env.unwrapped
    cfg = getattr(env_u, "cfg", None)
    if cfg is not None:
        setattr(cfg, "collect_terminal_stuck_constraint_stats", True)
        mpc_config = getattr(cfg, "mpc_config", None)
        if isinstance(mpc_config, dict):
            mpc_config["collect_terminal_stuck_constraint_stats"] = True


def _parse_weight_stage_response(raw: str) -> dict[str, Any]:
    cleaned = _strip_llm_json(raw)
    idx = cleaned.find("{")
    if 0 < idx < 200:
        cleaned = cleaned[idx:]
    try:
        parsed = _load_json_like(cleaned)
    except ValueError as exc:
        raise ValueError(
            "[residual_rl] Could not parse weight-sweep LLM response as JSON. "
            f"Raw preview: {raw[:300]}"
        ) from exc
    if not isinstance(parsed, dict):
        raise TypeError(
            f"[residual_rl] Weight-sweep response must be a JSON object, got {type(parsed).__name__}."
        )
    return parsed


def _require_weight_stage_response_fields(parsed: dict[str, Any], *, stage_id: int) -> None:
    for key in WEIGHT_STAGE_RESPONSE_KEYS:
        if key not in parsed:
            raise KeyError(f"[residual_rl] Stage {stage_id} LLM response is missing {key!r}.")
    if parsed["Diagnosis"] is None:
        raise ValueError(f"[residual_rl] Stage {stage_id} Diagnosis must not be null.")
    if parsed["Reasoning"] is None:
        raise ValueError(f"[residual_rl] Stage {stage_id} Reasoning must not be null.")
    if not isinstance(parsed["Weights"], dict):
        raise TypeError(f"[residual_rl] Stage {stage_id} Weights must be a dict.")


def _format_bc_prior_end_to_end_feedback(eval_payload: dict[str, object]) -> str:
    true_success_count = int(eval_payload.get("true_success_count") or 0)
    false_complete_count = int(eval_payload.get("false_complete_count") or 0)
    timeout_count = int(eval_payload.get("timeout_count") or 0)
    total_episodes = int(eval_payload.get("total_episodes") or 0)
    if total_episodes <= 0:
        total_episodes = true_success_count + false_complete_count + timeout_count
    true_rate = true_success_count / float(total_episodes) if total_episodes else 0.0
    false_rate = false_complete_count / float(total_episodes) if total_episodes else 0.0
    timeout_rate = timeout_count / float(total_episodes) if total_episodes else 0.0
    return "\n".join(
        [
            "Baseline source: deterministic BC-prior diagnostic rollout.",
            f"prior_checkpoint: {eval_payload.get('prior_checkpoint', '<unknown>')}",
            f"total_steps: {int(eval_payload.get('total_steps') or 0)}",
            f"completed_episodes: {int(eval_payload.get('completed_episodes') or 0)}",
            f"true_success_count: {true_success_count}",
            f"false_complete_count: {false_complete_count}",
            f"timeout_count: {timeout_count}",
            f"total_episodes: {total_episodes}",
            f"true_success_rate: {true_rate:.6f}",
            f"false_complete_rate: {false_rate:.6f}",
            f"timeout_rate: {timeout_rate:.6f}",
        ]
    )


def _format_timeout_component_block(
    constraint_stats: dict[int, dict[int, dict[str, Any]]] | None,
    *,
    stage_id: int,
) -> str:
    not_satisfied = format_constraint_unsatisfaction_rates(constraint_stats, stage_id=stage_id)
    satisfied = format_constraint_satisfaction_rates(constraint_stats, stage_id=stage_id)
    return "\n".join(
        [
            "Timeout/stuck non-satisfaction rates (primary signal):",
            not_satisfied,
            "",
            "Timeout/stuck satisfaction rates (same samples, complementary view):",
            satisfied,
        ]
    )


def run_initial_weight_sweep(args: argparse.Namespace) -> str:
    if args.openai_api_key:
        os.environ["OPENAI_API_KEY"] = str(args.openai_api_key)
    _require_openai_credentials(args)

    prior_checkpoint = str(resolve_artifact_path(args.prior_checkpoint, "BC prior checkpoint"))
    prior_checkpoint, _ = _resolve_prior_checkpoint_path(prior_checkpoint)
    if not prior_checkpoint:
        raise ValueError("[residual_rl] --prior_checkpoint must be non-empty.")

    saved_stats_path = getattr(args, "bc_eval_stats_path", None)
    saved_stats = (
        load_bc_eval_stats(saved_stats_path, prior_checkpoint=prior_checkpoint, task=str(args.task))
        if saved_stats_path else None
    )

    launch_script = str(args.launch_script).strip() if args.launch_script else None
    run_dirs = make_run_dirs(
        task=str(args.task),
        argv=sys.argv,
        launch_script_path=launch_script,
        execution_robot_id=getattr(args, "execution_robot_id", None),
        artifact_robot_id=getattr(args, "artifact_robot_id", None),
    )
    transcript_path = os.path.join(run_dirs.root_dir, WEIGHT_SWEEP_TRANSCRIPT_FILENAME)
    print(
        f"[residual_rl] Initial weight sweep: run_dir={run_dirs.root_dir} "
        f"(best.json when done; prompt transcript -> {WEIGHT_SWEEP_TRANSCRIPT_FILENAME})",
        flush=True,
    )

    simulation_app = None
    env = None
    try:
        print("[residual_rl] Initial weight sweep: creating Isaac env…", flush=True)
        simulation_app, env, agent_cfg = create_env_and_app(
            task=str(args.task),
            device=str(args.device),
            env_seed=int(args.env_seed),
            num_envs=1 if saved_stats is not None else int(args.num_envs),
            headless=not bool(args.render),
            execution_robot_id=getattr(args, "execution_robot_id", None),
            initial_prior_checkpoint=prior_checkpoint,
        )
        _enable_timeout_constraint_stats(env)
        program_identity = learned_program_identity(env.unwrapped.cfg.mpc_config["load_program"])
        if saved_stats is not None:
            validate_program_metadata(saved_stats, program_identity)

        print("[residual_rl] Initial weight sweep: collecting stage metadata…", flush=True)
        task_info = _collect_task_info(env, task=str(args.task))
        overrides_path = str(args.initial_weights_params).strip() if args.initial_weights_params else None
        if overrides_path:
            _seed_task_info(
                task_info,
                _load_initial_stage_overrides(
                    overrides_path,
                    expected_task=str(args.task),
                    expected_prior_checkpoint=prior_checkpoint,
                ),
            )
        assign_automatic_tolerances(task_info)
        _apply_task_info_to_env(env, task_info)

        if saved_stats is not None:
            print(f"[residual_rl] Initial weight sweep: reusing completed BC eval: {saved_stats_path}", flush=True)
            eval_payload = saved_stats
            recorded_stage_count = eval_payload.get("stage_count")
            if recorded_stage_count is not None and int(recorded_stage_count) != int(task_info["stage_count"]):
                raise ValueError("BC eval stats stage count does not match the selected task program.")
        else:
            print("[residual_rl] Initial weight sweep: running deterministic BC-prior diagnostic rollout…", flush=True)
            eval_payload = evaluate_bc_prior_policy_rollout(
                env=env,
                task=str(args.task),
                device=str(args.device),
                prior_checkpoint=prior_checkpoint,
                agent_cfg=agent_cfg,
                prior_obs_stats_count=float(args.prior_obs_stats_count),
                prior_verbose=bool(args.prior_verbose),
                prior_freeze_input_running_mean_std=bool(args.prior_freeze_input_running_mean_std),
                seed=int(args.env_seed),
                max_eval_steps=args.max_eval_steps,
            )
        eval_payload["prior_checkpoint"] = prior_checkpoint
        eval_payload.update({"task": str(args.task), "prior_checkpoint_sha256": file_sha256(prior_checkpoint),
                             **program_identity})
        stats_path = os.path.join(run_dirs.root_dir, "bc_prior_rollout_stats.json")
        write_json(stats_path, eval_payload)
        print(f"[residual_rl] Initial weight sweep: BC-prior stats written to {stats_path}", flush=True)

        stage_count = int(task_info["stage_count"])
        stages = task_info["stages"]
        if not isinstance(stages, dict):
            raise RuntimeError("[residual_rl] task_info['stages'] must be a dict.")
        task_description = str(task_info["task_description"]).strip()
        detectable_objects_str = json.dumps(
            task_info["detectable_objects"],
            indent=2,
            sort_keys=True,
            default=str,
        )

        system_prompt = SYSTEM_PROMPT + BC_PRIOR_WEIGHT_SWEEP_NOTE + SINGLE_BOTH_TUNING_FORMATTING_INSTRUCTIONS
        end_to_end_feedback = _format_bc_prior_end_to_end_feedback(eval_payload)
        constraint_stats = eval_payload.get("constraint_met_rate_stats")
        if not isinstance(constraint_stats, dict):
            constraint_stats = {}
        stage_transition_stats = eval_payload.get("stage_transition_stats")
        if not isinstance(stage_transition_stats, dict):
            stage_transition_stats = {}
        highest = eval_payload.get("highest_stage_reached_per_env")
        if not isinstance(highest, list):
            highest = None

        stage_summaries: dict[str, dict[str, Any]] = {}
        print(
            f"[residual_rl] Initial weight sweep: calling OpenAI once per stage "
            f"({stage_count} stage(s)); starting stage 0…",
            flush=True,
        )
        for stage_id in range(stage_count):
            stage_meta = stages[int(stage_id)]
            if not isinstance(stage_meta, dict):
                raise TypeError(f"[residual_rl] Stage metadata {stage_id} must be a dict.")
            explanations = stage_meta["explanations"]
            if not isinstance(explanations, list) or not explanations:
                raise RuntimeError(f"[residual_rl] Stage {stage_id} explanations must be a non-empty list.")
            current_weights = {
                int(k): float(v) for k, v in dict(stage_meta["weights"]).items()
            }
            current_parameters = _stage_parameters(stage_meta)
            if not current_weights:
                raise RuntimeError(f"[residual_rl] Stage {stage_id} current_weights must not be empty.")

            stage_satisfaction_times = format_stage_satisfaction_times(
                stage_transition_stats,
                highest_stage_reached_per_env=highest,
                stage_id=stage_id,
                num_env=len(highest) if highest else int(args.num_envs),
            )
            timeout_component_block = _format_timeout_component_block(
                constraint_stats,
                stage_id=stage_id,
            )
            reward_values = format_reward_values(constraint_stats, stage_id=stage_id)
            user_prompt = SINGLE_BOTH_PROMPT.format(
                task_description=task_description,
                detectable_objects=detectable_objects_str,
                stage_id=stage_id,
                stage_count=stage_count,
                atomic_action=summarize_atomic_action(stage_meta["atomic_action"]),
                constraints_explanations=_format_indexed_lines(explanations),
                current_weights=json.dumps(current_weights, indent=2, sort_keys=True),
                end_to_end_feedback=end_to_end_feedback,
                stage_satisfaction_times=stage_satisfaction_times,
                constraint_satisfaction_rates=timeout_component_block,
                reward_values=reward_values,
            )

            print(
                f"[residual_rl] Initial weight sweep: calling OpenAI for stage {stage_id} "
                f"(model={args.gpt_model})…",
                flush=True,
            )
            llm = OpenAIChatClient(
                model=str(args.gpt_model),
                temperature=float(args.temperature),
                num_suggestions=1,
                system_prompt=system_prompt,
                reasoning_effort=getattr(args, "reasoning_effort", None),
            )
            llm_resp = llm.prompt(user_prompt=user_prompt)
            if len(llm_resp.raw_outputs) != 1:
                raise RuntimeError(
                    f"[residual_rl] Expected exactly one LLM response for stage {stage_id}, got {len(llm_resp.raw_outputs)}."
                )
            raw_output = llm_resp.raw_outputs[0]
            timestamp = datetime.now().isoformat()
            append_jsonl(
                run_dirs.openai_payloads_path,
                {
                    "timestamp": timestamp,
                    "kind": "initial_weight_sweep_stage",
                    "stage_id": int(stage_id),
                    "messages": llm_resp.messages,
                    "raw_outputs": llm_resp.raw_outputs,
                    "usage": llm_resp.usage,
                    "extra": {
                        "constraints_explanations": explanations,
                        "stage_satisfaction_times": stage_satisfaction_times,
                        "timeout_component_block": timeout_component_block,
                        "reward_values": reward_values,
                    },
                },
            )
            append_prompt_transcript(
                path=transcript_path,
                header={
                    "timestamp": timestamp,
                    "kind": "initial_weight_sweep_stage",
                    "stage_id": int(stage_id),
                    "iter": 0,
                },
                messages=llm_resp.messages,
                raw_outputs=llm_resp.raw_outputs,
            )

            parsed = _parse_weight_stage_response(raw_output)
            _require_weight_stage_response_fields(parsed, stage_id=stage_id)
            ignored_llm_parameters = "Parameters" in parsed
            normalized_weights = _normalize_weights(
                current_weights,
                parsed["Weights"],
                constrain_step=True,
            )
            _diag = parsed["Diagnosis"]
            diagnosis = (_diag if isinstance(_diag, str) else json.dumps(_diag)).strip()
            _reas = parsed["Reasoning"]
            reasoning = (_reas if isinstance(_reas, str) else json.dumps(_reas)).strip()
            stage_timeout_stats = constraint_stats.get(stage_id, {})
            append_jsonl(
                run_dirs.iterations_path,
                {
                    "timestamp": timestamp,
                    "kind": "initial_weight_sweep_stage",
                    "stage_id": int(stage_id),
                    "iter": 0,
                    "weights": normalized_weights,
                    "parameters": current_parameters,
                    "tolerance_source": "automatic",
                    "tolerance_decisions": stage_meta.get("tolerance_decisions", {}),
                    "ignored_llm_parameters": ignored_llm_parameters,
                    "diagnosis": diagnosis,
                    "reasoning": reasoning,
                    "timeout_constraint_stats": stage_timeout_stats,
                },
            )
            append_jsonl(
                run_dirs.trials_path,
                {
                    "timestamp": timestamp,
                    "kind": "initial_weight_sweep_stage",
                    "stage_id": int(stage_id),
                    "iter": 0,
                    "weights": normalized_weights,
                    "parameters": current_parameters,
                    "tolerance_source": "automatic",
                    "tolerance_decisions": stage_meta.get("tolerance_decisions", {}),
                    "ignored_llm_parameters": ignored_llm_parameters,
                    "raw_llm_output": raw_output,
                    "timeout_constraint_stats": stage_timeout_stats,
                },
            )
            stage_summaries[str(stage_id)] = {
                "iter": 0,
                "weights": normalized_weights,
                "parameters": current_parameters,
                "tolerance_source": "automatic",
                "tolerance_decisions": stage_meta.get("tolerance_decisions", {}),
                "ignored_llm_parameters": ignored_llm_parameters,
                "diagnosis": diagnosis,
                "reasoning": reasoning,
                "timeout_constraint_stats": stage_timeout_stats,
            }

        best_summary = {
            "kind": "initial_weight_sweep",
            "task": str(args.task),
            "prior_checkpoint": prior_checkpoint,
            "prior_checkpoint_sha256": file_sha256(prior_checkpoint),
            "automatic_tolerance_policy": policy_metadata(),
            **program_identity,
            "bc_prior_rollout_stats": {
                "true_success_count": int(eval_payload.get("true_success_count") or 0),
                "false_complete_count": int(eval_payload.get("false_complete_count") or 0),
                "timeout_count": int(eval_payload.get("timeout_count") or 0),
                "total_episodes": int(eval_payload.get("total_episodes") or 0),
                "total_steps": int(eval_payload.get("total_steps") or 0),
                "stats_path": stats_path,
            },
            "stages": stage_summaries,
        }
        write_json(run_dirs.best_path, best_summary)

        output_best_path_file = (
            str(resolve_artifact_path(args.output_best_path_file, "Initial-weight result pointer"))
            if args.output_best_path_file
            else ""
        )
        if output_best_path_file:
            with open(output_best_path_file, "w", encoding="utf-8") as f:
                f.write(run_dirs.best_path)
            written = Path(output_best_path_file).read_text(encoding="utf-8").strip()
            if written != run_dirs.best_path.strip():
                raise RuntimeError(
                    f"[residual_rl] Output path file mismatch: expected {run_dirs.best_path!r}, got {written!r}."
                )
        print(f"[residual_rl] Initial weight sweep complete: {run_dirs.root_dir}", flush=True)
        print(f"[residual_rl] best.json: {run_dirs.best_path}", flush=True)
        return run_dirs.root_dir
    finally:
        import traceback as _tb

        _exc_info = sys.exc_info()
        if _exc_info[1] is not None and not isinstance(_exc_info[1], SystemExit):
            print(
                f"[residual_rl] Initial weight sweep EXCEPTION (before cleanup): "
                f"{type(_exc_info[1]).__name__}: {_exc_info[1]}",
                file=sys.stderr,
                flush=True,
            )
            _tb.print_exc(file=sys.stderr)
        if env is not None:
            env.close()
        if simulation_app is not None:
            simulation_app.close()


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    try:
        run_initial_weight_sweep(args)
    except BaseException:
        print(
            "[residual_rl] Initial weight sweep aborted before writing best.json "
            "(see the external residual_rl artifact root if a run_dir was created).",
            file=sys.stderr,
            flush=True,
        )
        raise


if __name__ == "__main__":
    main()
