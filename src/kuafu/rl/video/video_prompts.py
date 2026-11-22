"""Prompt construction for Gemini video-to-text summarisation.

Gemini's job is to translate rollout mp4s into structured text that GPT can
act on during global finetune.  The more task context Gemini gets, the better
it can align with GPT's reasoning about stages, constraints, and failure modes.

This module is the single source of truth for what Gemini sees.
"""

from __future__ import annotations

import json
from typing import Any


def build_gemini_context(
    *,
    task_description: str,
    stage_count: int,
    stage_descriptions: str,
    constraints_explanations_by_stage: str,
    current_weights_by_stage: dict[int, dict[int, float]] | str | None = None,
    true_success_rate: str | float | None = None,
    false_complete_rate: str | float | None = None,
    constraint_stats_text: str | None = None,
    stage_transition_text: str | None = None,
    num_envs_total: int | None = None,
    num_envs_in_video: int | None = None,
    iteration_context: str | None = None,
) -> str:
    """Build the full context prompt that is sent alongside each video to Gemini.

    The returned string is prepended to the per-video structured-output
    instructions (see :func:`build_gemini_output_instructions`), then the
    concatenation is sent as ``context_text`` to ``summarize_video_with_gemini``.
    """

    # Weights as readable JSON. Completion thresholds are deliberately absent
    # from this function's API and from the resulting LLM query.
    if isinstance(current_weights_by_stage, dict):
        weights_str = json.dumps(current_weights_by_stage, indent=2, sort_keys=True)
    elif isinstance(current_weights_by_stage, str):
        weights_str = current_weights_by_stage
    else:
        weights_str = "(not provided)"

    # Performance context (if available).
    perf_lines: list[str] = []
    if true_success_rate is not None:
        perf_lines.append(f"- true_success_rate (across ALL {num_envs_total or '?'} envs): {true_success_rate}")
    if false_complete_rate is not None:
        perf_lines.append(f"- false_complete_rate (across ALL {num_envs_total or '?'} envs): {false_complete_rate}")
    if stage_transition_text:
        perf_lines.append(f"\nPer-stage transition stats (across ALL envs):\n{stage_transition_text}")
    if constraint_stats_text:
        perf_lines.append(f"\nPer-stage per-component constraint satisfaction (across ALL envs):\n{constraint_stats_text}")
    perf_block = "\n".join(perf_lines) if perf_lines else "(not available yet)"

    vid_envs = num_envs_in_video or 9
    total_envs = num_envs_total or "many more"

    iter_block = ""
    if iteration_context:
        iter_block = f"\nIteration context: {iteration_context}\n"

    return (
        "You are a vision-language analysis assistant for a reinforcement-learning "
        "training system called Kuafu, built on NVIDIA Isaac Lab.\n"
        "\n"
        "Kuafu trains a robot arm to complete a multi-stage manipulation task.  "
        "A separate LLM (GPT) reads YOUR text summary (not the video directly) "
        "and uses it to tune reward weights.  Your "
        "summary is GPT's only visual evidence, so it must be precise, "
        "stage-aware, and physically grounded.\n"
        "\n"
        "========================\n"
        "TASK\n"
        "========================\n"
        f"{task_description}\n"
        "\n"
        "========================\n"
        f"STAGES ({stage_count} sequential)\n"
        "========================\n"
        f"{stage_descriptions}\n"
        "\n"
        "A stage is complete when ALL of its constraint components are satisfied.  "
        "The robot must complete every stage in order "
        "to truly succeed.\n"
        "\n"
        "========================\n"
        "PER-STAGE REWARD/CONSTRAINT COMPONENTS\n"
        "========================\n"
        "Each component defines a goal requirement, a shaping reward, and a "
        "constraint check.  The index numbers below correspond to the Weights "
        "table that follows.\n"
        "\n"
        f"{constraints_explanations_by_stage}\n"
        "\n"
        "========================\n"
        "CURRENT WEIGHTS (reward-component importance, per stage)\n"
        "========================\n"
        "Higher weight = component contributes more to total reward.  "
        "Total reward is normalised by sum(weights), so only ratios matter.\n"
        "\n"
        f"{weights_str}\n"
        "\n"
        "========================\n"
        "QUANTITATIVE PERFORMANCE (from ALL envs — authoritative)\n"
        "========================\n"
        f"{perf_block}\n"
        "\n"
        "Terminology:\n"
        "- true success: the REAL task goal is achieved (correct end state).\n"
        "- false complete: all stages report 'done' but the real goal is NOT met.\n"
        "- incomplete: the robot did not finish all stages within the episode.\n"
        f"{iter_block}"
        "\n"
        "========================\n"
        "VIDEO FORMAT\n"
        "========================\n"
        f"The video shows a TILED GRID of only {vid_envs} environments from a "
        f"single camera angle.  Training runs {total_envs} environments in "
        "parallel.  The video is a SMALL SAMPLE and may not be representative "
        "of the overall population.\n"
        "\n"
        "RECONCILING VIDEO vs STATS: Always cross-check your visual observations "
        "with the quantitative stats provided above.  The stats cover ALL envs "
        "and are the ground truth for rates, counts, and which constraints are met.  "
        "If the video looks successful but the stats say otherwise, the video sample "
        "likely contains lucky seeds.  Report what you see honestly AND flag any "
        "disagreement with the stats.\n"
        "\n"
        "========================\n"
        "YOUR MISSION\n"
        "========================\n"
        "Watch the video carefully and produce a structured text summary that GPT "
        "can use to decide which relative reward weights to adjust.\n"
        "\n"
        "Focus on:\n"
        "1. STAGE PROGRESSION — Which stages does the robot appear to reach?  "
        "Look across ALL tiles in the grid to assess how many environments "
        "succeed at each stage.  Estimate approximate time boundaries "
        "(e.g. 'stage 0 lasts ~frames 0-40, stage 1 ~frames 40-90, ...').\n"
        "\n"
        "2. PHYSICAL EVENTS — Describe what you SEE the robot doing: approach "
        "direction, gripper state (open/closed), contact quality, object motion, "
        "collisions, oscillation, slipping, overshooting, etc.\n"
        "\n"
        "3. CONSTRAINT ALIGNMENT — For each stage the robot reaches, judge "
        "whether the listed position, orientation, state, and progress components "
        "appear satisfied. Note visible overshoot, oscillation, or misalignment.\n"
        "   - Cross-check against the per-component satisfaction rates above.  "
        "If a component shows 0% met rate in stats but looks satisfied on video, "
        "note the discrepancy and defer to the stats.\n"
        "\n"
        "4. TRUE SUCCESS vs FALSE COMPLETE — Does the final state actually match "
        "the task goal described above?  If the episode ended but the task is not "
        "truly done (goal object/state not achieved, object dropped, wrong position), label it "
        "'false complete' and explain what went wrong.\n"
        "\n"
        "5. EARLIEST FAILURE — Identify the earliest stage where behaviour is "
        "wrong, sloppy, or misaligned.  Even if later stages 'succeed' in the "
        "logs, a sloppy early stage can cause downstream false completes.\n"
        "\n"
        "6. SUGGESTIONS FOR GPT — Based on what you see, briefly suggest which "
        "components' relative reward weights should change and why.  "
        "Refer to components by stage and index number.\n"
        "\n"
        "IMPORTANT CONSTRAINTS ON SUGGESTIONS:\n"
        "- You may ONLY suggest changes to reward-component Weights.\n"
        "- You CANNOT suggest changing target distances, target positions, "
        "stage ordering, transition logic, or constraint definitions.\n"
        "- Weight bounds: [1.0, 10.0]. Reward is normalized by sum(weights), so "
        "recommend relative tradeoffs rather than raising every component.\n"
    )


GEMINI_OUTPUT_INSTRUCTIONS = (
    "\n"
    "========================\n"
    "OUTPUT FORMAT (follow exactly)\n"
    "========================\n"
    "Return a compact, structured plain-text summary.  "
    "Do NOT return JSON.  Use the headings below.\n"
    "\n"
    "stage_progression:\n"
    "  List each stage the robot reaches and approximate frame/time ranges.\n"
    "  Example: 'Stage 0 (approach): frames 0-35. Stage 1 (grasp): frames 36-80.'\n"
    "\n"
    "env_consistency: (1-2 sentences)\n"
    "  Across the tiled grid of environments, do most behave similarly or are\n"
    "  there outliers?  How many tiles appear to succeed vs fail?\n"
    "\n"
    "stats_vs_video_reconciliation:\n"
    "  Compare what you see in the video against the quantitative stats above.\n"
    "  If there is a discrepancy (e.g. video looks successful but stats say 0%\n"
    "  success), EXPLICITLY NOTE it and defer to the stats — the video sample\n"
    "  likely contains lucky seeds that are not representative.\n"
    "\n"
    "achieved_goal: yes / no / uncertain\n"
    "  Did the REAL task goal get achieved?  Answer based on BOTH the video AND\n"
    "  the stats.  If the stats say 0% success, answer 'no' even if the video\n"
    "  sample looks successful, and explain the discrepancy.\n"
    "\n"
    "classification: true_success / false_complete / incomplete\n"
    "  true_success = task goal achieved.  false_complete = all stages appear done "
    "but goal NOT achieved.  incomplete = robot did not finish all stages.\n"
    "  Classify based on BOTH the stats AND the video.  Stats take precedence.\n"
    "\n"
    "key_events: (3-6 bullets)\n"
    "  Describe the major physical events you observe.  Be concrete.\n"
    "\n"
    "constraint_assessment:\n"
    "  For each stage reached, note which constraint components (by index) "
    "look satisfied, marginal, or violated.\n"
    "  If per-component satisfaction stats are provided, cross-check and note\n"
    "  discrepancies (but remember stats may be sparse with few training epochs).\n"
    "\n"
    "failure_mode:\n"
    "  The earliest stage where behaviour is wrong/sloppy, and what specifically "
    "goes wrong.  Or 'none' if the rollout looks clean.\n"
    "  If stats show stage transition counts, use them to identify bottlenecks.\n"
    "\n"
    "false_complete_risk:\n"
    "  Does the robot appear to 'finish' without truly succeeding?  "
    "Which stage enabled this?  What physical evidence tells you?\n"
    "\n"
    "suggestions_for_tuning: (2-4 bullets)\n"
    "  Which stage/component reward weights should GPT adjust, and in which\n"
    "  direction (increase/decrease).\n"
    "  ONLY suggest changes to Weights. Do NOT suggest changing targets,\n"
    "  distances, stage logic, or constraint definitions.\n"
    "  Bounds: weight [1.0, 10.0]. Reward is normalized by sum(weights), so\n"
    "  recommend relative tradeoffs rather than increasing everything.\n"
)


def build_gemini_prompt(
    *,
    task_description: str,
    stage_count: int,
    stage_descriptions: str,
    constraints_explanations_by_stage: str,
    current_weights_by_stage: dict[int, dict[int, float]] | str | None = None,
    true_success_rate: str | float | None = None,
    false_complete_rate: str | float | None = None,
    constraint_stats_text: str | None = None,
    stage_transition_text: str | None = None,
    num_envs_total: int | None = None,
    num_envs_in_video: int | None = None,
    iteration_context: str | None = None,
) -> str:
    """Build the complete Gemini prompt (context + output instructions).

    This is the convenience function that callers should use.
    """
    ctx = build_gemini_context(
        task_description=task_description,
        stage_count=stage_count,
        stage_descriptions=stage_descriptions,
        constraints_explanations_by_stage=constraints_explanations_by_stage,
        current_weights_by_stage=current_weights_by_stage,
        true_success_rate=true_success_rate,
        false_complete_rate=false_complete_rate,
        constraint_stats_text=constraint_stats_text,
        stage_transition_text=stage_transition_text,
        num_envs_total=num_envs_total,
        num_envs_in_video=num_envs_in_video,
        iteration_context=iteration_context,
    )
    return ctx + GEMINI_OUTPUT_INSTRUCTIONS
