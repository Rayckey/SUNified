# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Format env_learned stats for LLM prompts (e.g. SINGLE_WEIGHT_PROMPT).

Expects dict shapes from Kuafu tasks/env_learned.py:
- get_stage_transition_stats() -> dict[int, {"avg_steps", "samples"}]
  - "avg_steps": average steps to complete this stage (may be None)
  - "samples": number of transition-event samples used to compute avg_steps
- get_highest_stage_reached_per_env() -> list[int]
  - highest_stage_reached_per_env[i] is the highest stage index reached by env i
- get_constraint_met_rate_stats() -> dict[int, dict[int, {"rate", "met", "total", "reward"}]]
  - aggregated over envs that terminated while stuck in each stage
"""

from __future__ import annotations

import math
from typing import Any


def _to_float_scalar(x: object | None) -> float | None:
    """Best-effort convert a python/torch/numpy scalar to float."""
    if x is None:
        return None
    item = getattr(x, "item", None)
    if callable(item):
        try:
            x = item()
        except Exception:
            pass
    try:
        return float(x)  # type: ignore[arg-type]
    except Exception:
        return None


def _format_rate(
    numer: float | None,
    denom: float | None,
    *,
    precision: int = 2,
    include_counts: bool = True,
) -> str:
    """Format rate as '12.34% (n/d)' or 'n/a'."""
    if numer is None or denom is None or denom <= 0:
        return "n/a"
    rate = numer / denom
    if include_counts:
        n = int(round(numer))
        d = int(round(denom))
        return f"{rate:.{precision}%} ({n}/{d})"
    return f"{rate:.{precision}%}"


def format_end_to_end_rate_stats(
    *,
    true_success_count: object | None,
    false_complete_count: object | None,
    total_episodes: object | None = None,
    precision: int = 2,
    include_counts: bool = True,
) -> dict[str, str]:
    """Format end-to-end rates for `MULTI_STAGE_END2END_PROMPT`.

    Args:
        true_success_count: Number of true successes (task truly achieved).
        false_complete_count: Number of false completes (env marked complete but task not truly achieved).
        total_episodes: Denominator. If None, defaults to (true_success_count + false_complete_count).
            If you want timeouts/truncations included, pass total_episodes explicitly.
        precision: Percent formatting precision (e.g. 2 -> 12.34%).
        include_counts: If True, append '(n/d)'.

    Returns:
        Dict matching prompt placeholders:
        - "true_success_rate"
        - "false_complete_rate"
    """
    t = _to_float_scalar(true_success_count)
    f = _to_float_scalar(false_complete_count)
    denom = _to_float_scalar(total_episodes)
    if denom is None:
        denom = (t or 0.0) + (f or 0.0)
    return {
        "true_success_rate": _format_rate(t, denom, precision=precision, include_counts=include_counts),
        "false_complete_rate": _format_rate(f, denom, precision=precision, include_counts=include_counts),
    }


def format_stage_satisfaction_times(
    stage_transition_stats: dict[int, dict[str, Any]] | None,
    *,
    highest_stage_reached_per_env: list[int] | None,
    stage_id: int | None = None,
    num_env: int | None = None,
) -> str:
    """Format env_learned.get_stage_transition_stats() for prompt text.

    If stage_id is provided, only that stage is reported; otherwise all stages are included.
    If num_env is set and samples < 5% of num_env, avg_steps is not reported and the line
    shows "not enough data".
    """
    if stage_transition_stats is None or len(stage_transition_stats) == 0:
        return "No stage transition stats available yet."
    if highest_stage_reached_per_env is None:
        return "No highest_stage_reached_per_env available yet."

    min_count = max(1, math.ceil(0.05 * num_env)) if num_env is not None and num_env > 0 else 0
    lines = []
    for s, d in sorted(stage_transition_stats.items()):
        if stage_id is not None and s != stage_id:
            continue
        avg = d["avg_steps"]
        samples = int(d["samples"])
        envs_passed = sum(1 for h in highest_stage_reached_per_env if int(h) > int(s))
        if num_env is not None and num_env > 0 and samples < min_count:
            lines.append(
                f"  Stage {s}: not enough data (samples={samples}, need at least {min_count}; envs_passed={envs_passed})"
            )
        elif avg is not None and samples > 0:
            lines.append(f"  Stage {s}: avg_steps={avg:.1f}, samples={samples}, envs_passed={envs_passed}")
        else:
            lines.append(f"  Stage {s}: avg_steps unavailable (samples={samples}), envs_passed={envs_passed}")
    return "\n".join(lines) if lines else "No stage transition stats available yet."


def format_constraint_satisfaction_rates(
    constraint_met_rate_stats: dict[int, dict[int, dict[str, Any]]] | None,
    stage_id: int | None = None,
) -> str:
    """Format env_learned.get_constraint_met_rate_stats() for prompt text (rate only)."""
    if not constraint_met_rate_stats:
        return "No stuck terminal envs recorded for this stage yet."
    lines = []
    for sid, cdict in sorted(constraint_met_rate_stats.items()):
        if stage_id is not None and sid != stage_id:
            continue
        for cid, stats in sorted(cdict.items()):
            rate = stats["rate"]
            met = stats.get("met", 0)
            total = stats["total"]
            lines.append(
                f"  Stage {sid} component {cid}: rate={rate:.2%} ({met}/{total})"
            )
    return "\n".join(lines) if lines else "No stuck terminal envs recorded for this stage yet."


def format_constraint_unsatisfaction_rates(
    constraint_met_rate_stats: dict[int, dict[int, dict[str, Any]]] | None,
    stage_id: int | None = None,
) -> str:
    """Format per-component timeout/stuck non-satisfaction rates for prompt text."""
    if not constraint_met_rate_stats:
        return "No stuck terminal envs recorded for this stage yet."
    lines = []
    for sid, cdict in sorted(constraint_met_rate_stats.items()):
        if stage_id is not None and sid != stage_id:
            continue
        for cid, stats in sorted(cdict.items()):
            rate = float(stats["rate"])
            met = int(stats.get("met", 0))
            total = int(stats["total"])
            not_met = max(0, total - met)
            not_rate = max(0.0, min(1.0, 1.0 - rate))
            lines.append(
                f"  Stage {sid} component {cid}: not_satisfied={not_rate:.2%} ({not_met}/{total})"
            )
    return "\n".join(lines) if lines else "No stuck terminal envs recorded for this stage yet."


def format_reward_values(
    constraint_met_rate_stats: dict[int, dict[int, dict[str, Any]]] | None,
    stage_id: int | None = None,
) -> str:
    """Format per-component reward values from env_learned.get_constraint_stats() for prompt text.

    Note: These are weighted reward contributions (raw residual × weight per constraint),
    not raw 0-1 rates. Values can exceed 1 when the constraint weight is > 1.
    They are averaged over stuck terminal envs for the given stage/constraint.
    """
    if not constraint_met_rate_stats:
        return "No stuck terminal envs recorded for this stage yet."
    lines = []
    for sid, cdict in sorted(constraint_met_rate_stats.items()):
        if stage_id is not None and sid != stage_id:
            continue
        for cid, stats in sorted(cdict.items()):
            reward = stats["reward"]
            lines.append(f"  Stage {sid} component {cid}: reward={reward:.4f}")
    return "\n".join(lines) if lines else "No stuck terminal envs recorded for this stage yet."


def format_single_weight_feedback_block(
    stage_transition_stats: dict[int, dict[str, Any]] | None,
    highest_stage_reached_per_env: list[int] | None,
    constraint_met_rate_stats: dict[int, dict[int, dict[str, Any]]] | None,
    stage_id: int | None = None,
) -> str:
    """Format both env stats into the block used after a run (Stage satisfaction times + Per-component rates)."""
    st = format_stage_satisfaction_times(
        stage_transition_stats,
        highest_stage_reached_per_env=highest_stage_reached_per_env,
        stage_id=stage_id,
    )
    cr = format_constraint_satisfaction_rates(constraint_met_rate_stats, stage_id=stage_id)
    if not stage_transition_stats and not constraint_met_rate_stats:
        return ""
    return (
        f"Stage satisfaction times:\n{st}\n\n"
        f"Per-component constraint satisfaction rates:\n{cr}\n\n"
    )

def format_single_run_feedback_block(
    stage_transition_stats: dict[int, dict[str, Any]] | None,
    highest_stage_reached_per_env: list[int] | None,
    constraint_met_rate_stats: dict[int, dict[int, dict[str, Any]]] | None,
    stage_id: int | None = None,
) -> str:
    """Format stage transition stats, constraint rates, and reward values for a single run (LLM feedback).
    Same structure as SINGLE_BOTH_PROMPT: stage_satisfaction_times, constraint_satisfaction_rates, reward_values.
    """
    stage_satisfaction_times = format_stage_satisfaction_times(
        stage_transition_stats,
        highest_stage_reached_per_env=highest_stage_reached_per_env,
        stage_id=stage_id,
    )
    constraint_satisfaction_rates = format_constraint_satisfaction_rates(
        constraint_met_rate_stats, stage_id=stage_id
    )
    reward_values = format_reward_values(constraint_met_rate_stats, stage_id=stage_id)
    if not stage_transition_stats and not constraint_met_rate_stats:
        return ""
    return (
        f"Stage satisfaction times:\n{stage_satisfaction_times}\n\n"
        f"Per-component constraint satisfaction rates:\n{constraint_satisfaction_rates}\n\n"
        f"Reward values:\n{reward_values}\n\n"
    )
