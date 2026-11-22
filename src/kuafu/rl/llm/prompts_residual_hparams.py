"""Prompt templates for residual-RL hyperparameter search."""

from __future__ import annotations


RESIDUAL_HPARAM_TUNING_FORMATTING_INSTRUCTIONS = """
You tune residual-RL agent hyperparameters for a manipulation task.

Allowed edits only: params.env.clip_actions, params.config.residual_rl.residual_bounds,
params.config.residual_rl.clip_composed_actions, params.config.residual_rl.alpha_final,
and params.config.residual_rl.alpha_warmup_epochs. All reward, constraint, task, stage,
robot, and BC-prior settings are fixed.

The prompt gives full detectable_objects JSON, short stage labels, action-dimension labels,
outcome/stage-satisfaction stats, compact TensorBoard scalar summaries, and when available
a 240x240 image of Episode/Kuafu/success_rate_window.

Collapse definition:
- A collapse requires the latest post-warmup success_rate_window to be at least 50% lower
  than the previous post-warmup peak: peak_to_latest_drop_fraction >= 0.5.
- Do not classify smaller late dips as collapse. If collapse_after_peak is false, explicitly
  mention that the curve may be a normal run unless other evidence strongly indicates instability.
- Inspect the attached success_rate_window plot, when present, and use the visual trend to
  sanity-check the numeric summary before deciding.

The scalar summary includes:
- Episode/Kuafu/success_rate_window after critic_warmup_epochs only, including first valid value,
  stable peak, collapse onset, latest value, peak-to-latest drop, and 9 or fewer trend samples
- Episode/Kuafu/false_complete_rate
- Episode/Kuafu/timeout_rate
- Kuafu/rollout_step_reward_mean_window/iter
- Kuafu/rollout_epoch_step_reward_mean/iter

Use the action-dimension labels when editing residual_bounds. residual_bounds[i] controls the
residual range for action dimension i.

Decision policy:
- First read the success_rate_window trend samples and inspect the attached plot, if present.
- Use collapse_onset_step, stable_peak_step, latest_post_warmup, and peak_to_latest_drop_fraction
  to decide whether the 50% collapse rule is actually met.
- If collapse_after_peak is true under the 50% rule, slow down residual authority by increasing
  alpha_warmup_epochs, up to 1000.
- If collapse happens near or after the current warmup limit, also reduce alpha_final or tighten
  the action dimensions most likely causing drift.
- If collapse_after_peak is false, do not reduce aggression solely because the latest value is
  slightly below the peak; say it may be a normal run and use outcome stats, plot shape, false
  completions, timeouts, and rewards to choose minimal changes.
- Treat enabling clip_composed_actions, reducing clip_actions, reducing alpha_final, and tightening
  residual_bounds as restrictive safety moves. Use them only when there is clear collapse,
  instability, reward-proxy drift, high false-complete rate, or direct action-clipping evidence.
- If there is no clear collapse/instability, do not make action clipping more restrictive. Preserve
  the current clip_composed_actions value, or set it false/unlock clipping if stable success appears
  limited by composed-action clipping.
- If success is low without a peak/drop pattern and rewards are flat, allow a small amount more
  correction: consider increasing clip_actions up to its hard limit, loosening the relevant
  residual_bounds within bounds, or keeping alpha_final high rather than reducing it.
- If reward improves but success_rate_window drops sharply, treat that as reward-proxy drift and make the residual safer.
- Change the fewest hyperparameters needed; prefer alpha_warmup_epochs/alpha_final changes before broad residual_bounds changes when the signature is alpha-collapse.

Hard numeric bounds:
- clip_actions must be finite and in [0.5, 2.0].
- alpha_final must be finite and in [0.05, 1.5].
- alpha_warmup_epochs must be an integer in [0, 1000].
- residual_bounds must be the same length as the current list.
- each residual_bounds value must be finite and in [0.02, 1.0].
- Per-iteration step limit: each numeric value should stay within [0.5 * current, 1.5 * current],
  unless the current value is outside the hard numeric bounds and must be corrected.

Return ONLY one valid JSON object with exactly:
"Diagnosis", "Reasoning", "Hyperparameters"

"Diagnosis": summarize whether the run appears under-corrected, over-corrected, unstable, or stable.
"Reasoning": justify each changed hyperparameter and why unchanged values should remain unchanged.
"Hyperparameters" MUST contain exactly these keys:
  "clip_actions", "residual_bounds", "clip_composed_actions", "alpha_final", "alpha_warmup_epochs"
Do not add, remove, or rename keys.
"""

RESIDUAL_HPARAM_PROMPT = """
Task:
{task_description}

Detectable objects:
{detectable_objects}

Stages:
{stage_labels}

Action mode:
{action_mode}

Action dimension labels:
{action_dimension_labels}

Current residual-RL hyperparameters:
{current_hyperparameters}

Outcome stats:
{outcome_stats}

Stage satisfaction summary:
{stage_satisfaction_summary}

Exact TensorBoard scalar summaries:
{reward_success_event_summary}

Success-rate plot:
{success_rate_plot_summary}
"""
