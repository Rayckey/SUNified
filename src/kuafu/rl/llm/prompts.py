"""Prompt templates for residual-RL LLM interactions.

Constraint tolerances never appear in these templates. They are derived by
``kuafu.skills.constraints.tolerance_policy`` before weight selection and then kept
fixed throughout later tuning.
"""


INIT_SWEEP_INSTRUCTIONS = """
You initialize reward-component Weights for ONE stage before RL.
Adjust ONLY reward-component Weights.

Do NOT change reward/constraint definitions, indices, ordering, logic
structure, targets, distances, or any other configuration.

Use only the provided task/stage semantics, observed objects, adjacent-stage
context, component explanations, and current weights.
Do NOT assume rollout statistics, success rates, reward traces, or hidden task
facts.

Weights:
- Set weights by semantic role, not performance. Reward is normalized by sum(weights), so tune relative ratios.
- Default init-sweep weight is 1.0. Use simple ratios.
- Broad free-space progress, broad-body ordinary pose holds, broad-body top-down/side orientation maintenance, and non-critical maintenance terms usually stay in [1.0, 1.5].
- For broad_body tasks with no small/slender, receiver/opening, constrained-feature, or final-object-validity requirement, avoid weights above 2.0. Grasp/object-state may be 2.0 if losing the state invalidates all progress, but do not inflate it to 2.5-3.0 on init sweep.
- Direct fit/contact, small-cross-section grasp-lock, constrained-feature manipulation, precision directional progress, mission-critical angular constraints, and final object validity may use [1.5, 3.0] depending on whether losing that component invalidates the stage.
- Open-gripper setup/contact orientation terms and broad sliding-pull orientation terms are usually [1.0, 1.5] and should not be weighted as mission-critical angular constraints unless the current stage truly requires angular fit, insertion, final object validity, or a force/torque axis.
- Gripper/object state components are important when they define contact, release, or grasp validity; usually use 1.5, use 2.0 for broad-body state validity when losing state invalidates progress, and reserve 2.5-3.0 for small/slender, receiver/opening, constrained-feature, or final-object-validity cases.
- Release_guard robot/tool pose holds usually stay in [1.0, 1.5]; do not give them narrow-feature/manipulation-level weights unless the stage explicitly says the held tool pose itself is the contact-validity condition.
- Avoid weight inflation: do not set every core component above 2.0 on the init sweep, and do not make broad-body orientation/maintenance terms higher than the primary progress/contact term.
- weight must be finite and in [0.1, 10.0]. Correct any out-of-bound value.

Return ONLY one valid JSON object with exactly:
"Diagnosis", "Reasoning", "Weights"

"Diagnosis": briefly list the semantic role of each tunable component.
"Reasoning": briefly justify the labels and relative weight ratios.
"Weights" MUST contain EXACTLY the same indices as current weights.
"""


SINGLE_BOTH_TUNING_FORMATTING_INSTRUCTIONS = """
You tune ONE stage by adjusting ONLY reward-component Weights.

Do NOT change reward/constraint definitions, indices, ordering, logic structure, or any other configuration.

Use stage semantics and population stats.
Reward is normalized by sum(weights), so tune relative ratios.

For each component, identify its semantic role and main blocker:
- poor progress / control
- weight tradeoff
- required state not maintained

Judge meaningful completion by whether the resulting state is actually suitable for the intended interaction, not merely whether predicates became true.
Do NOT use weighted reward magnitude alone as evidence of geometric closeness.

Return ONLY one valid JSON object (no markdown, no code fences, no extra text).
Top-level keys MUST be EXACTLY:
  "Diagnosis", "Reasoning", "Weights"

- "Weights" MUST contain EXACTLY the same indices as the provided current weights.
- Do NOT add/remove/rename indices.

Weights:
- finite, final value in [0.1, 10.0]
- per-index step limit: new_w[i] in [0.5*old_w[i], 2.0*old_w[i]]
- if a current weight lies outside the hard bounds, you MUST correct it into the hard-bound interval on this pass

Optimization priority:
1) Increase meaningful stage completion.
2) Preserve semantic validity of the interaction.
3) Improve precision only if (1) and (2) do not worsen.
4) Improve efficiency only if (1), (2), and (3) do not worsen.
"""


MULTI_STAGE_SANITY_TUNING_FORMATTING_INSTRUCTIONS = """
You tune ALL stages jointly to improve true end-to-end task success.
Adjust ONLY reward-component Weights.

Use only the provided task semantics, stage descriptions, component
explanations, current Weights, population stats, and optional videos.
Do NOT change reward/constraint definitions, indices, ordering, stage ordering,
transition logic, targets, distances, or any other configuration.

Evidence priority:
1) Population stats for rates, counts, bottlenecks, and false-complete prevalence.
2) Task/stage semantics for what true success means.
3) Videos only for qualitative mechanism; if videos conflict with stats, trust stats.

Decision procedure:
1) Identify the true-success failure mode: incomplete, false complete, or low true success.
2) Blame the earliest causal stage/component supported by stats/videos.
3) Classify the blocker as poor progress/control, a weight tradeoff, or a required state not maintained.
4) Edit the fewest stages/components needed. Do not patch downstream symptoms when an upstream stage caused them.

Weights:
- Rewards are normalized by sum(weights), so tune ratios.
- weight must be finite and in [0.1, 10.0].
- Per-index step limit: new_w[s][i] in [0.5*old_w[s][i], 2.0*old_w[s][i]], unless correcting an out-of-bound current value.

Optimization priority:
1) Increase true_success_rate and reduce false_complete_rate.
2) Preserve semantic validity.
3) Improve precision only if it does not hurt (1) or (2).
4) Improve efficiency only if it does not hurt (1)-(3).

Return ONLY one valid JSON object with exactly:
"Diagnosis", "Reasoning", "Weights"

Integrity:
- Include ALL provided stage keys in "Weights", even unchanged ones.
- Within each stage, include EXACTLY the same component indices as provided.
- Do not add/remove/rename any key.

"Diagnosis": state rates/failure pattern, blamed stage/component(s), and blocker type.
"Reasoning": give the earliest-cause rationale and explain each changed weight briefly.
"""


SYSTEM_PROMPT = """
You are a supervisor for a reinforcement learning agent. The agent is training on a multi-stage task where each stage has its own set of reward components.

These reward components are based on cost terms that have been effective in prior optimal control settings. Do not add, remove, or redefine reward components.

Each reward component *i* defines:

1. a goal requirement,
2. a shaping reward function that increases as the agent gets closer to meeting that requirement, and
3. a constraint check that measures whether the goal requirement has been met.

A stage is considered complete when all component constraints are satisfied.

Task names, object labels, stage numbers, and component indices identify the
provided data; they do not determine which component deserves more weight.
Base each decision on the current physical requirement and, when provided,
measured evidence. Do not infer a stored solution from a familiar name.

Short rule examples, not task solutions:
- Weights [1, 2] and [2, 4] express the same relative preference after normalization.
- A pose hold during release and a pose requirement that validates the final
  object state can have different roles even when both use the same operator.

Physical roles used by the weight rules:
- Broad body: roughly solid/boxlike/round grasp cross-section >= 0.05 m,
  without a narrow axis or receiver-fit requirement.
- Small solid body: cross-section above 0.03 m and below 0.05 m; size alone
  does not make orientation mission-critical.
- Small/slender: cross-section <= 0.03 m, or a long/thin/axis-critical shape.
- Release guard: robot/tool pose held while opening after a supported result;
  distinguish this from a component checking the object's final validity.
- Precision progress: the traveled amount itself validates current contact,
  fit, or actuation; free-space waypoints are not precision progress.
"""


INIT_SWEEP_PROMPT = """
Overall task description: {task_description}

Detectable objects: {detectable_objects}

Stage index/total number of stages: {stage_id}/{stage_count}

Stage description: {atomic_action}

Current subtask constraint ops/args:
{current_constraint_ops}

Adjacent subtask context (for transition semantics only; tune ONLY the current stage):
Previous subtask:
{previous_stage_context}

Next subtask:
{next_stage_context}

Index-ordered reward component explanations:
{constraints_explanations}

Current weights:
{current_weights}
"""


SINGLE_BOTH_PROMPT = """
Overall task description: {task_description}

Detectable objects: {detectable_objects}

Stage index/total number of stages: {stage_id}/{stage_count}

Stage description: {atomic_action}

Index-ordered reward component explanations:
{constraints_explanations}

Current weights:
{current_weights}

========================
END-TO-END PERFORMANCE (ALL ENVS)
========================
{end_to_end_feedback}

Stage satisfaction times:
{stage_satisfaction_times}

Per-component constraint satisfaction rates:
{constraint_satisfaction_rates}

Per-component reward values (multiplied by weights):
{reward_values}
"""


MULTI_STAGE_END2END_PROMPT = """
Overall task description:
{task_description}

Detectable objects: {detectable_objects}

Total number of stages:
{stage_count}

Index-ordered stage descriptions:
{stage_descriptions_reasoning}

Index-ordered reward/constraint component explanations
(grouped by stage, then component index):
{constraints_explanations_by_stage}

Current weights (grouped by stage, then component index):
{current_weights_by_stage}

========================
END-TO-END PERFORMANCE (ALL ENVS)
========================
Goal of this tuning pass:
- Strictly maximize true_success_rate
- Treat false completions as failures
- Stage/proxy completion cannot justify a true-success regression

End-to-end rates:
- true_success_rate: {true_success_rate}
- false_complete_rate: {false_complete_rate}

Termination and incumbent context:
{true_success_context}

Per-stage satisfaction time summary (index-ordered stages):
{per_stage_satisfaction_time_stats}

Video rollouts:
{video_rollouts}
"""


VIDEO_EVIDENCE_PROMPT = """
========================
VIDEO EVIDENCE (MULTI-CAMERA GRID)
========================
Below you will find a Gemini-generated analysis of multi-camera rollout videos.
Gemini watched tiled grids (one per camera angle) of a SMALL SAMPLE of parallel
environments (typically 9 out of 1024+) and produced a textual summary.

IMPORTANT — LIMITATIONS OF THE VIDEO ANALYSIS:

1. SMALL SAMPLE: The video shows only ~9 environments. Training runs 1024+.
   The sampled environments may not be representative of the overall population.
   If the video shows success but the stats above say 0% success across all envs,
   the video sample contains lucky seeds — trust the stats.

2. PIXEL-ONLY: Gemini can only see pixel-level observations. It does NOT have
   access to the internal constraint states, reward values, or stage-machine logs
   that YOU have. It may misidentify stages, misjudge distances, or confuse
   visually ambiguous events.

3. SUGGESTION SCOPE: Gemini may suggest changes that are outside your control
   (e.g., changing target distances, adding stages). Ignore any such suggestions.
   You may ONLY change reward-component Weights.

HOW TO USE THE VIDEO ANALYSIS:
- Use it for QUALITATIVE insight only: movement patterns, failure modes, timing.
- For QUANTITATIVE facts (which stages pass, satisfaction rates, success counts),
  ALWAYS prefer the stats sections above.
- If Gemini says "achieved_goal: yes" but per-stage constraint stats show 0% met
  rate on a component, the component is NOT met — Gemini is wrong about those envs.
  Identify the real bottleneck stage from the stats and tune its relative weight.

{video_rollouts}
"""


FAILURE_FEEDBACK = """
The previous response caused an error:
{traceback_msg}

Previous response:
{previous_response}

Please propose a corrected response.
"""
