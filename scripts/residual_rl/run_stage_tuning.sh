#!/usr/bin/env bash
# =============================================================================
# Residual-RL: Stage-wise LLM Tuning (from scratch)
# =============================================================================
#
# Legacy stage-wise residual-RL pipeline. Constraint tolerances are assigned by
# code before the first rollout. The LLM proposes only per-stage residual
# weights, workers train them in parallel, and the best proposal advances. Once
# all stages pass, an optional global end-to-end pass refines the weights.
#
# Outputs (all co-located):
#   $KUAFU_ARTIFACT_ROOT/residual_rl/<task>/<timestamp>/
#       best.json, prompt_context.json, rl_runs/, tb/, checkpoints/, ...
# =============================================================================
set -euo pipefail

# ── API Keys ─────────────────────────────────────────────────────────────────
# OpenAI: required — the LLM that proposes reward-weight changes.
# Gemini: optional — summarises rollout videos into text for the LLM.
#   Set here, or export them before running.  Existing env vars are kept.
export OPENAI_API_KEY="${OPENAI_API_KEY:?Set OPENAI_API_KEY}"
export GEMINI_API_KEY="${GEMINI_API_KEY:-}"      # ok to be empty (video text skipped)

# ── Task ─────────────────────────────────────────────────────────────────────
TASK="FR3-FinRay-Table-Drawer"

# ── Hardware ─────────────────────────────────────────────────────────────────
DEVICE="cuda:0"
NUM_ENVS=8192           # Parallel environments per worker.

# ── Parallelism ──────────────────────────────────────────────────────────────
# Each LLM call produces NUM_WORKERS proposals; all train in parallel.
NUM_WORKERS=3

# ── RL training length (progressive) ────────────────────────────────────────
# Epochs ramp linearly: iter 0 → INIT, iter 1 → INIT+INCREMENT, … capped at MAX.
INIT_TRAIN_EPOCH=50
MAX_TRAIN_EPOCH=100
EPOCH_TRAIN_INCREMENT=5

# ── Stage tuning budget ─────────────────────────────────────────────────────
# MAXIMUN_PROPOSALS = total LLM iterations allowed across *all* stages.
# The loop runs while (iteration_count < MAXIMUN_PROPOSALS && stage_id < stage_count).
# With NUM_WORKERS=3, 25 proposals ≈ 25 iterations × 3 = 75 actual RL trainings.
MAXIMUN_PROPOSALS=25

# A stage is "passed" when ≥ STAGE_SUCCESS_FRACTION of envs advance beyond it.
STAGE_SUCCESS_FRACTION=0.5

# Stop tuning entirely when (true_success + false_complete) / total_episodes ≥ this cap.
STAGE_ADVANCE_COMPLETION_RATE_CAP=0.5

# ── Global finetune (after all stages pass) ──────────────────────────────────
# Set to 0 to skip.  LLM tunes ALL stages jointly with video evidence.
GLOBAL_FINETUNE_ITERS=5

# Early-stop when true_success_rate >= STAGE_SUCCESS_FRACTION AND
# false_complete_rate / true_success_rate <= this ratio.
GLOBAL_FINETUNE_MAX_FALSE_TO_TRUE_RATIO=0.5

# ── LLM ──────────────────────────────────────────────────────────────────────
GPT_MODEL="gpt-5.5"
TEMPERATURE=1.0

# ── Gemini (video → text) ───────────────────────────────────────────────────
# Model used for summarising rollout mp4s into text that goes into the LLM prompt.
# Only active during global finetune when VIDEO_FEEDBACK is on.
GEMINI_MODEL="gemini-2.0-flash"    # e.g. gemini-2.0-flash, gemini-2.5-pro

# ── Video feedback ───────────────────────────────────────────────────────────
# Video is only used in global finetune (not stage tuning).
VIDEO_FEEDBACK="--video_feedback"   # "--no-video_feedback" to disable
VIDEO_LENGTH=200                    # Steps to record per rollout.
VIDEO_NUM_ENVS_TO_RECORD=4          # Number of env cameras to record.

# ── Reward reflection subsampling ────────────────────────────────────────────
# After each RL training, TensorBoard scalars (reward trajectories, constraint
# satisfaction rates) are formatted as compact arrays and shown to the LLM.
# FEEDBACK_SUBSAMPLING is the stride: with 100 logged steps and stride=10,
# the LLM sees every 10th value → 10 data points.  Lower = more detail in
# the prompt; higher = smaller prompt.
FEEDBACK_SUBSAMPLING=10

# ── Bootstrap / warm-start (optional) ───────────────────────────────────────
# Eval-only checkpoint: used once to seed rollout stats without further training.
INITIAL_EVAL_CHECKPOINT=""
# Warm-start source for training: choose exactly one checkpoint or prior, or leave both empty for scratch.
INITIAL_CHECKPOINT=""               # .pth to warm-start RL training from
INITIAL_PRIOR_CHECKPOINT=""         # BC prior checkpoint/run dir to seed RL training from
INITIAL_PRIOR_OBS_STATS_COUNT="1000000"
INITIAL_PRIOR_VERBOSE=""
INITIAL_WEIGHTS_PARAMS=""           # JSON with per-stage weights/params (same format as best.json "stages")
# Restart each LLM iteration from the selected warm-start source instead of scratch.
RESTART_EACH_ITER_FROM_INITIAL_WARM_START="" # "--restart_each_iter_from_initial_warm_start" to enable

# ── Misc ─────────────────────────────────────────────────────────────────────
ENV_SEED=42
RENDER=""                           # "--render" for non-headless

# =============================================================================
if [[ -n "${INITIAL_CHECKPOINT}" && -n "${INITIAL_PRIOR_CHECKPOINT}" ]]; then
    echo "ERROR: INITIAL_CHECKPOINT and INITIAL_PRIOR_CHECKPOINT are mutually exclusive." >&2
    exit 1
fi

if [[ -n "${RESTART_EACH_ITER_FROM_INITIAL_WARM_START}" && -z "${INITIAL_CHECKPOINT}" && -z "${INITIAL_PRIOR_CHECKPOINT}" ]]; then
    echo "ERROR: RESTART_EACH_ITER_FROM_INITIAL_WARM_START requires INITIAL_CHECKPOINT or INITIAL_PRIOR_CHECKPOINT." >&2
    exit 1
fi

CMD=(
    python scripts/residual_rl/train.py
    --task              "${TASK}"
    --device            "${DEVICE}"
    --num_workers       "${NUM_WORKERS}"
    --num_envs          "${NUM_ENVS}"
    --env_seed          "${ENV_SEED}"

    --init_train_epoch      "${INIT_TRAIN_EPOCH}"
    --max_train_epoch       "${MAX_TRAIN_EPOCH}"
    --epoch_train_increment "${EPOCH_TRAIN_INCREMENT}"
    --feedback_subsampling  "${FEEDBACK_SUBSAMPLING}"

    --stage_success_fraction              "${STAGE_SUCCESS_FRACTION}"
    --maximun_proposals                   "${MAXIMUN_PROPOSALS}"
    --stage_advance_completion_rate_cap   "${STAGE_ADVANCE_COMPLETION_RATE_CAP}"

    --global_finetune_iters               "${GLOBAL_FINETUNE_ITERS}"
    --global_finetune_max_false_to_true_ratio "${GLOBAL_FINETUNE_MAX_FALSE_TO_TRUE_RATIO}"

    --gpt_model         "${GPT_MODEL}"
    --temperature       "${TEMPERATURE}"
    --gemini_model      "${GEMINI_MODEL}"

    ${VIDEO_FEEDBACK}
    --video_length              "${VIDEO_LENGTH}"
    --video_num_envs_to_record  "${VIDEO_NUM_ENVS_TO_RECORD}"

    ${RENDER}
)

[[ -n "${INITIAL_EVAL_CHECKPOINT}" ]]            && CMD+=(--initial_eval_checkpoint "${INITIAL_EVAL_CHECKPOINT}")
[[ -n "${INITIAL_CHECKPOINT}" ]]               && CMD+=(--initial_checkpoint "${INITIAL_CHECKPOINT}")
[[ -n "${INITIAL_PRIOR_CHECKPOINT}" ]]         && CMD+=(--initial_prior_checkpoint "${INITIAL_PRIOR_CHECKPOINT}")
[[ -n "${INITIAL_PRIOR_OBS_STATS_COUNT}" ]]    && CMD+=(--initial_prior_obs_stats_count "${INITIAL_PRIOR_OBS_STATS_COUNT}")
[[ -n "${INITIAL_PRIOR_VERBOSE}" ]]            && CMD+=(${INITIAL_PRIOR_VERBOSE})
[[ -n "${INITIAL_WEIGHTS_PARAMS}" ]]           && CMD+=(--initial_weights_params "${INITIAL_WEIGHTS_PARAMS}")
[[ -n "${RESTART_EACH_ITER_FROM_INITIAL_WARM_START}" ]] && CMD+=(${RESTART_EACH_ITER_FROM_INITIAL_WARM_START})

echo "========================================"
echo " Residual-RL: Stage-wise LLM Tuning"
echo "========================================"
echo "Task:              ${TASK}"
echo "Workers:           ${NUM_WORKERS}  (× ${MAXIMUN_PROPOSALS} max iterations)"
echo "Epochs:            ${INIT_TRAIN_EPOCH} → ${MAX_TRAIN_EPOCH} (+${EPOCH_TRAIN_INCREMENT})"
echo "Stage threshold:   ${STAGE_SUCCESS_FRACTION}"
echo "Global finetune:   ${GLOBAL_FINETUNE_ITERS} iters"
echo "LLM:               ${GPT_MODEL} (temp=${TEMPERATURE})"
echo "Gemini:            ${GEMINI_MODEL}"
echo "Initial eval ckpt: ${INITIAL_EVAL_CHECKPOINT:-<none>}"
echo "Warm-start ckpt:   ${INITIAL_CHECKPOINT:-<none>}"
echo "Warm-start prior:  ${INITIAL_PRIOR_CHECKPOINT:-<none>}"
echo "Warm-start weights:${INITIAL_WEIGHTS_PARAMS:-<none>}"
echo "Restart from warm: ${RESTART_EACH_ITER_FROM_INITIAL_WARM_START:-<off>}"
echo "========================================"
echo ""
echo "Running: ${CMD[*]}"
echo ""

exec "${CMD[@]}"
