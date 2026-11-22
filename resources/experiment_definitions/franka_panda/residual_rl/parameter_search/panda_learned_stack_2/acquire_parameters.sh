#!/usr/bin/env bash
# =============================================================================
# Kuafu sweep — generated from Priors (see train_config.yaml in this run dir)
# Prior dir: Priors/thor/stack2_damped/stack2_chunking_10_f_damped/train_config.yaml
# =============================================================================
set -euo pipefail

# ── API Keys ─────────────────────────────────────────────────────────────────
export OPENAI_API_KEY="${OPENAI_API_KEY:?Set OPENAI_API_KEY}"
export GEMINI_API_KEY="${GEMINI_API_KEY:-}"     # optional; video summaries skipped if empty
export CUDA_VISIBLE_DEVICES=0,1,2,3

# ── Task ─────────────────────────────────────────────────────────────────────
TASK="${KUAFU_TASK:-Panda-Table-Stack-2}"

# ── Hardware ─────────────────────────────────────────────────────────────────
DEVICE="cuda:0,1,2,3"
INIT_SWEEP_DEVICE="cuda:0"
NUM_ENVS=8192

# ── Parallelism ──────────────────────────────────────────────────────────────
NUM_WORKERS=4

# ── RL training length (progressive) ────────────────────────────────────────
INIT_TRAIN_EPOCH=800
MAX_TRAIN_EPOCH=1500
EPOCH_TRAIN_INCREMENT=50

# ── Stage tuning budget ─────────────────────────────────────────────────────
MAXIMUN_PROPOSALS=5
STAGE_SUCCESS_FRACTION=0.5
STAGE_ADVANCE_COMPLETION_RATE_CAP=0.8

# ── Global finetune (after all stages pass) ─────────────────────────────────
GLOBAL_FINETUNE_ITERS=5
GLOBAL_FINETUNE_MAX_FALSE_TO_TRUE_RATIO=0.2

# ── LLM ──────────────────────────────────────────────────────────────────────
GPT_MODEL="gpt-5.5"
TEMPERATURE=1.0

# ── Gemini (video → text) ───────────────────────────────────────────────────
GEMINI_MODEL="gemini-3.1-pro-preview"

# ── Video feedback ───────────────────────────────────────────────────────────
VIDEO_FEEDBACK="--video_feedback"
VIDEO_LENGTH=400
VIDEO_NUM_ENVS_TO_RECORD=9

# ── Reward reflection subsampling ────────────────────────────────────────────
FEEDBACK_SUBSAMPLING=10

# ── Bootstrap sources (optional) ────────────────────────────────────────────
# Eval-only checkpoint: used once to seed rollout stats without further training.
INITIAL_EVAL_CHECKPOINT=""
# Warm-start source for training: pick exactly one checkpoint or prior, or leave both empty for scratch.
INITIAL_CHECKPOINT=""
INITIAL_PRIOR_CHECKPOINT="Priors/thor/stack2_damped/stack2_chunking_10_f_damped/best.pth"
INITIAL_PRIOR_OBS_STATS_COUNT="1000000"
INITIAL_PRIOR_VERBOSE=""
INITIAL_WEIGHTS_PARAMS=""
# Restart each LLM iteration from the selected warm-start source instead of scratch.
RESTART_EACH_ITER_FROM_INITIAL_WARM_START="--restart_each_iter_from_initial_warm_start"
STOP_ON_ERROR=""

# ── Misc ─────────────────────────────────────────────────────────────────────
ENV_SEED=42
RENDER=""
PRINT_PROGRAM_FUNCTIONS=""

# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCH_SCRIPT="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
PYTHON_BIN="${PYTHON_BIN:-${HOME}/miniconda3/envs/kuafu/bin/python}"

EXECUTION_ROBOT_ID="${KUAFU_EXECUTION_ROBOT_ID:-${EXECUTION_ROBOT_ID:-franka_panda}}"
ARTIFACT_ROBOT_ID="${KUAFU_ARTIFACT_ROBOT_ID:-${ARTIFACT_ROBOT_ID:-franka_panda}}"
KUAFU_SWEEP_ROBOT_ARGS=()
[[ -n "${EXECUTION_ROBOT_ID}" ]] && KUAFU_SWEEP_ROBOT_ARGS+=(--execution_robot_id "${EXECUTION_ROBOT_ID}")
[[ -n "${ARTIFACT_ROBOT_ID}" ]] && KUAFU_SWEEP_ROBOT_ARGS+=(--artifact_robot_id "${ARTIFACT_ROBOT_ID}")

if [[ -n "${INITIAL_CHECKPOINT}" && -n "${INITIAL_PRIOR_CHECKPOINT}" ]]; then
    echo "ERROR: Choose only one warm-start source: INITIAL_CHECKPOINT or INITIAL_PRIOR_CHECKPOINT." >&2
    exit 1
fi

if [[ -n "${RESTART_EACH_ITER_FROM_INITIAL_WARM_START}" && -z "${INITIAL_CHECKPOINT}" && -z "${INITIAL_PRIOR_CHECKPOINT}" ]]; then
    echo "ERROR: RESTART_EACH_ITER_FROM_INITIAL_WARM_START requires INITIAL_CHECKPOINT or INITIAL_PRIOR_CHECKPOINT." >&2
    exit 1
fi

if [[ -n "${INITIAL_WEIGHTS_PARAMS}" ]]; then
    echo "========================================"
    echo " Kuafu: Initial Constraint Sweep — SKIPPED"
    echo "========================================"
    echo "Using initial weights from: ${INITIAL_WEIGHTS_PARAMS}"
    echo ""
else
    INIT_SWEEP_BEST_PATH_FILE="$(mktemp)"
    cleanup() {
        rm -f "${INIT_SWEEP_BEST_PATH_FILE}"
    }
    trap cleanup EXIT

    INIT_SWEEP_CMD=(
        "${PYTHON_BIN}" src/kuafu/rl/isaac/initial_constraint_sweep.py
        --task "${TASK}"
        "${KUAFU_SWEEP_ROBOT_ARGS[@]}"
        --device "${INIT_SWEEP_DEVICE}"
        --num_envs "${NUM_ENVS}"
        --env_seed "${ENV_SEED}"
        --gpt_model "${GPT_MODEL}"
        --temperature "${TEMPERATURE}"
        --launch_script "${LAUNCH_SCRIPT}"
        --output_best_path_file "${INIT_SWEEP_BEST_PATH_FILE}"
        ${RENDER}
    )

    echo "========================================"
    echo " Kuafu: Initial Constraint Sweep — ${TASK}"
    echo "========================================"
    echo "Execution robot: ${EXECUTION_ROBOT_ID:-<inferred>}"
    echo "Artifact robot:  ${ARTIFACT_ROBOT_ID:-<inferred>}"
    echo "Running: ${INIT_SWEEP_CMD[*]}"
    echo ""

    if [[ "${KUAFU_DRY_RUN:-${DRY_RUN:-0}}" == "1" ]]; then
        exit 0
    fi

    "${INIT_SWEEP_CMD[@]}"

    INITIAL_WEIGHTS_PARAMS="$(<"${INIT_SWEEP_BEST_PATH_FILE}")"
    if [[ -z "${INITIAL_WEIGHTS_PARAMS}" ]]; then
        echo "ERROR: Initial constraint sweep did not produce a best.json path." >&2
        exit 1
    fi
fi

BEST_JSON_PATH="${INITIAL_WEIGHTS_PARAMS}"
if [[ -z "${BEST_JSON_PATH}" ]]; then
    echo "ERROR: Initial constraint sweep did not produce a best.json path." >&2
    exit 1
fi

cp "${BEST_JSON_PATH}" "${SCRIPT_DIR}/best.json"

echo ""
echo "========================================"
echo " Kuafu: Initial Constraint Sweep Done"
echo "========================================"
echo "best.json:          ${BEST_JSON_PATH}"
echo "local copy:         ${SCRIPT_DIR}/best.json"
