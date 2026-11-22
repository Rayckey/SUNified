#!/usr/bin/env bash
# =============================================================================
# Kuafu initial constraint sweep for FR3-FinRay-Table-Hang-Cup.
#
# This only acquires an initial weights/parameters best.json. It does not launch
# the full Kuafu RL training loop.
# =============================================================================
set -euo pipefail

# API Keys
export OPENAI_API_KEY="${OPENAI_API_KEY:?Set OPENAI_API_KEY}"
export GEMINI_API_KEY="${GEMINI_API_KEY:-}"     # optional; video summaries skipped if empty
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Task
TASK="${KUAFU_TASK:-FR3-FinRay-Table-Hang-Cup}"

# Hardware
DEVICE="cuda:0,1,2,3"
INIT_SWEEP_DEVICE="cuda:0"
NUM_ENVS=8192

# LLM
GPT_MODEL="gpt-5.5"
TEMPERATURE=1.0

# Optional seed/bootstrap
ENV_SEED=42
INITIAL_WEIGHTS_PARAMS=""

# Set RENDER="--render" if you want a visible Isaac app.
RENDER="${RENDER:-}"

# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCH_SCRIPT="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
PYTHON_BIN="${PYTHON_BIN:-${HOME}/miniconda3/envs/kuafu/bin/python}"

EXECUTION_ROBOT_ID="${KUAFU_EXECUTION_ROBOT_ID:-${EXECUTION_ROBOT_ID:-}}"
ARTIFACT_ROBOT_ID="${KUAFU_ARTIFACT_ROBOT_ID:-${ARTIFACT_ROBOT_ID:-fr3_finray}}"
KUAFU_SWEEP_ROBOT_ARGS=()
[[ -n "${EXECUTION_ROBOT_ID}" ]] && KUAFU_SWEEP_ROBOT_ARGS+=(--execution_robot_id "${EXECUTION_ROBOT_ID}")
[[ -n "${ARTIFACT_ROBOT_ID}" ]] && KUAFU_SWEEP_ROBOT_ARGS+=(--artifact_robot_id "${ARTIFACT_ROBOT_ID}")

if [[ -n "${INITIAL_WEIGHTS_PARAMS}" ]]; then
    echo "========================================"
    echo " Kuafu: Initial Constraint Sweep - SKIPPED"
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
    echo " Kuafu: Initial Constraint Sweep - ${TASK}"
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
