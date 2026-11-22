#!/usr/bin/env bash
set -euo pipefail

KUAFU_WEIGHT_SWEEP_TASK="${KUAFU_TASK:-${KUAFU_WEIGHT_SWEEP_TASK:-FR3-FinRay-Table-Stack-2-f}}"
KUAFU_WEIGHT_SWEEP_DEVICE="${KUAFU_WEIGHT_SWEEP_DEVICE:-cuda:0}"
KUAFU_WEIGHT_SWEEP_NUM_ENVS="${KUAFU_WEIGHT_SWEEP_NUM_ENVS:-8192}"
KUAFU_WEIGHT_SWEEP_GPT_MODEL="${KUAFU_WEIGHT_SWEEP_GPT_MODEL:-gpt-5.5}"
KUAFU_WEIGHT_SWEEP_TEMPERATURE="${KUAFU_WEIGHT_SWEEP_TEMPERATURE:-1.0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../../.." && pwd)"
cd "${REPO_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-${HOME}/miniconda3/envs/kuafu/bin/python}"
LAUNCH_SCRIPT="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"

# Load credentials from the caller's environment; never read secrets from a
# tracked launcher.
: "${OPENAI_API_KEY:?Set OPENAI_API_KEY before running this script.}"
export OPENAI_API_KEY
export GEMINI_API_KEY="${GEMINI_API_KEY:-}"

EXECUTION_ROBOT_ID="${KUAFU_EXECUTION_ROBOT_ID:-${EXECUTION_ROBOT_ID:-}}"
ARTIFACT_ROBOT_ID="${KUAFU_ARTIFACT_ROBOT_ID:-${ARTIFACT_ROBOT_ID:-fr3_finray}}"
KUAFU_SWEEP_ROBOT_ARGS=()
[[ -n "${EXECUTION_ROBOT_ID}" ]] && KUAFU_SWEEP_ROBOT_ARGS+=(--execution_robot_id "${EXECUTION_ROBOT_ID}")
[[ -n "${ARTIFACT_ROBOT_ID}" ]] && KUAFU_SWEEP_ROBOT_ARGS+=(--artifact_robot_id "${ARTIFACT_ROBOT_ID}")

PRIOR_CHECKPOINT="${KUAFU_WEIGHT_SWEEP_PRIOR_CHECKPOINT:-${INITIAL_PRIOR_CHECKPOINT:-}}"
if [[ -z "${PRIOR_CHECKPOINT}" ]]; then
    if [[ "${KUAFU_DRY_RUN:-${DRY_RUN:-0}}" == "1" ]]; then
        PRIOR_CHECKPOINT="/tmp/kuafu_sweep_dry_run_prior.pth"
    else
        echo "ERROR: Set KUAFU_WEIGHT_SWEEP_PRIOR_CHECKPOINT=/path/to/prior.pth for acquire_weights.sh." >&2
        exit 1
    fi
fi
if [[ ! -e "${PRIOR_CHECKPOINT}" && "${KUAFU_DRY_RUN:-${DRY_RUN:-0}}" != "1" ]]; then
    echo "ERROR: prior checkpoint not found: ${PRIOR_CHECKPOINT}" >&2
    exit 1
fi

INITIAL_WEIGHTS_PARAMS="${KUAFU_WEIGHT_SWEEP_INITIAL_WEIGHTS_PARAMS:-${SCRIPT_DIR}/best.json}"
BEST_PATH_FILE="$(mktemp)"
cleanup() {
    rm -f "${BEST_PATH_FILE}"
}
trap cleanup EXIT

CMD=(
    "${PYTHON_BIN}" src/kuafu/rl/isaac/initial_weight_sweep.py
    --task "${KUAFU_WEIGHT_SWEEP_TASK}"
    "${KUAFU_SWEEP_ROBOT_ARGS[@]}"
    --device "${KUAFU_WEIGHT_SWEEP_DEVICE}"
    --num_envs "${KUAFU_WEIGHT_SWEEP_NUM_ENVS}"
    --env_seed "${KUAFU_WEIGHT_SWEEP_ENV_SEED:-42}"
    --prior_checkpoint "${PRIOR_CHECKPOINT}"
    --prior_obs_stats_count "${KUAFU_WEIGHT_SWEEP_PRIOR_OBS_STATS_COUNT:-1000000}"
    --gpt_model "${KUAFU_WEIGHT_SWEEP_GPT_MODEL}"
    --temperature "${KUAFU_WEIGHT_SWEEP_TEMPERATURE}"
    --launch_script "${LAUNCH_SCRIPT}"
    --output_best_path_file "${BEST_PATH_FILE}"
)
[[ "${KUAFU_WEIGHT_SWEEP_PRIOR_VERBOSE:-0}" == "1" ]] && CMD+=(--prior_verbose)
[[ "${KUAFU_WEIGHT_SWEEP_PRIOR_FREEZE_INPUT_RUNNING_MEAN_STD:-0}" == "1" ]] && CMD+=(--prior_freeze_input_running_mean_std)
[[ -f "${INITIAL_WEIGHTS_PARAMS}" ]] && CMD+=(--initial_weights_params "${INITIAL_WEIGHTS_PARAMS}")
[[ -n "${KUAFU_WEIGHT_SWEEP_MAX_EVAL_STEPS:-}" ]] && CMD+=(--max_eval_steps "${KUAFU_WEIGHT_SWEEP_MAX_EVAL_STEPS}")
[[ "${KUAFU_WEIGHT_SWEEP_RENDER:-0}" == "1" ]] && CMD+=(--render)

echo "========================================"
echo " Kuafu: Initial Weight Sweep - ${KUAFU_WEIGHT_SWEEP_TASK}"
echo "========================================"
echo "Execution robot: ${EXECUTION_ROBOT_ID:-<inferred>}"
echo "Artifact robot:  ${ARTIFACT_ROBOT_ID:-<inferred>}"
echo "Prior:           ${PRIOR_CHECKPOINT}"
echo "Seed params:     ${INITIAL_WEIGHTS_PARAMS}"
echo "Running: ${CMD[*]}"
echo ""

if [[ "${KUAFU_DRY_RUN:-${DRY_RUN:-0}}" == "1" ]]; then
    exit 0
fi

"${CMD[@]}"
BEST_JSON_PATH="$(<"${BEST_PATH_FILE}")"
if [[ -z "${BEST_JSON_PATH}" || ! -f "${BEST_JSON_PATH}" ]]; then
    echo "ERROR: Initial weight sweep did not produce a best.json path." >&2
    exit 1
fi
cp "${BEST_JSON_PATH}" "${SCRIPT_DIR}/best.json"

echo "========================================"
echo " Kuafu: Initial Weight Sweep - DONE"
echo "========================================"
echo "best.json:  ${BEST_JSON_PATH}"
echo "local copy: ${SCRIPT_DIR}/best.json"
