#!/usr/bin/env bash
# =============================================================================
# Kuafu residual-hparam run - generated from BC prior sweep CSV
# Selected BC prior row: drawer_damped_noise_f3_chunking_30_ff3_stride3
# Prior config: resources/experiment_definitions/fr3_finray/prior_configs/drawer/generated/drawer_damped_noise_f3/drawer_damped_noise_f3_chunking_30_ff3_stride3.yaml
# Prior checkpoint: Priors/fr3_finray/drawer/drawer_damped_noise_f3_chunking_30_ff3_stride3/best.pth
# =============================================================================
set -euo pipefail

KUAFU_EXECUTION_ROBOT_ID="${KUAFU_EXECUTION_ROBOT_ID:-fr3_finray}"
KUAFU_ARTIFACT_ROBOT_ID="${KUAFU_ARTIFACT_ROBOT_ID:-fr3_finray}"

: "${OPENAI_API_KEY:?Set OPENAI_API_KEY before running this script.}"
export OPENAI_API_KEY
export GEMINI_API_KEY="${GEMINI_API_KEY:-}"
export CUDA_VISIBLE_DEVICES="${KUAFU_CUDA_VISIBLE_DEVICES:-0,1,2,3}"

TASK="${KUAFU_TASK:-FR3-FinRay-Table-Drawer-ff3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KUAFU_REPO_ROOT="${KUAFU_REPO_ROOT:-}"
if [[ -z "${KUAFU_REPO_ROOT}" ]]; then
    KUAFU_SEARCH_DIR="${SCRIPT_DIR}"
    while [[ "${KUAFU_SEARCH_DIR}" != "/" && ! -f "${KUAFU_SEARCH_DIR}/src/kuafu/rl/launchers/prior_conditioned/dual_launcher_common.sh" ]]; do
        KUAFU_SEARCH_DIR="$(dirname "${KUAFU_SEARCH_DIR}")"
    done
    KUAFU_REPO_ROOT="${KUAFU_SEARCH_DIR}"
fi
source "${KUAFU_REPO_ROOT}/src/kuafu/rl/launchers/prior_conditioned/dual_launcher_common.sh"
kuafu_dual_init_runtime_options

DEVICE="${KUAFU_DEVICE:-cuda:0,1,2,3}"
NUM_ENVS=8192
NUM_WORKERS=4

INIT_TRAIN_EPOCH=2000
MAX_TRAIN_EPOCH=8000
EPOCH_TRAIN_INCREMENT=500

MAXIMUN_PROPOSALS=0
GLOBAL_FINETUNE_ITERS=0
RESIDUAL_HPARAM_ITERS=5
GLOBAL_FINETUNE_MAX_FALSE_TO_TRUE_RATIO=0.2
STAGE_SUCCESS_FRACTION=0.5
STAGE_ADVANCE_COMPLETION_RATE_CAP=0.8

GPT_MODEL="gpt-5.5"
TEMPERATURE=1.0
GEMINI_MODEL="gemini-3.1-pro-preview"
VIDEO_FEEDBACK="--no-video_feedback"
VIDEO_LENGTH=400
VIDEO_NUM_ENVS_TO_RECORD=9
FEEDBACK_SUBSAMPLING=10

INITIAL_EVAL_CHECKPOINT=""
INITIAL_CHECKPOINT=""
INITIAL_PRIOR_CHECKPOINT="Priors/fr3_finray/drawer/drawer_damped_noise_f3_chunking_30_ff3_stride3/best.pth"
INITIAL_PRIOR_MODE="residual_rl"
INITIAL_PRIOR_OBS_STATS_COUNT="1000000"
INITIAL_PRIOR_VERBOSE=""
INITIAL_PRIOR_CRITIC_INIT_MODE="actor_only_random_critic"
INITIAL_WEIGHTS_PARAMS="${INITIAL_WEIGHTS_PARAMS:-/home/zhi/isaac/IsaacLab/resources/experiment_definitions/fr3_finray/residual_rl/parameter_search/fr3_finray_learned_drawer/best.json}"
kuafu_dual_require_precomputed_weights_params

RESTART_EACH_ITER_FROM_INITIAL_WARM_START="--restart_each_iter_from_initial_warm_start"
STOP_ON_ERROR=""
ENV_SEED=42
RENDER=""
PRINT_PROGRAM_FUNCTIONS=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCH_SCRIPT="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"

if [[ -n "${INITIAL_CHECKPOINT}" && -n "${INITIAL_PRIOR_CHECKPOINT}" ]]; then
    echo "ERROR: Choose only one warm-start source: INITIAL_CHECKPOINT or INITIAL_PRIOR_CHECKPOINT." >&2
    exit 1
fi

CMD=(
    python scripts/residual_rl/train_residual_hparams.py
    --launch_script "${LAUNCH_SCRIPT}"
    --task "${TASK}"
    --device "${DEVICE}"
    --num_workers "${NUM_WORKERS}"
    --num_envs "${NUM_ENVS}"
    --env_seed "${ENV_SEED}"
    --init_train_epoch "${INIT_TRAIN_EPOCH}"
    --max_train_epoch "${MAX_TRAIN_EPOCH}"
    --epoch_train_increment "${EPOCH_TRAIN_INCREMENT}"
    --feedback_subsampling "${FEEDBACK_SUBSAMPLING}"
    --stage_success_fraction "${STAGE_SUCCESS_FRACTION}"
    --maximun_proposals "${MAXIMUN_PROPOSALS}"
    --stage_advance_completion_rate_cap "${STAGE_ADVANCE_COMPLETION_RATE_CAP}"
    --global_finetune_iters "${GLOBAL_FINETUNE_ITERS}"
    --residual_hparam_iters "${RESIDUAL_HPARAM_ITERS}"
    --global_finetune_max_false_to_true_ratio "${GLOBAL_FINETUNE_MAX_FALSE_TO_TRUE_RATIO}"
    --gpt_model "${GPT_MODEL}"
    --temperature "${TEMPERATURE}"
    --gemini_model "${GEMINI_MODEL}"
    "${KUAFU_ROBOT_ARGS[@]}"
    ${VIDEO_FEEDBACK}
    --video_length "${VIDEO_LENGTH}"
    --video_num_envs_to_record "${VIDEO_NUM_ENVS_TO_RECORD}"
    ${RENDER}
)

[[ -n "${INITIAL_EVAL_CHECKPOINT}" ]] && CMD+=(--initial_eval_checkpoint "${INITIAL_EVAL_CHECKPOINT}")
[[ -n "${INITIAL_CHECKPOINT}" ]] && CMD+=(--initial_checkpoint "${INITIAL_CHECKPOINT}")
[[ -n "${INITIAL_PRIOR_CHECKPOINT}" ]] && CMD+=(--initial_prior_checkpoint "${INITIAL_PRIOR_CHECKPOINT}")
[[ -n "${INITIAL_PRIOR_MODE}" ]] && CMD+=(--initial_prior_mode "${INITIAL_PRIOR_MODE}")
[[ -n "${INITIAL_PRIOR_OBS_STATS_COUNT}" ]] && CMD+=(--initial_prior_obs_stats_count "${INITIAL_PRIOR_OBS_STATS_COUNT}")
[[ -n "${INITIAL_PRIOR_CRITIC_INIT_MODE}" ]] && CMD+=(--initial_prior_critic_init_mode "${INITIAL_PRIOR_CRITIC_INIT_MODE}")
[[ -n "${INITIAL_PRIOR_VERBOSE}" ]] && CMD+=(${INITIAL_PRIOR_VERBOSE})
[[ -n "${INITIAL_WEIGHTS_PARAMS}" ]] && CMD+=(--initial_weights_params "${INITIAL_WEIGHTS_PARAMS}")
[[ -n "${RESTART_EACH_ITER_FROM_INITIAL_WARM_START}" ]] && CMD+=(${RESTART_EACH_ITER_FROM_INITIAL_WARM_START})
[[ -n "${STOP_ON_ERROR}" ]] && CMD+=(${STOP_ON_ERROR})
[[ -n "${PRINT_PROGRAM_FUNCTIONS}" ]] && CMD+=(${PRINT_PROGRAM_FUNCTIONS})

echo "========================================"
echo " Kuafu: Residual Hyperparameter Run - ${TASK}"
echo "========================================"
echo "Workers:           ${NUM_WORKERS}"
echo "Envs:              ${NUM_ENVS}"
echo "Epochs:            ${INIT_TRAIN_EPOCH} -> ${MAX_TRAIN_EPOCH} (+${EPOCH_TRAIN_INCREMENT})"
echo "Residual hparams:  ${RESIDUAL_HPARAM_ITERS} iters"
echo "Stage tuning:      off"
echo "Global finetune:   off"
echo "LLM:               ${GPT_MODEL} (temp=${TEMPERATURE})"
echo "Initial prior:     ${INITIAL_PRIOR_CHECKPOINT}"
echo "Weights/params:    ${INITIAL_WEIGHTS_PARAMS}"
echo "Execution robot:   ${EXECUTION_ROBOT_ID:-<inferred from task>}"
echo "Artifact robot:    ${ARTIFACT_ROBOT_ID:-<default>}"
echo "Restart from warm: ${RESTART_EACH_ITER_FROM_INITIAL_WARM_START:-<off>}"
echo "Stop on error:     ${STOP_ON_ERROR:-<off>}"
echo "========================================"
echo ""
echo "Running: ${CMD[*]}"
echo ""

if kuafu_dual_bool_enabled "${KUAFU_LAUNCH_DRY_RUN:-}"; then
    kuafu_dual_print_command "train" "${CMD[@]}"
    exit 0
fi

exec "${CMD[@]}"
