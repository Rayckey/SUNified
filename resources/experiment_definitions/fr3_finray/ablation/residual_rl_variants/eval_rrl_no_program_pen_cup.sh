#!/usr/bin/env bash
set -euo pipefail

TASK="${KUAFU_TASK:-FR3-FinRay-Table-Pen-Cup-ff}"
CHECKPOINT_TAG="rrl_no_program_pen_cup"
CHECKPOINT_SELECTOR="${CHECKPOINT_SELECTOR:-named_best}"
WORKER_IDX="${WORKER_IDX:-2}"
PRIOR_CHECKPOINT="${NO_STAGE_PEN_CUP_PRIOR:-${NO_STAGE_PRIOR_CHECKPOINT:-resources/experiment_definitions/fr3_finray/ablation/prior_variants/logs/pen_cup_chunking_15_ff_no_stage/2026-05-27_00-33-42/checkpoints/best.pth}}"
if [[ -z "${PRIOR_CHECKPOINT}" || ! -f "${PRIOR_CHECKPOINT}" ]]; then
  echo "ERROR: no-stage prior checkpoint not found. Set NO_STAGE_PEN_CUP_PRIOR=/path/to/best.pth." >&2
  exit 1
fi
PRIOR_MODE="residual_rl"
WEIGHTS_PARAMS_JSON="${PEN_CUP_WEIGHTS_PARAMS_JSON:-${WEIGHTS_PARAMS_JSON:-Residuals/FR3-FinRay-Table-Pen-Cup-ff/2026-05-23_17-52-24/best.json}}"
REWARD_PRESET="${REWARD_PRESET:-config}"
CLIP_ACTIONS="${CLIP_ACTIONS:-2.0}"
RESIDUAL_CLIP_COMPOSED_ACTIONS="${RESIDUAL_CLIP_COMPOSED_ACTIONS:-1}"
RESIDUAL_ALPHA_WARMUP_EPOCHS="${RESIDUAL_ALPHA_WARMUP_EPOCHS:-0}"
RESIDUAL_HIDE_STAGE_OBSERVATION=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
source "${REPO_ROOT}/src/kuafu/rl/launchers/ablation/eval_common.sh"
