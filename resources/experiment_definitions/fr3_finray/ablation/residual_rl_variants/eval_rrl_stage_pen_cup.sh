#!/usr/bin/env bash
set -euo pipefail

TASK="${KUAFU_TASK:-FR3-FinRay-Table-Pen-Cup-ff}"
CHECKPOINT_TAG="rrl_stage_pen_cup"
CHECKPOINT_SELECTOR="${CHECKPOINT_SELECTOR:-named_best}"
WORKER_IDX="${WORKER_IDX:-2}"
PRIOR_CHECKPOINT="${STAGE_PEN_CUP_PRIOR:-${STAGE_PRIOR_CHECKPOINT:-Priors/aws/pen_cup_noise_state_random/pen_cup_chunking_15_ff_damped_noise_state/best.pth}}"
if [[ -z "${PRIOR_CHECKPOINT}" || ! -f "${PRIOR_CHECKPOINT}" ]]; then
  echo "ERROR: stage-aware prior checkpoint not found. Set STAGE_PEN_CUP_PRIOR=/path/to/best.pth." >&2
  exit 1
fi
PRIOR_MODE="residual_rl"
WEIGHTS_PARAMS_JSON="${PEN_CUP_WEIGHTS_PARAMS_JSON:-${WEIGHTS_PARAMS_JSON:-Residuals/FR3-FinRay-Table-Pen-Cup-ff/2026-05-23_17-52-24/best.json}}"
REWARD_PRESET="${REWARD_PRESET:-config}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
source "${REPO_ROOT}/src/kuafu/rl/launchers/ablation/eval_common.sh"
