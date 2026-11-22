#!/usr/bin/env bash
set -euo pipefail

TASK="${KUAFU_TASK:-FR3-FinRay-Table-Stack-2-ff}"
CHECKPOINT_TAG="rrl_stage_stack2"
CHECKPOINT_SELECTOR="${CHECKPOINT_SELECTOR:-named_best}"
WORKER_IDX="${WORKER_IDX:-0}"
PRIOR_CHECKPOINT="${STAGE_STACK2_PRIOR:-${STAGE_PRIOR_CHECKPOINT:-Priors/thor/stack_2_damped_higher_noise_state/stack2_chunking_15_shift_ff_damped_higher/best.pth}}"
if [[ -z "${PRIOR_CHECKPOINT}" || ! -f "${PRIOR_CHECKPOINT}" ]]; then
  echo "ERROR: stage-aware prior checkpoint not found. Set STAGE_STACK2_PRIOR=/path/to/best.pth." >&2
  exit 1
fi
PRIOR_MODE="residual_rl"
WEIGHTS_PARAMS_JSON="${STACK2_WEIGHTS_PARAMS_JSON:-${WEIGHTS_PARAMS_JSON:-Residuals/FR3-FinRay-Table-Stack-2-ff/2026-05-17_11-53-49/best.json}}"
REWARD_PRESET="${REWARD_PRESET:-config}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
source "${REPO_ROOT}/src/kuafu/rl/launchers/ablation/eval_common.sh"
