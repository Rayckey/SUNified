#!/usr/bin/env bash
set -euo pipefail

TASK="${KUAFU_TASK:-FR3-FinRay-Table-Drawer-ff}"
CHECKPOINT_TAG="rrl_stage_drawer"
CHECKPOINT_SELECTOR="${CHECKPOINT_SELECTOR:-named_best}"
WORKER_IDX="${WORKER_IDX:-3}"
PRIOR_CHECKPOINT="${STAGE_DRAWER_PRIOR:-${STAGE_PRIOR_CHECKPOINT:-Priors/herm/drawer_damped_noise/drawer_chunking_30_ff_damped_noise/best.pth}}"
if [[ -z "${PRIOR_CHECKPOINT}" || ! -f "${PRIOR_CHECKPOINT}" ]]; then
  echo "ERROR: stage-aware prior checkpoint not found. Set STAGE_DRAWER_PRIOR=/path/to/best.pth." >&2
  exit 1
fi
PRIOR_MODE="residual_rl"
WEIGHTS_PARAMS_JSON="${DRAWER_WEIGHTS_PARAMS_JSON:-${WEIGHTS_PARAMS_JSON:-Residuals/FR3-FinRay-Table-Drawer-ff/2026-05-16_23-54-23/best.json}}"
REWARD_PRESET="${REWARD_PRESET:-config}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
source "${REPO_ROOT}/src/kuafu/rl/launchers/ablation/eval_common.sh"
