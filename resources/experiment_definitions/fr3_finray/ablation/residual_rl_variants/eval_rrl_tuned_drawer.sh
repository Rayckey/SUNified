#!/usr/bin/env bash
set -euo pipefail

TASK="${KUAFU_TASK:-FR3-FinRay-Table-Drawer-ff}"
CHECKPOINT_TAG="rrl_tuned_drawer"
WORKER_IDX="${WORKER_IDX:-3}"
WEIGHTS_PARAMS_JSON="Residuals/FR3-FinRay-Table-Drawer-ff/2026-05-16_23-54-23/best.json"
PRIOR_CHECKPOINT="Priors/herm/drawer_damped_noise/drawer_chunking_30_ff_damped_noise/best.pth"
PRIOR_MODE="residual_rl"
REWARD_PRESET="config"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
source "${REPO_ROOT}/src/kuafu/rl/launchers/ablation/eval_common.sh"
