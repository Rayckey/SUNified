#!/usr/bin/env bash
set -euo pipefail

TASK="${KUAFU_TASK:-FR3-FinRay-Table-Drawer-Stack-fff}"
CHECKPOINT_TAG="rrl_tuned_drawer_stack"
WORKER_IDX="${WORKER_IDX:-2}"
WEIGHTS_PARAMS_JSON="Residuals/FR3-FinRay-Table-Drawer-Stack-fff/2026-05-14_23-58-02/best.json"
PRIOR_CHECKPOINT="Priors/herm/drawer_stack/drawer_stack_chunking_15_fff_stride3/best.pth"
PRIOR_MODE="residual_rl"
REWARD_PRESET="config"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
source "${REPO_ROOT}/src/kuafu/rl/launchers/ablation/eval_common.sh"
