#!/usr/bin/env bash
set -euo pipefail

TASK="${KUAFU_TASK:-FR3-FinRay-Table-Stack-2-ff}"
CHECKPOINT_TAG="rrl_tuned_stack2"
WORKER_IDX="${WORKER_IDX:-0}"
WEIGHTS_PARAMS_JSON="Residuals/FR3-FinRay-Table-Stack-2-ff/2026-05-17_11-53-49/best.json"
PRIOR_CHECKPOINT="Priors/thor/stack_2_damped_higher_noise_state/stack2_chunking_15_shift_ff_damped_higher/best.pth"
PRIOR_MODE="residual_rl"
REWARD_PRESET="config"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
source "${REPO_ROOT}/src/kuafu/rl/launchers/ablation/eval_common.sh"
