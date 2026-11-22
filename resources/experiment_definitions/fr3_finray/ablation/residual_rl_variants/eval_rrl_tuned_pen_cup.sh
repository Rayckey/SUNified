#!/usr/bin/env bash
set -euo pipefail

TASK="${KUAFU_TASK:-FR3-FinRay-Table-Pen-Cup-ff}"
CHECKPOINT_TAG="rrl_tuned_pen_cup"
WORKER_IDX="${WORKER_IDX:-2}"
WEIGHTS_PARAMS_JSON="Residuals/FR3-FinRay-Table-Pen-Cup-ff/2026-05-23_17-52-24/best.json"
PRIOR_CHECKPOINT="Priors/aws/pen_cup_noise_state_random/pen_cup_chunking_15_ff_damped_noise_state/best.pth"
PRIOR_MODE="residual_rl"
REWARD_PRESET="config"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
source "${REPO_ROOT}/src/kuafu/rl/launchers/ablation/eval_common.sh"
