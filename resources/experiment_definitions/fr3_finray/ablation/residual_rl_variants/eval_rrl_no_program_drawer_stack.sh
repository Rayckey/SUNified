#!/usr/bin/env bash
set -euo pipefail

TASK="${KUAFU_TASK:-FR3-FinRay-Table-Drawer-Stack-fff}"
CHECKPOINT_TAG="rrl_no_program_drawer_stack"
CHECKPOINT_SELECTOR="${CHECKPOINT_SELECTOR:-named_best}"
WORKER_IDX="${WORKER_IDX:-2}"
PRIOR_CHECKPOINT="${NO_STAGE_DRAWER_STACK_PRIOR:-${NO_STAGE_PRIOR_CHECKPOINT:-resources/experiment_definitions/fr3_finray/ablation/prior_variants/logs/drawer_stack_chunking_15_fff_stride3_no_stage/2026-05-27_00-28-21/checkpoints/best.pth}}"
if [[ -z "${PRIOR_CHECKPOINT}" || ! -f "${PRIOR_CHECKPOINT}" ]]; then
  echo "ERROR: no-stage prior checkpoint not found. Set NO_STAGE_DRAWER_STACK_PRIOR=/path/to/best.pth." >&2
  exit 1
fi
PRIOR_MODE="residual_rl"
WEIGHTS_PARAMS_JSON="${DRAWER_STACK_WEIGHTS_PARAMS_JSON:-${WEIGHTS_PARAMS_JSON:-Residuals/FR3-FinRay-Table-Drawer-Stack-fff/2026-05-14_23-58-02/best.json}}"
REWARD_PRESET="${REWARD_PRESET:-config}"
CLIP_ACTIONS="${CLIP_ACTIONS:-2.0}"
RESIDUAL_CLIP_COMPOSED_ACTIONS="${RESIDUAL_CLIP_COMPOSED_ACTIONS:-1}"
RESIDUAL_ALPHA_WARMUP_EPOCHS="${RESIDUAL_ALPHA_WARMUP_EPOCHS:-0}"
RESIDUAL_HIDE_STAGE_OBSERVATION=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
source "${REPO_ROOT}/src/kuafu/rl/launchers/ablation/eval_common.sh"
