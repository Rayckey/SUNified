#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Collect pointcloud demos with the best Hang-Cup success-epoch EMA checkpoint.
#
# Best checkpoint source:
#   Episode/Kuafu/success_rate_combined_mean_lag_window = 0.50047325
#   run: Residuals/FR3-FinRay-Table-Hang-Cup-fff/2026-05-22_17-09-20
#   worker: 2026-05-22_17-13-28_Run-3, checkpoint epoch 2750
# ---------------------------------------------------------------------------

set -euo pipefail

COLLECT_LOOP_NAME="${COLLECT_LOOP_NAME:-collect_and_upload_loop_hang_cup_best_ema}"

GPUS="0 1 2 3"
DEMOS_PER_CHUNK=200
NUM_ENVS_PER_WORKER=128
TASK="${TASK:-FR3-FinRay-Table-Hang-Cup-fff}"
CHECKPOINT="${CHECKPOINT:-${KUAFU_ARTIFACT_ROOT:-../KuafuArtifacts}/Residuals/FR3-FinRay-Table-Hang-Cup-fff/2026-05-22_17-09-20/rl_runs/2026-05-22_17-13-28_Run-3/nn/last_fr3-finray-hang-cup_ep_2750_rew_15378.236.pth}"
PRIOR_CHECKPOINT="${PRIOR_CHECKPOINT:-${KUAFU_ARTIFACT_ROOT:-../KuafuArtifacts}/Priors/herm/hang_cup/hang_cup_chunking_10_state_fff/best.pth}"
PRIOR_MODE="${PRIOR_MODE:-residual_rl}"
WEIGHTS_PARAMS_JSON="${WEIGHTS_PARAMS_JSON:-${KUAFU_ARTIFACT_ROOT:-../KuafuArtifacts}/Residuals/FR3-FinRay-Table-Hang-Cup-fff/2026-05-22_17-07-48/best.json}"
NUM_ITERS=10
DATASET_NAME="${TASK}-best-ema-downsampled"
STRIDE=1
SAVER_CHUNK_SIZE=64
SAVE_POINT_CLOUDS=0
POINT_CLOUD_NUM_POINTS="16384"
DISABLE_POINT_CLOUD_WORKSPACE_CROP=1
CAMERA_WARMUP_FRAMES_AFTER_RESET=2
HIDE_TABLE_VISUALS=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/collect_and_upload_loop_common.sh"
