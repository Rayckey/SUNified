#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Collect pointcloud demos with the best EStop-Open success-epoch checkpoint.
#
# Best checkpoint source:
#   losses/kuafu_success_rate_epoch = 0.96824199
#   run: Residuals/FR3-FinRay-Table-EStop-Open-ff/2026-04-30_12-01-28
#   worker: 2026-04-30_21-35-13_Run-0, checkpoint epoch 1100
# ---------------------------------------------------------------------------

set -euo pipefail

COLLECT_LOOP_NAME="${COLLECT_LOOP_NAME:-collect_and_upload_loop_estop_open_best_ema}"

GPUS="0 1 2 3"
DEMOS_PER_CHUNK=200
NUM_ENVS_PER_WORKER=128
TASK="${TASK:-FR3-FinRay-Table-EStop-Open-ff}"
CHECKPOINT="${CHECKPOINT:-${KUAFU_ARTIFACT_ROOT:-../KuafuArtifacts}/Residuals/FR3-FinRay-Table-EStop-Open-ff/2026-04-30_12-01-28/rl_runs/2026-04-30_21-35-13_Run-0/nn/last_Open-FR3FinRay-EStop_ep_1100_rew__4772.532_.pth}"
PRIOR_CHECKPOINT="${PRIOR_CHECKPOINT:-${KUAFU_ARTIFACT_ROOT:-../KuafuArtifacts}/Priors/thor/estop_open/estop_open_chunking_15_ff/best.pth}"
PRIOR_MODE="${PRIOR_MODE:-residual_rl}"
WEIGHTS_PARAMS_JSON="${WEIGHTS_PARAMS_JSON:-${KUAFU_ARTIFACT_ROOT:-../KuafuArtifacts}/Residuals/FR3-FinRay-Table-EStop-Open-ff/2026-04-30_12-01-28/best.json}"
NUM_ITERS=10
DATASET_NAME="${TASK}-best-ema-downsampled"
STRIDE=1
SAVER_CHUNK_SIZE=64
SAVE_POINT_CLOUDS=0
POINT_CLOUD_NUM_POINTS="16384"
DISABLE_POINT_CLOUD_WORKSPACE_CROP=1
POINT_CLOUD_EXCLUDE_BOX_MIN="none -0.20 none"
POINT_CLOUD_EXCLUDE_BOX_MAX="0.14 none none"
CAMERA_WARMUP_FRAMES_AFTER_RESET=2
HIDE_TABLE_VISUALS=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/collect_and_upload_loop_common.sh"
