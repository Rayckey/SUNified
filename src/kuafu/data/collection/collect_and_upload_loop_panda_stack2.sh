#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Collect and upload point-cloud demonstrations with the accepted Panda
# Stack-2 fff3 residual.
#
# Selected artifact source:
#   Episode/Kuafu/success_rate_window = 0.9631717801094055 at epoch 1602
#   run: Residuals/Panda-Table-Stack-2-fff3/2026-08-02_17-32-02
#   worker: 2026-08-02_17-32-41_Run-0 (seed 4200)
#   selected checkpoint: epoch 1600, immediately before the window peak
# ---------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
export KUAFU_ARTIFACT_ROOT="${KUAFU_ARTIFACT_ROOT:-${PROJECT_ROOT}/../KuafuArtifacts}"

export KUAFU_COLLECTION_ROBOT_ID="franka_panda"
COLLECT_LOOP_NAME="${COLLECT_LOOP_NAME:-franka_panda_collect_and_upload_loop_stack2_fff3}"

# Collection workload. Each GPU gets one independent collector worker.
GPUS="${GPUS:-0 1 2 3}"
DEMOS_PER_CHUNK="${DEMOS_PER_CHUNK:-1000}"
NUM_ENVS_PER_WORKER="${NUM_ENVS_PER_WORKER:-16}"
NUM_ITERS="${NUM_ITERS:-1}"

# Exact accepted residual, prior, and learned-program weights.
TASK="${TASK:-Panda-Table-Stack-2-fff3}"
CHECKPOINT="${CHECKPOINT:-${KUAFU_ARTIFACT_ROOT}/Residuals/Panda-Table-Stack-2-fff3/2026-08-02_17-32-02/checkpoints/best_global.pth}"
PRIOR_CHECKPOINT="${PRIOR_CHECKPOINT:-${KUAFU_ARTIFACT_ROOT}/Priors/franka_panda/stack2/stack2_left_workspace_j1p1_mpc_1000demo_2026-08-01_v1_chunking_10_fff3/best.pth}"
PRIOR_MODE="${PRIOR_MODE:-residual_rl}"
WEIGHTS_PARAMS_JSON="${WEIGHTS_PARAMS_JSON:-${KUAFU_ARTIFACT_ROOT}/Residuals/Panda-Table-Stack-2-fff3/2026-08-02_17-32-02/inputs/initial_weights_params.json}"
CHECKPOINT_PARAMS_DIR="${CHECKPOINT_PARAMS_DIR:-${KUAFU_ARTIFACT_ROOT}/Residuals/Panda-Table-Stack-2-fff3/2026-08-02_17-32-02/params}"
CHECKPOINT_PARAMS_REQUIRED_FILES="agent.yaml env.yaml runtime.yaml"

# Point-cloud HDF5 settings. Crop only in Z in the robot-base frame, retaining
# the complete task workspace in X/Y. Remove the robot-base/body column with a
# 0.22 m-radius vertical cylinder centered at the robot origin.
STRIDE="${STRIDE:-1}"
SAVER_CHUNK_SIZE="${SAVER_CHUNK_SIZE:-16}"
SAVE_POINT_CLOUDS=1
SAVE_RGB="${SAVE_RGB:-0}"
POINT_CLOUD_NUM_POINTS="${POINT_CLOUD_NUM_POINTS:-16384}"
DISABLE_POINT_CLOUD_WORKSPACE_CROP="${DISABLE_POINT_CLOUD_WORKSPACE_CROP:-0}"
POINT_CLOUD_WORKSPACE_MIN="${POINT_CLOUD_WORKSPACE_MIN:-none none -0.1}"
POINT_CLOUD_WORKSPACE_MAX="${POINT_CLOUD_WORKSPACE_MAX:-none none 0.5}"
# The simulator support surface is invisible, so the tilted-table plane filter
# remains disabled.
DISABLE_POINT_CLOUD_TABLE_PLANE_FILTER="${DISABLE_POINT_CLOUD_TABLE_PLANE_FILTER:-1}"
POINT_CLOUD_EXCLUDE_BOX_MIN="${POINT_CLOUD_EXCLUDE_BOX_MIN:-}"
POINT_CLOUD_EXCLUDE_BOX_MAX="${POINT_CLOUD_EXCLUDE_BOX_MAX:-}"
POINT_CLOUD_EXCLUDE_CYLINDER_CENTER="${POINT_CLOUD_EXCLUDE_CYLINDER_CENTER:-0.0 0.0}"
POINT_CLOUD_EXCLUDE_CYLINDER_RADIUS="${POINT_CLOUD_EXCLUDE_CYLINDER_RADIUS:-0.22}"
POINT_CLOUD_EXCLUDE_CYLINDER_Z_MIN="${POINT_CLOUD_EXCLUDE_CYLINDER_Z_MIN:--0.1}"
POINT_CLOUD_EXCLUDE_CYLINDER_Z_MAX="${POINT_CLOUD_EXCLUDE_CYLINDER_Z_MAX:-0.5}"
# Use all three calibrated Panda views and fail fast if the environment does
# not register this exact simulated-camera bank.
CAMERA_KEYS="${CAMERA_KEYS:-table_cam_0 table_cam_1 table_cam_2}"
EXPECTED_NUM_CAMERAS=3
USE_TILED_CAMERAS=1
CAMERA_WARMUP_FRAMES_AFTER_RESET="${CAMERA_WARMUP_FRAMES_AFTER_RESET:-2}"
HIDE_TABLE_VISUALS=1
export KUAFU_EAGER_POINT_CLOUD="${KUAFU_EAGER_POINT_CLOUD:-1}"

# The shared loop stages only directories created by the current iteration,
# syncs them to Wasabi, and removes local staged copies only after sync succeeds.
DATASET_NAME="${DATASET_NAME:-${TASK}-best-residual-panda-3cam-zcrop-m0p1-0p5-cylinder-r0p22-downsampled}"
COLLECT_OUT="${COLLECT_OUT:-}"
POINTCLOUD_DIR="${POINTCLOUD_DIR:-}"
S3_ROOT="${S3_ROOT:-s3://kuafu/pointclouds-match}"
S3_DEST="${S3_DEST:-}"
AWS_ENDPOINT="${AWS_ENDPOINT:-https://s3.wasabisys.com}"
AWS_PROFILE="${AWS_PROFILE:-kuafu}"
LOCAL_ORGANIZE_ZARR_BEFORE_UPLOAD=0

# Avoid large auxiliary compiler/BLAS thread pools in each Isaac worker.
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

source "${SCRIPT_DIR}/collect_and_upload_loop_common.sh"
