#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Collect and upload point-cloud demonstrations with the accepted Kinova Gen3
# Stack-3 fff3 residual.
#
# Selected corrected artifact:
#   Episode/Kuafu/success_rate_window = 0.36746490001678467 at epoch 1956
#   run: Residuals/Kinova-Table-Gen3-Stack-3-fff3/2026-08-03_15-57-50
#   worker: 2026-08-04_22-39-10_Run-1 (seed 43)
#   checkpoint: epoch 1950, immediately before the window peak
#   residual network: [4096, 2048, 4096] (fff3)
# ---------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
export KUAFU_ARTIFACT_ROOT="${KUAFU_ARTIFACT_ROOT:-${PROJECT_ROOT}/../KuafuArtifacts}"

export KUAFU_COLLECTION_ROBOT_ID="kinova_gen3"
COLLECT_LOOP_NAME="${COLLECT_LOOP_NAME:-kinova_gen3_collect_and_upload_loop_stack3_fff3}"

# Use all four GPUs, with one independent collector process per GPU.
GPUS="${GPUS:-0 1 2 3}"
DEMOS_PER_CHUNK="${DEMOS_PER_CHUNK:-1000}"
NUM_ENVS_PER_WORKER="${NUM_ENVS_PER_WORKER:-16}"
NUM_ITERS="${NUM_ITERS:-1}"

# Exact residual, fff3 prior, LLM sweep JSON, and selected worker parameters.
TASK="${TASK:-Kinova-Table-Gen3-Stack-3-fff3}"
CHECKPOINT="${CHECKPOINT:-${KUAFU_ARTIFACT_ROOT}/Residuals/Kinova-Table-Gen3-Stack-3-fff3/2026-08-03_15-57-50/checkpoints/best_global.pth}"
PRIOR_CHECKPOINT="${PRIOR_CHECKPOINT:-${KUAFU_ARTIFACT_ROOT}/Priors/kinova_gen3/stack3/stack3_2026-08-01_17-44-14_mpc1800_spawn30_fff3_c10_chunking_10_fff3/best.pth}"
PRIOR_MODE="${PRIOR_MODE:-residual_rl}"
WEIGHTS_PARAMS_JSON="${WEIGHTS_PARAMS_JSON:-${KUAFU_ARTIFACT_ROOT}/Residuals/Kinova-Table-Gen3-Stack-3-fff3/2026-08-03_15-57-50/inputs/initial_weights_params.json}"
CHECKPOINT_PARAMS_DIR="${CHECKPOINT_PARAMS_DIR:-${KUAFU_ARTIFACT_ROOT}/Residuals/Kinova-Table-Gen3-Stack-3-fff3/2026-08-03_15-57-50/params}"
CHECKPOINT_PARAMS_REQUIRED_FILES="agent.yaml env.yaml runtime.yaml"

# Kinova point-cloud capture and robot-base filtering.
STRIDE="${STRIDE:-1}"
SAVER_CHUNK_SIZE="${SAVER_CHUNK_SIZE:-16}"
SAVE_POINT_CLOUDS=1
SAVE_RGB="${SAVE_RGB:-0}"
POINT_CLOUD_NUM_POINTS="${POINT_CLOUD_NUM_POINTS:-16384}"
DISABLE_POINT_CLOUD_WORKSPACE_CROP="${DISABLE_POINT_CLOUD_WORKSPACE_CROP:-0}"
POINT_CLOUD_WORKSPACE_MIN="${POINT_CLOUD_WORKSPACE_MIN:-0.10 -0.60 -0.10}"
POINT_CLOUD_WORKSPACE_MAX="${POINT_CLOUD_WORKSPACE_MAX:-0.65 0.60 0.50}"
DISABLE_POINT_CLOUD_TABLE_PLANE_FILTER="${DISABLE_POINT_CLOUD_TABLE_PLANE_FILTER:-1}"
POINT_CLOUD_EXCLUDE_BOX_MIN="${POINT_CLOUD_EXCLUDE_BOX_MIN:-}"
POINT_CLOUD_EXCLUDE_BOX_MAX="${POINT_CLOUD_EXCLUDE_BOX_MAX:-}"
POINT_CLOUD_EXCLUDE_CYLINDER_CENTER="${POINT_CLOUD_EXCLUDE_CYLINDER_CENTER:-0 0}"
POINT_CLOUD_EXCLUDE_CYLINDER_RADIUS="${POINT_CLOUD_EXCLUDE_CYLINDER_RADIUS:-0.2}"
POINT_CLOUD_EXCLUDE_CYLINDER_Z_MIN="${POINT_CLOUD_EXCLUDE_CYLINDER_Z_MIN:-}"
POINT_CLOUD_EXCLUDE_CYLINDER_Z_MAX="${POINT_CLOUD_EXCLUDE_CYLINDER_Z_MAX:-}"
CAMERA_WARMUP_FRAMES_AFTER_RESET="${CAMERA_WARMUP_FRAMES_AFTER_RESET:-2}"
HIDE_TABLE_VISUALS=1

# Locally organize worker shards, create DP3 XYZ zarrs, then upload the zarrs
# and organized HDF5 dataset. Refuse to mix with an existing destination.
DATASET_NAME="${DATASET_NAME:-${TASK}-corrected-fff3}"
COLLECT_OUT="${COLLECT_OUT:-}"
POINTCLOUD_DIR="${POINTCLOUD_DIR:-}"
S3_ROOT="${S3_ROOT:-s3://kuafu/pointclouds-kinova}"
S3_DEST="${S3_DEST:-}"
AWS_ENDPOINT="${AWS_ENDPOINT:-https://s3.wasabisys.com}"
AWS_PROFILE="${AWS_PROFILE:-kuafu}"
LOCAL_ORGANIZE_ZARR_BEFORE_UPLOAD="${LOCAL_ORGANIZE_ZARR_BEFORE_UPLOAD:-1}"
REQUIRE_CLEAN_S3_DEST="${REQUIRE_CLEAN_S3_DEST:-1}"
CLEAN_S3_DEST_BEFORE_UPLOAD="${CLEAN_S3_DEST_BEFORE_UPLOAD:-0}"

# Avoid large auxiliary compiler/BLAS pools in each Isaac worker.
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

source "${SCRIPT_DIR}/collect_and_upload_loop_common.sh"
