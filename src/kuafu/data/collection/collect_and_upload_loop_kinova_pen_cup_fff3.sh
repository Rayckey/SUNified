#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Collect, locally organize, zarr-convert, and upload pointcloud demos for the
# Kinova Gen3 Pen-Cup fff3 residual.
# ---------------------------------------------------------------------------

set -euo pipefail

COLLECT_LOOP_NAME="${COLLECT_LOOP_NAME:-collect_and_upload_loop_kinova_pen_cup_fff3}"

GPUS="${GPUS:-0 1 2 3}"
DEMOS_PER_CHUNK="${DEMOS_PER_CHUNK:-1500}"
NUM_ENVS_PER_WORKER="${NUM_ENVS_PER_WORKER:-16}"
TASK="${TASK:-Kinova-Table-Gen3-Pen-Cup-fff3}"
CHECKPOINT="${CHECKPOINT:-${KUAFU_ARTIFACT_ROOT:-../KuafuArtifacts}/Residuals/Kinova-Table-Gen3-Pen-Cup-fff3/2026-06-23_19-48-07/checkpoints/best_global.pth}"
PRIOR_CHECKPOINT="${PRIOR_CHECKPOINT:-${KUAFU_ARTIFACT_ROOT:-../KuafuArtifacts}/Priors/kinova_gen3/pen_cup/pen_cup_2026-06-21_23-44-13_chunking_10_fff3/best.pth}"
PRIOR_MODE="${PRIOR_MODE:-residual_rl}"
WEIGHTS_PARAMS_JSON="${WEIGHTS_PARAMS_JSON:-${KUAFU_ARTIFACT_ROOT:-../KuafuArtifacts}/Residuals/Kinova-Table-Gen3-Pen-Cup-fff3/2026-06-23_19-48-07/best.json}"
NUM_ITERS="${NUM_ITERS:-1}"
DATASET_NAME="${DATASET_NAME:-${TASK}-4}"

STRIDE="${STRIDE:-1}"
SAVER_CHUNK_SIZE="${SAVER_CHUNK_SIZE:-16}"
SAVE_POINT_CLOUDS="${SAVE_POINT_CLOUDS:-1}"
POINT_CLOUD_NUM_POINTS="${POINT_CLOUD_NUM_POINTS:-16384}"
DISABLE_POINT_CLOUD_WORKSPACE_CROP="${DISABLE_POINT_CLOUD_WORKSPACE_CROP:-0}"
# Kinova workspace crop plus robot-base removal cylinder.
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
HIDE_TABLE_VISUALS="${HIDE_TABLE_VISUALS:-1}"

COLLECT_OUT="${COLLECT_OUT:-}"
POINTCLOUD_DIR="${POINTCLOUD_DIR:-}"
S3_ROOT="${S3_ROOT:-s3://kuafu/pointclouds-kinova}"
S3_DEST="${S3_DEST:-}"
AWS_ENDPOINT="${AWS_ENDPOINT:-https://s3.wasabisys.com}"
AWS_PROFILE="${AWS_PROFILE:-kuafu}"

# For this Kinova task, do not raw-upload worker shards. The shared loop stages
# local shards, organizes them locally, creates DP3 XYZ zarrs, uploads zarrs
# first, then uploads the organized H5 dataset. Destination prefixes must be
# clean unless CLEAN_S3_DEST_BEFORE_UPLOAD=1 is explicitly set.
LOCAL_ORGANIZE_ZARR_BEFORE_UPLOAD="${LOCAL_ORGANIZE_ZARR_BEFORE_UPLOAD:-1}"
REQUIRE_CLEAN_S3_DEST="${REQUIRE_CLEAN_S3_DEST:-1}"
CLEAN_S3_DEST_BEFORE_UPLOAD="${CLEAN_S3_DEST_BEFORE_UPLOAD:-0}"

# Keep four Isaac workers from each spawning a large Torch Inductor pool.
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/collect_and_upload_loop_common.sh"
