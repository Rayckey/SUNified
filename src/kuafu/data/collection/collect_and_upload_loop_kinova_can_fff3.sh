#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Collect, locally organize, zarr-convert, and upload pointcloud demos for the
# Kinova Gen3 Can fff3 residual.
#
# Window-success selection provenance:
#   worker 1 peak = 0.5887415409 at epoch 763
#   selected checkpoint = epoch 750 (strictly before the window peak)
# ---------------------------------------------------------------------------

set -euo pipefail

COLLECT_LOOP_NAME="${COLLECT_LOOP_NAME:-collect_and_upload_loop_kinova_can_fff3}"

GPUS="${GPUS:-0 1 2 3}"
DEMOS_PER_CHUNK="${DEMOS_PER_CHUNK:-1500}"
NUM_ENVS_PER_WORKER="${NUM_ENVS_PER_WORKER:-16}"
TASK="${TASK:-Kinova-Table-Gen3-Can-fff3}"
CHECKPOINT="${CHECKPOINT:-${KUAFU_ARTIFACT_ROOT:-../KuafuArtifacts}/Residuals/Kinova-Table-Gen3-Can-fff3/2026-07-07_19-37-33/rl_runs/2026-07-07_19-39-23_Run-1/nn/last_kinova-can_ep_750_rew_11459.768.pth}"
PRIOR_CHECKPOINT="${PRIOR_CHECKPOINT:-${KUAFU_ARTIFACT_ROOT:-../KuafuArtifacts}/Priors/kinova_gen3/right_can/right_can_2026-07-05_12-46-51_plus_2026-07-06_extra1000_fff3_chunking_10_fff3/best.pth}"
PRIOR_MODE="${PRIOR_MODE:-residual_rl}"
WEIGHTS_PARAMS_JSON="${WEIGHTS_PARAMS_JSON:-${KUAFU_ARTIFACT_ROOT:-../KuafuArtifacts}/Residuals/Kinova-Table-Gen3-Can-fff3/2026-07-07_19-37-33/inputs/initial_weights_params.json}"
CHECKPOINT_PARAMS_DIR="${CHECKPOINT_PARAMS_DIR:-${KUAFU_ARTIFACT_ROOT:-../KuafuArtifacts}/Residuals/Kinova-Table-Gen3-Can-fff3/2026-07-07_19-37-33/rl_runs/2026-07-07_19-39-23_Run-1/params}"
CHECKPOINT_PARAMS_REQUIRED_FILES="${CHECKPOINT_PARAMS_REQUIRED_FILES:-agent.yaml env.yaml runtime.yaml}"
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

# Do not raw-upload worker shards. Organize them locally, create DP3 XYZ zarrs,
# upload zarrs first, then upload the organized H5 dataset.
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
