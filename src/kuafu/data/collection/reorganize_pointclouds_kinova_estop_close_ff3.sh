#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Consolidate uploaded Kinova Gen3 EStop Close ff3 pointcloud worker folders
# into one DemonstrationRecorder-style datasets/ directory on S3.
# ---------------------------------------------------------------------------

set -euo pipefail

DATASET_NAME="${DATASET_NAME:-Kinova-Table-Gen3-EStop-Close-ff3-best-residual-kinova-crop-downsampled}"
S3_ROOT="${S3_ROOT:-s3://kuafu/pointclouds-kinova}"
S3_DEST="${S3_DEST:-${S3_ROOT%/}/${DATASET_NAME}/}"
WORKERS="${WORKERS:-64}"

# Default to execution because this wrapper is intended to be run after the
# upload loop. Set EXECUTE=0 to preview the plan without copying metadata/files.
EXECUTE="${EXECUTE:-1}"
KEEP_SOURCE="${KEEP_SOURCE:-1}"

cmd=(
    python scripts/data_transfer/reorganize_pointcloud_s3_workers.py
    --prefix "${S3_DEST}"
    --workers "${WORKERS}"
)

if [[ "${EXECUTE}" == "1" ]]; then
    cmd+=(--execute)
fi

if [[ "${KEEP_SOURCE}" == "1" ]]; then
    cmd+=(--keep-source)
fi

echo "=========================================="
echo " reorganize_pointclouds_kinova_estop_close_ff3"
echo "  s3 prefix   : ${S3_DEST}"
echo "  workers     : ${WORKERS}"
echo "  execute     : ${EXECUTE}"
echo "  keep source : ${KEEP_SOURCE}"
echo "=========================================="

"${cmd[@]}"
