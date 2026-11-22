#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Consolidate uploaded Kinova Gen3 EStop Close ff3 RGB + pointcloud worker folders
# into one DemonstrationRecorder-style datasets/ directory on S3.
# ---------------------------------------------------------------------------

set -euo pipefail

DATASET_NAME="${DATASET_NAME:-Kinova-Table-Gen3-EStop-Close-ff3-best-residual-kinova-crop-downsampled-rgb}"
S3_ROOT="${S3_ROOT:-s3://kuafu/pointclouds-kinova}"
S3_DEST="${S3_DEST:-${S3_ROOT%/}/${DATASET_NAME}/}"
WORKERS="${WORKERS:-64}"

# Preview by default. Set EXECUTE=1 only after reviewing the plan; successful
# execution verifies canonical copies before deleting worker-source objects.
EXECUTE="${EXECUTE:-0}"
KEEP_SOURCE="${KEEP_SOURCE:-0}"

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
echo " reorganize_rgb_kinova_estop_close_ff3"
echo "  s3 prefix   : ${S3_DEST}"
echo "  workers     : ${WORKERS}"
echo "  execute     : ${EXECUTE}"
echo "  keep source : ${KEEP_SOURCE}"
echo "=========================================="

"${cmd[@]}"
