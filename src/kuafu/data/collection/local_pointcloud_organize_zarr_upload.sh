#!/usr/bin/env bash
# Organize local pointcloud worker shards, create DP3 zarrs, then upload zarrs
# before the organized H5 dataset.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
KUAFU_ARTIFACT_ROOT="$("${KUAFU_PYTHON:-python}" -c 'from kuafu.paths import external_artifact_root; print(external_artifact_root())')"
export KUAFU_ARTIFACT_ROOT

usage() {
    cat <<'EOF'
Usage:
  local_pointcloud_organize_zarr_upload.sh --task TASK [options]

Required:
  --task TASK

Options:
  --source-root PATH              Worker shard root. Default: $POINTCLOUD_DIR or $KUAFU_ARTIFACT_ROOT/pointclouds/$TASK.
  --organized-root PATH           Organized H5 root. Default: $KUAFU_ARTIFACT_ROOT/organized/pointclouds-kinova/$TASK.
  --task-root PATH                DP3 task output root. Default: $KUAFU_ARTIFACT_ROOT/dp3/tasks/$TASK.
  --pointcloud-s3-dest S3_URI     Pointcloud task root. Default: $S3_DEST or s3://kuafu/pointclouds-kinova/$TASK/.
  --zarr-s3-dest S3_URI           Zarr destination. Default: s3://kuafu/kuafusim2real-kinova/$TASK/zarr/.
  --aws-profile NAME              Default: $AWS_PROFILE or kuafu.
  --aws-endpoint URL              Default: $AWS_ENDPOINT or https://s3.wasabisys.com.
  --dp3-python PATH               Default: $DP3_PYTHON, ~/miniconda3/envs/dp3/bin/python, or python.
  --method hardlink|copy|move     Local organization method. Default: hardlink.
  --table-plane-clearance-m M     Override table-plane zarr crop clearance. Default: use table-plane clip config.
  --uniform-lowest-clearance-m M  Uniform zarr crop clearance above each episode first-frame lowest z. Default: 0.01.
  --clean-s3-dest                 Delete existing objects under destination prefixes before uploading.
  --allow-nonempty-s3-dest        Upload without requiring empty destination prefixes.
  --skip-upload                   Organize and export only.
  -h, --help

The upload stage intentionally syncs only *_xyz.zarr stores. The raw XYZRGB
zarr stores are local intermediates and are not uploaded.
EOF
}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

require_file() {
    local path="$1"
    local label="$2"
    [[ -f "${path}" ]] || die "${label} not found: ${path}"
}

require_dir() {
    local path="$1"
    local label="$2"
    [[ -d "${path}" ]] || die "${label} not found: ${path}"
}

TASK_NAME="${TASK:-${DATASET_NAME:-}}"
SOURCE_ROOT="${SOURCE_ROOT:-${POINTCLOUD_DIR:-}}"
KUAFU2REAL_ROOT="${KUAFU2REAL_ROOT:-${REPO_ROOT}/../Kuafu2Real}"
ORGANIZED_ROOT="${ORGANIZED_ROOT:-}"
TASK_ROOT="${TASK_ROOT:-}"
POINTCLOUD_S3_DEST="${POINTCLOUD_S3_DEST:-${S3_DEST:-}}"
ZARR_S3_DEST="${ZARR_S3_DEST:-}"
AWS_PROFILE_NAME="${AWS_PROFILE:-kuafu}"
AWS_ENDPOINT_URL="${AWS_ENDPOINT:-https://s3.wasabisys.com}"
DP3_PYTHON_BIN="${DP3_PYTHON:-}"
ORGANIZE_METHOD="${ORGANIZE_METHOD:-hardlink}"
CLEAN_S3_DEST_BEFORE_UPLOAD="${CLEAN_S3_DEST_BEFORE_UPLOAD:-0}"
REQUIRE_CLEAN_S3_DEST="${REQUIRE_CLEAN_S3_DEST:-1}"
SKIP_UPLOAD="${SKIP_UPLOAD:-0}"

DP3_NUM_POINTS="${DP3_NUM_POINTS:-4096}"
DP3_ACTION_FORMAT="${DP3_ACTION_FORMAT:-eff-simple}"
DP3_ACTION_SHIFT="${DP3_ACTION_SHIFT:-1}"
NO_TABLE_CLIP_CONFIG="${NO_TABLE_CLIP_CONFIG:-}"
TABLEPLANE_CLIP_CONFIG="${TABLEPLANE_CLIP_CONFIG:-}"
TABLEPLANE_CLEARANCE_M="${TABLEPLANE_CLEARANCE_M:-}"
UNIFORM_LOWEST_CLEARANCE_M="${UNIFORM_LOWEST_CLEARANCE_M:-0.01}"

while (( $# > 0 )); do
    case "$1" in
        --task)
            TASK_NAME="$2"
            shift 2
            ;;
        --source-root)
            SOURCE_ROOT="$2"
            shift 2
            ;;
        --organized-root)
            ORGANIZED_ROOT="$2"
            shift 2
            ;;
        --task-root)
            TASK_ROOT="$2"
            shift 2
            ;;
        --pointcloud-s3-dest)
            POINTCLOUD_S3_DEST="$2"
            shift 2
            ;;
        --zarr-s3-dest)
            ZARR_S3_DEST="$2"
            shift 2
            ;;
        --aws-profile)
            AWS_PROFILE_NAME="$2"
            shift 2
            ;;
        --aws-endpoint)
            AWS_ENDPOINT_URL="$2"
            shift 2
            ;;
        --dp3-python)
            DP3_PYTHON_BIN="$2"
            shift 2
            ;;
        --method)
            ORGANIZE_METHOD="$2"
            shift 2
            ;;
        --clean-s3-dest)
            CLEAN_S3_DEST_BEFORE_UPLOAD=1
            shift
            ;;
        --allow-nonempty-s3-dest)
            REQUIRE_CLEAN_S3_DEST=0
            shift
            ;;
        --skip-upload)
            SKIP_UPLOAD=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown option: $1"
            ;;
    esac
done

[[ -n "${TASK_NAME}" ]] || die "--task is required"

[[ -n "${SOURCE_ROOT}" ]] || SOURCE_ROOT="${KUAFU_ARTIFACT_ROOT:-${REPO_ROOT}/../KuafuArtifacts}/pointclouds/${TASK_NAME}"
[[ -n "${ORGANIZED_ROOT}" ]] || ORGANIZED_ROOT="${KUAFU_ARTIFACT_ROOT}/organized/pointclouds-kinova/${TASK_NAME}"
[[ -n "${TASK_ROOT}" ]] || TASK_ROOT="${KUAFU_ARTIFACT_ROOT}/dp3/tasks/${TASK_NAME}"
for output_var in ORGANIZED_ROOT TASK_ROOT; do
    output_path="$("${KUAFU_PYTHON:-python}" -c 'import sys; from kuafu.paths import resolve_artifact_path; print(resolve_artifact_path(sys.argv[1], sys.argv[2]))' "${!output_var}" "${output_var}")"
    printf -v "${output_var}" '%s' "${output_path}"
done
[[ -n "${POINTCLOUD_S3_DEST}" ]] || POINTCLOUD_S3_DEST="s3://kuafu/pointclouds-kinova/${TASK_NAME}/"
[[ -n "${ZARR_S3_DEST}" ]] || ZARR_S3_DEST="s3://kuafu/kuafusim2real-kinova/${TASK_NAME}/zarr/"
[[ -n "${NO_TABLE_CLIP_CONFIG}" ]] || NO_TABLE_CLIP_CONFIG="${KUAFU2REAL_ROOT}/configs/pointcloud_clip.kinova_gen3.no_table.yaml"
[[ -n "${TABLEPLANE_CLIP_CONFIG}" ]] || TABLEPLANE_CLIP_CONFIG="${KUAFU2REAL_ROOT}/configs/pointcloud_clip.kinova_gen3.tableplane.yaml"

case "${ORGANIZE_METHOD}" in
    hardlink|copy|move) ;;
    *) die "--method must be hardlink, copy, or move" ;;
esac

if [[ -z "${DP3_PYTHON_BIN}" ]]; then
    if [[ -x "${HOME}/miniconda3/envs/dp3/bin/python" ]]; then
        DP3_PYTHON_BIN="${HOME}/miniconda3/envs/dp3/bin/python"
    else
        DP3_PYTHON_BIN="python"
    fi
fi

require_dir "${SOURCE_ROOT}" "source root"
require_file "${REPO_ROOT}/scripts/data_transfer/reorganize_pointcloud_local_workers.py" "local organizer"
require_file "${KUAFU2REAL_ROOT}/scripts/export_kuafu_to_dp3.py" "DP3 exporter"
require_file "${KUAFU2REAL_ROOT}/scripts/create_xyz_only_zarr.py" "XYZ zarr converter"
require_file "${NO_TABLE_CLIP_CONFIG}" "no-table clip config"
require_file "${TABLEPLANE_CLIP_CONFIG}" "table-plane clip config"

DATASET_DIR="${ORGANIZED_ROOT}/datasets"
ZARR_ROOT="${TASK_ROOT}/zarr"
NO_TABLE_RAW="${ZARR_ROOT}/kuafu_eff_simple_no_table.zarr"
NO_TABLE_XYZ="${ZARR_ROOT}/kuafu_eff_simple_no_table_xyz.zarr"
TABLEPLANE_RAW="${ZARR_ROOT}/kuafu_eff_simple_tableplane.zarr"
TABLEPLANE_XYZ="${ZARR_ROOT}/kuafu_eff_simple_tableplane_xyz.zarr"
POINTCLOUD_DATASET_S3_DEST="${POINTCLOUD_S3_DEST%/}/datasets/"

normalize_s3_uri() {
    local uri="$1"
    if [[ "${uri}" == */ ]]; then
        printf '%s\n' "${uri}"
    else
        printf '%s/\n' "${uri}"
    fi
}

POINTCLOUD_S3_DEST="$(normalize_s3_uri "${POINTCLOUD_S3_DEST}")"
POINTCLOUD_DATASET_S3_DEST="$(normalize_s3_uri "${POINTCLOUD_DATASET_S3_DEST}")"
ZARR_S3_DEST="$(normalize_s3_uri "${ZARR_S3_DEST}")"

s3_bucket_prefix() {
    local uri="$1"
    [[ "${uri}" == s3://* ]] || die "not an S3 URI: ${uri}"
    local rest="${uri#s3://}"
    local bucket="${rest%%/*}"
    local prefix=""
    if [[ "${rest}" != "${bucket}" ]]; then
        prefix="${rest#*/}"
    fi
    printf '%s\t%s\n' "${bucket}" "${prefix}"
}

s3_key_count() {
    local uri="$1"
    local bucket prefix
    IFS=$'\t' read -r bucket prefix < <(s3_bucket_prefix "${uri}")
    aws --endpoint-url "${AWS_ENDPOINT_URL}" --profile "${AWS_PROFILE_NAME}" \
        s3api list-objects-v2 \
        --bucket "${bucket}" \
        --prefix "${prefix}" \
        --max-keys 1 \
        --query KeyCount \
        --output text
}

ensure_s3_clean() {
    local uri="$1"
    local label="$2"
    local count

    if [[ "${REQUIRE_CLEAN_S3_DEST}" != "1" && "${CLEAN_S3_DEST_BEFORE_UPLOAD}" != "1" ]]; then
        echo "[s3] not requiring empty destination for ${label}: ${uri}"
        return 0
    fi

    count="$(s3_key_count "${uri}")"
    if [[ "${count}" == "0" ]]; then
        echo "[s3] ${label} destination is clean: ${uri}"
        return 0
    fi

    if [[ "${CLEAN_S3_DEST_BEFORE_UPLOAD}" == "1" ]]; then
        echo "[s3] cleaning non-empty ${label} destination: ${uri}"
        aws --endpoint-url "${AWS_ENDPOINT_URL}" --profile "${AWS_PROFILE_NAME}" \
            s3 rm "${uri}" --recursive --only-show-errors
        count="$(s3_key_count "${uri}")"
        [[ "${count}" == "0" ]] || die "${label} destination still has objects after cleaning: ${uri}"
        return 0
    fi

    die "${label} destination is not clean (${count} object(s)): ${uri}"
}

validate_organized_dataset() {
    "${DP3_PYTHON_BIN}" - "${DATASET_DIR}" <<'PY'
import json
import sys
from pathlib import Path

dataset_dir = Path(sys.argv[1])
metadata = json.loads((dataset_dir / "metadata.json").read_text())
h5_count = len(list(dataset_dir.glob("episode_*/episode_*.h5")))
assert h5_count == metadata["num_episodes"], (h5_count, metadata["num_episodes"])
assert metadata["num_episodes"] == len(metadata["num_timesteps"])
print(f"[validate] organized episodes={h5_count}")
PY
}

validate_xyz_zarrs() {
    "${DP3_PYTHON_BIN}" - "${NO_TABLE_XYZ}" "${TABLEPLANE_XYZ}" <<'PY'
import sys
import zarr

for path in sys.argv[1:]:
    root = zarr.open(path, mode="r")
    point_cloud = root["data/point_cloud"]
    action = root["data/action"]
    agent_pos = root["data/agent_pos"]
    assert point_cloud.shape[-1] == 3, (path, point_cloud.shape)
    assert action.shape == agent_pos.shape, (path, action.shape, agent_pos.shape)
    print(f"[validate] {path} point_cloud={point_cloud.shape} action={action.shape}")
PY
}

verify_uploaded_zarr_prefixes() {
    local count
    count="$(s3_key_count "${ZARR_S3_DEST}kuafu_eff_simple_no_table.zarr/")"
    [[ "${count}" == "0" ]] || die "raw no-table XYZRGB zarr was uploaded unexpectedly"
    count="$(s3_key_count "${ZARR_S3_DEST}kuafu_eff_simple_tableplane.zarr/")"
    [[ "${count}" == "0" ]] || die "raw table-plane XYZRGB zarr was uploaded unexpectedly"
    count="$(s3_key_count "${ZARR_S3_DEST}kuafu_eff_simple_no_table_xyz.zarr/")"
    [[ "${count}" != "0" ]] || die "no-table XYZ zarr missing from upload"
    count="$(s3_key_count "${ZARR_S3_DEST}kuafu_eff_simple_tableplane_xyz.zarr/")"
    [[ "${count}" != "0" ]] || die "table-plane XYZ zarr missing from upload"
}

echo "=========================================="
echo " local_pointcloud_organize_zarr_upload"
echo "  task              : ${TASK_NAME}"
echo "  source root       : ${SOURCE_ROOT}"
echo "  organized root    : ${ORGANIZED_ROOT}"
echo "  dataset dir       : ${DATASET_DIR}"
echo "  zarr root         : ${ZARR_ROOT}"
echo "  pointcloud s3 root: ${POINTCLOUD_S3_DEST}"
echo "  zarr s3 dest      : ${ZARR_S3_DEST}"
echo "  aws profile       : ${AWS_PROFILE_NAME}"
echo "  aws endpoint      : ${AWS_ENDPOINT_URL}"
echo "  dp3 python        : ${DP3_PYTHON_BIN}"
echo "  organize method   : ${ORGANIZE_METHOD}"
if [[ -n "${TABLEPLANE_CLEARANCE_M}" ]]; then
    echo "  table clearance   : ${TABLEPLANE_CLEARANCE_M} m override"
else
    echo "  table clearance   : from ${TABLEPLANE_CLIP_CONFIG}"
fi
echo "  uniform clearance : ${UNIFORM_LOWEST_CLEARANCE_M} m above episode first-frame lowest z"
echo "=========================================="

echo "[organize] ${SOURCE_ROOT} -> ${ORGANIZED_ROOT}"
"${DP3_PYTHON_BIN}" "${REPO_ROOT}/scripts/data_transfer/reorganize_pointcloud_local_workers.py" \
    --source-root "${SOURCE_ROOT}" \
    --output-root "${ORGANIZED_ROOT}" \
    --s3-prefix "${POINTCLOUD_S3_DEST}" \
    --method "${ORGANIZE_METHOD}" \
    --execute

validate_organized_dataset

mkdir -p "${ZARR_ROOT}"

echo "[zarr] exporting no-table raw XYZRGB store"
"${DP3_PYTHON_BIN}" "${KUAFU2REAL_ROOT}/scripts/export_kuafu_to_dp3.py" "${DATASET_DIR}" \
    --out "${NO_TABLE_RAW}" \
    --num-points "${DP3_NUM_POINTS}" \
    --action-format "${DP3_ACTION_FORMAT}" \
    --action-shift "${DP3_ACTION_SHIFT}" \
    --clip-config "${NO_TABLE_CLIP_CONFIG}"

echo "[zarr] creating no-table XYZ-only store"
"${DP3_PYTHON_BIN}" "${KUAFU2REAL_ROOT}/scripts/create_xyz_only_zarr.py" \
    "${NO_TABLE_RAW}" "${NO_TABLE_XYZ}" --overwrite

echo "[zarr] removing no-table raw XYZRGB intermediate"
rm -rf "${NO_TABLE_RAW}"

echo "[zarr] exporting table-plane raw XYZRGB store"
TABLEPLANE_EXPORT_ARGS=(
    "${DATASET_DIR}"
    --out "${TABLEPLANE_RAW}"
    --num-points "${DP3_NUM_POINTS}"
    --action-format "${DP3_ACTION_FORMAT}"
    --action-shift "${DP3_ACTION_SHIFT}"
    --clip-config "${TABLEPLANE_CLIP_CONFIG}"
)
if [[ -n "${TABLEPLANE_CLEARANCE_M}" ]]; then
    TABLEPLANE_EXPORT_ARGS+=(--table-plane-clearance-m "${TABLEPLANE_CLEARANCE_M}")
fi
"${DP3_PYTHON_BIN}" "${KUAFU2REAL_ROOT}/scripts/export_kuafu_to_dp3.py" "${TABLEPLANE_EXPORT_ARGS[@]}"

echo "[zarr] creating table-plane XYZ-only store"
"${DP3_PYTHON_BIN}" "${KUAFU2REAL_ROOT}/scripts/create_xyz_only_zarr.py" \
    "${TABLEPLANE_RAW}" "${TABLEPLANE_XYZ}" --overwrite

echo "[zarr] removing table-plane raw XYZRGB intermediate"
rm -rf "${TABLEPLANE_RAW}"

validate_xyz_zarrs

if [[ "${SKIP_UPLOAD}" == "1" ]]; then
    echo "[upload] skipped by --skip-upload"
    exit 0
fi

ensure_s3_clean "${ZARR_S3_DEST}" "zarr"
echo "[upload] syncing XYZ zarr stores first -> ${ZARR_S3_DEST}"
aws --endpoint-url "${AWS_ENDPOINT_URL}" --profile "${AWS_PROFILE_NAME}" \
    s3 sync "${ZARR_ROOT}/" "${ZARR_S3_DEST}" \
    --exclude '*' \
    --include 'kuafu_eff_simple_no_table_xyz.zarr/*' \
    --include 'kuafu_eff_simple_tableplane_xyz.zarr/*' \
    --only-show-errors
verify_uploaded_zarr_prefixes

ensure_s3_clean "${POINTCLOUD_S3_DEST}" "organized pointcloud"
echo "[upload] syncing organized H5 dataset -> ${POINTCLOUD_DATASET_S3_DEST}"
aws --endpoint-url "${AWS_ENDPOINT_URL}" --profile "${AWS_PROFILE_NAME}" \
    s3 sync "${DATASET_DIR}/" "${POINTCLOUD_DATASET_S3_DEST}" \
    --only-show-errors

echo "[done] local organization, zarr export, and uploads completed"
