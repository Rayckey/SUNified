#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Collect pointcloud demos with the best Stack-3 success-window checkpoint.
#
# Best checkpoint source:
#   Episode/Kuafu/success_rate_window = 0.76125497 at checkpoint epoch 2750
#   run: Residuals/FR3-FinRay-Table-Stack-3-fff/2026-05-18_19-23-40
#   worker: 2026-05-18_19-24-40_Run-0, checkpoint epoch 2750
# ---------------------------------------------------------------------------

set -euo pipefail

COLLECT_LOOP_NAME="${COLLECT_LOOP_NAME:-collect_and_upload_loop_stack3}"

GPUS="0 1 2 3"
DEMOS_PER_CHUNK=200
NUM_ENVS_PER_WORKER=4
TASK="${TASK:-FR3-FinRay-Table-Stack-3-fff}"
CHECKPOINT="${CHECKPOINT:-${KUAFU_ARTIFACT_ROOT:-../KuafuArtifacts}/Residuals/FR3-FinRay-Table-Stack-3-fff/2026-05-18_19-23-40/rl_runs/2026-05-18_19-24-40_Run-0/nn/last_stack-3_ep_2750_rew_106216.23.pth}"
PRIOR_CHECKPOINT="${PRIOR_CHECKPOINT:-${KUAFU_ARTIFACT_ROOT:-../KuafuArtifacts}/Priors/ares/stack3/stack3_chunking_15_fff_stride3/best.pth}"
PRIOR_MODE="${PRIOR_MODE:-residual_rl}"
WEIGHTS_PARAMS_JSON="${WEIGHTS_PARAMS_JSON:-${KUAFU_ARTIFACT_ROOT:-../KuafuArtifacts}/Residuals/FR3-FinRay-Table-Stack-3-fff/2026-05-18_19-18-23/best.json}"
NUM_ITERS=8
DATASET_NAME="${TASK}-downsampled"

STRIDE=1
SAVER_CHUNK_SIZE=64
SAVE_POINT_CLOUDS=0
POINT_CLOUD_NUM_POINTS="16384"
DISABLE_POINT_CLOUD_WORKSPACE_CROP=1
CAMERA_WARMUP_FRAMES_AFTER_RESET=2
HIDE_TABLE_VISUALS=1

COLLECT_OUT=""
POINTCLOUD_DIR=""
S3_ROOT="s3://kuafu/pointclouds-19"
S3_DEST=""
AWS_ENDPOINT="https://s3.wasabisys.com"
AWS_PROFILE="kuafu"

# Keep four Isaac workers from each spawning a 32-process Torch Inductor pool.
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"

case ":${PATH}:" in
    *":${HOME}/.local/bin:"*) ;;
    *) export PATH="${HOME}/.local/bin:${PATH}" ;;
esac

S3_ROOT="${S3_ROOT%/}"

[[ -z "${COLLECT_OUT}"    ]] && COLLECT_OUT="${KUAFU_ARTIFACT_ROOT:-${REPO_ROOT}/../KuafuArtifacts}/collection/${TASK}"
[[ -z "${POINTCLOUD_DIR}" ]] && POINTCLOUD_DIR="${KUAFU_ARTIFACT_ROOT:-${REPO_ROOT}/../KuafuArtifacts}/pointclouds/${DATASET_NAME}"
[[ -z "${S3_DEST}"        ]] && S3_DEST="${S3_ROOT}/${DATASET_NAME}/"
[[ -z "${LOOP_LOG_DIR:-}" ]] && LOOP_LOG_DIR="${COLLECT_OUT}/loop_logs"

require_file() {
    local path="$1"
    local label="$2"
    if [[ ! -f "${path}" ]]; then
        echo "ERROR: ${label} does not exist: ${path}" >&2
        exit 1
    fi
}

require_positive_int() {
    local value="$1"
    local label="$2"
    if [[ ! "${value}" =~ ^[0-9]+$ ]] || (( value <= 0 )); then
        echo "ERROR: ${label} must be a positive integer, got '${value}'." >&2
        exit 1
    fi
}

require_nonnegative_int() {
    local value="$1"
    local label="$2"
    if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
        echo "ERROR: ${label} must be a nonnegative integer, got '${value}'." >&2
        exit 1
    fi
}

require_bool01() {
    local value="$1"
    local label="$2"
    if [[ "${value}" != "0" && "${value}" != "1" ]]; then
        echo "ERROR: ${label} must be 0 or 1, got '${value}'." >&2
        exit 1
    fi
}

require_file "${CHECKPOINT}" "CHECKPOINT"
require_file "${PRIOR_CHECKPOINT}" "PRIOR_CHECKPOINT"
require_file "${WEIGHTS_PARAMS_JSON}" "WEIGHTS_PARAMS_JSON"
require_positive_int "${DEMOS_PER_CHUNK}" "DEMOS_PER_CHUNK"
require_positive_int "${NUM_ENVS_PER_WORKER}" "NUM_ENVS_PER_WORKER"
require_positive_int "${STRIDE}" "STRIDE"
require_positive_int "${SAVER_CHUNK_SIZE}" "SAVER_CHUNK_SIZE"
require_nonnegative_int "${NUM_ITERS}" "NUM_ITERS"
if [[ -n "${POINT_CLOUD_NUM_POINTS}" ]]; then
    require_positive_int "${POINT_CLOUD_NUM_POINTS}" "POINT_CLOUD_NUM_POINTS"
fi
require_bool01 "${DISABLE_POINT_CLOUD_WORKSPACE_CROP}" "DISABLE_POINT_CLOUD_WORKSPACE_CROP"
require_nonnegative_int "${CAMERA_WARMUP_FRAMES_AFTER_RESET}" "CAMERA_WARMUP_FRAMES_AFTER_RESET"
require_bool01 "${HIDE_TABLE_VISUALS}" "HIDE_TABLE_VISUALS"

read -r -a _GPU_ARR <<< "${GPUS}"
NUM_WORKERS=${#_GPU_ARR[@]}
if (( NUM_WORKERS <= 0 )); then
    echo "ERROR: GPUS is empty." >&2
    exit 1
fi

for gpu in "${_GPU_ARR[@]}"; do
    require_nonnegative_int "${gpu}" "GPUS entry"
done

NUM_DEMOS_PER_WORKER=$(( (DEMOS_PER_CHUNK + NUM_WORKERS - 1) / NUM_WORKERS ))
EFFECTIVE_CHUNK=$(( NUM_DEMOS_PER_WORKER * NUM_WORKERS ))

mkdir -p "${COLLECT_OUT}" "${POINTCLOUD_DIR}" "${LOOP_LOG_DIR}"

INTERRUPTED=0
GOT_HUP=0

on_interrupt() {
    echo
    echo "[interrupt] signal received. Will upload current chunk and exit."
    INTERRUPTED=1
}

on_hup() {
    echo
    echo "[hup] terminal hangup received. Will try to finish current cleanup path."
    INTERRUPTED=1
    GOT_HUP=1
}

trap on_interrupt INT TERM
trap on_hup HUP

PRE_ENTRIES=()

snapshot_collect_out() {
    PRE_ENTRIES=()
    if [[ ! -d "${COLLECT_OUT}" ]]; then
        return 0
    fi

    shopt -s nullglob dotglob
    local p
    for p in "${COLLECT_OUT}"/*; do
        PRE_ENTRIES+=( "$(basename "$p")" )
    done
    shopt -u nullglob dotglob
}

run_collect() {
    snapshot_collect_out

    local save_pc_args=()
    if [[ "${SAVE_POINT_CLOUDS}" == "0" ]]; then
        save_pc_args+=(--skip_point_cloud_write)
    fi

    local pc_args=()
    if [[ -n "${POINT_CLOUD_NUM_POINTS}" ]]; then
        pc_args+=(--point_cloud_num_points "${POINT_CLOUD_NUM_POINTS}")
    fi
    if [[ "${DISABLE_POINT_CLOUD_WORKSPACE_CROP}" == "1" ]]; then
        pc_args+=(--disable_point_cloud_workspace_crop)
    fi
    if [[ "${HIDE_TABLE_VISUALS}" == "1" ]]; then
        pc_args+=(--hide_table_visuals)
    fi
    pc_args+=(--camera_warmup_frames_after_reset "${CAMERA_WARMUP_FRAMES_AFTER_RESET}")

    local run_stamp parent_log
    run_stamp="$(date +%Y-%m-%d_%H-%M-%S)"
    parent_log="${LOOP_LOG_DIR}/${COLLECT_LOOP_NAME}_${run_stamp}_parent.log"

    echo "[run] parent launcher log: ${parent_log}"

    # shellcheck disable=SC2086
    python src/kuafu/data/collection/launch_collect_with_checkpoint.py \
        --gpus ${GPUS} \
        --num_demos_per_worker "${NUM_DEMOS_PER_WORKER}" \
        --num_envs_per_worker "${NUM_ENVS_PER_WORKER}" \
        --task "${TASK}" \
        --checkpoint "${CHECKPOINT}" \
        --prior_checkpoint "${PRIOR_CHECKPOINT}" \
        --prior_mode "${PRIOR_MODE}" \
        --weights_params_json "${WEIGHTS_PARAMS_JSON}" \
        --stride "${STRIDE}" \
        --skip_dense_reward_computation \
        --saver_chunk_size "${SAVER_CHUNK_SIZE}" \
        "${save_pc_args[@]}" \
        "${pc_args[@]}" 2>&1 | tee "${parent_log}"

    return "${PIPESTATUS[0]}"
}

_is_pre_entry() {
    local needle="$1"
    local p
    for p in "${PRE_ENTRIES[@]}"; do
        [[ "$p" == "$needle" ]] && return 0
    done
    return 1
}

stage_pointclouds() {
    if [[ ! -d "${COLLECT_OUT}" ]]; then
        echo "[stage] ${COLLECT_OUT} does not exist, nothing to stage."
        return 0
    fi

    shopt -s nullglob dotglob
    local entries=( "${COLLECT_OUT}"/* )
    shopt -u nullglob dotglob

    local to_move=()
    local p name
    for p in "${entries[@]}"; do
        name="$(basename "$p")"
        if _is_pre_entry "${name}"; then
            continue
        fi
        if [[ ! -d "$p" ]]; then
            continue
        fi
        if [[ "${name}" =~ _worker[0-9]+$ ]]; then
            to_move+=( "$p" )
        fi
    done

    if (( ${#to_move[@]} == 0 )); then
        echo "[stage] no new *_worker## dirs to stage."
        return 0
    fi

    echo "[stage] new worker dirs detected:"
    for p in "${to_move[@]}"; do
        echo "          $p"
    done

    for p in "${to_move[@]}"; do
        find "$p" -depth -name 'episode_tmp_env*' \( -type f -o -type d \) -exec rm -rf {} +
    done

    mkdir -p "${POINTCLOUD_DIR}" || return $?
    echo "[stage] moving ${#to_move[@]} dir(s) -> ${POINTCLOUD_DIR}/"
    mv -f -- "${to_move[@]}" "${POINTCLOUD_DIR}/"
}

upload_and_clean() {
    if [[ ! -d "${POINTCLOUD_DIR}" ]] || [[ -z "$(ls -A "${POINTCLOUD_DIR}")" ]]; then
        echo "[upload] ${POINTCLOUD_DIR} is empty, nothing to upload."
        return 0
    fi

    echo "[upload] syncing ${POINTCLOUD_DIR}/ -> ${S3_DEST}"
    aws s3 sync "${POINTCLOUD_DIR}/" "${S3_DEST}" \
        --endpoint-url="${AWS_ENDPOINT}" \
        --profile "${AWS_PROFILE}" \
        --exclude '*/episode_tmp_env*' \
        --exclude 'episode_tmp_env*' || return $?

    echo "[upload] sync ok, removing local copies under ${POINTCLOUD_DIR}"
    rm -rf "${POINTCLOUD_DIR:?}"/*
}

capture_status() {
    local __result_var="$1"
    shift

    local restore_errexit=0
    case "$-" in
        *e*)
            restore_errexit=1
            set +e
            ;;
    esac

    "$@"
    local __capture_rc=$?

    if (( restore_errexit )); then
        set -e
    fi

    printf -v "${__result_var}" '%s' "${__capture_rc}"
    return 0
}

reorganize_uploaded_folder() {
    echo "[reorganize] consolidating uploaded worker folders under ${S3_DEST}"
    python scripts/data_transfer/reorganize_pointcloud_s3_workers.py \
        --prefix "${S3_DEST}" \
        --execute \
        --keep-source \
        --workers 64
}

iter=0
echo "=========================================="
echo " ${COLLECT_LOOP_NAME}"
echo "  task                 : ${TASK}"
echo "  checkpoint           : ${CHECKPOINT}"
echo "  prior checkpoint     : ${PRIOR_CHECKPOINT}"
echo "  weights/params       : ${WEIGHTS_PARAMS_JSON}"
echo "  gpus                 : ${GPUS}"
echo "  envs/worker          : ${NUM_ENVS_PER_WORKER}"
echo "  stride               : ${STRIDE}"
echo "  saver chunk size     : ${SAVER_CHUNK_SIZE} frame(s)"
echo "  save point clouds    : ${SAVE_POINT_CLOUDS}"
echo "  demos per chunk      : ${DEMOS_PER_CHUNK} (effective ${EFFECTIVE_CHUNK} = ${NUM_DEMOS_PER_WORKER}/worker x ${NUM_WORKERS} workers)"
echo "  num_iters (0=inf)    : ${NUM_ITERS}"
echo "  collect output dir   : ${COLLECT_OUT}"
echo "  pointcloud dir       : ${POINTCLOUD_DIR}"
echo "  loop log dir         : ${LOOP_LOG_DIR}"
echo "  s3 dest              : ${S3_DEST}"
echo "  aws profile/endpoint : ${AWS_PROFILE} @ ${AWS_ENDPOINT}"
if [[ -n "${POINT_CLOUD_NUM_POINTS}" ]]; then
    echo "  point_cloud cap/frame: ${POINT_CLOUD_NUM_POINTS}"
else
    echo "  point_cloud cap/frame: none"
fi
if [[ "${DISABLE_POINT_CLOUD_WORKSPACE_CROP}" == "1" ]]; then
    echo "  point_cloud xyz crop : disabled"
else
    echo "  point_cloud xyz crop : workspace limits"
fi
if [[ "${HIDE_TABLE_VISUALS}" == "1" ]]; then
    echo "  table visuals        : hidden"
else
    echo "  table visuals        : visible"
fi
echo "  camera reset warmup  : ${CAMERA_WARMUP_FRAMES_AFTER_RESET} render frame(s)"
echo "=========================================="

while :; do
    iter=$((iter + 1))
    echo
    echo "------ iteration ${iter} ------"

    capture_status collect_rc run_collect

    if (( collect_rc != 0 )); then
        echo "[loop] collection exited with rc=${collect_rc}; staging/uploading any completed worker dirs before exiting." >&2
    fi

    capture_status stage_rc stage_pointclouds
    if (( stage_rc != 0 )); then
        echo "[loop] staging failed with rc=${stage_rc}." >&2
        exit "${stage_rc}"
    fi

    capture_status upload_rc upload_and_clean
    if (( upload_rc != 0 )); then
        echo "[loop] upload/cleanup failed with rc=${upload_rc}." >&2
        exit "${upload_rc}"
    fi

    if (( collect_rc != 0 )); then
        exit "${collect_rc}"
    fi

    if (( INTERRUPTED )); then
        echo "[loop] interrupted, exiting after iteration ${iter}."
        break
    fi

    if (( NUM_ITERS > 0 )) && (( iter >= NUM_ITERS )); then
        echo "[loop] reached NUM_ITERS=${NUM_ITERS}, done."
        break
    fi
done

reorganize_uploaded_folder
