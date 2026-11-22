#!/usr/bin/env bash
# Shared collection/upload loop. Source this from a task-specific wrapper after
# defining the task checkpoint, prior, and output naming variables.

set -uo pipefail

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "ERROR: collect_and_upload_loop_common.sh must be sourced from a task wrapper." >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}" || return $?
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
COLLECT_PYTHON="${KUAFU_PYTHON:-python}"
KUAFU_ARTIFACT_ROOT="$("${COLLECT_PYTHON}" -c 'from kuafu.paths import external_artifact_root; print(external_artifact_root())')" || return $?
export KUAFU_ARTIFACT_ROOT

case ":${PATH}:" in
    *":${HOME}/.local/bin:"*) ;;
    *) export PATH="${HOME}/.local/bin:${PATH}" ;;
esac

require_var() {
    local name="$1"
    if [[ -z "${!name:-}" ]]; then
        echo "ERROR: ${name} must be set by the task wrapper." >&2
        return 1
    fi
}

require_file() {
    local path="$1"
    local label="$2"
    if [[ ! -f "${path}" ]]; then
        echo "ERROR: ${label} does not exist: ${path}" >&2
        return 1
    fi
}

require_positive_int() {
    local value="$1"
    local label="$2"
    if [[ ! "${value}" =~ ^[0-9]+$ ]] || (( value <= 0 )); then
        echo "ERROR: ${label} must be a positive integer, got '${value}'." >&2
        return 1
    fi
}

require_nonnegative_int() {
    local value="$1"
    local label="$2"
    if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
        echo "ERROR: ${label} must be a nonnegative integer, got '${value}'." >&2
        return 1
    fi
}

require_bool01() {
    local value="$1"
    local label="$2"
    if [[ "${value}" != "0" && "${value}" != "1" ]]; then
        echo "ERROR: ${label} must be 0 or 1, got '${value}'." >&2
        return 1
    fi
}

require_crop_bounds() {
    local value="$1"
    local label="$2"
    [[ -z "${value}" ]] && return 0

    local _bounds_arr=()
    read -r -a _bounds_arr <<< "${value}"
    if (( ${#_bounds_arr[@]} != 3 )); then
        echo "ERROR: ${label} must contain exactly 3 values, got '${value}'." >&2
        return 1
    fi

    local bound
    for bound in "${_bounds_arr[@]}"; do
        if [[ "${bound}" == "none" || "${bound}" == "null" ]]; then
            continue
        fi
        if [[ ! "${bound}" =~ ^-?([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]]; then
            echo "ERROR: ${label} values must be floats or none, got '${bound}' in '${value}'." >&2
            return 1
        fi
    done
}

require_vec2_bounds() {
    local value="$1"
    local label="$2"
    [[ -z "${value}" ]] && return 0

    local _bounds_arr=()
    read -r -a _bounds_arr <<< "${value}"
    if (( ${#_bounds_arr[@]} != 2 )); then
        echo "ERROR: ${label} must contain exactly 2 values, got '${value}'." >&2
        return 1
    fi

    local bound
    for bound in "${_bounds_arr[@]}"; do
        if [[ ! "${bound}" =~ ^-?([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]]; then
            echo "ERROR: ${label} values must be floats, got '${bound}' in '${value}'." >&2
            return 1
        fi
    done
}

require_float() {
    local value="$1"
    local label="$2"
    [[ -z "${value}" ]] && return 0
    if [[ ! "${value}" =~ ^-?([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]]; then
        echo "ERROR: ${label} must be a float, got '${value}'." >&2
        return 1
    fi
}

require_positive_float() {
    local value="$1"
    local label="$2"
    require_float "${value}" "${label}" || return $?
    [[ -z "${value}" ]] && return 0
    awk -v v="${value}" 'BEGIN { exit !(v > 0.0) }' || {
        echo "ERROR: ${label} must be positive, got '${value}'." >&2
        return 1
    }
}

require_var GPUS || return $?
require_var DEMOS_PER_CHUNK || return $?
require_var NUM_ENVS_PER_WORKER || return $?
require_var TASK || return $?
require_var CHECKPOINT || return $?
require_var PRIOR_CHECKPOINT || return $?
require_var PRIOR_MODE || return $?
require_var WEIGHTS_PARAMS_JSON || return $?
require_var NUM_ITERS || return $?

if [[ ! ${STRIDE+x} ]]; then STRIDE=1; fi
if [[ ! ${SAVER_CHUNK_SIZE+x} ]]; then SAVER_CHUNK_SIZE=64; fi
if [[ ! ${SAVE_POINT_CLOUDS+x} ]]; then SAVE_POINT_CLOUDS=0; fi
if [[ ! ${SAVE_RGB+x} ]]; then SAVE_RGB=0; fi
if [[ ! ${POINT_CLOUD_NUM_POINTS+x} ]]; then POINT_CLOUD_NUM_POINTS="16384"; fi
if [[ ! ${DISABLE_POINT_CLOUD_WORKSPACE_CROP+x} ]]; then
    if [[ "${TASK}" == Panda-* ]]; then
        DISABLE_POINT_CLOUD_WORKSPACE_CROP=0
    else
        DISABLE_POINT_CLOUD_WORKSPACE_CROP=1
    fi
fi
if [[ ! ${POINT_CLOUD_WORKSPACE_MIN+x} ]]; then POINT_CLOUD_WORKSPACE_MIN=""; fi
if [[ ! ${POINT_CLOUD_WORKSPACE_MAX+x} ]]; then POINT_CLOUD_WORKSPACE_MAX=""; fi
if [[ ! ${DISABLE_POINT_CLOUD_TABLE_PLANE_FILTER+x} ]]; then
    if [[ "${TASK}" == Panda-* ]]; then
        DISABLE_POINT_CLOUD_TABLE_PLANE_FILTER=1
    else
        DISABLE_POINT_CLOUD_TABLE_PLANE_FILTER=0
    fi
fi
if [[ ! ${POINT_CLOUD_EXCLUDE_BOX_MIN+x} ]]; then POINT_CLOUD_EXCLUDE_BOX_MIN=""; fi
if [[ ! ${POINT_CLOUD_EXCLUDE_BOX_MAX+x} ]]; then POINT_CLOUD_EXCLUDE_BOX_MAX=""; fi
if [[ ! ${POINT_CLOUD_EXCLUDE_CYLINDER_CENTER+x} ]]; then POINT_CLOUD_EXCLUDE_CYLINDER_CENTER=""; fi
if [[ ! ${POINT_CLOUD_EXCLUDE_CYLINDER_RADIUS+x} ]]; then POINT_CLOUD_EXCLUDE_CYLINDER_RADIUS=""; fi
if [[ ! ${POINT_CLOUD_EXCLUDE_CYLINDER_Z_MIN+x} ]]; then POINT_CLOUD_EXCLUDE_CYLINDER_Z_MIN=""; fi
if [[ ! ${POINT_CLOUD_EXCLUDE_CYLINDER_Z_MAX+x} ]]; then POINT_CLOUD_EXCLUDE_CYLINDER_Z_MAX=""; fi
if [[ ! ${CAMERA_WARMUP_FRAMES_AFTER_RESET+x} ]]; then CAMERA_WARMUP_FRAMES_AFTER_RESET=2; fi
if [[ ! ${CAMERA_KEYS+x} ]]; then CAMERA_KEYS=""; fi
if [[ ! ${EXPECTED_NUM_CAMERAS+x} ]]; then EXPECTED_NUM_CAMERAS=""; fi
if [[ ! ${USE_TILED_CAMERAS+x} ]]; then USE_TILED_CAMERAS=0; fi
if [[ ! ${CHECKPOINT_PARAMS_DIR+x} ]]; then CHECKPOINT_PARAMS_DIR=""; fi
if [[ ! ${CHECKPOINT_PARAMS_REQUIRED_FILES+x} ]]; then
    CHECKPOINT_PARAMS_REQUIRED_FILES="agent.yaml env.yaml"
fi
if [[ ! ${HIDE_TABLE_VISUALS+x} ]]; then HIDE_TABLE_VISUALS=1; fi
if [[ ! ${DISABLE_LEFT_RIGHT_RANDOMIZATION+x} ]]; then DISABLE_LEFT_RIGHT_RANDOMIZATION=0; fi
if [[ ! ${FORCE_LEFT_RIGHT_SWAP+x} ]]; then FORCE_LEFT_RIGHT_SWAP=0; fi
if [[ ! ${AWS_ENDPOINT+x} ]]; then AWS_ENDPOINT="https://s3.wasabisys.com"; fi
if [[ ! ${AWS_PROFILE+x} ]]; then AWS_PROFILE="kuafu"; fi
if [[ ! ${DATASET_NAME+x} ]]; then DATASET_NAME="${TASK}-downsampled"; fi
if [[ ! ${S3_ROOT+x} ]]; then S3_ROOT="s3://kuafu/pointclouds-match"; fi
if [[ ! ${COLLECT_LOOP_NAME+x} ]]; then
    COLLECT_LOOP_NAME="$(basename "${BASH_SOURCE[1]}" .sh)"
fi
if [[ ! ${LOCAL_ORGANIZE_ZARR_BEFORE_UPLOAD+x} ]]; then LOCAL_ORGANIZE_ZARR_BEFORE_UPLOAD=0; fi
if [[ ! ${LOCAL_UPLOAD_CLEAN_SOURCE_AFTER+x} ]]; then LOCAL_UPLOAD_CLEAN_SOURCE_AFTER=0; fi

S3_ROOT="${S3_ROOT%/}"

[[ -z "${COLLECT_OUT:-}"    ]] && COLLECT_OUT="${KUAFU_ARTIFACT_ROOT:-${REPO_ROOT}/../KuafuArtifacts}/collection/${TASK}"
[[ -z "${POINTCLOUD_DIR:-}" ]] && POINTCLOUD_DIR="${KUAFU_ARTIFACT_ROOT:-${REPO_ROOT}/../KuafuArtifacts}/pointclouds/${DATASET_NAME}"
[[ -z "${S3_DEST:-}"        ]] && S3_DEST="${S3_ROOT}/${DATASET_NAME}/"
[[ -z "${LOOP_LOG_DIR:-}"   ]] && LOOP_LOG_DIR="${COLLECT_OUT}/loop_logs"

require_file "${CHECKPOINT}" "CHECKPOINT" || return $?
require_file "${PRIOR_CHECKPOINT}" "PRIOR_CHECKPOINT" || return $?
require_file "${WEIGHTS_PARAMS_JSON}" "WEIGHTS_PARAMS_JSON" || return $?
_CHECKPOINT_PARAMS_REQUIRED_FILE_ARR=()
if [[ -n "${CHECKPOINT_PARAMS_DIR}" ]]; then
    if [[ ! -d "${CHECKPOINT_PARAMS_DIR}" ]]; then
        echo "ERROR: CHECKPOINT_PARAMS_DIR does not exist: ${CHECKPOINT_PARAMS_DIR}" >&2
        return 1
    fi
    read -r -a _CHECKPOINT_PARAMS_REQUIRED_FILE_ARR <<< "${CHECKPOINT_PARAMS_REQUIRED_FILES}"
    for params_file in "${_CHECKPOINT_PARAMS_REQUIRED_FILE_ARR[@]}"; do
        require_file "${CHECKPOINT_PARAMS_DIR}/${params_file}" "CHECKPOINT_PARAMS_DIR/${params_file}" || return $?
    done
fi
require_positive_int "${DEMOS_PER_CHUNK}" "DEMOS_PER_CHUNK" || return $?
require_positive_int "${NUM_ENVS_PER_WORKER}" "NUM_ENVS_PER_WORKER" || return $?
require_positive_int "${STRIDE}" "STRIDE" || return $?
require_positive_int "${SAVER_CHUNK_SIZE}" "SAVER_CHUNK_SIZE" || return $?
require_bool01 "${SAVE_POINT_CLOUDS}" "SAVE_POINT_CLOUDS" || return $?
require_bool01 "${SAVE_RGB}" "SAVE_RGB" || return $?
require_nonnegative_int "${NUM_ITERS}" "NUM_ITERS" || return $?
if [[ -n "${POINT_CLOUD_NUM_POINTS}" ]]; then
    require_positive_int "${POINT_CLOUD_NUM_POINTS}" "POINT_CLOUD_NUM_POINTS" || return $?
fi
require_bool01 "${DISABLE_POINT_CLOUD_WORKSPACE_CROP}" "DISABLE_POINT_CLOUD_WORKSPACE_CROP" || return $?
require_bool01 "${DISABLE_POINT_CLOUD_TABLE_PLANE_FILTER}" "DISABLE_POINT_CLOUD_TABLE_PLANE_FILTER" || return $?
require_crop_bounds "${POINT_CLOUD_WORKSPACE_MIN}" "POINT_CLOUD_WORKSPACE_MIN" || return $?
require_crop_bounds "${POINT_CLOUD_WORKSPACE_MAX}" "POINT_CLOUD_WORKSPACE_MAX" || return $?
require_crop_bounds "${POINT_CLOUD_EXCLUDE_BOX_MIN}" "POINT_CLOUD_EXCLUDE_BOX_MIN" || return $?
require_crop_bounds "${POINT_CLOUD_EXCLUDE_BOX_MAX}" "POINT_CLOUD_EXCLUDE_BOX_MAX" || return $?
require_vec2_bounds "${POINT_CLOUD_EXCLUDE_CYLINDER_CENTER}" "POINT_CLOUD_EXCLUDE_CYLINDER_CENTER" || return $?
require_positive_float "${POINT_CLOUD_EXCLUDE_CYLINDER_RADIUS}" "POINT_CLOUD_EXCLUDE_CYLINDER_RADIUS" || return $?
require_float "${POINT_CLOUD_EXCLUDE_CYLINDER_Z_MIN}" "POINT_CLOUD_EXCLUDE_CYLINDER_Z_MIN" || return $?
require_float "${POINT_CLOUD_EXCLUDE_CYLINDER_Z_MAX}" "POINT_CLOUD_EXCLUDE_CYLINDER_Z_MAX" || return $?
if [[ -n "${POINT_CLOUD_EXCLUDE_CYLINDER_CENTER}" || -n "${POINT_CLOUD_EXCLUDE_CYLINDER_RADIUS}" ]]; then
    if [[ -z "${POINT_CLOUD_EXCLUDE_CYLINDER_CENTER}" || -z "${POINT_CLOUD_EXCLUDE_CYLINDER_RADIUS}" ]]; then
        echo "ERROR: POINT_CLOUD_EXCLUDE_CYLINDER_CENTER and POINT_CLOUD_EXCLUDE_CYLINDER_RADIUS must be set together." >&2
        return 1
    fi
fi
require_nonnegative_int "${CAMERA_WARMUP_FRAMES_AFTER_RESET}" "CAMERA_WARMUP_FRAMES_AFTER_RESET" || return $?
require_bool01 "${USE_TILED_CAMERAS}" "USE_TILED_CAMERAS" || return $?
require_bool01 "${HIDE_TABLE_VISUALS}" "HIDE_TABLE_VISUALS" || return $?
require_bool01 "${DISABLE_LEFT_RIGHT_RANDOMIZATION}" "DISABLE_LEFT_RIGHT_RANDOMIZATION" || return $?
require_bool01 "${FORCE_LEFT_RIGHT_SWAP}" "FORCE_LEFT_RIGHT_SWAP" || return $?
if [[ "${DISABLE_LEFT_RIGHT_RANDOMIZATION}" == "1" && "${FORCE_LEFT_RIGHT_SWAP}" == "1" ]]; then
    echo "ERROR: DISABLE_LEFT_RIGHT_RANDOMIZATION and FORCE_LEFT_RIGHT_SWAP are mutually exclusive." >&2
    return 1
fi
require_bool01 "${LOCAL_ORGANIZE_ZARR_BEFORE_UPLOAD}" "LOCAL_ORGANIZE_ZARR_BEFORE_UPLOAD" || return $?
require_bool01 "${LOCAL_UPLOAD_CLEAN_SOURCE_AFTER}" "LOCAL_UPLOAD_CLEAN_SOURCE_AFTER" || return $?

read -r -a _GPU_ARR <<< "${GPUS}"
NUM_WORKERS=${#_GPU_ARR[@]}
if (( NUM_WORKERS <= 0 )); then
    echo "ERROR: GPUS is empty." >&2
    return 1
fi

for gpu in "${_GPU_ARR[@]}"; do
    require_nonnegative_int "${gpu}" "GPUS entry" || return $?
done

_CAMERA_KEY_ARR=()
if [[ -n "${CAMERA_KEYS//[[:space:]]/}" ]]; then
    read -r -a _CAMERA_KEY_ARR <<< "${CAMERA_KEYS}"
fi
if [[ -n "${EXPECTED_NUM_CAMERAS}" ]]; then
    require_positive_int "${EXPECTED_NUM_CAMERAS}" "EXPECTED_NUM_CAMERAS" || return $?
    if (( ${#_CAMERA_KEY_ARR[@]} != EXPECTED_NUM_CAMERAS )); then
        echo "ERROR: expected ${EXPECTED_NUM_CAMERAS} CAMERA_KEYS entries, got ${#_CAMERA_KEY_ARR[@]}: '${CAMERA_KEYS}'." >&2
        return 1
    fi
fi
for (( i = 0; i < ${#_CAMERA_KEY_ARR[@]}; i++ )); do
    for (( j = i + 1; j < ${#_CAMERA_KEY_ARR[@]}; j++ )); do
        if [[ "${_CAMERA_KEY_ARR[i]}" == "${_CAMERA_KEY_ARR[j]}" ]]; then
            echo "ERROR: CAMERA_KEYS contains duplicate sensor '${_CAMERA_KEY_ARR[i]}'." >&2
            return 1
        fi
    done
done

NUM_DEMOS_PER_WORKER=$(( (DEMOS_PER_CHUNK + NUM_WORKERS - 1) / NUM_WORKERS ))
EFFECTIVE_CHUNK=$(( NUM_DEMOS_PER_WORKER * NUM_WORKERS ))

# Validate output overrides before creating directories or staging/deleting files.
for output_var in COLLECT_OUT POINTCLOUD_DIR LOOP_LOG_DIR; do
    output_path="$("${COLLECT_PYTHON}" -c 'import sys; from kuafu.paths import resolve_artifact_path; print(resolve_artifact_path(sys.argv[1], sys.argv[2]))' "${!output_var}" "${output_var}")" || return $?
    printf -v "${output_var}" '%s' "${output_path}"
done
mkdir -p "${COLLECT_OUT}" "${POINTCLOUD_DIR}" "${LOOP_LOG_DIR}" || return $?

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
    local save_rgb_args=()
    if [[ "${SAVE_RGB}" == "1" ]]; then
        save_rgb_args+=(--save_rgb)
    fi

    local camera_args=()
    if (( ${#_CAMERA_KEY_ARR[@]} > 0 )); then
        camera_args+=(--camera_keys "${_CAMERA_KEY_ARR[@]}")
    fi
    if [[ "${USE_TILED_CAMERAS}" == "1" ]]; then
        camera_args+=(--use_tiled_cameras)
    fi

    local checkpoint_params_args=()
    if [[ -n "${CHECKPOINT_PARAMS_DIR}" ]]; then
        checkpoint_params_args+=(--checkpoint_params_dir "${CHECKPOINT_PARAMS_DIR}")
    fi

    local pc_args=()
    if [[ -n "${POINT_CLOUD_NUM_POINTS}" ]]; then
        pc_args+=(--point_cloud_num_points "${POINT_CLOUD_NUM_POINTS}")
    fi
    if [[ "${DISABLE_POINT_CLOUD_WORKSPACE_CROP}" == "1" ]]; then
        pc_args+=(--disable_point_cloud_workspace_crop)
    else
        if [[ -n "${POINT_CLOUD_WORKSPACE_MIN}" ]]; then
            local _pc_crop_min=()
            read -r -a _pc_crop_min <<< "${POINT_CLOUD_WORKSPACE_MIN}"
            pc_args+=(--point_cloud_workspace_min "${_pc_crop_min[@]}")
        fi
        if [[ -n "${POINT_CLOUD_WORKSPACE_MAX}" ]]; then
            local _pc_crop_max=()
            read -r -a _pc_crop_max <<< "${POINT_CLOUD_WORKSPACE_MAX}"
            pc_args+=(--point_cloud_workspace_max "${_pc_crop_max[@]}")
        fi
    fi
    if [[ "${DISABLE_POINT_CLOUD_TABLE_PLANE_FILTER}" == "1" ]]; then
        pc_args+=(--disable_point_cloud_table_plane_filter)
    fi
    if [[ -n "${POINT_CLOUD_EXCLUDE_BOX_MIN}" ]]; then
        local _pc_exclude_min=()
        read -r -a _pc_exclude_min <<< "${POINT_CLOUD_EXCLUDE_BOX_MIN}"
        pc_args+=(--point_cloud_exclude_box_min "${_pc_exclude_min[@]}")
    fi
    if [[ -n "${POINT_CLOUD_EXCLUDE_BOX_MAX}" ]]; then
        local _pc_exclude_max=()
        read -r -a _pc_exclude_max <<< "${POINT_CLOUD_EXCLUDE_BOX_MAX}"
        pc_args+=(--point_cloud_exclude_box_max "${_pc_exclude_max[@]}")
    fi
    if [[ -n "${POINT_CLOUD_EXCLUDE_CYLINDER_CENTER}" ]]; then
        local _pc_exclude_cylinder_center=()
        read -r -a _pc_exclude_cylinder_center <<< "${POINT_CLOUD_EXCLUDE_CYLINDER_CENTER}"
        pc_args+=(--point_cloud_exclude_cylinder_center_xy "${_pc_exclude_cylinder_center[@]}")
    fi
    if [[ -n "${POINT_CLOUD_EXCLUDE_CYLINDER_RADIUS}" ]]; then
        pc_args+=(--point_cloud_exclude_cylinder_radius "${POINT_CLOUD_EXCLUDE_CYLINDER_RADIUS}")
    fi
    if [[ -n "${POINT_CLOUD_EXCLUDE_CYLINDER_Z_MIN}" ]]; then
        pc_args+=(--point_cloud_exclude_cylinder_z_min "${POINT_CLOUD_EXCLUDE_CYLINDER_Z_MIN}")
    fi
    if [[ -n "${POINT_CLOUD_EXCLUDE_CYLINDER_Z_MAX}" ]]; then
        pc_args+=(--point_cloud_exclude_cylinder_z_max "${POINT_CLOUD_EXCLUDE_CYLINDER_Z_MAX}")
    fi
    if [[ "${HIDE_TABLE_VISUALS}" == "1" ]]; then
        pc_args+=(--hide_table_visuals)
    fi
    if [[ "${FORCE_LEFT_RIGHT_SWAP}" == "1" ]]; then
        pc_args+=(--force_left_right_swap)
    elif [[ "${DISABLE_LEFT_RIGHT_RANDOMIZATION}" == "1" ]]; then
        pc_args+=(--disable_left_right_randomization)
    fi
    pc_args+=(--camera_warmup_frames_after_reset "${CAMERA_WARMUP_FRAMES_AFTER_RESET}")

    local run_stamp parent_log
    run_stamp="$(date +%Y-%m-%d_%H-%M-%S)_${iter}"
    parent_log="${LOOP_LOG_DIR}/${COLLECT_LOOP_NAME}_${run_stamp}_parent.log"

    echo "[run] parent launcher log: ${parent_log}"

    # shellcheck disable=SC2086
    "${COLLECT_PYTHON}" src/kuafu/data/collection/launch_collect_with_checkpoint.py \
        --gpus ${GPUS} \
        --output_root "${COLLECT_OUT}/${run_stamp}" \
        --num_demos_per_worker "${NUM_DEMOS_PER_WORKER}" \
        --num_envs_per_worker "${NUM_ENVS_PER_WORKER}" \
        --task "${TASK}" \
        --checkpoint "${CHECKPOINT}" \
        --prior_checkpoint "${PRIOR_CHECKPOINT}" \
        --prior_mode "${PRIOR_MODE}" \
        --weights_params_json "${WEIGHTS_PARAMS_JSON}" \
        "${checkpoint_params_args[@]}" \
        --stride "${STRIDE}" \
        --skip_dense_reward_computation \
        --saver_chunk_size "${SAVER_CHUNK_SIZE}" \
        "${camera_args[@]}" \
        "${save_rgb_args[@]}" \
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

    local to_move=() destinations=()
    local p name worker worker_name
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
            destinations+=( "${POINTCLOUD_DIR}/${name}" )
        elif [[ -d "${p}/workers" ]]; then
            for worker in "${p}"/workers/worker_*; do
                [[ -d "${worker}/datasets" ]] || continue
                worker_name="${worker##*/}"
                to_move+=( "${worker}" )
                destinations+=( "${POINTCLOUD_DIR}/${name}_${worker_name/_/}" )
            done
        fi
    done

    if (( ${#to_move[@]} == 0 )); then
        echo "[stage] no new worker datasets to stage."
        return 0
    fi

    echo "[stage] new worker dirs detected:"
    for p in "${destinations[@]}"; do
        if [[ -e "${p}" ]]; then
            echo "ERROR: refusing to overwrite staged collection: ${p}" >&2
            return 1
        fi
    done
    for p in "${to_move[@]}"; do
        echo "          $p"
    done

    for p in "${to_move[@]}"; do
        find "$p" -depth -name 'episode_tmp_env*' \( -type f -o -type d \) -exec rm -rf {} +
    done

    mkdir -p "${POINTCLOUD_DIR}" || return $?
    echo "[stage] moving ${#to_move[@]} dir(s) -> ${POINTCLOUD_DIR}/"
    local index
    for index in "${!to_move[@]}"; do
        mv -- "${to_move[index]}" "${destinations[index]}" || return $?
    done
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

organize_zarr_upload_and_optionally_clean() {
    if [[ ! -d "${POINTCLOUD_DIR}" ]] || [[ -z "$(ls -A "${POINTCLOUD_DIR}")" ]]; then
        echo "[postprocess] ${POINTCLOUD_DIR} is empty, nothing to organize/upload."
        return 0
    fi

    local postprocess_script="${SCRIPT_DIR}/local_pointcloud_organize_zarr_upload.sh"
    if [[ ! -x "${postprocess_script}" ]]; then
        echo "ERROR: postprocess script is not executable: ${postprocess_script}" >&2
        return 1
    fi

    "${postprocess_script}" \
        --task "${DATASET_NAME}" \
        --source-root "${POINTCLOUD_DIR}" \
        --pointcloud-s3-dest "${S3_DEST}" \
        --aws-profile "${AWS_PROFILE}" \
        --aws-endpoint "${AWS_ENDPOINT}" || return $?

    if [[ "${LOCAL_UPLOAD_CLEAN_SOURCE_AFTER}" == "1" ]]; then
        echo "[postprocess] upload ok, removing staged worker shards under ${POINTCLOUD_DIR}"
        rm -rf "${POINTCLOUD_DIR:?}"/*
    else
        echo "[postprocess] upload ok, leaving local staged worker shards in ${POINTCLOUD_DIR}"
    fi
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

iter=0
echo "=========================================="
echo " ${COLLECT_LOOP_NAME}"
echo "  task                 : ${TASK}"
echo "  checkpoint           : ${CHECKPOINT}"
echo "  prior checkpoint     : ${PRIOR_CHECKPOINT}"
echo "  weights/params       : ${WEIGHTS_PARAMS_JSON}"
if [[ -n "${CHECKPOINT_PARAMS_DIR}" ]]; then
    echo "  checkpoint params    : ${CHECKPOINT_PARAMS_DIR} (${CHECKPOINT_PARAMS_REQUIRED_FILES})"
else
    echo "  checkpoint params    : archive auto-discovery"
fi
echo "  gpus                 : ${GPUS}"
echo "  envs/worker          : ${NUM_ENVS_PER_WORKER}"
echo "  stride               : ${STRIDE}"
echo "  saver chunk size     : ${SAVER_CHUNK_SIZE} frame(s)"
echo "  save point clouds    : ${SAVE_POINT_CLOUDS}"
echo "  save RGB images      : ${SAVE_RGB}"
if (( ${#_CAMERA_KEY_ARR[@]} > 0 )); then
    echo "  simulation cameras   : ${CAMERA_KEYS} (${#_CAMERA_KEY_ARR[@]} views)"
else
    echo "  simulation cameras   : auto-discover registered camera sensors"
fi
echo "  tiled camera setup   : ${USE_TILED_CAMERAS}"
echo "  demos per chunk      : ${DEMOS_PER_CHUNK} (effective ${EFFECTIVE_CHUNK} = ${NUM_DEMOS_PER_WORKER}/worker x ${NUM_WORKERS} workers)"
echo "  num_iters (0=inf)    : ${NUM_ITERS}"
echo "  collect output dir   : ${COLLECT_OUT}"
echo "  pointcloud dir       : ${POINTCLOUD_DIR}"
echo "  loop log dir         : ${LOOP_LOG_DIR}"
echo "  s3 dest              : ${S3_DEST}"
echo "  aws profile/endpoint : ${AWS_PROFILE} @ ${AWS_ENDPOINT}"
if [[ "${LOCAL_ORGANIZE_ZARR_BEFORE_UPLOAD}" == "1" ]]; then
    echo "  postprocess upload   : local organize -> zarr -> zarr upload -> H5 upload"
else
    echo "  postprocess upload   : raw staged worker sync"
fi
if [[ -n "${POINT_CLOUD_NUM_POINTS}" ]]; then
    echo "  point_cloud cap/frame: ${POINT_CLOUD_NUM_POINTS}"
else
    echo "  point_cloud cap/frame: none"
fi
if [[ "${DISABLE_POINT_CLOUD_WORKSPACE_CROP}" == "1" ]]; then
    echo "  point_cloud xyz crop : disabled"
elif [[ -n "${POINT_CLOUD_WORKSPACE_MIN}" || -n "${POINT_CLOUD_WORKSPACE_MAX}" ]]; then
    echo "  point_cloud xyz crop : custom workspace limits"
    echo "  point_cloud crop min : ${POINT_CLOUD_WORKSPACE_MIN:-default}"
    echo "  point_cloud crop max : ${POINT_CLOUD_WORKSPACE_MAX:-default}"
elif [[ "${TASK}" == Panda-* ]]; then
    echo "  point_cloud xyz crop : Panda default workspace limits"
else
    echo "  point_cloud xyz crop : workspace limits"
fi
if [[ "${DISABLE_POINT_CLOUD_TABLE_PLANE_FILTER}" == "1" ]]; then
    echo "  point_cloud plane crop: disabled"
elif [[ "${TASK}" == Panda-* ]]; then
    echo "  point_cloud plane crop: tilted table plane"
fi
if [[ -n "${POINT_CLOUD_EXCLUDE_BOX_MIN}" || -n "${POINT_CLOUD_EXCLUDE_BOX_MAX}" ]]; then
    echo "  point_cloud exclude min: ${POINT_CLOUD_EXCLUDE_BOX_MIN:-none}"
    echo "  point_cloud exclude max: ${POINT_CLOUD_EXCLUDE_BOX_MAX:-none}"
fi
if [[ -n "${POINT_CLOUD_EXCLUDE_CYLINDER_CENTER}" || -n "${POINT_CLOUD_EXCLUDE_CYLINDER_RADIUS}" ]]; then
    echo "  point_cloud exclude cylinder center: ${POINT_CLOUD_EXCLUDE_CYLINDER_CENTER:-none}"
    echo "  point_cloud exclude cylinder radius: ${POINT_CLOUD_EXCLUDE_CYLINDER_RADIUS:-none}"
    echo "  point_cloud exclude cylinder z min : ${POINT_CLOUD_EXCLUDE_CYLINDER_Z_MIN:-none}"
    echo "  point_cloud exclude cylinder z max : ${POINT_CLOUD_EXCLUDE_CYLINDER_Z_MAX:-none}"
fi
if [[ "${HIDE_TABLE_VISUALS}" == "1" ]]; then
    echo "  table visuals        : hidden"
else
    echo "  table visuals        : visible"
fi
if [[ "${FORCE_LEFT_RIGHT_SWAP}" == "1" ]]; then
    echo "  left/right assignment: fixed (swap=True)"
elif [[ "${DISABLE_LEFT_RIGHT_RANDOMIZATION}" == "1" ]]; then
    echo "  left/right assignment: fixed (swap=False)"
else
    echo "  left/right assignment: randomized"
fi
echo "  camera reset warmup  : ${CAMERA_WARMUP_FRAMES_AFTER_RESET} render frame(s)"
echo "=========================================="

while :; do
    iter=$((iter + 1))
    echo
    echo "------ iteration ${iter} ------"

    capture_status collect_rc run_collect

    if (( collect_rc != 0 )); then
        echo "[loop] collection exited with rc=${collect_rc}; staging/uploading any completed worker dirs before returning." >&2
    fi

    capture_status stage_rc stage_pointclouds
    if (( stage_rc != 0 )); then
        echo "[loop] staging failed with rc=${stage_rc}." >&2
        return "${stage_rc}"
    fi

    if [[ "${LOCAL_ORGANIZE_ZARR_BEFORE_UPLOAD}" == "1" ]]; then
        capture_status upload_rc organize_zarr_upload_and_optionally_clean
    else
        capture_status upload_rc upload_and_clean
    fi
    if (( upload_rc != 0 )); then
        echo "[loop] upload/cleanup failed with rc=${upload_rc}." >&2
        return "${upload_rc}"
    fi

    if (( collect_rc != 0 )); then
        echo "[loop] returning rc=${collect_rc} instead of hard-exiting the shell."
        return "${collect_rc}"
    fi

    if (( INTERRUPTED )); then
        echo "[loop] interrupted, returning after iteration ${iter}."
        return 0
    fi

    if (( NUM_ITERS > 0 )) && (( iter >= NUM_ITERS )); then
        echo "[loop] reached NUM_ITERS=${NUM_ITERS}, done."
        return 0
    fi
done
