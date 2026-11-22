#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Collect a small pointcloud preview with the best Stack-2 checkpoint, then
# render the collected HDF5 episodes to MP4 locally. No upload or staged-data
# deletion.
#
# Best checkpoint source:
#   Episode/Kuafu/success_rate_window = 0.83528435 at epoch 1489
#   run: Residuals/FR3-FinRay-Table-Stack-2-ff/2026-05-17_11-55-03
#   worker: 2026-05-17_11-56-02_Run-0, checkpoint epoch 1450
# ---------------------------------------------------------------------------

set -euo pipefail

# =========================== EDIT ME =====================================
# Forwarded to launch_collect_with_checkpoint.py
GPUS="${GPUS:-0 1 2}"
PYTHON_BIN="${PYTHON_BIN:-${KUAFU_PYTHON:-python}}"
DEMOS_TO_COLLECT="${DEMOS_TO_COLLECT:-12}"
NUM_ENVS_PER_WORKER="${NUM_ENVS_PER_WORKER:-4}"                       # keep 1 for exact small preview counts
TASK="${TASK:-FR3-FinRay-Table-Stack-2-ff}"
CHECKPOINT="${CHECKPOINT:-${KUAFU_ARTIFACT_ROOT:-../KuafuArtifacts}/Residuals/FR3-FinRay-Table-Stack-2-ff/2026-05-17_11-55-03/rl_runs/2026-05-17_11-56-02_Run-0/nn/last_FR3-FinRay-Table-Stack-2_ep_1450_rew_3375.3599.pth}"
PRIOR_CHECKPOINT="${PRIOR_CHECKPOINT:-${KUAFU_ARTIFACT_ROOT:-../KuafuArtifacts}/Priors/thor/stack_2_damped_higher_noise_state/stack2_chunking_15_shift_ff_damped_higher/best.pth}"
PRIOR_MODE="${PRIOR_MODE:-residual_rl}"
WEIGHTS_PARAMS_JSON="${WEIGHTS_PARAMS_JSON:-${KUAFU_ARTIFACT_ROOT:-../KuafuArtifacts}/Residuals/FR3-FinRay-Table-Stack-2-ff/2026-05-17_11-53-49/best.json}"
STRIDE="${STRIDE:-1}"


# Point cloud HDF5: empty = use collect_with_checkpoint.py default.
# Set a positive integer for uniform random subsampling per frame.
POINT_CLOUD_NUM_POINTS="${POINT_CLOUD_NUM_POINTS:-16384}"
# Keep point-cloud workspace cropping enabled by default so X < 0 is rejected,
# then sample POINT_CLOUD_NUM_POINTS from the valid fused point cloud.
DISABLE_POINT_CLOUD_WORKSPACE_CROP="${DISABLE_POINT_CLOUD_WORKSPACE_CROP:-0}"
# Optional custom crop bounds used when DISABLE_POINT_CLOUD_WORKSPACE_CROP=0.
# Use "none" for an unbounded axis, e.g. x >= 0.02, y <= -0.02, no z crop:
#   POINT_CLOUD_WORKSPACE_MIN="0.02 none none"
#   POINT_CLOUD_WORKSPACE_MAX="none -0.02 none"
POINT_CLOUD_WORKSPACE_MIN="${POINT_CLOUD_WORKSPACE_MIN:-0 none none}"
POINT_CLOUD_WORKSPACE_MAX="${POINT_CLOUD_WORKSPACE_MAX:-none none none}"
POINT_CLOUD_EXCLUDE_BOX_MIN="${POINT_CLOUD_EXCLUDE_BOX_MIN:-}"
POINT_CLOUD_EXCLUDE_BOX_MAX="${POINT_CLOUD_EXCLUDE_BOX_MAX:-}"
# Render a couple of frames after reset/auto-reset so the first saved point
# cloud does not reuse stale camera buffers from the previous episode.
CAMERA_WARMUP_FRAMES_AFTER_RESET="${CAMERA_WARMUP_FRAMES_AFTER_RESET:-2}"
# Set to 1 to hide table render geometry while keeping collision behavior.
HIDE_TABLE_VISUALS="${HIDE_TABLE_VISUALS:-1}"

# Output / cleanup
# launch_collect_with_checkpoint.py writes under $KUAFU_ARTIFACT_ROOT/collection/<TASK>/.
# After the run, new *_worker## dirs are moved into POINTCLOUD_DIR and kept
# locally with their rendered MP4s.
COLLECT_OUT="${COLLECT_OUT:-}"                                # empty -> $KUAFU_ARTIFACT_ROOT/collection/<TASK>
POINTCLOUD_DIR="${POINTCLOUD_DIR:-}"                             # empty -> preview dir under $KUAFU_ARTIFACT_ROOT/pointclouds

# Visualization preview settings.
VISUALIZE_FPS="${VISUALIZE_FPS:-30}"
VISUALIZE_STRIDE="${VISUALIZE_STRIDE:-1}"
VISUALIZE_N_POINTS="${VISUALIZE_N_POINTS:-16384}"
VISUALIZE_WIDTH="${VISUALIZE_WIDTH:-640}"
VISUALIZE_HEIGHT="${VISUALIZE_HEIGHT:-640}"
VISUALIZE_AZIM="${VISUALIZE_AZIM:-45}"
VISUALIZE_ELEV="${VISUALIZE_ELEV:-25}"

# =========================== END EDIT ME =================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"

case ":${PATH}:" in
    *":${HOME}/.local/bin:"*) ;;
    *) export PATH="${HOME}/.local/bin:${PATH}" ;;
esac

[[ -z "${COLLECT_OUT}"    ]] && COLLECT_OUT="${KUAFU_ARTIFACT_ROOT:-${REPO_ROOT}/../KuafuArtifacts}/collection/${TASK}"
[[ -z "${POINTCLOUD_DIR}" ]] && POINTCLOUD_DIR="${KUAFU_ARTIFACT_ROOT:-${REPO_ROOT}/../KuafuArtifacts}/pointclouds/${TASK}-best-ema-preview"

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
        echo "ERROR: ${label} must be 0 or 1, got ${value}." >&2
        exit 1
    fi
}

require_crop_bounds() {
    local value="$1"
    local label="$2"
    [[ -z "${value}" ]] && return 0
    read -r -a _bounds_arr <<< "${value}"
    if (( ${#_bounds_arr[@]} != 3 )); then
        echo "ERROR: ${label} must contain exactly 3 values, got ${value}." >&2
        exit 1
    fi
    local bound
    for bound in "${_bounds_arr[@]}"; do
        if [[ "${bound}" == "none" || "${bound}" == "null" ]]; then
            continue
        fi
        if [[ ! "${bound}" =~ ^-?([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]]; then
            echo "ERROR: ${label} values must be floats or none, got ${bound} in ${value}." >&2
            exit 1
        fi
    done
}

require_file "${CHECKPOINT}" "CHECKPOINT"
require_file "${PRIOR_CHECKPOINT}" "PRIOR_CHECKPOINT"
require_file "${WEIGHTS_PARAMS_JSON}" "WEIGHTS_PARAMS_JSON"
require_positive_int "${DEMOS_TO_COLLECT}" "DEMOS_TO_COLLECT"
require_positive_int "${NUM_ENVS_PER_WORKER}" "NUM_ENVS_PER_WORKER"
require_positive_int "${STRIDE}" "STRIDE"
require_positive_int "${VISUALIZE_FPS}" "VISUALIZE_FPS"
require_positive_int "${VISUALIZE_STRIDE}" "VISUALIZE_STRIDE"
require_positive_int "${VISUALIZE_N_POINTS}" "VISUALIZE_N_POINTS"
require_positive_int "${VISUALIZE_WIDTH}" "VISUALIZE_WIDTH"
require_positive_int "${VISUALIZE_HEIGHT}" "VISUALIZE_HEIGHT"
if [[ -n "${POINT_CLOUD_NUM_POINTS}" ]]; then
    require_positive_int "${POINT_CLOUD_NUM_POINTS}" "POINT_CLOUD_NUM_POINTS"
fi
require_nonnegative_int "${CAMERA_WARMUP_FRAMES_AFTER_RESET}" "CAMERA_WARMUP_FRAMES_AFTER_RESET"
require_bool01 "${DISABLE_POINT_CLOUD_WORKSPACE_CROP}" "DISABLE_POINT_CLOUD_WORKSPACE_CROP"
require_bool01 "${HIDE_TABLE_VISUALS}" "HIDE_TABLE_VISUALS"
require_crop_bounds "${POINT_CLOUD_WORKSPACE_MIN}" "POINT_CLOUD_WORKSPACE_MIN"
require_crop_bounds "${POINT_CLOUD_WORKSPACE_MAX}" "POINT_CLOUD_WORKSPACE_MAX"
require_crop_bounds "${POINT_CLOUD_EXCLUDE_BOX_MIN}" "POINT_CLOUD_EXCLUDE_BOX_MIN"
require_crop_bounds "${POINT_CLOUD_EXCLUDE_BOX_MAX}" "POINT_CLOUD_EXCLUDE_BOX_MAX"

read -r -a _GPU_ARR <<< "${GPUS}"
NUM_WORKERS=${#_GPU_ARR[@]}
if (( NUM_WORKERS <= 0 )); then
    echo "ERROR: GPUS is empty." >&2
    exit 1
fi
if (( DEMOS_TO_COLLECT % NUM_WORKERS != 0 )); then
    echo "ERROR: DEMOS_TO_COLLECT=${DEMOS_TO_COLLECT} must divide evenly across ${NUM_WORKERS} worker(s)." >&2
    echo "       Edit GPUS or DEMOS_TO_COLLECT so the script collects exactly the requested count." >&2
    exit 1
fi
NUM_DEMOS_PER_WORKER=$(( DEMOS_TO_COLLECT / NUM_WORKERS ))

mkdir -p "${COLLECT_OUT}" "${POINTCLOUD_DIR}"

PRE_ENTRIES=()
STAGED_DIRS=()

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
    local pc_args=()
    if [[ -n "${POINT_CLOUD_NUM_POINTS}" ]]; then
        pc_args+=(--point_cloud_num_points "${POINT_CLOUD_NUM_POINTS}")
    fi
    if [[ "${DISABLE_POINT_CLOUD_WORKSPACE_CROP}" == "1" ]]; then
        pc_args+=(--disable_point_cloud_workspace_crop)
    else
        if [[ -n "${POINT_CLOUD_WORKSPACE_MIN}" ]]; then
            read -r -a _pc_crop_min <<< "${POINT_CLOUD_WORKSPACE_MIN}"
            pc_args+=(--point_cloud_workspace_min "${_pc_crop_min[@]}")
        fi
        if [[ -n "${POINT_CLOUD_WORKSPACE_MAX}" ]]; then
            read -r -a _pc_crop_max <<< "${POINT_CLOUD_WORKSPACE_MAX}"
            pc_args+=(--point_cloud_workspace_max "${_pc_crop_max[@]}")
        fi
    fi
    if [[ -n "${POINT_CLOUD_EXCLUDE_BOX_MIN}" ]]; then
        read -r -a _pc_exclude_min <<< "${POINT_CLOUD_EXCLUDE_BOX_MIN}"
        pc_args+=(--point_cloud_exclude_box_min "${_pc_exclude_min[@]}")
    fi
    if [[ -n "${POINT_CLOUD_EXCLUDE_BOX_MAX}" ]]; then
        read -r -a _pc_exclude_max <<< "${POINT_CLOUD_EXCLUDE_BOX_MAX}"
        pc_args+=(--point_cloud_exclude_box_max "${_pc_exclude_max[@]}")
    fi
    if [[ "${HIDE_TABLE_VISUALS}" == "1" ]]; then
        pc_args+=(--hide_table_visuals)
    fi
    pc_args+=(--camera_warmup_frames_after_reset "${CAMERA_WARMUP_FRAMES_AFTER_RESET}")

    # shellcheck disable=SC2086
    "${PYTHON_BIN}" src/kuafu/data/collection/launch_collect_with_checkpoint.py \
        --python "${PYTHON_BIN}" \
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
        "${pc_args[@]}"
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
    STAGED_DIRS=()
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
        find "$p" -depth -name 'episode_tmp_env*' \( -type f -o -type d \) -exec rm -rf {} + 2>/dev/null || true
    done

    mkdir -p "${POINTCLOUD_DIR}"
    echo "[stage] moving ${#to_move[@]} dir(s) -> ${POINTCLOUD_DIR}/"
    for p in "${to_move[@]}"; do
        name="$(basename "$p")"
        mv -f -- "$p" "${POINTCLOUD_DIR}/"
        STAGED_DIRS+=( "${POINTCLOUD_DIR}/${name}" )
    done
}

visualize_pointclouds() {
    if (( ${#STAGED_DIRS[@]} == 0 )); then
        echo "[visualize] no staged dirs, nothing to render."
        return 0
    fi

    local h5_paths=()
    mapfile -t h5_paths < <(
        find "${STAGED_DIRS[@]}" \
            -type f \
            -name 'episode_*.h5' \
            ! -name '*tmp*' \
            | sort
    )

    if (( ${#h5_paths[@]} == 0 )); then
        echo "[visualize] no episode_*.h5 files found under staged dirs."
        return 0
    fi

    echo "[visualize] rendering ${#h5_paths[@]} episode(s) with src/kuafu/data/visualize_pointcloud.py"
    local h5_path
    for h5_path in "${h5_paths[@]}"; do
        echo "[visualize] ${h5_path}"
        "${PYTHON_BIN}" src/kuafu/data/visualize_pointcloud.py "${h5_path}" \
            --fps "${VISUALIZE_FPS}" \
            --stride "${VISUALIZE_STRIDE}" \
            --n_points "${VISUALIZE_N_POINTS}" \
            --width "${VISUALIZE_WIDTH}" \
            --height "${VISUALIZE_HEIGHT}" \
            --azim "${VISUALIZE_AZIM}" \
            --elev "${VISUALIZE_ELEV}"
    done
}

echo "=========================================="
echo " collect_and_visualize_stack2_best_ema"
echo "  task                 : ${TASK}"
echo "  checkpoint           : ${CHECKPOINT}"
echo "  prior checkpoint     : ${PRIOR_CHECKPOINT}"
echo "  weights/params       : ${WEIGHTS_PARAMS_JSON}"
echo "  gpus                 : ${GPUS}"
echo "  workers              : ${NUM_WORKERS}"
echo "  envs/worker          : ${NUM_ENVS_PER_WORKER}"
echo "  demos total          : ${DEMOS_TO_COLLECT} (${NUM_DEMOS_PER_WORKER}/worker)"
echo "  stride               : ${STRIDE}"
echo "  collect output dir   : ${COLLECT_OUT}"
echo "  pointcloud dir       : ${POINTCLOUD_DIR}"
echo "  upload               : disabled"
if [[ -n "${POINT_CLOUD_NUM_POINTS}" ]]; then
    echo "  point_cloud cap/frame: ${POINT_CLOUD_NUM_POINTS}"
else
    echo "  point_cloud cap/frame: none"
fi
if [[ "${DISABLE_POINT_CLOUD_WORKSPACE_CROP}" == "1" ]]; then
    echo "  point_cloud xyz crop : disabled"
else
    if [[ -n "${POINT_CLOUD_WORKSPACE_MIN}" || -n "${POINT_CLOUD_WORKSPACE_MAX}" ]]; then
        echo "  point_cloud xyz crop : custom"
        echo "  point_cloud crop min: ${POINT_CLOUD_WORKSPACE_MIN:-default}"
        echo "  point_cloud crop max: ${POINT_CLOUD_WORKSPACE_MAX:-default}"
    else
        echo "  point_cloud xyz crop : workspace limits"
    fi
fi
if [[ -n "${POINT_CLOUD_EXCLUDE_BOX_MIN}" || -n "${POINT_CLOUD_EXCLUDE_BOX_MAX}" ]]; then
    echo "  point_cloud exclude min: ${POINT_CLOUD_EXCLUDE_BOX_MIN:-none}"
    echo "  point_cloud exclude max: ${POINT_CLOUD_EXCLUDE_BOX_MAX:-none}"
fi
if [[ "${HIDE_TABLE_VISUALS}" == "1" ]]; then
    echo "  table visuals       : hidden"
else
    echo "  table visuals       : visible"
fi
echo "  camera reset warmup  : ${CAMERA_WARMUP_FRAMES_AFTER_RESET} render frame(s)"
echo "  visualize            : fps=${VISUALIZE_FPS}, stride=${VISUALIZE_STRIDE}, n_points=${VISUALIZE_N_POINTS}"
echo "=========================================="

run_collect
stage_pointclouds
visualize_pointclouds

echo
echo "[done] staged pointclouds and MP4s are under ${POINTCLOUD_DIR}"
