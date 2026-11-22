#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Collect a small pointcloud preview with the best Drawer EMA checkpoint, then
# render the collected HDF5 episodes to MP4 locally. No upload or staged-data
# deletion.
#
# Best checkpoint source:
#   Episode/Kuafu/success_rate_ema = 0.922387064
#   run: Residuals/FR3-FinRay-Table-Drawer-ff/2026-04-29_01-07-34
#   worker: 2026-04-29_01-09-42_Run-1, checkpoint epoch 1000
# ---------------------------------------------------------------------------

set -euo pipefail

# =========================== EDIT ME =====================================
# Forwarded to launch_collect_with_checkpoint.py
GPUS="0 1"
DEMOS_TO_COLLECT=10
NUM_ENVS_PER_WORKER=16                      # keep 1 for exact small preview counts
TASK="${TASK:-FR3-FinRay-Table-Drawer-ff}"
CHECKPOINT="${CHECKPOINT:-${KUAFU_ARTIFACT_ROOT:-../KuafuArtifacts}/Residuals/FR3-FinRay-Table-Drawer-ff/2026-04-29_01-07-34/rl_runs/2026-04-29_01-09-42_Run-1/nn/last_Open-FR3FinRay-Drawer_ep_1000_rew_6391.739.pth}"
PRIOR_CHECKPOINT="${PRIOR_CHECKPOINT:-${KUAFU_ARTIFACT_ROOT:-../KuafuArtifacts}/Priors/herm/drawer_damped_noise/drawer_chunking_15_ff_stride3_damped_noise/best.pth}"
PRIOR_MODE="${PRIOR_MODE:-residual_rl}"
WEIGHTS_PARAMS_JSON="${WEIGHTS_PARAMS_JSON:-${KUAFU_ARTIFACT_ROOT:-../KuafuArtifacts}/Residuals/FR3-FinRay-Table-Drawer-ff/2026-05-05_09-01-52/best.json}"
STRIDE=1


# Point cloud HDF5: empty = use collect_with_checkpoint.py default.
# Set a positive integer for uniform random subsampling per frame.
POINT_CLOUD_NUM_POINTS="16384"
# Disable all saver.py WORKSPACE_LIMITS point-cloud cropping,
# then sample POINT_CLOUD_NUM_POINTS from the valid fused point cloud.
DISABLE_POINT_CLOUD_WORKSPACE_CROP=1
# Render a couple of frames after reset/auto-reset so the first saved point
# cloud does not reuse stale camera buffers from the previous episode.
CAMERA_WARMUP_FRAMES_AFTER_RESET=2
# Set to 1 to hide table render geometry while keeping collision behavior.
HIDE_TABLE_VISUALS=1

# Upload / cleanup
# launch_collect_with_checkpoint.py writes under $KUAFU_ARTIFACT_ROOT/collection/<TASK>/.
# After each chunk, new *_worker## dirs are moved into POINTCLOUD_DIR, synced,
# and removed locally only after a successful sync.
COLLECT_OUT=""                                # empty -> $KUAFU_ARTIFACT_ROOT/collection/<TASK>
POINTCLOUD_DIR=""                             # empty -> <repo>/

# Visualization preview settings.
VISUALIZE_FPS=30
VISUALIZE_STRIDE=1
VISUALIZE_N_POINTS=16384
VISUALIZE_WIDTH=640
VISUALIZE_HEIGHT=640
VISUALIZE_AZIM=45
VISUALIZE_ELEV=25

# =========================== END EDIT ME =================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"

case ":${PATH}:" in
    *":${HOME}/.local/bin:"*) ;;
    *) export PATH="${HOME}/.local/bin:${PATH}" ;;
esac

[[ -z "${COLLECT_OUT}"    ]] && COLLECT_OUT="${KUAFU_ARTIFACT_ROOT:-${REPO_ROOT}/../KuafuArtifacts}/collection/${TASK}"
[[ -z "${POINTCLOUD_DIR}" ]] && POINTCLOUD_DIR="${KUAFU_ARTIFACT_ROOT:-${REPO_ROOT}/../KuafuArtifacts}/pointclouds/${TASK}-best-ema-preview-4"

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
    fi
    if [[ "${HIDE_TABLE_VISUALS}" == "1" ]]; then
        pc_args+=(--hide_table_visuals)
    fi
    pc_args+=(--camera_warmup_frames_after_reset "${CAMERA_WARMUP_FRAMES_AFTER_RESET}")

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
        python src/kuafu/data/visualize_pointcloud.py "${h5_path}" \
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
echo " collect_and_visualize_drawer_best_ema_4"
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
    echo "  point_cloud xyz crop : workspace limits"
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
