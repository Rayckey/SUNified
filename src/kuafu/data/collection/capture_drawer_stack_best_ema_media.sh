#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Capture review screenshots and MP4 videos for the Drawer-Stack best EMA
# policy. This does not collect or upload point clouds.
# ---------------------------------------------------------------------------

set -euo pipefail

# =========================== EDIT ME =====================================
GPU="${GPU:-0}"
# Override PYTHON_BIN if your IsaacLab environment lives somewhere else.

TASK="${TASK:-FR3-FinRay-Table-Drawer-Stack-fff}"
CHECKPOINT="${CHECKPOINT:-${KUAFU_ARTIFACT_ROOT:-../KuafuArtifacts}/Residuals/FR3-FinRay-Table-Drawer-Stack-fff/2026-05-15_00-01-56/rl_runs/2026-05-15_00-04-12_Run-2/nn/last_drawer-stack_ep_1550_rew_3839.0168.pth}"
PRIOR_CHECKPOINT="${PRIOR_CHECKPOINT:-${KUAFU_ARTIFACT_ROOT:-../KuafuArtifacts}/Priors/herm/drawer_stack/drawer_stack_chunking_15_fff_stride3/best.pth}"
PRIOR_MODE="${PRIOR_MODE:-residual_rl}"
WEIGHTS_PARAMS_JSON="${WEIGHTS_PARAMS_JSON:-${KUAFU_ARTIFACT_ROOT:-../KuafuArtifacts}/Residuals/FR3-FinRay-Table-Drawer-Stack-fff/2026-05-14_23-58-02/best.json}"

# Fixed seed keeps cabinet/strawberry randomization stable between camera tests.
SEED="${SEED:-20260516}"

# Camera bank choices: best, between_pillars_close, between_pillars_refined, between_pillars, drawer_close, sweep_a, sweep_b, hero.
CAMERA_BANK="${CAMERA_BANK:-best}"
# Capture backend choices: viewport for nice RGB review media, sensor for debugging env cameras.
CAPTURE_BACKEND="${CAPTURE_BACKEND:-viewport}"
# Set DEBUG_FRAMES=1 to render task frame-transformer debug markers.
DEBUG_FRAMES="${DEBUG_FRAMES:-0}"
# Optional comma-separated subset, for example: panda_fingertip,top_drawer_handle_frame.
DEBUG_FRAME_SENSORS="${DEBUG_FRAME_SENSORS:-}"
# Keep viewport captures deterministic and quick; increase only if the viewport lags on your machine.
VIEWPORT_SETTLE_FRAMES="${VIEWPORT_SETTLE_FRAMES:-0}"
# Save an extra final still with a tighter camera focused only on the drawer task.
DRAWER_TASK_SNAPSHOT="${DRAWER_TASK_SNAPSHOT:-1}"
DRAWER_TASK_SNAPSHOT_NAME="${DRAWER_TASK_SNAPSHOT_NAME:-}"

MAX_STEPS="${MAX_STEPS:-900}"
FPS="${FPS:-30}"
VIDEO_EVERY="${VIDEO_EVERY:-1}"
WIDTH="${WIDTH:-1280}"
HEIGHT="${HEIGHT:-720}"
OUTPUT_DIR="${OUTPUT_DIR:-${KUAFU_ARTIFACT_ROOT:-../KuafuArtifacts}/media/drawer_stack_best_ema}"
# Deterministic playback with seed 20260516 reached success during view testing.
STOCHASTIC_POLICY="${STOCHASTIC_POLICY:-0}"
PYTHON_BIN="${PYTHON_BIN:-${KUAFU_PYTHON:-python}}"
# =========================== END EDIT ME =================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"
if [[ -z "${TERM:-}" || "${TERM}" == "dumb" ]]; then
    export TERM=xterm
fi

require_file() {
    local path="$1"
    local label="$2"
    if [[ ! -f "${path}" ]]; then
        echo "ERROR: ${label} does not exist: ${path}" >&2
        exit 1
    fi
}

require_file "${CHECKPOINT}" "CHECKPOINT"
require_file "${PRIOR_CHECKPOINT}" "PRIOR_CHECKPOINT"
require_file "${WEIGHTS_PARAMS_JSON}" "WEIGHTS_PARAMS_JSON"

echo "=========================================="
echo " capture_drawer_stack_best_ema_media"
echo "  task            : ${TASK}"
echo "  checkpoint      : ${CHECKPOINT}"
echo "  prior checkpoint: ${PRIOR_CHECKPOINT}"
echo "  weights/params  : ${WEIGHTS_PARAMS_JSON}"
echo "  gpu             : ${GPU}"
echo "  seed            : ${SEED}"
echo "  camera bank     : ${CAMERA_BANK}"
echo "  backend         : ${CAPTURE_BACKEND}"
echo "  debug frames    : ${DEBUG_FRAMES}"
echo "  frame sensors   : ${DEBUG_FRAME_SENSORS:-all}"
echo "  settle frames   : ${VIEWPORT_SETTLE_FRAMES}"
echo "  drawer snapshot : ${DRAWER_TASK_SNAPSHOT}"
if [[ -n "${DRAWER_TASK_SNAPSHOT_NAME}" ]]; then
    echo "  drawer snap name: ${DRAWER_TASK_SNAPSHOT_NAME}"
fi
echo "  max steps       : ${MAX_STEPS}"
echo "  resolution/fps  : ${WIDTH}x${HEIGHT} @ ${FPS}"
echo "  stochastic      : ${STOCHASTIC_POLICY}"
echo "  output dir      : ${OUTPUT_DIR}"
echo "=========================================="

EXTRA_ARGS=()
if [[ "${STOCHASTIC_POLICY}" == "1" ]]; then
    EXTRA_ARGS+=(--stochastic_policy)
fi
if [[ "${DEBUG_FRAMES}" == "1" ]]; then
    EXTRA_ARGS+=(--debug_frames)
fi
if [[ -n "${DEBUG_FRAME_SENSORS}" ]]; then
    EXTRA_ARGS+=(--debug_frame_sensors "${DEBUG_FRAME_SENSORS}")
fi
if [[ "${DRAWER_TASK_SNAPSHOT}" == "0" ]]; then
    EXTRA_ARGS+=(--no-drawer-task-snapshot)
fi
if [[ -n "${DRAWER_TASK_SNAPSHOT_NAME}" ]]; then
    EXTRA_ARGS+=(--drawer-task-snapshot-name "${DRAWER_TASK_SNAPSHOT_NAME}")
fi

CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" src/kuafu/data/collection/capture_drawer_stack_best_ema_media.py \
    --headless \
    --device cuda:0 \
    --task "${TASK}" \
    --checkpoint "${CHECKPOINT}" \
    --prior_checkpoint "${PRIOR_CHECKPOINT}" \
    --prior_mode "${PRIOR_MODE}" \
    --weights_params_json "${WEIGHTS_PARAMS_JSON}" \
    --seed "${SEED}" \
    --camera_bank "${CAMERA_BANK}" \
    --capture_backend "${CAPTURE_BACKEND}" \
    --max_steps "${MAX_STEPS}" \
    --fps "${FPS}" \
    --video_every "${VIDEO_EVERY}" \
    --width "${WIDTH}" \
    --height "${HEIGHT}" \
    --output_dir "${OUTPUT_DIR}" \
    --viewport_settle_frames "${VIEWPORT_SETTLE_FRAMES}" \
    "${EXTRA_ARGS[@]}"
