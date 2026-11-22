#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Capture review screenshots and MP4 videos for the Pen-Cup best true-success
# policy. This does not collect or upload point clouds.
# ---------------------------------------------------------------------------

set -euo pipefail

# =========================== EDIT ME =====================================
GPU="${GPU:-0}"
# Override PYTHON_BIN if your IsaacLab environment lives somewhere else.

TASK="${TASK:-FR3-FinRay-Table-Pen-Cup-ff}"
CHECKPOINT="${CHECKPOINT:-${KUAFU_ARTIFACT_ROOT:-../KuafuArtifacts}/Residuals/FR3-FinRay-Table-Pen-Cup-ff/2026-04-29_20-41-13/rl_runs/2026-04-29_20-43-38_Run-3/nn/last_pen-cup_ep_1500_rew__20532.975__np126.pth}"
PRIOR_CHECKPOINT="${PRIOR_CHECKPOINT:-${KUAFU_ARTIFACT_ROOT:-../KuafuArtifacts}/Priors/aws/pen_cup_noise_state_random/pen_cup_chunking_15_ff_damped_noise_state/best.pth}"
PRIOR_MODE="${PRIOR_MODE:-residual_rl}"
WEIGHTS_PARAMS_JSON="${WEIGHTS_PARAMS_JSON:-${KUAFU_ARTIFACT_ROOT:-../KuafuArtifacts}/Residuals/FR3-FinRay-Table-Pen-Cup-ff/2026-04-29_20-40-07/best.json}"

# Fixed seed keeps pen/holder randomization stable between camera tests.
SEED="${SEED:-5}"

# Camera bank choices: best, front_pen_cup_close, front_pen_cup, side_pen_cup, sweep_a, sweep_b, hero.
CAMERA_BANK="${CAMERA_BANK:-best}"
# Capture backend choices: viewport for nice RGB review media, sensor for debugging env cameras.
CAPTURE_BACKEND="${CAPTURE_BACKEND:-viewport}"
# Set DEBUG_FRAMES=1 to render all task frame-transformer debug markers.
DEBUG_FRAMES="${DEBUG_FRAMES:-0}"
# Keep viewport captures deterministic and quick; increase only if the viewport lags on your machine.
VIEWPORT_SETTLE_FRAMES="${VIEWPORT_SETTLE_FRAMES:-0}"

MAX_STEPS="${MAX_STEPS:-350}"
FPS="${FPS:-30}"
VIDEO_EVERY="${VIDEO_EVERY:-1}"
# Pen-Cup review is screenshots-first by default; enable video after a still run is verified.
NO_VIDEO="${NO_VIDEO:-1}"
NO_SCREENSHOTS="${NO_SCREENSHOTS:-0}"
SCREENSHOT_STEPS="${SCREENSHOT_STEPS:-0,80,120,160,200,220,235,240,245,246,247}"
WIDTH="${WIDTH:-1280}"
HEIGHT="${HEIGHT:-720}"
OUTPUT_DIR="${OUTPUT_DIR:-${KUAFU_ARTIFACT_ROOT:-../KuafuArtifacts}/media/pen_cup_best_true_success}"
# Deterministic playback with seed 5 reached true success at policy step 247 during view testing.
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
echo " capture_pen_cup_best_true_success_media"
echo "  task            : ${TASK}"
echo "  checkpoint      : ${CHECKPOINT}"
echo "  prior checkpoint: ${PRIOR_CHECKPOINT}"
echo "  weights/params  : ${WEIGHTS_PARAMS_JSON}"
echo "  gpu             : ${GPU}"
echo "  seed            : ${SEED}"
echo "  camera bank     : ${CAMERA_BANK}"
echo "  backend         : ${CAPTURE_BACKEND}"
echo "  debug frames    : ${DEBUG_FRAMES}"
echo "  settle frames   : ${VIEWPORT_SETTLE_FRAMES}"
echo "  max steps       : ${MAX_STEPS}"
echo "  no video        : ${NO_VIDEO}"
echo "  screenshot steps: ${SCREENSHOT_STEPS}"
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
if [[ "${NO_VIDEO}" == "1" ]]; then
    EXTRA_ARGS+=(--no_video)
fi
if [[ "${NO_SCREENSHOTS}" == "1" ]]; then
    EXTRA_ARGS+=(--no_screenshots)
fi
if [[ -n "${SCREENSHOT_STEPS}" ]]; then
    EXTRA_ARGS+=(--screenshot_steps "${SCREENSHOT_STEPS}")
fi

CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" src/kuafu/data/collection/capture_pen_cup_best_true_success_media.py \
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
