#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Capture review screenshots and MP4 videos for the EStop-Open residual
# policy. This does not collect or upload point clouds.
# ---------------------------------------------------------------------------

set -euo pipefail

# =========================== EDIT ME =====================================
GPU="${GPU:-0}"
# Override PYTHON_BIN if your IsaacLab environment lives somewhere else.

TASK="${TASK:-FR3-FinRay-Table-EStop-Open-ff}"
CHECKPOINT="${CHECKPOINT:-${KUAFU_ARTIFACT_ROOT:-../KuafuArtifacts}/Residuals/FR3-FinRay-Table-EStop-Open-ff/2026-05-25_19-11-40/rl_runs/2026-05-25_19-12-48_Run-0/nn/last_Open-FR3FinRay-EStop_ep_1000_rew_1675.231.pth}"
PRIOR_CHECKPOINT="${PRIOR_CHECKPOINT:-${KUAFU_ARTIFACT_ROOT:-../KuafuArtifacts}/Priors/thor/estop_open/estop_open_chunking_15_ff/best.pth}"
PRIOR_MODE="${PRIOR_MODE:-residual_rl}"
WEIGHTS_PARAMS_JSON="${WEIGHTS_PARAMS_JSON:-${KUAFU_ARTIFACT_ROOT:-../KuafuArtifacts}/Residuals/FR3-FinRay-Table-EStop-Open-ff/2026-05-25_19-10-41/best.json}"

# Fixed seed keeps EStop randomization stable between camera tests.
SEED="${SEED:-20260516}"

# Camera bank choices: best, front_estop_close, side_estop, sweep_a, sweep_b, hero.
CAMERA_BANK="${CAMERA_BANK:-best}"
# Capture backend choices: viewport for nice RGB review media, sensor for debugging env cameras.
CAPTURE_BACKEND="${CAPTURE_BACKEND:-viewport}"
# Set DEBUG_FRAMES=1 to render all task frame-transformer debug markers.
DEBUG_FRAMES="${DEBUG_FRAMES:-0}"
# Keep viewport captures deterministic and quick; increase only if the viewport lags on your machine.
VIEWPORT_SETTLE_FRAMES="${VIEWPORT_SETTLE_FRAMES:-2}"

MAX_STEPS="${MAX_STEPS:-180}"
FPS="${FPS:-30}"
VIDEO_EVERY="${VIDEO_EVERY:-1}"
# EStop-Open review is screenshots-first by default; enable video after a still run is verified.
NO_VIDEO="${NO_VIDEO:-1}"
NO_SCREENSHOTS="${NO_SCREENSHOTS:-0}"
SCREENSHOT_STEPS="${SCREENSHOT_STEPS:-137}"
WIDTH="${WIDTH:-1280}"
HEIGHT="${HEIGHT:-720}"
OUTPUT_DIR="${OUTPUT_DIR:-${KUAFU_ARTIFACT_ROOT:-../KuafuArtifacts}/media/estop_open_best_ema}"
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
if [[ -n "${WEIGHTS_PARAMS_JSON}" ]]; then
    require_file "${WEIGHTS_PARAMS_JSON}" "WEIGHTS_PARAMS_JSON"
fi

echo "=========================================="
echo " capture_estop_open_best_ema_media"
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
if [[ -n "${WEIGHTS_PARAMS_JSON}" ]]; then
    EXTRA_ARGS+=(--weights_params_json "${WEIGHTS_PARAMS_JSON}")
fi

CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" src/kuafu/data/collection/capture_estop_open_best_ema_media.py \
    --headless \
    --device cuda:0 \
    --task "${TASK}" \
    --checkpoint "${CHECKPOINT}" \
    --prior_checkpoint "${PRIOR_CHECKPOINT}" \
    --prior_mode "${PRIOR_MODE}" \
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
