#!/usr/bin/env bash
# Recreate the final EStop-Open show screenshots.
# Usage:
#   ./resources/media/estop_open_media/recreate_estop_open_media.sh clean
#   ./resources/media/estop_open_media/recreate_estop_open_media.sh debug
#   ./resources/media/estop_open_media/recreate_estop_open_media.sh both

set -euo pipefail

MODE="${1:-both}"
case "${MODE}" in
    clean|debug|both) ;;
    *)
        echo "Usage: $0 [clean|debug|both]" >&2
        exit 2
        ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

# Exact final recipe values. The capture script has this view promoted to CAMERA_BANK=best:
# CameraSpec("front_close_estop_open_action", (0.10, -0.72, 0.50), (0.00, 0.05, 0.05), 28.0)
# This sensitive rollout uses one scheduled screenshot at step 137 and two viewport settle frames.
COMMON_ENV=(
    GPU="${GPU:-0}"
    SEED="${SEED:-20260516}"
    CAMERA_BANK="${CAMERA_BANK:-best}"
    CAPTURE_BACKEND="${CAPTURE_BACKEND:-viewport}"
    MAX_STEPS="${MAX_STEPS:-180}"
    WIDTH="${WIDTH:-1280}"
    HEIGHT="${HEIGHT:-720}"
    FPS="${FPS:-30}"
    VIDEO_EVERY="${VIDEO_EVERY:-1}"
    VIEWPORT_SETTLE_FRAMES="${VIEWPORT_SETTLE_FRAMES:-2}"
    STOCHASTIC_POLICY="${STOCHASTIC_POLICY:-0}"
    NO_VIDEO="${NO_VIDEO:-1}"
    WEIGHTS_PARAMS_JSON="${WEIGHTS_PARAMS_JSON:-Residuals/FR3-FinRay-Table-EStop-Open-ff/2026-04-30_12-01-28/best.json}"
    SCREENSHOT_STEPS="${SCREENSHOT_STEPS:-137}"
)

run_variant() {
    local debug_frames="$1"
    env "${COMMON_ENV[@]}" DEBUG_FRAMES="${debug_frames}" \
        ./src/kuafu/data/collection/capture_estop_open_best_ema_media.sh
}

if [[ "${MODE}" == "clean" || "${MODE}" == "both" ]]; then
    run_variant 0
fi

if [[ "${MODE}" == "debug" || "${MODE}" == "both" ]]; then
    run_variant 1
fi
