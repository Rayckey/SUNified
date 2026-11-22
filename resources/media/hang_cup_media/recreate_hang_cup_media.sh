#!/usr/bin/env bash
# Recreate Hang-Cup show screenshots.
# Usage:
#   ./resources/media/hang_cup_media/recreate_hang_cup_media.sh clean
#   ./resources/media/hang_cup_media/recreate_hang_cup_media.sh debug
#   ./resources/media/hang_cup_media/recreate_hang_cup_media.sh both

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

# Initial show recipe values. The capture script has this view promoted to CAMERA_BANK=best:
# CameraSpec("front_close_hang_cup_success", (0.12, -0.78, 0.56), (0.00, 0.06, 0.05), 24.0)
# Hang-Cup is screenshots-first by default; set NO_VIDEO=0 to also record MP4s.
COMMON_ENV=(
    GPU="${GPU:-0}"
    SEED="${SEED:-20260516}"
    CAMERA_BANK="${CAMERA_BANK:-best}"
    CAPTURE_BACKEND="${CAPTURE_BACKEND:-viewport}"
    MAX_STEPS="${MAX_STEPS:-1100}"
    WIDTH="${WIDTH:-1280}"
    HEIGHT="${HEIGHT:-720}"
    FPS="${FPS:-30}"
    VIDEO_EVERY="${VIDEO_EVERY:-1}"
    VIEWPORT_SETTLE_FRAMES="${VIEWPORT_SETTLE_FRAMES:-2}"
    STOCHASTIC_POLICY="${STOCHASTIC_POLICY:-0}"
    NO_VIDEO="${NO_VIDEO:-1}"
    SCREENSHOT_STEPS="${SCREENSHOT_STEPS:-final}"
)

run_variant() {
    local debug_frames="$1"
    env "${COMMON_ENV[@]}" DEBUG_FRAMES="${debug_frames}" \
        ./src/kuafu/data/collection/capture_hang_cup_best_ema_media.sh
}

if [[ "${MODE}" == "clean" || "${MODE}" == "both" ]]; then
    run_variant 0
fi

if [[ "${MODE}" == "debug" || "${MODE}" == "both" ]]; then
    run_variant 1
fi
