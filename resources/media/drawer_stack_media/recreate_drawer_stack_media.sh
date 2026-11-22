#!/usr/bin/env bash
# Recreate the final drawer-stack show videos.
# Usage:
#   ./resources/media/drawer_stack_media/recreate_drawer_stack_media.sh clean
#   ./resources/media/drawer_stack_media/recreate_drawer_stack_media.sh debug
#   ./resources/media/drawer_stack_media/recreate_drawer_stack_media.sh both

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

# Exact final recipe values. The capture script currently has this view promoted to CAMERA_BANK=best:
# CameraSpec("between_pillars_close_action", (0.00, -0.62, 0.56), (-0.10, 0.22, -0.03), 19.0)
COMMON_ENV=(
    GPU="${GPU:-0}"
    SEED="${SEED:-20260516}"
    CAMERA_BANK="${CAMERA_BANK:-best}"
    MAX_STEPS="${MAX_STEPS:-700}"
    WIDTH="${WIDTH:-1280}"
    HEIGHT="${HEIGHT:-720}"
    FPS="${FPS:-30}"
    VIDEO_EVERY="${VIDEO_EVERY:-1}"
    VIEWPORT_SETTLE_FRAMES="${VIEWPORT_SETTLE_FRAMES:-0}"
    STOCHASTIC_POLICY="${STOCHASTIC_POLICY:-0}"
)

run_variant() {
    local debug_frames="$1"
    env "${COMMON_ENV[@]}" DEBUG_FRAMES="${debug_frames}"         ./src/kuafu/data/collection/capture_drawer_stack_best_ema_media.sh
}

if [[ "${MODE}" == "clean" || "${MODE}" == "both" ]]; then
    run_variant 0
fi

if [[ "${MODE}" == "debug" || "${MODE}" == "both" ]]; then
    run_variant 1
fi
