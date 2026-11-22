#!/usr/bin/env bash
# Recreate the final Right-Can show screenshots.
# Usage:
#   ./resources/media/right_can_media/recreate_right_can_media.sh clean
#   ./resources/media/right_can_media/recreate_right_can_media.sh debug
#   ./resources/media/right_can_media/recreate_right_can_media.sh both

set -euo pipefail

MODE="${1:-both}"
ROBOT_ID="${ROBOT_ID:-fr3_finray}"
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
# CameraSpec("front_close_can_tight", (0.08, -0.62, 0.46), (0.00, 0.04, 0.05), 30.0)
# The final show artifact is screenshots-only by default.
COMMON_ENV=(
    GPU="${GPU:-0}"
    SEED="${SEED:-20260516}"
    CAMERA_BANK="${CAMERA_BANK:-best}"
    MAX_STEPS="${MAX_STEPS:-450}"
    WIDTH="${WIDTH:-1280}"
    HEIGHT="${HEIGHT:-720}"
    FPS="${FPS:-30}"
    VIDEO_EVERY="${VIDEO_EVERY:-1}"
    VIEWPORT_SETTLE_FRAMES="${VIEWPORT_SETTLE_FRAMES:-0}"
    STOCHASTIC_POLICY="${STOCHASTIC_POLICY:-0}"
    NO_VIDEO="${NO_VIDEO:-1}"
    ROBOT_ID="${ROBOT_ID}"
    SCREENSHOT_STEPS="${SCREENSHOT_STEPS:-0,100,200,300,340,360,380,390,391,394,395,396}"
)
if [[ -n "${WEIGHTS_PARAMS_JSON:-}" ]]; then
    COMMON_ENV+=(WEIGHTS_PARAMS_JSON="${WEIGHTS_PARAMS_JSON}")
elif [[ "${ROBOT_ID}" == "fr3_finray" ]]; then
    COMMON_ENV+=(WEIGHTS_PARAMS_JSON="resources/experiment_definitions/fr3_finray/residual_rl/parameter_search/fr3_finray_learned_can/best.json")
fi

run_variant() {
    local debug_frames="$1"
    env "${COMMON_ENV[@]}" DEBUG_FRAMES="${debug_frames}"         ./src/kuafu/data/collection/capture_right_can_best_media.sh
}

if [[ "${MODE}" == "clean" || "${MODE}" == "both" ]]; then
    run_variant 0
fi

if [[ "${MODE}" == "debug" || "${MODE}" == "both" ]]; then
    run_variant 1
fi
