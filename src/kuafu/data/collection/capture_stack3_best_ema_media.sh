#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Capture review screenshots and MP4 videos for the Stack-3 best EMA
# policy. This does not collect or upload point clouds.
# ---------------------------------------------------------------------------

set -euo pipefail

# =========================== EDIT ME =====================================
GPU="${GPU:-0}"
# Override PYTHON_BIN if your IsaacLab environment lives somewhere else.

TASK="${TASK:-FR3-FinRay-Table-Stack-3-fff}"
CHECKPOINT="${CHECKPOINT:-${KUAFU_ARTIFACT_ROOT:-../KuafuArtifacts}/Residuals/FR3-FinRay-Table-Stack-3-fff/2026-05-18_19-23-40/rl_runs/2026-05-18_19-24-40_Run-0/nn/last_stack-3_ep_2750_rew_106216.23.pth}"
PRIOR_CHECKPOINT="${PRIOR_CHECKPOINT:-${KUAFU_ARTIFACT_ROOT:-../KuafuArtifacts}/Priors/ares/stack3/stack3_chunking_15_fff_stride3/best.pth}"
PRIOR_MODE="${PRIOR_MODE:-residual_rl}"
WEIGHTS_PARAMS_JSON="${WEIGHTS_PARAMS_JSON:-${KUAFU_ARTIFACT_ROOT:-../KuafuArtifacts}/Residuals/FR3-FinRay-Table-Stack-3-fff/2026-05-18_19-18-23/best.json}"

# Fixed seed for the known-good 8192-env probe. Env 2524 reached true success at policy step 237.
SEED="${SEED:-20260516}"
# Saved by running the Python script with --probe_success_only --num_envs 8192 --probe_save_env_id 2524.
REPLAY_ACTIONS_FILE="${REPLAY_ACTIONS_FILE-${KUAFU_ARTIFACT_ROOT:-../KuafuArtifacts}/media/stack3_best_ema_probe/probe_seed20260516_env2524_trajectory.pt}"
RENDER_REPLAY_STATES="${RENDER_REPLAY_STATES:-1}"

# Camera bank choices: best, front_stack3_close, front_stack3, side_stack3, sweep_a, sweep_b, hero.
CAMERA_BANK="${CAMERA_BANK:-front_stack3_close}"
# Capture backend choices: viewport for nice RGB review media, sensor for debugging env cameras.
CAPTURE_BACKEND="${CAPTURE_BACKEND:-viewport}"
# Set DEBUG_FRAMES=1 to render all task frame-transformer debug markers.
DEBUG_FRAMES="${DEBUG_FRAMES:-0}"
# Keep viewport captures deterministic and quick; increase only if the viewport lags on your machine.
VIEWPORT_SETTLE_FRAMES="${VIEWPORT_SETTLE_FRAMES:-0}"

MAX_STEPS="${MAX_STEPS:-238}"
MAX_EPISODES="${MAX_EPISODES:-1}"
FPS="${FPS:-30}"
VIDEO_EVERY="${VIDEO_EVERY:-1}"
# Default now writes MP4s from the saved successful replay.
NO_VIDEO="${NO_VIDEO:-0}"
NO_SCREENSHOTS="${NO_SCREENSHOTS:-0}"
SCREENSHOT_STEPS="${SCREENSHOT_STEPS:-0,50,100,150,200,237}"
WIDTH="${WIDTH:-1280}"
HEIGHT="${HEIGHT:-720}"
OUTPUT_DIR="${OUTPUT_DIR:-${KUAFU_ARTIFACT_ROOT:-../KuafuArtifacts}/media/stack3_best_ema_full_replay}"
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
echo " capture_stack3_best_ema_media"
echo "  task            : ${TASK}"
echo "  checkpoint      : ${CHECKPOINT}"
echo "  prior checkpoint: ${PRIOR_CHECKPOINT}"
echo "  weights/params  : ${WEIGHTS_PARAMS_JSON}"
echo "  gpu             : ${GPU}"
echo "  seed            : ${SEED}"
echo "  camera bank     : ${CAMERA_BANK}"
echo "  backend         : ${CAPTURE_BACKEND}"
echo "  replay file     : ${REPLAY_ACTIONS_FILE:-<none>}"
echo "  replay states   : ${RENDER_REPLAY_STATES}"
echo "  debug frames    : ${DEBUG_FRAMES}"
echo "  settle frames   : ${VIEWPORT_SETTLE_FRAMES}"
echo "  max steps       : ${MAX_STEPS}"
echo "  max episodes    : ${MAX_EPISODES}"
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
if [[ -n "${REPLAY_ACTIONS_FILE}" ]]; then
    if [[ -f "${REPLAY_ACTIONS_FILE}" ]]; then
        EXTRA_ARGS+=(--replay_actions_file "${REPLAY_ACTIONS_FILE}")
        if [[ "${RENDER_REPLAY_STATES}" == "1" ]]; then
            EXTRA_ARGS+=(--render_replay_states)
        fi
    else
        echo "WARNING: replay file not found; falling back to normal seeded rollout: ${REPLAY_ACTIONS_FILE}" >&2
    fi
fi

CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" src/kuafu/data/collection/capture_stack3_best_ema_media.py \
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
    --max_episodes "${MAX_EPISODES}" \
    --fps "${FPS}" \
    --video_every "${VIDEO_EVERY}" \
    --width "${WIDTH}" \
    --height "${HEIGHT}" \
    --output_dir "${OUTPUT_DIR}" \
    --viewport_settle_frames "${VIEWPORT_SETTLE_FRAMES}" \
    "${EXTRA_ARGS[@]}"
