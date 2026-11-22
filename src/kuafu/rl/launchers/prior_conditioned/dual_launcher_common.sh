#!/usr/bin/env bash
# Shared helper for active Kuafu dual residual-hyperparameter launchers.

kuafu_dual_bool_enabled() {
    case "${1:-}" in
        1|true|TRUE|yes|YES|on|ON)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

kuafu_dual_init_runtime_options() {
    EXECUTION_ROBOT_ID="${KUAFU_EXECUTION_ROBOT_ID:-${EXECUTION_ROBOT_ID:-}}"
    ARTIFACT_ROBOT_ID="${KUAFU_ARTIFACT_ROBOT_ID:-${ARTIFACT_ROBOT_ID:-fr3_finray}}"
    KUAFU_LAUNCH_DRY_RUN="${KUAFU_LAUNCH_DRY_RUN:-${KUAFU_DRY_RUN:-${DRY_RUN:-}}}"

    KUAFU_ROBOT_ARGS=()
    [[ -n "${EXECUTION_ROBOT_ID}" ]] && KUAFU_ROBOT_ARGS+=(--execution_robot_id "${EXECUTION_ROBOT_ID}")
    [[ -n "${ARTIFACT_ROBOT_ID}" ]] && KUAFU_ROBOT_ARGS+=(--artifact_robot_id "${ARTIFACT_ROBOT_ID}")
}

kuafu_dual_require_precomputed_weights_params() {
    if [[ -z "${INITIAL_WEIGHTS_PARAMS:-}" ]]; then
        echo "ERROR: INITIAL_WEIGHTS_PARAMS must point to best.json from automatic tolerance assignment and the BC/post-BC weight query." >&2
        echo "The active residual-hparam pipeline does not run initial_constraint_sweep.py." >&2
        exit 1
    fi
    if ! kuafu_dual_bool_enabled "${KUAFU_LAUNCH_DRY_RUN:-}" && [[ ! -f "${INITIAL_WEIGHTS_PARAMS}" ]]; then
        echo "ERROR: INITIAL_WEIGHTS_PARAMS does not exist: ${INITIAL_WEIGHTS_PARAMS}" >&2
        exit 1
    fi
}

kuafu_dual_print_command() {
    local label="$1"
    shift
    echo "Kuafu dual dry-run: ${label}"
    printf '  %q' "$@"
    echo
}
