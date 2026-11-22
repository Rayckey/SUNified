#!/usr/bin/env bash
# Shared shell helpers for Kuafu ablation launchers.

kuafu_abla_bool_enabled() {
    case "${1:-}" in
        1|true|TRUE|yes|YES|on|ON) return 0 ;;
        *) return 1 ;;
    esac
}

kuafu_abla_init_runtime_options() {
    EXECUTION_ROBOT_ID="${KUAFU_EXECUTION_ROBOT_ID:-${EXECUTION_ROBOT_ID:-}}"
    ARTIFACT_ROBOT_ID="${KUAFU_ARTIFACT_ROBOT_ID:-${ARTIFACT_ROBOT_ID:-fr3_finray}}"
    KUAFU_ABLA_DRY_RUN="${KUAFU_ABLA_DRY_RUN:-${KUAFU_DRY_RUN:-${DRY_RUN:-}}}"

    KUAFU_ABLA_ROBOT_ARGS=()
    [[ -n "${EXECUTION_ROBOT_ID}" ]] && KUAFU_ABLA_ROBOT_ARGS+=(--execution_robot_id "${EXECUTION_ROBOT_ID}")
    [[ -n "${ARTIFACT_ROBOT_ID}" ]] && KUAFU_ABLA_ROBOT_ARGS+=(--artifact_robot_id "${ARTIFACT_ROBOT_ID}")
}

kuafu_abla_print_command() {
    local label="$1"
    shift
    printf '\nKuafu Abla dry-run: %s\n  ' "${label}"
    printf '%q  ' "$@"
    printf '\n'
}
