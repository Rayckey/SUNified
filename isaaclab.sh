#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_RUNTIME_ROOT="${ISAACLAB_ROOT:-$(cd "${PROJECT_ROOT}/.." && pwd)}"
ISAACLAB_ENTRYPOINT="${ISAACLAB_RUNTIME_ROOT}/isaaclab.sh"

if [[ ! -x "${ISAACLAB_ENTRYPOINT}" ]]; then
    echo "ERROR: Isaac Lab launcher is not executable: ${ISAACLAB_ENTRYPOINT}" >&2
    echo "Set ISAACLAB_ROOT to the external IsaacLab checkout." >&2
    exit 2
fi

export KUAFU_PROJECT_ROOT="${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

exec "${ISAACLAB_ENTRYPOINT}" "$@"
