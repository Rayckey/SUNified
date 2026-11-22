#!/usr/bin/env bash
# Compatibility entrypoint for the accepted Drawer artifact, deployed as fff.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/collect_and_upload_loop_kinova_drawer_fff.sh" "$@"
