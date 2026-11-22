#!/usr/bin/env bash
set -euo pipefail

# Alias for the EStop environment file name. The registered task is EStop-Open.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/pure_rl_no_program_estop_open.sh"
