#!/usr/bin/env bash
# train_fasttd3_pipeline.sh — BC prior then FastTD3 RL, driven by env vars.
#
# Required env vars:
#   AGENT_CFG           Path to *_fasttd3.yaml (shared by BC prior and FastTD3)
#   TASK                IsaacLab registered task id (e.g. FR3-FinRay-Table-Drawer-f)
#
# Optional env vars:
#   RESIDUAL_WEIGHTS    Path to a residual-RL best.json for --residual_weights (skip if empty)
#   BUFFER_SIZE         --buffer_size for FastTD3 (default: 12288)
#   NUM_ENVS            --num_envs for FastTD3 (default: 3072)
#   SKIP_BC             Set to 1 to skip BC prior training and reuse the newest
#                       best.pth under the YAML's logging.log_dir
#   PRIOR_CKPT          Explicit path to best.pth; overrides auto-discovery
#   EXTRA_BC_ARGS       Extra args appended to the BC command
#   EXTRA_RL_ARGS       Extra args appended to the FastTD3 command
#
# Usage:
#   AGENT_CFG=... TASK=... bash scripts/prior_training/fasttd3/train_fasttd3_pipeline.sh
# or invoke one of the per-task wrappers in this folder.

set -euo pipefail

: "${AGENT_CFG:?set AGENT_CFG to path of the *_fasttd3.yaml}"
: "${TASK:?set TASK to an IsaacLab registered task id}"
RESIDUAL_WEIGHTS="${RESIDUAL_WEIGHTS:-}"
BUFFER_SIZE="${BUFFER_SIZE:-12288}"
NUM_ENVS="${NUM_ENVS:-3072}"
SKIP_BC="${SKIP_BC:-0}"
PRIOR_CKPT="${PRIOR_CKPT:-}"
EXTRA_BC_ARGS="${EXTRA_BC_ARGS:-}"
EXTRA_RL_ARGS="${EXTRA_RL_ARGS:-}"

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

BC_SCRIPT="${REPO_ROOT}/src/kuafu/bc/training/train_bc_prior_mlp.py"
RL_SCRIPT="${REPO_ROOT}/src/kuafu/isaac/tasks/train_fasttd3.py"

[[ -f "${BC_SCRIPT}" ]] || { echo "[pipeline] missing ${BC_SCRIPT}" >&2; exit 1; }
[[ -f "${RL_SCRIPT}" ]] || { echo "[pipeline] missing ${RL_SCRIPT}" >&2; exit 1; }
[[ -f "${AGENT_CFG}" ]] || { echo "[pipeline] missing AGENT_CFG=${AGENT_CFG}" >&2; exit 1; }

cd "${REPO_ROOT}"

# Pull logging.log_dir out of the YAML so we can auto-discover the BC checkpoint.
BC_LOG_DIR="$(python - <<PYEOF
import yaml
with open("${AGENT_CFG}") as f:
    cfg = yaml.safe_load(f)
print((cfg.get("logging") or {}).get("log_dir", ""))
PYEOF
)"
[[ -n "${BC_LOG_DIR}" ]] || { echo "[pipeline] logging.log_dir missing from ${AGENT_CFG}" >&2; exit 1; }

# -------- 1) BC prior --------
if [[ "${SKIP_BC}" != "1" ]]; then
  echo "[pipeline] (1/2) BC prior training with ${AGENT_CFG}"
  # shellcheck disable=SC2086
  python "${BC_SCRIPT}" --config "${AGENT_CFG}" ${EXTRA_BC_ARGS}
else
  echo "[pipeline] SKIP_BC=1 → skipping BC prior training"
fi

# -------- 2) Discover BC checkpoint --------
if [[ -z "${PRIOR_CKPT}" ]]; then
  PRIOR_CKPT="$(ls -1dt "${BC_LOG_DIR}"/*/checkpoints/best.pth 2>/dev/null | head -n1 || true)"
fi
if [[ -z "${PRIOR_CKPT}" || ! -f "${PRIOR_CKPT}" ]]; then
  echo "[pipeline] could not find a best.pth under ${BC_LOG_DIR}/*/checkpoints/" >&2
  exit 1
fi
echo "[pipeline] using BC prior: ${PRIOR_CKPT}"

# -------- 3) FastTD3 RL --------
CMD=( python "${RL_SCRIPT}"
  --task "${TASK}"
  --agent_cfg "${AGENT_CFG}"
  --prior_checkpoint "${PRIOR_CKPT}"
  --buffer_size "${BUFFER_SIZE}"
)
[[ -n "${RESIDUAL_WEIGHTS}" ]] && CMD+=( --residual_weights "${RESIDUAL_WEIGHTS}" )
[[ -n "${NUM_ENVS}" ]]         && CMD+=( --num_envs "${NUM_ENVS}" )
# shellcheck disable=SC2206
[[ -n "${EXTRA_RL_ARGS}" ]]    && CMD+=( ${EXTRA_RL_ARGS} )

echo "[pipeline] (2/2) FastTD3 RL: ${CMD[*]}"
"${CMD[@]}"
