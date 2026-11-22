#!/usr/bin/env bash
# Right-Can (Can-f): BC prior then FastTD3.
# Set ROBOT_ID=franka_panda or ROBOT_ID=kinova_gen3 to use that robot's
# artifact config root and task prefix. Defaults stay FR3FinRay for legacy runs.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOT_ID="${ROBOT_ID:-fr3_finray}"
ARTIFACT_ROOT="${KUAFU_ARTIFACT_ROOT:-../KuafuArtifacts}"

case "${ROBOT_ID}" in
  fr3_finray)
    TASK_ROOT="${TASK_ROOT:-FR3-FinRay-Table}"
    ;;
  franka_panda)
    TASK_ROOT="${TASK_ROOT:-Panda-Table}"
    ;;
  kinova_gen3)
    TASK_ROOT="${TASK_ROOT:-Kinova-Table-Gen3}"
    ;;
  *)
    echo "[fasttd3] unknown ROBOT_ID=${ROBOT_ID}" >&2
    exit 1
    ;;
esac

CONFIG_ROOT="${CONFIG_ROOT:-resources/experiment_definitions/${ROBOT_ID}/prior_configs}"
export AGENT_CFG="${AGENT_CFG:-${CONFIG_ROOT}/right_can/right_can_fasttd3.yaml}"
export TASK="${TASK:-${TASK_ROOT}-Can}"
if [[ -z "${RESIDUAL_WEIGHTS+x}" && "${ROBOT_ID}" == "fr3_finray" ]]; then
  _latest_can_weights="$(ls -1dt "${ARTIFACT_ROOT}/residual_rl/${TASK_ROOT}-Can-f"/*/best.json 2>/dev/null | head -n1 || true)"
  RESIDUAL_WEIGHTS="${_latest_can_weights}"
fi
export RESIDUAL_WEIGHTS="${RESIDUAL_WEIGHTS:-}"
exec "${SCRIPT_DIR}/train_fasttd3_pipeline.sh" "$@"
