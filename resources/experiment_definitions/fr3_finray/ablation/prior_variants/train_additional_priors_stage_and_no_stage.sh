#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

usage() {
  cat <<'EOF'
Usage:
  resources/experiment_definitions/fr3_finray/ablation/prior_variants/train_additional_priors_stage_and_no_stage.sh [options]

Trains the four additional BC priors selected from
scripts/data_transfer/download_success_epoch_ema_checkpoints.sh and its prior CSV:
  estop_open, stack3, hang_cup, right_can

Each task has both a normal stage-aware prior and a stage-blind no-stage prior.

Options:
  --mode MODE    stage, no_stage, or both. Default: both.
  --only LIST    Comma-separated task names, e.g. estop_open,stack3.
  --dry-run      Print commands without executing them.
  -h, --help     Show this help.

Environment:
  CONDA_ENV      If set, run with: conda run -n "$CONDA_ENV" python.
  PYTHON_BIN     Python executable when CONDA_ENV is unset. Default: python.

Examples:
  CONDA_ENV=isaac resources/experiment_definitions/fr3_finray/ablation/prior_variants/train_additional_priors_stage_and_no_stage.sh
  CUDA_VISIBLE_DEVICES=1 CONDA_ENV=isaac resources/experiment_definitions/fr3_finray/ablation/prior_variants/train_additional_priors_stage_and_no_stage.sh --mode no_stage --only stack3
EOF
}

MODE="${MODE:-both}"
ONLY="${ONLY:-}"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --only)
      ONLY="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "${MODE}" in
  stage|no_stage|both) ;;
  *)
    echo "Invalid --mode: ${MODE}. Use stage, no_stage, or both." >&2
    exit 2
    ;;
esac

if [[ -n "${CONDA_ENV:-}" ]]; then
  PYTHON_CMD=(conda run -n "${CONDA_ENV}" python)
else
  PYTHON_CMD=("${PYTHON_BIN:-python}")
fi

should_run_task() {
  local name="$1"
  [[ -z "${ONLY}" || ",${ONLY}," == *",${name},"* ]]
}

should_run_mode() {
  local mode="$1"
  [[ "${MODE}" == "both" || "${MODE}" == "${mode}" ]]
}

RUNS=(
  "estop_open:stage:resources/experiment_definitions/fr3_finray/ablation/prior_variants/configs/estop_open_chunking_15_ff_stage.yaml:src/kuafu/bc/training/train_bc_prior_mlp.py"
  "estop_open:no_stage:resources/experiment_definitions/fr3_finray/ablation/prior_variants/configs/estop_open_chunking_15_ff_no_stage.yaml:src/kuafu/bc/training/ablation/train_bc_prior_mlp_no_stage.py"
  "stack3:stage:resources/experiment_definitions/fr3_finray/ablation/prior_variants/configs/stack3_chunking_15_fff_stride3_stage.yaml:src/kuafu/bc/training/train_bc_prior_mlp.py"
  "stack3:no_stage:resources/experiment_definitions/fr3_finray/ablation/prior_variants/configs/stack3_chunking_15_fff_stride3_no_stage.yaml:src/kuafu/bc/training/ablation/train_bc_prior_mlp_no_stage.py"
  "hang_cup:stage:resources/experiment_definitions/fr3_finray/ablation/prior_variants/configs/hang_cup_chunking_10_state_fff_stage.yaml:src/kuafu/bc/training/train_bc_prior_mlp.py"
  "hang_cup:no_stage:resources/experiment_definitions/fr3_finray/ablation/prior_variants/configs/hang_cup_chunking_10_state_fff_no_stage.yaml:src/kuafu/bc/training/ablation/train_bc_prior_mlp_no_stage.py"
  "right_can:stage:resources/experiment_definitions/fr3_finray/ablation/prior_variants/configs/right_can_chunking_15_ff_damped_stage.yaml:src/kuafu/bc/training/train_bc_prior_mlp.py"
  "right_can:no_stage:resources/experiment_definitions/fr3_finray/ablation/prior_variants/configs/right_can_chunking_15_ff_damped_no_stage.yaml:src/kuafu/bc/training/ablation/train_bc_prior_mlp_no_stage.py"
)

echo "[Abla] Repo root: ${ROOT}"
echo "[Abla] Python: ${PYTHON_CMD[*]}"
echo "[Abla] Mode: ${MODE}"
echo "[Abla] Only: ${ONLY:-all}"

for item in "${RUNS[@]}"; do
  IFS=":" read -r task mode config trainer <<<"${item}"
  if ! should_run_task "${task}" || ! should_run_mode "${mode}"; then
    continue
  fi

  cmd=("${PYTHON_CMD[@]}" "${trainer}" --config "${config}")
  echo
  echo "[Abla] Training ${task} (${mode})"
  echo "[Abla] Config: ${config}"
  echo "[Abla] Command: ${cmd[*]}"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    continue
  fi
  "${cmd[@]}"
done

echo
echo "[Abla] Done. Checkpoints are under resources/experiment_definitions/fr3_finray/ablation/prior_variants/logs/<run>/<timestamp>/checkpoints/."
