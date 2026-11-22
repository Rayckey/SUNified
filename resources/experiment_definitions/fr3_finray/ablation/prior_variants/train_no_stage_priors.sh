#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

usage() {
  cat <<'EOF'
Usage:
  resources/experiment_definitions/fr3_finray/ablation/prior_variants/train_no_stage_priors.sh [options]

Trains the four stage-blind BC prior ablations selected from
scripts/data_transfer/download_success_epoch_ema_checkpoints.sh and its prior CSV:
  drawer, stack2, pen_cup, drawer_stack

Options:
  --only LIST    Comma-separated run names to train, e.g. drawer,pen_cup.
  --dry-run      Print commands without executing them.
  -h, --help     Show this help.

Environment:
  CONDA_ENV      If set, run with: conda run -n "$CONDA_ENV" python.
  PYTHON_BIN     Python executable when CONDA_ENV is unset. Default: python.

Examples:
  CONDA_ENV=isaac resources/experiment_definitions/fr3_finray/ablation/prior_variants/train_no_stage_priors.sh
  CUDA_VISIBLE_DEVICES=1 CONDA_ENV=isaac resources/experiment_definitions/fr3_finray/ablation/prior_variants/train_no_stage_priors.sh --only drawer
EOF
}

ONLY="${ONLY:-}"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
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

if [[ -n "${CONDA_ENV:-}" ]]; then
  PYTHON_CMD=(conda run -n "${CONDA_ENV}" python)
else
  PYTHON_CMD=("${PYTHON_BIN:-python}")
fi

should_run() {
  local name="$1"
  [[ -z "${ONLY}" || ",${ONLY}," == *",${name},"* ]]
}

RUNS=(
  "drawer:resources/experiment_definitions/fr3_finray/ablation/prior_variants/configs/drawer_chunking_30_ff_no_stage.yaml"
  "stack2:resources/experiment_definitions/fr3_finray/ablation/prior_variants/configs/stack2_chunking_15_shift_ff_no_stage.yaml"
  "pen_cup:resources/experiment_definitions/fr3_finray/ablation/prior_variants/configs/pen_cup_chunking_15_ff_no_stage.yaml"
  "drawer_stack:resources/experiment_definitions/fr3_finray/ablation/prior_variants/configs/drawer_stack_chunking_15_fff_stride3_no_stage.yaml"
)

echo "[Abla] Repo root: ${ROOT}"
echo "[Abla] Python: ${PYTHON_CMD[*]}"
echo "[Abla] Only: ${ONLY:-all}"

for item in "${RUNS[@]}"; do
  name="${item%%:*}"
  config="${item#*:}"
  if ! should_run "${name}"; then
    continue
  fi

  cmd=("${PYTHON_CMD[@]}" src/kuafu/bc/training/ablation/train_bc_prior_mlp_no_stage.py --config "${config}")
  echo
  echo "[Abla] Training ${name}"
  echo "[Abla] Config: ${config}"
  echo "[Abla] Command: ${cmd[*]}"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    continue
  fi
  "${cmd[@]}"
done

echo
echo "[Abla] Done. Checkpoints are under resources/experiment_definitions/fr3_finray/ablation/prior_variants/logs/<run>/<timestamp>/checkpoints/."
