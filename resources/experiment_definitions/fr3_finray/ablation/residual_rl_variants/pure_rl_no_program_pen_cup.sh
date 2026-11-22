#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
source "${REPO_ROOT}/src/kuafu/rl/launchers/ablation/abla_launcher_common.sh"
kuafu_abla_init_runtime_options

TASK="${KUAFU_TASK:-FR3-FinRay-Table-Pen-Cup-ff}"


# T3 Pen Cup: Pure RL + No Program.
# Loads downloaded best.json only for stage parameters/monitoring; reward is external_rewards().
# Winning-run source: resources/experiment_definitions/fr3_finray/residual_rl/prior_conditioned_training/fr3_finray_learned_pen_cup_random/ff-15-random.sh

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

WEIGHTS_PARAMS_JSON="${PEN_CUP_WEIGHTS_PARAMS_JSON:-${WEIGHTS_PARAMS_JSON:-Residuals/FR3-FinRay-Table-Pen-Cup-ff/2026-05-23_17-52-24/best.json}}"
if [[ -z "${WEIGHTS_PARAMS_JSON}" || ! -f "${WEIGHTS_PARAMS_JSON}" ]]; then
  echo "ERROR: fixed weights/params JSON not found. Set PEN_CUP_WEIGHTS_PARAMS_JSON=/path/to/best.json." >&2
  exit 1
fi
echo "Using weights/params JSON for stage monitoring: ${WEIGHTS_PARAMS_JSON}"

python src/kuafu/rl/launchers/ablation/train_fixed_program.py \
  --task "${TASK}" \
  "${KUAFU_ABLA_ROBOT_ARGS[@]}" \
  --device "${DEVICE:-cuda:0}" \
  --num_envs "${NUM_ENVS:-8192}" \
  --env_seed "${ENV_SEED:-42}" \
  --worker_idx "${WORKER_IDX:-3}" \
  --max_train_epoch "${MAX_TRAIN_EPOCH:-3000}" \
  --agent_cfg "resources/experiment_definitions/fr3_finray/ablation/residual_rl_variants/agents/FR3-FinRay-Table-Pen-Cup-ff-pure-rl.yaml" \
  --weights_params_json "${WEIGHTS_PARAMS_JSON}" \
  --reward_preset external \
  --tag "pure_rl_no_program_pen_cup"
