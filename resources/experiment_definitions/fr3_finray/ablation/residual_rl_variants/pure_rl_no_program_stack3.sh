#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
source "${REPO_ROOT}/src/kuafu/rl/launchers/ablation/abla_launcher_common.sh"
kuafu_abla_init_runtime_options

TASK="${KUAFU_TASK:-FR3-FinRay-Table-Stack-3-fff}"


# Stack 3: Pure RL + No Program. Uses fr3_finray_stack3_env_learned.external_rewards().
# Winning-run source family: resources/experiment_definitions/fr3_finray/residual_rl/prior_conditioned_training/fr3_finray_learned_stack3_state/stack3_chunking_15_fff_stride3.sh

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python src/kuafu/rl/launchers/ablation/train_fixed_program.py \
  --task "${TASK}" \
  "${KUAFU_ABLA_ROBOT_ARGS[@]}" \
  --device "${DEVICE:-cuda:0}" \
  --num_envs "${NUM_ENVS:-8192}" \
  --env_seed "${ENV_SEED:-42}" \
  --worker_idx "${WORKER_IDX:-3}" \
  --max_train_epoch "${MAX_TRAIN_EPOCH:-8000}" \
  --agent_cfg "resources/experiment_definitions/fr3_finray/ablation/residual_rl_variants/agents/FR3-FinRay-Table-Stack-3-fff-pure-rl.yaml" \
  --reward_preset external \
  --tag "pure_rl_no_program_stack3"
