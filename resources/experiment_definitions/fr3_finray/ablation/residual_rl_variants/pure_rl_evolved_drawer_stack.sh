#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
source "${REPO_ROOT}/src/kuafu/rl/launchers/ablation/abla_launcher_common.sh"
kuafu_abla_init_runtime_options

TASK="${KUAFU_TASK:-FR3-FinRay-Table-Drawer-Stack-fff}"


# Drawer Stack: Pure RL + Evolved Program.
# Uses the exact best.json weights/parameters referenced by the winning run.
# Winning-run source: resources/experiment_definitions/fr3_finray/residual_rl/prior_conditioned_training/fr3_finray_learned_drawer_stack/drawer_stack_chunking_15_fff_stride3.sh

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python src/kuafu/rl/launchers/ablation/train_fixed_program.py \
  --task "${TASK}" \
  "${KUAFU_ABLA_ROBOT_ARGS[@]}" \
  --device "${DEVICE:-cuda:0}" \
  --num_envs "${NUM_ENVS:-8192}" \
  --env_seed "${ENV_SEED:-42}" \
  --worker_idx "${WORKER_IDX:-2}" \
  --max_train_epoch "${MAX_TRAIN_EPOCH:-2500}" \
  --agent_cfg "resources/experiment_definitions/fr3_finray/ablation/residual_rl_variants/agents/FR3-FinRay-Table-Drawer-Stack-fff-pure-rl.yaml" \
  --weights_params_json "Residuals/FR3-FinRay-Table-Drawer-Stack-fff/2026-05-14_23-58-02/best.json" \
  --reward_preset config \
  --tag "pure_rl_evolved_drawer_stack"
