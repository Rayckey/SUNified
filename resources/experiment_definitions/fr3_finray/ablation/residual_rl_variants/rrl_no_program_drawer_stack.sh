#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
source "${REPO_ROOT}/src/kuafu/rl/launchers/ablation/abla_launcher_common.sh"
kuafu_abla_init_runtime_options

TASK="${KUAFU_TASK:-FR3-FinRay-Table-Drawer-Stack-fff}"


# Drawer Stack: RRL with the no-stage prior and downloaded best.json.
# Hide scalar stage observations during residual training.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PRIOR_CHECKPOINT="${NO_STAGE_DRAWER_STACK_PRIOR:-${NO_STAGE_PRIOR_CHECKPOINT:-resources/experiment_definitions/fr3_finray/ablation/prior_variants/logs/drawer_stack_chunking_15_fff_stride3_no_stage/2026-05-27_00-28-21/checkpoints/best.pth}}"
if [[ -z "${PRIOR_CHECKPOINT}" || ! -f "${PRIOR_CHECKPOINT}" ]]; then
  echo "ERROR: no-stage prior checkpoint not found. Set NO_STAGE_DRAWER_STACK_PRIOR=/path/to/best.pth." >&2
  exit 1
fi
echo "Using no-stage prior: ${PRIOR_CHECKPOINT}"

WEIGHTS_PARAMS_JSON="${DRAWER_STACK_WEIGHTS_PARAMS_JSON:-${WEIGHTS_PARAMS_JSON:-Residuals/FR3-FinRay-Table-Drawer-Stack-fff/2026-05-14_23-58-02/best.json}}"
if [[ -z "${WEIGHTS_PARAMS_JSON}" || ! -f "${WEIGHTS_PARAMS_JSON}" ]]; then
  echo "ERROR: fixed weights/params JSON not found. Set DRAWER_STACK_WEIGHTS_PARAMS_JSON=/path/to/best.json." >&2
  exit 1
fi
echo "Using weights/params JSON: ${WEIGHTS_PARAMS_JSON}"

python src/kuafu/rl/launchers/ablation/train_fixed_program.py \
  --task "${TASK}" \
  "${KUAFU_ABLA_ROBOT_ARGS[@]}" \
  --device "${DEVICE:-cuda:0}" \
  --num_envs "${NUM_ENVS:-8192}" \
  --env_seed "${ENV_SEED:-42}" \
  --worker_idx "${WORKER_IDX:-2}" \
  --max_train_epoch "${MAX_TRAIN_EPOCH:-2500}" \
  --weights_params_json "${WEIGHTS_PARAMS_JSON}" \
  --clip_actions "${CLIP_ACTIONS:-2.0}" \
  --prior_checkpoint "${PRIOR_CHECKPOINT}" \
  --prior_mode residual_rl \
  --prior_obs_stats_count "${INITIAL_PRIOR_OBS_STATS_COUNT:-15000000}" \
  --prior_critic_init_mode actor_only_random_critic \
  --prior_actor_lr_cap 0.0 \
  --prior_bc_anchor_coef 0.0 \
  --prior_bc_dataset_coef 0.0 \
  --prior_bc_dataset_batch_size 1024 \
  --reward_preset config \
  --residual_hide_stage_observation \
  --residual_clip_composed_actions \
  --residual_alpha_warmup_epochs "${RESIDUAL_ALPHA_WARMUP_EPOCHS:-0}" \
  --tag "rrl_no_program_drawer_stack"
