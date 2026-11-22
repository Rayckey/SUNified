#!/usr/bin/env bash
set -euo pipefail

# Drawer Stack: RRL + Tuned Program (Kuafu), fixed-program replay without LLM/VLM.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python src/kuafu/rl/launchers/ablation/train_fixed_program.py \
  --task "FR3-FinRay-Table-Drawer-Stack-fff" \
  --device "${DEVICE:-cuda:0}" \
  --num_envs "${NUM_ENVS:-8192}" \
  --env_seed "${ENV_SEED:-42}" \
  --worker_idx "${WORKER_IDX:-2}" \
  --max_train_epoch "${MAX_TRAIN_EPOCH:-2500}" \
  --weights_params_json "Residuals/FR3-FinRay-Table-Drawer-Stack-fff/2026-05-14_23-58-02/best.json" \
  --prior_checkpoint "Priors/herm/drawer_stack/drawer_stack_chunking_15_fff_stride3/best.pth" \
  --prior_mode residual_rl \
  --prior_obs_stats_count "${INITIAL_PRIOR_OBS_STATS_COUNT:-15000000}" \
  --prior_critic_init_mode actor_only_random_critic \
  --prior_actor_lr_cap 0.0 \
  --prior_bc_anchor_coef 0.0 \
  --prior_bc_dataset_coef 0.0 \
  --prior_bc_dataset_batch_size 1024 \
  --reward_preset config \
  --tag "rrl_tuned_drawer_stack"
