#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
source "${REPO_ROOT}/src/kuafu/rl/launchers/ablation/abla_launcher_common.sh"
kuafu_abla_init_runtime_options

TASK="${KUAFU_TASK:-FR3-FinRay-Table-Stack-2-ff}"


# T6 Stack 2: RRL + Stage Prior + downloaded best.json params + manual external_rewards(), no LLM/VLM.
# Winning-run source: resources/experiment_definitions/fr3_finray/residual_rl/prior_conditioned_training/fr3_finray_learned_stack2_state/ff-15-shift-higher.sh

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PRIOR_CHECKPOINT="${STAGE_STACK2_PRIOR:-${STAGE_PRIOR_CHECKPOINT:-Priors/thor/stack_2_damped_higher_noise_state/stack2_chunking_15_shift_ff_damped_higher/best.pth}}"
if [[ -z "${PRIOR_CHECKPOINT}" || ! -f "${PRIOR_CHECKPOINT}" ]]; then
  echo "ERROR: stage-aware prior checkpoint not found. Set STAGE_STACK2_PRIOR=/path/to/best.pth." >&2
  exit 1
fi
echo "Using stage-aware prior: ${PRIOR_CHECKPOINT}"

WEIGHTS_PARAMS_JSON="${STACK2_WEIGHTS_PARAMS_JSON:-${STAGE_WEIGHTS_PARAMS_JSON:-${WEIGHTS_PARAMS_JSON:-Residuals/FR3-FinRay-Table-Stack-2-ff/2026-05-17_11-53-49/best.json}}}"
if [[ -z "${WEIGHTS_PARAMS_JSON}" || ! -f "${WEIGHTS_PARAMS_JSON}" ]]; then
  echo "ERROR: fixed weights/params JSON not found. Set STACK2_WEIGHTS_PARAMS_JSON, STAGE_WEIGHTS_PARAMS_JSON, or WEIGHTS_PARAMS_JSON=/path/to/best.json." >&2
  exit 1
fi
echo "Using weights/params JSON: ${WEIGHTS_PARAMS_JSON}"

python src/kuafu/rl/launchers/ablation/train_fixed_program.py \
  --task "${TASK}" \
  "${KUAFU_ABLA_ROBOT_ARGS[@]}" \
  --device "${DEVICE:-cuda:0}" \
  --num_envs "${NUM_ENVS:-8192}" \
  --env_seed "${ENV_SEED:-42}" \
  --worker_idx "${WORKER_IDX:-0}" \
  --max_train_epoch "${MAX_TRAIN_EPOCH:-1500}" \
  --weights_params_json "${WEIGHTS_PARAMS_JSON}" \
  --prior_checkpoint "${PRIOR_CHECKPOINT}" \
  --prior_mode residual_rl \
  --prior_obs_stats_count "${INITIAL_PRIOR_OBS_STATS_COUNT:-15000000}" \
  --prior_critic_init_mode actor_only_random_critic \
  --prior_actor_lr_cap 0.0 \
  --prior_bc_anchor_coef 0.0 \
  --prior_bc_dataset_coef 0.0 \
  --prior_bc_dataset_batch_size 1024 \
  --reward_preset config \
  --tag "rrl_stage_stack2"
