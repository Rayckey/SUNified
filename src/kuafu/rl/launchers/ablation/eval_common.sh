#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/abla_launcher_common.sh"
kuafu_abla_init_runtime_options

: "${TASK:?Set TASK before sourcing eval_common.sh}"
: "${CHECKPOINT_TAG:?Set CHECKPOINT_TAG before sourcing eval_common.sh}"
: "${REWARD_PRESET:?Set REWARD_PRESET before sourcing eval_common.sh}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

CMD=(
  python src/kuafu/rl/launchers/ablation/eval_fixed_program_checkpoint.py
  --task "${TASK}"
  "${KUAFU_ABLA_ROBOT_ARGS[@]}"
  --device "${DEVICE:-cuda:0}"
  --num_envs "${NUM_ENVS:-8192}"
  --env_seed "${ENV_SEED:-42}"
  --worker_idx "${WORKER_IDX:-0}"
  --reward_preset "${REWARD_PRESET}"
  --checkpoint_tag "${CHECKPOINT_TAG}"
  --checkpoint_selector "${CHECKPOINT_SELECTOR:-named_best}"
  --tag "${EVAL_TAG:-eval_${CHECKPOINT_TAG}}"
)

[[ -n "${AGENT_CFG:-}" ]] && CMD+=(--agent_cfg "${AGENT_CFG}")
[[ -n "${WEIGHTS_PARAMS_JSON:-}" ]] && CMD+=(--weights_params_json "${WEIGHTS_PARAMS_JSON}")
[[ -n "${CLIP_ACTIONS:-}" ]] && CMD+=(--clip_actions "${CLIP_ACTIONS}")
case "${RESIDUAL_CLIP_COMPOSED_ACTIONS:-}" in
  1|true|TRUE|yes|YES|on|ON) CMD+=(--residual_clip_composed_actions) ;;
  0|false|FALSE|no|NO|off|OFF) CMD+=(--no-residual_clip_composed_actions) ;;
esac
[[ -n "${RESIDUAL_ALPHA_WARMUP_EPOCHS:-}" ]] && CMD+=(--residual_alpha_warmup_epochs "${RESIDUAL_ALPHA_WARMUP_EPOCHS}")
[[ -n "${PRIOR_CHECKPOINT:-}" ]] && CMD+=(--prior_checkpoint "${PRIOR_CHECKPOINT}")
[[ -n "${PRIOR_MODE:-}" ]] && CMD+=(--prior_mode "${PRIOR_MODE}")
[[ "${RESIDUAL_HIDE_STAGE_OBSERVATION:-0}" == "1" ]] && CMD+=(--residual_hide_stage_observation)
[[ -n "${CHECKPOINT:-}" ]] && CMD+=(--checkpoint "${CHECKPOINT}")
[[ -n "${RUN_DIR:-}" ]] && CMD+=(--run_dir "${RUN_DIR}")
[[ -n "${SEARCH_ROOT:-}" ]] && CMD+=(--search_root "${SEARCH_ROOT}")
[[ -n "${MIN_CHECKPOINT_EPOCH:-}" ]] && CMD+=(--min_checkpoint_epoch "${MIN_CHECKPOINT_EPOCH}")
[[ -n "${CHECKPOINT_METRIC_TAG:-}" ]] && CMD+=(--checkpoint_metric_tag "${CHECKPOINT_METRIC_TAG}")
[[ -n "${OUTPUT_DIR:-}" ]] && CMD+=(--output_dir "${OUTPUT_DIR}")
[[ -n "${MAX_EVAL_STEPS:-}" ]] && CMD+=(--max_eval_steps "${MAX_EVAL_STEPS}")
[[ -n "${PROGRESS_METRIC_EXPR:-}" ]] && CMD+=(--progress_metric_expr "${PROGRESS_METRIC_EXPR}")
case "${NAIVE_REWARD_EVAL:-0}" in
  1|true|TRUE|yes|YES|on|ON) CMD+=(--naive_reward_eval) ;;
esac
[[ -n "${NAIVE_REWARD_PRESET:-}" ]] && CMD+=(--naive_reward_preset "${NAIVE_REWARD_PRESET}")
[[ -n "${RENDER:-}" ]] && CMD+=(--render)

echo "Running: ${CMD[*]}"
if kuafu_abla_bool_enabled "${KUAFU_ABLA_DRY_RUN:-}"; then
  kuafu_abla_print_command "eval" "${CMD[@]}"
  exit 0
fi
exec "${CMD[@]}"
