#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# BC prior (MLP 256-512-64, activation selectable, Adam) utilities
# ============================================================================
#
# This script documents *all supported flags* for the offline BC prior pipeline
# and provides runnable example commands.
#
# NOTE (PPO config match):
# - Your RL-Games Pen-Cup PPO config uses MLP units [256, 512, 64] and sets
#   normalize_input: True (see `src/kuafu/isaac/tasks/pen_cup/agents/rl_games_ppo_cfg.yaml`).
# - That config uses `activation: relu` for the PPO MLP. You can train the BC
#   prior with either:
#     --activation relu       (exact activation match to PPO)
#     --activation leaky_relu (your requested activation; PPO still uses ReLU)
#   Prior injection copies *Linear weights only*; PPO activations remain ReLU.
#
# Requirements:
# - Use conda env `isaac` (recommended). Commands below use `conda run -n isaac`.
#
# Paths:
ARTIFACT_ROOT="${KUAFU_ARTIFACT_ROOT:-../KuafuArtifacts}"
ROBOT_ID="${ROBOT_ID:-fr3_finray}"
TASK_SLUG="${TASK_SLUG:-pen_cup}"
CONFIG_NAME="${CONFIG_NAME:-pen_cup_chunking_15_l.yaml}"
DATASET_DEFAULT="${DATASET_DEFAULT:-}"
if [[ -z "${DATASET_DEFAULT}" && "${ROBOT_ID}" == "fr3_finray" ]]; then
  DATASET_DEFAULT="${ARTIFACT_ROOT}/Trajectories/fr3_finray_pen_cup_damped_noise.hdf5"
fi
OBS_KEY_DEFAULT="${OBS_KEY_DEFAULT:-full_state}"   # choices: state | full_state

BC_DIR="src/kuafu/bc/training"
CONFIG_PATH="${CONFIG_PATH:-resources/experiment_definitions/${ROBOT_ID}/prior_configs/${TASK_SLUG}/${CONFIG_NAME}}"
LOG_DIR="${ARTIFACT_ROOT}/standalone/bc_prior_mlp"

# ----------------------------------------------------------------------------
# 0) Inspect HDF5 keys + value ranges (sanity checks)
# ----------------------------------------------------------------------------
# Script: ${BC_DIR}/inspect_hdf5_keys.py
#
# Flags:
#   --dataset PATH              (required)
#   --max_tree_lines INT        (default: 500)
#   --sample_rows INT           (default: 4096)

# conda run -n isaac python "${BC_DIR}/inspect_hdf5_keys.py" \
#   --dataset "${DATASET_DEFAULT}" \
#   --max_tree_lines 500 \
#   --sample_rows 4096

# ----------------------------------------------------------------------------
# 1) Train BC prior MLP (offline)
# ----------------------------------------------------------------------------
# Script: ${BC_DIR}/train_bc_prior_mlp.py
#
# YAML config files in this folder:
# Example config:
#   - resources/experiment_definitions/${ROBOT_ID}/prior_configs/${TASK_SLUG}/${CONFIG_NAME}
#
# Only supported flag:
#   --config PATH                  (required YAML file with all training settings)
#
# Example (recommended for a single long demo):
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[run.sh] missing CONFIG_PATH=${CONFIG_PATH}" >&2
  echo "[run.sh] set ROBOT_ID, TASK_SLUG, CONFIG_NAME, or CONFIG_PATH explicitly" >&2
  exit 1
fi
python "${BC_DIR}/train_bc_prior_mlp.py" \
  --config "${CONFIG_PATH}"

# Outputs:
# - TensorBoard logs under ${LOG_DIR}/<timestamp>/
# - Checkpoints: ${LOG_DIR}/<timestamp>/checkpoints/{best.pth,final.pth,...}
# - Gradient PNG: ${LOG_DIR}/<timestamp>/grad_norm.png
#
# TensorBoard:
#   tensorboard --logdir "${LOG_DIR}"

# ----------------------------------------------------------------------------
# 2) Evaluate BC prior checkpoint on the dataset (offline)
# ----------------------------------------------------------------------------
# Script: ${BC_DIR}/eval_bc_prior_on_dataset.py
#
# Flags:
#   --checkpoint PATH              (required) path to best.pth or final.pth
#   --dataset PATH                 (default: ${DATASET_DEFAULT})
#   --demo DEMO_KEY                (optional)
#   --obs_key {state,full_state}   (optional; defaults to checkpoint's obs_key)
#   --num_samples INT              (default: 512)
#   --seed INT                     (default: 42)
#   --eval_all_shifts              (if set, evaluates shifts 0,+/-1,+/-2)
#
# Example:
# CKPT="${ARTIFACT_ROOT}/standalone/bc_prior_mlp/<RUN>/checkpoints/best.pth"
# conda run -n isaac python "${BC_DIR}/eval_bc_prior_on_dataset.py" \
#   --checkpoint "${CKPT}" \
#   --dataset "${DATASET_DEFAULT}" \
#   --obs_key "${OBS_KEY_DEFAULT}" \
#   --eval_all_shifts \
#   --num_samples 512

# ----------------------------------------------------------------------------
# 3) Use BC checkpoint as PPO prior (RL-Games)
# ----------------------------------------------------------------------------
# Integration patch was added to:
#   scripts/reinforcement_learning/rl_games/train.py
#
# New RL-Games flags:
#   --prior_ckpt PATH              (BC checkpoint .pth from train_bc_prior_mlp.py)
#   --prior_obs_stats_count FLOAT  (default: 1e6; sets RunningMeanStd count)
#   --prior_verbose                (prints copied layer shapes)
#
# Example (fill TASK/AGENT as you normally run RL-Games):
# PRIOR_CKPT="${ARTIFACT_ROOT}/standalone/bc_prior_mlp/<RUN>/checkpoints/best.pth"
# conda run -n isaac python scripts/reinforcement_learning/rl_games/train.py \
#   --task "<YOUR_TASK_NAME>" \
#   --agent "rl_games_cfg_entry_point" \
#   --prior_ckpt "${PRIOR_CKPT}" \
#   --prior_verbose
#
# Notes:
# - Your Pen-Cup config sets `normalize_input: True`, so prior injection will also
#   set rl-games RunningMeanStd to the BC obs mean/std.
# - If you also pass --checkpoint (RL-Games resume), RL-Games will resume from that
#   checkpoint instead of applying the BC prior init.

echo "[run.sh] Done (this script is mainly documentation + examples)."