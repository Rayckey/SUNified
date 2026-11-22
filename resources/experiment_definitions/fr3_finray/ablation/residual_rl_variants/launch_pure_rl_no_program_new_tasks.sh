#!/usr/bin/env bash
set -euo pipefail

# Launch the four new Pure RL + No Program ablations on GPUs 0-3.
# Each run sees one GPU as cuda:0 through CUDA_VISIBLE_DEVICES.

STAMP="${STAMP:-$(date +%Y-%m-%d_%H-%M-%S)}"
LOG_DIR="${LOG_DIR:-resources/experiment_definitions/fr3_finray/ablation/residual_rl_variants/launch_logs/pure_rl_no_program_new_tasks_${STAMP}}"
mkdir -p "${LOG_DIR}"

pids=()

launch() {
  local gpu="$1"
  local worker_idx="$2"
  local name="$3"
  local script="$4"
  local log="${LOG_DIR}/${name}_gpu${gpu}.log"

  echo "Launching ${name} on GPU ${gpu}; log: ${log}"
  CUDA_VISIBLE_DEVICES="${gpu}" DEVICE="${DEVICE:-cuda:0}" WORKER_IDX="${worker_idx}" \
    bash "${script}" > "${log}" 2>&1 &
  pids+=("$!")
}

launch 0 0 "can" "resources/experiment_definitions/fr3_finray/ablation/residual_rl_variants/pure_rl_no_program_can.sh"
launch 1 1 "estop_open" "resources/experiment_definitions/fr3_finray/ablation/residual_rl_variants/pure_rl_no_program_estop_open.sh"
launch 2 2 "hang_cup" "resources/experiment_definitions/fr3_finray/ablation/residual_rl_variants/pure_rl_no_program_hang_cup.sh"
launch 3 3 "stack3" "resources/experiment_definitions/fr3_finray/ablation/residual_rl_variants/pure_rl_no_program_stack3.sh"

echo "Launched PIDs: ${pids[*]}"
echo "Logs: ${LOG_DIR}"

wait "${pids[@]}"
