#!/usr/bin/env bash
set -euo pipefail

TASK="${KUAFU_TASK:-FR3-FinRay-Table-Stack-2-ff}"
CHECKPOINT_TAG="pure_rl_evolved_stack2"
CHECKPOINT_SELECTOR="${CHECKPOINT_SELECTOR:-named_best}"
WORKER_IDX="${WORKER_IDX:-0}"
AGENT_CFG="resources/experiment_definitions/fr3_finray/ablation/residual_rl_variants/agents/FR3-FinRay-Table-Stack-2-ff-pure-rl.yaml"
WEIGHTS_PARAMS_JSON="Residuals/FR3-FinRay-Table-Stack-2-ff/2026-05-17_11-53-49/best.json"
REWARD_PRESET="${REWARD_PRESET:-config}"
NAIVE_REWARD_EVAL="${NAIVE_REWARD_EVAL:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
source "${REPO_ROOT}/src/kuafu/rl/launchers/ablation/eval_common.sh"
