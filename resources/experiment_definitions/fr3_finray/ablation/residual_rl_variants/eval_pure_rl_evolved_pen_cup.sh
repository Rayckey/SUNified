#!/usr/bin/env bash
set -euo pipefail

TASK="${KUAFU_TASK:-FR3-FinRay-Table-Pen-Cup-ff}"
CHECKPOINT_TAG="pure_rl_evolved_pen_cup"
CHECKPOINT_SELECTOR="${CHECKPOINT_SELECTOR:-named_best}"
WORKER_IDX="${WORKER_IDX:-2}"
AGENT_CFG="resources/experiment_definitions/fr3_finray/ablation/residual_rl_variants/agents/FR3-FinRay-Table-Pen-Cup-ff-pure-rl.yaml"
WEIGHTS_PARAMS_JSON="Residuals/FR3-FinRay-Table-Pen-Cup-ff/2026-05-23_17-52-24/best.json"
REWARD_PRESET="${REWARD_PRESET:-config}"
NAIVE_REWARD_EVAL="${NAIVE_REWARD_EVAL:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
source "${REPO_ROOT}/src/kuafu/rl/launchers/ablation/eval_common.sh"
