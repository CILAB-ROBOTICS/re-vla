#!/usr/bin/env bash
# Roll out a fine-tuned SmolVLA checkpoint on LIBERO and collect
# success / failure / recoverable-failure episodes (see collect_libero_rollouts.py).
# Any extra args are passed through, e.g.:
#   ./collect_libero_rollouts_smolvla.sh --task libero_spatial --episodes-per-task 10
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Defaults match finetune_smolvla_libero.sh's --output-dir.
POLICY_PATH="${POLICY_PATH:-../fine_tuning/outputs/libero_smolvla/checkpoints/last/pretrained_model}"
TASK="${TASK:-libero_10}"
EPISODES_PER_TASK="${EPISODES_PER_TASK:-5}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/libero_rollouts_smolvla}"
REPO_ID="${REPO_ID:-local/libero_smolvla_rollouts}"

python "${SCRIPT_DIR}/collect_libero_rollouts.py" \
    --policy-path "${POLICY_PATH}" \
    --task "${TASK}" \
    --episodes-per-task "${EPISODES_PER_TASK}" \
    --output-dir "${OUTPUT_DIR}" \
    --repo-id "${REPO_ID}" \
    "$@"
