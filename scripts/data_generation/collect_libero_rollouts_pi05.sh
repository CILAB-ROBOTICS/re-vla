#!/usr/bin/env bash
# Roll out a fine-tuned Pi0.5 checkpoint on LIBERO and collect
# success / failure / recoverable-failure episodes (see collect_libero_rollouts.py).
# collect_libero_rollouts.py is policy-agnostic (it reads the policy type from the
# checkpoint's own config.json), so this only differs from the SmolVLA wrapper in
# its defaults.
# Any extra args are passed through, e.g.:
#   ./collect_libero_rollouts_pi05.sh --task libero_spatial --episodes-per-task 10
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Defaults match finetune_pi05_libero.sh's --output-dir.
POLICY_PATH="${POLICY_PATH:-./outputs/libero_pi05/checkpoints/last/pretrained_model}"
TASK="${TASK:-libero_10}"
EPISODES_PER_TASK="${EPISODES_PER_TASK:-5}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/libero_rollouts_pi05}"
REPO_ID="${REPO_ID:-local/libero_pi05_rollouts}"

python "${SCRIPT_DIR}/collect_libero_rollouts.py" \
    --policy-path "${POLICY_PATH}" \
    --task "${TASK}" \
    --episodes-per-task "${EPISODES_PER_TASK}" \
    --output-dir "${OUTPUT_DIR}" \
    --repo-id "${REPO_ID}" \
    "$@"
