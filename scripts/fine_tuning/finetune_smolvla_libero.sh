#!/usr/bin/env bash
# Fine-tune SmolVLA on LIBERO (lerobot/libero dataset, VLM-init from pretrained weights).
# Any extra args are passed through, e.g.:
#   ./finetune_smolvla_libero.sh --steps 50000 --batch-size 16
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_REPO_ID="${DATASET_REPO_ID:-lerobot/libero_goal_image}" # libero, libero_spatial_image, libero_object_image
STEPS="${STEPS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-64}"   # conservative default for ~8GB GPUs

python "${SCRIPT_DIR}/finetune_libero.py" \
    --policy-type smolvla \
    --dataset-repo-id "${DATASET_REPO_ID}" \
    --output-dir ./outputs/libero_smolvla/libero_goal \
    --wandb \
    --wandb-project libero-vla \
    --steps 20000 \
    --batch-size "${BATCH_SIZE}" \
    --load-vlm-weights \
    "$@"