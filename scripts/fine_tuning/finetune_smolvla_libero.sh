#!/usr/bin/env bash
# Fine-tune SmolVLA on LIBERO (lerobot/libero dataset, VLM-init from pretrained weights).
# Any extra args are passed through, e.g.:
#   ./finetune_smolvla_libero.sh --steps 50000 --batch-size 16
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

OUTPUT_DIR="${OUTPUT_DIR:-/mnt/nas/cilab_robotics/re_vla/libero_smolvla/libero_goal}"
DATASET_REPO_ID="${DATASET_REPO_ID:-lerobot/libero_goal_image}" # libero, libero_spatial, libero_object
STEPS="${STEPS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-8}"   # conservative default for ~8GB GPUs

python "${SCRIPT_DIR}/finetune_libero.py" \
    --policy-type smolvla \
    --dataset-repo-id "${DATASET_REPO_ID}" \
    --output-dir "${OUTPUT_DIR}" \
    --wandb \
    --wandb-project libero-vla \
    --steps "${STEPS}" \
    --batch-size "${BATCH_SIZE}" \
    --load-vlm-weights \
    --val-freq 2000 \
    --val-ratio 0.02 \
    --max-val-episodes 20 \
    "$@"