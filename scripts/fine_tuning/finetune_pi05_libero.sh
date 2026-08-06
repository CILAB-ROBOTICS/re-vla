#!/usr/bin/env bash
# Fine-tune Pi0.5 on LIBERO, continuing from the LIBERO-pretrained base checkpoint
# (lerobot/pi05_libero_base), matching lerobot/docs/source/pi05.mdx.
#
# NOTE: Pi0.5 is a ~3B-parameter VLA. The reference full fine-tune recipe is "sized for
# a single 80GB GPU"; even the frozen-VLM variant used here (--freeze-vision-encoder,
# train_expert_only) is unlikely to fit on an 8GB consumer GPU. Use a larger/cloud GPU,
# or drop --batch-size further / rely on gradient_checkpointing (already on by default).
#
# Any extra args are passed through, e.g.:
#   ./finetune_pi05_libero.sh --steps 6000 --batch-size 4
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

OUTPUT_DIR="${OUTPUT_DIR:-./outputs/libero_pi05}"
DATASET_REPO_ID="${DATASET_REPO_ID:-lerobot/libero}"
PRETRAINED_PATH="${PRETRAINED_PATH:-lerobot/pi05_libero_base}"
STEPS="${STEPS:-6000}"          # docs' reproduction: 6k additional steps from pi05_libero_base
BATCH_SIZE="${BATCH_SIZE:-2}"   # very conservative default; raise if your GPU allows

python "${SCRIPT_DIR}/finetune_libero.py" \
    --policy-type pi05 \
    --dataset-repo-id "${DATASET_REPO_ID}" \
    --output-dir "${OUTPUT_DIR}" \
    --pretrained-path "${PRETRAINED_PATH}" \
    --steps "${STEPS}" \
    --batch-size "${BATCH_SIZE}" \
    --freeze-vision-encoder \
    "$@"
