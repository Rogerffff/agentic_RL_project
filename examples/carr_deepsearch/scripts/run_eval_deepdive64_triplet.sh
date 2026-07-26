#!/bin/bash
# Evaluate SFT / step70 / step90 checkpoints on the same sampled DeepDive val subset64.
#
# Usage:
#   NGPUS=8 bash run_eval_deepdive64_triplet.sh <sft_ckpt> <step70_ckpt> <step90_ckpt> <run_prefix>

set -euo pipefail

SFT_CKPT="${1:?Usage: run_eval_deepdive64_triplet.sh <sft_ckpt> <step70_ckpt> <step90_ckpt> <run_prefix>}"
STEP70_CKPT="${2:?Usage: run_eval_deepdive64_triplet.sh <sft_ckpt> <step70_ckpt> <step90_ckpt> <run_prefix>}"
STEP90_CKPT="${3:?Usage: run_eval_deepdive64_triplet.sh <sft_ckpt> <step70_ckpt> <step90_ckpt> <run_prefix>}"
RUN_PREFIX="${4:?Usage: run_eval_deepdive64_triplet.sh <sft_ckpt> <step70_ckpt> <step90_ckpt> <run_prefix>}"

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"

bash "$PROJECT_DIR/examples/carr_deepsearch/scripts/run_eval_deepdive64_sampled.sh" \
  "$SFT_CKPT" "${RUN_PREFIX}_sft"

bash "$PROJECT_DIR/examples/carr_deepsearch/scripts/run_eval_deepdive64_sampled.sh" \
  "$STEP70_CKPT" "${RUN_PREFIX}_step70"

bash "$PROJECT_DIR/examples/carr_deepsearch/scripts/run_eval_deepdive64_sampled.sh" \
  "$STEP90_CKPT" "${RUN_PREFIX}_step90"
