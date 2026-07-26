#!/bin/bash
set -euxo pipefail

START_TIME=$(date +%s)

PROJECT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$PROJECT_DIR"

DATA_DIR="$PROJECT_DIR/examples/carr_deepsearch/data"
if [ ! -f "$DATA_DIR/sft_train.parquet" ]; then
    echo "Running SFT data preprocessing..."
    python examples/carr_deepsearch/data_preprocess/preprocess_carr_sft.py \
        --input_file CaRR/data/deepdive-sft-glm46-trajectory-1k.jsonl \
        --output_dir "$DATA_DIR" \
        --val_ratio 0.05 \
        --seed 42
fi

NGPUS="${NGPUS:-8}"
torchrun --nnodes=1 --nproc_per_node="$NGPUS" \
    -m verl.trainer.fsdp_sft_trainer \
    --config-path="$PROJECT_DIR/examples/carr_deepsearch/config" \
    --config-name='carr_sft' \
    "$@"

END_TIME=$(date +%s); ELAPSED=$((END_TIME - START_TIME))
echo "Completed in ${ELAPSED}s ($(( ELAPSED / 3600 ))h$(( (ELAPSED % 3600) / 60 ))m$(( ELAPSED % 60 ))s)"
