#!/bin/bash
# Sampled DeepDive val-subset64 evaluation launcher for Thinking checkpoints.
# Reuses run_eval_integration.sh so tool/reward logs and traces are preserved.
#
# Usage:
#   bash run_eval_deepdive64_sampled.sh <model_path> <run_name> [extra hydra overrides...]
#
# Examples:
#   NGPUS=8 bash run_eval_deepdive64_sampled.sh /root/hf_sft_591step sft_step591_dd64
#   NGPUS=4 CARR_MAX_TOOL_RESPONSE_LENGTH=5000 \
#     bash run_eval_deepdive64_sampled.sh /root/checkpoints/.../actor/huggingface step90_dd64

set -euo pipefail

MODEL_PATH="${1:?Usage: run_eval_deepdive64_sampled.sh <model_path> <run_name> [extra hydra overrides...]}"
RUN_NAME="${2:?Usage: run_eval_deepdive64_sampled.sh <model_path> <run_name> [extra hydra overrides...]}"
shift 2

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
DATA_DIR="${PROJECT_DIR}/examples/carr_deepsearch/data"
SUBSET_SIZE="${CARR_EVAL_SUBSET_SIZE:-64}"
SUBSET_SEED="${CARR_EVAL_SUBSET_SEED:-42}"
EVAL_FILE="${DATA_DIR}/rl_val_subset_${SUBSET_SIZE}_seed${SUBSET_SEED}.parquet"
NGPUS="${NGPUS:-8}"

if [ ! -f "$EVAL_FILE" ]; then
  python "$PROJECT_DIR/examples/carr_deepsearch/scripts/prepare_rl_val_subset.py" \
    --input "$DATA_DIR/rl_val.parquet" \
    --subset-size "$SUBSET_SIZE" \
    --seed "$SUBSET_SEED"
fi

case "$NGPUS" in
  8)
    : "${CARR_VAL_BATCH_SIZE:=32}"
    ;;
  4)
    : "${CARR_VAL_BATCH_SIZE:=16}"
    ;;
  *)
    echo "ERROR: unsupported NGPUS=$NGPUS (expected 4 or 8)" >&2
    exit 1
    ;;
esac

# Recommended sampled validation recipe for Qwen Thinking checkpoints.
: "${CARR_VAL_N:=1}"
: "${CARR_VAL_TEMPERATURE:=0.6}"
: "${CARR_VAL_TOP_P:=0.95}"
: "${CARR_VAL_TOP_K:=20}"
: "${CARR_VAL_DO_SAMPLE:=true}"
: "${CARR_VAL_REPETITION_PENALTY:=1.0}"
: "${CARR_VAL_PRESENCE_PENALTY:=0.0}"
: "${CARR_VAL_FREQUENCY_PENALTY:=0.0}"

# Formal DeepDive eval envelope: 64k + turn/tool budgets.
: "${CARR_MAX_RESPONSE_LENGTH:=61440}"
: "${CARR_MAX_ASSISTANT_TURNS:=120}"
: "${CARR_MAX_TOOL_RESPONSE_LENGTH:=6000}"
: "${CARR_ROLLOUT_WALL_TIME_S:=360}"
: "${CARR_MAX_TOOL_CALLS:=88}"
: "${CARR_MAX_SEARCH_CALLS:=40}"
: "${CARR_MAX_OPEN_CALLS:=32}"
: "${CARR_MAX_FIND_CALLS:=20}"
: "${CARR_TP_SIZE:=1}"
: "${CARR_SGLANG_ATTENTION_BACKEND:=flashinfer}"

OUT_DIR="${OUT_DIR:-$HOME/eval_results/$RUN_NAME}"

# run_eval_integration.sh reads these knobs from the environment, so export the
# wrapper-resolved values explicitly instead of relying on shell locals.
export NGPUS
export CARR_VAL_BATCH_SIZE
export CARR_VAL_N
export CARR_VAL_TEMPERATURE
export CARR_VAL_TOP_P
export CARR_VAL_TOP_K
export CARR_VAL_DO_SAMPLE
export CARR_VAL_REPETITION_PENALTY
export CARR_VAL_PRESENCE_PENALTY
export CARR_VAL_FREQUENCY_PENALTY
export CARR_MAX_RESPONSE_LENGTH
export CARR_MAX_ASSISTANT_TURNS
export CARR_MAX_TOOL_RESPONSE_LENGTH
export CARR_ROLLOUT_WALL_TIME_S
export CARR_MAX_TOOL_CALLS
export CARR_MAX_SEARCH_CALLS
export CARR_MAX_OPEN_CALLS
export CARR_MAX_FIND_CALLS
export CARR_TP_SIZE
export CARR_SGLANG_ATTENTION_BACKEND
export OUT_DIR

bash "$PROJECT_DIR/examples/carr_deepsearch/scripts/run_eval_integration.sh" \
  "$MODEL_PATH" \
  "$EVAL_FILE" \
  "$RUN_NAME" \
  data.validation_shuffle=false \
  data.val_max_samples="$SUBSET_SIZE" \
  trainer.validation_data_dir="$OUT_DIR" \
  "$@"
