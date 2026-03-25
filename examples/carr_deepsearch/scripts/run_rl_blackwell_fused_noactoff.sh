#!/bin/bash
# Wrapper for the current best Blackwell RL recipe:
# - 4 x RTX PRO 6000 96GB
# - fused kernels (torch backend)
# - activation offload disabled
# - actor/ref logprob token packing increased
#
# Usage:
#   bash run_rl_blackwell_fused_noactoff.sh [model_path] [extra hydra overrides...]
#
# Behavior:
# - Each run gets its own log directory under $LOG_ROOT/$RUN_NAME
# - ENABLE_ENTROPY_LOG=1 adds actor.calculate_entropy=true while keeping entropy_coeff=0
#   so entropy is logged from old_log_prob without turning on entropy regularization.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
SCRIPT_PATH="$PROJECT_DIR/examples/carr_deepsearch/scripts/run_rl_integration.sh"

MODEL_PATH_DEFAULT="${MODEL_PATH:-/workspace/hf_sft_591step}"
if [[ $# -gt 0 && "$1" != *=* ]]; then
  MODEL_PATH="$1"
  shift
else
  MODEL_PATH="$MODEL_PATH_DEFAULT"
fi

RUN_TAG="${RUN_TAG:-blackwell_fused_noactoff_b4n4_actor280k_log360k}"
TIMESTAMP="${TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_NAME="${RUN_NAME:-${RUN_TAG}_${TIMESTAMP}}"

LOG_ROOT="${LOG_ROOT:-$HOME/logs/carr_blackwell_fused_noactoff}"
LOG_DIR="${LOG_DIR:-$LOG_ROOT/$RUN_NAME}"
OUT_DIR="${OUT_DIR:-$HOME/checkpoints/carr_deepsearch_rl}"
mkdir -p "$LOG_DIR" "$OUT_DIR"

export PROJECT_DIR
export LOG_DIR
export OUT_DIR
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export CARR_SGLANG_ATTENTION_BACKEND="${CARR_SGLANG_ATTENTION_BACKEND:-flashinfer}"

OVERRIDES=(
  "trainer.n_gpus_per_node=${TRAINER_N_GPUS_PER_NODE:-4}"
  "trainer.total_training_steps=${TOTAL_STEPS:-1}"
  "trainer.val_before_train=${VAL_BEFORE_TRAIN:-false}"
  "trainer.test_freq=${TEST_FREQ:-0}"
  "trainer.save_freq=${SAVE_FREQ:-0}"
  "trainer.logger=${TRAINER_LOGGER:-[console]}"
  "data.train_batch_size=${TRAIN_BATCH_SIZE:-4}"
  "actor_rollout_ref.rollout.n=${ROLLOUT_N:-4}"
  "actor_rollout_ref.rollout.tensor_model_parallel_size=${TENSOR_PARALLEL_SIZE:-1}"
  "actor_rollout_ref.rollout.enforce_eager=${ENFORCE_EAGER:-true}"
  "actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${MAX_ASSISTANT_TURNS:-120}"
  "actor_rollout_ref.rollout.multi_turn.max_tool_response_length=${MAX_TOOL_RESPONSE_LENGTH:-6000}"
  "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-4}"
  "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
  "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU:-1}"
  "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU:-1}"
  "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ROLLOUT_LOGPROB_MAX_TOKEN_LEN_PER_GPU:-360000}"
  "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${REF_LOGPROB_MAX_TOKEN_LEN_PER_GPU:-360000}"
  "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU:-280000}"
  "actor_rollout_ref.nccl_timeout=${NCCL_TIMEOUT:-1800}"
  "actor_rollout_ref.model.use_fused_kernels=true"
  "actor_rollout_ref.model.fused_kernel_options.impl_backend=${FUSED_IMPL_BACKEND:-torch}"
  "actor_rollout_ref.model.enable_activation_offload=${ENABLE_ACTIVATION_OFFLOAD:-false}"
  "actor_rollout_ref.actor.ulysses_sequence_parallel_size=${ULYSSES_SEQUENCE_PARALLEL_SIZE:-1}"
  "+actor_rollout_ref.rollout.custom.carr_budget.max_rollout_wall_time_s=${MAX_ROLLOUT_WALL_TIME_S:-240}"
  "+actor_rollout_ref.rollout.custom.carr_budget.max_tool_calls=${MAX_TOOL_CALLS:-72}"
  "+actor_rollout_ref.rollout.custom.carr_budget.max_search_calls=${MAX_SEARCH_CALLS:-32}"
  "+actor_rollout_ref.rollout.custom.carr_budget.max_open_calls=${MAX_OPEN_CALLS:-32}"
  "+actor_rollout_ref.rollout.custom.carr_budget.max_find_calls=${MAX_FIND_CALLS:-16}"
)

if [[ "${ENABLE_ENTROPY_LOG:-0}" == "1" ]]; then
  OVERRIDES+=(
    "actor_rollout_ref.actor.calculate_entropy=true"
    "actor_rollout_ref.actor.entropy_coeff=0"
  )
fi

echo "RUN_NAME=$RUN_NAME"
echo "LOG_DIR=$LOG_DIR"

bash "$SCRIPT_PATH" "$MODEL_PATH" "$RUN_NAME" "${OVERRIDES[@]}" "$@"
