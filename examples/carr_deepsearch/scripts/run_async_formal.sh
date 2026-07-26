#!/bin/bash
# Formal async RL launcher.
#
# Modes:
#   eval_gate
#   gate_b4_n8_64k
#   short_b8_n8_64k
#   validate_b4_n8_64k_2x6_dynbsz
#   formal_best_1day
#   stretch_b16_n8_64k
#   fallback_b8_n4_64k

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
cd "$PROJECT_DIR"

RUN_MODE="${1:-}"
if [ -z "$RUN_MODE" ]; then
  echo "Usage: bash run_async_formal.sh <mode> [extra hydra overrides...]" >&2
  exit 1
fi
shift || true

die() {
  echo "ERROR: $*" >&2
  exit 1
}

warn() {
  echo "WARN: $*" >&2
}

expand_home() {
  local path="$1"
  if [ -z "$path" ]; then
    return 0
  fi
  if [[ "$path" == "~"* ]]; then
    path="${HOME}${path:1}"
  fi
  printf '%s\n' "$path"
}

is_hf_model_dir() {
  local dir="$1"
  [ -d "$dir" ] || return 1
  [ -f "$dir/config.json" ] || return 1
  find "$dir" -maxdepth 1 \
    \( -name "*.safetensors" -o -name "model.safetensors.index.json" -o -name "pytorch_model.bin" -o -name "pytorch_model*.bin" \) \
    -print -quit | grep -q .
}

resolve_model_dir() {
  local raw
  raw="$(expand_home "$1")"
  [ -n "$raw" ] || return 1

  local candidates=(
    "$raw"
    "$raw/huggingface_merged"
    "$raw/huggingface"
    "$raw/actor/huggingface_merged"
    "$raw/actor/huggingface"
  )

  local candidate
  for candidate in "${candidates[@]}"; do
    if is_hf_model_dir "$candidate"; then
      (
        cd "$candidate"
        pwd
      )
      return 0
    fi
  done
  return 1
}

make_phase_paths() {
  local mode="$1"
  local timestamp
  timestamp="$(date +%Y%m%d_%H%M%S)"
  PHASE_NAME="${PHASE_NAME:-${mode}_${timestamp}}"
  TRAINER_DEFAULT_LOCAL_DIR="${TRAINER_DEFAULT_LOCAL_DIR:-$HOME/checkpoints/$PHASE_NAME}"
  TRAINER_VALIDATION_DATA_DIR="${TRAINER_VALIDATION_DATA_DIR:-$HOME/eval_results/$PHASE_NAME}"
  TRAINER_EXPERIMENT_NAME="${TRAINER_EXPERIMENT_NAME:-$PHASE_NAME}"
  export PHASE_NAME TRAINER_DEFAULT_LOCAL_DIR TRAINER_VALIDATION_DATA_DIR TRAINER_EXPERIMENT_NAME
}

run_eval_gate() {
  make_phase_paths "$RUN_MODE"

  local sft_model_path step90_model_path step70_model_path
  sft_model_path="$(resolve_model_dir "${SFT_MODEL_PATH:-}")" || die "SFT_MODEL_PATH must point to a HF checkpoint (or a checkpoint root containing actor/huggingface[_merged])."
  step90_model_path="$(resolve_model_dir "${STEP90_MODEL_PATH:-}")" || die "STEP90_MODEL_PATH must point to a HF checkpoint (or a checkpoint root containing actor/huggingface[_merged])."

  step70_model_path=""
  if [ -n "${STEP70_MODEL_PATH:-}" ]; then
    if ! step70_model_path="$(resolve_model_dir "$STEP70_MODEL_PATH")"; then
      warn "STEP70_MODEL_PATH does not resolve to a HF checkpoint; skipping step70 in eval_gate."
      step70_model_path=""
    fi
  fi

  local eval_gpus
  eval_gpus="${FORMAL_EVAL_GPUS:-8}"

  local gate_root
  gate_root="${TRAINER_VALIDATION_DATA_DIR}"
  mkdir -p "$gate_root"

  local extra_eval_args=("$@")

  export CARR_MAX_RESPONSE_LENGTH=61440
  export CARR_MAX_ASSISTANT_TURNS=120
  export CARR_MAX_TOOL_RESPONSE_LENGTH=6000
  export CARR_ROLLOUT_WALL_TIME_S=360
  export CARR_MAX_TOOL_CALLS=88
  export CARR_MAX_SEARCH_CALLS=40
  export CARR_MAX_OPEN_CALLS=32
  export CARR_MAX_FIND_CALLS=20
  export CARR_VAL_N=1
  export CARR_VAL_TEMPERATURE=0.6
  export CARR_VAL_TOP_P=0.95
  export CARR_VAL_TOP_K=20
  export CARR_VAL_DO_SAMPLE=true
  export CARR_SGLANG_ATTENTION_BACKEND=flashinfer

  OUT_DIR="$gate_root/sft" \
  NGPUS="$eval_gpus" \
  bash "$PROJECT_DIR/examples/carr_deepsearch/scripts/run_eval_deepdive64_sampled.sh" \
    "$sft_model_path" "${PHASE_NAME}_sft" "${extra_eval_args[@]}"

  if [ -n "$step70_model_path" ]; then
    OUT_DIR="$gate_root/step70" \
    NGPUS="$eval_gpus" \
    bash "$PROJECT_DIR/examples/carr_deepsearch/scripts/run_eval_deepdive64_sampled.sh" \
      "$step70_model_path" "${PHASE_NAME}_step70" "${extra_eval_args[@]}"
  fi

  OUT_DIR="$gate_root/step90" \
  NGPUS="$eval_gpus" \
  bash "$PROJECT_DIR/examples/carr_deepsearch/scripts/run_eval_deepdive64_sampled.sh" \
    "$step90_model_path" "${PHASE_NAME}_step90" "${extra_eval_args[@]}"

  local selector_args=(
    --candidate "sft::${sft_model_path}::${gate_root}/sft"
    --candidate "step90::${step90_model_path}::${gate_root}/step90"
    --output-json "${gate_root}/selection_summary.json"
    --selected-path-out "${gate_root}/selected_async_start_model.txt"
    --export-sh "${gate_root}/export_async_start.env"
  )
  if [ -n "$step70_model_path" ]; then
    selector_args+=(--candidate "step70::${step70_model_path}::${gate_root}/step70")
  fi

  python3 "$PROJECT_DIR/examples/carr_deepsearch/scripts/select_async_start_checkpoint.py" \
    "${selector_args[@]}"

  echo "Eval gate outputs:" >&2
  echo "  summary: ${gate_root}/selection_summary.json" >&2
  echo "  selected model path: ${gate_root}/selected_async_start_model.txt" >&2
  echo "  export snippet: ${gate_root}/export_async_start.env" >&2
}

resolve_async_start_model() {
  local start_model_path
  if [ -n "${ASYNC_START_MODEL_PATH:-}" ]; then
    start_model_path="$ASYNC_START_MODEL_PATH"
  elif [ -n "${ASYNC_START_MODEL_FILE:-}" ] && [ -f "$(expand_home "$ASYNC_START_MODEL_FILE")" ]; then
    start_model_path="$(tr -d '\n' < "$(expand_home "$ASYNC_START_MODEL_FILE")")"
  else
    die "Set ASYNC_START_MODEL_PATH or ASYNC_START_MODEL_FILE before running a training mode."
  fi

  resolve_model_dir "$start_model_path" || die "Failed to resolve ASYNC_START_MODEL_PATH to a HF checkpoint directory."
}

TRAIN_EXTRA_OVERRIDES=()

apply_budget_variant() {
  local variant="$1"
  case "$variant" in
    default)
      ;;
    wall480)
      TRAIN_EXTRA_OVERRIDES+=(
        actor_rollout_ref.rollout.custom.carr_budget.max_rollout_wall_time_s=480
        actor_rollout_ref.rollout.custom.carr_budget.max_real_rollout_wall_time_s=960
      )
      ;;
    tool5000)
      export MAX_TOOL_RESPONSE_LENGTH=5000
      ;;
    *)
      die "Unsupported FORMAL_VARIANT=${variant}. Use default, wall480, or tool5000."
      ;;
  esac
}

run_train_mode() {
  local mode="$1"
  local ppo_mini_batch_size="$2"
  local rollout_n="$3"
  local staleness="$4"
  local max_concurrent="$5"
  local max_queue="$6"
  local total_rollout_steps="$7"
  local save_freq="$8"
  shift 8
  local extra_overrides=("$@")

  make_phase_paths "$mode"

  local start_model_path
  start_model_path="$(resolve_async_start_model)"

  export SFT_MODEL_PATH="$start_model_path"
  export TRAIN_GPUS="${FORMAL_TRAIN_GPUS:-4}"
  export ROLLOUT_GPUS="${FORMAL_ROLLOUT_GPUS:-4}"
  export RAY_NUM_CPUS="${RAY_NUM_CPUS:-128}"
  export TRAINER_RESUME_MODE=disable
  # Async tool-using runs are CPU-heavy (ray workers, tool server, reward loop, request schedulers).
  # Keep a wider default than run_rl_async.sh so Stage A/B/Formal do not inherit the too-tight 32 CPU default.
  export RAY_NUM_CPUS="${FORMAL_RAY_NUM_CPUS:-${RAY_NUM_CPUS:-128}}"
  export ASYNC_PROFILE=async_partial
  export ASYNC_PARTIAL=true
  export ASYNC_TRIGGER_SYNC_STEP="${FORMAL_ASYNC_TRIGGER_SYNC_STEP:-2}"
  export ASYNC_REQUIRE_BATCHES=1
  export ASYNC_STALENESS="$staleness"
  export ASYNC_MAX_CONCURRENT_SAMPLES="$max_concurrent"
  export ASYNC_MAX_QUEUE_SIZE="$max_queue"
  export ROLLOUT_TOTAL_STEPS="$total_rollout_steps"
  export TRAINER_SAVE_FREQ="$save_freq"
  export TRAINER_TEST_FREQ=0
  export ROLLOUT_TEST_FREQ=0
  export TRAINER_VAL_BEFORE_TRAIN=false
  export NCCL_P2P_DISABLE=1
  export ATTENTION_BACKEND=flashinfer
  export MAX_RESPONSE_LENGTH=61440
  export PPO_MAX_TOKEN_LEN_PER_GPU="${FORMAL_PPO_MAX_TOKEN_LEN_PER_GPU:-24576}"
  export MAX_ASSISTANT_TURNS=120
  export MAX_TOOL_RESPONSE_LENGTH=6000
  export FORMAL_ACTOR_USE_DYNAMIC_BSZ="${FORMAL_ACTOR_USE_DYNAMIC_BSZ:-false}"

  local budget_variant
  budget_variant="${FORMAL_VARIANT:-default}"
  TRAIN_EXTRA_OVERRIDES=("${extra_overrides[@]}")
  apply_budget_variant "$budget_variant"

  bash "$PROJECT_DIR/examples/carr_deepsearch/scripts/run_rl_async.sh" \
    actor_rollout_ref.actor.use_dynamic_bsz="$FORMAL_ACTOR_USE_DYNAMIC_BSZ" \
    actor_rollout_ref.actor.ppo_mini_batch_size="$ppo_mini_batch_size" \
    actor_rollout_ref.rollout.n="$rollout_n" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.custom.carr_budget.max_rollout_wall_time_s=360 \
    actor_rollout_ref.rollout.custom.carr_budget.max_real_rollout_wall_time_s=720 \
    actor_rollout_ref.rollout.custom.carr_budget.max_tool_calls=88 \
    actor_rollout_ref.rollout.custom.carr_budget.max_search_calls=40 \
    actor_rollout_ref.rollout.custom.carr_budget.max_open_calls=32 \
    actor_rollout_ref.rollout.custom.carr_budget.max_find_calls=20 \
    trainer.save_freq="$save_freq" \
    "${TRAIN_EXTRA_OVERRIDES[@]}"
}

run_validate_b4_n8_64k_2x6_dynbsz() {
  export FORMAL_TRAIN_GPUS=2
  export FORMAL_ROLLOUT_GPUS=6
  export FORMAL_ASYNC_TRIGGER_SYNC_STEP=4
  export FORMAL_ACTOR_USE_DYNAMIC_BSZ=true
  export FORMAL_PPO_MAX_TOKEN_LEN_PER_GPU="${FORMAL_PPO_MAX_TOKEN_LEN_PER_GPU:-24576}"

  run_train_mode \
    "$RUN_MODE" \
    4 \
    8 \
    0.5 \
    6 \
    12 \
    48 \
    1 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=true \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true \
    critic.use_dynamic_bsz=true \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
    critic.ulysses_sequence_parallel_size=2 \
    actor_rollout_ref.model.enable_activation_offload=true \
    critic.model.enable_activation_offload=true \
    critic.model.use_remove_padding=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    critic.model.fsdp_config.optimizer_offload=false \
    "$@"
}

compute_formal_rollout_steps() {
  local seconds_per_rollout_step="$1"
  python3 - "$seconds_per_rollout_step" <<'PY'
import math
import sys

seconds_per_rollout_step = float(sys.argv[1])
if seconds_per_rollout_step <= 0:
    raise SystemExit("seconds_per_rollout_step must be > 0")
steps = int(math.floor(0.8 * 86400 / seconds_per_rollout_step))
steps = max(32, steps - (steps % 32))
print(steps)
PY
}

compute_formal_save_freq() {
  local rollout_steps="$1"
  local ppo_mini_batch_size="$2"
  local require_batches="$3"
  local trigger_sync_step="$4"
  python3 - "$rollout_steps" "$ppo_mini_batch_size" "$require_batches" "$trigger_sync_step" <<'PY'
import math
import sys

rollout_steps = int(sys.argv[1])
ppo_mini_batch_size = int(sys.argv[2])
require_batches = int(sys.argv[3])
trigger_sync_step = int(sys.argv[4])
param_versions = math.floor(rollout_steps / (ppo_mini_batch_size * require_batches * trigger_sync_step))
save_freq = max(2, math.floor(param_versions / 4))
print(save_freq)
PY
}

run_formal_best_1day() {
  local source_mode
  source_mode="${FORMAL_SOURCE_MODE:-}"
  [ -n "$source_mode" ] || die "formal_best_1day requires FORMAL_SOURCE_MODE=gate_b4_n8_64k|short_b8_n8_64k|fallback_b8_n4_64k"

  local ppo_mini_batch_size rollout_n staleness max_concurrent max_queue reference_steps
  case "$source_mode" in
    gate_b4_n8_64k)
      ppo_mini_batch_size=4
      rollout_n=8
      staleness=0.5
      max_concurrent=4
      max_queue=12
      reference_steps=48
      ;;
    short_b8_n8_64k)
      ppo_mini_batch_size=8
      rollout_n=8
      staleness=0.5
      max_concurrent=16
      max_queue=16
      reference_steps=128
      ;;
    fallback_b8_n4_64k)
      ppo_mini_batch_size=8
      rollout_n=4
      staleness=0.5
      max_concurrent=16
      max_queue=16
      reference_steps=128
      ;;
    *)
      die "Unsupported FORMAL_SOURCE_MODE=${source_mode}"
      ;;
  esac

  if [ -n "${FORMAL_SOURCE_STALENESS:-}" ]; then
    staleness="${FORMAL_SOURCE_STALENESS}"
  fi

  local seconds_per_rollout_step
  if [ -n "${FORMAL_SECONDS_PER_ROLLOUT_STEP:-}" ]; then
    seconds_per_rollout_step="${FORMAL_SECONDS_PER_ROLLOUT_STEP}"
  elif [ -n "${FORMAL_REFERENCE_TOTAL_WALL_TIME_S:-}" ]; then
    seconds_per_rollout_step="$(python3 - "${FORMAL_REFERENCE_TOTAL_WALL_TIME_S}" "${reference_steps}" <<'PY'
import sys

total_wall_time = float(sys.argv[1])
reference_steps = int(sys.argv[2])
print(total_wall_time / reference_steps)
PY
)"
  else
    die "Set FORMAL_SECONDS_PER_ROLLOUT_STEP or FORMAL_REFERENCE_TOTAL_WALL_TIME_S for formal_best_1day."
  fi

  local total_rollout_steps
  if [ -n "${FORMAL_ROLLOUT_STEPS:-}" ]; then
    total_rollout_steps="${FORMAL_ROLLOUT_STEPS}"
  else
    total_rollout_steps="$(compute_formal_rollout_steps "$seconds_per_rollout_step")"
  fi

  local save_freq
  save_freq="$(compute_formal_save_freq "$total_rollout_steps" "$ppo_mini_batch_size" 1 2)"

  echo "Formal Day-1 derived config:" >&2
  echo "  source_mode=$source_mode" >&2
  echo "  ppo_mini_batch_size=$ppo_mini_batch_size" >&2
  echo "  rollout.n=$rollout_n" >&2
  echo "  staleness=$staleness" >&2
  echo "  max_concurrent=$max_concurrent" >&2
  echo "  max_queue=$max_queue" >&2
  echo "  rollout.total_rollout_steps=$total_rollout_steps" >&2
  echo "  trainer.save_freq=$save_freq" >&2

  run_train_mode \
    "$RUN_MODE" \
    "$ppo_mini_batch_size" \
    "$rollout_n" \
    "$staleness" \
    "$max_concurrent" \
    "$max_queue" \
    "$total_rollout_steps" \
    "$save_freq" \
    "$@"
}

case "$RUN_MODE" in
  eval_gate)
    run_eval_gate "$@"
    ;;
  gate_b4_n8_64k)
    run_train_mode "$RUN_MODE" 4 8 0.5 4 12 48 1 "$@"
    ;;
  short_b8_n8_64k)
    run_train_mode "$RUN_MODE" 8 8 0.5 16 16 128 2 "$@"
    ;;
  validate_b4_n8_64k_2x6_dynbsz)
    run_validate_b4_n8_64k_2x6_dynbsz "$@"
    ;;
  formal_best_1day)
    run_formal_best_1day "$@"
    ;;
  stretch_b16_n8_64k)
    run_train_mode "$RUN_MODE" 16 8 1.0 16 16 64 1 "$@"
    ;;
  fallback_b8_n4_64k)
    run_train_mode "$RUN_MODE" 8 4 0.5 16 16 128 2 "$@"
    ;;
  *)
    die "Unknown mode ${RUN_MODE}. Expected eval_gate, gate_b4_n8_64k, short_b8_n8_64k, validate_b4_n8_64k_2x6_dynbsz, formal_best_1day, stretch_b16_n8_64k, or fallback_b8_n4_64k."
    ;;
esac
