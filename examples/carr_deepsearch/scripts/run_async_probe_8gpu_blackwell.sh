#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
cd "$PROJECT_DIR"

PROBE_MODE="${PROBE_MODE:-all}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-$HOME/logs}"
mkdir -p "$LOG_DIR"

ULIMIT_NOFILE="${ULIMIT_NOFILE:-65535}"
ulimit -n "$ULIMIT_NOFILE" || true

export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export VERL_USE_EXTERNAL_MODULES=examples.carr_deepsearch.tools.carr_agent_loop,examples.carr_deepsearch.reward.cgrpo_advantage

SFT_PROBE_MODEL_PATH="${SFT_PROBE_MODEL_PATH:-${SFT_MODEL_PATH:-/root/sft_checkpoint}}"

if [ -n "${RL_PROBE_MODEL_PATH:-}" ]; then
  :
elif [ -d /root/checkpoints/carr_8gpu_formal/global_step_90/actor/huggingface ]; then
  RL_PROBE_MODEL_PATH=/root/checkpoints/carr_8gpu_formal/global_step_90/actor/huggingface
elif [ -d /root/CaRR_90step_checkpoint/actor/huggingface_merged ]; then
  RL_PROBE_MODEL_PATH=/root/CaRR_90step_checkpoint/actor/huggingface_merged
elif [ -d /root/CaRR_90step_checkpoint/actor/huggingface ]; then
  RL_PROBE_MODEL_PATH=/root/CaRR_90step_checkpoint/actor/huggingface
else
  RL_PROBE_MODEL_PATH=/root/checkpoints/carr_8gpu_formal/global_step_90/actor/huggingface
fi

ATTENTION_BACKEND_INITIAL="${ATTENTION_BACKEND_INITIAL:-flashinfer}"
BACKEND_FALLBACK_ENABLED="${BACKEND_FALLBACK_ENABLED:-1}"
RUN_FOLLOWUP64K="${RUN_FOLLOWUP64K:-1}"

TRAIN_GPUS="${TRAIN_GPUS:-4}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:-4}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-128}"
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-1}"
ROLLOUT_ENFORCE_EAGER="${ROLLOUT_ENFORCE_EAGER:-true}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.3}"
ASYNC_MAX_CONCURRENT_SAMPLES="${ASYNC_MAX_CONCURRENT_SAMPLES:-}"
ASYNC_MAX_QUEUE_SIZE="${ASYNC_MAX_QUEUE_SIZE:-}"
DEFAULT_MAX_PROMPT_LENGTH="${DEFAULT_MAX_PROMPT_LENGTH:-4096}"

RUN_SCRIPT="$PROJECT_DIR/examples/carr_deepsearch/scripts/run_rl_async.sh"
SELECTED_ATTENTION_BACKEND=""

require_dir() {
  local path="$1"
  local label="$2"
  if [ ! -d "$path" ]; then
    echo "Missing ${label}: $path" >&2
    exit 1
  fi
}

resolve_ppo_max_token_len() {
  local explicit_value="$1"
  local max_response_length="$2"

  if [ -n "$explicit_value" ]; then
    printf '%s\n' "$explicit_value"
    return
  fi

  if [ -n "$max_response_length" ]; then
    printf '%s\n' "$((max_response_length + DEFAULT_MAX_PROMPT_LENGTH))"
    return
  fi

  printf '\n'
}

run_preflight() {
  echo "===== PRECHECK $(date -Iseconds) ====="
  echo "PROJECT_DIR=$PROJECT_DIR"
  echo "GIT_COMMIT=$(git -C "$PROJECT_DIR" rev-parse HEAD)"
  echo "NPROC=$(nproc)"
  echo "RAY_NUM_CPUS=$RAY_NUM_CPUS"
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
  python - <<'PY'
import sys
import torch
import sglang

print("PYTHON", sys.version.split()[0])
print("TORCH", torch.__version__)
print("SGLANG", sglang.__version__)
print("GPU", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
try:
    import flashinfer
    print("FLASHINFER", flashinfer.__version__)
except Exception as exc:
    print("FLASHINFER_MISSING", repr(exc))
PY
}

run_phase() {
  local profile="$1"
  local phase_name="$2"
  local model_path="$3"
  local attention_backend="$4"
  local total_steps="$5"
  local max_response_length="$6"
  local ppo_max_token_len="$7"
  local max_assistant_turns="$8"
  local max_tool_response_length="$9"

  require_dir "$model_path" "model checkpoint"

  local log_path="$LOG_DIR/${phase_name}.log"
  echo "===== RUN ${phase_name} ====="
  echo "LOG_PATH=$log_path"

  (
    set -x
    env \
      ASYNC_PROFILE="$profile" \
      PHASE_NAME="$phase_name" \
      SFT_MODEL_PATH="$model_path" \
      TRAIN_GPUS="$TRAIN_GPUS" \
      ROLLOUT_GPUS="$ROLLOUT_GPUS" \
      TRAINER_RESUME_MODE=disable \
      TRAINER_VAL_BEFORE_TRAIN=false \
      RAY_NUM_CPUS="$RAY_NUM_CPUS" \
      ROLLOUT_TP_SIZE="$ROLLOUT_TP_SIZE" \
      ROLLOUT_ENFORCE_EAGER="$ROLLOUT_ENFORCE_EAGER" \
      ROLLOUT_GPU_MEMORY_UTILIZATION="$ROLLOUT_GPU_MEMORY_UTILIZATION" \
      ASYNC_MAX_CONCURRENT_SAMPLES="$ASYNC_MAX_CONCURRENT_SAMPLES" \
      ASYNC_MAX_QUEUE_SIZE="$ASYNC_MAX_QUEUE_SIZE" \
      ATTENTION_BACKEND="$attention_backend" \
      ROLLOUT_TOTAL_STEPS="$total_steps" \
      MAX_RESPONSE_LENGTH="$max_response_length" \
      PPO_MAX_TOKEN_LEN_PER_GPU="$ppo_max_token_len" \
      MAX_ASSISTANT_TURNS="$max_assistant_turns" \
      MAX_TOOL_RESPONSE_LENGTH="$max_tool_response_length" \
      ULIMIT_NOFILE="$ULIMIT_NOFILE" \
      bash "$RUN_SCRIPT"
  ) 2>&1 | tee "$log_path"
}

run_base_with_fallback() {
  local phase_name="phase0_async_base_8gpu_bw_tp1_${TIMESTAMP}"
  if run_phase base "$phase_name" "$SFT_PROBE_MODEL_PATH" "$ATTENTION_BACKEND_INITIAL" 32 "" "" "" ""; then
    SELECTED_ATTENTION_BACKEND="$ATTENTION_BACKEND_INITIAL"
    return 0
  fi

  if [ "$BACKEND_FALLBACK_ENABLED" != "1" ]; then
    return 1
  fi

  local fallback_phase_name="phase0_async_base_8gpu_bw_tp1_auto_${TIMESTAMP}"
  if run_phase base "$fallback_phase_name" "$SFT_PROBE_MODEL_PATH" auto 32 "" "" "" ""; then
    SELECTED_ATTENTION_BACKEND="auto"
    return 0
  fi

  return 1
}

run_sync_stream() {
  local total_steps="${SYNC_STREAM_TOTAL_STEPS:-24}"
  local max_response_length="${SYNC_STREAM_MAX_RESPONSE_LENGTH:-16384}"
  local ppo_max_token_len="${SYNC_STREAM_PPO_MAX_TOKEN_LEN_PER_GPU:-}"
  local max_assistant_turns="${SYNC_STREAM_MAX_ASSISTANT_TURNS:-10}"
  local max_tool_response_length="${SYNC_STREAM_MAX_TOOL_RESPONSE_LENGTH:-4000}"

  run_phase \
    sync_stream \
    "phase1_sync_stream_8gpu_bw_tp1_${TIMESTAMP}" \
    "$RL_PROBE_MODEL_PATH" \
    "$SELECTED_ATTENTION_BACKEND" \
    "$total_steps" \
    "$max_response_length" \
    "$(resolve_ppo_max_token_len "$ppo_max_token_len" "$max_response_length")" \
    "$max_assistant_turns" \
    "$max_tool_response_length"
}

run_async_partial() {
  local total_steps="${ASYNC_PARTIAL_TOTAL_STEPS:-24}"
  local max_response_length="${ASYNC_PARTIAL_MAX_RESPONSE_LENGTH:-16384}"
  local ppo_max_token_len="${ASYNC_PARTIAL_PPO_MAX_TOKEN_LEN_PER_GPU:-}"
  local max_assistant_turns="${ASYNC_PARTIAL_MAX_ASSISTANT_TURNS:-10}"
  local max_tool_response_length="${ASYNC_PARTIAL_MAX_TOOL_RESPONSE_LENGTH:-4000}"

  run_phase \
    async_partial \
    "phase2_async_partial_8gpu_bw_tp1_${TIMESTAMP}" \
    "$RL_PROBE_MODEL_PATH" \
    "$SELECTED_ATTENTION_BACKEND" \
    "$total_steps" \
    "$max_response_length" \
    "$(resolve_ppo_max_token_len "$ppo_max_token_len" "$max_response_length")" \
    "$max_assistant_turns" \
    "$max_tool_response_length"
}

run_followup_64k() {
  local total_steps="${FOLLOWUP64K_TOTAL_STEPS:-16}"
  local max_response_length="${FOLLOWUP64K_MAX_RESPONSE_LENGTH:-61440}"
  local ppo_max_token_len="${FOLLOWUP64K_PPO_MAX_TOKEN_LEN_PER_GPU:-}"
  local max_assistant_turns="${FOLLOWUP64K_MAX_ASSISTANT_TURNS:-30}"
  local max_tool_response_length="${FOLLOWUP64K_MAX_TOOL_RESPONSE_LENGTH:-10000}"

  run_phase \
    async_partial \
    "phase3_async_followup_64k_8gpu_bw_tp1_${TIMESTAMP}" \
    "$RL_PROBE_MODEL_PATH" \
    "$SELECTED_ATTENTION_BACKEND" \
    "$total_steps" \
    "$max_response_length" \
    "$(resolve_ppo_max_token_len "$ppo_max_token_len" "$max_response_length")" \
    "$max_assistant_turns" \
    "$max_tool_response_length"
}

case "$PROBE_MODE" in
  preflight)
    run_preflight
    ;;
  base)
    run_preflight
    run_base_with_fallback
    ;;
  sync_stream)
    run_preflight
    SELECTED_ATTENTION_BACKEND="${ATTENTION_BACKEND_OVERRIDE:-$ATTENTION_BACKEND_INITIAL}"
    run_sync_stream
    ;;
  async_partial)
    run_preflight
    SELECTED_ATTENTION_BACKEND="${ATTENTION_BACKEND_OVERRIDE:-$ATTENTION_BACKEND_INITIAL}"
    run_async_partial
    ;;
  followup64k)
    run_preflight
    SELECTED_ATTENTION_BACKEND="${ATTENTION_BACKEND_OVERRIDE:-$ATTENTION_BACKEND_INITIAL}"
    run_followup_64k
    ;;
  all)
    run_preflight
    run_base_with_fallback
    run_sync_stream
    run_async_partial
    if [ "$RUN_FOLLOWUP64K" = "1" ]; then
      run_followup_64k
    fi
    ;;
  *)
    echo "Unknown PROBE_MODE=$PROBE_MODE" >&2
    exit 1
    ;;
esac
