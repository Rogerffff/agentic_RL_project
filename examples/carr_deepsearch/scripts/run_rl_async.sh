#!/bin/bash
set -euxo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
cd "$PROJECT_DIR"

if [ -f "$HOME/.env" ]; then
  set -a
  . "$HOME/.env"
  set +a
fi

ASYNC_CONFIG_NAME="${ASYNC_CONFIG_NAME:-carr_grpo_async}"
ASYNC_PROFILE="${ASYNC_PROFILE:-sync_stream}"
TRAIN_GPUS="${TRAIN_GPUS:-4}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:-4}"
PHASE_NAME="${PHASE_NAME:-async-${ASYNC_PROFILE}-$(date +%Y%m%d_%H%M%S)}"
TRAINER_RESUME_MODE="${TRAINER_RESUME_MODE:-disable}"
TRAINER_VAL_BEFORE_TRAIN="${TRAINER_VAL_BEFORE_TRAIN:-false}"
TRAINER_DEFAULT_LOCAL_DIR="${TRAINER_DEFAULT_LOCAL_DIR:-$HOME/checkpoints/$PHASE_NAME}"
TRAINER_VALIDATION_DATA_DIR="${TRAINER_VALIDATION_DATA_DIR:-$HOME/eval_results/$PHASE_NAME}"
TRAINER_EXPERIMENT_NAME="${TRAINER_EXPERIMENT_NAME:-$PHASE_NAME}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-32}"
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-1}"
ROLLOUT_ENFORCE_EAGER="${ROLLOUT_ENFORCE_EAGER:-true}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.3}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-}"
ROLLOUT_TOTAL_STEPS="${ROLLOUT_TOTAL_STEPS:-}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-}"
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-}"
ASYNC_MAX_CONCURRENT_SAMPLES="${ASYNC_MAX_CONCURRENT_SAMPLES:-}"
ASYNC_MAX_QUEUE_SIZE="${ASYNC_MAX_QUEUE_SIZE:-}"
MAX_ASSISTANT_TURNS="${MAX_ASSISTANT_TURNS:-}"
MAX_TOOL_RESPONSE_LENGTH="${MAX_TOOL_RESPONSE_LENGTH:-}"
ULIMIT_NOFILE="${ULIMIT_NOFILE:-65535}"

case "${TRAIN_GPUS}:${ROLLOUT_GPUS}" in
  4:4|2:6)
    ;;
  3:5)
    if [ "${ALLOW_ASYNC_3_5:-0}" != "1" ]; then
      echo "3:5 split is disabled by default. Set ALLOW_ASYNC_3_5=1 and override actor.ppo_micro_batch_size_per_gpu=1 if you want to try it." >&2
      exit 1
    fi
    ;;
  *)
    echo "Unsupported async split ${TRAIN_GPUS}:${ROLLOUT_GPUS}. Supported defaults are 4:4 and 2:6." >&2
    exit 1
    ;;
esac

case "$ASYNC_PROFILE" in
  sync_stream)
    ASYNC_STALENESS="${ASYNC_STALENESS:-0.0}"
    ASYNC_PARTIAL="${ASYNC_PARTIAL:-false}"
    ASYNC_TRIGGER_SYNC_STEP="${ASYNC_TRIGGER_SYNC_STEP:-2}"
    ASYNC_REQUIRE_BATCHES="${ASYNC_REQUIRE_BATCHES:-1}"
    ;;
  async_partial)
    ASYNC_STALENESS="${ASYNC_STALENESS:-0.25}"
    ASYNC_PARTIAL="${ASYNC_PARTIAL:-true}"
    ASYNC_TRIGGER_SYNC_STEP="${ASYNC_TRIGGER_SYNC_STEP:-2}"
    ASYNC_REQUIRE_BATCHES="${ASYNC_REQUIRE_BATCHES:-1}"
    ;;
  base)
    if [ "$ASYNC_CONFIG_NAME" = "carr_grpo_async" ]; then
      ASYNC_CONFIG_NAME="carr_grpo_async_base"
    fi
    ASYNC_STALENESS="${ASYNC_STALENESS:-0.0}"
    ASYNC_PARTIAL="${ASYNC_PARTIAL:-false}"
    ASYNC_TRIGGER_SYNC_STEP="${ASYNC_TRIGGER_SYNC_STEP:-2}"
    ASYNC_REQUIRE_BATCHES="${ASYNC_REQUIRE_BATCHES:-1}"
    ;;
  *)
    echo "Unknown ASYNC_PROFILE=${ASYNC_PROFILE}" >&2
    exit 1
    ;;
esac

export VERL_USE_EXTERNAL_MODULES=examples.carr_deepsearch.tools.carr_agent_loop,examples.carr_deepsearch.reward.cgrpo_advantage
export RAY_enable_open_telemetry=0
export RAY_ENABLE_OPEN_TELEMETRY=0

ulimit -n "$ULIMIT_NOFILE" || true
mkdir -p "$TRAINER_DEFAULT_LOCAL_DIR" "$TRAINER_VALIDATION_DATA_DIR"

if [ -n "${MODEL_PATH:-}" ] && [ -z "${SFT_MODEL_PATH:-}" ]; then
  export SFT_MODEL_PATH="$MODEL_PATH"
fi

if [ -z "${SFT_MODEL_PATH:-}" ]; then
  SFT_CKPT_ROOT="$HOME/checkpoints/carr_deepsearch_sft"
  LATEST_STEP=$(cat "$SFT_CKPT_ROOT/latest_checkpointed_iteration.txt")
  export SFT_MODEL_PATH="$SFT_CKPT_ROOT/global_step_${LATEST_STEP}/huggingface"
fi

PIDS=()
cleanup() {
  for pid in "${PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT

clear_stale_listener() {
  local port="$1"

  if ! command -v lsof >/dev/null 2>&1; then
    return 0
  fi

  local pids
  pids="$(lsof -ti tcp:"$port" -sTCP:LISTEN 2>/dev/null || true)"
  if [ -n "$pids" ]; then
    echo "Clearing stale listener on port $port: $pids" >&2
    kill $pids 2>/dev/null || true
    sleep 1
  fi
}

if [ "$ASYNC_CONFIG_NAME" != "carr_grpo_async_base" ]; then
  if [ -z "${SERPER_API_KEY:-}" ] && [ -z "${SERPAPI_API_KEY:-}" ]; then
    echo "ERROR: Must set SERPER_API_KEY or SERPAPI_API_KEY" >&2
    exit 1
  fi
  : "${JINA_API_KEY:?Must set JINA_API_KEY}"
  : "${DEEPSEEK_API_KEY:?Must set DEEPSEEK_API_KEY}"

  export CARR_REWARD_SERVER_URL="${CARR_REWARD_SERVER_URL:-http://localhost:8888}"
  export CARR_REWARD_TIMEOUT="${CARR_REWARD_TIMEOUT:-650}"

  clear_stale_listener 7230
  clear_stale_listener 8888

  if [ -n "${SERPER_API_KEY:-}" ]; then
    SEARCH_ARGS=(--search_backend serper --serper_api_key "$SERPER_API_KEY")
  else
    SEARCH_ARGS=(--serp_api_key "$SERPAPI_API_KEY")
  fi

  python "$PROJECT_DIR/CaRR/tool_server/launch_server.py" \
    "${SEARCH_ARGS[@]}" \
    --jina_api_key "$JINA_API_KEY" \
    --port 7230 &
  PIDS+=($!)

  (
    cd "$PROJECT_DIR/CaRR/deepsearch_rm_with_rubrics"
    python launch_server.py \
      --port 8888 \
      --model_name deepseek-chat \
      --base_url https://api.deepseek.com \
      --api_key "$DEEPSEEK_API_KEY"
  ) &
  PIDS+=($!)

  tool_ready=0
  for i in {1..30}; do
    if curl -sf -X POST http://localhost:7230 \
      -H "Content-Type: application/json" \
      -d '{"session_id":"health","name":"start_session","arguments":{},"remote_env_info":{}}' >/dev/null; then
      tool_ready=1
      break
    fi
    sleep 2
  done
  if [ "$tool_ready" != "1" ]; then
    echo "Tool server failed health check" >&2
    exit 1
  fi

  reward_ready=0
  for i in {1..30}; do
    if curl -sf -X POST http://localhost:8888/evaluate \
      -H "Content-Type: application/json" \
      -d '{"history":[{"role":"user","content":"q"},{"role":"assistant","content":"a"}],"label":"a","task_unfinished":true,"remote_env_info":{"search_forbidden_strs":["q"],"rubrics":[],"rubric_reward_ratio":0.3}}' >/dev/null; then
      reward_ready=1
      break
    fi
    sleep 2
  done
  if [ "$reward_ready" != "1" ]; then
    echo "Reward server failed health check" >&2
    exit 1
  fi
fi

CMD=(
  python3 -m verl.experimental.fully_async_policy.fully_async_main
  --config-path="$PROJECT_DIR/examples/carr_deepsearch/config"
  --config-name="$ASYNC_CONFIG_NAME"
  actor_rollout_ref.model.path="$SFT_MODEL_PATH"
  data.train_files="$PROJECT_DIR/examples/carr_deepsearch/data/rl_train.parquet"
  data.val_files="$PROJECT_DIR/examples/carr_deepsearch/data/rl_val.parquet"
  actor_rollout_ref.hybrid_engine=false
  data.train_batch_size=0
  data.gen_batch_size=1
  data.return_raw_chat=true
  async_training.checkpoint_engine.enable=false
  async_training.staleness_threshold="$ASYNC_STALENESS"
  async_training.partial_rollout="$ASYNC_PARTIAL"
  async_training.trigger_parameter_sync_step="$ASYNC_TRIGGER_SYNC_STEP"
  async_training.require_batches="$ASYNC_REQUIRE_BATCHES"
  trainer.n_gpus_per_node="$TRAIN_GPUS"
  rollout.n_gpus_per_node="$ROLLOUT_GPUS"
  trainer.resume_mode="$TRAINER_RESUME_MODE"
  trainer.val_before_train="$TRAINER_VAL_BEFORE_TRAIN"
  trainer.default_local_dir="$TRAINER_DEFAULT_LOCAL_DIR"
  trainer.validation_data_dir="$TRAINER_VALIDATION_DATA_DIR"
  trainer.experiment_name="$TRAINER_EXPERIMENT_NAME"
  trainer.save_freq=0
  trainer.test_freq=0
  rollout.test_freq=0
  ray_kwargs.ray_init.num_cpus="$RAY_NUM_CPUS"
  actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP_SIZE"
  actor_rollout_ref.rollout.enforce_eager="$ROLLOUT_ENFORCE_EAGER"
  actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_GPU_MEMORY_UTILIZATION"
)

if [ -n "$ASYNC_MAX_CONCURRENT_SAMPLES" ]; then
  CMD+=(async_training.max_concurrent_samples="$ASYNC_MAX_CONCURRENT_SAMPLES")
fi

if [ -n "$ASYNC_MAX_QUEUE_SIZE" ]; then
  CMD+=(async_training.max_queue_size="$ASYNC_MAX_QUEUE_SIZE")
fi

if [ "$ASYNC_CONFIG_NAME" = "carr_grpo_async_base" ]; then
  CMD+=(
    reward.custom_reward_function.path="$PROJECT_DIR/examples/carr_deepsearch/reward/async_base_reward.py"
  )
else
  CMD+=(
    reward.custom_reward_function.path="$PROJECT_DIR/examples/carr_deepsearch/reward/carr_reward.py"
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$PROJECT_DIR/examples/carr_deepsearch/config/tool_config/carr_browser_tools.yaml"
  )
fi

if [ "${TRAIN_GPUS}:${ROLLOUT_GPUS}" = "3:5" ]; then
  CMD+=(actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1)
fi

if [ -n "$ROLLOUT_TOTAL_STEPS" ]; then
  CMD+=(rollout.total_rollout_steps="$ROLLOUT_TOTAL_STEPS")
fi

if [ -n "$MAX_RESPONSE_LENGTH" ]; then
  CMD+=(data.max_response_length="$MAX_RESPONSE_LENGTH")
fi

if [ -n "$PPO_MAX_TOKEN_LEN_PER_GPU" ]; then
  CMD+=(actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$PPO_MAX_TOKEN_LEN_PER_GPU")
fi

if [ -n "$MAX_ASSISTANT_TURNS" ] && [ "$ASYNC_CONFIG_NAME" != "carr_grpo_async_base" ]; then
  CMD+=(actor_rollout_ref.rollout.multi_turn.max_assistant_turns="$MAX_ASSISTANT_TURNS")
fi

if [ -n "$MAX_TOOL_RESPONSE_LENGTH" ] && [ "$ASYNC_CONFIG_NAME" != "carr_grpo_async_base" ]; then
  CMD+=(actor_rollout_ref.rollout.multi_turn.max_tool_response_length="$MAX_TOOL_RESPONSE_LENGTH")
fi

if [ -n "$ATTENTION_BACKEND" ] && [ "$ATTENTION_BACKEND" != "auto" ]; then
  CMD+=(+actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend="$ATTENTION_BACKEND")
fi

"${CMD[@]}" "$@"
