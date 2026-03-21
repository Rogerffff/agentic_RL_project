#!/bin/bash
set -euxo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
cd "$PROJECT_DIR"

ASYNC_CONFIG_NAME="${ASYNC_CONFIG_NAME:-carr_grpo_async}"
ASYNC_PROFILE="${ASYNC_PROFILE:-sync_stream}"
TRAIN_GPUS="${TRAIN_GPUS:-4}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:-4}"

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

if [ "$ASYNC_CONFIG_NAME" != "carr_grpo_async_base" ]; then
  if [ -z "${SERPER_API_KEY:-}" ] && [ -z "${SERPAPI_API_KEY:-}" ]; then
    echo "ERROR: Must set SERPER_API_KEY or SERPAPI_API_KEY" >&2
    exit 1
  fi
  : "${JINA_API_KEY:?Must set JINA_API_KEY}"
  : "${DEEPSEEK_API_KEY:?Must set DEEPSEEK_API_KEY}"

  export CARR_REWARD_SERVER_URL="${CARR_REWARD_SERVER_URL:-http://localhost:8888}"
  export CARR_REWARD_TIMEOUT="${CARR_REWARD_TIMEOUT:-650}"

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
  trainer.save_freq=0
  trainer.test_freq=0
  rollout.test_freq=0
)

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

"${CMD[@]}" "$@"
