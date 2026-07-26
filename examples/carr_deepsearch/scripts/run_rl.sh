#!/bin/bash
set -euxo pipefail

START_TIME=$(date +%s)

PROJECT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$PROJECT_DIR"

if [ -z "${SERPER_API_KEY:-}" ] && [ -z "${SERPAPI_API_KEY:-}" ]; then
    echo "ERROR: Must set SERPER_API_KEY or SERPAPI_API_KEY" >&2; exit 1
fi
: "${JINA_API_KEY:?Must set JINA_API_KEY}"
: "${DEEPSEEK_API_KEY:?Must set DEEPSEEK_API_KEY}"

export VERL_USE_EXTERNAL_MODULES=examples.carr_deepsearch.tools.carr_agent_loop,examples.carr_deepsearch.reward.cgrpo_advantage
export CARR_REWARD_SERVER_URL="http://localhost:8888"
export CARR_REWARD_TIMEOUT="650"
# Ray 2.53 can crash natively in OpenTelemetry metrics init on some remote setups.
# Disable it explicitly so RL startup does not depend on that fragile path.
export RAY_enable_open_telemetry=0
export RAY_ENABLE_OPEN_TELEMETRY=0

# Resolve SFT checkpoint path dynamically
if [ -z "${SFT_MODEL_PATH:-}" ]; then
    SFT_CKPT_ROOT="$HOME/checkpoints/carr_deepsearch_sft"
    LATEST_STEP=$(cat "$SFT_CKPT_ROOT/latest_checkpointed_iteration.txt")
    export SFT_MODEL_PATH="$SFT_CKPT_ROOT/global_step_${LATEST_STEP}/huggingface"
fi
echo "Using SFT model from: $SFT_MODEL_PATH"

DATA_DIR="$PROJECT_DIR/examples/carr_deepsearch/data"
if [ ! -f "$DATA_DIR/rl_train.parquet" ]; then
    echo "Running RL data preprocessing..."
    python examples/carr_deepsearch/data_preprocess/preprocess_carr_rl.py \
        --input_file CaRR/data/deepdive-rl-2k-rubrics.jsonl \
        --output_dir "$DATA_DIR" \
        --val_ratio 0.05 \
        --seed 42
fi

PIDS=()
cleanup() {
    echo "Cleaning up background processes..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
}
trap cleanup EXIT

echo "Starting CaRR tool server on port 7230..."
if [ -n "${SERPER_API_KEY:-}" ]; then
    SEARCH_ARGS="--search_backend serper --serper_api_key $SERPER_API_KEY"
else
    SEARCH_ARGS="--serp_api_key $SERPAPI_API_KEY"
fi
python "$PROJECT_DIR/CaRR/tool_server/launch_server.py" \
    $SEARCH_ARGS \
    --jina_api_key "$JINA_API_KEY" \
    --port 7230 &
PIDS+=($!)

echo "Starting CaRR reward server on port 8888..."
(
  cd "$PROJECT_DIR/CaRR/deepsearch_rm_with_rubrics"
  python launch_server.py \
    --port 8888 \
    --model_name deepseek-chat \
    --base_url https://api.deepseek.com \
    --api_key "$DEEPSEEK_API_KEY"
) &
PIDS+=($!)

# Wait for tool server readiness
echo "Waiting for tool server..."
for i in {1..30}; do
  if curl -sf -X POST http://localhost:7230 \
    -H "Content-Type: application/json" \
    -d '{"session_id":"health","name":"start_session","arguments":{},"remote_env_info":{}}' >/dev/null; then
    echo "Tool server ready."
    break
  fi
  sleep 2
done
if ! curl -sf -X POST http://localhost:7230 \
  -H "Content-Type: application/json" \
  -d '{"session_id":"health","name":"start_session","arguments":{},"remote_env_info":{}}' >/dev/null; then
  echo "ERROR: Tool server failed to start" >&2
  exit 1
fi

# Wait for reward server readiness
echo "Waiting for reward server..."
for i in {1..30}; do
  if curl -sf -X POST http://localhost:8888/evaluate \
    -H "Content-Type: application/json" \
    -d '{"history":[{"role":"user","content":"q"},{"role":"assistant","content":"a"}],"label":"a","task_unfinished":true,"remote_env_info":{"search_forbidden_strs":["q"],"rubrics":[],"rubric_reward_ratio":0.3}}' >/dev/null; then
    echo "Reward server ready."
    break
  fi
  sleep 2
done
if ! curl -sf -X POST http://localhost:8888/evaluate \
  -H "Content-Type: application/json" \
  -d '{"history":[{"role":"user","content":"q"},{"role":"assistant","content":"a"}],"label":"a","task_unfinished":true,"remote_env_info":{"search_forbidden_strs":["q"],"rubrics":[],"rubric_reward_ratio":0.3}}' >/dev/null; then
  echo "ERROR: Reward server failed to start" >&2
  exit 1
fi

echo "Starting RL training..."
python3 -m verl.trainer.main_ppo \
    --config-path="$PROJECT_DIR/examples/carr_deepsearch/config" \
    --config-name='carr_grpo' \
    data.train_files="$PROJECT_DIR/examples/carr_deepsearch/data/rl_train.parquet" \
    data.val_files="$PROJECT_DIR/examples/carr_deepsearch/data/rl_val.parquet" \
    reward.custom_reward_function.path="$PROJECT_DIR/examples/carr_deepsearch/reward/carr_reward.py" \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$PROJECT_DIR/examples/carr_deepsearch/config/tool_config/carr_browser_tools.yaml" \
    "$@"

END_TIME=$(date +%s); ELAPSED=$((END_TIME - START_TIME))
echo "Completed in ${ELAPSED}s ($(( ELAPSED / 3600 ))h$(( (ELAPSED % 3600) / 60 ))m$(( ELAPSED % 60 ))s)"
