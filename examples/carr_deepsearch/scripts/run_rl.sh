#!/bin/bash
set -euxo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$PROJECT_DIR"

: "${SERPAPI_API_KEY:?Must set SERPAPI_API_KEY}"
: "${JINA_API_KEY:?Must set JINA_API_KEY}"
: "${DEEPSEEK_API_KEY:?Must set DEEPSEEK_API_KEY}"

export VERL_USE_EXTERNAL_MODULES=examples.carr_deepsearch.tools.carr_agent_loop,examples.carr_deepsearch.reward.cgrpo_advantage
export CARR_REWARD_SERVER_URL="http://localhost:8888"
export CARR_REWARD_TIMEOUT="650"

# Resolve SFT checkpoint path dynamically
SFT_CKPT_ROOT="$HOME/checkpoints/carr_deepsearch_sft"
LATEST_STEP=$(cat "$SFT_CKPT_ROOT/latest_checkpointed_iteration.txt")
export SFT_MODEL_PATH="$SFT_CKPT_ROOT/global_step_${LATEST_STEP}/huggingface"
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
python CaRR/tool_server/launch_server.py \
    --serp_api_key "$SERPAPI_API_KEY" \
    --jina_api_key "$JINA_API_KEY" \
    --port 7230 &
PIDS+=($!)

echo "Starting CaRR reward server on port 8888..."
(
  cd CaRR/deepsearch_rm_with_rubrics
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
    "$@"
