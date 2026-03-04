#!/bin/bash
# BrowseComp evaluation using main_ppo val_only mode.
# Usage: bash run_eval_browsecomp.sh <model_path> [max_samples] [context]
#   model_path  - path to HF model checkpoint
#   max_samples - number of samples to evaluate (-1 for all, default: -1)
#   context     - context window: 64k or 128k (default: 64k)
set -euxo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$PROJECT_DIR"

MODEL_PATH="${1:?Usage: run_eval_browsecomp.sh <model_path> [max_samples] [context]}"
MAX_SAMPLES="${2:--1}"
CONTEXT="${3:-64k}"

: "${SERPAPI_API_KEY:?Must set SERPAPI_API_KEY}"
: "${JINA_API_KEY:?Must set JINA_API_KEY}"
: "${DEEPSEEK_API_KEY:?Must set DEEPSEEK_API_KEY}"

export VERL_USE_EXTERNAL_MODULES=examples.carr_deepsearch.tools.carr_agent_loop,examples.carr_deepsearch.reward.cgrpo_advantage
export CARR_REWARD_SERVER_URL="http://localhost:8888"
export CARR_REWARD_TIMEOUT="650"

# Map context to max_response_length
case "$CONTEXT" in
  64k)  MAX_RESP_LEN=61440 ;;
  128k) MAX_RESP_LEN=122880 ;;
  *)    echo "ERROR: context must be 64k or 128k" >&2; exit 1 ;;
esac

OUTPUT_DIR="$HOME/eval_results/browsecomp_$(basename "$MODEL_PATH")_${CONTEXT}"
mkdir -p "$OUTPUT_DIR"

DATA_DIR="$PROJECT_DIR/examples/carr_deepsearch/data"

# Preprocess BrowseComp data if needed
if [ ! -f "$DATA_DIR/browsecomp_eval.parquet" ]; then
    echo "Running BrowseComp data preprocessing..."
    python examples/carr_deepsearch/data_preprocess/preprocess_browsecomp.py \
        --output_dir "$DATA_DIR"
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

echo "Running BrowseComp evaluation (context=$CONTEXT, max_samples=$MAX_SAMPLES)..."
python3 -m verl.trainer.main_ppo \
    --config-path="$PROJECT_DIR/examples/carr_deepsearch/config" \
    --config-name='carr_grpo' \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    data.train_files="$DATA_DIR/rl_train.parquet" \
    data.val_files="$DATA_DIR/browsecomp_eval.parquet" \
    data.max_response_length="$MAX_RESP_LEN" \
    data.val_batch_size=32 \
    data.val_max_samples="$MAX_SAMPLES" \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=false \
    trainer.val_only=True \
    trainer.val_before_train=True \
    trainer.validation_data_dir="$OUTPUT_DIR" \
    "$@"

echo "Results saved to: $OUTPUT_DIR"
