#!/bin/bash
# Unified CaRR evaluation script for BrowseComp and DeepDive RL val.
# Usage: bash run_eval.sh <model_path> <eval_set> [context] [extra hydra overrides...]
#   model_path - path to HF model checkpoint
#   eval_set   - browsecomp_subset256 | browsecomp_full | browsecomp_smoke10 | deepdive_val
#   context    - context window: 64k or 128k (default: 64k)
set -euxo pipefail

START_TIME=$(date +%s)

PROJECT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$PROJECT_DIR"

MODEL_PATH="${1:?Usage: run_eval.sh <model_path> <eval_set> [context]}"
EVAL_SET="${2:?Must specify eval_set: browsecomp_subset256|browsecomp_full|browsecomp_smoke10|deepdive_val}"
CONTEXT="${3:-64k}"
shift 3 2>/dev/null || true

if [ -z "${SERPER_API_KEY:-}" ] && [ -z "${SERPAPI_API_KEY:-}" ]; then
    echo "ERROR: Must set SERPER_API_KEY or SERPAPI_API_KEY" >&2; exit 1
fi
: "${JINA_API_KEY:?Must set JINA_API_KEY}"
: "${DEEPSEEK_API_KEY:?Must set DEEPSEEK_API_KEY}"

export VERL_USE_EXTERNAL_MODULES=examples.carr_deepsearch.tools.carr_agent_loop,examples.carr_deepsearch.reward.cgrpo_advantage
export CARR_REWARD_SERVER_URL="http://localhost:8888"
export CARR_REWARD_TIMEOUT="650"

# Route eval_set to val_files + extra args
DATA_DIR="$PROJECT_DIR/examples/carr_deepsearch/data"
EXTRA_ARGS=()
case "$EVAL_SET" in
  browsecomp_subset256)
    EVAL_FILE="$DATA_DIR/browsecomp_eval_subset_256_seed42.parquet"
    ;;
  browsecomp_full)
    EVAL_FILE="$DATA_DIR/browsecomp_eval.parquet"
    ;;
  browsecomp_smoke10)
    EVAL_FILE="$DATA_DIR/browsecomp_eval.parquet"
    EXTRA_ARGS+=(data.val_max_samples=10)
    ;;
  deepdive_val)
    EVAL_FILE="$DATA_DIR/rl_val.parquet"
    ;;
  *)
    echo "ERROR: unknown eval_set '$EVAL_SET'" >&2
    echo "Valid options: browsecomp_subset256, browsecomp_full, browsecomp_smoke10, deepdive_val" >&2
    exit 1
    ;;
esac

# Preprocess BrowseComp data if needed
if [[ "$EVAL_SET" == browsecomp_* ]] && [ ! -f "$DATA_DIR/browsecomp_eval.parquet" ]; then
    echo "Running BrowseComp data preprocessing..."
    python "$PROJECT_DIR/examples/carr_deepsearch/data_preprocess/preprocess_browsecomp.py" \
        --output_dir "$DATA_DIR"
fi

# Map context to max_response_length
case "$CONTEXT" in
  64k)  MAX_RESP_LEN=61440 ;;
  128k) MAX_RESP_LEN=122880 ;;
  *)    echo "ERROR: context must be 64k or 128k" >&2; exit 1 ;;
esac

OUTPUT_DIR="$HOME/eval_results/${EVAL_SET}_$(basename "$MODEL_PATH")_${CONTEXT}"
mkdir -p "$OUTPUT_DIR"

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

echo "Running evaluation (eval_set=$EVAL_SET, context=$CONTEXT)..."
python3 -m verl.trainer.main_ppo \
    --config-path="$PROJECT_DIR/examples/carr_deepsearch/config" \
    --config-name='carr_grpo' \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    data.train_files="$DATA_DIR/rl_train.parquet" \
    data.val_files="$EVAL_FILE" \
    data.max_response_length="$MAX_RESP_LEN" \
    data.val_batch_size=32 \
    reward.custom_reward_function.path="$PROJECT_DIR/examples/carr_deepsearch/reward/carr_reward.py" \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$PROJECT_DIR/examples/carr_deepsearch/config/tool_config/carr_browser_tools.yaml" \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=false \
    trainer.val_only=True \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node="${NGPUS:-4}" \
    trainer.validation_data_dir="$OUTPUT_DIR" \
    "${EXTRA_ARGS[@]}" \
    "$@"

echo "Results saved to: $OUTPUT_DIR"

END_TIME=$(date +%s); ELAPSED=$((END_TIME - START_TIME))
echo "Completed in ${ELAPSED}s ($(( ELAPSED / 3600 ))h$(( (ELAPSED % 3600) / 60 ))m$(( ELAPSED % 60 ))s)"
