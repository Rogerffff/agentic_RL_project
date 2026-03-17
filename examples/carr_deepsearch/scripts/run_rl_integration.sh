#!/bin/bash
# Manual RL probe launcher that keeps tool/reward server artifacts and status
# around so step completion and cache/early-exit evidence can be inspected.
#
# Usage:
#   bash run_rl_integration.sh <model_path> <run_name> [extra hydra overrides...]

set -euo pipefail

MODEL_PATH="${1:?Usage: run_rl_integration.sh <model_path> <run_name> [extra hydra overrides...]}"
RUN_NAME="${2:?Usage: run_rl_integration.sh <model_path> <run_name> [extra hydra overrides...]}"
shift 2

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
OUT_DIR="${OUT_DIR:-$HOME/checkpoints/carr_deepsearch_rl}"
LOG_DIR="${LOG_DIR:-$HOME/logs}"
MAIN_LOG="$LOG_DIR/${RUN_NAME}.log"
TOOL_LOG="$LOG_DIR/${RUN_NAME}_tool.log"
REWARD_LOG="$LOG_DIR/${RUN_NAME}_reward.log"
TRACE_LOG="$LOG_DIR/${RUN_NAME}_reward_trace.jsonl"
TOOL_STATS="$LOG_DIR/${RUN_NAME}_tool_stats.json"
STATUS_FILE="$LOG_DIR/${RUN_NAME}.status"
LAUNCHER_LOG="$LOG_DIR/${RUN_NAME}.launcher.log"

mkdir -p "$LOG_DIR" "$OUT_DIR"

log_step() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" >> "$LAUNCHER_LOG"
}

detect_gpu_names() {
  nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | paste -sd ';' - || true
}

if [ -f "$HOME/.env" ]; then
  set -a
  . "$HOME/.env"
  set +a
fi

if [ -z "${SERPER_API_KEY:-}" ] && [ -z "${SERPAPI_API_KEY:-}" ]; then
  echo "missing search api key" > "$STATUS_FILE"
  exit 1
fi
: "${JINA_API_KEY:?Must set JINA_API_KEY}"
: "${DEEPSEEK_API_KEY:?Must set DEEPSEEK_API_KEY}"

export VERL_USE_EXTERNAL_MODULES=examples.carr_deepsearch.tools.carr_agent_loop,examples.carr_deepsearch.reward.cgrpo_advantage
export CARR_REWARD_SERVER_URL="${CARR_REWARD_SERVER_URL:-http://localhost:8888}"
export CARR_REWARD_TIMEOUT="${CARR_REWARD_TIMEOUT:-650}"
export CARR_REWARD_TRACE_LOG=1
export CARR_REWARD_TRACE_LOG_PATH="$TRACE_LOG"
export CARR_TOOL_CLIENT_TIMEOUT_S="${CARR_TOOL_CLIENT_TIMEOUT_S:-120}"

GPU_NAMES="$(detect_gpu_names)"
# Some Blackwell setups require explicitly disabling NCCL P2P, but the NVLink hosts
# used for H100/H200/A100 probes in this project should keep the default NCCL path.
if [ -n "${NCCL_P2P_DISABLE:-}" ]; then
  case "$GPU_NAMES" in
    *H100*|*H200*|*A100*)
      if [ "${ALLOW_NCCL_P2P_DISABLE_ON_NVLINK:-0}" != "1" ]; then
        echo "Refusing to run with NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE} on $GPU_NAMES. Unset it or set ALLOW_NCCL_P2P_DISABLE_ON_NVLINK=1 to override." > "$STATUS_FILE"
        echo "$(cat "$STATUS_FILE")" >&2
        exit 1
      fi
      ;;
  esac
  export NCCL_P2P_DISABLE
fi
export RAY_enable_open_telemetry=0
export RAY_ENABLE_OPEN_TELEMETRY=0

: > "$LAUNCHER_LOG"
log_step "STEP:begin"
log_step "STEP:env gpu_names=${GPU_NAMES:-unknown} NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-unset} NCCL_DEBUG=${NCCL_DEBUG:-unset}"
ray stop --force >> "$LAUNCHER_LOG" 2>&1 || true
log_step "STEP:ray_stopped"

PIDS=()
cleanup() {
  log_step "STEP:cleanup"
  curl -sf http://localhost:7230/stats > "$TOOL_STATS" || true
  for pid in "${PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT

cd "$PROJECT_DIR"

if [ -n "${SERPER_API_KEY:-}" ]; then
  SEARCH_ARGS=(--search_backend serper --serper_api_key "$SERPER_API_KEY")
else
  SEARCH_ARGS=(--serp_api_key "$SERPAPI_API_KEY")
fi

python "$PROJECT_DIR/CaRR/tool_server/launch_server.py" \
  "${SEARCH_ARGS[@]}" \
  --jina_api_key "$JINA_API_KEY" \
  --port 7230 > "$TOOL_LOG" 2>&1 &
PIDS+=($!)
log_step "STEP:tool_started pid=${PIDS[-1]}"

(
  cd "$PROJECT_DIR/CaRR/deepsearch_rm_with_rubrics"
  python launch_server.py \
    --port 8888 \
    --model_name deepseek-chat \
    --base_url https://api.deepseek.com \
    --api_key "$DEEPSEEK_API_KEY"
) > "$REWARD_LOG" 2>&1 &
PIDS+=($!)
log_step "STEP:reward_started pid=${PIDS[-1]}"

for i in {1..30}; do
  if curl -sf -X POST http://localhost:7230 -H "Content-Type: application/json" -d '{"session_id":"health","name":"start_session","arguments":{},"remote_env_info":{}}' >/dev/null; then
    log_step "STEP:tool_ready attempt=$i"
    break
  fi
  sleep 2
done

for i in {1..30}; do
  if curl -sf -X POST http://localhost:8888/evaluate -H "Content-Type: application/json" -d '{"history":[{"role":"user","content":"q"},{"role":"assistant","content":"a"}],"label":"a","task_unfinished":true,"remote_env_info":{"search_forbidden_strs":["q"],"rubrics":[],"rubric_reward_ratio":0.3}}' >/dev/null; then
    log_step "STEP:reward_ready attempt=$i"
    break
  fi
  sleep 2
done

ATTN_OVERRIDE="+actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=${CARR_SGLANG_ATTENTION_BACKEND:-flashinfer}"
CMD=(
  python3 -m verl.trainer.main_ppo
  --config-path="$PROJECT_DIR/examples/carr_deepsearch/config"
  --config-name=carr_grpo
  actor_rollout_ref.model.path="$MODEL_PATH"
  data.train_files="$PROJECT_DIR/examples/carr_deepsearch/data/rl_train.parquet"
  data.val_files="$PROJECT_DIR/examples/carr_deepsearch/data/rl_val.parquet"
  reward.custom_reward_function.path="$PROJECT_DIR/examples/carr_deepsearch/reward/carr_reward.py"
  actor_rollout_ref.rollout.multi_turn.tool_config_path="$PROJECT_DIR/examples/carr_deepsearch/config/tool_config/carr_browser_tools.yaml"
  "$ATTN_OVERRIDE"
)

printf 'CMD:' >> "$LAUNCHER_LOG"
printf ' %q' "${CMD[@]}" "$@" >> "$LAUNCHER_LOG"
printf '\n' >> "$LAUNCHER_LOG"
log_step "STEP:main_ppo_start"
set +e
"${CMD[@]}" "$@" > "$MAIN_LOG" 2>&1
RC=$?
set -e
log_step "STEP:main_ppo_end rc=$RC"
printf '%s\n' "$RC" > "$STATUS_FILE"
exit "$RC"
