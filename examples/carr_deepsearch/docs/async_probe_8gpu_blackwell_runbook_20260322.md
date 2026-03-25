# 8x RTX PRO 6000 Blackwell Async Probe Runbook

This runbook is the executable version of the 8-GPU async probe plan for the current `sglang 0.5.6.post2` stack.

## Entry Points

- Common launcher: `examples/carr_deepsearch/scripts/run_rl_async.sh`
- Probe wrapper: `examples/carr_deepsearch/scripts/run_async_probe_8gpu_blackwell.sh`

## What The Wrapper Enforces

- Fresh run isolation for every phase:
  - `trainer.resume_mode=disable`
  - unique `trainer.default_local_dir`
  - unique `trainer.validation_data_dir`
  - unique `trainer.experiment_name`
- Blackwell-safe rollout defaults:
  - `4:4`
  - `TP1`
  - `enforce_eager=true`
  - `gpu_memory_utilization=0.3`
  - `NCCL_P2P_DISABLE=1`
  - `ray_kwargs.ray_init.num_cpus=128`
- Backend fallback:
  - base probe tries `flashinfer` first
  - if that fails, it retries once with no attention backend override

## Required Inputs

- SFT checkpoint directory:
  - default `SFT_PROBE_MODEL_PATH=/root/sft_checkpoint`
- RL checkpoint directory:
  - default `/root/checkpoints/carr_8gpu_formal/global_step_90/actor/huggingface`
  - fallback `/root/CaRR_90step_checkpoint/actor/huggingface`
- API keys for non-base phases:
  - `SERPER_API_KEY` or `SERPAPI_API_KEY`
  - `JINA_API_KEY`
  - `DEEPSEEK_API_KEY`
  - `WANDB_API_KEY`

## Recommended Commands

Preflight only:

```bash
PROBE_MODE=preflight \
bash examples/carr_deepsearch/scripts/run_async_probe_8gpu_blackwell.sh
```

Base probe with backend fallback:

```bash
PROBE_MODE=base \
SFT_PROBE_MODEL_PATH=/root/sft_checkpoint \
bash examples/carr_deepsearch/scripts/run_async_probe_8gpu_blackwell.sh
```

Full runbook:

```bash
PROBE_MODE=all \
SFT_PROBE_MODEL_PATH=/root/sft_checkpoint \
RL_PROBE_MODEL_PATH=/root/checkpoints/carr_8gpu_formal/global_step_90/actor/huggingface \
bash examples/carr_deepsearch/scripts/run_async_probe_8gpu_blackwell.sh
```

## Phase Mapping

- `phase0_async_base_8gpu_bw_tp1_<timestamp>`
  - `ASYNC_PROFILE=base`
  - `rollout.total_rollout_steps=32`
- `phase1_sync_stream_8gpu_bw_tp1_<timestamp>`
  - `ASYNC_PROFILE=sync_stream`
  - `rollout.total_rollout_steps=24`
  - `max_response_length=16384`
  - `max_assistant_turns=10`
  - `max_tool_response_length=4000`
- `phase2_async_partial_8gpu_bw_tp1_<timestamp>`
  - same shape as phase 1
  - `ASYNC_PROFILE=async_partial`
- `phase3_async_followup_64k_8gpu_bw_tp1_<timestamp>`
  - `ASYNC_PROFILE=async_partial`
  - `rollout.total_rollout_steps=16`
  - `max_response_length=61440`
  - `max_assistant_turns=30`
  - `max_tool_response_length=10000`

## Logs

- Per-phase logs go to `$HOME/logs/<phase_name>.log`
- Training outputs go to `$HOME/checkpoints/<phase_name>`
- Validation outputs go to `$HOME/eval_results/<phase_name>`
