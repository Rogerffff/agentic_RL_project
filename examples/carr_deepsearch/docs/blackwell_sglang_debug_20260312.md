## Blackwell SGLang Compatibility Debug (2026-03-12)

Target machine:

- `8x RTX PRO 6000 Blackwell Server Edition`
- SSH: `ssh -p 47918 root@147.185.60.9`
- `torch 2.9.1+cu129`
- `sglang 0.5.6.post2`
- `flashinfer 0.5.3`
- Model: `/root/checkpoints/carr_deepsearch_sft_sp4_64k/global_step_591/huggingface`

### Final conclusion

This machine is not currently suitable for formal `verl + SGLang TP2` agent evaluation / RL.

What is confirmed:

1. Multi-GPU startup hang was partly caused by NCCL P2P on this host.
2. Setting `NCCL_P2P_DISABLE=1` fixes TP2 server startup.
3. But after startup, real TP2 `/generate` requests still hang on Blackwell under current SGLang stack.
4. The problem is not specific to `verl`; it reproduces in standalone SGLang.
5. The problem is not "long prompt only"; the hanging standalone test used a tiny 10-token prompt and `max_new_tokens=16`.
6. TP1 generation works on the same host with the same model / attention backend / skip-tokenizer-init mode.

So the current blocker is:

- Blackwell + SGLang 0.5.6 + TP2 generation path

Not:

- checkpoint corruption
- `verl` agent loop logic
- reward / tool server chain
- missing `disable_cuda_graph`

### Key findings

#### 1. NCCL P2P is broken on this host for TP startup

A raw 2-GPU NCCL smoke test hung until `NCCL_P2P_DISABLE=1` was added.

Implication:

- All future Blackwell tests on this host must export `NCCL_P2P_DISABLE=1`.

#### 2. Standalone TP2 SGLang can start after disabling P2P

This worked:

- TP2
- `attention_backend=flashinfer`
- `disable_cuda_graph=True`
- `NCCL_P2P_DISABLE=1`

Health endpoint succeeded:

- `/get_model_info`

Relevant remote log:

- `/root/logs/sglang_direct_tp2_p2p_off.log`

#### 3. `verl` gate still failed on first real sample

Minimal `run_eval.sh` gate with:

- `NCCL_P2P_DISABLE=1`
- `enforce_eager=true`
- `attention_backend=flashinfer`

progressed much further than before, but eventually failed when a real SGLang actor died.

Failure stack:

- `ray::TaskRunner.run()`
- `ray::AgentLoopWorker.generate_sequences()`
- `SGLangHttpServer` actor died

The worker log shows:

- request entered scheduler
- then SGLang watchdog fired

Relevant remote log:

- `/root/logs/deepdive_gate1_p2p_off_8gpu_64k_t10_tool4k.log`

Relevant Ray worker error evidence:

- `/tmp/ray/session_latest/logs/worker-6336a353695d75e849c80bb00ad7fdd810e76034272d54a91d2a5900-01000000-114970.err`

Important detail:

- the hanging request had only `514` input tokens
- so this was not a genuine "64k prompt is too large" case

#### 4. `disable_cuda_graph` was already active in `verl`

`enforce_eager=true` already maps to SGLang `disable_cuda_graph=true`.

Code path:

- `verl/workers/rollout/sglang_rollout/async_sglang_server.py`

So the Blackwell failure is not caused by forgetting to disable CUDA graph.

#### 5. Increasing SGLang `watchdog_timeout` does not solve it

A new minimized `verl` gate used:

- `data.max_response_length=2048`
- `max_assistant_turns=1`
- `max_tool_response_length=1000`
- `watchdog_timeout=1800`

`watchdog_timeout=1800` was confirmed in config / worker logs, but the gate still never completed a real sample.

This ruled out the earlier `300s` watchdog as the root cause by itself.

#### 6. Standalone TP2 generate reproduces the issue

Standalone control server:

- port `30003`
- TP2
- `flashinfer`
- `disable_cuda_graph`
- `skip_tokenizer_init`
- `watchdog_timeout=1800`

Server startup was successful, but an `input_ids`-based tiny generate request hung.

The server log reached:

- `Prefill batch, #new-seq: 1, #new-token: 10`

and then stopped making progress.

Relevant remote log:

- `/root/logs/sglang_blackwell_control.log`

This is the most important isolation result:

- the failure reproduces without `verl`
- therefore the issue is in SGLang TP2 generation on this Blackwell stack

#### 7. `fa3` is not usable on Blackwell

Standalone TP2 with `--attention-backend fa3` failed immediately.

Exact error:

- `FlashAttention v3 Backend requires SM>=80 and SM<=90. Please use --attention-backend flashinfer.`

Implication:

- `fa3` is not a viable workaround on `SM120`

#### 8. `fa4` can only be used for prefill, and it still does not fix TP2 generate

`--attention-backend fa4` is invalid.

Error:

- `FA4 backend is only supported for prefill. Please use --prefill-attention-backend fa4 instead.`

Then a hybrid server was tested:

- `prefill_attention_backend=fa4`
- `decode_attention_backend=flashinfer`

Startup succeeded, but tiny `input_ids` TP2 generate still hung after:

- `Prefill batch, #new-seq: 1, #new-token: 128`

Relevant remote log:

- `/root/logs/sglang_blackwell_control_fa4prefill.log`

Implication:

- replacing prefill with `fa4` is not sufficient

#### 9. TP1 works

Standalone TP1 server:

- port `30007`
- `flashinfer`
- `disable_cuda_graph`
- `skip_tokenizer_init`

The same `input_ids` tiny generate request succeeded.

Observed result:

- HTTP `200`
- `e2e_latency ~= 0.80s`
- returned 16 output tokens

Relevant remote log:

- `/root/logs/sglang_blackwell_control_tp1.log`

This is the final localization result:

- TP1 works
- TP2 hangs

### Practical recommendation

For this Vast Blackwell instance:

- Do not use it for formal `verl` agent RL with current SGLang TP2 rollout.

If Blackwell must be used in the future, start from these facts:

1. export `NCCL_P2P_DISABLE=1`
2. `fa3` is unavailable on `SM120`
3. `fa4` prefill alone does not fix TP2 generation
4. current failure reproduces in standalone TP2 `/generate`

So the next meaningful fixes would have to come from:

- a newer SGLang build with Blackwell TP support
- a different supported attention backend for Blackwell TP decode
- or a setup that avoids TP2 entirely

For current project execution, prefer:

- `8x A100`
- or `8x H200`

instead of this `8x RTX PRO 6000 Blackwell` host.
