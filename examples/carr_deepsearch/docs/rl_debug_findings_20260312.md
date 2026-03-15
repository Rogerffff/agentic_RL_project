# RL Debug Findings (2026-03-12)

## Summary

This note records the root causes behind the misleading "PPO/log_prob stall" seen during CaRR DeepSearch RL dry runs on `4 x A100-80G`, the code fixes applied, and the current experiment recommendation.

The main conclusion is:

- The primary blocker was not raw `A100` compute.
- Two execution-path issues made dry runs look stuck:
  - small batches could fail or misbehave when `len(batch)` was not divisible by `rollout.agent.num_workers`
  - validation duplicated real samples by padding to worker count before rollout/reward
- After fixing these, one-step RL runs completed on `4 x A100` with real PPO updates.

## Root Causes

### 1. Agent-loop chunking assumed exact divisibility

Original behavior in `verl/experimental/agent_loop/agent_loop.py`:

- `AsyncAgentLoopManager.generate_sequences()` always did:
  - `prompts.chunk(len(self.agent_loop_workers))`
- `DataProto.chunk()` requires equal chunk sizes unless auto-padding is enabled
- This made small RL dry runs fail when:
  - `train_batch_size * rollout.n` was not divisible by `agent.num_workers`

Observed failure pattern:

- `b5 / n2 => 10 trajectories`
- default `agent.num_workers = 8`
- previous chain validation could fail at the worker split stage

### 2. Validation padded by duplicating real samples

Original behavior in `verl/trainer/ppo/ray_trainer.py::_validate()`:

- validation batch was padded to `agent.num_workers`
- padded samples were sent through full agent rollout and reward
- only after that, outputs were unpadded

This inflated validation-time API usage and made dry runs appear more expensive/slower than they actually were.

For small validation sets, this also confused reward-call accounting because the extra calls came from duplicated real samples, not just the original validation items.

### 3. Earlier "no train metrics" observations were partly due to step-end validation

In the trainer loop:

- metrics are logged after step-end work
- if `test_freq` fires on the same step, validation runs before `logger.log(...)`

So in earlier interrupted runs, we killed the process during post-step validation and incorrectly concluded that PPO never finished step 1.

## Code Changes

### A. Dynamic active worker count for agent loop

File:

- `verl/experimental/agent_loop/agent_loop.py`

Change:

- choose `active_workers = min(num_workers, len(prompts))`
- reduce `active_workers` until `len(prompts) % active_workers == 0`
- only dispatch to the active worker subset

Result:

- small RL debug runs no longer require `len(batch)` to be divisible by the full configured worker count
- `b5 / n2 / w8` now runs instead of failing at the split stage

### B. Remove validation padding/unpadding around agent rollout

File:

- `verl/trainer/ppo/ray_trainer.py`

Change:

- validation now directly calls:
  - `self.async_rollout_manager.generate_sequences(test_gen_batch)`
- removed pad/unpad logic around validation rollout

Result:

- validation no longer duplicates real samples just to satisfy worker count
- validation-time API usage is closer to the true number of validation samples

## Validation Evidence

### Run 1: minimal no-validation PPO step

Config:

- `train_batch_size=2`
- `rollout.n=2`
- `data.max_response_length=16384`
- `max_assistant_turns=10`
- `max_tool_response_length=4000`
- `actor/ref SP=2`
- `actor/ref no torch.compile`
- `agent.num_workers=1`
- `reward.num_workers=1`
- `test_freq=0`
- `save_freq=0`

Observed:

- run completed successfully
- `step:1` logged
- `timing_s/step = 40.54s`
- real PPO update happened on `4 x A100`

Important metrics:

- `outcome_reward/mean = 0.0`
- `rubric_reward/mean = 0.0`
- `task_unfinished/ratio = 0.75`
- `num_turns/mean = 18.5`
- `tool_call_counts/mean = 8.25`

Interpretation:

- the RL chain works end-to-end
- policy signal is weak, but the run is no longer blocked by infrastructure

### Run 2: regression test for original divisibility problem

Config:

- `train_batch_size=5`
- `rollout.n=2`
- `agent.num_workers=8`
- other settings kept in the same reduced dry-run regime
- `test_freq=0`
- `save_freq=0`

Observed:

- run completed successfully
- `step:1` logged
- `timing_s/step = 103.02s`

Important metrics:

- `outcome_reward/mean = 0.0`
- `rubric_reward/mean = 0.0`
- `task_unfinished/ratio = 0.9`
- `num_turns/mean = 18.2`

Interpretation:

- the previous worker-divisibility failure path is fixed
- higher concurrency alone does not produce reward signal yet

### Run 3: 64k no-validation signal probe

Config:

- `train_batch_size=2`
- `rollout.n=2`
- `data.max_response_length=61400`
- `max_assistant_turns=30`
- `max_tool_response_length=10000`
- `actor/ref SP=2`
- `actor/ref no torch.compile`
- `agent.num_workers=1`
- `reward.num_workers=1`
- `test_freq=0`
- `save_freq=0`

Observed:

- run completed successfully
- `step:1` logged
- wall clock `226s`
- `timing_s/step = 132.02s`
- WandB run: `j8v0asiu`

Important metrics:

- `outcome_reward/mean = 0.0`
- `rubric_reward/mean = 0.0`
- `task_unfinished/ratio = 1.0`
- `hit_limit/ratio = 1.0`
- `num_turns/min=max=mean = 60.0`
- `tool_call_counts/min=max=mean = 29.0`
- `response_length/mean = 31740`
- `response_length/max = 38730`
- `response_length/clip_ratio = 0.0`
- `response/aborted_ratio = 0.0`
- `search_count/mean = 13.0`
- `open_count/mean = 9.0`
- `find_count/mean = 7.0`

Interpretation:

- reward computation still happened under `64k`; the reward chain did not disappear
- however, the policy produced zero useful reward signal on this batch
- the dominant failure mode was **turn-limit termination**, not context-length exhaustion

Why this is clear:

- `num_turns` hit the exact hard ceiling (`60`)
- `tool_call_counts` also saturated at the exact ceiling-driven pattern (`29`)
- `response_length` stayed well below the `61400` response budget
- `response_length/clip_ratio = 0.0` and `response/aborted_ratio = 0.0`

This means the current `64k` probe does **not** support jumping straight into formal concurrency validation. The next targeted experiment should vary `max_assistant_turns`, not just increase batch/concurrency.

### Run 4: 64k turn-limit ablation (`t45`, `tool8k`)

Config:

- same as Run 3 except:
  - `max_assistant_turns=45`
  - `max_tool_response_length=8000`

Observed:

- run completed successfully
- `step:1` logged
- wall clock `241s`
- `timing_s/step = 150.36s`
- WandB run: `3azf15c6`

Important metrics:

- `outcome_reward/mean = 0.0`
- `rubric_reward/mean = 0.0`
- `task_unfinished/ratio = 0.5`
- `hit_limit/ratio = 0.5`
- `num_turns/min = 44`
- `num_turns/max = 90`
- `num_turns/mean = 70.0`
- `tool_call_counts/mean = 34.0`
- `response_length/mean = 24354.25`
- `response_length/max = 31966.0`
- `response_length/clip_ratio = 0.0`
- `response/aborted_ratio = 0.0`
- `search_count/mean = 24.25`
- `open_count/mean = 9.0`
- `find_count/mean = 0.75`

Interpretation:

- increasing `max_assistant_turns` materially reduced unfinished rate:
  - from `1.0` to `0.5`
- the dominant constraint in Run 3 was indeed the turn limit
- even after giving more turns, the policy still failed to produce positive reward
- the new failure mode is **not** context truncation:
  - response lengths remained far below `61400`
  - no response clipping occurred

Practical implication:

- the current SFT policy benefits from a looser turn budget
- but simply raising turn count does not recover useful RL reward signal on its own
- the next decision should focus on policy quality / rewardability, not just concurrency scaling

### Run 5: DeepDive RL val small eval (`64k`, `t45`, `tool8k`, 5 samples)

Purpose:

- validate whether the same checkpoint can finish real `DeepDive` validation tasks
- check whether `DeepSeek` judge is actually invoked when the eval set contains rubric-bearing tasks

Config:

- model: `/root/checkpoints/carr_deepsearch_sft_sp4_64k/global_step_396/huggingface`
- dataset: `examples/carr_deepsearch/data/rl_val.parquet`
- `data.val_max_samples=5`
- `data.max_response_length=61400`
- `max_assistant_turns=45`
- `max_tool_response_length=8000`
- WandB run: `6nx0fouk`

Observed:

- eval completed successfully in `171s`
- result file: `/root/eval_results/deepdive_calib5_sft_step396_64k_t45_tool8k_r61400/0.jsonl`
- WandB summary:
  - `outcome_reward/mean@1 = 0.0`
  - `rubric_reward/mean@1 = 0.0`
  - `task_unfinished/mean@1 = 1.0`
  - `hit_limit/mean@1 = 1.0`
  - `num_turns/min=max=mean = 90`
  - `tool_call_counts/mean@1 = 44`
  - `search_count/mean@1 = 36.8`
  - `open_count/mean@1 = 4.2`
  - `find_count/mean@1 = 3.0`
  - `parse_error_count/mean@1 = 1.0`

Per-sample result file summary:

- `reward = [0, 0, 0, 0, 0]`
- `task_unfinished = [True, True, True, True, True]`
- `hit_limit = [True, True, True, True, True]`
- `tool_call_counts = [44, 44, 44, 44, 44]`
- `search_count = [44, 33, 35, 41, 31]`
- `open_count = [0, 7, 6, 2, 6]`
- `find_count = [0, 4, 3, 1, 7]`

Interpretation:

- this run proves `browser.open` is definitely reachable under the same checkpoint:
  - several samples performed multiple successful `open` calls
- however, **all five samples still hit the turn limit**
- the current `t45` setting is still not enough for `DeepDive` validation
- because every sample had `task_unfinished=True`, the reward server short-circuited before judge invocation

Why `DeepSeek` appeared absent:

- main eval log showed `POST /evaluate ... 200` but `DEEPSEEK_HTTPX=0`
- this is expected under the current reward server logic:
  - if `task_unfinished=True`, `/evaluate` immediately returns zero reward
  - no judge model call is made
- see `CaRR/deepsearch_rm_with_rubrics/launch_server.py`:
  - `if task_unfinished: return {"reward": 0, ...}`

Practical implication:

- the absence of `DeepSeek` dashboard traffic in this eval does **not** imply the reward chain is broken
- it implies the policy never finished a sample, so judge invocation was skipped by design
- if the goal is to test whether finished `DeepDive` samples trigger judge requests, the next probe should raise turn budget again
- the next turn-budget probe should not be unbounded:
  - a reasonable next step is `max_assistant_turns=60`
  - pair it with a tighter `max_tool_response_length` such as `6000`
  - then re-check `task_unfinished`, `num_turns`, and whether any sample finally reaches judge

### Run 6: DeepDive RL val small eval (`64k`, `t60`, `tool8k`, 5 samples, reward trace on)

Purpose:

- test whether a more aggressive turn budget is enough to produce any completed `DeepDive` sample
- verify with trace logs whether any sample really reaches DeepSeek judge

Config:

- same model and dataset as Run 5
- `data.val_max_samples=5`
- `data.max_response_length=61400`
- `max_assistant_turns=60`
- `max_tool_response_length=8000`
- `CARR_REWARD_TRACE_LOG=1`
- `CARR_REWARD_TRACE_LOG_PATH=/root/logs/carr_reward_trace_t60.jsonl`
- WandB run: `0iq60rcq`

Observed:

- eval completed successfully in `576s`
- result file: `/root/eval_results/deepdive_calib5_sft_step396_64k_t60_tool8k_r61400/0.jsonl`
- WandB summary:
  - `outcome_reward/mean@1 = 0.0`
  - `rubric_reward/mean@1 = 0.0`
  - `task_unfinished/mean@1 = 1.0`
  - `hit_limit/mean@1 = 1.0`
  - `num_turns/min=max=mean = 120`
  - `tool_call_counts/mean@1 = 59`
  - `search_count/mean@1 = 23.6`
  - `open_count/mean@1 = 15.6`
  - `find_count/mean@1 = 19.8`
  - `parse_error_count/mean@1 = 1.0`

Per-sample result file summary:

- `reward = [0, 0, 0, 0, 0]`
- `task_unfinished = [True, True, True, True, True]`
- `hit_limit = [True, True, True, True, True]`
- `tool_call_counts = [59, 59, 59, 59, 59]`
- `search_count = [47, 5, 4, 7, 55]`
- `open_count = [7, 5, 53, 11, 2]`
- `find_count = [5, 49, 2, 41, 2]`

Reward trace summary:

- `TRACE_SHORT = 7`
  - `2` startup health checks
  - `5` real evaluation samples
- `TRACE_RESULT = 0`

Interpretation:

- even with `max_assistant_turns=60`, **no sample reached judge**
- reward trace proves all five real samples still short-circuited at the reward server
- this removes the ambiguity from dashboard-based observation:
  - DeepSeek was not missing due to logging or account delay
  - DeepSeek was skipped because every sample still ended unfinished
- raising turn budget from `45` to `60` increased cost and latency sharply:
  - `171s` at `t45`
  - `576s` at `t60`
- but it did **not** recover any reward signal

Practical implication:

- do not keep linearly increasing `max_assistant_turns` and expect reward to appear automatically
- the current policy appears able to use tools, but not to converge to completion
- removing turn limit and relying only on `max_response_length` would most likely make runs slower and more expensive while still short-circuiting at context exhaustion

## Current Experiment Readout

What is now confirmed:

- RL rollout -> tool server -> reward server -> PPO update is runnable on `4 x A100`
- the earlier "PPO/log_prob path is stuck" conclusion was too pessimistic
- `4 x A100` is not currently the main blocker

What is not yet confirmed:

- that the formal `64k` RL setting produces stable non-zero reward signal
- that increasing `max_assistant_turns` will actually recover useful reward signal
- that formal-concurrency settings are cost-effective enough for a long run

## Recommendation

Do **not** jump straight from this debugging result to full `Run 2b` as originally written.

Recommended next step:

1. Run a revised `64k` dry signal probe with:
   - formal `max_response_length`
   - `test_freq=0`
   - `save_freq=0`
   - a small train batch
2. Measure:
   - `outcome_reward`
   - `rubric_reward`
   - `task_unfinished`
   - `num_turns`
   - `search/open/find`
3. Then vary:
   - `max_assistant_turns`
   - optionally `max_tool_response_length`
4. Only after this, move to formal concurrency validation

Practical implication:

- The original Step 5 should be considered replaced by a narrower `64k no-validation signal probe`
- If that probe still yields near-zero reward with very high `task_unfinished`, the next problem is policy/SFT behavior, not RL infrastructure

## Files Modified

- `verl/experimental/agent_loop/agent_loop.py`
- `verl/trainer/ppo/ray_trainer.py`
- `CaRR/deepsearch_rm_with_rubrics/launch_server.py`

## Reward Trace Logging

Added a dedicated reward-trace switch to the reward server for experiment debugging:

- `CARR_REWARD_TRACE_LOG=1`
- `CARR_REWARD_TRACE_LOG_PATH=/path/to/reward_trace.jsonl`

Behavior:

- when `task_unfinished=True`, the server writes an `evaluate_short_circuit` JSONL record
- when judge evaluation really happens, the server writes an `evaluate_result` JSONL record
- the `evaluate_result` record includes:
  - `reward`
  - `outcome_reward`
  - `rubric_reward`
  - `outcome_reward_result.judgement`
  - extracted final answer / gold answer
  - rubric judge outputs when present

Verified on the remote machine with a temporary server on port `8898`:

- unfinished sample -> trace recorded `evaluate_short_circuit`
- completed toy sample (`What is the capital of France?`) -> trace recorded `evaluate_result`
- the logged `judgement` contained the actual DeepSeek judge output and returned `reward=1.0`

## Termination Instrumentation

Added pass-through fields for limit diagnosis:

- `termination_reason`
- `termination_response_limit`
- `termination_assistant_turn_limit`
- `termination_user_turn_limit`
- `response_length`
- `response_length_max`
- `response_length_ratio`

Implementation:

- agent loop sets these fields in `examples/carr_deepsearch/tools/carr_agent_loop.py`
- reward client passes them through in `examples/carr_deepsearch/reward/carr_reward.py`

This makes them visible in:

- validation JSONL dumps
- WandB validation metrics

### Run 7: DeepDive RL val small eval (`64k`, `t120`, `tool8k`, 5 samples, termination instrumentation on)

Purpose:

- determine whether the previous failures were purely due to `max_assistant_turns`
- distinguish `assistant_turn_limit` from `response_limit`

Config:

- same model and dataset as prior DeepDive probes
- `data.val_max_samples=5`
- `data.max_response_length=61400`
- `max_assistant_turns=120`
- `max_tool_response_length=8000`
- `CARR_REWARD_TRACE_LOG=1`
- `CARR_REWARD_TRACE_LOG_PATH=/root/logs/carr_reward_trace_t120.jsonl`
- WandB run: `qr8td0cn`

Observed:

- eval completed successfully
- `task_unfinished/mean@1 = 1.0`
- `hit_limit/mean@1 = 1.0`
- `outcome_reward/mean@1 = 0.0`
- `rubric_reward/mean@1 = 0.0`
- `tool_call_counts/mean@1 = 89.2`
- `search_count/mean@1 = 30.4`
- `open_count/mean@1 = 8.6`
- `find_count/mean@1 = 50.2`
- `termination_response_limit/mean@1 = 0.4`
- `termination_assistant_turn_limit/mean@1 = 0.6`
- `termination_user_turn_limit/mean@1 = 0.0`
- `response_length/mean@1 = 42252.2`
- `response_length_max/mean@1 = 61400.0`
- `response_length_ratio/mean@1 = 0.6881`
- `num_turns/min = 58`
- `num_turns/max = 240`
- `num_turns/mean = 179.6`

Per-sample result file summary:

- sample 0:
  - `termination_reason = assistant_turn_limit`
  - `response_length = 13530`
  - `response_length_ratio = 0.220`
  - `tool_call_counts = 119`
- sample 1:
  - `termination_reason = assistant_turn_limit`
  - `response_length = 42309`
  - `response_length_ratio = 0.689`
  - `tool_call_counts = 119`
- sample 2:
  - `termination_reason = response_limit`
  - `response_length = 60156`
  - `response_length_ratio = 0.980`
  - `tool_call_counts = 29`
- sample 3:
  - `termination_reason = assistant_turn_limit`
  - `response_length = 34164`
  - `response_length_ratio = 0.556`
  - `tool_call_counts = 119`
- sample 4:
  - `termination_reason = response_limit`
  - `response_length = 61102`
  - `response_length_ratio = 0.995`
  - `tool_call_counts = 60`

Reward trace summary:

- `TRACE_SHORT = 7`
  - `2` startup health checks
  - `5` real evaluation samples
- `TRACE_RESULT = 0`

Interpretation:

- increasing to `t120` disproves the overly simple claim that *all* failures were caused by a too-small turn cap
- failure mode is now mixed:
  - `60%` assistant-turn-limit
  - `40%` response-limit
- therefore:
  - `max_assistant_turns=60` was indeed too restrictive for some tasks
  - but simply removing turn cap would not solve the overall issue
  - some trajectories already consume nearly the full `64k` response budget and still fail
- the current policy is not converging efficiently enough; it can spend either too many turns or too much context without finishing

## Run 7 Deep Trajectory Analysis

Detailed inspection of the 5 output trajectories revealed the true root cause of policy failure.

### Key Observation: Zero Reasoning Text

All 5 samples produced **zero non-tool-call assistant text**. Every assistant turn was a bare `<tool_call>{...}</tool_call>` with no `<think>` block, no analysis, no final answer attempt.

Per-sample breakdown:

| Sample | GT | Termination | Tool Calls | Failure Pattern |
|--------|-----|-------------|------------|-----------------|
| 0 | "Over 70,000" | `assistant_turn_limit` | 120 (118 search, 1 open) | Search brute-force: iterated "Bloody Water" + year 1847→1863, all "No results found" |
| 1 | "A Survey of Privacy..." | `assistant_turn_limit` | 120 (16 search, 12 open, 91 find) | `find("PDF")` repeated 80+ times on empty responses |
| 2 | "Bifurcation in a..." | `response_limit` | 29 (5 search, 24 open) | Found exact answer on 1st search, then chased author chain, `open(id=0)` repeated 20+ times |
| 3 | "Panic" | `assistant_turn_limit` | 120 (12 search, 5 open, 102 find) | `find("retired")` repeated 100 times on empty responses |
| 4 | "18" | `response_limit` | 60 (1 search, 1 open, 58 find) | `find("Polish")`/`find("Polish language")` alternating 58 times, same matches each time |

Three distinct failure modes:

1. **Search brute-force loop** (sample 0): model mechanically varies one query parameter when search returns nothing
2. **find() infinite loop** (samples 1, 3, 4): model repeats identical find calls on empty/repeated responses without noticing
3. **Found answer but didn't answer** (sample 2): model found the paper title immediately but continued chasing the multi-hop chain without ever producing an answer

### Why `max_assistant_turns` Is Not the Bottleneck

No sample was "almost done" — none were close to producing a final answer. The turn-limited samples were in infinite loops, not in productive research. Increasing turns would only increase cost.

## Root Cause Analysis: SFT Reasoning Was Never Learned

Investigation identified **two independent bugs** that together explain why the model only produces tool calls with no reasoning or answers.

### Bug 1: SFT Per-Message Tokenization Drops `reasoning_content` (Critical)

**Discovery**: The raw SFT data (`deepdive-sft-glm46-trajectory-1k.jsonl`) contains complete agent trajectories with `reasoning_content` in every assistant turn (832/832 samples, 29216/29216 assistant turns) and a final answer in every sample (832/832).

The preprocessing script (`preprocess_carr_sft.py`) correctly preserves `reasoning_content` in the parquet. The parquet (`sft_train.parquet`) contains all fields.

**But**: verl's `MultiTurnSFTDataset` tokenizes each message in isolation:

```python
# multiturn_sft_dataset.py (old code)
inputs = processor.apply_chat_template(
    [message],  # single message, no conversation context
    ...
)
```

Qwen3's Jinja chat template requires full conversation context to render `reasoning_content` as `<think>...</think>` blocks. When a single assistant message with `reasoning_content + tool_calls` is rendered in isolation, the template silently drops `reasoning_content`:

```
# Single-message rendering (WRONG - what training actually saw):
<|im_start|>assistant
<tool_call>
{"name": "browser.search", "arguments": {"query": "test"}}
</tool_call><|im_end|>

# Full-conversation rendering (CORRECT - what should have been trained):
<|im_start|>assistant
<think>
We need to search for the paper title. Let me look for it.
</think>

<tool_call>
{"name": "browser.search", "arguments": {"query": "test"}}
</tool_call><|im_end|>
```

**Impact**: The SFT checkpoint learned only tool-call syntax. It never learned reasoning (`<think>` blocks) or when to stop and produce a final answer. This is the primary cause of all observed eval failures.

**Note**: The existing `test_sft_tokenization.py` validated the whole-conversation rendering path (which produces correct output), not the actual per-message training path. This created a false-positive: the test passed while training was broken.

### Bug 2: RL Config Missing `enable_thinking` (Important)

**Discovery**: `carr_grpo.yaml` had no `apply_chat_template_kwargs` section. The default from `legacy_data.yaml` is `{}`.

Code path:

```
carr_grpo.yaml → no apply_chat_template_kwargs
→ legacy_data.yaml:131 default: {}
→ agent_loop.py:221 → self.apply_chat_template_kwargs = {}
→ agent_loop.py:274,302 → apply_chat_template(**{}) → no enable_thinking
→ RL rollout prompt lacks thinking mode → model never produces <think> blocks
```

Even if Bug 1 were fixed (model learns reasoning in SFT), this config gap would suppress thinking during RL inference.

### Bug 3: Right Truncation Cuts Final Answer (Moderate)

SFT config used `truncation: right` with `max_length=65536`. For long trajectories, this cuts off the tail — which contains the final answer.

Measured on sample 0: un-truncated length = 65,986 tokens, training keeps 65,536, cutting 450 tokens from the final answer turn (which was 997 tokens). The `## Exact Answer` section was partially or fully removed.

This weakens the supervision signal for "when and how to produce a final answer."

## Fixes Applied

### Fix A: Prefix-Based Rendering for Text-Only Multi-Turn SFT

File: `verl/utils/dataset/multiturn_sft_dataset.py`

New method `_process_messages_with_context()`:

- renders the conversation prefix up to each turn: `apply_chat_template(messages[:end_idx])`
- computes the delta (newly added tokens) by subtracting the previous prefix
- applies loss mask to the delta
- includes a prefix stability assertion with actionable error message

This preserves the full in-context serialization including `<think>` blocks, tool calls, and final answers. The multimodal path (using `processor`) retains the old per-message approach because it needs per-message auxiliary tensors (pixel_values, image_grid_thw).

Verified:

- intermediate assistant turns now produce `<think>reasoning</think>\n<tool_call>...` (was: only `<tool_call>...`)
- final assistant turns now produce `<think>reasoning</think>\nfinal answer` (was: only `final answer`)

### Fix B: `loss_window` Truncation Strategy

File: `verl/utils/dataset/multiturn_sft_dataset.py`

New truncation method `loss_window`:

- computes a prefix-sum of the loss mask
- slides a window of size `max_length` across the sequence
- selects the window position with the highest count of supervised (loss=1) tokens
- on ties, prefers the latest window (biased toward keeping the final answer)
- O(n) time and space

Verified:

- right truncation on a 100-token sequence with loss at the end and max_length=50 keeps 0 loss tokens
- `loss_window` on the same sequence keeps 30 loss tokens
- sample 0's `## Exact Answer` is now retained under `loss_window`

### Fix C: RL `enable_thinking` Config

File: `examples/carr_deepsearch/config/carr_grpo.yaml`

Added:

```yaml
data:
  apply_chat_template_kwargs:
    enable_thinking: true
```

This propagates through `agent_loop.py:221` → `self.apply_chat_template_kwargs` → all `apply_chat_template()` calls in the RL rollout.

### Fix D: SFT Config Truncation Mode

File: `examples/carr_deepsearch/config/carr_sft.yaml`

Changed `truncation: right` → `truncation: loss_window`.

### Regression Tests Added

1. `examples/carr_deepsearch/scripts/test_sft_tokenization.py`:
   - extended to check `MultiTurnSFTDataset` loss-mask text for `<think>` reasoning and `## Exact Answer`
   - this now tests the actual training path, not just the whole-conversation rendering

2. `tests/utils/dataset/test_multiturn_sft_dataset_on_cpu.py`:
   - `test_reasoning_content_preserved`: verifies that a `FakeReasoningTokenizer` (simulating Qwen3's per-message reasoning drop) produces correct output under prefix-based rendering
   - `test_loss_window_preserves_tail`: verifies that `loss_window` retains more supervised tokens than `right` truncation when the answer is at the end

## Files Modified

- `verl/utils/dataset/multiturn_sft_dataset.py` — prefix-based rendering + `loss_window` truncation
- `examples/carr_deepsearch/config/carr_sft.yaml` — `truncation: loss_window`
- `examples/carr_deepsearch/config/carr_grpo.yaml` — `apply_chat_template_kwargs.enable_thinking: true`
- `examples/carr_deepsearch/scripts/test_sft_tokenization.py` — extended coverage
- `tests/utils/dataset/test_multiturn_sft_dataset_on_cpu.py` — new CPU regression tests

## Next Steps

1. **Re-run SFT training** with the fixed tokenization and `loss_window` truncation
2. **Re-run DeepDive eval** with the new checkpoint + `enable_thinking` RL config
3. Verify that the model now produces `<think>` reasoning blocks and final answers
4. If eval shows improved completion rate, proceed to formal RL training

## 8x H200 Clean RL Probe (No Val / No Save / TP1)

On the 8x H200 instance, we ran a clean 1-step RL probe to remove the previous
"last-step validation/save" confound:

- `trainer.val_before_train=false`
- `trainer.test_freq=0`
- `trainer.save_freq=0`
- `data.train_batch_size=8`
- `actor_rollout_ref.rollout.n=4`
- `actor_rollout_ref.rollout.tensor_model_parallel_size=1`
- `data.max_response_length=61400`
- `max_assistant_turns=80`
- `max_tool_response_length=6000`

Remote artifacts:

- log: `/workspace/logs/carr_cgrpo_h200_probe_clean_tp1.log`
- reward trace: `/workspace/logs/carr_reward_trace_h200_probe_clean_tp1.jsonl`

Observed result:

- all `32` real trajectories reached reward evaluation
- `15` were `evaluate_result`
- `17` were `evaluate_short_circuit`
- mean reward over `evaluate_result` was about `0.516`
- after all `32` real reward events completed, the run still did not log
  `step:1`
- GPUs remained near full utilization with about `132 GB` allocated per card

Interpretation:

- This is stronger evidence than the earlier probes because validation and
  checkpoint saving were explicitly disabled.
- `rollout + tool + reward` are no longer the only explanation for the delay;
  after rollout completion, the remaining bottleneck is downstream, most likely
  `old_log_prob` / PPO update.
- Adding more reward servers may help earlier-stage latency, but it will not
  fix this specific post-rollout stall by itself.

Practical implication:

- Do not launch formal RL yet on this recipe.
- The next useful probes should reduce training-side work further (for example
  `b4/n4`) or profile the post-rollout path directly.
