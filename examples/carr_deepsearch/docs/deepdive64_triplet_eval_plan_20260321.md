# DeepDive Subset64 Triplet Eval Plan (2026-03-21)

目标：

- 在继续 RL 之前，先用同一套 sampled eval recipe 比较三个 checkpoint：
  - `SFT checkpoint`
  - `global_step_70`
  - `global_step_90`
- 固定使用 `DeepDive rl_val` 的同一个 `subset64`
- 保留 Thinking checkpoint 的 sampled eval 配置，不退回 greedy

## 评估口径

- dataset: `examples/carr_deepsearch/data/rl_val_subset_64_seed42.parquet`
- context: `64k` (`data.max_response_length=61440`)
- sampled eval:
  - `temperature=0.6`
  - `top_p=0.95`
  - `top_k=20`
  - `do_sample=true`
- turn/tool envelope:
  - `max_assistant_turns=120`
  - `max_tool_response_length=6000` by default
- formal budgets:
  - `wall=360`
  - `tool=88`
  - `search=40`
  - `open=32`
  - `find=20`
- `validation_shuffle=false`
- `val_n=1`

## 机器 profile

### 8 x RTX 6000

- `NGPUS=8`
- `data.val_batch_size=32`
- 理由：
  - 比默认 `16` 少一半 validation batch
  - 比 `64` 更保守，避免把 tool/reward 单点压力一次性拉满

### 4 x RTX 6000

- `NGPUS=4`
- `data.val_batch_size=16`
- 理由：
  - 保持每批并发压力适中
  - 避免 4 卡机器在 full-budget deep-search eval 上过早进入 server queueing

## 执行脚本

单 checkpoint：

```bash
NGPUS=8 bash examples/carr_deepsearch/scripts/run_eval_deepdive64_sampled.sh \
  /abs/path/to/checkpoint \
  run_name_here
```

三 checkpoint 串行：

```bash
NGPUS=8 bash examples/carr_deepsearch/scripts/run_eval_deepdive64_triplet.sh \
  /abs/path/to/sft \
  /abs/path/to/step70 \
  /abs/path/to/step90 \
  dd64_cmp
```

## 结果优先看什么

主指标：

- `outcome_reward/mean@1`
- `rubric_reward/mean@1`
- `task_unfinished/mean@1`

失败模式：

- `termination_response_limit`
- `termination_rollout_timeout`
- `termination_search_budget`
- `termination_find_budget`

成本与行为：

- `rollout_elapsed_s/mean@1`
- `tool_call_counts/mean@1`
- `search/open/find_count`

## 建议比较方式

- 先直接比较 `SFT vs step70 vs step90`
- 如果 `step90` 至少在下面两项上不差于 `step70`，再考虑从 `step90` 继续 RL：
  - `outcome_reward`
  - `task_unfinished`
- 如果 `step90` 只是在 sampled eval 下略优，但 `timeout`/`budget` 明显更坏，则后续 resume 需要先做短 probe，而不是直接长跑
