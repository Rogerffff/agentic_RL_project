# CaRR DeepSearch / verl Agent Handoff Background (2026-03-21)

## 1. 这份文档的用途

这是一份给后续 agent 的 handoff 文档，目标是减少重复口头交代。

- 只写当前阶段最有用的背景信息
- 已经在其他文档里写清楚的内容，直接引用，不重复展开
- 如果某些结论来自本地归档日志和近期分析，也会在这里直接写明

---

## 2. 项目目标

当前项目目标是：

- 在 `verl` 框架下复现/实现 CaRR 风格的 deep-search agent RL
- 基于 `search/open/find` 多轮工具调用训练多跳问答 agent 的 deep search 能力
- 最终形成一个可用于简历和面试叙述的项目

项目核心不是普通单轮 RLHF，而是：

- 多轮 agent loop
- 外部工具调用
- CaRR rubric-aware reward
- C-GRPO 优势计算

论文与数据背景优先看：

- [Carr_paper_data.md](./Carr_paper_data.md)

实现与调试背景优先看：

- [project_retrospective_20260312.md](./project_retrospective_20260312.md)
- [rl_debug_findings_20260312.md](./rl_debug_findings_20260312.md)

---

## 3. 当前代码与关键入口

和当前阶段最相关的代码/脚本：

- RL 启动脚本: [../scripts/run_rl.sh](../scripts/run_rl.sh)
- async RL 启动脚本: [../scripts/run_rl_async.sh](../scripts/run_rl_async.sh)
- 8 卡 async probe wrapper: [../scripts/run_async_probe_8gpu_blackwell.sh](../scripts/run_async_probe_8gpu_blackwell.sh)
- eval 基础 launcher: [../scripts/run_eval_integration.sh](../scripts/run_eval_integration.sh)
- sampled subset64 eval: [../scripts/run_eval_deepdive64_sampled.sh](../scripts/run_eval_deepdive64_sampled.sh)
- triplet eval: [../scripts/run_eval_deepdive64_triplet.sh](../scripts/run_eval_deepdive64_triplet.sh)
- 固定子集生成脚本: [../scripts/prepare_rl_val_subset.py](../scripts/prepare_rl_val_subset.py)
- CaRR 自定义 agent loop: [../tools/carr_agent_loop.py](../tools/carr_agent_loop.py)
- CaRR reward client: [../reward/carr_reward.py](../reward/carr_reward.py)
- async base reward: [../reward/async_base_reward.py](../reward/async_base_reward.py)
- C-GRPO advantage: [../reward/cgrpo_advantage.py](../reward/cgrpo_advantage.py)
- 默认 RL 配置: [../config/carr_grpo.yaml](../config/carr_grpo.yaml)
- async 通用配置: [../config/carr_grpo_async_common.yaml](../config/carr_grpo_async_common.yaml)
- async base probe 配置: [../config/carr_grpo_async_base.yaml](../config/carr_grpo_async_base.yaml)
- async CaRR 配置: [../config/carr_grpo_async.yaml](../config/carr_grpo_async.yaml)
- 默认 SFT 配置: [../config/carr_sft.yaml](../config/carr_sft.yaml)

`verl` 侧最关键的调度路径：

- trainer rollout / reward / PPO 主流程: [../../../verl/trainer/ppo/ray_trainer.py](../../../verl/trainer/ppo/ray_trainer.py)
- agent loop manager: [../../../verl/experimental/agent_loop/agent_loop.py](../../../verl/experimental/agent_loop/agent_loop.py)
- base tool agent loop: [../../../verl/experimental/agent_loop/tool_agent_loop.py](../../../verl/experimental/agent_loop/tool_agent_loop.py)
- reward loop: [../../../verl/experimental/reward_loop/reward_loop.py](../../../verl/experimental/reward_loop/reward_loop.py)
- naive reward manager: [../../../verl/experimental/reward_loop/reward_manager/naive.py](../../../verl/experimental/reward_loop/reward_manager/naive.py)
- fully async 入口: [../../../verl/experimental/fully_async_policy/fully_async_main.py](../../../verl/experimental/fully_async_policy/fully_async_main.py)
- fully async rollouter: [../../../verl/experimental/fully_async_policy/fully_async_rollouter.py](../../../verl/experimental/fully_async_policy/fully_async_rollouter.py)
- fully async trainer: [../../../verl/experimental/fully_async_policy/fully_async_trainer.py](../../../verl/experimental/fully_async_policy/fully_async_trainer.py)
- 参数同步: [../../../verl/experimental/fully_async_policy/param_sync.py](../../../verl/experimental/fully_async_policy/param_sync.py)
- detach sync / weight sync helper: [../../../verl/experimental/fully_async_policy/base_detach_sync.py](../../../verl/experimental/fully_async_policy/base_detach_sync.py)
- async agent loop assembly: [../../../verl/experimental/fully_async_policy/agent_loop/agent_loop.py](../../../verl/experimental/fully_async_policy/agent_loop/agent_loop.py)

---

## 4. 当前状态 TL;DR

当前阶段的真实状态：

- SFT 已完成
- 同步正式 RL 训练曾在 `8 x RTX 6000 (96GB)` 上跑到约 `step 96`
- 最近一个确认存在的完整 checkpoint 是远端历史路径 `/root/checkpoints/carr_8gpu_formal/global_step_90`
- 当前主工作流已经从“继续同步 RL”转到“把 CaRR 接进 fully async policy 并验证 async_partial 是否真实生效”
- `2026-03-22` 的 `Phase 2p` 远端 probe 已经通过 async 的关键 correctness gate：
  - `fully_async/partial/total_partial_num > 0`
  - completed sample 中出现 `param_version_start != param_version_end`
  - 多次 param sync 成功，rollout 侧权重 fingerprint 实际变化
  - run 正常结束，不是被 `SIGTERM` 或 SGLang weight update 崩溃打断
- 因此当前最推荐的下一步已经不是继续修 async 基座，也不是先回到 sync long run，而是：
  - 做一轮正式 async RL short run
  - 先验证稳定性和收益，再决定是否放大到更长上下文或更长训练

本地已经保留了 formal RL 的关键归档，不必重新从远端查历史日志：

- 训练日志归档目录: [../CaRR_formal_training_log](../CaRR_formal_training_log)
- step40 eval: [../CaRR_formal_training_log/40.jsonl](../CaRR_formal_training_log/40.jsonl)
- step80 eval: [../CaRR_formal_training_log/80.jsonl](../CaRR_formal_training_log/80.jsonl)
- 旧段 wandb 输出: [../CaRR_formal_training_log/files/output.log](../CaRR_formal_training_log/files/output.log)
- 新段 wandb 输出: [../CaRR_formal_training_log/output.log](../CaRR_formal_training_log/output.log)

### 4.1 `2026-03-22` async 改造补充

这一段是给后续 agent 最重要的 async 交接。

当前已经验证过的 async 主线环境：

- `8 x RTX PRO 6000 Blackwell`
- `verlai/verl:sgl056.latest`
- `sglang 0.5.6.post2`
- `torch 2.9.x`
- 单机 `4:4` split
- `TP=1`
- `NCCL_P2P_DISABLE=1`
- `attention_backend=flashinfer`

这轮 async 改造里，真正修过并已经被 runtime 验证过的关键点：

1. `fully_async` 基座修通了真实 param sync，不再只是 version bump
2. `rollout.test_freq=0` / `trainer.test_freq=0` 的 probe 路径已安全
3. CaRR multi-turn 已接入独立的 `carr_async_partial_tool_agent`
4. partial 模式下，`max_required_samples`、`max_concurrent_samples`、`max_queue_size` 已经解耦
5. partial 模式下的 `staleness` 已经降成 soft signal，不再直接卡住 dispatch
6. `cancel_queue` 优先级已补齐，resume 样本不会再被 staleness gate 卡死
7. async probe 下显式关闭了 `actor.use_dynamic_bsz`，修掉了 FSDP `update_actor` 的 rank 间 micro-batch 不一致 hang
8. SGLang 在线权重同步已经改成：
   - `update_weights_from_tensor(..., flush_cache=False)`
   - `pause()` 时显式 `clear_kv_cache()`
   - post-sync `flush_cache()` 结果强校验
9. `DONE` 逻辑已修，不会再在 partial 样本尚未 drain 完时过早结束 processor

`Phase 2p` 最关键的 probe 结论：

- `step:2` 时 `fully_async/partial/total_partial_num = 1`
- `step:3` 时 `fully_async/partial/total_partial_num = 29`
- 已出现跨版本 completed sample：
  - `param_version_start=0, param_version_end=1`
  - `param_version_start=1, param_version_end=2`
- param sync 成功且 rollout 侧权重实际变化
- run 正常收尾

这意味着：

- async partial 不是“看起来像在 cancel/resume”，而是真的形成了 completed cross-version sample
- 当前已具备进入正式 async RL short run 的条件

当前仍需关注、但不再是 blocker 的点：

- `pause()` 等待 rollout drain 的耗时仍然偏高
- `stale_trajectory_processed` 已经非零，说明 async 确实在用 staleness 换吞吐
- 还没有完成正式 async short run 的收益评估，因此不能直接跳到昂贵长训

### 4.2 `2026-03-23` 最新状态 TL;DR

这一段优先级高于后文 `2026-03-21` 的旧建议。

当前已经完成的关键里程碑：

- `Stage 0 eval` 已完成：SFT / step70 / step90 fixed subset64 sampled eval，async 起点选 `step70`
- async infra 修复已全部验证通过：真实 param sync、partial cross-version sample、queue_full cancel-based drain、dynbsz dp_group 对齐、互斥 unfinished 指标
- 偶发的 `SGLang flush_cache failed: empty response` 会导致训练中断，直接 resume 即可恢复
- 当前最新 checkpoint 请直接查远端 `/root/checkpoints/` 下最新的 `formal_mainline_gs*` 目录
- 当前最新可直接用于 model-only restart 的 HF actor 权重：`/root/checkpoints/formal_short_gs5_qfullfix_20260323_002038/global_step_6/actor/huggingface_merged`

当前最重要的工程判断：

- `2:6 + SP=2 + dynbsz + tok36864 + wall480/960 + gmu=0.5 + b3` 是已知最稳且信号最好的 async 训练主线
- `gpu_memory_utilization=0.5` 已证实有效：timeout 可压到 0%（稳定态），rollout 不再是瓶颈
- `param_sync` 通过 queue_full cancel-based drain 从 323s 降到 1.7-20s（偶发回到 ~100s）
- 当前主瓶颈：trainer `update_actor`（SP=2 通信开销 ~250-300s/gs）+ unfinished 中 response_limit + search_budget 截断
- `3:5 + SP=1` 已证明 OOM 失败，不是正式配置

### 4.3 当前 active mainline

最新 active training 请查远端 `/root/logs/` 下最新的 `formal_mainline_gs*.launcher.log`。

主线口径（自 step 9 起稳定不变）：

- `2:6`（2 GPU trainer + 6 GPU rollout）
- `SP=2`（Ulysses Sequence Parallel）
- `use_dynamic_bsz=true` + `ppo_mini_batch_size=3` + `rollout.n=8`
- `ppo_max_token_len_per_gpu=36864`
- `activation_offload=true` / `optimizer_offload=false`
- `gpu_memory_utilization=0.5`
- `wall=480 / real_wall=960`
- `max_concurrent_samples=6` / `trigger_parameter_sync_step=4` / `staleness_threshold=0.5`

### 4.4 async RL 全 step 指标历史（step 1 → 15）

每个 step 对应 `trigger_sync_step` 个 global_steps（sync=2 时 2 gs/step，sync=4 时 4 gs/step）。

| Step | 配置变更 | 每 gs | outcome | timeout | unfinished | param_sync | 备注 |
|------|---------|-------|---------|---------|------------|-----------|------|
| 1 | 4:4 conc=12 n=8 | - | 0.0 | 100% | 100% | - | SGLang 过载，全 timeout |
| 1' | 4:4 conc=4 n=8 sync=2 | 273s | 0.44 | 53% | 53% | - | 首次有训练信号 |
| 2 | 同上 | 370s | 0.11 | 69% | 83% | 248s | param_sync 长尾暴露 |
| 3 | 4:4 conc=4 sync=4 | 315s | 0.11 | 72% | 79% | 2.1s | sync=4 绕过 self-pause |
| 4 | **2:6 SP=2** wall=480 b=4 | 515s | **0.43** | 31% | 54% | 323s | SP=2 通信慢但信号好 |
| 5 | 同上 | 406s | 0.21 | 16% | 68% | 63s | pipeline 满 |
| 6 | + queue_full fix | 408s | 0.16 | 59% | 78% | **1.7s** | qfullfix 验证 |
| 7 | + **gmu=0.5** | 382s | **0.37** | **0%** | 39% | 47s | gmu 突破！零 timeout |
| 8 | 同上 | 443s | 0.20 | 0% | 68% | - | 长序列 batch |
| 9 | + **b=3** | **317s** | **0.39** | **0%** | **31%** | 72s | b=3 加速 + 信号最佳 |
| 10 | 同上 | 298s | 0.33 | 0% | 59% | 8s | pipeline 满 |
| 11 | 同上 | 296s | 0.22 | 0% | 66% | - | 磁盘满 OOM → resume |
| 11' | resume | 335s | 0.35 | 0% | 53% | 38s | 首步效应 |
| 12 | + **指标修复** | 325s | 0.31 | 0% | 53% | - | 新 unfinished 子原因 |
| 13 | 同上 | 324s | 0.24 | 0% | 59% | 21s | 稳定 |
| 14 | 同上 | 283s | **0.39** | 0% | 53% | 209s | search_budget 12.5% |
| 15 | resume(flush_cache 崩) | 339s | 0.13→0.38 | 0→21% | 65→51% | 102s | 首步冷启动 + 恢复 |

关键趋势：
- **timeout**：从 100% → 0%（gmu=0.5 后稳定消除）
- **每 gs 速度**：从 515s → ~300s（比 sync 493s 快 37-43%）
- **outcome**：在 0.2-0.4 波动（b=3 batch 小，统计噪声大），均值 ~0.3，高于 sync 基线 0.27
- **主要截断原因**：从 timeout 主导 → response_limit + search_budget 主导（模型学到更长搜索策略）

**如何查看完整 step metrics**：

上表只列了关键指标。每个 step 的完整 metrics（包含 40+ 字段）记录在远端 launcher logs 中：

- 远端日志目录：`/root/logs/`
- 每轮训练对应一个 `*.launcher.log`
- 在日志中搜索 `step:N -` 可以找到该 step 的完整输出
- 具体哪个 step 在哪个日志里，参考 §4.4 的 step-run 映射：
  - step 1-3: `gate_b4_n8_64k_conc4_*.launcher.log` 和 `gate_b4_n8_64k_conc4_sync4_resume_*.launcher.log`
  - step 4-5: `validate_b4_n8_64k_2x6_dynbsz_sp2_tok36864_wall480_*.launcher.log`
  - step 6: `formal_short_gs5_qfullfix_*.launcher.log`
  - step 7-8: `formal_mainline_gs6_sp2_dynbsz_gmu05_*.launcher.log`
  - step 9-11: `formal_mainline_gs8_sp2_dynbsz_gmu05_b3_*.launcher.log`
  - step 11': `formal_mainline_gs10_sp2_dynbsz_gmu05_b3_*.launcher.log`
  - step 12-14: `formal_mainline_gs11_sp2_dynbsz_gmu05_b3_reasonfix_*.launcher.log`
  - step 15+: `formal_mainline_gs14_sp2_dynbsz_gmu05_b3_reasonfix_*.launcher.log`

注意：wandb 未正常工作（`/root/wandb/` 为空），所有 metrics 仅在 launcher logs 和 Ray worker logs 中。

step 15 的详细截断分布（新增互斥指标首次完整输出）：

```
step 15 (resume 后第一步，含冷启动效应):
  outcome_reward/mean:     0.125   （冷启动偏低，后续恢复）
  rubric_reward/mean:      0.011
  task_unfinished/ratio:   0.653
    unfinished_limit/ratio:  0.361   ← response_limit 主导
    unfinished_budget/ratio: 0.292   ← rollout_timeout 回升（冷启动）
    unfinished_fallback:     0.0
    unfinished_empty_history: 0.0
    unfinished_no_final_assistant: 0.0
  completion_finished_early_stop: 0.181
  completion_finished_natural:    0.167
  termination_rollout_timeout:    0.208  ← resume 首步冷启动，后续应回到 0%
  termination_response_limit:     0.083
  termination_search_budget:      0.0
  termination_unknown_limit:      0.0    ← 没有未解释的截断
  termination_unknown_budget:     0.0
  rollout_elapsed_s/mean:  356s
  timing_s/step (4gs):     1356s → 每 gs 339s
  param_sync:              102s
```

### 4.5 最近一次失败的 throughput probe

最近一次失败 probe：

- phase: `probe_b3_n8_64k_3x5_dynbsz_sp1_gs6merged_20260323_122825`
- 口径：
  - `3:5`
  - `SP=1`
  - `use_dynamic_bsz=true`
  - `ppo_mini_batch_size=3`
  - `rollout.n=8`
  - `token_len=65536`
  - `activation_offload=true`
  - `optimizer_offload=false`
  - `wall=480/960`
- 起点：
  - model-only restart from `global_step_6/actor/huggingface_merged`

必须注意：

- `3:5 + SP=1 + ppo_mini_batch_size=4 + n=8` 是**非法几何**
  - 因为 `4 * 8 = 32`
  - `balance_batch(equal_size=True)` 会按 `dp_size=3` 做等分
  - 最终触发 `AssertionError: 32 % 3 != 0`
- 因此当前 `3:5 + SP=1` 的合法 probe 至少应改成：
  - `ppo_mini_batch_size=3`
  - 或者改 `n`
- 但即便切成 `b3/n8`，也已确认会在 `loss.backward()` OOM

---

## 5. 论文口径和当前实现的关系

论文给出的 RL 数值关系是：

- rollout size = 16
- 8 samples per prompt
- global batch size = 128

见 [Carr_paper_data.md](./Carr_paper_data.md)。

当前默认 RL 配置也确实是：

- `train_batch_size = 16`
- `rollout.n = 8`

见 [../config/carr_grpo.yaml](../config/carr_grpo.yaml)。

因此：

- `b16/n8` 在语义上并不是“论文口径写错了”
- 问题不在“数字不对”
- 问题在于当前工程实现对高并发 deep-search rollout 不经济

---

## 6. 已经确认的核心结论

### 6.1 `b16/n8` 不是理论错误，而是当前实现里不经济

这是当前最重要的工程判断之一。

原因不是单点，而是组合问题：

- rollout 是多轮 agent loop，不是普通单轮 LM generate
- 整步存在同步栅栏，慢样本决定整步 wall time
- 当前实现是单 tool server、单 reward server
- 当前没有全局 cache
- deep-search 任务长尾重，样本间方差大

详见：

- [project_retrospective_20260312.md](./project_retrospective_20260312.md)

### 6.2 当前 rollout 的真正流程

当前训练 step 的简化真实流程是：

1. trainer 取一个 prompt batch
2. 按 `rollout.n` 重复成 `b * n` 个 rollout
3. `AgentLoopManager` 把这些 rollout 分发给 worker，并 `ray.get(...)` 等所有 chunk 返回
4. 每条样本进入多轮状态机：
   - `PENDING`
   - `GENERATING`
   - `PROCESSING_TOOLS`
   - 重复
   - `TERMINATED`
5. rollout 全部返回后，trainer 才继续：
   - reward
   - old log prob
   - ref log prob
   - values / advantage
   - PPO update

重要含义：

- rollout 是整步 barrier
- 最慢样本会放大整个 step wall time

### 6.3 budget 不是纯“控成本开关”，而是 reward 路径的一部分

当前 CaRR 自定义 loop 中：

- hit limit 或 hit budget 会把样本记成 `task_unfinished=True`
- reward server 收到 `task_unfinished=True` 会直接 short-circuit 返回 0 分

所以：

- budget 不能被当成纯测速 guardrail
- 如果 budget hit 比例过高，它是在改训练目标，不只是降成本

### 6.4 `max_tool_response_length=6000 -> 5000` 的改动前期有效，后期发生了 failure-mode transfer

step70 之后把：

- `max_tool_response_length: 6000 -> 5000`

结果不是纯收益。

前期效果：

- `termination_response_limit` 下降
- `task_unfinished` 下降
- 中期训练窗口表现更好

后期问题：

- 主要失败模式从 `response_limit` 转成了 `rollout_timeout`
- `hit_budget` 也明显抬升

换句话说：

- `5000` 不是立刻错误
- 但它没有根治长尾，只是把失败模式迁移了

### 6.5 Thinking checkpoint 的主 eval 不应使用 greedy

这是已经确认的结论。

对 `Qwen/Qwen3-4B-Thinking-2507`：

- greedy eval 会放大重复失败模式
- sampled eval 更接近真实行为，也更便宜

推荐 eval recipe：

- `temperature=0.6`
- `top_p=0.95`
- `top_k=20`
- `do_sample=true`

见：

- [rl_debug_findings_20260312.md](./rl_debug_findings_20260312.md)

### 6.6 当前 reward 并不是“整批完全串行逐条调用”

这个点后面有人容易误判。

真实情况是：

- `reward_loop.py` 在 batch 内部会为每个样本创建 task，并 `asyncio.gather(...)`
- 你的 `carr_reward.py` 也是 `aiohttp` 异步请求

因此 reward 不是简单的：

- `128 samples * 每样本若干 judge 调用 = 线性串行总和`

但这不改变主结论，因为当前主要瓶颈仍然通常是 rollout/barrier 和长尾，而不是“reward 完全串行”。

### 6.7 `<think>` history replay 问题仍未修

这是当前仍然存在的非修复项。

简化理解：

- 上一轮生成出的 `<think>` token 仍会残留在下一轮上下文里

这个问题没有在 sampling parameter patch 中被解决。见：

- [rl_debug_findings_20260312.md](./rl_debug_findings_20260312.md)

---

## 7. 正式 RL 训练的关键历史

正式 RL 的核心参数大意：

- `8 GPU`
- `data.train_batch_size = 8`
- `rollout.n = 4`
- `max_assistant_turns = 120`
- budgets:
  - `wall = 360`
  - `tool = 88`
  - `search = 40`
  - `open = 32`
  - `find = 20`
- step70 之后：
  - `max_tool_response_length = 5000`

### 7.1 训练窗口总结

根据本地归档日志，关键窗口大致如下。

`step21-70` 平均：

- `outcome_reward ~= 0.2656`
- `rubric_reward ~= 0.0499`
- `task_unfinished ~= 0.5719`
- `termination_response_limit ~= 0.3731`
- `termination_rollout_timeout ~= 0.0694`
- `timing_s/step ~= 493s`

`step71-90` 平均：

- `outcome_reward ~= 0.3203`
- `rubric_reward ~= 0.0545`
- `task_unfinished ~= 0.4969`
- `termination_response_limit ~= 0.2656`
- `termination_rollout_timeout ~= 0.0844`
- `timing_s/step ~= 506s`

`step91-96` 平均：

- `outcome_reward ~= 0.1667`
- `rubric_reward ~= 0.0219`
- `task_unfinished ~= 0.7344`
- `termination_response_limit ~= 0.1146`
- `termination_rollout_timeout ~= 0.5208`
- `timing_s/step ~= 531s`

解释：

- `71-90` 这段中期比前面更健康
- `91-96` 开始明显坏掉，主要坏在 timeout / budget，而不是 response limit

### 7.2 训练是否值得原样继续

当前共识判断是：

- 不建议按原配置直接继续长跑
- 如果继续，应该从 `step90` 做短 probe，而不是无脑续训

原因：

- 训练已经明显进入 `timeout/budget` 主导区间
- 成本高
- 剩余正式长跑预算不划算

### 7.3 `Stage 0 eval` 结果（2026-03-23）

`Stage 0` 已按 fixed `subset64` + sampled eval 完成，统一口径：

- `max_assistant_turns = 120`
- `max_tool_response_length = 6000`
- formal budgets 保留

三组关键结果：

- `SFT`
  - `outcome_reward = 0.296875`
  - `rubric_reward = 0.044752`
  - `task_unfinished = 0.656250`
  - `termination_rollout_timeout = 0.0`
- `step70`
  - `outcome_reward = 0.328125`
  - `rubric_reward = 0.052262`
  - `task_unfinished = 0.640625`
  - `termination_rollout_timeout = 0.0`
- `step90`
  - `outcome_reward = 0.265625`
  - `rubric_reward = 0.063315`
  - `task_unfinished = 0.671875`
  - `termination_rollout_timeout = 0.0`

结论：

- async 起点最终选 `step70`
- `step90` 不是“完全学坏”，但 sampled eval 下 finishing 明显不如 `step70`
- 后续所有 async 正式训练 / probe 都应默认以 `step70` 或其后续 async checkpoint 为起点，而不是再默认押 `step90`

### 7.4 async RL 调参与关键实验链路（2026-03-22 ~ 2026-03-23）

这一段是当前最重要的工程背景，后续 agent 不应重复走已经踩过的坑。

#### 7.4.1 CPU / 调度层

- 最早的 `4:4 + b4/n8` async run 一度卡在初始化，不是 rollout 逻辑错，而是 `RAY_NUM_CPUS=32` 不够
- 症状是：
  - `FullyAsyncRollouter` 已 `ALIVE`
  - `FullyAsyncTrainer` 长时间 `PENDING_CREATION`
  - rollout 只占 `0-3` 卡，`4-7` 卡空闲
- 已修复：
  - `examples/carr_deepsearch/scripts/run_rl_async.sh`
  - `examples/carr_deepsearch/scripts/run_async_formal.sh`
  默认 `RAY_NUM_CPUS` 提高到 `128`

#### 7.4.2 `4:4 + conc12` → rollout 过载

- `4:4 + b4/n8 + max_concurrent_samples=12` 在 CPU 修复后能真正进训练
- 但第一条有效 step 基本是：
  - `outcome_reward = 0`
  - `task_unfinished = 1.0`
  - `termination_rollout_timeout = 1.0`
- 根因不是 response/tool budget，而是 rollout 并发过高：
  - `12 concurrent samples * n=8 = 96` 条 in-flight trajectory
  - `4` 个 rollout server 平摊约 `24/server`
- 结论：
  - async 不是先放大 budget
  - 先把 rollout 并发压回可用区间

#### 7.4.3 `conc4` 生效，但暴露 `param_sync` 长尾

- `4:4 + conc4` 后，训练信号显著恢复：
  - `reward > 0`
  - `unfinished` 从 `1.0` 降到约 `0.53`
  - `timeout` 从 `1.0` 降到约 `0.53`
- 但随后暴露出真正的 infra 问题：
  - `timing_s/param_sync = 248s`
  - `timing_s/param_sync = 282s`
- 拆分后发现慢点主要在 `pause/drain`，不是 `sync_weights` 本身
- 因此：
  - `trigger_parameter_sync_step = 2 -> 4`
  - 这一步是正确且保留至今的关键改动

#### 7.4.4 `2:6 + conc6 + sync4`：trainer OOM 链路

随后开始试图把 async 做出真正吞吐收益，切到：

- `2:6`
- `conc6`
- `sync4`

这条线的结论：

- rollout 侧确实更顺
- 但 trainer 在 `2 GPU` 下反复 OOM
- 尝试过：
  - `ppo_max_token_len_per_gpu: 24576 -> 18432 -> 16384`
  - `activation_offload=true`
  - `optimizer_offload=true`
- 真正把 `2 GPU trainer` 跑稳的关键不是继续盲压 token，而是：
  - `SP=2`

#### 7.4.5 dynbsz 修复

为了解决 `use_dynamic_bsz=true` 在 FSDP / DP 下 rank 间 micro-batch 不一致的问题，本地已修：

- `dp_group` 从 FSDP worker 显式传入 actor / ref / critic
- 4 个 `prepare_dynamic_batch(...)` 调用点统一走：
  - `dp_group=self.dp_group`
  - `same_micro_num_in_dp=True`
- 补了 debug 观测：
  - `len(micro_batches)`
  - `max_token_len`
  - `local max_seq_len`
  - `total_response_tokens`

当前对 dynbsz 的判断：

- `SP=1` 时已经跑到真实 dynbsz actor update
- 两个 rank 的 `len(micro_batches)` 已经对齐
- 说明这次 dynbsz 修复大概率是对的
- 但 `2:6 + SP=1` 仍然会因为纯显存峰值在 backward OOM

#### 7.4.6 目前质量最好的 async 主线

截至目前，质量最好的 async 主线配置见 §4.3。

在 `gmu=0.5 + b=3` 之后（step 9 起），稳定态指标大致为：

- `outcome_reward` 均值 ~0.30-0.39（波动大，batch 小）
- `rubric_reward` 最高达 0.12（step 9）
- `task_unfinished` 均值 ~0.50-0.65
- `termination_rollout_timeout = 0%`（稳定态，resume 首步可能短暂回升）
- `timing_s/step` 每 gs ~280-335s（比 sync 493s 快 32-43%）
- `trainer/idle_ratio` ~0.001（pipeline 满时）到 ~0.36（首步效应）
- `param_sync` ~1.7-20s（偶发 ~100-200s）

#### 7.4.7 已验证失败的配置（不要再试）

- `4:4 + conc=12`：SGLang 过载，100% timeout
- `3:5 + SP=1`：3 GPU trainer backward OOM（64k 序列太大）
- `2:6 + SP=1`：2 GPU trainer backward OOM
- `2:6 + SP=2 + tok=36864 + 无 dynbsz`：间歇性 OOM（dynbsz 对齐问题导致 FSDP hang 或 OOM）
- `optimizer_offload`：不加速 update_actor（瓶颈是 SP=2 通信不是 optimizer 传输）

#### 7.4.8 已知偶发问题

- `SGLang flush_cache failed: empty response`：param_sync 后 SGLang HTTP server 偶发返回空响应，导致训练中断。直接 resume 即可。如果频繁出现（>10%），考虑在 `ensure_sglang_flush_cache_succeeded()` 里加重试。
- 磁盘满导致 `torch.save` 崩溃：每个 checkpoint ~24GB，`save_freq=1` 时需要定期清理旧 checkpoint，只保留最新 2-3 个。
- 但吞吐仍不够理想，`update_actor` 仍是主瓶颈之一

#### 7.4.7 `queue_full self-pause` 问题与修复

在 `fully_async_rollouter.py` 里确认过一个真实 infra 问题：

- `queue_full` 内部 pause 的语义和公开 `pause()/resume()` 不一致
- 旧逻辑会在 queue 满时等待 active tasks 自然完成，放大长尾
- `cancel_queue` 检查还会绕过真正的 queue 背压

当前修复内容：

- `_get_pause_reason()` 改成先判 `queue_full`，再看 `cancel_queue`
- `queue_full + partial_rollout` 改成 cancel-based drain
- 引入 `_queue_full_cancel_pending`
- 只有在真正退出 `paused` 状态后才清 `cancellation_event`
- 避免与公开 param-sync `pause()/resume()` 产生竞态

当前结论：

- 这次修复**没有引入回归**
- 但还没有在一个明确触发 `queue_full` 的成功 short run 里被完整 exercise
- 所以现阶段应写成：
  - “已修且未回归”
  - 不应写成“已被完整正式验证”

#### 7.4.8 `global_step_5 -> global_step_6` 的 qfullfix short run

当前最近一次正式 short run 是：

- phase: `formal_short_gs5_qfullfix_20260323_002038`
- 起点：
  - `/root/checkpoints/validate_b4_n8_64k_2x6_dynbsz_sp2_tok36864_wall480_resume_gs3_20260322_225400/global_step_5`
- 已正常结束并保存：
  - `/root/checkpoints/formal_short_gs5_qfullfix_20260323_002038/global_step_6`

这轮最关键的聚合 step（`step:6`）是：

- `outcome_reward = 0.15625`
- `rubric_reward = 0.02289`
- `task_unfinished = 0.78125`
- `unfinished_budget = 0.76042`
- `unfinished_limit = 0.02083`
- `unfinished_fallback = 0.0`
- `completion_finished_early_stop = 0.13542`
- `completion_finished_natural = 0.08333`
- `termination_rollout_timeout = 0.59375`
- `termination_response_limit = 0.0`
- `termination_real_rollout_timeout = 0.0`
- `termination_max_param_span = 0.0`
- `timing_s/gen = 710.67`
- `timing_s/update_actor = 920.89`
- `timing_s/step = 1631.59`
- `timing_s/param_sync = 1.70`
- `trainer/idle_ratio = 0.4356`
- `perf/throughput = 310.15`

这轮的重要解释：

- `param_sync` 长尾问题已经基本压住
- 当前 unfinished 绝大部分来自 `budget`
- 其中最具体的 unfinished 原因是：
  - `termination_rollout_timeout`
- 不是：
  - `64K response`
  - `search/open/find/tool_call budget`

#### 7.4.9 `3:5 + SP=1 + dynbsz` throughput probe

为了判断 `update_actor` 能否继续提速，开始试：

- `3:5`
- `SP=1`
- `use_dynamic_bsz=true`

需要特别注意的几件事：

- `3:5 + SP=1 + ppo_mini_batch_size=4 + n=8` 不合法
  - 因为总轨迹数 `4 * 8 = 32`
  - `balance_batch(equal_size=True)` 要按 `dp_size=3` 等分
  - 触发：
    - `AssertionError: 32 % 3 != 0`
- 因此这条线的合法 probe 至少应改成：
  - `ppo_mini_batch_size=3`
  - `n=8`
  - 即总轨迹数 `24`

当前最新 `3:5 + SP=1` probe：

- phase: `probe_b3_n8_64k_3x5_dynbsz_sp1_gs6merged_20260323_122825`
- 起点：
  - model-only restart from `global_step_6/actor/huggingface_merged`
- 这条线的目标不是直接取代主线，而是单独验证：
  - `update_actor` 能否明显下降
  - `SP=1 + dp_group=3` 下 dynbsz 是否仍稳定

但当前真实结果已经可以写死：

- `3:5 + SP=1 + dynbsz + b3/n8` 虽然几何合法，且能进入真实 `update_actor`
- 但会在 `loss.backward()` 处 OOM
- 因此这条线当前不是“待定候选”，而是已知失败的 throughput probe

#### 7.4.10 当前正式主线：`b3 + gmu=0.5 + reasonfix`

当前正式主线已经推进到：

- phase:
  - `formal_mainline_gs11_sp2_dynbsz_gmu05_b3_reasonfix_20260323_091930`
- 最新 durable checkpoint:
  - `/root/checkpoints/formal_mainline_gs11_sp2_dynbsz_gmu05_b3_reasonfix_20260323_091930/global_step_13`

当前正式主线配置：

- `2:6`
- `SP=2`
- `use_dynamic_bsz=true`
- `ppo_mini_batch_size=3`
- `rollout.n=8`
- `token_len=36864`
- `activation_offload=true`
- `optimizer_offload=false`
- `gpu_memory_utilization=0.5`
- `wall=480`
- `real_wall=960`

这条线已经稳定证明两件事：

1. `gpu_memory_utilization=0.5` 值得保留

- rollout 卡显存从早期 `~34GB` 提升到 `~52GB`
- `termination_rollout_timeout` 可压到 `0`
- rollout 不再是主瓶颈

2. `ppo_mini_batch_size=3` 比 `4` 更适合作为当前主线

- `update_actor` 明显下降
- trainer 显存压力明显下降
- 训练信号没有坏掉，整体更稳

当前最关键的两条聚合 step：

`step:12`

- `outcome_reward = 0.3056`
- `task_unfinished = 0.5278`
- `unfinished_limit = 0.2778`
- `unfinished_budget = 0.2500`
- `unfinished_empty_history = 0`
- `unfinished_no_final_assistant = 0`
- `termination_response_limit = 0.1667`
- `termination_search_budget = 0.0417`
- `termination_rollout_timeout = 0`
- `termination_unknown_limit = 0`
- `termination_unknown_budget = 0`
- `timing_s/gen = 496.45`
- `timing_s/update_actor = 802.96`
- `timing_s/step = 1299.44`
- `timing_s/param_sync = 144.03`

`step:13`

- `outcome_reward = 0.2396`
- `task_unfinished = 0.5938`
- `unfinished_limit = 0.3750`
- `unfinished_budget = 0.2188`
- `unfinished_empty_history = 0`
- `unfinished_no_final_assistant = 0`
- `termination_response_limit = 0.2917`
- `termination_search_budget = 0.0`
- `termination_rollout_timeout = 0`
- `termination_unknown_limit = 0`
- `termination_unknown_budget = 0`
- `timing_s/gen = 0.84`
- `timing_s/update_actor = 1149.77`
- `timing_s/step = 1294.68`
- `timing_s/param_sync = 20.73`
- `trainer/idle_ratio = 0.00065`
- `fully_async/count/stale_samples_processed = 7`
- `fully_async/count/stale_trajectory_processed = 70`
- `fully_async/count/dropped_stale_samples = 0`
- `fully_async/partial/partial_ratio = 0.0208`

当前主线的结论：

- unfinished 已确认不是：
  - `rollout_timeout`
  - `real_rollout_timeout`
  - `max_param_span`
  - `empty_history`
  - `no_final_assistant`
- 也没有证据表明当前 unfinished 来自 async stale 样本被真正丢弃
- 当前最稳定的主瓶颈已经是 `update_actor`
- `param_sync` 仍会有窗口级长尾，但不再是最核心瓶颈

---

## 8. 已有评估结果

### 8.1 本地已有两次 formal eval 结果

- step40: [../CaRR_formal_training_log/40.jsonl](../CaRR_formal_training_log/40.jsonl)
- step80: [../CaRR_formal_training_log/80.jsonl](../CaRR_formal_training_log/80.jsonl)

### 8.2 关键数字

step40 eval:

- `n = 32`
- `score / outcome_reward mean = 0.34375`
- `rubric_reward mean ~= 0.0570`
- `task_unfinished = 0.59375`

step80 eval:

- `n = 64`
- `score / outcome_reward mean = 0.265625`
- `rubric_reward mean ~= 0.0533`
- `task_unfinished = 0.671875`

解释：

- 不能把这两个点当作严格同分布 benchmark 曲线，因为样本数和抽样窗口不同
- 但方向上并不支持“继续原样长跑就会自然变好”

---

## 9. 当前最推荐的下一步

当前最推荐的动作顺序已经更新成：

1. 保留 `2:6 + SP=2 + dynbsz + gmu=0.5 + b3` 作为当前正式训练主线
2. 继续 strict resume 跑这条主线，不要再频繁切 split / SP
3. `3:5 + SP=1` 暂停，不再当主线继续烧卡
4. 不再回到同步 RL 主线做长跑

当前主线 checkpoint / 权重基线：

- 最新 strict-resume 基线：
  - `/root/checkpoints/formal_mainline_gs11_sp2_dynbsz_gmu05_b3_reasonfix_20260323_091930/global_step_13`
- 最新 model-only restart 基线：
  - `/root/checkpoints/formal_short_gs5_qfullfix_20260323_002038/global_step_6/actor/huggingface_merged`

当前最推荐的主线训练口径：

- `2:6`
- `conc6`
- `sync4`
- `SP=2`
- `use_dynamic_bsz=true`
- `ppo_mini_batch_size=3`
- `activation_offload=true`
- `optimizer_offload=false`
- `token_len = 36864`
- `gpu_memory_utilization = 0.5`
- `wall = 480`
- `real_wall = 960`

这条线的定位：

- 它不是理论上的最终吞吐最优
- 但它是截至目前唯一同时满足：
  - 可 strict resume 连续推进
  - rollout 不再是主瓶颈
  - `termination_rollout_timeout` 可以压到 `0`
  - `b3` 后 trainer 显存压力更健康
  - 训练信号仍然可用

当前 throughput probe 口径：

- `3:5`
- `SP=1`
- `use_dynamic_bsz=true`
- `ppo_mini_batch_size=3`
- `n=8`
- `token_len=65536`
- `activation_offload=true`
- `optimizer_offload=false`

这条 probe 的目标非常明确：

- 只回答：
  - `update_actor` 能否明显比 `2:6 + SP=2` 更快
  - `dp_group=3` 下 dynbsz 是否依旧稳定
- 当前它已经证明：
  - 几何可以合法
  - 但 `b3/n8` 下仍会 OOM
- 因此它不是正式主线，不要直接拿它替代当前 `2:6 + SP=2`

不建议当前直接做的事：

- 回到同步 `b16/n8` 长训
- 继续大改 partial / queue / sync 语义
- 在 `queue_full` 修复还没被真实 exercise 前就宣称它“正式验证通过”
- 把 `actor/huggingface` 这种只含 tokenizer/config 的目录误当成可直接加载的 HF 权重
- 改 split / SP / world size 后仍然尝试 strict resume

## 10. 如果后续继续 RL，当前建议是什么

当前建议已经从“先修 async correctness”切换成“基于已有 async 主线做更有效的 short validation / probe”。

### 10.1 严格恢复与 model-only restart 的区别

后续 agent 必须先分清：

- `strict resume`
  - 用 `trainer.resume_mode=resume_path`
  - 恢复：
    - model
    - optimizer
    - lr scheduler
    - rng
    - `current_param_version`
    - dataloader / trainer state
- `model-only restart`
  - 只恢复 actor 权重
  - 不恢复 optimizer / scheduler / async runtime state

因此：

- 训练 world size / split / SP **不变**时，优先 strict resume
- 一旦改：
  - `2:6 -> 3:5`
  - `SP=2 -> SP=1`
  - trainer GPU 数变化
  就**不能** strict resume，只能 model-only restart

### 10.2 当前建议的训练分工

建议把当前训练工作分成两条线：

1. 质量主线

- 基于：
  - `/root/checkpoints/formal_mainline_gs11_sp2_dynbsz_gmu05_b3_reasonfix_20260323_091930/global_step_13`
- 用 strict resume 继续：
  - `2:6 + SP=2 + dynbsz + tok36864 + gmu=0.5 + b3 + wall480/960`
- 目标：
  - 继续稳定拿训练信号
  - 观察 `unfinished_limit / unfinished_budget / update_actor / param_sync`

2. 吞吐 probe 线

- 基于：
  - `/root/checkpoints/formal_short_gs5_qfullfix_20260323_002038/global_step_6/actor/huggingface_merged`
- 用 model-only restart 继续：
  - `3:5 + SP=1 + dynbsz + b3/n8`
- 目标：
  - 判断 `update_actor` 能否明显下降
  - 判断 `dp_group=3` dynbsz 是否稳定
- 当前状态：
  - 已确认会 OOM
  - 暂停，不继续当主线推进

### 10.3 当前判断中最重要的三个工程结论

1. `param_sync` 已不再是当前主瓶颈

- 旧问题曾到 `248s / 282s`
- 现在健康窗口已降到 `~1.7s - 3s`

2. 当前 unfinished 的主坏因已经转成 `limit + budget`

- 已不再是 `rollout_timeout`
- 也不是 async stale / cross-version drop
- 但 unfinished family 与 `termination_*` one-hot 仍未完全一一对齐，需要继续排

3. 当前想提速，主要矛头应指向：

- `update_actor`
- 以及 rollout 长尾

而不是再去重复怀疑：

- reward 是否串行
- response cap 是否主因
- old `param_sync` 是否仍是唯一问题

---

## 11. 这次 triplet eval 脚本的重要实现状态

这是后续 agent 很容易忽略的一个点。

`run_eval_deepdive64_sampled.sh` 之前有过一个 bug：

- wrapper 里解析好的默认 `CARR_*` 参数没有传到子脚本

这个问题已经修过。

当前状态：

- `NGPUS`
- `CARR_VAL_BATCH_SIZE`
- sampled eval 参数
- formal budget
- `max_tool_response_length`
- `flashinfer`

都已经在 wrapper 中显式 `export` 给 `run_eval_integration.sh`。

所以现在：

- `4 卡 -> val_batch_size=16`
- `8 卡 -> val_batch_size=32`

会真实生效，不再是 shell local 丢失。

## 12. 当前需要注意的问题清单（给后续 agent）

### 12.1 fully async checkpoint 语义

fully async 里：

- 日志里的 `global_steps` 不是 durable checkpoint id
- durable checkpoint 实际按 `current_param_version` 编号
- `latest_checkpointed_iteration.txt` 写的也是 `current_param_version`

因此：

- “已经跑到 `global_steps: 24`” 不等于磁盘上会有 `global_step_24`
- 当前这类 run 常见的是：
  - 4 个 local updates
  - 才对应 1 次参数版本推进

### 12.2 `actor/huggingface` 不一定有真正权重

后续 agent 不要再踩这个坑：

- `.../actor/huggingface` 目录有时只有：
  - tokenizer
  - config
  - 但没有 `model.safetensors` / `pytorch_model.bin`
- 真正可直接 model-only restart 的路径应优先检查：
  - `.../actor/huggingface_merged`

当前最近一次可直接用的 merged actor 是：

- `/root/checkpoints/formal_short_gs5_qfullfix_20260323_002038/global_step_6/actor/huggingface_merged`

### 12.3 磁盘空间是真风险

远端曾出现：

- `No space left on device`

并且一度只剩 `~5GB` 可用空间。

当前已经清过一批旧 checkpoint，最近一次确认空闲空间已回到 `~184GB`。  
但后续如果继续：

- merge 新 actor
- 保留多轮 short run
- 再做 eval

仍要定期检查磁盘。

### 12.4 `queue_full` 修复的验证状态

当前必须写得准确：

- 修复已经合入
- 且当前 short run 没有表现出回归
- 但还没有在一个“明确触发 `queue_full` 且训练成功完成”的窗口里被完整 exercise

所以：

- 可以写“已修且未见回归”
- 不要写“已被完全正式验证通过”

### 12.5 `3:5 + SP=1` 是 throughput probe，不是当前正式主线

当前正式主线仍是：

- `2:6 + SP=2 + dynbsz`

`3:5 + SP=1` 的价值是：

- 探测 `update_actor` 能不能更快

它当前还没有被证明：

- 更稳
- 更快
- 更适合作为正式长训主线

### 12.6 新日志语义优先读这些字段

现在判断 unfinished/completion，优先看：

- `unfinished_budget`
- `unfinished_limit`
- `unfinished_fallback`
- `completion_finished_early_stop`
- `completion_finished_natural`
- `termination_rollout_timeout`
- `termination_real_rollout_timeout`
- `termination_max_param_span`
- `termination_unknown_limit`
- `termination_unknown_budget`

不要再只盯：

- `hit_budget`
- `hit_limit`

因为那两个是聚合标签，不够细。

### 12.7 unfinished 归因仍有一个未完全解决的对齐问题

当前新日志已经能确认：

- unfinished 不是 `empty_history`
- 不是 `no_final_assistant`
- 不是 `real_rollout_timeout`
- 不是 `max_param_span`
- 也不是 `termination_unknown_limit` / `termination_unknown_budget`

但在最近窗口里仍然能看到：

- `unfinished_limit` 大于显式 `termination_response_limit + turn_limit`
- `unfinished_budget` 大于显式 `termination_search/open/find/tool_call/rollout_timeout`

这说明：

- 当前 unfinished family 与更细 one-hot termination 指标之间，仍有一部分聚合语义没有完全对齐
- 这不是 async stale / cross-version 丢样本的证据
- 后续如果要精确分析 unfinished 构成，仍需要继续排这层日志语义

## 13. 给下一个 agent 的建议阅读顺序

如果你刚接手这个项目，建议按下面顺序进入：

1. 先看本文件
2. 再看 [project_retrospective_20260312.md](./project_retrospective_20260312.md)
3. 再看 [rl_debug_findings_20260312.md](./rl_debug_findings_20260312.md)
4. 再看 [deepdive64_triplet_eval_plan_20260321.md](./deepdive64_triplet_eval_plan_20260321.md)
5. 再看：
   - [../scripts/run_async_formal.sh](../scripts/run_async_formal.sh)
   - [../scripts/run_rl_async.sh](../scripts/run_rl_async.sh)
   - [../tools/carr_agent_loop.py](../tools/carr_agent_loop.py)
   - [../../../verl/experimental/fully_async_policy/fully_async_rollouter.py](../../../verl/experimental/fully_async_policy/fully_async_rollouter.py)
   - [../../../verl/experimental/fully_async_policy/fully_async_trainer.py](../../../verl/experimental/fully_async_policy/fully_async_trainer.py)
6. 如果接手 dynbsz / trainer 内存问题，再看：
   - [../../../verl/workers/actor/dp_actor.py](../../../verl/workers/actor/dp_actor.py)
   - [../../../verl/workers/critic/dp_critic.py](../../../verl/workers/critic/dp_critic.py)
   - [../../../verl/workers/fsdp_workers.py](../../../verl/workers/fsdp_workers.py)

## 14. 一句话交接

当前项目已经进入“以 `2:6 + SP=2 + dynbsz + gmu=0.5 + b3` 为正式训练主线、从 `global_step_13` 继续 strict resume，同时把 unfinished 细粒度归因继续补齐”的阶段。
