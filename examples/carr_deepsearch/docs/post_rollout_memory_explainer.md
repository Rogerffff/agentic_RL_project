# verl RL 训练完整显存分析：从 Rollout 到 Policy Update

本文面向初学者，从数学原理到代码实现，完整讲解 verl 框架中 RL 训练**每个阶段**的显存使用、关键参数、以及调优方法。

---

## 目录

**Part I — 全流程显存分析**

- [A. 完整 RL 训练 step 的显存时间线](#a-完整-rl-训练-step-的显存时间线)
- [B. Phase 1: Rollout（SGLang 推理）](#b-phase-1-rollout-sglang-推理)
- [C. Phase 2: 模式切换（Rollout → Training）](#c-phase-2-模式切换-rollout--training)
- [D. Phase 3: Reward 计算](#d-phase-3-reward-计算)
- [E. Phase 4: old_log_prob 计算](#e-phase-4-old_log_prob-计算)
- [F. Phase 5: ref_log_prob 计算](#f-phase-5-ref_log_prob-计算)
- [G. Phase 6: Advantage 计算](#g-phase-6-advantage-计算)
- [H. Phase 7: update_actor（PPO 梯度更新）](#h-phase-7-update_actor-ppo-梯度更新)
- [I. Rollout 阶段 GPU 利用率波动的原因](#i-rollout-阶段-gpu-利用率波动的原因)
- [J. 所有显存相关配置参数速查表](#j-所有显存相关配置参数速查表)

**Part II — Post-Rollout 深度解析（old_log_prob / entropy / dynamic batching）**

1. [训练主循环中 post-rollout 的完整顺序](#1-训练主循环中-post-rollout-的完整顺序)
2. [old_log_prob 的完整代码路径](#2-old_log_prob-的完整代码路径)
3. [log_prob 是怎么算的](#3-log_prob-是怎么算的)
4. [entropy 是怎么算的](#4-entropy-是怎么算的)
5. [显存如何估算](#5-显存如何估算)
6. [dynamic batching 是如何实现的](#6-dynamic-batching-是如何实现的)
7. [为什么 b2/n4 在 8 卡 RTX 6000 上会 OOM](#7-为什么-b2n4-在-8-卡-rtx-6000-上会-oom)
8. [解决方向](#8-解决方向)

---

# Part I — 全流程显存分析

## A. 完整 RL 训练 step 的显存时间线

以 CaRR 配置（8x GPU, TP=1, FSDP=8, Qwen3-4B bf16）为例，一个 training step 的显存变化如下：

```
时间 ──────────────────────────────────────────────────────────────────────────►

Phase:    rollout_mode  |  rollout (gen)  |  trainer_mode  |  reward  |  old_log_prob  |  ref_log_prob  |  adv  |  update_actor  |
              |              |                  |              |             |                |            |            |
显存组成:     |              |                  |              |             |                |            |            |
              |              |                  |              |             |                |            |            |
SGLang 权重   ████████████████████████████████████              |             |                |            |            |
KV cache      ████████████████████████████████████              |             |                |            |            |
FSDP actor    ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
FSDP ref      ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
优化器                                                                                                         ████████████████
梯度                                                                                                           ████████████████
logits 峰值                                                     ▲▲▲▲▲▲▲▲      ▲▲▲▲▲▲                          ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
entropy 峰值                                                    ▲▲▲▲▲▲▲▲                                       (可能有)

总显存   ──╱──────────────╲──────────────────────────────╱▲╲──────────────╱▲╲──────────────────────╱▲▲▲╲──
                  rollout 峰值           reward 几乎      old_log_prob    ref_log_prob             update_actor
                  (SGLang 占主)          不用 GPU          峰值(entropy)   峰值(无 entropy)         峰值(梯度+优化器)
```

**关键观察：显存不是恒定的，而是在各阶段之间"时间复用"。** SGLang 占用的显存在训练模式下被释放，
然后被 FSDP 前向/反向传播的激活和 logits 张量所占用。

---

## B. Phase 1: Rollout（SGLang 推理）

代码位置：`verl/workers/fsdp_workers.py` 第 740-845 行（`rollout_mode`），
`verl/workers/rollout/sglang_rollout/async_sglang_server.py`

### 显存组成

SGLang 作为推理引擎，显存主要由三部分组成：

```
SGLang 显存 = 模型权重(bf16) + KV cache + 临时激活
```

KV cache 的大小由 `gpu_memory_utilization` 控制（`async_sglang_server.py` 第 186 行）：

```python
"mem_fraction_static": self.config.gpu_memory_utilization,  # 默认 0.5
```

这意味着 SGLang 会占用 **50% 的 GPU 显存**作为静态分配（模型权重 + KV cache）。
剩余 50% 留给 FSDP 训练权重和优化器状态。

### 与 FSDP 权重共存

在 `hybrid_engine=true`（CaRR 默认配置）下，rollout 期间 GPU 上同时存在：

| 张量 | 大小（Qwen3-4B, 8卡） | 说明 |
|------|----------------------|------|
| SGLang 模型权重 | ~8 GiB（完整副本） | SGLang 每个 replica 加载完整模型 |
| KV cache | `gpu_mem * 0.5 - 模型权重` | 剩余静态空间全给 KV cache |
| FSDP sharded actor 权重 | ~1 GiB (8G/8卡) | 如果 `param_offload=false`，仍留在 GPU |
| FSDP sharded ref 权重 | ~1 GiB (8G/8卡) | 同上 |
| 优化器状态 | ~2 GiB (16G/8卡) | Adam 的 m 和 v，sharded |

**Rollout 峰值估算（96 GB RTX 6000 Pro）：**

```
SGLang 静态分配: 96 * 0.5 = 48 GiB
FSDP actor sharded: ~1 GiB
FSDP ref sharded: ~1 GiB
优化器状态: ~2 GiB
总计: ~52 GiB / 96 GiB  (利用率约 54%)
```

### 多轮 agentic rollout 的特殊性

CaRR 使用多轮 agent loop（`multi_turn.enable=true`），每个样本会经历多轮：
生成 → 工具调用 → 生成 → 工具调用 → ... → 终止。

每一轮生成都会增长 KV cache。如果一个 request 经历 20 轮工具调用，
它的 KV cache 会从 prompt 长度逐渐增长到接近 `max_response_length`。

---

## C. Phase 2: 模式切换（Rollout → Training）

代码位置：`verl/workers/fsdp_workers.py` 第 1048-1078 行

### 切换过程

rollout 完成后，hybrid engine 执行 `trainer_mode()` 切换：

```
1. aggressive_empty_cache()          -- 清理 CUDA cache
2. SGLang release(tags=["weights"])  -- 释放 SGLang 模型权重
3. SGLang release(tags=["kv_cache"])-- 释放 KV cache
4. output = output.to("cpu")        -- rollout 结果移到 CPU
```

配置 `free_cache_engine=true`（默认）时，SGLang 的权重和 KV cache 都会被释放，
腾出的空间用于后续训练阶段。

### 瞬态峰值

在 `rollout_mode()` 进入时（下一步开始前的准备），需要同时做权重同步：
- 从 FSDP 提取完整 state_dict
- 拷贝到 SGLang 的权重 buffer

这个过程有一个瞬态峰值，但 `update_weights_bucket_megabytes=2048` 控制每次只同步 2 GB，
避免一次性复制全部权重。

---

## D. Phase 3: Reward 计算

代码位置：`verl/workers/reward_manager/naive.py` 第 46-122 行

### GPU 显存：几乎为零

reward 阶段完全在 CPU 上运行：
- rollout 结果已经被移到 CPU（`output.to("cpu")`）
- `NaiveRewardManager` 逐样本 decode tokens，调用 reward function
- CaRR 的 reward function 是发 HTTP 请求到 reward server（`CARR_REWARD_SERVER_URL`）
- 返回的 `reward_tensor` 是 float32 CPU 张量

GPU 上此时只有 FSDP sharded 的 actor/ref 权重 + 优化器状态（不活跃，只是驻留）。

**这是整个 step 中 GPU 显存最低的阶段。**

---

## E. Phase 4: old_log_prob 计算

代码位置：`verl/trainer/ppo/ray_trainer.py` 第 1132-1155 行

### 显存组成

```
old_log_prob 峰值 = FSDP 权重(all-gather 峰值)
                  + 前向激活(no_grad, 无反向图)
                  + logits(total_nnz * vocab_size * 2)
                  + entropy 峰值(logits * 3，如果 calculate_entropy=True)
```

**关键发现：`calculate_entropy=True` 是硬编码的（`ray_trainer.py` 第 1140 行），
但计算出的 entropy 只用于记录 `actor/entropy` 指标，随后被 `pop` 丢弃（第 1416 行）。
这是纯粹的显存浪费。**

详细分析见 Part II 第 2-5 节。

---

## F. Phase 5: ref_log_prob 计算

代码位置：`verl/trainer/ppo/ray_trainer.py` 第 1105-1130 行，
`verl/workers/fsdp_workers.py` 第 1132-1164 行

### 与 old_log_prob 的区别

| 对比项 | old_log_prob | ref_log_prob |
|--------|-------------|--------------|
| 模型 | actor (当前 policy) | ref (冻结的参考 policy) |
| calculate_entropy | **True**（硬编码） | **False** |
| logits 峰值 | total_nnz * V * 6 (含 entropy) | total_nnz * V * 2 (仅 logits) |
| micro_batch_size | `rollout.log_prob_micro_batch_size_per_gpu` | `ref.log_prob_micro_batch_size_per_gpu` (CaRR: 2) |

### 显存特殊考虑

ref 模型和 actor 模型是**两套独立的 FSDP 权重**，同时驻留在 GPU 上
（除非使用 LoRA，此时 `ref_in_actor=true`，ref 复用 actor 权重）。

CaRR 不用 LoRA，所以：

```
ref_log_prob 显存 = FSDP sharded actor 权重 (~1G)    <-- 仍然驻留
                  + FSDP sharded ref 权重 (~1G)       <-- 正在使用
                  + ref all-gather 临时权重 (~8G)      <-- 前向时 all-gather
                  + 优化器状态 (~2G)                    <-- 仍然驻留
                  + logits (total_nnz * V * 2)         <-- 无 entropy！
                  + 前向激活 (~3-5G)
```

因为没有 entropy，显存比 old_log_prob 阶段低很多。

---

## G. Phase 6: Advantage 计算

代码位置：`verl/trainer/ppo/ray_trainer.py` 第 1489-1504 行

**完全在 driver 进程（CPU）上运行，不消耗 GPU 显存。**

CaRR 使用 `adv_estimator=cgrpo`，调用注册在 `examples.carr_deepsearch.reward.cgrpo_advantage`
中的自定义优势估计器。计算公式：

```
R = (1-alpha)*R_outcome + alpha*R_outcome*R_rubric_normalized
```

所有操作都是 CPU 上的标量/向量运算。

---

## H. Phase 7: update_actor（PPO 梯度更新）

代码位置：`verl/workers/actor/dp_actor.py` 第 508-676 行

### 这是显存最紧张的阶段

因为需要同时存在：模型权重 + 优化器状态 + 梯度 + 前向激活（需要保留用于反向传播）。

```
update_actor 峰值 = FSDP sharded actor 权重 (~1G)
                  + FSDP sharded ref 权重 (~1G)       <-- 仍然驻留
                  + all-gather 临时完整权重 (~8G)       <-- 前向时需要
                  + 优化器状态 (~2G, Adam m+v)
                  + 梯度 (~1G, sharded)
                  + 前向激活 (需要保留给 backward!)
                  + logits (micro_batch_tokens * V * 2)
                  + entropy 计算 (如果 entropy_coeff != 0)
```

### 与 old_log_prob 的关键区别

| 对比项 | old_log_prob | update_actor |
|--------|-------------|--------------|
| `torch.no_grad()` | 是（纯推理） | 否（需要梯度） |
| 反向传播图 | 不保存 | 必须保存 |
| 前向激活 | 即用即丢 | 保留到 backward |
| gradient checkpointing | 不适用 | `enable_gradient_checkpointing=true` 有效 |
| 优化器 | 不参与 | step + 更新 |
| entropy | 硬编码 True | 取决于 `entropy_coeff`（CaRR 默认 0，不算） |

### 显存控制

CaRR 配置：
```yaml
actor:
  use_dynamic_bsz: true
  ppo_max_token_len_per_gpu: 24576     # 比 log_prob 阶段的默认 16384 更大
  ppo_micro_batch_size_per_gpu: 2
  enable_gradient_checkpointing: true  # 用重算换显存
```

`gradient_checkpointing` 的效果：不保存中间层激活，backward 时重算。
对 transformer 模型，激活显存从 O(层数 * 序列长度) 降到 O(sqrt(层数) * 序列长度)。

### 峰值估算（CaRR, Qwen3-4B, 8 卡, micro-batch 24k tokens）

```
FSDP actor sharded:           ~1 GiB
FSDP ref sharded:             ~1 GiB
all-gather 临时权重:           ~8 GiB
优化器 (Adam m+v):            ~2 GiB
梯度 (sharded):               ~1 GiB
前向激活 (checkpointed):      ~5-8 GiB
logits (24k * 152k * 2):      ~6.8 GiB
反向传播临时张量:              ~3-5 GiB
──────────────────────────────
总计:                          ~28-35 GiB  (在 96G 下很安全)
```

---

## I. Rollout 阶段 GPU 利用率波动的原因

你观察到 rollout 期间部分 GPU 利用率降到 0%，这不是 bug，而是 agentic multi-turn rollout 的
固有特征。以下是 5 个根本原因：

### 原因 1：工具调用期间 GPU 完全空闲

代码位置：`verl/experimental/agent_loop/tool_agent_loop.py` 第 286-383 行

每个样本的状态机是严格顺序的：

```
GENERATING (GPU 活跃) → PROCESSING_TOOLS (GPU 空闲) → GENERATING (GPU 活跃) → ...
```

在 `PROCESSING_TOOLS` 状态，样本通过 HTTP 调用工具服务器（search/open/find），
这是纯 CPU/网络 I/O 操作，**GPU 完全空闲**。

如果某个时刻大量样本同时进入工具调用阶段（它们往往在类似时间完成一轮生成），
就会形成"波浪"式的 GPU 空闲期。

### 原因 2：8 个独立 SGLang replica 的负载不均衡

代码位置：`verl/experimental/agent_loop/agent_loop.py` 第 56-121 行

TP=1 + 8 卡 = 8 个独立的 SGLang 推理实例。
`AsyncLLMServerManager` 使用最少请求数（least-requests）负载均衡：

```python
# agent_loop.py 第 87 行
count, idx, server = heapq.heappop(self.weighted_serveres)
```

但每个 `AgentLoopWorker` 有自己独立的负载均衡堆（第 363 行），
**没有跨 worker 的全局协调**。这意味着：

- Worker A 把它的请求都发到 GPU 0 和 GPU 1
- Worker B 也把请求发到 GPU 0 和 GPU 1
- GPU 6 和 GPU 7 可能完全空闲

### 原因 3：Straggler（掉队样本）导致大部分 GPU 空闲

代码位置：`verl/experimental/agent_loop/agent_loop.py` 第 471-475 行

```python
tasks = [asyncio.create_task(self._process_single_sample(i, ...)) for i in range(batch_size)]
results = await asyncio.gather(*tasks)  # 等待 ALL 样本完成
```

`asyncio.gather` 等待所有样本完成。在 rollout 后期：
- 大部分样本已经终止（模型输出了最终答案，或撞上 turn/token 限制）
- 只有 1-2 个"掉队"样本还在继续搜索
- 为这些 straggler 服务的 1-2 个 GPU 活跃，其余 6-7 个 GPU 空闲

**`over_sample_rate` 参数（`rollout.py` 第 149-151 行）本意是解决这个问题，**
当 `(1 - over_sample_rate) * total_requests` 个请求完成后 abort 剩余请求。
但在当前 Python 代码中**尚未实现**。

### 原因 4：多轮对话的序列化特性

对于单个样本，其多轮对话是严格顺序的：

```
第 1 轮生成 (GPU) → 工具调用 (网络) → 第 2 轮生成 (GPU) → 工具调用 (网络) → ...
```

一个样本在一个时刻只会使用一个 GPU replica。当这个样本在等待工具响应时，
它之前使用的那个 GPU replica 只有在有其他样本的请求时才会被利用。

**理想情况：** 当样本 A 在等工具调用时，样本 B、C、D 的生成请求填满 GPU。
**实际情况：** 如果大部分样本同步地进入工具调用，GPU 集体空闲。

### 原因 5：Sticky session 导致 KV cache 亲和

代码位置：`verl/experimental/agent_loop/agent_loop.py` 第 79-85 行

```python
self.request_id_to_server = LRUCache(capacity=max_cache_size)
# ...
if request_id in self.request_id_to_server:
    server = self.request_id_to_server[request_id]  # 复用同一个 replica
```

为了利用 SGLang 的 prefix caching，同一个 request_id 的多轮对话会被路由到同一个
GPU replica。这提高了推理效率，但也意味着：
- 如果 8 个 straggler 样本都被 stick 到同一个 GPU，那个 GPU 满载而其他空闲
- 负载均衡只在首次分配时生效，后续轮次固定在同一 GPU

### GPU 利用率波动的典型模式

```
时间 ─────────────────────────────────────────────────────────►

GPU 0: ████████░░░░████████░░░░████░░░░░░░░░░░░░░░░░░░░░░░░░
GPU 1: ████████░░░░████████░░░░████████░░░░████░░░░░░░░░░░░░░
GPU 2: ████████░░░░████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
GPU 3: ████████░░░░████████░░░░████████░░░░████████░░░░████░░  <- straggler
GPU 4: ████████░░░░████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
GPU 5: ████████░░░░████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
GPU 6: ████████░░░░████░░░░████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
GPU 7: ████████░░░░████████░░░░████░░░░░░░░░░░░░░░░░░░░░░░░░
       ↑                              ↑                    ↑
       初期：所有样本                   中期：部分样本已完成    后期：只剩 straggler
       同时生成，GPU 全满              部分 GPU 空闲          大部分 GPU 空闲
```

---

## J. 所有显存相关配置参数速查表

### Rollout 阶段参数

| 参数 | 默认值 | CaRR 值 | 影响 |
|------|--------|---------|------|
| `rollout.gpu_memory_utilization` | 0.5 | 0.5 | SGLang 静态内存占比（权重 + KV cache） |
| `rollout.free_cache_engine` | true | true | rollout 后释放 SGLang 显存 |
| `rollout.enable_sleep_mode` | true | true | 启用 SGLang sleep 模式节省显存 |
| `rollout.max_num_seqs` | 1024 | 默认 | 最大并发序列数，影响 KV cache 大小 |
| `rollout.max_model_len` | None | None | 最大序列长度，影响 KV cache |
| `rollout.max_num_batched_tokens` | 8192 | 默认 | 批量调度 token 上限 |
| `rollout.enforce_eager` | true | 默认 | 禁用 CUDA graphs，省图捕获显存 |
| `rollout.enable_chunked_prefill` | true | 默认 | 分块 prefill，平滑显存峰值 |
| `rollout.tensor_model_parallel_size` | 2 | 默认 | TP 并行度（影响 replica 数） |
| `rollout.over_sample_rate` | 0.0 | 默认 | 提前终止阈值（当前未实现） |

### old_log_prob / ref_log_prob 阶段参数

| 参数 | 默认值 | CaRR 值 | 影响 |
|------|--------|---------|------|
| `rollout.log_prob_micro_batch_size_per_gpu` | None | 未设 | old_log_prob 的 micro batch 大小 |
| `rollout.log_prob_use_dynamic_bsz` | false | 未设 | 是否启用 dynamic batching |
| `rollout.log_prob_max_token_len_per_gpu` | 16384 | 未设 | dynamic batch 每 GPU 最大 token 数 |
| `ref.log_prob_micro_batch_size_per_gpu` | None | **2** | ref_log_prob 的 micro batch 大小 |

### update_actor 阶段参数

| 参数 | 默认值 | CaRR 值 | 影响 |
|------|--------|---------|------|
| `actor.use_dynamic_bsz` | false | **true** | PPO 更新时启用 dynamic batching |
| `actor.ppo_max_token_len_per_gpu` | 16384 | **24576** | PPO 更新每 GPU 最大 token 数 |
| `actor.ppo_mini_batch_size` | 256 | **16** | 每次优化器 step 的样本数 |
| `actor.ppo_micro_batch_size_per_gpu` | None | **2** | 每 GPU micro batch 样本数 |
| `actor.ppo_epochs` | 1 | 1 | PPO epoch 数（数据重复使用次数） |
| `actor.entropy_coeff` | 0 | 0 | entropy 正则系数（0 时 update_actor 不算 entropy） |
| `actor.entropy_checkpointing` | false | 未设 | entropy checkpointing 开关 |

### 模型/FSDP 通用参数

| 参数 | 默认值 | CaRR 值 | 影响 |
|------|--------|---------|------|
| `model.enable_gradient_checkpointing` | false | **true** | 用重算换显存（仅 update_actor 有效） |
| `model.use_remove_padding` | true | **true** | 去除 padding 减少浪费 |
| `actor.fsdp_config.model_dtype` | fp32 | **bf16** | 模型精度（bf16 = 显存减半） |
| `actor.fsdp_config.param_offload` | false | false | 权重 offload 到 CPU |
| `actor.fsdp_config.optimizer_offload` | false | false | 优化器 offload 到 CPU |
| `actor.fsdp_config.grad_offload` | false | false | 梯度 offload 到 CPU |

### 数据维度参数（间接影响显存）

| 参数 | CaRR 值 | 影响 |
|------|---------|------|
| `data.train_batch_size` | 16 | 每步总 prompt 数（* rollout.n = 总 trajectory 数） |
| `data.max_prompt_length` | 4096 | prompt 最大长度 |
| `data.max_response_length` | 61440 | response 最大长度（决定单条 trajectory 上限） |
| `rollout.n` | 8 | 每 prompt 采样数（放大总 trajectory 数） |
| `rollout.multi_turn.max_assistant_turns` | 30 | 最大工具调用轮数 |
| `rollout.multi_turn.max_tool_response_length` | 10000 | 工具响应截断长度 |

---

# Part II — Post-Rollout 深度解析（old_log_prob / entropy / dynamic batching）

---

## 1. 训练主循环中 post-rollout 的完整顺序

代码位置：`verl/trainer/ppo/ray_trainer.py` 第 1310-1489 行

一个 training step 的主流程如下：

```
rollout (gen)           <-- SGLang 推理引擎生成 trajectory
    |
reward                  <-- 调 reward server 打分
    |
old_log_prob            <-- 用当前 actor 重算 log_prob + entropy  <== OOM 在这里
    |
ref_log_prob            <-- 用 ref policy 算 log_prob（无 entropy）
    |
advantage               <-- 在 driver 端计算优势
    |
update_actor            <-- PPO mini-batch 梯度更新
```

每个阶段都是顺序执行的。rollout 阶段在 SGLang 推理引擎上完成，显存模式和训练不同（只做推理，用 KV cache）。一旦 rollout 完成，SGLang 引擎会释放推理显存，然后切换到 FSDP 训练模式做后续计算。

---

## 2. old_log_prob 的完整代码路径

### 2.1 trainer 发起调用

代码位置：`verl/trainer/ppo/ray_trainer.py` 第 1132-1155 行

```python
def _compute_old_log_prob(self, batch: DataProto):
    # 把 DataProto 转成 TensorDict
    batch_td = batch.to_tensordict()
    # 去掉 left-padding，转成紧凑格式（每条样本只保留有效 token）
    batch_td = left_right_2_no_padding(batch_td)

    # 关键标记：calculate_entropy=True
    # 这是 old_log_prob 和 ref_log_prob 的核心区别！
    # ref_log_prob 设的是 calculate_entropy=False
    tu.assign_non_tensor(batch_td, calculate_entropy=True, compute_loss=False)

    # 通过 RPC 分发到各个 actor worker
    output = self.actor_rollout_wg.compute_log_prob(batch_td)

    # 收集结果
    entropy = tu.get(output, "entropy")
    log_probs = tu.get(output, "log_probs")
```

对比 `_compute_ref_log_prob`（第 1105-1130 行）：它设的是 `calculate_entropy=False`。
所以 ref 路径不算 entropy，显存需求低很多。

### 2.2 数据如何分发到各卡

代码位置：`verl/single_controller/base/decorator.py` 第 279-281 行

```python
def dispatch_nd_compute_dataproto(dp_rank_mapping, dp_size, worker_group, *args, **kwargs):
    # 按 DP rank 数量等分 batch（按样本数切，不是按 token 数）
    splitted_args, splitted_kwargs = _split_args_kwargs_data_proto(dp_size, *args, **kwargs)
```

对于 b2/n4, 8 卡（假设 TP=1, DP=8）的例子：
- 总共 2 x 4 = 8 条 trajectory
- 8 卡 DP，每卡分到 **1 条 trajectory**

**关键点：分发是按样本数均分，不考虑 token 长度。**
如果某卡分到一条特别长的 trajectory（比如 55k tokens），它要独自扛全部显存。

### 2.3 单卡上的 compute_log_prob

代码位置：`verl/workers/actor/dp_actor.py` 第 425-506 行

```python
def compute_log_prob(self, data, calculate_entropy=False):
    # 从 meta_info 读取配置
    use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]   # 是否启用 dynamic batching
    micro_batch_size = data.meta_info["micro_batch_size"]

    if use_dynamic_bsz:
        # max_token_len 就是你配的 log_prob_max_token_len_per_gpu
        max_token_len = data.meta_info["max_token_len"]
        # 把本卡的样本按 token 总量分成若干 micro-batch
        micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
    else:
        # 不启用 dynamic batching 时，直接按固定样本数切
        micro_batches = data.split(micro_batch_size)

    log_probs_lst = []
    entropy_lst = []

    # 逐个 micro-batch 做前向传播
    for micro_batch in micro_batches:
        micro_batch = micro_batch.to(get_device_id())   # 移到 GPU
        with torch.no_grad():                           # 注意：无梯度，纯推理
            outputs = self._forward_micro_batch(
                model_inputs, temperature=temperature,
                calculate_entropy=calculate_entropy
            )
        log_probs_lst.append(outputs["log_probs"])
        if calculate_entropy:
            entropy_lst.append(outputs["entropys"])

    # 拼接所有 micro-batch 的结果
    log_probs = torch.concat(log_probs_lst, dim=0)
    if calculate_entropy:
        entropys = torch.concat(entropy_lst, dim=0)

    # 如果用了 dynamic batching，恢复原始样本顺序
    if use_dynamic_bsz:
        log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
```

### 2.4 单个 micro-batch 的前向传播

代码位置：`verl/workers/actor/dp_actor.py` 第 160-278 行

这是显存消耗最大的地方，分步讲解：

**步骤 A：去除 padding (remove padding / rmpad)**

```python
# input_ids 形状: (batch_size, seq_len)   -- 这里 batch_size 是 micro-batch 中的样本数
# attention_mask 形状: (batch_size, seq_len) -- 1 表示有效 token，0 表示 padding

# unpad_input 把所有有效 token 拼成一维
input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
    input_ids.unsqueeze(-1), attention_mask
)
# input_ids_rmpad 形状: (total_nnz, 1)
# total_nnz = 所有样本的有效 token 数之和（Non-Zero 的意思）
# indices: 记录每个有效 token 在原始 padding 矩阵中的位置，用于后续 pad 回去
# cu_seqlens: cumulative sequence lengths，告诉 flash attention 每条样本的边界

input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # -> (1, total_nnz)
```

**为什么要 transpose 成 (1, total_nnz)?**
因为 flash attention 的 varlen (variable length) 模式要求输入是 batch=1 的单条长序列，
然后用 `cu_seqlens` 告诉 kernel 哪些 token 属于哪个原始样本。这样可以完全避免 padding 的浪费。

**步骤 B：模型前向**

```python
output = self.actor_module(
    input_ids=input_ids_rmpad,      # (1, total_nnz)
    attention_mask=None,            # varlen 模式不需要 attention_mask
    position_ids=position_ids_rmpad,
    use_cache=False,
)
```

模型输出 `output.logits` 的形状是 **(1, total_nnz, vocab_size)**。
那个 `1` 就是前面人为构造的假 batch 维度。

**步骤 C：squeeze 并计算 log_prob 和 entropy**

```python
logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
# squeeze(0) 去掉假 batch 维度，变成 (total_nnz, vocab_size)

logits_rmpad.div_(temperature)  # in-place 除以 temperature，不额外分配显存

# 计算 log_prob（详见第 3 节）
log_probs = logprobs_from_logits(
    logits=logits_rmpad,            # (total_nnz, vocab_size)
    labels=input_ids_rmpad_rolled,  # (total_nnz,) -- 左移一位的 token id
    inplace_backward=False,         # 因为还要算 entropy，不能 in-place 销毁 logits
)

# 计算 entropy（详见第 4 节）
if calculate_entropy:
    entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)
    # entropy_rmpad 形状: (total_nnz,)
```

**步骤 D：pad 回原始形状**

```python
# 把 rmpad 的结果还原成 (batch_size, seq_len) 的 padding 格式
full_log_probs = pad_input(
    hidden_states=log_probs.unsqueeze(-1),  # (total_nnz, 1)
    indices=indices,
    batch=batch_size,
    seqlen=seqlen,
)  # -> (batch_size, seq_len, 1)
# padding 位置自动填 0
```

---

## 3. log_prob 是怎么算的

### 3.1 数学原理

给定模型对某个位置输出的 logits 向量 z（长度为 V，V 是词表大小），我们要计算模型给真实 token y 分配的对数概率。

**softmax 概率：**

```
p(y) = exp(z_y) / sum_j(exp(z_j))
```

其中分母是对词表中所有 V 个 token 的 exp 求和。

**对数概率：**

```
log p(y) = z_y - log(sum_j(exp(z_j)))
```

即 `log_softmax(z)[y]`。

后面那个 `log(sum_j(exp(z_j)))` 叫做 **logsumexp**，它是一个归一化常数。

### 3.2 举例

假设词表只有 5 个 token: [A, B, C, D, E]，模型输出的 logits 是：

```
z = [2.0, 1.0, 0.5, -1.0, 3.0]
```

真实的下一个 token 是 B（index=1），那么：

```
logsumexp = log(exp(2.0) + exp(1.0) + exp(0.5) + exp(-1.0) + exp(3.0))
          = log(7.389 + 2.718 + 1.649 + 0.368 + 20.086)
          = log(32.21)
          = 3.473

log p(B) = z[1] - logsumexp = 1.0 - 3.473 = -2.473
```

这意味着模型给 B 这个 token 的概率是 exp(-2.473) = 0.084，即 8.4%。

### 3.3 在 RL 中的意义

在 PPO 中，我们需要两个 log_prob：

1. **old_log_prob**：用"当前"policy 在训练前算一次，作为更新的基准
2. **new_log_prob**：每个 mini-batch 更新时重算，和 old 比较得到 importance ratio

```
ratio = exp(new_log_prob - old_log_prob)
```

这个 ratio 就是 PPO clipping 的核心。如果 ratio 偏离 1 太多（说明 policy 变化太大），就会被 clip 住。

### 3.4 代码实现

代码位置：`verl/utils/torch_functional.py` 第 72-100 行

```python
def logprobs_from_logits(logits, labels, inplace_backward=True):
    """
    logits: 模型输出，形状 (total_nnz, vocab_size)
    labels: 真实 token id，形状 (total_nnz,)
    返回: 每个位置真实 token 的 log 概率，形状 (total_nnz,)
    """

    # 方案 1：使用 flash-attn 的 triton cross-entropy（最高效）
    if FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE:
        output = logprobs_from_logits_flash_attn(logits, labels, inplace_backward=inplace_backward)
        # 内部调用 cross_entropy_loss(logits, labels) 返回 (loss, z_loss)
        # cross_entropy_loss = -log_softmax(logits)[label]
        # 所以 log_prob = -loss
        #
        # 如果 inplace_backward=True，logits 的梯度会 in-place 写回 logits 张量本身，
        # 节省一份显存。但如果后面还要用 logits 算 entropy，就不能 in-place。

    # 方案 2：NPU（华为昇腾）专用
    elif NPU_CROSS_ENTROPY_LOSS_AVAILABLE:
        output = logprobs_from_logits_torch_npu(logits, labels)

    # 方案 3：纯 PyTorch fallback（逐行处理，最省显存但最慢）
    else:
        output = logprobs_from_logits_v2(logits, labels)
    return output
```

`logprobs_from_logits_v2`（第 166 行）的逐行处理方式：

```python
def logprobs_from_logits_v2(logits, labels):
    # logits: (batch_size, seq_len, vocab_size) 或 (batch_size, vocab_size)
    # 逐行计算 logsumexp，然后 gather 出 label 对应的值
    # 好处是不需要同时 materialize 整个 log_softmax 矩阵
    for row in range(logits.shape[0]):
        logp = logits[row, labels[row]] - torch.logsumexp(logits[row], dim=-1)
        # 只取了 label 对应的那一个值，不需要存整个 softmax 结果
```

**关键点：** 如果只算 log_prob（不算 entropy），用 flash-attn fused kernel 时 logits 可以被 in-place 消耗（`inplace_backward=True`），不需要额外显存。但如果同时需要算 entropy，logits 必须保留完整，所以 `inplace_backward` 被设成 `False`。

---

## 4. entropy 是怎么算的

### 4.1 数学原理

Shannon entropy 衡量一个概率分布的"不确定性"：

```
H(p) = -sum_j(p_j * log(p_j))
```

其中 p_j = softmax(z)_j 是第 j 个 token 的概率。

**数值稳定的等价形式：**

```
H(p) = logsumexp(z) - sum_j(softmax(z)_j * z_j)
```

这个等价可以这样推导：

```
-sum(p_j * log(p_j))
= -sum(p_j * (z_j - logsumexp(z)))          # 因为 log(softmax(z)_j) = z_j - logsumexp(z)
= -sum(p_j * z_j) + logsumexp(z) * sum(p_j) # sum(p_j) = 1
= logsumexp(z) - sum(p_j * z_j)
```

### 4.2 举例

沿用上面的例子：

```
z = [2.0, 1.0, 0.5, -1.0, 3.0]
logsumexp = 3.473

softmax(z) = [exp(z_j) / sum(exp(z))]
           = [7.389/32.21, 2.718/32.21, 1.649/32.21, 0.368/32.21, 20.086/32.21]
           = [0.229, 0.084, 0.051, 0.011, 0.624]

sum(softmax(z) * z) = 0.229*2.0 + 0.084*1.0 + 0.051*0.5 + 0.011*(-1.0) + 0.624*3.0
                    = 0.458 + 0.084 + 0.026 - 0.011 + 1.872
                    = 2.429

H = logsumexp - sum(softmax(z) * z) = 3.473 - 2.429 = 1.044
```

entropy = 1.044 nats。这个值越大，说明模型越"犹豫"（概率分布越均匀）；越小说明模型越"确定"（概率集中在少数 token 上）。

### 4.3 在 RL 中的意义

entropy 在 PPO 中用作**正则化项**：

```
loss = policy_loss - entropy_coeff * entropy
```

加入 entropy bonus 鼓励模型保持探索性，避免过早坍缩到确定性策略。
`entropy_coeff` 越大，模型越被鼓励输出均匀的分布。

verl 在 `_compute_old_log_prob` 阶段就计算好 entropy 并记录为 metric（`actor/entropy`），
同时在 `_update_actor` 阶段根据 `entropy_coeff` 决定是否需要再次计算。

### 4.4 代码实现

代码位置：`verl/utils/torch_functional.py` 第 224-238 行

```python
def entropy_from_logits(logits):
    """
    logits 形状: (total_nnz, vocab_size)
    返回: (total_nnz,)   -- 每个 token 位置一个 entropy 值
    """
    # 第 1 步：计算 softmax
    pd = torch.nn.functional.softmax(logits, dim=-1)
    # pd 形状: (total_nnz, vocab_size)
    # 这里新分配了一个和 logits 一样大的张量！

    # 第 2 步：用等价公式计算 entropy
    entropy = torch.logsumexp(logits, dim=-1)   # (total_nnz,) -- 这个很小
              - torch.sum(pd * logits, dim=-1)  # pd * logits 是 (total_nnz, vocab_size) 的临时张量！
    return entropy
```

### 4.5 为什么 entropy 是显存杀手

在 `entropy_from_logits` 执行期间，GPU 上同时存在以下张量：

```
张量                      形状                      生命周期
--------------------------------------------------------------
logits (输入，不能释放)    (total_nnz, vocab_size)   整个函数
pd = softmax(logits)      (total_nnz, vocab_size)   从计算到函数结束
pd * logits (临时)         (total_nnz, vocab_size)   torch.sum 期间
```

也就是说**峰值时有三份 (total_nnz, vocab_size) 大小的张量同时在显存中**。

以 Qwen3-4B 为例（vocab_size = 151,936），bf16 数据类型（2 bytes/element）：

```
单份张量大小 = total_nnz * 151,936 * 2 bytes
峰值 = total_nnz * 151,936 * 2 * 3  (三份)
     = total_nnz * 151,936 * 6 bytes
```

### 4.6 chunked 版本（更省显存）

代码位置：`verl/utils/torch_functional.py` 第 241-263 行

```python
def entropy_from_logits_with_chunking(logits, chunk_size=2048):
    """分块计算 entropy，大幅降低峰值显存"""
    entropy = torch.zeros(logits.shape[0], device=logits.device)
    for i in range(0, logits.shape[0], chunk_size):
        # 每次只处理 chunk_size 行
        logits_chunk = logits[i:i+chunk_size].float()   # 注意：转成 float32 提高精度
        pd_chunk = torch.nn.functional.softmax(logits_chunk, dim=-1)
        entropy_chunk = (torch.logsumexp(logits_chunk, dim=-1)
                        - torch.sum(pd_chunk * logits_chunk, dim=-1))
        entropy[i:i+chunk_size] = entropy_chunk
        # 每轮循环结束后，logits_chunk / pd_chunk / 临时张量都被释放
    return entropy
```

使用 chunking 时，峰值从 `total_nnz * V * 6` 降到 `chunk_size * V * 6`：

```
chunk_size=2048 时: 2048 * 151,936 * 6 = 约 1.7 GiB
```

---

## 5. 显存如何估算

### 5.1 显存的主要组成

post-rollout 阶段（old_log_prob）单卡显存由以下几部分组成：

```
总显存 = 模型权重 + 前向激活 + logits 张量 + entropy 计算峰值 + 其他 buffer
```

#### (a) 模型权重

FSDP 模式下，前向传播时需要 all-gather 到完整权重：
- Qwen3-4B, bf16: 约 **8 GiB**
- 平时只存 1/N（N 是 FSDP 并行度），但前向时临时 all-gather 到完整大小

#### (b) 前向激活 (activations)

模型各层的中间输出。由于 old_log_prob 用的是 `torch.no_grad()`（不算梯度），
不需要保存反向传播用的激活，所以比训练时的前向小很多。

粗略估计：
- 主要是各层的 hidden states + attention 中间结果
- 对 4B 模型大约 **2-5 GiB**（取决于序列长度和 flash attention 实现）

#### (c) logits 张量

```
logits 大小 = total_nnz * vocab_size * 2 bytes (bf16)
```

| total_nnz (有效 token 数) | vocab=151,936, bf16 | 大小 |
|--------------------------|---------------------|------|
| 10,000                   | 10k * 152k * 2      | 2.8 GiB |
| 30,000                   | 30k * 152k * 2      | 8.5 GiB |
| 50,000                   | 50k * 152k * 2      | 14.2 GiB |
| 70,000                   | 70k * 152k * 2      | 19.8 GiB |

#### (d) entropy 计算峰值

如上分析，峰值时有 3 份 logits 大小的张量：

| total_nnz | logits * 3 (entropy 峰值) |
|-----------|--------------------------|
| 10,000    | 8.5 GiB |
| 30,000    | 25.4 GiB |
| 50,000    | 42.6 GiB |
| 70,000    | 59.4 GiB |

### 5.2 完整估算公式

```
单卡峰值显存 ≈ 模型权重 (8G)
             + 前向激活 (3-5G)
             + entropy 峰值 (total_nnz * vocab_size * 6 bytes)
             + 其他 buffer (1-2G)
```

### 5.3 举例

假设 8 卡 DP=8，b2/n4 (8 条 trajectory)，某卡分到 1 条 40k token 的 trajectory：

```
模型权重:        ~8 GiB
前向激活:        ~4 GiB
entropy 峰值:    40,000 * 151,936 * 6 / (1024^3) = ~33.9 GiB
其他 buffer:     ~2 GiB
-------------------------------
总计:            ~47.9 GiB
```

如果那条 trajectory 有 50k token：

```
entropy 峰值:    50,000 * 151,936 * 6 / (1024^3) = ~42.4 GiB
总计:            ~56 GiB
```

在 48 GB 显存的 GPU 上会 OOM，在 96 GB 的 RTX 6000 Pro 上还有余量。
但如果关掉 entropy（`calculate_entropy=False`），50k token 场景下总计只需约 ~28 GiB。

---

## 6. dynamic batching 是如何实现的

### 6.1 核心思想

固定 batch size 切分的问题：如果一条样本 5k token，另一条 55k token，放在同一个 micro-batch 里会被 pad 到 55k，浪费 50k token 的计算。

Dynamic batching 的做法是：**按 token 总量而不是样本数来决定 micro-batch 的大小**。

### 6.2 代码流程

代码位置：`verl/utils/seqlen_balancing.py` 第 348-424 行

```python
def rearrange_micro_batches(batch, max_token_len, ...):
    """
    batch: 本卡分到的所有样本
    max_token_len: 单个 micro-batch 允许的最大 token 总量
                   （= 你配的 log_prob_max_token_len_per_gpu）
    """

    # 第 1 步：计算每条样本的有效 token 数
    seq_len_effective = batch["attention_mask"].sum(dim=1)
    # 例如 3 条样本: [20000, 35000, 15000]

    total_seqlen = seq_len_effective.sum().item()
    # 总 token 数: 70000

    # 第 2 步：决定要切成几个 micro-batch
    num_micro_batches = min(
        len(seq_len_effective),                     # 不超过样本数
        ceildiv(total_seqlen, max_token_len)        # ceil(总 token / 每 batch 上限)
    )
    # 如果 max_token_len=70000: ceil(70000/70000) = 1 -> 1 个 micro-batch（全塞进去）
    # 如果 max_token_len=40000: ceil(70000/40000) = 2 -> 2 个 micro-batch
    # 如果 max_token_len=20000: ceil(70000/20000) = 4 -> 但只有 3 条样本，所以 min(3,4)=3

    # 第 3 步：用 workload 均衡分组
    workloads = calculate_workload(seq_len_effective)
    # workload 近似 seq_len^2（因为 attention 计算量和序列长度的平方成正比）
    # 用贪心算法把样本分到 num_micro_batches 个桶里，让各桶 workload 尽量均衡

    micro_bsz_idx = get_seqlen_balanced_partitions(workloads, num_micro_batches, equal_size=False)
    # 返回类似 [[0, 2], [1]] -- 表示第 0、2 条样本放一个 micro-batch，第 1 条单独一个

    # 第 4 步：排序，把大 micro-batch 放中间，小的放两端
    # 目的是减少计算 pipeline 的 warm-up / cool-down bubble
    if use_dynamic_bsz_balance:
        micro_bsz_idx.sort(key=lambda partition: sum(workloads[idx] for idx in partition), reverse=True)
        micro_bsz_idx = micro_bsz_idx[::2][::-1] + micro_bsz_idx[1::2]
```

### 6.3 举例

假设本卡有 3 条 trajectory，有效 token 数分别是：

```
样本 0: 20,000 tokens
样本 1: 35,000 tokens
样本 2: 15,000 tokens
总计: 70,000 tokens
```

**场景 A：max_token_len = 70,000**

```
num_micro_batches = ceil(70000 / 70000) = 1
-> 只有 1 个 micro-batch，3 条全塞进去
-> 这个 micro-batch 的 total_nnz = 70,000
-> logits 张量: 70,000 * 151,936 * 2 = 19.8 GiB
-> entropy 峰值: 19.8 * 3 = 59.4 GiB  <-- 很可能 OOM
```

**场景 B：max_token_len = 40,000**

```
num_micro_batches = ceil(70000 / 40000) = 2
贪心分组（按 workload = seq_len^2 均衡）：
  micro-batch 1: [样本 1] -> 35,000 tokens
  micro-batch 2: [样本 0, 样本 2] -> 35,000 tokens
-> 最大 micro-batch 的 total_nnz = 35,000
-> entropy 峰值: 35,000 * 151,936 * 6 = 29.7 GiB  <-- 还是很大
```

**场景 C：max_token_len = 20,000**

```
num_micro_batches = min(3, ceil(70000 / 20000)) = min(3, 4) = 3
每条样本单独一个 micro-batch：
  micro-batch 1: [样本 1] -> 35,000 tokens  <-- 注意：单条 35k 已超过 max_token_len
  micro-batch 2: [样本 0] -> 20,000 tokens
  micro-batch 3: [样本 2] -> 15,000 tokens
-> 最大 micro-batch 的 total_nnz = 35,000  <-- 单条样本没法再拆
```

**关键限制：dynamic batching 是按样本级别分组的，不会把一条样本切成两半。**
所以如果单条 trajectory 就有 50k tokens，即使 max_token_len 设得再小，
这条样本自己就要占 50k * 151,936 * 6 = 42.6 GiB 的 entropy 峰值。

### 6.4 restore 阶段

代码位置：`verl/utils/seqlen_balancing.py` 第 484-499 行

```python
def restore_dynamic_batch(data, batch_idx_list):
    """
    dynamic batching 会打乱样本顺序（按 workload 排序），
    这个函数把结果恢复到原始的样本顺序。
    """
    indices = list(chain.from_iterable(batch_idx_list))
    # 例如 batch_idx_list = [[1], [0, 2]]
    # indices = [1, 0, 2]
    revert_indices = get_reverse_idx(indices)
    # revert_indices = [1, 0, 2] -> 把位置 0 的结果放回原始位置 1，以此类推
    return data[revert_indices]
```

---

## 7. 在 8 卡 RTX 6000 Pro (96 GB) 上的 OOM 分析

### 7.1 RTX 6000 Pro 的显存

RTX PRO 6000 (Blackwell) 有 **96 GiB** 显存。

### 7.2 配置示例（之前 b2/n4 的 probe）

```yaml
data.train_batch_size: 2
actor_rollout_ref.rollout.n: 4
# 总 trajectory 数: 2 * 4 = 8
# 8 卡 DP -> 每卡 1 条 trajectory

actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu: 70000
actor_rollout_ref.actor.ppo_max_token_len_per_gpu: 70000
actor_rollout_ref.ref.log_prob_max_token_len_per_gpu: 70000

data.max_response_length: 61440  # 约 60k
```

### 7.3 OOM 分析

即使每卡只分到 1 条 trajectory，如果它有较长的 token（deep search agent 经常产生长 trajectory）：

**50k token 场景：**

```
模型权重 (all-gather):                                    ~8 GiB
前向激活 (no_grad):                                       ~4 GiB
logits (50k * 152k * 2):                                  ~14.2 GiB
+ entropy 额外 2 份 (softmax + pd*logits):                ~28.4 GiB
优化器状态 (驻留):                                         ~2 GiB
FSDP sharded 权重 (actor+ref):                            ~2 GiB
其他 buffer:                                              ~2 GiB
--------------------------------------------------------------
总计:                                                     ~60.6 GiB / 96 GiB  ✓ 安全
```

**60k token 场景（接近 max_response_length）：**

```
logits (60k * 152k * 2):                                  ~17 GiB
+ entropy 额外 2 份:                                       ~34 GiB
+ 模型/激活/优化器:                                        ~18 GiB
--------------------------------------------------------------
总计:                                                     ~69 GiB / 96 GiB  ✓ 安全但余量不大
```

在 96 GB 下，b2/n4 的 old_log_prob 阶段通常不会 OOM。
但如果使用更大 batch（如 b4/n8，每卡分到多条长 trajectory），
或者 `max_token_len_per_gpu` 设得过大让 dynamic batch 把多条样本塞进同一个 micro-batch，
显存仍可能超限。

### 7.4 为什么 FSDP 救不了 logits 显存

FSDP 分片的是**模型权重**（parameter sharding）。
`logits` 和 entropy 的中间张量是**计算输出**，每张卡对自己分到的样本独立产生，不会被 FSDP 分片。

也就是说：
- 权重：8 卡分摊，每卡只存 1/8 （但前向时 all-gather 到完整）
- logits / entropy：每卡对自己的样本本地计算，不分摊

---

## 8. 解决方向

### 方案 0（最推荐）：关掉 old_log_prob 阶段的 entropy

`ray_trainer.py` 第 1140 行的 `calculate_entropy=True` 是硬编码的，
但计算出的 entropy 只用于记录一个 metric（`actor/entropy`），然后被 `pop` 丢弃。
将其改为 `False` 可以立即消除 entropy 的 3 倍 logits 显存峰值：

```python
# ray_trainer.py 第 1140 行
# 改前：
tu.assign_non_tensor(batch_td, calculate_entropy=True, compute_loss=False)
# 改后：
tu.assign_non_tensor(batch_td, calculate_entropy=False, compute_loss=False)
```

效果：50k token 场景下，old_log_prob 显存从 ~56 GiB 降到 ~28 GiB（减少约 50%）。
代价：丢失 `actor/entropy` 监控指标。

### 方案 A：降低 max_token_len_per_gpu

```yaml
actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu: 20000
actor_rollout_ref.ref.log_prob_max_token_len_per_gpu: 20000
actor_rollout_ref.actor.ppo_max_token_len_per_gpu: 20000
```

效果：dynamic batching 会把样本拆成更多 micro-batch，每个 micro-batch 的 logits 更小。
代价：更多轮前向传播，单步训练时间变长。
局限：对单条超长 trajectory 无效（一条样本不会被拆开）。

**重要补充（2026-03-17，4x RTX PRO 6000 RL probe）**:

- 仅仅降低 `rollout/ref.log_prob_max_token_len_per_gpu`，并不能避免所有 `max_token_len` 相关失败
- `old_log_prob/ref_log_prob` 用的是 `rollout/ref.log_prob_max_token_len_per_gpu`
- 但 `update_actor` 用的是 `actor.ppo_max_token_len_per_gpu`
- 当 `data.max_prompt_length=4096`、`data.max_response_length=61440` 时，实际 `max_seq_len` 会到 `65536`
- 如果此时 `actor.ppo_max_token_len_per_gpu < 65536`，会在 `update_actor` 阶段触发：

```text
AssertionError: max_token_len must be greater than the sequence length.
Got max_token_len=24576 and max_seq_len=65536
```

结论：

- `log_prob_max_token_len_per_gpu` 不等于 `ppo_max_token_len_per_gpu`
- 后续任何 `64k RL` 训练/探针，只要保留 `data.max_response_length=61440`，就必须把 `actor_rollout_ref.actor.ppo_max_token_len_per_gpu` 提到 `>= 65536`，当前实践上直接设 `70000`
- 如果显存不允许，就应该先降 `data.max_response_length`，而不是只压 `log_prob_max_token_len_per_gpu`

### 方案 B：启用 entropy checkpointing

代码位置：`verl/workers/actor/dp_actor.py` 第 276-278 行

```python
if not self.config.entropy_checkpointing:
    entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)
else:
    # 用 gradient checkpointing 重算，减少中间张量的保存
    entropy_rmpad = torch.utils.checkpoint.checkpoint(
        self.compute_entropy_from_logits, logits_rmpad
    )
```

可通过配置 `actor_rollout_ref.actor.entropy_checkpointing=true` 开启。

### 方案 C：使用 chunked entropy

将 `entropy_from_logits` 替换为 `entropy_from_logits_with_chunking`（已在代码中存在），
将峰值从 `total_nnz * V * 6` 降到 `chunk_size * V * 6`（约 1.7 GiB）。

### 方案 D：使用 fused kernel

代码位置：`verl/workers/actor/dp_actor.py` 第 253-255 行

```python
if self.use_fused_kernels:
    log_probs = output.log_probs.squeeze(0)     # (total_nnz,)
    entropy_rmpad = output.entropy.squeeze(0)    # (total_nnz,)
```

如果模型支持 fused kernel（在模型内部同时输出 log_probs 和 entropy），
就不需要在外部 materialize 完整的 logits 矩阵，显存可以大幅下降。

### 方案 E：缩短 max_response_length

减少单条 trajectory 的最大长度，从根源减小 total_nnz 的上限。
但这会影响 agent 的搜索深度。

---

## 附录：关键代码文件索引

| 文件 | 关键行号 | 内容 |
|------|---------|------|
| `verl/trainer/ppo/ray_trainer.py` | 1132-1155 | `_compute_old_log_prob` 入口 |
| `verl/trainer/ppo/ray_trainer.py` | 1105-1130 | `_compute_ref_log_prob` 入口（无 entropy） |
| `verl/trainer/ppo/ray_trainer.py` | 1398-1431 | 调用 old_log_prob 并记录 entropy metric |
| `verl/single_controller/base/decorator.py` | 279-281 | 按 DP rank 分发数据 |
| `verl/workers/actor/dp_actor.py` | 425-506 | `compute_log_prob` 主逻辑 |
| `verl/workers/actor/dp_actor.py` | 160-278 | `_forward_micro_batch` 前向传播 |
| `verl/utils/seqlen_balancing.py` | 348-424 | `rearrange_micro_batches` dynamic batching |
| `verl/utils/seqlen_balancing.py` | 445-481 | `prepare_dynamic_batch` 封装 |
| `verl/utils/torch_functional.py` | 72-100 | `logprobs_from_logits` log_prob 计算 |
| `verl/utils/torch_functional.py` | 224-238 | `entropy_from_logits` entropy 计算 |
| `verl/utils/torch_functional.py` | 241-263 | `entropy_from_logits_with_chunking` 省显存版 |
