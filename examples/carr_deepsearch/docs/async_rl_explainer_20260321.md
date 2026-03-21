# CaRR DeepSearch Async RL 详细讲解

> **目标读者**：对 verl 的同步训练流程有基本了解，但对 `fully_async_policy` 完全不了解的开发者。
>
> **配套文档**：实施计划见同目录下的实施版文档。

---

## 目录

1. [为什么需要异步 RL](#1-为什么需要异步-rl)
2. [当前同步架构的瓶颈](#2-当前同步架构的瓶颈)
3. [异步架构总览](#3-异步架构总览)
4. [四大核心组件详解](#4-四大核心组件详解)
5. [一个样本的完整生命周期（异步版）](#5-一个样本的完整生命周期异步版)
6. [同步 vs 异步：逐步对比](#6-同步-vs-异步逐步对比)
7. [参数版本与 Staleness 控制](#7-参数版本与-staleness-控制)
8. [Partial Rollout：中断与恢复](#8-partial-rollout中断与恢复)
9. [CaRR 接入异步的关键挑战](#9-carr-接入异步的关键挑战)
10. [实施计划各 Phase 对应的代码改动](#10-实施计划各-phase-对应的代码改动)
11. [关键文件索引](#11-关键文件索引)

---

## 1. 为什么需要异步 RL

### 一句话

当前训练的 **73% 样本因为超时被截断**，reward 直接归零，训练信号基本废掉。根本原因是同步栅栏——最慢的 deep-search 样本拖住了整个 step。异步 RL 消除这个栅栏。

### 数字证据

从 handoff 文档中的正式训练数据：

| 窗口 | outcome_reward | task_unfinished | termination_rollout_timeout | step wall time |
|------|---------------|-----------------|----------------------------|----------------|
| step 21-70 | 0.266 | 57.2% | 6.9% | ~493s |
| step 71-90 | 0.320 | 49.7% | 8.4% | ~506s |
| **step 91-96** | **0.167** | **73.4%** | **52.1%** | **~531s** |

step 91-96 明确表明：训练信号已经被截断毒化。`task_unfinished=73%` 意味着只有约 1/4 的样本能产生非零 reward。

---

## 2. 当前同步架构的瓶颈

### 同步训练的一个 step 长什么样

```
时间轴 ──────────────────────────────────────────────────────────>

┌─────────────────── 一个 training step ───────────────────────┐

│ Rollout (全部样本)  │ Reward │ LogProb │ Advantage │ Update │ Sync │
│ ████████████████████│████    │████     │██         │████    │██   │
│                     │        │         │           │        │     │
│ ← 同步栅栏：必须    │        │         │           │        │     │
│   等所有样本完成     │        │         │           │        │     │
│                     │        │         │           │        │     │
│ sample₁ ████        │        │         │           │        │     │
│ sample₂ ██████████  │        │         │           │        │     │
│ sample₃ ██          │        │         │           │        │     │
│ sample₄ ████████████████████ ← 长尾样本，拖住整个 step      │     │
│                     ↑        │         │           │        │     │
│              所有样本完成后   │         │           │        │     │
│              才能继续         │         │           │        │     │
└─────────────────────────────────────────────────────────────────┘
```

### 代码路径

在 `RayPPOTrainer.fit()` 中（`verl/trainer/ppo/ray_trainer.py`），关键的同步栅栏是：

```python
# 这一行会阻塞，直到所有 b*n 个 rollout 全部完成
gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)
# ↑ 对于 b=8, n=4 就是 32 个样本，每个样本要运行完整的多轮 agent loop
# 最慢的一个决定了这一步的 wall time
```

### 为什么 deep-search 特别痛

普通单轮 LLM 生成的样本方差小（都是一次 forward pass）。但 CaRR deep-search：
- 每个样本要经历 **几十轮** search → open → find 的工具调用循环
- 涉及外部 API 调用（Serper 搜索、Jina 网页提取）
- 样本间方差极大：简单问题 3 轮结束，难问题 50+ 轮还没完
- **一个长尾样本就能让整个 step 多等几分钟**

---

## 3. 异步架构总览

### 核心思想

**把 Rollout 和 Training 放到独立的 GPU 组上，通过消息队列解耦。**

```
异步架构：

  Rollouter GPU 组 (如 4 卡)          Trainer GPU 组 (如 4 卡)
  ┌──────────────────────┐           ┌──────────────────────┐
  │                      │           │                      │
  │  逐样本流式生成       │  队列     │  攒够就训练           │
  │  sample₁ ████ ──────►├──────────►│  取 4 个样本 → 训练   │
  │  sample₂ ██ ────────►│           │  取 4 个样本 → 训练   │
  │  sample₃ ████████ ──►│           │  取 4 个样本 → 训练   │
  │  sample₄ ██████ ────►│           │  ...                  │
  │  sample₅ ██ ────────►│           │                      │
  │  ...                 │           │  每 K 步触发参数同步   │
  │                      │           │                      │
  └──────────────────────┘           └──────────────────────┘
          ↑                                    │
          │          参数同步 (NCCL)            │
          └────────────────────────────────────┘
```

### 关键差异

| 维度 | 同步 (当前) | 异步 (目标) |
|------|------------|------------|
| GPU 分配 | 所有卡共享 rollout + train（hybrid engine） | 独立分组：rollout 卡 + train 卡 |
| 数据流 | 批量生成 → 批量训练 → 批量生成 ... | 流式生成 → 队列缓冲 → 流式消费 |
| 同步栅栏 | 每步都有，最慢样本决定 wall time | **无栅栏**，快样本先完成先被训练 |
| 长尾影响 | 拖慢整个 step | 只影响该样本自己 |
| 参数新鲜度 | 始终最新（on-policy） | 允许轻微滞后（staleness 可控） |

---

## 4. 四大核心组件详解

### 4.1 FullyAsyncRollouter —— 流式样本生产者

**文件**：`verl/experimental/fully_async_policy/fully_async_rollouter.py`

**职责**：在独立 GPU 组上，逐个样本地运行 agent loop 并把完成的样本放入队列。

**工作方式**：

```
FullyAsyncRollouter 内部有三个并发协程：

1. _feed_samples()    —— 从 dataloader 取 prompt，构造 RolloutSample，放入 pending_queue
2. _processor_worker() —— 从 pending_queue 取样本，提交给 agent loop 异步执行
3. _async_monitor_loop() —— 监控统计、触发恢复
```

**核心流程**：

```python
# 1. 每个 prompt 变成 RolloutSample
rollout_sample = RolloutSample(
    full_batch=full_batch,   # DataProto，包含 prompt tokens
    agent_loop_output_list=[None] * rollout_n,  # 预留 n 个生成槽位
    sample_id=f"sample_{epoch}_{step}",
    param_version=0,         # 稍后用实际版本覆盖
    ...
)

# 2. 提交给 agent loop manager 异步生成
ret, is_cancel = await self.async_rollout_manager.generate_single_sample_async(
    rollout_sample.full_batch,
    rollout_sample.agent_loop_output_list
)

# 3. 完成后放入 MessageQueue
await self.message_queue_client.put_sample(
    sample=ray.cloudpickle.dumps(rollout_sample),
    param_version=self.current_param_version,
)
```

**关键属性**：
- `max_concurrent_samples`：同时在飞的最大样本数（避免 OOM）
- `staleness_samples`：当前版本下已生成的样本数（用于 staleness 控制）
- `paused`：参数同步期间暂停生成

### 4.2 MessageQueue —— 消息队列

**文件**：`verl/experimental/fully_async_policy/message_queue.py`

**职责**：Ray Actor，在 Rollouter 和 Trainer 之间缓冲样本。

**设计**：

```python
@ray.remote(num_cpus=2, max_concurrency=20)
class MessageQueue:
    def __init__(self, config, max_queue_size):
        self.queue = deque(maxlen=max_queue_size)  # 有界队列
        self._lock = asyncio.Lock()
        self._consumer_condition = asyncio.Condition(self._lock)
```

**核心接口**：

| 方法 | 调用者 | 作用 |
|------|--------|------|
| `put_sample(sample, param_version)` | Rollouter | 生产者放入样本 |
| `get_sample()` | Trainer | 消费者取出样本（阻塞等待） |
| `put_validate(data)` | Rollouter | 放入 validation 结果 |
| `get_validate()` | Trainer | 取出 validation 结果 |

**流控**：队列满时 `put_sample` 会丢弃最旧的样本（`deque(maxlen=...)` 行为）。

### 4.3 FullyAsyncTrainer —— 异步消费者

**文件**：`verl/experimental/fully_async_policy/fully_async_trainer.py`

**职责**：在独立 GPU 组上，从队列取样本做 PPO/GRPO 训练。

**核心改动（对比同步 Trainer）**：

```python
# 同步版的 _fit_generate()：
def _fit_generate(self, batch):
    # 阻塞！等所有 rollout 完成
    gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)
    return batch

# 异步版的 _fit_generate()：
def _fit_generate(self, batch=None):
    # 非阻塞！从队列取已完成的样本
    epoch, batch = self._get_samples_from_queue()
    if batch is None:
        raise TrainingStopException()
    return batch
```

**训练循环**：

```python
async def fit(self):
    while True:
        try:
            await self.fit_step()  # 取样本 → reward → log_prob → advantage → update
        except TrainingStopException:
            break  # 队列发来终止信号
```

**参数同步触发**：

```python
# 每 trigger_parameter_sync_step 步触发一次参数同步
async def _fit_update_weights(self):
    if self.local_trigger_step < self.trigger_parameter_sync_step:
        self.local_trigger_step += 1
        return  # 还没到，跳过

    # 到了！触发同步
    self.current_param_version += 1
    self.local_trigger_step = 1
    ray.get(self.param_synchronizer.sync_weights.remote(self.current_param_version))
```

### 4.4 ParameterSynchronizer —— 参数同步器

**文件**：`verl/experimental/fully_async_policy/param_sync.py`

**职责**：协调 Trainer → Rollouter 的模型参数传输。

**同步流程**（修复后应该是这样）：

```
Trainer 触发 sync_weights(version=N)
  │
  ├─ 1. Rollouter.pause()     ← 暂停生成，等待在飞样本完成
  │
  ├─ 2. MessageQueue.update_param_version(N)
  │
  ├─ 3. actor_wg.sync_rollout_weights()     ← NCCL 广播：Trainer GPU → Rollouter GPU
  │     rollout_wg.sync_rollout_weights()    ← 两侧同时参与集合通信
  │
  ├─ 4. Rollouter.update_param_version(N)   ← 更新版本号
  │
  └─ 5. Rollouter.resume()    ← 恢复生成，使用新参数
```

**当前 P0 bug**：第 3 步的 NCCL 同步调用**被完全注释掉了**，导致只做了 pause + resume，参数实际没传。这是 Phase 0 的首要修复项。

---

## 5. 一个样本的完整生命周期（异步版）

以下是一个 CaRR deep-search 样本在异步架构中的完整旅程：

```
时刻 T=0: Rollouter 从 dataloader 取到一个 prompt
  │
  ├─ prepare_single_generation_data(batch_dict)
  │   → 构造 DataProto，repeat rollout.n=4 次
  │   → 设置 agent_name（当前硬编码 async_partial_tool_agent，CaRR 需改）
  │
  ├─ 构造 RolloutSample(full_batch, param_version=当前版本)
  │
  └─ 放入 pending_queue

时刻 T=1: _processor_worker() 取出样本
  │
  ├─ 检查是否 paused（参数同步中？） → 等待
  ├─ 检查并发数是否超限 → 等待
  │
  └─ asyncio.create_task(
       _process_single_sample_streaming(rollout_sample)
     )

时刻 T=2-T=N: Agent Loop 多轮执行（在 Rollouter GPU 上）
  │
  ├─ AgentLoopWorker.generate_sequences_no_post(batch, partial_output_list)
  │   │
  │   └─ 对 batch 中的每个样本（4 个）：
  │       asyncio.create_task(_partial_run_agent_loop(...))
  │       │
  │       └─ CaRRToolAgentLoop.run(sampling_params, ...)
  │           │
  │           ├─ PENDING: apply_chat_template → prompt_ids
  │           ├─ GENERATING: LLM 推理 → response_ids (mask=1)
  │           ├─ PROCESSING_TOOLS:
  │           │   ├─ 调用 browser.search → Serper API
  │           │   ├─ 调用 browser.open → Jina API
  │           │   ├─ 调用 browser.find → 本地匹配
  │           │   └─ 工具结果 tokenize → (mask=0)
  │           ├─ GENERATING: 继续推理...
  │           ├─ ... (多轮循环)
  │           └─ TERMINATED: 组装 AgentLoopOutput
  │
  ├─ 如果期间收到 cancel 信号（partial rollout）：
  │   → 保存 AgentData + AgentState → 放入 cancel_queue
  │   → 参数同步完成后从 cancel_queue 恢复继续
  │
  └─ 正常完成：组装 DataProto（含 response_mask, log_probs 等）

时刻 T=N+1: 样本完成，放入 MessageQueue
  │
  ├─ rollout_sample.param_version = current_param_version
  ├─ message_queue_client.put_sample(serialized_sample, param_version)
  │
  └─ total_generated_samples += 1

时刻 T=M: Trainer 从 MessageQueue 取出样本
  │
  ├─ _get_samples_from_queue()
  │   → 循环调用 get_sample_sync() 直到攒够 required_samples 个
  │   → 反序列化 RolloutSample 列表
  │   → assemble_batch_from_rollout_samples() → 合并为一个 DataProto
  │
  └─ 得到一个可训练的 batch

时刻 T=M+1: Trainer 执行训练步
  │
  ├─ _fit_compute_reward(batch)      ← 调用 reward manager（CaRR reward server）
  ├─ _fit_compute_log_prob(batch)    ← 用 rollout 时的 log_prob 作为 old_log_prob
  ├─ _fit_compute_ref_log_prob(batch) ← ref policy forward
  ├─ _fit_compute_advantage(batch)   ← C-GRPO 优势估计
  ├─ _fit_update_critic(batch)       ← PPO only
  ├─ _fit_update_actor(batch)        ← policy gradient update
  │
  └─ _fit_update_weights()
      → 如果达到 trigger_parameter_sync_step：触发参数同步
      → 否则 local_trigger_step += 1，继续下一个训练步
```

---

## 6. 同步 vs 异步：逐步对比

### GPU 时间利用率对比

```
同步架构（当前）：
时间 ──────────────────────────────────────────────────────>
GPU组  ┌── Rollout ──────────────┐┌─ Train ─┐┌── Rollout ──────────────┐┌─ Train ─┐
全部卡  │████████████ idle ██████ ││█████████││████████████ idle ██████ ││█████████│
       └─────────────────────────┘└─────────┘└─────────────────────────┘└─────────┘
                                 ↑
                            同步栅栏
                        长尾样本造成 idle

异步架构（目标）：
时间 ──────────────────────────────────────────────────────>
Rollout卡 │████ ██ ████████ ██████ ████ ██ ████████ ██████ ████ │  ← 持续生成
          │s1   s2  s3       s4    s5   s6  s7       s8    s9  │
          └────────────────────────────────────────────────────┘
                ↓ 完成就放入队列 ↓
              ┌─────────────────┐
              │   MessageQueue  │
              └─────────────────┘
                ↓ 攒够就取走 ↓
Train卡   │  ████  ████  ████  ████  ████  ████  ████  │  ← 持续训练
          │  t1    t2    t3    t4    t5    t6    t7    │
          └───────────────────────────────────────────┘
                               ↑
                        每 K 步参数同步（短暂暂停）
```

### 数据流对比表

| 步骤 | 同步版 | 异步版 |
|------|--------|--------|
| **数据来源** | `train_dataloader` 逐 batch | `_feed_samples()` 逐样本流式 |
| **Rollout** | `generate_sequences(全部样本)` 阻塞 | `generate_single_sample_async(单样本)` 非阻塞 |
| **样本传递** | 内存直传（同进程） | MessageQueue（Ray Actor 序列化） |
| **Batch 组装** | `batch.repeat(n).union(gen_output)` | `assemble_batch_from_rollout_samples(queue_samples)` |
| **old_log_prob** | Trainer 用当前参数重算 | **直接用 rollout 时计算的 log_prob**（`use_rollout_log_probs=True`） |
| **参数同步** | 每步 `checkpoint_manager.update_weights()` | 每 K 步 `param_synchronizer.sync_weights()` |
| **验证** | Trainer 侧同步执行 | Rollouter 侧或 Trainer 侧异步执行 |

---

## 7. 参数版本与 Staleness 控制

### 什么是 Staleness

异步训练中，Trainer 的参数版本可能领先于某些样本生成时的版本：

```
时间线：
  Rollouter 用 v0 生成 sample₁ ────────────────────► 放入队列
  Rollouter 用 v0 生成 sample₂ ────────────────────► 放入队列
  ───────── 参数同步 v0→v1 ─────────
  Rollouter 用 v1 生成 sample₃ ────────────────────► 放入队列
  Rollouter 用 v1 生成 sample₄ ────────────────────► 放入队列

  Trainer 现在是 v1，从队列取到 [sample₁(v0), sample₂(v0), sample₃(v1), sample₄(v1)]
  → sample₁ 和 sample₂ 是 "stale" 的（版本差 1）
```

### Staleness 控制参数

```yaml
async_training:
  staleness_threshold: 0.25   # 允许的最大 stale 样本比例
  # 0 = 完全同步（Rollouter 生成固定数量后必须等参数同步）
  # 0.25 = 允许 25% 的 stale 样本
  # 1.0 ≈ one-step off-policy
```

**工作原理**：Rollouter 在每次参数同步之间最多生成 `(1 + staleness_threshold) * required_samples * trigger_parameter_sync_step` 个样本。生成够了就 pause 等参数同步。

### old_log_prob 的处理

在异步架构中，有两种处理 old_log_prob 的方式：

**方式 A（默认，bypass_mode=True）**：直接用 rollout 时计算的 log_prob

```python
# Rollouter 在生成时就计算了 log_prob（calculate_log_probs=True）
# Trainer 直接用这个作为 old_log_prob，不需要重算
# 好处：简单、快
# 代价：如果 staleness 大，IS 权重可能不准
```

**方式 B（bypass_mode=False）**：Trainer 用版本 1 的参数重算

```python
# 保存当前参数到 CPU
actor_wg.save_model_to_cpu(current_version)
# 恢复 rollout 时的参数版本
actor_wg.restore_model_from_cpu(version_1)
# 用旧参数重算 log_prob
old_log_prob = actor_wg.compute_log_prob(batch)
# 恢复当前参数
actor_wg.restore_model_from_cpu(current_version)
```

**我们 CaRR 的初始 probe 用方式 A**（bypass_mode=True），因为更简单且 staleness_threshold 设得很小。

---

## 8. Partial Rollout：中断与恢复

### 什么是 Partial Rollout

当参数同步触发时，Rollouter 上可能有正在执行的 agent loop（已经跑了好几轮工具调用）。有两种处理方式：

**不启用 partial（partial_rollout=false）**：等待所有在飞任务完成后再同步。

```
参数同步请求到达
  │
  ├─ 等待所有 active_tasks 完成（可能等很久！）
  │
  └─ 然后才开始参数同步
```

**启用 partial（partial_rollout=true）**：中断在飞任务，同步后恢复。

```
参数同步请求到达
  │
  ├─ 设置 cancellation_event（通知所有 agent loop 停下来）
  │
  ├─ agent loop 在下一次 LLM 推理前检查 cancellation_event
  │   → 如果已设置：保存当前状态（AgentData + AgentState）到 cancel_queue
  │   → 立即返回 is_cancel=True
  │
  ├─ 所有 active_tasks 很快完成（不用等工具调用结束）
  │
  ├─ 执行参数同步（NCCL 广播）
  │
  ├─ 清除 cancellation_event
  │
  └─ 从 cancel_queue 取出中断的样本，继续执行（用新参数！）
```

### Partial Rollout 代码路径

```python
# 在 AsyncPartialToolAgentLoop._run_state_machine() 中：
while state != AgentState.TERMINATED:
    if cancellation_event and cancellation_event.is_set():
        # 被中断了！返回当前状态，不继续
        return state  # 不是 TERMINATED

    if state == AgentState.GENERATING:
        state = await self._handle_generating_state_partial(...)
    elif state == AgentState.PROCESSING_TOOLS:
        state = await self._handle_processing_tools_state(...)
    ...

# 中断后构造 cancelled output：
def _build_cancelled_output(self, agent_data, state):
    return AgentLoopOutput(
        extra_fields={
            "is_cancel": True,
            "agent_data": agent_data,   # 完整状态！
            "agent_state": state,       # 中断时的状态
        }
    )

# 恢复时：
def run(self, sampling_params, *, cancellation_event=None, **kwargs):
    output = kwargs.get("output", None)
    if output and output.extra_fields.get("is_cancel", False):
        # 从中断处恢复
        agent_data, state = self._restore_from_output(output)
        # 继续状态机
        state = await self._run_state_machine(agent_data, state, sampling_params, cancellation_event)
```

---

## 9. CaRR 接入异步的关键挑战

### 9.1 Agent Name 硬编码

**问题**：`detach_utils.py:prepare_single_generation_data()` 硬编码了 agent_name：

```python
# 当前代码（detach_utils.py:85-91）
if config.actor_rollout_ref.rollout.multi_turn.enable:
    full_batch.non_tensor_batch["agent_name"] = np.array(
        ["async_partial_tool_agent"] * len(full_batch), dtype=object
    )
```

**修复**：改为从配置读取，CaRR 配置指定 `carr_async_partial_tool_agent`。

### 9.2 CaRR 状态的 Cancel/Resume

基座的 `AsyncPartialToolAgentLoop` 只保存通用的 `AgentData`。但 CaRR 额外有：

| 状态 | 说明 | Cancel 时需要保存 |
|------|------|-------------------|
| `reward_history` | CaRR 格式的完整对话历史，含 tool_call_id 绑定 | 是 |
| `turn_idx` | 当前轮次号（用于生成 tool_call_id） | 是 |
| `search/open/find_count` | 工具调用计数 | 是 |
| `hit_limit/hit_budget` | 是否已触发限制 | 是 |
| `termination_reason` | 终止原因 | 是 |
| `rollout_start` | 开始时间（用于 wall budget） | 是 |
| tool session | tool server 端的会话状态 | **不需要**（server-side） |

Tool session 是 server-side stateful 的（以 `request_id` 标识），只要 resume 后带着相同的 `request_id`，就能继续使用同一个 session。

### 9.3 C-GRPO 组结构

GRPO/C-GRPO 需要同一 prompt 的 n 个样本做组内归一化。在异步架构中：

```python
# detach_utils.py:94
full_batch = full_batch.repeat(repeat_times=config.actor_rollout_ref.rollout.n, interleave=True)
```

同一 prompt 的 n 个样本被 repeat 后作为一个整体放入队列，所以**组结构天然保持**。Trainer 从队列取出的每个 RolloutSample 本身就包含 n 个 trajectory。

### 9.4 Reward 计算位置

Reward 在 **Trainer 侧** 计算（`_fit_compute_reward`），不在 Rollouter 侧。这意味着：
- CaRR reward server 仍然由 Trainer 调用
- reward_history 需要跟着样本通过 MessageQueue 传到 Trainer
- 不需要在 Rollouter 侧启动 reward server

---

## 10. 实施计划各 Phase 对应的代码改动

### Phase 0：基座修复

| 改动 | 文件 | 具体内容 |
|------|------|----------|
| **P0: 恢复真实 NCCL 同步** | `param_sync.py:130-141` | 取消注释 `sync_rollout_weights()` 调用，sglang 走直接同步路径 |
| **P0: 修 ZeroDivisionError** | `fully_async_trainer.py:367` | `% test_freq` 前判断 `> 0` |
| **P0: 修 TypeError** | `fully_async_trainer.py:368` | `_trigger_parameter_sync_after_step` 签名补齐或去掉错误 kwargs |
| **P0: checkpoint_engine 配置** | `carr_grpo_async_base.yaml` | sglang 主线设 `checkpoint_engine.enable=false` |
| **P1: fingerprint 验证** | 新增 debug RPC | actor/rollout 侧取参数摘要，sync 后对比 |

### Phase 0 Probe

| 改动 | 文件 | 具体内容 |
|------|------|----------|
| 新增 dummy reward | `reward/async_base_reward.py` | 只依赖 solution_str，不访问外部服务 |
| 新增 async base config | `config/carr_grpo_async_base.yaml` | multi_turn.enable=false，不启工具/奖励服务器 |

### Phase 1：CaRR Agent Loop 接入

| 改动 | 文件 | 具体内容 |
|------|------|----------|
| 新增 CaRR async agent | `tools/carr_agent_loop.py` | 注册 `carr_async_partial_tool_agent`，管理 CaRR 状态的 cancel/resume |
| 修 agent_name 硬编码 | `detach_utils.py:85-91` | 从配置读取 agent_name |
| session 幂等关闭 | `tools/carr_session_manager.py` | `close()` 做 best-effort，不依赖本地状态 |

### Phase 2：Budget 语义

| 改动 | 文件 | 具体内容 |
|------|------|----------|
| 新增 max_param_span | `tools/carr_agent_loop.py` | 防止坏 trajectory 跨太多版本 |
| 两层 wall time | `tools/carr_agent_loop.py` | active vs real wall time（可简化为只加 max_param_span） |

### Phase 3：CaRR Probe 配置

| 改动 | 文件 | 具体内容 |
|------|------|----------|
| CaRR async config | `config/carr_grpo_async.yaml` | 完整的 CaRR multi-turn + async 配置 |
| CaRR async 启动脚本 | `scripts/run_rl_async.sh` | 启动 tool/reward server + async main |

---

## 11. 关键文件索引

### 异步基座代码（verl/experimental/fully_async_policy/）

| 文件 | 作用 |
|------|------|
| `fully_async_main.py` | 入口：创建 Rollouter + Trainer + MessageQueue + ParameterSynchronizer |
| `fully_async_rollouter.py` | 流式样本生产者，管理并发和暂停 |
| `fully_async_trainer.py` | 队列消费者，管理训练步和参数同步触发 |
| `message_queue.py` | Ray Actor 消息队列 |
| `param_sync.py` | 参数同步协调器（**当前有 P0 bug**） |
| `detach_utils.py` | RolloutSample 数据结构、batch 组装、MetricsAggregator |
| `base_detach_sync.py` | NCCL 同步原语、动态 bucket、sglang/vllm 权重同步 |
| `checkpoint_engine.py` | 加速参数同步的 pinned memory + bucket 引擎（sglang 暂不支持） |
| `fsdp_workers.py` | FSDP 分离模式的 Actor/Rollout Worker |
| `config/fully_async_ppo_trainer.yaml` | 默认异步训练配置 |

### 异步 Agent Loop（verl/experimental/fully_async_policy/agent_loop/）

| 文件 | 作用 |
|------|------|
| `agent_loop.py` | FullyAsyncAgentLoopManager + FullyAsyncAgentLoopWorker（支持 cancel/resume） |
| `partial_tool_agent_loop.py` | AsyncPartialToolAgentLoop（支持中断恢复的工具 agent） |
| `partial_single_turn_agent_loop.py` | 单轮 agent 的 partial 版本 |

### 异步推理后端

| 文件 | 作用 |
|------|------|
| `sglang_rollout/sglang_async_server.py` | SGLang 可中断推理服务器 |
| `vllm_rollout/vllm_async_server.py` | vLLM 可中断推理服务器 |

### CaRR 项目代码（examples/carr_deepsearch/）

| 文件 | 作用 |
|------|------|
| `tools/carr_agent_loop.py` | CaRR 自定义 agent loop（**Phase 1 在此新增 async 版**） |
| `tools/carr_session_manager.py` | Tool server session 管理 |
| `tools/carr_browser_tool.py` | 浏览器工具适配器 |
| `reward/carr_reward.py` | CaRR reward 客户端 |
| `reward/cgrpo_advantage.py` | C-GRPO 优势估计器 |
| `config/carr_grpo.yaml` | 当前同步 RL 配置 |

---

## 附录：四种异步模式

`fully_async_policy` 支持从保守到激进的四种模式：

```
模式 a: On-policy pipeline
  trigger_parameter_sync_step=1, staleness_threshold=0
  → Rollouter 生成固定数量后停下等同步
  → 最保守，等价于"分离版同步训练"

模式 b: Stream off-policy pipeline
  trigger_parameter_sync_step>1, staleness_threshold=0
  → Rollouter 一次多生成几批，Trainer 分批消费
  → 中间步骤不同步参数

模式 c: Async stream + staleness
  trigger_parameter_sync_step>=1, staleness_threshold>0, partial_rollout=false
  → Rollouter 可以超额生成 stale 样本
  → 参数同步时等待在飞任务完成

模式 d: Async stream + partial rollout  ← CaRR 目标模式
  trigger_parameter_sync_step>=1, staleness_threshold>0, partial_rollout=true
  → 参数同步时中断在飞任务，同步后恢复
  → 最大化 GPU 利用率
```

**CaRR 的 probe 路径**：先用模式 b 验证基本语义（Phase 3 sync_stream），再用模式 d 验证 partial rollout（Phase 3 async_partial）。
