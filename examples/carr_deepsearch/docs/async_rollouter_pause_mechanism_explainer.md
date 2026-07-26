# verl 异步 RL 完整指南

> 本文档从零开始解释 verl `fully_async_policy` 的完整工作流程。
> 所有概念都配有具体数字实例。假设读者了解 PPO/GRPO 基本原理，但对异步架构零基础。
>
> 相关代码目录：`verl/experimental/fully_async_policy/`

---

## 1. 为什么要异步 RL

### 同步 RL 的长尾问题

同步 RL（verl 默认的 `RayPPOTrainer`）是一条严格的流水线：

```
一个 step 的流程：
  Rollout（所有样本） → Reward（所有样本） → Advantage → PPO 更新 → 下一个 step
     ↑                                                        ↓
     └────────────────── 全部完成后才继续 ──────────────────────┘
```

**问题**：一个 batch 里最慢的样本决定整步时间。

```
例：batch_size=128 的一步训练

样本  1: ████  30s 完成 → 等...
样本  2: ██████  45s 完成 → 等...
样本  3: ████████  60s 完成 → 等...
  ...
样本 127: ████████████████████████████████████  300s 完成 → 等...
样本 128: ██████████████████████████████████████████  360s（timeout！）

整步耗时 = 360s（最慢的那个）
其中样本 1 完成后空等了 330s，它占用的 GPU 全在发呆
```

deep search 任务长尾特别重：有的问题一次搜索就找到答案（30s），有的要搜 20 多轮（360s timeout）。

### 异步 RL 的解耦

异步 RL 把 Rollouter（生产者）和 Trainer（消费者）拆成两个独立进程：

```
同步 RL（工厂流水线全等最慢工人）：
  [所有工人干完] → [质检全部产品] → [包装全部产品] → [下一批]

异步 RL（完成一个交付一个）：
  工人A 完成 → 立刻送去质检 → 立刻包装
  工人B 完成 → 立刻送去质检 → 立刻包装
  工人C 还在干...（不影响 A、B 的交付）
```

快完成的样本立刻被消费，不用等慢的。GPU 利用率提高。

---

## 2. 四种异步模式（从保守到激进）

verl 的 `fully_async_policy` 支持四种模式，通过参数组合控制：

### 模式 a：on-policy（完全同步）

```
配置：staleness=0, trigger_sync_step=1

Rollouter: [生成4个样本]────────────────→ [等参数同步] → [生成4个样本]...
Trainer:   [等样本]  [训练] → [sync参数] → [等样本]  [训练] → [sync参数]...

特点：生成固定数量，等同步，再生成。最保守，无 stale 样本。
```

### 模式 b：sync_stream（流式但同步）

```
配置：staleness=0, trigger_sync_step>1（如 4）

Rollouter: [生成16个样本（一次性多生成）]─────────────────→ [等sync] → [再生成]
Trainer:   [取4个训练][取4个训练][取4个训练][取4个训练] → [sync] → ...

特点：一次生成更多，Trainer 流式消费，但所有样本仍是当前参数版本生成的。
```

### 模式 c：async + staleness（允许过期样本）

```
配置：staleness>0（如 0.5）, partial=false

Rollouter: [持续生成，不等 Trainer]──────────→ [sync时等在飞任务完成] → [继续]
Trainer:   [取4个训练][取4个训练]... → [触发sync] → [等在飞完成+传权重] → [继续]

特点：允许用旧参数生成的样本。sync 时等在飞任务自然完成（可能几分钟）。
```

### 模式 d：async + partial（中断并恢复）

```
配置：staleness>0, partial=true

Rollouter: [持续生成]────→ [sync时中断在飞任务] → [新参数恢复继续] → [持续生成]
Trainer:   [训练]... → [触发sync] → [cancel+传权重（几秒）] → [继续]

特点：最激进。sync 时中断在飞任务，保存中间状态，sync 完恢复继续。
      把 sync 从"等几分钟"缩短到"等几秒"。
```

### 模式对照表

| 模式 | staleness | trigger_sync_step | partial | sync 耗时 | stale 样本 |
|------|-----------|-------------------|---------|-----------|-----------|
| a: on-policy | 0 | 1 | - | 几秒 | 无 |
| b: sync_stream | 0 | >1 | - | 几秒 | 无 |
| c: async+staleness | >0 | >=1 | false | **几分钟** | 有 |
| d: async+partial | >0 | >=1 | **true** | **几秒** | 有 |

---

## 3. 整体架构

### 四个核心组件

```
┌───────────────────────┐          ┌───────────────────────┐
│       Rollouter       │          │        Trainer        │
│   （生产者，GPU 0-5） │  Queue   │   （消费者，GPU 6-7） │
│                       │ ──────▶  │                       │
│  持续生成样本         │          │  从 Queue 取样本       │
│  放入 MessageQueue    │          │  做 PPO 更新           │
└───────────┬───────────┘          └───────────┬───────────┘
            │                                  │
            │         ParameterSynchronizer    │
            │     ┌─────────────────────────┐  │
            └─────│ pause → 传权重 → resume │──┘
                  └─────────────────────────┘
```

### 请求分配：AsyncLLMServerManager

每张 rollout GPU 上起一个 SGLang 推理服务器（replica）。`AsyncLLMServerManager` 管理这些 server：

```
以 2:6 配置（2 GPU trainer + 6 GPU rollout）为例：
conc=6 个 prompt × n=8 个回答 = 48 个并发生成请求

AsyncLLMServerManager（最小堆负载均衡）：

  SGLang Server 0 (GPU 0): 分配 ~8 个请求
  SGLang Server 1 (GPU 1): 分配 ~8 个请求
  SGLang Server 2 (GPU 2): 分配 ~8 个请求
  SGLang Server 3 (GPU 3): 分配 ~8 个请求
  SGLang Server 4 (GPU 4): 分配 ~8 个请求
  SGLang Server 5 (GPU 5): 分配 ~8 个请求
```

两个调度策略：

- **最少请求负载均衡**：用最小堆跟踪每台 server 的请求数，新请求派给最空闲的
- **粘性会话**：同一个 prompt 的多轮对话总是路由到同一台 server，利用 prefix cache 加速（LRU 缓存 request_id→server 映射）

代码位置：`verl/experimental/agent_loop/agent_loop.py` line 61，`AsyncLLMServerManager`

---

## 4. 所有参数详解

### 训练侧参数

| 参数 | 含义 | 例子 |
|------|------|------|
| `ppo_mini_batch_size` (b) | Trainer 一次 PPO 更新需要多少个 prompt | b=4：每次取 4 个 prompt |
| `rollout.n` (n) | 每个 prompt 并行生成多少个不同回答 | n=8：每个问题生成 8 个回答互相对比 |
| `require_batches` | 每次更新前要收集几个 mini-batch | 1：凑够 1 个 batch 就开始训练 |

```
一次训练的实际数据量 = b × n = 4 × 8 = 32 条轨迹
```

### 异步控制参数

| 参数 | 含义 | 例子 |
|------|------|------|
| `staleness_threshold` | 允许多大比例的"过期样本"缓冲 | 0.5：允许 50% 额外缓冲 |
| `trigger_parameter_sync_step` | Trainer 训练几次后同步一次参数 | 4：每 4 次训练同步一次 |
| `max_concurrent_samples` (conc) | Rollouter 最多同时处理几个 prompt | 6：最多 6 个 prompt 同时在飞 |
| `max_queue_size` | MessageQueue 最多积压几个完成样本 | 12：防止内存爆炸 |

### Partial Rollout 参数

| 参数 | 含义 | 例子 |
|------|------|------|
| `partial_rollout` | 参数同步时是否可以中断/恢复在飞任务 | true：启用 cancel/resume |
| `max_param_span` | 一个样本最多跨几个参数版本 | 2：从 v1 到 v3 还没完就强制终止 |

### 引擎配置参数

| 参数 | 含义 | 为什么这样设 |
|------|------|------------|
| `gpu_memory_utilization=0.3` | SGLang 用 30% 显存做 KV cache | 96GB 卡上约 29GB，剩余给模型权重 |
| `use_dynamic_bsz=false` | 关闭动态 micro-batch 打包 | 防止不同 rank 的 micro-batch 数量不一致导致 NCCL hang（见第 9 节） |
| `enforce_eager=true` | SGLang 不用 CUDA graph | async 场景下 CUDA graph 有递归问题 |
| `checkpoint_engine.enable=false` | 关闭 GPU 加速参数广播 | 当前用直接权重同步，不需要额外引擎 |

### 关键推导公式

```
以 b=4, staleness=0.5, trigger_sync_step=4 为例：

required_samples = b × require_batches = 4 × 1 = 4
  → Trainer 每个 global_step 消费 4 个 prompt

max_required_samples = required_samples × (1 + staleness) × trigger_sync_step
                     = 4 × (1 + 0.5) × 4
                     = 4 × 1.5 × 4
                     = 24
  → 一个 sync 周期内 Rollouter 最多生成 24 个 prompt
  → 超过 24 个就自行暂停（但 partial 模式下这个条件被跳过）

实际并发请求数 = conc × n = 6 × 8 = 48 个生成请求同时跑在 SGLang 上
```

---

## 5. 三个计数器

async 训练里有三个容易混淆的计数器：

| 计数器 | 含义 | 类比 |
|--------|------|------|
| `global_steps` | PPO mini-batch 训练次数 | 工人搬了多少趟砖 |
| `local_trigger_step` | 本轮 sync 周期内已做了几次训练 | 搬到第几趟了（满 4 趟就歇一次） |
| `current_param_version` | 同步了多少次参数（日志中的 step:N） | 工人歇了几次（每次歇时把新图纸给生产线） |

### 完整时间表（trigger_sync_step=4）

```
global_steps:        1     2     3     4     5     6     7     8
local_trigger_step:  1     2     3     4     1     2     3     4
                     训练  训练  训练  训练  训练  训练  训练  训练
                                       ↓                       ↓
                                  param sync              param sync
param_version:                    0 → 1                   1 → 2
日志输出:                         step:1                  step:2
checkpoint:                    global_step_1           global_step_2
```

### step:N 日志为什么不是每次训练都打

因为日志用 `param_version` 作为 step 编号，而 `param_version` 只在 param sync 时才 +1。中间的 global_step 2, 3 只是在积累梯度更新，还没同步给 Rollouter，不单独记录。

代码位置：`fully_async_trainer.py` line 701-714

```python
async def _trigger_parameter_sync_after_step(self):
    if self.local_trigger_step < self.trigger_parameter_sync_step:
        self.local_trigger_step += 1
        return  # ← 还没攒够，不 sync，不输出 step 日志

    self.current_param_version += 1
    self.logger.log(step=self.current_param_version)  # ← 这里才输出 step:N
    # ... 然后做 param sync
```

### global_step_0 是什么

`global_step_0` 是**训练开始前的初始 checkpoint**——原始模型参数的快照，还没做任何梯度更新。目的是：如果后续训练崩了，可以从这个点恢复，不用重新做初始化。

它在初始 param sync 完成后、训练循环开始前保存（`param_version=0`, `save_freq` 条件满足时）。

### 具体例子

```
日志看到：global_steps: 11, local_trigger_step: 3, trigger_parameter_sync_step: 4

意味着：
- 已经做了 11 次 mini-batch 训练
- 本轮已攒了 3 次，还差 1 次触发 sync
- 已经同步过 2 次参数（step:1 在 global_step 4 后，step:2 在 global_step 8 后）
- global_step 12 完成后 → local_trigger_step 达到 4 → 触发 sync → 输出 step:3
```

---

## 6. 端到端时间线走读

**配置**：b=4, n=8, conc=6, trigger_sync_step=4, staleness=0.5, partial=true, 2 GPU trainer + 6 GPU rollout

### 阶段 0：初始化（t=0）

```
① Trainer 加载模型权重 → current_param_version = 0
② Rollouter 加载模型权重
③ 初始 param sync → NCCL 广播确保两边权重一致（~3s）
④ 保存 global_step_0 checkpoint
⑤ Rollouter 启动 _processor_worker 开始生产
```

**这里的 param sync 不是训练中的 sync，而是启动时的初始化校验。**

### 阶段 1：Rollouter 开始并发生产（t=10s）

```
_processor_worker 从 pending_queue 取 prompt 并派发到 SGLang：

  prompt_0 → SGLang（8 个回答并发生成，多轮 search→open→find）
  prompt_1 → SGLang
  prompt_2 → SGLang
  prompt_3 → SGLang
  prompt_4 → SGLang
  prompt_5 → SGLang
  ← 到达 conc=6 上限，等有样本完成才能调度新的

此时 Trainer 在 _fit_generate() 里等 Queue 凑够 4 个...
```

### 阶段 2：样本陆续完成（t=150s~300s）

```
t=150s  prompt_2 完成（快样本，30s 搜到答案） → 进 Queue
t=200s  prompt_0 完成 → 进 Queue
        Rollouter 空出 2 个位 → 立即调度 prompt_6, prompt_7
t=250s  prompt_5 完成 → 进 Queue
t=300s  prompt_1 完成 → 进 Queue（第 4 个！）
```

### 阶段 3：Trainer 取到第一批（t=300s, global_step 1）

```
t=300s  Queue 凑够 4 个 → Trainer 取走！等了 380s（含冷启动）

  Trainer 执行 fit_step():
    ① compute_reward      → 调 DeepSeek Judge 打分
    ② compute_log_prob    → 当前策略的概率
    ③ compute_ref_log_prob → 参考策略的概率
    ④ compute_advantage    → C-GRPO 组内对比（8 个回答互比）
    ⑤ update_actor         → PPO 梯度更新（反向传播 + 优化器 step）
    ⑥ check sync: local_trigger_step=1 < 4 → 不同步
    ⑦ global_steps: 1 → 2

  耗时约 80s

  与此同时 Rollouter 完全不受影响，继续生产 prompt_6, 7, 8...
```

### 阶段 4：后续 global_step 更快（t=380s~800s）

```
t=380s  global_step 1 完成 → 立刻从 Queue 取第二批
        Queue 里已有积压 → 只等 ~137s 凑够 4 个（pipeline 已预热）

t=520s  global_step 2 完成  local_trigger_step=2/4

t=660s  global_step 3 完成  local_trigger_step=3/4

t=800s  global_step 4 完成  local_trigger_step=4/4 → 触发 param sync！
```

**为什么第二批只要 137s？**
第一批 380s 包含了冷启动开销（SGLang warmup、从零开始生成）。第一批等待期间，Rollouter 并发生产了更多样本。后续批次到 Queue 时已经有存货，凑够 4 个的间隔缩短。

### 阶段 5：第一次 param sync（t=800s）

```
_trigger_parameter_sync_after_step():
  current_param_version: 0 → 1
  输出聚合指标 → 日志中出现 step:1 ← 第一条正式指标！

  sync_weights() 内部（详见第 7 节）：
    ① rollouter.pause()        → cancel 在飞任务 + 等当前轮完成（~15s）
    ② NCCL broadcast 新权重   → Trainer → Rollouter（~3s）
    ③ 验证 fingerprint         → 确保两边一致
    ④ rollouter.resume()       → 恢复被中断的样本 + 用新参数继续

  local_trigger_step 重置为 1
  Rollouter 用 param version 1 继续生产
```

### 双列并行时间线图

```
时间     Rollouter (6 GPU)                              Trainer (2 GPU)
─────────────────────────────────────────────────────────────────────────
t=0      初始化 + 初始 param sync (3s) + 保存 global_step_0
         │                                               │
t=10     并发生成 prompt 0~5                              等样本...
         │ [prompt_0 ████████████████]→Q                  │
         │ [prompt_1 ██████████████████████]→Q            │
         │ [prompt_2 ████████]→Q                          │
         │ [prompt_3 ████████████████████]→Q              │
         │ [prompt_4 ██████████████████████████]→Q        │
         │ [prompt_5 ██████████████]→Q                    │
         │                                               │
t=300    继续生成 6,7,8...                    ← 取4个 → global_step 1
         ████████████████████                  [训练80s]  local=1/4
t=380    继续生成...                          ← 取4个 → global_step 2
         ████████████████                      [训练80s]  local=2/4
t=520    继续生成...                          ← 取4个 → global_step 3
         ████████████████                      [训练80s]  local=3/4
t=660    继续生成...                          ← 取4个 → global_step 4
         ████████████████                      [训练80s]  local=4/4
         │                                               │
t=800    ┣━━━━━━ param sync (v0→v1, ~18s) ━━━━━━━━━━━━━━┫
         │ pause→cancel→传权重→resume                     │ 输出 step:1
t=818    │                                               │
         用新参数继续                         ← 取4个 → global_step 5
         (从 cancel_queue 恢复半成品)          [训练80s]  local=1/4
         ████████████████████                              ...
```

### 一个样本的完整生命周期

```
prompt_0 的一生：

t=10s    被 _processor_worker 从 pending_queue 取出
         → 创建 asyncio task，加入 active_tasks
         → AsyncLLMServerManager 选择负载最低的 SGLang server
         → 派发 8 个回答的生成请求（n=8）

t=10s    Agent Loop 开始：
         轮次 1: 生成 → search("量子计算") → 拿到 10 条结果 → [检查 cancel] → 继续
         轮次 2: 生成 → open(result_3) → 拿到网页内容 → [检查 cancel] → 继续
         轮次 3: 生成 → find("量子纠缠") → 找到段落 → [检查 cancel] → 继续
         ...
         轮次 15: 生成 → 输出最终答案 → state=TERMINATED

t=200s   8 个回答全部完成 → 打包成 RolloutSample
         → message_queue_client.put_sample()
         → 进入 MessageQueue 等 Trainer 取走
         → 从 active_tasks 中移除
         → Rollouter 空出一个并发位，调度下一个 prompt
```

---

## 7. 参数同步（param sync）详解

### sync_weights 的 5 步流程

代码位置：`param_sync.py` line 163-203

```
sync_weights(version=1) 内部：

  步骤               做什么                              耗时
  ─────────────────────────────────────────────────────────────
  ① pause()         暂停 Rollouter                       见下文
  ② update_version  更新 MessageQueue 版本号              <0.1s
  ③ NCCL broadcast  Trainer 权重 → Rollouter 权重        ~3s
                    （4B 参数 × 2 bytes bf16 ≈ 8GB 传输）
  ④ fingerprint     校验两边权重一致                       <0.1s
  ⑤ resume()        恢复 Rollouter 生产                   <0.1s
```

### cancel 的真实语义："挂红旗"

**cancel 不是强制终止，而是一个异步通知。**

```
pause() 调用 cancel():
  → 设置 cancellation_event（挂一面红旗）
  → 不会中断正在运行的代码

在飞的 agent loop 任务内部（每一轮循环开头）：
  while state != TERMINATED:
      if cancellation_event.is_set():    ← 检查红旗
          保存中间状态（搜索历史、budget、工具记录）
          放入 cancel_queue
          return                          ← 优雅退出

      if state == GENERATING:
          调 SGLang 生成...              ← 这一步可能要 5-60s
      elif state == PROCESSING_TOOLS:
          调工具服务器...                 ← 这一步可能要 1-30s
```

**关键**：正在进行的那一轮 LLM 生成或工具调用必须完成，任务才会在**下一个检查点**看到红旗并退出。

```
例：cancel 信号到达时的 3 个在飞任务

prompt_18: 正在第 13 轮，SGLang 正在生成 token...
           → 生成完成（5s）→ 检查红旗 → 看到了 → 保存状态 → 退出

prompt_19: 正在第 6 轮，工具服务器正在请求 Jina 抓网页...
           → HTTP 返回（15s）→ 检查红旗 → 看到了 → 保存状态 → 退出

prompt_20: 正在第 1 轮，SGLang 正在生成...
           → 生成完成（8s）→ 检查红旗 → 看到了 → 保存状态 → 退出

gather 等最慢的 = 15s → 然后传权重 3s → 总计 ~18s
```

### 对比：partial 生效 vs 未生效

```
partial 正常工作时（只等当前轮完成）：
  cancel → 3 个任务各自完成当前轮 → 最慢的 15s → 传权重 3s
  总计 = 18s

partial 未生效时（等所有轮次自然完成）：
  没有 cancel 信号 → 3 个任务要跑完所有剩余轮次
  prompt_18 还有 7 轮 → 120s
  prompt_19 还有 16 轮 → 200s
  prompt_20 还有 24 轮 → 248s（接近 timeout）
  等最慢的 = 248s → 传权重 3s
  总计 = 251s
```

**差别：18s vs 251s。这就是 partial rollout 的价值。**

### 248s 的精确拆解

之前实际观测到的 `timing_s/param_sync=248s`，到底在等什么？

```
timing_s/param_sync 的测量范围（fully_async_trainer.py line 721-729）：
  with marked_timer("timing_s/param_sync"):
      ray.get(param_synchronizer.sync_weights.remote(...))

  sync_weights 内部（param_sync.py line 163-203）：
      ray.get(rollouter.pause.remote())  ← Trainer 侧在这里阻塞！
      NCCL broadcast                     ← 这只要 3s
      resume                             ← 这是异步的

所以 248s ≈ pause() 的耗时
```

pause() 里面发生了什么？ → 见第 8 节的两种暂停机制。

---

## 8. 两种暂停机制

Rollouter 有两种完全不同的"停"的方式。这是理解 async RL 最关键的部分。

### 暂停 A：Self-Pause（Rollouter 自己暂停自己）

**谁触发**：`_processor_worker` 在每次循环开头自己检查

**触发条件**：
1. `queue_full`：MessageQueue 满了（`queue_size >= max_queue_size`）
2. `staleness`：已生产样本超过阈值（`staleness_samples >= max_required_samples`）
   - 但 `partial_rollout=true` 时，**staleness 条件被跳过**！

**行为**：等在飞任务**自然完成所有轮次**（不发 cancel）

```
工厂比喻：
  工人数了数仓库里的产品："够了，我自己停下来休息"
  → 手里正在做的活（几件产品）做完再停
  → 每件产品必须做完所有工序（可能要几分钟）
  → 然后坐下等主管来叫
```

代码位置：`fully_async_rollouter.py` line 567-595

```python
async def _processor_worker(self):
    while True:
        pause_reason = await self._get_pause_reason()
        if self.paused or pause_reason is not None:
            self.paused = True
            while self.active_tasks:                  # ← 等所有在飞任务自然完成
                done_tasks, self.active_tasks = await asyncio.wait(
                    self.active_tasks, return_when=asyncio.FIRST_COMPLETED
                )
            while self.paused:
                await self.condition.wait()            # ← 睡眠，等别人唤醒
```

### 暂停 B：External Pause（Trainer 从外部要求暂停）

**谁触发**：Trainer 完成训练后，调 `sync_weights()` → 内部调 `pause()`

**行为**：发 cancel 信号（红旗），在飞任务只需等**当前这一轮**完成

```
工厂比喻：
  主管跑过来喊："停！要换新图纸了！"
  → 工人放下手里正在做的那一道工序（等这道做完）
  → 把半成品放到旁边架子上（cancel_queue）
  → 换完新图纸后，拿起半成品继续做
```

代码位置：`fully_async_rollouter.py` line 861-893

```python
async def pause(self):
    self.paused = True
    if self.config.async_training.partial_rollout:
        await self.async_rollout_manager.cancel()    # ① 挂红旗
    if self.active_tasks:
        await asyncio.gather(*self.active_tasks)     # ② 等任务退出（几秒）
        self.active_tasks.clear()
    await self.async_rollout_manager.clear_kv_cache() # ③ 清 KV cache
```

### 对比表

| | Self-Pause (暂停 A) | External Pause (暂停 B) |
|---|---|---|
| **谁触发** | `_processor_worker` 自己 | Trainer 调 `pause()` |
| **触发条件** | queue 满 或 staleness 超限 | Trainer 要同步参数 |
| **是否发 cancel** | **不发** | **发**（挂红旗） |
| **等待什么** | 在飞任务**跑完所有轮次** | 在飞任务**跑完当前这一轮** |
| **等待时间** | 可能 **几分钟** | 通常 **几秒到十几秒** |
| **任务去向** | 正常完成 → message_queue | 中断保存 → cancel_queue → 后续恢复 |

### 设计缺陷：Self-Pause 抢先于 External Pause

当 Self-Pause 先触发时，它把在飞任务全部等完（drain），等 External Pause 到达时已经没有在飞任务可以 cancel 了。

```
有缺陷的时序（trigger_sync_step=1, staleness=0.25）：

t=0      Rollouter 生产了 5 个，staleness_samples=5 >= max_required_samples=5
         → Self-Pause 触发！
         → _processor_worker 开始等在飞任务自然完成（不发 cancel）

t=0~248s 3 个在飞任务各自跑完所有轮次... 248s 后全部完成
         → active_tasks 变空
         → _processor_worker 进入 condition.wait()

t=248s   Trainer 训练完，调 sync_weights() → pause()
         → cancel() → 挂红旗 → 但没有任务在看红旗了！
         → gather(空集) → 0s
         → 传权重 → 3s
         → resume()

total timing_s/param_sync ≈ 251s（其中 248s 是 Self-Pause 白等的）
```

```
正确的时序（Self-Pause 没有抢先）：

t=0      Trainer 训练完，调 pause()
         → cancel() → 挂红旗
         → 3 个在飞任务在下一个检查点看到红旗 → 保存中间状态 → 退出
         → gather 等最慢的 = 15s
         → 传权重 3s

total timing_s/param_sync ≈ 18s
```

### 当前代码的规避方式

**规避 1：partial 模式下跳过 staleness self-pause**

```python
# _get_pause_reason() 中（line 840-850）：
if self.staleness_samples >= self.max_required_samples:
    if self.enable_partial_rollout:
        return None  # ← partial 模式下不触发 staleness self-pause！
    return "staleness"
```

在 `partial_rollout=true` 下，staleness 条件**永远不会**触发 Self-Pause。唯一能触发的只剩 `queue_full`。

**规避 2：trigger_sync_step=4 让阈值更大**

```
max_required_samples = 4 × 1.5 × 4 = 24
conc=6，同时最多 6 个在飞 + queue 里几个 ≈ 远小于 24
→ 即使在非 partial 模式下，staleness self-pause 也很难触发
```

**残留风险**：`queue_full` 仍然可以触发 Self-Pause，且不发 cancel。如果 Trainer 消费极慢导致 queue 积满，Rollouter 仍会走 drain 逻辑。

---

## 9. 已知问题与陷阱

### 问题 1：Livelock（活锁）

**场景**：`trigger_sync_step=1` + `partial_rollout=true` + 样本生成时间长

```
sync 间隔 ≈ 15s（训练 1 次很快）
样本生成时间 ≈ 35s

时间线：
  t=0    resume → 从 cancel_queue 恢复 sample_0_13 → 开始生成
  t=5    sample_0_13 才跑了 5s...
  t=15   Trainer 训练完 → 触发 sync → pause → cancel！
         → sample_0_13 又被 cancel → 放回 cancel_queue

  t=18   resume → 恢复 sample_0_13 → 又开始生成
  t=23   才跑了 5s...
  t=33   Trainer 又训练完 → 又 cancel！
         → sample_0_13 再次放回 cancel_queue

  → 永远循环，sample_0_13 永远跑不完
```

**根因**：sync 间隔（15s）< 样本生成时间（35s），样本永远在下一次 sync 之前被打断。

**解法**：增大 `trigger_sync_step`（当前设 4），或给被 cancel 过的样本设保护期。

### 问题 2：Dynamic Batch 导致 NCCL Hang

**场景**：`use_dynamic_bsz=true` 时，FSDP 训练的 PPO update 阶段

`use_dynamic_bsz` 是什么：不按固定样本数切分 micro-batch，而是按 token 总量打包，让每个 micro-batch 的计算量接近，提高 GPU 利用率。

```
问题：各 rank 的样本长度不同 → 切出不同数量的 micro-batch

Rank 0: 总 30000 token ÷ 8192 = 4 个 micro-batch
Rank 1: 总 29000 token ÷ 8192 = 4 个 micro-batch
Rank 2: 总 28000 token ÷ 8192 = 4 个 micro-batch
Rank 3: 总 22000 token ÷ 8192 = 3 个 micro-batch  ← 少一个！

Rank 3 跑完 3 个 micro-batch → 进入 clip_grad_norm（需要跨 rank allreduce）
Rank 0/1/2 还在跑第 4 个 micro-batch（forward 里有 FSDP allgather）

两边在不同的 NCCL 操作上互相等 → 死锁
```

**解法**：设 `use_dynamic_bsz=false`（当前配置已经这样设了）。

### 问题 3：trigger_sync_step 两难困境

| trigger_sync_step | pause 耗时 | partial 效果 | off-policy 程度 |
|-------------------|-----------|-------------|----------------|
| 1 | 很长（在飞任务多）或 livelock | 有机会触发 | 低（接近 on-policy） |
| 2 | 中等 | 部分触发 | 中等 |
| 4（当前） | 很短（~3s，任务已自然完成） | **几乎不触发** | 较高 |
| 8 | 极短 | 完全不触发 | 高 |

**矛盾**：
- sync 频率越高 → 越可能撞上在飞任务（partial 有用武之地）→ 但 pause 慢或 livelock
- sync 频率越低 → 在飞任务自然跑完（partial 无用武之地）→ 但 pause 快

当前配置 `trigger_sync_step=4` 选择了"sync 快但 partial 失效"的 trade-off。在样本 72% timeout 的现状下，这是务实的选择——因为大部分样本都是慢的，partial 的收益有限。

---

## 10. 关键指标速查表

### Timing 指标

| 指标 | 含义 | 健康值 | 异常时说明什么 |
|------|------|--------|--------------|
| `timing_s/param_sync` | 一次参数同步的总时间 | < 30s | > 100s：Self-Pause drain 太慢，或 partial 未生效 |
| `timing_s/gen` | 等 Queue 凑够一个 mini-batch 的时间 | < 200s | > 500s：Rollouter 产能不足（GPU 太少或并发太低） |
| `timing_s/update_actor` | PPO 梯度更新时间 | < 100s | > 200s：显存瓶颈或序列太长 |
| `timing_s/step` | 一个完整 step（含多个 global_step）的时间 | 看配置 | 除以 trigger_sync_step 得到单个 global_step 时间 |

```
例：timing_s/step=1262s, trigger_sync_step=4
→ 每个 global_step ≈ 1262/4 = 315s
→ 其中 gen_wait ≈ 237s（等样本）+ train ≈ 79s（训练）
```

### 异步专属指标

| 指标 | 含义 | 关注点 |
|------|------|--------|
| `trainer/idle_ratio` | Trainer 空闲率 | 高 → Rollouter 产能不足 |
| `rollouter/idle_ratio` | Rollouter 空闲率 | 高 → Trainer 消费太慢 |
| `fully_async/count/stale_samples_processed` | 累计使用的过期样本数 | 持续增长说明有 stale 样本进入训练 |
| `fully_async/partial/total_partial_num` | 被 cancel/resume 的样本数 | =0 说明 partial 没触发 |
| `fully_async/partial/partial_ratio` | 跨版本样本占比 | =0 同上 |
| `monitor/queue/cancel_queue_size` | cancel_queue 中的半成品数 | > 0 说明有样本被中断等待恢复 |

### 如何判断瓶颈

```
trainer/idle_ratio 高 + rollouter/idle_ratio 低
  → Rollouter 是瓶颈（生产慢）
  → 增加 rollout GPU 或降低并发压力

trainer/idle_ratio 低 + rollouter/idle_ratio 高
  → Trainer 是瓶颈（消费慢）
  → 增加 trainer GPU 或降低 batch size

两个都高
  → param sync 太频繁，两边都在等同步
  → 增大 trigger_sync_step

两个都低
  → 理想状态，两边都在忙
```

### 训练信号指标

| 指标 | 含义 | 关注点 |
|------|------|--------|
| `outcome_reward/mean` | 平均奖励分 | 应该随训练逐步上升 |
| `timeout_ratio` | 超时样本占比 | > 50% 说明样本太难或 budget 太小 |
| `unfinished_ratio` | 未完成样本占比 | 高 → reward 被 short-circuit 为 0 |
| `response_length/mean` | 平均回答长度 | 不应持续膨胀（可能是无效循环） |

---

## 附录：Stale（过期）样本解释

在 RL 训练中，rollout 生成样本时用的是**某一版本的策略参数**。如果 Trainer 已经更新了参数（比如从 v2 → v3），但 Rollouter 还在用 v2 的参数生成——这些样本就是 **stale（陈旧/过期的）**。

```
类比：
  厨师按旧版菜谱做了 5 道菜 → 菜谱更新了 → 这 5 道菜是"过期版本"
  用过期菜做质量评估 → 结论可能不准确（on-policy 偏差）

技术影响：
  PPO 的 importance sampling ratio = π_new / π_old
  如果 π_old 用的是 stale 参数计算的 → ratio 偏离 → 梯度不准确
```

`staleness_threshold` 控制允许多大比例的 stale 样本缓冲。它不是直接控制"过期程度"，而是通过影响 Rollouter 的暂停门槛间接控制：

```
staleness=0.0 → max_required_samples = b × 1.0 × sync_step = 刚好够 Trainer 用
               → 没有缓冲，Rollouter 生产够了就停 → 等同同步

staleness=0.5 → max_required_samples = b × 1.5 × sync_step = 多 50% 缓冲
               → Rollouter 可以多生产一些 → 部分样本可能是旧参数生成的

staleness=1.0 → max_required_samples = b × 2.0 × sync_step = 双倍缓冲
               → Rollouter 几乎不暂停 → stale 样本最多
```
