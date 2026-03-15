# CaRR Deep Search 训练链路 Infra 全解析

本文档系统解析 CaRR Deep Search 在 verl 框架下的完整训练链路架构，包括数据流、并发模型、同步屏障和瓶颈分析。目标是为后续 infra 优化提供清晰的架构理解。

---

## 1. 训练主循环概览

训练入口 `python -m verl.trainer.main_ppo`，核心循环在 [ray_trainer.py](../../../verl/trainer/ppo/ray_trainer.py) 的 `RayPPOTrainer.fit()` 方法中（约 line 1224 起）。

### 1.1 每个 training step 的 6 个阶段

```
┌────────────────────────────────────────────────────────────────────────────┐
│                          一个 Training Step                                │
│                                                                            │
│  ① Rollout (gen)         ← GPU busy (推理) + GPU idle (等工具/API)         │
│  │  └─ sleep_replicas()  ← 释放推理引擎 GPU 显存                          │
│  │                                                                         │
│  ② Reward                ← GPU idle (等外部 reward server HTTP)            │
│  │                                                                         │
│  ③ Old Log Prob          ← GPU busy (前向推理，计算 π_old)                 │
│  │                                                                         │
│  ④ Advantage             ← CPU only (driver 侧计算，不用 GPU)             │
│  │                                                                         │
│  ⑤ Update Actor          ← GPU busy (前向+反向+优化器)                     │
│  │                                                                         │
│  ⑥ Update Weights        ← GPU busy (权重同步到推理引擎)                   │
│  │  └─ wake_replicas()   ← 恢复推理引擎                                   │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 实测耗时分布（8×H200, b4/n4, TP=1）

```
阶段                  耗时         占比      GPU 状态
─────────────────────────────────────────────────────
① gen (Rollout)      426.81s      85.7%     推理时 busy, 等工具时 idle
③ old_log_prob        19.18s       3.9%     busy
⑤ update_actor        37.15s       7.5%     busy
⑥ update_weights      14.59s       2.9%     busy
─────────────────────────────────────────────────────
total step           497.81s     100.0%
wall clock           578.00s              (含 metrics/logging)
```

**结论**：Rollout 阶段占总时间 85%+，是唯一值得重点优化的阶段。

### 1.3 关键代码路径

```python
# ray_trainer.py:1312-1316
with marked_timer("gen", timing_raw):
    gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)  # ← 阻塞等待所有样本完成
    self.checkpoint_manager.sleep_replicas()  # ← 释放推理引擎显存

# ray_trainer.py:1374-1378
with marked_timer("reward", timing_raw):
    if self.use_rm and "rm_scores" not in batch.batch.keys():
        batch_reward = self._compute_reward_colocate(batch)  # ← 如果用 reward model
    reward_tensor, reward_extra_infos_dict = extract_reward(batch)

# ray_trainer.py:1400
with marked_timer("old_log_prob", timing_raw):
    old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)

# ray_trainer.py:1489-1497
batch = compute_advantage(batch, adv_estimator=self.config.algorithm.adv_estimator, ...)

# ray_trainer.py:1513-1514
with marked_timer("update_actor", timing_raw):
    actor_output = self._update_actor(batch)

# ray_trainer.py:1541-1542
with marked_timer("update_weights", timing_raw):
    self.checkpoint_manager.update_weights()
```

---

## 2. Rollout 阶段详解：三层并发模型

Rollout 是一个多层嵌套的异步并发架构。理解每一层的并发语义和同步屏障是优化的前提。

### 2.1 架构图

```
RayPPOTrainer.fit()
  │
  ▼
AgentLoopManager.generate_sequences()          ← Layer 0: 入口
  │  将 batch 切分到 N 个 worker
  │  ray.get([worker.generate_sequences.remote(chunk) ...])  ← 同步屏障 A
  │
  ├── AgentLoopWorker[0].generate_sequences()  ← Layer 1: Ray Worker
  │     │  asyncio.gather(*tasks)              ← 同步屏障 B
  │     │
  │     ├── _run_agent_loop(sample_0)          ← Layer 2: 样本级 asyncio 协程
  │     │     │
  │     │     ├── GENERATING → server_manager.generate()     ← GPU 推理（异步 HTTP）
  │     │     ├── PROCESSING_TOOLS → asyncio.gather(*tools)  ← Layer 3: 工具并发
  │     │     │     ├── CaRRBrowserTool.execute(search)       ← HTTP → Tool Server
  │     │     │     ├── CaRRBrowserTool.execute(open)         ← HTTP → Tool Server → Jina
  │     │     │     └── CaRRBrowserTool.execute(find)         ← 本地字符串匹配
  │     │     ├── GENERATING → ...  (循环多轮)
  │     │     └── TERMINATED → 输出 extra_fields
  │     │
  │     ├── _run_agent_loop(sample_1)
  │     └── ...
  │
  ├── AgentLoopWorker[1].generate_sequences()
  └── ...
```

### 2.2 Layer 0: AgentLoopManager — batch 分发与 Ray 同步

**文件**: [agent_loop.py:944-964](../../../verl/experimental/agent_loop/agent_loop.py#L944-L964)

```python
def generate_sequences(self, prompts: DataProto) -> DataProto:
    num_workers = len(self.agent_loop_workers)
    active_workers = min(num_workers, len(prompts))
    # 确保能整除
    while active_workers > 1 and len(prompts) % active_workers != 0:
        active_workers -= 1

    chunkes = prompts.chunk(active_workers)
    outputs = ray.get([                                    # ← 同步屏障 A
        worker.generate_sequences.remote(chunk)
        for worker, chunk in zip(self.agent_loop_workers[:active_workers], chunkes)
    ])
    output = DataProto.concat(outputs)
```

**关键点**：
- `ray.get()` 是**阻塞调用**，等待所有 worker 完成才返回
- 如果某个 worker 分到的样本中有一个特别慢（长尾样本），所有 worker 都得等它
- `num_workers` 由 `actor_rollout_ref.rollout.agent.num_workers` 配置，默认 4
- Worker 按 round-robin 分布到所有 Ray 节点（line 931-942）

**并发数学**：
```
effective_batch = train_batch_size × rollout.n
samples_per_worker = effective_batch / num_workers
```

### 2.3 Layer 1: AgentLoopWorker — 样本级 asyncio 并发

**文件**: [agent_loop.py:394-474](../../../verl/experimental/agent_loop/agent_loop.py#L394-L474)

```python
async def generate_sequences(self, batch: DataProto) -> DataProto:
    tasks = []
    for i in range(len(batch)):
        tasks.append(
            asyncio.create_task(
                self._run_agent_loop(sampling_params, trajectory_info[i], **kwargs)
            )
        )
    outputs = await asyncio.gather(*tasks)  # ← 同步屏障 B
```

**关键点**：
- 单个 Python event loop 运行**所有该 worker 分到的样本**
- `asyncio.gather()` 等待最慢的那个样本完成
- 样本间的 LLM 推理请求通过 `AsyncLLMServerManager` 分发到不同的 SGLang 推理实例
- 样本间共享同一个 event loop，当一个样本在等工具 HTTP 响应时，其他样本可以执行 LLM 推理

### 2.4 Layer 2: AsyncLLMServerManager — Sticky Session 负载均衡

**文件**: [agent_loop.py:56-121](../../../verl/experimental/agent_loop/agent_loop.py#L56-L121)

```python
class AsyncLLMServerManager:
    def __init__(self, config, server_handles):
        self.weighted_serveres = [(0, i, handle) for i, handle in enumerate(server_handles)]
        heapq.heapify(self.weighted_serveres)
        self.request_id_to_server = {}  # LRU cache, max 10000

    async def generate(self, request_id, ...):
        # Sticky session: 同一 request_id 始终路由到同一个 server（prefix caching）
        if request_id in self.request_id_to_server:
            server_index = self.request_id_to_server[request_id]
        else:
            _, server_index, _ = heapq.heappop(self.weighted_serveres)
            self.request_id_to_server[request_id] = server_index
```

**关键点**：
- **Sticky session**：同一个样本的多轮推理始终路由到同一个 SGLang 实例，利用 **prefix caching**
- **最少请求负载均衡**：新样本分配到当前请求最少的 server
- LRU cache 最大 10000 条目
- 每个 SGLang 实例 = 1 个 RolloutReplica = `TP_size` 个 GPU

### 2.5 Layer 3: ToolAgentLoop 状态机

**文件**: [tool_agent_loop.py](../../../verl/experimental/agent_loop/tool_agent_loop.py)

```
PENDING → GENERATING → PROCESSING_TOOLS → GENERATING → ... → TERMINATED
            ↑              │
            │              ├── 工具并发: asyncio.gather(*tools)
            │              │   最多 max_parallel_calls 个
            └──────────────┘
```

**每一轮（turn）的执行**：

| 阶段 | 操作 | I/O 类型 | 耗时估算 |
|------|------|----------|---------|
| GENERATING | HTTP POST 到 SGLang，模型生成 tokens | GPU 推理 | 2-10s |
| 解析 tool_calls | 正则/XML 解析生成文本 | CPU | <1ms |
| PROCESSING_TOOLS | HTTP POST 到 Tool Server | 外部 API I/O | 5-30s |
| 拼接 tool response | tokenize + 更新 response_mask | CPU | <10ms |

**终止条件**（任一触发即终止）：
1. `response_mask` 长度 ≥ `max_response_length`（61440 tokens）
2. `assistant_turns` ≥ `max_assistant_turns`（30 或 80）
3. 模型生成了不含 tool_calls 的最终回答
4. 模型生成了 EOS token

### 2.6 GPU 利用率时序图（单个样本，10 轮）

```
时间轴 →
GPU:  ████░░░░░░████░░░░████░░░░░░░░████░░████░░░░░░████░░████████
      gen1 wait  gen2 wait gen3  wait     gen4 w gen5  wait  gen6...
      (推理) (工具) (推理)(工具)(推理) (工具)  (推理)()(推理)(工具)(final)

█ = GPU busy (LLM 推理)
░ = GPU idle (等待外部工具 HTTP 响应)
```

**核心问题**：每一轮 PROCESSING_TOOLS 阶段 GPU 完全空闲。如果一个样本有 10 轮工具调用，每轮工具等待 10s，仅工具等待就 100s。在此期间该 GPU 上的推理引擎可以服务其他样本，但受限于 asyncio event loop 的调度效率和 SGLang 的并发请求处理能力。

---

## 3. 同步屏障与长尾效应

### 3.1 同步屏障链

```
AgentLoopManager
  ray.get()  ←─── 屏障 A: 等所有 worker
    │
    AgentLoopWorker[k]
      asyncio.gather()  ←─── 屏障 B: 等该 worker 内所有样本
        │
        _run_agent_loop(sample_i)
          asyncio.gather(*tools)  ←─── 屏障 C: 等该样本内所有工具
```

**step 完成时间 = max(所有 trajectory 完成时间) + post-rollout 阶段**

### 3.2 长尾效应公式

```
effective_batch = train_batch_size × rollout.n
T_step = max(T_trajectory_1, T_trajectory_2, ..., T_trajectory_effective_batch) + T_post_rollout

其中每个 trajectory:
T_trajectory = Σ_{turn=1}^{N_turns} (T_generate(turn) + T_tool_io(turn))
```

| 配置 | effective_batch | 最慢样本概率 | 预期效果 |
|------|---:|---:|---:|
| b4/n4 | 16 | P(长尾 in 16) = 低 | 稳定，498s/step |
| b8/n4 | 32 | P(长尾 in 32) = 中 | 勉强完成 |
| b16/n8 | 128 | P(长尾 in 128) = 高 | 几乎必然有极端长尾 |

**关键洞察**：batch 越大，包含极端长尾样本的概率越高。128 个 trajectory 中，几乎一定有某个样本进入"无限搜索循环"（反复调用 search 但找不到答案），它的完成时间决定了整个 step 的时间。

### 3.3 Post-rollout 阶段不是免费的

b4/n4 下 post-rollout 仅 ~70s（old_log_prob 19s + update_actor 37s + update_weights 15s）。但 effective_batch 增大后：
- old_log_prob 需要对更多 token 做前向推理
- update_actor 的 mini-batch 数量增加
- FSDP 同步开销增加

这就是为什么 b8/n4（32 个 trajectory）在 reward 全部完成后仍然没有打出 `step:1`——post-rollout 阶段本身也被放大了。

---

## 4. 工具服务器链路详解

### 4.1 调用链

```
CaRRToolAgentLoop._handle_processing_tools_state()
  │
  ▼
CaRRBrowserTool.execute()
  │  提取 agent_data.request_id → session_id
  │  提取 agent_data.tools_kwargs → search_forbidden_strs
  │
  ▼
CaRRSessionManager.ensure_started()  ← 首次工具调用时发送 start_session
  │
  ▼
CaRRSessionManager.call_server()
  │  aiohttp POST → http://localhost:7230
  │  timeout: 60s
  │  连接池: TCPConnector(limit=512, keepalive=30s)
  │
  ▼
CaRR Tool Server (Quart, 单进程异步)
  │
  ├── browser.search → Serper.dev HTTP API (30s × 3 retries)
  ├── browser.open → Jina Reader HTTP API (30s × 3 retries, Semaphore=128)
  └── browser.find → 本地字符串匹配（无外部 I/O）
```

### 4.2 会话生命周期

```
样本开始 (request_id = "req_xxx")
  │
  ├── 第1次工具调用 → ensure_started() → POST start_session → session2sandbox[req_xxx] = {}
  ├── 第2次工具调用 → session 已存在，直接 call_server
  ├── ...
  └── 样本结束 → agent loop finally → CaRRSessionManager.close() → POST close_session
```

**设计要点**：
- 三个浏览器工具（search/open/find）共享同一个 session_id
- Tool Server 内的 `session2sandbox` 字典保存每个 session 的状态（搜索结果 idx→url 映射、当前打开的网页内容）
- 这意味着 `browser.open(id=3)` 能正确打开第 3 个搜索结果的 URL，因为 session 保持了 search 的结果

### 4.3 Tool Server 内部实现

**文件**: [CaRR/tool_server/launch_server.py](../../../CaRR/tool_server/launch_server.py)

```python
# 单进程 Quart 异步服务器
app = Quart(__name__)

# 全局 session 状态
session2sandbox = defaultdict(dict)

@app.route("/", methods=["POST"])
async def handle_request():
    data = await request.json
    session_id = data["session_id"]
    name = data["name"]
    arguments = data["arguments"]
    remote_env_info = data.get("remote_env_info", {})

    sandbox = session2sandbox[session_id]

    if name == "browser.search":
        result = await search(query, num, sandbox, remote_env_info)
    elif name == "browser.open":
        result = await open_page(id, sandbox, remote_env_info)
    elif name == "browser.find":
        result = find(pattern, sandbox)  # 同步，纯本地
    elif name == "start_session":
        sandbox.clear()
    elif name == "close_session":
        session2sandbox.pop(session_id, None)
```

**文件**: [CaRR/tool_server/web_search.py](../../../CaRR/tool_server/web_search.py)

| 工具 | 外部 API | 超时 | 重试 | 并发限制 | 典型延迟 |
|------|----------|------|------|---------|---------|
| browser.search | Serper.dev POST | 30s | 3次 | 无（仅 HTTP 连接池 256） | 1-5s |
| browser.open | Jina Reader GET | 30s | 3次 | Semaphore=128 | 3-15s |
| browser.find | 无（本地） | - | - | - | <1ms |

### 4.4 工具服务器瓶颈分析

**问题 1: 单实例瓶颈**
- Quart 单进程，所有请求共享一个 Python event loop
- 当 128 个 trajectory 同时发起工具调用时，排队延迟显著

**问题 2: 无缓存**
- 相同 query 的搜索不缓存结果
- 同一个 URL 被多个 trajectory 重复 open，每次都调用 Jina API
- GRPO 中同一个 prompt 生成 n 个 trajectory，它们很可能搜索相似的 query

**问题 3: Jina 并发受限**
- Semaphore=128，超过后请求排队
- Jina API 本身可能有速率限制

**问题 4: 重试放大**
- 每个失败请求最多重试 3 次，30s 超时
- 最坏情况：一个 search 调用耗时 90s（3 × 30s）

---

## 5. 奖励服务器链路详解

### 5.1 调用链

```
NaiveRewardManager.run_single()
  │  合并 extra_info = parquet 静态字段 + agent loop extra_fields
  │
  ▼
carr_reward.compute_score()
  │  构建 payload: {history, label, task_unfinished, remote_env_info}
  │  aiohttp POST → http://localhost:8888/evaluate
  │  timeout: 650s (> 服务器内部 600s)
  │  连接池: TCPConnector(limit=512)
  │
  ▼
CaRR Reward Server (FastAPI, 单实例)
  │
  ├── task_unfinished=True → 短路返回 {reward: 0} （不调用 LLM）
  │
  └── task_unfinished=False:
      ├── Stage 1: get_outcome_reward() → 1次 DeepSeek API (是/否判断)
      ├── Stage 2: identify_entity()   → 1次 DeepSeek API (实体提取)
      └── Stage 3: judge_rubric()      → 1次 DeepSeek API (rubric 判断)
                                         + BFS 连通性检查
```

### 5.2 Reward Server 内部实现

**文件**: [CaRR/deepsearch_rm_with_rubrics/launch_server.py](../../../CaRR/deepsearch_rm_with_rubrics/launch_server.py)

```python
# FastAPI 单实例
app = FastAPI()

# 全局 LLM 客户端（单例）
reward_model = GPTModel(model_name, base_url, api_key)

@app.post("/evaluate")
async def evaluate(request: EvalRequest):
    if request.task_unfinished:
        return {"reward": 0, "outcome_reward": 0, "rubric_reward": 0}  # 短路

    # 三阶段串行评估
    result = await asyncio.wait_for(
        get_reward(history, label, rubrics, ...),
        timeout=600  # 内部 600s 超时
    )
    return result

async def get_reward(...):
    outcome = await get_outcome_reward(...)   # 1次 LLM 调用
    rubric = await get_rubric_reward(...)     # 2次 LLM 调用（identify + judge）
    return combine(outcome, rubric)
```

### 5.3 Reward 计算时序（单个样本）

```
时间轴 →
┌──────────────────────────────────────────────────────────────┐
│ outcome_reward    │ identify_entity     │ judge_rubric       │
│ DeepSeek API ×1   │ DeepSeek API ×1     │ DeepSeek API ×1    │
│ ~3-10s            │ ~3-10s              │ ~5-15s             │
└──────────────────────────────────────────────────────────────┘
                    总计: ~10-35s 每个样本
```

### 5.4 Reward 并发模型

当前 reward 的调用发生在两个可能的位置：

**路径 A: Streaming Reward（agent loop 内）**
- 如果配置了 `reward_loop_worker_handles`
- 在 rollout 阶段内就计算 reward，与 rollout 并行
- AgentLoopWorker 通过 `_compute_score()` 方法提交到 reward worker

**路径 B: Post-rollout Reward（训练循环内）**
- 当前 CaRR 使用 `reward.reward_model.enable: false` + `custom_reward_function`
- Reward 在 rollout 结束后、通过 `extract_reward()` 从 `non_tensor_batch` 中提取
- 实际的 HTTP 调用发生在 agent loop 内的 `_compute_score()`

**当前实际路径**：CaRR 走的是 streaming reward 路径（NaiveRewardManager 在 agent loop worker 中调用），每个 sample 完成 rollout 后立即计算 reward。这意味着 reward 计算与 rollout 是**交叉并行**的，而不是完全串行。

### 5.5 Reward Server 瓶颈分析

**问题 1: 串行 LLM 调用**
- 每个样本的 3 次 DeepSeek API 调用是串行的（outcome → identify → judge）
- 无法并行化（identify 依赖 outcome，judge 依赖 identify）

**问题 2: 单实例全局模型**
- 一个 FastAPI 进程 + 一个 AsyncOpenAI client
- 虽然 AsyncOpenAI 支持并发，但所有请求共享连接池
- DeepSeek API 本身可能有并发限制

**问题 3: 长尾 timeout**
- 内部 600s 全局超时，客户端 650s 超时
- 如果 DeepSeek API 响应慢，一个样本可能阻塞 600s

**问题 4: task_unfinished 短路**
- 好消息：未完成的样本直接返回 0，不调用 LLM
- 在 b4/n4 clean probe 中，17/32 个样本是 short_circuit

---

## 6. 自定义 Agent Loop (CaRRToolAgentLoop)

### 6.1 与 ToolAgentLoop 的关系

**文件**: [carr_agent_loop.py](../tools/carr_agent_loop.py)

```python
@register("carr_tool_agent")
class CaRRToolAgentLoop(ToolAgentLoop):
    """
    扩展 ToolAgentLoop，并行维护两套消息历史：
    1. agent_data.messages — verl 标准格式，用于 tokenization 和训练
    2. reward_history — CaRR 格式，用于发送给 reward server
    """
```

### 6.2 为什么需要自定义 Agent Loop

CaRR reward server 的 `/evaluate` 端点要求特定的 `history` 格式：

```python
# CaRR 格式（reward_history）
[
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "...", "tool_calls": [
        {"tool_call_id": "req_x_tc_0_0", "name": "browser.search", "arguments": "{...}"}
    ]},
    {"role": "tool", "content": [
        {"tool_call_id": "req_x_tc_0_0", "output": "search results..."}
    ]},
    {"role": "assistant", "content": "final answer"}
]

# verl 标准格式（agent_data.messages）
[
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "...", "tool_calls": [
        {"type": "function", "id": "...", "function": {"name": "...", "arguments": "..."}}
    ]},
    {"role": "tool", "content": "search results..."}  # 纯字符串，无 tool_call_id
]
```

两种格式的区别：
- tool_calls 结构不同（CaRR 用 `tool_call_id/name/arguments`，verl 用 OpenAI function calling 格式）
- tool 消息格式不同（CaRR 用 `[{tool_call_id, output}]` 列表，verl 用纯字符串）
- CaRR reward server 要求 `history[-1].role == "assistant"` 才执行评估

### 6.3 数据传播路径

```
CaRRToolAgentLoop.run()
  │  维护 reward_history（CaRR 格式）
  │  维护 agent_data.messages（verl 格式，由 base class 管理）
  │
  ▼ 输出
extra_fields = {
    "messages": reward_history,          → 传给 reward function
    "task_unfinished": bool,             → 传给 reward function
    "tool_call_counts": {...},           → 传给 metrics
    "termination_reason": "...",         → 传给 metrics
}
  │
  ▼ 进入 NaiveRewardManager
compute_score(extra_info={
    ...parquet 静态字段 (rubrics, search_forbidden_strs)...,
    ...agent loop extra_fields (messages, task_unfinished)...,
})
  │
  ▼ 返回
{"score": outcome_reward, "outcome_reward": ..., "rubric_reward": ...}
  │  三个 key 都写入 non_tensor_batch
  │
  ▼ 进入 compute_advantage()
cgrpo_advantage.compute_cgrpo_advantage(
    non_tensor_batch={"outcome_reward": [...], "rubric_reward": [...]}
)
```

---

## 7. C-GRPO Advantage 计算

### 7.1 与 verl 核心的集成

**文件**: [cgrpo_advantage.py](../reward/cgrpo_advantage.py)

```python
@register_adv_est("cgrpo")
def compute_cgrpo_advantage(token_level_rewards, response_mask, index, config, non_tensor_batch, ...):
    # 从 non_tensor_batch 提取 outcome 和 rubric reward
    outcome = non_tensor_batch["outcome_reward"]
    rubric = non_tensor_batch["rubric_reward"]

    # 组内归一化 rubric
    for group_id, indices in id2indices.items():
        max_r = rubric[indices].max()
        if max_r > 0:
            norm_rubric[indices] = rubric[indices] / max_r

    # C-GRPO 融合
    cgrpo_rewards = (1 - alpha) * outcome + alpha * outcome * norm_rubric

    # 放到最后一个有效 token
    new_rewards[i, valid_len - 1] = cgrpo_rewards[i]

    # 委托给标准 GRPO 做组内标准化
    return compute_grpo_outcome_advantage(token_level_rewards=new_rewards, ...)
```

### 7.2 non_tensor_batch 传递机制

**文件**: [ray_trainer.py:215-220](../../../verl/trainer/ppo/ray_trainer.py#L215-L220)

```python
# compute_advantage() 中通过 inspect 检测是否接受 non_tensor_batch
import inspect
_sig = inspect.signature(adv_estimator_fn)
if "non_tensor_batch" in _sig.parameters or any(
    p.kind == inspect.Parameter.VAR_KEYWORD for p in _sig.parameters.values()
):
    adv_kwargs["non_tensor_batch"] = data.non_tensor_batch
```

这确保只有声明了 `non_tensor_batch` 参数的 advantage estimator（如 cgrpo）才会收到该参数，不影响标准 GRPO/GAE。

---

## 8. 为什么 b16/n8 不能线性加速

### 8.1 核心公式

```
T_step = T_rollout + T_post_rollout

T_rollout = max_{i=1..N}(T_trajectory_i)     ← N = batch_size × rollout.n

T_trajectory_i = Σ_{turn=1}^{K_i} (T_gen(turn) + T_tool(turn))
                 其中 K_i ~ [2, 80]（实测 2-10 轮常见，偶尔 30+）

T_post_rollout = T_old_log_prob + T_advantage + T_update_actor + T_update_weights
                 ≈ 70s (b4/n4) → 随 N 增长
```

### 8.2 长尾概率分析

假设每个 trajectory 的完成时间服从某分布，p99 延迟 = 300s：

| N (effective_batch) | P(至少1个≥p99) | 预期最大延迟 |
|---:|---:|---:|
| 16 (b4/n4) | 15% | ~150s |
| 32 (b8/n4) | 27% | ~200s |
| 128 (b16/n8) | 72% | ~400s+ |

### 8.3 成本分析

```
b4/n4:   498s/step × 50 steps = 6.9 小时 × 8×H200
b16/n8:  如果能跑通，每步可能 30-60 分钟 × 50 步 = 25-50 小时 × 8×H200

但 b16/n8 每步处理 128 个 trajectory（b4/n4 只有 16 个）
所以 b16/n8 只需 50/8 ≈ 7 步就等价于 b4/n4 的 50 步数据量

关键问题：b16/n8 的 7 步总时间 vs b4/n4 的 50 步总时间？
→ b16/n8: 7 × 40min = 4.7h（乐观）到 7 × 60min = 7h（悲观）
→ b4/n4:  50 × 8.3min = 6.9h
→ 理论上差不多，但 b16/n8 的方差更大，且长尾概率高
```

### 8.4 真正的瓶颈不是 batch size

问题的本质是：
1. **工具 I/O 延迟不可压缩**：Serper + Jina 的 API 延迟是固定的
2. **无缓存导致重复调用**：同一 prompt 的 n 个 trajectory 搜索类似 query，但不共享结果
3. **单实例服务器串行化**：工具服务器和奖励服务器各只有一个实例
4. **同步屏障放大长尾**：最慢的一个 trajectory 决定整个 step 的时间

---

## 9. 关键文件索引

### 训练框架（verl 核心）

| 文件 | 角色 | 与性能相关的关键行 |
|------|------|-------------------|
| [ray_trainer.py](../../../verl/trainer/ppo/ray_trainer.py) | 训练主循环 | L1312-1316: rollout 阻塞; L1542: 权重同步 |
| [agent_loop.py](../../../verl/experimental/agent_loop/agent_loop.py) | AgentLoopManager/Worker | L470: asyncio.gather; L959: ray.get; L974-999: 性能指标 |
| [tool_agent_loop.py](../../../verl/experimental/agent_loop/tool_agent_loop.py) | 状态机基类 | L298: 工具并发; L183: LLM 推理调用 |
| [naive.py](../../../verl/workers/reward_manager/naive.py) | Reward Manager | L59: 逐样本计算 reward |

### CaRR 自定义层

| 文件 | 角色 | 关键设计 |
|------|------|---------|
| [carr_agent_loop.py](../tools/carr_agent_loop.py) | 自定义 AgentLoop | 维护 reward_history; finally 关闭 session |
| [carr_session_manager.py](../tools/carr_session_manager.py) | HTTP 会话管理 | TCPConnector(512); 惰性启动; 60s timeout |
| [carr_browser_tool.py](../tools/carr_browser_tool.py) | 浏览器工具适配器 | open.id 类型转换; search_forbidden_strs 提取 |
| [carr_reward.py](../reward/carr_reward.py) | 奖励函数 | TCPConnector(512); 650s timeout; 短路检测 |
| [cgrpo_advantage.py](../reward/cgrpo_advantage.py) | C-GRPO 优势估计器 | 组内归一化 rubric; 乘性融合 |

### CaRR 外部服务

| 文件 | 角色 | 并发模型 |
|------|------|---------|
| [CaRR/tool_server/launch_server.py](../../../CaRR/tool_server/launch_server.py) | 工具服务器 | Quart 单进程异步 |
| [CaRR/tool_server/web_search.py](../../../CaRR/tool_server/web_search.py) | 搜索/打开/查找 | HTTP pool 256; Jina Semaphore 128 |
| [CaRR/deepsearch_rm_with_rubrics/launch_server.py](../../../CaRR/deepsearch_rm_with_rubrics/launch_server.py) | 奖励服务器 | FastAPI 单实例; 3 次串行 LLM 调用 |

---

## 10. 瓶颈总结与优化方向预览

| 瓶颈 | 影响 | 优化方向 |
|------|------|---------|
| 工具服务器单实例 | 高并发下排队 | 多实例 + 负载均衡 |
| 无搜索/URL 缓存 | 重复 API 调用，浪费时间和费用 | query 级缓存 + single-flight |
| 奖励服务器串行 LLM | 每样本 3 次 DeepSeek 调用 | 并发评估 + batch 化 |
| 同步屏障（ray.get） | 最慢样本决定 step 时间 | 超时截断 + 异步训练 |
| 无效工具循环 | 长尾样本浪费 budget | 早期终止策略 |
| SFT 起点质量 | 影响 RL 策略收敛 | 使用 Qwen3-4B-Thinking |

> 本文档仅做架构解析。具体优化方案将在后续文档中讨论。
