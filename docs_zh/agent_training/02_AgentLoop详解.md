# Agent Loop 系统详解

本文档深入解析 verl 中 Agent Loop 系统的核心架构与实现细节。Agent Loop 是 verl 支持多轮对话、工具调用和环境交互式强化学习训练的关键组件，它负责编排 LLM 推理、工具执行和环境交互的完整流程。

---

## 1. AgentLoopBase 抽象基类

**源码**: `verl/experimental/agent_loop/agent_loop.py`

`AgentLoopBase` 是所有 Agent Loop 实现的抽象基类，定义了统一的接口规范。每个样本（sample）会创建一个独立的 `AgentLoopBase` 实例来执行其 agent loop 流程。

```python
class AgentLoopBase(ABC):
    def __init__(
        self,
        trainer_config: DictConfigWrap,
        server_manager: AsyncLLMServerManager,
        tokenizer: AutoTokenizer,
        processor: AutoProcessor,
        dataset_cls: type[RLHFDataset],
        dataset_config: DictConfigWrap,
        **kwargs,
    ):
        ...

    @abstractmethod
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        raise NotImplementedError
```

### 构造参数说明

| 参数 | 类型 | 说明 |
|------|------|------|
| `trainer_config` | `DictConfigWrap` | 包装后的 Hydra 训练配置，通过 `.config` 访问底层 `DictConfig` |
| `server_manager` | `AsyncLLMServerManager` | LLM server 管理器，负责负载均衡和推理请求路由 |
| `tokenizer` | `AutoTokenizer` | HuggingFace tokenizer，用于文本编解码 |
| `processor` | `AutoProcessor` | HuggingFace processor，用于多模态数据处理（纯文本模型时为 `None`） |
| `dataset_cls` | `type[RLHFDataset]` | 数据集类，用于处理视觉信息等数据预处理 |
| `dataset_config` | `DictConfigWrap` | 数据集配置 |

### run() 方法

`run()` 是核心入口方法，接收以下参数：

- **`sampling_params`**: LLM 采样参数字典，包含 `temperature`、`top_p`、`top_k`、`logprobs` 等
- **`**kwargs`**: 来自数据集 `RLHFDataset` 的字段，主要包括：
  - `raw_prompt`: 原始对话消息列表（`list[dict]`）
  - `tools_kwargs`: 工具调用所需的额外参数
  - `extra_info`: 附加信息字典（可包含 `interaction_kwargs` 等）
  - `agent_name`: Agent Loop 类型名称

### 辅助方法

`AgentLoopBase` 还提供了两个重要的辅助方法供子类使用：

- **`process_vision_info(messages)`**: 从消息中提取图片和视频等多模态数据
- **`apply_chat_template(messages, tools, images, videos, remove_system_prompt)`**: 将消息列表应用 chat template 并编码为 token IDs

---

## 2. AgentLoopOutput 数据结构

**源码**: `verl/experimental/agent_loop/agent_loop.py`

`AgentLoopOutput` 基于 Pydantic `BaseModel`，是所有 Agent Loop 的标准输出格式。它包含了训练所需的全部信息。

```python
class AgentLoopOutput(BaseModel):
    prompt_ids: list[int]                       # 初始 prompt 的 token IDs
    response_ids: list[int]                     # 所有响应 token（LLM 生成 + 工具响应）
    response_mask: list[int]                    # 1=LLM 生成 token, 0=工具/交互响应 token
    response_logprobs: Optional[list[float]]    # LLM 生成 token 的 log 概率
    routed_experts: Optional[Any]               # MoE 模型的专家路由信息
    multi_modal_data: Optional[dict[str, Any]]  # 图片/视频等多模态数据
    reward_score: Optional[float]               # 可选的奖励分数
    num_turns: int                              # 总对话轮次数（user + assistant + 1）
    metrics: AgentLoopMetrics                   # 性能指标（生成耗时、工具调用耗时等）
    extra_fields: dict[str, Any]                # 动态扩展字段（turn_scores, tool_rewards 等）
```

### response_mask 机制详解

`response_mask` 是 Agent Loop 最核心的设计之一。它标识了 `response_ids` 中每个 token 的来源，从而控制哪些 token 参与策略梯度的损失计算：

```
response_ids:  [LLM 生成 tokens] [工具响应 tokens] [LLM 生成 tokens] [交互响应 tokens] [LLM 生成 tokens]
response_mask: [1, 1, ..., 1   ] [0, 0, ..., 0   ] [1, 1, ..., 1   ] [0, 0, ..., 0    ] [1, 1, ..., 1   ]
                ↑ 纳入损失计算     ↑ 排除损失计算     ↑ 纳入损失计算     ↑ 排除损失计算      ↑ 纳入损失计算
```

**设计原理**：

- **mask=1 的 token**：由 LLM 生成，对应模型的策略分布，需要通过 policy gradient 进行优化
- **mask=0 的 token**：来自外部环境（工具返回值、交互响应），不属于模型策略分布的一部分，因此不参与损失计算

这种设计使得多轮对话中 LLM 生成的所有 token 都能被正确地纳入梯度更新，而环境反馈的 token 仅作为上下文信息存在于序列中。

### extra_fields 常见字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `turn_scores` | `list[float]` | 每轮交互的奖励分数（来自 Interaction） |
| `tool_rewards` | `list[float]` | 每次工具调用的奖励分数 |
| `raw_prompt` | `list[dict]` | 原始 prompt 消息（后处理时添加） |
| `is_cancel` | `Any` | 是否被取消的标记 |
| `reward_extra_info` | `dict` | 异步 reward 计算的额外信息 |

---

## 3. ToolAgentLoop 状态机详解

**源码**: `verl/experimental/agent_loop/tool_agent_loop.py`

`ToolAgentLoop` 通过 `@register("tool_agent")` 注册，是 verl 中最重要的多轮 Agent Loop 实现。它采用有限状态机（Finite State Machine）模式来管理复杂的多轮对话流程。

### 状态转移图

```
                          ┌─────────────────────────────────────────────────┐
                          │                                                 │
                          ↓                                                 │
PENDING ──→ GENERATING ──→ PROCESSING_TOOLS ──→ GENERATING ──→ ...         │
                │                                     │                     │
                │                                     │                     │
                ├──→ INTERACTING ──→ GENERATING ──────┤                     │
                │         │                           │                     │
                │         └──→ TERMINATED             │                     │
                │                                     │                     │
                └──→ TERMINATED ←─────────────────────┘                     │
                          ↑                                                 │
                          └─────────────────────────────────────────────────┘
```

### 状态详解

#### PENDING 状态

**职责**：初始化阶段，准备 prompt token 序列。

```python
async def _handle_pending_state(self, agent_data, sampling_params) -> AgentState:
    prompt_ids = await self.apply_chat_template(
        agent_data.messages,
        tools=self.tool_schemas,
        images=agent_data.image_data,
        videos=agent_data.video_data,
    )
    agent_data.prompt_ids = prompt_ids
    return AgentState.GENERATING
```

**执行逻辑**：
1. 将原始消息（`messages`）与工具 schema（`tool_schemas`）一起通过 chat template 编码
2. 如果有多模态数据（图片/视频），一并传入 processor 处理
3. 将编码后的 `prompt_ids` 存入 `AgentData`
4. 无条件转移到 `GENERATING` 状态

#### GENERATING 状态

**职责**：调用 LLM server 生成回复，并根据生成结果决定下一步。

**执行逻辑**：
1. 调用 `server_manager.generate()` 发送推理请求，传入当前累积的 `prompt_ids`
2. 将生成的 `response_ids` 追加到 `prompt_ids`（用于下一轮的 prefix caching）
3. 更新 `response_mask`，将所有 LLM 生成的 token 标记为 **1**
4. 记录 `logprobs` 和 `routed_experts`（如有）
5. 递增 `assistant_turns` 计数器
6. **检查终止条件**（满足任一则转移到 `TERMINATED`）：
   - `len(response_mask) >= response_length`：总响应长度达到上限
   - `assistant_turns >= max_assistant_turns`：助手回复轮次达到上限
   - `user_turns >= max_user_turns`：用户/环境轮次达到上限
7. 调用 `tool_parser.extract_tool_calls()` 从生成的 token 中提取工具调用
8. **决定下一状态**：
   - 存在 `tool_calls` --> `PROCESSING_TOOLS`
   - 配置了 interaction 且无工具调用 --> `INTERACTING`
   - 否则 --> `TERMINATED`

#### PROCESSING_TOOLS 状态

**职责**：并行执行工具调用，处理工具返回结果。

**执行逻辑**：
1. 从待执行的 `tool_calls` 中取出最多 `max_parallel_calls` 个，通过 `asyncio.gather()` 并行执行
2. 对每个工具调用：
   - 解析工具名和 JSON 参数
   - 创建工具实例（`tool.create()`）
   - 执行工具（`tool.execute()`），获取 `ToolResponse`、`tool_reward` 和附加信息
   - 执行完毕后释放工具实例（`tool.release()`）
   - 异常时返回错误信息作为工具响应
3. 处理工具响应内容：
   - **文本响应**：如果超过 `max_tool_response_length`，按配置的截断策略处理
   - **多模态响应**：提取图片数据追加到 `agent_data.image_data`
4. **截断策略**（`tool_response_truncate_side`）：
   - `left`：保留前 N 个字符，尾部追加 `"...(truncated)"`
   - `right`：保留后 N 个字符，头部追加 `"(truncated)..."`
   - `middle`：保留首尾各 N/2 个字符，中间插入 `"...(truncated)..."`
5. 将工具响应编码为 token，追加到 `prompt_ids`
6. 更新 `response_mask`，将工具响应 token 全部标记为 **0**
7. 如果有 `response_logprobs`，为工具响应 token 填充 `0.0`
8. 收集 `tool_rewards`，递增 `user_turns`
9. 检查长度限制：如果 `len(response_mask) + len(response_ids) >= response_length`，转移到 `TERMINATED`
10. 否则返回 `GENERATING` 继续生成

#### INTERACTING 状态

**职责**：与外部环境交互，获取环境反馈。

**执行逻辑**：
1. 调用 `interaction.generate_response()` 获取环境反馈
   - 返回值包括：`should_terminate_sequence`（是否终止）、`interaction_responses`（反馈文本）、`reward`（奖励分数）、`metrics`（指标）
2. 递增 `user_turns` 计数器
3. 将环境反馈构造为 `{"role": "user", "content": interaction_responses}` 消息
4. 通过 `apply_chat_template()` 编码反馈消息
5. 更新 `prompt_ids` 和 `response_mask`（交互响应 token 标记为 **0**）
6. 如果有 `reward`，追加到 `agent_data.turn_scores`
7. **决定下一状态**：
   - `should_terminate_sequence == True` --> `TERMINATED`
   - 否则 --> `GENERATING`

#### TERMINATED 状态

**职责**：组装最终的 `AgentLoopOutput` 并返回。

在状态机循环退出后，`run()` 方法执行最终的输出组装：

```python
# 从累积的 prompt_ids 中分离出初始 prompt 和响应部分
response_ids = agent_data.prompt_ids[-len(agent_data.response_mask):]
prompt_ids = agent_data.prompt_ids[:len(agent_data.prompt_ids) - len(agent_data.response_mask)]

output = AgentLoopOutput(
    prompt_ids=prompt_ids,
    response_ids=response_ids[:self.response_length],       # 截断到最大响应长度
    response_mask=agent_data.response_mask[:self.response_length],
    multi_modal_data=multi_modal_data,
    response_logprobs=agent_data.response_logprobs[:self.response_length],
    num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
    metrics=agent_data.metrics,
    routed_experts=agent_data.routed_experts,
    extra_fields={
        "turn_scores": agent_data.turn_scores,
        "tool_rewards": agent_data.tool_rewards,
    },
)
```

### 完整的多轮示例

以下是一个两轮工具调用的完整 token 序列示意：

```
第 1 轮:
  PENDING:          prompt_ids = [system + user message tokens]
  GENERATING:       prompt_ids = [..., assistant response tokens(含 tool_call)]
                    response_mask = [1, 1, ..., 1]
  PROCESSING_TOOLS: prompt_ids = [..., tool response tokens]
                    response_mask = [1, 1, ..., 1, 0, 0, ..., 0]

第 2 轮:
  GENERATING:       prompt_ids = [..., assistant final response tokens]
                    response_mask = [1, 1, ..., 1, 0, 0, ..., 0, 1, 1, ..., 1]
  TERMINATED:       组装输出
```

---

## 4. AgentData 状态封装

**源码**: `verl/experimental/agent_loop/tool_agent_loop.py`

`AgentData` 封装了单个样本在整个 Agent Loop 执行过程中的所有状态。它作为参数在各个状态处理函数之间传递，也会被传递给工具调用，以便工具可以访问完整的历史上下文。

```python
class AgentData:
    # 对话状态
    messages: list[dict]                        # 完整消息历史（包含 system/user/assistant/tool 消息）
    image_data: list[Image.Image]               # 累积的图片数据
    video_data: list[tuple[Tensor, dict]]       # 累积的视频数据

    # Token 序列
    prompt_ids: list[int]                       # 累积的完整 token 序列（prompt + 所有响应）
    response_ids: list[int]                     # 当前轮 LLM 生成的 token（临时变量）
    response_mask: list[int]                    # 响应掩码序列
    response_logprobs: list[float]              # 响应 token 的 log 概率

    # 奖励信号
    turn_scores: list[float]                    # 每轮交互的奖励分数
    tool_rewards: list[float]                   # 每次工具调用的奖励分数

    # 轮次计数
    user_turns: int                             # 用户/环境/工具轮次计数
    assistant_turns: int                        # 助手回复轮次计数

    # 工具调用
    tool_calls: list[FunctionCall]              # 当前待执行的工具调用列表

    # MoE 路由
    routed_experts: Optional[Any]               # MoE 模型的专家路由信息

    # 扩展
    extra_fields: dict[str, Any]                # 自定义扩展字段（如工具会话数据）
    request_id: str                             # 请求唯一标识（用于 sticky session）
    tools_kwargs: dict[str, Any]                # 工具调用的额外参数
    interaction: Optional[BaseInteraction]      # 交互环境实例
    interaction_kwargs: dict[str, Any]          # 交互环境参数
    metrics: dict[str, Any]                     # 性能指标字典
```

### prompt_ids 的累积机制

`AgentData.prompt_ids` 在整个 Agent Loop 过程中不断累积，这是为了支持 prefix caching：

```
初始:           [system_tokens, user_tokens]
GENERATING 后:  [system_tokens, user_tokens, assistant_tokens]
TOOL 响应后:    [system_tokens, user_tokens, assistant_tokens, tool_tokens]
再次 GENERATING: [system_tokens, user_tokens, assistant_tokens, tool_tokens, assistant_tokens_2]
```

最终输出时，通过 `response_mask` 的长度从 `prompt_ids` 尾部切分出 `response_ids`，剩余部分作为 `prompt_ids`。

---

## 5. SingleTurnAgentLoop

**源码**: `verl/experimental/agent_loop/single_turn_agent_loop.py`

`SingleTurnAgentLoop` 通过 `@register("single_turn_agent")` 注册，是最简单的 Agent Loop 实现，适用于标准的单轮问答场景。

### 特点

- **无状态机**：直接执行 "编码 --> 生成 --> 返回" 三步流程
- **无工具调用**：不涉及工具执行和多轮交互
- **默认 agent loop 类型**：当数据集中未指定 `agent_name` 时，系统使用 `config.actor_rollout_ref.rollout.agent.default_agent_loop` 指定的 agent loop，通常为 `single_turn_agent`
- **固定 num_turns=2**：一轮用户消息 + 一轮助手回复

### 执行流程

```python
@register("single_turn_agent")
class SingleTurnAgentLoop(AgentLoopBase):
    async def run(self, sampling_params, **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])

        # 1. 提取多模态数据
        multi_modal_data = await self.process_vision_info(messages)

        # 2. 应用 chat template 编码为 token
        prompt_ids = await self.apply_chat_template(messages, tools=self.tool_schemas, ...)

        # 3. 调用 LLM server 生成
        output = await self.server_manager.generate(
            request_id=uuid4().hex,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            ...
        )

        # 4. 构造输出，response_mask 全为 1
        response_mask = [1] * len(output.token_ids)
        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=output.token_ids[:self.response_length],
            response_mask=response_mask[:self.response_length],
            ...
            extra_fields={"turn_scores": [], "tool_rewards": []},  # 保持与 ToolAgentLoop 一致的 schema
        )
```

注意：`SingleTurnAgentLoop` 也支持加载工具 schema（通过 `data.tool_config_path`），工具 schema 会被传入 chat template，但实际不会执行工具调用。这使得模型可以在单轮场景下学习何时以及如何生成工具调用格式。

---

## 6. AsyncLLMServerManager 负载均衡

**源码**: `verl/experimental/agent_loop/agent_loop.py`

`AsyncLLMServerManager` 管理多个 OpenAI 兼容的 LLM server 实例（由 vLLM、SGLang 等 backend 提供），实现请求的智能路由。

### 核心设计

```python
class AsyncLLMServerManager:
    def __init__(self, config, server_handles, max_cache_size=10000):
        self.server_handles = server_handles           # Ray actor handles 列表
        self.weighted_serveres = [[0, idx, server] ...] # 最小堆，按请求数排序
        self.request_id_to_server = LRUCache(maxsize=max_cache_size)  # LRU 缓存
```

### 负载均衡策略

#### Least-Request（最少请求）策略

首轮请求使用最小堆选择当前负载最低的 server：

```python
def _choose_server(self, request_id):
    if request_id in self.request_id_to_server:
        return self.request_id_to_server[request_id]  # 粘性会话命中

    # 从最小堆顶部取出负载最低的 server
    _, _, server = self.weighted_serveres[0]
    self.weighted_serveres[0][0] += 1   # 增加请求计数
    heapq.heapreplace(self.weighted_serveres, self.weighted_serveres[0])  # 重新堆化
    self.request_id_to_server[request_id] = server  # 建立绑定
    return server
```

#### Sticky Session（粘性会话）

多轮对话的后续请求会路由到同一个 server。这是一个关键优化：

- **原理**：同一样本的多轮请求发送到同一 server，使 server 可以复用前几轮的 KV cache（prefix caching）
- **实现**：通过 LRU cache 维护 `request_id` 到 `server` 的映射
- **缓存大小**：默认 `max_cache_size=10000`，使用 LRU 策略淘汰最久未使用的映射

### generate() 方法

```python
async def generate(self, request_id, *, prompt_ids, sampling_params,
                   image_data=None, video_data=None) -> TokenOutput:
    server = self._choose_server(request_id)          # 路由到合适的 server
    output = await server.generate.remote(
        request_id=uuid4().hex,    # 每轮使用新的 request_id（server 层面）
        prompt_ids=prompt_ids,
        sampling_params=sampling_params,
        image_data=image_data,
        video_data=video_data,
    )
    return output
```

注意区分两层 `request_id`：
- **Agent Loop 层**：同一样本的所有轮次使用同一个 `request_id`，用于 sticky session 路由
- **Server 层**：每轮推理使用新的 `uuid4().hex`，因为 server 不需要跟踪跨轮次的状态

---

## 7. AgentLoopWorker 与 AgentLoopManager

### AgentLoopWorker

**源码**: `verl/experimental/agent_loop/agent_loop.py`

`AgentLoopWorker` 是 Ray actor，负责执行一批样本的 agent loop。它是实际创建和运行 `AgentLoopBase` 实例的工作单元。

#### 核心职责

1. **初始化**：加载 tokenizer/processor，初始化 `AsyncLLMServerManager`，配置 agent loop registry
2. **并发执行**：为每个样本创建独立的 `AgentLoopBase` 实例，通过 `asyncio.gather()` 并发运行
3. **后处理**：对各个 agent loop 的输出进行 padding 和批处理

#### generate_sequences() 流程

```python
async def generate_sequences(self, batch: DataProto) -> DataProto:
    # 1. 构建 sampling_params
    sampling_params = dict(temperature=..., top_p=..., top_k=..., logprobs=...)

    # 2. 设置默认 agent_name（如数据集未指定）
    if "agent_name" not in batch.non_tensor_batch:
        batch.non_tensor_batch["agent_name"] = np.array(
            [config.agent.default_agent_loop] * len(batch))

    # 3. 为每个样本创建异步任务
    tasks = []
    for i in range(len(batch)):
        kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items()}
        tasks.append(asyncio.create_task(
            self._run_agent_loop(sampling_params, ..., **kwargs)
        ))

    # 4. 并发执行所有 agent loop
    outputs = await asyncio.gather(*tasks)

    # 5. 后处理：padding、组装 DataProto
    output = self._postprocess(outputs)
    return output
```

#### 后处理详解

`_agent_loop_postprocess()` 将每个 `AgentLoopOutput` 转换为 `_InternalAgentLoopOutput`：

- **prompt_ids**: 左侧 padding 到 `prompt_length`（例如 `[0,0,0,0,1,2,3,4]`）
- **response_ids**: 右侧 padding 到 `response_length`（例如 `[5,6,7,8,0,0,0,0]`）
- **input_ids**: `prompt_ids` + `response_ids` 拼接
- **attention_mask**: padding 位置为 0，其余为 1
- **response_mask**: LLM 生成 token 为 1，工具/交互/padding token 为 0
- **position_ids**: 根据 `attention_mask` 计算递增位置编码（多模态场景使用 3D rope index）

### AgentLoopManager

**源码**: `verl/experimental/agent_loop/agent_loop.py`

`AgentLoopManager` 是顶层编排器，负责协调 LLM server 和 `AgentLoopWorker` 集群。

#### 初始化流程

```
AgentLoopManager.__init__()
    │
    ├── _initialize_llm_servers(rollout_resource_pool)
    │       ├── 计算 replica 数量 = world_size / rollout_world_size
    │       ├── 为每个 replica 创建 RolloutReplica 实例
    │       ├── 初始化 server（hybrid 模式或 standalone 模式）
    │       └── 收集所有 server handles 和 addresses
    │
    └── _init_agent_loop_workers()
            ├── 创建 num_workers 个 AgentLoopWorker（Ray actor）
            ├── 使用 round-robin 策略分布到各 node
            └── 传入 server_handles 和 reward_loop_worker_handles
```

#### generate_sequences() 编排

```python
def generate_sequences(self, prompts: DataProto) -> DataProto:
    # 1. 将 batch 按 worker 数量切分
    chunks = prompts.chunk(len(self.agent_loop_workers))

    # 2. 分发到各 worker 并行执行
    outputs = ray.get([
        worker.generate_sequences.remote(chunk)
        for worker, chunk in zip(self.agent_loop_workers, chunks)
    ])

    # 3. 聚合结果
    output = DataProto.concat(outputs)

    # 4. 收集性能指标
    metrics = [output.meta_info.pop("metrics") for output in outputs]
    timing = self._performance_metrics(metrics, output)

    output.meta_info = {"timing": timing, **outputs[0].meta_info}
    return output
```

#### 性能指标

`AgentLoopManager` 收集以下关键性能指标：

| 指标 | 说明 |
|------|------|
| `agent_loop/generate_sequences/{min,max,mean}` | 各样本 LLM 推理总耗时 |
| `agent_loop/tool_calls/{min,max,mean}` | 各样本工具调用总耗时 |
| `agent_loop/num_preempted/{min,max,mean}` | 各样本被 preempt 的次数 |
| `agent_loop/slowest/*` | 最慢样本的详细信息（瓶颈分析） |

由于 batch 的整体执行时间由最慢的样本决定（木桶效应），`slowest` 指标对于诊断性能瓶颈非常重要。

---

## 8. 自定义 AgentLoop 注册

verl 提供了灵活的 registry 机制，允许用户实现自定义的 Agent Loop。

### 注册方式

```python
from verl.experimental.agent_loop.agent_loop import register, AgentLoopBase, AgentLoopOutput

@register("my_custom_agent")
class MyAgentLoop(AgentLoopBase):
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        # 获取输入消息
        messages = list(kwargs["raw_prompt"])
        tools_kwargs = kwargs.get("tools_kwargs", {})

        # 应用 chat template
        prompt_ids = await self.apply_chat_template(messages)

        # 自定义多轮逻辑
        all_response_ids = []
        all_response_mask = []

        for turn in range(max_turns):
            # 调用 LLM server 生成
            output = await self.server_manager.generate(
                request_id="my_request",
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
            )
            all_response_ids.extend(output.token_ids)
            all_response_mask.extend([1] * len(output.token_ids))

            # 自定义逻辑：调用工具、与环境交互等
            # ...

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=all_response_ids,
            response_mask=all_response_mask,
            multi_modal_data={},
            num_turns=max_turns * 2,
            metrics=AgentLoopMetrics(),
        )
```

### 使用方式

在数据集中为每个样本设置 `agent_name` 字段即可指定使用哪个 Agent Loop：

```python
# 在数据预处理阶段，为 parquet 文件中的每行添加 agent_name 列
dataset["agent_name"] = "my_custom_agent"
```

或者通过配置设置默认的 agent loop：

```yaml
actor_rollout_ref:
  rollout:
    agent:
      default_agent_loop: my_custom_agent
```

### Registry 内部机制

Registry 基于 Hydra `instantiate` 实现动态类加载：

```python
_agent_loop_registry: dict[str, dict] = {}

def register(agent_name: str):
    def decorator(subclass: type[AgentLoopBase]) -> type[AgentLoopBase]:
        fqdn = f"{subclass.__module__}.{subclass.__qualname__}"
        _agent_loop_registry[agent_name] = {"_target_": fqdn}
        return subclass
    return decorator
```

此外，还支持通过 YAML 配置文件注册自定义 Agent Loop（通过 `agent_loop_config_path` 配置项），适用于需要额外初始化参数的场景。

---

## 9. ToolParser 工具调用解析

**源码**: `verl/experimental/agent_loop/tool_parser.py`

`ToolParser` 负责从 LLM 生成的 token 序列中解析出工具调用信息。不同的模型使用不同的工具调用格式，因此 verl 通过 registry 模式支持多种解析器。

### 基类定义

```python
class ToolParser(ABC):
    _registry: dict[str, type["ToolParser"]] = {}

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    @abstractmethod
    async def extract_tool_calls(self, responses_ids: list[int]) -> tuple[str, list[FunctionCall]]:
        """从 response token IDs 中提取工具调用。
        返回: (剩余文本内容, 工具调用列表)
        """
        raise NotImplementedError

    @classmethod
    def get_tool_parser(cls, name: str, tokenizer):
        """根据名称获取已注册的解析器实例。"""
        return cls._registry[name](tokenizer)

    @classmethod
    def register(cls, name: str):
        """注册解析器的装饰器。"""
        def decorator(subclass):
            cls._registry[name] = subclass
            return subclass
        return decorator
```

### FunctionCall 数据结构

```python
class FunctionCall(BaseModel):
    name: str            # 要调用的函数名称
    arguments: str       # JSON 格式的函数参数（注意：模型可能生成无效 JSON）
```

### 内置解析器

#### HermesToolParser

**注册名**: `hermes`

解析 Hermes 格式的工具调用标签，广泛用于 Qwen、Mistral 等开源模型。

```
输入格式:
<tool_call>{"name": "search", "arguments": {"query": "verl framework"}}</tool_call>
```

解析逻辑：
1. 将 `response_ids` 解码为文本
2. 检查是否包含 `<tool_call>` 和 `</tool_call>` 标签
3. 使用正则表达式 `<tool_call>(.*?)</tool_call>` 提取所有匹配
4. 解析 JSON，提取 `name` 和 `arguments` 字段
5. 返回去除工具调用标签后的剩余文本和 `FunctionCall` 列表

#### GptOssToolParser

**注册名**: `gpt-oss`

解析 OpenAI Harmony 格式的工具调用，适用于 GPT 系列兼容模型。

```
输入格式:
<|start|>assistant<|channel|>analysis<|message|>思考过程...<|end|>
<|start|>assistant<|channel|> to=functions.search <|constrain|>json<|message|>{"query": "verl"}<|call|>
```

解析逻辑：
1. 将 `response_ids` 解码为文本（**保留** special tokens）
2. 去除 padding token
3. 去除 Chain-of-Thought（CoT）部分（`<|start|>assistant<|channel|>analysis<|message|>...<|end|>`），因为 CoT 中可能包含类似工具调用的文本但非真实调用
4. 使用正则表达式匹配 `to=functions.{name}` 和 `<|message|>{arguments}<|call|>` 模式
5. 返回去除工具调用后的文本和 `FunctionCall` 列表

### 自定义解析器注册

```python
@ToolParser.register("my_format")
class MyToolParser(ToolParser):
    async def extract_tool_calls(self, responses_ids: list[int]) -> tuple[str, list[FunctionCall]]:
        text = self.tokenizer.decode(responses_ids)
        # 自定义解析逻辑
        function_calls = [...]
        content = "..."
        return content, function_calls
```

### 配置方式

通过训练配置文件选择使用的解析器：

```yaml
actor_rollout_ref:
  rollout:
    multi_turn:
      format: hermes          # 或 gpt-oss，或自定义注册名
      tool_config_path: ...   # 工具配置文件路径
      max_user_turns: 5
      max_assistant_turns: 5
      max_parallel_calls: 4
      max_tool_response_length: 4096
      tool_response_truncate_side: middle  # left / right / middle
```
