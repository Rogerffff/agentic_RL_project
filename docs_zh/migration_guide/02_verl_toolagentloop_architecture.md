# verl 原生 ToolAgentLoop 系统架构详解

本文档详细记录 verl 框架中 ToolAgentLoop 的完整架构、核心文件职责、数据流、以及关键实现细节。这是 verl 原生的多轮 Agent 训练基础设施，基于异步状态机设计。

---

## 1. 系统总览

verl 的 ToolAgentLoop 采用 **Server-based 异步架构**：
- LLM 推理由独立的 vLLM/SGLang Server 提供（支持多 Server 负载均衡）
- 每个样本独立运行一个异步状态机
- 通过 Ray Actor 实现分布式并行
- 使用 **Delta-based Tokenization** 处理多轮 chat template 的 token 拼接

### 与 Agent-R1 的核心区别

| 方面 | Agent-R1 | verl ToolAgentLoop |
|------|----------|-------------------|
| 执行模型 | 同步 for 循环，batch 统一处理 | 异步状态机，每个样本独立 |
| 推理后端 | actor_rollout_wg（Ray WorkerGroup 同步调用） | AsyncLLMServerManager（异步 + 多 Server 负载均衡） |
| 工具解析 | BaseToolEnv 统一处理 extract/execute/format | ToolParser（独立解析）+ 状态处理器（执行） |
| 工具接口 | BaseTool.execute(args) → dict | BaseTool.execute(id, params) → (ToolResponse, reward, metrics) |
| 工具响应格式 | 手动拼接特殊 token | role="tool" message → apply_chat_template |

---

## 2. 核心文件及其职责

### 2.1 ToolAgentLoop 状态机 — `verl/experimental/agent_loop/tool_agent_loop.py`

这是核心状态机实现，479 行代码。

#### AgentState 枚举

```python
class AgentState(Enum):
    PENDING = "pending"            # 初始状态，等待构建 prompt
    GENERATING = "generating"      # LLM 正在生成
    PROCESSING_TOOLS = "processing_tools"  # 执行工具调用
    TERMINATED = "terminated"      # 终止状态
    INTERACTING = "interacting"    # 与外部环境交互（非工具场景）
```

状态转移图：
```
PENDING → GENERATING → PROCESSING_TOOLS → GENERATING → ... → TERMINATED
                   ↘                                    ↗
                    INTERACTING → GENERATING → ...
```

#### AgentData — 单个轨迹的完整状态

```python
class AgentData:
    def __init__(self, messages, image_data, video_data, metrics,
                 request_id, tools_kwargs, interaction, interaction_kwargs):
        self.messages = messages          # chat messages 列表
        self.image_data = image_data      # 图像数据
        self.video_data = video_data      # 视频数据
        self.metrics = metrics            # 性能指标
        self.request_id = request_id      # 唯一请求 ID（用于 sticky session）
        self.tools_kwargs = tools_kwargs  # 工具初始化参数

        # 状态变量（核心！）
        self.prompt_ids: list[int] = []          # 累积的所有 token IDs
        self.response_ids: list[int] = []        # 当前轮 LLM 生成的 token IDs
        self.response_mask: list[int] = []       # 1=LLM 生成, 0=工具/交互
        self.response_logprobs: list[float] = [] # log 概率（可选）
        self.turn_scores: list[float] = []       # 每轮得分（交互场景）
        self.tool_rewards: list[float] = []      # 工具级别奖励
        self.user_turns = 0                      # user/tool 轮数
        self.assistant_turns = 0                 # assistant 轮数

        self.tool_calls: list[FunctionCall] = [] # 当前轮提取的工具调用
        self.extra_fields: dict[str, Any] = {}   # 扩展字段
```

**关键理解**：
- `prompt_ids` 是**累积**的——每轮生成后，新的 response_ids 和 tool_response_ids 都会追加到 prompt_ids
- `response_mask` 也是累积的——与 Agent-R1 的 action_mask 语义完全等价
- `response_ids` 是当前轮的临时变量，不是累积的

#### ToolAgentLoop 类

```python
@register("tool_agent")  # 注册为 "tool_agent"，数据集中 agent_name 字段指定
class ToolAgentLoop(AgentLoopBase):
    def __init__(self, trainer_config, server_manager, tokenizer, processor, **kwargs):
        # 从配置读取参数
        self.max_user_turns = config...multi_turn.max_user_turns
        self.max_assistant_turns = config...multi_turn.max_assistant_turns
        self.max_parallel_calls = config...multi_turn.max_parallel_calls
        self.max_tool_response_length = config...multi_turn.max_tool_response_length
        self.tool_response_truncate_side = config...multi_turn.tool_response_truncate_side

        # 从 YAML 配置初始化工具
        tool_config_path = config...multi_turn.tool_config_path
        tool_list = initialize_tools_from_config(tool_config_path)
        self.tools = {tool.name: tool for tool in tool_list}
        self.tool_schemas = [tool.tool_schema.model_dump(...) for tool in tool_list]

        # 初始化工具解析器
        self.tool_parser = ToolParser.get_tool_parser(
            config...multi_turn.format, self.tokenizer
        )  # 例如 "hermes" → HermesToolParser

        self.prompt_length = config...rollout.prompt_length
        self.response_length = config...rollout.response_length
```

#### run() — 状态机主循环

```python
async def run(self, sampling_params, **kwargs) -> AgentLoopOutput:
    messages = list(kwargs["raw_prompt"])
    # 提取多模态数据
    multi_modal_data = await self.process_vision_info(messages)

    # 创建 AgentData
    agent_data = AgentData(
        messages=messages,
        image_data=images, video_data=videos,
        metrics={}, request_id=uuid4().hex,
        tools_kwargs=kwargs.get("tools_kwargs", {}),
    )

    # 状态机循环
    state = AgentState.PENDING
    while state != AgentState.TERMINATED:
        if state == AgentState.PENDING:
            state = await self._handle_pending_state(agent_data, sampling_params)
        elif state == AgentState.GENERATING:
            state = await self._handle_generating_state(agent_data, sampling_params)
        elif state == AgentState.PROCESSING_TOOLS:
            state = await self._handle_processing_tools_state(agent_data)
        elif state == AgentState.INTERACTING:
            state = await self._handle_interacting_state(agent_data)

    # 最终输出
    response_ids = agent_data.prompt_ids[-len(agent_data.response_mask):]
    prompt_ids = agent_data.prompt_ids[:len(agent_data.prompt_ids) - len(agent_data.response_mask)]

    return AgentLoopOutput(
        prompt_ids=prompt_ids,
        response_ids=response_ids[:self.response_length],
        response_mask=agent_data.response_mask[:self.response_length],
        response_logprobs=...,
        num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
        metrics=agent_data.metrics,
        extra_fields={"turn_scores": ..., "tool_rewards": ...},
    )
```

#### _handle_pending_state — 初始化 Prompt

```python
async def _handle_pending_state(self, agent_data, sampling_params):
    prompt_ids = await self.apply_chat_template(
        agent_data.messages,
        tools=self.tool_schemas,    # ← 自动注入工具描述
        images=agent_data.image_data,
        videos=agent_data.video_data,
    )
    agent_data.prompt_ids = prompt_ids
    return AgentState.GENERATING
```

**关键点**：通过 `apply_chat_template(messages, tools=self.tool_schemas)` 自动将工具描述注入到 system prompt 中，这与 Agent-R1 的 `ToolRLDataset.use_default_tool_template` 等价。

#### _handle_generating_state — LLM 生成

```python
async def _handle_generating_state(self, agent_data, sampling_params):
    # 异步调用 LLM Server 生成
    output = await self.server_manager.generate(
        request_id=agent_data.request_id,
        prompt_ids=agent_data.prompt_ids,
        sampling_params=sampling_params,
    )

    agent_data.assistant_turns += 1
    agent_data.response_ids = output.token_ids

    # 累积更新
    agent_data.prompt_ids += agent_data.response_ids    # 追加到 prompt
    agent_data.response_mask += [1] * len(agent_data.response_ids)  # 全部标记为 LLM 生成
    if output.log_probs:
        agent_data.response_logprobs += output.log_probs

    # 检查终止条件
    if len(agent_data.response_mask) >= self.response_length:
        return AgentState.TERMINATED
    if self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
        return AgentState.TERMINATED

    # 提取工具调用
    _, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(
        agent_data.response_ids
    )

    if agent_data.tool_calls:
        return AgentState.PROCESSING_TOOLS
    else:
        return AgentState.TERMINATED
```

**关键点**：
- `prompt_ids` 是累积式更新：每轮生成后 `prompt_ids += response_ids`
- `response_mask` 同步累积：LLM 生成的部分标记为 `[1, 1, ..., 1]`
- 终止条件：总响应长度超限 / assistant 轮数超限 / 无工具调用

#### _handle_processing_tools_state — 执行工具

```python
async def _handle_processing_tools_state(self, agent_data):
    add_messages = []

    # 并行执行多个工具调用（最多 max_parallel_calls 个）
    tasks = []
    for tool_call in agent_data.tool_calls[:self.max_parallel_calls]:
        tasks.append(self._call_tool(tool_call, agent_data.tools_kwargs, agent_data))

    responses = await asyncio.gather(*tasks)

    # 处理工具响应
    for tool_response, tool_reward, _ in responses:
        message = {"role": "tool", "content": tool_response.text or ""}
        add_messages.append(message)
        if tool_reward is not None:
            agent_data.tool_rewards.append(tool_reward)

    agent_data.messages.extend(add_messages)

    # 使用 chat template 将工具响应转为 token IDs
    response_ids = await self.apply_chat_template(
        add_messages,
        images=new_images,
        remove_system_prompt=True,  # ← Delta tokenization: 去除重复的 system prompt
    )

    # 检查长度
    if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
        return AgentState.TERMINATED

    # 累积更新（工具响应部分 mask = 0）
    agent_data.prompt_ids += response_ids
    agent_data.response_mask += [0] * len(response_ids)  # ← 工具响应不计算梯度
    if agent_data.response_logprobs:
        agent_data.response_logprobs += [0.0] * len(response_ids)

    agent_data.user_turns += 1
    return AgentState.GENERATING  # 回到生成状态
```

**关键点**：
- 工具响应用 `role="tool"` 消息表示，通过 `apply_chat_template` 自动转为正确的 token 格式
- **Delta-based Tokenization**：`remove_system_prompt=True` 去掉重复的 system prompt tokens
- response_mask 中工具响应部分为 `[0, 0, ..., 0]`（不计算 RL 梯度）
- 工具级别奖励 `tool_reward` 被收集到 `agent_data.tool_rewards`

#### _call_tool — 工具生命周期管理

```python
async def _call_tool(self, tool_call, tools_kwargs, agent_data):
    try:
        tool_name = tool_call.name
        tool_args = json.loads(tool_call.arguments)
        tool = self.tools[tool_name]

        # 完整生命周期: create → execute → release
        kwargs = tools_kwargs.get(tool_name, {})
        instance_id, _ = await tool.create(
            create_kwargs=kwargs.get("create_kwargs", {})
        )
        tool_response, tool_reward, res = await tool.execute(
            instance_id, tool_args, agent_data=agent_data
        )
    except Exception as e:
        return ToolResponse(text=f"Error: {e}"), 0.0, {}
    finally:
        if tool and instance_id:
            await tool.release(instance_id)

    # 截断过长的工具响应
    if tool_response_text and len(tool_response_text) > self.max_tool_response_length:
        if self.tool_response_truncate_side == "left":
            tool_response_text = tool_response_text[:self.max_tool_response_length] + "...(truncated)"
        elif self.tool_response_truncate_side == "right":
            tool_response_text = "(truncated)..." + tool_response_text[-self.max_tool_response_length:]
        else:  # middle
            length = self.max_tool_response_length // 2
            tool_response_text = tool_response_text[:length] + "...(truncated)..." + tool_response_text[-length:]

    return ToolResponse(text=tool_response_text), tool_reward, res
```

**关键点**：
- `create → execute → release` 三阶段生命周期（Agent-R1 只有 execute）
- `create_kwargs` 从数据集的 `extra_info.tools_kwargs` 传入（如 ground_truth）
- 支持三种截断策略：左截断、右截断、中间截断
- 异常处理：工具执行失败时返回错误文本而不是抛出异常

---

### 2.2 基础设施层 — `verl/experimental/agent_loop/agent_loop.py`

约 1014 行代码，提供了整个 Agent Loop 系统的基础设施。

#### AsyncLLMServerManager — 异步 LLM 服务管理器

```python
class AsyncLLMServerManager:
    def __init__(self, config, server_handles, max_cache_size=10000):
        self.server_handles = server_handles  # 多个 vLLM/SGLang Server 的 Ray Actor 句柄
        # 最少请求负载均衡
        self.weighted_servers = [[0, idx, server] for idx, server in enumerate(server_handles)]
        heapq.heapify(self.weighted_servers)
        # LRU cache 实现 sticky session
        self.request_id_to_server = LRUCache(maxsize=max_cache_size)

    def _choose_server(self, request_id):
        # Sticky session: 同一 request_id 总是路由到同一 Server
        if request_id in self.request_id_to_server:
            return self.request_id_to_server[request_id]
        # 否则选择请求最少的 Server
        _, _, server = self.weighted_servers[0]
        self.weighted_servers[0][0] += 1
        heapq.heapreplace(self.weighted_servers, self.weighted_servers[0])
        self.request_id_to_server[request_id] = server
        return server

    async def generate(self, request_id, *, prompt_ids, sampling_params, ...):
        server = self._choose_server(request_id)
        output = await server.generate.remote(
            request_id=uuid4().hex,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
        )
        return output
```

**关键点**：
- 支持多个推理 Server 的负载均衡（最少请求策略）
- **Sticky session**：同一轨迹的多轮请求路由到同一 Server，利用 prefix caching
- 完全异步调用

#### AgentLoopBase — 抽象基类

```python
class AgentLoopBase(ABC):
    def __init__(self, trainer_config, server_manager, tokenizer, processor,
                 dataset_cls, dataset_config, **kwargs):
        self.server_manager = server_manager
        self.tokenizer = tokenizer
        self.processor = processor
        self.system_prompt = initialize_system_prompt(self.tokenizer, ...)

    async def apply_chat_template(self, messages, tools=None, images=None,
                                   videos=None, remove_system_prompt=False):
        """
        将消息列表转为 token IDs。
        remove_system_prompt=True 时去掉前缀的 system prompt tokens，
        实现 Delta-based tokenization。
        """
        prompt_ids = tokenizer.apply_chat_template(
            messages, tools=tools, add_generation_prompt=True, tokenize=True
        )
        if remove_system_prompt:
            prompt_ids = prompt_ids[len(self.system_prompt):]
        return prompt_ids

    @abstractmethod
    async def run(self, sampling_params, **kwargs) -> AgentLoopOutput:
        raise NotImplementedError
```

**Delta-based Tokenization 解释**：

多轮对话中，每次调用 `apply_chat_template` 都会生成完整的 token 序列（包含 system prompt）。但我们只需要新增部分的 tokens（delta）。通过记录初始 system prompt 的长度，在非首次调用时截掉前缀，就能得到 delta tokens。

```
首次: [system_tokens | user_tokens | ... ]  ← 完整保留
后续: [system_tokens | tool_response_tokens | ...]
                       ↑ 去掉 system_tokens 前缀
→ 只保留 [tool_response_tokens | ...]
```

#### AgentLoopOutput — 输出数据结构

```python
class AgentLoopOutput(BaseModel):
    prompt_ids: list[int]               # Prompt token IDs
    response_ids: list[int]             # 响应 token IDs（含 LLM + tool 响应）
    response_mask: list[int]            # 1=LLM, 0=tool/interaction
    response_logprobs: Optional[list[float]]  # Log 概率
    multi_modal_data: Optional[dict]    # 多模态数据
    reward_score: Optional[float]       # 奖励分数（streaming reward 计算）
    num_turns: int                      # 总对话轮数
    metrics: AgentLoopMetrics           # 性能指标
    extra_fields: dict[str, Any]        # 扩展字段
```

#### AgentLoopWorker — Ray Worker

```python
class AgentLoopWorker:
    """Ray Actor，处理一个 batch 的样本。"""

    def __init__(self, config, server_handles, reward_loop_worker_handles=None):
        self.server_manager = AsyncLLMServerManager(config, server_handles)
        self.tokenizer = hf_tokenizer(model_path)
        self.processor = hf_processor(model_path)

    async def generate_sequences(self, batch: DataProto) -> DataProto:
        """处理一个 batch"""
        # 为每个样本创建异步任务
        tasks = []
        for i in range(len(batch)):
            kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items()}
            tasks.append(asyncio.create_task(
                self._run_agent_loop(sampling_params, ..., **kwargs)
            ))
        outputs = await asyncio.gather(*tasks)  # 并行执行所有样本

        output = self._postprocess(outputs, ...)
        return output

    async def _run_agent_loop(self, sampling_params, ..., agent_name, **kwargs):
        """根据 agent_name 创建并运行对应的 AgentLoop"""
        agent_loop = hydra.utils.instantiate(
            config=_agent_loop_registry[agent_name],
            trainer_config=..., server_manager=..., tokenizer=..., ...
        )
        output = await agent_loop.run(sampling_params, **kwargs)
        return await self._agent_loop_postprocess(output, **kwargs)
```

**关键点**：
- 每个 Worker 是一个 Ray Actor
- batch 中的每个样本独立异步执行
- 通过 `agent_name` 字段（数据集中指定）选择使用哪个 AgentLoop 实现
- 支持 streaming reward 计算（通过 `reward_loop_worker_handles`）

#### _agent_loop_postprocess — 后处理

将 AgentLoopOutput（变长 list）转为 _InternalAgentLoopOutput（固定长度 tensor）：

```python
async def _agent_loop_postprocess(self, output, **kwargs):
    # 左填充 prompt
    tokenizer.padding_side = "left"
    prompt_output = tokenizer.pad({"input_ids": output.prompt_ids},
                                    padding="max_length", max_length=prompt_length)

    # 右填充 response
    tokenizer.padding_side = "right"
    response_output = tokenizer.pad({"input_ids": output.response_ids},
                                      padding="max_length", max_length=response_length)

    # response_mask = 原始 mask × attention_mask（padding 位置为 0）
    response_mask = response_mask_output["input_ids"] * response_output["attention_mask"]

    # 拼接
    input_ids = cat(prompt_output["input_ids"], response_output["input_ids"])
    attention_mask = cat(prompt_attention_mask, response_attention_mask)

    # 可选：计算 streaming reward
    await self._compute_score(output, ...)
```

#### AgentLoopManager — 顶层编排器

```python
class AgentLoopManager:
    def __init__(self, config, worker_group, rollout_resource_pool, ...):
        self._initialize_llm_servers(rollout_resource_pool)  # 启动 vLLM/SGLang
        self._init_agent_loop_workers()  # 创建 N 个 AgentLoopWorker

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """分发 batch 到多个 Worker，收集结果"""
        chunks = prompts.chunk(len(self.agent_loop_workers))
        outputs = ray.get([
            worker.generate_sequences.remote(chunk)
            for worker, chunk in zip(self.agent_loop_workers, chunks)
        ])
        output = DataProto.concat(outputs)
        return output
```

**关键点**：
- `AgentLoopManager` 替代了 Agent-R1 中 `ToolGenerationManager` 的角色
- 但它不直接执行循环，而是通过 Ray 分发到多个 Worker
- 每个 Worker 内部异步处理多个样本

---

### 2.3 工具解析器 — `verl/experimental/agent_loop/tool_parser.py`

#### ToolParser 注册机制

```python
class ToolParser(ABC):
    _registry: dict[str, type["ToolParser"]] = {}

    @abstractmethod
    async def extract_tool_calls(self, responses_ids: list[int]) -> tuple[str, list[FunctionCall]]:
        """从响应 token IDs 中提取工具调用"""

    @classmethod
    def get_tool_parser(cls, name: str, tokenizer):
        return cls._registry[name](tokenizer)

    @classmethod
    def register(cls, name: str):
        def decorator(subclass):
            cls._registry[name] = subclass
            return subclass
        return decorator
```

#### HermesToolParser — `<tool_call>` 格式解析器

```python
@ToolParser.register("hermes")
class HermesToolParser(ToolParser):
    def __init__(self, tokenizer):
        self.tool_call_start_token = "<tool_call>"
        self.tool_call_end_token = "</tool_call>"
        self.tool_call_regex = regex.compile(
            r"<tool_call>(.*?)</tool_call>", regex.DOTALL
        )

    async def extract_tool_calls(self, responses_ids):
        text = tokenizer.decode(responses_ids)

        if "<tool_call>" not in text or "</tool_call>" not in text:
            return text, []

        matches = self.tool_call_regex.findall(text)
        function_calls = []
        for match in matches:
            function_call = json.loads(match)
            name = function_call["name"]
            arguments = function_call["arguments"]
            function_calls.append(FunctionCall(
                name=name,
                arguments=json.dumps(arguments, ensure_ascii=False)
            ))
        return content, function_calls
```

**关键点**：
- `hermes` 格式与 Agent-R1 的 NousToolEnv 解析的格式**完全一致**
- 两者都解析 `<tool_call>{"name": "...", "arguments": {...}}</tool_call>`
- 但 verl 的 FunctionCall 中 arguments 是 JSON string，而 Agent-R1 直接是 dict

#### FunctionCall 数据结构

```python
class FunctionCall(BaseModel):
    arguments: str   # JSON 字符串格式的参数
    name: str        # 函数名
```

---

### 2.4 工具接口 — `verl/tools/base_tool.py`

```python
class BaseTool:
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        self.config = config
        self.tool_schema = tool_schema
        self.name = self.tool_schema.function.name

    async def create(self, instance_id=None, **kwargs) -> tuple[str, ToolResponse]:
        """创建工具实例（用于有状态的工具）"""
        return str(uuid4()), ToolResponse()

    async def execute(self, instance_id, parameters, **kwargs) -> tuple[ToolResponse, float, dict]:
        """
        执行工具
        返回: (tool_response, step_reward, metrics)
          - tool_response: ToolResponse 对象
          - step_reward: 该步骤的奖励（可为 None）
          - metrics: 监控指标
        """
        return ToolResponse(text="..."), 0.0, {}

    async def calc_reward(self, instance_id, **kwargs) -> float:
        """计算最终奖励"""
        return 0.0

    async def release(self, instance_id, **kwargs) -> None:
        """释放工具实例"""
        pass
```

**与 Agent-R1 BaseTool 的关键区别**：
- verl 的方法都是 **async**
- verl 有 **create/release 生命周期**（适合有状态的工具，如 sandbox session）
- verl 的 execute 返回 **(ToolResponse, reward, metrics)** 而不是简单的 dict
- verl 通过 `config` 和 `tool_schema` 初始化，而 Agent-R1 用类属性

---

### 2.5 ToolResponse 和 Schema — `verl/tools/schemas.py`

```python
class ToolResponse(BaseModel):
    text: Optional[str] = None     # 文本响应
    image: Optional[Any] = None    # 图像响应（支持多模态）
    video: Optional[Any] = None    # 视频响应

class OpenAIFunctionToolSchema(BaseModel):
    type: str = "function"
    function: FunctionSchema

class FunctionSchema(BaseModel):
    name: str
    description: str
    parameters: dict
```

---

### 2.6 SearchTool 参考实现 — `verl/tools/search_tool.py`

verl 已有一个搜索工具实现，可以直接参考用于 MultihopQA 的 WikiSearchTool 适配。

```python
class SearchTool(BaseTool):
    def __init__(self, config, tool_schema):
        super().__init__(config, tool_schema)
        self.num_workers = config.get("num_workers", 120)
        self.rate_limit = config.get("rate_limit", 120)
        self.timeout = config.get("timeout", 30)
        self.retrieval_service_url = config.get("retrieval_service_url")

        # Ray 并发控制
        self.execution_pool = init_search_execution_pool(
            num_workers=self.num_workers,
            rate_limit=self.rate_limit,
        )

    async def create(self, instance_id=None, **kwargs):
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {"response": "", "reward": []}
        return instance_id, ToolResponse()

    async def execute(self, instance_id, parameters, **kwargs):
        query_list = parameters.get("query_list")
        result_text, metadata = await self.execution_pool.execute.remote(
            self.execute_search, instance_id, query_list,
            self.retrieval_service_url, self.topk, self.timeout
        )
        return ToolResponse(text=result_text), 0.0, metrics

    async def release(self, instance_id, **kwargs):
        del self._instance_dict[instance_id]
```

**关键点**：
- 使用 `TokenBucketWorker` (Ray Actor) 做全局并发限流
- `SearchExecutionWorker` 通过 `ray.remote` 实现线程池
- 参数从 `parameters.get("query_list")` 读取（注意参数名与 Agent-R1 不同）

---

### 2.7 工具配置文件 — YAML 格式

以搜索工具为例 (`examples/sglang_multiturn/config/tool_config/search_tool_config.yaml`)：

```yaml
tools:
  - class_name: verl.tools.search_tool.SearchTool
    config:
      retrieval_service_url: http://127.0.0.1:8000/retrieve
      num_workers: 120
      rate_limit: 120
      timeout: 30
      type: native
    tool_schema:
      type: function
      function:
        name: search
        description: Searches the web for relevant information based on the given query.
        parameters:
          type: object
          properties:
            query_list:
              type: array
              item:
                type: string
              description: A list of fully-formed semantic queries.
          required:
            - query_list
```

**结构**：
- `class_name`: Python 类的完整路径（通过 `initialize_tools_from_config` 动态加载）
- `config`: 传递给 `__init__` 的 config dict
- `tool_schema`: OpenAI function calling 格式的 schema

---

### 2.8 数据格式 — GSM8K 参考

参考 `examples/data_preprocess/gsm8k_tool_agent_loop.py`：

```python
data = {
    "data_source": "openai/gsm8k",
    "agent_name": "tool_agent",       # ← 指定使用 ToolAgentLoop
    "prompt": [
        {"role": "system", "content": "You are a math expert..."},
        {"role": "user", "content": question},
    ],
    "ability": "math",
    "reward_model": {"style": "rule", "ground_truth": solution},
    "extra_info": {
        "split": split,
        "index": idx,
        "need_tools_kwargs": True,    # ← 标记需要传递 tools_kwargs
        "tools_kwargs": {
            "calc_gsm8k_reward": {    # ← 按工具名组织
                "create_kwargs": {"ground_truth": solution},
            },
        },
    },
}
```

**关键字段**：
- `agent_name`: 决定使用哪个 AgentLoop（`"tool_agent"` → ToolAgentLoop）
- `extra_info.need_tools_kwargs`: 告诉框架需要从 extra_info 提取 tools_kwargs
- `extra_info.tools_kwargs`: 按工具名组织的参数，传递到 `_call_tool` 中

---

### 2.9 训练配置 — YAML

GSM8K 多轮训练配置 (`examples/sglang_multiturn/config/gsm8k_multiturn_grpo.yaml`)：

```yaml
hydra:
  searchpath:
    - file://verl/trainer/config

defaults:
  - ppo_trainer       # 基础配置
  - _self_

data:
  max_prompt_length: 1024
  max_response_length: 1024
  train_batch_size: 256
  return_raw_chat: True       # ← 必须为 True，传递 raw chat messages

actor_rollout_ref:
  hybrid_engine: True
  rollout:
    name: sglang
    multi_turn:
      enable: True              # ← 启用多轮
      max_assistant_turns: 5    # ← 最大 assistant 轮数
```

训练脚本中通过 CLI 覆盖更多参数：
```bash
python3 -m verl.trainer.main_ppo \
    --config-path=examples/sglang_multiturn/config \
    --config-name=gsm8k_multiturn_grpo \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=.../tool_config.yaml \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=2048 \
    algorithm.adv_estimator=grpo \
    ...
```

---

## 3. 完整数据流图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           数据准备阶段                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  gsm8k_tool_agent_loop.py → train.parquet                              │
│    {data_source, agent_name: "tool_agent",                             │
│     prompt: [{role:"system",...}, {role:"user",...}],                   │
│     reward_model: {style:"rule", ground_truth: answer},                │
│     extra_info: {need_tools_kwargs: True, tools_kwargs: {...}}}         │
│      ↓                                                                  │
│  RLHFDataset (return_raw_chat=True)                                    │
│      ↓ 保留 raw_prompt (chat messages 列表)                             │
│  DataProto:                                                             │
│    non_tensor_batch["raw_prompt"] = messages                           │
│    non_tensor_batch["agent_name"] = "tool_agent"                       │
│    non_tensor_batch["tools_kwargs"] = {...}                            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                   训练循环 (RayPPOTrainer)                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  for epoch in range(total_epochs):                                      │
│    for batch in dataloader:                                             │
│                                                                         │
│    ┌────────────────────────────────────────────────────────────────┐   │
│    │ 1. 生成阶段: AgentLoopManager.generate_sequences(DataProto)   │   │
│    │                                                                │   │
│    │    DataProto → chunk() → 分发到 N 个 AgentLoopWorker          │   │
│    │                                                                │   │
│    │    每个 Worker 内部:                                           │   │
│    │    for each sample (异步并行):                                 │   │
│    │      agent_loop = ToolAgentLoop(...)                          │   │
│    │      output = await agent_loop.run(sampling_params, **kwargs)  │   │
│    │                                                                │   │
│    │    ToolAgentLoop 状态机:                                       │   │
│    │    PENDING                                                     │   │
│    │      → apply_chat_template(messages, tools=schemas)           │   │
│    │    GENERATING                                                  │   │
│    │      → server_manager.generate(prompt_ids, ...)               │   │
│    │      → tool_parser.extract_tool_calls(response_ids)           │   │
│    │      → response_mask += [1] * len(response_ids)               │   │
│    │    PROCESSING_TOOLS                                            │   │
│    │      → _call_tool(): create → execute → release               │   │
│    │      → apply_chat_template([{role:"tool",...}])                │   │
│    │      → response_mask += [0] * len(tool_response_ids)          │   │
│    │    GENERATING (loop back)                                      │   │
│    │      ...                                                      │   │
│    │    TERMINATED                                                  │   │
│    │                                                                │   │
│    │    Worker 后处理:                                              │   │
│    │      _agent_loop_postprocess(): padding → tensor              │   │
│    │      _postprocess(): stack → DataProto                        │   │
│    │                                                                │   │
│    │    输出: DataProto                                             │   │
│    │      prompts: (B, prompt_length) — 左填充                     │   │
│    │      responses: (B, response_length) — 右填充                 │   │
│    │      response_mask: (B, response_length) — 1=LLM, 0=tool     │   │
│    │      input_ids: (B, prompt_length + response_length)          │   │
│    │      attention_mask: (B, total_length)                        │   │
│    │      rm_scores: (B, response_length) — 如有 streaming reward  │   │
│    └────────────────────────────────────────────────────────────────┘   │
│                              ↓                                          │
│    ┌────────────────────────────────────────────────────────────────┐   │
│    │ 2. 奖励计算                                                    │   │
│    │                                                                │   │
│    │    方式 A: Streaming Reward (通过 RewardLoopWorker)            │   │
│    │      在 _agent_loop_postprocess 中异步计算                     │   │
│    │      结果存入 rm_scores                                        │   │
│    │                                                                │   │
│    │    方式 B: Custom Reward Function                              │   │
│    │      custom_reward_function.path = "path.to.reward_fn"        │   │
│    │      在 RayPPOTrainer 中调用                                   │   │
│    └────────────────────────────────────────────────────────────────┘   │
│                              ↓                                          │
│    ┌────────────────────────────────────────────────────────────────┐   │
│    │ 3. GRPO 优势计算                                               │   │
│    │                                                                │   │
│    │    compute_grpo_outcome_advantage():                          │   │
│    │      按 prompt group 分组                                      │   │
│    │      组内: advantage = (reward - mean) / std                   │   │
│    │                                                                │   │
│    │    RL loss 只作用于 response_mask=1 的 token                   │   │
│    └────────────────────────────────────────────────────────────────┘   │
│                              ↓                                          │
│    ┌────────────────────────────────────────────────────────────────┐   │
│    │ 4. 策略更新 (FSDP)                                             │   │
│    │                                                                │   │
│    │    policy_loss = agg_loss(log_ratio * advantage, mask=response_mask) │
│    │    + optional kl_loss                                          │   │
│    └────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. response_mask 语义详解

```
responses 张量 (右填充):
  |<- Turn1 LLM 生成 ->|<- Turn1 tool 响应 ->|<- Turn2 LLM 生成 ->|<- padding ->|

response_mask:
  | 1, 1, ..., 1, 1    | 0, 0, ..., 0, 0     | 1, 1, ..., 1, 1    | 0, 0, ..., 0|
    ↑ 计算 RL 梯度         ↑ 不计算               ↑ 计算 RL 梯度       ↑ 不计算

最终: response_mask = raw_mask * response_attention_mask
  (response_attention_mask 在 padding 位置为 0)
```

这与 Agent-R1 的 `action_mask` 语义**完全等价**。

---

## 5. 关键实现细节总结

### 5.1 注册机制

ToolAgentLoop 通过 `@register("tool_agent")` 装饰器注册。数据集中 `agent_name = "tool_agent"` 会触发使用该 AgentLoop。如果未指定 agent_name，使用 `config.actor_rollout_ref.rollout.agent.default_agent_loop` 配置的默认值。

### 5.2 采样参数

从 rollout 配置读取：
```python
sampling_params = {
    "temperature": config.rollout.temperature,
    "top_p": config.rollout.top_p,
    "top_k": config.rollout.top_k,
    "repetition_penalty": 1.0,
    "logprobs": config.rollout.calculate_log_probs,
}
```

### 5.3 性能指标收集

每个 AgentLoopOutput 包含 metrics：
- `generate_sequences`: LLM 生成总耗时
- `tool_calls`: 工具调用总耗时
- `num_preempted`: 被抢占的次数

AgentLoopManager 聚合所有样本的指标，计算 min/max/mean 以及最慢样本的详情。

### 5.4 Streaming Reward

如果配置了 `reward_loop_worker_handles`，奖励会在每个样本完成后立即异步计算，结果存入 `rm_scores` 字段。这避免了所有样本完成后再批量计算奖励的延迟。
