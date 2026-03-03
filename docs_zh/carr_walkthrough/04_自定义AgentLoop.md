# Part 4: 自定义 Agent Loop

本部分详细介绍 verl `ToolAgentLoop` 的完整运作流程，为什么需要自定义 AgentLoop，以及 `CaRRToolAgentLoop` 的设计细节。

---

## 4.1 ToolAgentLoop 状态机完整解析

### 4.1.1 五个状态

`tool_agent_loop.py:46-51` 定义了状态枚举：

```python
class AgentState(Enum):
    PENDING = "pending"            # 初始化 prompt
    GENERATING = "generating"      # LLM 推理生成
    PROCESSING_TOOLS = "processing_tools"  # 执行工具调用
    INTERACTING = "interacting"    # 交互式反馈（我们不用）
    TERMINATED = "terminated"      # 终止
```

### 4.1.2 状态转换图

```
                    ┌─────────┐
                    │ PENDING │
                    └────┬────┘
                         │ apply_chat_template(messages, tools=...)
                         │ → prompt_ids
                         ▼
              ┌──────────────────────┐
         ┌───>│    GENERATING        │
         │    │ server_manager       │
         │    │  .generate()         │
         │    │ → response_ids       │
         │    │ → tool_calls?        │
         │    └──────────┬───────────┘
         │               │
         │     ┌─────────┴─────────┐
         │     │                   │
         │  有 tool_calls      无 tool_calls
         │     │                   │
         │     ▼                   ▼
         │  ┌──────────────┐  ┌────────────┐
         │  │ PROCESSING   │  │ TERMINATED │
         │  │ _TOOLS       │  └────────────┘
         │  │              │
         │  │ _call_tool() │
         │  │ × N 个工具   │
         │  └──────┬───────┘
         │         │
         └─────────┘ → GENERATING（下一轮对话）
```

**终止条件**（任一触发即进入 TERMINATED）：
1. `response_mask` 长度达到 `response_length` 上限
2. `assistant_turns` 达到 `max_assistant_turns`（配置为 30）
3. `user_turns` 达到 `max_user_turns`
4. 生成内容中没有 tool_call 且没有 interaction

### 4.1.3 run() 方法逐行解析

`tool_agent_loop.py:136-212` — 这是整个 Agent 一条轨迹的入口：

```python
@rollout_trace_op
async def run(self, sampling_params, **kwargs):
    # ① 获取 messages（来自 parquet 的 prompt 字段）
    messages = list(kwargs["raw_prompt"])

    # ② 提取多模态数据（我们不需要）
    multi_modal_data = await self.process_vision_info(messages)
    images = multi_modal_data.get("images")
    videos = multi_modal_data.get("videos")

    # ③ 生成唯一 request_id
    metrics = {}
    request_id = uuid4().hex
    tools_kwargs = kwargs.get("tools_kwargs", {})   # ← 来自 parquet extra_info

    # ④ 创建 AgentData（封装所有轨迹状态）
    agent_data = AgentData(
        messages=messages,
        image_data=images,
        video_data=videos,
        metrics=metrics,
        request_id=request_id,
        tools_kwargs=tools_kwargs,
        ...
    )

    # ⑤ 状态机主循环
    state = AgentState.PENDING
    while state != AgentState.TERMINATED:
        if state == AgentState.PENDING:
            state = await self._handle_pending_state(agent_data, sampling_params)
        elif state == AgentState.GENERATING:
            state = await self._handle_generating_state(agent_data, sampling_params)
        elif state == AgentState.PROCESSING_TOOLS:
            state = await self._handle_processing_tools_state(agent_data)
        ...

    # ⑥ 组装输出
    output = AgentLoopOutput(
        prompt_ids=prompt_ids,
        response_ids=response_ids[:self.response_length],
        response_mask=agent_data.response_mask[:self.response_length],
        response_logprobs=...,
        num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
        metrics=agent_data.metrics,
        extra_fields={},     # ← 可以放自定义数据
    )
    output.extra_fields.update({
        "turn_scores": agent_data.turn_scores,
        "tool_rewards": agent_data.tool_rewards,
    })
    return output
```

### 4.1.4 AgentData —— 轨迹状态容器

`tool_agent_loop.py:54-94`，每个样本一个 AgentData 实例：

```python
class AgentData:
    def __init__(self, messages, image_data, video_data, metrics,
                 request_id, tools_kwargs, interaction=None, interaction_kwargs=None):
        self.messages = messages              # 完整对话历史
        self.request_id = request_id          # 唯一标识（用于 sticky session 和 session 管理）
        self.tools_kwargs = tools_kwargs      # 工具参数（来自 parquet）

        # 运行时累积状态
        self.prompt_ids: list[int] = []       # 累积 token 序列（prompt + 所有 response）
        self.response_ids: list[int] = []     # 当前轮生成的 token
        self.response_mask: list[int] = []    # 1=LLM生成, 0=工具/系统插入
        self.response_logprobs: list[float] = []  # 对数概率
        self.turn_scores: list[float] = []    # 每轮交互的分数
        self.tool_rewards: list[float] = []   # 工具步级奖励
        self.user_turns = 0                   # 用户/工具轮次计数
        self.assistant_turns = 0              # 助手轮次计数

        self.tool_calls: list[FunctionCall] = []  # 当前轮解析出的工具调用
        self.extra_fields: dict[str, Any] = {}    # ← 自定义扩展字段
```

### 4.1.5 各状态处理器详解

**PENDING**（`_handle_pending_state`）：
```python
async def _handle_pending_state(self, agent_data, sampling_params):
    prompt_ids = await self.apply_chat_template(
        agent_data.messages, tools=self.tool_schemas, ...
    )
    agent_data.prompt_ids = prompt_ids   # 初始 prompt token 序列
    return AgentState.GENERATING
```

**GENERATING**（`_handle_generating_state`）：
```python
async def _handle_generating_state(self, agent_data, sampling_params, ...):
    # 调用推理服务器生成
    output = await self.server_manager.generate(
        request_id=agent_data.request_id,
        prompt_ids=agent_data.prompt_ids,   # 含之前所有轮次的 token
        sampling_params=sampling_params,
    )

    # 累积 token 和 mask
    agent_data.assistant_turns += 1
    agent_data.response_ids = output.token_ids
    agent_data.prompt_ids += agent_data.response_ids        # ← 累积
    agent_data.response_mask += [1] * len(agent_data.response_ids)  # ← LLM 生成的 token 标记为 1
    if output.log_probs:
        agent_data.response_logprobs += output.log_probs

    # 检查终止条件
    if len(agent_data.response_mask) >= self.response_length:
        return AgentState.TERMINATED

    # 解析 tool_calls
    _, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(agent_data.response_ids)

    if agent_data.tool_calls:
        return AgentState.PROCESSING_TOOLS
    else:
        return AgentState.TERMINATED     # 无工具调用 → 生成完毕
```

**PROCESSING_TOOLS**（`_handle_processing_tools_state`）：
```python
async def _handle_processing_tools_state(self, agent_data):
    # 并行执行所有工具调用
    tasks = []
    for tool_call in agent_data.tool_calls[:self.max_parallel_calls]:
        tasks.append(self._call_tool(tool_call, agent_data.tools_kwargs, agent_data))
    responses = await asyncio.gather(*tasks)

    # 处理工具响应，构建 tool messages
    add_messages = [...]
    agent_data.messages.extend(add_messages)

    # 工具响应的 token 标记为 0（不参与梯度）
    response_ids = await self.apply_chat_template(add_messages, remove_system_prompt=True)
    agent_data.prompt_ids += response_ids
    agent_data.response_mask += [0] * len(response_ids)  # ← 工具 token 标记为 0
    agent_data.user_turns += 1

    return AgentState.GENERATING   # 回到生成状态
```

### 4.1.6 response_mask 的累积过程

这是理解训练损失计算的关键：

```
第 1 轮 GENERATING:  mask=[1,1,1,1,1,1,1,1]        ← LLM 生成 8 个 token
第 1 轮 PROCESSING:  mask=[1,1,1,1,1,1,1,1, 0,0,0,0,0]  ← 工具响应 5 个 token（不参与训练）
第 2 轮 GENERATING:  mask=[1,1,1,1,1,1,1,1, 0,0,0,0,0, 1,1,1,1,1,1]  ← 又生成 6 个 token
第 2 轮 PROCESSING:  mask=[..., 0,0,0,0]            ← 又插入工具响应
...
最终 GENERATING:     mask=[..., 1,1,1,1,1,1,1,1,1,1]  ← 最终回答
```

只有 `mask=1` 的 token 参与 PPO 策略梯度更新。

---

## 4.2 为什么需要自定义 AgentLoop？

基类 `ToolAgentLoop` 不能直接用于 CaRR，有两个核心原因：

### 4.2.1 原因 1：CaRR Reward Server 需要特定的 history 格式

CaRR reward server 的 `/evaluate` 端点期望的 `history` 格式：

```json
[
  {"role": "user", "content": "问题文本"},
  {
    "role": "assistant",
    "content": "思考 + 可见文本",
    "tool_calls": [
      {
        "tool_call_id": "req_001_tc_0_0",
        "name": "browser.search",
        "arguments": "{\"query\":\"...\",\"num\":10}"
      }
    ]
  },
  {
    "role": "tool",
    "content": [
      {
        "tool_call_id": "req_001_tc_0_0",
        "output": "搜索结果文本"
      }
    ]
  },
  {"role": "assistant", "content": "最终回答..."}
]
```

**关键约束**：
1. `tool_call_id` 必须在 assistant 和 tool 消息之间匹配
2. tool 消息的 `content` 是 `list[dict]` 格式（不是纯字符串）
3. `arguments` 是 JSON 字符串（不是 dict）
4. `history[-1].role` **必须是 `"assistant"`**，否则 reward 直接为 0

但 `ToolAgentLoop` 维护的 `agent_data.messages` 并不包含 `tool_call_id` 绑定信息，也不是这种格式。

### 4.2.2 原因 2：Session 生命周期需要在 run() 的 finally 中管理

基类的 `run()` 方法没有 `try/finally` 结构来确保 session 清理。我们需要在轨迹结束（无论正常还是异常）时关闭 CaRR Tool Server 的 session。

### 4.2.3 原因 3：extra_fields 需要传递 reward history 和 task_unfinished

基类只在 `extra_fields` 中放了 `turn_scores` 和 `tool_rewards`。我们还需要：
- `messages`：CaRR 格式的 reward history（传给 reward server）
- `task_unfinished`：轨迹是否被截断（被截断时 reward=0）

---

## 4.3 extra_fields → reward 函数的数据流

这是一条关键路径。理解了它就理解了为什么要维护 reward_history。

### 4.3.1 AgentLoop 输出 → AgentLoopWorker 收集

```python
# tool_agent_loop.py:198-211
output = AgentLoopOutput(
    ...,
    extra_fields={},
)
output.extra_fields.update({
    "turn_scores": agent_data.turn_scores,
    "tool_rewards": agent_data.tool_rewards,
})
# 我们的 CaRRToolAgentLoop 还会添加：
# output.extra_fields["messages"] = reward_history
# output.extra_fields["task_unfinished"] = bool
```

### 4.3.2 AgentLoopWorker 收集 extra_fields → non_tensor_batch

`agent_loop.py:780-796`：

```python
# _postprocess() 中
extra_fields = {}
all_keys = set(key for input_item in inputs for key in input_item.extra_fields)
for key in all_keys:
    temp_arr = np.empty(len(inputs), dtype=object)
    temp_arr[:] = [input.extra_fields.get(key) for input in inputs]
    extra_fields[key] = temp_arr

non_tensor_batch.update(extra_fields)
```

所以 `extra_fields` 中的所有 key 都会变成 `non_tensor_batch` 的一部分。

### 4.3.3 Streaming Reward 路径（关键！）

当使用 streaming reward 时（`reward_loop_worker_handles` 不为 None），数据流不同。

`agent_loop.py:698-718`：

```python
# 在 _agent_loop_postprocess() 中（每个样本 rollout 完立即执行）
non_tensor_batch = {
    **{k: np.array([v]) for k, v in kwargs.items()},
    "__num_turns__": np.array([output.num_turns]),
    "tool_extra_fields": np.array([output.extra_fields], dtype=object),  # ← 整个 extra_fields 打包
}
data = DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

# 发送给 RewardLoopWorker 计算分数
result = await selected_reward_loop_worker_handle.compute_score.remote(data)
output.reward_score = result["reward_score"]
output.extra_fields["reward_extra_info"] = result["reward_extra_info"]
```

### 4.3.4 NaiveRewardManager 解包

`verl/experimental/reward_loop/reward_manager/naive.py:34-99`：

```python
async def run_single(self, data):
    data_item = data[0]

    data_source = data_item.non_tensor_batch["data_source"]
    ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
    extra_info = data_item.non_tensor_batch.get("extra_info", {})

    # ← 关键！tool_extra_fields 合并到 extra_info 中
    tool_extra_fields = data_item.non_tensor_batch.get("tool_extra_fields", None)
    if tool_extra_fields is not None:
        extra_info.update(tool_extra_fields.items())

    # 调用我们的 compute_score
    result = await self.compute_score(
        data_source=data_source,
        solution_str=response_str,      # LLM 生成的文本
        ground_truth=ground_truth,      # 标准答案
        extra_info=extra_info,          # ← 包含 messages, task_unfinished, rubrics 等
    )
```

**完整数据流**：

```
CaRRToolAgentLoop.run()
    output.extra_fields["messages"] = reward_history
    output.extra_fields["task_unfinished"] = True/False
        │
        ▼
AgentLoopWorker._agent_loop_postprocess()
    non_tensor_batch["tool_extra_fields"] = output.extra_fields
        │
        ▼
NaiveRewardManager.run_single()
    extra_info = parquet 的 extra_info (含 rubrics, search_forbidden_strs)
    extra_info.update(tool_extra_fields)  ← 合并！
    # 现在 extra_info 同时包含：
    #   - rubrics (来自 parquet)
    #   - search_forbidden_strs (来自 parquet)
    #   - messages (来自 AgentLoop)
    #   - task_unfinished (来自 AgentLoop)
        │
        ▼
carr_reward.compute_score(extra_info=extra_info)
    messages = extra_info["messages"]       # CaRR reward history
    rubrics = extra_info["rubrics"]         # 评估标准
    task_unfinished = extra_info["task_unfinished"]
```

---

## 4.4 CaRRToolAgentLoop 设计

### 4.4.1 注册

```python
from verl.experimental.agent_loop.agent_loop import register
from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop

@register("carr_tool_agent")
class CaRRToolAgentLoop(ToolAgentLoop):
    ...
```

注册名 `"carr_tool_agent"` 与 parquet 中 `agent_name` 字段对应。运行时通过 `VERL_USE_EXTERNAL_MODULES` 环境变量触发导入。

### 4.4.2 需要维护的额外状态

对每条轨迹（每个 request_id），我们需要维护：

```python
# 在 run() 方法中
reward_history = []              # CaRR 格式的完整对话历史
pending_tool_calls = []          # 当前轮次的待匹配 tool_calls
turn_idx = 0                     # 轮次计数（用于生成 tool_call_id）
```

### 4.4.3 reward_history 的构建规则

**① 初始化**——添加 user 消息：
```python
reward_history.append({
    "role": "user",
    "content": messages[0]["content"]  # 用户问题
})
```

**② GENERATING 结束后**——添加 assistant 消息：
```python
# 解码 LLM 生成的 token
response_text = tokenizer.decode(response_ids, skip_special_tokens=True)

# 如果有工具调用
if tool_calls:
    reward_history.append({
        "role": "assistant",
        "content": response_text,       # 可能包含 thinking 内容
        "tool_calls": [
            {
                "tool_call_id": f"{request_id}_tc_{turn_idx}_{i}",   # 确定性 ID
                "name": tc.name,
                "arguments": tc.arguments,   # 保持 JSON 字符串
            }
            for i, tc in enumerate(tool_calls)
        ]
    })
```

**③ PROCESSING_TOOLS 结束后**——添加 tool 消息：
```python
# 每个工具调用的结果
reward_history.append({
    "role": "tool",
    "content": [
        {
            "tool_call_id": f"{request_id}_tc_{turn_idx}_{i}",   # 与 assistant 的一致
            "output": tool_response_text,
        }
        for i, tool_response_text in enumerate(tool_results)
    ]
})
turn_idx += 1
```

**④ 最终回答**（无工具调用的 assistant 消息）：
```python
reward_history.append({
    "role": "assistant",
    "content": response_text,   # 最终回答
})
```

### 4.4.4 tool_call_id 生成规则

使用**确定性规则**：`{request_id}_tc_{turn}_{index}`

```
第 0 轮第 0 个工具: req_abc123_tc_0_0
第 0 轮第 1 个工具: req_abc123_tc_0_1  （如果一轮有多个工具调用）
第 1 轮第 0 个工具: req_abc123_tc_1_0
...
```

这样 assistant 消息中的 `tool_call_id` 和 tool 消息中的 `tool_call_id` 自然匹配。

### 4.4.5 task_unfinished 判定

```python
task_unfinished = (
    len(reward_history) == 0
    or reward_history[-1].get("role") != "assistant"
)
```

什么情况下 `task_unfinished = True`？
1. reward_history 为空（极端异常情况）
2. 最后一条消息不是 assistant 角色——这发生在轨迹被 `max_turns` 或 `response_length` 截断时（此时最后一条可能是 tool 消息）

**CaRR reward server 的行为**：
- `task_unfinished=True` → 直接返回 `reward=0`
- `history[-1].role != "assistant"` → 也返回 `reward=0`

这两个条件互为保险。

### 4.4.6 覆盖的方法

`CaRRToolAgentLoop` 需要覆盖以下方法：

| 方法 | 覆盖原因 |
|------|----------|
| `run()` | 添加 try/finally（session 关闭）；初始化 reward_history；构建 extra_fields 输出 |
| `_handle_generating_state()` | 在 LLM 生成后提取文本，构建 reward_history 的 assistant 条目 |
| `_handle_processing_tools_state()` | 在工具执行后构建 reward_history 的 tool 条目 |

**不需要覆盖**：
- `__init__()`：基类初始化已足够（工具加载、parser 配置等）
- `_handle_pending_state()`：prompt tokenize 逻辑不变
- `_call_tool()`：工具调用逻辑不变（CaRRBrowserTool 已处理好 session）

### 4.4.7 run() 方法实现框架

```python
@rollout_trace_op
async def run(self, sampling_params, **kwargs):
    # 初始化 reward_history
    messages = list(kwargs["raw_prompt"])
    request_id = uuid4().hex

    # 用于维护 CaRR reward history 的状态
    self._reward_history = []
    self._turn_idx = 0
    self._request_id = request_id

    # 添加初始 user 消息到 reward_history
    # 注意：messages 可能只有一个 user 消息，也可能有 system + user
    for msg in messages:
        self._reward_history.append({"role": msg["role"], "content": msg["content"]})

    try:
        # 调用父类逻辑（但使用覆盖后的状态处理器）
        # 实际上需要复制 run() 的主体逻辑，因为需要在状态机内部插入 reward_history 维护
        ...（状态机循环）...

        # 构建输出
        output = AgentLoopOutput(...)
        output.extra_fields.update({
            "turn_scores": agent_data.turn_scores,
            "tool_rewards": agent_data.tool_rewards,
            "messages": self._reward_history,          # ← 传给 reward server
            "task_unfinished": task_unfinished,         # ← 是否被截断
        })
        return output

    finally:
        # 无论成功还是异常，都关闭 session
        from .carr_session_manager import CaRRSessionManager
        session_manager = CaRRSessionManager.get_instance()
        tool_server_url = self._get_tool_server_url()
        await session_manager.close(request_id, tool_server_url)
```

### 4.4.8 _handle_generating_state 覆盖要点

```python
async def _handle_generating_state(self, agent_data, sampling_params, ...):
    # ① 调用父类逻辑完成 LLM 生成
    # （或复制其核心逻辑）

    # ② 在状态转换前，提取文本并更新 reward_history
    response_text = self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)

    if agent_data.tool_calls:
        # assistant with tool_calls
        self._reward_history.append({
            "role": "assistant",
            "content": response_text,
            "tool_calls": [
                {
                    "tool_call_id": f"{self._request_id}_tc_{self._turn_idx}_{i}",
                    "name": tc.name,
                    "arguments": tc.arguments,  # 已经是 JSON 字符串
                }
                for i, tc in enumerate(agent_data.tool_calls)
            ]
        })
    else:
        # final assistant response
        self._reward_history.append({
            "role": "assistant",
            "content": response_text,
        })

    # ③ 返回下一个状态
    return next_state
```

### 4.4.9 _handle_processing_tools_state 覆盖要点

```python
async def _handle_processing_tools_state(self, agent_data):
    # ① 调用父类逻辑完成工具执行
    # 需要收集每个工具的原始响应文本

    # ② 构建 tool 消息
    tool_contents = []
    for i, (tool_response, _, _) in enumerate(responses):
        tool_contents.append({
            "tool_call_id": f"{self._request_id}_tc_{self._turn_idx}_{i}",
            "output": tool_response.text or "",
        })
    self._reward_history.append({
        "role": "tool",
        "content": tool_contents,
    })
    self._turn_idx += 1

    # ③ 返回 GENERATING 继续下一轮
    return AgentState.GENERATING
```

---

## 4.5 注册机制详解

### 4.5.1 @register 装饰器

`agent_loop.py:333-341`：

```python
_agent_loop_registry: dict[str, dict] = {}

def register(agent_name: str):
    def decorator(subclass: type[AgentLoopBase]) -> type[AgentLoopBase]:
        fqdn = f"{subclass.__module__}.{subclass.__qualname__}"
        _agent_loop_registry[agent_name] = {"_target_": fqdn}
        return subclass
    return decorator
```

`@register("carr_tool_agent")` 会把 `"carr_tool_agent"` 映射到类的全限定名（如 `"examples.carr_deepsearch.tools.carr_agent_loop.CaRRToolAgentLoop"`），存储在全局 registry 中。

### 4.5.2 如何触发注册

模块必须被 import 才能执行 `@register` 装饰器。通过环境变量触发：

```bash
export VERL_USE_EXTERNAL_MODULES=examples.carr_deepsearch.tools.carr_agent_loop
```

`verl/__init__.py` 在启动时会 `import` 这些模块，从而触发装饰器执行。

### 4.5.3 运行时查找

当 `AgentLoopWorker` 处理一个 `agent_name="carr_tool_agent"` 的样本时：

```python
# agent_loop.py 中的 AgentLoopWorker
agent_loop_cls = _agent_loop_registry["carr_tool_agent"]  # {"_target_": "examples.carr_deepsearch..."}
agent_loop = hydra.utils.instantiate(agent_loop_cls, ...)  # 实例化
output = await agent_loop.run(sampling_params, **kwargs)
```

---

## 4.6 关键注意事项

### 4.6.1 不能访问 agent_data 作为实例变量

`agent_data` 是 `run()` 方法中的**局部变量**，不是 `self` 的属性。如果在 `run()` 返回后尝试访问 `self.agent_data`，会找不到。

解决方案：在 `run()` 内部通过参数传递，或使用实例变量（如 `self._reward_history`）在 `run()` 的作用域内维护。

### 4.6.2 response_text 的解码时机

在 `_handle_generating_state` 中，`agent_data.response_ids` 是当前轮次生成的 token。需要在状态转换前解码：

```python
response_text = self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)
```

注意 `skip_special_tokens=True` 会去掉特殊 token（如 EOS），但保留工具调用标签（因为它们不是"特殊 token"，而是普通文本 token）。ToolParser 在 `response_ids` 层面提取 tool_calls，所以 `response_text` 可能包含 `<tool_call>...</tool_call>` 标签的文本形式。

### 4.6.3 工具响应截断

基类在 `_call_tool` 返回后会截断工具响应文本（`tool_agent_loop.py:448-457`）：

```python
if tool_response_text and len(tool_response_text) > self.max_tool_response_length:
    if self.tool_response_truncate_side == "left":
        tool_response_text = tool_response_text[:self.max_tool_response_length] + "...(truncated)"
    ...
```

我们的 reward_history 应该记录**截断前**的原始工具响应文本（因为 CaRR reward server 需要完整引用内容来验证），还是截断后的？

建议记录**截断后**的，因为：
- 这是 LLM 实际"看到"的内容
- reward server 评估的应该是 Agent 可用信息范围内的表现

---

## 4.7 完整的一条轨迹示例

```
reward_history 构建过程：

[1] 初始化：
    [{"role": "user", "content": "What is the title of the paper..."}]

[2] LLM 第1次生成（带 tool_call）：
    [...,
     {"role": "assistant", "content": "<think>需要搜索...</think>",
      "tool_calls": [{"tool_call_id": "req_001_tc_0_0", "name": "browser.search",
                       "arguments": "{\"query\":\"breakpoint graph 2017\",\"num\":10}"}]}
    ]

[3] 工具执行（search 结果）：
    [...,
     {"role": "tool", "content": [
       {"tool_call_id": "req_001_tc_0_0", "output": "[0] Title: Can a Breakpoint..."}
     ]}
    ]

[4] LLM 第2次生成（打开网页）：
    [...,
     {"role": "assistant", "content": "<think>找到了相关结果...</think>",
      "tool_calls": [{"tool_call_id": "req_001_tc_1_0", "name": "browser.open",
                       "arguments": "{\"id\":\"0\"}"}]}
    ]

[5] 工具执行（open 结果）：
    [...,
     {"role": "tool", "content": [
       {"tool_call_id": "req_001_tc_1_0", "output": "Title: Can a Breakpoint Graph..."}
     ]}
    ]

... (更多轮次)

[最终] LLM 给出答案：
    [...,
     {"role": "assistant", "content": "## Explanation with Citations\n...\n## Exact Answer\nCan a Breakpoint Graph Be Decomposed into None Other Than 2-Cycles?"}
    ]

→ task_unfinished = False  (最后是 assistant)
→ 传给 CaRR reward server 评估
```

---

## 4.8 小结

| 方面 | 说明 |
|------|------|
| **为什么自定义** | CaRR reward server 需要特定格式的 history；需要管理 session 生命周期 |
| **核心数据结构** | `reward_history`（CaRR 格式的消息列表）|
| **覆盖方法** | `run()`（try/finally + extra_fields）、`_handle_generating_state()`、`_handle_processing_tools_state()` |
| **注册方式** | `@register("carr_tool_agent")` + `VERL_USE_EXTERNAL_MODULES` |
| **输出** | `extra_fields["messages"]` = reward_history, `extra_fields["task_unfinished"]` = bool |
| **数据流** | extra_fields → tool_extra_fields → NaiveRewardManager.extra_info → compute_score |

---

## 下一部分预告

Part 5 将详细介绍**奖励系统**：
- CaRR 三步 rubric reward 计算的完整流程
- Reward Server 的 `/evaluate` API 格式
- `carr_reward.py` 的实现
- streaming reward 路径的前提条件

有任何关于 Part 4 的问题，请随时提出！
