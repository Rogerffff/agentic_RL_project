# 两个系统的架构对比

本文档逐层对比 Agent-R1 MultihopQA 系统与 verl 原生 ToolAgentLoop 系统的等价组件，帮助你快速理解两个系统之间的映射关系。

---

## 1. 核心组件对照表

| 功能层级 | Agent-R1 | verl ToolAgentLoop | 差异要点 |
|---------|----------|-------------------|---------|
| **入口点** | `python3 -m agent_r1.src.main_agent` | `python3 -m verl.trainer.main_ppo` | 不同的入口，但底层都用 verl 的 DataProto |
| **训练器** | `RayAgentTrainer` (agent_ray_trainer.py) | `RayPPOTrainer` (ray_trainer.py) | Agent-R1 的训练器是 verl 训练器的扩展版 |
| **多轮循环** | `ToolGenerationManager.run_llm_loop()` | `ToolAgentLoop.run()` | 同步 for 循环 vs 异步状态机 |
| **工具接口** | `BaseTool.execute(args) → dict` | `BaseTool.execute(id, params) → (ToolResponse, reward, metrics)` | verl 多了 create/release 生命周期 + step reward |
| **环境/解析** | `BaseToolEnv` 统一 step/extract/format | `ToolParser` (独立解析) + `_handle_processing_tools_state` (执行) | verl 将解析和执行**分离** |
| **工具调用格式** | `<tool_call>JSON</tool_call>` | `<tool_call>JSON</tool_call>` (HermesToolParser) | **完全兼容** |
| **工具响应格式** | 手动拼接 `<tool_response>` + `<\|im_start\|>user` | `role="tool"` message → `apply_chat_template` | verl 用 chat template 自动生成 |
| **Action/Response Mask** | `_create_response_action_mask()` 手动拼接 | `AgentData.response_mask` 增量更新 | 等价语义，不同实现方式 |
| **推理后端** | `actor_rollout_wg.generate_sequences()` (同步 Ray call) | `AsyncLLMServerManager.generate()` (异步 + 负载均衡) | verl 异步 + multi-server |
| **奖励计算** | `AgentRewardManager` → 放在最后 token | `RewardLoopWorker` (streaming) 或 `custom_reward_function` | verl 支持 streaming reward |
| **数据集** | `ToolRLDataset` (自己处理 tool template) | `RLHFDataset` + `agent_name` / `tools_kwargs` in extra_info | verl 通过 extra_info 传递工具配置 |
| **配置格式** | `agent_trainer.yaml` + `tool.*` | `ppo_trainer.yaml` + `multi_turn.*` + `agent.*` | 字段名不同，语义等价 |

---

## 2. 逐层详细对比

### 2.1 多轮循环架构

#### Agent-R1: 同步批处理 for 循环

```python
# ToolGenerationManager.run_llm_loop()
for turn in range(max_turns):
    if not active_mask.sum(): break

    # 筛选活跃样本（batch 维度操作）
    rollings_active = rollings[active_mask]

    # 同步调用 Ray WorkerGroup 生成
    gen_output = actor_rollout_wg.generate_sequences(rollings_active)

    # 批量执行工具
    tool_responses, _, new_active = env.batch_step(raw_responses)

    # 更新状态
    rollings = _update_rolling_state(rollings, responses_ids, tool_responses)
```

**特点**：
- 整个 batch 统一推进，活跃样本和非活跃样本在同一个循环中
- 如果某些样本提前终止，需要 `_example_level_pad` 做填充
- 工具调用是批量同步执行

#### verl: 异步状态机 + 独立并行

```python
# AgentLoopWorker.generate_sequences()
tasks = []
for i in range(len(batch)):
    tasks.append(asyncio.create_task(
        self._run_agent_loop(sampling_params, **kwargs[i])
    ))
outputs = await asyncio.gather(*tasks)  # 每个样本独立异步执行

# ToolAgentLoop.run() — 每个样本独立的状态机
state = AgentState.PENDING
while state != AgentState.TERMINATED:
    if state == AgentState.PENDING:
        state = await self._handle_pending_state(...)
    elif state == AgentState.GENERATING:
        state = await self._handle_generating_state(...)
    elif state == AgentState.PROCESSING_TOOLS:
        state = await self._handle_processing_tools_state(...)
```

**特点**：
- 每个样本独立运行，不需要 batch 对齐
- 样本可以在不同轮数终止，无需填充
- 工具调用是独立异步执行
- 支持多 Server 负载均衡和 prefix caching

### 2.2 工具接口对比

#### Agent-R1 BaseTool

```python
class BaseTool(ABC):
    name: str = ''
    description: str = ''
    parameters: dict = {}

    def execute(self, args: Dict) -> Dict[str, Any]:
        """同步执行"""
        return {"content": "result", "success": True}

    def batch_execute(self, args_list) -> List[Dict]:
        """批量执行"""
        return [self.execute(args) for args in args_list]
```

#### verl BaseTool

```python
class BaseTool:
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        self.config = config
        self.name = tool_schema.function.name

    async def create(self, instance_id=None, **kwargs):
        """创建实例"""
        return str(uuid4()), ToolResponse()

    async def execute(self, instance_id, parameters, **kwargs):
        """异步执行"""
        return ToolResponse(text="result"), 0.0, {}

    async def calc_reward(self, instance_id, **kwargs):
        """计算奖励"""
        return 0.0

    async def release(self, instance_id, **kwargs):
        """释放实例"""
        pass
```

**差异对照**：

| 方面 | Agent-R1 | verl |
|------|----------|------|
| 初始化 | 类属性 (name, description, parameters) | 构造函数 (config, tool_schema) |
| 工具描述格式 | `tool_description` property → dict | `OpenAIFunctionToolSchema` Pydantic model |
| 执行方式 | 同步 `execute(args)` | 异步 `execute(instance_id, parameters)` |
| 返回格式 | `{"content": str, "success": bool}` | `(ToolResponse, float, dict)` |
| 批量执行 | `batch_execute()` | 无（通过异步并行替代） |
| 生命周期 | 无 | create → execute → release |
| 参数校验 | `validate_args()` 内置 | 无内置校验 |
| Step Reward | 无 | execute 返回 float reward |

### 2.3 工具调用解析

#### Agent-R1: NousToolEnv 统一处理

```python
class NousToolEnv(BaseToolEnv):
    def extract_tool_calls(self, raw_response: str) -> List[Any]:
        pattern = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
        for match in re.findall(pattern, raw_response):
            tool_call = json.loads(match)  # → {"name": str, "arguments": dict}

    def step(self, raw_response):
        tool_calls = self.extract_tool_calls(raw_response)
        # 直接在 step 中执行工具
        tool_result = tool.execute(tool_call["arguments"])
        # 格式化响应
        return self.format_tool_response(tool_responses), successes, active
```

**特点**：extract + execute + format 在同一个 `step()` 中完成。

#### verl: ToolParser + 状态处理器分离

```python
# 解析阶段 (在 _handle_generating_state 中)
_, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(
    agent_data.response_ids
)
# tool_calls 是 [FunctionCall(name="search", arguments='{"query":"..."}')]

# 执行阶段 (在 _handle_processing_tools_state 中)
for tool_call in agent_data.tool_calls:
    response = await self._call_tool(tool_call, ...)
    # _call_tool 内部: create → execute → release

# 格式化阶段 (也在 _handle_processing_tools_state 中)
response_ids = await self.apply_chat_template(
    [{"role": "tool", "content": response.text}],
    remove_system_prompt=True,
)
```

**特点**：解析和执行在不同的状态处理器中完成，通过 `AgentData.tool_calls` 传递。

### 2.4 工具响应格式

#### Agent-R1: 手动拼接特殊 token

```python
def format_tool_response(self, tool_responses):
    tool_message = "<|im_end|>\n<|im_start|>user\n"
    for tool_response in tool_responses:
        tool_message += f"<tool_response>\n{tool_response}\n</tool_response>"
    tool_message += "<|im_end|>\n<|im_start|>assistant\n<think>\n"
    return tool_message
```

生成的格式：
```
<|im_end|>
<|im_start|>user
<tool_response>
{搜索结果}
</tool_response><|im_end|>
<|im_start|>assistant
<think>
```

#### verl: 使用 chat template 自动生成

```python
add_messages = [{"role": "tool", "content": tool_response.text}]
response_ids = await self.apply_chat_template(
    add_messages,
    remove_system_prompt=True,
)
```

Qwen2.5-Instruct 的 chat template 会自动生成类似的 token 序列，但具体格式取决于 tokenizer 的 chat_template 设置。

**关键区别**：
- Agent-R1 将工具响应包装为 `role="user"` 消息（带 `<tool_response>` 标签）
- verl 使用 `role="tool"` 消息，由 chat template 决定具体格式
- Agent-R1 手动添加了 `<think>` 引导标记
- verl 依赖模型自行决定是否开始思考

### 2.5 Mask 对比

#### Agent-R1: action_mask

```python
# 每轮生成后，构建当前轮的 mask
action_masks = []
for model_ids, tool_ids in zip(responses_ids_list, tool_responses_ids_list):
    action_mask = [1] * len(model_ids) + [0] * len(tool_ids)
    action_masks.append(action_mask)

# 追加到累积 mask
rollings.non_tensor_batch["action_mask"][i] = old_mask + new_mask

# 最终输出
final_output["action_mask"][i, :mask_len] = (
    tensor(action_mask[:mask_len]) * response_mask[i, :mask_len]
)
```

存储方式：`non_tensor_batch` 中的 numpy array of lists（变长），最终转为固定长度 tensor。

#### verl: response_mask

```python
# 在 _handle_generating_state 中
agent_data.response_mask += [1] * len(agent_data.response_ids)

# 在 _handle_processing_tools_state 中
agent_data.response_mask += [0] * len(response_ids)

# 最终输出
response_mask = response_mask_output["input_ids"] * response_output["attention_mask"]
```

存储方式：list[int]（增量追加），最终 padding 为固定长度 tensor。

**语义完全等价**：两者都是 1 表示 LLM 生成的 token（计算梯度），0 表示工具/环境的 token（不计算梯度）。

### 2.6 奖励计算

#### Agent-R1: AgentRewardManager

```python
class AgentRewardManager:
    def __call__(self, data: DataProto):
        for i in range(len(data)):
            sequences_str = tokenizer.decode(prompt + response, skip_special_tokens=False)
            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            score = self.compute_score(data_source, sequences_str, ground_truth, extra_info)
            reward_tensor[i, valid_response_length - 1] = score  # 最后一个 token
```

**特点**：
- 解码完整轨迹（含特殊 token）后计算奖励
- 奖励放在最后一个有效 response token 位置
- 通过 `data_source` 字段分发到不同的评分函数

#### verl: 两种方式

**方式 A — Streaming Reward (RewardLoopWorker)**：
```python
# 在 _agent_loop_postprocess 中
result = await reward_loop_worker.compute_score.remote(data)
output.reward_score = result["reward_score"]
# 后续 _postprocess 中放入 rm_scores
rm_scores[i, response_length - 1] = score
```

**方式 B — Custom Reward Function**：
```python
# 在 RayPPOTrainer 中
custom_reward_function.path = "path.to.reward_fn"
# 框架自动调用
```

**特点**：
- Streaming reward 可以在样本完成后立即计算，不需要等所有样本完成
- 奖励同样放在最后一个有效 token 位置
- 可以传递 extra_fields（如 tool_rewards, turn_scores）做更丰富的奖励计算

### 2.7 数据集对比

#### Agent-R1: ToolRLDataset

```python
# 数据格式 (hotpotqa.py)
{
    "data_source": "hotpotqa/hotpot_qa",
    "prompt": [{"role": "user", "content": question}],
    "reward_model": {"style": "rule", "ground_truth": answer},
    "ability": "multihop_qa",
}

# ToolRLDataset 在 __getitem__ 中:
if self.use_default_tool_template:
    raw_prompt = tokenizer.apply_chat_template(messages, tools=self.tools, ...)
# → 工具描述由 ToolRLDataset 在 tokenize 时注入
```

#### verl: RLHFDataset + extra_info

```python
# 数据格式 (gsm8k_tool_agent_loop.py)
{
    "data_source": "openai/gsm8k",
    "agent_name": "tool_agent",
    "prompt": [
        {"role": "system", "content": "You are a math expert..."},
        {"role": "user", "content": question},
    ],
    "reward_model": {"style": "rule", "ground_truth": solution},
    "extra_info": {
        "need_tools_kwargs": True,
        "tools_kwargs": {
            "tool_name": {"create_kwargs": {"ground_truth": solution}},
        },
    },
}

# RLHFDataset 只需 return_raw_chat=True
# 工具描述由 ToolAgentLoop._handle_pending_state 中的 apply_chat_template(tools=schemas) 注入
```

**差异对照**：

| 字段 | Agent-R1 | verl |
|------|----------|------|
| 工具描述注入时机 | Dataset tokenize 时 | AgentLoop 运行时 |
| System prompt | 不在数据中，由 tool template 自动生成 | 可以在 prompt 中显式指定 |
| 工具参数传递 | 不需要（工具由配置初始化） | `extra_info.tools_kwargs` |
| Agent 类型指定 | 不需要（只有一种 AgentLoop） | `agent_name: "tool_agent"` |
| 额外标记 | `ability` 字段 | `extra_info.index` 等 |

### 2.8 配置参数映射

| Agent-R1 配置 | verl 配置 | 含义 |
|--------------|----------|------|
| `tool.max_turns=5` | `multi_turn.max_assistant_turns=5` | 最大 LLM 生成轮数 |
| `tool.tools=['wiki_search']` | `multi_turn.tool_config_path=xxx.yaml` | 工具列表/配置 |
| `tool.max_tool_response_length=2048` | `multi_turn.max_tool_response_length=2048` | 工具响应截断长度 |
| N/A | `multi_turn.max_user_turns=N` | 最大 user/tool 轮数 |
| N/A | `multi_turn.max_parallel_calls=N` | 并行工具调用数 |
| N/A | `multi_turn.format=hermes` | 工具调用解析格式 |
| `data.use_default_tool_template=True` | `data.return_raw_chat=True` | chat template 处理方式 |
| `data.max_response_length_single_turn=1024` | N/A (由 Server sampling_params 控制) | 单轮生成限制 |
| `actor_rollout_ref.rollout.stop_token_ids=[151658]` | N/A (由 ToolParser 格式决定) | 停止 token |
| `actor_rollout_ref.rollout.n_repeat=5` | `actor_rollout_ref.rollout.n=5` | GRPO 重复采样数 |
| `algorithm.adv_estimator=grpo` | `algorithm.adv_estimator=grpo` | RL 算法（完全相同） |

---

## 3. 等价性验证清单

以下是验证迁移正确性的检查点：

| 检查项 | Agent-R1 行为 | verl 预期行为 | 验证方法 |
|--------|-------------|-------------|---------|
| 工具调用解析 | regex `<tool_call>(.*?)</tool_call>` | HermesToolParser 相同 regex | 比较解析结果 |
| 工具响应格式 | 手动 `<\|im_end\|>...<tool_response>...` | `apply_chat_template(role="tool")` | 比较 token IDs |
| Response mask | 1=LLM, 0=tool | 1=LLM, 0=tool | 比较最终 mask tensor |
| 奖励位置 | `reward_tensor[i, valid_response_length-1]` | `rm_scores[i, response_length-1]` | 检查 reward 位置 |
| 奖励范围 | [-1.0, 1.0] | [-1.0, 1.0] | 检查 score 统计 |
| 终止条件 | `env.stop()` = 无 tool_call | `tool_parser.extract_tool_calls` 为空 | 对比终止样本数 |
| 序列长度 | max_prompt=8192, max_response=8192 | prompt_length + response_length | 比较截断行为 |
