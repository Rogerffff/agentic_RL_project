# 关键差异和注意事项

本文档记录迁移 MultihopQA Agent 从 Agent-R1 到 verl ToolAgentLoop 时容易踩的坑和需要特别注意的地方。

---

## 1. 工具响应格式差异（最关键的差异）

### 问题描述

Agent-R1 的 `NousToolEnv.format_tool_response()` 手动拼接 Qwen 特殊 token：

```
<|im_end|>                      ← 结束 assistant 轮
<|im_start|>user                ← 用 "user" 角色包装工具响应
<tool_response>
{搜索结果}
</tool_response>
<|im_end|>                      ← 结束 user 轮
<|im_start|>assistant           ← 开始新 assistant 轮
<think>                         ← 引导思考
```

verl 的 ToolAgentLoop 使用 `role="tool"` 消息通过 `apply_chat_template` 生成：

```python
add_messages = [{"role": "tool", "content": tool_response_text}]
response_ids = await self.apply_chat_template(add_messages, remove_system_prompt=True)
```

### 影响

1. **奖励函数中的正则表达式**：`compute_score_format()` 使用 `<|im_start|>assistant\n(.*?)<|im_end|>` 提取 assistant blocks。如果 verl 的 chat template 对 tool role 生成了不同的 token 结构（例如 `<|im_start|>tool` 而不是 `<|im_start|>user`），中间轮次的结构会改变。

2. **`<think>` 引导标记缺失**：Agent-R1 在每个工具响应后手动添加 `<think>` 标签来引导模型继续思考。verl 的 chat template 不会添加这个标签。模型可能需要自行学会在工具响应后开始 `<think>` 块。

### 解决方案

**方案 A（推荐）**：先验证 Qwen2.5-Instruct 的 chat template 对 `role="tool"` 的处理方式。在 Python 中运行：

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

# 模拟一个完整的多轮对话
messages = [
    {"role": "user", "content": "Who directed Film Y?"},
    {"role": "assistant", "content": '<think>I need to search.</think>\n<tool_call>{"name":"search","arguments":{"query":"Film Y director"}}</tool_call>'},
    {"role": "tool", "content": '{"results":[{"content":"Film Y was directed by Jane Smith."}]}'},
]

text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
print(text)
```

检查输出中 tool role 的格式，然后相应调整 `compute_score_format()` 的正则。

**方案 B**：如果 Qwen chat template 不支持 `role="tool"` 或格式不理想，可以在 ToolAgentLoop 中自定义 tool 响应的格式化方式。参考 `gpt-oss` 模式中手动构建 tool response text 的做法（`tool_agent_loop.py` 第 349-354 行）。

---

## 2. 奖励函数中的轨迹解析

### 问题描述

Agent-R1 的 `AgentRewardManager` 解码完整轨迹时使用 `skip_special_tokens=False`：

```python
sequences_str = tokenizer.decode(sequences, skip_special_tokens=False)
```

这保留了 `<|im_start|>`, `<|im_end|>` 等特殊 token。奖励函数依赖这些 token 来提取 assistant blocks：

```python
assistant_blocks = re.findall(
    r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>',
    solution_str, re.DOTALL
)
```

### 潜在问题

1. verl 的 `custom_reward_function` 接收的 `DataProto` 中的 token IDs 可能经过不同的 padding 和截断处理。
2. 如果 tokenizer 的 decode 行为不同（例如 padding token 处理），解码结果可能有差异。

### 解决方案

在 reward function 中添加清理逻辑：

```python
sequences_str = tokenizer.decode(sequences, skip_special_tokens=False)
# 去除 padding token 的影响
pad_token = tokenizer.decode([tokenizer.pad_token_id])
sequences_str = sequences_str.replace(pad_token, "")
# 确保以 EOS 结尾
if not sequences_str.endswith(tokenizer.eos_token):
    sequences_str += tokenizer.eos_token
```

---

## 3. 异步 vs 同步的工具执行

### 问题描述

Agent-R1 的 `NousToolEnv.batch_step()` 是同步批处理，一次性发送所有搜索请求：

```python
# 收集所有合法的工具调用
for tool_name, args_list in success_tool_calls_arguments.items():
    batch_results = tool.batch_execute(args_list)  # 单次 POST 请求
```

verl 的 ToolAgentLoop 中每个样本独立异步执行工具调用：

```python
# 每个样本各自调用 _call_tool
tasks = [self._call_tool(tc, ...) for tc in tool_calls]
responses = await asyncio.gather(*tasks)
```

### 影响

1. **搜索 API 并发压力**：Agent-R1 的批量 API 调用合并为少数几个请求，而 verl 会产生大量并发单条请求。
2. **需要并发控制**：verl 的 WikiSearchTool 必须有 `TokenBucketWorker` 限流，否则可能压垮搜索服务。

### 解决方案

已在 WikiSearchTool 实现中添加了 `TokenBucketWorker` 限流。建议配置：

```yaml
config:
  num_workers: 60        # 最大并发执行线程数
  rate_limit: 60         # 全局最大并发数
```

同时确保 KILT 搜索服务端也能支持相应的并发量。如果搜索服务成为瓶颈，可以：
1. 部署多个搜索服务实例
2. 降低 rate_limit
3. 增加搜索服务的 worker 数

---

## 4. tool_parser 格式选择

### 问题描述

verl 支持多种工具调用格式：

| 格式 | Parser | 模型支持 |
|------|--------|---------|
| `hermes` | HermesToolParser | Qwen2.5-Instruct, Mistral |
| `gpt-oss` | GptOssToolParser | GPT-OSS 模型 |

### 推荐

使用 `format: hermes`，原因：

1. **与 Agent-R1 完全兼容**：NousToolEnv 解析的就是 `<tool_call>JSON</tool_call>` 格式，HermesToolParser 使用相同的正则表达式
2. **Qwen2.5-Instruct 原生支持**：Qwen2.5 的 chat template 会自动引导模型使用 `<tool_call>` 格式
3. **无需额外适配**：不需要改任何格式相关的代码

### 注意

如果使用 `format: qwen`（在 search_multiturn_grpo.yaml 中出现过），需要确认这个格式的 parser 行为。`hermes` 是经过验证的安全选择。

---

## 5. 数据字段差异

### 问题描述

Agent-R1 的数据只有 user message，工具描述由 `ToolRLDataset` 自动注入：

```python
# Agent-R1 数据:
{"prompt": [{"role": "user", "content": "Question: ..."}]}

# ToolRLDataset.__getitem__ 中:
tokenizer.apply_chat_template(messages, tools=self.tools, ...)
# → 自动在 system prompt 中注入工具描述
```

verl 的数据可以包含 system message，工具描述在 AgentLoop 运行时注入：

```python
# verl 数据:
{"prompt": [
    {"role": "system", "content": "..."},  # 可选
    {"role": "user", "content": "Question: ..."}
]}

# ToolAgentLoop._handle_pending_state 中:
apply_chat_template(messages, tools=self.tool_schemas, ...)
# → 工具描述也会注入
```

### 潜在问题

1. **双重 system prompt**：如果数据中已有 system message，且 `apply_chat_template` 又注入了工具描述的 system prompt，可能导致 system prompt 被覆盖或重复。

2. **system prompt 内容差异**：Agent-R1 通过 `tools` 参数自动生成的 system prompt 内容可能与手动写的不同。

### 解决方案

**推荐**：数据中不添加 system message，让 `apply_chat_template(tools=schemas)` 自动生成。这样最接近 Agent-R1 的行为。

如果需要添加额外的任务指令（如 "think step by step"），可以：
1. 把指令放在 user message 中（Agent-R1 就是这样做的——instruction_following 在 question 后面）
2. 或在 tool_schema 的 description 中包含更详细的指导

---

## 6. stop_token_ids 处理差异

### 问题描述

Agent-R1 显式配置 `stop_token_ids=[151658]`（`</tool_call>` 的 token ID），vLLM 在生成时遇到该 token 会停止。停止后由 `NousToolEnv` 处理后续逻辑。

verl 的 ToolAgentLoop 不需要配置 stop_token_ids，因为 SGLang/vLLM Server 会生成完整响应直到 EOS 或最大长度，然后由 `ToolParser` 在已生成的文本中提取工具调用。

### 影响

1. **生成效率**：Agent-R1 在遇到 `</tool_call>` 时立即停止，不浪费计算。verl 可能会在 `</tool_call>` 后继续生成一些 token 直到 EOS。

2. **生成质量**：如果模型在 `</tool_call>` 后继续生成文本，这些额外的 token 可能影响工具调用的解析。

### 解决方案

verl 的 SGLang rollout 支持在 sampling_params 中设置 stop tokens。可以在训练脚本中添加：

```bash
actor_rollout_ref.rollout.stop_token_ids=[151658]
```

或者在 ToolAgentLoop 中确保 ToolParser 能正确处理 `</tool_call>` 后面的额外文本（HermesToolParser 已经处理了，因为它用 regex 提取 `<tool_call>...</tool_call>` 之间的内容）。

---

## 7. n_repeat vs rollout.n

### 问题描述

Agent-R1 使用 `actor_rollout_ref.rollout.n_repeat=5` 表示每个 prompt 生成 5 个响应。verl 使用 `actor_rollout_ref.rollout.n=5` 表示相同含义。

### 注意

确保在 verl 脚本中使用 `rollout.n=5` 而不是 `rollout.n_repeat=5`，否则可能不生效或报错。

---

## 8. 工具参数格式差异

### 问题描述

Agent-R1 的 WikiSearchTool 接受 `{"query": "search string"}`（单个字符串）。

verl 已有的 SearchTool 接受 `{"query_list": ["query1", "query2"]}`（字符串列表）。

### 影响

如果复用 verl 的 SearchTool 而不是创建新的 WikiSearchTool，需要修改数据或模型提示来生成 `query_list` 格式的工具调用。这可能导致：
1. 模型需要学习新的参数格式
2. 与 Agent-R1 的行为不一致，难以对比基线

### 解决方案

创建新的 WikiSearchTool（保持 `{"query": str}` 格式），而不是复用 SearchTool。详见迁移指南第 1 节。

---

## 9. chat_template 的 `add_generation_prompt` 行为

### 问题描述

verl 的 `apply_chat_template` 在 `_handle_pending_state` 和 `_handle_processing_tools_state` 中都使用 `add_generation_prompt=True`。这会在 token 序列末尾添加 `<|im_start|>assistant\n`，引导模型开始生成 assistant 消息。

但在 Agent-R1 中，`format_tool_response` 手动添加了 `<|im_start|>assistant\n<think>\n`，多了一个 `<think>` 标签。

### 影响

verl 生成的 prompt 在工具响应后只有 `<|im_start|>assistant\n`，而 Agent-R1 有 `<|im_start|>assistant\n<think>\n`。

这意味着在 verl 中，模型需要自行学会在接收到工具响应后输出 `<think>` 标签。如果使用未经 SFT cold start 的模型，可能需要更多训练才能学会这个行为。

### 解决方案

1. **使用 SFT 模型作为起点**：如果有 Agent-R1 的 SFT cold start 模型（如 `russwest404/Qwen3-4B-ReTool-SFT`），模型已经学会了 `<think>` 格式。
2. **在 instruction 中明确要求**：在 user message 的 instruction_following 中已有 "The reasoning process MUST BE enclosed within `<think>` `</think>` tags"。
3. **修改 chat template**（高级）：自定义 Qwen 的 chat template，在 tool role 后自动添加 `<think>` 标签。

---

## 10. 验证模式差异

### 问题描述

Agent-R1 的 `ToolGenerationManager` 通过 `is_validation` 参数区分训练和验证模式。

verl 通过 `batch.meta_info.get("validate", False)` 判断：

```python
if batch.meta_info.get("validate", False):
    sampling_params["top_p"] = config.val_kwargs.top_p
    sampling_params["temperature"] = config.val_kwargs.temperature
```

### 注意

确保训练配置中设置了合理的验证参数：

```bash
actor_rollout_ref.rollout.val_kwargs.top_p=0.95
actor_rollout_ref.rollout.val_kwargs.top_k=-1
actor_rollout_ref.rollout.val_kwargs.temperature=0.1
```

---

## 11. 调试建议

### 第一步：验证工具调用链路

在正式训练前，先用一个简单的测试验证整个工具调用链路：

```python
# 测试 WikiSearchTool
import asyncio
from verl.tools.wiki_search_tool import WikiSearchTool
from verl.tools.schemas import OpenAIFunctionToolSchema

schema = OpenAIFunctionToolSchema.model_validate({
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search Wikipedia",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
})

tool = WikiSearchTool(
    config={"api_url": "http://localhost:8000", "num_workers": 1, "rate_limit": 1},
    tool_schema=schema,
)

async def test():
    instance_id, _ = await tool.create()
    response, reward, metrics = await tool.execute(
        instance_id, {"query": "capital of France"}
    )
    print("Response:", response.text[:200])
    await tool.release(instance_id)

asyncio.run(test())
```

### 第二步：验证解码后的轨迹格式

在训练的前几个 step 中，打印解码后的轨迹文本，确认格式与奖励函数的正则匹配：

```python
# 在 reward function 中临时添加
for i in range(min(3, len(data))):
    # ... 解码 ...
    print(f"=== Sample {i} ===")
    print(sequences_str[:500])
    print(f"Format score: {compute_score_format(sequences_str)}")
    print(f"Answer score: {compute_score_answer(sequences_str, ground_truth)}")
```

### 第三步：检查 response_mask

确认 response_mask 的 1/0 分布合理：

```python
# 在训练循环中添加
mask = data.batch["response_mask"]  # verl 中叫 response_mask
print(f"Mask ones ratio: {mask.sum() / mask.numel():.3f}")
print(f"Mask shape: {mask.shape}")
```

正常情况下，ones ratio 应该在 0.3-0.7 之间（取决于工具响应的长度占比）。
如果接近 1.0，说明工具响应部分没有被正确 mask 掉。
如果接近 0.0，说明可能有 bug。

---

## 12. 性能预期

### 训练速度

- Agent-R1（同步 batch）：每个 step 的时间取决于最慢的样本
- verl（异步并行）：每个 worker 内部样本独立运行，整体吞吐量通常更高
- 搜索 API 延迟（约 50-200ms/query）是主要瓶颈

### 显存需求

| 模型 | 训练 (FSDP) | 推理 (TP=2) | 总计 (8x GPU) |
|------|------------|------------|-------------|
| Qwen2.5-7B-Instruct | ~14GB/GPU | ~7GB/GPU | 约 21GB/GPU |

### 收敛速度

- Agent-R1 论文中 Qwen2.5-1.5B 在 HotpotQA 上约 200 steps 开始收敛
- Qwen2.5-7B 预期更快收敛，但每 step 更慢
- 建议先用小数据集（1000 条）验证流程，再全量训练
