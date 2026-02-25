# Agent-R1 MultihopQA 系统架构详解

本文档详细记录 Agent-R1 中 MultihopQA（多跳问答）Agent 的完整架构、核心文件职责、数据流、以及关键实现细节。阅读本文档后，你可以理解该系统的所有核心组件及其交互方式，无需重新阅读源代码。

---

## 1. 系统总览

Agent-R1 的 MultihopQA Agent 使用端到端强化学习训练 LLM 与 Wikipedia 搜索工具交互，回答需要跨多个文档推理的复杂问题。

**核心思想**：LLM 在推理过程中可以调用搜索工具获取信息，将搜索结果融入思考链，最终给出答案。训练通过 GRPO（Group Relative Policy Optimization）算法，以答案正确性和格式合规性作为奖励信号。

### 典型交互轨迹示例

```
[User] Question: Who directed the film that starred the actor born in 1990 who appeared in "Movie X"?

[Assistant] <think>我需要分步查找信息。首先找出 Movie X 中 1990 年出生的演员。</think>
<tool_call>{"name": "search", "arguments": {"query": "Movie X cast 1990"}}</tool_call>

[Tool Response] <tool_response>
{"results": [{"content": "John Doe, born 1990, starred in Movie X and Film Y..."}]}
</tool_response>

[Assistant] <think>John Doe 是 1990 年出生的演员，他还出演了 Film Y。现在查找 Film Y 的导演。</think>
<tool_call>{"name": "search", "arguments": {"query": "Film Y director"}}</tool_call>

[Tool Response] <tool_response>
{"results": [{"content": "Film Y was directed by Jane Smith..."}]}
</tool_response>

[Assistant] <think>根据搜索结果，Film Y 的导演是 Jane Smith。</think>
<answer>Jane Smith</answer>
```

---

## 2. 核心文件及其职责

### 2.1 工具基类 — `agent_r1/tool/base.py`

定义了工具系统的两个核心抽象：

#### BaseTool（工具执行器）

```python
class BaseTool(ABC):
    name: str = ''              # 工具名称，用于 function calling
    description: str = ''       # 工具描述
    parameters: dict = {}       # JSON Schema 定义的参数格式

    def execute(self, args: Dict, **kwargs) -> Dict[str, Any]:
        """执行工具调用，返回 {"content": str, "success": bool}"""
        pass

    def batch_execute(self, args_list: List[Dict], **kwargs) -> List[Dict[str, Any]]:
        """批量执行，默认循环调用 execute"""
        return [self.execute(args, **kwargs) for args in args_list]

    @property
    def tool_description(self) -> Dict:
        """返回 OpenAI function calling 格式的工具描述"""
        return {"type": "function", "function": {"name": self.name, ...}}

    def validate_args(self, args: Dict) -> bool:
        """用 jsonschema 验证参数合法性"""
```

**关键点**：
- `execute()` 是同步方法，返回简单的 dict
- `tool_description` 属性用于注入 chat template 的 tools 参数
- `validate_args()` 在执行前做参数校验

#### BaseToolEnv（环境编排器 / 状态转移函数）

```python
class BaseToolEnv(ABC):
    def step(self, raw_response: str) -> Tuple[str, List[bool], bool]:
        """
        核心状态转移函数
        输入: LLM 的原始响应文本
        输出: (tool_response, success_list, active)
            - tool_response: 格式化后的工具响应字符串
            - success_list: 每个工具调用是否成功
            - active: 该轨迹是否仍然活跃（是否继续生成）
        """

    def batch_step(self, raw_responses: List[str]) -> Tuple[...]:
        """批量执行 step"""

    def process_responses_ids(self, tokenizer, raw_responses_ids) -> torch.Tensor:
        """预处理 response token ids（大多数情况下直接返回原值）"""

    def stop(self, raw_response: str) -> bool:
        """判断是否应该停止生成"""

    def extract_tool_calls(self, raw_response: str) -> List[Any]:
        """从响应中提取工具调用参数"""

    def format_tool_response(self, tool_response: str) -> str:
        """格式化工具响应文本"""
```

**关键点**：
- `step()` 是整个状态转移函数的入口，编排了 extract → execute → format 的完整流程
- `stop()` 的返回值决定该样本是否终止：`True` = 停止（没有更多工具调用），`False` = 继续
- `BaseToolEnv` 同时负责解析和执行，verl 将这两者分离了

---

### 2.2 WikiSearch 搜索工具 — `agent_r1/tool/tools/wiki_search_tool.py`

```python
class WikiSearchTool(BaseTool):
    name = "search"
    description = "Search for information on the internet using Wikipedia as a knowledge source."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"}
        },
        "required": ["query"]
    }
```

**初始化**：
- 从环境变量 `WIKI_SEARCH_API_URL` 读取搜索 API 地址（默认 `http://localhost:8000`）
- 启动时执行健康检查 `GET /health`
- 禁用 HTTP 代理以避免网络问题

**单条执行 `execute(args)`**：
```python
def execute(self, args: Dict) -> Dict[str, Any]:
    query = args.get("query", "").strip()
    limit = args.get("limit", 5)  # 默认返回 5 条结果
    response = requests.get(f"{self.api_url}/search", params={"query": query, "top_k": limit})
    # 返回格式: {"content": json_str, "success": True/False}
```

**批量执行 `batch_execute(args_list)`**（重要优化）：
```python
def batch_execute(self, args_list: List[Dict]) -> List[Dict[str, Any]]:
    queries = [args.get("query", "").strip() for args in args_list]
    # 使用 POST 批量接口
    response = requests.post(f"{self.api_url}/search", json={"queries": queries, "top_k": max_limit})
```

**返回格式**：
```json
{"results": [{"content": "文档正文...", "title": "文档标题"}, ...]}
```

**关键点**：
- 单条用 `GET /search`，批量用 `POST /search`
- 搜索服务是外部 KILT 检索服务（基于 FAISS 索引 + Wikipedia 语料库）
- `_format_results()` 只保留 `content` 和 `title`，去除分数等元数据

---

### 2.3 Nous 工具环境 — `agent_r1/tool/envs/nous.py`

这是 MultihopQA 任务的核心环境类，定义了工具调用的解析、执行和格式化逻辑。

```python
class NousToolEnv(BaseToolEnv):
    def __init__(self, tools: List[BaseTool], max_tool_response_length: int):
        self.tools = tools
        self.tool_map = {tool.name: tool for tool in self.tools}
        self.tool_call_start = "<tool_call>"
        self.tool_call_end = "</tool_call>"
        self.tool_response_start = "<tool_response>"
        self.tool_response_end = "</tool_response>"
        self.eos_token = "<|im_end|>"
        self.parallel_tool_calls = False    # 不支持并行工具调用
        self.max_tool_response_length = max_tool_response_length
```

#### extract_tool_calls — 提取工具调用

```python
def extract_tool_calls(self, raw_response: str) -> List[Any]:
    pattern = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
    for tool_call in re.findall(pattern, raw_response):
        try:
            tool_call = json.loads(tool_call)  # 解析 JSON
            tool_calls.append(tool_call)
        except json.JSONDecodeError:
            tool_calls.append(None)  # JSON 解析失败标记为 None
    return tool_calls
```

**解析的格式**：
```
<tool_call>{"name": "search", "arguments": {"query": "some query"}}</tool_call>
```

#### format_tool_response — 格式化工具响应

```python
def format_tool_response(self, tool_responses: List[str]) -> str:
    tool_message = "<|im_end|>\n<|im_start|>user\n"
    for tool_response in tool_responses:
        if len(tool_response) > self.max_tool_response_length:
            tool_response = tool_response[:self.max_tool_response_length] + "..."
        tool_message += f"<tool_response>\n{tool_response}\n</tool_response>"
    tool_message += "<|im_end|>\n<|im_start|>assistant\n<think>\n"
    return tool_message
```

**极其重要** — 这个格式化包含了 Qwen chat template 的特殊 token：

```
<|im_end|>               ← 结束当前 assistant 轮
<|im_start|>user         ← 开始一个新的 user 轮（模拟工具响应作为用户消息）
<tool_response>
{搜索结果}
</tool_response>
<|im_end|>               ← 结束 user 轮
<|im_start|>assistant    ← 开始新的 assistant 轮
<think>                  ← 引导模型继续思考
```

**关键点**：
- 工具响应被包装成了一个伪 user 消息，而不是 role="tool" 消息
- 手动拼接了 `<|im_start|>/<|im_end|>` 等 Qwen 特殊 token
- 末尾的 `<think>` 标签引导模型继续进行推理

#### step — 单步状态转移

```python
def step(self, raw_response: str) -> Tuple[str, List[bool], bool]:
    tool_calls = self.extract_tool_calls(raw_response)
    if len(tool_calls) == 0:
        return "", [], False  # 无工具调用 → 轨迹不活跃

    if not self.parallel_tool_calls:
        tool_calls = [tool_calls[0]]  # 只取第一个工具调用

    for tool_call in tool_calls:
        if tool_call is None:  # JSON 解析失败
            → "Error: JSONDecodeError"
        elif "name" not in tool_call:
            → "Error: No tool name"
        elif tool_name not in self.tool_map:
            → "Error: ToolNotFoundError"
        elif not tool.validate_args(tool_call["arguments"]):
            → "Error: Invalid tool arguments"
        else:
            tool_result = tool.execute(tool_call["arguments"])
            → tool_result["content"]

    tool_response = self.format_tool_response(tool_responses)
    return tool_response, tool_successes, True  # True = 轨迹继续活跃
```

#### batch_step — 批量状态转移（训练时使用）

`batch_step` 比简单循环更高效：
1. 遍历所有响应，提取工具调用参数
2. 按工具名分组，收集所有合法的参数
3. 调用 `tool.batch_execute(args_list)` 批量执行
4. 将结果回填到对应位置
5. 调用 `format_tool_response` 格式化

这样 WikiSearchTool 的 `batch_execute` 可以发送一个批量 HTTP 请求，而不是逐条请求。

#### stop — 停止判断

```python
def stop(self, raw_response: str) -> bool:
    tool_calls = self.extract_tool_calls(raw_response)
    return len(tool_calls) == 0  # 没有工具调用 → 停止
```

---

### 2.4 多轮生成管理器 — `agent_r1/llm_agent/generation.py`

**这是 Agent-R1 最核心的文件**，849 行代码，实现了 LLM 与工具之间的多轮交互循环。

#### ToolGenerationConfig 配置

```python
@dataclass
class ToolGenerationConfig:
    max_turns: int                      # 最大交互轮数（如 5）
    max_prompt_length: int              # 最大 prompt 长度
    max_response_length: int            # 所有轮次的总响应长度
    max_response_length_single_turn: int  # 单轮最大响应长度
    use_batch_tool_calls: bool = False  # 是否批量执行工具
```

#### ToolGenerationManager 类

初始化参数：
- `tokenizer`: HuggingFace tokenizer
- `processor`: 多模态处理器（MultihopQA 不需要）
- `actor_rollout_wg`: Actor/Rollout Worker 组（Ray WorkerGroup）
- `config`: ToolGenerationConfig
- `is_validation`: 是否为验证模式

#### run_llm_loop — 核心多轮循环

这是整个生成阶段的入口方法。完整流程如下：

```
输入: gen_batch (DataProto, 包含 prompt) + env (NousToolEnv)

初始化:
  batch_size = gen_batch.batch["input_ids"].shape[0]
  active_mask = [True] * batch_size      # 所有样本初始活跃
  turns = [0] * batch_size               # 轮数计数器
  rollings = gen_batch                   # 滚动状态

for turn in range(max_turns):
    if active_mask.sum() == 0: break

    # 步骤 1: 检查序列长度
    effective_len = rollings.batch["attention_mask"].sum(dim=1)
    length_exceeded = effective_len > max_prompt_length
    active_mask[length_exceeded] = False

    # 步骤 2: 筛选活跃样本（节省计算）
    rollings_active = rollings[active_mask]

    # 步骤 3: 分布式生成
    rollings_active = pad_to_divisor(rollings_active, world_size)
    gen_output = actor_rollout_wg.generate_sequences(rollings_active)
    gen_output = unpad(gen_output)

    # 步骤 4: 执行工具
    responses_ids = gen_output.batch["responses"]
    raw_responses = tokenizer.batch_decode(responses_ids)
    if use_batch_tool_calls:
        tool_responses, _, new_active_masks = env.batch_step(raw_responses)
    else:
        for raw_response in raw_responses:
            tool_response, _, active = env.step(raw_response)

    # 步骤 5: 更新 active_mask
    responses_ids = _example_level_pad(responses_ids, active_mask)
    tool_responses = _example_level_pad(tool_responses, active_mask)
    active_mask[active_mask.clone()] = tensor(new_active_masks)
    turns[active_mask] += 1

    # 步骤 6: 更新滚动状态
    rollings = _update_rolling_state(rollings, responses_ids, tool_responses, ...)

输出: final_output (DataProto)
  - prompts: 原始 prompt
  - responses: 完整响应（包含所有轮的 LLM 生成 + 工具响应）
  - action_mask: 1=LLM 生成的 token, 0=工具响应/环境反馈的 token
  - turns: 每个样本的交互轮数
```

#### _create_response_action_mask — 构建 Action Mask

```python
def _create_response_action_mask(self, responses_ids_list, tool_responses_ids_list):
    for model_ids, tool_ids in zip(responses_ids_list, tool_responses_ids_list):
        action_mask = [1] * len(model_ids) + [0] * len(tool_ids)
        #              ↑ LLM 生成的 token     ↑ 工具响应的 token
```

**Action Mask 示意**：
```
Turn 1 LLM:    [1, 1, 1, 1, 1, 1, 1, 1, 1]    ← 需要计算 RL 梯度
Tool Response:  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  ← 不计算梯度
Turn 2 LLM:    [1, 1, 1, 1, 1, 1]               ← 需要计算 RL 梯度
Padding:        [0, 0, 0, 0]                     ← 不计算
```

#### _update_rolling_state — 状态累积更新

每轮生成后，将新生成的响应和工具响应追加到序列中：

```
Turn N-1 结束时:
  input_ids = [Prompt | Resp_1 | ToolResp_1 | ... | Resp_{N-1} | ToolResp_{N-1}]

Turn N 后:
  input_ids = [Prompt | ... | Resp_{N-1} | ToolResp_{N-1} | Resp_N | ToolResp_N]
  action_mask = [... | 1...1 | 0...0 | 1...1 | 0...0]
```

更新步骤：
1. 将工具响应文本 tokenize 为 `tool_responses_ids`
2. 将 `responses_ids` 和 `tool_responses_ids` 拼接到 `rollings.batch["responses"]`
3. 构建该轮的 action_mask 并追加到 `rollings.non_tensor_batch["action_mask"]`
4. 更新 `input_ids` = 拼接所有历史
5. 重新计算 `attention_mask` 和 `position_ids`
6. 更新 `raw_prompt_ids`（用于下一轮生成的 prompt）

#### _example_level_pad — 处理不同结束时间

当 batch 中部分样本已终止时，需要将活跃样本的数据填充回完整 batch：

```python
# 例如: active_mask = [True, False, True, False]
# 活跃样本的 data = ["response1", "response3"]
# 填充后 = ["response1", "", "response3", ""]
```

#### 最终输出组装

```python
final_output["prompts"] = prompts                    # (batch_size, prompt_len)
final_output["responses"] = rollings.batch["responses"]  # (batch_size, response_len)
final_output["input_ids"] = cat(prompts, responses)  # (batch_size, total_len)
final_output["attention_mask"] = ...                 # (batch_size, total_len)
final_output["position_ids"] = ...                   # (batch_size, total_len)
final_output["action_mask"] = ...                    # (batch_size, response_len)
final_output["turns"] = turns                        # (batch_size,)
```

**action_mask 最终处理**：
```python
# 初始化为 response 的 attention_mask（有效 token 为 1，padding 为 0）
final_output["action_mask"] = response_mask.clone()
# 将工具响应位置设为 0
for i, action_mask in enumerate(rollings.non_tensor_batch["action_mask"]):
    mask_len = min(len(action_mask), response_mask.shape[1])
    final_output["action_mask"][i, :mask_len] = (
        tensor(action_mask[:mask_len]) * response_mask[i, :mask_len]
    )
```

---

### 2.5 奖励管理器 — `agent_r1/src/agent_reward_manager.py`

```python
class AgentRewardManager:
    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source"):
        self.compute_score = compute_score or _default_compute_score
```

**奖励计算流程**：

```python
def __call__(self, data: DataProto, return_dict=False):
    reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)

    for i in range(len(data)):
        # 1. 解码完整轨迹
        prompt_ids = data_item.batch["prompts"]
        response_ids = data_item.batch["responses"]
        sequences = cat(valid_prompt_ids, valid_response_ids)
        sequences_str = tokenizer.decode(sequences, skip_special_tokens=False)

        # 2. 获取 ground truth
        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        data_source = data_item.non_tensor_batch["data_source"]

        # 3. 计算奖励分数
        score = self.compute_score(
            data_source=data_source,
            solution_str=sequences_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )

        # 4. 将奖励放在最后一个有效 token 位置
        reward_tensor[i, valid_response_length - 1] = reward
```

**关键点**：
- 奖励只放在 **最后一个有效 response token** 的位置（稀疏奖励）
- `sequences_str` 包含完整的 prompt + response，包括特殊 token（`skip_special_tokens=False`）
- 奖励函数通过 `data_source` 字段分发到不同的计算函数

---

### 2.6 奖励评分函数 — `agent_r1/src/reward_score/qa_em_and_format.py`

MultihopQA 使用 `compute_score_format_answer` 作为奖励函数，由两部分组成：

#### 格式奖励 `compute_score_format(solution_str)`

检查模型输出是否遵循 `<think>...<tool_call>...<answer>` 的结构化格式。

```python
def compute_score_format(solution_str):
    # 提取所有 assistant blocks
    assistant_blocks = re.findall(
        r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>', solution_str, re.DOTALL
    )

    format_reward = 0.0

    # 检查非最后一个 block: 是否有 <think>...</think><tool_call>...</tool_call>
    for i, block in enumerate(assistant_blocks[:-1]):
        if (block.count('<think>') == 1 and block.count('</think>') == 1
            and block.count('<tool_call>') == 1 and block.count('</tool_call>') == 1):
            think_match = re.search(
                r'^<think>(.*?)</think>(\s*)<tool_call>(.*?)</tool_call>$',
                block, re.DOTALL
            )
            if think_match:
                format_reward += 0.5  # 每个合格的中间 block +0.5

    # 检查最后一个 block: 是否有 <think>...</think>...<answer>...</answer>
    last_block = assistant_blocks[-1]
    think_answer_match = re.search(
        r'^<think>(.*?)</think>(.*?)<answer>(.*?)</answer>$',
        last_block, re.DOTALL
    )
    if think_answer_match:
        format_reward += 0.5

    return format_reward  # 可能 > 1.0（多轮工具调用时）
```

#### 答案奖励 `compute_score_answer(solution_str, ground_truth)`

```python
def compute_score_answer(solution_str, ground_truth):
    # 提取最后一个 assistant block
    assistant_blocks = re.findall(
        r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>', solution_str, re.DOTALL
    )
    solution_str = assistant_blocks[-1]

    # 从 <answer> 标签提取答案
    answer = extract_solution(solution_str)  # regex: <answer>(.*?)</answer>

    answer_reward = 0.0
    if answer is not None:
        if subem_check(answer, ground_truth):  # SubEM: 子串包含匹配
            answer_reward = 1.0

    # 如果 <answer> 中没匹配，检查整个 solution
    if answer_reward == 0.0:
        if subem_check(solution_str, ground_truth):
            answer_reward = 0.2  # 降级奖励

    return answer_reward
```

**SubEM (Substring Exact Match)**：
```python
def subem_check(prediction, golden_answers):
    normalized_prediction = normalize_answer(prediction)
    for golden_answer in golden_answers:
        if normalize_answer(golden_answer) in normalized_prediction:
            return True  # golden answer 是 prediction 的子串
    return False
```

**normalize_answer**：小写 → 去冠词 → 去标点 → 合并空格

#### 组合奖励 `compute_score_format_answer`

```python
def compute_score_format_answer(solution_str, ground_truth):
    format_reward = compute_score_format(solution_str)     # 0 ~ ∞
    answer_reward = compute_score_answer(solution_str, ground_truth)  # 0 ~ 1.0

    format_reward = min(format_reward, 1.0)  # 截断为 [0, 1]

    if format_reward >= 0.5:
        return -1.0 + format_reward + answer_reward  # 范围: [-0.5, 1.0]
    else:
        return -1.0 + format_reward                   # 范围: [-1.0, -0.5)
```

**奖励设计逻辑**：
- 基础分 = -1.0（惩罚基线）
- 格式合规 ≥ 0.5 时才能获得答案奖励
- 最终范围：**[-1.0, 1.0]**
  - -1.0: 完全错误的格式
  - -0.5: 部分格式正确但无答案
  - 0.0: 格式完全正确但答案错误
  - 1.0: 格式完全正确 + 答案正确

---

### 2.7 数据集 — `agent_r1/src/agent_rl_dataset.py`

```python
class ToolRLDataset(RLHFDataset):
    def __init__(self, data_files, tokenizer, config, processor=None, env=None):
        self.env = env
        self.use_default_tool_template = config.get("use_default_tool_template", True)
        if self.use_default_tool_template and self.env is not None:
            self.tools = [tool.tool_description for tool in self.env.tools]
        self.use_custom_system_prompt = config.get("use_custom_system_prompt", False)
```

**关键行为**：

1. **工具描述注入**：当 `use_default_tool_template=True` 时：
   ```python
   raw_prompt = tokenizer.apply_chat_template(
       messages, tools=self.tools, add_generation_prompt=True, tokenize=False
   )
   ```
   这会将 `WikiSearchTool` 的 tool_description 通过 Qwen 的 chat template 自动注入到 system prompt 中。

2. **输出字段**：
   ```python
   row_dict["input_ids"] = input_ids      # tokenized prompt
   row_dict["attention_mask"] = attention_mask
   row_dict["position_ids"] = position_ids
   row_dict["raw_prompt_ids"] = raw_prompt_ids  # 用于多轮生成的起点
   ```

---

### 2.8 数据预处理 — `agent_r1/examples/data_preprocess/hotpotqa.py`

下载 HotpotQA 数据集并转换为训练格式：

```python
# 输入：HotpotQA 数据集 (question, answer, supporting_facts, level, type)
# 输出：train.parquet, validation.parquet

instruction_following = (
    'You FIRST think about the reasoning process as an internal monologue '
    'and then provide the final answer. '
    'The reasoning process MUST BE enclosed within <think> </think> tags. '
    'The final answer MUST BE put in <answer> </answer> tags.'
)

data = {
    "data_source": "hotpotqa/hotpot_qa",
    "prompt": [{
        "role": "user",
        "content": "Question: " + question + "\n" + instruction_following,
    }],
    "ability": "multihop_qa",
    "reward_model": {
        "style": "rule",
        "ground_truth": answer_raw  # e.g., "Jane Smith"
    },
}
```

**关键点**：
- prompt 只有一个 user message，没有 system message
- system prompt（包含工具描述）由 `ToolRLDataset` 的 `use_default_tool_template` 自动注入
- `reward_model.ground_truth` 是答案字符串
- 默认 `train_size=25600`, `val_size=128`

---

### 2.9 训练脚本 — `agent_r1/examples/trainer/run_grpo_multihopqa.sh`

```bash
export BASE_MODEL='Qwen/Qwen2.5-1.5B-Instruct'

python3 -m agent_r1.src.main_agent \
    algorithm.adv_estimator=grpo \
    data.train_files=['data/hotpotqa/train.parquet','data/2wiki/train_processed.parquet'] \
    data.val_files=['data/hotpotqa/validation.parquet',...] \
    data.train_batch_size=128 \
    data.max_prompt_length=8192 \
    data.max_response_length=8192 \
    data.max_response_length_single_turn=1024 \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.stop_token_ids=[151658] \
    actor_rollout_ref.rollout.n_repeat=5 \
    trainer.n_gpus_per_node=4 \
    tool.max_turns=5 \
    tool.tools=['wiki_search'] \
    tool.max_tool_response_length=2048
```

**关键配置解析**：

| 参数 | 值 | 含义 |
|------|-----|------|
| `algorithm.adv_estimator` | `grpo` | 使用 GRPO 算法 |
| `data.max_prompt_length` | `8192` | prompt 最大长度 |
| `data.max_response_length` | `8192` | 所有轮次总响应最大长度 |
| `data.max_response_length_single_turn` | `1024` | 单轮生成最大长度 |
| `actor_rollout_ref.rollout.name` | `vllm` | 使用 vLLM 做推理 |
| `actor_rollout_ref.rollout.stop_token_ids` | `[151658]` | `</tool_call>` 的 token ID |
| `actor_rollout_ref.rollout.n_repeat` | `5` | GRPO 每个 prompt 生成 5 个响应 |
| `tool.max_turns` | `5` | 最多 5 轮工具调用 |
| `tool.tools` | `['wiki_search']` | 使用 wiki_search 工具 |
| `tool.max_tool_response_length` | `2048` | 工具响应最大 2048 字符 |
| `trainer.n_gpus_per_node` | `4` | 4 GPU 训练 |

**入口点**：`python3 -m agent_r1.src.main_agent`（不是 `verl.trainer.main_ppo`）

---

## 3. 完整数据流图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           数据准备阶段                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  hotpotqa.py / 2wikimultihopqa.py / musique.py                         │
│      ↓                                                                  │
│  train.parquet:                                                         │
│    {data_source, prompt: [{role:"user", content:question}],            │
│     reward_model: {style:"rule", ground_truth: answer},                │
│     ability: "multihop_qa"}                                             │
│      ↓                                                                  │
│  ToolRLDataset(use_default_tool_template=True)                         │
│      ↓ apply_chat_template(messages, tools=[wiki_search])              │
│  tokenized input_ids (包含自动注入的工具描述 system prompt)               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                           训练循环 (RayAgentTrainer)                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  for epoch in range(total_epochs):                                      │
│    for batch in dataloader:                                             │
│                                                                         │
│    ┌────────────────────────────────────────────────────────────────┐   │
│    │ 1. 生成阶段: ToolGenerationManager.run_llm_loop()              │   │
│    │                                                                │   │
│    │    for turn in range(max_turns=5):                             │   │
│    │      ├→ actor_rollout_wg.generate_sequences(rollings_active)  │   │
│    │      │    → vLLM 生成直到 </tool_call> 或 EOS                  │   │
│    │      ├→ NousToolEnv.batch_step(raw_responses)                 │   │
│    │      │    → extract <tool_call> JSON                          │   │
│    │      │    → WikiSearchTool.batch_execute(queries)             │   │
│    │      │    → format: <|im_end|>...<tool_response>...</tool_response>│
│    │      └→ _update_rolling_state()                               │   │
│    │           → 拼接 response + tool_response                     │   │
│    │           → 更新 action_mask: [1...1, 0...0]                  │   │
│    │                                                                │   │
│    │    输出: DataProto                                             │   │
│    │      prompts: (B, prompt_len)                                 │   │
│    │      responses: (B, response_len)                             │   │
│    │      action_mask: (B, response_len) — 1=LLM, 0=tool          │   │
│    └────────────────────────────────────────────────────────────────┘   │
│                              ↓                                          │
│    ┌────────────────────────────────────────────────────────────────┐   │
│    │ 2. 奖励计算: AgentRewardManager                               │   │
│    │                                                                │   │
│    │    for each sample:                                           │   │
│    │      sequences_str = decode(prompt + response)                │   │
│    │      score = compute_score_format_answer(                     │   │
│    │                sequences_str, ground_truth)                   │   │
│    │      reward_tensor[i, last_valid_token] = score               │   │
│    │                                                                │   │
│    │    奖励范围: [-1.0, 1.0]                                      │   │
│    │      -1.0: 格式错误                                           │   │
│    │       0.0: 格式正确但答案错误                                  │   │
│    │       1.0: 格式正确且答案正确                                  │   │
│    └────────────────────────────────────────────────────────────────┘   │
│                              ↓                                          │
│    ┌────────────────────────────────────────────────────────────────┐   │
│    │ 3. GRPO 优势计算                                               │   │
│    │                                                                │   │
│    │    同一 prompt 的 n_repeat=5 个响应分为一组                      │   │
│    │    组内计算相对优势: advantage = (reward - mean) / std          │   │
│    │    只对 action_mask=1 的 token 计算 policy loss                 │   │
│    └────────────────────────────────────────────────────────────────┘   │
│                              ↓                                          │
│    ┌────────────────────────────────────────────────────────────────┐   │
│    │ 4. 策略更新                                                    │   │
│    │                                                                │   │
│    │    loss = policy_loss(action_mask=1 tokens) + kl_loss          │   │
│    │    optimizer.step()                                            │   │
│    └────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. 关键实现细节总结

### 4.1 stop_token_ids 配置

训练脚本中配置 `actor_rollout_ref.rollout.stop_token_ids=[151658]`，这是 `</tool_call>` 在 Qwen tokenizer 中的 token ID。vLLM 在生成时遇到这个 token 会停止，然后由 NousToolEnv 提取工具调用并执行。

### 4.2 Action Mask 的精确语义

```
responses 张量:
  [Turn1_LLM_tokens | Turn1_Tool_tokens | Turn2_LLM_tokens | ... | padding]

action_mask:
  [    1, 1, ..., 1 |     0, 0, ..., 0  |     1, 1, ..., 1 | ... |  0, 0  ]
   ↑ LLM 生成         ↑ 工具响应           ↑ LLM 生成              ↑ padding
   (计算梯度)          (不计算)             (计算梯度)               (不计算)
```

action_mask = response_attention_mask × per-token-action-flag

### 4.3 GRPO 的 n_repeat 机制

`n_repeat=5` 表示每个 prompt 生成 5 个独立的完整轨迹（包括不同的工具调用路径），然后在组内计算相对优势。这增加了训练的探索性和稳定性。

### 4.4 工具响应长度截断

`max_tool_response_length=2048`：如果搜索结果超过 2048 字符，会被截断并附加 `"..."`。这在 `NousToolEnv.format_tool_response()` 中实现。
