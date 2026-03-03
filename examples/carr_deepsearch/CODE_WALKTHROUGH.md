# CaRR Deep Search — 代码讲解文档

本文档逐部分讲解实施 CaRR Deep Search Agent 所需了解的所有代码和 API。

---

## 第一部分：CaRR 工具服务器

**相关文件**：
- `CaRR/tool_server/launch_server.py` — 服务器主体
- `CaRR/tool_server/web_search.py` — 搜索/解析/查找的具体实现

### 1. 总体架构

CaRR 工具服务器是一个 Quart (异步 Flask) HTTP 应用，运行在端口 7230。它提供浏览器工具的后端服务：搜索网页、打开网页、在网页内查找。

所有请求发送到同一个端点 `POST /`，通过请求体中的 `name` 字段区分操作类型。

### 2. 会话状态管理

```python
session2sandbox = defaultdict(dict)
```

服务器用一个全局字典管理所有会话的状态。每个 `session_id` 对应一个 `sandbox` 字典，存储：
- `idx2url`：搜索结果中序号到 URL 的映射（由 `browser.search` 写入）
- `cur_web_content`：当前打开的网页内容（由 `browser.open` 写入）

这意味着 **三个工具共享同一个 session 的状态**。典型调用流程：

```
browser.search("query")  →  sandbox["idx2url"] = {0: "url1", 1: "url2", ...}
browser.open(0)           →  读取 sandbox["idx2url"][0]，下载内容，存入 sandbox["cur_web_content"]
browser.find("pattern")  →  在 sandbox["cur_web_content"] 中搜索
```

### 3. 请求格式

所有请求都是相同的 JSON 结构：

```json
{
    "session_id": "unique_session_id",
    "name": "browser.search",
    "arguments": {"query": "...", "num": 10},
    "remote_env_info": {
        "search_forbidden_strs": ["原始问题文本"]
    }
}
```

- `session_id`：会话标识，用于查找对应的 sandbox
- `name`：操作名称（`start_session` / `close_session` / `browser.search` / `browser.open` / `browser.find`）
- `arguments`：操作参数
- `remote_env_info`：环境信息，主要用于传递防作弊的 `search_forbidden_strs`

### 4. 五个操作

#### 4.1 `start_session`
创建会话。由于使用 `defaultdict(dict)`，实际上不需要显式创建。返回 `"Sucess start session"`。

#### 4.2 `close_session`
删除会话的 sandbox：`del session2sandbox[session_id]`。

#### 4.3 `browser.search`

```python
result, idx2url = await search(query, num=num, forbidden_strs=forbidden_strs,
                                proxy=args.http_proxy, serp_api_key=args.serp_api_key)
sandbox['idx2url'] = idx2url
```

调用 `web_search.py` 中的 `search()` 函数：
- 使用 SerpAPI（Google 搜索的 API 封装）进行网页搜索
- 对每条搜索结果，检查是否包含"禁止字符串"（防作弊），如果包含则跳过
- 返回格式化的搜索结果文本和 `idx2url` 映射

返回的文本格式：
```
[0] Title: Some Title
[0] URL Source: https://example.com
[0] Description: Some description

[1] Title: Another Title
[1] URL Source: https://another.com
[1] Description: Another description
```

#### 4.4 `browser.open`

```python
idx2url = sandbox.get("idx2url", {})
idx = arguments.get("id", None)
url = idx2url.get(idx) if isinstance(idx, int) else idx
result = await parse_url(url=url, ...)
sandbox["cur_web_content"] = result
result = result[:10000]  # 截断到 10000 字符
```

关键细节：
- `id` 参数可以是整数（搜索结果序号）或字符串（直接 URL）
- 使用 Jina Reader API (`r.jina.ai/URL`) 将网页转换为 Markdown 文本
- 结果存入 `sandbox["cur_web_content"]`（完整内容），但返回给 agent 时截断为 10000 字符
- **类型判断用 `isinstance(idx, int)`**：如果模型输出的是字符串 `"0"`，不会匹配。我们的工具实现需要做 `int()` 转换

#### 4.5 `browser.find`

```python
find_results = find(pattern=pattern, parse_content=cur_web_content,
                    max_results=20, context_length=200, word_overlap_threshold=0.8)
```

在当前打开的网页内容中搜索模式：
- 先做精确匹配（大小写不敏感）
- 如果精确匹配不够，做模糊匹配（基于词级重叠度 ≥ 0.8）
- 返回每个匹配位置前后各 200 字符的上下文

### 5. 防作弊机制 — `contain_forbidden_str()`

```python
def contain_forbidden_str(text, forbidden_str, ngram_size=13):
    normalized_text = normalize_string(text)
    normalized_forbidden = normalize_string(forbidden_str)

    if len(normalized_forbidden.split()) < ngram_size:
        return normalized_forbidden in normalized_text  # 短文本：精确匹配

    # 长文本：13-gram 重叠检测
    forbidden_ngrams = word_ngrams(normalized_forbidden, ngram_size)
    text_ngrams = word_ngrams(normalized_text, ngram_size)
    return len(forbidden_ngrams & text_ngrams) > 0
```

这个函数防止 agent 直接搜索原始问题文本来找到答案。`search_forbidden_strs[0]` 通常就是原始问题。如果搜索结果中包含与问题文本相似的 13-gram，该结果会被过滤掉。

### 6. 对我们实现的关键影响

| 要点 | 影响 |
|------|------|
| 三个工具共享 session 状态 | 需要 `CaRRSessionManager` 确保同一轨迹的三个工具用同一个 `session_id` |
| `browser.open` 的 `id` 参数有类型敏感性 | 工具 `execute()` 中需要 `try: int(id_val)` 转换 |
| 搜索结果截断 10000 字符 | RL 配置中 `max_tool_response_length` 也应设为 10000 |
| 会话状态不持久化 | 服务器重启后状态丢失，训练期间不要重启 |
| `search_forbidden_strs` 需要在每次 search/open 请求中传递 | `create_kwargs` 中存储，每次 `execute()` 时传入 `remote_env_info` |

---

## 第二部分：CaRR 奖励服务器

**相关文件**：
- `CaRR/deepsearch_rm_with_rubrics/launch_server.py` — 服务器主体
- `CaRR/deepsearch_rm_with_rubrics/prompts/*.txt` — LLM Judge 的 prompt 模板

### 1. 总体架构

奖励服务器是一个 FastAPI 应用，运行在端口 8888，核心功能是评估 agent 的回答质量。它有两种奖励维度：

- **Outcome Reward（结果奖励）**：回答对不对？二元判断 (0 或 1)
- **Rubric Reward（评分标准奖励）**：推理链中的中间步骤对不对？细粒度评估 (0~1 之间的分数)

两种奖励都依赖 LLM Judge（默认用 DeepSeek）来做判断。

### 2. GPTModel — LLM Judge 封装

```python
class GPTModel:
    def __init__(self, model_name, base_url, api_key):
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url, ...)

    async def get_resp(self, message_list):
        for i in range(3):  # 最多重试 3 次
            try:
                chat_completion = await self.client.chat.completions.create(
                    messages=message_list, model=self.model_name)
                return chat_completion.choices[0].message.content
            except Exception as e:
                time.sleep(1)
                continue
        return ''
```

启动时通过命令行参数配置：
- `--model_name deepseek-chat`
- `--base_url https://api.deepseek.com`
- `--api_key $DEEPSEEK_API_KEY`

所有评判逻辑都通过给 LLM 发 prompt，让 LLM 做判断。

### 3. `/evaluate` 端点 — 入口

这是唯一的 API 端点。请求格式和前置检查：

```python
@app.post("/evaluate")
async def evaluate(request: Request):
    data = await request.json()

    # 必须包含 4 个字段
    for key in ["history", "label", "task_unfinished", "remote_env_info"]:
        if key not in data:
            raise HTTPException(...)

    # 检查 1: 任务未完成 → 全零奖励
    if data["task_unfinished"]:
        return {"reward": 0, "outcome_reward": 0, "rubric_reward": 0, ...}

    # 检查 2: 最后一条消息必须是 assistant 且有 content
    if history[-1]['role'] != 'assistant' or 'content' not in history[-1]:
        return {"reward": 0, ...}

    # 从 remote_env_info 提取关键信息
    question = remote_env_info["search_forbidden_strs"][0]  # ← 注意！问题文本从这里来
    rubrics = remote_env_info.get("rubrics", [])
    rubric_reward_ratio = remote_env_info.get("rubric_reward_ratio", 0)

    response = history[-1]['content']  # agent 的最终回答

    result = await asyncio.wait_for(
        get_reward(response, question, answer, history, rubrics, rubric_reward_ratio),
        timeout=600  # 10 分钟超时
    )
```

**关键发现**：
1. **`question` 来自 `search_forbidden_strs[0]`**，不是从 history 里取的。防作弊用的"禁止搜索字符串"就是原始问题本身。
2. **`history[-1]` 必须是 assistant role**，且必须有 `content` 字段。否则直接给 0 分。
3. **超时设置是 600 秒**（10 分钟），因为 LLM judge 调用可能很慢。

### 4. `get_outcome_reward()` — 结果正确性判断

```python
async def get_outcome_reward(response, question, answer):
    prompt = OUTCOME_REWARD_JUDGE_PROMPT.format(
        question=question, correct_answer=answer, response=response
    )
```

使用的 prompt 模板 (`prompts/get_outcome_reward.txt`) 要求 LLM judge：
1. 从 response 中**提取最终答案** (`extracted_final_answer`)
2. 与标准答案比较
3. 判断是否正确 (`correct: yes/no`)

LLM judge 的输出通过正则解析：
```python
correctness_pattern = r"(?i)\*{0,2}correct\*{0,2}\s*:\s*(no|yes)"
extracted_pattern = r"(?i)\*{0,2}extracted_final_answer\*{0,2}\s*:\s*(.+)"
```

最终返回 `1.0`（正确）或 `0`（错误）。最多重试 3 次。

### 5. `get_rubric_reward()` — 核心！三步评分流水线

这是 CaRR 论文的核心创新。以一个例子说明：

假设问题是："哪位科学家发明了万维网？他在哪所大学获得学士学位？"

Rubrics（评分标准）可能是：
```
C1. <E0> 是万维网的发明者
C2. <E1> 是 <E0> 获得学士学位的大学
C3. <E2> 是 <E0> 获得学士学位的年份
```

#### Step 1: 实体识别 (`identify_entity`)

用 LLM judge 从 agent 回答中提取每个实体占位符的真实值：

```python
identify_entity_prompt = IDENTIFY_ENTITY_PROMPT.format(
    question=question, constraints=rubric_text, response=response
)
```

LLM judge 返回类似：
```json
{"E0": "Tim Berners-Lee", "E1": "Queen's College, Oxford", "E2": "1976"}
```

通过 `check_identified_entities()` 验证：
- 返回值必须是 dict
- keys 必须与 rubrics 中出现的所有 `<Exx>` 完全匹配
- 值不能是 "null"/"none" 字符串

最多重试 5 次。如果 5 次都失败 → rubric_reward = 0。

#### Step 2: 填充 rubrics + 提取引文内容 + LLM 判断

**2.1 填充 rubrics**：将 `<E0>` 等占位符替换为实际值：
```
"<E0> 是万维网的发明者" → "Tim Berners-Lee 是万维网的发明者"
```

如果某个 rubric 中有未识别的实体（值为 null），该 rubric 直接判为 0 分。

**2.2 提取引文内容** — `extract_citation_content()`：

这是**最关键的函数**，单独在下面详细讲解。

**2.3 LLM 判断**：用填充后的 rubrics 和引文内容，让 LLM judge 判断每条 rubric 是否被引文支持：

```python
judge_rubric_prompt = JUDGE_RUBRIC_PROMPT.format(
    context=citation_content,      # 引文内容
    statements=rubric_text.strip() # 填充后的 rubrics
)
```

prompt 模板 (`prompts/judge_rubric.txt`) 要求 LLM：
- 逐条分析每个 statement 是否被 webpage 内容支持
- 找到证据的 URL
- 输出 JSON：`{"S1": true, "S2": false, ...}`

**核心思想**：不是简单看 agent 说了什么，而是看 agent 引用的网页是否真的支持它的说法。这就是 "Citation-Aware" 的含义。

#### Step 3: BFS 连通性检查

即使某些 rubrics 被引文支持了，还需要检查它们是否形成从 `E0`（最终答案）出发的**连通推理链**：

```python
# 建立 entity → rubric 的映射
entity2rubrics = defaultdict(set)
for idx, rubric in rubric_scores.items():
    if rubric["is_supported"] == 1:
        for entity in rubric["entity_values"]:
            entity2rubrics[entity].add(idx)

# 从 E0 出发做 BFS
arrivable_entities = {"E0"}
arrivable_rubrics = set()
queue = deque(["E0"])
while queue:
    entity = queue.popleft()
    for idx in entity2rubrics[entity]:
        arrivable_rubrics.add(idx)
        for other_entity in rubric_scores[idx]["entity_values"]:
            if other_entity not in arrivable_entities:
                arrivable_entities.add(other_entity)
                queue.append(other_entity)
```

**直觉理解**：假设有 3 条 rubrics：
- C1: `<E0> 是万维网的发明者` — 涉及 E0
- C2: `<E1> 是 <E0> 获得学士学位的大学` — 涉及 E0, E1
- C3: `<E2> 是某个无关事实` — 涉及 E2

即使 C3 被引文支持了，但从 E0 出发 BFS 到不了 E2（因为没有连接 E0/E1 和 E2 的 rubric），所以 C3 不算分。这保证了只有**与最终答案相关的推理步骤**才被奖励。

最终 rubric_reward = 可达且被支持的 rubric 数 / 总 rubric 数。

### 6. `extract_citation_content()` — 对我们实现影响最大的函数

这个函数从 agent 的回答中提取引用的 URL，然后从 history 的工具调用记录中找到这些 URL 对应的内容。

```python
def extract_citation_content(response, history, max_citation_num=20):
    # Step 1: 从 response 文本中用正则提取引用的 URL
    pattern = r'\[\d+\]\((?:view-source:)?(https?://.+?)\)[\],]'
    citation_urls = re.findall(pattern, response)
    # 去重、清理 jina 前缀...

    # Step 2: 遍历 history，建立 URL → 内容 的映射
    tool_calls = {}
    for item in history:
        if item["role"] == "assistant" and "tool_calls" in item:
            for tool_call in item["tool_calls"]:
                tool_calls[tool_call["tool_call_id"]] = tool_call  # ← 关键！

        if item["role"] == "tool":
            for tool_output in item["content"]:     # ← content 是列表！
                tool_call_id = tool_output["tool_call_id"]
                tool_call = tool_calls[tool_call_id] # ← 通过 tool_call_id 关联
                tool_name = tool_call["name"]
                arguments = json.loads(tool_call["arguments"])  # ← arguments 是 JSON 字符串
                ...
```

**CaRR 期望的 history 格式**：

```python
# assistant 消息中的 tool_calls:
{
    "role": "assistant",
    "tool_calls": [
        {
            "tool_call_id": "tc-xxx",      # 唯一标识符
            "name": "browser.search",       # 工具名
            "arguments": "{\"query\": \"...\"}"  # JSON 字符串
        }
    ]
}

# tool 消息中的 content 是列表:
{
    "role": "tool",
    "content": [
        {
            "tool_call_id": "tc-xxx",   # 与上面的 tool_call_id 对应
            "output": "search results..." # 工具输出文本
        }
    ]
}
```

**这与 verl ToolAgentLoop 产生的消息格式不同！** verl/Qwen3 的格式是：
```python
# verl/Qwen3 格式的 tool 消息:
{
    "role": "tool",
    "content": "工具输出文本",  # ← 纯字符串，不是列表
    "tool_call_id": "tc-xxx"   # ← 在消息顶层，不在 content 内
}
```

**这就是为什么实施计划中需要维护一个单独的 `reward_history`** — 必须将 verl 格式的消息转换为 CaRR 期望的格式，才能正确调用奖励服务器。

### 7. `get_reward()` — 最终奖励合成

```python
async def get_reward(response, question, answer, history, rubrics, rubric_reward_ratio=0):
    # 去掉 References 部分再评判
    if '## References' in response:
        response = response.rsplit('## References', 1)[0].strip()

    outcome_reward = 0
    rubric_reward = 0
    if rubric_reward_ratio < 1:
        outcome_reward = (await get_outcome_reward(...))["reward"]
    if rubric_reward_ratio > 0:
        rubric_reward = (await get_rubric_reward(...))["reward"]

    final_reward = (1-rubric_reward_ratio) * outcome_reward + rubric_reward_ratio * rubric_reward
```

**两层混合逻辑**：
- **服务器端**：`(1-α)*outcome + α*rubric`，这是简单的线性加权
- **我们的 C-GRPO**：`(1-α)*outcome + α*outcome*normalized_rubric`，rubric 是对 outcome 的乘性调节，且 rubric 是组内归一化后的

我们需要**分别拿到** `outcome_reward` 和 `rubric_reward` 两个原始值，然后在 C-GRPO advantage estimator 中做自己的混合。`rubric_reward_ratio` 只要是 0 < x < 1 的值就行（确保两个 reward 都被计算），具体值不影响我们——因为我们取的是分开返回的两个值，不是混合后的 `reward`。

### 8. 返回值格式

`/evaluate` 返回：
```json
{
    "reward": 0.7,           // 混合后的最终奖励（我们不直接用这个）
    "outcome_reward": 1.0,   // 结果正确性 (0 或 1)
    "rubric_reward": 0.4,    // rubric 评分 (0~1)
    "rubric_scores": {...}   // 每条 rubric 的详细评分
}
```

### 9. 对我们实现的关键影响

| 要点 | 影响 |
|------|------|
| `/evaluate` 需要 `history` 字段，且格式含 `tool_call_id` 绑定 | 必须在 AgentLoop 中构建 CaRR 格式的 `reward_history` |
| `question` 来自 `search_forbidden_strs[0]` | 数据预处理时确保该字段就是原始问题 |
| `history[-1].role` 必须是 `assistant` | AgentLoop 结束时检查，否则标记 `task_unfinished=True` |
| tool 消息的 content 是 `[{"tool_call_id": "...", "output": "..."}]` 列表 | 不能直接传 verl 的 tool 消息，需要格式转换 |
| `arguments` 是 JSON 字符串 | 转换时要 `json.dumps(dict)` |
| 超时 600 秒 | 我们的 reward 函数超时应 >= 650 秒 |
| 依赖 `./prompts/` 目录 | 启动服务器时必须 `cd` 到正确目录 |

---

## 第三部分：verl 的 BaseTool 接口和工具系统

**相关文件**：
- `verl/tools/base_tool.py` — 工具基类
- `verl/tools/schemas.py` — 数据结构定义（ToolResponse, OpenAIFunctionToolSchema 等）
- `verl/tools/utils/tool_registry.py` — 工具注册与加载
- `verl/tools/gsm8k_tool.py` — GSM8K 工具（参考实现）
- `examples/sglang_multiturn/config/tool_config/gsm8k_tool_config.yaml` — 工具配置示例

### 1. 工具系统总览

verl 的工具系统由三部分组成：

```
YAML 配置文件 (tool_config.yaml)
    ↓ 解析
tool_registry.py (initialize_tools_from_config)
    ↓ 实例化
BaseTool 子类 (如 Gsm8kTool, 我们的 CaRRBrowserTool)
    ↓ 被调用
ToolAgentLoop (状态机，下一部分讲)
```

### 2. BaseTool — 工具基类

BaseTool 定义了工具的**完整生命周期**，有 4 个核心方法：

```
create() → execute() → ... → execute() → calc_reward() → release()
  创建        执行(可多次)         计算奖励        释放
```

#### 2.1 `__init__(self, config, tool_schema)`

```python
def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
    self.config = config          # YAML 中的 config 字典
    self.tool_schema = tool_schema # OpenAI 格式的工具 schema
    self.name = self.tool_schema.function.name  # 工具名，如 "browser.search"
```

**注意**：一个 BaseTool 子类只会被实例化**一次**（全局单例）。多个轨迹共用同一个工具实例，通过 `instance_id` 区分不同轨迹的状态。

#### 2.2 `create(instance_id, **kwargs)` → `(instance_id, ToolResponse)`

```python
async def create(self, instance_id=None, **kwargs) -> tuple[str, ToolResponse]:
```

**作用**：为一条新的轨迹创建状态。每条轨迹开始时，ToolAgentLoop 为每个工具调用一次 `create`。

**参数来源**：parquet 数据中 `extra_info.tools_kwargs` 里对应工具的 `create_kwargs`。以 GSM8K 为例：

```python
# parquet 数据中：
"tools_kwargs": {
    "calc_gsm8k_reward": {
        "create_kwargs": {"ground_truth": "42"}
    }
}
```

ToolAgentLoop 调用时：
```python
instance_id, response = await tool.create(
    instance_id=some_id,
    create_kwargs={"ground_truth": "42"}
)
```

看 Gsm8kTool 的实现：

```python
async def create(self, instance_id=None, ground_truth=None, **kwargs):
    if instance_id is None:
        instance_id = str(uuid4())
    if ground_truth is None:
        ground_truth = kwargs.get("create_kwargs", {}).get("ground_truth", None)
    self._instance_dict[instance_id] = {
        "response": "",
        "ground_truth": ground_truth,
        "reward": 0.0,
    }
    return instance_id, ToolResponse()
```

它把每条轨迹的状态存在 `self._instance_dict[instance_id]` 字典里。

#### 2.3 `execute(instance_id, parameters, **kwargs)` → `(ToolResponse, float, dict)`

```python
async def execute(self, instance_id, parameters, **kwargs) -> tuple[ToolResponse, float, dict]:
```

**作用**：当模型生成了一个 tool_call 时，AgentLoop 解析出工具名和参数，调用对应工具的 `execute`。

**返回三个值**：
- `ToolResponse`：工具返回给模型的文本/图片/视频
- `float`：步级奖励（step reward），可以在每次工具调用时给予奖励/惩罚
- `dict`：指标字典（metrics），用于日志记录

GSM8K 的例子：

```python
async def execute(self, instance_id, parameters, **kwargs):
    answer = parameters.get("answer", "")
    # 存储模型提交的答案
    self._instance_dict[instance_id]["response"] = "#### " + answer
    # 计算是否正确
    reward = await self.calc_reward(instance_id)
    # 如果答案没改进，给 -0.05 惩罚
    tool_reward = 0.0 if reward > self._instance_dict[instance_id]["reward"] else -0.05
    return ToolResponse(text=f"Current parsed {answer=} {reward=}"), tool_reward, {}
```

#### 2.4 `calc_reward(instance_id)` → `float`

轨迹结束时可以调用，计算基于工具状态的最终奖励。默认返回 0.0。

#### 2.5 `release(instance_id)`

释放轨迹状态，清理资源。GSM8K 的实现是 `del self._instance_dict[instance_id]`。

### 3. ToolResponse — 工具返回值

```python
class ToolResponse(BaseModel):
    text: str | None = None       # 文本内容
    image: list[Any] | None = None  # 图片（多模态场景）
    video: list[Any] | None = None  # 视频
```

对我们的场景，只用 `text` 字段就够了：
```python
return ToolResponse(text="search results here..."), 0.0, {}
```

### 4. OpenAI 格式的工具 Schema

工具 schema 使用标准 OpenAI function calling 格式，由 Pydantic model 定义：

```python
OpenAIFunctionToolSchema
  ├── type: "function"
  └── function: OpenAIFunctionSchema
        ├── name: "browser.search"
        ├── description: "Search the web..."
        └── parameters: OpenAIFunctionParametersSchema
              ├── type: "object"
              ├── properties: {"query": {"type": "string", "description": "..."}, ...}
              └── required: ["query"]
```

这个 schema 有两个用途：
1. **传给推理引擎**：让模型知道有哪些可用工具以及参数格式
2. **在 `__init__` 中设置 `self.name`**：`self.name = self.tool_schema.function.name`

### 5. 工具注册与加载 — `initialize_tools_from_config()`

这是工具系统的入口函数，接收 YAML 配置文件路径，返回工具实例列表。

```python
def initialize_tools_from_config(tools_config_file):
    tools_config = OmegaConf.load(tools_config_file)  # 加载 YAML
    tool_list = []

    for tool_config in tools_config.tools:
        cls_name = tool_config.class_name      # 如 "verl.tools.gsm8k_tool.Gsm8kTool"
        tool_type = ToolType(tool_config.config.type)  # "native" 或 "mcp"
        tool_cls = get_tool_class(cls_name)    # 动态导入类

        match tool_type:
            case ToolType.NATIVE:
                tool_schema = OpenAIFunctionToolSchema.model_validate(tool_schema_dict)
                tool = tool_cls(config=..., tool_schema=tool_schema)
                tool_list.append(tool)
            case ToolType.MCP:
                # MCP 工具（不在我们的场景中）
                ...

    return tool_list
```

`get_tool_class()` 通过 `importlib` 动态导入。它把 `"examples.carr_deepsearch.tools.carr_browser_tool.CaRRBrowserTool"` 拆分为模块路径和类名，然后导入。

### 6. YAML 配置文件格式

以 GSM8K 为例：

```yaml
tools:
  - class_name: "verl.tools.gsm8k_tool.Gsm8kTool"  # 工具类的完整路径
    config:
      type: native                                     # native 或 mcp
    tool_schema:                                       # OpenAI 格式 schema
      type: "function"
      function:
        name: "calc_gsm8k_reward"
        description: "A tool for calculating the reward of gsm8k."
        parameters:
          type: "object"
          properties:
            answer:
              type: "string"
              description: "The model's answer"
          required: ["answer"]
```

对我们的场景，一个 YAML 里定义**三个工具**，共用同一个类但 name 不同：

```yaml
tools:
  - class_name: "examples.carr_deepsearch.tools.carr_browser_tool.CaRRBrowserTool"
    config:
      type: native
      tool_server_url: "http://localhost:7230"
    tool_schema:
      type: "function"
      function:
        name: "browser.search"    # ← 工具名 1
        ...
  - class_name: "examples.carr_deepsearch.tools.carr_browser_tool.CaRRBrowserTool"
    config: ...
    tool_schema:
      type: "function"
      function:
        name: "browser.open"     # ← 工具名 2
        ...
  - class_name: "examples.carr_deepsearch.tools.carr_browser_tool.CaRRBrowserTool"
    config: ...
    tool_schema:
      type: "function"
      function:
        name: "browser.find"    # ← 工具名 3
        ...
```

这意味着 `CaRRBrowserTool` 会被实例化 **3 次**，分别对应 3 个工具名。每个实例的 `self.name` 不同，但 `execute()` 方法相同——都是把请求转发到 CaRR 工具服务器。

### 7. 关键设计理解：工具实例 vs 轨迹实例

```
                       全局（进程级）
                    ┌──────────────────┐
                    │  CaRRBrowserTool │  ← __init__ 时创建，全局唯一
                    │  name="browser.  │
                    │       search"    │
                    └──────────────────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
    instance_id_1     instance_id_2     instance_id_3
    (轨迹 1)          (轨迹 2)          (轨迹 3)
```

- **工具对象**是全局单例，`__init__` 只调用一次
- **轨迹状态**通过 `instance_id` 区分，存在工具对象的内部字典里
- 每条轨迹的生命周期：`create(id)` → 多次 `execute(id, ...)` → `release(id)`

### 8. 对我们 CaRRBrowserTool 的影响

| 要点 | 影响 |
|------|------|
| 3 个工具名用同一个类 | `execute()` 用 `self.name` 区分请求类型，转发到 CaRR 服务器 |
| `create()` 接收 `create_kwargs` | 从 parquet `tools_kwargs` 传入 `search_forbidden_strs` |
| `execute()` 返回 `(ToolResponse, reward, metrics)` | 我们的浏览器工具不给步级奖励，reward 返回 0.0 |
| `release()` 需要清理 | 关闭 CaRR 服务器的 session |
| 工具是全局单例，但每个轨迹有独立 instance_id | 需要 CaRRSessionManager 管理每个 instance_id 对应的 server-side session |
| `execute` 的 `parameters` 是解析后的 dict | CaRR 服务器的 `arguments` 就是 dict，可以直接传 |

---

（后续部分将继续补充）
