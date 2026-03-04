# CaRR Deep Search Agent: verl 集成实施计划（完整版）

## Context

目标: 在 `verl` 框架下训练一个多跳问答 Deep Search Agent，使用 CaRR (Citation-Aware Rubric Rewards) 的细粒度奖励框架和 C-GRPO 训练算法。

背景: CaRR 论文提出了一种分解多跳问题为原子 rubric、通过引文检查评估 agent 轨迹的奖励框架。`verl` 已有 `ToolAgentLoop` 支持多轮工具调用 RL 训练。需要将 CaRR 的工具服务器和奖励服务器接入 `verl` 的 agent loop 中。

模型: Qwen3-4B（dense 4B），8 GPU × 1 节点  
训练流程: SFT 冷启动 -> RL (C-GRPO)  
数据: `CaRR/data/deepdive-sft-glm46-trajectory-1k.jsonl` (SFT) + `CaRR/data/deepdive-rl-2k-rubrics.jsonl` (RL)

## 0. 核心修正（已合并到后续步骤）

本计划已将以下“会导致实现后仍跑不通”的问题纳入正式设计：

1. `ToolAgentLoop` 默认每次工具调用都会 `create->execute->release`，不能直接靠 `instance_id` 做跨轮会话共享。  
2. `CaRRToolAgentLoop` 不能在 `super().run()` 返回后访问 `agent_data`（它是局部变量）。  
3. CaRR reward server 需要特定 `history` 格式（`tool_call_id` 绑定 assistant/tool），不能直接复用默认 `messages`。  
4. `history[-1].role` 必须是 `assistant`，否则 reward 直接为 0。  
5. YAML/脚本里的 `~/...` 路径不可靠，统一改为 `${oc.env:HOME}/...`。  
6. 若 SFT 使用 `fsdp_sft_trainer`，配置应设置 `model.strategy` 和顶层 `use_remove_padding`，而不是 `engine.strategy` 或 `model.use_remove_padding`。

---

## 目标文件结构

```text
examples/
  __init__.py                     # 推荐（兼容性增强，非必须，namespace package 可正常导入）
  carr_deepsearch/
    __init__.py                   # 推荐（同上）
    tools/
      __init__.py
      carr_browser_tool.py        # 统一浏览器工具 (search/open/find)
      carr_session_manager.py     # CaRR 工具服务器会话管理（request级）
      carr_agent_loop.py          # 注册自定义 AgentLoop (扩展 ToolAgentLoop)
    reward/
      __init__.py
      carr_reward.py              # 异步调用 CaRR reward server 的 compute_score
      cgrpo_advantage.py          # C-GRPO advantage estimator (注册到 verl)
    data_preprocess/
      preprocess_carr_rl.py       # JSONL -> parquet (RL)
      preprocess_carr_sft.py      # JSONL -> parquet (SFT)
    config/
      carr_grpo.yaml              # RL 训练配置
      carr_sft.yaml               # SFT 训练配置
      tool_config/
        carr_browser_tools.yaml   # 工具定义
    scripts/
      run_sft.sh                  # SFT 训练脚本
      run_rl.sh                   # RL 训练脚本 (含服务启停)
      smoke_test.py               # 端到端冒烟测试
```

`verl` 核心改动（2 处）:

```text
1. verl/trainer/config/algorithm.py — AlgoConfig 新增 cgrpo_alpha: float = 0.3
2. verl/trainer/ppo/ray_trainer.py — compute_advantage 条件传递 non_tensor_batch（签名检查，~5行）
```

注意: pre-commit hook `autogen-trainer-cfg` 会自动重新生成 `_generated_*.yaml`。

---

## Phase 1: 数据预处理

### 1.1 RL 数据 (`preprocess_carr_rl.py`)

输入: `CaRR/data/deepdive-rl-2k-rubrics.jsonl`  
输出: `${oc.env:HOME}/data/carr_deepsearch/rl_train.parquet`, `${oc.env:HOME}/data/carr_deepsearch/rl_val.parquet`

源数据字段:

```json
{
  "source": "deepsearch_with_browser",
  "input_messages": [{"role": "user", "content": "..."}],
  "label": "correct answer",
  "tools": [
    {"name": "browser.search", "parameters": {"type": "object", "properties": {"query": "...", "num": "..."}}},
    {"name": "browser.open", "parameters": {"type": "object", "properties": {"id": {"type": ["integer", "string"]}}}},
    {"name": "browser.find", "parameters": {"type": "object", "properties": {"pattern": "..."}}}
  ],
  "metadata": {
    "remote_env_info": {
      "search_forbidden_strs": ["question text"],
      "rubrics": ["<E2> is a breakpoint graph...", "..."]
    }
  }
}
```

转换为 `verl` RL parquet 格式:

```json
{
  "data_source": "carr_deepsearch",
  "agent_name": "carr_tool_agent",
  "prompt": [
    {"role": "user", "content": "..."}
  ],
  "ability": "deepsearch",
  "reward_model": {
    "style": "rule",
    "ground_truth": "label"
  },
  "extra_info": {
    "split": "train",
    "index": 0,
    "rubrics": ["..."],
    "search_forbidden_strs": ["..."],
    "rubric_reward_ratio": 0.3,
    "need_tools_kwargs": true,
    "tools_kwargs": {
      "browser.search": {
        "create_kwargs": {
          "search_forbidden_strs": ["..."]
        }
      },
      "browser.open": {
        "create_kwargs": {
          "search_forbidden_strs": ["..."]
        }
      },
      "browser.find": {
        "create_kwargs": {}
      }
    }
  }
}
```

说明：

- `prompt` 字段在实现中应**直接复制原始 `input_messages`**，即 `prompt = input_messages`。
- 上面的示例只显示单条 `user` 消息，是因为当前 CaRR RL 数据的 2234 条样本全部如此；不要把预处理逻辑写成固定取 `input_messages[0]` 的有损转换。

System prompt 结论:

- `verl` 的 RL 数据格式不要求必须包含显式 `system` message；`prompt` 只要是合法的 chat messages 列表即可。
- 对当前 CaRR 集成，**首版建议不额外添加 system prompt**，直接沿用原始 RL 数据中的 `input_messages`（即单条 `user` 消息）。
- 原因：
  - CaRR 原始 SFT / RL 数据都不带显式 `system`，首版保持分布一致更稳妥。
  - 原始 `user` 问题里已经包含目标输出格式（`## Explanation with Citations / ## Exact Answer / ## References`）。
  - 工具调用协议不应手写进 system prompt；在 `verl` 中会由 `apply_chat_template(messages, tools=...)` 自动注入。
- 因此，RL 预处理建议：
  - `prompt = input_messages`
  - 不手写 `# Tools` / `<tools>` / `<tool_call>` 协议文本
  - 不在 system 中重复写答案格式要求

如需做 ablation，可额外尝试一个**极短** system prompt，但应保持 SFT 与 RL 一致同时加上。推荐仅使用：

```text
You are a research assistant. Use the available browser tools to gather evidence step by step when needed. Support claims with citations and follow the user's requested answer format.
```

关键:

- 固定随机种子（建议 `seed=42`）
- 从 2234 条中切 5% 作为 val（约 112 条）
- 保证 `search_forbidden_strs` 非空（CaRR reward server 依赖 `[0]`）

脚本参考: `examples/data_preprocess/gsm8k_tool_agent_loop.py`

### 1.2 SFT 数据 (`preprocess_carr_sft.py`)

输入: `CaRR/data/deepdive-sft-glm46-trajectory-1k.jsonl`（当前实测可用轨迹 832 条）  
输出: `${oc.env:HOME}/data/carr_deepsearch/sft_train.parquet`, `${oc.env:HOME}/data/carr_deepsearch/sft_val.parquet`

源数据字段:

```json
{
  "messages": [
    {"role": "user", "content": "question"},
    {"role": "assistant", "content": null, "reasoning_content": "thinking...",
     "tool_calls": [{"tool_call_id": "tc-xxx", "name": "browser.search", "arguments": {"query": "...", "num": 10}}]},
    {"role": "tool", "content": [{"output": "search results text"}]},
    {"role": "assistant", "content": "final answer with citations"}
  ],
  "tools": [{"name": "browser.search", "description": "...", "parameters": {...}}]
}
```

格式转换（关键）:

- `assistant.tool_calls`: CaRR 格式 `{"tool_call_id", "name", "arguments": dict}` -> Qwen3 格式 `{"type": "function", "id", "function": {"name", "arguments": json_string}}`
- `tool.content`: CaRR 格式 `[{"output": "text"}]` -> Qwen3 格式 `"text"`（纯字符串）
- `tools` 定义: CaRR 格式 -> OpenAI function tools 格式
- `reasoning_content`: 保留（Qwen3 thinking mode 支持）
- `assistant.content is None` 时转成空字符串 `""`

```python
import json


def convert_message(msg):
    """Convert CaRR message format to Qwen3 chat template format."""
    converted = {"role": msg["role"]}

    if msg["role"] == "assistant":
        converted["content"] = msg.get("content") if msg.get("content") is not None else ""
        if msg.get("reasoning_content"):
            converted["reasoning_content"] = msg["reasoning_content"]
        if msg.get("tool_calls"):
            converted["tool_calls"] = [
                {
                    "type": "function",
                    "id": tc.get("tool_call_id", f"call_{i}"),
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(tc["arguments"], ensure_ascii=False)
                        if isinstance(tc.get("arguments"), dict)
                        else str(tc.get("arguments", "{}")),
                    },
                }
                for i, tc in enumerate(msg["tool_calls"])
            ]
    elif msg["role"] == "tool":
        content = msg.get("content", "")
        if isinstance(content, list) and len(content) > 0 and isinstance(content[0], dict):
            converted["content"] = content[0].get("output", "")
        else:
            converted["content"] = str(content)
    else:
        converted["content"] = msg.get("content", "")

    return converted


def convert_tools(tools):
    """Convert CaRR tool schema to OpenAI format."""
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("parameters", {}),
            },
        }
        for t in tools
    ]
```

输出 parquet 格式:

```json
{
  "messages": [
    {"role": "user", "content": "..."},
    {
      "role": "assistant",
      "content": "",
      "reasoning_content": "thinking...",
      "tool_calls": [
        {
          "type": "function",
          "id": "tool-call-xxx",
          "function": {
            "name": "browser.search",
            "arguments": "{\"query\": \"...\", \"num\": 10}"
          }
        }
      ]
    },
    {"role": "tool", "content": "search results text..."},
    {"role": "assistant", "content": "## Explanation with Citations\n..."}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "browser.search",
        "description": "...",
        "parameters": {"type": "object", "properties": {}, "required": []}
      }
    }
  ],
  "enable_thinking": true
}
```

验证:

- 预处理脚本内应对**每条样本**做 `apply_chat_template` 验证，失败样本跳过并记录原因
- 预处理完成后，再用 Qwen3 tokenizer 对 5 条样本做 spot check
- 5% 切分 val（约 42 条，按实际 832 条计算）

---

## Phase 2: 浏览器工具实现

### 2.1 会话管理器 (`carr_session_manager.py`)

> 修正点：不能用“每个工具实例 ref_count 关闭会话”的方案。`ToolAgentLoop` 每次工具调用都会 `create/release`，必须使用 request 级 session。

设计:

- `session_id = agent_data.request_id`
- 第一次工具调用时 `start_session`
- 后续工具调用复用同一 `session_id`
- 会话关闭在 `CaRRToolAgentLoop.run()` 的 `finally` 里统一 `close_session`

```python
import aiohttp
from typing import Any


class CaRRSessionManager:
    _instance = None

    def __init__(self):
        self._started_sessions = set()                 # session_id set
        self._session_data: dict[str, dict[str, Any]] = {}  # session_id -> metadata

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def ensure_started(self, session_id: str, tool_server_url: str, **kwargs):
        if session_id not in self._started_sessions:
            await self.call_server(tool_server_url, session_id, "start_session", {}, {})
            self._started_sessions.add(session_id)
            self._session_data[session_id] = kwargs

    async def close(self, session_id: str, tool_server_url: str):
        if session_id in self._started_sessions:
            await self.call_server(tool_server_url, session_id, "close_session", {}, {})
            self._started_sessions.discard(session_id)
            self._session_data.pop(session_id, None)

    def get_session_data(self, session_id: str) -> dict[str, Any]:
        return self._session_data.get(session_id, {})

    async def call_server(self, url: str, session_id: str, name: str, arguments: dict, remote_env_info: dict):
        payload = {
            "session_id": session_id,
            "name": name,
            "arguments": arguments,
            "remote_env_info": remote_env_info,
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                    if resp.status != 200:
                        return f"Tool server HTTP {resp.status}"
                    result = await resp.json()
                    return result.get("output", "")
        except Exception as e:
            return f"Error calling tool server: {e}"
```

### 2.2 浏览器工具 (`carr_browser_tool.py`)

使用统一 `CaRRBrowserTool` 处理 `search/open/find`，但会话 id 从 `agent_data.request_id` 获取。

```python
import logging
from typing import Any
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

from .carr_session_manager import CaRRSessionManager

logger = logging.getLogger(__name__)


class CaRRBrowserTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.tool_server_url = config.get("tool_server_url", "http://localhost:7230")
        self.session_manager = CaRRSessionManager.get_instance()

    async def create(self, instance_id=None, **kwargs):
        # instance_id 在 ToolAgentLoop 中是每次调用级别，不用于会话生命周期
        if instance_id is None:
            instance_id = str(uuid4())
        return instance_id, ToolResponse()

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs):
        agent_data = kwargs.get("agent_data")
        if agent_data is None:
            return ToolResponse(text="Missing agent_data"), 0.0, {}

        session_id = agent_data.request_id
        create_kwargs = agent_data.tools_kwargs.get(self.name, {}).get("create_kwargs", {})
        search_forbidden_strs = create_kwargs.get("search_forbidden_strs", [])

        await self.session_manager.ensure_started(
            session_id,
            self.tool_server_url,
            search_forbidden_strs=search_forbidden_strs,
        )

        # browser.open 参数兼容: "0" -> 0
        if self.name == "browser.open" and "id" in parameters:
            try:
                parameters["id"] = int(parameters["id"])
            except (TypeError, ValueError):
                pass

        remote_env_info = {"search_forbidden_strs": search_forbidden_strs}
        result = await self.session_manager.call_server(
            self.tool_server_url,
            session_id,
            self.name,
            parameters,
            remote_env_info,
        )
        return ToolResponse(text=str(result)), 0.0, {}

    async def release(self, instance_id: str, **kwargs):
        # 不在这里 close session；统一在 AgentLoop 结束时 close
        return
```

### 2.3 工具配置 YAML (`carr_browser_tools.yaml`)

> 注意：`verl.tools.schemas` 中参数类型只支持 `type: string` 这类标量，不支持 `oneOf`，因此 `browser.open.id` 使用 `string`，再在 execute 中做 int 转换。

```yaml
tools:
  - class_name: "examples.carr_deepsearch.tools.carr_browser_tool.CaRRBrowserTool"
    config:
      type: native
      tool_server_url: "http://localhost:7230"
    tool_schema:
      type: "function"
      function:
        name: "browser.search"
        description: "Search the web for information"
        parameters:
          type: "object"
          properties:
            query:
              type: "string"
              description: "Search query"
            num:
              type: "integer"
              description: "Number of results to return"
          required: ["query"]

  - class_name: "examples.carr_deepsearch.tools.carr_browser_tool.CaRRBrowserTool"
    config:
      type: native
      tool_server_url: "http://localhost:7230"
    tool_schema:
      type: "function"
      function:
        name: "browser.open"
        description: "Open a webpage by search result ID or URL"
        parameters:
          type: "object"
          properties:
            id:
              type: "string"
              description: "ID(integer as string) or URL"
          required: ["id"]

  - class_name: "examples.carr_deepsearch.tools.carr_browser_tool.CaRRBrowserTool"
    config:
      type: native
      tool_server_url: "http://localhost:7230"
    tool_schema:
      type: "function"
      function:
        name: "browser.find"
        description: "Find a pattern in the currently opened webpage"
        parameters:
          type: "object"
          properties:
            pattern:
              type: "string"
              description: "Pattern to search for"
          required: ["pattern"]
```

---

## Phase 3: 自定义 Agent Loop

### 3.1 `CaRRToolAgentLoop` (`carr_agent_loop.py`)

> 关键修正：不能依赖 `self.agent_data`。必须在运行过程中自行维护 reward history。

目标:

1. 注册 `@register("carr_tool_agent")`
2. 在每条轨迹维护 `reward_history`（严格匹配 CaRR reward server）
3. 在 `run()` 结束时输出:
   - `extra_fields["messages"] = reward_history`
   - `extra_fields["task_unfinished"] = bool`，判定规则：
     ```python
     # task_unfinished = True 的条件（满足任一即可）:
     # 1. reward_history 为空
     # 2. reward_history[-1].role != "assistant"（被 max_turns/response_length 截断）
     task_unfinished = (
         len(reward_history) == 0
         or reward_history[-1].get("role") != "assistant"
     )
     ```
     注意：CaRR reward server 要求 `history[-1].role == "assistant"` 才执行评估，
     否则即使 `task_unfinished=false` 也只返回 `reward=0`。
4. 在 `finally` 里主动关闭工具 session

关键数据结构:

```python
# 每个样本一份
self._reward_state[request_id] = {
    "history": [...],
    "pending_tool_calls": [{"tool_call_id": "...", "name": "...", "arguments": "..."}],
    "turn_idx": 0,
}
```

`reward_history` 消息格式（必须）:

```json
[
  {"role": "user", "content": "..."},
  {
    "role": "assistant",
    "content": "thinking + visible text",
    "tool_calls": [
      {
        "tool_call_id": "req_x_tc_0_0",
        "name": "browser.search",
        "arguments": "{\"query\": \"...\", \"num\": 10}"
      }
    ]
  },
  {
    "role": "tool",
    "content": [
      {
        "tool_call_id": "req_x_tc_0_0",
        "output": "search result text"
      }
    ]
  },
  {"role": "assistant", "content": "final answer ..."}
]
```

实现建议:

- 继承 `ToolAgentLoop`
- 覆盖 `run`, `_handle_generating_state`, `_handle_processing_tools_state`, `_call_tool`
- 生成 tool_call_id 时使用 deterministic 规则：`{request_id}_tc_{turn}_{i}`
- `_call_tool` 返回时将 tool 输出绑定对应 tool_call_id

注册方式:

```bash
export VERL_USE_EXTERNAL_MODULES=examples.carr_deepsearch.tools.carr_agent_loop
```

数据流:

- RL parquet 中 `agent_name: "carr_tool_agent"`
- `AgentLoopWorker` 从 registry 实例化 `CaRRToolAgentLoop`

---

## Phase 4: CaRR Reward 函数

### 4.1 `carr_reward.py`

异步调用 CaRR reward server 的 `/evaluate`。

```python
import logging
import os
import aiohttp

logger = logging.getLogger(__name__)

REWARD_SERVER_URL = os.environ.get("CARR_REWARD_SERVER_URL", "http://localhost:8888")
# 建议 >= 650，避免服务端 wait_for(600) 时客户端先超时
REWARD_TIMEOUT = int(os.environ.get("CARR_REWARD_TIMEOUT", "650"))


async def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    extra_info = extra_info or {}

    messages = extra_info.get("messages", [])
    rubrics = extra_info.get("rubrics", [])
    search_forbidden_strs = extra_info.get("search_forbidden_strs", [])
    rubric_reward_ratio = extra_info.get("rubric_reward_ratio", 0.3)
    task_unfinished = extra_info.get("task_unfinished", False)

    if not messages:
        # fallback，避免空history导致服务端报错
        messages = [
            {"role": "user", "content": search_forbidden_strs[0] if search_forbidden_strs else "Question"},
            {"role": "assistant", "content": solution_str},
        ]

    if messages and messages[-1].get("role") != "assistant":
        task_unfinished = True

    payload = {
        "history": messages,
        "label": ground_truth,
        "task_unfinished": task_unfinished,
        "remote_env_info": {
            "search_forbidden_strs": search_forbidden_strs,
            "rubrics": rubrics,
            "rubric_reward_ratio": rubric_reward_ratio,
        },
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{REWARD_SERVER_URL}/evaluate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=REWARD_TIMEOUT),
            ) as resp:
                if resp.status != 200:
                    logger.error("Reward server status=%s", resp.status)
                    return {"score": 0.0, "outcome_reward": 0.0, "rubric_reward": 0.0}
                result = await resp.json()
    except Exception as e:
        logger.error("Reward server call failed: %s", e)
        return {"score": 0.0, "outcome_reward": 0.0, "rubric_reward": 0.0}

    outcome_reward = float(result.get("outcome_reward", 0.0))
    rubric_reward = float(result.get("rubric_reward", 0.0))

    # score 作为默认rm_scores，C-GRPO融合在advantage阶段完成
    return {
        "score": outcome_reward,
        "outcome_reward": outcome_reward,
        "rubric_reward": rubric_reward,
    }
```

数据流:

- `NaiveRewardManager.run_single()` -> `compute_score(...)`
- `extra_info` 包含：
  - `messages`（来自 `tool_extra_fields`，由 agent loop 的 `extra_fields` 传入）
  - `rubrics/search_forbidden_strs`（来自 parquet `extra_info`）
- 返回 dict 的 key 将进入 `non_tensor_batch`

**前提条件**（streaming reward 路径）:
- 必须走 streaming reward 路径：`enable_agent_reward_loop = True`（当 `reward.reward_model.enable = false` 时自动满足）
- 使用 `verl/experimental/reward_loop/reward_manager/naive.py` 中的 NaiveRewardManager（此版本在 line 44-47 合并 `tool_extra_fields` 到 `extra_info`）
- RL 配置中须显式设置 `reward.reward_model.enable: false`，防止被上层配置覆盖

---

## Phase 5: C-GRPO Advantage Estimator

### 5.1 `verl` 核心修改

**文件 1**: `verl/trainer/config/algorithm.py`
在 `AlgoConfig` 中新增 `cgrpo_alpha` 字段：
```python
    rollout_correction: Optional[RolloutCorrectionConfig] = None
    cgrpo_alpha: float = 0.3  # C-GRPO: rubric reward mixing ratio
```

**文件 2**: `verl/trainer/ppo/ray_trainer.py`
位置: `compute_advantage()` 的 `else` 分支

使用签名检查条件传递 `non_tensor_batch`，避免破坏 `OPTIMAL_TOKEN_BASELINE` 等不接受 `**kwargs` 的 estimator：

```python
# 在现有 adv_kwargs 构建后，advantages, returns = adv_estimator_fn(**adv_kwargs) 之前：
import inspect
_sig = inspect.signature(adv_estimator_fn)
if "non_tensor_batch" in _sig.parameters or any(
    p.kind == inspect.Parameter.VAR_KEYWORD for p in _sig.parameters.values()
):
    adv_kwargs["non_tensor_batch"] = data.non_tensor_batch
```

这样只有声明了 `non_tensor_batch` 参数或接受 `**kwargs` 的 estimator 才会收到该参数，确保向后兼容。

### 5.2 `cgrpo_advantage.py`

```python
"""C-GRPO advantage estimator.
R_i = (1-α) * R_outcome + α * R_outcome * R_hat_rubric
R_hat_rubric_i = R_rubric_i / max_group_rubric
"""

from collections import defaultdict
import numpy as np
import torch

from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage, register_adv_est


@register_adv_est("cgrpo")
def compute_cgrpo_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    config=None,
    non_tensor_batch: dict = None,
    norm_adv_by_std_in_grpo: bool = True,
    **kwargs,
):
    alpha = 0.3
    if config is not None:
        # DictConfig/BaseConfig 均可兼容
        alpha = float(config.get("cgrpo_alpha", 0.3)) if hasattr(config, "get") else 0.3

    has_carr = (
        non_tensor_batch is not None
        and "outcome_reward" in non_tensor_batch
        and "rubric_reward" in non_tensor_batch
    )

    if has_carr:
        outcome = np.array(non_tensor_batch["outcome_reward"], dtype=np.float32)
        rubric = np.array(non_tensor_batch["rubric_reward"], dtype=np.float32)
        bsz = len(outcome)

        id2indices = defaultdict(list)
        for i in range(bsz):
            id2indices[index[i]].append(i)

        norm_rubric = np.zeros(bsz, dtype=np.float32)
        for _, ids in id2indices.items():
            max_r = rubric[ids].max()
            if max_r > 0:
                norm_rubric[ids] = rubric[ids] / max_r

        cgrpo_rewards = (1 - alpha) * outcome + alpha * outcome * norm_rubric

        new_rewards = torch.zeros_like(token_level_rewards)
        for i in range(bsz):
            valid_len = int(response_mask[i].sum())
            if valid_len > 0:
                new_rewards[i, valid_len - 1] = cgrpo_rewards[i]
        token_level_rewards = new_rewards

    return compute_grpo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        config=config,
    )
```

注册:

```bash
export VERL_USE_EXTERNAL_MODULES=examples.carr_deepsearch.tools.carr_agent_loop,examples.carr_deepsearch.reward.cgrpo_advantage
```

---

## Phase 6: 训练配置

### 6.1 SFT 配置 (`carr_sft.yaml`)

> 以下配置面向 `torchrun -m verl.trainer.fsdp_sft_trainer`

```yaml
hydra:
  searchpath:
    - file://verl/trainer/config

defaults:
  - sft_trainer
  - _self_

model:
  partial_pretrain: Qwen/Qwen3-4B
  enable_gradient_checkpointing: true
  trust_remote_code: true
  strategy: fsdp

data:
  train_files: ${oc.env:HOME}/data/carr_deepsearch/sft_train.parquet
  val_files: ${oc.env:HOME}/data/carr_deepsearch/sft_val.parquet
  max_length: 65536
  train_batch_size: 4
  micro_batch_size_per_gpu: 1
  multiturn:
    enable: true
    messages_key: messages
    tools_key: tools
    enable_thinking_key: enable_thinking

# fsdp_sft_trainer 读取的是顶层 use_remove_padding
use_remove_padding: true

optim:
  lr: 4.0e-5

trainer:
  total_epochs: 3
  project_name: carr_deepsearch_sft
  experiment_name: carr-sft-qwen3-4b
  logger: ['console', 'wandb']
  default_local_dir: ${oc.env:HOME}/checkpoints/carr_deepsearch_sft
  save_freq: 1
  test_freq: 1
  nnodes: 1
  n_gpus_per_node: 8
  checkpoint:
    save_contents: ["model", "optimizer", "extra", "hf_model"]
```

### 6.2 RL 配置 (`carr_grpo.yaml`)

```yaml
hydra:
  searchpath:
    - file://verl/trainer/config

defaults:
  - ppo_trainer
  - _self_

data:
  train_files: ${oc.env:HOME}/data/carr_deepsearch/rl_train.parquet
  val_files: ${oc.env:HOME}/data/carr_deepsearch/rl_val.parquet
  return_raw_chat: true
  max_prompt_length: 4096
  max_response_length: 61440
  train_batch_size: 128

actor_rollout_ref:
  hybrid_engine: true
  model:
    path: ${oc.env:SFT_MODEL_PATH}  # 由 run_rl.sh 通过 latest_checkpointed_iteration.txt 动态设置
    enable_gradient_checkpointing: true
    trust_remote_code: true
    use_remove_padding: true
  actor:
    strategy: fsdp
    use_dynamic_bsz: true
    ppo_max_token_len_per_gpu: 24576
    optim:
      lr: 2.0e-6
    ppo_mini_batch_size: 16
    ppo_micro_batch_size_per_gpu: 2
  rollout:
    name: sglang
    mode: async
    temperature: 1.0
    top_p: 1.0
    n: 16
    gpu_memory_utilization: 0.5
    multi_turn:
      enable: true
      format: hermes
      max_assistant_turns: 30
      max_tool_response_length: 10000
      tool_response_truncate_side: right
      tool_config_path: examples/carr_deepsearch/config/tool_config/carr_browser_tools.yaml
    agent:
      default_agent_loop: carr_tool_agent
  ref:
    log_prob_micro_batch_size_per_gpu: 2

algorithm:
  adv_estimator: cgrpo
  cgrpo_alpha: 0.3
  use_kl_in_reward: false
  kl_penalty: kl

reward:
  reward_model:
    enable: false  # 使用 custom_reward_function，不需要单独的 reward model
  custom_reward_function:
    path: examples/carr_deepsearch/reward/carr_reward.py
    name: compute_score

trainer:
  nnodes: 1
  n_gpus_per_node: 8
  total_epochs: 3
  save_freq: 50
  test_freq: 50
  project_name: carr_deepsearch
  experiment_name: carr-cgrpo-qwen3-4b
  logger: ['console', 'wandb']
  default_local_dir: ${oc.env:HOME}/checkpoints/carr_deepsearch_rl
```

---

## Phase 7: Shell 脚本

### 7.1 SFT 训练 (`run_sft.sh`)

```bash
#!/bin/bash
set -euxo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$PROJECT_DIR"

if [ ! -f "$HOME/data/carr_deepsearch/sft_train.parquet" ]; then
    echo "Running SFT data preprocessing..."
    python examples/carr_deepsearch/data_preprocess/preprocess_carr_sft.py \
        --input_file CaRR/data/deepdive-sft-glm46-trajectory-1k.jsonl \
        --output_dir "$HOME/data/carr_deepsearch" \
        --val_ratio 0.05 \
        --seed 42
fi

torchrun --nnodes=1 --nproc_per_node=8 \
    -m verl.trainer.fsdp_sft_trainer \
    --config-path="$PROJECT_DIR/examples/carr_deepsearch/config" \
    --config-name='carr_sft' \
    "$@"
```

### 7.2 RL 训练 (`run_rl.sh`)

```bash
#!/bin/bash
set -euxo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$PROJECT_DIR"

: "${SERPAPI_API_KEY:?Must set SERPAPI_API_KEY}"
: "${JINA_API_KEY:?Must set JINA_API_KEY}"
: "${DEEPSEEK_API_KEY:?Must set DEEPSEEK_API_KEY}"

export VERL_USE_EXTERNAL_MODULES=examples.carr_deepsearch.tools.carr_agent_loop,examples.carr_deepsearch.reward.cgrpo_advantage
export CARR_REWARD_SERVER_URL="http://localhost:8888"
export CARR_REWARD_TIMEOUT="650"

# 从 SFT checkpoint 动态读取最新 HF 模型路径
SFT_CKPT_ROOT="$HOME/checkpoints/carr_deepsearch_sft"
LATEST_STEP=$(cat "$SFT_CKPT_ROOT/latest_checkpointed_iteration.txt")
export SFT_MODEL_PATH="$SFT_CKPT_ROOT/global_step_${LATEST_STEP}/huggingface"
echo "Using SFT model from: $SFT_MODEL_PATH"

if [ ! -f "$HOME/data/carr_deepsearch/rl_train.parquet" ]; then
    echo "Running RL data preprocessing..."
    python examples/carr_deepsearch/data_preprocess/preprocess_carr_rl.py \
        --input_file CaRR/data/deepdive-rl-2k-rubrics.jsonl \
        --output_dir "$HOME/data/carr_deepsearch" \
        --val_ratio 0.05 \
        --seed 42
fi

PIDS=()
cleanup() {
    echo "Cleaning up background processes..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
}
trap cleanup EXIT

echo "Starting CaRR tool server on port 7230..."
python CaRR/tool_server/launch_server.py \
    --serp_api_key "$SERPAPI_API_KEY" \
    --jina_api_key "$JINA_API_KEY" \
    --port 7230 &
PIDS+=($!)

echo "Starting CaRR reward server on port 8888..."
(
  cd CaRR/deepsearch_rm_with_rubrics
  python launch_server.py \
    --port 8888 \
    --model_name deepseek-chat \
    --base_url https://api.deepseek.com \
    --api_key "$DEEPSEEK_API_KEY"
) &
PIDS+=($!)

# 服务就绪检查（重试 + fail-fast）
echo "Waiting for tool server..."
for i in {1..30}; do
  if curl -s -X POST http://localhost:7230 \
    -H "Content-Type: application/json" \
    -d '{"session_id":"health","name":"start_session","arguments":{},"remote_env_info":{}}' >/dev/null; then
    echo "Tool server ready."
    break
  fi
  sleep 2
done
if ! curl -s -X POST http://localhost:7230 \
  -H "Content-Type: application/json" \
  -d '{"session_id":"health","name":"start_session","arguments":{},"remote_env_info":{}}' >/dev/null; then
  echo "ERROR: Tool server failed to start" >&2
  exit 1
fi

echo "Waiting for reward server..."
for i in {1..30}; do
  if curl -s -X POST http://localhost:8888/evaluate \
    -H "Content-Type: application/json" \
    -d '{"history":[{"role":"user","content":"q"},{"role":"assistant","content":"a"}],"label":"a","task_unfinished":true,"remote_env_info":{"search_forbidden_strs":["q"],"rubrics":[],"rubric_reward_ratio":0.3}}' >/dev/null; then
    echo "Reward server ready."
    break
  fi
  sleep 2
done
if ! curl -s -X POST http://localhost:8888/evaluate \
  -H "Content-Type: application/json" \
  -d '{"history":[{"role":"user","content":"q"},{"role":"assistant","content":"a"}],"label":"a","task_unfinished":true,"remote_env_info":{"search_forbidden_strs":["q"],"rubrics":[],"rubric_reward_ratio":0.3}}' >/dev/null; then
  echo "ERROR: Reward server failed to start" >&2
  exit 1
fi

echo "Starting RL training..."
python3 -m verl.trainer.main_ppo \
    --config-path="$PROJECT_DIR/examples/carr_deepsearch/config" \
    --config-name='carr_grpo' \
    "$@"
```

### 7.3 冒烟测试 (`smoke_test.py`)

```python
"""端到端冒烟测试:
1. 验证工具服务器连通性 (search -> open -> find)
2. 验证奖励服务器连通性 (固定样本)
3. 验证数据加载
"""
import asyncio
import aiohttp


async def test_tool_server(url="http://localhost:7230"):
    session_id = "smoke_test_001"

    async with aiohttp.ClientSession() as session:
        # start_session
        resp = await session.post(url, json={
            "session_id": session_id,
            "name": "start_session",
            "arguments": {},
            "remote_env_info": {},
        })
        result = await resp.json()
        assert "output" in result

        # browser.search
        resp = await session.post(url, json={
            "session_id": session_id,
            "name": "browser.search",
            "arguments": {"query": "breakpoint graph genome rearrangement", "num": 3},
            "remote_env_info": {"search_forbidden_strs": []},
        })
        result = await resp.json()
        assert len(result.get("output", "")) > 0

        # browser.open
        resp = await session.post(url, json={
            "session_id": session_id,
            "name": "browser.open",
            "arguments": {"id": 0},
            "remote_env_info": {"search_forbidden_strs": []},
        })
        _ = await resp.json()

        # browser.find
        resp = await session.post(url, json={
            "session_id": session_id,
            "name": "browser.find",
            "arguments": {"pattern": "breakpoint"},
            "remote_env_info": {},
        })
        _ = await resp.json()

        # close_session
        await session.post(url, json={
            "session_id": session_id,
            "name": "close_session",
            "arguments": {},
            "remote_env_info": {},
        })


async def test_reward_server(url="http://localhost:8888"):
    payload = {
        "history": [
            {"role": "user", "content": "What is the title of the paper?"},
            {"role": "assistant", "content": "The paper is titled 'Test Paper Title'."},
        ],
        "label": "Test Paper Title",
        "task_unfinished": False,
        "remote_env_info": {
            "search_forbidden_strs": ["What is the title"],
            "rubrics": ["<E0> is the exact title of a paper."],
            "rubric_reward_ratio": 0.3,
        },
    }

    async with aiohttp.ClientSession() as session:
        resp = await session.post(f"{url}/evaluate", json=payload, timeout=aiohttp.ClientTimeout(total=180))
        result = await resp.json()
        assert "reward" in result
        assert "outcome_reward" in result


if __name__ == "__main__":
    import sys

    if "--tool" in sys.argv or "--all" in sys.argv:
        asyncio.run(test_tool_server())
    if "--reward" in sys.argv or "--all" in sys.argv:
        asyncio.run(test_reward_server())
    if len(sys.argv) == 1:
        print("Usage: python smoke_test.py --tool | --reward | --all")
```

---

## Phase 8: 执行顺序与验证

### 8.0 环境准备

```bash
# 1. 安装 verl 基础依赖
pip install -e ".[test,sglang]"

# 2. 安装 CaRR 依赖
pip install -r CaRR/deepsearch_rm_with_rubrics/requirements.txt
pip install quart aiohttp

# 3. 设置 API Key 环境变量
export SERPAPI_API_KEY="your_key"
export JINA_API_KEY="your_key"
export DEEPSEEK_API_KEY="your_key"
```

### 8.1 数据预处理（无外部依赖）

```bash
# RL 数据
python examples/carr_deepsearch/data_preprocess/preprocess_carr_rl.py \
    --input_file CaRR/data/deepdive-rl-2k-rubrics.jsonl \
    --output_dir "$HOME/data/carr_deepsearch" \
    --val_ratio 0.05 \
    --seed 42

# SFT 数据
python examples/carr_deepsearch/data_preprocess/preprocess_carr_sft.py \
    --input_file CaRR/data/deepdive-sft-glm46-trajectory-1k.jsonl \
    --output_dir "$HOME/data/carr_deepsearch" \
    --val_ratio 0.05 \
    --seed 42
```

验证:

```python
import pandas as pd
from transformers import AutoTokenizer

# RL 数据检查
rl_df = pd.read_parquet(f"{__import__('os').path.expanduser('~')}/data/carr_deepsearch/rl_train.parquet")
assert "agent_name" in rl_df.columns
assert all(rl_df["agent_name"] == "carr_tool_agent")
assert all(rl_df["extra_info"].apply(lambda x: "rubrics" in x and "search_forbidden_strs" in x))

# SFT 数据检查
sft_df = pd.read_parquet(f"{__import__('os').path.expanduser('~')}/data/carr_deepsearch/sft_train.parquet")
assert "messages" in sft_df.columns
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True)
sample = sft_df.iloc[0]
tok.apply_chat_template(sample["messages"], tools=sample.get("tools"), tokenize=False)
```

### 8.2 服务启动与冒烟测试

```bash
# Terminal 1: 启动工具服务器
python CaRR/tool_server/launch_server.py --serp_api_key "$SERPAPI_API_KEY" --jina_api_key "$JINA_API_KEY" --port 7230

# Terminal 2: 启动奖励服务器
cd CaRR/deepsearch_rm_with_rubrics && python launch_server.py --port 8888 --model_name deepseek-chat --base_url https://api.deepseek.com --api_key "$DEEPSEEK_API_KEY"

# Terminal 3: 冒烟测试
python examples/carr_deepsearch/scripts/smoke_test.py --all
```

### 8.3 SFT 训练

```bash
bash examples/carr_deepsearch/scripts/run_sft.sh
```

验证:

- 训练 loss 下降
- 至少完成 1 个 epoch
- Checkpoint 保存到 `${HOME}/checkpoints/carr_deepsearch_sft/`

### 8.4 RL 训练（C-GRPO）

```bash
# 小规模验证
bash examples/carr_deepsearch/scripts/run_rl.sh \
    data.train_batch_size=16 \
    actor_rollout_ref.rollout.n=4 \
    trainer.total_epochs=1 \
    trainer.save_freq=10

# 全量训练
bash examples/carr_deepsearch/scripts/run_rl.sh
```

验证:

- `response/aborted_ratio < 0.2`
- `num_turns` 分布合理（平均 5-15）
- `outcome_reward` 和 `rubric_reward` 非全零
- 训练稳定，无 NaN 或 reward 崩塌

---

## 关键风险与缓解

| 风险 | 缓解措施 |
|---|---|
| 奖励全零（history 格式不匹配） | `carr_agent_loop.py` 中维护 CaRR 专用 `reward_history`; 保证 `history[-1].role == assistant` |
| 工具会话断裂（open/find 无上下文） | 使用 request_id 级 session；仅在 `run()` finally `close_session` |
| 工具服务超时阻塞训练 | tool execute 使用超时与错误文本兜底，不抛异常中断 rollout |
| SFT tokenize 失败 | 预处理阶段逐条 `apply_chat_template` 验证，失败样本跳过并记录 |
| `browser.open.id` 类型不匹配 | execute 中先尝试 `int()` 转换，失败回退字符串 URL |
| C-GRPO 奖励字段丢失 | `ray_trainer.compute_advantage` 透传 `non_tensor_batch`，并在日志检查 `outcome_reward/rubric_reward` |
| 长 context OOM | RL 开启 `use_dynamic_bsz` + `ppo_max_token_len_per_gpu` 控制；降低 `rollout.n` 先冒烟 |

---

## 关键文件引用

| 用途 | 文件路径 |
|---|---|
| BaseTool 接口 | `verl/tools/base_tool.py` |
| ToolResponse 定义 | `verl/tools/schemas.py` |
| 工具注册 | `verl/tools/utils/tool_registry.py` |
| Agent Loop 注册 | `verl/experimental/agent_loop/agent_loop.py` |
| ToolAgentLoop 基类 | `verl/experimental/agent_loop/tool_agent_loop.py` |
| Advantage 注册 | `verl/trainer/ppo/core_algos.py` |
| GRPO advantage 实现 | `verl/trainer/ppo/core_algos.py` |
| compute_advantage 入口 | `verl/trainer/ppo/ray_trainer.py` |
| Reward 函数加载 | `verl/trainer/ppo/reward.py` |
| NaiveRewardManager | `verl/experimental/reward_loop/reward_manager/naive.py` |
| CaRR 工具服务器 | `CaRR/tool_server/launch_server.py` |
| CaRR 奖励服务器 | `CaRR/deepsearch_rm_with_rubrics/launch_server.py` |
| GSM8K 工具示例 | `verl/tools/gsm8k_tool.py` |
| GSM8K RL 预处理参考 | `examples/data_preprocess/gsm8k_tool_agent_loop.py` |
| GSM8K SFT 预处理参考 | `examples/data_preprocess/gsm8k_multiturn_sft.py` |
| 多轮 GRPO 配置示例 | `examples/sglang_multiturn/config/gsm8k_multiturn_grpo.yaml` |
| SFT 配置模板 | `verl/trainer/config/sft_trainer.yaml` |
| MultiTurnSFTDataset | `verl/utils/dataset/multiturn_sft_dataset.py` |

---

## 监督执行清单（后续你让我逐步监督时使用）

- Gate 1: 数据预处理通过（字段/切分/tokenizer 验证）
- Gate 2: 工具链路通过（search->open->find）
- Gate 3: reward 通过（非全零且无格式错误）
- Gate 4: 小规模 RL 通过（无崩溃、指标正常）
- Gate 5: 全量训练启动并稳定
