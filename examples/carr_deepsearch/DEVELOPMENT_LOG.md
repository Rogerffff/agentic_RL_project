# CaRR Deep Search — 开发记录与代码说明

本文档记录 Phase 1（代码编写 + 本地验证）阶段的全部产出，面向下一阶段开发者。

---

## 一、总体概况

### 1.1 目标

在 verl 框架下实现 CaRR (Citation-Aware Rubric Rewards) Deep Search Agent 的完整训练流水线：

```
SFT 冷启动（Qwen3-4B）→ C-GRPO RL 训练 → BrowseComp 评测
```

### 1.2 完成状态

| 阶段 | 状态 | 说明 |
|------|------|------|
| 代码编写（18 个新文件） | ✅ 完成 | 全部文件已编写并通过逐步 code review |
| 本地 Mac 验证（Gate 1） | ✅ 通过 | 语法/schema/数据预处理/import 全部验证 |
| GPU 训练验证（Gate 2-7） | ⏳ 待做 | 正式训练前先在 `2 x 5090` 完成 Gate 2-5，再进入 8 卡正式训练 |

### 1.3 架构一句话

Agent（Qwen3-4B）使用 CaRR 工具服务器（browser.search/open/find）进行多轮网页搜索，CaRR 奖励服务器评估轨迹质量（outcome + rubric），C-GRPO 算法将两种奖励融合后做组内标准化训练。

---

## 二、数据文件位置

### 2.1 源数据（CaRR submodule 内）

```
CaRR/data/
├── deepdive-rl-2k-rubrics.jsonl          # RL 训练数据（2234 条，含 rubrics）    8.8 MB
└── deepdive-sft-glm46-trajectory-1k.jsonl # SFT 轨迹数据（832 条，含工具调用）  151 MB
```

BrowseComp 评测数据从 HuggingFace `smolagents/browse_comp` 自动下载（test split，1266 条）。

### 2.2 预处理产物（已生成，在仓库内）

```
examples/carr_deepsearch/data/
├── rl_train.parquet          # RL 训练集      2123 条   6.0 MB
├── rl_val.parquet            # RL 验证集       111 条   344 KB
├── sft_train.parquet         # SFT 训练集      791 条    64 MB
├── sft_val.parquet           # SFT 验证集       41 条   3.2 MB
└── browsecomp_eval.parquet   # BrowseComp 评测 1266 条   3.9 MB
```

### 2.3 Parquet Schema

**RL / BrowseComp 共用 schema**（6 列）:

| 列名 | 类型 | 说明 |
|------|------|------|
| `data_source` | str | `"carr_deepsearch"` 或 `"browsecomp"` |
| `agent_name` | str | `"carr_tool_agent"` — 对应注册的 AgentLoop |
| `prompt` | list[dict] | `[{"role": "user", "content": "..."}]` |
| `ability` | str | `"deepsearch"` |
| `reward_model` | dict | `{"style": "rule", "ground_truth": "..."}` |
| `extra_info` | dict | 包含 rubrics, search_forbidden_strs, tools_kwargs 等 |

`extra_info` 字段详情：

```python
{
    "rubrics": ["<E0> is the inventor of ...", ...],  # BrowseComp 为 []
    "search_forbidden_strs": ["原始问题文本"],          # 防作弊 + 奖励服务器用作 question
    "rubric_reward_ratio": 0.3,                        # BrowseComp 为 0.0
    "need_tools_kwargs": True,
    "tools_kwargs": {
        "browser.search": {"create_kwargs": {"search_forbidden_strs": [...]}},
        "browser.open":   {"create_kwargs": {"search_forbidden_strs": [...]}},
    },
    "split": "train" / "val" / "eval",
    "index": 0,
}
```

**SFT schema**（3 列）:

| 列名 | 类型 | 说明 |
|------|------|------|
| `messages` | list[dict] | OpenAI 格式的多轮对话，含 tool_calls 和 reasoning_content |
| `enable_thinking` | bool | 始终为 `True`（Qwen3 thinking mode） |
| `tools` | list[dict] | OpenAI function 格式的工具定义 |

---

## 三、文件清单与代码逻辑

### 3.1 目录结构

```
examples/carr_deepsearch/
├── __init__.py                              # 13 行
├── tools/
│   ├── __init__.py                          # 13 行
│   ├── carr_session_manager.py              # 109 行 — CaRR 工具服务器会话管理
│   ├── carr_browser_tool.py                 # 89 行  — 浏览器工具适配器
│   └── carr_agent_loop.py                   # 339 行 — 自定义 AgentLoop（核心）
├── reward/
│   ├── __init__.py                          # 13 行
│   ├── carr_reward.py                       # 105 行 — CaRR 奖励函数
│   └── cgrpo_advantage.py                   # 116 行 — C-GRPO 优势估计器
├── data_preprocess/
│   ├── preprocess_carr_rl.py                # 152 行 — RL 数据预处理
│   ├── preprocess_carr_sft.py               # 221 行 — SFT 数据预处理
│   └── preprocess_browsecomp.py             # 133 行 — BrowseComp 数据预处理
├── config/
│   ├── carr_grpo.yaml                       # 78 行  — C-GRPO RL 训练配置
│   ├── carr_sft.yaml                        # 48 行  — SFT 冷启动配置
│   └── tool_config/
│       └── carr_browser_tools.yaml          # 62 行  — 浏览器工具定义
└── scripts/
    ├── run_sft.sh                           # 20 行  — SFT 训练入口
    ├── run_rl.sh                            # 98 行  — RL 训练入口
    ├── run_eval_browsecomp.sh               # 123 行 — BrowseComp 评测入口
    └── smoke_test.py                        # 152 行 — 服务器连通性测试
```

### 3.2 工具层

#### `carr_session_manager.py` — 会话管理器（单例）

**职责**：管理 CaRR 工具服务器的 HTTP 会话生命周期。

```
CaRRSessionManager (单例)
├── _started_sessions: set          — 已启动的 session_id 集合
├── _session_data: dict             — 每个 session 的附加数据
├── ensure_started(session_id, url) — 懒启动：首次工具调用时自动发送 start_session
├── close(session_id, url)          — 发送 close_session 并清理
└── call_server(url, id, name, args, env_info) — 统一 HTTP POST 封装
```

**关键设计**：
- 三个浏览器工具共享同一个 session（CaRR 服务器要求相同 session_id 才能跨工具共享状态）
- 会话在首次 `execute()` 时懒启动，在 AgentLoop 的 `finally` 块中关闭（不在 `release()` 中关闭，因为 base class 每次工具调用都会 create→execute→release）

#### `carr_browser_tool.py` — 浏览器工具适配器

**职责**：将 verl BaseTool 接口适配到 CaRR 工具服务器。

```python
class CaRRBrowserTool(BaseTool):
    create()   → no-op（会话由 session_manager 管理）
    execute()  → 从 agent_data 获取 session_id → ensure_started → call_server
    release()  → no-op（会话由 agent loop finally 关闭）
```

**关键细节**：
- 同一个类被实例化 3 次（name 分别为 browser.search / browser.open / browser.find）
- `browser.open` 的 `id` 参数需要 `int()` 转换（CaRR 服务器用 `isinstance(idx, int)` 判断）
- `search_forbidden_strs` 从 `agent_data.tools_kwargs` 中提取，每次请求传入 `remote_env_info`

#### `carr_agent_loop.py` — 自定义 AgentLoop（最核心）

**职责**：扩展 ToolAgentLoop，维护 CaRR 格式的 reward_history。

**为什么需要自定义**：CaRR 奖励服务器的 `/evaluate` 端点要求特定的 `history` 格式（含 `tool_call_id` 绑定），而 verl 默认的 ToolAgentLoop 不维护这种格式。

```
@register("carr_tool_agent")    ← 通过 VERL_USE_EXTERNAL_MODULES 在启动时注册
class CaRRToolAgentLoop(ToolAgentLoop):
    run() — 重写主循环，并行维护 reward_history
```

**run() 状态机流程**：

```
PENDING → GENERATING ←→ PROCESSING_TOOLS → TERMINATED
   │           │              │                 │
   │           │              │                 ▼
   │           │              │           构建 output:
   │           │              │           extra_fields = {
   │           │              │             messages: reward_history,
   │           │              │             task_unfinished: bool,
   │           │              │           }
   │           ▼              ▼
   │    构建 assistant    构建 tool 条目
   │    reward_history    reward_history
   │    条目              条目
   ▼
调用 base handler
```

**CaRR reward_history 格式**（对比 verl 默认格式）：

```python
# CaRR 期望的格式（我们在 reward_history 中维护）
{"role": "assistant", "content": "...", "tool_calls": [
    {"tool_call_id": "xxx_tc_0_0", "name": "browser.search", "arguments": "{...}"}
]}
{"role": "tool", "content": [
    {"tool_call_id": "xxx_tc_0_0", "output": "search results..."}
]}

# verl 默认格式（agent_data.messages，用于 tokenization）
{"role": "assistant", "content": "...", "tool_calls": [
    {"type": "function", "id": "...", "function": {"name": "...", "arguments": "..."}}
]}
{"role": "tool", "content": "search results..."}  # 纯字符串
```

**三个关键修复**（code review 中发现并修复）：

1. **Stale tool_calls 清理**（line 127）：在调用 `_handle_generating_state()` 前清空 `agent_data.tool_calls = []`。原因：base handler 可能在到达 `extract_tool_calls` 之前就 TERMINATE，留下上一轮的旧值。

2. **task_unfinished 三重判定**（line 194-198）：
   ```python
   task_unfinished = (
       hit_limit                                      # 被 response_length/max_turns 截断
       or len(reward_history) == 0                     # 空历史（异常情况）
       or reward_history[-1].get("role") != "assistant"  # 最后消息不是 assistant
   )
   ```

3. **hit_limit 检测**（line 131-137, 176-177）：在 GENERATING 和 PROCESSING_TOOLS 两个状态中都检测截断。

**tool_call_id 生成**：确定性格式 `{request_id}_tc_{turn_idx}_{i}`（不用 uuid，方便调试）。

**会话清理**（line 186-189）：在 `finally` 块中调用 `session_manager.close()`。

### 3.3 奖励层

#### `carr_reward.py` — CaRR 奖励函数

**职责**：被 NaiveRewardManager 调用，将请求转发到 CaRR 奖励服务器。

```
NaiveRewardManager.run_single()
    ↓ 合并 extra_info（static parquet fields + dynamic agent loop fields）
    ↓
compute_score(data_source, solution_str, ground_truth, extra_info)
    ↓ 从 extra_info 提取 messages, rubrics, search_forbidden_strs, task_unfinished
    ↓ 构建 /evaluate payload
    ↓ POST to CARR_REWARD_SERVER_URL
    ↓
返回 {"score": outcome_reward, "outcome_reward": ..., "rubric_reward": ...}
    ↓ 三个 key 都传播到 non_tensor_batch（供 C-GRPO 使用）
```

**关键设计**：
- `score = outcome_reward`（不是混合后的 reward），C-GRPO 融合在 advantage 阶段做
- 650s 超时 > 服务器内部 600s 超时
- Fallback：如果 `messages` 为空（如单轮评测），构建最小 `[user, assistant]` history
- 环境变量：`CARR_REWARD_SERVER_URL`（默认 `http://localhost:8888`），`CARR_REWARD_TIMEOUT`（默认 650）

#### `cgrpo_advantage.py` — C-GRPO 优势估计器

**职责**：融合 outcome + rubric 两种奖励，然后调用标准 GRPO 归一化。

```
@register_adv_est("cgrpo")   ← 通过 VERL_USE_EXTERNAL_MODULES 注册
compute_cgrpo_advantage(token_level_rewards, response_mask, index, config, non_tensor_batch)
```

**C-GRPO 公式**（与 CaRR 论文一致）：

```
对于每个 prompt group g 内的 sample i：
  R̂_rubric_i = R_rubric_i / max(R_rubric in group g)      # 组内归一化
  R_i = (1-α) × R_outcome_i + α × R_outcome_i × R̂_rubric_i  # 乘性融合
```

**与服务器端混合的区别**：
- 服务器端：`reward = (1-α)*outcome + α*rubric`（线性加权）
- 我们的 C-GRPO：`R = (1-α)*outcome + α*outcome*norm_rubric`（乘性调节，rubric 是组内归一化后的）

**关键保护**（line 76-82）：
```python
assert not use_kl, (
    "C-GRPO advantage estimator is incompatible with use_kl_in_reward=True. "
    "KL penalty in token_level_rewards would be discarded by reward reconstruction."
)
```
原因：C-GRPO 用 `torch.zeros_like(token_level_rewards)` 重建 reward tensor，会丢弃预存的 KL penalty。

**Fallback**：如果 `non_tensor_batch` 中没有 `outcome_reward` / `rubric_reward`（如不使用 CaRR 奖励服务器的评测），退化为标准 GRPO。

### 3.4 数据预处理

#### `preprocess_carr_rl.py`

```
输入: CaRR/data/deepdive-rl-2k-rubrics.jsonl
输出: rl_train.parquet (2123), rl_val.parquet (111)
```

**转换逻辑**：
- `input_messages` → `prompt`
- `metadata.remote_env_info.{rubrics, search_forbidden_strs}` → `extra_info`
- 构建 `tools_kwargs`（browser.search 和 browser.open 都需要 `search_forbidden_strs`）
- 跳过 `search_forbidden_strs` 为空的记录
- 5% val split，seed=42

#### `preprocess_carr_sft.py`

```
输入: CaRR/data/deepdive-sft-glm46-trajectory-1k.jsonl
输出: sft_train.parquet (791), sft_val.parquet (41)
```

**格式转换**（CaRR → Qwen3/OpenAI）：

| 字段 | CaRR 格式 | 转换后格式 |
|------|-----------|-----------|
| assistant.tool_calls | `{tool_call_id, name, arguments: dict}` | `{type: "function", id, function: {name, arguments: json_string}}` |
| tool.content | `[{output: text}]`（列表） | `"text"`（纯字符串） |
| assistant.content | `null` | `""`（空字符串） |
| tools | `{name, description, parameters}` | `{type: "function", function: {name, description, parameters}}` |
| reasoning_content | 直传 | 直传 |

**可选验证**：`--model_name Qwen/Qwen3-4B` 加载 tokenizer 做 `apply_chat_template` 验证（需 GPU 环境）。

**已知待确认项**：CaRR 原始数据的 tool 消息不含 `tool_call_id`，当前按原样转换。如果 Qwen3 tokenizer 要求 tool 消息有 `tool_call_id`，需在 `convert_message()` 中从前一个 assistant 的 `tool_calls` 传播。

#### `preprocess_browsecomp.py`

```
输入: HuggingFace smolagents/browse_comp (test split, 1266 条)
输出: browsecomp_eval.parquet (1266)
```

**与 RL 的区别**：
- `rubric_reward_ratio = 0.0`（仅评估 outcome，不评估 rubric）
- `rubrics = []`（无 rubric 标注）
- `prompt = problem + FORMAT_SUFFIX`（含回答格式要求）
- `search_forbidden_strs[0] = problem`（不含 FORMAT_SUFFIX）

### 3.5 配置文件

#### `carr_grpo.yaml` — C-GRPO RL 训练配置

```yaml
# 关键配置项
algorithm:
  adv_estimator: cgrpo           # 使用自定义 C-GRPO 优势估计器
  cgrpo_alpha: 0.3               # rubric 混合权重
  use_kl_in_reward: false         # 必须 false（C-GRPO 不兼容）

actor_rollout_ref:
  rollout:
    name: sglang                  # SGLang 推理后端
    mode: async                   # 异步 rollout
    n: 16                         # 每个 prompt 采样 16 条
    multi_turn:
      enable: true
      format: hermes              # Qwen3 tool calling 格式
      max_assistant_turns: 30     # 最大工具调用轮数
      max_tool_response_length: 10000  # 与 CaRR 服务器截断一致
    agent:
      default_agent_loop: carr_tool_agent  # 使用自定义 AgentLoop

reward:
  custom_reward_function:
    path: examples/carr_deepsearch/reward/carr_reward.py
    name: compute_score

data:
  max_response_length: 61440     # 64k context
  train_batch_size: 128
```

#### `carr_sft.yaml` — SFT 冷启动配置

```yaml
model:
  partial_pretrain: Qwen/Qwen3-4B
  strategy: fsdp

data:
  max_length: 65536              # 64k
  multiturn:
    enable: true
    messages_key: messages        # 对应 SFT parquet 列名
    tools_key: tools
    enable_thinking_key: enable_thinking

use_remove_padding: true         # fsdp_sft_trainer 的顶层配置
```

#### `carr_browser_tools.yaml` — 工具定义

定义 3 个浏览器工具，全部使用 `CaRRBrowserTool` 类：
- `browser.search(query, num)` — 网页搜索
- `browser.open(id)` — 打开搜索结果
- `browser.find(pattern)` — 页面内查找

### 3.6 运行脚本

#### `run_sft.sh` — SFT 训练

```bash
# 自动预处理（如果数据不存在）→ torchrun SFT
torchrun --nnodes=1 --nproc_per_node=8 \
    -m verl.trainer.fsdp_sft_trainer \
    --config-path=examples/carr_deepsearch/config --config-name=carr_sft
```

#### `run_rl.sh` — RL 训练

```bash
# 1. 检查环境变量：SERPAPI_API_KEY, JINA_API_KEY, DEEPSEEK_API_KEY
# 2. 设置 VERL_USE_EXTERNAL_MODULES（注册 carr_tool_agent + cgrpo）
# 3. 从 SFT checkpoint 解析最新 global_step
# 4. 自动预处理 RL 数据
# 5. 启动 CaRR 工具服务器（7230）+ 奖励服务器（8888）
# 6. curl -sf 健康检查（带 HTTP 错误检测）
# 7. python -m verl.trainer.main_ppo
```

#### `run_eval_browsecomp.sh` — BrowseComp 评测

```bash
# 用法: run_eval_browsecomp.sh <model_path> [max_samples] [context]
#   context: 64k → max_response_length=61440
#            128k → max_response_length=122880
# 使用 trainer.val_only=True 模式运行 main_ppo（仅做推理+评测）
```

#### `smoke_test.py` — 服务器连通性测试

```bash
python smoke_test.py --tool     # 测试工具服务器 (search → open → find)
python smoke_test.py --reward   # 测试奖励服务器 (/evaluate)
python smoke_test.py --all      # 测试全部
```

---

## 四、verl 核心修改（已合并，无需操作）

以下两处修改已在 `feature/carr-deepsearch` 分支中完成：

### 4.1 AlgoConfig 新增 `cgrpo_alpha`

```python
# verl/trainer/config/algorithm.py:615
cgrpo_alpha: float = 0.3
```

### 4.2 non_tensor_batch 传递到 advantage estimator

```python
# verl/trainer/ppo/ray_trainer.py:215-220
# compute_advantage() 方法中，通过 inspect 检测 advantage function 是否接受 non_tensor_batch 参数
# 如果接受，将 non_tensor_batch（含 outcome_reward, rubric_reward）传入
```

---

## 五、数据流图

```
┌─────────────────────────────────────────────────────────────────┐
│                        训练数据流                                │
│                                                                 │
│  JSONL ──preprocess──→ Parquet ──DataLoader──→ DataProto       │
│                                                                 │
│  DataProto.non_tensor_batch = {                                │
│      prompt, agent_name, extra_info{                            │
│          rubrics, search_forbidden_strs,                        │
│          rubric_reward_ratio, tools_kwargs,                     │
│      }                                                          │
│  }                                                              │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      Rollout 阶段                                │
│                                                                 │
│  CaRRToolAgentLoop.run()                                       │
│      │                                                          │
│      ├─→ GENERATING: 模型生成 → 解析 tool_calls                 │
│      │   └─→ 构建 reward_history assistant 条目                  │
│      │                                                          │
│      ├─→ PROCESSING_TOOLS: 调用 CaRRBrowserTool.execute()      │
│      │   └─→ CaRRSessionManager → CaRR Tool Server (7230)      │
│      │   └─→ 构建 reward_history tool 条目（含 tool_call_id）     │
│      │                                                          │
│      └─→ TERMINATED: 输出 extra_fields = {                      │
│              messages: reward_history,                           │
│              task_unfinished: bool,                              │
│          }                                                      │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      Reward 阶段                                 │
│                                                                 │
│  NaiveRewardManager.run_single()                                │
│      │ 合并 extra_info = static (parquet) + dynamic (agent loop) │
│      ↓                                                          │
│  compute_score(extra_info)                                      │
│      │ POST /evaluate → CaRR Reward Server (8888)               │
│      ↓                                                          │
│  返回 {score, outcome_reward, rubric_reward}                     │
│      │ 传播到 non_tensor_batch                                   │
│      ↓                                                          │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    Advantage 阶段                                │
│                                                                 │
│  compute_cgrpo_advantage(token_level_rewards, non_tensor_batch) │
│      │                                                          │
│      ├─ 从 non_tensor_batch 提取 outcome_reward, rubric_reward  │
│      ├─ 组内归一化: R̂_rubric = R_rubric / max(group)            │
│      ├─ C-GRPO 融合: R = (1-α)*outcome + α*outcome*R̂_rubric    │
│      ├─ 放置到 last valid token                                  │
│      └─ 调用 compute_grpo_outcome_advantage() 做标准 GRPO 归一化 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 六、外部依赖

### 6.1 API Key

| 环境变量 | 用途 | 消耗者 |
|----------|------|--------|
| `SERPAPI_API_KEY` | Google 搜索 API | CaRR 工具服务器 (browser.search) |
| `JINA_API_KEY` | Jina Reader 网页解析 | CaRR 工具服务器 (browser.open) |
| `DEEPSEEK_API_KEY` | LLM Judge | CaRR 奖励服务器 (outcome/rubric 评分) |

### 6.2 外部服务器

| 服务 | 端口 | 启动命令 | 启动目录 |
|------|------|---------|---------|
| CaRR 工具服务器 | 7230 | `python CaRR/tool_server/launch_server.py` | 项目根目录 |
| CaRR 奖励服务器 | 8888 | `python launch_server.py` | `CaRR/deepsearch_rm_with_rubrics/` |

**注意**：奖励服务器必须在 `CaRR/deepsearch_rm_with_rubrics/` 目录下启动（依赖 `./prompts/` 目录）。

### 6.3 模块注册

RL 训练和评测必须设置：
```bash
export VERL_USE_EXTERNAL_MODULES=examples.carr_deepsearch.tools.carr_agent_loop,examples.carr_deepsearch.reward.cgrpo_advantage
```

这会在 verl 启动时导入这两个模块，触发 `@register("carr_tool_agent")` 和 `@register_adv_est("cgrpo")`。

---

## 七、本地验证结果（Gate 1）

| 检查项 | 结果 |
|--------|------|
| Python 语法（ast.parse） | 12/12 文件通过 |
| YAML 验证（yaml.safe_load） | 3/3 配置通过 |
| Shell 语法（bash -n） | 3/3 脚本通过 |
| verl import 路径（文件存在性） | 8/8 模块存在 |
| RL 预处理 | 2234 → 2123 train + 111 val，schema 正确 |
| SFT 预处理 | 832 → 791 train + 41 val，格式转换正确 |
| BrowseComp 预处理 | 1266 → 1266 eval，schema 与 RL 对齐 |

---

## 八、下一阶段 TODO（按设备分层）

> 完整的设备策略见 `IMPLEMENTATION_PLAN_DEVICE_ORDER.md`。当前本地 Mac 代码开发已完成，下一步直接进入同一台 `2 x 5090` 机器完成 Gate 2-5。

数据文件已在仓库内（`examples/carr_deepsearch/data/`），config 已指向该路径，无需额外预处理。

### 8.1 阶段 BC — `2 x 5090` 联调 + 小规模冒烟（Gate 2/3/4/5）

**目标**：在正式训练前解决所有关键问题。先证明链路可用，再证明小规模训练和 quick eval 可执行。

**执行顺序（一次租机连续完成）**：

1. **环境搭建与模块注册**
```bash
pip install -e .[test,sglang]
# 验证自定义模块能正确注册
VERL_USE_EXTERNAL_MODULES=examples.carr_deepsearch.tools.carr_agent_loop,examples.carr_deepsearch.reward.cgrpo_advantage \
    python -c "from verl.experimental.agent_loop.agent_loop import _AGENT_LOOP_REGISTRY; print(_AGENT_LOOP_REGISTRY.keys())"
```

2. **服务连通性**
```bash
# 启动 CaRR 工具服务器 + 奖励服务器（需要 SERPAPI / JINA / DEEPSEEK API key）
# 然后运行：
python examples/carr_deepsearch/scripts/smoke_test.py --all
```

3. **单样本 rollout 链路（最关键）**
- 用极小 batch 跑一次 `main_ppo`，验证 `generate → tool_call → tool_server → reward_server → advantage` 全链路
- 重点检查：
  - `history` 格式是否被 CaRR reward server 接受（非格式性全零）
  - `tool_call_id` 绑定是否正确
  - `non_tensor_batch` 中是否出现 `outcome_reward` 和 `rubric_reward`
  - `cgrpo` advantage estimator 是否被调用

**待确认项（在本阶段尽早排除）**：SFT parquet 中 tool 消息缺少 `tool_call_id` 字段。如果 Qwen3 tokenizer 报错，需在 `preprocess_carr_sft.py` 的 `convert_message()` 中从前一个 assistant 的 tool_calls 传播。可先运行 `--model_name Qwen/Qwen3-4B` 验证：
```bash
python examples/carr_deepsearch/data_preprocess/preprocess_carr_sft.py \
    --input_file CaRR/data/deepdive-sft-glm46-trajectory-1k.jsonl \
    --output_dir examples/carr_deepsearch/data \
    --model_name Qwen/Qwen3-4B
```

4. **SFT 微型冒烟**
```bash
bash examples/carr_deepsearch/scripts/run_sft.sh \
    trainer.n_gpus_per_node=2 \
    data.train_batch_size=2 \
    data.max_length=16384 \
    trainer.total_epochs=1
```

5. **BrowseComp 快速评测冒烟**（用 SFT checkpoint）
```bash
bash examples/carr_deepsearch/scripts/run_eval_browsecomp.sh <sft_ckpt_path> 5 64k \
    trainer.n_gpus_per_node=2
```

6. **GRPO baseline 微型冒烟**
```bash
bash examples/carr_deepsearch/scripts/run_rl.sh \
    algorithm.adv_estimator=grpo \
    trainer.n_gpus_per_node=2 \
    data.train_batch_size=8 \
    actor_rollout_ref.rollout.n=2 \
    data.max_response_length=16384 \
    trainer.total_epochs=1
```

7. **C-GRPO 微型冒烟**
```bash
bash examples/carr_deepsearch/scripts/run_rl.sh \
    trainer.n_gpus_per_node=2 \
    data.train_batch_size=8 \
    actor_rollout_ref.rollout.n=2 \
    data.max_response_length=16384 \
    trainer.total_epochs=1
```

**本阶段退出条件**：
- FSDP / `torchrun` 是否正常
- rollout 是否能稳定结束（不死锁、不 OOM）
- `run_eval_browsecomp.sh` 能否输出 reward 指标
- 训练 loss 是否在下降（不要求收敛，但不能是 NaN 或常数）
- Gate 2/3/4/5 全部通过后，才进入 8 卡阶段

### 8.2 阶段 D — 8 卡正式训练与评测（Gate 6/7）

> **进入条件**：Gate 1~5 全部通过后才租 8 卡。此时不应再排查 history 格式、tool_call_id 绑定、estimator 注册、shell 路径等低级问题。

**正式 SFT**：
```bash
bash examples/carr_deepsearch/scripts/run_sft.sh
```

**正式 RL（三条线）**：

| 实验 | 命令差异 |
|------|---------|
| GRPO baseline | `bash run_rl.sh algorithm.adv_estimator=grpo` |
| C-GRPO | `bash run_rl.sh`（默认配置） |

**正式 BrowseComp 评测（三条线）**：
```bash
# SFT-only
bash run_eval_browsecomp.sh <sft_ckpt> -1 64k

# GRPO baseline
bash run_eval_browsecomp.sh <grpo_ckpt> -1 64k

# C-GRPO
bash run_eval_browsecomp.sh <cgrpo_ckpt> -1 64k
```

**最终产出**：SFT / GRPO / C-GRPO 三模型在 BrowseComp 上的对比表。

### 8.3 Gate 定义速查

| Gate | 说明 | 设备 | 前置 |
|------|------|------|------|
| Gate 1 | 数据预处理通过 | Mac | — |
| Gate 2 | 工具链路通过（search → open → find） | `2 x 5090` | Gate 1 |
| Gate 3 | 奖励链路通过（reward 非格式性全零） | `2 x 5090` | Gate 1 |
| Gate 4 | 小规模 SFT + RL 可跑通 | `2 x 5090` | Gate 2, 3 |
| Gate 5 | 小规模 BrowseComp eval 可输出指标 | `2 x 5090` | Gate 4 |
| Gate 6 | 正式 BrowseComp 评测完成 | 8 卡 | Gate 5 |
| Gate 7 | 三模型对比表完成 | 8 卡 | Gate 6 |

---

## 九、参考文件索引

| 文件 | 用途 |
|------|------|
| `IMPLEMENTATION_PLAN.md` | 完整技术设计文档 |
| `CODE_WALKTHROUGH.md` | CaRR 服务器 + verl 工具系统的详细代码讲解 |
| `IMPLEMENTATION_PLAN_DEVICE_ORDER.md` | 按设备阶段排序的实施计划 |
| `verl/experimental/agent_loop/tool_agent_loop.py` | ToolAgentLoop 基类（理解状态机） |
| `verl/trainer/ppo/core_algos.py` | GRPO 优势估计器基础 |
| `verl/experimental/reward_loop/reward_manager/naive.py` | NaiveRewardManager 调用链 |
