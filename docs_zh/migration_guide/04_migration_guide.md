# MultihopQA → verl ToolAgentLoop 详细迁移指南

本文档逐步说明将 Agent-R1 的 MultihopQA Agent 迁移到 verl 原生 ToolAgentLoop 架构所需修改/创建的每个文件，包含具体的修改思路和代码示例。

---

## 迁移概览

### 需要新建的文件

| 文件 | 用途 |
|------|------|
| `verl/tools/wiki_search_tool.py` | WikiSearch 工具适配 |
| `config/tool_config/wiki_search_tool_config.yaml` | 工具配置 |
| `scripts/preprocess_hotpotqa_verl.py` | HotpotQA 数据预处理 |
| `reward_functions/multihopqa_reward.py` | 奖励函数 |
| `config/hotpotqa_multiturn_grpo.yaml` | 训练配置 |
| `scripts/run_hotpotqa_grpo.sh` | 训练脚本 |

### 不需要迁移的组件（verl 已内置）

| Agent-R1 组件 | verl 替代 |
|--------------|----------|
| `ToolGenerationManager` (generation.py, 849行) | `ToolAgentLoop` (tool_agent_loop.py, 479行) |
| `NousToolEnv` (nous.py) | `ToolAgentLoop` + `HermesToolParser` |
| `RayAgentTrainer` (agent_ray_trainer.py) | `RayPPOTrainer` (ray_trainer.py) |
| `ToolRLDataset` (agent_rl_dataset.py) | `RLHFDataset` + `agent_name`/`tools_kwargs` |

---

## 1. 工具适配：WikiSearchTool

### 源文件
`agent-R1/agent_r1/tool/tools/wiki_search_tool.py`

### 目标文件
新建 `verl/tools/wiki_search_tool.py`

### 修改思路

**核心变化**：
1. 从 Agent-R1 的 `BaseTool(name, description, parameters, execute)` 转为 verl 的 `BaseTool(config, tool_schema)`
2. 将同步 `execute(args) → dict` 改为 async `execute(instance_id, parameters) → (ToolResponse, reward, metrics)`
3. 添加 `create/release` 生命周期
4. 添加 `TokenBucketWorker` 并发控制（参考 `verl/tools/search_tool.py`）
5. API 调用逻辑可直接复用

### 示例代码

```python
# verl/tools/wiki_search_tool.py

import json
import logging
import os
import threading
from contextlib import ExitStack
from typing import Any, Optional
from uuid import uuid4

import ray
import requests

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# 复用 search_tool.py 的并发控制机制
@ray.remote(concurrency_groups={"acquire": 1, "release": 10})
class TokenBucketWorker:
    def __init__(self, rate_limit: int):
        self.rate_limit = rate_limit
        self.current_count = 0
        self._semaphore = threading.Semaphore(rate_limit)

    @ray.method(concurrency_group="acquire")
    def acquire(self):
        self._semaphore.acquire()
        self.current_count += 1

    @ray.method(concurrency_group="release")
    def release(self):
        self._semaphore.release()
        self.current_count -= 1


class WikiSearchExecutionWorker:
    def __init__(self, enable_global_rate_limit=True, rate_limit=10):
        self.rate_limit_worker = (
            TokenBucketWorker.options(
                name="wiki-search-rate-limiter", get_if_exists=True
            ).remote(rate_limit)
            if enable_global_rate_limit
            else None
        )

    def execute(self, fn, *args, **kwargs):
        if self.rate_limit_worker:
            with ExitStack() as stack:
                stack.callback(self.rate_limit_worker.release.remote)
                ray.get(self.rate_limit_worker.acquire.remote())
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"Error executing wiki search: {e}")
        else:
            return fn(*args, **kwargs)


class WikiSearchTool(BaseTool):
    """Wikipedia 搜索工具，适配 verl BaseTool 接口。"""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}

        # 配置参数
        self.api_url = config.get("api_url", "http://localhost:8000")
        self.num_workers = config.get("num_workers", 60)
        self.rate_limit = config.get("rate_limit", 60)
        self.top_k = config.get("top_k", 5)
        self.enable_global_rate_limit = config.get("enable_global_rate_limit", True)

        # 初始化执行池
        self.execution_pool = (
            ray.remote(WikiSearchExecutionWorker)
            .options(max_concurrency=self.num_workers)
            .remote(
                enable_global_rate_limit=self.enable_global_rate_limit,
                rate_limit=self.rate_limit,
            )
        )

        logger.info(f"WikiSearchTool initialized: api_url={self.api_url}")

    async def create(self, instance_id=None, **kwargs):
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {"reward": []}
        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(self, instance_id, parameters, **kwargs):
        """
        执行搜索查询。

        parameters 格式: {"query": "search query string"}
        注意: Agent-R1 的 WikiSearchTool 用 {"query": str}
              verl 已有的 SearchTool 用 {"query_list": list}
              这里保持与 Agent-R1 一致用 {"query": str}
        """
        query = parameters.get("query", "").strip()
        if not query:
            return ToolResponse(text="Error: empty query"), 0.0, {}

        try:
            result_text = await self.execution_pool.execute.remote(
                self._search, query
            )
            if result_text is None:
                return ToolResponse(text="Error: search failed"), 0.0, {}
            return ToolResponse(text=result_text), None, {}
        except Exception as e:
            return ToolResponse(text=f"Error: {e}"), 0.0, {}

    def _search(self, query):
        """同步搜索（在 Ray 线程池中执行）"""
        response = requests.get(
            f"{self.api_url}/search",
            params={"query": query, "top_k": self.top_k},
        )
        if response.status_code == 200:
            result = response.json()
            return self._format_results(result)
        else:
            return f"Search API error: {response.status_code}"

    def _format_results(self, api_result):
        """复用 Agent-R1 的格式化逻辑"""
        results_list = []
        if "query_results" in api_result and len(api_result["query_results"]) > 0:
            query_result = api_result["query_results"][0]
            for result in query_result["results"]:
                document = result["document"]
                clean_result = {
                    "content": document.get("contents", ""),
                    "title": document.get("title", ""),
                }
                results_list.append(clean_result)
        elif "results" in api_result:
            for result in api_result["results"]:
                if "document" in result:
                    document = result["document"]
                    clean_result = {
                        "content": document.get("contents", ""),
                        "title": document.get("title", ""),
                    }
                    results_list.append(clean_result)
                else:
                    results_list.append(result)

        return json.dumps({"results": results_list}, ensure_ascii=False)

    async def calc_reward(self, instance_id, **kwargs):
        return 0.0

    async def release(self, instance_id, **kwargs):
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
```

### 关键注意事项

1. **参数名 `query` vs `query_list`**：Agent-R1 的 WikiSearchTool 接受 `{"query": str}`，verl 已有的 SearchTool 接受 `{"query_list": list[str]}`。新的 WikiSearchTool 应保持 `{"query": str}` 格式，因为这与 HotpotQA 的训练数据和模型行为一致。

2. **tool_reward 返回 None**：搜索工具没有 step-level reward（搜索结果的质量难以即时评估），所以返回 `None`。最终奖励由外部 reward function 计算。

3. **_format_results 直接复用**：Agent-R1 的格式化逻辑可以直接复用，保持搜索结果的输出格式一致。

---

## 2. 工具配置 YAML

### 目标文件
新建 `config/tool_config/wiki_search_tool_config.yaml`

### 参考
`examples/sglang_multiturn/config/tool_config/search_tool_config.yaml`

### 内容

```yaml
tools:
  - class_name: verl.tools.wiki_search_tool.WikiSearchTool
    config:
      api_url: http://localhost:8000     # KILT 搜索服务地址
      num_workers: 60                     # 并发线程数
      rate_limit: 60                      # 全局速率限制
      top_k: 5                           # 返回 top-k 结果
      enable_global_rate_limit: true
    tool_schema:
      type: function
      function:
        name: search
        description: >
          Search for information on the internet using Wikipedia as a knowledge source.
          Use this tool to find facts, dates, names, and other information needed to
          answer multi-hop questions.
        parameters:
          type: object
          properties:
            query:
              type: string
              description: The search query to look up
          required:
            - query
```

### 关键注意事项

1. **function name = "search"**：必须与 Agent-R1 中 WikiSearchTool 的 `name = "search"` 保持一致，否则模型生成的 `<tool_call>{"name": "search", ...}` 会匹配不到工具。

2. **description 的重要性**：tool description 会通过 chat template 注入到 system prompt 中，影响模型的工具调用行为。建议先保持与 Agent-R1 一致，后续微调。

3. **api_url 参数**：替代了 Agent-R1 中的 `WIKI_SEARCH_API_URL` 环境变量，改为在配置文件中显式指定。

---

## 3. 数据预处理

### 源文件
`agent-R1/examples/data_preprocess/hotpotqa.py`

### 目标文件
新建 `scripts/preprocess_hotpotqa_verl.py`

### 修改思路

1. 添加 `agent_name: "tool_agent"` 字段
2. 保持 `reward_model.ground_truth` 不变
3. 添加 `extra_info` 字段（包含 index 等信息）
4. 可选：添加 system prompt

### 示例代码

```python
# scripts/preprocess_hotpotqa_verl.py

import os
import json
import random
import argparse
import datasets
import requests
from tqdm import tqdm


def download_file(url, local_path):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    with open(local_path, "wb") as file:
        for data in tqdm(response.iter_content(1024), total=total_size // 1024):
            file.write(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/hotpotqa_verl")
    parser.add_argument("--train_size", type=int, default=25600)
    parser.add_argument("--val_size", type=int, default=128)
    args = parser.parse_args()

    data_source = "hotpotqa/hotpot_qa"
    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    # 下载数据（复用 Agent-R1 的下载逻辑）
    train_url = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json"
    dev_url = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
    train_file = os.path.join(local_dir, "hotpot_train_v1.1.json")
    dev_file = os.path.join(local_dir, "hotpot_dev_distractor_v1.json")

    if not os.path.exists(train_file):
        download_file(train_url, train_file)
    if not os.path.exists(dev_file):
        download_file(dev_url, dev_file)

    with open(train_file, "r") as f:
        train_data = json.load(f)
    with open(dev_file, "r") as f:
        validation_data = json.load(f)

    train_dataset = datasets.Dataset.from_dict({
        "question": [item["question"] for item in train_data],
        "answer": [item["answer"] for item in train_data],
    })
    validation_dataset = datasets.Dataset.from_dict({
        "question": [item["question"] for item in validation_data],
        "answer": [item["answer"] for item in validation_data],
    })

    if args.train_size:
        indices = random.sample(range(len(train_dataset)), min(args.train_size, len(train_dataset)))
        train_dataset = train_dataset.select(indices)
    if args.val_size:
        indices = random.sample(range(len(validation_dataset)), min(args.val_size, len(validation_dataset)))
        validation_dataset = validation_dataset.select(indices)

    instruction_following = (
        "You FIRST think about the reasoning process as an internal monologue "
        "and then provide the final answer. "
        "The reasoning process MUST BE enclosed within <think> </think> tags. "
        "The final answer MUST BE put in <answer> </answer> tags."
    )

    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop("question")
            question = "Question: " + question_raw + "\n" + instruction_following
            answer_raw = example.pop("answer")

            data = {
                "data_source": data_source,
                "agent_name": "tool_agent",  # ← 新增：指定使用 ToolAgentLoop
                "prompt": [
                    # 可选：添加 system prompt
                    # {"role": "system", "content": "You are a helpful assistant..."},
                    {
                        "role": "user",
                        "content": question,
                    },
                ],
                "ability": "multihop_qa",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer_raw,
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    # WikiSearchTool 不需要 create_kwargs
                    # 但如果未来添加有状态工具，可以在这里传递
                    # "need_tools_kwargs": True,
                    # "tools_kwargs": {...},
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    validation_dataset = validation_dataset.map(function=make_map_fn("validation"), with_indices=True)

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    validation_dataset.to_parquet(os.path.join(local_dir, "validation.parquet"))
    print(f"Saved to {local_dir}")
```

### 与 Agent-R1 版本的差异

| 字段 | Agent-R1 | verl |
|------|----------|------|
| `agent_name` | 无 | `"tool_agent"` |
| `extra_info` | 被注释掉了 | 添加 `split`, `index` |
| `tools_kwargs` | 不需要（WikiSearchTool 无状态） | 可选（如需传 create_kwargs） |
| System prompt | 由 ToolRLDataset 自动注入 | 可选添加到 prompt 中 |

### 关键注意事项

1. **`agent_name: "tool_agent"`** 是必须的，否则 verl 会使用默认的 SingleTurnAgentLoop。

2. **System prompt 的选择**：
   - **不加 system prompt**（推荐）：工具描述由 ToolAgentLoop 的 `apply_chat_template(tools=schemas)` 自动注入
   - **加 system prompt**：可以添加额外的任务指令，但要注意不要与自动注入的工具描述冲突

3. **tools_kwargs**：WikiSearchTool 是无状态的（不需要 ground_truth 初始化），所以不需要 `need_tools_kwargs` 和 `tools_kwargs`。如果工具需要 per-sample 初始化参数（如 sandbox session），则需要这些字段。

---

## 4. 奖励函数

### 源文件
`agent-R1/agent_r1/src/reward_score/qa_em_and_format.py`

### 目标文件
新建 `reward_functions/multihopqa_reward.py`

### 修改思路

1. 核心评分逻辑（`normalize_answer`, `em_check`, `subem_check`, `extract_solution`, `compute_score_format`, `compute_score_answer`, `compute_score_format_answer`）**直接复用**
2. 包装为 verl 的 `custom_reward_function` 格式
3. **重要**：确认 verl ToolAgentLoop 生成的轨迹格式是否与 Agent-R1 一致（影响 format check 中的正则表达式）

### 示例代码

```python
# reward_functions/multihopqa_reward.py

import re
import string
from verl import DataProto


def normalize_answer(s):
    """标准化答案文本"""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    for golden_answer in golden_answers:
        if normalize_answer(golden_answer) == normalized_prediction:
            return 1.0
    return 0.0


def subem_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    for golden_answer in golden_answers:
        if normalize_answer(golden_answer) in normalized_prediction:
            return 1.0
    return 0.0


def extract_solution(solution_str):
    """从 <answer>...</answer> 标签中提取答案"""
    match = re.search(r'<answer>(.*?)</answer>', solution_str, re.DOTALL)
    return match.group(1).strip() if match else None


def compute_score_format(solution_str):
    """格式奖励：检查 <think>/<tool_call>/<answer> 结构"""
    if solution_str is None:
        return 0.0

    try:
        # 提取 assistant blocks
        # 注意：这里的正则可能需要根据 verl 的实际输出格式调整
        # Qwen chat template 使用 <|im_start|>assistant 和 <|im_end|>
        assistant_blocks = re.findall(
            r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>',
            solution_str, re.DOTALL
        )

        format_reward = 0.0
        if not assistant_blocks:
            return 0.0

        # 中间 blocks: <think>...</think><tool_call>...</tool_call>
        for i, block in enumerate(assistant_blocks[:-1]):
            if (block.count('<think>') == 1 and block.count('</think>') == 1
                    and block.count('<tool_call>') == 1
                    and block.count('</tool_call>') == 1):
                think_match = re.search(
                    r'^<think>(.*?)</think>(\s*)<tool_call>(.*?)</tool_call>$',
                    block, re.DOTALL
                )
                if think_match:
                    format_reward += 0.5

        # 最后 block: <think>...</think>...<answer>...</answer>
        last_block = assistant_blocks[-1]
        think_answer_match = re.search(
            r'^<think>(.*?)</think>(.*?)<answer>(.*?)</answer>$',
            last_block, re.DOTALL
        )
        if think_answer_match:
            format_reward += 0.5

    except Exception as e:
        return 0.0

    return format_reward


def compute_score_answer(solution_str, ground_truth):
    """答案奖励：SubEM 匹配"""
    if solution_str is None:
        return 0.0

    try:
        assistant_blocks = re.findall(
            r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>',
            solution_str, re.DOTALL
        )
        if not assistant_blocks:
            return 0.0

        solution_str = assistant_blocks[-1]
        answer = extract_solution(solution_str)

        answer_reward = 0.0
        if answer is not None:
            if subem_check(answer, ground_truth):
                answer_reward = 1.0

        if answer_reward == 0.0:
            if subem_check(solution_str, ground_truth):
                answer_reward = 0.2

    except Exception:
        return 0.0

    return answer_reward


def compute_score_format_answer(solution_str, ground_truth):
    """组合奖励 = -1.0 + format_reward + answer_reward"""
    if solution_str is None or ground_truth is None:
        return 0.0

    try:
        format_reward = compute_score_format(solution_str)
        answer_reward = compute_score_answer(solution_str, ground_truth)

        format_reward = min(format_reward, 1.0)
        if format_reward >= 0.5:
            return -1.0 + format_reward + answer_reward
        else:
            return -1.0 + format_reward
    except Exception:
        return -1.0


# ============================================================
# verl custom_reward_function 入口
# ============================================================

def compute_score(data: DataProto, tokenizer) -> DataProto:
    """
    verl custom_reward_function 接口。

    参数:
        data: DataProto，包含 batch 和 non_tensor_batch
        tokenizer: HuggingFace tokenizer

    返回:
        data: 添加了 reward_tensor 的 DataProto
    """
    import torch

    reward_tensor = torch.zeros_like(
        data.batch["responses"], dtype=torch.float32
    )

    for i in range(len(data)):
        data_item = data[i]

        # 解码完整轨迹
        prompt_ids = data_item.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]
        valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
        valid_prompt_ids = prompt_ids[-valid_prompt_length:]

        response_ids = data_item.batch["responses"]
        valid_response_length = (
            data_item.batch["attention_mask"][prompt_length:].sum()
        )
        valid_response_ids = response_ids[:valid_response_length]

        sequences = torch.cat((valid_prompt_ids, valid_response_ids))
        sequences_str = tokenizer.decode(sequences, skip_special_tokens=False)

        # 获取 ground truth
        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]

        # 计算奖励
        score = compute_score_format_answer(sequences_str, ground_truth)

        # 放在最后一个有效 response token 位置
        reward_tensor[i, valid_response_length - 1] = score

    data.batch["reward_tensor"] = reward_tensor
    return data
```

### 关键注意事项

1. **轨迹格式可能不同**：Agent-R1 的 `format_tool_response` 手动拼接了 `<|im_start|>user` + `<tool_response>` 标签，而 verl 用 `role="tool"` message 通过 `apply_chat_template` 生成。这意味着解码后的文本格式**可能不同**。

   **验证方法**：在迁移后的第一次训练中，打印几个解码后的 `sequences_str`，检查 `<|im_start|>assistant` 的结构是否仍然被正确匹配。

2. **格式正则可能需要调整**：如果 verl 的 chat template 对 tool role 生成了不同的 token 序列（例如用 `<|im_start|>tool` 而不是 `<|im_start|>user`），`compute_score_format` 中提取 assistant blocks 的正则可能需要调整。

3. **compute_score 接口**：verl 的 `custom_reward_function` 接口接收 `(data: DataProto, tokenizer)` 并返回修改后的 `DataProto`（添加了 `reward_tensor`）。这与 Agent-R1 的 `AgentRewardManager.__call__` 类似但签名不同。

---

## 5. 训练配置

### 目标文件
新建 `config/hotpotqa_multiturn_grpo.yaml`

### 参考
`examples/sglang_multiturn/config/gsm8k_multiturn_grpo.yaml`

### 内容

```yaml
hydra:
  searchpath:
    - file://verl/trainer/config

defaults:
  - ppo_trainer
  - _self_

data:
  max_prompt_length: 8192         # 与 Agent-R1 一致
  max_response_length: 8192       # 与 Agent-R1 一致
  train_batch_size: 128           # 与 Agent-R1 一致
  return_raw_chat: True           # 必须为 True
  shuffle: True

actor_rollout_ref:
  hybrid_engine: True
  model:
    path: Qwen/Qwen2.5-7B-Instruct
  rollout:
    name: sglang                  # 或 vllm
    multi_turn:
      enable: True
      max_assistant_turns: 5      # 对应 Agent-R1 的 tool.max_turns=5
      max_tool_response_length: 2048  # 对应 Agent-R1 的 tool.max_tool_response_length
      format: hermes              # 使用 HermesToolParser
```

### Agent-R1 → verl 配置字段映射

```
# Agent-R1:
tool.max_turns=5
→ actor_rollout_ref.rollout.multi_turn.max_assistant_turns=5

# Agent-R1:
tool.tools=['wiki_search']
→ actor_rollout_ref.rollout.multi_turn.tool_config_path=config/tool_config/wiki_search_tool_config.yaml

# Agent-R1:
tool.max_tool_response_length=2048
→ actor_rollout_ref.rollout.multi_turn.max_tool_response_length=2048

# Agent-R1:
data.use_default_tool_template=True
→ data.return_raw_chat=True (chat template 自动注入工具)

# Agent-R1:
actor_rollout_ref.rollout.stop_token_ids=[151658]
→ 不需要（ToolAgentLoop 由 ToolParser 控制停止）

# Agent-R1:
actor_rollout_ref.rollout.n_repeat=5
→ actor_rollout_ref.rollout.n=5

# Agent-R1:
algorithm.adv_estimator=grpo
→ algorithm.adv_estimator=grpo (完全相同)
```

---

## 6. 训练脚本

### 目标文件
新建 `scripts/run_hotpotqa_grpo.sh`

### 参考
`examples/sglang_multiturn/run_qwen2.5-3b_gsm8k_tool_agent_mlflow.sh`

### 内容

```bash
#!/bin/bash

export BASE_MODEL='Qwen/Qwen2.5-7B-Instruct'
export PROJECT_NAME='multihopqa-verl'
export EXPERIMENT_NAME='grpo-qwen2.5-7b-instruct'

# 确保 KILT 搜索服务已启动
# export WIKI_SEARCH_API_URL=http://localhost:8000

python3 -m verl.trainer.main_ppo \
    --config-path=../../config \
    --config-name=hotpotqa_multiturn_grpo \
    \
    data.train_files=$HOME/data/hotpotqa_verl/train.parquet \
    data.val_files=$HOME/data/hotpotqa_verl/validation.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=8192 \
    data.max_response_length=8192 \
    \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=5 \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=2048 \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=config/tool_config/wiki_search_tool_config.yaml \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.001 \
    \
    custom_reward_function.path=reward_functions.multihopqa_reward.compute_score \
    \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=10 \
    trainer.total_epochs=1 \
    trainer.val_before_train=True \
    "$@"
```

### 与 Agent-R1 脚本的关键差异

| 方面 | Agent-R1 | verl |
|------|----------|------|
| 入口 | `python3 -m agent_r1.src.main_agent` | `python3 -m verl.trainer.main_ppo` |
| 配置基础 | `agent_trainer.yaml` | `ppo_trainer.yaml` |
| 工具配置 | `tool.tools=['wiki_search']` | `multi_turn.tool_config_path=xxx.yaml` |
| 奖励函数 | `compute_score` 内置在 AgentRewardManager | `custom_reward_function.path=...` |
| 推理后端 | `rollout.name=vllm` | `rollout.name=sglang` (也支持 vllm) |
| stop token | `rollout.stop_token_ids=[151658]` | 不需要（ToolParser 处理） |
| rollout 模式 | 同步 | `rollout.mode=async` |
| 重复采样 | `rollout.n_repeat=5` | `rollout.n=5` |

---

## 7. 迁移执行检查清单

### 第一阶段：基础设施准备

- [ ] 启动 KILT 搜索服务（参考 agent-R1 文档）
- [ ] 安装 verl + SGLang 依赖
- [ ] 确认 Qwen2.5-7B-Instruct 模型已下载

### 第二阶段：代码实现

- [ ] 创建 `verl/tools/wiki_search_tool.py`
- [ ] 创建 `config/tool_config/wiki_search_tool_config.yaml`
- [ ] 创建 `scripts/preprocess_hotpotqa_verl.py`
- [ ] 运行数据预处理，生成 parquet 文件
- [ ] 创建 `reward_functions/multihopqa_reward.py`
- [ ] 创建 `config/hotpotqa_multiturn_grpo.yaml`
- [ ] 创建 `scripts/run_hotpotqa_grpo.sh`

### 第三阶段：验证

- [ ] 单独测试 WikiSearchTool（创建实例 → 执行搜索 → 检查返回格式）
- [ ] 验证数据预处理输出格式（检查 parquet 字段）
- [ ] 小规模试运行（2 GPU，少量数据，1 epoch）
- [ ] 检查生成的轨迹格式（打印解码后的文本）
- [ ] 确认 reward function 的正则匹配正确
- [ ] 验证 action_mask / response_mask 正确性
- [ ] 监控 wandb 指标（reward 分布、loss 收敛）

### 第四阶段：全量训练

- [ ] 8 GPU 全量训练
- [ ] 调参：learning rate, KL coefficient, temperature
- [ ] 对比 Agent-R1 基线的性能
