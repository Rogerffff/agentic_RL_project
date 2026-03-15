# CaRR Deep Search Training Pipeline

CaRR (Citation-Aware Rubric Rewards) Deep Search 是基于 verl 框架的 agentic RL 训练流水线。训练一个 LLM agent 学会使用浏览器工具（search/open/find）进行多轮深度搜索，并通过 C-GRPO 算法（融合 outcome reward 和 rubric reward）优化搜索策略。

## 训练流程

```
SFT 冷启动 → C-GRPO RL 训练 → BrowseComp 评测
                ↑                    ↑
         工具服务器 (7230)      奖励服务器 (8888)
      Serper/SerpAPI + Jina      DeepSeek Judge
```

## 快速运行

```bash
# SFT (建议显式 bf16，避免 fp32 带来的显存和吞吐问题)
torchrun --nproc_per_node=4 -m verl.trainer.fsdp_sft_trainer \
  --config-path=examples/carr_deepsearch/config --config-name=carr_sft \
  model.fsdp_config.model_dtype=bf16

# RL（推荐用脚本自动启动工具/奖励服务器）
export SERPER_API_KEY=...        # 或 SERPAPI_API_KEY=...
export JINA_API_KEY=...
export DEEPSEEK_API_KEY=...
export VERL_USE_EXTERNAL_MODULES=examples.carr_deepsearch.tools.carr_agent_loop,examples.carr_deepsearch.reward.cgrpo_advantage
export CARR_REWARD_SERVER_URL=http://localhost:8888
export CARR_REWARD_TIMEOUT=650
export SFT_MODEL_PATH=/abs/path/to/sft_checkpoint/huggingface
bash examples/carr_deepsearch/scripts/run_rl.sh \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
  actor_rollout_ref.ref.fsdp_config.model_dtype=bf16
```

## 外部 API 依赖

| API | 用途 | 调用方 |
|-----|------|--------|
| Serper.dev（首选）/ SerpAPI（兼容） | 网页搜索 (`browser.search`) | 工具服务器 |
| Jina Reader | 网页内容提取 (`browser.open`) | 工具服务器 |
| DeepSeek | LLM Judge (outcome + rubric 评分) | 奖励服务器 |

`browser.find` 是纯本地字符串匹配，不调用外部 API。

---

## 本目录文件结构

### config/ — Hydra 训练配置

| 文件 | 说明 |
|------|------|
| `carr_sft.yaml` | SFT 冷启动配置。模型 Qwen/Qwen3-4B，multiturn SFT，max_length=65536，3 epochs |
| `carr_grpo.yaml` | C-GRPO RL 配置。train_batch_size=128, rollout.n=16, max_response_length=61440, adv_estimator=cgrpo, cgrpo_alpha=0.3 |
| `tool_config/carr_browser_tools.yaml` | 三个浏览器工具的 schema 定义（SFT + RL 唯一真值）。建议保持稳定 key 顺序；最终一致性由 `normalize_tool_schema()` 保证 |

### tools/ — Agent 工具与会话管理

| 文件 | 说明 |
|------|------|
| `carr_agent_loop.py` | 自定义 AgentLoop，继承 ToolAgentLoop。注册名 `carr_tool_agent`。维护 CaRR 格式的 reward_history（含 tool_call_id），在 finally 中关闭工具服务器 session |
| `carr_browser_tool.py` | BaseTool 适配器，三个工具共用一个类（按 self.name 区分）。转换参数类型（如 open.id str→int），提取 search_forbidden_strs，委托 CaRRSessionManager 管理 session |
| `carr_session_manager.py` | 单例 session 管理器。每个 request_id 一个 session，惰性启动（首次工具调用时 start_session），agent 结束时 close_session |

### reward/ — 奖励函数与优势估计

| 文件 | 说明 |
|------|------|
| `carr_reward.py` | 异步奖励函数，由 NaiveRewardManager 调用。发送 agent history 到奖励服务器 `/evaluate`，返回 {score, outcome_reward, rubric_reward}。env: `CARR_REWARD_SERVER_URL`, `CARR_REWARD_TIMEOUT` |
| `cgrpo_advantage.py` | C-GRPO 优势估计器，注册名 `cgrpo`。融合公式: `R = (1-α)*R_outcome + α*R_outcome*R̂_rubric`，α=0.3。组内归一化 rubric 后融合，重建 token_level_rewards |

### data_preprocess/ — 数据预处理脚本

| 文件 | 说明 |
|------|------|
| `preprocess_carr_sft.py` | CaRR SFT JSONL → verl parquet。从 `carr_browser_tools.yaml` 加载 canonical tool schemas（而非 record 中的 tools），确保 SFT-RL schema 一致。支持 `--tool_config` 参数 |
| `preprocess_carr_rl.py` | CaRR RL JSONL → verl parquet。提取 rubrics、search_forbidden_strs，构建 tools_kwargs，设置 agent_name="carr_tool_agent" |
| `preprocess_browsecomp.py` | BrowseComp 评测数据 → verl parquet |

### data/ — Parquet 数据文件

| 文件 | 行数 | 说明 |
|------|------|------|
| `sft_train.parquet` | 791 | SFT 训练集。列: messages, tools, enable_thinking |
| `sft_val.parquet` | 41 | SFT 验证集 |
| `rl_train.parquet` | 2123 | RL 训练集。列: data_source, agent_name, prompt, ability, extra_info(rubrics, search_forbidden_strs, rubric_reward_ratio, tools_kwargs) |
| `rl_val.parquet` | 111 | RL 验证集 |
| `browsecomp_eval.parquet` | 1266 | BrowseComp 评测集 |

### scripts/ — 运行脚本与测试

| 文件 | 说明 |
|------|------|
| `run_sft.sh` | SFT 一键训练（含数据预处理 + torchrun） |
| `run_rl.sh` | RL 一键训练（启动工具/奖励服务器 + main_ppo）。支持 Serper/SerpAPI 双后端 |
| `run_eval_browsecomp.sh` | BrowseComp 评测（启动服务器 + val_only 模式） |
| `smoke_test.py` | Gate 2/3 冒烟测试：验证工具服务器 search→open→find 链路 + 奖励服务器 outcome_reward > 0 |
| `verify_tool_consistency.py` | 验证 SFT parquet 和 RL YAML 经各自真实路径渲染后的 system prompt 完全一致 |
| `test_sft_tokenization.py` | SFT tokenization 正确性验证（含 enable_thinking） |

### docs/

| 文件 | 说明 |
|------|------|
| `progress_2x5090.md` | 2x RTX 5090 环境验证进度报告。含所有 Gate 验证记录、调试问题汇总、正式训练规划（GPU 选择、训练时长、API 用量估算、完整训练命令） |
| `project_retrospective_20260312.md` | 本轮 RL 训练前复盘长文。系统解释为什么大多数配置失败、为什么成本失控、为什么 `b4/n4, TP1` 是当前唯一稳定口径，以及下一轮最该先修什么 |

### 根目录项目文档

| 文件 | 说明 |
|------|------|
| `CODE_WALKTHROUGH.md` | 核心代码路径导览，便于快速定位训练、工具链路、奖励链路实现 |
| `DEVELOPMENT_LOG.md` | 开发阶段问题、修复与阶段性验收记录 |
| `api.md` | 工具/奖励服务接口与调用示例说明 |

---

## verl 框架关键文件

### 训练入口

| 文件 | 说明 |
|------|------|
| `verl/trainer/main_ppo.py` | RL 训练 Hydra 入口。加载 ppo_trainer 配置，初始化 Ray，实例化 RayPPOTrainer |
| `verl/trainer/fsdp_sft_trainer.py` | FSDP SFT 训练器。加载 SFTDataset + apply_chat_template，FSDP 分布式训练，checkpoint 管理 |
| `verl/trainer/config/ppo_trainer.yaml` | PPO 基础配置模板（所有默认值在此定义） |

### RL 训练循环

| 文件 | 说明 |
|------|------|
| `verl/trainer/ppo/ray_trainer.py` | RayPPOTrainer 主循环: rollout → reward → advantage → policy update。处理 data batching、mini-batch 更新、checkpoint、metrics |
| `verl/trainer/ppo/core_algos.py` | 核心算法函数。`@register_adv_est` 注册优势估计器（gae, grpo, reinforce_plus_plus, cgrpo 等）。compute_grpo_outcome_advantage(), compute_policy_loss(), agg_loss() |
| `verl/trainer/ppo/metric_utils.py` | 指标计算：data_metrics, throughput_metrics, timing_metrics, validation_metrics |

### Agent Loop（多轮 agentic RL）

| 文件 | 说明 |
|------|------|
| `verl/experimental/agent_loop/agent_loop.py` | AgentLoopBase 抽象基类。AsyncLLMServerManager 负载均衡推理服务器。AgentLoopWorker/AgentLoopManager 管理 Ray agent 工作组 |
| `verl/experimental/agent_loop/tool_agent_loop.py` | ToolAgentLoop 状态机: PENDING→GENERATING→PROCESSING_TOOLS→TERMINATED。AgentData 追踪每个样本的 messages, response_ids, response_mask, tool_calls, assistant_turns |
| `verl/experimental/agent_loop/tool_parser.py` | 工具调用解析器注册表。HermesToolParser 解析 Qwen `<tool_call>` 格式 |

### 工具系统

| 文件 | 说明 |
|------|------|
| `verl/tools/schemas.py` | OpenAIFunctionToolSchema 数据结构 + `normalize_tool_schema()` 归一化函数（drop None + 递归字母序排列 key），在 SFT/RL 4 个消费点调用确保一致性 |
| `verl/tools/base_tool.py` | BaseTool 抽象接口：create() → execute() → release() 生命周期 |

### 数据加载

| 文件 | 说明 |
|------|------|
| `verl/utils/dataset/sft_dataset.py` | SFTDataset：加载 parquet，apply_chat_template 渲染 system prompt（含 tools），tokenize。支持 truncation, max_length 过滤 |
| `verl/utils/dataset/multiturn_sft_dataset.py` | 多轮 SFT 数据集，message 级处理。调用 normalize_tool_schema() 归一化 tools |
| `verl/utils/dataset/rl_dataset.py` | RLHFDataset：RL 训练数据加载，prompt tokenization，支持 checkpoint 恢复。调用 normalize_tool_schema() 归一化 tools |

### Rollout 推理引擎

| 文件 | 说明 |
|------|------|
| `verl/workers/rollout/sglang_rollout/sglang_rollout.py` | SGLang ServerAdapter (BaseRollout 子类)。管理 weight sync, KV cache, 模型初始化。传递 attention_backend 到 AsyncHttpServerAdapter（SM120 兼容修改） |
| `verl/workers/rollout/sglang_rollout/http_server_engine.py` | AsyncHttpServerAdapter：与 SGLang 推理服务器的 HTTP 通信 |

### Reward 管理

| 文件 | 说明 |
|------|------|
| `verl/workers/reward_manager/naive.py` | NaiveRewardManager：逐样本调用 reward function（如 carr_reward.compute_score）。合并 parquet extra_info + agent loop extra_fields |
| `verl/workers/reward_manager/registry.py` | Reward manager 注册表，根据配置动态实例化 |

### FSDP 分布式训练

| 文件 | 说明 |
|------|------|
| `verl/workers/fsdp_workers.py` | ActorRolloutRefWorker, CriticWorker：FSDP 分布式 actor/ref/critic。weight 初始化, forward/backward, 梯度累积, checkpoint save/load, train/inference 模式切换 |

### 外部模块加载

| 文件 | 说明 |
|------|------|
| `verl/__init__.py` | `import_external_libs()`：通过 `VERL_USE_EXTERNAL_MODULES` env var 动态导入外部模块（逗号分隔），使自定义 agent loop / advantage estimator 自动注册 |

---

## CaRR 子模块关键文件

### 工具服务器 (`CaRR/tool_server/`)

| 文件 | 说明 |
|------|------|
| `launch_server.py` | Quart 异步 HTTP 服务器 (端口 7230)。路由 POST 请求到 browser.search/open/find。维护 session2sandbox 会话状态，每 15 秒打印工具调用日志。支持 `--search_backend serper` 和 `--serp_api_key` / `--serper_api_key` |
| `web_search.py` | 工具实现。`search_serper()`: POST 到 Serper.dev API, retry_times=3。`parse_url()`: GET Jina Reader API (`r.jina.ai`), retry_times=3。`find()`: 本地精确+模糊字符串匹配（免费）。`contain_forbidden_str()`: n-gram 反作弊过滤 |

### 奖励服务器 (`CaRR/deepsearch_rm_with_rubrics/`)

| 文件 | 说明 |
|------|------|
| `launch_server.py` | FastAPI 服务器 (端口 8888)。`/evaluate` 端点接收 history + label + remote_env_info。task_unfinished=True 直接返回 0。正常流程: `get_outcome_reward()` (1 次 LLM 调用, 二分类判断) + `get_rubric_reward()` (2 次 LLM 调用: 实体识别 + rubric 评判 + BFS 连通性检查)。每样本约 3 次 DeepSeek API 调用，内部 timeout=600s |
| `prompts/` | LLM Judge 提示词: `get_outcome_reward.txt`, `identify_entity.txt`, `judge_rubric.txt` |
| `requirements.txt` | 奖励服务器依赖: fastapi, uvicorn, openai, httpx |

---

## 关键设计决策与注意事项

### RL 正式训练前问题索引

下面这份列表只做**简要索引**，帮助后续 agent 快速知道在正式 RL 训练前已经踩过哪些坑。  
如果需要完整背景、证据和根因分析，请优先阅读 [docs/project_retrospective_20260312.md](./docs/project_retrospective_20260312.md)。

#### 1. 错误的 SFT 起点会直接污染 RL 判断

- 早期曾经使用过有问题的 SFT 起点，导致模型容易：
  - 疯狂调用工具
  - 不会稳定收束到最终答案
  - 让人误以为是 RL 链路坏了
- 当前应以**修正后的 SFT 主链路 checkpoint**作为唯一 RL 起点。
- 详见复盘中的“监督微调起点错误”。

#### 2. `task_unfinished=True` 会让 reward server 直接返回 0

- 这会造成一种错觉：看起来像是 DeepSeek judge 没工作。
- 真实情况通常是：
  - rollout 没在限制条件内完成
  - reward server 直接 short-circuit 为零分
- 早期多轮 probe 中，大量零分其实是 unfinished，不是 reward server 宕掉。
- 详见复盘中的“`task_unfinished` 与 reward short-circuit”。

#### 3. assistant turn limit 和 response limit 都曾是主要失败模式

- 早期 probe 中，`max_assistant_turns` 偏小会让样本在还没接近 64k 时就先被截断。
- 后续提高 turn limit 后，又确认一部分样本会转而撞上 `max_response_length`。
- 说明问题不是单一阈值，而是：
  - turn limit
  - response limit
  - 无效工具循环
  三者共同作用。
- 详见复盘中关于 termination reason 与 unfinished 的部分。

#### 4. “没看到 `step:1`”不能直接判定 PPO 卡死

- 多轮排查后确认，以下因素都曾误导 probe 结论：
  - `trainer.val_before_train`
  - 最后一步 validation
  - 最后一步 save
  - reward trace 里混入 health check
- 因此后续所有 `1-step probe` 必须显式设为 clean probe：
  - `trainer.val_before_train=false`
  - `trainer.test_freq=0`
  - `trainer.save_freq=0`
- 详见复盘中的“validation/save/Ray 对 probe 的误导”。

#### 5. RL 早期有过配置级硬错误，不是策略问题

- 典型问题包括：
  - `train_batch_size * rollout.n` 与 agent loop worker 数不整除
  - `ppo_max_token_len_per_gpu` 太小，导致 `max_token_len must be greater than the sequence length`
  - `actor_rollout_ref.model.path` 需要显式 CLI override，不能只赌环境变量
- 这些问题会让 run 在 rollout 之后或更早阶段直接失败，不能拿来判断 reward signal。
- 详见复盘与 `docs/rl_debug_findings_20260312.md`。

#### 6. 4xA100 上不是“完全跑不动”，而是吞吐不经济

- 4xA100 已证明：
  - RL 主链路能走通
  - 可以产生真实 reward
- 但正式并发口径 wall time 过长，不适合作为正式 RL 主训练机型。
- 这不是“显存不够”，而是系统吞吐结构不经济。
- 详见复盘中的成本章节。

#### 7. 8x RTX PRO 6000 Blackwell 主要卡在 SGLang `TP=2` generate 路径

- 该机器不是单纯“训练慢”，而是 standalone SGLang 就能复现：
  - `TP=1` 正常
  - `TP=2` server 可启动，但真实 `/generate` 会卡住
- 因此它不适合作为当前正式 RL 机器。
- 详见复盘和 `docs/blackwell_sglang_debug_20260312.md`。

#### 8. 8xH200 上真正稳定打出 `step:1` 的是 `b4/n4, TP1`

- 这套 clean probe 是第一套已确认：
  - rollout 完成
  - reward 完成
  - old_log_prob / update_actor / update_weights 全部完成
  - 明确打出 `step:1`
- 但它仍然太贵，不能直接当正式经济配方。
- 详见复盘中的 `b4/n4, TP1` 成本分析。

#### 9. `b16/n8` 不是“显存够了就该更快”

- 当前实现里，放大 `train_batch_size` 和 `rollout.n` 主要是在放大：
  - rollout 队列深度
  - 外部工具长尾
  - reward judge 长尾
  - 同步栅栏等待
- 因此 `b16/n8` 在当前 infra 下不经济，不应直接作为正式训练默认值。
- 详见复盘中的“为什么 `b16/n8` 不是显存够了就该更快”。

#### 10. H200 上还踩过系统层问题：`Too many open files`

- 远端 H200 机器曾出现：
  - `ulimit -n = 1024`
  - `ray_init.num_cpus = null` / Ray 默认吃满整机 CPU
  联合作用导致 Ray 启动期 `Too many open files`
- 这类问题和模型策略无关，但会直接破坏 probe 解释。
- 后续租机时应优先固定：
  - `ulimit -n 65535`
  - `ray start --num-cpus=32`
  - `+ray_kwargs.ray_init.address=auto`
- 详见复盘中的“Ray 启动期问题”。

#### 11. 当前最大问题是系统吞吐结构，不是单个组件“坏掉”

- 不能简单说：
  - “GPU 不是瓶颈”
  - “reward server 就是唯一瓶颈”
  - “PPO 后半段一定卡死”
- 更准确的说法是：
  - 瓶颈会随配置迁移
  - 低并发稳定口径里主瓶颈更像 rollout/generation
  - 更高并发下 post-rollout 也会被放大成新瓶颈
- 详见复盘的“为什么成本失控”与“为什么 `b16/n8` 不经济”。

### SFT-RL Tool Schema 一致性

SFT 和 RL 的 system prompt 中 tools JSON 必须完全一致，否则模型在 RL 阶段无法正确调用工具。

- **唯一真值**: `config/tool_config/carr_browser_tools.yaml`
- **归一化函数**: `verl/tools/schemas.py:normalize_tool_schema()` — drop None + 递归字母序
- **SFT 预处理**: `preprocess_carr_sft.py` 从 YAML 加载 canonical tools（不用 record 自带的 tools）
- **4 个消费点** 都调用 normalize_tool_schema: multiturn_sft_dataset, rl_dataset, tool_agent_loop, single_turn_agent_loop

### 环境变量

```bash
# 必须在 ray start 之前 export（Ray worker 继承启动时的环境）
export VERL_USE_EXTERNAL_MODULES=examples.carr_deepsearch.tools.carr_agent_loop,examples.carr_deepsearch.reward.cgrpo_advantage
export CARR_REWARD_SERVER_URL=http://localhost:8888
export CARR_REWARD_TIMEOUT=650
```

### RTX 5090 (SM120) 特殊配置

```bash
# FA3 不支持 SM120，必须用 flashinfer
+actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=flashinfer
# A100/H100 不需要此配置（默认 fa3 即可）
```

### Hydra 路径注意事项

Ray TaskRunner 的 cwd 是 `/root/`，所有文件路径必须用绝对路径 CLI override：
- `data.train_files`, `data.val_files`
- `reward.custom_reward_function.path`
- `actor_rollout_ref.rollout.multi_turn.tool_config_path`
- `actor_rollout_ref.model.path`

### bf16 必须显式设置

```bash
# SFT
model.fsdp_config.model_dtype=bf16
# RL（actor + ref 两个都要设）
actor_rollout_ref.actor.fsdp_config.model_dtype=bf16
actor_rollout_ref.ref.fsdp_config.model_dtype=bf16
```
