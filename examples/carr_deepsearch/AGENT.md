# CaRR Deep Search Training Pipeline

CaRR (Citation-Aware Rubric Rewards) Deep Search 是基于 verl 框架的 agentic RL 训练流水线。训练一个 LLM agent 学会使用浏览器工具（search/open/find）进行多轮深度搜索，并通过 C-GRPO 算法（融合 outcome reward 和 rubric reward）优化搜索策略。

## 当前最终状态（2026-07-26）

这个项目已经完成当前简历目标所需的开发、训练和评测，不是仍在等待继续训练的现场。

| 范围 | 最终状态 |
|------|----------|
| 代码 | 已完成 CaRR AgentLoop、browser tools、reward bridge、C-GRPO，以及 `verl/experimental/fully_async_policy/` 的异步训练扩展 |
| 训练 | 已完成 SFT、同步 RL 至 step 96，以及 23 次异步参数版本更新（每 4 个 trainer global step 同步一次参数，约对应 92 次 trainer update） |
| 内部评测 | `DeepDive rl_val 111` matched sampled eval：`outcome_reward` 从 `0.1802` 提升到 `0.3243` |
| 外部评测 | `BrowseComp subset256 64k` matched sampled eval：judge-pass 从 `4/256` 提升到 `9/256`；这是子集结果，不是完整 BrowseComp |
| 当前阶段 | 迁移、简历编写和面试准备；没有必须继续执行的 GPU 训练或评测 |
| 可选扩展 | `BrowseComp full 64k`、128K 评测、GRPO/C-GRPO 本地消融；这些不是当前项目的未收尾任务 |

系统吞吐方面，同步后段的实测中位数约为每步 `506s`，异步健康窗口按每个参数版本包含的 4 次 trainer update 归一化后约为 `280-320s`，对应约 `37%-45%` 的健康窗口迭代时间缩短。两阶段的 batch、rollout 数量和 budget 不完全相同，因此这是系统工程对比，不是严格的同配置消融实验。

事实源优先级：

1. 对外 claim 和数字以 `docs/resume_source_of_truth_20260412.md` 为唯一事实源。
2. 完整训练配置与时间线以 `docs/training_full_history_20260404.md` 为准。
3. 最终实测结果以 `docs/eval_analysis_20260415.md` 为准。
4. 指标语义和样本级解释以后写的 `docs/agent_loop_run_explained.md`、`docs/eval_result_deep_analysis.md` 和 `docs/eval_result_deep_analysis_followup.md` 为准。
5. `docs/agent_handoff_background_20260321.md` 只记录异步训练期间的历史现场，其中的继续训练指令已经失效。

注意：

- `docs/final_resume_ready.md` 是候选对外文案，不应覆盖 `resume_source_of_truth`，其中偏强的吞吐和 timeout 表述需要在正式使用前再次核对。
- `docs/interview_qa_bank_20260412.md` 已回填真实 `BrowseComp subset256 64k` 结果；full/128K 仍只是可选扩展，回答数字时仍应回查 `resume_source_of_truth`。
- `early_stop` 表示输出命中了 trim 模板，不等于模型主动决定停止；`parse_error_count` 也不能直接解释为模型反复提交错误工具参数。具体语义见上述代码级解释文档。

## 项目训练与评测结构

```
SFT 冷启动 → C-GRPO RL 训练 → BrowseComp 评测
                ↑                    ↑
         工具服务器 (7230)      奖励服务器 (8888)
      Serper/SerpAPI + Jina      DeepSeek Judge
```

## 历史运行入口

以下命令用于需要重新训练时恢复环境，不代表当前仍有运行中的任务。正式复现实验前，应先根据 `docs/training_full_history_20260404.md` 核对实际历史配置，而不是直接把 YAML 默认值当作最终运行值。

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

## 资产索引（建议先看这里）

### 0. 先分清主工作树和历史快照

- `examples/carr_deepsearch/` 是当前主工作树，最新代码和文档结论都以这里为准。
- `examples/carr_deepsearch/CaRR_repo/` 是从远端机器下载下来的完整 repo 快照，适合核对历史脚本、旧环境和远端目录结构，不应默认当作当前事实源。
- `examples/carr_deepsearch/eval_results/` 保存样本级最终输出；`examples/carr_deepsearch/CaRR_log/` 保存 launcher、tool、reward、trace 等运行时原始日志。它们是旧 Mac 本地归档，不是远程 Git 分支的完整组成部分。
- 新机器通过 Git 克隆时，`eval_results/`、部分原始日志、Parquet 数据、密钥和 `CaRR_repo/` 不会自动出现。这些路径属于旧 Mac 本地资产，具体范围和恢复方式见 `docs/new_mac_migration_20260726.md`。

### 1. 想快速理解“项目当前状态”，先看这些文件

| 目标 | 首选文件 | 用途 |
|------|----------|------|
| 看允许对外使用的 claim 和数字 | `docs/resume_source_of_truth_20260412.md` | 唯一简历事实源，优先级最高 |
| 看完整训练时间线（sync → async、budget、b/n 变化） | `docs/training_full_history_20260404.md` | 追溯所有关键配置和训练阶段 |
| 看最新真实 eval 结论 | `docs/eval_analysis_20260415.md` | 汇总 DeepDive / BrowseComp 的最终结果和主要结论 |
| 看样本级 case study 与 failure mode | `docs/eval_result_deep_analysis.md`、`docs/eval_result_deep_analysis_followup.md` | paired case、retention、parse-error caveat、thinking-only failure |
| 看 AgentLoop 和指标的最终代码语义 | `docs/agent_loop_run_explained.md` | early-stop、natural completion、trim rescue、reward history |
| 看候选简历文案 | `docs/final_resume_ready.md` | 仅作候选表达，使用数字前仍需回查事实源 |
| 看面试问答素材 | `docs/interview_qa_bank_20260412.md` | 已回填真实 subset256 结果；数字仍以简历事实源为准 |
| 看 async 调试历史 | `docs/agent_handoff_background_20260321.md` | 训练期间的历史交接，不是当前状态页 |

### 2. 评测资产（旧 Mac 本地归档，普通 Git clone 不包含）

#### `eval_results/` — 样本级最终输出

| 路径 | 说明 |
|------|------|
| `eval_results/dd111_sft/0.jsonl` | `DeepDive rl_val 111` 的 SFT 基线输出 |
| `eval_results/dd111_async23/0.jsonl` | `DeepDive rl_val 111` 的 async23 输出 |
| `eval_results/async23_dd64_sanity/0.jsonl` | `DeepDive subset64` 的 async23 sanity eval |
| `eval_results/bc256_sft_relaxed_v2/0.jsonl` | `BrowseComp subset256 64k` 的 SFT 输出 |
| `eval_results/bc256_async23_relaxed_v2/0.jsonl` | `BrowseComp subset256 64k` 的 async23 输出 |

#### `CaRR_log/20260415_gpu_eval/` — 与上述 eval 对应的原始日志

- 每个 eval run 通常有 6 类文件：
  - `*.launcher.log`: 启动与 trainer 主日志
  - `*.log`: eval 主输出
  - `*.status`: 运行状态
  - `*_reward.log` / `*_reward_trace.jsonl`: reward server 日志与逐样本 trace
  - `*_tool.log` / `*_tool_stats.json`: tool server 日志与聚合统计
- 这些原始日志对应当前最重要的 5 个 eval：
  - `dd111_sft`
  - `dd111_async23`
  - `bc256_sft_relaxed_v2`
  - `bc256_async23_relaxed_v2`
  - `async23_dd64_sanity`

#### 旧评测 / checkpoint selection 日志

- `CaRR_log/step70_dd64_8gpu_20260322_153257.*`: `step70` 的 `DeepDive subset64 sampled` eval
- `CaRR_log/step90_dd64_8gpu_20260322_151340.*`: `step90` 的 `DeepDive subset64 sampled` eval
- `CaRR_log/eval_gate_20260323_144500_*`: `SFT / step90` gate 相关日志
- 这些日志主要用于说明为什么 `step70` 被选作 async 起点，以及早期 gate 的真实指标来源

### 3. 训练日志与运行历史（大部分是旧 Mac 本地归档）

| 路径 | 说明 |
|------|------|
| `CaRR_formal_training_log/output.log` | 同步 RL 正式训练主日志 |
| `CaRR_formal_training_log/40.jsonl`、`CaRR_formal_training_log/80.jsonl` | sync 训练中的中间 eval / dump 产物 |
| `CaRR_log/formal_mainline_*` | 正式 async mainline 的关键训练日志 |
| `CaRR_log/phase0_*`、`phase1_*`、`phase2*` | fully-async probe / bring-up 各阶段日志 |
| `CaRR_log/gate_b4_n8_*`、`validate_b4_n8_*` | async 配置探索、resume 和 gate 日志 |
| `CaRR_log/latest_checkpointed_iteration.txt` | 最新 durable checkpoint 记录 |

### 4. 代码实现资产

| 路径 | 说明 |
|------|------|
| `tools/carr_agent_loop.py` | 同步/异步 CaRR agent loop，含 reward history、budget、partial rollout 状态持久化 |
| `tools/carr_browser_tool.py` | browser.search/open/find 适配层 |
| `tools/carr_session_manager.py` | request-level session 生命周期管理 |
| `reward/carr_reward.py` | reward bridge，连接 reward server |
| `reward/cgrpo_advantage.py` | C-GRPO reward fusion / advantage 实现 |
| `reward/async_base_reward.py` | async base probe 的 dummy reward |
| `config/` | SFT、sync RL、async RL、tool schema 配置 |
| `scripts/` | 训练、eval、subset 准备、checkpoint 选择和一致性验证脚本 |

如果要理解 async infra，本目录外还需要同时看：

- `verl/experimental/fully_async_policy/fully_async_main.py`
- `verl/experimental/fully_async_policy/fully_async_rollouter.py`
- `verl/experimental/fully_async_policy/fully_async_trainer.py`
- `verl/experimental/fully_async_policy/param_sync.py`

### 5. 数据资产（旧 Mac 本地文件，可复制或重新生成）

| 路径 | 说明 |
|------|------|
| `data/sft_train.parquet`、`data/sft_val.parquet` | SFT 训练 / 验证数据 |
| `data/rl_train.parquet`、`data/rl_val.parquet` | RL 训练 / 验证数据 |
| `data/rl_val_subset_64_seed42.parquet` | `DeepDive subset64` gate / sanity eval 数据 |
| `data/browsecomp_eval.parquet` | 完整 BrowseComp eval 数据 |
| `data/browsecomp_eval_subset_256_seed42.parquet` | 当前最常用的 BrowseComp subset256 |
| `data_preprocess/` | CaRR SFT / RL / BrowseComp 到 verl parquet 的预处理脚本 |

### 6. 文档资产

#### 项目与训练背景

- `CODE_WALKTHROUGH.md`: 代码入口导览
- `DEVELOPMENT_LOG.md`: 开发阶段记录
- `docs/project_retrospective_20260312.md`: 训练前复盘与问题归因
- `docs/rl_debug_findings_20260312.md`: RL / eval 细节坑位索引
- `docs/async_rl_explainer_20260321.md`: async 结构化解释
- `docs/sync_vs_async_grpo_update_mechanics.md`: sync / async 更新机制对比

#### 评测与结果分析

- `docs/eval.md`: 评测总说明
- `docs/eval_analysis_20260415.md`: 当前最重要的结果总结
- `docs/eval_result_deep_analysis.md`: 深入行为分析
- `docs/eval_result_deep_analysis_followup.md`: finished-but-wrong、retention、oracle 等补充分析
- `docs/agent_loop_run_explained.md`: early-stop、natural completion、trim rescue 和 reward history 的最终代码级语义
- `docs/carr_paper_training_analysis_20260416.md`: 论文训练现象与本项目结果对照

#### 简历与面试材料

- `docs/resume_source_of_truth_20260412.md`: 简历事实源
- `docs/final_resume_ready.md`: 对外展示版 resume 文案
- `docs/interview_qa_bank_20260412.md`: 面试 Q&A 题库
- `resume_design_and_question/`: 更早期的简历草稿与面试资料备份

#### GPU / 执行计划

- `docs/post_gpu_revalidation_plan_20260412.md`: 已执行完毕的 GPU 复验历史 runbook
- `docs/gpu_execution_plan.md`: 已执行完毕的 GPU 评测历史计划
- `docs/gpu_eval_recommendations_20260404.md`: 评测执行前的历史建议

### 7. 远端快照与备份资产

| 路径 | 说明 |
|------|------|
| `CaRR_repo/` | 远端整仓快照；适合核对历史 repo 状态、旧脚本和原始文档 |
| `remote_artifacts/h200_20260312/` | H200 机器上的 ray / wandb / logs 备份 |
| `artifacts/huggingface/` | 本地保存的 HuggingFace 产物目录 |

### 8. 查找建议

1. 先看 `docs/resume_source_of_truth_20260412.md`、`docs/training_full_history_20260404.md`、`docs/eval_analysis_20260415.md`。
2. 要看真实样本行为，先看 `eval_results/*.jsonl`，再对照 `CaRR_log/20260415_gpu_eval/` 的 tool / reward trace。
3. 要理解“为什么会这样”，再看 `docs/agent_loop_run_explained.md`、`docs/eval_result_deep_analysis.md` 和 `docs/eval_result_deep_analysis_followup.md`。
4. 要改代码时，优先在当前主工作树里看 `tools/`、`reward/`、`config/`、`scripts/`，不要默认去 `CaRR_repo/` 里改。

## 本目录文件结构

### config/ — Hydra 训练配置

| 文件 | 说明 |
|------|------|
| `carr_sft.yaml` | SFT 冷启动配置。模型 Qwen/Qwen3-4B，multiturn SFT，max_length=65536，3 epochs |
| `carr_grpo.yaml` | 同步 C-GRPO 的当前 YAML 默认值：`train_batch_size=16`、`rollout.n=8`、`max_response_length=61440`、`adv_estimator=cgrpo`、`cgrpo_alpha=0.3`。实际同步正式训练使用 `b8/n4`，不要混淆默认值和历史实测值 |
| `carr_grpo_async_common.yaml` | async 共用基础配置；其中短上下文和 `n=4` 是 probe 默认值，正式脚本会覆盖，不代表最终主线 |
| `carr_grpo_async_base.yaml` | fully-async 基座 probe 配置；关闭多轮工具，只验证 trainer、rollouter 和参数同步 |
| `carr_grpo_async.yaml` | CaRR async RL 配置；启用多轮 AgentLoop、partial rollout、queue 和 budget 控制 |
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
| `run_rl_async.sh` | async RL 底层启动脚本；管理 `fully_async_main`、tool/reward server、队列和 GPU 划分 |
| `run_async_formal.sh` | 历史正式 async 编排脚本；实际最终 `b3/n8 + 2:6 + SP=2 + sync4` 仍需结合训练历史和日志中的 overrides 复原 |
| `run_eval_integration.sh` | 正式 sampled eval 通用入口，接口为 `<model_path> <eval_file> <run_name>` |
| `run_eval_deepdive64_sampled.sh` | 固定 sampled recipe 的 DeepDive subset64 gate |
| `run_eval_browsecomp.sh` | 旧的 BrowseComp greedy eval 脚本，硬编码 `temperature=0`、`do_sample=false`；不能用于复现最终 Thinking checkpoint 结果 |
| `smoke_test.py` | Gate 2/3 冒烟测试：验证工具服务器 search→open→find 链路 + 奖励服务器 outcome_reward > 0 |
| `verify_tool_consistency.py` | 验证 SFT parquet 和 RL YAML 经各自真实路径渲染后的 system prompt 完全一致 |
| `test_sft_tokenization.py` | SFT tokenization 正确性验证（含 enable_thinking） |

### docs/

| 文件 | 说明 |
|------|------|
| `progress_2x5090.md` | 2x RTX 5090 环境验证进度报告。含所有 Gate 验证记录、调试问题汇总、正式训练规划（GPU 选择、训练时长、API 用量估算、完整训练命令） |

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
