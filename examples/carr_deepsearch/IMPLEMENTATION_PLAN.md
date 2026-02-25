# CaRR × verl 多跳问答 Deep Search Agent 实施计划（修订版）

## 0. 文档目的

本文档用于指导在 `verl` 框架下实现 CaRR 风格的多轮工具调用 RL 训练，目标是：

1. 在 `verl` 的 `ToolAgentLoop` 中稳定运行 `browser.search/open/find`。
2. 将完整轨迹送入 CaRR reward server，获得 `outcome_reward` 与 `rubric_reward`。
3. 先跑通可训练的 MVP，再迭代到论文等价的 C-GRPO（组内归一化 + gated rubric）。

本修订版修复了上一版中的关键问题：

- 轨迹 `messages` 不完整导致奖励恒 0 的风险。
- reward server 启动目录错误导致 `prompts` 读取失败。
- SFT 配置与 `fsdp_sft_trainer` 不匹配。
- RL 未配置 `val_files` 导致默认读取无关数据。
- `browser.open.id` 参数类型与 CaRR 服务端预期不一致。

---

## 1. 范围与非目标

## 1.1 范围

- 代码范围：`examples/carr_deepsearch/` + 少量 `verl/experimental/agent_loop/` 扩展。
- 训练范围：SFT 冷启动 + RL（GRPO/C-GRPO）。
- 服务范围：CaRR 工具服务器 + CaRR 奖励服务器。

## 1.2 非目标

- 本阶段不追求复现论文所有最终数值（先保证链路正确、稳定）。
- 本阶段不改 CaRR 原始服务端逻辑（除非必须修 bug）。

---

## 2. 可行性结论（简版）

可行，且可分阶段落地。

- `verl` 已支持多轮 agent loop、工具调用、custom reward function。
- `CaRR` 已提供工具服务与 reward 服务。
- 数据供给满足起步：SFT 轨迹（文档提到 832 条可训练轨迹）+ RL QA/rubrics（2234）。

关键工程工作不在“有没有能力”，而在“接口桥接是否严谨”。

---

## 3. 关键接口契约（必须遵守）

## 3.1 工具服务契约（CaRR）

- 入口：`POST http://<tool_server>/`
- 请求字段：
  - `session_id: str`
  - `name: start_session | close_session | browser.search | browser.open | browser.find`
  - `arguments: dict`
  - `remote_env_info.search_forbidden_strs: list[str]`
- 返回字段：`{"output": "..."}`

`browser.open` 的 `arguments.id` 必须支持 `int | str`：

- `int`：表示 search 结果索引。
- `str`：表示 URL。

## 3.2 奖励服务契约（CaRR）

- 入口：`POST http://<reward_server>/evaluate`
- 请求字段：
  - `history: list[message]`（完整多轮）
  - `label: str`（标准答案）
  - `task_unfinished: bool`
  - `remote_env_info`：
    - `search_forbidden_strs`
    - `rubrics`
    - `rubric_reward_ratio`
- 返回字段：
  - `reward`
  - `outcome_reward`
  - `rubric_reward`
  - `rubric_scores`

注意：CaRR 服务端要求 `history[-1]` 为 assistant message，否则直接给 0 分。

## 3.3 verl 侧数据契约（RL parquet）

每条样本至少包含：

- `data_source`
- `agent_name: tool_agent`
- `prompt`（list[chat message]）
- `reward_model.ground_truth`
- `extra_info`：
  - `rubrics`
  - `rubric_reward_ratio`
  - `search_forbidden_strs`
  - `need_tools_kwargs: true`
  - `tools_kwargs`

---

## 4. 实施路线（分阶段）

## Phase A: 环境与依赖基线（0.5-1 天）

目标：所有脚本可启动，不因依赖或路径失败。

任务：

1. 安装 `verl` 基础依赖：`pip install -r requirements.txt`。
2. 安装 CaRR reward 依赖：`pip install -r CaRR/deepsearch_rm_with_rubrics/requirements.txt`。
3. 补充工具服务依赖（`quart`, `aiohttp`，若环境中不存在）。
4. 固定运行目录策略：
   - 工具服务可在仓库根目录启动。
   - 奖励服务必须在 `CaRR/deepsearch_rm_with_rubrics` 目录内启动，避免 `./prompts` 路径错误。

验收：

- 两个服务均可独立启动并通过健康测试请求。

---

## Phase B: 数据预处理（1-2 天）

目标：得到可直接用于 `verl` 的 SFT/RL parquet。

任务：

1. `preprocess_carr_rl.py`
   - 转换 `question/answer/rubrics/search_forbidden_strs`。
   - 输出 `prompt + reward_model + extra_info`。
   - 生成 `rl_train.parquet` 与 `rl_val.parquet`（建议从 train 切分 2%-5% 做 val）。
2. `preprocess_carr_sft.py`
   - 输出 `messages` 字段（多轮格式）。
   - 若源数据缺 system，可补统一 system prompt。
3. 产出数据检查脚本：
   - 字段完整率检查。
   - 抽样检查 `rubrics` 非空率。

验收：

- RL parquet 可被 `create_rl_dataset` 正常加载。
- SFT parquet 在 `multiturn.enable=true` 配置下可被 `MultiTurnSFTDataset` 加载。

---

## Phase C: 工具层集成（2-3 天）

目标：在 `ToolAgentLoop` 中稳定执行 `browser.search/open/find`。

任务：

1. 新建工具实现：
   - `BrowserSearchTool`
   - `BrowserOpenTool`
   - `BrowserFindTool`
2. 工具配置 `carr_browser_tools.yaml`：
   - `browser.open.id` schema 使用 `oneOf: [integer, string]`（或允许 number/string 的等价表达）。
3. 会话管理：
   - 每个 trajectory 只启动一次 `start_session`。
   - 在 trajectory 结束时执行 `close_session`，防止 `session2sandbox` 泄漏。
4. 错误处理：
   - 网络异常、空结果、超时统一转为可读文本，不能抛异常中断训练。

验收：

- 单条样本可跑通 `search -> open -> find`。
- 100 条离线冒烟中工具调用成功率 >= 95%。

---

## Phase D: 轨迹与奖励桥接（2-4 天）

目标：CaRR reward 请求拿到“完整且格式正确”的 history。

任务：

1. 扩展 `ToolAgentLoop` 输出字段：
   - 保留 `turn_scores/tool_rewards`。
   - 增加可供 reward 使用的完整消息历史（必须包含 assistant/tool/user 的时序）。
2. 修正历史构造逻辑：
   - 确保最终 assistant 回复进入 history 末尾。
   - 正确绑定 `tool_call_id` 与 tool output（避免“全部绑定最后一个 id”）。
3. `carr_reward.py`：
   - 调用 `/evaluate`。
   - 返回 `score/outcome_reward/rubric_reward/rubric_scores`。
   - 服务不可用时给安全降级分并打日志。

验收：

- 100 条样本 reward 请求成功率 100%。
- `history[-1].role == assistant` 覆盖率 100%。
- `outcome_reward` 与 `rubric_reward` 分布非全零。

---

## Phase E: 训练配置与脚本（2-3 天）

目标：一键执行 SFT/RL，配置与 trainer 严格对齐。

任务：

1. SFT 配置 `carr_sft.yaml`（关键）：
   - 继承 `sft_trainer` 默认结构。
   - 使用 `model.partial_pretrain`（不是 `model.path`）。
   - `data.multiturn.enable: true`。
   - 明确 `data.train_files` 与 `data.val_files`。
2. RL 配置 `carr_grpo.yaml`：
   - 明确 `data.train_files` 与 `data.val_files`，避免落回默认 gsm8k。
   - `data.return_raw_chat: true`。
   - `actor_rollout_ref.rollout.multi_turn.enable: true`。
   - `format: hermes`（当前源码已注册）。
   - `reward.custom_reward_function.path/name` 指向 CaRR reward 封装。
3. 启动脚本：
   - `run_sft.sh`。
   - `run_rl.sh`（加入 `trap` 清理进程，保证异常退出也能 kill 服务）。
   - `run_rl.sh` 内 reward server 采用 `(cd CaRR/deepsearch_rm_with_rubrics && python launch_server.py ...)`。

验收：

- SFT 可启动并至少完成 1 个 epoch（小样本）。
- RL 可启动并跑完 1k step 冒烟。

---

## Phase F: C-GRPO 论文对齐（3-5 天）

目标：从“可训练 MVP”升级为“论文等价逻辑”。

任务：

1. 明确当前奖励链路：
   - CaRR server 返回单条 `outcome/rubric`。
   - 训练侧按 rollout group 重算最终奖励。
2. 实现组内 rubric 归一化：
   - 同一 prompt 的 G 条 rollout，做 `rubric / max(rubric)`。
3. 实现 gated rubric：
   - `final = (1-alpha)*outcome + alpha*outcome*normalized_rubric`。
4. 与现有 GRPO advantage 流程对齐，保证张量维度和 group 映射正确。

验收：

- 日志中可见 `outcome/rubric/final` 三类统计。
- 消融验证符合预期：关闭 gated 或归一化时性能劣化。

---

## 5. 目录与文件清单（目标态）

```text
examples/carr_deepsearch/
  IMPLEMENTATION_PLAN.md
  README.md
  tools/
    __init__.py
    carr_session_manager.py
    browser_search_tool.py
    browser_open_tool.py
    browser_find_tool.py
  reward/
    __init__.py
    carr_reward.py
  data_preprocess/
    preprocess_carr_rl.py
    preprocess_carr_sft.py
    validate_carr_data.py
  config/
    carr_grpo.yaml
    carr_sft.yaml
    tool_config/
      carr_browser_tools.yaml
  scripts/
    run_sft.sh
    run_rl.sh
    smoke_test_tools.sh
    smoke_test_reward.sh
```

---

## 6. 配置模板（最小可运行示例）

## 6.1 `carr_grpo.yaml` 关键项

```yaml
hydra:
  searchpath:
    - file://verl/trainer/config

defaults:
  - ppo_trainer
  - _self_

data:
  train_files: ~/data/carr_deepsearch/rl_train.parquet
  val_files: ~/data/carr_deepsearch/rl_val.parquet
  return_raw_chat: true
  max_prompt_length: 4096
  max_response_length: 61440
  train_batch_size: 128

actor_rollout_ref:
  hybrid_engine: true
  model:
    path: Qwen/Qwen3-4B
  actor:
    optim:
      lr: 2.0e-6
  rollout:
    name: sglang
    n: 16
    temperature: 1.0
    multi_turn:
      enable: true
      format: hermes
      max_assistant_turns: 30
      max_tool_response_length: 10000
      tool_response_truncate_side: right
      tool_config_path: examples/carr_deepsearch/config/tool_config/carr_browser_tools.yaml

algorithm:
  adv_estimator: grpo
  use_kl_in_reward: false

reward:
  custom_reward_function:
    path: examples/carr_deepsearch/reward/carr_reward.py
    name: compute_score

trainer:
  nnodes: 1
  n_gpus_per_node: 8
  total_epochs: 3
  project_name: carr_deepsearch
  experiment_name: carr-grpo
```

## 6.2 `carr_sft.yaml` 关键项

```yaml
hydra:
  searchpath:
    - file://verl/trainer/config

defaults:
  - sft_trainer
  - _self_

model:
  partial_pretrain: Qwen/Qwen3-4B

data:
  train_files: ~/data/carr_deepsearch/sft_train.parquet
  val_files: ~/data/carr_deepsearch/sft_val.parquet
  max_length: 131072
  multiturn:
    enable: true
    messages_key: messages

optim:
  lr: 4.0e-5

trainer:
  total_epochs: 3
  project_name: carr_deepsearch_sft
  experiment_name: carr-sft
```

---

## 7. 脚本规范（关键约束）

## 7.1 `run_rl.sh` 关键约束

1. 必须在脚本开始设置进程清理：`trap 'kill ...' EXIT`。
2. reward server 启动必须切目录：
   - `cd CaRR/deepsearch_rm_with_rubrics && python launch_server.py ...`
3. 训练命令必须传 `data.train_files` 与 `data.val_files`。

## 7.2 API Key 约束

- 工具服务参数名是 `--serp_api_key`，底层调用 `serpapi.com`。
- 推荐环境变量名统一为：
  - `SERPAPI_API_KEY`
  - `JINA_API_KEY`
  - `DEEPSEEK_API_KEY`

---

## 8. 验证计划（必须执行）

## 8.1 工具侧

1. `browser.search`：检查结果格式包含 `[idx] URL Source:`。
2. `browser.open`：分别验证 `id=int` 和 `id=url`。
3. `browser.find`：先 open 再 find，验证上下文命中。

## 8.2 奖励侧

1. 用固定样本直调 `/evaluate`，验证非 500。
2. 验证 `history[-1]` 为 assistant。
3. 验证 `outcome_reward` 与 `rubric_reward` 能返回非零样本。

## 8.3 端到端

1. 2 条样本、1 step 的 RL 干跑。
2. 100-500 step 小规模训练，观察：
   - `response/aborted_ratio`
   - `num_turns`
   - `outcome_reward` / `rubric_reward`
3. 1k-3k step 冒烟：确认无奖励崩塌、无会话泄漏。

---

## 9. 风险与回滚

1. 风险：奖励全零。
   - 处理：先打印并抽样落盘 `history`，定位末条消息与 tool_call_id 绑定。
2. 风险：服务端超时导致训练阻塞。
   - 处理：reward 调用设置超时与异常降级分。
3. 风险：会话泄漏。
   - 处理：训练轮次结束强制 `close_session`，并定期统计 active sessions。
4. 风险：C-GRPO 实现偏离。
   - 处理：先保留 MVP，再用单元测试校验组内归一化公式。

---

## 10. 里程碑与通过标准

- M1（链路打通）
  - 工具、奖励、训练三条链路都可运行。
- M2（MVP 可训练）
  - 1k+ step 无崩溃，奖励分布合理。
- M3（C-GRPO 对齐）
  - 组内归一化 + gated rubric 上线，完成消融。

通过标准：

1. 训练可复现（脚本 + 配置 + 数据版本固定）。
2. 关键指标可观测（至少包含 outcome/rubric/final reward）。
3. 失败模式有日志可追溯（history、tool calls、reward 请求体）。

---

## 11. 实施顺序（建议）

1. 先完成 Phase A-B（依赖与数据）。
2. 再完成 Phase C-D（工具与奖励桥接）。
3. 完成 Phase E 后先跑 MVP。
4. 最后进入 Phase F 做 C-GRPO 论文对齐。

