# CaRR DeepSearch Resume Source of Truth (2026-04-12)

## 1. 项目一句话定义

这是一个基于 `verl` 的 `agentic RL / post-training` 系统工程项目：把 CaRR 的 citation-aware deep-search 训练链路接入 `verl`，实现多轮 browser-tool agent、CaRR reward、`C-GRPO`、以及 `fully_async_policy` 异步训练稳定化，并围绕 `DeepDive + BrowseComp` 建立统一评测与诊断口径。

这份文档是当前项目对外表述的唯一事实源。任何后续写进简历、面试稿、项目介绍里的数字，都必须先在本文件中出现，并带上状态标签：

- `Observed`: 本地日志、现有代码或已有文档可直接支持
- `Estimated`: 当前无远端真实评测，按论文参考值做单点估计，仅用于简历草稿占位
- `Pending`: 必须等新 GPU 上机后补齐，当前不写数字

---

## 2. 中文简历 Bullets

### 2.1 Safe

1. 在 `verl` 中构建 CaRR-style citation-aware browser-tool RL 栈，打通多轮 `search/open/find` `AgentLoop`、CaRR-compatible `reward_history`、CaRR reward bridge 与 `C-GRPO` final-token reward fusion，并为 `64k` 长轨迹训练补齐 `unfinished/termination/tool-use` 诊断指标。
2. 稳定化 `fully_async_policy` 长轨迹训练主线，修复 `queue_full` partial drain、param-sync pause/resume、FSDP dynamic-batch rank alignment、checkpoint strict resume 与 `FSDP -> HuggingFace` 评测链路，推进到 durable `global_step_23`。
3. 设计 fixed sampled `DeepDive subset64` checkpoint-selection gate，基于 `outcome/rubric`、`unfinished_limit/budget`、`response_limit`、`rollout_timeout` 与 tool-use diagnostics 选择 async 起点，而不是默认使用更晚 checkpoint。

### 2.2 Evidence-Leaning

1. 在 `verl` 中构建 CaRR-style citation-aware browser-tool RL 栈，维护 CaRR-compatible `reward_history`，实现 `C-GRPO` final-token reward fusion，并把 `unfinished/termination/tool-use` 诊断指标透传到训练与评测链路。
2. 稳定化 `fully_async_policy` 长轨迹训练主线，修复 `queue_full` partial drain、param-sync pause/resume、FSDP dynamic-batch rank alignment 与 `FSDP -> HuggingFace` 评测链路；最终 durable run 落在 `global_step_23`，`param_sync` 从病态 `~323s` 窗口收敛到常见 `1.7-20s` 健康窗口，但最终主线仍保留 `85-144s` 级长尾。
3. 设计 fixed sampled `DeepDive subset64` gate：`SFT outcome=0.296875 / rubric=0.04475 / unfinished=0.65625`，`step70 outcome=0.328125 / rubric=0.05226 / unfinished=0.640625`，`step90 outcome=0.265625 / rubric=0.06332 / unfinished=0.671875`；按 outcome-first gate 规则选择 `step70` 作为 async 起点。

### 2.3 Draft-With-Placeholder

1. 在 `verl` 中构建 CaRR-style browser-tool RL 栈，覆盖 `reward_history` 组装、`C-GRPO` reward fusion、async partial rollouts 和长轨迹诊断。
2. 在 `8 x 96GB GPU` 条件下推进 async 主线到 durable `global_step_23`，并形成 `FSDP checkpoint -> huggingface_merged -> sampled eval` 的完整后处理链路。
3. 外部 `BrowseComp` 占位指标已在文档单独记录为 `Estimated`，仅用于简历草稿推演，不能作为正式简历 bullet 或面试结论。

### 2.4 Final Resume-Ready（中文终版）

以下 3 条是当前最推荐直接写进中文简历的终版 bullets，目标是同时展示：

- `RL infra` 能力：你不仅能调参，而且能把 async RL 主线真正稳定到可训练、可恢复、可评测
- `research / method` 能力：你理解并实现了 CaRR 的奖励设计，不是只会复现一个现成表格

推荐终版 bullets：

1. 在 `verl` 中实现面向 deep-search agent 的多轮 browser-tool RL：自定义 `search/open/find` `AgentLoop` 维护可恢复的 `reward_history`，把最终答案正确性奖励（`outcome reward`）与基于引用支撑、事实完整性和证据链连通性的 `rubric reward` 接入统一训练与评测链路。
2. 实现 `C-GRPO` 奖励设计：在组内对 `rubric reward` 做归一化，并只对答案正确的 rollout 注入加权 `rubric` 信号，在 final token 上重建奖励，避免纯二元结果奖励诱导 shortcut，也避免错误轨迹因命中局部事实被误奖励。
3. 稳定化 `fully_async_policy` 长轨迹 RL infra，修复 `queue_full` cancel-based partial drain、param-sync pause/resume、FSDP dynamic-batch rank alignment、strict resume 与 `FSDP -> HuggingFace merged` 评测链路，在 `8 x 96GB GPU` 上推进到 durable `global_step_23`，并用 fixed sampled `DeepDive subset64` gate 选择 `step70` 作为 async 起点。

使用建议：

- 如果简历更偏 `MLE / infra`，优先保留第 `3` 条，再在第 `1` 条里保留 `reward_history`、`outcome reward`、`rubric reward` 这些关键词。
- 如果简历更偏 `post-training / research engineering`，优先保留第 `2` 条，因为它最能体现你理解并实现了“为什么 CaRR 比纯 outcome reward 更合理”。
- 如果版面只够写 `2` 条，优先保留 `1 + 3`；第 `2` 条可以在面试展开时补充。

---

## 3. English Resume Bullets

### 3.1 Safe

1. Built a CaRR-style citation-aware browser-tool RL stack on `verl`, implementing a multi-turn `search/open/find` `AgentLoop`, CaRR-compatible `reward_history`, CaRR reward bridging, and `C-GRPO` final-token reward fusion with long-trajectory diagnostics.
2. Stabilized `verl` `fully_async_policy` for long-horizon agentic RL by fixing queue-full partial drain, param-sync pause/resume, FSDP dynamic-batch alignment, checkpoint recovery, and `FSDP -> HuggingFace` evaluation paths, reaching a durable `global_step_23`.
3. Built a fixed sampled `DeepDive subset64` checkpoint-selection gate using outcome, rubric, unfinished, timeout, and tool-use metrics to choose the async start point instead of defaulting to a later checkpoint.

### 3.2 Evidence-Leaning

1. Built a CaRR-style citation-aware browser-tool RL stack on `verl`, maintaining CaRR-compatible `reward_history`, implementing `C-GRPO` final-token reward fusion, and passing long-trajectory `unfinished/termination/tool-use` diagnostics into both training and evaluation.
2. Stabilized async training from pathological param-sync stalls to healthy windows: a worst-case `~323s` sync window was reduced to frequently observed `1.7-20s` healthy windows, but the final mainline still retained `85-144s` tails; the durable async run ended at `global_step_23`.
3. Ran a fixed sampled `DeepDive subset64` gate with `SFT outcome=0.296875 / rubric=0.04475 / unfinished=0.65625`, `step70 outcome=0.328125 / rubric=0.05226 / unfinished=0.640625`, and `step90 outcome=0.265625 / rubric=0.06332 / unfinished=0.671875`, selecting `step70` by the outcome-first gate rule rather than treating it as universally superior.

### 3.3 Draft-With-Placeholder

1. Built and stabilized a CaRR-style browser-tool `agentic RL` training system on `verl`, covering reward-history assembly, `C-GRPO` reward fusion, async partial rollouts, and long-trajectory diagnostics.
2. Ran the async mainline on `8 x 96GB GPU` through a durable `global_step_23` endpoint while preserving a clean `FSDP -> huggingface_merged -> sampled eval` path.
3. Paper-referenced `BrowseComp` placeholders are tracked separately for internal drafting only and must not be copied into the final resume or used as interview results.

---

## 4. Claims Matrix

| Claim | Status | Current statement | Resume-safe now? | Evidence |
|------|--------|-------------------|------------------|----------|
| 实现了 `verl + CaRR` 的多轮 browser-tool RL 流水线 | Observed | 自定义 `AgentLoop`、CaRR reward、`C-GRPO`、多轮工具调用均已接通 | Yes | 代码、实现文档 |
| async 主线最终 durable checkpoint 为 `global_step_23` | Observed | `latest_checkpointed_iteration=23`，最终 run 已写出 `global_step_23` dataloader 与 actor checkpoint | Yes | `latest_checkpointed_iteration.txt`、final async log |
| 最佳内部 async 训练窗口与最终 checkpoint 不是同一个概念 | Observed | `step16` 是目前最强的内部训练窗口，但最终 durable checkpoint 仍是 `global_step_23` | No | final async log |
| async 参数同步已显著收敛，但仍保留长尾 | Observed | 从病态 `~323s` 问题窗口收敛到常见 `1.7-20s` 健康窗口；final mainline 仍出现 `85-144s` 级 tail | Yes, with caveat | 本地 async 日志、handoff 汇总 |
| sync formal 稳定窗口 `timing_s/step` 中位数约 `506s` | Observed | 取同步 formal `step 71-90` 作为对比基线 | Yes | 本地 sync 训练日志 |
| async 主线健康窗口 `~280-320s / gs` | Observed | 这是 `trigger_sync_step=4` 下的健康窗口 `per-global-step` 归一化，不是全程平均 | Yes, with caveat | handoff 文档、local async logs |
| `step70` 被选为 async 起点 | Observed | 按 outcome-first sampled gate 规则，`step70` 被选为 async 起点 | Yes | eval 日志 |
| 已记录 paper-referenced `BrowseComp` placeholders | Estimated | placeholder 只用于内部草稿，不是项目真实外部结果 | No | `Carr_paper_data.md` |
| `BrowseComp full / 128k` 的真实外部指标 | Pending | 必须在新 GPU 上完成 merge + sampled eval 后补齐 | No | 待执行 |

---

## 5. 技术解释与证据映射

### 5.1 Built

**简历对应表述**

- 在 `verl` 上实现 citation-aware 多轮 browser-tool `agentic RL` 流水线，打通自定义 `AgentLoop`、CaRR reward 和 `C-GRPO`。

**这条话在项目里具体指什么**

- 不是简单“把论文配置搬进 verl”。
- 真正的难点是把多轮 browser-tool 轨迹整理成 reward server 真正能消费的结构：
  - 多轮 `search/open/find` 轨迹被整理成 CaRR-compatible `reward_history`
  - `unfinished_limit / unfinished_budget / termination_* / parse_error_count` 被显式编码为可训练、可评测的诊断字段
  - `C-GRPO` 不是抽象概念，而是明确的 final-token reward fusion：binary outcome 和 group-normalized rubric reward 在 advantage 阶段融合

**关键代码**

- `examples/carr_deepsearch/tools/carr_agent_loop.py:61-170`
  - `CaRRToolAgentLoop` 注册为 `carr_tool_agent`
  - 负责 rollout budget、tool budget、unfinished reason flags
- `examples/carr_deepsearch/tools/carr_agent_loop.py:655-673`
  - 在 async agent state 中维护 `reward_history`、`pending_tool_calls`、`param_version_start`、`last_param_version`
- `examples/carr_deepsearch/tools/carr_agent_loop.py:792-864`
  - 在 completed output 中拼装 `messages`、`unfinished_*`、`termination_*`、`tool_call_counts`、`param_version_span`
- `examples/carr_deepsearch/reward/carr_reward.py:104-142`
  - 把 `unfinished/termination/tool-use/response-length` 指标透传回 reward path
- `examples/carr_deepsearch/reward/cgrpo_advantage.py:15-120`
  - 通过 `@register_adv_est("cgrpo")` 注册 `C-GRPO`
  - 用 final-token reward fusion `R = (1-alpha) * outcome + alpha * outcome * norm_rubric` 重建 token-level reward

**为什么这条对求职有价值**

- 这证明你不是只会调超参，而是真的把 `reward / agent loop / PPO advantage` 三段链路接入了现成 post-training 框架。
- 对 `agentic RL` 岗位来说，这比单纯“跑过一次 RL”更有区分度。

**相关背景文档**

- `docs_zh/agent_training/01_架构设计.md`
- `docs_zh/agent_training/02_AgentLoop详解.md`
- `docs_zh/agent_training/05_奖励系统.md`
- `CaRR/docs/01_论文概述与核心贡献.md`
- `CaRR/docs/02_CaRR奖励框架详解.md`
- `CaRR/docs/03_C-GRPO训练算法详解.md`
- `IMPLEMENTATION_PLAN.md`
- `examples/carr_deepsearch/CLAUDE.md`

### 5.2 Stabilized

**简历对应表述**

- 扩展并稳定化 `fully_async_policy`，解决参数同步、partial rollout/backpressure、resume 与 checkpoint 可用性问题。

**这条话在项目里具体指什么**

- async 训练不是“把 sync 训练改成异步启动脚本”。
- 真正做的事情包括：
  - trainer 在 param sync 前 pause rollout，再同步 actor 和 rollout 权重，并校验 fingerprint
  - async partial rollout 需要在 cancellation 之后继续恢复，因此 agent state 里要带 `param_version`、budget 状态和当前 assistant turn
  - queue-full 时通过 cancel-based partial drain 做 backpressure，而不是把系统直接拖死
  - `dynbsz + FSDP` 下 actor/critic 的 micro-batch 数必须在 DP rank 间对齐
  - checkpoint 既要能 strict resume，也要能走 model-only eval

**关键代码**

- `verl/experimental/fully_async_policy/param_sync.py:163-219`
  - `sync_weights()` 里明确先 `pause`，再 sync，后做 fingerprint validation，最后 `update_param_version` 与 `resume`
- `verl/experimental/fully_async_policy/fully_async_rollouter.py:249-308`
  - `update_param_version()` 负责 async 版本推进、staleness 重置、validate 触发
- `verl/experimental/fully_async_policy/fully_async_rollouter.py:568-610`
  - `_processor_worker()` 在 `queue_full` 时执行 partial drain cancellation
- `examples/carr_deepsearch/tools/carr_agent_loop.py:655-673`
  - 记录 `param_version_start / last_param_version / real_rollout_start_s`
- `examples/carr_deepsearch/tools/carr_agent_loop.py:712-767`
  - cancellation-aware generation；partial rollout 被 cancel 后继续累计 response ids
- `examples/carr_deepsearch/tools/carr_agent_loop.py:877-899`
  - canceled output 以显式 `is_cancel=True` 方式返回，便于后续恢复
- `verl/workers/actor/dp_actor.py:518-527`
  - actor 侧 dynamic batch 使用 `dp_group=self.dp_group` 且 `same_micro_num_in_dp=True`
- `verl/workers/critic/dp_critic.py:201-210`
  - critic 侧采用相同的 dynamic batch 对齐约束
- `verl/experimental/fully_async_policy/fully_async_main.py`
  - async 入口，组织 `MessageQueue`、`Rollouter`、`Trainer`、`ParameterSynchronizer`
- `examples/carr_deepsearch/scripts/run_rl_async.sh`
  - async 主启动脚本
- `examples/carr_deepsearch/scripts/run_async_formal.sh:45-177`
  - formal async 训练和 eval gate 的资产解析、HF checkpoint 路径解析、sampled gate 参数
- `verl/model_merger/__main__.py:15-73`
  - 明确支持 `FSDP checkpoint -> HuggingFace merged model` 的标准 merge 入口

**当前已观测到的系统结果**

- `Observed`: async 最终 durable checkpoint 是 `global_step_23`
- `Observed`: `param_sync` 从问题窗口约 `323s` 降到常见健康窗口 `1.7-20s`，但 final mainline 仍保留 `85-144s` 级长尾
- `Observed`: sync formal `step 71-90` 的 `timing_s/step` 中位数约 `506s`
- `Observed`: async 主线健康窗口约 `280-320s / gs`，但这是健康窗口、且是 `per-global-step` 归一化，不是全程平均

**为什么这条对求职有价值**

- 这条卖点是“你能让一个不稳定的 agentic RL infra 真正跑起来”，而不只是“能看懂论文”。
- 对后训练岗位而言，`resume / checkpoint / param sync / queue backpressure / sampled eval` 都是非常真实的工程问题。

**相关背景文档**

- `docs_zh/agent_training/06_训练流程.md`
- `docs_zh/agent_training/07_配置参考.md`
- `examples/carr_deepsearch/docs/agent_handoff_background_20260321.md`
- `examples/carr_deepsearch/docs/async_rl_explainer_20260321.md`
- `examples/carr_deepsearch/docs/sync_vs_async_grpo_update_mechanics.md`
- `examples/carr_deepsearch/DEVELOPMENT_LOG.md`

### 5.3 Evaluated

**简历对应表述**

- 建立 `DeepDive + BrowseComp` 的统一评测与诊断思路，明确哪些结果已证实、哪些仍待远端补证据。

**这条话在项目里具体指什么**

- 对 Thinking checkpoint，eval 不能默认 greedy。
- 内部 gate 和外部 gate 也不能混用不同 recipe。
- 评测不仅要看 `outcome_reward`，还要同时看：
  - `rubric_reward`
  - `task_unfinished`
  - `termination_response_limit`
  - `termination_rollout_timeout`
  - `tool_call_counts / num_turns`

**关键代码与脚本**

- `examples/carr_deepsearch/scripts/run_eval_integration.sh:1-189`
  - 启动 tool/reward server 并以 `val_only=True` 方式跑集成评测
  - sampled eval 参数从 `CARR_VAL_*` 环境变量读取
- `examples/carr_deepsearch/scripts/run_async_formal.sh:91-165`
  - `eval_gate` 模式下固定 `DeepDive subset64` sampled eval recipe
- `examples/carr_deepsearch/reward/carr_reward.py:104-142`
  - 把 unfinished / termination / tool counts 等 agent-specific 指标透传回评测结果

**关键背景文档**

- `examples/carr_deepsearch/docs/rl_debug_findings_20260312.md:845-912`
  - 明确说明对 `Qwen3-4B-Thinking-2507`，greedy eval 会误导，sampled recipe 才是主 gate
- `examples/carr_deepsearch/docs/agent_handoff_background_20260321.md:158-170`
  - 说明 `Stage 0 eval` 已完成、async 起点选 `step70`
- `examples/carr_deepsearch/docs/agent_handoff_background_20260321.md:197-233`
  - 记录 async 全 step 指标历史、健康窗口和主瓶颈
- `examples/carr_deepsearch/docs/Carr_paper_data.md`
  - 给出论文侧的 `BrowseComp`、`C-GRPO`、tool-call 行为解释，可用作 placeholder 来源

**当前应如何解释 `step70 vs step90`**

- `step70` 不是“所有指标都更好”。
- `Observed`: `step90` 的 `rubric_reward` 更高，但 `step70` 的 `outcome_reward` 和 `task_unfinished` 更适合作为 async 起点。
- 因此文档里应始终写成“按 outcome-first sampled gate 规则，`step70` 被选为 async 起点”，而不是笼统写成 “`step70` 全面优于 `step90`”。

**为什么这条对求职有价值**

- 面试官通常不只关心“有没有分数”，更关心你如何定义可信评测。
- 这条体现的是你的实验设计、指标选择、failure-mode diagnosis 能力。

---

## 6. 当前已观测结果

### 6.1 DeepDive Subset64 Sampled Eval

| Checkpoint | outcome | rubric | unfinished | Status | Source |
|-----------|---------|--------|------------|--------|--------|
| SFT | `0.296875` | `0.04475` | `0.65625` | Observed | `examples/carr_deepsearch/CaRR_log/eval_gate_20260323_144500_sft.log` |
| sync step70 | `0.328125` | `0.05226` | `0.640625` | Observed | `examples/carr_deepsearch/CaRR_log/step70_dd64_8gpu_20260322_153257.log` |
| sync step90 | `0.265625` | `0.06332` | `0.671875` | Observed | `examples/carr_deepsearch/CaRR_log/step90_dd64_8gpu_20260322_151340.log` |

当前可对外表述：

- `Observed`: 在统一 sampled-eval 口径下，按 outcome-first gate 规则，`step70` 被选为 async 起点。
- `Observed`: `step90` 的 `rubric_reward` 更高，但它在 `outcome_reward` 和 `unfinished` 上不如 `step70`，因此没有被选作 async 起点。

### 6.2 Sync Formal Baseline

| Metric | Value | Status | Source |
|-------|-------|--------|--------|
| `timing_s/step` median on sync formal `step 71-90` | `~506s` | Observed | `examples/carr_deepsearch/CaRR_formal_training_log/output.log` |
| `timing_s/step` sample points | `520.24s`, `494.24s`, `531.17s` | Observed | 同上 |

当前可对外表述：

- `Observed`: 同步 formal 后段稳定窗口是 `~506s / step` 量级。

### 6.3 Final Durable Async Checkpoint

| Metric | Value | Status | Source |
|-------|-------|--------|--------|
| `latest_checkpointed_iteration` | `23` | Observed | `examples/carr_deepsearch/CaRR_log/latest_checkpointed_iteration.txt` |
| Final durable checkpoint folder | `global_step_23` | Observed | `examples/carr_deepsearch/CaRR_log/formal_mainline_gs14_sp2_dynbsz_gmu05_b3_reasonfix_20260323_215746.launcher.log` |
| Final-step `param_sync` tail | `85.94s` | Observed | 同上 |
| Healthy floor observed in final mainline | `1.69-1.78s` on steps `16/18/20/22` | Observed | 同上 |
| Residual long tails observed across mainline | `101.81s`, `144.03s`, `208.96s` | Observed | `gs14/gs11` async logs |
| Healthy `per-global-step` window | `~280-320s / gs` | Observed | `examples/carr_deepsearch/docs/agent_handoff_background_20260321.md` |

当前可对外表述：

- `Observed`: async 主线已经具备“能稳定训练、能 resume、能留下可评测 checkpoint”的工程可用性。
- `Observed`: `global_step_23` 是最终 durable checkpoint，但它不等于“最佳 checkpoint by quality”。
- `Observed`: async 系统结果必须按“健康窗口 + 剩余长尾”一起解释，不能把 `1.7s` 或 `280-320s / gs` 写成全程平均。

### 6.4 Best Observed Async Training Window

| Metric | Value | Status | Source |
|-------|-------|--------|--------|
| Best observed internal training window | `step16` | Observed | `examples/carr_deepsearch/CaRR_log/formal_mainline_gs14_sp2_dynbsz_gmu05_b3_reasonfix_20260323_215746.launcher.log` |
| `outcome_reward/mean` | `0.59375` | Observed | 同上 |
| `rubric_reward/mean` | `0.09308` | Observed | 同上 |
| `task_unfinished/ratio` | `0.29167` | Observed | 同上 |
| `param_sync` | `1.69s` | Observed | 同上 |
| `per-global-step` healthy window | `~284s / gs` | Observed | `examples/carr_deepsearch/docs/agent_handoff_background_20260321.md` |

当前可对外表述：

- `Observed`: `step16` 是目前最强的内部训练窗口，但它只是训练期窗口，不是 formal sampled eval checkpoint。
- `Observed`: 面试里可以把它当作“最佳内部训练信号”，不能把它直接写成最终 benchmark 结果。

---

## 7. 外部 Benchmark Placeholder

These are paper-referenced placeholders for resume drafting only and must be replaced by real BrowseComp eval before final use.

### 7.1 Placeholder Rule

- 只对 `BrowseComp` 外部 benchmark 做占位，不对 `DeepDive subset64`、`sync/async timing`、`param_sync` 做估计。
- 占位值来源：`examples/carr_deepsearch/docs/Carr_paper_data.md` 中的 CaRR 论文 4B `BrowseComp` 结果。
- 固定估计公式：
  - `paper_uplift = paper_cgrpo - paper_sft`
  - `estimated_step70 = paper_sft + 0.25 * paper_uplift`
  - `estimated_async_best = paper_sft + 0.40 * paper_uplift`

### 7.2 BrowseComp Placeholder Table

| Eval target | Context | paper_sft | paper_cgrpo | estimated_step70 | estimated_async_best | Status |
|------------|---------|-----------|-------------|------------------|----------------------|--------|
| BrowseComp subset256 | `64k` | `7.7` | `13.9` | `9.3` | `10.2` | Estimated |
| BrowseComp full | `64k` | `7.7` | `13.9` | `9.3` | `10.2` | Estimated |
| BrowseComp subset256 | `128k` | `14.1` | `17.5` | `15.0` | `15.5` | Estimated |
| BrowseComp full | `128k` | `14.1` | `17.5` | `15.0` | `15.5` | Estimated |

使用说明：

- 这些数值只是一种“合理的简历草稿占位”，不是当前项目已经测出来的外部 benchmark。
- `subset256` 与 `full` 当前都借用了论文 full benchmark 的 4B 数值作为 provisional prior，因此绝不能原样写进正式简历。
- 一旦真实 `BrowseComp` 结果到位，本节应整体替换，不保留任何 placeholder。

---

## 8. 面试讲法

### 8.1 三句话版本

1. 这是一个 `agentic RL / post-training` 系统工程项目，不是单纯复现论文表格。
2. 我真正完成的工作是把 CaRR 的多轮工具调用、citation-aware reward 和 `C-GRPO` 接进 `verl`，并把 async RL 主线稳定到可训练、可 resume、可评测。
3. 当前内部 `DeepDive` 评测和系统日志已经足够支撑工程能力表述；外部 `BrowseComp` 还在等待新 GPU 上机后补齐，所以文档里严格区分了 `Observed` 和 `Estimated`。

### 8.2 展开版

- 不要把项目讲成“我复现了 CaRR 并得到论文同等结果”。
- 要讲成：
  - 为什么这个任务属于典型 `agentic RL`
  - 你在 `verl` 里具体改了哪些层
  - 为什么 async 稳定化比再多刷几十 step 更有工程价值
  - 你如何判断一个 Thinking checkpoint 的 eval 是可信的
  - 为什么 “最终 durable checkpoint” 和 “最佳内部训练窗口” 不能混为一谈
- 面试官如果追问“为什么没有直接给 BrowseComp 真值”，就直接回答：
  - 当前远端机器不可用，外部 benchmark 正在等新 GPU 环境恢复
  - 所以我在文档里只把本地可验证的内部指标和系统指标标为 `Observed`
  - 所有 placeholder 都集中放在外部 benchmark 小节，并明确标注 `Estimated`

### 8.3 不要这样讲

- 不要说“我复现了论文结果”
- 不要说“async 一定比 sync 效果更好”
- 不要把 raw step 数当作主卖点
- 不要把 `timeout=0%` 当成唯一 headline
- 不要把 healthy-window 数字写成 whole-run average

---

## 9. 参考文档与代码索引

### 9.1 实现代码

| Path | 用途 |
|------|------|
| `examples/carr_deepsearch/tools/carr_agent_loop.py` | `CaRRToolAgentLoop` 与 `CaRRAsyncPartialToolAgentLoop` 实现 |
| `examples/carr_deepsearch/reward/carr_reward.py` | CaRR reward server bridge 与 agent-specific metrics pass-through |
| `examples/carr_deepsearch/reward/cgrpo_advantage.py` | `C-GRPO` 优势估计器 |
| `verl/experimental/fully_async_policy/fully_async_main.py` | async 主入口 |
| `verl/experimental/fully_async_policy/fully_async_rollouter.py` | async rollout、pause/resume、partial drain |
| `verl/experimental/fully_async_policy/fully_async_trainer.py` | async trainer 端训练推进 |
| `verl/experimental/fully_async_policy/param_sync.py` | 参数同步、fingerprint validation、resume 调度 |
| `verl/workers/actor/dp_actor.py` | actor 侧 dynbsz 与 DP rank 对齐 |
| `verl/workers/critic/dp_critic.py` | critic 侧 dynbsz 与 DP rank 对齐 |
| `examples/carr_deepsearch/scripts/run_rl_async.sh` | async RL 主启动脚本 |
| `examples/carr_deepsearch/scripts/run_async_formal.sh` | formal async launcher、eval gate、HF 目录解析 |
| `examples/carr_deepsearch/scripts/run_eval_integration.sh` | sampled integration eval 入口 |
| `verl/model_merger/__main__.py` | FSDP -> `huggingface_merged` 标准 merge 入口 |

### 9.2 背景文档

| Path | 用途 |
|------|------|
| `docs_zh/agent_training/01_架构设计.md` | `verl` agent training 架构背景 |
| `docs_zh/agent_training/02_AgentLoop详解.md` | AgentLoop 语义 |
| `docs_zh/agent_training/05_奖励系统.md` | reward 系统与扩展方式 |
| `docs_zh/agent_training/06_训练流程.md` | 训练流程总览 |
| `docs_zh/agent_training/07_配置参考.md` | 配置项背景 |
| `CaRR/docs/01_论文概述与核心贡献.md` | CaRR 方法概览 |
| `CaRR/docs/02_CaRR奖励框架详解.md` | rubric / citation reward 逻辑 |
| `CaRR/docs/03_C-GRPO训练算法详解.md` | `C-GRPO` 算法解释 |
| `CaRR/docs/08_实验结果与消融分析.md` | 论文结果和消融 |
| `IMPLEMENTATION_PLAN.md` | 项目初始实施方案 |
| `examples/carr_deepsearch/DEVELOPMENT_LOG.md` | 开发历史与修复过程 |
| `examples/carr_deepsearch/CLAUDE.md` | 项目总览 |
| `examples/carr_deepsearch/docs/agent_handoff_background_20260321.md` | async handoff 总文档与关键结论 |
| `examples/carr_deepsearch/docs/async_rl_explainer_20260321.md` | async 机制解释 |
| `examples/carr_deepsearch/docs/rl_debug_findings_20260312.md` | Thinking checkpoint 的 sampled eval 原因 |
| `examples/carr_deepsearch/docs/Carr_paper_data.md` | 论文结果和 placeholder 来源 |
| `examples/carr_deepsearch/docs/sync_vs_async_grpo_update_mechanics.md` | sync vs async 更新机制对比 |

### 9.3 本地日志证据

| Path | 用途 |
|------|------|
| `examples/carr_deepsearch/CaRR_log/latest_checkpointed_iteration.txt` | durable checkpoint iteration 标记 |
| `examples/carr_deepsearch/CaRR_log/eval_gate_20260323_144500_sft.log` | `SFT` 的 `DeepDive subset64` sampled eval |
| `examples/carr_deepsearch/CaRR_log/step70_dd64_8gpu_20260322_153257.log` | `step70` 的 `DeepDive subset64` sampled eval |
| `examples/carr_deepsearch/CaRR_log/step90_dd64_8gpu_20260322_151340.log` | `step90` 的 `DeepDive subset64` sampled eval |
| `examples/carr_deepsearch/CaRR_formal_training_log/output.log` | sync formal `step 71-96` 日志与 `timing_s/step` |
| `examples/carr_deepsearch/CaRR_log/formal_mainline_gs14_sp2_dynbsz_gmu05_b3_reasonfix_20260323_215746.launcher.log` | async 主线最终 run、`global_step_23` 与 queue/full、param sync 相关日志 |
| `examples/carr_deepsearch/CaRR_log/formal_mainline_gs11_sp2_dynbsz_gmu05_b3_reasonfix_20260323_091930.launcher.log` | async 主线 earlier tail 与 best-window 对照日志 |

### 9.4 当前对外使用规则

- 正式简历默认只使用 `Safe` 或 `Evidence-Leaning`。
- `Draft-With-Placeholder` 只能保留在内部草稿或这份文档中。
- 任何涉及 `BrowseComp` uplift 的表述，必须先从 `Estimated` 更新成 `Observed`。
