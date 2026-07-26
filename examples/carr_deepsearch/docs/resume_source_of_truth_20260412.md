# CaRR DeepSearch Resume Source of Truth (2026-04-12, status updated 2026-07-26)

## 1. 项目一句话定义

这是一个基于 `verl` 的 `agentic RL / post-training` 系统工程项目：把 CaRR 的 citation-aware deep-search 训练链路接入 `verl`，实现多轮 browser-tool agent、CaRR reward、`C-GRPO`、以及 `fully_async_policy` 异步训练稳定化，并围绕 `DeepDive + BrowseComp` 建立统一评测与诊断口径。

这份文档是当前项目对外表述的唯一事实源。任何后续写进简历、面试稿、项目介绍里的数字，都必须先在本文件中出现，并带上状态标签：

- `Observed`: 本地日志、现有代码或已有文档可直接支持
- `Estimated`: 只允许用于尚未执行的 full/128K 等可选扩展的内部情景草稿，不得进入正式简历
- `Pending`: 尚未执行的可选扩展，当前不写数字；不代表本项目还没有完成简历闭环

Recruiter-facing 简历 bullets 默认优先写“可理解的效果指标”，例如训练迭代时间、内部 anchor 的 outcome/unfinished 变化、和外部 sampled gate 的真实正向证据；`global_step_23`、`step70` 这类内部 checkpoint 语义只保留在证据层和面试展开里，不直接写进最终简历。

---

## 2. 中文简历 Bullet 写法建议

下面给的是“建议写法模式”，不是最终定稿。目标是：

- 先把最强的 `Observed` 证据写进去
- 故意保留 1 个方法 hook、1 个系统 hook、1 个评测 hook，让面试官有东西可追问
- 不把 `step70`、`global_step_23` 这种内部语义直接暴露在 recruiter-facing 文本里

### 2.1 Safe

1. 在 `verl` 中构建 CaRR-style citation-aware browser-tool RL 栈，打通多轮 `search/open/find` `AgentLoop`、CaRR-compatible `reward_history`、CaRR reward bridge 与 `C-GRPO` final-token reward fusion，并为 `64k` 长轨迹训练补齐 `unfinished/termination/tool-use` 诊断指标。
2. 稳定化 `fully_async_policy` 长轨迹训练主线，修复 `queue_full` partial drain、param-sync pause/resume、FSDP dynamic-batch rank alignment、strict resume 与 `FSDP -> HuggingFace` 评测链路，把系统从同步栅栏主导的长尾截断推进到可持续训练、可恢复、可评测的异步 RL 主线。
3. 设计统一 sampled eval 与 failure-mode diagnosis 流程，用 `outcome/rubric`、`unfinished`、`response_limit`、`rollout_timeout` 和 tool-use diagnostics 评估长轨迹质量，并用同一口径连接 `DeepDive` 内部评测与 `BrowseComp` 外部 gate。

### 2.2 Evidence-Leaning

1. 在 `verl` 中构建 CaRR-style citation-aware browser-tool RL 栈，维护 CaRR-compatible `reward_history`，实现 `C-GRPO` final-token reward fusion，并把 `unfinished/termination/tool-use` 诊断指标透传到训练与评测链路。
2. 在 `8 x 96GB GPU` 的健康窗口中，async 主线把训练迭代时间从同步 formal 后段约 `506s / step` 压到约 `280-320s` 的 per-global-step 等效区间，约为 `37-45%` 的健康窗口迭代时间缩短，并把 rollout group size 从同步实测 `n=4` 恢复到异步主线 `n=8`。
3. 在统一 sampled eval 下，`async23` 在 `DeepDive rl_val(111)` 上把 `outcome_reward` 从 `0.180` 提到 `0.324`、把 `unfinished` 从 `0.730` 降到 `0.631`；在 matched sampled `BrowseComp subset256 64k` 外部 gate 上，把 judge-pass 样本从 `4/256` 提到 `9/256`，finished answers 从 `11` 提到 `17`。

说明：

- 第 `2` 条是系统工程结果，不是 strict matched ablation，因为 async 阶段同时改变了 batch geometry 并放宽了 rollout budget：同步 formal 为观测到的 `b8/n4 + wall360`，async 主线为 `b3/n8 + wall480/real960 + trigger_sync_step=4`。
- 第 `3` 条应始终写成 “matched sampled subset eval 的真实正向证据”，不要写成“复现论文结果”或“显著提升 full benchmark”。

### 2.3 Hook-Oriented

1. 方法 hook 写法建议：保留 `CaRR-compatible reward_history` 或 `C-GRPO final-token reward fusion` 这种词，不在简历正文里解释公式。这样会自然引出“为什么 rubric 只给正确轨迹”或“为什么 reward 放在 final token”这类高质量追问。
2. 系统 hook 写法建议：保留 `queue_full cancel-based partial drain`、`param-sync pause/resume`、`async partial rollout` 这类词，但不要在简历正文里解释队列机制。这样会自然引出“跨 param-version 的 cancel/resume 怎么做”。
3. 评测 hook 写法建议：保留 `matched sampled eval`、`unfinished/response_limit diagnosis`、`BrowseComp subset256` 这类词，但不要在简历正文里展开 greedy-vs-sampled 或 budget 细节。这样会自然引出“为什么 Thinking checkpoint 不能直接 greedy eval”。

### 2.4 选词建议与不要踩的坑

建议故意留下的 hook：

- `reward_history`
- `C-GRPO final-token reward fusion`
- `queue_full partial drain`
- `matched sampled eval`
- `failure-mode diagnostics`

不要留下歧义的写法：

- 不要只写 `BrowseComp`，必须写 `BrowseComp subset256 64k`
- 不要只写 “benchmark uplift”，要明确它是 sampled external subset gate
- 不要把 `step70`、`step16`、`global_step_23` 写进 recruiter-facing bullet
- 不要把 `~506s -> ~280-320s` 写成 whole-run average；要写成 healthy-window systems result

---

## 3. English Bullet Writing Guidance

### 3.1 Safe

1. Built a CaRR-style citation-aware browser-tool RL stack on `verl`, implementing a multi-turn `search/open/find` `AgentLoop`, CaRR-compatible `reward_history`, CaRR reward bridging, and `C-GRPO` final-token reward fusion with long-trajectory diagnostics.
2. Stabilized `verl` `fully_async_policy` for long-horizon agentic RL by fixing queue-full partial drain, param-sync pause/resume, FSDP dynamic-batch alignment, strict checkpoint recovery, and `FSDP -> HuggingFace` evaluation paths, turning a sync-barrier-dominated workflow into a trainable, resumable, and evaluable async RL mainline.
3. Built a unified sampled evaluation and failure-mode diagnosis workflow using outcome, rubric, unfinished, timeout, and tool-use metrics to connect internal `DeepDive` validation with external `BrowseComp` gates.

### 3.2 Evidence-Leaning

1. Built a CaRR-style citation-aware browser-tool RL stack on `verl`, maintaining CaRR-compatible `reward_history`, implementing `C-GRPO` final-token reward fusion, and passing long-trajectory `unfinished/termination/tool-use` diagnostics into both training and evaluation.
2. In observed healthy windows on `8 x 96GB` GPUs, the async mainline reduced per-global-step-equivalent iteration time from about `506s` in the sync formal late window to roughly `280-320s`, while restoring rollout group size from observed sync `n=4` to async `n=8`.
3. Under matched sampled evaluation, `async23` improved `DeepDive rl_val(111)` `outcome_reward` from `0.180` to `0.324` and reduced `unfinished` from `0.730` to `0.631`; on `BrowseComp subset256 64k`, it increased judge-pass samples from `4/256` to `9/256`, with finished answers rising from `11` to `17`.

Notes:

- The timing result is a systems claim, not a strict matched ablation: async also changed batch geometry and widened rollout budgets from observed sync `wall360` to async `wall480/real960`.
- The external result should always be framed as matched sampled evidence on `BrowseComp subset256 64k`, not as paper reproduction or a full-benchmark claim.

### 3.3 Hook-Oriented

1. Method hook to leave in the bullet: `CaRR-compatible reward_history` or `C-GRPO final-token reward fusion`. This naturally invites follow-ups about why rubric rewards are only injected into correct trajectories.
2. Infra hook to leave in the bullet: `cancel-based partial drain`, `param-sync pause/resume`, or `async partial rollouts`. This naturally invites follow-ups about cancel/resume across parameter versions.
3. Eval hook to leave in the bullet: `matched sampled eval`, `failure-mode diagnostics`, or `BrowseComp subset256`. This naturally invites follow-ups about why Thinking checkpoints should not be judged with greedy decoding.

---

## 4. Claims Matrix

| Claim | Status | Current statement | Resume-safe now? | Evidence |
|------|--------|-------------------|------------------|----------|
| 实现了 `verl + CaRR` 的多轮 browser-tool RL 流水线 | Observed | 自定义 `AgentLoop`、CaRR reward、`C-GRPO`、多轮工具调用均已接通 | Yes | 代码、实现文档 |
| async 健康窗口的训练迭代时间明显短于同步 formal 后段 | Observed | 同步 formal 后段约 `506s / step`，async 健康窗口约 `280-320s / global-step-equivalent`，约 `37-45%` healthy-window iteration-time reduction | Yes, with caveat | sync log、async logs |
| async 参数同步已显著收敛，但仍保留长尾 | Observed | 从病态 `~323s` 问题窗口收敛到常见 `1.7-20s` 健康窗口；当前仍可见 `85.94s / 101.81s / 144.03s / 208.96s` 级 residual tails | Yes, with caveat | 本地 async 日志 |
| sync formal 稳定窗口 `timing_s/step` 中位数约 `506s` | Observed | 取同步 formal `step 71-90` 作为对比基线；实测同步 formal run 为 `b8/n4`、`wall=360` | Yes | 本地 sync 训练日志、W&B artifact（strings extracted） |
| `DeepDive rl_val(111)` 内部 anchor 已给出真实正向结果 | Observed | `SFT -> async23` 的 `outcome_reward` 为 `0.180 -> 0.324`，`unfinished` 为 `0.730 -> 0.631` | Yes | `dd111_*_reward_trace.jsonl` |
| `BrowseComp subset256 64k` 外部 gate 已给出真实正向结果 | Observed | matched sampled eval 下，judge-pass `4/256 -> 9/256`，finished answers `11 -> 17` | Yes, with subset caveat | `bc256_*_reward_trace.jsonl` |
| `step70` 被选为 async 起点 | Observed | 按 outcome-first sampled Stage 0 gate 规则，`step70` 被选为 async 起点；它是训练起点，不是最终效果指标 | No | eval 日志 |
| async 主线最终 durable checkpoint 为 `global_step_23` | Observed | `latest_checkpointed_iteration=23`，最终 run 已写出 `global_step_23` dataloader 与 actor checkpoint | No | `latest_checkpointed_iteration.txt`、final async log |
| `BrowseComp full 64k / subset256 128k / full 128k` 的真实外部指标 | Pending | 当前 source-of-truth 不再需要 placeholder；这几项保持 pending 即可 | No | 待执行 |

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

- `Observed`: 本地可确认的同步 formal 实测几何是 `b8/n4`，`max_rollout_wall_time_s=360`，`max_assistant_turns=120`，`max_tool_response_length=6000`
- `Observed`: 当前 async 主线稳定几何是 `2:6 + SP=2 + b3/n8 + wall480/real960 + trigger_sync_step=4`
- `Observed`: sync formal `step 71-90` 的 `timing_s/step` 中位数约 `506s`；async 主线健康窗口约 `280-320s / gs`
- `Observed`: 这组 `~506s -> ~280-320s` 只能解释为系统工程改善，不是严格同配 ablation，因为 async 阶段同时恢复了 `n=8` rollout groups 并放宽了 rollout wall budget
- `Observed`: `param_sync` 从问题窗口约 `323s` 降到常见健康窗口 `1.7-20s`，但当前仍可见 `85.94s / 101.81s / 144.03s / 208.96s` 级 residual tails
- `Observed`: `global_step_23` 只用于证明 async 主线已经能稳定落 checkpoint，不是 recruiter-facing 简历指标

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

- 建立 `DeepDive + BrowseComp` 的统一评测与诊断思路，明确哪些结果已证实、哪些仍是 pending。

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
- `examples/carr_deepsearch/docs/eval_analysis_20260415.md`
  - 汇总 `dd111` 与 `BrowseComp subset256 64k` 的真实 sampled eval 结果
- `examples/carr_deepsearch/docs/training_full_history_20260404.md`
  - 汇总 sync/async 几何、budget 与 checkpoint-selection 背景

**当前应如何解释 `step70 vs step90`**

- `step70` 不是“所有指标都更好”。
- `Observed`: `step90` 的 `rubric_reward` 更高，但 `step70` 的 `outcome_reward` 和 `task_unfinished` 更适合作为 async 起点。
- 因此文档里应始终写成“按 outcome-first sampled gate 规则，`step70` 被选为 async 起点”，而不是笼统写成 “`step70` 全面优于 `step90`”。

**当前已完成、可直接引用的真实评测**

- `Observed`: `DeepDive rl_val(111)` 已完成 `SFT vs async23` matched sampled eval，可作为当前最强的内部质量证据。
- `Observed`: `BrowseComp subset256 64k` 已完成 `SFT vs async23` matched sampled eval，可作为当前真实外部证据。
- `Observed`: `BrowseComp` 不使用 CaRR rubric，因此外部 uplift 反映的是 final-answer correctness 与 completion behavior，而不是 rubric-score inflation。

**为什么这条对求职有价值**

- 面试官通常不只关心“有没有分数”，更关心你如何定义可信评测。
- 这条体现的是你的实验设计、指标选择、failure-mode diagnosis 能力。

---

## 6. 当前已观测结果

### 6.1 Stage 0: DeepDive Subset64 Sampled Gate

| Checkpoint | outcome | rubric | unfinished | Status | Source |
|-----------|---------|--------|------------|--------|--------|
| SFT | `0.296875` | `0.04475` | `0.65625` | Observed | `examples/carr_deepsearch/CaRR_log/eval_gate_20260323_144500_sft.log` |
| sync step70 | `0.328125` | `0.05226` | `0.640625` | Observed | `examples/carr_deepsearch/CaRR_log/step70_dd64_8gpu_20260322_153257.log` |
| sync step90 | `0.265625` | `0.06332` | `0.671875` | Observed | `examples/carr_deepsearch/CaRR_log/step90_dd64_8gpu_20260322_151340.log` |

当前可对外表述：

- `Observed`: 在统一 sampled-eval 口径下，按 outcome-first gate 规则，`step70` 被选为 async 起点。
- `Observed`: `step90` 的 `rubric_reward` 更高，但它在 `outcome_reward` 和 `unfinished` 上不如 `step70`，因此没有被选作 async 起点。
- 这一组数字只用于 checkpoint-selection 历史，不应作为“最终项目效果”主证据。

### 6.2 Sync Formal Baseline

| Metric | Value | Status | Source |
|-------|-------|--------|--------|
| Observed sync formal geometry | `b8/n4`, `ppo_mini_batch_size=4`, `wall=360`, `max_assistant_turns=120`, `max_tool_response_length=6000` | Observed | `examples/carr_deepsearch/CaRR_formal_training_log/run-fqv0jkxr.wandb`（strings extracted） |
| `timing_s/step` median on sync formal `step 71-90` | `506.46s` | Observed | `examples/carr_deepsearch/CaRR_formal_training_log/output.log` |
| `timing_s/step` sample points | `520.24s`, `494.24s`, `531.17s` | Observed | 同上 |

当前可对外表述：

- `Observed`: 同步 formal 后段稳定窗口是 `~506s / step` 量级。
- `Observed`: 这个同步基线是实测 `b8/n4 + wall360` 的 run，不是最初 YAML 里的 `b16/n8` 计划值。

### 6.3 Async Mainline System Window

| Metric | Value | Status | Source |
|-------|-------|--------|--------|
| Observed async mainline geometry | `2:6`, `SP=2`, `ppo_mini_batch_size=3`, `rollout.n=8`, `wall=480`, `real_wall=960`, `trigger_sync_step=4` | Observed | `examples/carr_deepsearch/CaRR_log/formal_mainline_gs14_sp2_dynbsz_gmu05_b3_reasonfix_20260323_215746.launcher.log` |
| `latest_checkpointed_iteration` | `23` | Observed | `examples/carr_deepsearch/CaRR_log/latest_checkpointed_iteration.txt` |
| Final durable checkpoint folder | `global_step_23` | Observed | `examples/carr_deepsearch/CaRR_log/formal_mainline_gs14_sp2_dynbsz_gmu05_b3_reasonfix_20260323_215746.launcher.log` |
| Final-run block timings (`4 gs` each) | `1136.32s`, `1216.17s`, `1267.85s`, `1281.15s`, `1275.74s` | Observed | 同上 |
| Per-global-step equivalent | `284.08s`, `304.04s`, `316.96s`, `320.29s`, `318.93s` | Observed | 同上 |
| Final-step `param_sync` tail | `85.94s` | Observed | 同上 |
| Healthy floor observed in final mainline | `1.68-1.76s` on steps `19-22` | Observed | 同上 |
| Residual long tails observed across mainline | `101.81s`, `144.03s`, `208.96s` | Observed | `formal_mainline_gs14...launcher.log`, `formal_mainline_gs11...launcher.log` |

当前可对外表述：

- `Observed`: async 主线在健康窗口里大致跑到 `~280-320s / global-step-equivalent`，对比同步 formal 后段 `~506s / step`，对应约 `37-45%` 的 healthy-window iteration-time reduction。
- `Observed`: 这条速度结果必须和配置变化一起解释，因为 async 主线同时从实测同步 `n=4` 恢复到 `n=8`，并把 rollout wall budget 从 `360` 放宽到 `480/960`。
- `Observed`: `global_step_23` 只是工程证据，证明 async 主线已经具备“能稳定训练、能 resume、能留下可评测 checkpoint”的可用性；它不是最终简历里要写的效果指标。

### 6.4 Internal Anchor: DeepDive `rl_val(111)`

| Metric | `SFT` | `async23` | Delta | Status | Source |
|-------|-------|-----------|-------|--------|--------|
| `outcome_reward` | `0.1802` | `0.3243` | `+0.1441` | Observed | `20260415_gpu_eval/dd111_{sft,async23}_reward_trace.jsonl` |
| `rubric_reward` | `0.0332` | `0.0710` | `+0.0378` | Observed | 同上 |
| `task_unfinished` | `0.7297` | `0.6306` | `-0.0991` | Observed | 同上 |
| `termination_response_limit` | `0.5135` | `0.3784` | `-0.1351` | Observed | 同上 |
| `termination_rollout_timeout` | `0.0090` | `0.0000` | `-0.0090` | Observed | 同上 |
| pass samples | `20 / 111` | `36 / 111` | `+16` | Observed | 同上 |
| finished answers | `30 / 111` | `41 / 111` | `+11` | Observed | 同上 |

当前可对外表述：

- `Observed`: 这组 `dd111` 数字是当前最强、最干净的内部质量证据，应优先于 Stage 0 的 subset64 gate。
- `Observed`: 这组结果不仅仅是 reward 抖动；`outcome`、`unfinished`、`response_limit` 和 finished answers 都是同向改善。

### 6.5 External Gate: `BrowseComp subset256 64k`

评测口径：

- `Observed`: sampled recipe 固定为 `temperature=0.6`, `top_p=0.95`, `top_k=20`, `do_sample=true`, `max_response_length=61440`, `max_assistant_turns=120`, `max_tool_response_length=6000`
- `Observed`: 外部 gate 使用 matched relaxed budgets：`wall=600`, `real_wall=1200`, `max_tool_calls=160`, `max_search_calls=80`, `max_open_calls=60`, `max_find_calls=40`

| Metric | `SFT` | `async23` | Delta | Status | Source |
|-------|-------|-----------|-------|--------|--------|
| `outcome_mean` | `0.015625` | `0.03515625` | `+0.01953125` | Observed | `20260415_gpu_eval/bc256_{sft_relaxed_v2,async23_relaxed_v2}_reward_trace.jsonl` |
| judge-pass samples | `4 / 256` | `9 / 256` | `+5` | Observed | 同上 |
| finished answers | `11 / 256` | `17 / 256` | `+6` | Observed | 同上 |
| finished rate | `0.04297` | `0.06641` | `+0.02344` | Observed | 同上 |
| unfinished rate | `0.95703` | `0.93359` | `-0.02344` | Observed | 同上 |

当前可对外表述：

- `Observed`: `BrowseComp subset256 64k` 的真实 external gate 已到位，当前 source-of-truth 不再需要论文 placeholder。
- `Observed`: 这组外部结果必须写成 “matched sampled subset eval 的真实正向证据”，不能写成 full benchmark，更不能写成论文复现。
- `Observed`: `BrowseComp` 不走 CaRR rubric，因此这组 uplift 反映的是 final-answer correctness 与 completion behavior，而不是 rubric-score inflation。
- `Observed`: generation dump 的行为统计显示提升不是简单来自更高的总 tool volume；这个点适合放在面试展开里，而不是简历 headline。

### 6.6 Best Observed Async Training Window

| Metric | Value | Status | Source |
|-------|-------|--------|--------|
| Best observed internal training window | `step16` | Observed | `examples/carr_deepsearch/CaRR_log/formal_mainline_gs14_sp2_dynbsz_gmu05_b3_reasonfix_20260323_215746.launcher.log` |
| `outcome_reward/mean` | `0.59375` | Observed | 同上 |
| `rubric_reward/mean` | `0.09308` | Observed | 同上 |
| `task_unfinished/ratio` | `0.29167` | Observed | 同上 |
| `param_sync` | `1.69s` | Observed | 同上 |
| `per-global-step` healthy window | `~284s / gs` | Observed | 同上 |

当前可对外表述：

- `Observed`: `step16` 是最佳内部训练窗口，但它是训练期窗口，不是最终 benchmark checkpoint。
- `Observed`: 面试里可以把它当作“训练过程里最强的内部训练信号”，不能把它直接写成最终项目效果。

---

## 7. 外部 Benchmark 状态

### 7.1 当前状态表

| Eval target | Context | Status | 备注 |
|------------|---------|--------|------|
| `BrowseComp subset256` | `64k` | Observed | 已完成 `SFT vs async23` matched sampled eval |
| `BrowseComp full` | `64k` | Pending | 非当前简历闭环所必需 |
| `BrowseComp subset256` | `128k` | Pending | 可作为后续加分项，不是当前必需项 |
| `BrowseComp full` | `128k` | Pending | 成本最高，最后再考虑 |

### 7.2 当前简历使用规则

- recruiter-facing 草稿已经不需要任何 `Estimated BrowseComp` placeholder。
- 当前可以直接使用的外部数字只有 `BrowseComp subset256 64k` 的真实 observed 结果。
- 如果后续补了 `full` 或 `128k`，应该作为新证据追加，而不是把现有 `subset256` 结果模糊写成 `BrowseComp`。

---

## 8. 面试讲法

### 8.1 三句话版本

1. 这是一个 `agentic RL / post-training` 系统工程项目，不是单纯复现论文表格。
2. 我真正完成的工作是把 CaRR 的多轮工具调用、citation-aware reward 和 `C-GRPO` 接进 `verl`，并把 async RL 主线稳定到可训练、可 resume、可评测。
3. 当前真实证据已经包括内部 `DeepDive rl_val(111)` 的 `outcome 0.180 -> 0.324`，以及外部 `BrowseComp subset256 64k` 的 judge-pass `4/256 -> 9/256`；full/128k 仍是 pending。
4. recruiter-facing 简历应优先写“约 `37-45%` 的健康窗口迭代时间缩短、`dd111` 内部 uplift、`BrowseComp subset256` 外部正向证据”这类可理解指标，而不是 `global_step_23`、`step70` 这种内部 checkpoint 语义。

### 8.2 展开版

- 不要把项目讲成“我复现了 CaRR 并得到论文同等结果”。
- 要讲成：
  - 为什么这个任务属于典型 `agentic RL`
  - 你在 `verl` 里具体改了哪些层
  - 为什么 async 稳定化比再多刷几十 step 更有工程价值
  - 你如何判断一个 Thinking checkpoint 的 eval 是可信的
  - 为什么 “最终 durable checkpoint” 和 “最佳内部训练窗口” 不能混为一谈
  - 为什么简历里优先写 `~506s -> ~280-320s` 这类系统结果，而不是 `global_step_23`
- 面试官如果追问“为什么外部只给 subset256，不是 full”，就直接回答：
  - 当前 `BrowseComp subset256 64k` 的真实结果已经足够闭合简历故事，所以先把最小必要外部证据落地
  - `full 64k / 128k` 仍可继续补，但它们是降低方差与增强说服力，不是证明项目成立的唯一条件
  - 你在文档里已经明确区分了 `Observed subset256` 与 `Pending full/128k`

### 8.3 不要这样讲

- 不要说“我复现了论文结果”
- 不要说“async 一定比 sync 效果更好”
- 不要把 `BrowseComp subset256` 模糊写成 `BrowseComp`
- 不要把 raw step 数当作主卖点
- 不要把 `timeout=0%` 当成唯一 headline
- 不要把 healthy-window 数字写成 whole-run average
- 不要在正式简历里直接写 `global_step_23`、`step70` 这类只有项目内部才看得懂的 checkpoint 语义

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
| `examples/carr_deepsearch/docs/training_full_history_20260404.md` | sync/async 历史、budget 与配置变化 |
| `examples/carr_deepsearch/docs/eval_analysis_20260415.md` | 真实内部/外部评测汇总 |
| `examples/carr_deepsearch/docs/rl_debug_findings_20260312.md` | Thinking checkpoint 的 sampled eval 原因 |
| `examples/carr_deepsearch/docs/Carr_paper_data.md` | 论文结果与方法论参考 |
| `examples/carr_deepsearch/docs/sync_vs_async_grpo_update_mechanics.md` | sync vs async 更新机制对比 |

### 9.3 本地日志证据

| Path | 用途 |
|------|------|
| `examples/carr_deepsearch/CaRR_log/latest_checkpointed_iteration.txt` | durable checkpoint iteration 标记 |
| `examples/carr_deepsearch/CaRR_log/eval_gate_20260323_144500_sft.log` | `SFT` 的 `DeepDive subset64` sampled eval |
| `examples/carr_deepsearch/CaRR_log/step70_dd64_8gpu_20260322_153257.log` | `step70` 的 `DeepDive subset64` sampled eval |
| `examples/carr_deepsearch/CaRR_log/step90_dd64_8gpu_20260322_151340.log` | `step90` 的 `DeepDive subset64` sampled eval |
| `examples/carr_deepsearch/CaRR_formal_training_log/run-fqv0jkxr.wandb` | sync formal 观测配置的 artifact 证据（需 strings extract） |
| `examples/carr_deepsearch/CaRR_formal_training_log/output.log` | sync formal `step 71-96` 日志与 `timing_s/step` |
| `examples/carr_deepsearch/CaRR_log/formal_mainline_gs14_sp2_dynbsz_gmu05_b3_reasonfix_20260323_215746.launcher.log` | async 主线最终 run、`global_step_23` 与 queue/full、param sync 相关日志 |
| `examples/carr_deepsearch/CaRR_log/formal_mainline_gs11_sp2_dynbsz_gmu05_b3_reasonfix_20260323_091930.launcher.log` | async 主线 earlier tail 与 best-window 对照日志 |
| `examples/carr_deepsearch/CaRR_log/20260415_gpu_eval/dd111_sft_reward_trace.jsonl` | `DeepDive rl_val(111)` 的 `SFT` 内部 anchor |
| `examples/carr_deepsearch/CaRR_log/20260415_gpu_eval/dd111_async23_reward_trace.jsonl` | `DeepDive rl_val(111)` 的 `async23` 内部 anchor |
| `examples/carr_deepsearch/CaRR_log/20260415_gpu_eval/bc256_sft_relaxed_v2_reward_trace.jsonl` | `BrowseComp subset256 64k` 的 `SFT` 外部 gate |
| `examples/carr_deepsearch/CaRR_log/20260415_gpu_eval/bc256_async23_relaxed_v2_reward_trace.jsonl` | `BrowseComp subset256 64k` 的 `async23` 外部 gate |
| `examples/carr_deepsearch/CaRR_log/20260415_gpu_eval/eval_results/bc256_sft_relaxed_v2/0.jsonl` | `BrowseComp subset256 64k` 的 `SFT` 行为级 generation dump |
| `examples/carr_deepsearch/CaRR_log/20260415_gpu_eval/eval_results/bc256_async23_relaxed_v2/0.jsonl` | `BrowseComp subset256 64k` 的 `async23` 行为级 generation dump |

### 9.4 当前对外使用规则

- 正式简历默认只使用 `Safe` 或 `Evidence-Leaning`。
- 当前不再需要 `BrowseComp` placeholder；任何外部指标都应直接写成 `Observed subset256` 或 `Pending full/128k`。
- 任何涉及 `BrowseComp` uplift 的表述，必须明确写出 `subset256 64k`，直到 full benchmark 真正完成。
- 任何涉及速度改善的表述，必须写成 healthy-window systems result，并附带“不是 strict matched ablation”的 caveat。
