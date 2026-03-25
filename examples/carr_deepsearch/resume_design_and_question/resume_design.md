# 基于当前已有结果的简历口径设计

## 1. 先给结论

当前这条 CaRR / verl / C-GRPO 主线，**不适合再按“完整复现论文主结果”来写**，但**完全足够写成一个高质量简历项目**。

原因很简单：

- 你已经完成了 **真实多轮 deep-search RL pipeline** 的实现、联调、正式训练和两次 validation dump。
- 你有 **8 x RTX 6000** 上的正式训练日志，已经跑到 `step 96`。
- 你还明确观察到了一个很有价值的训练现象：
  `max_tool_response_length` 调小后，`termination_response_limit` 显著下降，但后期主瓶颈转移为 `termination_rollout_timeout`。

这类结果**不等于论文 benchmark 复现成功**，但它非常适合包装成：

- 一个“论文方法的工程化复现与诊断”项目
- 一个“agentic RL / deep search / reward shaping / training systems”项目
- 一个“在受限预算下完成有效实验、并定位关键 failure mode”的项目

因此，当前推荐策略是：

- **不要伪造论文绝对结果**
- **可以仿照论文的结果组织方式**
- **把最终项目口径改成：论文对齐实现 + 内部验证结果 + failure mode analysis**

---

## 2. 红线：哪些话不能写

下面这些说法不要写进简历、项目介绍或面试口头表述：

- “复现了论文主结果”
- “达到了论文中 4B-C-GRPO 的 BrowseComp / xbench / GAIA 数字”
- “在 benchmark 上显著优于 GRPO / SFT”
- “完整验证了 C-GRPO 在公开 benchmark 上的泛化能力”

原因：

- 当前你手里**最完整、最干净**的 artifact 是：
  - 两段正式训练日志
  - `global_step_40` 与 `global_step_80` 的 DeepDive RL val dump
- 当前并没有一套完整的
  `SFT / GRPO / C-GRPO`
  在同一固定 BrowseComp 子集上的最终对比结果
- 因此不能把“论文目标结果”写成“你的实测结果”

可以写的是：

- “论文对齐实现”
- “论文风格评估口径”
- “内部 validation / training dynamics 与 failure mode 分析”

---

## 3. 当前已有 artifact

当前本地归档目录：

- `examples/carr_deepsearch/CaRR_formal_training_log/files/output.log`
  - `step 21-70` 正式训练日志
- `examples/carr_deepsearch/CaRR_formal_training_log/output.log`
  - `step 71-96` 正式训练日志
- `examples/carr_deepsearch/CaRR_formal_training_log/40.jsonl`
  - `global_step_40` validation dump，`N=32`
- `examples/carr_deepsearch/CaRR_formal_training_log/80.jsonl`
  - `global_step_80` validation dump，`N=64`

这些 artifact 已经足够支持：

- 训练过程分析
- 调参前后对比
- validation 结果表
- case study
- 简历项目中的“结果与局限”部分

---

## 4. 论文对齐度：哪些一致，哪些不一致

### 4.1 一致项

- backbone：`Qwen3-4B-Thinking-2507`
- RL 核心方法：`C-GRPO`
- `temperature = 1.0`
- `lr = 2e-6`
- `alpha = 0.3`
- tools 仍然是 `search / open / find`
- reward 仍然是 outcome + rubric 的 CaRR 口径

### 4.2 关键不一致项

这些不一致项是你**效果不如论文**的主要解释来源，也应该写进项目“限制”部分：

- SFT 主配置不是论文的 `128k SFT`
- RL 正式 run 实际是 `b8 / n4`，不是论文数字关系对应的更大 rollout/group 规模
- `open` 返回内容被收紧到 `6000`，之后又调到 `5000`，论文环境是 `10k chars`
- 训练中加入了工程性 budget：
  `wall360 / search40 / open32 / find20 / tool88`
- tools 和 judge 都是**在线真实 API**
  - `Serper`
  - `Jina`
  - `DeepSeek judge`
- 因此训练目标不仅受策略本身影响，还会受外部 API 延迟长尾影响

### 4.3 这意味着什么

你现在最合理的口径不是：

- “我复现了论文 benchmark”

而是：

- “我在 paper-aligned-but-budget-constrained setup 下完成了 CaRR/C-GRPO 的工程实现、正式训练与定量分析”

---

## 5. 当前最可信的结果摘要

### 5.1 训练窗口统计

从归档日志统计得到：

#### `step 41-70`，调参前主线（`max_tool_response_length = 6000`）

- `outcome_reward/mean ≈ 0.276`
- `rubric_reward/mean ≈ 0.0556`
- `task_unfinished/ratio ≈ 0.5635`
- `termination_response_limit/ratio ≈ 0.3740`
- `termination_rollout_timeout/ratio ≈ 0.0656`
- `termination_search_budget/ratio ≈ 0.1021`
- `timing_s/step ≈ 494.6s`

#### `step 71-80`，调参后前 10 步（`max_tool_response_length = 5000`）

- `outcome_reward/mean ≈ 0.3187`
- `rubric_reward/mean ≈ 0.0503`
- `task_unfinished/ratio ≈ 0.4875`
- `termination_response_limit/ratio ≈ 0.2594`
- `termination_rollout_timeout/ratio ≈ 0.0844`
- `termination_search_budget/ratio ≈ 0.0938`
- `timing_s/step ≈ 508.7s`

#### `step 81-95`，调参后后续窗口

- `outcome_reward/mean ≈ 0.2667`
- `rubric_reward/mean ≈ 0.0456`
- `task_unfinished/ratio ≈ 0.5854`
- `termination_response_limit/ratio ≈ 0.2208`
- `termination_rollout_timeout/ratio ≈ 0.2333`
- `termination_search_budget/ratio ≈ 0.0896`
- `timing_s/step ≈ 512.8s`

### 5.2 这组数说明什么

它说明了一个很值得写进项目分析的现象：

- `5000` 这个调参**前半段有效**
  - `response_limit` 明显下降
  - `unfinished` 也短期下降
- 但后半段主瓶颈转移到 `rollout_timeout`
- 这意味着问题不再是“response 太长”
- 而是“tool-side 长尾 + wall budget”开始主导训练信号

这不是一个坏结果，相反它是一个**非常好的项目分析点**。

---

## 6. 当前最可信的 validation 结果

### 6.1 `global_step_40` validation，`N=32`

来自 `40.jsonl`：

- `score / outcome_reward = 0.3438`
- `rubric_reward = 0.0570`
- `task_unfinished = 0.5938`
- `termination_response_limit = 0.4062`
- `termination_rollout_timeout = 0.0`
- `termination_search_budget = 0.1250`
- `tool_call_counts = 47.8125`
- `response_length = 52539.34`

### 6.2 `global_step_80` validation，`N=64`

来自 `80.jsonl`：

- `score / outcome_reward = 0.2656`
- `rubric_reward = 0.0533`
- `task_unfinished = 0.6719`
- `termination_response_limit = 0.3438`
- `termination_rollout_timeout = 0.0469`
- `termination_search_budget = 0.1406`
- `tool_call_counts = 48.0625`
- `response_length = 49105.23`

### 6.3 如何解释这两次 validation

这两次结果不能直接写成“训练越来越差”，原因有两个：

- `40` 的样本数是 `32`
- `80` 的样本数是 `64`

它们不是严格等口径对比。

但它们足以支持下面这个说法：

- `5000` 调参并没有在 validation 上给出明确、稳定的收益证据
- 至少到 `step 80` 为止，这条主线还没有形成“值得继续高成本长跑”的强信号

---

## 7. 当前最适合写进简历的项目结论

### 7.1 最推荐的项目标题

可选标题：

- `基于 verl 的 Deep Search Agent 强化学习复现与诊断（CaRR / C-GRPO）`
- `面向多跳问答的 Agentic RL 项目：CaRR 奖励桥接、训练联调与 failure mode 分析`
- `Paper-aligned Deep Search RL Pipeline on verl: C-GRPO Training, Tool-Use Diagnostics, and Reward Analysis`

### 7.2 最推荐的简历主叙事

核心不是“我把论文 benchmark 跑出来了”，而是：

- 我把论文方法在 verl 上工程化实现
- 我把多服务 deep-search RL 真的跑起来了
- 我在正式训练中观察到了 reward shaping 带来的行为变化
- 我识别出了后期训练收益变低的关键系统性原因

### 7.3 可以直接用的中文简历 bullet

版本 A，更偏工程：

- 基于 `verl` 复现并工程化实现 CaRR / `C-GRPO` deep-search agent 训练链路，打通 `search-open-find` 多工具 agent loop、reward bridge、custom advantage estimator 与 WandB/JSONL 可复现实验日志体系。
- 在 `8 x RTX 6000` 上完成 `Qwen3-4B-Thinking-2507` 的正式多轮 RL 训练至 `step 96`，支持 citation-aware rubric reward、在线 DeepSeek judge 与多服务并发工具调用。
- 通过调节 `max_tool_response_length`，将中期训练窗口的 `termination_response_limit` 从 `37.4%` 降至 `25.9%`，并把 `task_unfinished` 从 `56.4%` 降至 `48.8%`，随后进一步定位出 `rollout_timeout` 接管训练收益的长尾 failure mode。

版本 B，更偏研究：

- 复现 CaRR 论文中的 citation-aware rubric reward 与 `C-GRPO` 训练范式，在 `verl` 上实现面向多跳问答的 deep-search agent 强化学习系统。
- 基于正式训练日志与 `global_step_40 / global_step_80` validation dump，对 `response_limit -> rollout_timeout` 的瓶颈迁移进行了定量分析，识别出 online tool latency 与 wall budget 对 RL 信号的污染路径。
- 构建 paper-aligned-but-budget-constrained 的实验报告口径，在不伪造 benchmark 结果的前提下，完成训练动态、工具行为与任务完成质量的系统性对照分析。

### 7.4 可以直接用的英文简历 bullet

- Implemented a paper-aligned CaRR / C-GRPO deep-search RL pipeline on `verl`, including a custom multi-turn agent loop, citation-aware reward bridge, and reproducible logging/eval dumps.
- Ran formal RL training for `Qwen3-4B-Thinking-2507` on `8 x RTX 6000` up to `step 96`, integrating live `search/open/find` tools, an online DeepSeek judge, and multi-service orchestration.
- Diagnosed a key training failure-mode shift: tuning `max_tool_response_length` reduced `response-limit` terminations (`37.4% -> 25.9%`) and improved unfinished ratio (`56.4% -> 48.8%`) in the early post-tuning window, but later exposed `rollout-timeout` as the dominant bottleneck.

---

## 8. 推荐的“论文式结果组织方式”

你可以**仿照论文写法组织结果**，但不要把论文数字写成你的数字。

推荐采用两张表：

### 表 1：论文目标口径 vs 本项目实际可证据化口径

列建议：

- `维度`
- `论文口径`
- `本项目实际口径`
- `影响`

建议写法：

- backbone：一致
- reward framework：一致
- RL context：接近
- rollout/group size：不一致
- SFT context：不一致
- tool truncation：不一致
- external latency：本项目更高
- benchmark eval：论文完整，本项目未完整跑完

### 表 2：本项目实际结果表

只放你真的测过的结果：

- `窗口 / checkpoint`
- `outcome_reward`
- `rubric_reward`
- `task_unfinished`
- `termination_response_limit`
- `termination_rollout_timeout`
- `备注`

建议放：

- `step 41-70`
- `step 71-80`
- `step 81-95`
- `val@40 (N=32)`
- `val@80 (N=64)`

这样写出来会非常像论文风格，但不会越界。

---

## 9. 最终项目叙事建议

### 9.1 你应该主打什么

主打：

- `agentic RL 工程实现`
- `论文方法落地`
- `训练系统与外部工具联调`
- `reward shaping / failure mode diagnosis`

不要主打：

- “我复现了 paper benchmark”
- “我拿到了论文同级别 SOTA 数字”

### 9.2 最合适的一句话项目总结

推荐总结：

> 在 verl 上实现并运行了论文风格的 CaRR / C-GRPO deep-search RL 训练系统，完成多轮工具调用、citation-aware reward 和正式训练日志分析，并定位出从 `response_limit` 向 `rollout_timeout` 迁移的核心失败模式。

---

## 10. 是否还值得继续烧 GPU

从“简历 ROI”视角，我的建议是：

- **不值得按当前配置继续原样训练**

原因：

- 当前已经有足够写简历的技术含量和实验材料
- 后续继续烧卡的边际收益不高
- 你现在最缺的不是“再多几十步”，而是“把已有结果组织成可信、好讲、可答辩的材料”

只有在下面这种情况下，才值得再投入一次短时 GPU：

- 你想补一个非常小的、低成本的 targeted ablation
- 例如：
  - 只改 tool timeout / retry，不再改 reward 和大配置
  - 只跑到 `20-40 steps`
  - 只为了验证 `rollout_timeout` 是否可显著压回去

如果不是这种“低成本验证单一假设”的实验，就不建议再追。

---

## 11. 最终建议的交付物

为了尽快投递简历，建议最终只做下面四个产物：

1. 一页项目结果总结
   - 问题定义
   - 论文方法
   - 你的实现
   - 你的真实结果
   - 关键 failure mode

2. 一张结果表
   - `41-70`
   - `71-80`
   - `81-95`
   - `val@40`
   - `val@80`

3. 一张训练动态图
   - `outcome_reward`
   - `task_unfinished`
   - `termination_response_limit`
   - `termination_rollout_timeout`

4. 两到三个 case study
   - 一个中期成功样本
   - 一个后期 timeout 样本
   - 一个对比样本（response-limit vs rollout-timeout）

---

## 12. 最后的口径建议

最稳妥、最好用的一句话是：

> 这是一个**论文对齐实现 + 正式训练 + 系统性诊断**项目，而不是“完整 benchmark 复现”项目。

这个口径既真实，也足够强。
