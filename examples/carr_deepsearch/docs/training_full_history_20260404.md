# CaRR DeepSearch 完整训练历史 (2026-04-04)

本文档还原训练全流程的配置变化、budget 命中比例和与论文的差异分析。

---

## 1. 训练时间线概览

```
2026-03-04  初始代码实现（carr_grpo.yaml: b128/n16 原始配置）
2026-03-11  批量修正（b128/n16 → b16/n8，后续实际跑的是 b8/n4）
2026-03-12  项目复盘文档（project_retrospective）
2026-03-15  开始在 8xH200 上做正式训练前的 gate probe
2026-03-18  正式同步 RL 开始（wandb 记录起始时间）
            steps 1-20: 在 8xH200 上完成（日志不在本地，checkpoint 保存到 global_step_20）
            steps 21-70: 从 global_step_20 resume，日志在 files/output.log
            step 70: save checkpoint（用于后续 eval 和 async 起点）
            steps 71-96: 从 global_step_70 resume，日志在 output.log
            step 91+: 急剧恶化（timeout 飙升）
2026-03-22  Stage 0 eval（SFT/step70/step90 DeepDive subset64）
            async RL 基座验证 probe
2026-03-23  正式 async RL（从 step70 warm-start）
            23 param versions = 92 global steps
```

---

## 1.5 SFT 训练详情

### 1.5.1 SFT 配置

| 参数 | 实际值 | 论文值 | 差异 |
|------|--------|--------|------|
| `max_length` | **65536 (64k)** | **128k** | **论文的 1/2，最大差异** |
| 训练数据 | 791 条（train）+ 41 条（val） | 832 条（reject sampling） | 基本一致，同源 DeepDive |
| epochs | 3 | 3 | 一致 |
| learning_rate | 4e-5 | 4e-5 | 一致 |
| batch_size | 4 (global) | 16 | 论文的 1/4 |
| model_dtype | bf16 | 未说明 | — |
| backbone | Qwen3-4B-Thinking | 同 | 一致 |
| truncation | `loss_window`（滑动窗口） | 未说明（可能不截断，因为 128k 足够） | 关键差异 |
| enable_thinking | true | 未说明 | Qwen3-Thinking 模型需要 |

### 1.5.2 样本截断情况

- 总 SFT 训练样本：791 条
- 需要截断的样本：**~175 条（~22%）**
- 最长样本未截断长度：~68k tokens（超过 64k 上限）
- 截断方式：`loss_window` — 滑动窗口选取 supervised token (loss=1) 最多的 64k 窗口，平局时偏向保留末尾（final answer）

### 1.5.3 SFT 训练中发现并修复的关键 bug

**Bug 1：Per-message tokenization 丢失 reasoning content（严重，已修复）**

Qwen3 的 chat template 要求完整对话上下文才能正确渲染 `reasoning_content` 为 `<think>` blocks。早期实现逐条 message 独立 tokenize，导致 SFT 模型只学会了工具调用语法，没有学到任何 reasoning。

- 表现：早期 SFT checkpoint 在 eval 时零 reasoning 输出
- 修复：改为 prefix-based rendering（`_process_messages_with_context()`）
- Commit：`38ea7f59` (2026-03-12)

**Bug 2：Right truncation 截断 final answer（中等，已修复）**

初始 `truncation: right` 会从序列末尾截断，恰好切掉 `## Exact Answer` 部分。

- 示例：Sample 0 的 final answer turn (997 tokens) 被截掉 450 tokens
- 修复：改为 `loss_window` 截断策略
- Commit：`38ea7f59` (2026-03-12)

**Bug 3：RL 配置缺少 enable_thinking（重要，已修复）**

`carr_grpo.yaml` 最初没有 `apply_chat_template_kwargs.enable_thinking: true`，导致 RL rollout 的 prompt 不含 thinking mode。

- 修复：添加 `enable_thinking: true` 到 RL 配置
- Commit：`38ea7f59` (2026-03-12)

### 1.5.4 64k SFT 对 BrowseComp 评测的影响分析

论文用 128k 做 SFT，意味着所有 832 条 trace 完整保留、无截断。你用 64k，22% 样本被截断。

影响链：
1. 模型在 SFT 阶段没有见过 >64k 的完整搜索轨迹 → 不知道如何高效管理长轨迹
2. 被截断的样本丢失了部分 reasoning 和 final answer → loss_window 尽量保留但仍有信息损失
3. BrowseComp 的难题通常需要更长搜索轨迹 → 64k SFT 模型在这些题上能力不足
4. 论文 SFT 在 BrowseComp 64k 报告 7.7%，128k 报告 14.1% → 长上下文能力对 BrowseComp 极为重要

预期：你的 SFT 在 BrowseComp 64k 上会**低于论文的 7.7%**（因为 SFT 训练长度只有 64k 而非 128k，加上 bs=4 vs 16 的差异）。

---

## 2. 同步 RL 实际运行配置

### 2.1 Step 1-70 配置

| 参数 | 实际值 | 论文值 | 差异 |
|------|--------|--------|------|
| `train_batch_size` | 8 | 16 | 论文的 1/2 |
| `rollout.n` | 4 | 8 | 论文的 1/2 |
| `ppo_mini_batch_size` | 4 | — | — |
| `max_response_length` | 61440 (64k) | 64k | 一致 |
| `max_assistant_turns` | **120** | 无限制（论文未设） | 额外限制 |
| `max_tool_response_length` | **6000** | **10000** | 论文返回前 10k chars |
| `max_rollout_wall_time_s` | **360** | **无** | 论文没有 timeout |
| `max_tool_calls` | **88** | 无明确限制（论文未设） | 额外限制 |
| `max_search_calls` | **40** | 无明确限制 | 额外限制 |
| `max_open_calls` | **32** | 无明确限制 | 额外限制 |
| `max_find_calls` | **20** | 无明确限制 | 额外限制 |
| `temperature` | 1.0 | 1.0 | 一致 |
| `learning_rate` | 2e-6 | 2e-6 | 一致 |
| `gpu_memory_utilization` | 0.5 | — | SGLang 参数 |

### 2.2 Step 71-96 配置变更

**在 step 70 到 step 71 之间改了 `max_tool_response_length` 从 6000 → 5000**。其余配置不变。

这个改动的效果在训练指标中可以清楚看到：

| 区间 | outcome | unfinished | response_limit | rollout_timeout | hit_budget |
|------|---------|------------|---------------|----------------|-----------|
| step 21-70 (tool_resp=6000) | 0.266 | 0.572 | **0.373** | 0.069 | 0.199 |
| step 71-90 (tool_resp=5000) | **0.320** | **0.497** | **0.266** | 0.084 | 0.231 |
| step 91-96 (tool_resp=5000) | 0.167 | 0.734 | 0.115 | **0.521** | 0.620 |

**解读**：
- 降低 tool_response_length 从 6000→5000 后，response_limit 命中率从 37.3% 降到 26.6%（因为每次工具返回更短，总 token 更省）
- 训练指标在 step 71-90 确实改善了（outcome 0.266→0.320）
- 但主瓶颈被"转移"了：response_limit 减少后，rollout_timeout 成为新瓶颈
- Step 91+ timeout 飙升到 52%，模型的搜索时间更长但 wall=360s 不允许

### 2.3 Step 40 和 Step 80 的内部验证

训练过程中做了两次内部验证（与后来 Stage 0 eval 的口径不同）：

| 验证 | n | Outcome | Rubric | Unfinished | Response Limit | Timeout | Hit Budget |
|------|---|---------|--------|------------|---------------|---------|-----------|
| step 40 eval | 32 样本 | **0.3438** | 0.0570 | 0.5938 | 0.4063 | 0.0000 | 0.1875 |
| step 80 eval | 64 样本 | 0.2656 | 0.0533 | **0.6719** | 0.3438 | **0.0469** | **0.3281** |

注意：step 40 eval 只用了 32 样本，step 80 eval 用了 64 样本，**口径不完全一致**。但 step 80 的 outcome 退化和 budget 命中上升是清晰的。

### 2.4 关于 "step90 是否可能是更好的 checkpoint" 的分析

Stage 0 eval（SFT/step70/step90，统一 64 样本 sampled eval）的 timeout 全部为 **0.0**。所以 eval 时 wall=360s 并没有造成 timeout 截断。

step90 在 eval 中 outcome 更差（0.266 vs step70 的 0.328）不是因为 eval 时 timeout，而是因为：
1. Step 71-90 虽然训练指标改善，但 tool_response_length 从 6000→5000 改变了模型行为
2. Step 91+ 的 timeout 恶化说明模型在 step 80-90 已经开始学习更长搜索策略，但 wall=360s 无法承载
3. **Step 90 checkpoint 的模型已经适应了 5000 的 tool response + 360s wall，形成了一种"被约束的搜索策略"**
4. 在 eval 时虽然没有 timeout（因为 eval 的样本分布不同），但模型的行为模式可能已经不如 step 70 时的"未被 timeout 扭曲过的策略"

**总结**：step90 不太可能是更好的 checkpoint。虽然 step 71-90 的训练指标看起来更好，但那是因为 tool_response=5000 压低了 response_limit 命中率造成的"表面改善"。验证结果（step 80 eval）和 Stage 0 eval（step 90）都确认了 step 70 后的训练并没有在验证集上带来真实提升。

### 2.5 Stage 0 eval 和异步 RL 阶段的 max_tool_response_length

关键事实：**Stage 0 eval 和异步 RL 都把 `max_tool_response_length` 恢复到了 6000**，没有沿用 step 71-96 训练时的 5000。

日志验证：

| 阶段 | max_tool_response_length | 来源 |
|------|------------------------|------|
| 同步 step 1-70 训练 | **6000** | wandb 配置 |
| 同步 step 71-96 训练 | **5000** | codex 回复确认 |
| Stage 0 eval（SFT/step70/step90）| **6000** | eval log 中 `'max_tool_response_length': 6000` |
| 异步 RL 全程 | **6000** | async launcher log 中 `'max_tool_response_length': 6000` |

这意味着：
- Step70 checkpoint 在训练时（step 1-70）和 eval 时都用的是 6000 — **完全一致**
- Step90 checkpoint 在训练时（step 71-90）用的是 5000，但 eval 时用的是 6000 — **不一致**
- 这个不一致理论上应该对 step90 有利（eval 时工具返回更多内容），但 step90 的 eval 结果反而更差
- 这进一步证实 step 71-90 的训练（tool_resp=5000 + wall=360s）没有真正改善模型能力

异步 RL 从 step70 warm-start 后恢复到 6000，与 step70 训练时的配置一致，避免了 train-eval 口径不匹配的问题。

### 2.6 max_tool_response_length 变更的完整影响链

```
step 1-70: tool_resp=6000
  → 模型学到的搜索策略适配 6000 的工具返回长度
  → step70 checkpoint 的行为在 6000 下是自然的

step 71-90: tool_resp=5000（降低了）
  → 每次工具返回更短 → 总 token 更省 → response_limit 命中率下降
  → 训练指标表面改善，但模型行为被扭曲：
    - 模型可能学到了更频繁调用工具来弥补每次返回更短的信息
    - 这导致搜索时间变长 → 逐步逼近 wall=360s
  → step 91+ timeout 飙升，训练崩溃

Stage 0 eval: tool_resp=6000（恢复）
  → step70 在 6000 下评测：行为与训练一致，表现正常
  → step90 在 6000 下评测：行为与训练不一致（训练时 5000，评测时 6000）
    但结果更差，说明 5000 训练没有带来泛化能力提升

异步 RL: tool_resp=6000（恢复）
  → 从 step70 warm-start，配置与 step70 训练时一致
  → 加上 gmu=0.5 消除 timeout + n=8 恢复 GRPO 信号
  → 训练动态正向（tool_calls 增长，搜索深度加大）
```

**关键差异总结**：论文没有设置任何 timeout 或工具调用次数上限。论文唯一的截断条件是超过 64k token 上下文限制。你的实现额外增加了 6 种 budget 限制（wall_time、tool_calls、search、open、find、assistant_turns），这些是为了防止同步训练时长尾 rollout 阻塞 GPU。step 70→71 时还额外将 tool_response_length 从 6000 降到 5000，但 Stage 0 eval 和异步 RL 都恢复到了 6000。

---

## 3. 同步 RL 各 step 的 budget 命中比例

### 3.1 Step 21-70 汇总统计（旧段 wandb log）

| 指标 | 均值 | 最小 | 最大 | 说明 |
|------|------|------|------|------|
| outcome_reward/mean | 0.266 | 0.031 (step 47) | 0.688 (step 62) | 高方差 |
| task_unfinished/ratio | 0.571 | 0.219 (step 62) | 0.875 (step 32/66) | 超一半样本未完成 |
| termination_rollout_timeout | 0.071 | 0.000 | 0.188 (step 76) | timeout 还不严重 |
| termination_response_limit | 0.365 | 0.094 (step 62) | 0.625 (step 29) | **主要截断原因** |
| termination_search_budget | 0.098 | 0.000 | 0.313 (step 25/82) | 搜索 budget 命中 |
| termination_find_budget | 0.025 | 0.000 | 0.188 (step 75) | find 偶发 |
| termination_open_budget | 0.010 | 0.000 | 0.094 (step 57/88) | open 很少命中 |
| termination_tool_call_budget | 0.000 | 0.000 | 0.000 | 从未命中 |
| hit_limit/ratio | 0.365 | 0.094 | 0.625 | ≈ response_limit |
| hit_budget/ratio | 0.196 | 0.031 | 0.375 | budget 类截断 |
| tool_call_counts/mean | 45.0 | 38.6 (step 37) | 51.8 (step 66) | 稳定 |
| response_length/mean | 50.0k | 44.3k | 56.9k | 接近 64k 上限 |
| timing_s/step | 493s | 451s | 546s | 稳定 |

### 3.2 Step 71-90 详细数据（最佳训练窗口）

| Step | Outcome | Unfinished | Timeout | Response Limit | Search Budget | Find Budget | Open Budget |
|------|---------|------------|---------|---------------|--------------|-------------|-------------|
| 71 | 0.250 | 0.563 | 0.063 | 0.406 | 0.063 | 0.031 | 0.000 |
| 72 | **0.469** | 0.406 | 0.000 | 0.156 | 0.219 | 0.031 | 0.000 |
| 73 | 0.063 | 0.656 | 0.094 | 0.281 | 0.188 | 0.094 | 0.000 |
| 74 | 0.344 | 0.438 | 0.156 | 0.188 | 0.031 | 0.031 | 0.031 |
| 75 | 0.344 | 0.531 | 0.031 | 0.281 | 0.031 | 0.188 | 0.000 |
| 76 | 0.344 | 0.500 | 0.188 | 0.281 | 0.000 | 0.000 | 0.031 |
| 77 | 0.281 | 0.469 | 0.125 | 0.250 | 0.094 | 0.000 | 0.000 |
| 78 | 0.344 | 0.469 | 0.000 | 0.313 | 0.125 | 0.031 | 0.000 |
| 79 | 0.281 | 0.438 | 0.125 | 0.188 | 0.094 | 0.031 | 0.000 |
| 80 | **0.469** | 0.406 | 0.063 | 0.250 | 0.094 | 0.000 | 0.000 |
| 81 | 0.406 | 0.469 | 0.031 | 0.375 | 0.031 | 0.031 | 0.000 |
| 82 | 0.188 | 0.563 | 0.094 | 0.281 | 0.188 | 0.000 | 0.000 |
| 83 | 0.313 | 0.594 | 0.031 | 0.313 | 0.156 | 0.094 | 0.000 |
| 84 | **0.469** | 0.406 | 0.000 | 0.188 | 0.125 | 0.063 | 0.031 |
| 85 | 0.219 | 0.469 | 0.063 | 0.375 | 0.031 | 0.000 | 0.000 |
| 86 | 0.344 | 0.563 | 0.156 | 0.313 | 0.063 | 0.000 | 0.031 |
| 87 | 0.281 | 0.500 | 0.156 | 0.188 | 0.094 | 0.063 | 0.000 |
| 88 | 0.281 | 0.594 | 0.094 | 0.281 | 0.156 | 0.000 | 0.063 |
| 89 | **0.438** | 0.500 | 0.156 | 0.156 | 0.094 | 0.094 | 0.000 |
| 90 | 0.281 | 0.406 | 0.063 | 0.250 | 0.094 | 0.000 | 0.000 |

**Step 71-90 平均**：outcome 0.330, unfinished 0.490, timeout 0.084, response_limit 0.261, search_budget 0.098

### 3.3 Step 91-96 详细数据（恶化阶段）

| Step | Outcome | Unfinished | **Timeout** | Response Limit | Search Budget | Find Budget |
|------|---------|------------|-------------|---------------|--------------|-------------|
| 91 | 0.281 | 0.563 | 0.094 | 0.344 | 0.000 | 0.125 |
| 92 | 0.188 | 0.719 | **0.438** | 0.156 | 0.094 | 0.031 |
| 93 | 0.031 | **0.906** | **0.906** | 0.000 | 0.000 | 0.000 |
| 94 | 0.156 | 0.813 | **0.719** | 0.000 | 0.094 | 0.000 |
| 95 | 0.125 | 0.719 | **0.500** | 0.094 | 0.125 | 0.000 |
| 96 | 0.219 | 0.688 | **0.469** | 0.094 | 0.063 | 0.063 |

**关键观察**：
- Step 91 还相对正常（timeout 9.4%）
- Step 92 开始 timeout 突然飙升到 43.8%
- Step 93 达到 90.6% timeout（几乎全部样本超时）
- Step 93 的 outcome 跌到 0.031（接近零）
- 恶化不是渐进的，而是突然发生的（step 91→92 跳变）
- 可能原因：模型学到了更长的搜索策略但 wall=360s 来不及完成

---

## 4. 与论文的 Budget 差异分析

### 4.1 论文的截断行为

论文只提到一种截断：**超过 64k token 上下文限制**。论文原文（Section 2.3）：

> "rollouts with format error or overlength problem (i.e., exceeding token or tool-call limits) are assigned a reward of 0"

注意论文说了 "tool-call limits"，但**没有给出具体数值**。论文的 tool-call limit 不明确是否指 88 次还是其他数字。

### 4.2 你的额外 budget 限制如何影响训练

你的实现增加了 6 种 budget，其中实际命中的主要是 3 种：

| Budget | 同步 step 21-90 平均命中率 | 对 GRPO 信号的影响 |
|--------|--------------------------|-------------------|
| **response_limit** (64k token) | **~33%** | 与论文一致，这是论文也有的截断 |
| **rollout_timeout** (360s) | **~8%**（step 71-90），**>50%**（step 92+） | **论文没有**，你独有的截断 |
| **search_budget** (40 calls) | **~10%** | 论文可能有类似限制但未明确 |
| find_budget (20 calls) | ~3% | 影响较小 |
| open_budget (32 calls) | ~1% | 几乎不影响 |
| tool_call_budget (88 calls) | 0% | 从未命中 |

### 4.3 影响分析

**response_limit (~33%)**：这是与论文一致的截断条件。论文也是 64k context。这部分不会导致与论文的训练行为差异。

**rollout_timeout (~8% 正常态，>50% 恶化态)**：**这是你与论文最大的差异**。
- 论文没有 timeout，模型可以无限搜索直到 64k token 用完
- 你的 360s timeout 在模型学到更长搜索策略后成为瓶颈
- 正常态下（step 21-90）8% 的 timeout 率意味着每 batch 32 条中约 2-3 条因 timeout 被截断（reward=0）
- 恶化态下（step 92+）>50% 的 timeout 意味着 GRPO 的组内对比几乎全是 0 vs 0
- **这不只是"信号变少"，而是系统性地惩罚了"搜索更深"的行为**——模型需要更多时间搜索，但 timeout 不允许

**search_budget (~10%)**：
- 40 次搜索的上限在论文中没有明确提及
- 10% 的命中率意味着少部分样本因搜索次数不够而被截断
- 影响方向：轻微限制了模型学习更广泛搜索策略的能力

### 4.4 对 GRPO 信号的具体影响

以 step 71-90 的平均数据为例（n=4 per group）：

```
平均 unfinished = 49% → 每 group 约 2 条 unfinished (reward=0)
平均 outcome = 33% → 每 group 约 1.3 条 correct (reward>0)
→ 典型 group: [0, 0, 0.3, 1.0]
→ mean=0.325, std=0.47
→ advantage of correct: +1.44
→ advantage of unfinished: -0.69
```

**问题**：unfinished 的 2 条中，可能 1 条是 timeout（模型在认真搜索但时间不够），1 条是 response_limit（真正的 overlength）。GRPO 无法区分这两种失败原因——它们都获得相同的负 advantage。

而论文没有 timeout，所以论文的 unfinished 主要是 overlength（64k token 用完），这是一个更"干净"的截断信号。

---

## 5. 异步 RL 阶段配置变化

### 5.1 配置迭代时间线

| 时间 | Param Versions | 关键变更 | 效果 |
|------|---------------|---------|------|
| Step 1 | v1 | 4:4 split, conc=12, n=8, sync=2, **gmu=0.3** | SGLang 过载，100% timeout |
| Step 1' | v1 | **conc 降到 4** | 首次有训练信号（outcome 0.44） |
| Step 2 | v2 | 同上 | param_sync 长尾暴露（248s） |
| Step 3 | v3 | **sync=2→4** | sync=4 绕过 self-pause |
| Step 4 | v4 | **2:6 split + SP=2**, wall=480, b=4 | 信号好但 SP=2 通信慢 |
| Step 5-6 | v5-6 | + queue_full fix | param_sync 降到 1.7s |
| Step 7-8 | v7-8 | + **gmu=0.3→0.5** | **timeout 从 59%→0%**（关键突破） |
| Step 9 onwards | v9-23 | + **b=4→3** | 加速 + 信号最佳窗口 |
| Step 12 | v12 | + 指标修复（unfinished 子原因） | 诊断更精确 |
| Step 15 | v15 | flush_cache 崩溃 resume | 首步冷启动效应 |

**gmu 变化的解释**：
- 异步 probe 初始推荐口径是 `gpu_memory_utilization=0.3`（codex 聊天记录确认）
- Step 1-6 用 gmu=0.3，timeout 始终很高（16-72%），SGLang 的 KV cache 空间不够导致长序列被拒绝
- Step 7 改为 gmu=0.5 后，KV cache 空间翻倍，timeout 立刻降到 0%
- 注意：同步 RL（step 1-96）的 wandb 配置显示也是 gmu=0.5
- gmu 0.3→0.5 的变化只发生在异步阶段的 probe 期间（step 1-7），是异步 infra 调优的核心突破之一

### 5.2 异步阶段的 budget 配置

| 参数 | 异步值 | 同步值 | 变化 |
|------|--------|--------|------|
| `max_rollout_wall_time_s` | **480** | 360 | 放宽 33% |
| `max_real_rollout_wall_time_s` | **960** | — | 新增（含 pause 时间） |
| `max_tool_calls` | 88 | 88 | 不变 |
| `max_search_calls` | 40 | 40 | 不变 |
| `max_open_calls` | 32 | 32 | 不变 |
| `max_find_calls` | 20 | 20 | 不变 |
| `max_assistant_turns` | 120 | 120 | 不变 |
| `rollout.n` | **8** | 4 | 恢复到论文值 |
| `gpu_memory_utilization` | **0.5** | 0.5 | 不变但效果不同（异步下消除 timeout） |

### 5.3 异步阶段 budget 命中比例（step 9-23 稳定主线）

从 handoff 文档中的 step metrics：

| 指标 | 均值(step 9-23) | 与同步对比 |
|------|-----------------|----------|
| timeout | **0-8%**（gmu=0.5 后） | 同步 71-90 约 8%，91+ 飙到 50%+ |
| unfinished_limit (response_limit) | ~27-37% | 与同步相当 |
| unfinished_budget (search/find/open) | ~12-25% | 略低于同步 |
| task_unfinished 总计 | ~29-68% | 同步 40-91% |

**关键改善**：异步阶段通过 gmu=0.5 彻底消除了 timeout 截断，这是最大的 budget 差异。模型可以在 480s 内充分搜索，不再被 360s 截断惩罚。

---

## 6. 这些 Budget 差异对训练效果的影响预期

### 6.1 已确认的影响

1. **Timeout 截断是同步训练后期恶化的直接原因**：step 92+ 的 timeout 飙升导致 GRPO 信号退化到噪声水平
2. **消除 timeout 后训练信号立即改善**：异步阶段 gmu=0.5 后，outcome 从同步后期的 0.03-0.22 回升到 0.13-0.59
3. **n=4→8 恢复了 GRPO 组内对比的统计量**：从每 group 4 条恢复到 8 条，组内对比更稳定

### 6.2 预期但未验证的影响

1. **max_tool_response_length 6000 vs 论文 10000**：你的工具返回截断更多内容，模型获得的信息更少。这可能导致模型需要更多 open/find 调用来获取完整信息。但从数据看 open_budget 命中率很低（<2%），所以影响可能有限。

2. **search_budget 40 calls**：约 10% 的命中率说明确实限制了部分样本的搜索深度。论文没有明确的搜索次数限制。如果放宽到 60-80，可能允许模型探索更深层的搜索策略。

3. **max_assistant_turns 120**：论文没有 turn 限制。但从数据看 assistant_turn_limit 命中率为 0%（因为 120 turns 下通常先撞 64k token 或 timeout），所以实际无影响。

### 6.3 对 eval 预期的影响

做 BrowseComp eval 时，eval 也会使用 budget 限制。如果 eval 的 budget 和论文不一致（比如 wall=360s），可能人为压低了模型的表现。

gpu_eval_recommendations 中的 BrowseComp eval 使用了 `CARR_ROLLOUT_WALL_TIME_S=360`，这和同步训练一致但和论文不一致。**考虑在 eval 时适当放宽 wall_time（如 600s）以更公平地评估模型能力**。

---

## 7. 配置变更的起因

### 7.1 为什么从 b16/n8 降到 b8/n4

- 初始代码 (2026-03-04) 设置 b128/n16（明显错误，128 是 global batch 不是 mini batch）
- 修正后 (2026-03-11) 改为 b16/n8，更接近论文
- 但在 8xH200 上实际跑 b16/n8 时发现不经济：rollout 队列深度太大、外部工具长尾被放大、同步栅栏等待时间太长
- 最终降到 b8/n4 作为第一个稳定可跑的配置
- 项目复盘文档详细分析了为什么 b16/n8 在当前 infra 下不经济

### 7.2 为什么增加 timeout 和 budget 限制

- 同步训练中，所有 GPU 等待最慢的 rollout 完成
- 没有 timeout 时，单个复杂样本可能需要 10+ 分钟，导致整个 batch 的 step 时间失控
- 360s timeout 是在稳定性和信号质量之间的折中
- 工具调用 budget (88/40/32/20) 是防止模型陷入无效循环

### 7.3 为什么异步阶段能恢复 n=8

- async 解耦了 rollout 和 training，长尾不再阻塞 GPU
- 同时 gmu=0.5 通过限制 SGLang 的 GPU 显存使用率避免了 OOM 导致的 timeout
- 因此可以安全恢复到论文的 n=8，且 timeout 降到 0%

---

## 数据来源

| 数据 | 文件路径 |
|------|---------|
| Step 21-70 metrics | `CaRR_formal_training_log/files/output.log` |
| Step 71-96 metrics | `CaRR_formal_training_log/output.log` |
| wandb 实际运行配置 | `CaRR_formal_training_log/run-fqv0jkxr.wandb` (二进制提取) |
| Async step 1-23 metrics | `CaRR_log/formal_mainline_gs*.launcher.log` |
| 配置变更历史 | git history + `DEVELOPMENT_LOG.md` + `project_retrospective_20260312.md` |
| 论文配置参考 | `CaRR/2601.06021v1.pdf` Section 3.1 + Appendix A |
