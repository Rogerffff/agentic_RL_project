# Eval Result 深度分析补充任务书

## 1. 背景

当前已有主分析文档：

- [eval_result_deep_analysis.md](/Users/xiaohui/Desktop/agentic-RL-project/verl-carr-deepsearch/examples/carr_deepsearch/docs/eval_result_deep_analysis.md)

该文档已经足够支撑高层结论：

- RL 在 `dd111` 和 `bc256` 上都带来了正向 accuracy 提升
- paired `net gain` 为正
- `response_limit` 下降、`finished` 上升、`finished-to-correct` 上升
- timeout / parse error / rubric / response length 等辅助指标已初步覆盖

但当前版本还不足以作为最终面试素材，主要缺口有三类：

1. `finished-but-wrong` 分类可信度不足，容易把“明确答错”误归为“format non-compliant”
2. 缺少 `retention / overlap / oracle` 分析，无法完整描述 RL 对 baseline 正确样本集合的影响
3. case study 仍偏摘要化，尤其缺少可直接复述的 `BrowseComp` paired case

本任务只做这些缺口的补充，不重做整份主分析。

---

## 2. 目标

补齐当前主分析文档中最影响“面试可辩护性”的部分，产出可直接用于：

- 面试中回答“RL 到底学到了什么”
- 面试中回答“有哪些 regression / tradeoff”
- 面试中回答“BrowseComp 结果为什么低、低在哪里”
- 面试中给出 3-5 个真正可讲的 paired case

---

## 3. 输入数据

### 3.1 主分析文档

- [eval_result_deep_analysis.md](/Users/xiaohui/Desktop/agentic-RL-project/verl-carr-deepsearch/examples/carr_deepsearch/docs/eval_result_deep_analysis.md)

### 3.2 原始结果文件

- [dd111_sft/0.jsonl](/Users/xiaohui/Desktop/agentic-RL-project/verl-carr-deepsearch/examples/carr_deepsearch/eval_results/dd111_sft/0.jsonl)
- [dd111_async23/0.jsonl](/Users/xiaohui/Desktop/agentic-RL-project/verl-carr-deepsearch/examples/carr_deepsearch/eval_results/dd111_async23/0.jsonl)
- [bc256_sft_relaxed_v2/0.jsonl](/Users/xiaohui/Desktop/agentic-RL-project/verl-carr-deepsearch/examples/carr_deepsearch/eval_results/bc256_sft_relaxed_v2/0.jsonl)
- [bc256_async23_relaxed_v2/0.jsonl](/Users/xiaohui/Desktop/agentic-RL-project/verl-carr-deepsearch/examples/carr_deepsearch/eval_results/bc256_async23_relaxed_v2/0.jsonl)

### 3.3 已确认前提

- `dd111` 的 SFT / async23 两个文件行序对齐，可按行号 paired 对比
- `bc256` 的 SFT / async23 两个文件行序对齐，可按行号 paired 对比

---

## 4. 分析约束

### 4.1 严格口径

- 所有 “是否交了最终答案” 的判断，一律基于 `last_assistant_chunk`
- 禁止直接在整条 `output` 上搜 `## Exact Answer` 后下结论
- 所有比例必须报告分子/分母
- 样本数 `<20` 时必须标注小样本 caveat

### 4.2 `last_assistant_chunk` 定义

从 `output` 中取最后一个 `assistant\n` 之后的内容，作为最终 assistant turn。

原因：

- 有些样本在更早的轨迹里出现过 `## Exact Answer` 草稿
- 但最后一个 assistant turn 仍在搜索、planning 或 tool call 中
- 直接扫整条 `output` 会把“中途草稿”误判为“最终交卷”

### 4.3 问题摘要提取要求

- `Question` 必须提取用户实际问题
- 不允许直接把 `system prompt` 和 tool schema 带进 case study
- 最终 case 卡片中的 `Question` 必须是 1-2 句自然语言摘要

---

## 5. 补充任务

### 5.1 重做 `finished-but-wrong` 审计

**目标**：修正主分析文档中 `Task 3.3` 的分类偏差。

**当前问题**：

主文档里 `bc256` 的 `finished-but-wrong` 样本被大量归类为 `format non-compliant`，但根据原始 `0.jsonl` 复核，其中不少样本其实已经明确交了错误答案。

**必须完成的内容**：

对四个 eval 文件分别找出：

- `task_unfinished=False`
- `outcome_reward=0`

然后基于 `last_assistant_chunk` 对每个样本做人工或半自动分类，分类只能用以下四类：

1. `Answered but wrong`
   - 明确给出了 `## Exact Answer`
   - 但答案与 `gts` 不符

2. `No final answer`
   - 最后一个 assistant turn 没有真正交出最终答案
   - 停在 planning / explanation 草稿 / tool_call / 引用整理等阶段

3. `Format non-compliant`
   - 已经接近或已经给出答案
   - 但结构明显不符合评测器预期，导致无法正常判定

4. `Suspected false negative`
   - 最终答案与 `gts` 基本一致
   - 仅差大小写、别名、轻微标点、Unicode 变体等
   - 需要谨慎标注，不可滥用

**额外要求**：

- 每个数据集至少给出 2 个典型样本
- 对于 `BrowseComp`，必须给出 1 个“明确答错”案例和 1 个“高疑似 false negative / 近似匹配”案例

**特别提醒**：

当前已知可能需要重点复核的 `bc256` finished negative 样本包括：

- `bc256_sft`: row `79`, `185`, `189`, `217`, `242`, `249`
- `bc256_async23`: row `28`, `79`, `112`, `151`, `153`, `155`, `194`

这些样本里很多已经明确交了一个错误答案，不能简单归类为格式问题。

**期望产出**：

- 一张新的 `finished-but-wrong` 分类表
- 每个数据集的分类解释
- 一段结论：当前失败更多是“没交卷”还是“交了但答错”

---

### 5.2 增加 `Retention / Overlap / Oracle` 分析

**目标**：补齐当前文档缺失的“正确样本集合”视角。

当前主分析只有：

- gain (`01`)
- regression (`10`)
- net gain

但缺少以下更适合面试深挖的指标：

1. `SFT-correct retention`
   - RL 保留了多少原本 SFT 就答对的样本
   - 公式：`11 / (10 + 11)`

2. `RL precision over its own correct set`
   - 已在 scoreboard 中体现为 RL accuracy，不需重复定义

3. `Overlap / Jaccard`
   - SFT correct 集合与 RL correct 集合的重叠程度
   - 公式：`11 / (01 + 10 + 11)`

4. `Oracle union accuracy`
   - 如果有一个理想 selector 能在 SFT 和 RL 中二选一，理论上可以覆盖多少正确样本
   - 公式：`(01 + 10 + 11) / N`

**必须完成的内容**：

分别对 `dd111` 和 `bc256` 计算并解释：

- `01 / 10 / 11 / 00`
- retention
- overlap / Jaccard
- oracle union accuracy

**分析要求**：

- 明确说明 RL 是“在原正确样本上继续加点”，还是“改写了正确样本集合”
- 明确说明 `BrowseComp` 的 correct set overlap 是否极低
- 给出一段适合面试的 tradeoff 解释

**期望产出**：

- 一张 paired set-overlap 表
- 一段结论：RL 的提升是 additive 还是 redistributive

---

### 5.3 补充 2 个真正可讲的 `BrowseComp` gain case

**目标**：让外部 benchmark 的改善不只停留在宏观数字。

当前主文档的 top gain case 几乎都来自 `dd111`，这不够支撑“外部 benchmark 也有可解释改善”。

**必须完成的内容**：

从 `bc256` 的 `01` 样本中至少挑 2 个最佳 case，要求：

- 问题本身可以被人类快速理解
- SFT 和 RL 的搜索策略差异明显
- 能清楚说明 RL 为什么成功、SFT 为什么失败

每个 case 必须包含：

- `dataset`
- `row_index`
- `question` 的自然语言摘要
- `gts`
- `outcome_pair`
- `termination_reason` 对比
- SFT 的前 3 个 search query
- RL 的前 3 个 search query
- SFT 最后一个 assistant turn 摘要
- RL 最后一个 assistant turn 摘要
- 一句“面试可讲的核心故事”

**优先挑选的故事类型**：

- RL 更早锁定了搜索方向
- RL 规避了无效 query loop
- RL 更快进入 answer submission
- RL 在更短轨迹中完成

**期望产出**：

- 2 张完整的 `BrowseComp` paired case card
- 可直接用于面试的 2 段讲稿

---

### 5.4 把 2-3 个 `dd111` gain case 扩写成真正的 paired trajectory case

**目标**：把当前“只给摘要指标”的 top gain，升级成能在面试中复述的案例。

当前主分析中的 top gain case 已经有 row 和摘要，但还缺：

- SFT 到底搜了什么
- RL 的 query reformulation 是什么
- 关键 pivot 发生在哪一步
- 为什么说它体现的是“更好 stopping”而不是“运气好”

**必须完成的内容**：

从主文档中的 gain case 里选择 2-3 个，优先：

- row `100`
- row `105`
- row `69`

每个 case 扩成完整 paired case card，要求：

- 问题摘要
- SFT 前 3 个 query
- RL 前 3 个 query
- 若 query 明显变好，指出“变好在哪里”
- SFT 最后一个 assistant turn 在做什么
- RL 最后一个 assistant turn 在做什么
- 定义该 case 的主要胜因，只能选一个主标签：
  - `better query formulation`
  - `better stopping`
  - `avoids loop`
  - `faster convergence`

**期望产出**：

- 2-3 张高质量 `dd111` paired case card
- 每个 case 附一句 30 秒面试讲法

---

### 5.5 增加 `thinking-only / low-tool failure` 分析

**目标**：补齐当前主文档里已经隐约出现、但尚未系统化的失败模式。

主文档中最慢样本里出现了：

- `response_limit`
- `tool_call_counts=0`

这说明有一类失败不是“搜错了”，而是“在真正调用工具前就被长 thinking 吃掉了 budget / token”。

**必须完成的内容**：

定义并统计两类样本：

1. `thinking-only failure`
   - `tool_call_counts=0`
   - `termination_reason=response_limit`

2. `low-tool failure`
   - `tool_call_counts<=3`
   - `termination_reason=response_limit`

分别对 4 个 eval 统计：

- 样本数
- 占所有失败样本的比例
- 平均 `response_length`
- 平均 `rollout_elapsed_s`

再挑 1-2 个代表性样本，解释：

- 模型在没有真正展开搜索前就耗尽了什么
- 这对后续 prompt / stop rule / thinking budget 设计意味着什么

**期望产出**：

- 一个小表
- 一段工程含义解释

---

### 5.6 收紧 `content_early_stopped` 的因果表述

**目标**：把主文档里可能过强的表述改成更稳妥、更适合面试的版本。

当前主文档把 `content_early_stopped` 讲成了强机制信号，但更合理的说法是：

- 它是“成功完成的强相关信号”
- 不应直接当成单独的因果机制

**必须完成的内容**：

在现有统计基础上再补一层条件分析：

- 只看 `finished` 样本时：
  - `early_stop` 的正确率
  - `natural completion` 的正确率
- 对比 `finished` 内部的这两类，而不是直接和总体 outcome 对比

**分析要求**：

- 如果 `natural` 更准，要如实写
- 如果样本极小，必须标注 caveat
- 最终结论必须用“correlated with success”而不是“caused the gain”

**期望产出**：

- 一小节修正版结论
- 一段更稳妥的面试口径

---

## 6. 输出格式

### 6.1 输出文件

请将所有补充分析写入一个新文档：

- `examples/carr_deepsearch/docs/eval_result_deep_analysis_followup.md`

不要直接覆盖原文档。

### 6.2 文档结构建议

1. `Why Follow-up Was Needed`
2. `Re-audited Finished-but-Wrong Cases`
3. `Retention / Overlap / Oracle`
4. `BrowseComp Gain Cases`
5. `DeepDive Paired Trajectory Cases`
6. `Thinking-Only / Low-Tool Failures`
7. `Revised Interview Takeaways`

### 6.3 每个 case 的最小格式

```
Dataset: dd111 / bc256
Row Index: N
Question: [自然语言摘要]
Ground Truth: [gts]
Outcome Pair: SFT=X / RL=Y
Termination: SFT=[reason] / RL=[reason]

SFT:
- tool_calls=
- search/open/find=
- response_length=
- rollout_elapsed_s=
- first 3 queries=
- last assistant summary=

RL:
- tool_calls=
- search/open/find=
- response_length=
- rollout_elapsed_s=
- first 3 queries=
- last assistant summary=

Main Story:
[一句话总结为什么 RL 胜 / 败]
```

---

## 7. 验收标准

完成后，这份 follow-up 文档必须能回答下面这些面试问题：

1. `RL 提升是建立在保留原正确样本的基础上，还是换了一批正确样本？`
2. `BrowseComp 的 gain case 能具体举两个吗？`
3. `当前失败更多是没交最终答案，还是交了但答错？`
4. `有些极慢失败为什么几乎没怎么用工具？`
5. `content_early_stopped 到底是因果机制还是成功信号？`

如果这 5 个问题仍不能被直接回答，说明补充分析还不够。

---

## 8. 最终用途

这份补充分析的用途不是再写一版总报告，而是为了支持三件事：

- 修订简历 bullet 的措辞边界
- 充实面试 Q&A bank
- 形成 3-5 个真正可复述的 paired case story

因此请优先产出：

- 可辩护的结论
- 清晰的反例/回退样本
- 可直接讲述的 paired case

而不是再堆更多宽泛的描述性统计。
