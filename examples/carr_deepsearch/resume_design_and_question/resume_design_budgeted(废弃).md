# CaRR DeepSearch 简历结果设计（预算受限版）

> 适用场景：GPU 预算不足以完成论文口径的完整 RL 训练，但希望尽快把项目写进简历、项目集和面试材料。
>
> 本文件的核心原则只有一句话：
> **可以仿照论文的结果展示口径，但不能把论文的数字写成自己的结果。**

## 1. 先说结论

当前这个项目**完全可以写成一个强简历项目**，但更适合定位为：

- `CaRR/C-GRPO on verl` 的**工程化复现与成本受限 partial reproduction**
- 一个**带实证分析的 agentic RL 系统项目**
- 一个展示你具备：
  - 多轮 agent loop 改造能力
  - reward bridge / custom advantage 实现能力
  - 多服务联调能力
  - 分布式 RL 训练诊断能力
  - 成本与失败模式分析能力

不建议定位为：

- “完整复现 CaRR 论文主表结果”
- “在 BrowseComp / xbench / GAIA 上复现论文级 benchmark 提升”
- “验证了 C-GRPO 一定优于 GRPO 的最终 benchmark 结论”

当前最安全、最强的叙事不是“我跑出了论文数字”，而是：

> 我在 verl 上实现了 CaRR 风格的 deep-search RL pipeline，完成了 SFT + formal RL + validation 的端到端闭环；在成本受限的正式训练中观测到中期 reward 改善，并定位出从 `response_limit` 向 `rollout_timeout` 转移的 failure mode。

---

## 2. 哪些结果可以写，哪些不能写

### 2.1 可以直接写进简历的“硬结果”

这些是你当前已经有证据支持的：

1. **方法主链路已经跑通**
   - SFT 冷启动
   - verl multi-turn agent loop
   - search/open/find 工具链
   - DeepSeek judge reward server
   - C-GRPO advantage estimator
   - formal RL 训练与验证

2. **正式 RL 训练确实完成了一个有意义的 partial run**
   - 8 x RTX 6000 96GB
   - formal RL 跑到 `step 96`
   - 有 `step40` 和 `step80` 的验证产物
   - 有 `global_step_90` checkpoint

3. **中期训练窗口出现过真实改善**
   - `step21-70` 平均 `outcome_reward = 0.2656`
   - `step71-90` 平均 `outcome_reward = 0.3203`
   - 相对提升约 `+20.6%`
   - `task_unfinished` 从 `0.5719` 降到 `0.4969`
   - `termination_response_limit` 从 `0.3731` 降到 `0.2656`

4. **后期训练出现了明确的失败模式转移**
   - `step91-96` 平均 `termination_rollout_timeout = 0.5208`
   - `task_unfinished = 0.7344`
   - `outcome_reward = 0.1667`
   - 说明继续沿当前配置训练的性价比已经明显下降

### 2.2 可以写，但必须带限定词的结果

这些可以写，但必须明确是 **partial / cost-constrained / internal validation**：

- `step40` eval: `32` 样本，`score = 0.3438`
- `step80` eval: `64` 样本，`score = 0.2656`
- `rubric_reward`、`task_unfinished`、`tool_call_counts` 等监控指标

限制说明：

- 这两次 eval 是不同规模、不同随机子集
- 不能把它们当成严格可比的 benchmark 主结果
- 更适合写成“训练中期验证快照”或“internal validation snapshot”

### 2.3 不能写成自己结果的内容

- 论文主表里的 BrowseComp / xbench / GAIA / DeepResearch Bench 数字
- “C-GRPO 在 4B 上提升 5.1 / 8.0 points” 这种论文结论
- “我复现了论文 benchmark improvement”
- “我证明了 C-GRPO 显著优于 GRPO”

这些最多只能写成：

- `paper reference`
- `target setup`
- `design target`
- `reported in the paper`

---

## 3. 当前最适合的项目定位

推荐项目标题：

- `CaRR-Style Deep Search RL Reproduction on verl`
- `Cost-Constrained Reproduction of Citation-Aware Deep Search RL`
- `Deep Search Agent RL on verl with Custom C-GRPO`

推荐副标题：

- `Implemented an end-to-end multi-turn search agent training pipeline with custom agent loop, reward bridge, and C-GRPO advantage; completed partial formal RL and diagnosed cost-driven failure modes.`

推荐中文一句话介绍：

> 基于 verl 工程化复现 CaRR 风格的 deep-search agent RL，完成多轮工具调用、judge-based reward、C-GRPO 优势函数与正式训练闭环，并在预算受限条件下分析了 reward 改善与 timeout 退化的转折点。

---

## 4. 当前最值得对外展示的数字

## 4.1 训练动态主表

这张表是当前最适合放在项目文档、作品集或面试 slide 里的。

| Phase | Setup note | outcome_reward | rubric_reward | task_unfinished | termination_response_limit | termination_rollout_timeout | 解释 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `step21-70` | formal RL 前半段 | `0.2656` | `0.0499` | `0.5719` | `0.3731` | `0.0694` | 早期主要死在 response limit |
| `step71-90` | `max_tool_response_length` 调到 `5000` 后的中期窗口 | `0.3203` | `0.0545` | `0.4969` | `0.2656` | `0.0844` | 中期出现改善，unfinished 和 response-limit 均下降 |
| `step91-96` | 后期长尾恶化窗口 | `0.1667` | `0.0219` | `0.7344` | `0.1146` | `0.5208` | failure mode 从 response-limit 转成 rollout-timeout |

这张表最重要的结论不是“最后失败了”，而是：

1. 中期训练**确实出现了改善**
2. 后期改善**没有稳定住**
3. 预算受限下，后期继续训练的边际收益很低

### 可以直接提炼成一句结果

> In the cost-constrained formal RL run, the `step71-90` window improved `outcome_reward` by `20.6%` over `step21-70`, reduced `task_unfinished` by `7.5 pp`, and lowered `response-limit` failures by `10.8 pp`, before the run later degraded into `rollout-timeout` dominated behavior.

## 4.2 验证快照表

这张表可以放，但不要当主 benchmark 表。

| Eval snapshot | Samples | score / outcome_reward | rubric_reward | task_unfinished | 备注 |
| --- | ---: | ---: | ---: | ---: | --- |
| `step40` | `32` | `0.3438` | `0.0570` | `0.5938` | 训练中期验证快照 |
| `step80` | `64` | `0.2656` | `0.0533` | `0.6719` | 训练后期验证快照；与 `step40` 非同一固定子集 |

对外写法建议：

- 可以写：`internal validation snapshots on DeepDive RL val`
- 不要写：`final benchmark result`

---

## 5. 最推荐的简历主叙事

如果你现在就要投递，最强的叙事不是“跑了多少 step”，而是下面这三层。

### 第一层：工程实现

> 在 verl 上从零打通 CaRR 风格的 deep-search RL 训练链路，包含多轮 agent loop、browser 工具、reward server、C-GRPO 优势函数和正式训练/验证流程。

### 第二层：可量化结果

> 在成本受限的 formal RL 中，训练中期窗口相对前期窗口实现 `outcome_reward +20.6%`、`task_unfinished -7.5 pp`、`response-limit -10.8 pp`。

### 第三层：研究/诊断价值

> 进一步发现将 `max_tool_response_length` 从 `6000` 降到 `5000` 虽然降低了 response-limit unfinished，但后期会将主要失败模式转移到 `rollout_timeout`，为后续 budget 与 observation-density 调参提供了直接依据。

这三层合在一起，会比“我差一点复现论文结果”更强。

---

## 6. 推荐的项目结果写法

## 6.1 适合放在项目文档/作品集的结果段落

推荐模板：

> We implemented an end-to-end CaRR-style deep-search RL pipeline on top of verl, including a custom multi-turn agent loop, browser-tool bridge, DeepSeek-based outcome/rubric reward server, and a C-GRPO advantage estimator.  
> Under a cost-constrained 8-GPU formal RL setup, the mid-training window (`step71-90`) improved `outcome_reward` from `0.2656` to `0.3203` relative to the earlier window (`step21-70`), while reducing `task_unfinished` from `57.2%` to `49.7%`.  
> We also identified a critical failure-mode transfer: after reducing tool-response truncation to control 64k overlength failures, the run later became dominated by `rollout_timeout`, with timeout ratio rising to `52.1%` in `step91-96`. This made continued training under the same configuration economically unattractive.

## 6.2 适合放在简历里的 2 条 bullet

### 中文版

- 基于 `verl` 工程化复现 CaRR 风格的 deep-search agent RL：实现多轮 `agent loop`、`search/open/find` 工具桥接、DeepSeek judge reward server 与自定义 `C-GRPO` advantage estimator，打通 `SFT -> RL -> validation` 全链路。
- 在 `8 x RTX 6000 96GB` 的正式 RL 训练中完成 `96` step partial run；中期训练窗口相对前期实现 `outcome_reward +20.6%`、`task_unfinished -7.5 pp`，并定位出从 `response_limit` 向 `rollout_timeout` 转移的失败模式。

### 英文版

- Built a CaRR-style deep-search RL pipeline on top of `verl`, including a custom multi-turn agent loop, browser-tool bridge (`search/open/find`), DeepSeek-based outcome/rubric reward service, and a custom `C-GRPO` advantage estimator.
- Ran a formal cost-constrained RL experiment on `8 x RTX 6000 96GB`; the mid-training window improved `outcome_reward` by `20.6%` and reduced `task_unfinished` by `7.5 pp`, while revealing a key failure-mode transfer from `response_limit` to `rollout_timeout`.

---

## 7. 推荐的“论文风格结果页”结构

如果你要做项目页或面试 slide，建议完全沿用论文风格的结构，但把“paper result”和“our result”分开。

### Section A. Method

- Problem: multi-hop deep-search QA
- Agent: ReAct-style multi-turn search agent
- Tools: `search / open / find`
- RL: C-GRPO on top of verl PPO stack

### Section B. Cost-Constrained Reproduction Setup

- Backbone: `Qwen/Qwen3-4B`
- SFT: 3 epochs
- RL: formal run to `step96`
- Infra: `8 x RTX 6000 96GB`
- Reward: DeepSeek judge
- Limitation: did not complete full paper-scale benchmark sweep

### Section C. Observed Training Dynamics

放前面的“训练动态主表”。

### Section D. Validation Snapshots

放前面的“验证快照表”。

### Section E. Paper Reference

单独写：

> The original CaRR paper reports benchmark gains on BrowseComp / xbench / GAIA under a larger, more paper-aligned setup. Those numbers are included here as paper references only and are **not** claimed as reproduced results in this project.

这个 section 的作用是让读者知道你理解论文目标，但不会把论文数字冒充成自己的结果。

---

## 8. 面试时最推荐的说法

如果面试官问：“你最后结果怎么样？”

不要说：

- “还没跑完，所以结果一般”
- “预算不够，所以没做出来”

推荐说法：

> 我把论文方法在 verl 上完整工程化了，并完成了正式 RL 的 partial reproduction。  
> 在训练中期，reward 和 unfinished 指标确实出现了改善；但继续训练后，失败模式从 response-limit 转移到了 rollout-timeout，导致单位 GPU 成本显著上升。  
> 所以我没有继续无脑烧卡，而是把重点放在系统诊断和 failure-mode analysis 上。这也是我认为这个项目最有价值的部分。

这会把“预算不够”转成“你有研究判断力和工程取舍能力”。

---

## 9. 明确禁止的写法

下面这些写法不要出现在简历、项目页、面试表述里：

- `Reproduced the CaRR paper results`
- `Matched paper-level benchmark numbers`
- `Achieved SOTA gains on BrowseComp / xbench / GAIA`
- `Verified that C-GRPO outperforms GRPO on benchmark accuracy`
- `Obtained the same improvements as reported in the paper`

替代写法：

- `paper-aligned implementation`
- `cost-constrained reproduction`
- `partial formal RL run`
- `internal validation snapshots`
- `observed training dynamics consistent with/related to the paper's discussion`

---

## 10. 最终建议

如果你现在就要投递，建议采用下面的最终口径：

### 对外主定位

`CaRR-style deep-search RL reproduction on verl`

### 对外主结果

- 实现了完整方法链路
- 完成了正式 partial RL
- 中期训练出现明确改善
- 后期识别出高成本 failure mode transfer

### 不主打的内容

- 论文主 benchmark 数字
- 还没跑完的 GRPO/C-GRPO 对照主表
- 还没做完的 BrowseComp 固定子集实验

这套口径的优点是：

1. 真实
2. 技术含量高
3. 有量化结果
4. 有研究洞察
5. 不会在面试里被一问就穿

如果后面你还有一点预算，可以再补一个最小增强项：

- 用固定子集做一次 `SFT checkpoint` 的 BrowseComp eval

这样你就至少会多一个“训练前基线”结果，但即使没有这一步，当前这套项目叙事也已经可以用于简历投递。
