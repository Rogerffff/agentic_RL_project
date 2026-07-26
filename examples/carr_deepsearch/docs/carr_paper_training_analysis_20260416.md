# CaRR 论文训练现象与当前结果对照分析

> 日期: 2026-04-16
> 论文来源: `CaRR/2601.06021v1.pdf`
> 本地对照来源:
> - `examples/carr_deepsearch/docs/eval_result_deep_analysis.md`
> - `examples/carr_deepsearch/docs/eval_result_deep_analysis_followup.md`
> - `examples/carr_deepsearch/docs/eval_analysis_20260415.md`
> - `examples/carr_deepsearch/docs/training_full_history_20260404.md`
> - `examples/carr_deepsearch/scripts/run_eval_integration.sh`

## 1. 结论先行

这篇论文最重要的训练结论，不是“答对率提升”本身，而是下面 4 点：

1. 纯 outcome reward 的 GRPO 会学到 shortcut exploitation，64k 内可能涨分，但在更长 context 和更复杂问题上会暴露出不充分验证的问题。
2. C-GRPO 通过只对 `outcome=1` 的正确 rollout 叠加 citation-aware rubric reward，把优化目标从“答对”进一步收紧为“答对且证据链完整”。
3. 训练动态上，GRPO 与 C-GRPO 一开始都会减少 tool calls 来避免 overlength，但后续只有 C-GRPO 会重新增加 evidence gathering；GRPO 则继续向“少查一点、赌答案”收缩。
4. C-GRPO 的核心收益不仅体现在 accuracy，还体现在 cited pages 更多、supported rubrics 更多、connected rubrics 更多，以及 128k test-time scaling 更强。

和你当前本地结果相比，结论可以分成三类：

- `方向一致`：你的 `async23` 相对 `SFT` 在内部 `DeepDive` 和外部 `BrowseComp subset256` 都是正提升；`response_limit`/unfinished 仍是核心瓶颈；RL 提升不是靠粗暴增加 tool volume。
- `暂时无法验证`：论文最关键的 `C-GRPO > GRPO`、`128k scaling 更强`、`cited pages / rubric connectivity 更强`，你当前本地结果还没有直接复现这些比较。
- `存在明显差距`：你当前 `BrowseComp subset256 64k` 的绝对分数明显低于论文 4B 结果，而且这个差距在 `SFT` 阶段就已经存在，因此不能简单归因于“RL 还没训够”。

因此，更准确的判断是：

- 你的当前结果和论文在“rubric-aware RL 比纯 SFT 更有帮助”这个方向上是相符的。
- 但你的当前实验还不足以证明“已经复现了论文的完整算法结论”，尤其是 `C-GRPO vs GRPO` 和 `64k -> 128k scaling` 这两块。
- 你当前 `BrowseComp` 偏低，最可能不是工具类型错了，而是训练长度、SFT 长上下文能力、额外 budgets、评测 judge、以及 `open` 截断长度共同造成的协议差异。

---

## 2. 论文中明确报告的训练设置

### 2.1 数据与模型

论文使用 DeepDive 作为训练数据来源，并分别训练两个 backbone：

- `Qwen3-4B-Thinking-2507`
- `Qwen3-30B-A3B-Thinking-2507`

数据规模：

- SFT 样本: `1,016`
- RL 样本: `2,234`

但 SFT 不是直接用全部 `1,016` 条，而是先用 `GLM-4.6` 在 SFT split 上做 reject sampling，最终得到 `832` 条高质量 SFT traces。

### 2.2 SFT 超参数

| 项目 | 论文值 |
|---|---:|
| epochs | `3` |
| batch size | `16` |
| learning rate | `4e-5` |
| max context length | `128k` |
| SFT traces | `832` |

### 2.3 RL 超参数

| 项目 | 论文值 |
|---|---:|
| RL QA pairs | `2,234` |
| rollout size | `16` |
| samples per prompt | `8` |
| global batch size | `128` |
| temperature | `1.0` |
| learning rate | `2e-6` |
| max context length | `64k` |
| epochs | `3` |
| rubric reward weight `alpha` | `0.3` |
| judge LLM | `DeepSeek-v3.2` |

### 2.4 工具与输出协议

论文环境采用标准 ReAct deep-search agent：

- 工具只有 `search` / `open` / `find`
- `search` 使用 `Serper API`
- `open` 使用 `Jina API` 抓网页，并返回前 `10k chars`
- `find` 使用字符串匹配

最终回答要求包含两个固定段落：

- `## Explanation with Citations`
- `## Exact Answer`

这意味着论文中的 reward 不是只看最后一句答案，而是强依赖：

- 最终 response 中是否显式写出 hidden entities
- 最终 response 中是否包含 citations
- cited contents 是否真正支撑 rubric
- supported rubrics 是否能连到 final answer

---

## 3. 论文中观察到的全部训练现象与关键指标

## 3.1 核心算法现象

论文对现有 outcome-only RL 的主要批评有两个：

1. `shortcut exploitation`
2. `hallucination tolerance`

作者认为，纯二值 outcome reward 无法区分：

- “答对但证据链不完整”的 rollout
- “只抓住最后几跳信息、忽略前面约束”的 rollout
- “靠侥幸猜对”的 rollout

因此论文的核心现象不是“纯 GRPO 完全无效”，而是：

- GRPO 在 RL context length 内可以涨分
- 但它会学到更脆弱的策略
- 这种策略在更长 context、更多 tool budget、或者更难问题上会掉队

## 3.2 C-GRPO 的 reward 结构

CaRR 的 rubric reward 分 3 步：

1. hidden entity identification
2. citation-based rubric judgment
3. evidence connectivity check

最终 rubric reward 是：

- `connected & supported rubrics / total rubrics`

而 C-GRPO 不是给所有 rollout 都加 rubric reward，而是只给 `outcome=1` 的正确 rollout 加权：

- `R_i = (1 - alpha) * outcome + alpha * outcome * normalized_rubric`

论文明确指出，如果给所有 rollout 都加 rubric reward，训练会明显变差，因为：

- RL 早期 correct rollout 很少
- overlength rollout 很多
- 错误 rollout 也可能因为命中一些局部 rubric 而拿到正 advantage
- 这会把策略往错误方向推

## 3.3 主 benchmark 结果

论文在 4 个 benchmark 上报告了 `64k / 128k` 结果：

- BrowseComp
- BrowseComp-ZH
- xbench-DeepSearch
- GAIA text-only validation subset

### 4B 主结果

| Model | BrowseComp | BrowseComp-ZH | xbench-DS | GAIA |
|---|---:|---:|---:|---:|
| 4B-SFT | `7.7 / 14.1` | `10.1 / 16.6` | `34.0 / 44.3` | `39.5 / 46.0` |
| + GRPO | `12.9 / 14.7` | `16.6 / 17.5` | `41.0 / 41.3` | `40.5 / 41.1` |
| + E-GRPO | `11.5 / 14.5` | `16.5 / 20.2` | `43.7 / 45.0` | `42.4 / 42.4` |
| + C-GRPO | `13.9 / 17.5` | `18.2 / 24.7` | `50.3 / 54.0` | `48.9 / 50.2` |

### 30B 主结果

| Model | BrowseComp | BrowseComp-ZH | xbench-DS | GAIA |
|---|---:|---:|---:|---:|
| 30B-SFT | `12.2 / 20.5` | `15.8 / 24.7` | `43.0 / 54.3` | `46.0 / 50.8` |
| + GRPO | `16.0 / 18.9` | `24.1 / 26.1` | `51.3 / 52.0` | `51.1 / 51.1` |
| + E-GRPO | `13.1 / 18.5` | `17.1 / 24.0` | `51.7 / 55.7` | `52.8 / 55.3` |
| + C-GRPO | `17.9 / 24.8` | `26.0 / 33.3` | `55.3 / 57.7` | `53.7 / 56.3` |

论文明确给出的聚合结论：

- 4B: `C-GRPO - GRPO` 平均提升 `+5.1 @64k`, `+8.0 @128k`
- 30B: `C-GRPO - GRPO` 平均提升 `+2.6 @64k`, `+6.0 @128k`

对 BrowseComp 来说，论文 4B 的 SFT 到 C-GRPO 提升是：

- `7.7 -> 13.9` at `64k`, 绝对 `+6.2`
- `14.1 -> 17.5` at `128k`, 绝对 `+3.4`

这意味着论文的核心不是“RL 在 64k 内涨了一点”，而是：

- `C-GRPO` 在 `64k` 内优于 `GRPO`
- 更重要的是它在 `128k` 仍保持甚至放大优势

## 3.4 Figure 3: 论文对 test-time scaling 的观察

论文 Figure 3 给出的关键结论是：

- GRPO 在 `64k` context 内相对 SFT 有提升
- 但到了更长 context budget 和更高 tool-call budget，GRPO 的 scaling 不如 SFT 稳定
- C-GRPO 在更长 context 和更大 tool budget 下明显更能继续涨

换句话说，论文不是把 `64k` 当终点，而是把 `64k -> 128k` 的 scaling 作为检验 agent 是否真的学会“更深搜索”的关键证据。

## 3.5 Figure 4: 论文最重要的训练动态

这是整篇论文最值得和你当前项目对照的部分。

论文观察到：

1. GRPO 和 C-GRPO 的平均 tool-call steps 在训练初期都会下降。
2. 这不是坏事，说明模型先学会了减少无效搜索、避免 overlength rollouts。
3. 但之后两者分化：
   - `GRPO` 在小幅回升后继续下降
   - `C-GRPO` 在初期下降后重新上升
4. 论文把这种分化解释为：
   - `GRPO`: 学到 shortcut policy，只验证少量证据就尝试交答案
   - `C-GRPO`: 为了满足更多 rubrics，重新增加 evidence gathering
5. 在这段时期里：
   - `C-GRPO` 的 outcome reward 还略高于 `GRPO`
   - `C-GRPO` 的 rubric reward 持续上升

论文因此得出的训练现象是：

- “tool calls 一直降”并不一定是好事
- outcome reward 单独上涨也不够
- 更鲁棒的 deep-search policy 应该表现为：
  - 初期减少无效搜索
  - 后期重新增加有价值的证据收集
  - rubric reward 稳步提升

## 3.6 Table 2: comprehensiveness / factuality 指标

论文额外比较了 30B agents 在 BrowseComp solved subset 上的 cited pages 和 rubric satisfaction。

| Model | `|C^H|` cited pages | `|R_identify|` | `|R_support|` | `|R_connect|` | `|R_q|` |
|---|---:|---:|---:|---:|---:|
| SFT | `3.8` | `8.0` | `6.2` | `4.5` | `10.1` |
| GRPO | `3.5` | `7.5` | `5.3` | `4.0` | `10.1` |
| C-GRPO | `4.3` | `8.2` | `6.6` | `5.2` | `10.1` |

这组数非常关键，因为它直接支持了论文的机制性结论：

- C-GRPO 不只是“更容易 judge-pass”
- 它确实引用了更多网页
- 它确实满足了更多 rubrics
- 它确实构成了更完整的 evidence chain

而 GRPO 在这组指标上甚至低于 SFT，这正是论文所说的 shortcut exploitation。

## 3.7 Table 3: open-ended deep research 泛化

论文在 DeepResearch Bench 上还报告了开放式长报告生成结果。

### 4B

| Model | Overall | Comp. | Insight | Inst. | Read. |
|---|---:|---:|---:|---:|---:|
| SFT | `33.81` | `29.57` | `24.23` | `44.05` | `41.02` |
| GRPO | `34.79` | `31.29` | `26.79` | `43.81` | `41.58` |
| E-GRPO | `36.59` | `33.20` | `28.30` | `45.58` | `42.67` |
| C-GRPO | `37.51` | `33.88` | `30.01` | `45.72` | `43.82` |

### 30B

| Model | Overall | Comp. | Insight | Inst. | Read. |
|---|---:|---:|---:|---:|---:|
| SFT | `37.51` | `34.27` | `28.85` | `46.77` | `43.21` |
| GRPO | `39.30` | `36.10` | `31.66` | `47.65` | `44.92` |
| E-GRPO | `36.12` | `32.31` | `27.73` | `45.72` | `42.33` |
| C-GRPO | `41.99` | `39.75` | `35.87` | `48.51` | `46.63` |

论文把这个结果解释为：

- synthetic QA 上学到的 citation-aware search policy
- 可以泛化到 open-ended deep research tasks
- 尤其是 30B C-GRPO 甚至超过了一些 proprietary-data agents

## 3.8 Table 4 和 Table 5: 消融实验结论

### `alpha` 权重

| Setting | BC | BC-ZH | xbench-DS | GAIA |
|---|---:|---:|---:|---:|
| `alpha=0` | `14.7` | `17.5` | `41.3` | `41.1` |
| `alpha=0.1` | `13.0` | `18.0` | `46.0` | `46.3` |
| `alpha=0.3` | `17.5` | `24.7` | `54.0` | `50.2` |
| `alpha=0.5` | `17.0` | `20.8` | `49.3` | `42.4` |

论文结论：

- `alpha` 太小，rubric reward 不够
- `alpha` 太大，会分散“先答对”的主目标
- `alpha=0.3` 最优

### 组件消融

| Setting | BC | BC-ZH | xbench-DS | GAIA |
|---|---:|---:|---:|---:|
| Full C-GRPO | `17.5` | `24.7` | `54.0` | `50.2` |
| w/o hidden entity identification | `16.5` | `23.2` | `50.7` | `46.6` |
| w/o evidence connectivity check | `15.1` | `20.8` | `47.7` | `44.0` |
| rubric reward for all rollouts | `13.3` | `14.0` | `40.3` | `40.8` |

论文结论：

- hidden entity identification 有用
- evidence connectivity check 非常关键
- 给所有 rollout 加 rubric reward 会严重伤害训练

## 3.9 Judge 可靠性

论文附录 C 手工复核了 `10` 条 `DeepDive-30B-SFT` 轨迹，覆盖：

- `128` 个 hidden entities
- `164` 个 rubrics

judge LLM 相对人工标注的准确率：

- hidden entity identification: `97.7%`
- citation-based rubric evaluation: `95.1%`

这说明论文并不是把 judge 当作完全无误，而是给出了一个“足够可靠但并非完美”的量化先验。

---

## 4. 你当前本地 eval 结果的核心结论

## 4.1 当前对照对象

你当前已经拿到的稳定结果，主要是：

- `DeepDive rl_val 111`: `SFT vs async23`
- `BrowseComp subset256 64k`: `SFT vs async23`

当前本地的 sampled eval recipe 是：

- `temperature=0.6`
- `top_p=0.95`
- `top_k=20`
- `do_sample=true`
- `max_response_length=61440`
- `max_assistant_turns=120`
- `max_tool_response_length=6000`

外部 `BrowseComp subset256` 还使用了 relaxed-but-bounded eval envelope：

- `max_rollout_wall_time_s=600`
- `max_real_rollout_wall_time_s=1200`
- `max_tool_calls=160`
- `max_search_calls=80`
- `max_open_calls=60`
- `max_find_calls=40`

本地 agent loop 仍然是同一类工具接口：

- `browser.search`
- `browser.open`
- `browser.find`

因此，当前 `BrowseComp` 偏低不太像是“根本没给对工具”，而更像是：

- 训练规模差异
- context / budget / judge 协议差异
- `open` 返回长度更短
- 当前 checkpoint 仍然存在大量 response-limit unfinished

## 4.2 当前本地结果

### DeepDive rl_val 111

| Metric | SFT | async23 | Delta |
|---|---:|---:|---:|
| outcome | `0.1802` | `0.3243` | `+0.1441` |
| rubric | `0.0332` | `0.0710` | `+0.0378` |
| unfinished | `0.7297` | `0.6306` | `-0.0991` |
| response_limit | `0.5135` | `0.3784` | `-0.1351` |
| rollout_timeout | `0.0090` | `0.0000` | `-0.0090` |
| finished-to-correct | `20/30 = 66.7%` | `36/41 = 87.8%` | `+21.1 pts` |

### BrowseComp subset256 64k

| Metric | SFT | async23 | Delta |
|---|---:|---:|---:|
| judge-pass / outcome | `4/256 = 1.6%` | `9/256 = 3.5%` | `+5` |
| finished | `11` | `17` | `+6` |
| unfinished | `245` | `239` | `-6` |
| response_limit | `231/256 = 90.2%` | `223/256 = 87.1%` | `-3.1 pts` |
| rollout_timeout | `3/256 = 1.2%` | `2/256 = 0.8%` | `-0.4 pts` |

本地 follow-up 分析还给出了几个重要结论：

- `dd111` 上是明显正向 net gain: `24 gains vs 8 regressions`
- `bc256` 上也是正向 net gain: `9 gains vs 4 regressions`
- finished-but-wrong 的主因是“明确交了错误答案”，不是格式问题
- `content_early_stopped` 更像成功相关信号，而不是因果机制
- `thinking-only` failure 确实存在，但只占失败样本的一小部分

---

## 5. 与论文一致的现象

## 5.1 RL 相对 SFT 的方向性提升

这点和论文一致。

论文里：

- C-GRPO 相对 SFT 有明显 accuracy 提升

你本地当前：

- `DeepDive`: `0.180 -> 0.324`, 绝对 `+0.144`, 相对 `+80%`
- `BrowseComp subset256`: `1.6% -> 3.5%`, 绝对 `+1.9 pts`, 相对约 `+118.8%`

虽然绝对值远低于论文，但方向是一致的。

## 5.2 response-limit / overlength 仍然是核心瓶颈

这点也和论文一致。

论文训练动态明确强调：

- 初期模型会先减少 tool calls 来避免 overlength

你本地结果里：

- `DeepDive` 的最大改善之一就是 `response_limit 51.4% -> 37.8%`
- `BrowseComp` 上最主要的失败原因仍然是 `response_limit`，而且高达 `90.2% / 87.1%`

这说明你当前系统和论文一样，都被“长轨迹 + 64k 上下文”强烈约束。

## 5.3 RL 提升不是靠简单堆高 tool volume

这点和论文的机制解释基本相容。

你本地 `async23` 的提升，并不是来自总 tool_call 明显更高：

- `DeepDive`: `47.99 -> 47.20`
- `BrowseComp`: `57.03 -> 55.48`

但正确率、finished rate、response_limit 都改善了。

这意味着当前 RL 学到的更像是：

- 更有效的搜索分配
- 更好的停止时机
- 更高的 finished-to-correct conversion

这和论文“鲁棒策略不等于盲目多查，而是更有价值地搜证据”并不冲突。

## 5.4 策略变化带来 gain 和 regression 的 correct-set churn

这也和论文对 policy shift 的描述一致。

论文并没有声称 RL 只是对 SFT 做纯 additive 的逐样本增强；它强调的是策略会变得更 robust。

你本地结果里：

- `dd111` retention 只有 `60%`
- `bc256` retention 是 `0%`，但基数只有 `4`

这说明 RL 的确改变了搜索策略和成功样本集合，而不是单纯“原来会做的继续会做，再额外加一些新题”。

---

## 6. 当前无法直接验证论文结论的部分

## 6.1 没有本地 `GRPO` 基线

论文最核心的算法性 claim 是：

- `C-GRPO > GRPO`

你当前本地只有：

- `SFT vs async23`

因此你当前结果最多只能说：

- “使用 CaRR/C-GRPO 风格 reward 的 RL checkpoint 相对 SFT 有正向提升”

不能说：

- “已经本地验证了 C-GRPO 明显优于 GRPO”

## 6.2 没有本地 `128k` scaling 结果

论文一个很强的结论是：

- `GRPO` 可能在 `64k` 看起来能涨
- 但 `C-GRPO` 在 `128k` 才真正体现出更强的 scaling

你当前稳定结果只有：

- `64k`

所以论文最强的“deep search 能力”证据，目前本地还没有直接验证。

## 6.3 没有复刻论文的 cited pages / rubric-connectivity 分析

论文 Table 2 是机制证明的重要一环。

你本地当前虽然有：

- `rubric_reward`
- `unfinished`
- `termination reason`
- `tool counts`

但还没有直接复刻：

- cited pages 数
- identified / supported / connected rubrics 数

因此你当前可以讲“行为更有效、更容易完成”，但还不能像论文那样强讲“证据链更完整”。

## 6.4 judge 口径不同

论文 benchmark 使用：

- `GPT-5-Chat`

你当前本地 eval 路径使用的是：

- `deepseek-chat` 风格 reward/eval server

这不一定会改变所有结论方向，但会影响绝对分数和可比性。尤其是在 `BrowseComp` 这种低基数场景下，judge 差异不能忽略。

---

## 7. 与论文不一致或明显更弱的地方

## 7.1 BrowseComp 绝对分数明显更低，而且差距从 SFT 就开始存在

这是当前最重要的不一致。

论文 4B 在 `BrowseComp 64k` 上：

- `SFT = 7.7`
- `C-GRPO = 13.9`

你当前本地在 `BrowseComp subset256 64k` 上：

- `SFT = 1.6`
- `async23 = 3.5`

即使只和论文 `4B-SFT 64k` 比，你当前 `SFT` 也低了约：

- `7.7 - 1.6 = 6.1 pts`

这说明绝对分差不可能全部由“RL 还没训满 3 epochs”解释，因为在 RL 之前就已经有较大缺口。

补充一点：

- 你当前 `BrowseComp` 的相对提升比例看起来不低，约 `+118.8%`
- 但这是建立在 `1.6%` 的极低 baseline 上
- 对这种低基数任务，更有解释力的是绝对通过数和 finished 数，而不是相对百分比

## 7.2 当前本地的成功样本数仍然过少

论文在 BrowseComp 上已经能稳定到双位数 accuracy，而你当前：

- `SFT 4/256`
- `async23 9/256`

finished 数也仍然很低：

- `11 -> 17`

这说明当前策略虽然方向变好了，但还远没到论文那种“系统性解决”的水平。

## 7.3 论文强调 C-GRPO 后期会重新增加 evidence gathering；你当前 final eval 更像“更省地搜”

这是一处需要谨慎解释的“表面不一致”。

论文的关键观察是：

- C-GRPO 在训练中后期会重新增加 tool-call steps

你当前 final eval 结果则是：

- total tool calls 基本持平或略降
- response_limit 降低
- finished / judge-pass 提升

这不一定和论文矛盾，原因有 3 个：

1. 论文比较的是 `GRPO vs C-GRPO` 的训练曲线，你当前比较的是 `SFT vs async23` 的最终评测。
2. 论文看的是训练动态，你当前看的是最终 eval 统计。
3. 论文用 cited pages / connected rubrics 衡量 evidence gathering，你当前只看 tool counts，还没有复刻那套指标。

因此，这里更准确的说法是：

- 你当前还没有证实论文所说的“后期更多证据收集”机制
- 但也不能仅凭 final tool-count 下降就说与你的结果矛盾

## 7.4 你的训练过程受额外 budgets 的影响远大于论文

论文里明确强调的截断主要是：

- 超过 token / tool-call 限制的 rollout 记为 0 reward

但论文没有给出你这种显式的：

- wall-time budget
- search budget
- open budget
- find budget
- assistant-turn budget

你当前训练历史里，这些额外 budgets 不只是防护栏，而是实际塑形了训练行为：

- sync 阶段主失败模式曾从 `response_limit` 转移到 `rollout_timeout`
- `max_tool_response_length 6000 -> 5000` 的改动也显式改变了 failure-mode mix

所以从 reward-signal 纯度上看，你的训练环境比论文更“受工程约束干预”。

---

## 8. 为什么你当前 BrowseComp 会低于论文

下面按影响优先级排序。

## 8.1 SFT 本身就比论文弱很多

这是第一原因。

你当前 SFT 和论文的关键差异：

| 项目 | 论文 | 你当前 |
|---|---:|---:|
| SFT max context | `128k` | `64k` |
| batch size | `16` | `4` |
| SFT truncation | 未见显式截断问题 | 约 `22%` 样本需要截断 |

这意味着你的模型在进入 RL 前就已经：

- 少见过长轨迹
- 少见过完整证据链
- 部分样本 final answer / reasoning 被截断

而 `BrowseComp` 恰恰是最依赖长轨迹和长上下文管理的任务之一。

## 8.2 RL 训练规模和论文差距很大

论文 RL 是：

- `2,234` RL samples
- `3 epochs`
- `16 prompts x 8 samples`

你当前：

- sync 只跑到 `96` steps 左右
- async 是从 `step70` warm-start 后的 `23` 个 param versions
- 训练 geometry 也明显小于论文主设定

这会直接影响两个东西：

- 正确 rollout 在 group 里的覆盖率
- 模型学到稳定 citation-aware search policy 的速度

## 8.3 当前外部集仍然被 `response_limit` 严重压制

这是当前最直接的行为证据。

在 `BrowseComp subset256 64k` 上：

- `SFT response_limit = 90.2%`
- `async23 response_limit = 87.1%`

也就是说，绝大部分样本根本没顺利完成到可 judge 的阶段。

而且这个结果还是在已经放宽过的 external eval envelope 下得到的：

- `max_rollout_wall_time_s=600`
- `max_real_rollout_wall_time_s=1200`
- `max_tool_calls=160`
- `max_search_calls=80`
- `max_open_calls=60`
- `max_find_calls=40`

这说明当前低分并不只是“评测预算设得太紧”，而是 agent 本身在 `64k` 条件下仍然很难完成足够深的搜索。

这和论文强调的 `128k` scaling 很一致：

- 如果 agent 仍然大量死在 64k 限制上
- 那么它的真实深搜索能力就还没有被充分释放出来

## 8.4 `open` 返回长度更短

论文的 `open` 返回前 `10k chars`，你当前本地常用口径是：

- `6000 chars`

这会带来两个副作用：

1. 每次 open 能看到的信息更少
2. 模型可能需要更多 `open/find` 才能补足证据

虽然它不是唯一原因，但会进一步加剧 long-horizon search 的困难。

## 8.5 额外 budgets 会改变 reward signal

这也是很可能的重要因素。

论文没有像你这样在训练中显式依赖：

- `max_rollout_wall_time_s`
- `max_search_calls`
- `max_open_calls`
- `max_find_calls`
- `max_assistant_turns`

这些预算本来的工程目标是：

- 切掉极端长尾
- 防止同步 RL 被少数 rollout 卡住

但副作用是：

- 它会改变哪些 rollout 被记作 unfinished / reward 0
- 从而改变 group 内 reward 分布
- 进而改变 advantage 的排序

这对 citation-aware RL 尤其敏感，因为长问题本来就依赖多跳证据收集。

从你当前训练历史看，这种影响不是纯理论上的：

- sync 阶段 `wall=360` 时，后期确实出现了从 `response_limit` 向 `rollout_timeout` 的 failure-mode transfer
- async 阶段把 wall 放宽到 `480s` 后，这种问题有所缓解，但并不能证明 reward signal 已经完全恢复到论文那种“只受 64k token 上限约束”的状态

## 8.6 当前 benchmark judge 与论文不同

论文 benchmark judge:

- `GPT-5-Chat`

你当前本地:

- `deepseek-chat`

这不是你分数偏低的唯一原因，因为你的 `SFT` 已经很低、response-limit 也很高。

但它确实会让绝对数字更难一比一对齐。

## 8.7 “工具类型错了”不是主要嫌疑

从当前代码和运行脚本看，本地 agent loop 仍然是：

- `browser.search`
- `browser.open`
- `browser.find`

所以更合理的判断是：

- 你和论文在工具族上是一致的
- 真正的问题在工具返回长度、预算 envelope、SFT 长上下文能力、RL 训练规模，以及 judge 口径

---

## 9. 对你当前项目最稳的表述方式

基于论文和你当前结果，最稳的技术结论是：

1. 你的本地结果已经方向性支持了论文的一个核心点:
   - citation-aware RL 相对 SFT 能改善 deep-search agent 的完成率和正确率。
2. 你当前最强的本地证据不是“完全复现论文数字”，而是：
   - 在内部 `DeepDive` 和外部 `BrowseComp subset256` 上都得到正向结果
   - 并且能用 unfinished / response_limit / finished-to-correct 解释行为变化。
3. 你当前最明显还没补上的，是论文最强的两块证据：
   - `C-GRPO vs GRPO` 的直接算法比较
   - `64k -> 128k` 的 scaling 结果

如果后续还要继续补实验，优先级应该是：

1. `128k BrowseComp` 或更宽松 context/budget 下的 external eval
2. 复刻论文 Table 2 风格的 cited pages / rubric-connectivity 分析
3. 如果预算允许，再补一个纯 `GRPO` 对照

---

## 10. 一句话总结

论文表明，真正强的 deep-search RL 不只是“64k 内答对率更高”，而是“在更长 context 和更多 tool budget 下仍能保持完整证据链的搜索策略”；你当前结果已经复现了“RL 相对 SFT 的方向性收益”和“response-limit 是核心瓶颈”这两点，但还没有完整复现论文关于 `C-GRPO > GRPO`、`128k scaling`、以及“更多 cited evidence / stronger rubric connectivity”的强结论。

---

## 11. 参考路径

- `CaRR/2601.06021v1.pdf`
- `examples/carr_deepsearch/docs/Carr_paper_data.md`
- `examples/carr_deepsearch/docs/eval_result_deep_analysis.md`
- `examples/carr_deepsearch/docs/eval_result_deep_analysis_followup.md`
- `examples/carr_deepsearch/docs/eval_analysis_20260415.md`
- `examples/carr_deepsearch/docs/training_full_history_20260404.md`
- `examples/carr_deepsearch/tools/carr_agent_loop.py`
- `examples/carr_deepsearch/scripts/run_eval_integration.sh`
