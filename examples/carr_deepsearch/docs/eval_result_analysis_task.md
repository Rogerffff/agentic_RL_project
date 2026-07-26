# Eval Result 深度分析任务书 (v2)

## 1. 目标

对本轮 GPU 评测产出的 generation dump 进行深度分析，产出可直接用于简历 bullets 编写和面试 Q&A 准备的素材。

**分析原则**：
- 先验证假设再下结论，不预设正向结果
- 所有统计必须报告分子/分母/样本数，小样本（n<20）必须标注 caveat
- 同时分析增益和代价（regression），不只讲正面
- 每个 case study 必须带 `dataset + row_index + gts + outcome_pair + termination_reason`，便于追溯

---

## 2. 数据源

所有数据在本地 `examples/carr_deepsearch/eval_results/` 下，每个目录包含一个 `0.jsonl`，每行是一个样本的完整评测记录。

| 文件 | 模型 | 评测集 | 样本数 | 大小 |
|------|------|--------|--------|------|
| `dd111_sft/0.jsonl` | SFT | DeepDive rl_val 111 | 111 | 20MB |
| `dd111_async23/0.jsonl` | async23 (RL) | DeepDive rl_val 111 | 111 | 19MB |
| `bc256_sft_relaxed_v2/0.jsonl` | SFT | BrowseComp subset256 | 256 | 50MB |
| `bc256_async23_relaxed_v2/0.jsonl` | async23 (RL) | BrowseComp subset256 | 256 | 49MB |

**已确认**：dd111 和 bc256 的 SFT / async23 两组 0.jsonl 行序是对齐的（input 和 gts 完全一致），可安全按行号对齐。

### 每行 jsonl 的字段

```
input: str          — 输入 prompt（含系统 prompt 和用户问题）
output: str         — 模型完整输出（含所有 thinking、tool_call、tool_response 轮次）
gts: str            — ground truth 答案
score: float        — 最终得分
outcome_reward: float — binary 0/1 最终答案是否正确
rubric_reward: float  — rubric 过程质量分数 (DeepDive 有值, BrowseComp 为 0)
task_unfinished: bool — 是否未完成任务
termination_reason: str — 截断原因
tool_call_counts: int — 总工具调用次数
search_count: int    — search 调用次数
open_count: int      — open 调用次数
find_count: int      — find 调用次数
response_length: float — 回复 token 数
rollout_elapsed_s: float — rollout 耗时（秒），eval 时为主时间字段
active_rollout_elapsed_s: float — 活跃 rollout 时间（eval 时 == rollout_elapsed_s）
real_rollout_elapsed_s: float — 真实墙钟时间（eval 时无 pause，固定为 0.0）
termination_rollout_timeout: float — 是否因 rollout_elapsed_s 超 wall_time 被截断（0.0/1.0）
termination_real_rollout_timeout: float — 是否因 real wall 超时被截断（eval 时固定 0.0）
completion_finished_early_stop: bool — 模型主动提前终止
completion_finished_natural: bool — 自然完成（生成 EOS）
content_early_stopped: bool — 内容层面提前终止
unfinished_limit: bool — 因 response_limit 未完成
unfinished_budget: bool — 因 budget 未完成
parse_error_count: int — 工具调用解析错误次数
response_length_max: float — 最大允许 response 长度
response_length_ratio: float — response_length / response_length_max
```

---

## 3. 分析任务清单

### 3.0 Headline Scoreboard（必须最先完成）

**目标**：为所有后续分析提供全局数据基础，直接决定简历 bullet 怎么写。

**具体任务**：

对 `dd111` 和 `bc256` 各自产出一张 scoreboard：

1. **四象限矩阵**（paired，按行号对齐）：
   - `00`：SFT 错 + RL 错（两个都失败）
   - `01`：SFT 错 + RL 对（RL 增量正确 — 增益样本）
   - `10`：SFT 对 + RL 错（RL regression — 代价样本）
   - `11`：SFT 对 + RL 对（两个都成功）
   - 每个象限报告样本数和百分比

2. **汇总指标**：
   - outcome_reward 均值
   - rubric_reward 均值（仅 dd111）
   - task_unfinished 比例
   - finished 比例（= 1 - task_unfinished）
   - completion_finished_early_stop 比例
   - completion_finished_natural 比例

3. **Termination 分布**：
   - response_limit / search_budget / open_budget / find_budget / rollout_timeout / tool_call_budget 各自比例
   - 自然完成 + early_stop 的合计比例

**期望产出**：两张 scoreboard 表（dd111 + bc256），直接作为后续所有分析的引用基础

---

### 3.1 搜索效率对比分析

**待验证假设**：RL 模型的效果提升是否来自更高效的搜索策略，而非简单增加搜索量？

**具体任务**：

对 `dd111` 和 `bc256` 分别：
- 计算 SFT 和 async23 的 tool_call_counts / search_count / open_count / find_count 的均值、中位数
- 分别对 finished 和 unfinished 样本统计上述指标
- 对 outcome_reward=1 的样本统计工具调用均值（注意：bc256 中 outcome=1 的样本数很少，必须标注 n=?）
- 计算 finished-to-correct conversion rate：outcome=1 的样本数 / finished 样本数
- 计算总体答对率：outcome=1 的样本数 / 总样本数

**期望产出**：对比表 + 基于数据的结论（可能支持假设，也可能不支持）

---

### 3.2 Termination Reason 分布迁移分析

**待验证假设**：SFT → RL 后，样本的截断原因分布是否发生了有意义的变化？

**具体任务**：

对 `dd111` 和 `bc256` 分别：
- 统计每种 termination_reason 的样本数和百分比
- 计算 unfinished_limit vs unfinished_budget 的比例变化
- 分析：response_limit 截断减少的部分转移到了哪里？（自然完成？early_stop？还是其他 budget？）
- 注意区分 `completion_finished_early_stop` 和 `completion_finished_natural` 的含义差异

**期望产出**：termination 分布对比表 + 迁移方向分析

---

### 3.3 Final-Answer Delivery 与评测质量审计

**目标**：分析模型是否真正交出了最终答案，以及"未完成"是否包含假阴性。这对解释 BrowseComp 的高 unfinished 率至关重要。

**具体任务**：

**关键规则**：所有 final-answer / near-miss 审计一律基于 **last_assistant_chunk**（output 中最后一个 `<|im_start|>assistant` 到 `<|im_end|>` 或文件末尾的片段），禁止在整条 output 上直接搜索 `## Exact Answer`——因为中途 thinking 草稿中可能出现该标记但并非最终提交。

1. **Finished but score=0 审计**：
   - 找出所有 `task_unfinished=False` 且 `outcome_reward=0` 的样本
   - 提取每个样本的 last_assistant_chunk，分为以下 4 类：
     - **A. 真正交了答案但答错**：last_assistant_chunk 包含 `## Exact Answer` 且有明确答案文本，只是答案与 gts 不匹配
     - **B. 没有交最终答案**：last_assistant_chunk 停在 planning / tool_call / 半成品 explanation，没有 `## Exact Answer` 段落
     - **C. 交了答案但格式不合规**：last_assistant_chunk 有类似答案的内容但不符合 `## Explanation with Citations` + `## Exact Answer` 的标准格式
     - **D. 高疑似 false negative**：last_assistant_chunk 格式正确、答案文本与 gts 语义接近但被 judge 判错（如大小写/别名/单位差异）
   - 报告每类的样本数，并各挑 1 个典型案例

2. **Unfinished near-miss 审计**：
   - 找出所有 `task_unfinished=True` 且 `response_length_ratio > 0.95` 的样本
   - 提取每个样本的 last_assistant_chunk，判断：
     - **Near-miss**：last_assistant_chunk 已经开始写 `## Explanation` 或 `## Exact Answer`，被 response_limit 截断在交卷过程中
     - **Still searching**：last_assistant_chunk 仍然是 thinking / tool_call，还在搜索中被截断
   - 报告 near-miss 样本数，估算放宽 response_limit 后可能额外完成的样本数

3. **content_early_stopped 分析**：
   - 统计 `content_early_stopped=True` 的样本数和 outcome 分布
   - 这些样本是否倾向于更高的 outcome？（提前终止可能意味着模型有信心直接给答案）

**期望产出**：
- finished-but-wrong 的 4 类分布（A/B/C/D）
- near-miss 样本数和"假阴性"估算
- content_early_stopped 的效果分析

---

### 3.4 Bad Case 对比分析：增益样本（SFT 败 → RL 胜）

**目标**：找出高质量的对比案例，面试时可以展开讲。

**具体任务**：

从 3.0 的四象限中取 `01`（SFT 错 + RL 对）的所有样本：

对每个样本提取：
- `dataset`：dd111 或 bc256
- `row_index`：行号
- `gts`：ground truth
- `outcome_pair`：SFT outcome / RL outcome
- 问题摘要（从 input 提取用户问题，1-2 句）
- SFT 的 termination_reason、tool_call_counts、search/open/find、response_length
- async23 的同上指标
- SFT 的搜索行为摘要：从 output 提取前 3 次 search query 和最后 500 字符
- async23 的搜索行为摘要：同上

对每个 case 标注故事角度（可多选）：
- "更高效搜索"：RL 用更少工具调用完成
- "更好的搜索策略"：RL 的 search query 更精准/更有针对性
- "更快收敛"：RL 在更短 response_length 内完成
- "避免了无效循环"：SFT 重复搜索同类 query，RL 没有
- "更好的答案提交"：RL 更快决定停止搜索并提交答案

**期望产出**：全部 01 样本的列表 + 挑选 3-5 个最佳 case study 卡片

---

### 3.5 Regression Case 分析：代价样本（SFT 胜 → RL 败）

**目标**：找出 RL 训练带来的 regression，理解 tradeoff，面试被追问时不会穿帮。

**具体任务**：

从 3.0 的四象限中取 `10`（SFT 对 + RL 错）的所有样本：

对每个样本提取（格式同 3.4）：
- dataset、row_index、gts、outcome_pair
- 问题摘要
- SFT 和 async23 的指标对比
- SFT 和 async23 的搜索行为摘要

分析每个 regression 的原因（可多选）：
- RL 模型搜索策略变化导致错过关键信息
- RL 模型过早终止搜索（early_stop 但答错）
- RL 模型遇到了新的 budget 截断
- 随机性（sampled eval 的方差）
- 其他

**期望产出**：
- 全部 10 样本的列表（可能很少，也可能不少）
- net gain = 01 样本数 - 10 样本数
- 1-2 个典型 regression case 的详细分析
- 面试口径：如何解释 RL 训练的 tradeoff

---

### 3.6 Hard Case 分析：RL 仍然失败的典型样本

**目标**：展示诚实和技术判断力——当前模型的局限在哪里。

**具体任务**：

按 failure mode 分层抽样（不只按 tool_call_counts 排序）：

1. **response_limit 截断 + near-miss**：从 3.3 的 near-miss 中挑 1 个最典型的
2. **search_budget 截断**：找 1 个因 search_budget 达上限而失败的样本
3. **finished but wrong**：找 1 个自然完成但答案错误的样本（搜索方向错误）
4. **极长轨迹失败**：找 1 个 tool_call_counts 最高的失败样本

每个样本提取：dataset、row_index、gts、termination_reason、tool_call_counts、response_length、搜索行为摘要。

分析"如果给更多 budget/context 是否可能成功"的判断。

**期望产出**：3-4 个 failure case 卡片 + 局限性分析

---

### 3.7 轨迹长度与质量的关系分析

**待验证假设**：RL 模型是否在更短的轨迹中就能完成任务？

**具体任务**：

对 `dd111` 的 SFT 和 async23：
- 按 response_length 分桶（0-20k, 20k-40k, 40k-55k, 55k-61.4k），统计每桶的样本数和 outcome_reward 均值
- 分别对 finished 和 unfinished 样本统计 response_length 均值
- outcome=1 样本的平均 response_length（SFT vs async23）
- response_length_ratio 的分布对比

对 `bc256` 做同样分析（注意 outcome=1 样本数少，标注 n）。

**期望产出**：分桶对比表 + 基于数据的结论

---

### 3.8 Timeout 与延迟分析

**目标**：分析 eval 时的 rollout 时间分布，确认 relaxed budget 是否足够。

**主时间字段**：使用 `rollout_elapsed_s`。eval 时 `active_rollout_elapsed_s == rollout_elapsed_s`，`real_rollout_elapsed_s` 固定为 0.0（eval 无 pause/resume 机制）。

**具体任务**：

对所有 4 个 eval：
- 统计 rollout_elapsed_s 的均值、中位数、P90、P99、最大值
- finished vs unfinished 样本的时间分布差异
- 统计 termination_rollout_timeout 的比例（不预设为零，如实报告）
- 最慢的 5 个样本：row_index、termination_reason、tool_call_counts、rollout_elapsed_s
- 判断：timeout 比例是否足够低以至于不影响结论？

**期望产出**：时间分布表 + timeout 影响判断

---

### 3.9 Rubric Reward 深度分析（仅 DeepDive）

**目标**：分析 RL 训练后搜索过程质量的变化。

**具体任务**：

对 `dd111`（BrowseComp 没有 rubric，跳过）：
- rubric_reward 的均值、中位数、非零样本数和比例
- 对 outcome=1 的样本：rubric_reward 均值（SFT vs async23）——标注 n
- 对 outcome=0 的样本：rubric_reward 均值
- rubric_reward > 0 的样本中，outcome=1 vs outcome=0 的比例

**期望产出**：rubric 分布对比 + "正确回答的过程质量"是否也提升了

---

### 3.10 Parse Error 分析

**目标**：确认工具调用格式正确率。

**具体任务**：
- 所有 4 个 eval 的 parse_error_count 分布（均值、>0 的样本比例）
- parse_error > 0 的样本的 outcome 均值 vs parse_error=0 的样本的 outcome 均值
- 是否有 parse_error 导致轨迹提前中断的情况？

**期望产出**：一句话结论 + 相关性判断

---

### 3.11 跨评测集一致性验证

**目标**：验证内部 DeepDive 和外部 BrowseComp 的结论方向是否一致。

**具体任务**：
- 列出所有 SFT→RL 的指标变化方向（dd111 vs bc256）
- 标注方向一致 / 不一致的指标
- 分析不一致的原因（难度差异、rubric 有无、样本量差异等）

**期望产出**：一致性对比表 + 可信度判断

---

## 4. 输出格式要求

### 4.1 每个分析任务产出

1. **数据表格**：具体数字，所有比例同时报告 分子/分母
2. **小样本标注**：当 n<20 时，必须在数字旁标注 `(n=X, 小样本 caveat)`
3. **基于数据的结论**：先陈述数据，再给出判断，不预设方向
4. **面试展开版**：2-3 句话的说明，适合面试追问时使用

### 4.2 Case Study 卡片格式（3.4 / 3.5 / 3.6）

每个 case 必须包含：

```
Dataset: dd111 / bc256
Row Index: N
Question: [1-2句摘要]
Ground Truth: [gts]
Outcome Pair: SFT=X / RL=Y
Termination: SFT=[reason] / RL=[reason]

SFT: tool_calls=N, search=N, open=N, find=N, resp_len=N, elapsed=Ns
RL:  tool_calls=N, search=N, open=N, find=N, resp_len=N, elapsed=Ns

SFT 搜索行为: [前3次search query + 最后500字符摘要]
RL  搜索行为: [前3次search query + 最后500字符摘要]

故事角度: [标注]
```

### 4.3 输出文件

所有分析结果写入：`examples/carr_deepsearch/docs/eval_result_deep_analysis.md`

---

## 5. 分析注意事项

- BrowseComp 的 rubric_reward 全部为 0（没有 rubric 标注），不要分析 bc256 的 rubric
- output 字段非常长（10-100k 字符），提取 search query 时只需取前 3 次和最后 500 字符
- dd111 的 SFT 和 async23 按行号对齐（已验证）
- bc256 同理（已验证）
- BrowseComp 的 outcome=1 样本很少（SFT 4 个、RL 9 个），所有基于正确样本的统计必须标注 n 值
- eval 时使用的主时间字段为 `rollout_elapsed_s`（eval 无 pause，active 和 real 与之相同或为 0）
- 不要预设"RL 一定更好"——如果数据不支持某个假设，如实报告
