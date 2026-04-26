# CaRR DeepSearch Eval Analysis (2026-04-15)

## 1. 结论先行

这轮 GPU 收尾已经拿到了足够支撑简历叙事的真实结果。

- `async23` 在统一 sampled recipe 下的 `BrowseComp subset256 64k` 外部评测中，真实结果优于 `SFT`
- `async23` 在 `DeepDive rl_val 111` 内部 anchor 上也显著优于 `SFT`
- 因此当前不再需要为简历目的补跑 `step70` fallback、`BrowseComp full` 或 `128k`
- 下一步的最高 ROI 已经不是继续烧 GPU，而是更新 source-of-truth 文档、抽 case study、把真实指标替换掉 placeholder

当前推荐的项目收尾判断：

1. 把 `BrowseComp subset256 64k` 的真实结果写进内部文档与项目介绍
2. 不把这组数字包装成“论文复现”或“大幅超越 benchmark”
3. 把对外叙事固定为：
   - `verl + CaRR + async RL infra` 系统工程成立
   - 内部 `DeepDive` 和外部 `BrowseComp subset256` 都给出了正向证据
   - 外部 benchmark 目前是 `subset256`，不是 full benchmark

---

## 2. 本轮实际完成的评测

### 2.1 统一评测口径

这轮正式评测统一使用 sampled recipe，而不是 greedy：

- `temperature=0.6`
- `top_p=0.95`
- `top_k=20`
- `do_sample=true`
- `max_response_length=61440`
- `max_assistant_turns=120`
- `max_tool_response_length=6000`

`BrowseComp subset256` 的正式外部 gate 对 `SFT` 和 `async23` 使用同一套 relaxed-but-bounded eval envelope：

- `max_rollout_wall_time_s=600`
- `max_real_rollout_wall_time_s=1200`
- `max_tool_calls=160`
- `max_search_calls=80`
- `max_open_calls=60`
- `max_find_calls=40`

这意味着本轮 `SFT vs async23` 的外部比较是 matched eval，不是预算不一致导致的假提升。

### 2.2 已完成的评测项

| Eval | 目的 | 状态 |
|------|------|------|
| `async23_dd64_sanity` | 确认最终 async checkpoint 在 cheap internal gate 上不过线不会崩 | Completed |
| `dd111_sft` | 给 `DeepDive rl_val 111` 提供内部 anchor baseline | Completed |
| `dd111_async23` | 检查最终 async checkpoint 是否在较大 internal set 上优于 `SFT` | Completed |
| `bc256_sft_relaxed_v2` | `BrowseComp subset256 64k` 外部 baseline | Completed |
| `bc256_async23_relaxed_v2` | `BrowseComp subset256 64k` 外部 async gate | Completed |

---

## 3. 关键结果

### 3.1 Internal sanity: `async23_dd64_sanity`

`DeepDive subset64 sampled` 上，`async23` 的 sanity 结果是正向的：

| Metric | Value |
|------|------:|
| `outcome_reward` | `0.3750` |
| `rubric_reward` | `0.0960` |
| `task_unfinished` | `0.5625` |
| `termination_response_limit` | `0.3438` |
| `termination_rollout_timeout` | `0.0000` |
| `tool_call_counts` | `45.86` |
| `search_count` | `20.28` |
| `open_count` | `16.17` |
| `find_count` | `9.41` |
| `response_length` | `48987.20` |
| `num_turns` | `93.13` |

这个结果的意义不是“证明 external uplift”，而是证明最终 durable async checkpoint 不是坏 checkpoint。

### 3.2 Internal anchor: `DeepDive rl_val 111`

这是目前最强、最干净的内部真实效果证据。

| Metric | `SFT` | `async23` | Delta |
|------|------:|------:|------:|
| `outcome_reward` | `0.1802` | `0.3243` | `+0.1441` |
| `rubric_reward` | `0.0332` | `0.0710` | `+0.0378` |
| `task_unfinished` | `0.7297` | `0.6306` | `-0.0991` |
| `unfinished_limit` | `0.5135` | `0.3784` | `-0.1351` |
| `unfinished_budget` | `0.2162` | `0.2523` | `+0.0360` |
| `completion_finished_early_stop` | `0.1892` | `0.2613` | `+0.0721` |
| `completion_finished_natural` | `0.0811` | `0.1081` | `+0.0270` |
| `tool_call_counts` | `47.99` | `47.20` | `-0.79` |
| `search_count` | `22.16` | `23.48` | `+1.32` |
| `open_count` | `15.85` | `14.49` | `-1.36` |
| `find_count` | `9.98` | `9.23` | `-0.75` |
| `parse_error_count` | `0.0631` | `0.0450` | `-0.0180` |
| `termination_response_limit` | `0.5135` | `0.3784` | `-0.1351` |
| `termination_rollout_timeout` | `0.0090` | `0.0000` | `-0.0090` |
| `response_length` | `52840.86` | `50876.69` | `-1964.16` |
| `num_turns` | `97.16` | `95.80` | `-1.36` |

解读：

- 内部 anchor 上，`async23` 不是小幅抖动，而是 `outcome` 和 `rubric` 都明显提升
- 最大的正向变化来自：
  - 更低的 `unfinished`
  - 更低的 `response_limit`
  - `rollout_timeout` 归零
- 工具调用总量没有显著增加，说明提升不是靠“更暴力地乱搜”，而更像是更有效的搜索分配

### 3.3 External gate: `BrowseComp subset256 64k`

这是当前最重要的真实外部结果。

#### Source-of-truth judge metrics

以下 judge 结果以 reward trace 为准：

| Metric | `SFT` | `async23` | Delta |
|------|------:|------:|------:|
| `samples` | `256` | `256` | — |
| `outcome_mean` | `0.015625` | `0.03515625` | `+0.01953125` |
| `outcome_pos` | `4` | `9` | `+5` |
| `finished` | `11` | `17` | `+6` |
| `unfinished` | `245` | `239` | `-6` |
| `finished_rate` | `0.04297` | `0.06641` | `+0.02344` |

等价地说：

- `BrowseComp subset256` 上，`async23` 的 judge-pass 样本数从 `4/256` 提升到 `9/256`
- 相对 `SFT` 是 `2.25x` 的通过数
- 同时完成回答数从 `11` 提升到 `17`

#### Directional behavior metrics from generation dump

以下指标来自 `0.jsonl` generation dump，重点用于行为解释而不是 judge 结论。

`SFT` 的 dump 在逐行解析时出现了 `2` 个坏行，因此行为均值只做方向性参考；judge 结论仍以上面的 reward trace 为准。

| Metric | `SFT` | `async23` | Direction |
|------|------:|------:|------|
| parsed rows | `255` | `256` | — |
| `tool_call_counts` | `57.03` | `55.48` | slight down |
| `search_count` | `32.80` | `33.29` | slight up |
| `open_count` | `13.74` | `12.66` | down |
| `find_count` | `10.49` | `9.53` | down |
| `parse_error_count` | `0.1255` | `0.1211` | slight down |
| `termination_response_limit` | `0.9020` | `0.8711` | down |
| `termination_rollout_timeout` | `0.0118` | `0.0078` | down |
| `termination_search_budget` | `0.0392` | `0.0508` | slight up |
| `response_length` | `59527.94` | `59412.62` | flat |
| `response_length_ratio` | `0.9689` | `0.9670` | flat |
| `rollout_elapsed_s` | `323.81` | `299.10` | down |

外部集上的核心观察：

- `async23` 的提升并不是来自明显更大的总 tool volume
- 更像是：
  - 略多的 `search`
  - 更少的 `open/find`
  - 更低的 `response_limit`
  - 更短的平均 rollout 时间
- 这说明 RL 后行为更可能是“更有选择地查”和“更早形成可提交答案”，而不是简单把轨迹拉得更长

### 3.4 BrowseComp 的一个重要说明

这轮 `BrowseComp` 结果里，`rubric_reward=0`，因为外部集不走 CaRR rubric。

因此这轮外部 uplift 反映的是：

- binary final-answer correctness
- completion behavior
- termination pattern

不是因为在外部集里“刷 rubric 分”得到的假收益。

---

## 4. 能否写进简历

### 4.1 可以写的结论

当前已经可以把 placeholder 换成真实外部结果，但写法需要克制。

最稳的写法不是“显著提升 benchmark”，而是：

1. `async23` 在 matched sampled `BrowseComp subset256 64k` 外部评测中优于 `SFT`
2. `async23` 在 `DeepDive rl_val 111` 内部 anchor 上也显著优于 `SFT`
3. 外部结果的改善主要来自更多样本从 `response_limit/unfinished` 转成 finished answer，而不是简单增加工具调用总量

### 4.2 推荐写法

#### 中文技术版

在统一 sampled eval 下，最终 async checkpoint 在 `DeepDive rl_val 111` 上将 `outcome_reward` 从 `0.180` 提升到 `0.324`，并在 `BrowseComp subset256 64k` 外部评测上将 judge-pass 样本数从 `4/256` 提升到 `9/256`，完成回答数从 `11` 提升到 `17`。

#### English technical version

Under a matched sampled eval setup, the final async checkpoint improved `outcome_reward` on `DeepDive rl_val (111)` from `0.180` to `0.324`, and increased `BrowseComp subset256 64k` judge-pass samples from `4/256` to `9/256`, with finished answers rising from `11` to `17`.

### 4.3 不建议直接写进简历 headline 的内容

- 不要写“复现了 CaRR 论文结果”
- 不要把 `BrowseComp subset256` 写成 full benchmark
- 不要只报 `9/256` 而不解释这是 `subset256 sampled external gate`
- 不要把这轮外部结果包装成单纯“更多 tool calls 带来的提升”，因为数据不支持这个说法

---

## 5. 还需不需要继续烧 GPU

### 5.1 当前判断

不需要再补 mandatory GPU eval。

原因：

- 外部 `BrowseComp subset256` 已经给出了真实正结果
- 内部 `dd111` anchor 也强正
- `step70` fallback 的唯一价值，是在 `async23` 外部打平或打输时证明 RL 信号存在
- 现在 `async23` 已经明确优于 `SFT`，因此 `step70` 不是当前简历收尾的必需项

### 5.2 为什么不建议现在再开 `BrowseComp full`

`BrowseComp full 1266` 的价值是更低方差的外部数字，但它不是当前简历闭环的必要条件。

以当前结果看：

- 项目已经从“只有系统工程故事”升级到“系统工程 + 外部真实正证据”
- full benchmark 只会让这个证据更稳，不会改变项目是否成立
- 在实例可能被回收的前提下，更高 ROI 的动作是：
  - 先同步日志到本地
  - 更新简历 source-of-truth
  - 提取 case study

### 5.3 如果后面还想再补一轮

唯一值得补的额外 GPU 任务是：

1. `BrowseComp full 64k`: `SFT vs async23`

不建议优先补：

- `step70` fallback
- `BrowseComp subset256 128k`
- continuation 再训练

除非你后续的目标从“简历收尾”切换成“项目报告/博客/更低方差 benchmark”。

---

## 6. Case Study 候选

以下 case 都来自 `async23 win / SFT lose` 的真实外部样本，可作为后续项目介绍或面试手卡候选。

### 6.1 候选 A: Rwanda literacy / critically endangered mammal

- `async23`: `score=1`, `finished=True`, `tool=40`, `search=6`, `open=20`, `find=14`, `resp=56008`
- `SFT`: `score=0`, `finished=False`, `termination=response_limit`, `tool=59`, `search=16`, `open=25`, `find=18`, `resp=60913`

适合讲的点：

- `async23` 用更少的总工具调用完成了答案
- `SFT` 被拖进了长轨迹 response-limit
- 这是“更有效搜索”而不是“更长搜索”的典型例子

### 6.2 候选 B: interviewer / personal archive privilege

- `async23`: `score=1`, `finished=True`, `tool=30`, `search=12`, `open=8`, `find=10`, `resp=26214`
- `SFT`: `score=0`, `finished=False`, `termination=response_limit`, `tool=62`, `search=38`, `open=15`, `find=9`, `resp=61577`

适合讲的点：

- `async23` 在更短回答长度下完成任务
- `SFT` 明显存在 search 过量、收敛不足的问题

### 6.3 候选 C: recipe developer / blog traffic source

- `async23`: `score=1`, `finished=True`, `tool=37`, `search=13`, `open=9`, `find=15`, `resp=34854`
- `SFT`: `score=0`, `finished=False`, `termination=response_limit`, `tool=71`, `search=43`, `open=22`, `find=6`, `resp=59390`

适合讲的点：

- async RL 后的 agent 更快锁定高价值 source
- SFT 则表现出更明显的“重复搜索 + 拉长轨迹”倾向

### 6.4 候选 D: 仍然失败但很有代表性的 hard case

示例问题：

- `A dental cleaner was introduced in the mid-1810s ...`

`async23` 在这个样本上仍然失败：

- `termination=response_limit`
- `tool=108`
- `search=65`
- `open=18`
- `find=25`

适合讲的点：

- 当前主瓶颈已经不是 infra 不可用，而是少数 hard case 仍然会把 agent 拖进 `response_limit`
- 如果后续还做 continuation 或 128k，最优先要攻的就是这类长尾 hard case

---

## 7. 对简历故事的最终判断

当前项目已经可以稳定讲成下面这个版本：

1. 你把 `CaRR` 的 reward design 和 `C-GRPO` 接进了 `verl`
2. 你把 `fully_async_policy` 从容易被长轨迹拖死的状态，推进到可训练、可恢复、可评测的主线
3. 你不仅有内部 gate，而且现在已经拿到了真实外部正结果

因此简历上最合理的组合是：

- 用系统工程 bullet 当 headline
- 用内部 `dd111` + 外部 `BrowseComp subset256` 的真实数字当 supporting evidence
- 不再使用 paper placeholder 当外部结果占位

---

## 8. 当前建议的后续动作

1. 更新 [resume_source_of_truth_20260412.md](/Users/xiaohui/Desktop/agentic-RL-project/verl-carr-deepsearch/examples/carr_deepsearch/docs/resume_source_of_truth_20260412.md)，把 `BrowseComp` placeholder 换成真实结果。
2. 更新 [final_resume_ready.md](/Users/xiaohui/Desktop/agentic-RL-project/verl-carr-deepsearch/examples/carr_deepsearch/docs/final_resume_ready.md)，把第三条量化 evidence 改成真实 external gate。
3. 从本轮 `0.jsonl` 中精修 2-3 个 case study。
4. 如果后续还有 GPU 预算，再考虑 `BrowseComp full 64k`，不是现在的必需项。

---

## 9. 本轮产物位置

### 9.1 Remote artifacts

- `/root/logs/bc256_sft_relaxed_v2*`
- `/root/logs/bc256_async23_relaxed_v2*`
- `/root/logs/dd111_sft*`
- `/root/logs/dd111_async23*`
- `/root/logs/async23_dd64_sanity*`
- `/root/eval_results/bc256_sft_relaxed_v2/0.jsonl`
- `/root/eval_results/bc256_async23_relaxed_v2/0.jsonl`

### 9.2 Local archive

本轮关键日志和 generation dump 已归档到：

- `examples/carr_deepsearch/CaRR_log/20260415_gpu_eval/`

当前本地已经包含：

- `bc256_sft_relaxed_v2*`
- `bc256_async23_relaxed_v2*`
- `dd111_sft*`
- `dd111_async23*`
- `async23_dd64_sanity*`
- `eval_results/bc256_sft_relaxed_v2/0.jsonl`
- `eval_results/bc256_async23_relaxed_v2/0.jsonl`

因此后续即使停掉实例，也可以继续在本地完成 case study 抽取与文档回填。
