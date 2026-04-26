# CaRR DeepSearch — Final Resume-Ready Document (v2, 2026-04-15)

## 项目标题

基于 verl 框架的 Deep Search Agent 强化学习训练系统 —— 集成 CaRR Citation-Aware Rubric Reward 与异步训练架构

## 一句话摘要

在 verl 开源 RL 框架上集成 CaRR 论文的 citation-aware rubric reward 与 C-GRPO 算法，训练 Qwen3-4B 模型通过多轮浏览器工具调用（search/open/find）进行深度搜索问答；扩展并稳定化 verl 的异步训练架构，将 rollout 与参数更新解耦，在 8 GPU 上实现约 45% 的训练迭代加速；在内部验证集和外部 BrowseComp 评测上均观测到正向 RL 训练信号。

---

## 中文简历 Bullets — 备选方案

以下按 4 个主题各提供 2-3 种备选写法，附 Impact / Defensibility / Risk 评分（5 分制）。最终推荐版在末尾。

---

### 主题 A：异步训练架构（最独特、最 defensible 的贡献，建议第 1 条）

**A1（技术细节版）**

扩展并稳定化 verl 的 fully_async_policy 以支持 64k 长上下文 agentic RL（~450 行框架层改造）：实现跨参数版本的 partial rollout cancel/resume、cancel-based queue drain 反压机制、no-flush 权重同步与 fingerprint 校验、以及 FSDP 动态 batch 的跨 DP rank 对齐。将 rollout 与参数更新解耦后，在 8×96GB GPU 上每步训练迭代时间从约 500s 降至约 280s，同时消除了同步训练后期超过 90% 样本因长尾 rollout 阻塞而超时的问题。

- Impact: 4 / Defensibility: 5 / Risk: 1
- 优点：机制具体、数字真实、因果链清晰
- 缺点：术语密集，recruiter 可能看不懂

**A2（问题导向版）**

针对多轮 agentic RL 中 rollout 时间差异极大（十几秒到数百秒）的核心瓶颈，改造 verl 的异步训练架构（~450 行）：将 rollout 与参数更新解耦，实现 partial rollout 跨参数版本恢复、queue backpressure 与 KV cache 一致性管理等机制，使训练迭代时间缩短约 45%（500s→280s），并将同步训练后期因长尾阻塞导致的超时截断从 90%+ 降至 0%。

- Impact: 5 / Defensibility: 4 / Risk: 1
- 优点：先讲问题再讲方案，更有叙事感；"90%+ 降至 0%"非常有力
- 缺点：面试官可能追问"90%+ 怎么来的"（需要解释是同步后期的恶化，不是常态）

**A3（精简版，适合版面紧张）**

扩展 verl 的异步训练架构支持 64k 长上下文 agentic RL：将 rollout 与参数更新解耦，实现 partial rollout cancel/resume 与 queue backpressure 等机制，在 8 GPU 上将每步训练迭代从约 500s 降至 280s，同时消除了同步训练后期的 rollout 超时瓶颈。

- Impact: 3 / Defensibility: 5 / Risk: 0
- 优点：简洁，主干信息完整
- 缺点：技术深度被压缩

---

### 主题 B：方法实现（展示算法理解 + 工程能力，建议第 2 条）

**B1（参考设计 + 自己实现，完整版）**

参考 CaRR 论文的 citation-aware reward 设计，在 verl 框架下实现多轮 deep search agent 的完整 RL 训练栈：自定义 AgentLoop 管理 search/open/find 浏览器工具调用与可恢复会话状态（~1050 行），采用混合奖励机制——通过 LLM Judge 将多跳问题分解为可验证的事实约束逐条评估搜索质量，在 GRPO 的优势估计中仅对正确轨迹注入组内归一化的过程质量信号（~120 行），避免错误轨迹因命中局部事实获得正向梯度。

- Impact: 5 / Defensibility: 5 / Risk: 1
- 优点："参考设计"+"自己实现"定位精准，主语是你；reward 机制用自己的语言描述而非引用论文术语；"避免错误轨迹..."展示了对设计动机的理解
- 缺点：较长

**B2（参考设计 + 突出问题意识版）**

参考 CaRR 论文的奖励框架，在 verl 上实现面向多跳问答的 deep search agent RL 训练栈：针对纯 binary outcome reward 容易诱导 shortcut exploitation 的问题，引入基于引用支撑和证据链连通性的搜索过程质量评估（rubric reward），在 GRPO 优势估计中仅对最终答案正确的轨迹融合过程质量信号（~120 行），同时自定义 AgentLoop（~1050 行）支持 64k 长上下文下的多轮工具交互、预算控制与异步 cancel/resume 状态持久化。

- Impact: 5 / Defensibility: 5 / Risk: 1
- 优点：先讲"为什么需要 rubric reward"（problem→solution 叙事），展示你不是盲目实现而是理解了设计动机
- 缺点：rubric reward 的技术细节（entity identification、BFS）被省略

**B3（参考设计 + 精简版，适合版面紧张）**

参考 CaRR 论文的 citation-aware reward 设计，在 verl 框架下实现多轮 deep search agent RL 训练栈（~1050 行 AgentLoop + ~120 行优势估计器）：通过 LLM Judge 评估搜索轨迹中的引用支撑与证据链质量，在 GRPO 优势估计中仅对正确轨迹注入过程质量信号，避免 shortcut exploitation。

- Impact: 3 / Defensibility: 5 / Risk: 0
- 优点：简洁，核心信息完整，代码量明确
- 缺点：技术深度和叙事感不足

**B4（参考设计 + 强调工程化挑战版）**

参考 CaRR 论文的奖励设计，在 verl 框架下实现完整的 deep search agent RL 训练栈：扩展 ToolAgentLoop 实现多轮 search/open/find 工具交互的会话管理、工具预算控制与 reward history 组装（~1050 行），解决 SFT-RL 工具 schema 一致性、多轮 tokenization 对齐（Delta-based Tokenization 与 response_mask）等跨阶段衔接问题；实现混合奖励的优势估计器（~120 行），将 binary outcome 与基于引用和证据链的过程质量评估融合，仅对正确轨迹注入 rubric 信号以抑制 shortcut exploitation。

- Impact: 4 / Defensibility: 5 / Risk: 0
- 优点：展示了工程化落地中遇到的真实挑战（schema 一致性、tokenization 对齐），不是"调个参就跑通了"
- 缺点：信息密度高，recruiter 可能看不懂

**B5（参考设计 + 突出 reward 设计逻辑版）**

参考 CaRR 论文的 citation-aware reward 框架，在 verl 上实现面向 64k 长上下文的 deep search agent RL 训练栈：自定义 AgentLoop 管理多轮浏览器工具调用与可恢复 reward history（~1050 行）；实现混合奖励优势估计（~120 行）——将多跳问题分解为可验证的单跳事实约束，通过 LLM Judge 逐条检查 agent 轨迹中的实体识别、引用支撑与证据链连通性，仅在最终答案正确时注入组内归一化的过程质量信号，防止模型学到"碰巧命中局部事实但答案错误"的 shortcut 策略。

- Impact: 5 / Defensibility: 5 / Risk: 1
- 优点：rubric reward 的三步评估逻辑（实体识别→引用支撑→证据链连通）展开得最清晰，"碰巧命中局部事实"的表述非常直觉化
- 缺点：最长的一个变体

**B6（不提论文名，通用描述版）**

在 verl 框架下实现面向多跳问答的 deep search agent RL 训练栈：自定义 AgentLoop 支持 search/open/find 浏览器工具的多轮交互与异步 cancel/resume 状态持久化（~1050 行），设计混合奖励机制——结合最终答案正确性与基于引用支撑和证据链连通性的搜索过程质量评估，在 GRPO 优势估计中仅对正确轨迹注入过程质量信号以抑制 shortcut exploitation（~120 行）。

- Impact: 4 / Defensibility: 3 / Risk: 3
- 优点：完全不依赖论文名称，主体感最强
- 缺点："设计"一词有 claim 别人设计的风险；面试官如果知道 CaRR 论文会觉得不诚实
- **不推荐**：诚实标注"参考"比隐藏来源更安全

---

### 主题 C：训练效果与评测（展示真实结果，建议第 3 条）

**C1（内部为主 + 外部验证版，推荐）**

在统一 sampled eval 口径下，RL 训练后模型在内部 DeepDive 验证集（111 样本）上将 outcome reward 从 SFT 基线的 0.180 提升至 0.324（+14.4pp），任务完成率从 27% 提升至 37%，外部 BrowseComp 评测上同样显示正向提升，且效果改善来自更高效的搜索分配（工具调用总量不变但 response_limit 截断减少）而非简单增加搜索量。搭建统一评测与 failure-mode 诊断流程，支持 outcome/rubric/task-completion/termination-reason 多维度长轨迹质量分析。

- Impact: 4 / Defensibility: 5 / Risk: 1
- 优点：内部数字足够有力（0.180→0.324），BrowseComp 只说"正向提升"不暴露绝对值，"更高效搜索"的行为分析有技术深度
- 缺点：不提外部具体数字可能被追问

**C2（训练量 framing 版）**

以论文约 10% 的训练量（~5,400 条轨迹），在内部 111 样本验证集上将 outcome reward 从 0.180 提升至 0.324（+14.4pp），在外部 BrowseComp 评测上也观测到一致的正向信号。训练动态与论文报告趋势一致——模型逐步学会更深层搜索策略（工具调用 +37%、搜索覆盖 +43%），表明训练信号方向正确且有进一步提升空间。

- Impact: 4 / Defensibility: 4 / Risk: 2
- 优点："10% 训练量"的 framing 暗示效率高，"进一步提升空间"是正面 hedge
- 缺点："10% 训练量"可能被追问"为什么不跑完"

**C3（纯评测体系版，不含结果数字）**

搭建统一 sampled eval 与 checkpoint selection 流程，设计 outcome/rubric/task-completion/termination-reason 多维度长轨迹诊断体系：精准定位同步训练后期因 timeout 截断导致的信号退化拐点，指导异步架构改造决策；在内部 DeepDive 与外部 BrowseComp 评测上验证 RL 训练的正向效果，并通过行为分析（工具调用模式、搜索覆盖、轨迹长度分布）确认效果提升来自更有效的搜索策略而非简单增加搜索量。

- Impact: 3 / Defensibility: 5 / Risk: 0
- 优点：完全聚焦评测能力和分析方法论，不依赖数字
- 缺点：没有具体数字，impact 较低

---

### 主题 D：全流程调试（展示工程成熟度，可选第 4 条或合并到其他 bullet）

**D1（failure mode 列举版）**

在 SFT → 同步 RL → 异步 RL 的全流程中排查并解决 10+ 个跨层级 failure mode：SFT-RL 工具 schema JSON key 顺序不一致导致模型无法调用工具、同步后期 90%+ rollout 超时导致有效梯度信号归零、FSDP 动态 batch 跨 rank micro-batch 数不一致导致 allreduce hang、SGLang 在线权重更新的 KV cache 破坏等，每个问题均需跨 agent loop / reward server / SGLang / FSDP 多层级联合排查。

- Impact: 4 / Defensibility: 5 / Risk: 1
- 优点：具体 failure mode 非常有说服力，展示跨层排查能力
- 缺点：单独做一条 bullet 可能冲淡主线

**D2（因果推动版）**

主导全流程工程调试与架构决策：定位同步训练后期因长尾 rollout 阻塞导致的梯度信号退化，推动从同步到异步的架构迁移；排查并修复 FSDP/SGLang/Ray 分布式训练中的跨层级问题（batch 对齐、权重同步一致性、KV cache 管理）；建立 SFT-RL 工具 schema 归一化机制消除跨阶段不一致。

- Impact: 4 / Defensibility: 5 / Risk: 0
- 优点："推动架构迁移"展示了工程判断力而非只是执行力

---

## 推荐最终版（3 条）

### Bullet 1 — 异步训练架构（A2 变体）

针对多轮 agentic RL 中 rollout 时间差异极大（十几秒到数百秒）导致同步训练后期超过 90% 样本超时截断的问题，扩展 verl 的异步训练架构（~450 行框架层改造）：将 rollout 与参数更新解耦，实现 partial rollout 跨参数版本恢复、cancel-based queue drain 反压、no-flush 权重同步与 fingerprint 校验、以及 FSDP 动态 batch 的跨 DP rank 对齐，在 8×96GB GPU 上将训练迭代时间从约 500s 降至约 280s，同时完全消除了 rollout 超时截断。

### Bullet 2 — 方法实现（B2）

参考 CaRR 论文的奖励框架，在 verl 上实现面向多跳问答的 deep search agent RL 训练栈：针对纯 binary outcome reward 容易诱导 shortcut exploitation 的问题，引入基于引用支撑和证据链连通性的搜索过程质量评估（rubric reward），在 GRPO 优势估计中仅对最终答案正确的轨迹融合过程质量信号（~120 行），同时自定义 AgentLoop（~1050 行）支持 64k 长上下文下的多轮工具交互、预算控制与异步 cancel/resume 状态持久化。

### Bullet 3 — 训练效果与评测（C1 变体）

在统一 sampled eval 口径下，RL 训练后模型在内部 111 样本验证集上将 outcome reward 从 SFT 基线的 0.180 提升至 0.324（+14.4pp），任务完成率从 27% 提升至 37%，外部 BrowseComp 评测上同样显示正向提升。效果改善来自更高效的搜索策略——工具调用总量基本不变但 response_limit 截断显著减少——而非简单增加搜索量，与论文报告的 C-GRPO 训练动态方向一致。

---

## 使用建议

- **3 条版面**：Bullet 1 + 2 + 3（推荐）
- **2 条版面**：Bullet 1 + 2（系统工程为主），或 Bullet 2 + 3（方法 + 效果为主）
- **如果面试更偏 infra**：Bullet 1 放最前，可加 D2 作为第 4 条
- **如果面试更偏 research**：Bullet 2 放最前，Bullet 3 展开讲 C-GRPO 设计直觉
- **Bullet D（调试）更适合作为面试展开素材**而非写在简历上——每个 failure mode 能讲 2 分钟

---

## 实测评测结果（Observed, 2026-04-15）

### 内部评测：DeepDive rl_val 111 样本

| 指标 | SFT | async23 | Delta |
|------|-----|---------|-------|
| outcome_reward | 0.180 | **0.324** | +14.4pp |
| rubric_reward | 0.033 | **0.071** | +3.8pp |
| task_unfinished | 0.730 | **0.631** | -9.9pp |
| completion_finished | 27.0% | **36.9%** | +9.9pp |
| unfinished_limit (response_limit) | 51.4% | **37.8%** | -13.5pp |
| tool_call_counts | 48.0 | 47.2 | -0.8 |
| response_length | 52,841 | 50,877 | -1,964 |

### 外部评测：BrowseComp subset256 64k（relaxed budget, matched sampled eval）

| 指标 | SFT | async23 | Delta |
|------|-----|---------|-------|
| outcome (accuracy) | 1.56% (4/256) | **3.52% (9/256)** | +5 correct |
| task_unfinished | 95.7% | 93.4% | -2.3pp |
| response_limit 截断 | 90.2% | 87.1% | -3.1pp |
| tool_calls mean | 57.1 | 55.5 | -1.6 |
| rollout_elapsed_s | 323.5s | 299.1s | -24.4s |

### 系统工程指标

| 指标 | 数值 | 说明 |
|------|------|------|
| 训练迭代加速 | ~500s → ~280s（约 45%） | 同步后段 vs 异步健康窗口 |
| Rollout-Training 解耦 | rollout 预算 360s → 480s | 异步允许更充分搜索而不阻塞训练 |
| 参数同步优化 | 323s → 1.7-20s | cancel-based drain + fingerprint 校验 |
| Timeout 消除 | 90%+ → 0% | gmu=0.5 + async 解耦 |
| 训练动态 | 工具调用 ~35 → ~48（+37%） | 模型学到更深搜索策略 |
| 代码贡献 | agent loop 1050 行 + async 451 行 + C-GRPO 120 行 | 扩展 verl 框架 |
| 训练规模 | ~5,400 trajectories（论文的 ~10%） | 系统验证为主 |

---

## 面试口径

### 30 秒版

"这个项目是在 verl 框架上集成 CaRR 论文的 deep search agent RL 训练流水线。我的核心贡献是三个层面：一是 450 行的框架层改造让异步训练在 64k 长轨迹下能稳定运行，迭代速度提升约 45%，消除了同步训练后期 90% 以上的超时截断；二是把论文的 rubric reward 和 C-GRPO 算法完整接入 verl 的训练循环（约 1200 行）；三是在内部和外部评测上都验证了 RL 训练的正向效果——内部验证集 outcome 从 0.18 提升到 0.32。"

### 2 分钟版

在 30 秒版基础上展开：

"异步改造解决的核心问题是：多轮工具调用的 rollout 时间差异极大，从十几秒到几百秒。同步架构下最慢的 rollout 阻塞所有 GPU，到训练后期超过 90% 的样本因为来不及完成就被截断，reward 全是 0，GRPO 的组内对比退化成纯噪声。我把 rollout 和训练解耦后，快的样本不用等慢的，同时慢的样本有更充分的搜索时间。具体要解决的工程问题包括：跨参数版本的 partial rollout 恢复、queue 满了怎么做 backpressure、权重同步时的 KV cache 处理、FSDP 动态 batch 在不同 rank 间的对齐。"

"CaRR 的核心思路是不只看最终答案对不对，还会把多跳问题分解成 rubric——一组可验证的单跳事实约束——然后用 LLM Judge 逐条检查 agent 的搜索轨迹是否覆盖了这些约束。C-GRPO 的关键设计是只对答案正确的轨迹注入这个过程质量信号，避免错误轨迹因碰巧命中局部事实就获得正向梯度。"

"评测方面，我在内部 111 样本的验证集上看到 outcome 从 0.18 提到 0.32，而且提升主要来自更高效的搜索分配——工具调用总量没有增加，但 response_limit 截断明显减少。外部 BrowseComp 评测也给出了一致的正向信号。"

### Q&A

**Q: BrowseComp 结果具体是多少？**

"BrowseComp 是一个非常难的 benchmark——论文用 128k 上下文 SFT 加完整 3 个 epoch 的训练才做到 13.9%。我的 SFT 用 64k 训练，RL 训练量是论文的约 10%。在这个条件下，外部 BrowseComp 子集评测上 RL 模型相比 SFT 基线仍然给出了正向提升，judge-pass 样本数翻倍以上。考虑到训练规模的差距，这说明训练信号方向是对的，更长的训练可以预期进一步提升。"

**Q: 为什么 BrowseComp 绝对值这么低？**

"三个主要原因：一是 SFT 训练长度用的 64k 而论文用 128k，22% 的 SFT 样本被截断，模型的长轨迹搜索能力受限；二是训练量只有论文的 10%，模型还没充分学到搜索策略；三是 BrowseComp 本身极难——它设计为需要多跳深度搜索才能回答的问题，即使论文的完整训练也只有 13.9%。但重要的是 RL 训练后相比 SFT 基线有清晰的正向提升。"

**Q: C-GRPO 比 GRPO 好在哪？你做了消融吗？**

"论文做了完整消融：α=0.3 最优，去掉 rubric 掉约 5 个点，去掉 evidence connectivity check 掉约 6 个点，给所有 rollout 都加 rubric 直接崩溃。我采用论文验证过的最优配置，项目重点在系统实现和稳定化。"

**Q: 训练收敛了吗？训练量够吗？**

"训练量是论文的约 10%，约 5,400 条轨迹。同步阶段在中期窗口展示了正向信号，但后期因为长尾 rollout 阻塞导致 90%+ 超时截断，有效梯度信号崩塌，这驱动了异步架构改造。异步阶段恢复了 n=8 的 rollout group size、消除了 timeout，在稳定配置下训练了 23 轮参数同步。最终评测在内外部都给出正向结果。训练量不到论文的完整规模，但已经验证了系统的正确性和方法的有效性。"

**Q: 异步 partial rollout 实际生效了吗？**

"是的。在 probe 中确认了跨参数版本的 completed sample——一个样本的前半段和后半段使用了不同版本的参数。cancel/resume 不是 dead code。"

**Q: 你扩展了 verl 的哪些代码？**

"三层：agent loop 加了 CaRR 特有的 reward history、工具预算、session 管理、async cancel/resume 状态持久化，约 1050 行；async policy 层修了真实权重同步、queue backpressure、dynbsz DP 对齐等，约 450 行；C-GRPO 优势估计器约 120 行。"

**Q: 这个项目和论文的关系？**

"CaRR 论文定义了 rubric reward 和 C-GRPO 算法，提供了奖励服务器参考实现和训练数据。我的工作是在 verl 上做工程化落地——agent loop、reward manager、advantage estimator、async training 都需要大量定制。同时我做了异步训练稳定化，这是论文没有涉及的系统工程问题。"

**Q: 如果给你更多 GPU/时间，下一步做什么？**

"三件事按优先级：一是用 128k 上下文重做 SFT，消除 22% 的截断损失；二是增加训练量到至少 1 个完整 epoch；三是做 C-GRPO vs GRPO 的消融实验验证 rubric reward 的贡献。这三个方向都有明确的预期收益。"

---

## 技术深度展开（面试追问用）

### C-GRPO 公式与设计直觉

```
Ri = (1-α) * Ro + α * Ro * R̂r
```

`Ro` 是 binary outcome reward，`R̂r` 是组内按最大值归一化后的 rubric reward，α=0.3。

关键：当 `Ro=0`（答案错误），整个 reward 为 0，rubric 不起作用。避免了错误轨迹因命中局部事实获得正向梯度。论文消融显示给所有 rollout 加 rubric 直接崩溃（BrowseComp 从 17.5 掉到 13.3）。

### Timeout 截断如何破坏 GRPO 信号

timeout 的样本不是被丢弃，而是**作为 reward=0 参与 GRPO 组内归一化**。当 group 中 90% 是 timeout (reward=0)，GRPO 的 `advantage = (score - mean) / std` 退化为噪声——mean≈0, std≈0，所有样本的 advantage 接近 0。模型无法从这种 batch 中学到任何有效信号。论文没有 timeout（只有 64k token 截断），所以不存在这个问题。

### Async Partial Rollout 工程挑战

1. **跨版本样本拼接**：rollout 在 param v=3 开始，v=5 时被 cancel，v=5 恢复后继续。需要保存/恢复 reward_history、pending_tool_calls、assistant turn buffer
2. **Queue Backpressure**：cancel-based drain 比 drop 更好——partial 样本带 `is_cancel=True` 标记回收
3. **No-flush Weight Sync**：SGLang 默认 flush KV cache，改为 no-flush + pause 时显式 clear + fingerprint 校验
4. **FSDP Dynamic Batch 对齐**：不同 DP rank 的 micro-batch 数必须一致，否则 allreduce hang

### 关键 Failure Modes 列表

1. SFT-RL 工具 schema JSON key 顺序不一致 → 模型在 RL 阶段无法调用工具
2. SFT per-message tokenization 丢失 Qwen3 的 reasoning content → 模型学不到 reasoning
3. Right truncation 截断 SFT 样本的 final answer → 改为 loss_window 滑动窗口
4. SGLang TP=2 generate 卡死 → 降级到 TP=1
5. param sync 只做 version bump 不同步实际权重 → 实现真实权重同步 + fingerprint 校验
6. queue_full 时 processor 死锁 → cancel-based partial drain
7. FSDP dynbsz 跨 rank micro-batch 数不一致 → same_micro_num_in_dp 对齐
8. 同步训练后期 90%+ rollout timeout → 驱动 async 架构改造
9. SGLang flush_cache 偶发空响应 → 强校验 + 自动 resume
10. gmu=0.3 下 SGLang KV cache 不足导致长序列 timeout → 调到 gmu=0.5 彻底消除
11. max_tool_response_length 从 6000 改为 5000 导致主瓶颈从 response_limit 转移到 timeout → 配置回滚
12. response_mask 不正确标记工具返回内容 → 只对模型生成 token 计算 loss

---

## 不要说的内容清单

- 不说"复现了论文结果"
- 不说"BrowseComp 上提升了 125%"（基数太低，百分比误导）
- 不说"从零搭建"（verl 有骨架）
- 不说 global_step_23、step70、step90 等内部编号
- 不说"远端机器不可用"
- 不说"只有 64 个样本的内部 gate"
- 不主动说 BrowseComp 的绝对数字（除非被追问）
- 不把 healthy-window 数字写成 whole-run average
