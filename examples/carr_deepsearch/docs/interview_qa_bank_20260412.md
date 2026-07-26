# CaRR DeepSearch Interview Q&A Bank (2026-04-12, status updated 2026-07-26)

## 0. 使用说明

这份文档的目标不是“项目说明书”，而是面向面试场景的可直接回答问答库。

- 优先级：覆盖面 > 精简度
- 回答形式：优先给出可以直接口述的短答，再给 1-3 个展开点
- 引用规则：
  - `Observed`：已有代码、文档或本地日志支持
  - `Estimated`：只用于尚未执行的 full/128K 可选扩展，不得对外当作实测结果
  - `Pending`：可选的后续研究项，不等于当前项目尚未收尾
- 对外使用规则：
  - `DeepDive rl_val 111` 和 `BrowseComp subset256 64k` 已完成 matched sampled eval，可以按精确范围引用真实结果
  - 不得把 `BrowseComp subset256` 写成完整 BrowseComp，也不得把结果写成论文复现
  - 文档后文若仍有评测前的 placeholder 或 Pending 表述，以本节和 `resume_source_of_truth_20260412.md` 的最新状态为准

建议使用方式：

1. 先熟读第 1 节到第 5 节，形成主线叙述
2. 面试前快速过一遍第 6 节和第 7 节，准备追问
3. 凡是带数字的回答，优先和 `resume_source_of_truth_20260412.md` 保持一致

---

## 1. 项目定位与一句话回答

### Q1. 你这个项目一句话是做什么的？

**短答**

我在 `verl` 里实现并稳定化了一个面向 deep-search agent 的 browser-tool RL 系统：把多轮 `search/open/find`、citation-aware reward、`C-GRPO` 和 `fully_async_policy` 接起来，让模型能在 `64k` 长上下文里做多跳搜索式问答训练。

**展开**

- 这不是单轮 RLHF，而是典型的 `agentic RL`：模型要多轮调用工具、维护证据链、最后给出带 citation 的答案。
- 方法上接的是 CaRR 风格奖励：不仅看最终答对没答对，还看中间事实是否被引用网页支撑、是否形成连通证据链。
- 工程上最难的是把这套 reward / agent loop / async RL infra 真正接进 `verl`，并把它稳定到可训练、可 resume、可评测。

**引用**

- `docs_zh/carr_walkthrough/01_全景概览与数据流.md:3-12`
- `examples/carr_deepsearch/docs/resume_source_of_truth_20260412.md:15-52`

### Q2. 这个项目和普通 RLHF / PPO 项目有什么本质不同？

**短答**

普通 RLHF 主要优化“给定 prompt 输出一段 answer”；这个项目优化的是“模型在多轮工具交互中的完整行为”，包括搜索、打开网页、页内查找、组织证据、写最终答案。

**展开**

- 输出不是单轮文本，而是长轨迹：`thinking -> tool call -> observation -> ... -> final answer`
- reward 不只看 final answer，还看 citation grounding 和 evidence connectivity
- rollout 不再是单次生成，而是一个状态机驱动的 agent loop
- async 训练需要处理 pause/resume、queue backpressure、partial rollout、staleness 和 strict resume

**引用**

- `docs_zh/agent_training/01_架构设计.md:52-123`
- `docs_zh/agent_training/02_AgentLoop详解.md:112-205`
- `CaRR/docs/02_CaRR奖励框架详解.md:5-7`

### Q3. 你这个项目里“真正由你完成”的部分是什么？

**短答**

我没有声称自己发明了 CaRR 算法；我做的是把 CaRR 的奖励设计和 deep-search 轨迹格式落地到 `verl`，并把一条原本会 timeout / OOM / stall 的 async RL 主线稳定到可训练、可恢复、可评测。

**展开**

- 算法思想来自 CaRR 论文：`outcome reward + rubric reward + C-GRPO`
- 我的核心工作是工程化实现：
  - 自定义 `CaRRToolAgentLoop` / `CaRRAsyncPartialToolAgentLoop`
  - 自定义 reward bridge，把 `messages/rubrics/unfinished` 接到 reward server
  - 自定义 `C-GRPO` advantage estimator
  - 修复 async 主线里的 `queue_full` backpressure、param sync、dynbsz/FSDP、resume / merge / eval gate

**引用**

- `examples/carr_deepsearch/tools/carr_agent_loop.py:61-171`
- `examples/carr_deepsearch/tools/carr_agent_loop.py:609-850`
- `examples/carr_deepsearch/reward/carr_reward.py:50-180`
- `examples/carr_deepsearch/reward/cgrpo_advantage.py:38-114`
- `examples/carr_deepsearch/docs/resume_source_of_truth_20260412.md:91-176`

### Q4. 为什么这个项目适合讲成 “agentic RL infra + research engineering”？

**短答**

因为它同时包含了方法实现和系统稳定化两层难点：一层是把 CaRR 奖励和 `C-GRPO` 正确接入，另一层是把 async RL 在真实多轮工具场景下跑稳。

**展开**

- `research engineering` 部分：把论文里的 reward semantics、rubric normalization、正确 rollout 才加 rubric 这些关键点实现对
- `infra` 部分：处理 queue、pause/resume、partial cancel、version span、dynbsz、SP=2、resume、HF merge
- 实验设计部分：把 greedy eval 和 sampled eval 区分开，用 outcome / unfinished / timeout / response-limit 去诊断

**引用**

- `CaRR/docs/01_论文概述与核心贡献.md:42-60`
- `examples/carr_deepsearch/docs/async_rl_explainer_20260321.md:684-708`
- `examples/carr_deepsearch/docs/resume_source_of_truth_20260412.md:167-313`

### Q5. 如果面试官问“这项目最难的点是什么”，你怎么答？

**短答**

最难的不是某一个超参，而是把“长轨迹工具 agent + citation-aware reward + async RL”三件事同时做对，因为每一层都会把下一层的问题放大。

**展开**

- reward 层如果只看 final answer，模型会学 shortcut
- agent loop 层如果不能稳定维护 `reward_history`，reward server 根本拿不到正确输入
- async 层如果 pause/resume、queue_full、dynbsz 没处理好，训练会直接 hang、stall 或 OOM

**引用**

- `docs_zh/carr_walkthrough/01_全景概览与数据流.md:16-63`
- `examples/carr_deepsearch/docs/agent_handoff_background_20260321.md:156-172`

---

## 2. Research / Method：CaRR、Reward、C-GRPO

### Q6. 为什么不能只用 outcome reward？

**短答**

只用二元 outcome reward，会把“侥幸答对”和“真正做了完整证据搜索后答对”混在一起，模型容易学会跳过搜索、猜答案或者编造引用。

**展开**

- pure outcome reward 无法区分 reasoning quality
- deep-search agent 会出现 shortcut exploitation：减少搜索步数也可能拿到 1 分
- 还会出现 hallucinated citation：最终答案碰巧对，但引用内容并不支撑答案

**引用**

- `CaRR/docs/01_论文概述与核心贡献.md:20-32`
- `docs_zh/carr_walkthrough/01_全景概览与数据流.md:18-30`

### Q7. CaRR 到底是什么？不要只说缩写。

**短答**

CaRR 是一种 citation-aware rubric reward 设计。它先把多跳问题拆成一组原子事实约束，再判断 agent 的回答里这些事实是否被正确识别、是否有引用网页支撑、是否能连成通向最终答案的证据链。

**展开**

- `rubric` 不是普通 checklist，而是用隐藏实体占位符表示的原子事实陈述
- judge 会检查实体识别、citation support、evidence connectivity
- 最终 `rubric reward` 是“连到最终答案的支持 rubrics 数 / 总 rubrics 数”

**引用**

- `CaRR/docs/01_论文概述与核心贡献.md:42-52`
- `CaRR/docs/02_CaRR奖励框架详解.md:13-25`
- `CaRR/docs/02_CaRR奖励框架详解.md:54-145`

### Q8. 什么是 rubric？为什么要引入隐藏实体？

**短答**

Rubric 是把复杂多跳问题拆成若干原子事实陈述；隐藏实体占位符的作用是强制模型通过搜索去恢复中间实体，而不是直接猜最终答案。

**展开**

- `<E0>` 通常表示最终答案，`<E1>...<E8>` 表示中间实体
- rubric 让 reward 能覆盖中间推理步骤，不只看 final answer
- 隐藏实体机制本质上是在 reward 设计里对抗 shortcut exploitation

**引用**

- `docs_zh/carr_walkthrough/01_全景概览与数据流.md:43-63`
- `CaRR/docs/02_CaRR奖励框架详解.md:16-25`

### Q9. CaRR 的三步 judge 流程是什么？

**短答**

第一步识别隐藏实体，第二步验证引用网页是否支撑每条 rubric，第三步检查这些被支撑的 rubrics 是否能从最终答案实体出发形成连通证据链。

**展开**

- 实体识别防止模型跳过中间实体
- citation judgment 防止模型编造事实
- connectivity check 防止模型靠收集一堆和答案无关的局部事实刷分

**引用**

- `docs_zh/carr_walkthrough/01_全景概览与数据流.md:58-63`
- `CaRR/docs/02_CaRR奖励框架详解.md:54-145`
- `CaRR/docs/02_CaRR奖励框架详解.md:190-193`

### Q10. 为什么 CaRR 强调 citation-aware？

**短答**

因为 deep-search agent 的关键价值不只是“答对”，而是“用可验证的网页证据答对”。如果引用内容不支持回答，即使结果碰巧正确，也不应该得到高分。

**展开**

- citation-aware 是 CaRR 相比纯 outcome reward 和 E-GRPO 的关键差异
- 这个设计直接针对 hallucinated sources
- 对实际产品化也更接近要求，因为可审计性很重要

**引用**

- `CaRR/docs/01_论文概述与核心贡献.md:49-52`
- `CaRR/docs/03_C-GRPO训练算法详解.md:224-230`

### Q11. 为什么还要做 evidence connectivity，而不是把支持的 rubrics 加起来就完？

**短答**

如果不做 connectivity，模型可以找到一些局部正确但和最终答案无关的事实来刷 rubric 分。连通性检查保证 reward 真正鼓励“从中间事实走到最终答案”的完整证据链。

**展开**

- 连接性是从 `<E0>` 出发做 BFS
- 只有和最终答案实体连通的 supported rubrics 才算进奖励
- 这是 CaRR 对抗 rubric hacking 的核心机制

**引用**

- `CaRR/docs/02_CaRR奖励框架详解.md:115-145`
- `examples/carr_deepsearch/docs/Carr_paper_data.md:7-12`

### Q12. 你的实现里 reward server 返回的是什么？为什么不是最终混合后的 reward？

**短答**

reward server 返回的是 `outcome_reward` 和 `rubric_reward`，而不是最终混合 reward；最终的 `C-GRPO` 融合是在 advantage 阶段按组做归一化和 final-token 重建。

**展开**

- `carr_reward.py` 的 `score` 先等于 `outcome_reward`
- 这是为了把组内归一化和 `outcome * rubric` 的 gating 放到 `advantage` 阶段统一做
- 这样不会把单条 rollout 上的局部 rubric 分数过早当成最终优化目标

**引用**

- `examples/carr_deepsearch/reward/carr_reward.py:59-62`
- `examples/carr_deepsearch/reward/carr_reward.py:171-180`
- `examples/carr_deepsearch/reward/cgrpo_advantage.py:72-114`

### Q13. C-GRPO 的核心公式是什么？你要怎么用一句话讲清楚？

**短答**

一句话讲，就是：先保证“答对”，再在答对的 rollout 之间用 group-normalized rubric reward 排序，公式是 `R = (1-α) * outcome + α * outcome * norm_rubric`。

**展开**

- 当 `outcome=0` 时，rubric 项自动清零
- 当 `outcome=1` 时，rubric 只影响“正确轨迹之间”的相对优劣
- 这避免了给错误轨迹错误的正反馈

**引用**

- `CaRR/docs/01_论文概述与核心贡献.md:54-60`
- `CaRR/docs/03_C-GRPO训练算法详解.md:31-45`
- `examples/carr_deepsearch/reward/cgrpo_advantage.py:99-112`

### Q14. 为什么 `rubric reward` 只加到正确 rollout 上？

**短答**

因为如果错误 rollout 也拿到 rubric 分，模型在训练早期会被鼓励去产生“中间看起来像对、最终其实错”的轨迹，优化方向会被带偏。

**展开**

- 论文的关键发现之一就是：对所有 rollout 都加 rubric reward 会明显变差
- 错误 rollout 可能命中局部事实，但最终答案仍错
- RL 早期正确 rollout 本来就少，这种错误正反馈会很危险

**引用**

- `CaRR/docs/03_C-GRPO训练算法详解.md:90-105`
- `CaRR/docs/03_C-GRPO训练算法详解.md:238-246`

### Q15. 为什么要在组内对 `rubric reward` 做归一化？

**短答**

因为不同问题的 rubric 数量不同，原始 rubric score 不可比。组内用最大值归一化后，训练信号更稳定，也更符合 GRPO 的组内比较逻辑。

**展开**

- 一些问题可能有 5 条 rubric，一些有 15 条
- 不归一化会让不同问题的 advantage scale 混乱
- 在实现里是按 group index 把同一组 rollout 聚在一起做 `max` 归一化

**引用**

- `CaRR/docs/03_C-GRPO训练算法详解.md:47-59`
- `examples/carr_deepsearch/reward/cgrpo_advantage.py:88-101`

### Q16. 为什么 `α=0.3`？

**短答**

`α=0.3` 是论文和项目里都遵循的平衡点：太小则 rubric 信号太弱，太大则模型会偏离“先答对”这个主目标。

**展开**

- `α=0` 退化成纯 GRPO
- `α` 太大时，模型会为了 rubric 过度追求局部过程分数
- 你的实现里默认也是 `0.3`

**引用**

- `CaRR/docs/01_论文概述与核心贡献.md:58-60`
- `CaRR/docs/03_C-GRPO训练算法详解.md:119-133`
- `examples/carr_deepsearch/reward/cgrpo_advantage.py:62-64`

### Q17. 你在实现里如何保证 unfinished 的 rollout 不会污染 reward？

**短答**

unfinished rollout 会被显式标记成 `task_unfinished`，并把原因拆成 `unfinished_limit`、`unfinished_budget`、`unfinished_empty_history`、`unfinished_no_final_assistant` 等字段；reward 侧按这些字段统一透传并保持一致语义。

**展开**

- agent loop 在产出时就构造 unfinished / termination flags
- reward bridge 会再次按同一规则兜底计算，防止下游看到不一致标记
- 这样训练和评测都能区分“没做完是因为 response limit 还是 budget timeout”

**引用**

- `examples/carr_deepsearch/tools/carr_agent_loop.py:125-171`
- `examples/carr_deepsearch/tools/carr_agent_loop.py:818-850`
- `examples/carr_deepsearch/reward/carr_reward.py:70-142`

### Q18. 你的方法层实现里，最容易写错的地方是什么？

**短答**

最容易写错的是把 `rubric reward` 当成单条 rollout 的直接 final reward，而不是放到组内去做 `outcome * normalized_rubric` 融合；第二个容易错的是 `reward_history` 的格式和 unfinished 语义不一致。

**展开**

- 论文里 reward server 和 training-time fusion 是分开的
- 如果在 reward server 里直接混掉，组内归一化逻辑就丢了
- 如果 `reward_history` 最后一条不是 assistant，或 cancel 后状态没恢复好，reward 会被错误打成 0

**引用**

- `CaRR/docs/03_C-GRPO训练算法详解.md:83-86`
- `examples/carr_deepsearch/reward/carr_reward.py:97-103`
- `examples/carr_deepsearch/tools/carr_agent_loop.py:134-171`

### Q19. 如果面试官问“你这里的 research 创新是什么”，怎么答才不越界？

**短答**

算法思想本身来自 CaRR 论文；我项目里的“创新”主要是工程化创新和训练诊断创新：把 citation-aware reward 和 `C-GRPO` 无缝接进 `verl`，再用可解释的 unfinished / termination 指标把 agentic RL 的 failure mode 拆开。

**展开**

- 不要 claim “我发明了 CaRR”
- 可以讲“我把 CaRR 的关键设计正确落地，并把它和 async RL infra 组合起来”
- 也可以讲“我把 reward / loop / eval / async pause-resume 做成了一个可训练系统”

**引用**

- `examples/carr_deepsearch/docs/resume_source_of_truth_20260412.md:91-176`

---

## 3. verl 架构、数据流与接线点

### Q20. 你把 CaRR 接在了 `verl` 的哪几层？

**短答**

主要接了三层：Agent Loop、Reward、Advantage。

**展开**

- Agent Loop：自定义 `CaRRToolAgentLoop` / `CaRRAsyncPartialToolAgentLoop`
- Reward：`carr_reward.compute_score()` 作为 `NaiveRewardManager` 的 reward bridge
- Advantage：`@register_adv_est("cgrpo")` 的 `compute_cgrpo_advantage`

**引用**

- `examples/carr_deepsearch/tools/carr_agent_loop.py:61-171`
- `examples/carr_deepsearch/tools/carr_agent_loop.py:609-850`
- `examples/carr_deepsearch/reward/carr_reward.py:50-180`
- `examples/carr_deepsearch/reward/cgrpo_advantage.py:38-114`

### Q21. 训练主循环里，`verl` 这些组件分别负责什么？

**短答**

`PPOTrainer` 管训练主循环，`AgentLoopManager` 管 rollout 编排，`RewardLoop/RewardManager` 管奖励计算，`Actor/Critic/Ref` worker 负责 PPO/GRPO 的优化部分。

**展开**

- `AgentLoopManager` 负责把 batch 分发给多个 agent loop worker
- agent loop worker 内部用 `asyncio` 并发跑多个样本
- Reward manager 根据 rollout 产出调用 reward function
- async 模式下，`fully_async_main` 会再引入 `MessageQueue`、`Rollouter`、`Trainer`、`ParameterSynchronizer`

**引用**

- `docs_zh/agent_training/01_架构设计.md:7-49`
- `docs_zh/carr_walkthrough/01_全景概览与数据流.md:66-130`

### Q22. 为什么 `verl` 这里强调 token-based API，而不是 chat-completion API？

**短答**

因为 RL 训练需要 rollout 时的 token 序列和训练时重新算 logprob 的 token 序列严格一致，chat-completion 走文本 decode / encode 会引入 tokenization mismatch。

**展开**

- 工具调用会改写 content，文本重建可能和原始 token 不一致
- decode-encode 不是严格可逆
- PPO/GRPO 的 ratio 和 clip 机制依赖在同一 token 序列上计算 logprob

**引用**

- `docs_zh/agent_training/01_架构设计.md:125-170`
- `docs_zh/agent_training/01_架构设计.md:171-220`

### Q23. `response_mask` 为什么重要？

**短答**

因为多轮 agent 轨迹里既有模型自己生成的 token，也有工具返回的 token；只有模型生成的 token 才该参与策略梯度，所以必须用 `response_mask` 区分。

**展开**

- `mask=1` 表示 LLM 生成 token，参与 loss
- `mask=0` 表示工具响应或环境交互 token，只作为上下文存在
- 这让多轮工具轨迹仍然能被当成一个统一序列训练

**引用**

- `docs_zh/agent_training/02_AgentLoop详解.md:69-99`

### Q24. `reward_history` 到底是什么，为什么不能只传 final answer？

**短答**

因为 CaRR 的 reward 不只看 final answer，还要看中间引用、tool observation 和最终 assistant message 之间是否构成被网页支撑的证据链，所以必须把整条可评估历史传给 reward server。

**展开**

- `reward_history` 里会保留 user / assistant / tool 等关键信息
- reward server 需要基于这个 history 做 citation-based rubric judgment
- final answer 单独拿出来不够，因为你会丢掉 supporting evidence

**引用**

- `examples/carr_deepsearch/tools/carr_agent_loop.py:296-475`
- `examples/carr_deepsearch/tools/carr_agent_loop.py:645-673`
- `examples/carr_deepsearch/reward/carr_reward.py:65-80`

### Q25. 你自定义的 `CaRRToolAgentLoop` 比 `ToolAgentLoop` 多了什么？

**短答**

多了三类能力：CaRR-compatible history、budget / unfinished 原因跟踪、以及把 tool-use metrics 结构化塞进输出供 reward 和 eval 读取。

**展开**

- 单独维护 `reward_history`，而不是只依赖 prompt 拼接
- 显式管理 `tool/search/open/find` budget 和 `response/turn` limit
- 输出 `termination_*`、`unfinished_*`、`tool_call_counts`、`response_length_ratio` 等诊断字段

**引用**

- `examples/carr_deepsearch/tools/carr_agent_loop.py:92-171`
- `examples/carr_deepsearch/tools/carr_agent_loop.py:434-485`

### Q26. Async 版本的 `CaRRAsyncPartialToolAgentLoop` 额外做了什么？

**短答**

它把 agent loop 变成了可中断、可恢复的状态机：在 cancel 后保留 `reward_history`、pending tool calls、当前 assistant turn、budget 状态和参数版本，从而支持 partial rollout 跨版本继续生成。

**展开**

- async state 里维护 `param_version_start`、`last_param_version`
- 生成中被 cancel 时不会丢样本，而是带 `is_cancel=True` 返回中间态
- resume 后用相同 `request_id` 继续同一个 tool session

**引用**

- `examples/carr_deepsearch/tools/carr_agent_loop.py:621-673`
- `examples/carr_deepsearch/tools/carr_agent_loop.py:712-790`
- `examples/carr_deepsearch/tools/carr_agent_loop.py:792-850`

### Q27. `search/open/find` 工具是怎么接进来的？

**短答**

模型在生成时输出 tool call，agent loop 解析后通过 browser tool 封装发 HTTP 请求到外部 tool server，再把结果 tokenize 回序列继续下一轮生成。

**展开**

- 这是标准 `ToolAgentLoop` 的 `GENERATING -> PROCESSING_TOOLS -> GENERATING` 状态机
- CaRR 版本额外记录了各类 tool count 和 budget 命中情况
- 这些 tool observation 不参与 loss，但会进入上下文和 reward history

**引用**

- `docs_zh/agent_training/02_AgentLoop详解.md:180-205`
- `docs_zh/carr_walkthrough/01_全景概览与数据流.md:183-220`

### Q28. 为什么要对工具响应做长度截断？

**短答**

因为网页内容很长，工具响应如果不截断，会直接把上下文挤爆，既拖慢 rollout，也让后续 LLM 生成更容易 hit response limit。

**展开**

- `max_tool_response_length` 是长轨迹预算里很关键的一个杠杆
- 正式口径固定在 `6000`
- 只有在特定 failure mode 明确指向 response-limit 时才考虑调小做对照

**引用**

- `docs_zh/agent_training/02_AgentLoop详解.md:193-200`
- `examples/carr_deepsearch/scripts/run_async_formal.sh:115-127`

### Q29. 你的数据流从 parquet 到更新 actor，大致是怎样的？

**短答**

parquet 提供 prompt + rubrics + extra_info，agent loop 跑出多轮轨迹，reward manager 调 reward server 打 `outcome/rubric`，然后 `C-GRPO` 在 advantage 阶段重建 final-token reward，最后 actor 做 GRPO/PPO 更新。

**展开**

- rollout 输出里会有 `messages`、`response_mask`、`tool metrics`
- reward path 会把 unfinished / termination / tool-use 一并透传
- advantage 侧把 `outcome` 和 normalized `rubric` 融在最后一个 valid token 上

**引用**

- `docs_zh/carr_walkthrough/01_全景概览与数据流.md:174-220`
- `examples/carr_deepsearch/reward/cgrpo_advantage.py:99-114`

### Q30. 你为什么要单独写一个 `select_async_start_checkpoint.py`？

**短答**

因为对于 Thinking checkpoint，不能默认“step 越晚越好”。需要一个显式 selector 按 sampled eval 的 `outcome`、`unfinished`、`timeout` 去选 async 起点。

**展开**

- 候选首先要过 `unfinished <= 0.70` 和 `timeout <= 0.45`
- 主排序看 `outcome_reward`
- 前两名差距小于 `0.03` 时，再用 `unfinished` 和 `timeout` tie-break
- 这是历史上用于 `SFT / step70 / step90` 的起点 gate；后续新 GPU 上做 `async_best` 收尾时，规则会更严格，差值 `< 0.02` 时先看 `rubric_reward`，不能把两套 policy 混成一条

**引用**

- `examples/carr_deepsearch/scripts/select_async_start_checkpoint.py:12-18`
- `examples/carr_deepsearch/scripts/select_async_start_checkpoint.py:93-136`
- `examples/carr_deepsearch/docs/post_gpu_revalidation_plan_20260412.md:129-140`

---

## 4. Async RL Infra、参数同步与系统取舍

### Q31. 你为什么要做 async RL，而不是继续跑同步 RL？

**短答**

因为 multi-turn browser agent 的 rollout 太长，纯同步训练会把所有 GPU 绑在最慢样本上，工具调用等待期间 GPU 大量空闲；async 的核心目标是让 rollout 和 training 重叠起来，减少长尾浪费。

**展开**

- 工具调用延迟和样本长尾在 agentic RL 里比普通 RLHF 严重得多
- async 允许 trainer 消费已经完成的样本，同时 rollouter 继续生产
- 但代价是要处理 staleness、queue、param sync、resume 和 partial rollout

**引用**

- `docs_zh/agent_training/01_架构设计.md:52-84`
- `examples/carr_deepsearch/docs/async_rl_explainer_20260321.md:401-437`

### Q32. `fully_async_policy` 里最核心的新组件是什么？

**短答**

最核心是四个组件：`MessageQueue`、`Rollouter`、`Trainer`、`ParameterSynchronizer`。

**展开**

- `Rollouter` 持续产样本
- `MessageQueue` 负责跨 actor / process 传递完成样本和版本信息
- `Trainer` 从队列取样本并更新模型
- `ParameterSynchronizer` 负责 pause、传权重、做 fingerprint validation、更新 param version、resume

**引用**

- `examples/carr_deepsearch/docs/async_rl_explainer_20260321.md:639-645`
- `verl/experimental/fully_async_policy/param_sync.py:163-204`

### Q33. 你可以怎么解释 `trigger_parameter_sync_step`？

**短答**

它决定 trainer 做多少次 local update 才同步一次参数。值越小，参数越新但 pause 更频繁；值越大，吞吐更高但 stale 样本更多。

**展开**

- `trigger_sync_step=1` 更接近同步
- `trigger_sync_step=4` 是你最后主线的折中点
- 这个参数和 `staleness_threshold` 是 async 松弛程度的两个主控制杆

**引用**

- `examples/carr_deepsearch/docs/async_rollouter_pause_mechanism_explainer.md:183-192`
- `examples/carr_deepsearch/docs/sync_vs_async_grpo_update_mechanics.md:439`

### Q34. `staleness_threshold` 是什么？为什么不是越小越好？

**短答**

它控制 rollouter 允许额外缓存多少“旧版本生成的样本”。越小越接近 on-policy，但也更容易让 rollouter 过早停下来，吞吐下降。

**展开**

- 本质上是 on-policy 程度和吞吐的 trade-off
- `staleness=0.5` 允许适度缓冲，通常能换来更好的 pipeline 满度
- 如果过大，PPO ratio 被 clip 的概率会增加，训练可能变慢

**引用**

- `examples/carr_deepsearch/docs/async_rollouter_pause_mechanism_explainer.md:811-820`
- `examples/carr_deepsearch/docs/sync_vs_async_grpo_update_mechanics.md:360-377`

### Q35. 什么是 partial rollout？它解决什么问题？

**短答**

partial rollout 允许在参数同步时 cancel 正在生成的样本、同步完权重后再继续，从而把 “等所有长轨迹自然跑完再 sync” 变成 “只等当前生成轮次返回再 sync”。

**展开**

- 没有 partial 时，pause 可能要等几分钟
- 有 partial 时，理论上可以把 param sync 的等待压到几秒级
- 这要求样本状态、tool session、param version 和 budget 都能完整恢复

**引用**

- `examples/carr_deepsearch/docs/async_rollouter_pause_mechanism_explainer.md:96-115`
- `examples/carr_deepsearch/docs/async_rollouter_pause_mechanism_explainer.md:494-510`
- `examples/carr_deepsearch/tools/carr_agent_loop.py:621-673`

### Q36. 你最后为什么选 `2:6 + SP=2 + dynbsz + b=3` 作为主线？

**短答**

因为这是你实际验证里最稳的折中点：rollout 侧 6 卡足够把 timeout 压下去，trainer 侧 2 卡配合 `SP=2` 和 dynbsz 能避免 OOM，而 `ppo_mini_batch_size=3` 又把 update_actor 速度拉回到了可接受范围。

**展开**

- `3:5 + SP=1` 实测 OOM 或 batch divisibility 不稳定，不适合作正式配置
- `4:4` rollout 侧不够强，trainer / rollout 都偏弱
- `2:6 + SP=2 + gmu=0.5 + b=3` 能达到 durable `global_step_23`

**引用**

- `examples/carr_deepsearch/docs/agent_handoff_background_20260321.md:166-190`
- `examples/carr_deepsearch/docs/agent_handoff_background_20260321.md:197-239`

### Q37. 为什么 `ppo_mini_batch_size=4 -> 3` 会有帮助？

**短答**

因为在你这个 `2 trainer GPU + SP=2 + long sequence` 的设置下，`update_actor` 已经是显式瓶颈。把 mini-batch 从 4 降到 3 可以显著降低每次 update 的峰值成本和 wall-clock，同时不至于像降到 2 那样把吞吐打得太碎。

**展开**

- 最后主线 `b=3` 对应的是 step 9 以后更稳定的速度窗口
- 这不是单纯“显存不够”，也是为了平衡 update_actor 时长
- `b=3` 后出现了更好的健康窗口和更低的 unfinished 窗口

**引用**

- `examples/carr_deepsearch/docs/agent_handoff_background_20260321.md:208-223`

### Q38. 为什么 `gpu_memory_utilization=0.5` 是一个关键拐点？

**短答**

因为 rollout GPU 原先只给 KV cache 留了太少空间，长序列并发下 preemption 很严重；把 `gmu` 从 `0.3` 提到 `0.5` 后，KV cache 空间显著增大，timeout 从主导失败模式变成了基本消失。

**展开**

- 这个改动主要改善 rollout 侧，而不是 trainer 侧
- 本质是减少 SGLang 对长序列并发的 cache 抢占
- 在你的主线表里，`gmu=0.5` 后 timeout 明显下降

**引用**

- `examples/carr_deepsearch/docs/agent_handoff_background_20260321.md:168-171`
- `examples/carr_deepsearch/docs/agent_handoff_background_20260321.md:205-209`

### Q39. `queue_full` backpressure 是什么？为什么它会把训练拖死？

**短答**

当 `MessageQueue` 满了时，rollouter 会暂停 dispatch。如果只是等在飞长轨迹自然完成，就会把 param sync 和整体 pipeline 卡成分钟级长尾。

**展开**

- 这是 async 系统特有的 backpressure 问题
- 如果 queue_full 只做自然 drain，不做 cancel-based drain，慢样本会把 trainer 和 rollouter 一起拖住
- 这正是你后来修的核心问题之一

**引用**

- `examples/carr_deepsearch/docs/async_rollouter_pause_mechanism_explainer.md:542-603`
- `verl/experimental/fully_async_policy/fully_async_rollouter.py:578-631`

### Q40. 你具体怎么修 `queue_full` partial drain 的？

**短答**

我把 queue_full 场景从“等 active tasks 自然返回”改成了 cancel-based partial drain：先 `cancel()` 在飞任务，`gather()` 回流中间态，等真正恢复时再安全清掉 cancellation state，避免和外部 param-sync pause/resume 发生竞态。

**展开**

- queue_full 分支下会设置 `_queue_full_cancel_pending`
- 不直接在 queue_full 分支里 `resume()`，而是在真正 `paused=False` 之后再清 cancellation event
- 这样不会破坏外部 param-sync pause 的时序契约

**引用**

- `verl/experimental/fully_async_policy/fully_async_rollouter.py:578-631`

### Q41. `param_sync` 具体做了什么？

**短答**

它不是单纯 version bump，而是真正执行了：pause rollouter、更新队列版本号、同步 actor 权重到 rollout、做 fingerprint validation、更新 param version、再 resume。

**展开**

- 这也是为什么 `param_sync` 的时间里会包含 pause 等待成本
- 你的修复之后能把健康窗口压到 `1.7-20s`
- 但 residual tails 仍存在，observed 过约 `85s / 102s / 144s / 209s`

**引用**

- `verl/experimental/fully_async_policy/param_sync.py:163-204`
- `examples/carr_deepsearch/docs/resume_source_of_truth_20260412.md:263-295`

### Q42. 什么是 fingerprint validation？为什么要做？

**短答**

因为 async 训练里“版本号变了”不等于“权重真的同步过去了”。fingerprint validation 会比较 actor 和 rollout 侧同步后的权重签名，防止出现只做了 pause/resume、实际没传权重的假同步。

**展开**

- 这是 async correctness 的关键保证
- 没有这个校验，日志里看起来像同步成功，实际上 rollout 可能还在用旧权重
- 你在项目里把它当成 infra correctness gate，而不只是 debug print

**引用**

- `verl/experimental/fully_async_policy/param_sync.py:150-161`
- `verl/experimental/fully_async_policy/param_sync.py:183-185`

### Q43. `dynamic_bsz` 和 `SP=2` 分别解决什么问题？

**短答**

`dynamic_bsz` 解决的是长序列 batch 在 trainer 侧的 token packing 和峰值内存问题，`SP=2` 解决的是单条超长序列 backward 的每卡显存峰值问题。

**展开**

- 没有 dynbsz，长序列 batch 很容易在 trainer 侧不均匀爆炸
- 没有 `SP=2`，在 2 trainer GPU 的设置下直接 `SP=1` 很容易 OOM
- 两者一起用，才能把 `2:6` 这条线真正跑稳

**引用**

- `examples/carr_deepsearch/docs/agent_handoff_background_20260321.md:168-172`
- `verl/workers/actor/dp_actor.py:522-526`
- `verl/workers/critic/dp_critic.py:205-209`

### Q44. 你修的 dynbsz / FSDP rank alignment bug 是什么？

**短答**

问题本质是 dynamic batch 切分后，不同 DP rank 看到的 micro-batch 数不一致，FSDP collective 会 hang 或出错。我把 `dp_group=self.dp_group` 和 `same_micro_num_in_dp=True` 显式传进 actor / critic 的 `prepare_dynamic_batch()`，强制 rank 间对齐。

**展开**

- 这是典型的“局部看没问题，全局 collective 才炸”的 bug
- 修完后 dynamic bsz 才能在 DP 场景下稳定使用
- 这也是 later `use_dynamic_bsz=true` 能进入主线的必要前提

**引用**

- `verl/workers/actor/dp_actor.py:522-526`
- `verl/workers/critic/dp_critic.py:205-209`

### Q45. 为什么 `3:5 + SP=1` 最后没有成为主线？

**短答**

因为它虽然有机会降低 `SP=2` 的通信开销，但实际 probe 里暴露了 OOM 和 batch divisibility / trainer stability 问题，没有达到“已知稳定”的标准。

**展开**

- `32 trajectories / update` 对 `3 GPU trainer` 的整除性就有约束
- `SP=1` 下单卡显存峰值更高
- 在项目阶段，主线选择标准是“稳定性优先于理论更快”

**引用**

- `examples/carr_deepsearch/docs/agent_handoff_background_20260321.md:171-172`
- `examples/carr_deepsearch/docs/resume_source_of_truth_20260412.md:167-170`

### Q46. strict resume、model-only restart 和 HF merge 各是什么？

**短答**

- strict resume：恢复 optimizer、dataloader、global step 等完整训练状态
- model-only restart：只拿模型权重重新起训或评测，不恢复 optimizer state
- HF merge：把 FSDP shard 合并成 `huggingface_merged`，方便后续 eval 或 model-only 使用

**展开**

- strict resume 适合正式主线 continuation
- model-only restart 更适合 probe 或只关心模型行为时
- eval 前只认 `actor/huggingface_merged`

**引用**

- `examples/carr_deepsearch/docs/post_gpu_revalidation_plan_20260412.md:64-95`
- `examples/carr_deepsearch/scripts/run_async_formal.sh:45-78`
- `examples/carr_deepsearch/docs/agent_handoff_background_20260321.md:176-190`

### Q47. 为什么 `RAY_NUM_CPUS` 也会成为 async run 的关键参数？

**短答**

因为 async tool-using run 不只是 GPU 重，它还非常吃 CPU：Ray worker、tool server、reward loop、请求调度都跑在 CPU 上。CPU 太紧会把异步 pipeline 人为卡慢。

**展开**

- 你在 formal launcher 里明确把默认值放宽到了 `128`
- 这是从“把 RL 当纯 GPU 问题”升级成“把 agentic RL 当分布式系统问题”
- 这个点在面试里很能体现工程感

**引用**

- `examples/carr_deepsearch/scripts/run_async_formal.sh:219-227`

### Q48. 为什么固定 `TP=1`、`flashinfer`、`NCCL_P2P_DISABLE=1`？

**短答**

因为这条 async 主线要先优先稳定性和可复现性。`TP=1` 降低复杂度，`flashinfer` 是当时验证过可用的 attention backend，`NCCL_P2P_DISABLE=1` 是为当前硬件 / 通信栈下的稳定性保底。

**展开**

- 在长轨迹工具场景里，先把系统跑稳，比追求更激进的并行策略重要
- 这也是你 formal launcher 强写共同默认值的原因

**引用**

- `examples/carr_deepsearch/scripts/run_async_formal.sh:239-245`

---

## 5. 调参与排障：失败模式、定位过程、权衡

### Q49. Stage A 最开始为什么会 100% timeout？

**短答**

根因不是工具预算不够，而是 rollout server 并发过载。`n=8`、`max_concurrent_samples=12`、只有 4 个 rollout GPU 时，每个 server 实际在扛的长序列并发太高，SGLang preemption 和 KV cache 抢占严重，导致单次 generate 太慢，最后全部撞到 wall time。

**展开**

- 不是 “做太多事”，而是 “做每件事都太慢”
- 关键现象是 tool count 还远没到上限，response 也没打满，但 wall time 已经超了
- 后续通过降 concurrent、提高 `gpu_memory_utilization`、改 split 才逐步缓解

**引用**

- `examples/carr_deepsearch/docs/agent_handoff_background_20260321.md:199-204`
- `examples/carr_deepsearch/docs/async_rollouter_pause_mechanism_explainer.md:745-764`

### Q50. 为什么不直接把 `n=8` 改成 `n=4`？

**短答**

因为这个项目一开始的明确目标就是优先保住 `n=8` 的组内比较信号。只有当 `n=8` 三档都不稳定时，才允许回退 `n=4` 作为工程兜底。

**展开**

- `n=8` 对 `C-GRPO` / GRPO 这种组内相对比较类算法更有利
- 你先调的是 concurrent、split、gmu、SP、dynbsz，而不是先砍掉 `n`
- 这体现的是“保住方法目标，再在 infra 上找解”

**引用**

- `examples/carr_deepsearch/docs/agent_handoff_background_20260321.md:166-172`
- `examples/carr_deepsearch/scripts/run_async_formal.sh:202-260`

### Q51. 为什么最终 unfinished 的主要坏因从 timeout 变成了 limit / budget？

**短答**

因为当 rollout 侧速度问题被解决后，模型开始能跑得更深、更长，新的主限制变成了 response length、search budget 和 assistant turn 等上层预算，而不是 wall time 先把它砍死。

**展开**

- 这是“失败模式迁移”，本质上是进步而不是退步
- 说明模型不再卡在“还没来得及做事就 timeout”
- 后面 unfinished 更有解释性，也更适合做继续训练或 budget 调整

**引用**

- `examples/carr_deepsearch/docs/agent_handoff_background_20260321.md:225-232`
- `examples/carr_deepsearch/docs/resume_source_of_truth_20260412.md:237-313`

### Q52. 为什么 sampled eval 比 greedy 更合适？

**短答**

对 Thinking checkpoint，greedy 并不是“中立基线”，它会放大重复和 collapse failure mode，甚至更慢。sampled recipe 更贴近 backbone 的 intended behavior，也更适合作 gate。

**展开**

- 你本地对同一 checkpoint 做过 greedy vs sampled 对照
- greedy 下 `task_unfinished` 更高，wall time 更长
- 所以后来 `DeepDive subset64` 的 gate 固定成 `temperature=0.6, top_p=0.95, top_k=20, do_sample=true`

**引用**

- `examples/carr_deepsearch/docs/rl_debug_findings_20260312.md:847-912`
- `examples/carr_deepsearch/scripts/run_async_formal.sh:115-127`

### Q53. `trainer/idle_ratio`、`rollouter/idle_ratio` 分别怎么看？

**短答**

`trainer/idle_ratio` 高通常说明 trainer 在等样本，rollout 产能不够；`rollouter/idle_ratio` 高则更像 trainer 消费太慢或者 queue / sync 把 rollout 卡住了。

**展开**

- 这是你判断“现在该调 rollout 还是调 trainer”的关键指标
- 在你项目中，前期更多是 rollout 慢，后期更多是 `update_actor` 成为瓶颈

**引用**

- `examples/carr_deepsearch/docs/async_rollouter_pause_mechanism_explainer.md:760-774`
- `examples/carr_deepsearch/docs/agent_handoff_background_20260321.md:169-171`

### Q54. 为什么 `timing_s/param_sync` 有时能做到 1.7s，有时又会回到几十秒甚至 200s 左右？

**短答**

因为 `param_sync` 时间不只是“传权重”本身，它还吃 pause 成本；只要碰上 queue_full、慢样本尾部或者当前轮次尚未回流，`pause()` 就会被拖长，所以它天然有长尾。

**展开**

- 1.7s 说明 partial / queue drain 在那个窗口里工作得很好
- 但 residual tails 仍真实存在，observed 过约 `85s / 102s / 144s / 209s`
- final `gs14` 主线能看到 `~85s / ~102s`，更早的 `gs11` 线还出现过 `~144s / ~209s`
- 这也是为什么文档里要写成“健康窗口 + 剩余长尾”

**引用**

- `verl/experimental/fully_async_policy/param_sync.py:167-190`
- `examples/carr_deepsearch/docs/async_rollouter_pause_mechanism_explainer.md:517-526`
- `examples/carr_deepsearch/docs/resume_source_of_truth_20260412.md:263-279`

### Q55. 你是怎么判断“当前瓶颈已经从 rollout 转到 trainer”的？

**短答**

当 `timeout` 降下来、`trainer/idle_ratio` 接近 0、但 `timing_s/update_actor` 仍然占大头时，就说明 rollout 已经不是主瓶颈，trainer update 成了新的上限。

**展开**

- `gmu=0.5` 后 rollout 明显改善
- 到后期健康窗口里，`gen` 几乎不再占主要 wall-clock
- 这时再提速的主要空间在 trainer 侧而不是 rollout 侧

**引用**

- `examples/carr_deepsearch/docs/agent_handoff_background_20260321.md:216-223`
- `examples/carr_deepsearch/docs/resume_source_of_truth_20260412.md:280-294`

### Q56. 为什么 `step16` 很重要，但你又不能把它当最终结果讲？

**短答**

因为 `step16` 是训练中的最佳内部窗口，不是正式 external eval checkpoint。它说明系统在某个阶段确实跑出了很强的训练信号，但不等于你已经拿它做了完整 benchmark。

**展开**

- `step16` 的 `outcome`、`unfinished`、`param_sync` 都很好
- 但面试里要把它讲成 “best observed internal training window”
- 不能偷换成 “final benchmark result”

**引用**

- `examples/carr_deepsearch/docs/resume_source_of_truth_20260412.md:280-294`
- `examples/carr_deepsearch/CaRR_log/formal_mainline_gs14_sp2_dynbsz_gmu05_b3_reasonfix_20260323_215746.launcher.log:8730-8741`

### Q57. 如果面试官质疑“你是不是一直在调超参”，怎么回应？

**短答**

我会强调这不是盲调，而是按 failure mode 调。每次改动都对应一个明确瓶颈：并发过载、KV cache 抢占、queue_full backpressure、dynbsz rank mismatch、update_actor 峰值过高、response limit 主导等。

**展开**

- 你每次调的是机制，不是碰运气
- 比如 `gmu=0.5` 是针对 rollout cache/preemption
- `b=3` 是针对 trainer update_actor
- `queue_full` 修复是针对 param_sync 长尾

**引用**

- `examples/carr_deepsearch/docs/agent_handoff_background_20260321.md:197-239`

---

## 6. 评测设计、结果口径与数字

### Q58. 你们内部 gate 为什么选 `DeepDive subset64`？

**短答**

因为它便宜、固定、可复现，能在不烧太多预算的情况下对不同 checkpoint 做相对可靠的对比，并且它和训练数据 schema、reward 逻辑是一致的。

**展开**

- `subset64` 是内部 gate，不是最终外部 benchmark
- 采用 fixed sampled recipe，避免 greedy 对 Thinking checkpoint 的误导
- 用于选择 async 起点和后续 async checkpoint 对比都很合适

**引用**

- `examples/carr_deepsearch/scripts/run_async_formal.sh:91-159`
- `examples/carr_deepsearch/docs/post_gpu_revalidation_plan_20260412.md:106-140`

### Q59. 真实的 `SFT / step70 / step90` gate 结果是什么？

**短答**

- `SFT`: `outcome=0.296875`, `rubric=0.04475`, `unfinished=0.65625`
- `step70`: `outcome=0.328125`, `rubric=0.05226`, `unfinished=0.640625`
- `step90`: `outcome=0.265625`, `rubric=0.06332`, `unfinished=0.671875`

**展开**

- `step70` 在 outcome 和 unfinished 上最适合作为 async 起点
- `step90` 的 rubric 更高，但整体 gate 规则仍然不会选它

**引用**

- `examples/carr_deepsearch/CaRR_log/eval_gate_20260323_144500_sft.log:1212`
- `examples/carr_deepsearch/CaRR_log/step70_dd64_8gpu_20260322_153257.log:1270`
- `examples/carr_deepsearch/CaRR_log/step90_dd64_8gpu_20260322_151340.log:1269`
- `examples/carr_deepsearch/docs/resume_source_of_truth_20260412.md:237-250`

### Q60. 为什么最后选 `step70` 而不是 `step90`？

**短答**

因为你的 selector 是 outcome-first gate，不是 rubric-first gate。`step70` 的 `outcome_reward` 更高、`task_unfinished` 更低，而且两者都没有 timeout 问题，所以它更适合作为 async 起点。

**展开**

- `step90` 没有证明“训练更多步数就一定更好”
- 这个例子本身也是一个很好的面试点：checkpoint selection 不能靠 step number 想当然

**引用**

- `examples/carr_deepsearch/scripts/select_async_start_checkpoint.py:12-18`
- `examples/carr_deepsearch/scripts/select_async_start_checkpoint.py:109-136`
- `examples/carr_deepsearch/docs/resume_source_of_truth_20260412.md:224-250`

### Q61. async 主线目前有哪些真实可讲的结果？

**短答**

三个最安全的结果是：

- 最终 durable checkpoint 到了 `global_step_23`
- `param_sync` 从病态 `~323s` 收敛到常见 `1.7-20s` 健康窗口，但 observed residual tails 仍包括约 `85s / 102s / 144s / 209s`
- 在健康窗口里，async 的 `per-global-step` 归一化时间可以做到 `~280-320s / gs`

**引用**

- `examples/carr_deepsearch/docs/resume_source_of_truth_20260412.md:263-279`
- `examples/carr_deepsearch/docs/resume_source_of_truth_20260412.md:280-294`

### Q62. `global_step_23` 为什么可以算 durable checkpoint？

**短答**

因为本地日志和 checkpoint iteration 标记都能直接支撑它：`latest_checkpointed_iteration.txt=23`，最终 run 也明确写出了 `global_step_23` 的 dataloader 和 actor checkpoint 保存记录。

**展开**

- 这是一个“真正写到盘上”的 checkpoint，不是只在日志里出现 step number
- 这也是为什么它可以作为后续 strict resume / model-only eval 的锚点

**引用**

- `examples/carr_deepsearch/CaRR_log/latest_checkpointed_iteration.txt`
- `examples/carr_deepsearch/CaRR_log/formal_mainline_gs14_sp2_dynbsz_gmu05_b3_reasonfix_20260323_215746.launcher.log:36100-36640`

### Q63. 什么叫 `~280-320s / gs`？为什么你要强调这是“健康窗口”？

**短答**

因为日志原生打出来的是一个聚合 step 的 `timing_s/step`，而你主线配置里每个聚合 step 对应 `4` 个 global steps，所以这里的 `~280-320s / gs` 是把健康窗口的聚合 step 时间除以 4 得到的归一化值，不是全程平均。

**展开**

- 这是正确口径，能和同步 formal `~506s / step` 做粗粒度对比
- 但不能把它说成全 run average
- 文档里已经要求必须解释成 “healthy window + residual tails”

**引用**

- `examples/carr_deepsearch/docs/resume_source_of_truth_20260412.md:66-68`
- `examples/carr_deepsearch/CaRR_log/formal_mainline_gs14_sp2_dynbsz_gmu05_b3_reasonfix_20260323_215746.launcher.log:8730-8741`
- `examples/carr_deepsearch/CaRR_log/formal_mainline_gs14_sp2_dynbsz_gmu05_b3_reasonfix_20260323_215746.launcher.log:36087-36099`

### Q64. sync formal 的对比基线是什么？

**短答**

目前文档里固定用 sync formal `step 71-90` 的 `timing_s/step` 中位数，约 `506s / step` 作为稳定窗口基线。

**展开**

- 这个基线只用于系统工程层面的 wall-clock 对比
- 不能据此直接 claim “模型效果优于 sync”

**引用**

- `examples/carr_deepsearch/CaRR_formal_training_log/output.log`
- `examples/carr_deepsearch/docs/resume_source_of_truth_20260412.md:252-261`

### Q65. 你目前有哪些结果不能 claim 成既成事实？

**短答**

当前不能 claim 的主要是完整 `BrowseComp`、128K 上下文结果、GRPO/C-GRPO 本地消融，以及“异步架构在完全相同配置下因果性提升了 37%-45%”。已经完成并可以准确引用的是 `DeepDive rl_val 111` 与 `BrowseComp subset256 64k` 的 matched sampled eval。

**展开**

- `BrowseComp subset256 64k` 是真实结果，但不是 full benchmark
- 同步和异步阶段的 batch、rollout 数量与 budget 不完全相同，因此吞吐数字是系统工程对比，不是严格消融
- 论文中的算法消融可以用于解释设计依据，不能说成我在本项目里重新做过

**引用**

- `examples/carr_deepsearch/docs/resume_source_of_truth_20260412.md`
- `examples/carr_deepsearch/docs/eval_analysis_20260415.md`

### Q66. 如果面试官追问“BrowseComp 具体测到了多少”，怎么回答？

**短答**

真实测评是 `BrowseComp subset256 64k`：在完全一致的 sampled eval 配置和 relaxed-but-bounded budget 下，SFT 的 judge-pass 为 `4/256`，async23 为 `9/256`；记录到的完成回答数从 `11` 增加到 `17`。这是方向一致的外部正向证据，但正样本绝对数量较少，不能包装成完整 benchmark 的稳定大幅提升。

**展开**

- 评测使用 `temperature=0.6`、`top_p=0.95`、`top_k=20`、`do_sample=true` 和 64K response limit
- SFT 与 async23 使用同一评测数据、解码参数和 budget，因此两者可直接比较
- 不能把 `4/256 -> 9/256` 说成完整 BrowseComp，也不应只强调 125% 相对增幅而隐藏低基数

**引用**

- `examples/carr_deepsearch/docs/eval_analysis_20260415.md`
- `examples/carr_deepsearch/docs/resume_source_of_truth_20260412.md`

### Q67. 如果面试官说“外部 benchmark 正样本太少，怎么证明项目有效”，怎么答？

**短答**

我会把系统证据、内部质量证据和外部泛化证据分开讲。系统侧已经证明 async 主线可训练、可恢复、可评测；内部 `DeepDive 111` 的正确数从 `20/111` 提升到 `36/111`；外部 `BrowseComp subset256 64k` 的 judge-pass 从 `4/256` 提升到 `9/256`。外部正样本仍少，所以它是 supporting evidence，而不是 SOTA 或完整论文复现结论。

**展开**

- 不把小样本外部结果单独当成充分证明，而是看系统、内部评测和外部方向是否形成一致证据链
- `DeepDive 111` 的绝对增益更适合做质量 headline，BrowseComp 子集只做外部 supporting evidence
- 若继续投入 GPU，最有价值的是补 full benchmark 降低方差，而不是重复当前子集

**引用**

- `examples/carr_deepsearch/docs/resume_source_of_truth_20260412.md`
- `examples/carr_deepsearch/docs/eval_analysis_20260415.md`

### Q68. 如果面试官问“best checkpoint 是哪个”，怎么答最稳妥？

**短答**

要分两层回答：

- 最终 durable checkpoint：`global_step_23`
- 最佳内部训练窗口：`step16`

不能把这两个概念混成一个。

**引用**

- `examples/carr_deepsearch/docs/resume_source_of_truth_20260412.md:263-295`

### Q69. 如果面试官问“你最满意的一组训练信号是什么”，怎么答？

**短答**

我最满意的是 `step16` 这个内部窗口：`outcome_reward=0.59375`、`rubric_reward=0.09308`、`task_unfinished=0.29167`、`param_sync=1.69s`，而且是在 async 主线上拿到的。

**展开**

- 这说明模型行为和系统状态在那个窗口里同时比较健康
- 但要立刻补一句：这是训练窗口，不是 formal external eval

**引用**

- `examples/carr_deepsearch/docs/resume_source_of_truth_20260412.md:280-294`
- `examples/carr_deepsearch/CaRR_log/formal_mainline_gs14_sp2_dynbsz_gmu05_b3_reasonfix_20260323_215746.launcher.log:8730-8741`

---

## 7. 限制、后续工作与产品化追问

### Q70. 这个项目当前最大的未完成项是什么？

**短答**

当前简历闭环没有必须补做的训练或评测。若把项目继续扩展成更完整的研究复现，优先级最高的可选项是 `BrowseComp full 64k`，其次是 128K 上下文评测和本地 GRPO/C-GRPO 消融。

**引用**

- `examples/carr_deepsearch/docs/eval_analysis_20260415.md`
- `examples/carr_deepsearch/docs/resume_source_of_truth_20260412.md`

### Q71. 如果你现在拿到新 GPU，第一优先级会做什么？

**短答**

如果目标仍是提高证据强度，我会先确认现有 SFT 和 async23 checkpoint 可用，然后在相同 sampled recipe 下补 `BrowseComp full 64k` 的成对评测；只有结果仍为正且预算充足时，再考虑 128K 评测。已经完成的 `DeepDive 111` 和 `BrowseComp subset256 64k` 不需要为了简历闭环机械重跑。

**展开**

- 仍要按 runbook 做最小 preflight：tool server health check -> reward server health check -> `smoke_test.py --all` -> checkpoint 检查
- 先确认环境、密钥、SGLang / CUDA / Ray / flashinfer 正常
- 评测前所有 checkpoint 必须先 merge 成 `huggingface_merged`
- `BrowseComp` 不能直接拿 `run_eval_browsecomp.sh` 当正式结果，因为那个脚本固定 greedy；正式评测要走 `run_eval_integration.sh` 或 sampled wrapper
- SFT 与 async23 必须使用同一 full 数据集、解码参数、response limit 和 budget

**引用**

- `examples/carr_deepsearch/docs/eval_analysis_20260415.md`
- `examples/carr_deepsearch/docs/post_gpu_revalidation_plan_20260412.md`
- `examples/carr_deepsearch/scripts/smoke_test.py:15-25`

### Q72. 如果再给你一周，你最想补哪三个方向？

**短答**

1. 补 `BrowseComp full 64k` 的 matched sampled eval，降低当前 subset256 的方差
2. 继续压 `update_actor` 和 `param_sync` 尾部  
3. 在预算允许时补 GRPO/C-GRPO 消融或研究更稳的 `128k` 配置

**展开**

- 第一点补“对外结果”
- 第二点补“infra 完成度”
- 第三点补“test-time scaling / longer-context” 能力

**引用**

- `examples/carr_deepsearch/docs/post_gpu_revalidation_plan_20260412.md:185-190`
- `examples/carr_deepsearch/docs/post_gpu_revalidation_plan_20260412.md:196-218`

### Q73. 如果你要把这个系统往产品化推进，会监控哪些指标？

**短答**

我会同时监控四类指标：结果质量、轨迹质量、系统吞吐、失败原因。

**展开**

- 结果质量：`outcome_reward`, `rubric_reward`
- 轨迹质量：`task_unfinished`, `response_length_ratio`, `num_turns`, `tool_call_counts`
- 系统吞吐：`timing_s/gen`, `timing_s/update_actor`, `timing_s/param_sync`, `trainer/idle_ratio`
- 失败原因：`termination_response_limit`, `termination_rollout_timeout`, `termination_search_budget`, `unfinished_*`

**引用**

- `examples/carr_deepsearch/reward/carr_reward.py:104-142`
- `examples/carr_deepsearch/docs/async_rollouter_pause_mechanism_explainer.md:745-764`

### Q74. 你觉得这个项目最大的系统风险还剩什么？

**短答**

最大的残留系统风险是：`param_sync` 仍有长尾，trainer `update_actor` 仍是主瓶颈，而且当前 external evidence 只覆盖 `BrowseComp subset256 64k`，正样本数量较少。

**引用**

- `examples/carr_deepsearch/docs/resume_source_of_truth_20260412.md:263-295`
- `examples/carr_deepsearch/docs/post_gpu_revalidation_plan_20260412.md:224-239`

### Q75. 这个方法能迁移到别的 agent task 吗？

**短答**

能，但前提是目标任务也有“过程质量比单纯 outcome 更重要”的结构，并且你能定义可判定的中间约束。CaRR 的 citation-aware rubric 适合 deep search；别的任务需要换成自己的中间过程信号。

**展开**

- 框架层面是通用的：自定义 agent loop、reward bridge、advantage、async rollout
- 奖励语义不一定通用：rubric 设计要跟任务结构匹配
- 对 code agent、web agent、workflow agent 都有迁移空间

**引用**

- `docs_zh/agent_training/01_架构设计.md:43-49`
- `examples/carr_deepsearch/docs/resume_source_of_truth_20260412.md:91-176`

### Q76. 如果你继续做 128k，会最担心什么？

**短答**

我最担心三件事：rollout wall time 再次上升、trainer 侧长序列显存峰值重新爆炸、以及 external eval 成本过高导致验证节奏过慢。

**展开**

- `128k` 不是简单把 `max_response_length` 翻倍
- rollout、trainer、eval 三条链都会一起变贵
- 当前 `BrowseComp subset256 64k` 已经给出正向结果；若继续投入，应先用 full 64k 确认低方差趋势，再决定是否加测 `128k`

**引用**

- `examples/carr_deepsearch/docs/post_gpu_revalidation_plan_20260412.md:185-190`
- `examples/carr_deepsearch/docs/resume_source_of_truth_20260412.md:330-337`

---

## 8. 快问快答：容易被追问的小细节

### Q77. `prompt_ids` 和 `response_ids` 的区别是什么？

`prompt_ids` 是初始 prompt token；`response_ids` 是 rollout 过程中累计的响应 token，里面既可能有 LLM 生成，也可能有工具响应。  
**引用**：`docs_zh/agent_training/02_AgentLoop详解.md:69-81`

### Q78. `response_mask=0` 的 token 会参与 loss 吗？

不会。`response_mask=0` 表示工具或环境返回的 token，只作为上下文，不是策略分布的一部分。  
**引用**：`docs_zh/agent_training/02_AgentLoop详解.md:83-99`

### Q79. 为什么 `reward_history` 的最后一条必须是 assistant？

否则 reward 侧会把它视为 unfinished 或 fallback，因为说明轨迹没有形成一个可评估的最终 assistant answer。  
**引用**：`examples/carr_deepsearch/reward/carr_reward.py:74-81`

### Q80. `task_unfinished` 和 `termination_reason` 有什么区别？

`task_unfinished` 是“这条轨迹没做完”的总标记；`termination_reason` 是具体截断原因，比如 `response_limit`、`rollout_timeout`、`search_budget`。  
**引用**：`examples/carr_deepsearch/tools/carr_agent_loop.py:125-171`

### Q81. 为什么 `score` 先等于 `outcome_reward`？

因为最终 reward fusion 在 `C-GRPO` advantage 阶段做，reward function 这里只负责把 `outcome` 和 `rubric` 都透传出来。  
**引用**：`examples/carr_deepsearch/reward/carr_reward.py:171-179`

### Q82. 为什么 eval gate 里固定 `temperature=0.6` 而不是 0？

因为对 Thinking checkpoint，sampled recipe 更 faithful，也更便宜；greedy 会放大 collapse failure mode。  
**引用**：`examples/carr_deepsearch/docs/rl_debug_findings_20260312.md:847-912`

补一句工程细节：`run_eval_browsecomp.sh` 和通用 `run_eval.sh` 默认都是 greedy 配置，所以不能直接拿来给 Thinking checkpoint 下正式结论；正式 sampled gate 要走 `run_eval_integration.sh` 或 sampled wrapper。  
**引用**：`examples/carr_deepsearch/scripts/run_eval_browsecomp.sh:115-128`; `examples/carr_deepsearch/scripts/run_eval.sh:143-160`; `examples/carr_deepsearch/scripts/run_eval_integration.sh:145-177`

### Q83. 为什么要单独记录 `search_count/open_count/find_count`？

因为 deep-search agent 的学习信号不只在最终 reward，还体现在工具使用行为是否更合理。  
**引用**：`examples/carr_deepsearch/reward/carr_reward.py:114-117`

### Q84. 为什么 `response_length_ratio` 值得看？

它能帮助你区分“模型是真的更会搜证据了”还是“只是越来越容易打满上限”。  
**引用**：`examples/carr_deepsearch/reward/carr_reward.py:139-142`

### Q85. 为什么 formal launcher 要强写一堆默认值？

因为 async 行为对小差异很敏感。把关键默认值固化到 launcher 层，能防止 run 之间因为隐式继承而失去可比性。  
**引用**：`examples/carr_deepsearch/scripts/run_async_formal.sh:202-260`

### Q86. 为什么 `global_step_23` 不等于 “最优质量 checkpoint”？

因为 durable checkpoint 是“最终稳定写盘的训练端点”，而最优质量可能出现在中间某个内部窗口，例如 `step16`。  
**引用**：`examples/carr_deepsearch/docs/resume_source_of_truth_20260412.md:263-295`

### Q87. 为什么不把 `step90` 说成退化？

因为它不是全面退化。它的 `rubric_reward` 更高，只是按 outcome-first gate 规则不适合作 async 起点。  
**引用**：`examples/carr_deepsearch/docs/resume_source_of_truth_20260412.md:224-250`

### Q88. 你项目里最能体现“不是黑盒调参”的一个点是什么？

我会选 unfinished / termination 的原因拆分，因为它把“为什么失败”从黑盒 reward 里拉了出来，直接变成可解释的系统信号。  
**引用**：`examples/carr_deepsearch/tools/carr_agent_loop.py:125-171`; `examples/carr_deepsearch/reward/carr_reward.py:104-142`

---

## 9. 面试时可以主动抛出的加分点

- 我不是把 CaRR 当成“论文里的 reward 名词”来讲，而是把它拆成了 `outcome reward`、citation-grounded `rubric reward`、evidence chain、组内归一化和“只奖励正确 rollout”这几个可实现的设计点。
- 我不会把 async 主线讲成“速度一定更快”，而是讲成“通过可解释的失败模式定位，把系统从 timeout / OOM / stall 推到可训练、可恢复、可评测，再在健康窗口里看到吞吐改善”。
- 我会明确说外部实测范围是 `BrowseComp subset256 64k`，并同时报告 `4/256 -> 9/256` 的分子和分母，不把低基数相对提升包装成完整 benchmark 结论。

**引用**

- `examples/carr_deepsearch/docs/resume_source_of_truth_20260412.md`
- `examples/carr_deepsearch/docs/post_gpu_revalidation_plan_20260412.md`

---

## 10. 深水区追问：更偏论文细节、verl 内部机制与口径边界

### Q89. 如果面试官质疑“LLM judge 本身会不会不稳定”，你怎么答？

**短答**

会有噪声，所以 CaRR 不是让 judge 直接打一个整体主观分，而是把判断拆成实体识别、citation support、evidence connectivity 三步。当前本地没有单独重跑 judge-agreement 统计，但论文人工核验给过 prior：hidden entity identification `97.7%`、citation-rubric judgment `95.1%`。我会把这当作外部 prior，而不是 claim 本地 judge 已复验。

**展开**

- reward 不是单点黑盒分数，而是三步判定后的组合结果
- rubric 还要被 `outcome_reward` gating，错误 rollout 不会因为 judge 给了局部过程分就被正向强化
- unfinished / history 格式错误会直接走 0 分兜底，减少 judge 噪声传播

**引用**

- `CaRR/docs/02_CaRR奖励框架详解.md:54-142`
- `CaRR/docs/02_CaRR奖励框架详解.md:185-193`
- `examples/carr_deepsearch/reward/cgrpo_advantage.py:72-112`
- `CaRR/docs/08_实验结果与消融分析.md:197-201`

### Q90. 为什么 CaRR 要把引用 URL 限制在最多 20 个？

**短答**

这是为了同时控制 judge 上下文成本和防止 citation flooding。否则模型可以靠塞大量引用 URL 来“刷支持证据”，反而让 judge 更不稳定。

**展开**

- 这是显式对抗 rubric hacking 的设计，而不只是省 token
- URL 提取后还会去重，避免同源重复刷分
- 这个 cap 说明 reward 设计不仅管“对不对”，也管“怎么防作弊”

**引用**

- `CaRR/docs/02_CaRR奖励框架详解.md:88-100`

### Q91. 为什么 `reward_history` 里要维护 `tool_call_id` 绑定？

**短答**

因为一个 assistant turn 里可能发多个并行 tool call。没有 `tool_call_id`，reward 侧就无法可靠知道哪段 tool output 对应哪个调用，citation chain 也会变得不可信。

**展开**

- assistant message 里会给每个 tool call 生成稳定的 `tool_call_id`
- tool 返回时再把 `tool_call_id -> output` 绑定写回 `reward_history`
- 这让 reward server 能按同一轮调用关系重建可判定历史，而不是只看无结构文本

**引用**

- `examples/carr_deepsearch/tools/carr_agent_loop.py:357-378`
- `examples/carr_deepsearch/tools/carr_agent_loop.py:522-572`

### Q92. async 里怎么保证一个 prompt 的 `n` 条 rollout 还是一组，不会被打散成乱序样本？

**短答**

每个输入 prompt 在进入 async rollouter 时会先 `repeat(interleave=True)` 成 `n` 条轨迹，随后被包装成一个 `RolloutSample`。trainer 端消费的是 `RolloutSample`，不是单条 trajectory，因此组结构在跨队列传输过程中是保留下来的。

**展开**

- `prepare_single_generation_data()` 先把单条样本扩成 `n` 个 rollout
- `RolloutSample` 里保存这整个 prompt-group 的 batch 和 `agent_loop_output_list`
- assemble 时再按 sample 级拼回来，所以 `C-GRPO` 的组内归一化前提没有丢

**引用**

- `verl/experimental/fully_async_policy/detach_utils.py:27-47`
- `verl/experimental/fully_async_policy/detach_utils.py:60-97`
- `verl/experimental/fully_async_policy/detach_utils.py:100-196`

### Q93. `MessageQueue` 满了以后会阻塞还是丢样本？

**短答**

最后一道保险是“丢最老样本”，不是无限阻塞。`MessageQueue` 本身的行为是满了就 `popleft()` 最旧样本；但正常主线不应该靠这个工作，而是 rollouter 在更上层先感知 `queue_full` 并暂停 / partial drain。

**展开**

- queue actor 用 `deque(maxlen=max_queue_size)`
- 真正想要的是 rollouter 的 backpressure 先起作用，而不是等 queue actor 被动丢样本
- 所以“会不会 drop”这个答案是：机制上会，但设计目标是尽量别走到那一步

**引用**

- `verl/experimental/fully_async_policy/message_queue.py:32-40`
- `verl/experimental/fully_async_policy/message_queue.py:78-96`
- `verl/experimental/fully_async_policy/fully_async_rollouter.py:858-870`

### Q94. `FullyAsyncRollouter` 真正跑起来时，最关键的三个 coroutine 是哪三个？

**短答**

`_feed_samples()`、`_processor_worker()`、`_async_monitor_loop()`。

**展开**

- `_feed_samples()`：持续把 dataloader 样本转换成 `RolloutSample` 推进 `pending_queue`
- `_processor_worker()`：决定从 `pending_queue/cancel_queue` 取什么、何时 dispatch、何时因 `queue_full/staleness` 暂停
- `_async_monitor_loop()`：周期性打印状态，并在 pause reason 消失后触发恢复

**引用**

- `verl/experimental/fully_async_policy/fully_async_rollouter.py:529-566`
- `verl/experimental/fully_async_policy/fully_async_rollouter.py:568-699`
- `verl/experimental/fully_async_policy/fully_async_rollouter.py:829-896`

### Q95. 为什么 partial 模式下要优先消费 `cancel_queue`，而不是一直喂新样本？

**短答**

因为 partial rollout 的价值就在于“把被 cancel 的长轨迹接着跑完”。如果恢复后还先喂新样本，旧样本就会一直堆着，staleness 和 unfinished 都会变差。

**展开**

- processor 会优先检查 `cancel_queue`
- `_get_pause_reason()` 里也专门允许“即使有 cancel_queue，也先恢复 dispatch”
- 这说明 partial 模式不是单纯 cancel，而是 cancel-then-resume unfinished trajectories

**引用**

- `verl/experimental/fully_async_policy/fully_async_rollouter.py:650-669`
- `verl/experimental/fully_async_policy/fully_async_rollouter.py:871-895`

### Q96. `old_log_prob` 为什么默认直接用 rollout 侧的，而不是 trainer 重新算？

**短答**

因为 `old_log_prob` 和 rollout 时的参数版本、token 序列是强绑定的。异步场景下如果 trainer 侧用错了版本或重建了不同 token 序列，importance ratio 就不再严格正确。

**展开**

- 默认 `use_rollout_log_probs=True`
- `bypass_mode=True` 时直接用 rollout 记录下来的 log prob
- 如果开 `bypass_mode=False`，trainer 会按 local trigger step 把旧版本参数恢复出来重算 `old_log_prob`，这更接近 rollout importance sampling / decoupled PPO

**引用**

- `verl/experimental/fully_async_policy/README_zh.md:127-141`
- `verl/experimental/fully_async_policy/fully_async_trainer.py:488-508`

### Q97. async validation 是怎么拆给 rollout 和 trainer 的？

**短答**

有两种模式。默认是 rollouter 做 validate，然后把结果塞进 `val_queue`；如果 `use_trainer_do_validate=True`，trainer 自己持有验证集切片并在参数同步后做 validate。

**展开**

- trainer 模式下，验证集会按总 GPU 数 split，再把 rollout GPU 对应前几份剔掉，只保留 trainer 负责的那部分
- rollouter 模式下，validate 结果通过 `put_validate/get_validate` 走 message queue 回传
- 这也是为什么这个开关不只是“谁算 val”，而是会影响资源分配和时间重叠方式

**引用**

- `verl/experimental/fully_async_policy/fully_async_trainer.py:146-175`
- `verl/experimental/fully_async_policy/fully_async_rollouter.py:278-307`
- `verl/experimental/fully_async_policy/param_sync.py:192-213`
- `verl/experimental/fully_async_policy/README_zh.md:163-167`

### Q98. 为什么 async checkpoint 目录编号跟的是 `current_param_version`，不是原始 `global_steps`？

**短答**

因为 async 真正的“稳定边界”是参数同步边界，不是每个 local update。一个 `current_param_version` 对应一次完整的 sync/save 语义，所以 checkpoint、日志聚合和恢复都按它编号更一致。

**展开**

- save 路径直接写成 `global_step_{current_param_version}`
- resume 时会反推 `global_steps = current_param_version * trigger_sync_step + 1`
- 这也是为什么你要向面试官说明：这里的 `global_step_23` 实际上代表 23 次 param sync，不是普通同步训练里那种一步一更

**引用**

- `verl/experimental/fully_async_policy/fully_async_trainer.py:560-569`
- `verl/experimental/fully_async_policy/fully_async_trainer.py:617-621`
- `verl/experimental/fully_async_policy/fully_async_trainer.py:657-659`

### Q99. 为什么你们经常强调 `huggingface_merged`，而不是直接拿 FSDP shard 去 eval？

**短答**

因为 eval / model-only restart 需要的是标准 HuggingFace 权重目录，而训练中落盘的是 FSDP shard。launcher 会优先解析 checkpoint root、`huggingface_merged`、`actor/huggingface_merged` 这几种位置，只有解析成 HF 模型目录后才会继续。

**展开**

- 这也是之前某次 model-only restart 报 “no `pytorch_model.bin` / `model.safetensors`” 的根因
- `eval_gate` 和训练模式在入口上都共用这一层路径解析
- 所以“能不能 eval”不是只看有没有 checkpoint，而是看有没有 merge 成 HF 资产

**引用**

- `examples/carr_deepsearch/scripts/run_async_formal.sh:45-78`
- `examples/carr_deepsearch/scripts/run_async_formal.sh:91-159`
- `examples/carr_deepsearch/scripts/run_async_formal.sh:167-178`

### Q100. `unfinished_limit` 除了 `response_limit` 之外，还可能包含什么？

**短答**

还包括 `assistant_turn_limit` 和 `user_turn_limit`。reward 侧会把这些都归到 `unfinished_limit` 大类里，同时再单独保留更细的 `termination_*` 字段。

**展开**

- agent loop 在生成结束时会区分 `response_limit`、`assistant_turn_limit`、`user_turn_limit`
- reward bridge 里再用 `known_limit_reasons` 做统一归并
- 所以如果看到 `unfinished_limit` 高，不要自动等价成“全是 64k 打满”

**引用**

- `examples/carr_deepsearch/tools/carr_agent_loop.py:330-340`
- `examples/carr_deepsearch/tools/carr_agent_loop.py:463-487`
- `examples/carr_deepsearch/reward/carr_reward.py:84-95`
- `examples/carr_deepsearch/reward/carr_reward.py:121-133`

### Q101. 你会怎么解释 `C-GRPO` 和 `E-GRPO` 的区别？

**短答**

我会说：`E-GRPO` 更像 outcome 加实体匹配率，通常需要更强的中间标注；`C-GRPO` 则是 outcome 加 citation-aware rubric reward，只对正确 rollout 生效，而且不依赖人工金标中间实体。

**展开**

- `C-GRPO` 的关键不是“多一个 reward”，而是 citation grounding + connectivity + outcome gating 这三件事同时成立
- 这让它更适合 deep-search 这类“过程是否被证据支撑”非常重要的任务
- 也是为什么你可以把它讲成“research design + engineering implementation”的结合点

**引用**

- `CaRR/docs/03_C-GRPO训练算法详解.md:218-230`
- `CaRR/docs/03_C-GRPO训练算法详解.md:234-246`

### Q102. 如果面试官问“为什么不用 checkpoint engine 再继续压 param_sync”，怎么回答？

**短答**

可以说：框架文档里 checkpoint engine 确实给出了“同步时间开销可降低 60%+”的 prior，但它会引入额外临时显存开销。这个项目阶段优先解决的是 queue_full backpressure 和 partial pause/resume 正确性，因为当时长尾主因并不只是广播本身。

**展开**

- 这不是说 checkpoint engine 没价值，而是当时不是第一优先级
- 在长序列 + SP + dynbsz 的主线里，额外显存余量本来就敏感
- 所以后续可以把它当 next-step optimization，而不是当前成果的一部分

**引用**

- `verl/experimental/fully_async_policy/README_zh.md:149-161`

### Q103. 对 `queue_full` 修复，你最稳妥的 claim 边界是什么？

**短答**

最稳妥的说法是：我修掉了一个真实会把 `param_sync` 拖成分钟级长尾的 backpressure 路径，并在后续主线里观察到明显收敛；但我不会说“从此完全没有任何长尾”。

**展开**

- `323s -> 1.7-20s` 是健康窗口里的真实改善
- final mainline 和更早 probe 仍有约 `85s / 102s / 144s / 209s` 的残余长尾，所以不能吹成“完全根治”
- 正确讲法是：`queue_full` partial drain 不再是系统性 blocker，剩余尾部更多和样本长尾、当前轮次回流有关

**引用**

- `examples/carr_deepsearch/docs/agent_handoff_background_20260321.md:168-171`
- `examples/carr_deepsearch/docs/resume_source_of_truth_20260412.md:282-297`

### Q104. 为什么 `BrowseComp` 不能直接跑 `run_eval_browsecomp.sh` 当正式结论？

**短答**

因为那个脚本固定的是 greedy recipe：`temperature=0`、`do_sample=false`。对 Thinking checkpoint，这不是你想要的正式 gate，会系统性放大 repetition / unfinished failure mode。

**展开**

- `run_eval_browsecomp.sh` 硬写了 `temperature=0`、`do_sample=false`
- 本项目对 Thinking checkpoint 的统一结论口径是 sampled eval：`temperature=0.6, top_p=0.95, top_k=20, do_sample=true`
- 所以正式外部 gate 应该走 `run_eval_integration.sh` 或 sampled wrapper，并确保模型目录是 `huggingface_merged`

**引用**

- `examples/carr_deepsearch/scripts/run_eval_browsecomp.sh:115-128`
- `examples/carr_deepsearch/scripts/run_eval_integration.sh:145-177`
- `examples/carr_deepsearch/docs/rl_debug_findings_20260312.md:847-912`

### Q105. 历史 async 起点选择和后续 `async_best` 选择为什么规则不同？

**短答**

因为它们回答的是两个不同问题。历史 `SFT / step70 / step90` gate 的目标是给 async 找一个安全起点，所以规则更简单、更 outcome-first；后续 `async_best` 选择是简历收尾阶段的 checkpoint-selection policy，会在 outcome 接近时更重视 `rubric_reward` 和 finer-grained unfinished diagnostics。

**展开**

- 历史脚本规则：`outcome -> unfinished -> timeout`，`<0.03` 才 tie-break
- 新 GPU 收尾计划里：`outcome` 接近到 `<0.02` 时先看 `rubric_reward`，再看 `task_unfinished / termination_response_limit / termination_rollout_timeout`
- 这不是自相矛盾，而是阶段目标不同

**引用**

- `examples/carr_deepsearch/scripts/select_async_start_checkpoint.py:12-18`
- `examples/carr_deepsearch/scripts/select_async_start_checkpoint.py:93-136`
- `examples/carr_deepsearch/docs/post_gpu_revalidation_plan_20260412.md:129-140`

### Q106. RL parquet schema contract 是什么？

**短答**

parquet 层面最核心的列是：`data_source`、`agent_name`、`prompt`、`ability`、`reward_model`、`extra_info`。其中 `agent_name` 选 agent loop，`reward_model.ground_truth` 进 reward，`extra_info.rubrics/search_forbidden_strs/rubric_reward_ratio` 进 reward server，`extra_info.tools_kwargs.*.create_kwargs` 进 tool session。`raw_prompt` 是 `return_raw_chat=true` 后进入 agent loop 的 runtime form，不是 parquet 列。

**展开**

- parquet 列名是 `prompt`，它保存 chat messages 列表
- `raw_prompt` 是运行时形态：dataset/worker 在 `return_raw_chat=true` 路径下把 `prompt` 以 raw chat 形式送进 agent loop
- `agent_name` 决定 registry 里走哪个 agent loop
- `reward_model.ground_truth` 给 `compute_score()` 做 outcome 判断
- `extra_info` 里静态 rubric / forbidden strings 跟 agent loop 动态产出的 `tool_extra_fields` 会在 reward manager 汇合

**引用**

- `docs_zh/carr_walkthrough/02_数据预处理.md:130-179`
- `docs_zh/carr_walkthrough/02_数据预处理.md:205-229`
- `verl/experimental/reward_loop/reward_manager/naive.py:42-52`
- `examples/carr_deepsearch/reward/carr_reward.py:65-80`

### Q107. 为什么必须做 request-level session manager，而不能依赖 BaseTool 生命周期？

**短答**

因为 `ToolAgentLoop` 的工具对象生命周期是“每次调用 create -> execute -> release”，而 CaRR 的 `search/open/find` 需要共享同一个 server-side sandbox/session。session 如果绑在单次 tool instance 上，跨调用状态会丢。

**展开**

- `CaRRSessionManager` 以 `request_id` 为 session id，首次调用时 lazy start
- `create()` 和 `release()` 都是 no-op，真正的 close 在 agent loop `finally` 里统一做
- 这样同一条 rollout 里的 `search -> open -> find` 才能共享一个 server-side state

**引用**

- `examples/carr_deepsearch/tools/carr_session_manager.py:17-24`
- `examples/carr_deepsearch/tools/carr_session_manager.py:56-76`
- `examples/carr_deepsearch/tools/carr_browser_tool.py:21-25`
- `examples/carr_deepsearch/tools/carr_browser_tool.py:45-49`
- `examples/carr_deepsearch/tools/carr_browser_tool.py:87-88`

### Q108. 为什么 async 训练要关掉 `hybrid_engine`，改成 trainer/rollout 资源隔离？

**短答**

因为这个项目的目标是让 rollout 和 trainer 真正并行重叠。`hybrid_engine` 更适合同卡位的串行切换；fully async 则要求 trainer pool 和 rollout pool 分离，分别管理资源和节奏。

**展开**

- async README 明确把“资源隔离”列成核心特性
- 代码里 trainer 和 rollouter 都直接 `assert not hybrid_engine`
- launcher 也会显式把 `actor_rollout_ref.hybrid_engine=false` 写进去，避免隐式继承

**引用**

- `verl/experimental/fully_async_policy/README_zh.md:33`
- `verl/experimental/fully_async_policy/fully_async_trainer.py:70-72`
- `verl/experimental/fully_async_policy/fully_async_rollouter.py:66-69`
- `examples/carr_deepsearch/scripts/run_rl_async.sh:201-205`

### Q109. 为什么 rollout backend 选 `SGLang`，而不是 `vLLM`？

**短答**

不是因为框架不支持 `vLLM`，而是因为这条项目主线实际验证、修补和稳定化的是 `SGLang` 路径，尤其是 partial rollout、在线权重同步、no-flush update 和 KV cache 清理这几块。

**展开**

- fully async agent loop manager 同时支持 `sglang` 和 `vllm`
- 但项目配置明确选的是 `sglang`
- handoff 中真正被 runtime 验证过的修复也都是围绕 SGLang 在线权重同步、flush/no-flush、clear_kv_cache 展开

**引用**

- `verl/experimental/fully_async_policy/agent_loop/agent_loop.py:246-259`
- `examples/carr_deepsearch/config/carr_grpo_async_common.yaml:33-35`
- `examples/carr_deepsearch/docs/agent_handoff_background_20260321.md:105-128`

### Q110. 为什么训练后端选 `FSDP`，而不是 `Megatron`？

**短答**

同样不是说 `Megatron` 不行，而是你这条已知稳定主线的 dynbsz、SP、checkpoint merge、resume、OOM 排障全都围绕 `FSDP` 完成，面试里应该如实讲“validated path is FSDP”。

**展开**

- 项目配置里的 actor strategy 是 `fsdp`
- 关键修复点也是 `dynbsz + FSDP` 的 DP-rank micro-batch 对齐
- 如果没在本项目里真正把 `Megatron` 这条线跑稳，就不要把它说成你的主要实现资产

**引用**

- `examples/carr_deepsearch/config/carr_grpo_async_common.yaml:20-27`
- `examples/carr_deepsearch/config/carr_grpo.yaml:31-35`
- `examples/carr_deepsearch/docs/agent_handoff_background_20260321.md:124-125`
- `examples/carr_deepsearch/docs/agent_handoff_background_20260321.md:625-627`

### Q111. 为什么 trainer 用 `SP=2`，但 rollout 固定 `TP=1`？

**短答**

因为这两个旋钮解决的是完全不同的问题。trainer 侧 `SP=2` 是为了压长序列 backward 峰值、避免 OOM；rollout 侧 `TP=1` 是为了保持 serving/simple sync 路径稳定，降低并行复杂度。

**展开**

- handoff 里 `SP=1` 的 2:6 / 3:5 probe 都 OOM 了，所以 trainer 需要 `SP=2`
- rollout 侧主线一直是 `TP=1`
- launcher 里 eval 和 formal 训练都把 rollout TP 显式固定为 1

**引用**

- `examples/carr_deepsearch/docs/agent_handoff_background_20260321.md:661-663`
- `examples/carr_deepsearch/docs/agent_handoff_background_20260321.md:183-190`
- `examples/carr_deepsearch/scripts/run_eval_integration.sh:154-156`
- `examples/carr_deepsearch/scripts/run_async_formal.sh:241-245`

### Q112. 为什么 `SFT / RL / eval` 都要显式 `enable_thinking=true`？

**短答**

因为这是 Qwen3 prompt/render 行为的一部分。如果 SFT、RL、eval 对 `enable_thinking` 处理不一致，实际看到的 system prompt 和 generation behavior 就会漂移，评测也不再同口径。

**展开**

- SFT 数据通过 `enable_thinking_key` 从 parquet 读
- RL / async config 在 `apply_chat_template_kwargs` 里显式打开 `enable_thinking`
- `verify_tool_consistency.py` 也用 `enable_thinking=True` 去比对 SFT/RL 真实渲染路径

**引用**

- `examples/carr_deepsearch/config/carr_sft.yaml:29-34`
- `examples/carr_deepsearch/config/carr_grpo.yaml:13-23`
- `examples/carr_deepsearch/config/carr_grpo_async_common.yaml:10-12`
- `examples/carr_deepsearch/scripts/verify_tool_consistency.py:76-82`

### Q113. 你怎么保证 `SFT / RL / eval` 使用同一份 tool schema 和 system prompt？

**短答**

我不是靠“肉眼觉得差不多”，而是做了真实执行路径的一致性校验：SFT 从 parquet 走一遍 schema normalize，RL 从 YAML 走一遍 model_validate + normalize，再用同一个 tokenizer 和 `enable_thinking=true` 渲染 system prompt 做逐字比对。

**展开**

- tool config YAML 还要求所有 key 按字母序排列，因为 parquet round-trip 会排序 struct field
- 如果 YAML 顺序和 SFT parquet 里的 schema 不一致，最终 system prompt 就会漂
- `verify_tool_consistency.py` 就是为这个问题写的

**引用**

- `examples/carr_deepsearch/config/tool_config/carr_browser_tools.yaml:6-12`
- `examples/carr_deepsearch/scripts/verify_tool_consistency.py:14-18`
- `examples/carr_deepsearch/scripts/verify_tool_consistency.py:61-82`
- `examples/carr_deepsearch/scripts/verify_tool_consistency.py:100-118`

### Q114. 为什么 `browser.open.id` 在 schema 里写成 string，运行时再转 int？

**短答**

因为 `verl` 的 `OpenAIFunctionPropertySchema` 不支持 `oneOf`。`browser.open` 既可能接收 search result id，也可能接收 URL，所以 schema 层统一写 string，执行层再尝试转 int。

**展开**

- YAML 注释里已经写明了这个约束
- `CaRRBrowserTool.execute()` 会在 `browser.open` 且 `id` 存在时尝试 `int()`，失败就保留原字符串，当作 URL
- 这是工具 schema 与真实运行时兼容性之间的工程折中

**引用**

- `examples/carr_deepsearch/config/tool_config/carr_browser_tools.yaml:6-8`
- `examples/carr_deepsearch/tools/carr_browser_tool.py:70-75`

### Q115. 为什么 reward 走 `custom_reward_function + 外部 reward server`，而不是 `reward_model.enable=true`？

**短答**

因为这里需要的是结构化、带 history 和 rubric 的 rule/server-driven reward，不是一个本地 learned reward model。项目里真正的 reward 逻辑在 `carr_reward.py -> /evaluate`，所以配置明确是 `reward_model.enable=false`，同时挂 `custom_reward_function.path`。同时它还有明确的 failure fallback：客户端 `CARR_REWARD_TIMEOUT=650` 故意大于 server 内部 `600s`，如果 HTTP 非 `200` 或请求异常，就回退成 `score=0/outcome_reward=0/rubric_reward=0`，但保留 unfinished / termination / tool-use 的 pass-through 指标。

**展开**

- RL config 里 `reward_model.enable=false`
- `custom_reward_function` 指向 `carr_reward.py`
- `run_rl_async.sh` 和 `run_eval_integration.sh` 也都显式把这条路径写进去
- `carr_reward.py` 不是简单返回一个标量，而是把 `messages/history`、`rubrics`、`search_forbidden_strs`、`rubric_reward_ratio` 打成 `/evaluate` payload
- 客户端 timeout 设成 `650s`，故意大于 reward server 内部 `600s`，避免 client 先超时
- 如果 reward server 返回非 `200`，或者 HTTP/JSON 调用异常，reward 会安全回退到零分，但 unfinished、termination_reason、tool_call_counts 这类诊断字段仍然原样 pass through，不会把 failure attribution 一起丢掉

**引用**

- `examples/carr_deepsearch/config/carr_grpo.yaml:69-74`
- `examples/carr_deepsearch/config/carr_grpo_async_common.yaml:65-67`
- `examples/carr_deepsearch/scripts/run_rl_async.sh:237-245`
- `examples/carr_deepsearch/scripts/run_eval_integration.sh:149-154`
- `examples/carr_deepsearch/reward/carr_reward.py:34-37`
- `examples/carr_deepsearch/reward/carr_reward.py:104-142`
- `examples/carr_deepsearch/reward/carr_reward.py:144-169`

### Q116. 为什么 eval 走 `main_ppo + val_only=True`，而不是 `main_eval.py`？

**短答**

因为这里要评的是“真实 agent rollout + tools + reward server”这整条集成链，不是对一份已经生成好的 responses parquet 做离线打分。`main_ppo + val_only=True` 复用了训练时的 rollout/reward 路径，而 `main_eval.py` 是离线 evaluate generated file。

**展开**

- `run_eval_integration.sh` 和 `run_eval.sh` 都走 `python -m verl.trainer.main_ppo ... trainer.val_only=True`
- `main_eval.py` 读的是离线 parquet：`response_key`、`reward_model_key`
- 所以如果想验证 tool calling behavior、unfinished reasons、integration correctness，就该用 `val_only=True` 而不是 `main_eval.py`

**引用**

- `examples/carr_deepsearch/scripts/run_eval_integration.sh:145-177`
- `examples/carr_deepsearch/scripts/run_eval.sh:143-160`
- `verl/trainer/main_eval.py:15-18`
- `verl/trainer/main_eval.py:43-48`

### Q117. `rollout_timeout`、`real_rollout_timeout`、`max_param_span` 分别约束什么？

**短答**

- `rollout_timeout`：只算 active generation / tool loop 的累计 wall time
- `real_rollout_timeout`：从这条样本第一次启动开始的真实经过时间，包含 pause/resume 间隔
- `max_param_span`：一条 partial rollout 跨过的参数版本跨度上限

**展开**

- 这三个约束分别对应“单轮活跃时间太长”“样本在系统里活太久”“跨版本过多导致太 stale”
- async partial 模式里尤其需要后两者，不然样本可能被无限续命
- reward bridge 也会把这三种 termination 单独透传出来

**引用**

- `examples/carr_deepsearch/tools/carr_agent_loop.py:695-710`
- `examples/carr_deepsearch/tools/carr_agent_loop.py:849-851`
- `examples/carr_deepsearch/reward/carr_reward.py:84-95`
- `examples/carr_deepsearch/reward/carr_reward.py:125-138`

### Q118. 为什么 `C-GRPO` 不兼容 `algorithm.use_kl_in_reward=true`？

**短答**

因为 `C-GRPO` 在 advantage 阶段会把 token-level rewards 整个重建成“全 0 + 最后一个 valid token 上的 fused scalar reward”。如果你之前已经把 KL penalty 加进 token rewards，这一步会直接把它丢掉。

**展开**

- `cgrpo_advantage.py` 里有显式 assert
- 所以用 `adv_estimator=cgrpo` 时必须关 `use_kl_in_reward`
- 这是算法语义上的不兼容，不是实现疏漏

**引用**

- `examples/carr_deepsearch/reward/cgrpo_advantage.py:72-82`
- `examples/carr_deepsearch/reward/cgrpo_advantage.py:106-112`
- `examples/carr_deepsearch/config/carr_grpo.yaml:63-67`

### Q119. 为什么要做 completed-answer trimming / `content_early_stopped`？

**短答**

因为模型有时已经把完整 final answer 连同 `## Exact Answer / ## References` 都写出来了，但还在继续无意义生成。如果不及时截断，这类本来应该算“正常完成”的样本会被误记成 `response_limit` unfinished。

**展开**

- trim 逻辑是保守的：必须同时看到 `## Exact Answer` 和 `## References`，且没有 tool-call marker
- 还要通过 decode/encode round-trip 和 prefix 一致性检查
- 成功 trim 后会打 `content_early_stopped=true`，并把 completion 归到 finished 而不是 unfinished_limit

**引用**

- `examples/carr_deepsearch/tools/carr_agent_loop.py:173-254`
- `examples/carr_deepsearch/tools/carr_agent_loop.py:347-356`
- `examples/carr_deepsearch/reward/carr_reward.py:73-83`

### Q120. `SGLang flush_cache failed: empty response` 是什么风险？为什么通常直接 resume 就行？

**短答**

这是一次 param-sync 后的 rollout server 偶发性基础设施故障，不是训练语义本身崩了。它会中断当前 run，但已经落盘的 checkpoint 仍然可用，所以通常直接 resume 就能继续。

**展开**

- handoff 里明确把它归类成“已知偶发问题”
- 如果频率很低，工程上直接 resume 的性价比最高
- 只有当它频繁出现时，才值得去给 `ensure_sglang_flush_cache_succeeded()` 加重试或更深排障

**引用**

- `examples/carr_deepsearch/docs/agent_handoff_background_20260321.md:160`
- `examples/carr_deepsearch/docs/agent_handoff_background_20260321.md:668-669`

---

## 11. 设计选择、步骤安排与消融追问

### Q121. 为什么项目步骤是“先同步 RL 跑通，再迁 async，再做内部 gate，最后才做外部 gate / continuation”？

**短答**

因为这几步各自解决的是不同层级的风险，不能混在一起。先用同步 RL 证明 reward / agent loop / 数据链路是对的，再用 async 解决长轨迹吞吐和稳定性，随后先用便宜的内部 gate 选 checkpoint，最后才用更贵的外部 benchmark 和 continuation 判断“值不值得再烧预算”。

**展开**

- sync 阶段主要回答“方法链路和训练链路是否接通”
- async 阶段主要回答“长轨迹工具 agent 能不能稳定训练、resume、留下 checkpoint”
- `DeepDive subset64` gate 主要回答“下一步拿哪个 checkpoint 去做正式外部评测”
- `BrowseComp` 和 continuation 则回答“是否已经形成对外可讲的结果，还是只是内部系统有效”

**引用**

- `examples/carr_deepsearch/docs/agent_handoff_background_20260321.md:166-171`
- `examples/carr_deepsearch/docs/agent_handoff_background_20260321.md:197-233`
- `examples/carr_deepsearch/docs/post_gpu_revalidation_plan_20260412.md:129-218`

### Q122. 为什么 continuation 不是默认动作，而要等外部 gate 之后再决定？

**短答**

因为“多跑几步”不是一个可靠结论。当前项目的 `b/n`、sync/async step 语义和论文都不一致，所以 continuation 只有在 failure mode 明确指向“训练长度不够”，而不是 infra 回退、merge 错误或 eval recipe 错误时才成立。

**展开**

- continuation 前必须先排除：`BrowseComp` 结果只是因为 greedy 脚本、merge 资产不对、或 infra 回退而变差
- 只有当 `async_best` 内部不差于 `step70`、外部至少不差于 `SFT/step70`、主坏因是 `response_limit / unfinished` 时，才值得补 `+4` 到 `+8` 个 async step
- 最终 matched eval 已经形成简历闭环，因此本项目没有为了增加内部 step 数而机械执行 continuation

**引用**

- `examples/carr_deepsearch/docs/post_gpu_revalidation_plan_20260412.md:196-218`
- `examples/carr_deepsearch/docs/post_gpu_revalidation_plan_20260412.md:235-239`

### Q123. 你本地真正做过哪些“消融 / probe”，哪些没有做？

**短答**

本地真正做过的是系统层和训练层 probe，不是完整算法消融。做过的包括 `4:4 / 2:6 / 3:5` split、`sync=2 / 4`、`gmu=0.3 / 0.5`、`b=4 / 3`、`partial rollout`、`SP=1 / 2`、`dynbsz` 相关路径，以及 `queue_full` 修复后的集成验证；但没有做完整的 `GRPO vs C-GRPO`、`α sweep`、或 reward 组件的 controlled ablation。

**展开**

- 这些本地 probe 的目标是定位系统瓶颈和已知稳定主线，而不是写一篇算法消融论文
- 真正跑出主线价值的是：`queue_full` 修复、`gmu=0.5`、`2:6 + SP=2`、`b=3`
- `3:5 + SP=1` 属于失败 throughput probe，不是正向配置结论
- `queue_full` 修复有真实集成证据，但它不是一张“只改一个变量”的干净 ablation 表
- 所以面试里要清楚区分“我做过的系统 probe”和“我没有本地重做的算法 ablation”

**引用**

- `examples/carr_deepsearch/docs/agent_handoff_background_20260321.md:168-172`
- `examples/carr_deepsearch/docs/agent_handoff_background_20260321.md:197-223`
- `examples/carr_deepsearch/docs/agent_handoff_background_20260321.md:225-232`

### Q124. 为什么你没有本地做完整的 `GRPO vs C-GRPO`、`α` sweep、connectivity ablation？

**短答**

因为这个项目当前阶段的主目标不是重新做一轮 paper-style 算法研究，而是先把 `verl + CaRR + async RL` 这条真实训练系统跑通并稳定化。完整算法消融只有在 infra 稳定、外部 eval recipe 固定之后才有解释力。

**展开**

- 在 async 主线还会 timeout / OOM / stall 的阶段，做算法 sweep 很容易把 infra 噪声误当成方法差异
- 当前真实 sampled eval 已经补齐最小闭环；若继续做研究，下一步才是 full benchmark 或 controlled algorithm ablation
- 对未做的算法消融，可以直接引用论文结论，但要明确说“这是 paper prior，不是本地复验”

**引用**

- `examples/carr_deepsearch/docs/post_gpu_revalidation_plan_20260412.md:185-218`
- `examples/carr_deepsearch/docs/post_gpu_revalidation_plan_20260412.md:267-280`
- `CaRR/docs/08_实验结果与消融分析.md:147-194`

### Q125. 如果面试官追问“那你为什么还坚持 `α=0.3`、只奖励正确 rollout、保留 connectivity check”？

**短答**

因为你实现的是 paper-faithful `C-GRPO`，而论文对这三个设计点都给了明确 ablation 结论：`α=0.3` 最优；去掉 connectivity check 会明显变差；对所有 rollout 都加 rubric reward 甚至可能比纯 GRPO 更差。

**展开**

- `α=0.3`：平衡“先答对”和“在正确 rollout 之间区分过程质量”
- connectivity check：防止模型找一堆和最终答案无关的局部事实刷 rubric
- 只奖励正确 rollout：防止错误轨迹因为局部命中 rubric 而得到正向 advantage
- 这些在当前项目里是“按论文结论实现”，不是“本地重新证明了一遍”

**引用**

- `CaRR/docs/08_实验结果与消融分析.md:149-163`
- `CaRR/docs/08_实验结果与消融分析.md:165-194`
- `examples/carr_deepsearch/reward/cgrpo_advantage.py:88-112`

### Q126. 为什么简历和面试里优先讲系统工程结果，而不是硬写 benchmark uplift？

**短答**

因为最有区分度、也最容易被代码和日志完整证明的贡献仍然是 CaRR 接入、async partial rollout、参数同步、backpressure、恢复和诊断体系。外部 `BrowseComp subset256 64k` 已经有真实正向结果，但只有 `4` 和 `9` 个 judge-pass 样本，适合作为 supporting evidence，不适合脱离范围和基数写成夸张 headline。

**展开**

- 这是 claim discipline，不是回避结果
- 你现在能稳讲的是：系统可恢复、健康窗口吞吐缩短、`DeepDive 111` 从 `20/111` 到 `36/111`，以及 `BrowseComp subset256 64k` 从 `4/256` 到 `9/256`
- 外部 full benchmark、128K 和严格算法消融仍未执行，所以不越过证据边界

**引用**

- `examples/carr_deepsearch/docs/resume_source_of_truth_20260412.md`
- `examples/carr_deepsearch/docs/eval_analysis_20260415.md`

### Q127. 如果现在要补一个“最小可辩护”的算法消融矩阵，你会怎么设计？

**短答**

我会做一个非常小但解释力足够的矩阵，而不是大范围扫参：先固定 sampled eval recipe，再比较 `SFT`、纯 outcome `GRPO`、paper-faithful `C-GRPO(α=0.3)`；如果预算还够，再只补一个 `α` 小 sweep，例如 `0.1 / 0.3 / 0.5`。

**展开**

- 内部先用 `DeepDive subset64` 做 cheap gate，外部只把最值得的 1-2 个候选带去 `BrowseComp subset256`
- 如果要做 reward 组件消融，优先顺序是：`α`、`connectivity check`、`only-correct-rollouts`
- 但要明确：后两者已经不只是改配置，往往涉及 reward server / judge path 逻辑，不是一次轻量脚本实验

**引用**

- `examples/carr_deepsearch/docs/post_gpu_revalidation_plan_20260412.md:129-190`
- `CaRR/docs/08_实验结果与消融分析.md:147-194`

### Q128. 为什么 internal gate 先用 `DeepDive subset64`，而不是直接对所有候选跑 `BrowseComp`？

**短答**

因为 `DeepDive subset64` 更便宜、和训练分布更近，而且已经覆盖了真实 tool-use + reward integration 路径，足够先做 checkpoint 选择和 recipe 排错。`BrowseComp` 更贵，应该留给 1-2 个真正值得带出去的候选。

**展开**

- `DeepDive subset64` 可以用统一 sampled recipe 快速并排比较 `SFT / step70 / step90 / async_best`
- 它先回答“哪个 checkpoint 值得带去外部 benchmark”，也能先排除 merge、greedy 脚本、tool/reward integration 断链这类低级问题
- `BrowseComp subset256` 的职责不是做大规模 checkpoint 搜索，而是验证外部泛化是否成立
- 这样做的本质是先用低成本 gate 缩小候选，再把 GPU 预算花在更贵的外部 benchmark 上

**引用**

- `examples/carr_deepsearch/docs/post_gpu_revalidation_plan_20260412.md:36-53`
- `examples/carr_deepsearch/docs/post_gpu_revalidation_plan_20260412.md:129-190`
- `examples/carr_deepsearch/docs/post_gpu_revalidation_plan_20260412.md:257-258`
- `examples/carr_deepsearch/docs/resume_source_of_truth_20260412.md:21`
- `examples/carr_deepsearch/docs/resume_source_of_truth_20260412.md:27`

### Q129. 为什么 continuation 只从 `async_best` 恢复，而不是从 latest step 继续？

**短答**

因为 continuation 要回答的是“当前最好的 async 候选是不是只是训练长度还不够”，而不是“最新那个 checkpoint 能不能自己涨回来”。如果直接从 latest step 继续，会把 checkpoint 选择问题和训练长度问题混在一起。

**展开**

- `async_best` 是 internal gate 在统一 sampled recipe 下选出来的候选，语义最干净
- latest durable checkpoint 不一定等于 best-by-quality；文档里已经把 `global_step_23` 和最佳内部窗口 `step16` 明确分开
- continuation 的设计目的是固定主线配置，只额外测试“多跑几步”是否有价值；从 latest 恢复会把 late-step drift、resume 首步效应、unfinished 回升一起掺进来
- 所以 continuation 只能从 `async_best` 出发，而不是默认追最新 checkpoint

**引用**

- `examples/carr_deepsearch/docs/post_gpu_revalidation_plan_20260412.md:202-216`
- `examples/carr_deepsearch/docs/resume_source_of_truth_20260412.md:296-312`
