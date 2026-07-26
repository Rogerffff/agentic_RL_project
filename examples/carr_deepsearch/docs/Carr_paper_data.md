## 1. 你最需要先抓住的核心设定

这篇论文的 agent 是标准 **ReAct** 形式：模型反复执行 `thinking -> tool call -> observation`，直到最后输出带引用的解释和最终答案。工具只有 3 个：`search`、`open`、`find`。最终回答必须包含两个固定段落：`## Explanation with Citations` 和 `## Exact Answer`。这点对你用 verl 复现非常重要，因为 reward 计算依赖**最终回答里是否显式写出实体、是否包含 citation、是否有 final answer**。  

CaRR 的 rubric reward 不是简单“答对加分”，而是三步：
第一步，判断最终回答里是否**显式识别出 hidden entities**；
第二步，判断这些 rubric 是否被**引用到的网页内容**真正支持；
第三步，检查这些被支持的 rubrics 是否能通过一个图结构**连到最终答案实体**，形成完整 evidence chain。最终 rubric reward 是：`连到答案的支持 rubrics 数 / 总 rubrics 数`。他们还特别限制**最多提取 20 个 cited URLs**，防止 agent 靠狂引网页 hack reward。

C-GRPO 的关键不是“所有 rollout 都加 rubric reward”，而是**只有 outcome reward = 1 的正确 rollout 才加 weighted rubric reward**。公式是
`Ri = (1-α) * Ro + α * Ro * R̂r`，
其中 `R̂r` 是**组内按最大 rubric reward 归一化**后的 rubric reward。格式错误或超长（超过 token 或 tool-call 限制）的 rollout，reward 直接设为 0。这个设计是整篇论文最关键、也最容易在实现里写偏的地方。

## 2. 论文里明确给出的 SFT 设置

论文用的 backbone 一共有两个：
**Qwen3-4B-Thinking-2507** 和 **Qwen3-30B-A3B-Thinking-2507**，分别覆盖 dense 和 MoE 两种规模。训练数据来自 **DeepDive**。论文明确写了：DeepDive 一共包含 **1,016 个 SFT 样本**和 **2,234 个 RL 样本**。

SFT 阶段不是直接拿原始 1,016 条全训，而是先用 **GLM-4.6** 在 DeepDive 的 SFT split 上做 **reject sampling**，生成 **832 条高质量 SFT traces**，然后在这些 traces 上训练。SFT 的明确超参是：

* 训练 **3 epochs**
* **batch size = 16**
* **learning rate = 4e-5**
* **maximum context length = 128k**。

这意味着你如果对论文“严格对标”，SFT 最应该先核对的不是 dataset 名字，而是这 4 个点：
1）是否先做 reject sampling 产生高质量 traces；
2）是否只用 832 条 SFT traces；
3）是否 128k 上下文；
4）是否 3 epoch、bs16、lr4e-5。

## 3. 论文里明确给出的 RL 设置

RL 用的是 **DeepDive RL split 全部 2,234 个 QA pairs**。明确写出的 RL 配置有：

* **rollout size = 16**
* **8 samples per prompt**
* **global batch size = 128**
* **temperature = 1.0**
* **learning rate = 2e-6**
* **maximum context length = 64k tokens**
* **train 3 epochs**
* **rubric reward weight α = 0.3**。

论文没有把“rollout size=16”和“8 samples per prompt”的实现语义展开解释，但从数字关系看，很可能是一个 update 中有 16 个 prompts、每个 prompt 采 8 条 rollout，总 rollout 数对应 global batch 128。论文文本本身没有再更细解释，所以你在 verl 里要确认你自己的 `train_batch_size / n_samples / prompt batch` 映射方式，不要只看字面数字。这个是我基于论文数值关系做的实现层推断。支持这些数值本身的原文在这里。

另外，reward judge 用的是 **DeepSeek-v3.2**，而且它同时负责 **outcome reward 和 rubric reward**。这意味着如果你想更接近论文，reward server 里的 judge model 不能随便换；换了 judge，reward 分布很可能就会变。

## 4. 环境和工具设置

环境部分论文也写得比较具体：

* `search` 工具用 **Serper API**
* `open` 工具先用 **Jina API** 抓网页，再返回网页前 **10k chars**
* `find` 工具是**vanilla string matching**。

附录里又补充了工具接口定义：

* `browser.search(query, num=10)`
* `browser.open(id 或 url)`
* `browser.find(pattern)`。

如果你在 verl 里用的是别的工具 server，最容易造成偏差的地方有 4 个：
1）`open` 返回内容长度是不是前 10k chars；
2）`search` 默认结果数是不是 10；
3）`find` 是否只是字符串匹配而不是 BM25/embedding retrieval；
4）最终 response 的 citation 格式是否和 reward extractor 兼容。 

## 5. Baseline、比较对象、评测方式

论文真正做对比的 RL baseline 只有两个：

* **GRPO**：只用 outcome rewards
* **E-GRPO**：给错误 rollout 一个按 hidden entities 命中率算的细粒度奖励。

正式 benchmark 一共 4 个：
**BrowseComp、BrowseComp-ZH、xbench-DeepSearch、GAIA 的 text-only validation subset**。评测指标是**accuracy**。他们按照 benchmark 官方的 LLM-as-judge 方式，用 **GPT-5-Chat** 判断 agent 的最终输出是否和 ground truth 匹配。由于 BrowseComp-ZH、xbench-DeepSearch、GAIA 数据量较小，这三个 benchmark **重复评测 3 次取平均**。此外，每个模型都会在 **64k 和 128k context** 下都评一次：64k 对应 RL 训练时的长度，128k 用来看 test-time scaling。

open-ended 泛化评测用的是 **DeepResearch Bench**。这个 bench 要求 agent 写研究报告，由 **Gemini-2.5-Pro-preview** 按预定义 rubrics 打分，维度包括 **Overall、Comprehensiveness、Insight、Instruction Following、Readability**。

## 6. 你最该拿来对照的主结果

### 4B 模型

4B 的 SFT / GRPO / E-GRPO / C-GRPO 在 64k 与 128k 下的结果如下：

* **4B-SFT**：BC 7.7 / 14.1，BC-ZH 10.1 / 16.6，xbench-DS 34.0 / 44.3，GAIA 39.5 / 46.0
* **+GRPO**：12.9 / 14.7，16.6 / 17.5，41.0 / 41.3，40.5 / 41.1
* **+E-GRPO**：11.5 / 14.5，16.5 / 20.2，43.7 / 45.0，42.4 / 42.4
* **+C-GRPO**：13.9 / 17.5，18.2 / 24.7，50.3 / 54.0，48.9 / 50.2。



论文作者总结说：相对 GRPO，C-GRPO 在 **4B** 上平均提升 **5.1（64k）/ 8.0（128k）**，在 **30B** 上平均提升 **2.6（64k）/ 6.0（128k）**；而且纯 outcome reward 的 GRPO 虽然在 64k 内会提升，但会伤害更长上下文下的 test-time scaling。 

对你来说，这组数值就是**最直接的复现对标目标**。如果你做 4B 简历项目，最值得拿来对照的是：
SFT 64k 大约在 `7.7 / 10.1 / 34.0 / 39.5`，
C-GRPO 64k 大约在 `13.9 / 18.2 / 50.3 / 48.9`，
128k 下进一步到 `17.5 / 24.7 / 54.0 / 50.2`。

## 7. 训练动态和中间现象：你训练时应该观察什么

Figure 4 展示了训练曲线，横轴大约到 **400 training steps**。论文观察到：

* 一开始 GRPO 和 C-GRPO 的平均 tool calls 都先下降，说明模型先学会了减少无效搜索、避免 overlength rollout；
* 之后 **GRPO** 的 tool calls 又继续下降，说明策略掉进了 shortcut solution 的局部最优；
* 而 **C-GRPO** 在早期下降后，tool calls 会重新上升，说明模型为了满足更多 rubrics 开始主动收集更多证据；
* 同一时期 C-GRPO 的 outcome rewards 还会略高于 GRPO。 

这意味着你在 verl 里做训练监控时，**不能只看 final accuracy 或 outcome reward**。你至少还应该看：
1）平均 tool-call steps；
2）correct rollout 比例；
3）rubric reward 均值；
4）overlength / format error 比例。
因为按照论文的叙述，**“tool-call 持续下降 + outcome reward 看起来还行”反而可能是模型走偏了**。

## 8. 论文如何证明 C-GRPO 更“全面、更真实”

论文在 BrowseComp 的一个子集上，比较了 30B-SFT、30B-GRPO、30B-C-GRPO 的 cited pages 和 rubric satisfaction：

* **SFT**：`|CH|=3.8, |Ridentify|=8.0, |Rsupport|=6.2, |Rconnect|=4.5, |Rq|=10.1`
* **GRPO**：`3.5, 7.5, 5.3, 4.0, 10.1`
* **C-GRPO**：`4.3, 8.2, 6.6, 5.2, 10.1`。

作者据此认为，C-GRPO 会引用更多网页、满足更多 rubrics，所以 reasoning comprehensiveness 和 factuality 更强；而 GRPO 甚至在 cited pages 和 satisfied rubrics 上低于 SFT，进一步说明 pure outcome reward 会诱导 shortcut exploitation。

## 9. open-ended deep research 泛化结果

在 DeepResearch Bench 上，C-GRPO 也优于其他 RL 设定。
4B：

* SFT `33.81 / 29.57 / 24.23 / 44.05 / 41.02`
* GRPO `34.79 / 31.29 / 26.79 / 43.81 / 41.58`
* E-GRPO `36.59 / 33.20 / 28.30 / 45.58 / 42.67`
* C-GRPO `37.51 / 33.88 / 30.01 / 45.72 / 43.82`。

30B：

* SFT `37.51 / 34.27 / 28.85 / 46.77 / 43.21`
* GRPO `39.30 / 36.10 / 31.66 / 47.65 / 44.92`
* E-GRPO `36.12 / 32.31 / 27.73 / 45.72 / 42.33`
* C-GRPO `41.99 / 39.75 / 35.87 / 48.51 / 46.63`。

论文还特别说，**30B 的 C-GRPO 甚至超过了一些 proprietary-data 训练出来的 advanced agents**，把它作为泛化能力强的证据。

## 10. 消融实验：这些点非常值得你在自己的项目里保留

### α 的影响

4B 上：

* α=0：14.7 / 17.5 / 41.3 / 41.1
* α=0.1：13.0 / 18.0 / 46.0 / 46.3
* α=0.3：17.5 / 24.7 / 54.0 / 50.2
* α=0.5：17.0 / 20.8 / 49.3 / 42.4。

结论是 **α=0.3 最优**。太小，rubric reward 不够；太大，又会让模型偏离“先答对”的主目标。

### 去掉 hidden entity identification

从 `17.5 / 24.7 / 54.0 / 50.2` 掉到 `16.5 / 23.2 / 50.7 / 46.6`。作者认为严格要求先识别 hidden entities，再做 rubric judgment，会让奖励更干净。 

### 去掉 evidence connectivity check

掉到 `15.1 / 20.8 / 47.7 / 44.0`。这说明如果不要求 supported rubrics 必须连接到 final answer，模型会学会找一些“局部成立但和答案无关”的事实去 hack rubric。 

### 给所有 rollouts 都加 rubric reward

掉到 `13.3 / 14.0 / 40.3 / 40.8`。作者解释原因是：早期 RL 时 correct rollout 很少、overlength rollout 很多，如果错误 rollout 也带 rubric reward，GRPO 的 advantage 可能会让错误策略得到正向优化。 

## 11. 附录里和实现最相关的补充信息

论文附录 C 做了 judge LLM 的人工核验：在 **10 条 DeepDive-30B-SFT 轨迹**上，覆盖 **128 个 hidden entities** 和 **164 个 rubrics**，judge 在

* hidden entity identification 上准确率 **97.7%**
* citation-based rubric evaluation 上准确率 **95.1%**。

附录 D 的 case study 也很关键：

* **GRPO** 倾向于只根据问题后半段或最后几跳信息猜答案，不认真验证前面约束；
* **C-GRPO** 会继续搜证据，直到把每个约束都验证掉，而且确保回答中的陈述都有 citation 支撑。

附录 A 给了你需要完全对齐的轨迹格式和工具格式，这对 verl 里的 parser、prompt template、reward extraction 都很重要。

## 12. 论文没有说清楚、你不能强行“脑补”的地方

下面这些是**论文没有明确给出**或至少在我从 PDF 可检索到的正文/附录片段里**没有看到具体数值**的：

* SFT / RL 的 optimizer 类型
* scheduler / warmup / weight decay
* PPO/GRPO 的 `ϵ_high`、`ϵ_low` 数值
* gradient accumulation
* KL penalty / entropy bonus
* max tool-call limit 的具体数值
* format error 的严格判定规则
* reject sampling 的筛选阈值
* Figure 11–14 里的完整 prompt 文本。
  论文只明确说附录 E 展示了 prompts，但我当前检索到的行级内容没有把 prompt 正文完整展开出来，所以这部分我不想假装自己看到了。

## 13. 你在 verl 启动前，最值得逐项核对的清单

你可以直接按这几项查你自己的实现：

第一，**输出格式**是否严格是
`Explanation with Citations + Exact Answer`，并且最终回答里显式写出 hidden entities。

第二，**工具行为**是否和论文一致：3 个工具；`open` 返回前 10k chars；`find` 是 string match。

第三，**reward 逻辑**是否一致：
rubric reward 三步；最多 20 cited URLs；只有 correct rollout 才乘上 rubric reward；rubric reward 组内最大值归一化。 

第四，**训练超参**是否对齐：
SFT `3 epoch / bs16 / lr4e-5 / 128k / 832 traces`；
RL `3 epoch / rollout size 16 / 8 samples per prompt / global batch 128 / temp1.0 / lr2e-6 / 64k / α=0.3`。

第五，**judge 模型**是否一致：
训练时 reward judge 用 DeepSeek-v3.2；benchmark eval 用 GPT-5-Chat；DeepResearch Bench 用 Gemini-2.5-Pro-preview。  

第六，训练监控里不要只看 answer accuracy，还要看：
`avg tool calls / outcome reward / rubric reward / overlength ratio`。因为论文最重要的发现之一就是：**纯 GRPO 也会“看起来在学”，但其实是在学 shortcut**。

