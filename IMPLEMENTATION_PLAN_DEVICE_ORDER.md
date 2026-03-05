# CaRR Deep Search Agent: 开发顺序表（设备与验证分层版）

## 目的

本文件不替代原始实施方案。

- 原文件 [`IMPLEMENTATION_PLAN.md`](/Users/xiaohui/Desktop/agentic-RL-project/verl-carr-deepsearch/IMPLEMENTATION_PLAN.md) 继续作为“技术设计与实现细节”的主文档。
- 本文件只回答一件事：每一步应该在什么设备上编写、执行和验证，避免过早租大机器，也避免把第一次 GPU 验证拖得太晚。

适用目标保持不变：

- `verl` Agent Loop + CaRR Tool Server + CaRR Reward Server
- SFT 冷启动
- RL 训练（C-GRPO）
- `SFT / GRPO / C-GRPO` 三条线对比
- `BrowseComp` benchmark 评测与结果汇总
- 目标模型：Qwen3-4B

---

## 设备分层

### Device A: 本地 Mac

适合做：

- 阅读代码与设计文档
- 编写 Python 逻辑
- 数据预处理脚本开发
- YAML / shell 脚本编写
- message / history / parquet / schema 转换验证
- 不依赖 CUDA 的静态检查和小脚本验证

不建议依赖它做：

- `sglang` / `vllm` / FSDP / Ray 的真实训练联调
- 多轮 rollout 的真实 GPU 行为验证
- SFT / RL 冒烟训练

### Device BC: 远程 Linux `2 x 5090`（合并联调 + 冒烟）

推荐角色：正式训练前唯一的 GPU 调试阶段。

适合做：

- CUDA / 驱动 / Python 环境安装
- `verl` + `sglang` 依赖可用性验证
- 工具服务、奖励服务、环境变量、端口联通验证
- 自定义模块导入验证
- 单样本或极小 batch 的 rollout 链路验证
- 小规模 SFT / RL 冒烟
- BrowseComp subset quick eval
- `torchrun` / FSDP / rollout / reward 全链路联调
- OOM 边界、吞吐和关键指标的第一轮确认

不建议把它当作：

- 正式 SFT 训练机
- 正式 RL 训练机
- 正式 benchmark 结论机器

### Device D: 远程 Linux 8 卡 GPU

推荐角色：正式训练机。

适合做：

- 完整 SFT
- 完整 RL
- 全量 BrowseComp benchmark 评测
- checkpoint、日志、稳定性和指标收敛验证

---

## 总体策略

默认策略：

1. 先在 Device A 完成大部分代码编写。
2. 代码在 Mac 基本完成后，直接切到 Device BC（`2 x 5090`）一次性完成 Gate 2-5。
3. 只有在 Gate 1-5 都通过后，才上 Device D 做正式训练与正式 benchmark 评测。

成本原则：

- 不要在还没打通 tool/reward/history 协议前就租 8 卡。
- Mac 代码阶段结束后，优先进入同一台 `2 x 5090` 机器完成联调和小规模训练验证。
- 正式训练前的所有关键问题都应在 Device BC 清零，不要把基础问题带到 8 卡阶段。

---

## 开发顺序表

| 步骤 | 对应原计划 | 主要产物 | 编写设备 | 执行设备 | 验证设备 | GPU 需求 | 完成标准 |
|---|---|---|---|---|---|---|---|
| 0 | 准备阶段 | 开发分支、目录骨架、依赖清单 | Device A | Device A | Device A | 不需要 | 新目录结构明确，文件位置与原计划一致 |
| 1 | Phase 1.1 / 1.2 / 7.5.3 | `preprocess_carr_rl.py`、`preprocess_carr_sft.py`、`preprocess_browsecomp.py` | Device A | Device A | Device A | 不需要 | 能生成 `SFT / RL / BrowseComp` 三类 parquet |
| 2 | Phase 1 验证 | parquet 字段校验、chat template 校验脚本 | Device A | Device A | Device A | 不需要 | RL 数据字段完整；SFT 样本可过 tokenizer `apply_chat_template` |
| 3 | Phase 2 | `carr_session_manager.py`、`carr_browser_tool.py`、工具 YAML | Device A | Device A | Device A | 不需要 | 代码层面完成 session 设计、参数转换、tool schema 定义 |
| 4 | Phase 4 | `carr_reward.py` | Device A | Device A | Device A | 不需要 | reward payload、超时、兜底逻辑完成 |
| 5 | Phase 5 | `cgrpo_advantage.py`，以及 `algorithm.py` / `ray_trainer.py` 改动 | Device A | Device A | Device A | 不需要 | estimator 注册成功；静态阅读确认 `non_tensor_batch` 透传路径正确 |
| 6 | Phase 3 | `carr_agent_loop.py` | Device A | Device A | Device A | 不需要 | `reward_history`、`task_unfinished`、tool_call_id、finally close session 逻辑完成 |
| 7 | Phase 6 / 7 / 7.5.4 | `carr_sft.yaml`、`carr_grpo.yaml`、`run_sft.sh`、`run_rl.sh`、`run_eval_browsecomp.sh`、`smoke_test.py` | Device A | Device A | Device A | 不需要 | 配置、训练脚本与评测脚本接口一致 |
| 8 | Gate 1 | 数据 Gate | Device A | Device A | Device A | 不需要 | 数据预处理通过，样本抽检通过 |
| 9 | Phase 8.0 / 8.2 | Linux GPU 环境打底 | Device A | Device BC | Device BC | 需要 `2 x 5090` | `verl`、`sglang`、自定义模块、依赖安装成功 |
| 10 | Phase 8.2 | Tool / Reward Server 联通 | Device A 或 BC | Device BC | Device BC | 需要 `2 x 5090` | `smoke_test.py --tool`、`--reward`、`--all` 通过 |
| 11 | Phase 3 / 4 联调 | 单样本 agent rollout 链路 | Device A 或 BC | Device BC | Device BC | 需要 `2 x 5090` | 能真实走一轮 `generate -> tool -> reward payload`，且 history 格式正确 |
| 12 | Gate 2 / 3 | 工具链路 Gate、奖励 Gate | Device A 或 BC | Device BC | Device BC | 需要 `2 x 5090` | `search -> open -> find` 正常；reward 非格式性全零 |
| 13 | Phase 8.3 | SFT 微型冒烟 | Device A 或 BC | Device BC | Device BC | 需要 `2 x 5090` | 至少跑通 1 个 epoch 或最少若干 step，无 tokenizer/loader/FSDP 崩溃 |
| 14 | Phase 8.4 | BrowseComp 快速评测冒烟（SFT ckpt） | Device A 或 BC | Device BC | Device BC | 需要 `2 x 5090` | `run_eval_browsecomp.sh` 在 `max_samples=5/20` 下可完成 |
| 15 | Phase 8.5 / 8.6 | GRPO baseline 微型冒烟与快速评测 | Device A 或 BC | Device BC | Device BC | 需要 `2 x 5090` | `algorithm.adv_estimator=grpo` 路线可跑通，且输出 `val-core/browsecomp/reward/mean@1` |
| 16 | Phase 8.7 / 8.8 | C-GRPO 微型冒烟与快速评测 | Device A 或 BC | Device BC | Device BC | 需要 `2 x 5090` | C-GRPO 小规模训练与 subset eval 都可完成 |
| 17 | Gate 4 / 5 | 小规模训练与快速评测 Gate | Device A 或 BC | Device BC | Device BC | 需要 `2 x 5090` | `SFT / GRPO / C-GRPO` 路线都至少有一次 subset eval 结果 |
| 18 | Phase 8.3 正式版 | 完整 SFT | Device A 或 BC | Device D | Device D | 需要 8 卡 | SFT checkpoint 可用 |
| 19 | Phase 8.4 / 8.9 | SFT 正式 BrowseComp 评测 | Device A 或 BC | Device D | Device D | 需要 8 卡 | SFT 的 64k 正式 benchmark 数字产出 |
| 20 | Phase 8.5 正式版 | 完整 GRPO baseline | Device A 或 BC | Device D | Device D | 需要 8 卡 | GRPO checkpoint 可用 |
| 21 | Phase 8.6 / 8.9 | GRPO 正式 BrowseComp 评测 | Device A 或 BC | Device D | Device D | 需要 8 卡 | GRPO 的 64k 正式 benchmark 数字产出 |
| 22 | Phase 8.7 正式版 | 完整 C-GRPO | Device A 或 BC | Device D | Device D | 需要 8 卡 | C-GRPO checkpoint 可用 |
| 23 | Phase 8.8 / 8.9 | C-GRPO 正式 BrowseComp 评测 | Device A 或 BC | Device D | Device D | 需要 8 卡 | C-GRPO 的 64k 正式 benchmark 数字产出 |
| 24 | Phase 8.10 | 结果汇总与简历指标整理 | Device A | Device A | Device A | 不需要 | 三模型对比表完成 |

---

## 每一步更具体的设备建议

### Step 0-8: 完全在本地 Mac 完成

这些步骤不需要等 GPU：

- 建目录和文件骨架
- 写 `SFT / RL / BrowseComp` 数据预处理脚本
- 写工具适配器和 reward 函数
- 写 `cgrpo` estimator
- 写 `carr_agent_loop.py`
- 写 YAML、训练脚本和评测脚本
- 跑 parquet 校验和 tokenizer 校验

这一段的目标不是“跑训练”，而是把协议和数据格式写对。

如果在这一段就依赖远程 GPU，成本高，而且容易把简单问题拖成环境问题。

### Step 9-17: 使用同一台 `2 x 5090` 完成联调 + 小规模训练验证

这一步是 Mac 开发完成后的唯一预正式训练阶段，建议一次租机连续跑完。

先做联调（旧 Step 9-12）：

- 依赖是否能在 Linux + CUDA 环境中装起来
- `VERL_USE_EXTERNAL_MODULES` 是否能正确加载你的自定义模块
- Tool Server / Reward Server 是否能按脚本启动
- `smoke_test.py` 是否真的通过
- 单样本 history 格式是否能让 CaRR reward server 接受

再做小规模训练与 quick eval（旧 Step 13-17）：

- SFT:
  - 减小 `train_batch_size`
  - 减小 `max_length`
  - 减少 epoch 或只跑少量 step
- BrowseComp quick eval:
  - 先用 `max_samples=5/20/100`
  - 先只跑 `64k`
- RL:
  - `data.train_batch_size=8/16`
  - `actor_rollout_ref.rollout.n=2/4`
  - 缩短 `max_response_length`
  - 先跑 `trainer.total_epochs=1`

这一段要一次性确认：

- FSDP / `torchrun` 是否正常
- rollout 是否能稳定结束
- reward 字段是否进入 `non_tensor_batch`
- `cgrpo` advantage 是否被正确调用
- `run_eval_browsecomp.sh` 是否能稳定输出 `val-core/browsecomp/reward/mean@1`
- 训练是否出现明显 OOM、NaN、死锁或异常中止

### Step 18-24: 正式训练与正式评测，使用 8 卡 Linux

只有在前面几个 Gate 通过之后才建议进入这里。

正式训练前应满足：

- 数据格式已经确认
- tool / reward 协议已经确认
- 单样本 rollout 已确认
- 小规模 `SFT / GRPO / C-GRPO` 与 BrowseComp quick eval 已确认

进入 8 卡阶段后，不应再花时间排查如下低级问题：

- `history[-1].role != assistant`
- `tool_call_id` 没绑定
- `search_forbidden_strs[0]` 缺失
- 自定义 estimator 没注册
- shell 路径写错
- `run_eval_browsecomp.sh` 忘记设置 `data.train_files`

这些都应在前面的设备层完成。

---

## Gate 定义与设备要求

| Gate | 说明 | 必须在哪类设备完成 |
|---|---|---|
| Gate 1 | 数据预处理通过 | Device A |
| Gate 2 | 工具链路通过 | Device BC |
| Gate 3 | 奖励链路通过 | Device BC |
| Gate 4 | 小规模 RL 通过 | Device BC |
| Gate 5 | 小规模训练与快速评测通过 | Device BC |
| Gate 6 | 正式 BrowseComp 评测完成 | Device D |
| Gate 7 | `SFT / GRPO / C-GRPO` 三模型正式结果表完成 | Device D + Device A |

不能跳过的顺序：

`Gate 1 -> Gate 2 -> Gate 3 -> Gate 4 -> Gate 5 -> Gate 6 -> Gate 7`

---

## 推荐的实际执行节奏

### 阶段 A: 本地开发周

在 Device A 上连续完成：

1. 数据预处理
2. Tool / Reward 适配
3. Agent Loop
4. 配置与脚本
5. 数据和 tokenizer 校验

产出：

- 代码基本完成
- 至少能静态确认所有接口拼得上

### 阶段 BC: `2 x 5090` 联调 + 小规模训练验证（合并阶段）

切到 Device BC，一次连续完成：

1. 环境安装
2. Tool / Reward 服务启动
3. `smoke_test.py`
4. 单样本 rollout
5. SFT 微型冒烟
6. BrowseComp quick eval 冒烟
7. GRPO baseline 微型冒烟
8. C-GRPO 微型冒烟
9. 观察 reward、turn 数、aborted ratio、BrowseComp subset accuracy

产出：

- 确认代码不是“只在本地看起来正确”
- 确认方案既具备训练可行性，也具备首版 benchmark 可执行性

### 阶段 D: 正式训练与正式评测

切到 Device D，做：

1. 完整 SFT
2. SFT 正式 BrowseComp 评测
3. 完整 GRPO baseline
4. GRPO 正式 BrowseComp 评测
5. 完整 C-GRPO
6. C-GRPO 正式 BrowseComp 评测
7. 汇总三模型对比表

---

## 对 `2 x 5090` 合并阶段的定位

在本计划中，`2 x 5090` 同时承担原 Device B + Device C：

- 用于 Linux GPU 环境联调、服务联通、模块注册、单样本链路
- 用于小规模 `SFT / GRPO / C-GRPO` 冒烟与 BrowseComp quick eval
- 用于在正式训练前定位 OOM、NaN、死锁、协议错误等关键问题
- 不用于正式 benchmark 数字结论

如果预算紧张，推荐顺序是：

1. 先在 Mac 写完大部分代码
2. 直接租 `2 x 5090`，完成 Gate 2-5
3. 最后租 8 卡做正式训练与正式评测

如果只能拿到单张卡，它只能承担部分联调任务，不建议作为本计划默认路径。

---

## 与原计划的映射关系

| 本文件步骤 | 原计划对应内容 |
|---|---|
| Step 1-2 | Phase 1 数据预处理 + BrowseComp 预处理 |
| Step 3 | Phase 2 浏览器工具实现 |
| Step 4 | Phase 4 CaRR Reward 函数 |
| Step 5 | Phase 5 C-GRPO Advantage Estimator |
| Step 6 | Phase 3 自定义 Agent Loop |
| Step 7 | Phase 6 配置 + Phase 7 脚本 + 评测脚本设计 |
| Step 9-12 | Phase 8.0 / 8.2 环境和服务冒烟 |
| Step 13 | Phase 8.3 SFT 冒烟 |
| Step 14 / 15 / 16 | Phase 8.4 / 8.6 / 8.8 快速评测 |
| Step 18-24 | Phase 8.3-8.10 正式训练、正式评测与结果汇总 |

---

## 执行建议总结

最推荐的落地顺序：

1. Device A 完成代码与数据逻辑
2. Device BC（`2 x 5090`）完成联调 + 小规模训练 + quick eval 冒烟
3. Device D 做正式训练、正式 benchmark 与结果汇总

如果必须压缩成本，优先压缩的是：

- Device D 的租用时长

不应该压缩的是：

- Device BC 阶段（Gate 2-5）

因为它负责把“逻辑问题”和“训练系统问题”都在正式训练前提前拆开。
