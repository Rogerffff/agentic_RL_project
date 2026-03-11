**修订后的完整执行计划**

## Summary
- 主结果口径固定为：`BrowseComp subset (N=256, seed=42)` 上的 `outcome_reward/mean@1`。
- 主对比对象固定为：`SFT / GRPO / C-GRPO`。
- `Qwen/Qwen3-4B base` 只做 10-sample smoke，不进入正式主表。
- `DeepDive RL val` 用来证明方法有效性，重点看 `rubric_reward`、工具行为和任务完成质量。
- 所有正式结果都基于同一 judge 体系，强调项目内相对提升，不与论文绝对数字直接对齐。

## 锁定后的实验矩阵
- `Eval 0`：`Qwen/Qwen3-4B` base 跑 10 个 BrowseComp 样本的 smoke eval。
- `Run 1`：完整 SFT，checkpoint 规则固定为 `best val/loss`。
- `Run 2`：从同一 SFT checkpoint 出发跑 `GRPO`，结果使用 `epoch 3` 结束时的最终 checkpoint。
- `Run 3`：从同一 SFT checkpoint 出发跑 `C-GRPO`，结果使用 `epoch 3` 结束时的最终 checkpoint。
- `Eval 1`：`SFT / GRPO / C-GRPO` 在固定子集 `browsecomp_eval_subset_256_seed42.parquet` 上统一评测。
- `Eval 2`：`SFT / GRPO / C-GRPO` 在 `DeepDive RL val (111)` 上统一评测 `outcome_reward/mean@1`、`rubric_reward/mean@1`、`task_unfinished ratio`、`num_turns`、`tool_call_counts`、三类工具计数。
- 可选增强项只保留一个：只有当 `C-GRPO` 相对 `SFT` 在 `BrowseComp subset 64k` 上提升至少 `3 pp` 时，再补 `SFT / C-GRPO` 的 `128k subset`。
- 不做：BrowseComp 全量、Base 全量、E-GRPO、alpha sweep、额外 benchmark、30B。

## 关键配置决策
- BrowseComp 评测不用 `val_max_samples` 运行时截取，改为预生成固定文件：
  `examples/carr_deepsearch/data/browsecomp_eval_subset_256_seed42.parquet`。
- SFT `max_length` 锁定优先级：
  首选 `65536`，因为你实测 `79.1%` 样本零截断，剩余 `20.9%` 样本平均仍保留 `81%` 内容。
- SFT 不追求 `131072`，因为在 4 卡 A100 上高概率不划算或不稳定。
- 只有当 `65536` 在目标机器上 20-step smoke 明确 OOM，才降到 `32768`；若降到 `32768`，最终报告必须明确这是保守配置，因为此时 `78.4%` 样本被截断。
- RL 保持 `rollout.n=16`，结果说明里直接写清楚 `group size = 16 per prompt`。
- logger 改为 `['console', 'wandb']`，命名固定：
  `carr-sft-qwen3-4b`
  `carr-grpo-qwen3-4b`
  `carr-cgrpo-qwen3-4b`


## 训练与评测脚本要求
- [run_eval_browsecomp.sh](/Users/xiaohui/Desktop/agentic-RL-project/verl-carr-deepsearch/examples/carr_deepsearch/scripts/run_eval_browsecomp.sh) 默认正式评测使用固定 subset parquet，不再依赖 `val_max_samples`。
- `run_eval_browsecomp.sh` 保留一个显式参数用于切换 `subset256` 或 `full`，默认 `subset256`。
- `run_sft.sh` 和 `run_rl.sh` 在头尾自动记录开始时间、结束时间、总秒数，产出 wall-time，用于 GPU-hours 统计。
- 正式 RL 训练必须显式设置：
  `trainer.validation_data_dir=$HOME/eval_dumps/<exp>`
  `trainer.rollout_data_dir=$HOME/rollout_dumps/<exp>`
- BrowseComp 正式评测也必须设置 `validation_data_dir`，确保 case study 材料落盘。

## 正式要记录的指标
- `Base smoke`：
  `valid tool-call rate`
  `parse error rate`
  `num_turns`
  `tool_call_counts`
  少量 sample output
- `SFT 训练期`：
  `train/loss`
  `val/loss`
  `epoch time`
  `peak GPU memory`
  `tokens/s`
- `RL 训练期`：
  `critic/score/*`
  `critic/rewards/*`
  `response_length/*`
  `response/aborted_ratio`
  `num_turns/*`
  `tool_call_counts/*`
  `outcome_reward/*`
  `rubric_reward/*`
  `task_unfinished/mean`
  `search_count/open_count/find_count`
  `parse_error_count`
  `hit_limit`
  `throughput`
  `timing`
  `peak GPU memory`
- `正式 benchmark`：
  `BrowseComp subset (N=256) outcome_reward/mean@1`
- `过程质量`：
  `DeepDive RL val` 上的
  `outcome_reward/mean@1`
  `rubric_reward/mean@1`
  `task_unfinished/mean@1`
  `tool_call_counts/mean@1`
  `search_count/open_count/find_count/mean@1`
  `num_turns/mean`
- `成本`：
  SFT、GRPO、C-GRPO 的 wall time 与 GPU-hours
  DeepSeek / Serper / Jina 调用量和估算成本

## 结果产出物
- 主表：
  `SFT / GRPO / C-GRPO` 在 `BrowseComp subset (N=256)` 上的对比表。
- 补充表：
  `SFT / GRPO / C-GRPO` 在 `DeepDive RL val` 上的
  `outcome_reward`
  `rubric_reward`
  `task_unfinished`
  `tool_call_counts`
  `search/open/find`
  `num_turns`
  对比表。
- 曲线图：
  training-step 的 `outcome_reward`、`rubric_reward`、`tool_call_counts`、`num_turns`。
- Case study：
  2 个正例、1 个失败例，必须能从 dump 中还原完整 `search → open → find` 或等价轨迹。
- Base smoke 只作为附录或方法说明，不放主表。

## 简历叙事落点
- 主 bullet 聚焦：
  `C-GRPO 相对 SFT/GRPO 在 BrowseComp subset 上带来的相对提升`
- 第二层支撑：
  `DeepDive RL val` 上 `rubric_reward`、`task_unfinished` 和工具行为改善
- 第三层支撑：
  自定义 `agent loop + reward bridge + advantage estimator + fixed eval subset + reproducible logging`
- judge 差异必须在项目说明中注明：
  结果基于当前实现中的 DeepSeek judge，因此强调项目内相对比较，不直接对齐论文绝对数值

## fallback 叙事
- 如果 `C-GRPO vs GRPO` 提升 `< 2 pp`：
  主叙事转为“完整实现了 verl multi-turn deep-search RL pipeline，并观察到 rubric_reward、task completion 和工具行为的系统性改善”。
- 如果 `RL vs SFT` 也没有明显增益：
  不主打 benchmark 数字，改为强调工程深度：
  自定义 agent loop
  reward bridge
  C-GRPO advantage
  多服务联调
  固定评测集与可复现日志体系
- 同时把原因解释固定为：
  `4B 模型容量有限`
  `训练预算受限`
  `仅 3 epochs`
  `SFT context length 未完全对齐论文 128k`

## 开始正式训练前的验收门槛
- 5-sample dry run 必须确认 training-step 日志里出现：
  `outcome_reward`
  `rubric_reward`
  `task_unfinished`
  `tool_call_counts`
- 5-sample dry run 必须确认 validation metrics 里同样出现：
  `task_unfinished`
  `tool_call_counts`
  `search_count/open_count/find_count`
- 5-sample dry run 必须确认 `validation_data_dir` 的 JSONL 中包含足够 case study 素材，至少能还原一个样本的完整工具调用链路。
- 固定 subset parquet 生成后，三次独立读取必须校验文件行数、样本索引和哈希一致。
- SFT 20-step smoke 必须先验证 `max_length=65536` 可跑；只有在明确 OOM 时才允许切到 `32768`。
- `Base 10-sample smoke` 必须先跑完，再决定是否在最终材料中保留 base 结果。