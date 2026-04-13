# Post-GPU Revalidation Plan (2026-04-12)

## 1. 新机 Preflight

新 GPU 机器上线后，先做环境和依赖检查，不直接启动训练或评测。

### 1.1 环境检查

- Python 版本、`torch`、CUDA、NCCL
- `vLLM` 或 `SGLang` 是否与当前 `verl` 分支兼容
- `ray` 是否可正常启动与停止
- `flashinfer` / attention backend 是否可用

### 1.2 密钥与环境变量

- `SERPER_API_KEY` 或 `SERPAPI_API_KEY`
- `JINA_API_KEY`
- `DEEPSEEK_API_KEY`
- `WANDB_API_KEY`
- `$HOME/.env` 是否存在且权限合理

### 1.3 机器资源

- GPU 数量是否满足目标配置
- 单卡显存是否满足 `64k` agent rollout
- 磁盘空间是否足以容纳 checkpoint、merge 产物、eval dumps、logs
- checkpoint 和 logs 目录是否有写权限

### 1.4 入口脚本与接口

当前 runbook 只依赖以下接口：

- `bash examples/carr_deepsearch/scripts/run_eval_integration.sh <model_path> <eval_file> <run_name>`
- `python -m verl.model_merger merge --backend fsdp --local_dir ... --target_dir ...`
- 所有评测 checkpoint 必须是 `huggingface_merged`
- `DeepDive subset64` 和 `BrowseComp subset256` 的 sampled recipe 固定为：
  - `temperature=0.6`
  - `top_p=0.95`
  - `top_k=20`
  - `do_sample=true`

---

## 2. Checkpoint 资产准备

资产准备顺序固定如下，不再扩大 checkpoint 池：

1. `SFT`
2. `sync step70`
3. `async16`
4. `async19`
5. `async23`

每个 checkpoint 都要先确认：

- 路径是否存在
- 是否已有 `actor/huggingface_merged`
- 是否具备后续评测所需的 tokenizer/config

若 checkpoint 目录下只有 FSDP shard，则先进入 merge 流程，不跳过。

---

## 3. Merge 流程

### 3.1 原则

- 评测前只认 `actor/huggingface_merged`
- 不直接拿 FSDP shard 目录做 eval
- 不默认相信 `actor/huggingface` 一定包含完整权重

### 3.2 标准命令

```bash
python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir <checkpoint>/actor \
  --target_dir <checkpoint>/actor/huggingface_merged
```

### 3.3 唯一允许的 fallback

如果 merge 过程中模型初始化内存吃紧，只允许补：

```bash
--use_cpu_initialization
```

不在这一步更换其他超参或引入新脚本。

### 3.4 相关实现依据

- `verl/model_merger/__main__.py`
- `examples/carr_deepsearch/scripts/run_async_formal.sh`

---

## 4. Eval 顺序

### 4.1 总原则

- 先内部 gate，再外部 gate
- 先 sampled，再谈任何结果
- 不使用 `run_eval.sh` 或 `run_eval_browsecomp.sh` 的 greedy 默认作为最终结论

### 4.2 Phase A: DeepDive Internal Gate

执行对象：

- `async16`
- `async19`
- `async23`

数据集：

- `examples/carr_deepsearch/data/rl_val_subset_64_seed42.parquet`

口径：

- `64k`
- sampled eval
- `temperature=0.6`
- `top_p=0.95`
- `top_k=20`
- `do_sample=true`
- `max_assistant_turns=120`
- `max_tool_response_length=6000`

比较规则：

1. 先看 `outcome_reward`
2. 若差值 `< 0.02`，看 `rubric_reward`
3. 若仍接近，看 `task_unfinished`
4. 再看 `termination_response_limit`
5. 最后看 `termination_rollout_timeout`

输出：

- 选出唯一 `async_best`
- 若 top1 / top2 差距过小，可把两个候选一并带入 Phase B

### 4.3 Phase B: BrowseComp External Gate

默认至少跑 3 个 eval：

1. `SFT`
2. `sync step70`
3. `async_best`

若内部 gate 无法唯一决策，最多跑 4 个 eval。

数据集：

- `examples/carr_deepsearch/data/browsecomp_eval_subset_256_seed42.parquet`

推荐命令模板：

```bash
export CARR_VAL_TEMPERATURE=0.6
export CARR_VAL_TOP_P=0.95
export CARR_VAL_TOP_K=20
export CARR_VAL_DO_SAMPLE=true
export CARR_MAX_RESPONSE_LENGTH=61440
export CARR_MAX_ASSISTANT_TURNS=120
export CARR_MAX_TOOL_RESPONSE_LENGTH=6000
export CARR_ROLLOUT_WALL_TIME_S=360
export CARR_MAX_TOOL_CALLS=88
export CARR_MAX_SEARCH_CALLS=40
export CARR_MAX_OPEN_CALLS=32
export CARR_MAX_FIND_CALLS=20
export CARR_SGLANG_ATTENTION_BACKEND=flashinfer

OUT_DIR=$HOME/eval_results/<run_name> NGPUS=8 \
bash examples/carr_deepsearch/scripts/run_eval_integration.sh \
  <model_path>/actor/huggingface_merged \
  examples/carr_deepsearch/data/browsecomp_eval_subset_256_seed42.parquet \
  <run_name>
```

说明：

- 这里显式使用 `run_eval_integration.sh`
- 不允许因为图省事而直接走 greedy 默认脚本

### 4.4 Phase C: 条件性加分项

只有当 `BrowseComp subset256 64k` 对 `async_best` 给出清晰正结果时，才补：

1. `BrowseComp full 64k`
2. 如果仍为正且预算还够，再补 `BrowseComp subset256 128k`

若 `subset256 64k` 仍是 neutral，则停止，不继续烧预算。

---

## 5. Continuation 触发条件

只允许一次短 continuation，且不是默认动作。

### 5.1 必须同时满足

- `async_best` 在 `DeepDive subset64` 上不差于 `step70`
- `BrowseComp subset256 64k` 至少不差于 `SFT/step70`
- 当前主坏因是 `response_limit / unfinished`
- 不是 infra 回退、merge 问题、或 eval recipe 错误导致的差结果

### 5.2 允许的 continuation 范围

- 只从 `async_best` 恢复，不默认从最新 step 恢复
- 保持当前主线配置，不再开新配置搜索
- 只补 `+4` 到 `+8` 个 async step

### 5.3 continuation 后的唯一补测

- `DeepDive subset64`
- `BrowseComp subset256 64k`

如果 continuation 之后仍未形成清晰外部提升，则直接收尾。

---

## 6. 文档回填规则

新 GPU 评测结束后，更新顺序固定如下：

1. 先替换 `resume_source_of_truth_20260412.md` 中的 `External benchmark placeholder`
2. 再更新 `Claims Matrix` 中对应条目的状态：
  - `Estimated -> Observed`
  - `Pending -> Observed`
3. 再决定中文/英文第三条 bullet 使用哪一版：
  - `Evaluated / Diagnosed`
  - 或 `benchmark uplift`
4. 最后更新面试讲法

### 6.1 不允许的回填方式

- 不允许把 placeholder 和真实值并存
- 不允许把仍未实测的 `BrowseComp` uplift 改写成既成事实
- 不允许因为 continuation 多跑了几步就直接 claim “论文复现”

---

## 7. 产物归档清单

每次评测或 continuation 完成后，至少归档：

- merge 命令和产物路径
- eval 命令和 sampled 参数
- 主日志路径
- reward trace 路径
- tool stats 路径
- validation output 路径
- 最终用于文档回填的摘要表

推荐整理出的最终产物：

1. `DeepDive subset64` 对比表
2. `BrowseComp subset256 64k` 对比表
3. 若为正结果，再补 `BrowseComp full 64k` / `subset256 128k`
4. 3 个 case study
  - 1 个成功样例
  - 1 个 citation 更完整样例
  - 1 个失败样例（明确是 `response_limit` 或 `unfinished`）

---

## 8. Claim 约束

两份文档后续更新时都必须遵守同一套规则：

- 不写“复现论文结果”
- 不写未验证的 external uplift 为既成事实
- 不把 raw step 数当主卖点
- 不把 `timeout=0%` 单独写成 headline
- 不把 `sync vs async` 写成质量结论，只写成系统工程对比

如果某条内容将被复制进正式简历，必须满足以下之一：

- 它不是 `Estimated`
- 或它是纯系统工程事实，不依赖外部 benchmark placeholder
