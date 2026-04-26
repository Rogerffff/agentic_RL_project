# GPU Eval 计划建议 (2026-04-04)

本文档是对 `post_gpu_revalidation_plan_20260412.md` 的简历导向优化建议。核心目标：**用最少 GPU 预算产出对简历最有价值的评测证据**。

---

## 1. 当前简历策略对 eval 的需求分析

修正后的简历 bullets 核心卖点是**系统工程**（集成 + 异步架构 + 调试），不依赖 benchmark 数字成立。因此 eval 的角色是：

| eval 结果 | 对简历的影响 |
|-----------|------------|
| async_best > SFT（BrowseComp） | **加分**：可在 Bullet 3 追加一句实测 uplift |
| async_best ≈ SFT | **不影响**：当前 bullets 已经不含 benchmark claim |
| async_best < SFT | **不影响**：不提 BrowseComp 数字即可 |

结论：eval 是**可选加分项**，不是简历成立的前提。应以最小预算执行。

---

## 2. 对原计划各 Phase 的具体建议

### 2.1 Preflight（保留，无修改）

原计划的环境检查、密钥、磁盘、入口脚本验证都是必要的，不建议跳过。

### 2.2 Checkpoint 资产准备（精简）

**原计划**：准备 5 个 checkpoint（SFT、sync step70、async16、async19、async23）

**建议精简为 2 个**：

| Checkpoint | 是否需要 | 理由 |
|-----------|---------|------|
| SFT | 需要 | Phase B 的基线 |
| sync step70 | **不需要** | 简历不再做 sync vs async 质量对比 |
| async16 | **不需要** | 训练窗口最高点，但 step23 作为最终 checkpoint 更 defensible |
| async19 | **不需要** | 中间 checkpoint，没有特殊价值 |
| async23 | 需要 | 最终稳定 checkpoint，作为 async_best |

**节省**：3 个 checkpoint 的 merge 时间（如果需要 merge 的话）

注意：
- SFT checkpoint 应该已有现成的 HF 权重（`/root/hf_sft_591step`）
- async23 需要确认是否已有 `huggingface_merged`，若无则需要 merge
- 异步 RL 从 **step70** checkpoint warm-start（经 Stage 0 eval 选出，outcome 最高）
- 总训练量约 5,408 trajectories（论文的 ~10%），其中同步 3,072 + 异步 2,336

### 2.3 Phase A: DeepDive Internal Gate（大幅精简）

**原计划**：对 async16/19/23 三个 checkpoint 做 DeepDive subset64 eval，按 outcome-first 规则选出 async_best。

**建议**：**跳过 Phase A，直接选 async23 作为 async_best**。

理由：
1. 内部 DeepDive subset64 的数字不会写进简历（已决定不写 0.297→0.328 这类小数）
2. async23 是最终稳定运行到结束的 checkpoint，比挑训练窗口最高点更 defensible
3. 节省 ~3h GPU（3 个 eval × ~1h/each）

**如果确实想保险**，只 eval async23 一个，确认它不比 SFT 基线差即可（~1h）。

### 2.4 Phase B: BrowseComp External Gate（精简为 2 个 eval）

**原计划**：对 SFT、sync step70、async_best 三个做 BrowseComp subset256 64k。

**建议**：**只跑 SFT + async23**（2 个 eval）。

理由：
1. step70 的 BrowseComp eval 主要用于 sync→async 退化检查，但简历不再做这个对比
2. SFT vs async23 的对比已足够：如果正向，简历可加一句实测 uplift

**执行命令**（两个 eval 顺序执行）：

```bash
# --- Eval 1: SFT baseline ---
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

OUT_DIR=$HOME/eval_results/bc256_sft NGPUS=8 \
bash examples/carr_deepsearch/scripts/run_eval_integration.sh \
  /root/hf_sft_591step \
  examples/carr_deepsearch/data/browsecomp_eval_subset_256_seed42.parquet \
  bc256_sft

# --- Eval 2: async_best ---
OUT_DIR=$HOME/eval_results/bc256_async23 NGPUS=8 \
bash examples/carr_deepsearch/scripts/run_eval_integration.sh \
  <async23_merged_path>/actor/huggingface_merged \
  examples/carr_deepsearch/data/browsecomp_eval_subset_256_seed42.parquet \
  bc256_async23
```

注意：
- `<async23_merged_path>` 需要替换为 async23 checkpoint merge 后的实际路径
- SFT 路径需要确认远端是否仍为 `/root/hf_sft_591step`
- 确保使用 `run_eval_integration.sh`（sampled recipe），不要用 `run_eval_browsecomp.sh`（greedy 默认）

**预计 GPU 时间**：每个 eval ~2-3h，共 ~4-6h。

### 2.5 Phase C: BrowseComp full / 128k（建议跳过）

**原计划**：Phase B 为正时，补跑 BrowseComp full 64k 和 subset256 128k。

**建议**：跳过。理由：
1. 简历不依赖 benchmark uplift 成立
2. BrowseComp full (1266 samples) 需要 ~8-12h GPU，成本太高
3. subset256 的结果对于面试叙事已经足够（可以说"在 256 样本子集上评测"）

### 2.6 Continuation（建议跳过）

**原计划**：满足条件时从 async_best 继续训练 +4 到 +8 步。

**建议**：跳过。理由：
1. 在已有的训练规模上多加 4-8 步对结果影响有限
2. 需要额外 ~2.5h GPU + 后续再跑 eval
3. 简历 bullets 不依赖更多训练步数

---

## 3. 新增建议：Case Study 提取（最高 ROI）

**这是对简历和面试最有价值的产出，且不消耗 GPU。**

### 3.1 来源

Phase B 的 eval 会在 `$OUT_DIR` 下保存 generation dump（由 `trainer.validation_data_dir` 控制），reward trace 保存在 `$LOG_DIR/${RUN_NAME}_reward_trace.jsonl`。

### 3.2 目标

从 SFT 和 async23 的 BrowseComp eval dump 中，提取 2-3 个对比案例：

| Case | 描述 | 面试价值 |
|------|------|---------|
| Case 1: SFT 失败 → RL 成功 | 同一个问题，SFT 模型搜索不充分/走 shortcut，RL 模型系统性搜索后答对 | 最高：直接展示 RL 训练的效果 |
| Case 2: RL 搜索更全面 | RL 模型引用了更多网页、覆盖了更多 rubric 约束 | 高：展示 C-GRPO 的 rubric 引导效果 |
| Case 3: 仍然失败的 hard case | RL 模型搜索了很多但仍然没答对，展示当前局限性 | 中：面试中展示诚实和技术判断力 |

### 3.3 提取方法

```bash
# 在远端机器上，eval 完成后
# 查看 generation dump
ls $HOME/eval_results/bc256_sft/
ls $HOME/eval_results/bc256_async23/

# 查看 reward trace（包含 outcome_reward、rubric_reward、task_unfinished）
head -5 $HOME/logs/bc256_sft_reward_trace.jsonl
head -5 $HOME/logs/bc256_async23_reward_trace.jsonl
```

选择标准：
1. 找 SFT `outcome_reward=0` 且 async23 `outcome_reward=1` 的样本 → Case 1
2. 找两个都 `outcome_reward=1` 但 async23 `rubric_reward` 更高的样本 → Case 2
3. 找 async23 `tool_call_counts` 很高但 `outcome_reward=0` 的样本 → Case 3

---

## 4. 总结：修正后的执行流程

```
Step 1: Preflight（~30min）
  ├── 环境检查
  ├── 确认 SFT HF 权重路径
  └── 确认 async23 是否需要 merge
         │
Step 2: Merge async23（如需要，~15-30min）
  └── python -m verl.model_merger merge ...
         │
Step 3: Phase B — BrowseComp subset256 64k（~4-6h）
  ├── Eval SFT（~2-3h）
  └── Eval async23（~2-3h）
         │
Step 4: Case Study 提取（0 GPU，~1h 分析）
  ├── 从 generation dump 中选 2-3 个对比案例
  └── 整理成面试可讲的叙事
         │
Step 5: 文档回填
  ├── 如果 async23 > SFT：更新 Bullet 3 追加 BrowseComp uplift
  ├── 如果 ≈ 或 <：不改 bullets，只更新内部 source of truth
  └── 更新 resume_source_of_truth 中的 Estimated → Observed
```

**总 GPU 预算：~5-7h**（原计划 ~15-20h，节省约 65%）

---

## 5. 与原计划的差异总结

| 项目 | 原计划 | 建议 | 节省 |
|------|--------|------|------|
| Phase A: DeepDive subset64 | 3 个 async checkpoint | 跳过（直接选 async23） | ~3h |
| Phase B: BrowseComp subset256 | 3 个 checkpoint（SFT/step70/async_best） | 2 个（SFT/async23） | ~2-3h |
| Phase C: BrowseComp full/128k | 条件执行 | 跳过 | ~8-12h |
| Continuation | 条件执行 +4-8 步 | 跳过 | ~2.5h + eval |
| Case Study | 提到但未展开 | 新增为独立步骤 | 0 GPU |
| **合计** | **~15-20h** | **~5-7h** | **~65%** |
