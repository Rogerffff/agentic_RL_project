# Post-GPU Revalidation Plan (2026-04-14)

## 0. 结论先行

是，补一轮 **最小外部 eval** 对这个项目的简历收尾是合理的。

原因不是“没有外部 benchmark 项目就不成立”，而是：

- 当前 [final_resume_ready.md](examples/carr_deepsearch/docs/final_resume_ready.md) 已经能靠系统工程事实成立
- 但如果能补上 1 轮真实外部 benchmark，就能把故事从“我把系统跑通跑稳了”闭合成“我把系统跑通跑稳了，而且至少验证过它在外部分布上没有明显失效”
- 这对 `agentic RL / post-training` 岗位更完整，也更利于面试追问时落地

本计划因此固定为：

1. 不再开新配置搜索
2. 不再默认继续训练
3. 只做 **最少 GPU 成本、最大简历收益** 的验证闭环
4. 所有新的正式 eval 一律使用 **统一 sampled recipe + 统一 eval budget envelope**

默认闭环只包含：

1. 新机环境拉起
2. `async23` merge
3. `async23` 的 `DeepDive subset64` sampled sanity eval
4. `DeepDive rl_val` 全量 `111` 条上的 `SFT vs async23` sampled internal anchor eval
5. `BrowseComp subset256 64k` 上的 `SFT vs async23` sampled external gate
6. 如有需要，补 1 个 cheap `360 vs 480` budget-sensitivity probe
7. 2-3 个 case study 提取

`step70`、`BrowseComp full`、`128k` 都改为 **条件性 fallback / 加分项**，不是默认步骤。

---

## 1. 本次 GPU 收尾的目标

### 1.1 主目标

补足一个对简历叙事最有价值的外部验证：

- `BrowseComp subset256 64k`
- sampled recipe
- `SFT` vs `async23`

### 1.2 成功标准

满足以下任一条，就算这次 GPU 收尾成功：

- `async23` 在 `BrowseComp subset256` 上明显优于 `SFT`
- `async23` 与 `SFT` 接近，但 case study 能清楚展示 RL 后搜索行为更深、更系统
- 即便 `async23` 不优于 `SFT`，也能通过 `step70` fallback 说明问题出在 checkpoint 选择或训练长度，而不是评测链路错误

### 1.3 非目标

- 不追求“论文复现”
- 不追求重新调参
- 不默认补 continuation
- 不默认跑 `BrowseComp full 1266` 或 `128k`

### 1.4 关于 sync/async budget 差异的工程判断

当前最重要的判断先写清楚：

- **会影响训练信号**，而且最可能影响的是同步 RL 阶段，而不是异步阶段
- 同步阶段的 `wall=360` 是为了解决长尾 rollout 阻塞 GPU 的工程折中，不是论文设定
- 根据 [training_full_history_20260404.md](examples/carr_deepsearch/docs/training_full_history_20260404.md)，同步后段真正恶化的主因是 `rollout_timeout`，不是模型突然不会搜索了
- 这意味着同步阶段的 tighter budget 不是“中性截断”，而是会把更深、更慢的搜索轨迹系统性打成 unfinished / zero-reward
- 异步阶段把 `wall` 放宽到 `480`，同时保持 `tool/search/open/find` budget 不变，更像是在**减轻**这种偏差，而不是引入新的明显惩罚项

但这里不能做单因果归因：

- 异步阶段不只改了 `wall=360 -> 480`
- 还同时改了 `n=4 -> 8`、`gmu=0.3 -> 0.5`、`b=4 -> 3`、以及 async infra 本身
- 所以最终对外不能说“效果提升就是因为 wall 放宽”

这也是为什么本次 GPU 收尾的正式结论只依赖：

- 同一套 sampled eval recipe
- 同一套 eval budget envelope
- `SFT vs async23` 的 matched eval

而不是再拿同步训练过程中的 budget 设置做直接质量结论。

---

## 2. 当前最合理的评测范围

### 2.1 为什么不是重跑一堆 checkpoint

当前简历主卖点已经固定为：

- `verl + CaRR` 集成
- `fully_async_policy` 稳定化
- 长轨迹 agentic RL 的 failure-mode diagnosis

因此这次 GPU 评测只需要回答一个问题：

> 最终稳定 async checkpoint 在外部 benchmark 上，是否至少值得被拿来作为项目收尾版本？

这不需要重新做：

- `async16 / async19 / async23` 全部对比
- `SFT / step70 / step90 / async*` 全池筛选
- continuation + 再评测的大循环

### 2.2 默认评测对象

默认只准备这 3 个资产：

1. `SFT`
2. `async23`
3. `step70` 仅作 fallback 诊断，不默认评

### 2.3 默认评测顺序

1. `async23` on `DeepDive subset64` sampled sanity eval
2. `SFT` on `DeepDive rl_val` full `111`
3. `async23` on `DeepDive rl_val` full `111`
4. `SFT` on `BrowseComp subset256 64k`
5. `async23` on `BrowseComp subset256 64k`
6. 只有在结果需要解释时，才补 `step70`

---

## 3. 新机环境启动

## 3.1 Docker 起点

新机基础镜像固定为：

- `verlai/verl:sgl056.latest`

推荐启动方式：

```bash
docker run --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 --ulimit nofile=65535:65535 \
  -it --rm \
  -v <HOST_REPO>:/root/verl-carr-deepsearch \
  -v <HOST_CHECKPOINTS>:/root/checkpoints \
  -v <HOST_LOGS>:/root/logs \
  -v <HOST_EVAL_RESULTS>:/root/eval_results \
  -v <HOST_MODELS>:/root/models \
  --name carr-eval \
  verlai/verl:sgl056.latest /bin/bash
```

建议在宿主机预先准备：

- 当前项目 repo
- 所需 checkpoint
- `hf_sft_591step` 或对应 SFT HF 权重
- 挂载后的 `logs / eval_results / checkpoints`

## 3.2 用哪个 repo

新机上 **优先使用当前工作区 repo**：

- `/root/verl-carr-deepsearch`

原因：

- 你当前本地 repo 是这轮文档和代码的 source of truth
- 归档在 [CaRR_repo](examples/carr_deepsearch/CaRR_repo) 的旧机完整 repo 主要作为 reference / rollback
- 已核对关键脚本在当前 repo 与 `CaRR_repo` 中一致：
  - `run_async_formal.sh`
  - `run_eval_integration.sh`
  - `smoke_test.py`

因此，新机上不要双 repo 混用。默认只跑当前 repo；如果发现当前 repo 和旧机行为不一致，再回看 `CaRR_repo`。

## 3.3 依赖安装

进入容器后：

```bash
cd /root/verl-carr-deepsearch

# 安装 repo（不覆盖镜像内已配好的主依赖）
pip install --no-deps -e .

# CaRR 奖励服务依赖
pip install -r CaRR/deepsearch_rm_with_rubrics/requirements.txt
pip install quart aiohttp requests
pip install --ignore-installed blinker
```

这组命令来自旧机执行记录：

- [progress_2x5090.md](examples/carr_deepsearch/CaRR_repo/examples/carr_deepsearch/docs/progress_2x5090.md)
- [gpu_execution_plan.md](examples/carr_deepsearch/CaRR_repo/examples/carr_deepsearch/docs/gpu_execution_plan.md)

## 3.4 环境变量

在容器内准备 `$HOME/.env`，至少包含：

```bash
SERPER_API_KEY=...
JINA_API_KEY=...
DEEPSEEK_API_KEY=...
WANDB_API_KEY=...
```

Blackwell / RTX PRO 6000 机器建议显式设置：

```bash
export NCCL_P2P_DISABLE=1
export VERL_USE_EXTERNAL_MODULES=examples.carr_deepsearch.tools.carr_agent_loop,examples.carr_deepsearch.reward.cgrpo_advantage
```

说明：

- `VERL_USE_EXTERNAL_MODULES` 必须在 Ray worker 起之前就可见
- `NCCL_P2P_DISABLE=1` 是旧 Blackwell probe 脚本里的默认值；如果新机验证后不需要，再移除

## 3.5 模块注册检查

```bash
python - <<'PY'
from verl.experimental.agent_loop.agent_loop import _agent_loop_registry
from verl.trainer.ppo.core_algos import ADV_ESTIMATOR_REGISTRY
assert 'carr_tool_agent' in _agent_loop_registry
assert 'cgrpo' in ADV_ESTIMATOR_REGISTRY
print('carr_tool_agent and cgrpo registered OK')
PY
```

## 3.6 手动健康检查与冒烟测试

虽然 `run_eval_integration.sh` 会自动启动 tool/reward server，但新机第一次使用时仍建议先手动做一次健康检查：

```bash
# tool server
python CaRR/tool_server/launch_server.py \
  --search_backend serper \
  --serper_api_key "$SERPER_API_KEY" \
  --jina_api_key "$JINA_API_KEY" \
  --port 7230 &
TOOL_PID=$!

# reward server
(cd CaRR/deepsearch_rm_with_rubrics && python launch_server.py \
  --port 8888 \
  --model_name deepseek-chat \
  --base_url https://api.deepseek.com \
  --api_key "$DEEPSEEK_API_KEY") &
REWARD_PID=$!

sleep 15
python examples/carr_deepsearch/scripts/smoke_test.py --all

kill $TOOL_PID $REWARD_PID
```

如果 `smoke_test.py --all` 不通过，不进入 eval。

---

## 4. Checkpoint 与数据资产

## 4.1 必备 checkpoint

默认只导入：

1. `SFT`
2. `async23`

条件性 fallback：

3. `step70`

不默认导入：

- `async16`
- `async19`
- `step90`

## 4.2 Merge 原则

评测只认：

- `actor/huggingface_merged`

标准命令：

```bash
python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir <checkpoint>/actor \
  --target_dir <checkpoint>/actor/huggingface_merged
```

如果内存紧张，唯一允许的 fallback：

```bash
--use_cpu_initialization
```

## 4.3 BrowseComp 数据集核验结果

本地已确认：

- [browsecomp_eval_subset_256_seed42.parquet](examples/carr_deepsearch/data/browsecomp_eval_subset_256_seed42.parquet) 为 **自然语言题目**
- 它不是密文版数据
- 列结构与现有 eval 脚本兼容：
  - `data_source`
  - `agent_name`
  - `prompt`
  - `ability`
  - `reward_model`
  - `extra_info`
- subset256 共 `256` 条
- full BrowseComp 数据集 [browsecomp_eval.parquet](examples/carr_deepsearch/data/browsecomp_eval.parquet) 共 `1266` 条

因此数据准备不是 blocker。

---

## 5. 默认评测计划

## 5.0 Canonical Eval Recipe（本次 GPU 收尾统一口径）

从这一步开始，所有**正式写进简历或 source-of-truth 的新 eval** 都使用同一套口径。

### 5.0.1 为什么新的正式 eval 不再沿用历史 `wall=360`

历史 `Stage 0 eval` 用 `wall=360`，它适合做当时的 checkpoint gate 和问题排查，但不适合作为本次收尾的最终对外口径。

原因：

- `wall=360` 来自同步训练阶段的吞吐折中，不是论文设定
- async 正式训练主线已经改为 `wall=480 / real_wall=960`
- 如果新一轮正式 eval 继续压回 `360`，会系统性低估更深搜索 checkpoint 的能力
- 这会让最终简历故事混入“旧同步吞吐约束下的表现”，而不是“最终 async 系统在统一预算下的表现”

因此本次 GPU 收尾统一采用：

- `max_rollout_wall_time_s = 480`
- `max_real_rollout_wall_time_s = 960`

并对 `SFT` 和 `async23` **同时使用同一 eval envelope**。

历史 `Stage 0` 的 `subset64 / wall=360` 日志仍然保留，但只作为：

- merge/sampling sanity 的参考
- 旧 checkpoint 选择过程的背景材料

不再作为最终 matched comparator。

### 5.0.2 正式 eval 统一环境变量

```bash
export PROJECT_DIR=/root/verl-carr-deepsearch
export NGPUS=8
export CARR_TP_SIZE=1
export CARR_ENFORCE_EAGER=true
export CARR_SGLANG_ATTENTION_BACKEND=flashinfer

export CARR_VAL_BATCH_SIZE=32
export CARR_VAL_N=1
export CARR_VAL_TEMPERATURE=0.6
export CARR_VAL_TOP_P=0.95
export CARR_VAL_TOP_K=20
export CARR_VAL_DO_SAMPLE=true
export CARR_VAL_REPETITION_PENALTY=1.0
export CARR_VAL_PRESENCE_PENALTY=0.0
export CARR_VAL_FREQUENCY_PENALTY=0.0

export CARR_MAX_RESPONSE_LENGTH=61440
export CARR_MAX_ASSISTANT_TURNS=120
export CARR_MAX_TOOL_RESPONSE_LENGTH=6000
export CARR_ROLLOUT_WALL_TIME_S=480
export CARR_MAX_TOOL_CALLS=88
export CARR_MAX_SEARCH_CALLS=40
export CARR_MAX_OPEN_CALLS=32
export CARR_MAX_FIND_CALLS=20

export CARR_REWARD_TIMEOUT=650
export CARR_TOOL_CLIENT_TIMEOUT_S=120
```

额外 Hydra override 统一追加：

```bash
+actor_rollout_ref.rollout.custom.carr_budget.max_real_rollout_wall_time_s=960
data.validation_shuffle=false
```

说明：

- sampled 参数来自 [rl_debug_findings_20260312.md](examples/carr_deepsearch/docs/rl_debug_findings_20260312.md) 中对 Thinking checkpoint 的最终修正口径
- `run_eval_browsecomp.sh` 的 greedy 默认 **不允许** 用于正式结论
- 所有最终结果一律走 [run_eval_integration.sh](examples/carr_deepsearch/scripts/run_eval_integration.sh)

## 5.1 Phase A: `async23` Internal Sanity Eval

目的：

- 验证 merge 后的 `async23` 可以正常跑 sampled eval
- 验证新机 tool/reward server、Ray、SGLang、外部模块注册都正常
- 在跑更大内部 eval 和 BrowseComp 前，先用更便宜的 `DeepDive subset64` 做 sanity check

执行命令：

先执行 `5.0.2` 的统一环境变量块，再运行下面命令。

```bash
cd /root/verl-carr-deepsearch
bash examples/carr_deepsearch/scripts/run_eval_deepdive64_sampled.sh \
  <ASYNC23_MERGED_PATH> \
  async23_dd64_sanity \
  +actor_rollout_ref.rollout.custom.carr_budget.max_real_rollout_wall_time_s=960
```

通过标准：

- 脚本无 infra 级错误
- sampled eval 正常产出结果与 trace
- `async23` 没有明显灾难性退化

建议停止条件：

- 若 `async23` 在 `DeepDive subset64` 上明显低于 `SFT` 已有基线（例如 `outcome` 低超过 `0.03`），先排查 merge / recipe / model path，不直接进入 BrowseComp

说明：

- `subset64` 的统计方差确实偏大，所以这里只把它当 **sanity check**，不把它当最终质量结论
- 这一步也使用 `wall=480 / real_wall=960`，与本次正式 eval envelope 保持一致
- 本地已经保存了历史 `Stage 0 eval` 的完整日志，可直接作为对照：
  - [eval_gate_20260323_144500_sft.log](examples/carr_deepsearch/CaRR_log/eval_gate_20260323_144500_sft.log)
  - [step70_dd64_8gpu_20260322_153257.log](examples/carr_deepsearch/CaRR_log/step70_dd64_8gpu_20260322_153257.log)
  - [step90_dd64_8gpu_20260322_151340.log](examples/carr_deepsearch/CaRR_log/step90_dd64_8gpu_20260322_151340.log)
- 但这些历史日志的 `wall=360` 只用于 sanity 对照，不再作为本次正式 matched comparator

## 5.2 Phase B: DeepDive Internal Anchor Eval

这一步是为了降低 `subset64` 的高方差问题。

数据集：

- `examples/carr_deepsearch/data/rl_val.parquet`
- 本地已确认总样本数为 `111`

执行对象：

1. `SFT`
2. `async23`

统一 sampled recipe 与 eval envelope：

执行下面两个命令前，先完整执行一次 `5.0.2` 的统一环境变量块。

```bash
export PROJECT_DIR=/root/verl-carr-deepsearch
export NGPUS=8
export CARR_TP_SIZE=1
export CARR_ENFORCE_EAGER=true
export CARR_SGLANG_ATTENTION_BACKEND=flashinfer
export CARR_VAL_BATCH_SIZE=32
export CARR_VAL_N=1
export CARR_VAL_TEMPERATURE=0.6
export CARR_VAL_TOP_P=0.95
export CARR_VAL_TOP_K=20
export CARR_VAL_DO_SAMPLE=true
export CARR_VAL_REPETITION_PENALTY=1.0
export CARR_VAL_PRESENCE_PENALTY=0.0
export CARR_VAL_FREQUENCY_PENALTY=0.0
export CARR_MAX_RESPONSE_LENGTH=61440
export CARR_MAX_ASSISTANT_TURNS=120
export CARR_MAX_TOOL_RESPONSE_LENGTH=6000
export CARR_ROLLOUT_WALL_TIME_S=480
export CARR_MAX_TOOL_CALLS=88
export CARR_MAX_SEARCH_CALLS=40
export CARR_MAX_OPEN_CALLS=32
export CARR_MAX_FIND_CALLS=20
export CARR_REWARD_TIMEOUT=650
export CARR_TOOL_CLIENT_TIMEOUT_S=120
```

SFT：

```bash
OUT_DIR=$HOME/eval_results/dd111_sft \
bash examples/carr_deepsearch/scripts/run_eval_integration.sh \
  <SFT_HF_PATH> \
  examples/carr_deepsearch/data/rl_val.parquet \
  dd111_sft \
  data.validation_shuffle=false \
  data.val_max_samples=111 \
  +actor_rollout_ref.rollout.custom.carr_budget.max_real_rollout_wall_time_s=960
```

async23：

```bash
OUT_DIR=$HOME/eval_results/dd111_async23 \
bash examples/carr_deepsearch/scripts/run_eval_integration.sh \
  <ASYNC23_MERGED_PATH> \
  examples/carr_deepsearch/data/rl_val.parquet \
  dd111_async23 \
  data.validation_shuffle=false \
  data.val_max_samples=111 \
  +actor_rollout_ref.rollout.custom.carr_budget.max_real_rollout_wall_time_s=960
```

这一步的作用：

- 给 `BrowseComp` 之前增加一个比 `subset64` 更稳的内部 anchor
- 如果 `async23` 在 `rl_val` 全量 `111` 上也明显不如 `SFT`，就不必对 `BrowseComp` 抱过高期待
- 如果 `async23` 在 `111` 上至少不差于 `SFT`，再进入外部 gate，故事会完整得多

## 5.3 Phase C: BrowseComp External Gate

这是本次 GPU 收尾的核心步骤。

默认只跑 2 个 eval：

1. `SFT`
2. `async23`

数据集：

- `examples/carr_deepsearch/data/browsecomp_eval_subset_256_seed42.parquet`

统一 sampled recipe 与 eval envelope：

执行下面两个命令前，先完整执行一次 `5.0.2` 的统一环境变量块。

```bash
export PROJECT_DIR=/root/verl-carr-deepsearch
export NGPUS=8
export CARR_TP_SIZE=1
export CARR_ENFORCE_EAGER=true
export CARR_SGLANG_ATTENTION_BACKEND=flashinfer
export CARR_VAL_BATCH_SIZE=32
export CARR_VAL_N=1
export CARR_VAL_TEMPERATURE=0.6
export CARR_VAL_TOP_P=0.95
export CARR_VAL_TOP_K=20
export CARR_VAL_DO_SAMPLE=true
export CARR_VAL_REPETITION_PENALTY=1.0
export CARR_VAL_PRESENCE_PENALTY=0.0
export CARR_VAL_FREQUENCY_PENALTY=0.0
export CARR_MAX_RESPONSE_LENGTH=61440
export CARR_MAX_ASSISTANT_TURNS=120
export CARR_MAX_TOOL_RESPONSE_LENGTH=6000
export CARR_ROLLOUT_WALL_TIME_S=480
export CARR_MAX_TOOL_CALLS=88
export CARR_MAX_SEARCH_CALLS=40
export CARR_MAX_OPEN_CALLS=32
export CARR_MAX_FIND_CALLS=20
export CARR_REWARD_TIMEOUT=650
export CARR_TOOL_CLIENT_TIMEOUT_S=120
```

SFT：

```bash
OUT_DIR=$HOME/eval_results/bc256_sft \
bash examples/carr_deepsearch/scripts/run_eval_integration.sh \
  <SFT_HF_PATH> \
  examples/carr_deepsearch/data/browsecomp_eval_subset_256_seed42.parquet \
  bc256_sft \
  data.validation_shuffle=false \
  data.val_max_samples=256 \
  +actor_rollout_ref.rollout.custom.carr_budget.max_real_rollout_wall_time_s=960
```

async23：

```bash
OUT_DIR=$HOME/eval_results/bc256_async23 \
bash examples/carr_deepsearch/scripts/run_eval_integration.sh \
  <ASYNC23_MERGED_PATH> \
  examples/carr_deepsearch/data/browsecomp_eval_subset_256_seed42.parquet \
  bc256_async23 \
  data.validation_shuffle=false \
  data.val_max_samples=256 \
  +actor_rollout_ref.rollout.custom.carr_budget.max_real_rollout_wall_time_s=960
```

注意：

- 必须使用 `run_eval_integration.sh`
- 不允许使用 `run_eval_browsecomp.sh` 的 greedy 默认
- 本次正式 external gate 统一使用 `wall=480 / real_wall=960`
- 这不是为了“偏向 async”，而是为了在统一、更接近最终 async operating point 的 budget 下比较 `SFT` 和 `async23`
- `run_eval_integration.sh` 已包含：
  - `ray stop --force`
  - tool server 启动
  - reward server 启动
  - health check

## 5.3.1 Optional: Budget Sensitivity Probe（便宜但信息量高）

如果你想回答“`wall=360 -> 480` 到底影响多大”这个问题，最便宜的补充不是重训，而是补一个小型 sensitivity probe。

推荐只跑：

1. `async23` on `DeepDive subset64`, `wall=360 / real_wall=720`
2. `async23` on `DeepDive subset64`, `wall=480 / real_wall=960`

其余 sampled 参数完全一致。

执行方式：

```bash
OUT_DIR=$HOME/eval_results/async23_dd64_w360 \
CARR_ROLLOUT_WALL_TIME_S=360 \
bash examples/carr_deepsearch/scripts/run_eval_deepdive64_sampled.sh \
  <ASYNC23_MERGED_PATH> \
  async23_dd64_w360 \
  +actor_rollout_ref.rollout.custom.carr_budget.max_real_rollout_wall_time_s=720

OUT_DIR=$HOME/eval_results/async23_dd64_w480 \
CARR_ROLLOUT_WALL_TIME_S=480 \
bash examples/carr_deepsearch/scripts/run_eval_deepdive64_sampled.sh \
  <ASYNC23_MERGED_PATH> \
  async23_dd64_w480 \
  +actor_rollout_ref.rollout.custom.carr_budget.max_real_rollout_wall_time_s=960
```

它不能证明训练时的因果影响，但能回答一个非常实际的问题：

- 更宽的 wall budget 是否会显著释放当前 checkpoint 的 deep-search 能力

若 `w480` 明显优于 `w360`，面试里可以安全说：

- tighter sync-era wall budget 很可能压制了更深搜索行为

若差异很小，则说明：

- 对当前 checkpoint 来说，wall budget 不是 eval 结果的主导因素

## 5.4 Phase D: 仅在必要时跑 `step70`

`step70` 不再是默认评测对象，只在下面情况触发：

- `async23` 明显不如 `SFT`
- 或 `async23` 与 `SFT` 非常接近，但你需要判断问题出在 async checkpoint 本身还是“RL 本来也没带来外部增益”

触发后执行：

```bash
OUT_DIR=$HOME/eval_results/bc256_step70 \
bash examples/carr_deepsearch/scripts/run_eval_integration.sh \
  <STEP70_HF_PATH> \
  examples/carr_deepsearch/data/browsecomp_eval_subset_256_seed42.parquet \
  bc256_step70 \
  data.validation_shuffle=false \
  data.val_max_samples=256 \
  +actor_rollout_ref.rollout.custom.carr_budget.max_real_rollout_wall_time_s=960
```

解释逻辑：

- 若 `step70 > async23` 且 `step70 >= SFT`：
  - 说明 async23 未必是最佳质量 checkpoint
  - 简历继续以系统工程为主，不强写 async external uplift
- 若 `step70` 也不优于 `SFT`：
  - 说明这次 RL 训练在外部 benchmark 上没有拉开
  - 简历仍成立，但只保留系统/方法/评测诊断表述

## 5.5 Phase E: 只在外部结果明显为正时补加分项

只有当 `async23 > SFT` 且结果足够干净时，才允许继续跑：

1. `BrowseComp full 64k` (`1266` 条)
2. 如果仍然正，且你还想写 test-time scaling，再补 `128k`

默认不跑。

---

## 6. 本次不默认做 continuation

这次 GPU 收尾默认 **不继续训练**。

原因：

- 你的最终展示已经不依赖再多跑几步
- 继续训练会引入新的 checkpoint selection 问题
- 对简历故事来说，先补上“真实外部 eval + case study”比继续堆 step 更值钱

只有在下面条件同时满足时，才考虑单次 continuation：

- `async23` 在 `DeepDive` sanity 上不差
- `BrowseComp subset256` 对 `async23` 是中性而不是负面
- failure mode 明确指向 `response_limit / unfinished`
- 你愿意把它作为“训练长度验证”，而不是“收尾评测”

即便如此，也不属于本次默认计划。

---

## 7. 结果解释规则

## 7.1 如果 `async23 > SFT`

这是最佳情况。

后续动作：

- 更新 [resume_source_of_truth_20260412.md](examples/carr_deepsearch/docs/resume_source_of_truth_20260412.md)
- 更新 [final_resume_ready.md](examples/carr_deepsearch/docs/final_resume_ready.md)
- 把 `BrowseComp subset256` 的实测 uplift 写入对外材料

## 7.2 如果 `async23 ≈ SFT`

这是可接受结果。

后续动作：

- 不改核心简历 bullets
- 在 source-of-truth 里把 `Estimated` 替换成 `Observed neutral`
- 重点讲：
  - async 架构稳定化
  - 长轨迹工具使用行为
  - case study

## 7.3 如果 `async23 < SFT`

先跑 `step70` 诊断。

如果 `step70` 也不优于 `SFT`，则：

- 不在正式简历中写 external uplift
- 仍保留系统工程成果
- 在面试中诚实解释：
  - 训练量仍低于论文规模
  - 本项目完成了系统实现与稳定化验证
  - 外部 benchmark 没有形成足够强的提升

---

## 8. Case Study 提取

这一步对面试价值很高，而且可以在日志和 dump 下载到本地后慢慢做，不需要继续占用 GPU。

从 Phase B 的 eval 结果里至少提 2-3 个案例：

1. `SFT` 失败而 `async23` 成功
2. 两者都成功，但 `async23` 的搜索或引用更完整
3. `async23` 仍失败的 hard case

优先查看：

- `$HOME/eval_results/<run_name>/`
- `$HOME/logs/<run_name>_reward_trace.jsonl`
- `$HOME/logs/<run_name>_tool_stats.json`
- `$HOME/logs/<run_name>.log`
- `$HOME/logs/<run_name>.launcher.log`

建议在远端评测完成后统一下载以下产物到本地再做 case study：

1. `eval_results/<run_name>/`
2. `logs/<run_name>.log`
3. `logs/<run_name>.launcher.log`
4. `logs/<run_name>_tool.log`
5. `logs/<run_name>_reward.log`
6. `logs/<run_name>_reward_trace.jsonl`
7. `logs/<run_name>_tool_stats.json`

建议记录的字段：

- 问题文本
- 最终答案
- `outcome_reward`
- `task_unfinished`
- `termination_*`
- `tool_call_counts`
- `search/open/find_count`

---

## 9. 文档回填顺序

GPU 评测完成后，文档更新顺序固定为：

1. 更新 [resume_source_of_truth_20260412.md](examples/carr_deepsearch/docs/resume_source_of_truth_20260412.md)
2. 更新 [final_resume_ready.md](examples/carr_deepsearch/docs/final_resume_ready.md)
3. 如有必要，再更新 interview Q&A 文档

具体规则：

- `BrowseComp` 真实结果一旦到位，先替换 placeholder
- 若结果为正，再把 final resume 里的 benchmark 表述升级为实测值
- 若结果中性或负面，不改主 bullets，只补一段对结果的诚实解释

---

## 10. 最终执行清单

### 必做

1. 启动新机 Docker 环境
2. 安装 repo 与 CaRR 依赖
3. 准备 `.env`
4. 模块注册检查
5. 手动 `smoke_test.py --all`
6. merge `async23`
7. `async23` 跑 `DeepDive subset64` sanity eval
8. `SFT` 跑 `DeepDive rl_val` 全量 `111`
9. `async23` 跑 `DeepDive rl_val` 全量 `111`
10. `SFT` 跑 `BrowseComp subset256 64k`
11. `async23` 跑 `BrowseComp subset256 64k`
12. 如有需要，补 `async23 dd64 w360 vs w480` sensitivity probe
13. 下载 logs / eval dumps 到本地
14. 在本地提 2-3 个 case study
15. 回填 `resume_source_of_truth` 与 `final_resume_ready`

### 仅在必要时做

1. `step70` 的 `BrowseComp subset256 64k`
2. `BrowseComp full 64k`
3. `128k`
4. continuation

---

## 11. Claim 约束

这轮 GPU 收尾后，仍然遵守下面的对外规则：

- 不写“复现论文结果”
- 不把 `sync vs async` 写成质量结论，只写成系统工程对比
- 不把 raw step 数当主卖点
- 不把 `timeout=0%` 单独写成 headline
- 外部 benchmark 只有在真实 sampled eval 完成后才能写成 `Observed`

对外最重要的升级目标只有一个：

> 把当前项目从“系统已经跑通跑稳”升级成“系统已经跑通跑稳，并完成了最小可信外部验证”。
