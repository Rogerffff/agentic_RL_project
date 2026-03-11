# CaRR DeepSearch GPU Execution Plan

4 x A100-80G (推荐) / RTX Pro 6000 (96GB) 执行计划。
所有命令假设 `cd /root/verl-carr-deepsearch` 且环境变量已导出。

---

## 0. 前置准备：本地 Mac 操作

> GPU 服务器上不执行这些步骤，仅在本地 Mac 上完成。

### 0.1 代码提交与推送

```bash
cd ~/Desktop/agentic-RL-project/verl-carr-deepsearch
git add -A && git commit -m "准备正式训练"
git push origin feature/carr-deepsearch
```

### 0.2 数据文件 SCP 传输

`.parquet` 文件在 `.gitignore` 中，不会被 git 推送。需要手动 SCP 传输到 GPU 服务器：

```bash
# 传输所有 parquet 数据文件
scp -P <PORT> \
    examples/carr_deepsearch/data/sft_train.parquet \
    examples/carr_deepsearch/data/sft_val.parquet \
    examples/carr_deepsearch/data/rl_train.parquet \
    examples/carr_deepsearch/data/rl_val.parquet \
    examples/carr_deepsearch/data/browsecomp_eval.parquet \
    root@<HOST>:/root/verl-carr-deepsearch/examples/carr_deepsearch/data/

# 传输 CaRR 原始数据（SFT 数据重新生成需要）
scp -P <PORT> \
    CaRR/data/deepdive-sft-glm46-trajectory-1k.jsonl \
    root@<HOST>:/root/verl-carr-deepsearch/CaRR/data/
```

> **注意**: `browsecomp_eval_subset_256_seed42.parquet` 尚未生成，将在 GPU 服务器上的 Step 1 中生成。

---

## 1. GPU 服务器环境搭建

### 1.1 Docker 镜像

```
镜像: verlai/verl:sgl056.latest
基于: lmsysorg/sglang:v0.5.6.post2
PyTorch: 2.9.1+cu129
Python: 3.12
CUDA: 12.9
SGLang: 0.5.6.post2
```

### 1.2 硬件验证

```bash
# GPU 验证
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
python -c "import sglang; print('SGLang:', sglang.__version__)"
```


| GPU 架构       | SM  | attention_backend | 配置方式                                                                           |
| ------------ | --- | ----------------- | ------------------------------------------------------------------------------ |
| A100         | 80  | `fa3` (默认)        | 无需配置                                                                           |
| H100         | 90  | `fa3` (默认)        | 无需配置                                                                           |
| RTX 5090     | 120 | `flashinfer`      | `+actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=flashinfer` |
| RTX Pro 6000 | 未知  | 需测试               | 先试默认，失败则用 `flashinfer`                                                         |


### 1.3 代码部署

```bash
# 克隆代码（含 CaRR 子模块）
git clone --recurse-submodules <REPO_URL> verl-carr-deepsearch
cd verl-carr-deepsearch && git checkout feature/carr-deepsearch
git submodule update --init --recursive

# 安装 verl（--no-deps 避免覆盖 Docker 已有依赖）
pip install --no-deps -e .

# CaRR 依赖（quart 需特殊处理 blinker 冲突）
pip install -r CaRR/deepsearch_rm_with_rubrics/requirements.txt
pip install quart aiohttp requests
pip install --ignore-installed blinker

# 验证关键依赖可导入
python -c "import pandas, datasets, pyarrow, quart, aiohttp; print('All dependencies OK')"
```

### 1.4 验证模块注册

```bash
export VERL_USE_EXTERNAL_MODULES=examples.carr_deepsearch.tools.carr_agent_loop,examples.carr_deepsearch.reward.cgrpo_advantage
python -c "
from verl.experimental.agent_loop.agent_loop import _agent_loop_registry
from verl.trainer.ppo.core_algos import ADV_ESTIMATOR_REGISTRY
assert 'carr_tool_agent' in _agent_loop_registry
assert 'cgrpo' in ADV_ESTIMATOR_REGISTRY
print('carr_tool_agent and cgrpo registered OK')
"
```

### 1.5 环境变量

```bash
export SERPER_API_KEY="..."       # Serper.dev 搜索 API
export JINA_API_KEY="..."         # Jina Reader API
export DEEPSEEK_API_KEY="..."     # DeepSeek LLM Judge
export WANDB_API_KEY="..."        # WandB 日志
export NGPUS=4                    # GPU 数量
```

### 1.6 数据就绪检查

数据来源：

- **RL parquet** (`rl_train.parquet`, `rl_val.parquet`): 从本地 Mac 通过 SCP 传输（§0.2），**不在 GPU 上重建**
- **BrowseComp** (`browsecomp_eval.parquet`): 同上，通过 SCP 传输
- **SFT parquet**: 需在 GPU 上使用 canonical tool schemas **重新生成**（确保 SFT-RL 工具描述一致）
- **BrowseComp 256 / 64 固定子集**: 在 GPU 上生成（固定 seed 可复现）

```bash
cd /root/verl-carr-deepsearch

# 重新生成 SFT 数据（从 canonical YAML tool schemas）
python examples/carr_deepsearch/data_preprocess/preprocess_carr_sft.py \
    --input_file CaRR/data/deepdive-sft-glm46-trajectory-1k.jsonl \
    --output_dir examples/carr_deepsearch/data \
    --model_name Qwen/Qwen3-4B

# 生成 BrowseComp 256 子集（固定 seed 可复现）
python examples/carr_deepsearch/data_preprocess/preprocess_browsecomp.py \
    --output_dir examples/carr_deepsearch/data \
    --subset_size 256 --subset_seed 42

# 生成 BrowseComp 64 子集（RL 中后期单次 probe）
python examples/carr_deepsearch/data_preprocess/preprocess_browsecomp.py \
    --output_dir examples/carr_deepsearch/data \
    --subset_size 64 --subset_seed 42
```

验证（RL parquet 来自 SCP，不校验行数）：

```bash
python -c "
import pandas as pd
for f, n in [
    ('sft_train.parquet', 791), ('sft_val.parquet', 41),
    ('rl_train.parquet', None), ('rl_val.parquet', None),
    ('browsecomp_eval.parquet', None),
    ('browsecomp_eval_subset_256_seed42.parquet', 256),
    ('browsecomp_eval_subset_64_seed42.parquet', 64),
]:
    path = f'examples/carr_deepsearch/data/{f}'
    df = pd.read_parquet(path)
    if n: assert len(df) == n, f'FAIL: {f} has {len(df)} rows, expected {n}'
    print(f'OK: {f} ({len(df)} rows)')
"
```

### 1.7 磁盘空间

```bash
df -h /root   # 确认可用空间
```

**磁盘需求估算：**


| 内容                       | 大小           | 说明                                |
| ------------------------ | ------------ | --------------------------------- |
| Qwen3-4B 模型缓存            | ~8 GB        | HuggingFace 自动下载                  |
| SFT checkpoints (3个)     | ~24 GB       | 3 epochs × ~8 GB/HF model         |
| GRPO checkpoints (~4个, optional) | ~30 GB | 默认 `save_freq=99`，默认 b16 formal 为 step `99/198/297/396` |
| C-GRPO checkpoints (~4个) | ~30 GB       | 同上                                |
| 数据文件                     | ~1 GB        | parquet 文件                        |
| 验证 dump (JSONL)          | ~5 GB        | validation_data_dir               |
| WandB 本地缓存               | ~2 GB        |                                   |
| **合计**                   | **~100 GB**  | 完整方案含 optional GRPO 时按上限估算               |
| **建议最低**                 | **≥ 250 GB** | 留有余量应对中间产物、重复 eval 与临时日志                |


> **2x5090 教训**: Step 11 中 `save_freq=1` 导致 7 个 checkpoint × 32GB = 224GB 瞬间磁盘满。当前默认改为 `save_freq=99`，用于配合 b16/n8 formal 的 `396` 步口径控制磁盘增长。

---

## 2. 4-GPU 适配

`carr_grpo.yaml` 默认 `trainer.n_gpus_per_node: 8`。以下适配方式：


| 阶段   | 脚本            | 适配方式                                                                      |
| ---- | ------------- | ------------------------------------------------------------------------- |
| SFT  | `run_sft.sh`  | `$NGPUS` 控制 torchrun nproc，但需追加 `trainer.n_gpus_per_node=$NGPUS` 确保配置记录一致 |
| RL   | `run_rl.sh`   | 透传 `"$@"`，**必须追加** `trainer.n_gpus_per_node=$NGPUS`                       |
| Eval | `run_eval.sh` | 内置 `trainer.n_gpus_per_node="${NGPUS:-4}"`，已适配                            |


---

## 3. 已知风险与注意事项

> 以下问题来自 2x5090 冒烟测试（progress_2x5090.md）的 11 个已修复问题和 10 个未验证风险。

### 3.1 已修复问题（代码层面）


| #   | 问题                                  | 状态    | Commit     |
| --- | ----------------------------------- | ----- | ---------- |
| 7   | SGLang attention_backend 不支持 SM120  | ✅ 已修复 | `56392880` |
| 10  | numpy.float32 赋值给 torch.FloatTensor | ✅ 已修复 | `012f0575` |
| 11  | SFT vs RL tool schema 不一致           | ✅ 已修复 | `8ff047cd` |


### 3.2 运行时注意事项（每次部署都需检查）


| #   | 问题                                       | 影响                  | 解决方案                                                 |
| --- | ---------------------------------------- | ------------------- | ---------------------------------------------------- |
| 1   | Ray dashboard 崩溃导致 GCS 超时                | 训练无法启动              | `ray start --head --include-dashboard=false`         |
| 2   | Hydra struct 拒绝新 key                     | `ray_kwargs` 不在默认配置 | 使用 `+` 前缀：`+ray_kwargs.ray_init.address=auto`        |
| 3   | `SFT_MODEL_PATH` env var 不可见于 Ray worker | 模型加载失败              | CLI override 绝对路径                                    |
| 5/8 | 相对路径在 Ray worker 中失效                     | 数据/reward/tool 加载失败 | run_rl.sh/run_eval.sh 已用 `$PROJECT_DIR` 绝对路径覆盖       |
| 9   | `carr_tool_agent` 未注册                    | agent loop 找不到      | `ray start` **前** export `VERL_USE_EXTERNAL_MODULES` |


### 3.3 正式训练风险（冒烟测试未暴露）


| #   | 风险                  | 严重性 | 解决方案                                                    |
| --- | ------------------- | --- | ------------------------------------------------------- |
| 12  | NCCL 超时（默认 600s）    | 高   | 追加 `actor_rollout_ref.nccl_timeout=1800`                |
| 13  | 验证阶段 batch 过大        | 中   | 默认 RL 配置已设 `data.val_batch_size=16`，formal 与 dry run 保持该值 |
| 14  | Serper API 额度不足     | 高   | 免费 2500 次 vs 单次正式 RL 约需 `76K-127K` 次搜索，若补跑 `GRPO` 预算约翻倍，**必须购买付费套餐** |
| 15  | DeepSeek API 并发限制   | 中   | 默认 formal `b16/n8` 约 `128 trajectories/step`；若升到 `b32/n8` 约 `256/step`，需先在 `Run 2b` 观测重试/超时 |
| 16  | 工具服务器尾延迟与单实例压力     | 中   | 先用 `Run 2b` 在 `b16/n8` 与可选 `b32/n8` 下实测；不再直接规划 multi-worker 方案 |
| 17  | 高并发下外部 API 长尾/429 | 低   | 连接池、`asyncio.sleep`、Jina semaphore 已上线；正式前仍需看 `Run 2b` 的 timeout/retry 情况 |
| 18  | 验证阶段奖励服务器超时         | 中   | 每样本 650s timeout，大批量验证耗时长                               |
| 19  | FSDP→HF 转换仅支持 2 卡   | 低   | 当前 run_rl.sh 自动转换为 HF，4 卡需验证                            |
| 20  | Checkpoint 恢复       | 中   | 崩溃后需从最近 checkpoint 恢复（见 §7）                             |
| 21  | 磁盘空间不足              | 高   | 需 ≥250 GB；若同时保留 optional `GRPO` checkpoints、eval dumps 与重复重跑产物，500 GB 更从容 |


### 3.4 API 用量估算

以下估算分为四个层级：
- `Run 2a Dry`：只为链路验收，建议先小额采购。
- `Run 2b Dry`：按正式并发口径做 `1 step` 稳定性验证。
- `首轮正式执行`：默认先跑 `C-GRPO formal (b16/n8)`。
- `完整方案`：在 `C-GRPO` 跑出正向信号后，再补 `GRPO` baseline。

其中 `Jina Reader` 与 `DeepSeek` 优先按 `token` 预算，`Serper` 按 `request` 预算。

| API | Run 2a Dry 预计实际消耗 | Run 2a Dry 建议先采购 | 单次正式 C-GRPO 主估计 | 完整方案（补 GRPO）安全预算 | 依据 |
| --- | --- | --- | --- | --- | --- |
| Serper.dev 搜索 | `60-150` 次 | `>=500` 次 | `76K-127K` 次 | `152K-254K` 次 | `Run 2a Dry` 约 `30` trajectories（train `10` + val `20`，默认 `val_kwargs.n=1`），按 `2-5 search/traj`；单次正式 RL 按默认 `396 steps × 16 prompts × n=8 = 50,688 trajectories`，再按 `1.5-2.5 search/traj` |
| Jina Reader | `0.4M-3.2M` Reader tokens | `5M-8M` Reader tokens | `1.1B-1.4B` Reader tokens | `2.2B-2.9B` Reader tokens | `Run 2a Dry` 约 `30 traj × 1-2 open/traj`；正式 `C-GRPO` 按 `50,688 traj × 1.5-2.0 open/traj`，完整方案约翻倍 |
| DeepSeek Judge | `0.2M-0.5M` total tokens | `1M-2M` total tokens | `0.37B-0.50B` total tokens | `0.75B-1.0B` total tokens | `Run 2a Dry` 约 `10-20 finished traj × 3 calls × 4K-8K tokens/call`（history context）；正式规模按 `50,688 trajectories` 与冒烟账单均值外推 |

**Run 2b Dry 的额外预算口径**:
- 默认 `b16/n8`：train `16 × 8 = 128` trajectories，配合 `val_max_samples=8`、`val_kwargs.n=1`，总量约 **144 trajectories**
- 可选 `b32/n8`：train `32 × 8 = 256` trajectories，同口径总量约 **272 trajectories**
- 因此 `Run 2b` 的 API 消耗通常约为 `Run 2a` 的 `5-9x`，请在正式训练前留出这部分缓冲

**Jina 本地 shadow run 实测**:
- 脚本：`examples/carr_deepsearch/scripts/estimate_jina_usage.py`
- 样本：`rl_train.parquet` 抽样 `10` 个 query，每个 query 打开搜索结果前 `2` 个链接，共 `20` 次 `browser.open`
- 实测结果：`success_rate=100%`，`latency_mean=10.26s`
- `raw_reader_tokens`：`mean=52,590`，`median=7,466`，`p90=200,813`
- `truncated_reader_tokens`：`mean=1,641`，`p90=2,877`

**Jina 口径说明**:
- `raw_reader_tokens` 更接近 Reader 计费口径。
- `truncated_reader_tokens` 是 agent 真正看到的上下文口径，因为 `browser.open` 会在工具服务器中截断到前 `10,000` chars。
- `raw mean` 明显高于 `median`，说明 Reader token 分布重尾；正式采购不应只按 `median`，也不建议直接按 `raw mean` 做唯一预算。
- 当前文档的主采购口径采用 `20% trimmed mean ≈ 14.3K tokens/open` 作为基线，`raw mean / p90` 只作为长尾风险参考。

**为什么不用 `1.0 open/traj` 做主预算**:
- `1.0 open/traj` 可以保留为乐观下界，但不适合作为正式采购主估计。
- 论文成熟 agent 在 BrowseComp 子集上的平均 cited pages 为 `|CH|=4.3`，而 cited page 通常意味着至少一次成功 `open`。
- 当前项目在 `DeepDive RL val` 的内部预估也已经把 `C-GRPO open_count/mean` 放在 `1.2-2.2`。
- 因此正式采购主估计采用 `1.5-2.0 open/traj`，安全预算采用 `2.5-3.0 open/traj`。

**Run 2b Dry 结束后，必须立刻重估正式预算**:
- 使用 dry run 实际记录到的 `search_count/mean`
- 使用 dry run 实际记录到的 `open_count/mean`
- 使用 dry run 实际记录到的 `task_unfinished/ratio`
- 再结合各平台 dashboard 的真实消耗，更新正式 `Serper / Jina / DeepSeek` 采购量

**当前推荐执行顺序**:
1. `SFT`
2. `C-GRPO dry run（2a 链路验收 + 2b 正式并发验证）`
3. `C-GRPO formal`
4. `SFT + C-GRPO` 两模型 eval
5. 只有当 `C-GRPO` 对 `SFT` 出现足够正向信号时，再补 `GRPO` baseline


---

## 4. 执行步骤

### Eval 0: Base Model 10-Sample Smoke Test

**目的**: 验证端到端链路（模型加载→agent rollout→工具服务器→奖励服务器→指标输出）。

**预估时间**: `20-40 分钟`
- 组成：tool/reward server 启动与健康检查 `~5-10 分钟` + `10` 个 BrowseComp 样本 eval `~15-30 分钟`
- 不确定性主要来自工具调用延迟和 DeepSeek judge 返回速度

```bash
bash examples/carr_deepsearch/scripts/run_eval.sh \
    Qwen/Qwen3-4B browsecomp_smoke10 64k
```

> 如果 GPU 是 RTX Pro 6000/5090（SM120），需追加：
> `'+actor_rollout_ref.rollout.engine_kwargs={sglang: {attention_backend: flashinfer}}'`

**验收**: outcome_reward ≈ 0，tool calls 格式正确，无报错。

**如果失败**: 检查 §3.2 中的注意事项，尤其是 Ray 启动、环境变量、attention_backend。

**记录**:

| 指标 | 值 |
|------|-----|
| outcome_reward/mean@1 | |
| num_turns/mean | |
| tool_call_counts/mean | |
| parse_error_count/mean | |
| 样本 dump 路径 | |
| 是否有报错 | |

**指标来源**:
- `outcome_reward/mean@1`、`num_turns/mean`、`tool_call_counts/mean`、`parse_error_count/mean`：来自本次 eval run 的 console / WandB validation metrics。
- `样本 dump 路径`：来自 `$HOME/eval_results/browsecomp_smoke10_Qwen3-4B_64k/` 下的 JSONL dump。
- `是否有报错`：来自当前终端输出。


---

### Run 1: SFT Training

```bash
NGPUS=$NGPUS bash examples/carr_deepsearch/scripts/run_sft.sh \
    trainer.n_gpus_per_node=$NGPUS
```

**预估时间**: `20-30 分钟`
- 依据：2x5090 记录中的 4xA100 推算是 `~2-3s / step`，3 epochs 共 `~594` steps，对应 `~20-30 分钟`

> `model.fsdp_config.model_dtype=bf16` 已在 `carr_sft.yaml` 中配置，无需 CLI 追加。
> 如 OOM 于 `max_length=65536`，追加 `data.max_length=32768` 并在结果中注明。

**A100 vs 2x5090 差异**: A100 (80GB) 相比 RTX 5090 (32GB) 可尝试更长 `max_length`。2x5090 需降到 8192，4xA100 应可支持 32768-65536。

**监控**:

```bash
# 另开终端：采样 GPU 显存
nvidia-smi --query-gpu=memory.used --format=csv -l 30 | tee sft_gpu_mem.log &

# WandB: carr_deepsearch_sft / carr-sft-qwen3-4b
```

**完成后**:

1. 记录 wall-time（脚本自动输出）
2. 查看 WandB val/loss 曲线，选择最低 val/loss 对应的 epoch checkpoint
3. 设置 checkpoint 路径：

```bash
# 查看可用 checkpoint
ls $HOME/checkpoints/carr_deepsearch_sft/

# 选择最佳 epoch（示例：epoch 2 = step 396 有最低 val/loss）
export SFT_MODEL_PATH=$HOME/checkpoints/carr_deepsearch_sft/global_step_396/huggingface
```

**SFT 指标记录表**:


| 指标                           | 值   |
| ---------------------------- | --- |
| 总步数                          |     |
| Wall-time                    |     |
| GPU-hours                    |     |
| 最终 train/loss                |     |
| 最终 val/loss                  |     |
| 最佳 val/loss (WandB)          |     |
| 选择的 checkpoint step          |     |
| Peak GPU memory (nvidia-smi) |     |
| data.max_length 实际使用         |     |

**指标来源**:
- `总步数`、`最终 train/loss`、`最终 val/loss`、`最佳 val/loss`：来自 WandB run `carr_deepsearch_sft / carr-sft-qwen3-4b` 或 console 日志。
- `Wall-time`：来自 `run_sft.sh` 结束时的 `Completed in ...` 输出。
- `GPU-hours`：手动计算，公式为 `wall-time(hours) × NGPUS`。
- `选择的 checkpoint step`：来自最终导出的 `SFT_MODEL_PATH` 或 checkpoint 目录名。
- `Peak GPU memory`：来自 `sft_gpu_mem.log`。
- `data.max_length 实际使用`：来自最终执行命令。


**SFT OOM Fallback**: 如果 `max_length=65536` OOM，重新运行：

```bash
NGPUS=$NGPUS bash examples/carr_deepsearch/scripts/run_sft.sh \
    trainer.n_gpus_per_node=$NGPUS \
    data.max_length=32768
```

---

### Eval 1a: SFT -> BrowseComp Fixed Subset256

**目的**: 在进入 RL 前锁定一个外部 benchmark baseline，避免花完 RL 预算后才发现 SFT 起点异常。

**预估时间**: `40-80 分钟`

```bash
bash examples/carr_deepsearch/scripts/run_eval.sh \
    "$SFT_MODEL_PATH" browsecomp_subset256 64k
```

**记录**:
- 将该结果作为固定 `SFT baseline`
- 后续所有 `C-GRPO / GRPO` 的 BrowseComp 对比都以这个数为基准

**指标来源**:
- `outcome_reward/mean@1`：来自本次 eval run 的 console / WandB validation metrics
- case study 材料：来自 `$HOME/eval_results/browsecomp_subset256_<sft_ckpt_basename>_64k/` 下的 JSONL dump

---

### Run 2a Dry: C-GRPO 5-Sample Chain Validation

**目的**: 用最小 batch 验证完整 RL 训练链路（rollout→reward→advantage→update→checkpoint→validation metrics），确认所有新增指标正常输出。

**预估时间**: `20-40 分钟`
- 组成：外部服务启动 `~5-10 分钟` + step 0 validation + `1` 个 train step + step 1 validation
- 这是纯链路 gate，不用于判断正式并发是否稳定

```bash
export SFT_MODEL_PATH=<path from Run 1>
bash examples/carr_deepsearch/scripts/run_rl.sh \
    trainer.n_gpus_per_node=$NGPUS \
    trainer.experiment_name=carr-cgrpo-dryrun-chain \
    trainer.default_local_dir=$HOME/checkpoints/carr_deepsearch_rl_cgrpo_dryrun_chain \
    trainer.validation_data_dir=$HOME/eval_results/carr-cgrpo-dryrun-chain-val \
    data.train_batch_size=5 \
    data.train_max_samples=5 \
    data.val_max_samples=10 \
    actor_rollout_ref.rollout.n=2 \
    trainer.total_epochs=1 \
    trainer.save_freq=999 \
    trainer.test_freq=1
```

> A100 无需追加 attention_backend。RTX Pro 6000/5090 需追加 flashinfer。
> 如需手动启动 Ray：
>
> ```bash
> export VERL_USE_EXTERNAL_MODULES=examples.carr_deepsearch.tools.carr_agent_loop,examples.carr_deepsearch.reward.cgrpo_advantage
> export CARR_REWARD_SERVER_URL=http://localhost:8888
> export CARR_REWARD_TIMEOUT=650
> ray start --head --include-dashboard=false --num-gpus=$NGPUS --disable-usage-stats
> ```
>
> 然后在 main_ppo 命令中追加 `+ray_kwargs.ray_init.address=auto`。
> （run_rl.sh 已自动启动 tool/reward server，但不会预启动 Ray。）

**必须通过的验收门槛**:

- Training-step 日志包含: `outcome_reward`, `rubric_reward`, `task_unfinished`, `tool_call_counts`, `parse_error_count`
- Validation metrics 包含: `task_unfinished`, `tool_call_counts`, `search/open/find_count`
- `validation_data_dir` JSONL 的 `response` 字段包含完整多轮轨迹
- 无 CUDA OOM 错误

如有验收门失败：

- 缺少 metric → 检查 `carr_agent_loop.py` 的 `extra_fields.update()` 和 `carr_reward.py` 的 `_pass` dict
- OOM → 降低 `data.max_response_length`（默认 61440→32768）或 `actor_rollout_ref.actor.ppo_max_token_len_per_gpu`
- Tool server 报错 → 检查 Serper/Jina API key 有效性

**产物来源**:
- training-step 指标：来自 WandB run `carr_deepsearch / carr-cgrpo-dryrun-chain` 的最后一个 step 或 console 日志。
- validation metrics：来自同一 run 的 validation panel / console validation summary。
- case study 原始材料：来自 `$HOME/eval_results/carr-cgrpo-dryrun-chain-val/` 下的 JSONL dump。

**Run 2a Dry 的 API 预算（建议先小额采购）**:

> **Trajectory 拆解**：train `5 × n=2 = 10` + val `10 × n=1 × 2 rounds = 20` = **~30 trajectories**。
> validation 默认走 `actor_rollout_ref.rollout.val_kwargs.n=1`，不会继承训练用的 `rollout.n=2`。

| API | Dry run 预计实际消耗 | 当前建议先采购量 | 说明 |
| --- | --- | --- | --- |
| Serper.dev | `60-150` 次搜索 | `>=500` 次 | `~30 traj × 2-5 search/traj`；覆盖 dry run + 一次重跑 + 手工验证 |
| Jina Reader | `0.4M-3.2M` Reader tokens | `5M-8M` Reader tokens | `~30 traj × 1-2 open/traj`；trimmed mean 14.3K→下界，raw mean 52.6K→上界；含长尾和一次重跑 |
| DeepSeek Judge | `0.2M-0.5M` total tokens | `1M-2M` total tokens | `~10-20 finished traj × 3 calls × 4K-8K tokens/call`（history 作为 context）；含 retry 和一次重跑 |

### Run 2b Dry: Formal-Concurrency Validation

**目的**: 在正式训练前，用默认 `b16/n8` 验证 4xA100 下的真实并发、step 时间、tool/reward server 稳定性；若稳定，再额外试一次 `b32/n8`，决定 formal 的 prompt batch。

**默认命令（必须先跑）**:

```bash
bash examples/carr_deepsearch/scripts/run_rl.sh \
    trainer.n_gpus_per_node=$NGPUS \
    trainer.experiment_name=carr-cgrpo-dryrun-b16-n8 \
    trainer.default_local_dir=$HOME/checkpoints/carr_deepsearch_rl_cgrpo_dryrun_b16_n8 \
    trainer.validation_data_dir=$HOME/eval_results/carr-cgrpo-dryrun-b16-n8-val \
    data.train_batch_size=16 \
    data.train_max_samples=16 \
    data.val_batch_size=8 \
    data.val_max_samples=8 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    trainer.total_epochs=1 \
    trainer.save_freq=999 \
    trainer.test_freq=1
```

**可选加压命令（仅当 b16/n8 完全稳定后再跑）**:

```bash
bash examples/carr_deepsearch/scripts/run_rl.sh \
    trainer.n_gpus_per_node=$NGPUS \
    trainer.experiment_name=carr-cgrpo-dryrun-b32-n8 \
    trainer.default_local_dir=$HOME/checkpoints/carr_deepsearch_rl_cgrpo_dryrun_b32_n8 \
    trainer.validation_data_dir=$HOME/eval_results/carr-cgrpo-dryrun-b32-n8-val \
    data.train_batch_size=32 \
    data.train_max_samples=32 \
    data.val_batch_size=8 \
    data.val_max_samples=8 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    trainer.total_epochs=1 \
    trainer.save_freq=999 \
    trainer.test_freq=1
```

**Run 2b 必看的稳定性指标**:

- `timing_s/step`
- `task_unfinished/ratio`
- `parse_error_count/mean`
- `search_count/mean`、`open_count/mean`、`find_count/mean`
- tool / reward server 的 timeout、429、重试与长尾

**Run 2b 的决策规则**:

- 如果 `b16/n8` 不稳定，先修服务或降低并发，不进入正式训练
- 如果 `b16/n8` 稳定、`b32/n8` 不稳定，正式训练固定使用 `train_batch_size=16`
- 如果 `b16/n8` 与 `b32/n8` 都稳定，优先仍以 `b16` 为默认；只有在你明确想减少 optimizer steps 时才把 formal 升到 `b32`

**Do not proceed to formal training if Run 2a or Run 2b fails.**

**Dry run 后怎么更新正式预算**:
- `search_count/mean`：来自 dry run 的 WandB training-step / validation metrics
- `open_count/mean`：来自 dry run 的 WandB training-step / validation metrics
- `task_unfinished/ratio`：来自 dry run 的 WandB training-step / validation metrics
- `Serper / Jina / DeepSeek` 实际消耗：来自三家平台 dashboard / billing 页面
- 正式采购时，以 dry run 真实 `open_count/mean` 替换本节中的 `1.5-2.0 open/traj` 主估计

**通过 dry run 后的决策规则**:
- 如果 `C-GRPO` 的 reward/工具指标链路正常，且 `b16/n8` 稳定，即可进入 `Run 2: C-GRPO Formal`
- 不再把 `GRPO` 作为正式训练前置条件
- `GRPO` 仅作为后续可选 baseline，用于补全三模型主表或验证“是否优于纯 outcome RL”

---

### Run 2: C-GRPO Formal (3 epochs)

```bash
bash examples/carr_deepsearch/scripts/run_rl.sh \
    trainer.n_gpus_per_node=$NGPUS \
    trainer.experiment_name=carr-cgrpo-qwen3-4b \
    trainer.default_local_dir=$HOME/checkpoints/carr_deepsearch_rl_cgrpo \
    trainer.validation_data_dir=$HOME/eval_results/carr-cgrpo-train-val \
    data.train_batch_size=16 \
    data.val_batch_size=16 \
    actor_rollout_ref.rollout.n=8 \
    trainer.save_freq=99 \
    trainer.test_freq=99 \
    actor_rollout_ref.nccl_timeout=1800
```

**正式默认口径**:
- 默认 formal 使用 `train_batch_size=16, rollout.n=8`，即 **`128 trajectories/step`**
- 若 `Run 2b` 里 `b32/n8` 稳定，最多只把 `data.train_batch_size` 升到 `32`；`rollout.n` 仍固定为 `8`
- `train_batch_size` 改成 `32` 主要改变的是 **每步并发与总步数**，不是总 API 用量的量级

> `run_rl.sh` 已自动启动 tool/reward server 并等待就绪。
> `nccl_timeout=1800`（30 min）防止长 rollout 期间 GPU 间无通信导致 NCCL 超时（默认 600s）。
> 默认 `save_freq=test_freq=99`。对默认 b16 formal，这会落在 step `99/198/297/396`；如果后续手动升到 b32，则落在 step `99/198`。

**监控** (另开终端):

```bash
# GPU 显存采样
nvidia-smi --query-gpu=memory.used --format=csv -l 60 | tee cgrpo_gpu_mem.log &

# WandB: carr_deepsearch / carr-cgrpo-qwen3-4b
# 关注: critic/score/mean, outcome_reward/mean, rubric_reward/mean,
#        tool_call_counts/mean, search_count/mean, num_turns/mean,
#        task_unfinished/ratio, perf/max_memory_allocated_gb
```

**训练规模参考**:


| 口径 | train_batch_size | rollout.n | trajectories/step | 每 epoch 步数 | 3 epochs 总步数 | 总 trajectories |
| --- | --- | --- | --- | --- | --- | --- |
| 默认 formal | 16 | 8 | 128 | floor(2123 / 16) = **132** | **396** | **50,688** |
| 可选 promoted formal | 32 | 8 | 256 | floor(2123 / 32) = **66** | **198** | **50,688** |

**时长估算方法**:
- 先记录 `Run 2b` 的真实 `timing_s/step`
- 如果正式沿用默认 `b16/n8`：`formal wall-time ≈ timing_s/step × 396`
- 如果正式升到 `b32/n8`：`formal wall-time ≈ timing_s/step × 198`
- 因为 `3 epochs` 下总 trajectories 基本不变，`b16` 与 `b32` 的主要差异是 **每步并发** 与 **optimizer step 数**


**完成后**:

```bash
export CGRPO_CKPT=$HOME/checkpoints/carr_deepsearch_rl_cgrpo/global_step_<final>/actor/huggingface
```

**C-GRPO 指标记录表**:


| 指标                         | 值   |
| -------------------------- | --- |
| 总步数                        |     |
| Wall-time                  |     |
| GPU-hours                  |     |
| 最终 critic/score/mean       |     |
| 最终 outcome_reward/mean     |     |
| 最终 rubric_reward/mean      |     |
| task_unfinished/ratio (最终) |     |
| tool_call_counts/mean (最终) |     |
| search_count/mean (最终)     |     |
| open_count/mean (最终)       |     |
| find_count/mean (最终)       |     |
| parse_error_count/mean (最终) |     |
| hit_limit/ratio (最终)      |     |
| num_turns/mean (最终)        |     |
| Peak GPU memory            |     |
| validation dump 目录         |     |

**指标来源**:
- `critic/score/mean`、`outcome_reward/mean`、`rubric_reward/mean`、`task_unfinished/ratio`、`tool_call_counts/mean`、`search/open/find_count/mean`、`parse_error_count/mean`、`hit_limit/ratio`、`num_turns/mean`：来自 WandB run `carr_deepsearch / carr-cgrpo-qwen3-4b` 的最终 training-step 指标。
- `Wall-time`：来自 `run_rl.sh` 结束时的 `Completed in ...` 输出。
- `GPU-hours`：手动计算，公式为 `wall-time(hours) × NGPUS`。
- `Peak GPU memory`：来自 `cgrpo_gpu_mem.log`。
- `validation dump 目录`：固定为 `$HOME/eval_results/carr-cgrpo-train-val/`，用于 case study 和训练期验证样本追踪。


---

### Run 2 Mid-Probe: Fixed Subset64 on One Mid-Late Checkpoint

**目的**: 在不频繁打扰训练和不反复烧 benchmark 成本的前提下，只对一个中后期 checkpoint 做一次轻量外部 benchmark probe。

**推荐 checkpoint**:
- 默认 b16 formal：首选 `global_step_297`
- 如果你把 formal 手动升到 `b32`，则改用 `global_step_99`

```bash
MID_CKPT=$HOME/checkpoints/carr_deepsearch_rl_cgrpo/global_step_297/actor/huggingface

bash examples/carr_deepsearch/scripts/run_eval.sh \
    "$MID_CKPT" browsecomp_subset256 64k \
    data.val_files=/root/verl-carr-deepsearch/examples/carr_deepsearch/data/browsecomp_eval_subset_64_seed42.parquet \
    trainer.validation_data_dir=$HOME/eval_results/browsecomp_subset64_step297_64k
```

**预估时间**: `15-30 分钟`

**注意**:
- 这个 `subset64` probe 只做一次，不作为频繁 early-stop 机制
- 其作用是确认 `C-GRPO` 在外部 benchmark 上是否已经明显优于 `SFT baseline`
- 训练过程中继续依赖 `RL val`（默认 `test_freq=99`）做阶段性监控，不把完整 BrowseComp 当作过程监控指标

---

### Eval 1b: C-GRPO Candidate Checkpoints -> Same BrowseComp Subset256

**目的**: 训练结束后，从后半程保留下来的 `2-3` 个候选 checkpoint 中选出最终用于对外汇报的 `C-GRPO` checkpoint。

**推荐候选点**:
- 默认 b16 formal：`global_step_198`、`global_step_297`、`global_step_396`（final）
- 如果 formal 手动升到 `b32`：改为 `global_step_99`、`global_step_198`（final）

```bash
for MODEL in \
  $HOME/checkpoints/carr_deepsearch_rl_cgrpo/global_step_198/actor/huggingface \
  $HOME/checkpoints/carr_deepsearch_rl_cgrpo/global_step_297/actor/huggingface \
  $HOME/checkpoints/carr_deepsearch_rl_cgrpo/global_step_396/actor/huggingface
do
    bash examples/carr_deepsearch/scripts/run_eval.sh \
        "$MODEL" browsecomp_subset256 64k
done
```

**预估时间**: `2-4 小时总计`
- 约 `40-80 分钟 / checkpoint`
- 所有候选点必须跑同一个固定 `subset256`，保证可比性

**完成后**:
- 记录候选 checkpoint 的 `BrowseComp subset256 outcome_reward/mean@1`
- 选择最佳的 `C-GRPO` checkpoint 作为 `CGRPO_CKPT`
- `SFT` 的对照使用 `Eval 1a` 已记录好的同一 `subset256` 结果

结果目录: `$HOME/eval_results/browsecomp_subset256_<model_basename>_64k/`

**首轮主表 (Primary Result, same subset256)**:

| Model  | outcome_reward/mean@1 | Δ vs SFT (pp) | Δ vs GRPO (pp, optional) |
| ------ | --------------------- | ------------- | -------------- |
| SFT    |                       | 0.0           | —              |
| C-GRPO |                       |               |                |
| GRPO (optional)   |                       |               | 0.0            |

**指标来源**:
- `outcome_reward/mean@1`：来自对应 eval run 的 console / WandB validation metrics。
- `Δ vs SFT (pp)`：由 `Eval 1a` 的 SFT baseline 与本节选中的最佳 `C-GRPO` checkpoint 手动相减得到。
- `Δ vs GRPO (pp)`：仅在补跑 `GRPO` 后填写。
- case study 材料：来自 `$HOME/eval_results/browsecomp_subset256_<model_basename>_64k/` 下的 JSONL dump。

**是否补跑 GRPO 的决策建议**:
- 如果 `C-GRPO` 相对 `SFT` 在 BrowseComp subset 上提升明显（例如 `>= 3 pp`），优先保留结果并进入 `Eval 2`
- 如果你仍需要三模型对比主表，或想证明“优于 pure outcome RL”，再进入 `Run 3 (Optional): GRPO Formal`

**可选增强项（仅当 C-GRPO 相对 SFT 在 64k 上提升 >= 3 pp）**:

```bash
for MODEL in "$SFT_MODEL_PATH" "$CGRPO_CKPT"; do
    bash examples/carr_deepsearch/scripts/run_eval.sh \
        "$MODEL" browsecomp_subset256 128k
done
```

**预估时间**: `1.5-3 小时总计`
- 仅 `2` 个模型，但 `128k` context 会明显拉长单样本 rollout 和 judge 时间

| Model | 64k outcome_reward/mean@1 | 128k outcome_reward/mean@1 | Δ(128k-64k) |
|-------|---------------------------|----------------------------|-------------|
| SFT | | | |
| C-GRPO | | | |

**指标来源**:
- `64k`：来自上方主表。
- `128k`：来自对应 `context=128k` eval run 的 console / WandB validation metrics。
- `Δ(128k-64k)`：由同一模型的 `128k - 64k` 手动相减得到，用于说明 test-time scaling。


---

### Eval 2: DeepDive RL Val (111) x 2 Models（使用最终选中的 C-GRPO checkpoint）

```bash
for MODEL in "$SFT_MODEL_PATH" "$CGRPO_CKPT"; do
    bash examples/carr_deepsearch/scripts/run_eval.sh \
        "$MODEL" deepdive_val 64k
done
```

**预估时间**: `40-90 分钟总计`
- 约 `20-40 分钟 / 模型`
- DeepDive `111` 条，小于 BrowseComp subset `256` 条，因此总耗时明显更短

结果目录: `$HOME/eval_results/deepdive_val_<model_basename>_64k/`

**补充表 (Supplementary Result)**:


| Model  | outcome_reward | rubric_reward | task_unfinished | tool_call_counts | search | open | find | num_turns |
| ------ | -------------- | ------------- | --------------- | ---------------- | ------ | ---- | ---- | --------- |
| SFT    |                |               |                 |                  |        |      |      |           |
| C-GRPO |                |               |                 |                  |        |      |      |           |
| GRPO (optional)   |                |               |                 |                  |        |      |      |           |

**指标来源**:
- 所有表项均来自对应 `deepdive_val` eval run 的 console / WandB validation metrics。
- 轨迹样本、失败例、工具调用链：来自 `$HOME/eval_results/deepdive_val_<model_basename>_64k/` 下的 JSONL dump。

---

### Run 3 (Optional): GRPO Formal Baseline

仅在以下情况之一满足时执行：
- 你需要把主表升级为 `SFT / GRPO / C-GRPO` 三模型完整对比
- `C-GRPO` 相对 `SFT` 已出现明确正向信号，值得再投入成本验证“是否优于纯 outcome RL”

```bash
bash examples/carr_deepsearch/scripts/run_rl.sh \
    algorithm.adv_estimator=grpo \
    trainer.n_gpus_per_node=$NGPUS \
    trainer.experiment_name=carr-grpo-qwen3-4b \
    trainer.default_local_dir=$HOME/checkpoints/carr_deepsearch_rl_grpo \
    trainer.validation_data_dir=$HOME/eval_results/carr-grpo-train-val \
    data.train_batch_size=16 \
    data.val_batch_size=16 \
    actor_rollout_ref.rollout.n=8 \
    trainer.save_freq=99 \
    trainer.test_freq=99 \
    actor_rollout_ref.nccl_timeout=1800
```

**预估时间**: 与 `C-GRPO formal` 同量级，按 `Run 2b` 的实测 `timing_s/step` 回填

**完成后**:

```bash
export GRPO_CKPT=$HOME/checkpoints/carr_deepsearch_rl_grpo/global_step_<final>/actor/huggingface
```

**补跑 GRPO 后需要追加的评估**:
- `BrowseComp subset` 上补 `GRPO`
- `DeepDive RL val` 上补 `GRPO`
- 将首轮两模型主表升级为三模型主表

### 4.1 结果预期区间（内部预估，不用于对外汇报）

以下区间用于两件事：
- 租机前估算“跑出来的数字大概是否正常”。
- 正式训练中做跑偏诊断。

**预估依据**:
- 论文 4B 主结果锚点：
  `BrowseComp 64k: SFT 7.7%, GRPO 12.9%, C-GRPO 13.9%`
  `BrowseComp 128k: SFT 14.1%, C-GRPO 17.5%`
- 论文训练现象锚点：
  `C-GRPO` 在训练中后期应出现比 `GRPO` 更高的 `rubric_reward`，且 `tool_call_counts` 不应持续塌缩。
- 论文锚点的本地整理来源：
  `examples/carr_deepsearch/docs/Carr_paper_data.md`
  `CaRR/2601.06021v1.pdf`
- 当前项目与论文存在显著差异：
  backbone 为 `Qwen/Qwen3-4B` 而非论文的 `Qwen3-4B-Thinking-2507`；
  SFT 主配置为 `64k` 而非论文 `128k`；
  benchmark judge 为当前实现中的 `DeepSeek judge`，不等于论文 benchmark judge。

> 下表中的项目内指标采用 `0-1` 小数口径；论文原表是 `%`。

**论文锚点（4B, BrowseComp）**:

| Setting | Paper BrowseComp |
| ------- | ---------------- |
| 4B-SFT 64k | 7.7% |
| 4B-GRPO 64k | 12.9% |
| 4B-C-GRPO 64k | 13.9% |
| 4B-SFT 128k | 14.1% |
| 4B-C-GRPO 128k | 17.5% |

**各阶段内部预估**:

| Stage | 关键指标 | 内部预估区间 / 现象 |
| ----- | -------- | ------------------- |
| Eval 0: base smoke | `outcome_reward/mean@1` | `0.00-0.05`。`10` 个样本里 `0-1` 个答对都算正常。 |
| Eval 0: base smoke | `tool_call_counts/mean` | `0.5-3.0`。base 能调用出工具即可，不要求稳定 deep search。 |
| Eval 0: base smoke | `parse_error_count/mean` | `0.0-0.8`。若明显高于 `1.0`，优先查 parser / prompt / stop token。 |
| Eval 0: base smoke | `num_turns/mean` | `1.5-4.0`。过低可能没进工具环；过高可能卡在无效搜索。 |
| Run 1: SFT | `最终 train/loss` | `0.7-1.1`。 |
| Run 1: SFT | `最终 val/loss` / `最佳 val/loss` | `0.8-1.3`。若长期 `>1.5`，优先查数据格式、packing、loss mask。 |
| Run 2 dry | `outcome_reward/mean` | `0.00-0.20`，高方差，仅用于确认 reward 链路非全零。 |
| Run 2 dry | `rubric_reward/mean` | `0.00-0.15`，高方差，仅用于确认 rubric 指标被正确记录。 |
| Run 2 dry | `task_unfinished/ratio` | `0.20-0.80`，只要非 `NaN` 且能随样本变化即可。 |
| Run 2 dry | `tool_call_counts/mean` | `1.0-4.0`，只要非零且能看到 `search/open/find` 拆分即可。 |

**正式 RL 训练期的相对预期**:
- `GRPO` 和 `C-GRPO` 在前 `5-10` 个 step 的 `tool_call_counts/mean` 都可能先下降，这与论文 Figure 4 一致。
- 到训练中后段，`C-GRPO` 应出现比 `GRPO` 更高的 `rubric_reward/mean`，且 `tool_call_counts/mean` 不应持续低于 `GRPO`。
- 一个可接受的末段相对关系是：
  `C-GRPO rubric_reward/mean` 比 `GRPO` 高 `0.05-0.12`，
  `C-GRPO tool_call_counts/mean` 比 `GRPO` 高 `0.5-1.5`，
  `C-GRPO task_unfinished/ratio` 比 `GRPO` 低 `0.03-0.10`。

**Eval 1 预估主表（BrowseComp subset, N=256, 64k）**:

| Model | 预估 outcome_reward/mean@1 | 说明 |
| ----- | -------------------------- | ---- |
| SFT | `0.04-0.09` | 保守低于或接近论文 `4B-SFT 64k = 7.7%`。 |
| GRPO | `0.06-0.12` | 相对 SFT 提升 `+1-4 pp`。 |
| C-GRPO | `0.08-0.14` | 相对 SFT 提升 `+3-6 pp`，相对 GRPO 再提升 `+1-3 pp`。 |

**Eval 2 预估补充表（DeepDive RL val, N=111, 64k）**:

| Model | outcome_reward | rubric_reward | task_unfinished | tool_call_counts | num_turns |
| ----- | -------------- | ------------- | --------------- | ---------------- | --------- |
| SFT | `0.15-0.28` | `0.18-0.30` | `0.25-0.45` | `3.0-5.0` | `3.0-5.0` |
| GRPO | `0.18-0.32` | `0.15-0.28` | `0.20-0.40` | `2.5-4.5` | `2.5-4.5` |
| C-GRPO | `0.22-0.36` | `0.24-0.38` | `0.15-0.35` | `3.5-5.5` | `3.5-5.5` |

**Eval 2 结构性现象预期**:
- `C-GRPO rubric_reward` 应明显高于 `GRPO`，这是最重要的过程质量信号。
- `GRPO` 可能出现 `outcome_reward` 略升但 `tool_call_counts` 和 `rubric_reward` 偏低的 shortcut 现象。
- `C-GRPO` 的 `task_unfinished` 应不高于 `GRPO`，否则说明 agent loop 或 rollout 限制存在问题。
- 若需要拆分到 `search/open/find` 三列，保守预估可用：
  `SFT ≈ 1.0-2.0 / 1.0-2.0 / 0.5-1.5`
  `GRPO ≈ 0.8-1.8 / 0.8-1.8 / 0.4-1.2`
  `C-GRPO ≈ 1.2-2.2 / 1.2-2.2 / 0.6-1.6`。

**可选 128k eval 预估（仅当 64k 主结果够好时再跑）**:

| Model | 64k outcome_reward/mean@1 | 128k outcome_reward/mean@1 | 预估增幅 |
| ----- | ------------------------- | -------------------------- | -------- |
| SFT | `0.04-0.09` | `0.05-0.11` | `+1-3 pp` |
| C-GRPO | `0.08-0.14` | `0.10-0.17` | `+1-4 pp` |

> 由于当前项目的 SFT 主配置是 `64k` 而不是论文的 `128k SFT`，这里对 `128k` 增幅的预估明显比论文保守。

**红线/跑偏诊断**:
- 若完整 `SFT` 后 `BrowseComp subset` 仍低于 `0.03`，优先检查评测 prompt、judge、工具链和 SFT 数据格式。
- 若 `GRPO` 和 `C-GRPO` 都没有高于 `SFT`，优先检查 reward server、`task_unfinished`、`parse_error_count`、`hit_limit` 是否异常。
- 若 `C-GRPO` 的 `rubric_reward` 不高于 `GRPO`，同时 `tool_call_counts` 也更低，基本可判定为复现出了论文所说的 shortcut 问题而非 deep search 改善。


---

## 5. Checkpoint 管理


| Stage | save_freq | 总步数  | Expected checkpoints | Disk per ckpt      | 总磁盘    |
| ----- | --------- | ---- | -------------------- | ------------------ | ------ |
| SFT   | 198 steps | ~594 | 3 (one per epoch)    | ~8 GB (HF model)   | ~24 GB |
| RL（默认 b16 formal） | 99 steps | 396 | 4 (step 99, 198, 297, 396) | ~7.5 GB (HF model) | ~30 GB |


**策略**:

- SFT: 手动从 WandB val/loss 曲线选最佳 epoch checkpoint
- RL: 使用 epoch 3 最终 checkpoint
- **Run 2 完成后、Run 3 开始前**: 清理 C-GRPO 中间 checkpoint 释放磁盘
  ```bash
  # 保留最终 checkpoint，删除中间的
  ls $HOME/checkpoints/carr_deepsearch_rl_cgrpo/
  # 仅保留 global_step_<final>/，删除其余
  ```

### SFT Checkpoint 保存说明

`fsdp_sft_trainer.py` 的 `save_freq` 逻辑：

- `save_freq=198` → 在 step 198, 396, 594 保存（≈每 epoch 末）
- 保存内容: `["model", "optimizer", "extra", "hf_model"]`（见 `carr_sft.yaml`）
- HF model 自动导出到 `global_step_XXX/huggingface/`

### RL Checkpoint 保存说明

- `save_freq=99` → 默认 b16 formal 会在 step `99/198/297/396` 保存；若 formal 手动升到 b32，则保存 `99/198`
- Checkpoint 包含 actor model，自动转换 HF 格式到 `actor/huggingface/`
- `latest_checkpointed_iteration.txt` 记录最新 step 编号

---

## 6. 必须配置参数清单

### 6.1 bf16 精度（所有阶段必须）

已在 YAML 配置中设置，无需 CLI 追加：


| 阶段         | YAML 配置                                                                   | 说明          |
| ---------- | ------------------------------------------------------------------------- | ----------- |
| SFT        | `carr_sft.yaml`: `model.fsdp_config.model_dtype: bf16`                    | 否则 fp32 OOM |
| RL (actor) | `carr_grpo.yaml`: `actor_rollout_ref.actor.fsdp_config.model_dtype: bf16` | 同上          |
| RL (ref)   | `carr_grpo.yaml`: `actor_rollout_ref.ref.fsdp_config.model_dtype: bf16`   | 同上          |


> 基础配置默认 fp32（`sft_trainer.yaml`, `_generated_ppo_trainer.yaml`），CaRR YAML 通过 `_self_` 优先级覆盖为 bf16。

### 6.2 绝对路径（RL/Eval 阶段）

Ray TaskRunner 将 cwd 设为 `/root/`，YAML 中的相对路径会失效。以下 4 个路径在 `run_rl.sh` 和 `run_eval.sh` 中已用 `$PROJECT_DIR` 覆盖：

- `data.train_files`
- `data.val_files`
- `reward.custom_reward_function.path`
- `actor_rollout_ref.rollout.multi_turn.tool_config_path`

### 6.3 Ray 预启动（如果不由脚本管理）

```bash
# 环境变量必须在 ray start 前 export
export VERL_USE_EXTERNAL_MODULES=examples.carr_deepsearch.tools.carr_agent_loop,examples.carr_deepsearch.reward.cgrpo_advantage
export CARR_REWARD_SERVER_URL=http://localhost:8888
export CARR_REWARD_TIMEOUT=650

ray start --head --include-dashboard=false --num-gpus=$NGPUS --disable-usage-stats
```

然后 main_ppo 命令追加 `+ray_kwargs.ray_init.address=auto`。

---

## 7. 中断恢复

### 7.1 SFT 中断恢复

`fsdp_sft_trainer.py` 自动检测 `default_local_dir` 中的 checkpoint 并恢复：

```bash
# 直接重新运行相同命令即可
NGPUS=$NGPUS bash examples/carr_deepsearch/scripts/run_sft.sh \
    trainer.n_gpus_per_node=$NGPUS
```

它会自动从 `latest_checkpointed_iteration.txt` 读取最新 step 并继续。

### 7.2 RL 中断恢复

verl RL trainer 支持**真正的训练恢复**，不是只恢复模型权重。

checkpoint 中包含：
- actor / critic 权重与优化器状态
- `global_steps`
- `train_dataloader.state_dict()`（保存为 `global_step_xxx/data.pt`）

这意味着只要 checkpoint 完整，重新启动后会从**上次已训练过的数据之后继续跑**，而不是从 RL 数据集开头重扫。

**默认恢复模式**:
- `ppo_trainer.yaml` 默认 `trainer.resume_mode=auto`
- 只要 `trainer.default_local_dir` 下存在 `global_step_*`，重新运行就会自动恢复最新 checkpoint

检查最新 checkpoint：

```bash
# C-GRPO
cat $HOME/checkpoints/carr_deepsearch_rl_cgrpo/latest_checkpointed_iteration.txt
# GRPO (optional)
cat $HOME/checkpoints/carr_deepsearch_rl_grpo/latest_checkpointed_iteration.txt
```

重新运行相同命令（**不变任何关键参数**），trainer 会自动检测并恢复：

```bash
bash examples/carr_deepsearch/scripts/run_rl.sh \
    trainer.n_gpus_per_node=$NGPUS \
    trainer.experiment_name=carr-cgrpo-qwen3-4b \
    trainer.default_local_dir=$HOME/checkpoints/carr_deepsearch_rl_cgrpo
```

**推荐做法**:
- `GRPO` 和 `C-GRPO` 使用**不同的** `trainer.default_local_dir`
- 同一个实验中断恢复时，保持以下参数不变：
  - `data.train_files`
  - `data.train_batch_size`
  - `data.train_max_samples`
  - `data.shuffle / data.seed`
  - `actor_rollout_ref.rollout.n`
  - `trainer.default_local_dir`
- 恢复时优先显式传：
  - `trainer.default_local_dir=<该实验自己的 checkpoint 目录>`

**强烈建议的目录隔离**:

```bash
# GRPO
trainer.default_local_dir=$HOME/checkpoints/carr_deepsearch_rl_grpo

# C-GRPO
trainer.default_local_dir=$HOME/checkpoints/carr_deepsearch_rl_cgrpo
```

当前 `carr_grpo.yaml` 默认把 RL checkpoint 写到同一个目录。如果 `GRPO` 和 `C-GRPO` 共用目录，`resume_mode=auto` 可能会误捡到另一个实验的最新 checkpoint。

**指定某个 checkpoint 恢复**:

```bash
bash examples/carr_deepsearch/scripts/run_rl.sh \
    trainer.n_gpus_per_node=$NGPUS \
    trainer.default_local_dir=$HOME/checkpoints/carr_deepsearch_rl_cgrpo \
    trainer.resume_mode=resume_path \
    trainer.resume_from_path=$HOME/checkpoints/carr_deepsearch_rl_cgrpo/global_step_198
```

适用场景：
- `latest` 不是你想恢复的点
- 同目录里有多个实验残留
- 你想从某个稳定 checkpoint 重开一轮

**不是恢复训练的情况**:
- 如果你只是把 `global_step_xxx/actor/huggingface` 作为新的 `actor_rollout_ref.model.path` 去启动 RL，
  这只是 **warm start**，不是 resume。
- 这种情况下模型参数会继承，但 RL 数据会从头开始重新跑。

**会退化为“模型恢复，但数据从头开始”的情况**:
- `global_step_xxx/data.pt` 丢失
- 你手动只保留了 `actor/huggingface`，删掉了完整 checkpoint

出现这类情况时，trainer 仍可加载模型，但 dataloader 无法恢复到原位置，会从数据集开头重新开始。

**恢复后看到先做 validation 是正常现象**:
- `val_before_train=True` 时，恢复后 trainer 会先跑一次 validation，再进入下一训练 step
- 这不代表它从头训练，只是恢复后的正常启动流程

### 7.3 磁盘满处理

如果磁盘满导致 checkpoint 保存失败：

1. 清理不再需要的中间 checkpoint
2. 清理 WandB 缓存: `rm -rf /root/wandb/`
3. 清理 HuggingFace 缓存: `rm -rf ~/.cache/huggingface/hub/`（注意：会删除下载的模型）
4. 重新运行（自动从最近 checkpoint 恢复）

**清理时必须保留**:
- 目标恢复点对应的完整 `global_step_xxx/`
- 其中的 `actor/`
- 其中的 `data.pt`
- `latest_checkpointed_iteration.txt`（如果你打算继续用 `resume_mode=auto`）

**不要只保留** `actor/huggingface/`：
- 这只够做 eval 或 warm start
- 不够做真正的 RL 续训

---

## 8. WandB 监控指标

### 8.1 SFT 阶段


| WandB Key    | 说明   | 期望趋势                     |
| ------------ | ---- | ------------------------ |
| `train/loss` | 训练损失 | 持续下降                     |
| `val/loss`   | 验证损失 | 持续下降（选最低点对应的 checkpoint） |
| `train/lr`   | 学习率  | 按 schedule 变化            |


> Peak GPU memory 和 tokens/s 不是 WandB 结构化指标，需从 `nvidia-smi` 和 wall-time 手动计算。

### 8.2 RL 阶段


| WandB Key                      | 说明            | 期望趋势       |
| ------------------------------ | ------------- | ---------- |
| `critic/score/mean`            | 训练 reward 均值  | 上升         |
| `outcome_reward/mean`          | 准确率 reward    | 上升         |
| `rubric_reward/mean`           | Rubric reward | 上升         |
| `tool_call_counts/mean`        | 每样本工具调用数      | 稳定或上升      |
| `search_count/mean`            | search 调用数    |            |
| `open_count/mean`              | open 调用数      | 上升（学会链式搜索） |
| `find_count/mean`              | find 调用数      | 上升         |
| `task_unfinished/ratio`        | 未完成任务比例       | 下降         |
| `parse_error_count/mean`       | 解析错误数         | 下降         |
| `num_turns/mean`               | 平均对话轮数        |            |
| `perf/max_memory_allocated_gb` | GPU 峰值显存      | 稳定         |
| `timing_s/step`                | 每步耗时          | 稳定         |
| `perf/throughput`              | tokens/s      | 稳定         |


---

## 9. 训练时长与成本估算

### 9.1 时长


| 阶段 | 4xA100 预估 |
| --- | --- |
| 环境搭建 + 冒烟测试 | 2-3 小时 |
| SFT (3 epochs) | 20-30 分钟 |
| Eval 1a: SFT -> subset256 | 40-80 分钟 |
| C-GRPO dry run（2a + 2b） | 30-90 分钟 |
| C-GRPO (3 epochs) | 先用 `Run 2b` 的 `timing_s/step` 回填；默认 b16 为 `396` steps，若升到 b32 为 `198` steps |
| 中后期 checkpoint 单次 subset64 probe | 15-30 分钟 |
| 训练后 2~3 个候选 checkpoint 跑同一 subset256 | 2-4 小时 |
| DeepDive eval x2（SFT + 选中的 C-GRPO） | 40-90 分钟 |
| **首轮最小闭环总计** | **Run 2b 后按实测回填** |
| 可选 128k 放大确认 | 1.5-3 小时 |
| GRPO (3 epochs, optional) | 与对应 formal 配置同量级；默认 b16 为 `396` steps |
| BrowseComp / DeepDive 补 GRPO eval（optional） | 1.5-3 小时 |
| **完整三模型方案总计** | **Run 2b 后按实测回填** |


> 瓶颈不在 GPU 计算，而在工具服务器和奖励服务器的 I/O 延迟。`train_batch_size=16` 与 `32` 的主要区别是每步并发和总步数，不是总 trajectories 的数量级。

### 9.2 GPU 显存预算 (Qwen3-4B, 4xA100-80G)


| 组件                         | 大小             | 说明            |
| -------------------------- | -------------- | ------------- |
| Actor 参数 (bf16)            | 8 GB / FSDP均分  | 4B × 2 bytes  |
| Actor 梯度 (bf16)            | 8 GB / FSDP均分  |               |
| Actor 优化器 (AdamW)          | 32 GB / FSDP均分 | 2 个 fp32 状态   |
| Ref 模型 (bf16)              | 8 GB / FSDP均分  |               |
| **FSDP 静态/卡**              | **~14 GB**     | 56 GB / 4     |
| SGLang KV cache (util=0.5) | ~40 GB         | 80 × 0.5      |
| 激活值                        | ~15 GB         | 梯度检查点已开启      |
| **合计/卡**                   | **~69 GB**     | **余量 ~11 GB** |


---

### 9.3 实际成本记录表

**GPU / 机器成本**:

| Stage | Wall-time (h) | GPUs | GPU-hours | 机器租价 ($/h) | 机器成本 ($) |
|-------|---------------|------|-----------|----------------|---------------|
| SFT | | 4 | | | |
| Eval 1a: SFT -> subset256 | | 4 | | | |
| C-GRPO dry run | | 4 | | | |
| C-GRPO | | 4 | | | |
| subset64 probe（中后期 1 次） | | 4 | | | |
| 候选 checkpoint subset256 eval | | 4 | | | |
| DeepDive eval x2（SFT + 选中的 C-GRPO） | | 4 | | | |
| 可选 128k 放大确认 | | 4 | | | |
| GRPO（optional） | | 4 | | | |
| 补 GRPO eval（optional） | | 4 | | | |

**来源**:
- `Wall-time (h)`：来自各脚本 `Completed in ...` 输出，换算为小时。
- `GPU-hours`：手动计算，公式为 `wall-time(hours) × 4`。
- `机器租价 ($/h)`：来自租机平台订单页面。
- `机器成本 ($)`：手动计算，公式为 `wall-time(hours) × machine_price_per_hour`。

**API / Judge 成本**:

| Provider | 记录项 | 实际调用量 | 单价 | 成本 ($) | 来源 |
|----------|--------|------------|------|----------|------|
| Serper.dev | search API calls | | | | 平台 dashboard / billing 页面 |
| Jina Reader | reader requests / tokens | | | | 平台 dashboard / billing 页面 |
| DeepSeek | judge requests / tokens | | | | 平台 dashboard / billing 页面 |

> 这张表是简历项目里“成本感知”和“工程权衡”最直接的数据来源，正式跑完必须补齐。

---

## 10. 结果产出物清单

完成所有步骤后，收集以下产出物：

1. **首轮主表**: `SFT / C-GRPO` 在 BrowseComp subset (256) 上的 `outcome_reward/mean@1`
2. **主表升级项（optional）**: 若补跑 `GRPO`，将主表扩展为 `SFT / GRPO / C-GRPO`
3. **主表增强列**: `Δ vs SFT (pp)`；`Δ vs GRPO (pp)` 仅在补跑 `GRPO` 后填写
4. **补充表**: DeepDive RL val 上 outcome/rubric_reward, task_unfinished, tool counts, num_turns
5. **可选 128k 表**: 仅在 `C-GRPO vs SFT >= 3 pp` 时补 `64k vs 128k`
6. **训练曲线**: WandB 截图
  - outcome_reward, rubric_reward 随 training step 变化
  - tool_call_counts, num_turns 随 training step 变化
- 其中 `C-GRPO` 是首轮必须保留的训练曲线；`GRPO` 曲线为 optional
7. **Case studies**: 从 `validation_data_dir` JSONL dump 中选取 2 正例 + 1 失败例
  - 还原 search→open→find 完整轨迹
8. **运行时统计**: wall-time, GPU-hours, peak GPU memory (from nvidia-smi log)
9. **成本统计**: Serper / Jina / DeepSeek 实际调用量与估算成本

### 总览指标填写表


| 指标                          | SFT | GRPO (optional) | C-GRPO |
| --------------------------- | --- | ---- | ------ |
| **BrowseComp subset (256)** |     |      |        |
| outcome_reward/mean@1       |     |      |        |
| Δ vs SFT (pp)              | 0.0 |      |        |
| Δ vs GRPO (pp, optional)   | —   | 0.0  |        |
| **DeepDive RL val**         |     |      |        |
| outcome_reward/mean@1       |     |      |        |
| rubric_reward/mean@1        |     |      |        |
| task_unfinished/ratio       |     |      |        |
| tool_call_counts/mean       |     |      |        |
| search_count/mean           |     |      |        |
| open_count/mean             |     |      |        |
| find_count/mean             |     |      |        |
| num_turns/mean              |     |      |        |
| **Runtime**                 |     |      |        |
| Training wall-time          |     |      |        |
| GPU-hours                  |     |      |        |
| Peak GPU memory             |     |      |        |

### 指标来源速查

| 指标类型 | 主要来源 | 备注 |
|----------|----------|------|
| SFT `train/loss` / `val/loss` | WandB + console | `carr_deepsearch_sft / carr-sft-qwen3-4b` |
| RL training-step 指标 | WandB + console | `carr_deepsearch / carr-grpo-qwen3-4b` 或 `carr-cgrpo-qwen3-4b` |
| Eval 主表 / 补充表指标 | 对应 eval run 的 console / WandB validation metrics | `run_eval.sh` 产出 |
| Case study 轨迹 | `validation_data_dir` JSONL dump | 看 `response` 字段和样本索引 |
| Wall-time | 脚本末尾 `Completed in ...` | `run_sft.sh` / `run_rl.sh` / `run_eval.sh` |
| Peak GPU memory | `nvidia-smi` 采样日志 | `sft_gpu_mem.log`, `grpo_gpu_mem.log` 等 |
| GPU-hours / 机器成本 | 手动计算 | 基于 wall-time 和租价 |
| API 调用量 / 成本 | Serper / Jina / DeepSeek dashboard | 用于成本统计和简历工程权衡说明 |
