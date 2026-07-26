# CaRR DeepSearch 新 Mac 迁移说明

本文档记录 2026-07-26 从旧 Mac 向新 Mac 迁移项目时的 Git 状态、推荐克隆方式，以及没有进入 Git 仓库的大型本地资产。后续如果在新机器上发现某个目录不存在，应先查看本文档，不要直接判断为仓库内容丢失。

## 1. 远程仓库与工作分支

- 远程仓库：`https://github.com/Rogerffff/agentic_RL_project.git`
- 当前项目工作分支：`feature/carr-deepsearch`
- GitHub 默认分支：`main`
- CaRR 子模块仓库：`https://github.com/Rogerffff/CaRR.git`
- CaRR 子模块分支：`feature/add-serper-backend`

这个项目的完整实现和最新文档目前位于 `feature/carr-deepsearch`，因此新 Mac 不应只停留在默认的 `main` 分支。

## 2. 新 Mac 推荐克隆命令

```bash
git clone --recurse-submodules \
  --branch feature/carr-deepsearch \
  https://github.com/Rogerffff/agentic_RL_project.git \
  verl-carr-deepsearch

cd verl-carr-deepsearch
git submodule update --init --recursive
git status --short --branch
```

如果已经先克隆了默认分支，可以执行：

```bash
git fetch origin
git switch feature/carr-deepsearch
git submodule update --init --recursive
```

## 3. Git 仓库中包含的内容

远程分支包含以下可复用资产：

- CaRR DeepSearch 的 agent loop、browser tool、session manager、reward bridge 和 C-GRPO 实现
- 同步与异步强化学习相关配置、启动脚本和评测脚本
- `verl/experimental/fully_async_policy/` 下的异步训练实现
- 项目的开发日志、完整训练历史、评测结论、行为分析、简历材料和面试问答文档
- 已经纳入 Git 的小型 reward trace、tool statistics、状态文件和 checkpoint 选择证据
- 指向可访问远程提交的 `CaRR`、`agent-R1` 等 Git 子模块

## 4. 不进入 Git 仓库的旧 Mac 本地资产

以下内容由 `.gitignore` 排除，或者体积、敏感性不适合直接放进公开 GitHub 仓库。它们不会随普通 `git clone` 自动下载。

| 路径 | 旧 Mac 约占用空间 | 不提交原因 | 新 Mac 恢复方式 |
|------|------------------|------------|----------------|
| `examples/carr_deepsearch/api.md`、各级 `.env` | 很小 | 包含 API 密钥，禁止进入公开仓库 | 在新 Mac 手工创建 `.env`，重新填入有效密钥 |
| `examples/carr_deepsearch/eval_results/` | 148MB | 样本级完整模型输出，体积较大 | 如需继续逐样本分析，从旧 Mac 单独复制 |
| `examples/carr_deepsearch/CaRR_log/` 中被忽略的日志 | 165MB | 大量运行日志，可由文档和已跟踪摘要替代 | 仅在需要原始日志审计时单独复制 |
| `examples/carr_deepsearch/CaRR_formal_training_log/` 中被忽略的日志 | 23MB | 同步训练原始日志，非运行代码依赖 | 按需从旧 Mac 单独复制 |
| `examples/carr_deepsearch/data/` | 76MB | Parquet 数据被全局忽略，可由预处理脚本重建 | 从原始 CaRR 数据重新生成，或从旧 Mac 复制 |
| `examples/carr_deepsearch/CaRR_repo/` | 204MB | 远端机器的重复仓库快照，不是当前事实源 | 通常不需要恢复；历史核对时再从旧 Mac 复制 |
| `CaRR/data/deepdive-sft-glm46-trajectory-1k.jsonl` | 151MB | 子模块内未跟踪的大型数据文件，超过普通 GitHub 单文件限制 | 从原始数据源重新下载，或从旧 Mac 复制 |
| `__pycache__/`、`.DS_Store`、Ray、Weights & Biases 和临时缓存 | 不固定 | 可再生成缓存 | 不需要迁移 |
| 模型 checkpoint、`*.pt`、`*.ckpt` | 可能很大 | GitHub 不适合存储模型权重 | 从对象存储、模型仓库或旧训练机器恢复 |

## 5. 建议额外保留的本地归档

如果旧 Mac 之后会被清理，建议至少单独备份以下两个目录。它们不是继续开发代码的前置条件，但对将来复核简历数字、样本案例和 judge 行为很有价值：

```text
examples/carr_deepsearch/eval_results/
examples/carr_deepsearch/CaRR_log/20260415_gpu_eval/
```

推荐使用外置硬盘、局域网传输或私有云盘保存，不要直接上传到当前公开 GitHub 仓库。传输完成后，应核对目录大小和文件数量。

## 6. 新 Mac 首次检查

```bash
git status --short --branch
git submodule status --recursive
python --version
```

然后依次确认：

1. 当前分支是 `feature/carr-deepsearch`。
2. `CaRR` 子模块已经初始化，并位于仓库记录的提交。
3. 根据实际用途重新创建 `.env`，不要从 Git 历史查找密钥。
4. 如果只需要回顾项目、准备简历或继续修改代码，克隆后的 Git 内容已经足够。
5. 如果需要重新训练或执行完整评测，再恢复数据集、模型 checkpoint 和 API 密钥。

## 7. 项目恢复后的阅读顺序

1. `examples/carr_deepsearch/AGENT.md`
2. `examples/carr_deepsearch/docs/training_full_history_20260404.md`
3. `examples/carr_deepsearch/docs/eval_analysis_20260415.md`
4. `examples/carr_deepsearch/docs/eval_result_deep_analysis.md`
5. `examples/carr_deepsearch/docs/resume_source_of_truth_20260412.md`
6. `examples/carr_deepsearch/docs/final_resume_ready.md`
