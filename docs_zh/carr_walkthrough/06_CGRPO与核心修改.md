# Part 6: C-GRPO Advantage Estimator 与 verl 核心修改

本部分讲解 CaRR 论文中的 C-GRPO 算法公式、verl 的 advantage estimator 注册机制、`cgrpo_advantage.py` 的完整实现，以及我们对 verl 框架的两处核心修改。

---

## 6.1 为什么需要 C-GRPO？

### 6.1.1 标准 GRPO 的局限

标准 GRPO（Group Relative Policy Optimization）的 advantage 估计：

```
A_i = (R_i - mean_g(R)) / std_g(R)
```

其中 `g` 是同一 prompt 的 rollout 组（例如 n=16 条轨迹）。

对于 Deep Search Agent，`R_i` 通常是 outcome reward（答案正确 1，错误 0）。问题是：

1. **稀疏信号**：同一 prompt 的 16 条轨迹可能全部答错（R=0）或全部答对（R=1），此时 `std_g = 0`，advantage 为 0，无法学习
2. **缺乏过程反馈**：两条都答对的轨迹得到相同的 advantage，但一条可能有充分的推理证据链，另一条可能是"蒙对"的——标准 GRPO 无法区分

### 6.1.2 C-GRPO 的解决方案

C-GRPO（Citation-aware GRPO）引入 rubric reward 来区分推理质量：

```
R_i = (1 - α) · R_outcome + α · R_outcome · R̂_rubric
```

其中：
- `R_outcome` ∈ {0, 1}：答案正确性
- `R̂_rubric`：组内归一化的 rubric reward
- `α = 0.3`：混合系数

**关键设计**：`R_outcome` 作为乘法因子出现在 rubric 项中。这意味着：
- **答错时**：`R_outcome = 0`，整个 rubric 项为 0。即使推理过程很好，但答案错了就不奖励
- **答对时**：`R_outcome = 1`，rubric 项 = `α · R̂_rubric`，区分推理质量

这样：同样答对的两条轨迹，推理更充分（更多 rubric 满足）的那条获得更高 advantage。

### 6.1.3 Group-wise Rubric 归一化

`R̂_rubric` 的计算（组内归一化）：

```
R̂_rubric_i = R_rubric_i / max(R_rubric_g)
```

- 分母是同组中 rubric reward 的最大值
- 如果 `max = 0`（组内无人获得任何 rubric 分），`R̂_rubric = 0`
- 归一化到 [0, 1] 范围，避免 rubric reward 的绝对值影响 advantage 的尺度

**为什么不在 reward server 端做归一化？**

因为归一化需要看到同组所有 rollout 的 rubric reward，而 reward server 是逐条评估的。归一化只能在 advantage 阶段（看到整个 batch 后）进行。

---

## 6.2 verl Advantage Estimator 注册机制

### 6.2.1 Registry 模式

`core_algos.py` 使用 decorator-based registry 管理 advantage estimator：

```python
# core_algos.py:112
ADV_ESTIMATOR_REGISTRY: dict[str, Any] = {}

# core_algos.py:115-133
def register_adv_est(name_or_enum: str | AdvantageEstimator):
    def decorator(fn):
        name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
        if name in ADV_ESTIMATOR_REGISTRY and ADV_ESTIMATOR_REGISTRY[name] != fn:
            raise ValueError(f"Adv estimator {name} has already been registered: ...")
        ADV_ESTIMATOR_REGISTRY[name] = fn
        return fn
    return decorator

# core_algos.py:136-149
def get_adv_estimator_fn(name_or_enum):
    name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
    if name not in ADV_ESTIMATOR_REGISTRY:
        raise ValueError(f"Unknown advantage estimator: {name}")
    return ADV_ESTIMATOR_REGISTRY[name]
```

内置的 estimator 直接在 `core_algos.py` 中注册（如 `@register_adv_est("grpo")`）。我们的 `cgrpo` 则是**外部注册**。

### 6.2.2 外部模块加载机制

verl 支持通过环境变量加载外部模块：

```bash
export VERL_USE_EXTERNAL_MODULES=examples.carr_deepsearch.reward.cgrpo_advantage
```

在 `verl/__init__.py` 中，启动时会 `importlib.import_module()` 加载这些模块。加载时，模块顶层的 `@register_adv_est("cgrpo")` 装饰器执行，将函数注册到 `ADV_ESTIMATOR_REGISTRY`。

### 6.2.3 调用链路

```
ray_trainer.py:compute_advantage()
    │
    │  adv_estimator = "cgrpo"  (来自 YAML algorithm.adv_estimator)
    │
    │  # 不是 GAE 也不是 GRPO，进入 else 分支
    ├── adv_estimator_fn = core_algos.get_adv_estimator_fn("cgrpo")
    │   → 返回 compute_cgrpo_advantage 函数
    │
    ├── adv_kwargs = {
    │       "token_level_rewards": data.batch["token_level_rewards"],
    │       "response_mask": data.batch["response_mask"],
    │       "config": config,
    │       "index": data.non_tensor_batch["uid"],         # if "uid" in non_tensor_batch
    │   }
    │
    ├── # 签名检查 → compute_cgrpo_advantage 有 non_tensor_batch 参数
    │   adv_kwargs["non_tensor_batch"] = data.non_tensor_batch
    │
    └── advantages, returns = adv_estimator_fn(**adv_kwargs)
```

---

## 6.3 verl 的两处核心修改（已完成）

这两处修改已经在当前代码中完成，是支持 C-GRPO 的前提。

### 6.3.1 修改 1: AlgoConfig 新增 cgrpo_alpha

**文件**: `verl/trainer/config/algorithm.py:615`

```python
class AlgoConfig(BaseConfig):
    # ... 其他字段 ...
    rollout_correction: Optional[RolloutCorrectionConfig] = None
    cgrpo_alpha: float = 0.3  # C-GRPO: rubric reward mixing ratio  ← 新增
```

这使得 `cgrpo_alpha` 可以通过 Hydra YAML 配置：

```yaml
algorithm:
  adv_estimator: cgrpo
  cgrpo_alpha: 0.3
```

在 `compute_advantage()` 调用时，`config` 参数就是 `AlgoConfig` 实例，`cgrpo_advantage.py` 通过 `config.get("cgrpo_alpha", 0.3)` 读取。

### 6.3.2 修改 2: ray_trainer.py 条件传递 non_tensor_batch

**文件**: `verl/trainer/ppo/ray_trainer.py:215-220`

```python
# Conditionally pass non_tensor_batch only to estimators that accept it
_sig = inspect.signature(adv_estimator_fn)
if "non_tensor_batch" in _sig.parameters or any(
    p.kind == inspect.Parameter.VAR_KEYWORD for p in _sig.parameters.values()
):
    adv_kwargs["non_tensor_batch"] = data.non_tensor_batch
```

**为什么需要签名检查？**

verl 的 `else` 分支处理所有非 GAE/GRPO 的 estimator。不是所有 estimator 都接受 `non_tensor_batch`——例如 `OPTIMAL_TOKEN_BASELINE` 不接受 `**kwargs`，强行传入会报错。

签名检查的逻辑：
1. 如果函数参数列表中有 `non_tensor_batch` → 传入
2. 如果函数接受 `**kwargs`（`VAR_KEYWORD`）→ 传入
3. 都不满足 → 不传入

我们的 `compute_cgrpo_advantage` 显式声明了 `non_tensor_batch: dict = None`，所以条件满足。

**向后兼容性**：这个修改对现有 estimator 完全无影响——不接受该参数的函数根本不会收到它。

---

## 6.4 cgrpo_advantage.py 实现详解

### 6.4.1 完整伪代码

**文件**: `examples/carr_deepsearch/reward/cgrpo_advantage.py`

```python
"""C-GRPO advantage estimator.

R_i = (1 - α) * R_outcome + α * R_outcome * R̂_rubric
R̂_rubric_i = R_rubric_i / max_group_rubric

当 non_tensor_batch 中没有 outcome_reward/rubric_reward 时，
fallback 到标准 GRPO（确保与非 CaRR 数据兼容）。
"""

from collections import defaultdict

import numpy as np
import torch

from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage, register_adv_est


@register_adv_est("cgrpo")
def compute_cgrpo_advantage(
    token_level_rewards: torch.Tensor,    # (bsz, response_length)
    response_mask: torch.Tensor,          # (bsz, response_length)
    index: np.ndarray,                    # (bsz,) — uid, 同 prompt 的 rollout 共享同一 uid
    config=None,                          # AlgoConfig
    non_tensor_batch: dict = None,        # 包含 outcome_reward, rubric_reward
    norm_adv_by_std_in_grpo: bool = True,
    **kwargs,
):
    # 1. 读取 alpha
    alpha = 0.3
    if config is not None:
        alpha = float(config.get("cgrpo_alpha", 0.3)) if hasattr(config, "get") else 0.3

    # 2. 检查是否有 CaRR 专用字段
    has_carr = (
        non_tensor_batch is not None
        and "outcome_reward" in non_tensor_batch
        and "rubric_reward" in non_tensor_batch
    )

    if has_carr:
        # 3. 提取 outcome 和 rubric reward
        outcome = np.array(non_tensor_batch["outcome_reward"], dtype=np.float32)
        rubric = np.array(non_tensor_batch["rubric_reward"], dtype=np.float32)
        bsz = len(outcome)

        # 4. 按 uid 分组
        id2indices = defaultdict(list)
        for i in range(bsz):
            id2indices[index[i]].append(i)

        # 5. 组内归一化 rubric
        norm_rubric = np.zeros(bsz, dtype=np.float32)
        for _, ids in id2indices.items():
            max_r = rubric[ids].max()
            if max_r > 0:
                norm_rubric[ids] = rubric[ids] / max_r
            # max_r == 0 → norm_rubric 保持 0

        # 6. C-GRPO 公式
        cgrpo_rewards = (1 - alpha) * outcome + alpha * outcome * norm_rubric

        # 7. 将标量 reward 放到 response 最后一个有效 token 位置
        new_rewards = torch.zeros_like(token_level_rewards)
        for i in range(bsz):
            valid_len = int(response_mask[i].sum())
            if valid_len > 0:
                new_rewards[i, valid_len - 1] = cgrpo_rewards[i]
        token_level_rewards = new_rewards

    # 8. 使用标准 GRPO 计算 group-relative advantage
    return compute_grpo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        config=config,
    )
```

### 6.4.2 逐步解析

**Step 1-2: 读取配置 + 检查可用性**

```python
alpha = float(config.get("cgrpo_alpha", 0.3))
```

`config` 是 `AlgoConfig` 实例（继承自 `BaseConfig`，提供 dict-like `get()` 方法）。如果 YAML 中配置了 `algorithm.cgrpo_alpha: 0.5`，这里就读到 0.5。

`has_carr` 检查 `non_tensor_batch` 中是否有 `outcome_reward` 和 `rubric_reward`。这两个值来自 `carr_reward.py` 的返回：

```python
# carr_reward.py 返回:
return {
    "score": outcome_reward,       # → rm_scores
    "outcome_reward": outcome_reward,  # → non_tensor_batch["outcome_reward"]
    "rubric_reward": rubric_reward,    # → non_tensor_batch["rubric_reward"]
}
```

如 Part 5 所述，这些值通过 `reward_extra_info` → `_postprocess` → `non_tensor_batch` 传递到这里。

**Step 3-5: 提取 + 分组 + 归一化**

```
假设一组（同一 prompt，n=4）的 rubric reward:
  rollout_0: rubric = 0.8
  rollout_1: rubric = 0.4
  rollout_2: rubric = 0.0
  rollout_3: rubric = 0.6

max_r = 0.8

归一化后:
  norm_rubric_0 = 0.8 / 0.8 = 1.0
  norm_rubric_1 = 0.4 / 0.8 = 0.5
  norm_rubric_2 = 0.0 / 0.8 = 0.0
  norm_rubric_3 = 0.6 / 0.8 = 0.75
```

**Step 6: C-GRPO 公式**

```
假设 α = 0.3, outcome = [1, 1, 0, 1]:

cgrpo_rewards = (1-0.3) * outcome + 0.3 * outcome * norm_rubric

rollout_0: 0.7 * 1 + 0.3 * 1 * 1.0  = 1.0   (答对 + 推理最好)
rollout_1: 0.7 * 1 + 0.3 * 1 * 0.5  = 0.85  (答对 + 推理一般)
rollout_2: 0.7 * 0 + 0.3 * 0 * 0.0  = 0.0   (答错)
rollout_3: 0.7 * 1 + 0.3 * 1 * 0.75 = 0.925 (答对 + 推理较好)
```

对比标准 GRPO（只用 outcome = [1, 1, 0, 1]），C-GRPO 区分了三条答对轨迹的推理质量。

**Step 7: 放置 reward 到 token 位置**

verl 的 `token_level_rewards` 是 `(bsz, response_length)` 张量。标量 outcome reward 通常放在 response 最后一个有效 token 位置。C-GRPO 用融合后的 reward 替换原始位置：

```python
new_rewards[i, valid_len - 1] = cgrpo_rewards[i]
```

**Step 8: 委托给标准 GRPO**

融合后的 reward 传给 `compute_grpo_outcome_advantage()`，由标准 GRPO 完成 group-relative normalization：

```
GRPO 的处理:
  scores = token_level_rewards.sum(dim=-1)  → [1.0, 0.85, 0.0, 0.925]
  mean_g = 0.694
  std_g = 0.389
  advantage = (scores - mean) / std → [0.787, 0.401, -1.784, 0.594]
```

这样最终的 advantage 既反映了答案正确性，又区分了推理质量。

### 6.4.3 Fallback 行为

当 `has_carr = False`（例如混合训练数据中有非 CaRR 的样本），函数直接跳过 C-GRPO 融合，使用原始 `token_level_rewards` 走标准 GRPO。这保证了兼容性。

---

## 6.5 数据流全景图

将 Part 5 和 Part 6 的数据流串联起来：

```
                       RL 训练的一次迭代
                       ═══════════════

┌─────────────────────────────────────────────────────────┐
│  1. Rollout (Agent Loop)                                │
│                                                         │
│  CaRRToolAgentLoop.run() → extra_fields:                │
│    messages: [{role, content, tool_calls}, ...]         │
│    task_unfinished: bool                                │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│  2. Streaming Reward                                    │
│                                                         │
│  NaiveRewardManager.run_single():                       │
│    extra_info = parquet.extra_info                       │
│      + tool_extra_fields (from AgentLoop)               │
│                                                         │
│    carr_reward.compute_score(extra_info=extra_info)      │
│      → HTTP POST /evaluate                              │
│      → {"score": 1.0, "outcome_reward": 1.0,           │
│          "rubric_reward": 0.67}                         │
│                                                         │
│    reward_extra_info = {score, outcome_reward,           │
│                         rubric_reward}                   │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│  3. Postprocess (_postprocess)                          │
│                                                         │
│  rm_scores[i, last_token] = score (= outcome_reward)    │
│  non_tensor_batch["outcome_reward"] = [1.0, ...]        │
│  non_tensor_batch["rubric_reward"] = [0.67, ...]        │
│  non_tensor_batch["uid"] = ["prompt_hash_0", ...]       │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│  4. Compute Advantage (ray_trainer.py)                  │
│                                                         │
│  token_level_rewards = rm_scores + optional KL penalty  │
│                                                         │
│  adv_estimator = "cgrpo"                                │
│  → compute_cgrpo_advantage(                             │
│      token_level_rewards,                               │
│      response_mask,                                     │
│      index=non_tensor_batch["uid"],                     │
│      config=AlgoConfig(cgrpo_alpha=0.3),                │
│      non_tensor_batch=non_tensor_batch,  ← 签名检查通过  │
│    )                                                    │
│                                                         │
│  C-GRPO 融合:                                            │
│    cgrpo_rewards = 0.7*outcome + 0.3*outcome*norm_rubric │
│  → GRPO group normalization                             │
│  → advantages, returns                                  │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│  5. Policy Update                                       │
│                                                         │
│  PPO/GRPO policy loss with advantages                   │
│  → 更新 actor 参数                                       │
└─────────────────────────────────────────────────────────┘
```

---

## 6.6 与原始 CaRR 的算法差异

| 对比项 | CaRR 原始论文 | 我们的实现 |
|--------|-------------|-----------|
| reward 融合位置 | reward server 端（`get_reward()`） | advantage estimator 端（`cgrpo_advantage.py`） |
| rubric 归一化 | 不做归一化 | Group-wise max 归一化 |
| advantage 基线 | 未明确说明 | GRPO group-relative（同 prompt 组内均值/方差） |
| `score` 语义 | = final_reward | = outcome_reward（rubric 另存） |
| Fallback | 无 | 自动退化到标准 GRPO |

**关于归一化的注意点**：CaRR 论文中的公式是 `R_i = (1-α)R_outcome + α·R_outcome·R̂_rubric`，其中 `R̂_rubric` 的具体归一化方式在实现中选择为 group-wise max normalization。这是因为：
- rubric reward 的绝对值跨 prompt 差异大（不同题目的 rubric 数量不同）
- Group-wise normalization 使得 rubric 的贡献在不同 prompt 间保持一致的相对尺度

---

## 6.7 注册与配置

### 6.7.1 外部模块注册

```bash
export VERL_USE_EXTERNAL_MODULES=examples.carr_deepsearch.tools.carr_agent_loop,examples.carr_deepsearch.reward.cgrpo_advantage
```

加载顺序：
1. verl 启动时 import `examples.carr_deepsearch.reward.cgrpo_advantage`
2. 模块顶层执行 `@register_adv_est("cgrpo")` 装饰器
3. `compute_cgrpo_advantage` 注册到 `ADV_ESTIMATOR_REGISTRY["cgrpo"]`
4. `ray_trainer.py` 中 `get_adv_estimator_fn("cgrpo")` 能找到它

### 6.7.2 YAML 配置

```yaml
algorithm:
  adv_estimator: cgrpo     # 使用 C-GRPO
  cgrpo_alpha: 0.3          # rubric 混合系数
  norm_adv_by_std_in_grpo: true  # 保持标准 GRPO 的 std normalization
```

### 6.7.3 验证要点

| # | 检查项 | 验证方式 |
|---|--------|----------|
| 1 | `cgrpo` 已注册到 registry | 启动时无 `Unknown advantage estimator` 错误 |
| 2 | `non_tensor_batch` 被传入 | 在 `compute_cgrpo_advantage` 开头打印 `non_tensor_batch.keys()` |
| 3 | `outcome_reward` / `rubric_reward` 值合理 | 打印几条样本检查范围 |
| 4 | 融合后 reward 值合理 | `cgrpo_rewards` 应在 [0, 1] 范围内 |
| 5 | Fallback 正常工作 | 用非 CaRR 数据测试，应退化为标准 GRPO |

---

下一部分（Part 7）将讲解完整的训练配置（SFT + RL YAML）和执行流程。
