# 同步 vs 异步 GRPO：更新机制深度对比

> 本文档解释 verl 中同步 GRPO 和异步 GRPO 在 PPO 更新机制上的区别。
> 重点回答：异步 RL 是否只是拆分了 Rollouter 和 Trainer？还是算法本身也变了？
>
> 结论先放这里：**核心算法没变，但 old_log_prob 的来源和计算方式变了，
> 导致重要性采样的 ratio 行为不同。在合理参数下差异可控。**

---

## 1. 同步 GRPO 的完整更新流程

### 一个 step 的流程

以 `batch_size=128, n=8, ppo_epochs=1` 为例：

```
                           用参数 θ_v0 做所有事情
                           ─────────────────────
① Rollout（生成）
   128 个 prompt × 8 个回答 = 1024 条轨迹
   用 θ_v0 生成，顺便记录 rollout_log_probs

② Compute Reward
   调 DeepSeek Judge 打分

③ Compute old_log_prob                     ← 关键步骤
   用 θ_v0 对所有 1024 条轨迹重新算一遍 log_prob
   得到 old_log_probs
   此时 old_log_probs ≈ rollout_log_probs（因为参数没变，都是 θ_v0）

④ Compute ref_log_prob
   用冻结的参考策略 θ_ref 算 log_prob（KL 约束用）

⑤ Compute Advantage
   C-GRPO 组内对比：同一 prompt 的 8 个回答互相比较
   好的 advantage > 0，差的 < 0

⑥ Update Actor（PPO 梯度更新）
   for mini_batch in mini_batches:
     new_log_prob = forward(mini_batch, θ_v0)    # 用当前参数算
     ratio = exp(new_log_prob - old_log_prob)     # ≈ 1.0（还没更新过）
     loss = -min(ratio × A, clip(ratio) × A)
     loss.backward()
     optimizer.step()                              # θ_v0 变成 θ_v0.01

⑦ 下一个 step
   用 θ_v0.01 重新做 rollout...
```

### ppo_epochs > 1 时

如果 `ppo_epochs=4`，步骤 ⑥ 会在**同一批数据**上反复训练 4 次：

```
old_log_prob = compute(θ_v0)  ← 固定不动

epoch 1: new_log_prob(θ_v0)   → ratio ≈ 1.00 → update → θ 变了一点
epoch 2: new_log_prob(θ_v0.1) → ratio ≈ 1.02 → update → θ 又变了
epoch 3: new_log_prob(θ_v0.2) → ratio ≈ 1.05 → update → θ 又变了
epoch 4: new_log_prob(θ_v0.3) → ratio ≈ 1.08 → update → θ 又变了

ratio 逐渐偏离 1.0，但 PPO 的 clip(ratio, 1-ε, 1+ε) 限制了偏离幅度。
```

**这是理解异步模式的关键对比点**——异步的 `trigger_sync_step > 1` 和同步的 `ppo_epochs > 1` 本质上是同一种"用同一个 π_old 做多步更新"。

---

## 2. 异步 GRPO 的更新流程

### 一个 sync 周期的流程

以 `b=4, n=8, trigger_sync_step=4, bypass_mode=True` 为例：

```
param sync 后，Rollouter 用 θ_v0 开始生成
────────────────────────────────────────────

global_step 1（local_trigger_step=1）：
  ① 从 Queue 取 4 个 prompt（都是 θ_v0 生成的）
  ② Compute Reward
  ③ old_log_prob = rollout_log_probs          ← bypass 模式：直接用 rollout 记录的
  ④ ref_log_prob
  ⑤ Advantage（C-GRPO 组内对比，4×8=32 条轨迹）
  ⑥ Update Actor → θ_v0 变成 θ_0.1
  ⑦ local_trigger_step=1 < 4 → 不 sync

global_step 2（local_trigger_step=2）：
  ① 从 Queue 取 4 个新 prompt（还是 θ_v0 生成的）
  ② Compute Reward
  ③ old_log_prob = rollout_log_probs          ← 还是 θ_v0 生成时的 log_prob
  ④ ref_log_prob
  ⑤ Advantage
  ⑥ Update Actor
     new_log_prob = forward(θ_0.1)            ← 用的是已更新过 1 次的参数
     ratio = exp(new_log_prob - old_log_prob)  ← ratio 开始偏离 1.0
     → θ_0.1 变成 θ_0.2
  ⑦ local_trigger_step=2 < 4 → 不 sync

global_step 3：类似，ratio 偏离更大
global_step 4：类似，ratio 偏离最大
  → local_trigger_step=4 = trigger_sync_step → 触发 param sync！
  → 把 θ_0.4 同步给 Rollouter
  → param_version: v0 → v1
```

### 和同步模式的逐步对比

```
                    同步（ppo_epochs=4）              异步（trigger_sync_step=4）
─────────────────────────────────────────────────────────────────────────────────
数据来源          同一个 batch 重复用 4 次          4 个不同 batch（不同 prompt）
old_log_prob      算一次，4 轮共享                  每个 global_step 各自的 rollout_log_probs
优势函数          算一次，4 轮共享                  每个 global_step 独立计算（不同 prompt）
ratio 演变        1.00 → 1.02 → 1.05 → 1.08       类似趋势
信息量            低（同样的数据）                   高（不同的 prompt 和回答）
```

**异步的优势**：虽然 ratio 行为类似，但每个 global_step 用的是**不同的 prompt**，信息量更大。同步的 ppo_epochs > 1 是在同一批数据上反复榨取。

---

## 3. old_log_prob 的计算方式对比

这是同步和异步最核心的区别。

### 模式 1：同步模式

```
old_log_prob = forward(batch, θ_current)

θ_current 就是 rollout 时的参数（因为 rollout 刚做完，参数还没变）
→ old_log_prob ≈ rollout_log_probs
→ 在整个 ppo_epochs 中固定不变
```

### 模式 2：异步 bypass_mode=True（默认）

```
old_log_prob = rollout_log_probs（直接赋值，不做任何计算）

优点：零成本，不需要额外的 forward pass
缺点：rollout_log_probs 是 Rollouter 生成时算的，可能来自旧版本参数

系统中只有 2 个策略：
  π_rollout（生成样本时的策略）
  π_θ（当前训练中的策略）
  ratio = π_θ / π_rollout
```

代码位置：`core_algos.py` line 2064+，`compute_policy_loss_bypass_mode()`

### 模式 3：异步 bypass_mode=False（Decoupled PPO）

这是更精确但更昂贵的模式。

```
每个 global_step 都需要：
  1. 把当前参数 θ_current 存到 CPU
  2. 从 CPU 恢复 θ_v0（rollout 时的参数版本）
  3. 用 θ_v0 做 forward pass 算 old_log_prob
  4. 从 CPU 恢复 θ_current 回来

系统中有 3 个策略：
  π_rollout（生成样本时的策略）
  π_old = π_v0（通过 save/restore 恢复的参考策略）
  π_θ（当前训练中的策略）
```

代码实现（`fully_async_trainer.py` line 488-508）：

```python
def _compute_old_log_prob(self, batch):
    if self.local_trigger_step == 1:
        # 第 1 步：存一份 θ_v0 的副本到 CPU
        self.actor_rollout_wg.save_model_to_cpu(1)
        return super()._compute_old_log_prob(batch)  # 用当前参数算
    else:
        # 第 2,3,4 步：参数已经被更新过了
        self.actor_rollout_wg.save_model_to_cpu(self.local_trigger_step)  # 存当前 θ
        self.actor_rollout_wg.restore_model_from_cpu(1)   # 恢复 θ_v0 ← "时间旅行"
        old_log_prob = super()._compute_old_log_prob(batch)  # 用 θ_v0 算
        self.actor_rollout_wg.restore_model_from_cpu(self.local_trigger_step)  # 恢复当前 θ
        return old_log_prob
```

底层实现通过 FSDP 的 sharded state dict 存取（`fsdp_workers.py` line 238-252）。

### 三种模式的对比

```
                      同步              异步 bypass=True      异步 bypass=False
─────────────────────────────────────────────────────────────────────────────────
old_log_prob 来源    forward(θ_current)  rollout_log_probs     forward(θ_v0, 恢复的)
额外计算成本        1 次 forward        0                     1 次 forward + save/restore
策略数量            2（π_old, π_θ）     2（π_rollout, π_θ）   3（π_rollout, π_old, π_θ）
精确度              最高                中等                   较高
适用场景            同步训练            async 快速训练         async 高精度训练
```

---

## 4. 重要性采样 ratio 的行为对比

### ratio 是什么

PPO 的核心公式（`core_algos.py` line 1210-1213）：

```
ratio = exp(new_log_prob - old_log_prob) = π_θ(a|s) / π_old(a|s)

loss = -min(ratio × A, clip(ratio, 1-ε, 1+ε) × A)
```

ratio = 1.0 表示当前策略和旧策略一样。偏离越大，说明策略变化越大。
PPO clip（通常 ε=0.2）把 ratio 限制在 [0.8, 1.2] 范围内。

### ratio 在不同模式下的演变

```
同步 ppo_epochs=4（同一批数据，old_log_prob 固定为 θ_v0 算的）：
  epoch 1: ratio ≈ 1.00  ← 刚开始，θ 还没变
  epoch 2: ratio ≈ 1.02  ← θ 更新了一次
  epoch 3: ratio ≈ 1.05  ← θ 更新了两次
  epoch 4: ratio ≈ 1.08  ← θ 更新了三次
  → clip 机制：超出 [0.8, 1.2] 的部分被裁掉

异步 trigger_sync_step=4, bypass_mode=True：
  step 1: ratio = exp(forward(θ_v0) - rollout(θ_v0)) ≈ 1.00
  step 2: ratio = exp(forward(θ_0.1) - rollout(θ_v0)) ≈ 1.02
  step 3: ratio = exp(forward(θ_0.2) - rollout(θ_v0)) ≈ 1.05
  step 4: ratio = exp(forward(θ_0.3) - rollout(θ_v0)) ≈ 1.08
  → 趋势一样，但每步用的是不同 prompt 的 rollout_log_probs

异步 + stale 样本（staleness>0，样本来自更旧的 θ_v-1）：
  step 1: ratio = exp(forward(θ_v0) - rollout(θ_v-1)) ≈ 1.10  ← 起点就偏离了！
  step 2: ratio ≈ 1.13
  step 3: ratio ≈ 1.16
  step 4: ratio ≈ 1.20  ← 接近 clip 边界
  → 更容易被 clip 裁掉，有效梯度信号减弱
```

### bypass 模式下 PPO-clip vs REINFORCE 的不同处理

```
PPO-clip（默认，core_algos.py line 2064+）：
  ratio = π_θ / π_rollout
  loss = -min(ratio × A, clip(ratio) × A)
  → 不额外施加 IS 权重（clipping 本身就在处理 off-policy）

REINFORCE（可选）：
  IS_weight = π_θ / π_rollout
  loss = IS_weight × log π_θ(a|s) × A
  → 显式施加重要性采样权重修正 off-policy 偏差
```

---

## 5. 优势函数：完全一样

**GRPO 的优势计算不受 sync/async 影响。**

无论同步还是异步，C-GRPO 的计算都是（`core_algos.py` line 267-297）：

```
对于每个 prompt（组）的 n 个回答：
  reward_i = (1-α) × outcome_i + α × outcome_i × rubric_hat_i
  advantage_i = (reward_i - mean(rewards)) / std(rewards)

  好回答（advantage > 0）被鼓励
  差回答（advantage < 0）被抑制
```

异步模式下，每个 global_step 的 4 个 prompt 各自独立做组内对比，和同步完全一样。

---

## 6. Partial Rollout 的特殊问题

### 跨版本样本

partial rollout 允许在参数同步时中断正在生成的样本，sync 后用新参数继续。这导致一个样本可能**跨越两个参数版本**：

```
prompt_18 的 token 序列：

  [用 θ_v0 生成的 token...][用 θ_v1 生成的 token...]
  ├── 轮次 1-12 ──────────┤├── 轮次 13-21 ──────────┤
       search, open, find        search, open, 最终答案
       ~8000 tokens              ~5000 tokens

  理论上正确的 old_log_prob：
    前 8000 个 token → 应该用 θ_v0 算
    后 5000 个 token → 应该用 θ_v1 算

  实际代码的处理：
    整个 13000 个 token → 用同一个版本算（bypass 模式用 rollout_log_probs）
```

**这是一个近似**。为什么可以接受？

1. 参数版本之间差异通常很小（只差 1-4 步梯度更新）
2. `max_param_span=2` 限制了最大跨越版本数——超过就强制终止样本
3. PPO clip 天然容忍一定程度的 ratio 偏离

### param_version_start / param_version_end

代码通过这两个字段追踪每个样本的跨版本情况（`detach_utils.py` line 27-48）：

```python
@dataclass
class RolloutSample:
    param_version: int                 # 样本开始时的参数版本
    param_version_start: list[int]     # 每条轨迹开始时的版本
    param_version_end: list[int]       # 每条轨迹结束时的版本
```

```
例：prompt_18 在 θ_v0 开始，θ_v1 结束
  param_version_start = [0, 0, 0, 0, 0, 0, 0, 0]  （8 个回答都从 v0 开始）
  param_version_end   = [1, 1, 1, 1, 1, 0, 0, 1]  （部分在 v1 结束，部分在 v0 完成）
```

这些字段用于计算 `fully_async/partial/partial_ratio` 和 `max_partial_span` 指标。

---

## 7. Off-Policy 程度的控制旋钮

### 哪些因素增加 off-policy 程度

```
off-policy 程度从低到高：

  同步 RL (ppo_epochs=1)
    → ratio ≈ 1.0，完全 on-policy
    │
  同步 RL (ppo_epochs=4)
    → ratio 逐渐偏离到 ~1.08
    │
  异步 RL (trigger_sync_step=4, staleness=0)
    → 类似 ppo_epochs=4，但用不同 prompt
    │
  异步 RL (trigger_sync_step=4, staleness=0.5)
    → 部分样本来自更旧版本，ratio 起点就偏离
    │
  异步 RL (trigger_sync_step=4, staleness=1.0)
    → 更多旧版本样本混入
    │
  异步 RL (trigger_sync_step=8, staleness=1.0)
    → 参数变化大 + 旧样本多 → ratio 可能频繁触及 clip 边界
```

### 控制旋钮总结

| 旋钮 | 调大的效果 | 调大的代价 |
|------|-----------|-----------|
| `trigger_sync_step` | 减少 sync 次数，提高吞吐 | ratio 偏离更大，训练可能不稳定 |
| `staleness_threshold` | Rollouter 不暂停，吞吐更高 | 更多旧版本样本，off-policy 程度增加 |
| `ppo_epochs` | 同一批数据榨取更多梯度信号 | ratio 偏离更大 |
| `max_param_span` | 允许样本跨更多版本 | 跨版本近似误差增大 |

### PPO Clip 的天然容忍度

PPO clip（ε=0.2）的设计初衷就是允许一定程度的 off-policy：

```
ratio 在 [0.8, 1.2] 内 → 梯度正常传播
ratio 超出 [0.8, 1.2] → 梯度被截断

这意味着：
  如果 θ 从 v0 更新了 4 步变成 θ_0.4，
  只要 ratio 还在 [0.8, 1.2] 内，训练效果和 on-policy 接近。

  通常 trigger_sync_step=2~4 + staleness=0.5 不会让 ratio 频繁越界。
  但 trigger_sync_step=8 + staleness=1.0 可能会。
```

### rollout_correction 的额外保护

bypass 模式下，代码还会计算拒绝采样（rejection sampling）掩码（`rollout_corr_helper.py`）：

```
对每个 token：
  IS_weight = π_θ / π_rollout
  如果 IS_weight 偏离太大（某些 token 的 ratio 极端）
  → response_mask 设为 0 → 这个 token 的梯度不参与更新

这提供了 clip 之外的额外安全网。
```

---

## 8. 结论：量变到质变的边界在哪里

### 没有问题（纯架构变化）

| 方面 | 说明 |
|------|------|
| Rollouter/Trainer 拆分 | 不影响算法。只是把 rollout 和 train 放到不同 GPU 上 |
| 流式消费 | 不影响算法。只是样本一个一个到，而不是一批到 |
| 优势函数计算 | 完全一样。GRPO 组内对比不受 async 影响 |
| trigger_sync_step=1 + staleness=0 | 等价于同步模式 |

### 有差异但可控（合理参数下）

| 方面 | 差异 | 为什么可控 |
|------|------|-----------|
| trigger_sync_step=2~4 | ratio 逐步偏离，类似 ppo_epochs>1 | PPO clip 限制偏离幅度 |
| staleness=0.5 | 部分样本来自旧版本 | bypass_mode + 拒绝采样处理 |
| bypass_mode=True | old_log_prob 用 rollout 记录的近似值 | 省了 forward pass，精度损失小 |
| partial rollout 跨版本 | 同一样本前后用不同参数生成 | max_param_span=2 限制跨度 |

### 需要注意的风险（参数激进时）

| 风险 | 触发条件 | 表现 |
|------|---------|------|
| ratio 频繁被 clip | trigger_sync_step=8 + staleness=1.0 | 有效梯度信号减弱，训练变慢 |
| response length 膨胀 | 长期 off-policy 累积 | 模型学会生成很长但无用的内容 |
| 训练不稳定 | ratio 极端偏离 | loss 震荡、reward 下降 |

### 实践建议

```
保守（接近 on-policy）：
  trigger_sync_step=1~2, staleness=0, bypass_mode=True
  → 几乎等价同步，但有 async 的吞吐优势

平衡（推荐）：
  trigger_sync_step=2~4, staleness=0.5, bypass_mode=True
  → off-policy 程度可控，吞吐提升明显

激进（高吞吐优先）：
  trigger_sync_step=4~8, staleness=1.0, bypass_mode=False
  → 需要 rollout_correction 保护，监控 ratio 分布和 response length
```

**一句话总结：异步 RL 不是一个不同的算法，而是同步 RL 的一种松弛。松弛程度由 `trigger_sync_step` 和 `staleness_threshold` 控制。PPO clip 天然能容忍一定的松弛。在 trigger_sync_step=2~4 + staleness=0.5 下，训练效果接近同步，但吞吐高得多。**
