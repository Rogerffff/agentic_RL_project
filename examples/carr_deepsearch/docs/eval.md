评估流程完整分析
整体流程图

run_eval_browsecomp.sh
  ├── 启动工具服务器 (port 7230)         ← Serper + Jina
  ├── 启动奖励服务器 (port 8888)         ← DeepSeek Judge
  └── python -m verl.trainer.main_ppo    ← val_only=True
        └── RayPPOTrainer._validate()
              │
              │  对每个 val batch:
              ├── 1. AgentLoop 多轮 rollout
              │     ├── LLM 生成 → 解析 tool_call
              │     ├── 调用工具服务器 (search/open/find)
              │     ├── 工具返回 → 拼入 messages → 下一轮
              │     └── 达到终止条件 → 产出 reward_history
              │
              ├── 2. 在 AgentLoop 内部计算 reward (streaming)
              │     └── RewardLoopManager.compute_score()
              │           └── NaiveRewardManager.__call__()
              │                 └── carr_reward.compute_score()  ← 调用奖励服务器
              │                       POST http://localhost:8888/evaluate
              │                       └── 返回 {score, outcome_reward, rubric_reward}
              │
              ├── 3. rm_scores + reward_extra_info 写入 batch
              │     ├── batch["rm_scores"] = reward tensor
              │     └── non_tensor_batch["outcome_reward"] = [...]
              │         non_tensor_batch["rubric_reward"] = [...]
              │         non_tensor_batch["score"] = [...]
              │         meta_info["reward_extra_keys"] = ["score","outcome_reward","rubric_reward"]
              │
              ├── 4. extract_reward() 提取 rm_scores + extra_info
              │
              └── 5. process_validation_metrics() 汇总指标
                    └── 按 data_source 分组，计算 mean@N / std@N / best@N / maj@N
关键回答
1. 评估是否使用 DeepSeek 评判？

是的，完全一样的奖励服务器。 评估和训练用的是同一个 carr_reward.compute_score() 函数，同一个奖励服务器 /evaluate 端点，同一个 DeepSeek LLM Judge。

具体链路：

carr_grpo.yaml 中 reward.reward_model.enable: false
因此 use_rm = False，enable_agent_reward_loop = True（ray_trainer.py:835）
reward 计算在 AgentLoop 内部以 streaming 方式完成（agent_loop.py:604, 716）
最终调用 carr_reward.compute_score() → POST 到奖励服务器
2. 评估记录哪些指标？

不只是准确率，也记录 rubric_reward 等所有额外指标。

carr_reward.compute_score() 返回 dict：


{"score": outcome_reward, "outcome_reward": outcome_reward, "rubric_reward": rubric_reward}
这三个 key 全部通过 reward_extra_info → non_tensor_batch → reward_extra_infos_dict → process_validation_metrics() 流入最终指标。

输出的完整指标列表：

指标	前缀	含义
val-core/{ds}/reward/mean@N	core	reward 均值（= outcome_reward，主指标）
val-aux/{ds}/outcome_reward/mean@N	aux	outcome 准确率（0 或 1）
val-aux/{ds}/rubric_reward/mean@N	aux	rubric 评分（0~1 连续值）
val-aux/{ds}/score/mean@N	aux	同 outcome_reward
val-aux/num_turns/min,max,mean	aux	agent 对话轮数
其中 {ds} = data_source，对 BrowseComp 评测是 browsecomp，对 RL val 是 carr_deepsearch。

val-core vs val-aux 的区分逻辑（ray_trainer.py:633-646）：只有 reward（或 acc）变量的 mean@N 被标为 val-core，其余全部是 val-aux。

3. 奖励服务器内部对每个样本做什么？

对每个已完成的 trajectory（task_unfinished=False）：


get_reward()
├── get_outcome_reward()     ← 1 次 DeepSeek 调用
│   └── 二分类判断: "Correct: yes/no"
│   └── 返回 0 或 1
│
└── get_rubric_reward()      ← 2 次 DeepSeek 调用 (rubric_reward_ratio=0.3 > 0)
    ├── identify_entity()    ← 从 response 中提取 rubric 实体
    ├── judge_rubric()       ← 判断每条 rubric 是否有引用支撑
    └── BFS 连通性检查       ← 纯本地计算（不调用 API）
    └── 返回 0~1 连续值 (supported_rubrics / total_rubrics)
task_unfinished=True 的样本直接返回全零，不调用 DeepSeek。

4. 训练时 vs 评测时的 reward 差异

评测时 reward 仅用于指标统计，不参与梯度计算。训练时 reward 还会经过 C-GRPO advantage 融合：


训练: advantage = (1-α)*R_outcome + α*R_outcome*R̂_rubric  (用于 policy loss)
评测: 直接输出 outcome_reward 和 rubric_reward 的均值  (仅统计)
5. validation_data_dir 会保存什么？

如果设了 trainer.validation_data_dir（脚本中第 130 行），会调用 _dump_generations() 将每个样本的 input/output/ground_truth/score/reward_extra_info 存为文件，可事后分析每道题的模型回答和得分。

