# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
C-GRPO (Citation-aware GRPO) advantage estimator.

Fuses outcome and rubric rewards before standard GRPO normalization:
    R_i = (1 - alpha) * R_outcome + alpha * R_outcome * R_hat_rubric
    R_hat_rubric_i = R_rubric_i / max(R_rubric in group)

Falls back to standard GRPO if outcome_reward / rubric_reward not present
in non_tensor_batch (e.g. when running eval without CaRR reward server).

Registration:
    @register_adv_est("cgrpo") — activated via VERL_USE_EXTERNAL_MODULES
    Config: algorithm.adv_estimator = "cgrpo", algorithm.cgrpo_alpha = 0.3
"""

from collections import defaultdict
from typing import Optional

import numpy as np
import torch

from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage, register_adv_est


@register_adv_est("cgrpo")
def compute_cgrpo_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    config=None,
    non_tensor_batch: Optional[dict] = None,
    norm_adv_by_std_in_grpo: bool = True,
    **kwargs,
):
    """Compute C-GRPO advantage by fusing outcome and rubric rewards.

    Args:
        token_level_rewards: (bs, response_length) — token-level rewards from reward manager.
        response_mask: (bs, response_length) — 1 for valid response tokens.
        index: (bs,) — group index for GRPO normalization (uid).
        config: AlgoConfig with cgrpo_alpha field.
        non_tensor_batch: Dict with outcome_reward and rubric_reward arrays from carr_reward.py.
        norm_adv_by_std_in_grpo: Whether to normalize by std (True=GRPO, False=Dr.GRPO).

    Returns:
        advantages: (bs, response_length)
        returns: (bs, response_length)
    """
    alpha = 0.3
    if config is not None:
        alpha = float(config.get("cgrpo_alpha", 0.3)) if hasattr(config, "get") else 0.3

    has_carr = (
        non_tensor_batch is not None
        and "outcome_reward" in non_tensor_batch
        and "rubric_reward" in non_tensor_batch
    )

    if has_carr:
        # C-GRPO rebuilds token_level_rewards from scratch (zeros + fused scalar at last token).
        # This discards any pre-existing dense rewards such as KL penalty added by
        # ray_trainer.py when use_kl_in_reward=True. Guard against silent misuse.
        if config is not None and hasattr(config, "get"):
            use_kl = config.get("use_kl_in_reward", False)
            assert not use_kl, (
                "C-GRPO advantage estimator is incompatible with use_kl_in_reward=True. "
                "KL penalty in token_level_rewards would be discarded by reward reconstruction. "
                "Set algorithm.use_kl_in_reward=false when using adv_estimator=cgrpo."
            )

        outcome = np.array(non_tensor_batch["outcome_reward"], dtype=np.float32)
        rubric = np.array(non_tensor_batch["rubric_reward"], dtype=np.float32)
        bsz = len(outcome)

        # Normalize rubric within each group: R_hat = R_rubric / max(group)
        id2indices = defaultdict(list)
        for i in range(bsz):
            id2indices[index[i]].append(i)

        norm_rubric = np.zeros(bsz, dtype=np.float32)
        for _, ids in id2indices.items():
            max_r = rubric[ids].max()
            if max_r > 0:
                norm_rubric[ids] = rubric[ids] / max_r

        # C-GRPO fusion: R = (1-alpha)*outcome + alpha*outcome*norm_rubric
        cgrpo_rewards = (1 - alpha) * outcome + alpha * outcome * norm_rubric

        # Place fused reward at last valid token position
        new_rewards = torch.zeros_like(token_level_rewards)
        for i in range(bsz):
            valid_len = int(response_mask[i].sum())
            if valid_len > 0:
                new_rewards[i, valid_len - 1] = cgrpo_rewards[i]
        token_level_rewards = new_rewards

    return compute_grpo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        config=config,
    )
