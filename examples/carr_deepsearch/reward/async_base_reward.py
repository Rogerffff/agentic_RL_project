# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
Cheap local reward for fully-async base probes.

This deliberately avoids the CaRR reward/tool stack. It only exists to verify
that fully-async can collect samples, perform a real update, synchronize the
new weights, and shut down cleanly with `rollout.test_freq=0`.
"""


async def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    text = (solution_str or "").strip()
    if not text:
        return {
            "score": 0.0,
            "dummy_reward_length": 0.0,
            "dummy_reward_unique_chars": 0.0,
        }

    text_len = len(text)
    unique_chars = len(set(text))
    length_score = min(1.0, text_len / 256.0)
    diversity_bonus = min(0.25, unique_chars / 128.0)
    score = float(length_score + diversity_bonus)
    return {
        "score": score,
        "dummy_reward_length": float(text_len),
        "dummy_reward_unique_chars": float(unique_chars),
    }
