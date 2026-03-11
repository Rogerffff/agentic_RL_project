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
CaRR reward function for verl NaiveRewardManager.

Called by NaiveRewardManager.run_single() with merged extra_info containing:
- Static fields from parquet: rubrics, search_forbidden_strs, rubric_reward_ratio
- Dynamic fields from AgentLoop (via tool_extra_fields): messages, task_unfinished

Sends history to CaRR reward server /evaluate endpoint and returns
{score, outcome_reward, rubric_reward} dict. All keys propagate to
non_tensor_batch for downstream C-GRPO advantage estimation.
"""

import asyncio
import logging
import os

import aiohttp

logger = logging.getLogger(__name__)

REWARD_SERVER_URL = os.environ.get("CARR_REWARD_SERVER_URL", "http://localhost:8888")
# Must be > server's internal 600s timeout to avoid client-side timeout first
REWARD_TIMEOUT = int(os.environ.get("CARR_REWARD_TIMEOUT", "650"))
_HTTP_SESSIONS: dict[asyncio.AbstractEventLoop, aiohttp.ClientSession] = {}


def _get_http_session() -> aiohttp.ClientSession:
    loop = asyncio.get_running_loop()
    session = _HTTP_SESSIONS.get(loop)
    if session is None or session.closed:
        connector = aiohttp.TCPConnector(limit=512, keepalive_timeout=30, ttl_dns_cache=300)
        session = aiohttp.ClientSession(connector=connector)
        _HTTP_SESSIONS[loop] = session
    return session


async def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """Compute CaRR reward by calling the reward server.

    Args:
        data_source: Dataset identifier (e.g. "carr_deepsearch", "browsecomp").
        solution_str: Decoded model response text.
        ground_truth: Ground truth answer string.
        extra_info: Merged dict with rubrics, messages, search_forbidden_strs, etc.

    Returns:
        Dict with score, outcome_reward, rubric_reward.
        score = outcome_reward (C-GRPO fusion happens in advantage stage).
    """
    extra_info = extra_info or {}

    messages = extra_info.get("messages", [])
    rubrics = extra_info.get("rubrics", [])
    search_forbidden_strs = extra_info.get("search_forbidden_strs", [])
    rubric_reward_ratio = extra_info.get("rubric_reward_ratio", 0.3)
    task_unfinished = extra_info.get("task_unfinished", False)

    # Fallback: build minimal history if messages is empty (e.g. single-turn eval)
    if not messages:
        messages = [
            {"role": "user", "content": search_forbidden_strs[0] if search_forbidden_strs else "Question"},
            {"role": "assistant", "content": solution_str},
        ]

    # Force task_unfinished if last message is not from assistant
    if messages and messages[-1].get("role") != "assistant":
        task_unfinished = True

    # Pass-through metrics (constructed AFTER task_unfinished override so values are consistent)
    _pass = {
        "task_unfinished": float(task_unfinished),
        "tool_call_counts": float(extra_info.get("tool_call_counts", 0)),
        "search_count": float(extra_info.get("search_count", 0)),
        "open_count": float(extra_info.get("open_count", 0)),
        "find_count": float(extra_info.get("find_count", 0)),
        "hit_limit": float(extra_info.get("hit_limit", False)),
        "parse_error_count": float(extra_info.get("parse_error_count", 0)),
    }

    payload = {
        "history": messages,
        "label": ground_truth,
        "task_unfinished": task_unfinished,
        "remote_env_info": {
            "search_forbidden_strs": search_forbidden_strs,
            "rubrics": rubrics,
            "rubric_reward_ratio": rubric_reward_ratio,
        },
    }

    try:
        session = _get_http_session()
        async with session.post(
            f"{REWARD_SERVER_URL}/evaluate",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=REWARD_TIMEOUT),
        ) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                logger.error("Reward server HTTP %s: %s", resp.status, error_text[:200])
                return {"score": 0.0, "outcome_reward": 0.0, "rubric_reward": 0.0, **_pass}
            result = await resp.json()
    except Exception as e:
        logger.error("Reward server call failed: %s", e)
        return {"score": 0.0, "outcome_reward": 0.0, "rubric_reward": 0.0, **_pass}

    outcome_reward = float(result.get("outcome_reward", 0.0))
    rubric_reward = float(result.get("rubric_reward", 0.0))

    # score = outcome_reward; C-GRPO fusion (mixing rubric) happens in advantage stage
    return {
        "score": outcome_reward,
        "outcome_reward": outcome_reward,
        "rubric_reward": rubric_reward,
        **_pass,
    }
