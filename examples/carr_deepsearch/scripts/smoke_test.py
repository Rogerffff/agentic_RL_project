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
End-to-end smoke test for CaRR Deep Search servers.

Tests:
1. Tool server connectivity (search -> open -> find -> close)
2. Reward server connectivity (fixed sample -> outcome + rubric reward)

Usage:
    python smoke_test.py --tool       # test tool server only
    python smoke_test.py --reward     # test reward server only
    python smoke_test.py --all        # test both
"""

import asyncio
import sys

import aiohttp


async def test_tool_server(url="http://localhost:7230"):
    """Test CaRR tool server: start_session -> search -> open -> find -> close."""
    session_id = "smoke_test_001"
    print(f"Testing tool server at {url} ...")

    async with aiohttp.ClientSession() as session:
        # start_session
        resp = await session.post(url, json={
            "session_id": session_id,
            "name": "start_session",
            "arguments": {},
            "remote_env_info": {},
        })
        result = await resp.json()
        assert "output" in result, f"start_session failed: {result}"
        print("  start_session: OK")

        # browser.search
        resp = await session.post(url, json={
            "session_id": session_id,
            "name": "browser.search",
            "arguments": {"query": "breakpoint graph genome rearrangement", "num": 3},
            "remote_env_info": {"search_forbidden_strs": []},
        })
        assert resp.status == 200, f"browser.search HTTP {resp.status}"
        result = await resp.json()
        search_output = result.get("output", "")
        assert len(search_output) > 0, f"browser.search returned empty: {result}"
        # Distinguish real results from error/fallback text
        assert "No results found" not in search_output, f"browser.search found no results: {search_output[:200]}"
        print(f"  browser.search: OK ({len(search_output)} chars)")

        # browser.open
        resp = await session.post(url, json={
            "session_id": session_id,
            "name": "browser.open",
            "arguments": {"id": 0},
            "remote_env_info": {"search_forbidden_strs": []},
        })
        assert resp.status == 200, f"browser.open HTTP {resp.status}"
        result = await resp.json()
        open_output = result.get("output", "")
        assert len(open_output) > 0, f"browser.open returned empty: {result}"
        print(f"  browser.open: OK ({len(open_output)} chars)")

        # browser.find
        resp = await session.post(url, json={
            "session_id": session_id,
            "name": "browser.find",
            "arguments": {"pattern": "breakpoint"},
            "remote_env_info": {},
        })
        assert resp.status == 200, f"browser.find HTTP {resp.status}"
        result = await resp.json()
        find_output = result.get("output", "")
        # find may legitimately return empty if pattern not on page, so only check HTTP ok
        print(f"  browser.find: OK ({len(find_output)} chars)")

        # close_session
        await session.post(url, json={
            "session_id": session_id,
            "name": "close_session",
            "arguments": {},
            "remote_env_info": {},
        })
        print("  close_session: OK")

    print("Tool server smoke test PASSED.\n")


async def test_reward_server(url="http://localhost:8888"):
    """Test CaRR reward server with a fixed evaluation payload."""
    print(f"Testing reward server at {url} ...")

    payload = {
        "history": [
            {"role": "user", "content": "What is the title of the paper?"},
            {"role": "assistant", "content": "The paper is titled 'Test Paper Title'."},
        ],
        "label": "Test Paper Title",
        "task_unfinished": False,
        "remote_env_info": {
            "search_forbidden_strs": ["What is the title"],
            "rubrics": ["<E0> is the exact title of a paper."],
            "rubric_reward_ratio": 0.3,
        },
    }

    async with aiohttp.ClientSession() as session:
        resp = await session.post(
            f"{url}/evaluate",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=180),
        )
        assert resp.status == 200, f"Reward server HTTP {resp.status}"
        result = await resp.json()
        assert "reward" in result, f"Missing 'reward' in response: {result}"
        assert "outcome_reward" in result, f"Missing 'outcome_reward' in response: {result}"
        assert "rubric_reward" in result, f"Missing 'rubric_reward' in response: {result}"
        # Verify types are numeric (not None or string)
        assert isinstance(result["outcome_reward"], (int, float)), (
            f"outcome_reward is not numeric: {result['outcome_reward']}"
        )
        assert isinstance(result["rubric_reward"], (int, float)), (
            f"rubric_reward is not numeric: {result['rubric_reward']}"
        )
        assert result["outcome_reward"] > 0, (
            f"Expected positive outcome_reward for exact-match test case, got {result['outcome_reward']}. "
            "Check DEEPSEEK_API_KEY validity."
        )
        print(f"  outcome_reward: {result['outcome_reward']}")
        print(f"  rubric_reward:  {result['rubric_reward']}")
        print(f"  reward:         {result['reward']}")

    print("Reward server smoke test PASSED.\n")


if __name__ == "__main__":
    if "--tool" in sys.argv or "--all" in sys.argv:
        asyncio.run(test_tool_server())
    if "--reward" in sys.argv or "--all" in sys.argv:
        asyncio.run(test_reward_server())
    if len(sys.argv) == 1:
        print("Usage: python smoke_test.py --tool | --reward | --all")
