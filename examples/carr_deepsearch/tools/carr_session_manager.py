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
CaRR Tool Server session manager.

Manages request-level sessions against the CaRR tool server. Each agent rollout
(identified by request_id) gets one session that is lazily started on first tool
call and explicitly closed when the agent loop finishes.

ToolAgentLoop creates/releases tool instances per call, so session lifecycle
cannot be tied to individual tool instances. Instead, the session is managed
at the request level and closed in the agent loop's finally block.
"""

import logging
from typing import Any, Dict, Set, Tuple

import aiohttp

logger = logging.getLogger(__name__)


class CaRRSessionManager:
    """Singleton manager for CaRR tool server sessions."""

    _instance = None

    def __init__(self):
        self._started_sessions: Set[str] = set()
        self._session_data: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def get_instance(cls) -> "CaRRSessionManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def ensure_started(self, session_id: str, tool_server_url: str, **kwargs):
        """Lazily start a session if not already started.

        Only marks the session as started if the server responds successfully.
        On failure, the next tool call will retry start_session.
        """
        if session_id not in self._started_sessions:
            ok, _ = await self.call_server(tool_server_url, session_id, "start_session", {}, {})
            if ok:
                self._started_sessions.add(session_id)
                self._session_data[session_id] = kwargs
            else:
                logger.warning("start_session failed for %s, will retry on next tool call", session_id)

    async def close(self, session_id: str, tool_server_url: str):
        """Close a session and clean up tracking state."""
        if session_id in self._started_sessions:
            ok, _ = await self.call_server(tool_server_url, session_id, "close_session", {}, {})
            if not ok:
                logger.warning("close_session failed for %s", session_id)
            self._started_sessions.discard(session_id)
            self._session_data.pop(session_id, None)

    def get_session_data(self, session_id: str) -> Dict[str, Any]:
        return self._session_data.get(session_id, {})

    async def call_server(
        self,
        url: str,
        session_id: str,
        name: str,
        arguments: dict,
        remote_env_info: dict,
    ) -> Tuple[bool, str]:
        """Make a POST request to the CaRR tool server.

        Returns:
            (ok, output): ok is True if HTTP 200, output is the response text.
        """
        payload = {
            "session_id": session_id,
            "name": name,
            "arguments": arguments,
            "remote_env_info": remote_env_info,
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        logger.error("Tool server HTTP %s: %s", resp.status, error_text[:200])
                        return False, "Tool server HTTP %d" % resp.status
                    result = await resp.json()
                    return True, result.get("output", "")
        except Exception as e:
            logger.error("Error calling tool server: %s", e)
            return False, "Error calling tool server: %s" % e
