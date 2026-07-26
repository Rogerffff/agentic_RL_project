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
CaRR browser tool adapter for verl agent loop.

A single CaRRBrowserTool class handles all three CaRR browser operations
(browser.search, browser.open, browser.find), differentiated by self.name
from the tool schema. The tool server URL comes from the tool config YAML.

Session lifecycle:
- create(): no-op — session is managed at the request level by CaRRSessionManager
- execute(): lazily starts session via session_manager.ensure_started(), then calls tool server
- release(): no-op — session is closed by CaRRToolAgentLoop in its finally block
"""

import logging
from typing import Any, Dict, Optional, Tuple
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

from .carr_session_manager import CaRRSessionManager

logger = logging.getLogger(__name__)


class CaRRBrowserTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.tool_server_url = config.get("tool_server_url", "http://localhost:7230")
        self.session_manager = CaRRSessionManager.get_instance()

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> Tuple[str, ToolResponse]:
        # Session lifecycle is managed at request level, not per tool call
        if instance_id is None:
            instance_id = str(uuid4())
        return instance_id, ToolResponse()

    async def execute(self, instance_id: str, parameters: Dict[str, Any], **kwargs) -> Tuple[ToolResponse, float, Dict]:
        agent_data = kwargs.get("agent_data")
        if agent_data is None:
            return ToolResponse(text="Missing agent_data"), 0.0, {}

        session_id = agent_data.request_id

        # Get search_forbidden_strs from tools_kwargs (uses .get() for safety — browser.find has no entry)
        tool_kwargs = agent_data.tools_kwargs.get(self.name, {})
        create_kwargs = tool_kwargs.get("create_kwargs", {})
        search_forbidden_strs = create_kwargs.get("search_forbidden_strs", [])

        # Lazily start session on first tool call for this request
        await self.session_manager.ensure_started(
            session_id,
            self.tool_server_url,
            search_forbidden_strs=search_forbidden_strs,
        )

        # browser.open: convert string id to int (model may output "0" instead of 0)
        if self.name == "browser.open" and "id" in parameters:
            try:
                parameters["id"] = int(parameters["id"])
            except (TypeError, ValueError):
                pass  # keep as-is if it's a URL string

        remote_env_info = {"search_forbidden_strs": search_forbidden_strs}
        ok, result = await self.session_manager.call_server(
            self.tool_server_url,
            session_id,
            self.name,
            parameters,
            remote_env_info,
        )
        return ToolResponse(text=str(result)), 0.0, {}

    async def release(self, instance_id: str, **kwargs):
        # Session is closed by CaRRToolAgentLoop in its finally block, not here
        return
