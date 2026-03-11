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
CaRR custom agent loop for deep search.

Extends ToolAgentLoop to maintain a separate ``reward_history`` that matches
the format expected by the CaRR reward server's ``/evaluate`` endpoint:

    history = [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "...", "tool_calls": [
            {"tool_call_id": "...", "name": "...", "arguments": "..."}
        ]},
        {"role": "tool", "content": [
            {"tool_call_id": "...", "output": "..."}
        ]},
        {"role": "assistant", "content": "final answer"},
    ]

The base ToolAgentLoop does not track ``tool_call_id`` or maintain a history
with this shape, so we override the state handlers to build it ourselves.

Registration: ``@register("carr_tool_agent")`` — activated via
    ``VERL_USE_EXTERNAL_MODULES=examples.carr_deepsearch.tools.carr_agent_loop``

Session lifecycle:
    The CaRR tool session is closed in the ``finally`` block of ``run()``,
    not in individual tool ``release()`` calls.
"""

import asyncio
import json
import logging
import os
import re
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, register
from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState, ToolAgentLoop
from verl.utils.rollout_trace import rollout_trace_op

from .carr_session_manager import CaRRSessionManager

logger = logging.getLogger(__name__)


@register("carr_tool_agent")
class CaRRToolAgentLoop(ToolAgentLoop):
    """Agent loop that maintains CaRR-compatible reward history."""

    def __init__(self, trainer_config, server_manager, tokenizer, processor, **kwargs):
        super().__init__(trainer_config, server_manager, tokenizer, processor, **kwargs)
        self.session_manager = CaRRSessionManager.get_instance()
        # tool_server_url is read from the first tool's config (all share the same URL)
        self.tool_server_url = None
        for tool in self.tools.values():
            if hasattr(tool, "tool_server_url"):
                self.tool_server_url = tool.tool_server_url
                break

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])

        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")

        metrics = {}
        request_id = uuid4().hex
        tools_kwargs = kwargs.get("tools_kwargs", {})

        # Initialize interaction if needed
        interaction = None
        interaction_kwargs = {}
        if self.interaction_config_file:
            interaction_kwargs = kwargs["extra_info"]["interaction_kwargs"]
            if "name" not in interaction_kwargs:
                raise ValueError("'name' key is required in interaction_kwargs")
            interaction_name = interaction_kwargs["name"]
            if interaction_name not in self.interaction_map:
                raise ValueError(
                    f"Interaction '{interaction_name}' not found. "
                    f"Available: {list(self.interaction_map.keys())}"
                )
            interaction = self.interaction_map[interaction_name]
            await interaction.start_interaction(request_id, **interaction_kwargs)

        agent_data = AgentData(
            messages=messages,
            image_data=images,
            video_data=videos,
            metrics=metrics,
            request_id=request_id,
            tools_kwargs=tools_kwargs,
            interaction=interaction,
            interaction_kwargs=interaction_kwargs,
        )

        # CaRR-specific state: reward history and turn tracking
        reward_history = [{"role": "user", "content": messages[0]["content"]}] if messages else []
        pending_tool_calls = []
        turn_idx = 0
        hit_limit = False  # True if terminated by response_length / max_turns limit
        total_tool_calls = 0
        search_count = 0
        open_count = 0
        find_count = 0
        parse_error_count = 0

        try:
            state = AgentState.PENDING
            while state != AgentState.TERMINATED:
                if state == AgentState.PENDING:
                    state = await self._handle_pending_state(agent_data, sampling_params)

                elif state == AgentState.GENERATING:
                    # Clear stale tool_calls before entering base handler.
                    # Base _handle_generating_state() may return TERMINATED before
                    # reaching extract_tool_calls (line 265), leaving old values.
                    agent_data.tool_calls = []
                    state = await self._handle_generating_state(agent_data, sampling_params)

                    # Detect if terminated by limit (before tool_calls extraction)
                    if state == AgentState.TERMINATED:
                        if (
                            len(agent_data.response_mask) >= self.response_length
                            or (self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns)
                            or (self.max_user_turns and agent_data.user_turns >= self.max_user_turns)
                        ):
                            hit_limit = True

                    # Build assistant entry for reward_history from this turn's generation
                    assistant_text = await self.loop.run_in_executor(
                        None, lambda: self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)
                    )

                    if agent_data.tool_calls:
                        # Assistant message with tool calls (only if base actually parsed them)
                        tc_entries = []
                        for i, tc in enumerate(agent_data.tool_calls):
                            tc_id = f"{request_id}_tc_{turn_idx}_{i}"
                            tc_entries.append({
                                "tool_call_id": tc_id,
                                "name": tc.name,
                                "arguments": tc.arguments,
                            })
                        pending_tool_calls = tc_entries
                        reward_history.append({
                            "role": "assistant",
                            "content": assistant_text,
                            "tool_calls": tc_entries,
                        })
                    else:
                        # Final assistant message or truncated (no tool calls parsed)
                        reward_history.append({
                            "role": "assistant",
                            "content": assistant_text,
                        })

                    # Count parse errors: complete blocks that failed JSON parse + truncated blocks
                    complete_blocks = len(re.findall(r"<tool_call>.*?</tool_call>", assistant_text, re.DOTALL))
                    incomplete_blocks = assistant_text.count("<tool_call>") - complete_blocks
                    parse_error_count += max(0, complete_blocks - len(agent_data.tool_calls)) + incomplete_blocks

                elif state == AgentState.PROCESSING_TOOLS:
                    # Count ACTUALLY EXECUTED tool calls (after max_parallel_calls slice)
                    executed = agent_data.tool_calls[:self.max_parallel_calls]
                    total_tool_calls += len(executed)
                    for tc in executed:
                        if tc.name == "browser.search":
                            search_count += 1
                        elif tc.name == "browser.open":
                            open_count += 1
                        elif tc.name == "browser.find":
                            find_count += 1

                    prev_state_len = len(agent_data.response_mask)
                    state = await self._handle_processing_tools_state_with_history(
                        agent_data, reward_history, pending_tool_calls,
                    )
                    pending_tool_calls = []
                    turn_idx += 1
                    # _handle_processing_tools_state_with_history can also TERMINATE
                    # on response_length overflow
                    if state == AgentState.TERMINATED:
                        hit_limit = True

                elif state == AgentState.INTERACTING:
                    state = await self._handle_interacting_state(agent_data)

                else:
                    logger.error("Invalid state: %s", state)
                    state = AgentState.TERMINATED

        finally:
            # Close tool session for this request
            if self.tool_server_url:
                await self.session_manager.close(request_id, self.tool_server_url)

        # Determine task_unfinished:
        # 1. Explicitly truncated by limits → always unfinished
        # 2. Empty history or last message not assistant → unfinished (fallback)
        task_unfinished = (
            hit_limit
            or len(reward_history) == 0
            or reward_history[-1].get("role") != "assistant"
        )

        # Build output (same as base class)
        response_ids = agent_data.prompt_ids[-len(agent_data.response_mask):]
        prompt_ids = agent_data.prompt_ids[:len(agent_data.prompt_ids) - len(agent_data.response_mask)]
        multi_modal_out = {}
        if agent_data.image_data is not None:
            multi_modal_out["images"] = agent_data.image_data
        if agent_data.video_data is not None:
            multi_modal_out["videos"] = agent_data.video_data

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[:self.response_length],
            response_mask=agent_data.response_mask[:self.response_length],
            multi_modal_data=multi_modal_out,
            response_logprobs=agent_data.response_logprobs[:self.response_length]
            if agent_data.response_logprobs else None,
            num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
            metrics=agent_data.metrics,
            routed_experts=agent_data.routed_experts,
            extra_fields={},
        )
        output.extra_fields.update({
            "turn_scores": agent_data.turn_scores,
            "tool_rewards": agent_data.tool_rewards,
            "messages": reward_history,
            "task_unfinished": task_unfinished,
            "tool_call_counts": total_tool_calls,
            "search_count": search_count,
            "open_count": open_count,
            "find_count": find_count,
            "hit_limit": hit_limit,
            "parse_error_count": parse_error_count,
        })
        return output

    async def _handle_processing_tools_state_with_history(
        self,
        agent_data: AgentData,
        reward_history: list,
        pending_tool_calls: list,
    ) -> AgentState:
        """Execute tool calls and append tool entries to reward_history.

        This mirrors the base _handle_processing_tools_state() logic but also
        builds the CaRR-format tool message with tool_call_id binding.
        """
        from verl.experimental.agent_loop.utils import build_gpt_oss_tool_response_text
        from verl.utils.profiler import simple_timer

        add_messages = []
        new_images_this_turn = []

        tasks = []
        tool_call_names = []
        for tool_call in agent_data.tool_calls[:self.max_parallel_calls]:
            tasks.append(self._call_tool(tool_call, agent_data.tools_kwargs, agent_data))
            tool_call_names.append(tool_call.name)

        with simple_timer("tool_calls", agent_data.metrics):
            responses = await asyncio.gather(*tasks)

        # Build CaRR-format tool content entries and regular messages
        tool_content_entries = []
        for idx, (tool_response, tool_reward, _) in enumerate(responses):
            response_text = tool_response.text or ""

            # CaRR tool entry with tool_call_id binding
            if idx < len(pending_tool_calls):
                tool_content_entries.append({
                    "tool_call_id": pending_tool_calls[idx]["tool_call_id"],
                    "output": response_text,
                })

            # Build standard message for tokenization (same as base class)
            if tool_response.image or tool_response.video:
                if not getattr(self.processor, "image_processor", None):
                    raise ValueError(
                        "Multimedia data can only be processed by `processor`, but the processor is None."
                    )
                content = []
                if tool_response.image:
                    content.append({"type": "image"})
                if tool_response.video:
                    content.append({"type": "video"})
                if tool_response.text:
                    content.append({"type": "text", "text": tool_response.text})
                message = {"role": "tool", "content": content}
            else:
                message = {"role": "tool", "content": response_text}

            add_messages.append(message)

            if tool_response.image:
                if isinstance(tool_response.image, list):
                    for img in tool_response.image:
                        if img is not None:
                            new_images_this_turn.append(img)
                elif tool_response.image is not None:
                    new_images_this_turn.append(tool_response.image)

            if tool_response.video:
                raise NotImplementedError("Multimedia type 'video' is not currently supported.")

            if tool_reward is not None:
                agent_data.tool_rewards.append(tool_reward)

        # Append CaRR tool message to reward_history
        if tool_content_entries:
            reward_history.append({
                "role": "tool",
                "content": tool_content_entries,
            })

        agent_data.messages.extend(add_messages)

        # Tokenize tool responses (same as base class)
        if self.tool_parser_name == "gpt-oss":
            tool_response_text = build_gpt_oss_tool_response_text(add_messages, tool_call_names)
            response_ids = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.encode(tool_response_text, add_special_tokens=False)
            )
        else:
            response_ids = await self.apply_chat_template(
                add_messages,
                images=new_images_this_turn,
                videos=None,
                remove_system_prompt=True,
            )

        if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
            return AgentState.TERMINATED

        if new_images_this_turn:
            if agent_data.image_data is None:
                agent_data.image_data = []
            elif not isinstance(agent_data.image_data, list):
                agent_data.image_data = [agent_data.image_data]
            for img in new_images_this_turn:
                agent_data.image_data.append(img)

        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)
        agent_data.user_turns += 1
        return AgentState.GENERATING
