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
import time
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

        rollout_custom = self.config.actor_rollout_ref.rollout.get("custom") or {}
        carr_budget = rollout_custom.get("carr_budget") or {}
        self.max_rollout_wall_time_s = carr_budget.get("max_rollout_wall_time_s")
        self.max_tool_calls = carr_budget.get("max_tool_calls")
        self.max_search_calls = carr_budget.get("max_search_calls")
        self.max_open_calls = carr_budget.get("max_open_calls")
        self.max_find_calls = carr_budget.get("max_find_calls")

    def _remaining_rollout_time_s(self, rollout_start: float) -> float | None:
        if self.max_rollout_wall_time_s is None:
            return None
        return float(self.max_rollout_wall_time_s - (time.monotonic() - rollout_start))

    def _rollout_timeout_reached(self, rollout_start: float) -> bool:
        remaining_time = self._remaining_rollout_time_s(rollout_start)
        return remaining_time is not None and remaining_time <= 0

    def _tool_budget_termination_reason(
        self,
        tool_calls,
        total_tool_calls: int,
        search_count: int,
        open_count: int,
        find_count: int,
    ) -> str | None:
        next_total = total_tool_calls
        next_search = search_count
        next_open = open_count
        next_find = find_count

        for tool_call in tool_calls[:self.max_parallel_calls]:
            next_total += 1
            if self.max_tool_calls is not None and next_total > self.max_tool_calls:
                return "tool_call_budget"

            if tool_call.name == "browser.search":
                next_search += 1
                if self.max_search_calls is not None and next_search > self.max_search_calls:
                    return "search_budget"
            elif tool_call.name == "browser.open":
                next_open += 1
                if self.max_open_calls is not None and next_open > self.max_open_calls:
                    return "open_budget"
            elif tool_call.name == "browser.find":
                next_find += 1
                if self.max_find_calls is not None and next_find > self.max_find_calls:
                    return "find_budget"

        return None

    def _extract_completed_answer_text(self, assistant_text: str) -> str | None:
        """Return a safely trimmable final answer prefix, or ``None``.

        This is intentionally conservative: it only fires when the assistant has
        already emitted the expected final-answer sections and there are no
        tool-call markers in the text.
        """
        if "<tool_call>" in assistant_text or "</tool_call>" in assistant_text:
            return None

        exact_pos = assistant_text.find("## Exact Answer")
        refs_pos = assistant_text.find("## References")
        if exact_pos == -1 or refs_pos == -1 or refs_pos <= exact_pos:
            return None

        ref_line_pattern = re.compile(r"^\s*(\[\d+\]|\d+\.\s+|[-*]\s+|https?://)")
        ref_section = assistant_text[refs_pos:]
        lines = ref_section.splitlines(keepends=True)
        if not lines:
            return None

        kept_chars = len(lines[0])  # Keep the "## References" heading.
        saw_reference_entry = False

        for line in lines[1:]:
            stripped = line.strip()
            if not saw_reference_entry:
                if not stripped:
                    kept_chars += len(line)
                    continue
                if ref_line_pattern.match(stripped):
                    saw_reference_entry = True
                    kept_chars += len(line)
                    continue
                return None

            if not stripped:
                kept_chars += len(line)
                continue
            if ref_line_pattern.match(stripped):
                kept_chars += len(line)
                continue
            break

        if not saw_reference_entry:
            return None

        return assistant_text[: refs_pos + kept_chars].rstrip()

    def _try_trim_completed_answer(self, agent_data: AgentData, assistant_text: str) -> str | None:
        """Trim the current assistant turn if a complete final answer is present.

        The trim is applied only when the decoded prefix can be mapped back to a
        prefix of the current token ids with a stable decode/encode round trip.
        """
        trimmed_text = self._extract_completed_answer_text(assistant_text)
        if trimmed_text is None:
            return None

        trimmed_ids = self.tokenizer.encode(trimmed_text, add_special_tokens=False)
        trimmed_len = len(trimmed_ids)
        current_len = len(agent_data.response_ids)
        if trimmed_len <= 0 or trimmed_len > current_len:
            return None

        current_prefix_text = self.tokenizer.decode(agent_data.response_ids[:trimmed_len], skip_special_tokens=True)
        normalize = lambda s: re.sub(r"\s+", " ", s).strip()
        if normalize(current_prefix_text) != normalize(trimmed_text):
            return None

        current_turn_logprobs = None
        if agent_data.response_logprobs:
            current_turn_logprobs = agent_data.response_logprobs[-current_len:]

        prefix_prompt_len = len(agent_data.prompt_ids) - current_len
        agent_data.response_ids = agent_data.response_ids[:trimmed_len]
        agent_data.prompt_ids = agent_data.prompt_ids[:prefix_prompt_len] + agent_data.response_ids
        agent_data.response_mask = agent_data.response_mask[:-current_len] + [1] * trimmed_len
        if current_turn_logprobs is not None:
            agent_data.response_logprobs = agent_data.response_logprobs[:-current_len] + current_turn_logprobs[:trimmed_len]

        return trimmed_text

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
        hit_budget = False  # True if terminated by rollout wall-clock / tool budgets
        termination_reason = None
        total_tool_calls = 0
        search_count = 0
        open_count = 0
        find_count = 0
        parse_error_count = 0
        content_early_stopped = False
        rollout_start = time.monotonic()

        try:
            state = AgentState.PENDING
            while state != AgentState.TERMINATED:
                if state in {AgentState.GENERATING, AgentState.PROCESSING_TOOLS, AgentState.INTERACTING}:
                    if self._rollout_timeout_reached(rollout_start):
                        hit_budget = True
                        termination_reason = "rollout_timeout"
                        state = AgentState.TERMINATED
                        continue

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
                        if len(agent_data.response_mask) >= self.response_length:
                            hit_limit = True
                            termination_reason = "response_limit"
                        elif self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
                            hit_limit = True
                            termination_reason = "assistant_turn_limit"
                        elif self.max_user_turns and agent_data.user_turns >= self.max_user_turns:
                            hit_limit = True
                            termination_reason = "user_turn_limit"

                    # Build assistant entry for reward_history from this turn's generation
                    assistant_text = await self.loop.run_in_executor(
                        None, lambda: self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)
                    )

                    if not agent_data.tool_calls:
                        trimmed_text = self._try_trim_completed_answer(agent_data, assistant_text)
                        if trimmed_text is not None:
                            assistant_text = trimmed_text
                            content_early_stopped = True
                            # If we already have a complete final answer, do not
                            # treat this turn as an unfinished hard-limit failure.
                            hit_limit = False
                            termination_reason = None

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
                    budget_reason = self._tool_budget_termination_reason(
                        agent_data.tool_calls,
                        total_tool_calls,
                        search_count,
                        open_count,
                        find_count,
                    )
                    if budget_reason is not None:
                        hit_budget = True
                        termination_reason = budget_reason
                        state = AgentState.TERMINATED
                        continue

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
                        termination_reason = "response_limit"

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
        # 2. Explicitly terminated by rollout/tool budgets → always unfinished
        # 3. Empty history or last message not assistant → unfinished (fallback)
        task_unfinished = (
            hit_limit
            or hit_budget
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
            "hit_budget": hit_budget,
            "parse_error_count": parse_error_count,
            "termination_reason": termination_reason,
            "termination_response_limit": 1.0 if termination_reason == "response_limit" else 0.0,
            "termination_assistant_turn_limit": 1.0 if termination_reason == "assistant_turn_limit" else 0.0,
            "termination_user_turn_limit": 1.0 if termination_reason == "user_turn_limit" else 0.0,
            "termination_rollout_timeout": 1.0 if termination_reason == "rollout_timeout" else 0.0,
            "termination_tool_call_budget": 1.0 if termination_reason == "tool_call_budget" else 0.0,
            "termination_search_budget": 1.0 if termination_reason == "search_budget" else 0.0,
            "termination_open_budget": 1.0 if termination_reason == "open_budget" else 0.0,
            "termination_find_budget": 1.0 if termination_reason == "find_budget" else 0.0,
            "content_early_stopped": content_early_stopped,
            "rollout_elapsed_s": float(time.monotonic() - rollout_start),
            "response_length": float(len(agent_data.response_mask)),
            "response_length_max": float(self.response_length),
            "response_length_ratio": float(len(agent_data.response_mask) / self.response_length)
            if self.response_length
            else 0.0,
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


@register("carr_async_partial_tool_agent")
class CaRRAsyncPartialToolAgentLoop(CaRRToolAgentLoop):
    """CaRR async agent loop with resumable assistant turns and budget accounting."""

    def __init__(self, trainer_config, server_manager, tokenizer, processor, **kwargs):
        super().__init__(trainer_config, server_manager, tokenizer, processor, **kwargs)
        rollout_custom = self.config.actor_rollout_ref.rollout.get("custom") or {}
        carr_budget = rollout_custom.get("carr_budget") or {}
        self.enable_partial_rollout = trainer_config.config.async_training.get("partial_rollout", False)
        self.max_real_rollout_wall_time_s = carr_budget.get("max_real_rollout_wall_time_s")
        self.max_param_span = carr_budget.get("max_param_span")

    async def _init_async_agent_data(self, kwargs: dict[str, Any], param_version: int) -> AgentData:
        messages = list(kwargs["raw_prompt"])
        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")
        metrics = {}
        request_id = uuid4().hex
        tools_kwargs = kwargs.get("tools_kwargs", {})

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
        agent_data.extra_fields["carr_async_state"] = {
            "reward_history": [{"role": "user", "content": messages[0]["content"]}] if messages else [],
            "pending_tool_calls": [],
            "turn_idx": 0,
            "hit_limit": False,
            "hit_budget": False,
            "termination_reason": None,
            "total_tool_calls": 0,
            "search_count": 0,
            "open_count": 0,
            "find_count": 0,
            "parse_error_count": 0,
            "content_early_stopped": False,
            "real_rollout_start_s": time.monotonic(),
            "active_rollout_wall_time_s": 0.0,
            "assistant_turn_in_progress": False,
            "param_version_start": param_version,
            "last_param_version": param_version,
        }
        return agent_data

    def _restore_from_output(self, output: AgentLoopOutput) -> tuple[AgentData, AgentState]:
        agent_data = output.extra_fields.get("agent_data")
        agent_state = output.extra_fields.get("agent_state")
        if agent_data is None or agent_state is None:
            raise ValueError(f"Unexpected async CaRR partial output: agent_data={agent_data}, agent_state={agent_state}")
        return agent_data, agent_state

    def _get_carr_async_state(self, agent_data: AgentData) -> dict[str, Any]:
        carr_state = agent_data.extra_fields.get("carr_async_state")
        if carr_state is None:
            raise ValueError("Missing carr_async_state in async CaRR agent_data")
        return carr_state

    def _current_active_elapsed_s(self, carr_state: dict[str, Any], run_started_s: float) -> float:
        return float(carr_state.get("active_rollout_wall_time_s", 0.0) + max(0.0, time.monotonic() - run_started_s))

    def _current_real_elapsed_s(self, carr_state: dict[str, Any]) -> float:
        return float(max(0.0, time.monotonic() - carr_state.get("real_rollout_start_s", time.monotonic())))

    def _async_budget_termination_reason(
        self, carr_state: dict[str, Any], run_started_s: float, param_version: int
    ) -> str | None:
        if self.max_rollout_wall_time_s is not None and self._current_active_elapsed_s(carr_state, run_started_s) >= float(
            self.max_rollout_wall_time_s
        ):
            return "rollout_timeout"
        if self.max_real_rollout_wall_time_s is not None and self._current_real_elapsed_s(carr_state) >= float(
            self.max_real_rollout_wall_time_s
        ):
            return "real_rollout_timeout"
        if self.max_param_span is not None and param_version - int(carr_state.get("param_version_start", param_version)) > int(
            self.max_param_span
        ):
            return "max_param_span"
        return None

    async def _handle_generating_state_partial(
        self, agent_data: AgentData, sampling_params: dict[str, Any]
    ) -> AgentState:
        from verl.utils.profiler import simple_timer

        carr_state = self._get_carr_async_state(agent_data)
        if not carr_state.get("assistant_turn_in_progress", False):
            agent_data.response_ids = []
            agent_data.tool_calls = []
            carr_state["assistant_turn_in_progress"] = True

        routed_experts = None
        with simple_timer("generate_sequences", agent_data.metrics):
            if self.enable_partial_rollout:
                response_ids, log_probs, is_cancel = await self.server_manager.generate_for_partial(
                    request_id=agent_data.request_id,
                    prompt_ids=agent_data.prompt_ids,
                    sampling_params=sampling_params,
                    image_data=agent_data.image_data,
                    video_data=agent_data.video_data,
                )
            else:
                output = await self.server_manager.generate(
                    request_id=agent_data.request_id,
                    prompt_ids=agent_data.prompt_ids,
                    sampling_params=sampling_params,
                    image_data=agent_data.image_data,
                    video_data=agent_data.video_data,
                )
                if agent_data.metrics.get("num_preempted") is None:
                    agent_data.metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
                else:
                    agent_data.metrics["num_preempted"] += output.num_preempted if output.num_preempted is not None else 0
                response_ids = output.token_ids
                log_probs = output.log_probs
                routed_experts = output.routed_experts
                is_cancel = False

        response_ids = list(response_ids)
        if response_ids:
            agent_data.response_ids += response_ids
            agent_data.prompt_ids += response_ids
            agent_data.response_mask += [1] * len(response_ids)
            if log_probs:
                agent_data.response_logprobs += list(log_probs)

        if routed_experts is not None:
            agent_data.routed_experts = routed_experts

        if is_cancel:
            if len(agent_data.response_mask) >= self.response_length:
                carr_state["assistant_turn_in_progress"] = False
                agent_data.assistant_turns += 1
                return AgentState.TERMINATED
            return AgentState.GENERATING

        carr_state["assistant_turn_in_progress"] = False
        agent_data.assistant_turns += 1

        if len(agent_data.response_mask) >= self.response_length:
            return AgentState.TERMINATED
        if self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
            return AgentState.TERMINATED
        if self.max_user_turns and agent_data.user_turns >= self.max_user_turns:
            return AgentState.TERMINATED

        _, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(agent_data.response_ids)

        if self.interaction_config_file:
            assistant_message = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)
            )
            agent_data.messages.extend([{"role": "assistant", "content": assistant_message}])

        if agent_data.tool_calls:
            return AgentState.PROCESSING_TOOLS
        if self.interaction_config_file:
            return AgentState.INTERACTING
        return AgentState.TERMINATED

    def _build_completed_output(self, agent_data: AgentData, param_version: int, run_started_s: float) -> AgentLoopOutput:
        carr_state = self._get_carr_async_state(agent_data)
        response_ids = agent_data.prompt_ids[-len(agent_data.response_mask):]
        prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - len(agent_data.response_mask)]
        multi_modal_out = {}
        if agent_data.image_data is not None:
            multi_modal_out["images"] = agent_data.image_data
        if agent_data.video_data is not None:
            multi_modal_out["videos"] = agent_data.video_data

        active_elapsed_s = self._current_active_elapsed_s(carr_state, run_started_s)
        real_elapsed_s = self._current_real_elapsed_s(carr_state)
        termination_reason = carr_state.get("termination_reason")
        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=agent_data.response_mask[: self.response_length],
            multi_modal_data=multi_modal_out,
            response_logprobs=agent_data.response_logprobs[: self.response_length]
            if agent_data.response_logprobs
            else None,
            num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
            metrics=agent_data.metrics,
            routed_experts=agent_data.routed_experts,
            extra_fields={},
        )
        reward_history = carr_state.get("reward_history", [])
        task_unfinished = (
            carr_state.get("hit_limit", False)
            or carr_state.get("hit_budget", False)
            or len(reward_history) == 0
            or reward_history[-1].get("role") != "assistant"
        )
        output.extra_fields.update(
            {
                "turn_scores": agent_data.turn_scores,
                "tool_rewards": agent_data.tool_rewards,
                "messages": reward_history,
                "task_unfinished": task_unfinished,
                "tool_call_counts": carr_state.get("total_tool_calls", 0),
                "search_count": carr_state.get("search_count", 0),
                "open_count": carr_state.get("open_count", 0),
                "find_count": carr_state.get("find_count", 0),
                "hit_limit": carr_state.get("hit_limit", False),
                "hit_budget": carr_state.get("hit_budget", False),
                "parse_error_count": carr_state.get("parse_error_count", 0),
                "termination_reason": termination_reason,
                "termination_response_limit": 1.0 if termination_reason == "response_limit" else 0.0,
                "termination_assistant_turn_limit": 1.0 if termination_reason == "assistant_turn_limit" else 0.0,
                "termination_user_turn_limit": 1.0 if termination_reason == "user_turn_limit" else 0.0,
                "termination_rollout_timeout": 1.0 if termination_reason == "rollout_timeout" else 0.0,
                "termination_tool_call_budget": 1.0 if termination_reason == "tool_call_budget" else 0.0,
                "termination_search_budget": 1.0 if termination_reason == "search_budget" else 0.0,
                "termination_open_budget": 1.0 if termination_reason == "open_budget" else 0.0,
                "termination_find_budget": 1.0 if termination_reason == "find_budget" else 0.0,
                "termination_real_rollout_timeout": 1.0 if termination_reason == "real_rollout_timeout" else 0.0,
                "termination_max_param_span": 1.0 if termination_reason == "max_param_span" else 0.0,
                "content_early_stopped": carr_state.get("content_early_stopped", False),
                "rollout_elapsed_s": active_elapsed_s,
                "active_rollout_elapsed_s": active_elapsed_s,
                "real_rollout_elapsed_s": real_elapsed_s,
                "param_version_span": float(param_version - int(carr_state.get("param_version_start", param_version))),
                "response_length": float(len(agent_data.response_mask)),
                "response_length_max": float(self.response_length),
                "response_length_ratio": float(len(agent_data.response_mask) / self.response_length)
                if self.response_length
                else 0.0,
                "is_cancel": False,
                "param_version_start": carr_state.get("param_version_start", param_version),
                "param_version_end": param_version,
            }
        )
        return output

    def _build_cancelled_output(self, agent_data: AgentData, state: AgentState) -> AgentLoopOutput:
        return AgentLoopOutput(
            prompt_ids=[],
            response_ids=[],
            response_mask=[],
            multi_modal_data={},
            response_logprobs=None,
            num_turns=0,
            metrics=agent_data.metrics,
            extra_fields={
                "is_cancel": True,
                "agent_data": agent_data,
                "agent_state": state,
            },
        )

    @rollout_trace_op
    async def run(
        self, sampling_params: dict[str, Any], *, cancellation_event: asyncio.Event = None, **kwargs
    ) -> AgentLoopOutput:
        param_version = kwargs.get("param_version", 0)
        output: AgentLoopOutput | None = kwargs.get("output")
        should_close_session = False

        if output and not output.extra_fields.get("is_cancel", False):
            return output

        if output and output.extra_fields.get("is_cancel", False):
            agent_data, state = self._restore_from_output(output)
        else:
            agent_data = await self._init_async_agent_data(kwargs, param_version)
            state = AgentState.PENDING

        carr_state = self._get_carr_async_state(agent_data)
        carr_state["last_param_version"] = param_version
        run_started_s = time.monotonic()

        try:
            while state != AgentState.TERMINATED:
                if state in {AgentState.GENERATING, AgentState.PROCESSING_TOOLS, AgentState.INTERACTING}:
                    budget_reason = self._async_budget_termination_reason(carr_state, run_started_s, param_version)
                    if budget_reason is not None:
                        carr_state["hit_budget"] = True
                        carr_state["termination_reason"] = budget_reason
                        state = AgentState.TERMINATED
                        continue

                if cancellation_event and cancellation_event.is_set():
                    return self._build_cancelled_output(agent_data, state)

                if state == AgentState.PENDING:
                    state = await self._handle_pending_state(agent_data, sampling_params)

                elif state == AgentState.GENERATING:
                    agent_data.tool_calls = []
                    state = await self._handle_generating_state_partial(agent_data, sampling_params)

                    if carr_state.get("assistant_turn_in_progress", False):
                        continue

                    if state == AgentState.TERMINATED:
                        if len(agent_data.response_mask) >= self.response_length:
                            carr_state["hit_limit"] = True
                            carr_state["termination_reason"] = "response_limit"
                        elif self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
                            carr_state["hit_limit"] = True
                            carr_state["termination_reason"] = "assistant_turn_limit"
                        elif self.max_user_turns and agent_data.user_turns >= self.max_user_turns:
                            carr_state["hit_limit"] = True
                            carr_state["termination_reason"] = "user_turn_limit"

                    assistant_text = await self.loop.run_in_executor(
                        None, lambda: self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)
                    )

                    if not agent_data.tool_calls:
                        trimmed_text = self._try_trim_completed_answer(agent_data, assistant_text)
                        if trimmed_text is not None:
                            assistant_text = trimmed_text
                            carr_state["content_early_stopped"] = True
                            carr_state["hit_limit"] = False
                            carr_state["termination_reason"] = None

                    if agent_data.tool_calls:
                        tc_entries = []
                        turn_idx = int(carr_state.get("turn_idx", 0))
                        for i, tc in enumerate(agent_data.tool_calls):
                            tc_id = f"{agent_data.request_id}_tc_{turn_idx}_{i}"
                            tc_entries.append(
                                {
                                    "tool_call_id": tc_id,
                                    "name": tc.name,
                                    "arguments": tc.arguments,
                                }
                            )
                        carr_state["pending_tool_calls"] = tc_entries
                        carr_state["reward_history"].append(
                            {
                                "role": "assistant",
                                "content": assistant_text,
                                "tool_calls": tc_entries,
                            }
                        )
                    else:
                        carr_state["reward_history"].append(
                            {
                                "role": "assistant",
                                "content": assistant_text,
                            }
                        )

                    complete_blocks = len(re.findall(r"<tool_call>.*?</tool_call>", assistant_text, re.DOTALL))
                    incomplete_blocks = assistant_text.count("<tool_call>") - complete_blocks
                    carr_state["parse_error_count"] += max(0, complete_blocks - len(agent_data.tool_calls)) + incomplete_blocks

                elif state == AgentState.PROCESSING_TOOLS:
                    budget_reason = self._tool_budget_termination_reason(
                        agent_data.tool_calls,
                        int(carr_state.get("total_tool_calls", 0)),
                        int(carr_state.get("search_count", 0)),
                        int(carr_state.get("open_count", 0)),
                        int(carr_state.get("find_count", 0)),
                    )
                    if budget_reason is not None:
                        carr_state["hit_budget"] = True
                        carr_state["termination_reason"] = budget_reason
                        state = AgentState.TERMINATED
                        continue

                    executed = agent_data.tool_calls[: self.max_parallel_calls]
                    carr_state["total_tool_calls"] += len(executed)
                    for tc in executed:
                        if tc.name == "browser.search":
                            carr_state["search_count"] += 1
                        elif tc.name == "browser.open":
                            carr_state["open_count"] += 1
                        elif tc.name == "browser.find":
                            carr_state["find_count"] += 1

                    state = await self._handle_processing_tools_state_with_history(
                        agent_data,
                        carr_state["reward_history"],
                        carr_state.get("pending_tool_calls", []),
                    )
                    carr_state["pending_tool_calls"] = []
                    carr_state["turn_idx"] += 1
                    if state == AgentState.TERMINATED:
                        carr_state["hit_limit"] = True
                        carr_state["termination_reason"] = "response_limit"

                elif state == AgentState.INTERACTING:
                    state = await self._handle_interacting_state(agent_data)

                else:
                    logger.error("Invalid state: %s", state)
                    state = AgentState.TERMINATED

            should_close_session = True
            return self._build_completed_output(agent_data, param_version, run_started_s)
        except Exception:
            should_close_session = True
            raise
        finally:
            carr_state["active_rollout_wall_time_s"] = self._current_active_elapsed_s(carr_state, run_started_s)
            if should_close_session and self.tool_server_url:
                await self.session_manager.close(agent_data.request_id, self.tool_server_url)
