# CaRR Agent Loop `run()` 方法详解

> 本文档解析 `examples/carr_deepsearch/tools/carr_agent_loop.py` 中 `CaRRToolAgentLoop.run()` 的完整执行流程，重点说明各状态的处理逻辑、终止条件分支，以及 `early_stopped` 与 `natural completion` 在实现层面的真实区别。
>
> 涉及文件：
> - [carr_agent_loop.py](../tools/carr_agent_loop.py)（CaRR 自定义 agent loop）
> - [verl/experimental/agent_loop/tool_agent_loop.py](../../../verl/experimental/agent_loop/tool_agent_loop.py)（基类 `ToolAgentLoop`）
> - [verl/experimental/agent_loop/agent_loop.py](../../../verl/experimental/agent_loop/agent_loop.py)（`AgentLoopBase` / `AgentData` / `AgentState`）

---

## 1. 概览

`CaRRToolAgentLoop` 继承自 verl 的 `ToolAgentLoop`，是单条 trajectory 的执行器。每个样本一次 `run()` 调用，完成「**一段多轮 agent 交互**」：模型生成 → 工具调用 → 工具返回 → 模型生成 → ... → 终止。

继承的同时，CaRR 在基类 `run()` 之上覆盖加了三件事：

1. **CaRR-format `reward_history`**：维护 `[user, assistant, tool, assistant, ...]` 格式的对话历史，发给奖励服务器做 outcome / rubric 评分
2. **细粒度 budget**：除了基类的 `response_length` / `assistant_turn_limit`，还增加 `tool_call / search / open / find / rollout_wall_time` 五种 budget
3. **early-stop trim 机制**：在生成结果中识别"已写完最终答案 + 引用列表"的尾部，把多余内容裁掉，让"撞到 response_limit 但答案已写完"的样本被视为成功完成

输出是 `AgentLoopOutput`，里面通过 `extra_fields` 携带二十多个 metrics 字段（见第 7 节）。

---

## 2. 状态机概览

`AgentState` 枚举（定义在 [`agent_loop.py`](../../../verl/experimental/agent_loop/agent_loop.py)）共 5 个状态：

```
PENDING → GENERATING → PROCESSING_TOOLS → GENERATING → ... → TERMINATED
                    ↘ INTERACTING ↗
                    ↘ TERMINATED
```

| 状态 | 含义 | CaRR 中谁触发 |
|------|------|--------------|
| `PENDING` | 准备 prompt | 入口默认状态 |
| `GENERATING` | 调用 LLM 生成 assistant turn | `_handle_generating_state` (基类) |
| `PROCESSING_TOOLS` | 并行执行解析出的 tool_call | `_handle_processing_tools_state_with_history` (CaRR 重写) |
| `INTERACTING` | 多轮交互（环境反馈非工具响应） | CaRR 没用，`interaction_config_file=None` |
| `TERMINATED` | 终止，跳出 while 循环 | 由各种 limit/budget 触发 |

CaRR 在 deep search 任务里**不使用 `INTERACTING`** —— 因为环境反馈就是 tool_response 本身，不需要额外的 interaction 角色。

---

## 3. `run()` 主循环逐段解析

`run()` 大体可分成 6 段：

### 3.1 输入处理与请求初始化（L257-293）

```python
async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
    messages = list(kwargs["raw_prompt"])
    multi_modal_data = await self.process_vision_info(messages)
    images = multi_modal_data.get("images")
    videos = multi_modal_data.get("videos")

    metrics = {}
    request_id = uuid4().hex
    tools_kwargs = kwargs.get("tools_kwargs", {})

    # Initialize interaction if needed (CaRR 不使用，一般跳过)
    interaction = None
    interaction_kwargs = {}
    if self.interaction_config_file:
        ...

    agent_data = AgentData(
        messages=messages, image_data=images, video_data=videos,
        metrics=metrics, request_id=request_id, tools_kwargs=tools_kwargs,
        interaction=interaction, interaction_kwargs=interaction_kwargs,
    )
```

**关键点**：
- `request_id` 是该 trajectory 的唯一 ID，后续 session_manager / tool_call_id 都基于它
- `tools_kwargs` 来自 parquet 数据的 `extra_info.tools_kwargs`，包含 `search_forbidden_strs`（反作弊 n-gram 过滤）等
- `AgentData` 是跨状态共享的可变状态容器，包含 `prompt_ids` / `response_ids` / `response_mask` / `tool_calls` / `assistant_turns` 等

### 3.2 CaRR-specific 状态初始化（L296-308）

```python
reward_history = [{"role": "user", "content": messages[0]["content"]}] if messages else []
pending_tool_calls = []
turn_idx = 0
hit_limit = False         # response_length / max_turns 超限
hit_budget = False        # rollout wall-clock / 工具 budget 超限
termination_reason = None
total_tool_calls = 0
search_count = 0
open_count = 0
find_count = 0
parse_error_count = 0
content_early_stopped = False
rollout_start = time.monotonic()
```

**`hit_limit` vs `hit_budget` 的语义区分**：

| 标志 | 触发条件 | 终止原因取值 |
|------|---------|------------|
| `hit_limit` | 撞到了"硬上限" | `response_limit` / `assistant_turn_limit` / `user_turn_limit` |
| `hit_budget` | 撞到了"额度上限" | `rollout_timeout` / `tool_call_budget` / `search_budget` / `open_budget` / `find_budget` |

这两个标志后面会喂给 `_build_unfinished_reason_flags`，决定 `task_unfinished` 是不是 True。

### 3.3 主循环：状态机驱动（L310-432）

```python
try:
    state = AgentState.PENDING
    while state != AgentState.TERMINATED:
        # (a) Wall-time 检查（每轮迭代开头）
        if state in {AgentState.GENERATING, AgentState.PROCESSING_TOOLS, AgentState.INTERACTING}:
            if self._rollout_timeout_reached(rollout_start):
                hit_budget = True
                termination_reason = "rollout_timeout"
                state = AgentState.TERMINATED
                continue

        # (b) 各状态分发
        if state == AgentState.PENDING:
            state = await self._handle_pending_state(agent_data, sampling_params)
        elif state == AgentState.GENERATING:
            ...   # 见第 4.2 节
        elif state == AgentState.PROCESSING_TOOLS:
            ...   # 见第 4.3 节
        elif state == AgentState.INTERACTING:
            state = await self._handle_interacting_state(agent_data)
        else:
            logger.error("Invalid state: %s", state)
            state = AgentState.TERMINATED
finally:
    # (c) 不论怎么终止，都要关闭工具 session
    if self.tool_server_url:
        await self.session_manager.close(request_id, self.tool_server_url)
```

**循环开头的 wall-time 检查（关键）**：

每轮迭代开头都先检查 `_rollout_timeout_reached(rollout_start)`，且**在分发到任何状态之前**就检查。这意味着 wall-time 超时**优先级高于其他终止条件**——只要超时，无论当前在哪个状态，都立刻 TERMINATE。

这个检查只对 `GENERATING / PROCESSING_TOOLS / INTERACTING` 生效，PENDING 状态不检查（因为 PENDING 还没真正开始计时意义上的工作）。

`finally` 块保证**任何路径退出循环都会关闭 session**，避免泄漏。

### 3.4 终止后：构造 unfinished flags（L434-440）

```python
unfinished_flags = self._build_unfinished_reason_flags(
    hit_limit=hit_limit,
    hit_budget=hit_budget,
    termination_reason=termination_reason,
    reward_history=reward_history,
    content_early_stopped=content_early_stopped,
)
```

这一步把循环中累积的标志位翻译成最终的 `task_unfinished` / `completion_finished_*` 等语义。详见第 5 节。

### 3.5 构造 `AgentLoopOutput`（L442-462）

```python
response_ids = agent_data.prompt_ids[-len(agent_data.response_mask):]
prompt_ids = agent_data.prompt_ids[:len(agent_data.prompt_ids) - len(agent_data.response_mask)]

output = AgentLoopOutput(
    prompt_ids=prompt_ids,
    response_ids=response_ids[:self.response_length],
    response_mask=agent_data.response_mask[:self.response_length],
    multi_modal_data=multi_modal_out,
    response_logprobs=...,
    num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
    metrics=agent_data.metrics,
    routed_experts=agent_data.routed_experts,
    extra_fields={},
)
```

**注意**：`response_ids` / `response_mask` 都被截到 `self.response_length`，这是兜底防御（理论上循环里也保证不会超过）。

### 3.6 写入 `extra_fields`（L463-493）

把 23 个 CaRR-specific metrics 都塞进 `extra_fields`，下游 reward manager / metrics 收集器靠这些字段做评测分析。详见第 7 节。

---

## 4. 各状态处理详解

### 4.1 `PENDING` 状态

```python
# verl/experimental/agent_loop/tool_agent_loop.py
async def _handle_pending_state(self, agent_data, sampling_params) -> AgentState:
    prompt_ids = await self.apply_chat_template(
        agent_data.messages,
        tools=self.tool_schemas,
        images=agent_data.image_data,
        videos=agent_data.video_data,
    )
    agent_data.prompt_ids = prompt_ids
    return AgentState.GENERATING
```

**职责**：
- 把 `messages`（含 system prompt 和 user query）+ `tool_schemas` 渲染成 token IDs
- 写到 `agent_data.prompt_ids`
- 转入 `GENERATING`

**只执行一次**（trajectory 开头）。

### 4.2 `GENERATING` 状态（CaRR 在基类基础上扩展）

CaRR 没有重写 `_handle_generating_state`，而是**在调用基类后做了大量 post-processing**。完整逻辑见 L323-383：

```python
elif state == AgentState.GENERATING:
    # (1) 清掉旧的 tool_calls，避免 stale 数据
    agent_data.tool_calls = []
    state = await self._handle_generating_state(agent_data, sampling_params)

    # (2) 检测是否被基类的 limit 终止
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

    # (3) 解码当前 turn 生成的文本
    assistant_text = await self.loop.run_in_executor(
        None, lambda: self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)
    )

    # (4) 关键：early-stop trim 检测
    if not agent_data.tool_calls:
        trimmed_text = self._try_trim_completed_answer(agent_data, assistant_text)
        if trimmed_text is not None:
            assistant_text = trimmed_text
            content_early_stopped = True
            # 关键：撤销 hit_limit 判定
            hit_limit = False
            termination_reason = None

    # (5) 把 assistant turn 写入 reward_history
    if agent_data.tool_calls:
        # 有 tool_call → 含 tool_call_id 的 message
        tc_entries = [{"tool_call_id": ..., "name": ..., "arguments": ...} for tc in ...]
        pending_tool_calls = tc_entries
        reward_history.append({"role": "assistant", "content": assistant_text, "tool_calls": tc_entries})
    else:
        # 无 tool_call → 普通 final assistant message
        reward_history.append({"role": "assistant", "content": assistant_text})

    # (6) parse error 计数
    complete_blocks = len(re.findall(r"<tool_call>.*?</tool_call>", assistant_text, re.DOTALL))
    incomplete_blocks = assistant_text.count("<tool_call>") - complete_blocks
    parse_error_count += max(0, complete_blocks - len(agent_data.tool_calls)) + incomplete_blocks
```

#### 基类 `_handle_generating_state` 的完整执行顺序

要理解 CaRR 的 (1) (2) (4) 三步为什么要这么写，必须先看清楚基类内部的**真实执行顺序**。完整代码见 [tool_agent_loop.py:252-308](../../../verl/experimental/agent_loop/tool_agent_loop.py#L252-L308)：

```python
async def _handle_generating_state(self, agent_data, sampling_params, ignore_termination=False):
    # ───────── Phase A: LLM 生成 ─────────
    output = await self.server_manager.generate(
        request_id=agent_data.request_id,
        prompt_ids=agent_data.prompt_ids,
        sampling_params=sampling_params,
        ...
    )

    # ───────── Phase B: 写入生成结果 ─────────
    agent_data.assistant_turns += 1
    agent_data.response_ids = output.token_ids
    agent_data.prompt_ids += agent_data.response_ids
    agent_data.response_mask += [1] * len(agent_data.response_ids)
    if output.log_probs:
        agent_data.response_logprobs += output.log_probs

    # ───────── Phase C: 三个硬上限检查（关键！）─────────
    if not ignore_termination and len(agent_data.response_mask) >= self.response_length:
        return AgentState.TERMINATED                                       # ★ 出口 1
    if self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
        return AgentState.TERMINATED                                       # ★ 出口 2
    if self.max_user_turns and agent_data.user_turns >= self.max_user_turns:
        return AgentState.TERMINATED                                       # ★ 出口 3

    # ───────── Phase D: 解析工具调用 ─────────
    _, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(agent_data.response_ids)

    # ───────── Phase E: interaction 路径（CaRR 不走）─────────
    if self.interaction_config_file:
        assistant_message = self.tokenizer.decode(...)
        add_messages.append({"role": "assistant", "content": assistant_message})
        agent_data.messages.extend(add_messages)

    # ───────── Phase F: 根据 tool_calls 决定下一个状态 ─────────
    if agent_data.tool_calls:
        return AgentState.PROCESSING_TOOLS                                 # ★ 出口 4
    elif self.interaction_config_file:
        return AgentState.INTERACTING                                      # ★ 出口 5
    else:
        return AgentState.TERMINATED                                       # ★ 出口 6
```

注意基类**总共有 6 个返回出口**，按是否解析过 tool_calls 可以分成两组：

| 出口 | 触发条件 | `agent_data.tool_calls` 是什么状态 |
|------|---------|----------------------------------|
| ★1 response_limit | 生成完后 token 数超上限 | **未被赋值，保留上一轮的旧值** |
| ★2 assistant_turn_limit | assistant 轮数超上限 | **未被赋值，保留上一轮的旧值** |
| ★3 user_turn_limit | user 轮数超上限 | **未被赋值，保留上一轮的旧值** |
| ★4 PROCESSING_TOOLS | 解析到了工具调用 | 刚被 Phase D 重新赋值 |
| ★5 INTERACTING | 没工具调用 + 有 interaction | 刚被 Phase D 重新赋值（空列表） |
| ★6 TERMINATED（自然结束） | 没工具调用 + 无 interaction | 刚被 Phase D 重新赋值（空列表） |

**关键观察**：Phase C 的硬上限检查**早于** Phase D 的 tool_calls 解析。也就是说，**只要从 ★1/★2/★3 出口走，`agent_data.tool_calls` 这个字段是不会被刷新的**——它会保留上一次调用时留下的旧值。

#### 这造成了什么问题？

`agent_data` 是跨 turn 共享的同一个对象。设想以下场景：

```
Turn 1: GENERATING
  - LLM 生成: <think>...</think><tool_call>{search('X')}</tool_call>
  - Phase C: 没超限
  - Phase D: tool_calls = [search('X')]   ← 写入了
  - Phase F: 出口 ★4 → 返回 PROCESSING_TOOLS

Turn 1: PROCESSING_TOOLS
  - 执行 search('X')，写回 tool_response
  - 返回 GENERATING

Turn 2: GENERATING
  - LLM 生成: 又一段长长的 thinking + 完整的 ## Exact Answer + ## References
  - Phase B: response_mask 累加后达到 61440
  - Phase C: response_mask >= response_length → 返回 ★1（TERMINATED）
  - ⚠️ Phase D 没被执行！
  - ⚠️ agent_data.tool_calls 仍然是 Turn 1 留下的 [search('X')]
```

回到 CaRR 主循环 [L323-355](../tools/carr_agent_loop.py#L323-L355)：

```python
elif state == AgentState.GENERATING:
    # (1) 清掉 stale tool_calls          ← 如果不清，下面会出大事
    agent_data.tool_calls = []
    state = await self._handle_generating_state(agent_data, sampling_params)

    # (2) 检测是否被 limit 终止
    if state == AgentState.TERMINATED:
        if len(agent_data.response_mask) >= self.response_length:
            hit_limit = True
            termination_reason = "response_limit"
        ...

    # (3) 解码 assistant_text
    assistant_text = self.tokenizer.decode(agent_data.response_ids, ...)

    # (4) Trim 检测：判定逻辑是 "if not agent_data.tool_calls"
    if not agent_data.tool_calls:                  # ← 这里依赖 tool_calls 为空
        trimmed_text = self._try_trim_completed_answer(agent_data, assistant_text)
        if trimmed_text is not None:
            content_early_stopped = True
            hit_limit = False
            termination_reason = None
```

**如果不在 (1) 处主动清空 `agent_data.tool_calls`**，会发生：

- Turn 2 因 response_limit 被截断时，基类没刷新 `tool_calls`
- 但实际上 Turn 2 的 `response_ids` 里**根本没有 `<tool_call>` 块**（模型这次写的是答案）
- 然而 `agent_data.tool_calls` 残留着 Turn 1 的 `[search('X')]`
- (4) 处的 `if not agent_data.tool_calls` 判定为 False（因为列表非空）
- **trim 检测被跳过** → 模型明明写完了答案，却被打成 unfinished

更糟的是，主循环的 [L357-372](../tools/carr_agent_loop.py#L357-L372) 之后会基于这个残留的 `tool_calls` 写 `reward_history`：

```python
if agent_data.tool_calls:
    # 误以为模型这一轮调用了工具
    tc_entries = [{"tool_call_id": ..., "name": "search", ...}]
    reward_history.append({"role": "assistant", "content": assistant_text, "tool_calls": tc_entries})
```

→ 把 Turn 1 的旧 tool_call 错误地塞到 Turn 2 的 reward_history 里，最后送给奖励服务器评分时数据是脏的。

**所以 CaRR 第 (1) 步的 `agent_data.tool_calls = []` 不是冗余防御，是必须的修正**——它把基类"limit 出口不刷新 tool_calls"这个隐含 bug 兜住了。

#### 第 (2) 步：复读基类的 limit 判定

基类返回 TERMINATED 后**只告诉你"终止了"**，不告诉你"为什么终止"——出口 ★1/★2/★3/★6 都返回同样的 TERMINATED。CaRR 需要自己反推哪个 limit 触发了：

```python
if state == AgentState.TERMINATED:
    if len(agent_data.response_mask) >= self.response_length:
        hit_limit = True
        termination_reason = "response_limit"     # ← 推断是 ★1
    elif self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
        hit_limit = True
        termination_reason = "assistant_turn_limit"   # ← 推断是 ★2
    elif self.max_user_turns and agent_data.user_turns >= self.max_user_turns:
        hit_limit = True
        termination_reason = "user_turn_limit"        # ← 推断是 ★3
    # 注意：如果三个 if 都没命中，说明走的是 ★6（自然结束），
    #       此时 hit_limit 保持 False，termination_reason 保持 None
```

**判定顺序**就是 [tool_agent_loop.py:284-289](../../../verl/experimental/agent_loop/tool_agent_loop.py#L284-L289) 里基类的检查顺序——保持一致才能正确反推。

这里有个微妙之处：基类的三个 limit 是**先到先返回**（response_length 检查最先），所以理论上如果两个 limit 同时满足（比如刚好同一轮既超 response_length 又超 turn limit），CaRR 会标成 `response_limit` 而不是 `turn_limit`——和基类的优先级一致。

#### 第 (4) 步：trim 检测的入口条件

现在能看清第 (4) 步的判定为什么是 `if not agent_data.tool_calls` 了：

| 进入 (4) 时的 `tool_calls` | 含义 | 处理 |
|---------------------------|------|------|
| 非空 | 模型生成了 tool_call → 转 PROCESSING_TOOLS | **跳过 trim**，因为这一轮还要继续工具调用 |
| 空 + 走 ★1/★2/★3 出口 | 撞 limit 但未解析（在 (1) 处被主动清空保证了空）| **尝试 trim 救援** |
| 空 + 走 ★6 出口 | 模型自然结束（基类 Phase D 解析后确实没 tool_call）| **尝试 trim 救援** |

最后两类都会进入 trim 分支。**trim 成功时撤销 hit_limit**（[L353-355](../tools/carr_agent_loop.py#L353-L355)）—— 这就是为什么"撞到 response_limit 但答案写完了"的样本最终被打成 `early_stop` 而不是 unfinished。

#### 一句话总结

> CaRR 的 (1) 主动清空 `tool_calls`、(2) 反推 termination_reason、(4) trim 检测——这三步都是因为基类 `_handle_generating_state` 的 6 个返回出口里**有 3 个出口（★1/★2/★3）会跳过 tool_calls 解析直接返回 TERMINATED**，CaRR 必须在外层补齐缺失的状态正确性和救援逻辑。

### 4.3 `PROCESSING_TOOLS` 状态（CaRR 重写）

CaRR 替换了基类的 `_handle_processing_tools_state`，调用自己的 `_handle_processing_tools_state_with_history`。L385-420：

```python
elif state == AgentState.PROCESSING_TOOLS:
    # (1) Budget 预检（关键：在 tokenize 之前先检查）
    budget_reason = self._tool_budget_termination_reason(
        agent_data.tool_calls, total_tool_calls, search_count, open_count, find_count,
    )
    if budget_reason is not None:
        hit_budget = True
        termination_reason = budget_reason
        state = AgentState.TERMINATED
        continue

    # (2) 计数（仅记录实际执行的，截到 max_parallel_calls）
    executed = agent_data.tool_calls[:self.max_parallel_calls]
    total_tool_calls += len(executed)
    for tc in executed:
        if tc.name == "browser.search":
            search_count += 1
        elif tc.name == "browser.open":
            open_count += 1
        elif tc.name == "browser.find":
            find_count += 1

    # (3) 实际执行 + 写 reward_history
    state = await self._handle_processing_tools_state_with_history(
        agent_data, reward_history, pending_tool_calls,
    )
    pending_tool_calls = []
    turn_idx += 1

    # (4) 如果工具响应 tokenize 后超长，也算 response_limit
    if state == AgentState.TERMINATED:
        hit_limit = True
        termination_reason = "response_limit"
```

**`_tool_budget_termination_reason` 的精妙之处**（L92-123）：它是**预检**——在执行工具之前就模拟"如果把这一批工具调用加上去会不会超 budget"，超了就直接 abort，**不执行**这一批。

这意味着 search_count / open_count / find_count 都是"实际执行了的次数"，不是"模型尝试调用的次数"。

**`_handle_processing_tools_state_with_history` 内部**（L496-606）：

1. 并行 `await asyncio.gather(*tasks)` 执行所有 tool_calls
2. 把工具响应写两遍：
   - **CaRR 格式**：`{"role": "tool", "content": [{"tool_call_id": ..., "output": ...}, ...]}` → 进 `reward_history`
   - **标准格式**：`{"role": "tool", "content": "..."}` → 进 `agent_data.messages`（用于下一轮 LLM tokenization）
3. 把工具响应 tokenize 后追加到 `agent_data.prompt_ids`，对应位置 `response_mask = 0`（不参与 loss）
4. **超长检查**（L590-591）：如果加上工具响应后 `response_mask` 长度会超过 `response_length`，return `TERMINATED`
5. 否则 `agent_data.user_turns += 1`，return `GENERATING`

**`response_mask` 的语义**：1 = 模型生成的 token（参与 RL 梯度），0 = 工具响应或注入的 token（不参与）。这是 Delta-based Tokenization 的核心。

### 4.4 `INTERACTING` 状态

CaRR 不使用，但代码里保留了入口：

```python
elif state == AgentState.INTERACTING:
    state = await self._handle_interacting_state(agent_data)
```

由基类处理，仅在 `interaction_config_file` 非空时启用。CaRR 配置里这个字段是 None，所以这条分支永远不会被走。

### 4.5 终止路径汇总

所有可能的终止入口：

| 入口 | 触发条件 | 设置 |
|------|---------|------|
| 循环开头 wall-time | `time.monotonic() - rollout_start ≥ max_rollout_wall_time_s` | `hit_budget=True, reason=rollout_timeout` |
| `GENERATING` 后 response_length | `len(response_mask) ≥ response_length` 且 trim 失败 | `hit_limit=True, reason=response_limit` |
| `GENERATING` 后 turn limits | `assistant_turns / user_turns` 超限 | `hit_limit=True, reason=assistant_turn_limit / user_turn_limit` |
| `GENERATING` 后无 tool_call + 无 interaction | 本轮生成返回了非工具答案；若未触发 trim 则落入 finished-without-trim bucket | 不设 hit_*，`termination_reason=None`（可能是 `natural`，也可能 trim 后变成 `early_stop`） |
| `PROCESSING_TOOLS` budget 预检 | tool_call/search/open/find_budget 超限 | `hit_budget=True, reason=*_budget` |
| `PROCESSING_TOOLS` 工具响应过长 | 加上 tool response 后 response_mask 超长 | `hit_limit=True, reason=response_limit` |

---

## 5. `early_stopped` vs `natural completion` 详解

这是本文档的核心。两者**都属于 `task_unfinished=False` 的完成样本**，区别**不在于是否撞到 token 上限**，而在于 **trim 是否命中**。

### 5.1 三层定义

#### Layer 1: `task_unfinished`（成败二元判定）

[`carr_agent_loop.py:136`](../tools/carr_agent_loop.py#L136)：

```python
empty_history = len(reward_history) == 0
no_final_assistant = bool((not empty_history) and reward_history[-1].get("role") != "assistant")
task_unfinished = hit_limit or hit_budget or empty_history or no_final_assistant
```

四个条件任一满足就是 unfinished：
- `hit_limit`：撞到 response/turn 硬上限
- `hit_budget`：撞到 wall-time/tool budget
- `empty_history`：空历史（异常情况）
- `no_final_assistant`：最后一条不是 assistant（说明在 tool_response 之后被截断，模型还没机会回复）

#### Layer 2: `content_early_stopped`（trim 命中标志）

只在一处被设为 True，[L347-355](../tools/carr_agent_loop.py#L347-L355)：

```python
if not agent_data.tool_calls:
    trimmed_text = self._try_trim_completed_answer(agent_data, assistant_text)
    if trimmed_text is not None:
        assistant_text = trimmed_text
        content_early_stopped = True
        hit_limit = False           # ← 关键：撤销 hit_limit
        termination_reason = None
```

只在以下条件全满足时为 True：
1. 当前 turn **没有解析出任何 tool_call**
2. `_try_trim_completed_answer()` 成功裁剪

#### Layer 3: 最终 metric 标签（[L157-158](../tools/carr_agent_loop.py#L157-L158)）

```python
completion_finished_early_stop = bool((not task_unfinished) and content_early_stopped)
completion_finished_natural    = bool((not task_unfinished) and (not content_early_stopped))
```

四种组合：

| `task_unfinished` | `content_early_stopped` | `completion_finished_early_stop` | `completion_finished_natural` |
|:---:|:---:|:---:|:---:|
| True | False | False | False |
| True | True | False | False |
| **False** | **True** | **True** ✓ | False |
| **False** | **False** | False | **True** ✓ |

### 5.2 `_try_trim_completed_answer` 详解

这是整个 early_stop 机制的"判定大脑"。它分两个函数：

#### `_extract_completed_answer_text`（[L173-220](../tools/carr_agent_loop.py#L173-L220)）

从 `assistant_text` 中提取一段"安全可裁剪"的前缀，返回字符串或 `None`。完整代码按 5 个 step 拆解：

##### Step 1：拒绝含 `<tool_call>` 残留的文本

```python
if "<tool_call>" in assistant_text or "</tool_call>" in assistant_text:
    return None
```

只要文本**任何位置**含 `<tool_call>` 或 `</tool_call>`，立即拒绝。目的是避免半截 tool_call（比如 token 截断刚好在中间）造成的脏数据被当成答案送给 reward server。

##### Step 2：定位两个 anchor

```python
exact_pos = assistant_text.find("## Exact Answer")
refs_pos = assistant_text.find("## References")
if exact_pos == -1 or refs_pos == -1 or refs_pos <= exact_pos:
    return None
```

三个失败条件：
- `## Exact Answer` 不存在
- `## References` 不存在
- `## References` 出现在 `## Exact Answer` 之前

注意用的是 `find()`，**只匹配第一次出现**。如果模型在 thinking 中提到 "## Exact Answer" 这个字符串（比如做格式规划），会被误匹配——但由于 Step 1 已排除 tool_call，加上 SFT 训练让模型规范使用这两个 heading，实际误匹配很少。

##### Step 3：准备扫描 `## References` 段

```python
ref_line_pattern = re.compile(r"^\s*(\[\d+\]|\d+\.\s+|[-*]\s+|https?://)")
ref_section = assistant_text[refs_pos:]
lines = ref_section.splitlines(keepends=True)
if not lines:
    return None

kept_chars = len(lines[0])  # 保留 "## References" 标题这一行
saw_reference_entry = False
```

`ref_line_pattern` 接受 5 种引用格式：

| 模式 | 例子 |
|------|------|
| `[N]` | `[1]`, `[42]` |
| `N. ` | `1. https://...`, `2. Smith et al.` |
| `- ` | `- https://...` |
| `* ` | `* https://...` |
| `https?://` | 直接以 URL 开头的行 |

`splitlines(keepends=True)` 保留换行符，方便后面精确计算字符数（最终用 `kept_chars` 切原文本）。

`kept_chars = len(lines[0])` 先把 `## References` 这一行本身的长度记上。

##### Step 4：扫描 References 段后续行（核心逻辑）

```python
for line in lines[1:]:
    stripped = line.strip()

    if not saw_reference_entry:
        # 还没看到第一条引用
        if not stripped:
            kept_chars += len(line)        # 空行：跳过但保留
            continue
        if ref_line_pattern.match(stripped):
            saw_reference_entry = True     # 找到第一条规范引用
            kept_chars += len(line)
            continue
        return None                        # ← 第一条非空行不规范，整个 trim 失败

    # 已经看到至少一条引用
    if not stripped:
        kept_chars += len(line)            # 空行：保留
        continue
    if ref_line_pattern.match(stripped):
        kept_chars += len(line)            # 规范引用：保留
        continue
    break                                  # ← 遇到非规范行就停（但不算失败）
```

注意两种"停止"行为的区别：

- **没找到首条引用前** 遇到不规范行 → `return None`（整个 trim 失败）
- **已找到首条引用后** 遇到不规范行 → `break`（trim 成功，但裁掉这行及之后的内容）

##### Step 5：终检与切片

```python
if not saw_reference_entry:
    return None

return assistant_text[: refs_pos + kept_chars].rstrip()
```

如果整个 References 段一条规范引用都没有，失败。否则返回从开头到 `refs_pos + kept_chars` 的前缀，去掉尾部空白。

##### 5 个例子帮助理解判定结果

**例 A：标准模板 → trim 成功**
```
## Explanation with Citations
The paper is from Few-Body Systems 2004 [[1]].
## Exact Answer
Photodisintegration of Light Nuclei
## References
[1] https://link.springer.com/article/10.1007/s006010400006
[2] https://arxiv.org/abs/nucl-th/0406080
```
→ 5 个 step 全过，返回完整文本

**例 B：非模板答案 → trim 失败**
```
The answer is "Photodisintegration of Light Nuclei", published in Few-Body Systems Vol 35 (2004).
```
→ Step 2 找不到 `## Exact Answer` → 返回 None

**例 C：缺 References → trim 失败**
```
## Exact Answer
April 23, 1970

## Explanation
This is the date of the protocol's introduction.
```
→ Step 2 找不到 `## References` → 返回 None

**例 D：References 段格式不规范 → trim 失败**
```
## Exact Answer
X
## References
According to multiple sources I found through Wikipedia and ACM Digital Library...
```
→ Step 4 第一条非空行 "According..." 不匹配 `ref_line_pattern` → `return None`

**例 E：模板规范 + 末尾多写了几行 → trim 成功 + 裁切**
```
## Exact Answer
X
## References
[1] https://example.com
[2] https://other.com

Note: I'm fairly confident in this based on triangulating multiple sources.
```
→ Step 4 处理 `[1]` `[2]` → saw_reference_entry=True；遇到 "Note:" → break
→ 返回到 `[2]` 末尾的文本，"Note:" 那行**被裁掉**

#### `_try_trim_completed_answer`（[L222-254](../tools/carr_agent_loop.py#L222-L254)）

外层包装，加了一层 token-level 安全验证：

```python
def _try_trim_completed_answer(self, agent_data, assistant_text):
    trimmed_text = self._extract_completed_answer_text(assistant_text)
    if trimmed_text is None:
        return None

    # 安全校验 1：encode 后的 token 数不能超过当前 response_ids 长度
    trimmed_ids = self.tokenizer.encode(trimmed_text, add_special_tokens=False)
    trimmed_len = len(trimmed_ids)
    current_len = len(agent_data.response_ids)
    if trimmed_len <= 0 or trimmed_len > current_len:
        return None

    # 安全校验 2：decode→encode round-trip 必须稳定
    current_prefix_text = self.tokenizer.decode(
        agent_data.response_ids[:trimmed_len], skip_special_tokens=True
    )
    normalize = lambda s: re.sub(r"\s+", " ", s).strip()
    if normalize(current_prefix_text) != normalize(trimmed_text):
        return None

    # 通过校验：实际裁剪 response_ids / prompt_ids / response_mask / response_logprobs
    ...
    agent_data.response_ids = agent_data.response_ids[:trimmed_len]
    agent_data.prompt_ids = agent_data.prompt_ids[:prefix_prompt_len] + agent_data.response_ids
    agent_data.response_mask = agent_data.response_mask[:-current_len] + [1] * trimmed_len
    if current_turn_logprobs is not None:
        agent_data.response_logprobs = agent_data.response_logprobs[:-current_len] + current_turn_logprobs[:trimmed_len]

    return trimmed_text
```

**两个安全校验为什么重要**：
1. 防止 tokenizer 编码偏移导致 `response_mask` / `response_ids` 错位
2. 保证 RL 梯度计算时所有 tensor 长度一致

### 5.3 4 × 1 = 5 种实际场景对照表

trim 触发与否 + 基类出口 = 共 5 种可能（tool_call 非空时直接跳过 trim 算第 5 种）：

| # | 模型行为 | 基类出口 | `tool_calls` | trim 是否被尝试 | trim 结果 | 最终标签 |
|---|---------|---------|-------------|--------------|----------|---------|
| 1 | 写了 `## Exact Answer` + `## References` + 可裁剪引用段；本轮生成正常返回 | ★6 finished | 空 | 是 | ✅ 成功 | **early_stop** |
| 2 | 写了 `## Exact Answer` + `## References` + 可裁剪引用段；但撞 response_limit | ★1 limit | 空（被 (1) 清空）| 是 | ✅ 成功 | **early_stop** |
| 3 | 本轮生成正常返回，但输出没命中 trim 模式（散文化答案、没写 References、引用段不匹配） | ★6 finished | 空 | 是 | ❌ 失败 | **natural** |
| 4 | 没用模板 / 没写完答案 + 撞 response_limit | ★1 limit | 空 | 是 | ❌ 失败 | **unfinished** |
| 5 | 解析出了 tool_call | ★4 PROCESSING_TOOLS | 非空 | **否**（被 `if not tool_calls` 短路） | — | 转 PROCESSING_TOOLS |

#### 关键观察

**早停标签不区分基类出口**——场景 1 和场景 2 都被打成 `early_stop`，意味着：

- 模型按 SFT 模板正常完成（场景 1）
- 模型撞到 response_limit 但答案模板写完了（场景 2）

**两者最终标签一样**。这是因为 trim 一旦成功就会撤销 `hit_limit`：

```python
if trimmed_text is not None:
    content_early_stopped = True
    hit_limit = False           # ← 关键：场景 1 本来就是 False，场景 2 被强制改成 False
    termination_reason = None
```

#### 触发分布与命名陷阱

| 标签 | 包含场景 | 真实含义 |
|------|---------|---------|
| `early_stop` | 1 + 2 | **输出命中了 trim 可识别的完整答案格式**（不论是否撞 limit） |
| `natural` | 3 | **finished 但未命中 trim 的残差类别** |
| `unfinished` | 4 | 撞 limit 且输出不符合模板 |

命名上 `early_stop` / `natural` 让人以为是"主动早停"vs"自然结束"，但代码里**没有消费 rollout 返回的 `stop_reason`**，也没有任何"主动判断停止"的硬路径。两者真实区分的是 **trim 是否成功**——本质上是**trim 格式信号**，而不是停止时机信号。详见 5.4 节。

### 5.4 标签的真实语义（重要！容易被命名误导）

#### `early_stop` 的真实语义

回到表格：

| 进入 trim 前的状态 | trim 结果 | 最终标签 |
|-------------------|----------|---------|
| ★1/★2/★3 撞 limit | ✅ 成功 | early_stop |
| ★1/★2/★3 撞 limit | ❌ 失败 | unfinished |
| ★6 本轮生成返回且未触发 unfinished | ✅ 成功 | early_stop |
| ★6 本轮生成返回且未触发 unfinished | ❌ 失败 | natural |

注意 **early_stop 在两种基类出口下都可能产生**——只要 trim 成功就标 early_stop。换句话说：

> **`early_stop` 的真实语义不是"模型主动早停"，而是"模型的输出命中了 trim 可识别的完整答案格式（`## Exact Answer` + `## References` + 可裁剪引用段）"。**

这是命名最大的陷阱。代码里**没有任何"模型主动判断该停了"的硬路径**——`early_stop` 是个 post-hoc 标签，靠 trim 模板检测决定。

#### `natural` 的真实语义（残差类别）

`natural` 是个**残差类别**——它捕获的是「finished + trim 失败」的样本。很多情况下，这意味着模型本轮自己收束并返回了非工具答案；但**当前代码没有持久化 `finish_reason/stop_reason`**，所以不能严格断言一定是 EOS。这意味着 natural 样本可能是：

| 子类 | 描述 | 答案质量 |
|------|------|---------|
| 答对了但用非模板格式 | "The answer is X, from..." | 高 |
| 写了 `## Exact Answer` 但没写 `## References` | 简短答案 | 高/中 |
| References 段用散文化叙述（不匹配规范引用格式） | 描述性而非列表式 | 中 |
| 模型在 thinking 阶段就自行收束返回（罕见） | 没真正答完 | 低 |

这是**杂糅类别**，单凭 natural 这个标签**没法**对样本质量做强结论。

#### 为什么命中 trim 格式的回答都会被打成 early_stop？

是的，这是个反直觉的事实：

- 模型按可裁剪答案格式写完答案 + 本轮生成正常返回 → trim 成功 → **早停标签 early_stop**
- 模型按可裁剪答案格式写完答案 + 撞 response_limit → trim 救援成功 → **早停标签 early_stop**

两者最终标签一样。也就是说：**早停 ratio 升高 ≠ "RL 学会了主动早停"，而是"RL 模型更频繁地输出命中 trim 格式的回答"**。

#### 评测分析中"natural > early_stop 准确率"的诚实解读

数据：
- dd111 SFT：natural=9 (outcome 0.89) vs early_stop=21 (outcome 0.57)
- dd111 RL：natural=12 (outcome 1.00) vs early_stop=29 (outcome 0.83)

**两者方向一致**（natural 在 SFT 和 RL 上都更高）。但这**主要不是因为 natural 是更优的成功机制**，而是 selection bias 的结果：

1. **格式宽松性 selection bias**：natural 类别要求"finished 且没命中 trim 模式"。能进这个类别的样本，往往对应**较容易/较确定的问题**（答案短到不需要长 reference 列表，或直接一句话作答），所以正确率高
2. **小样本噪声**：n=9-29，1.00 和 0.89 的差异（0.11）和 0.83 的差异（0.17）都不是统计显著
3. **judge 容错性**：DeepSeek judge 不严格要求模板，所以非模板答案不被惩罚

**真实驱动因素的猜测**：natural 类别更可能集中"问题难度低 + 答案简单"的样本，所以正确率高——这是问题分布造成的，不是停止策略造成的。

#### 所以应该怎么解读 RL 训练对这些标签的影响？

**有意义的指标**：
- ✅ `finished ratio` 上升（任务完成率）
- ✅ `finished-to-correct conversion rate` 上升（完成→正确的转化率）
- ✅ 整体 `outcome_reward` 上升

**容易误读的指标**（不应直接作为 RL 学到能力的证据）：
- ⚠️ `early_stop ratio` 上升 → 实际反映"模型更稳定地输出命中 trim 的交卷格式"，而不是"学会了主动早停"
- ⚠️ `natural` vs `early_stop` 的准确率对比 → 主要是 selection bias，不是策略差别

**简历/面试避坑**：
- ❌ 不要写"RL 让模型学会了 X% 的早停能力"
- ❌ 不要写"natural completion 准确率 100%，证明自然结束是更优策略"
- ✅ 可以写"RL 提升了任务完成率（finished ratio: 27%→37%）和完成→正确转化率（67%→88%）"

### 5.5 为什么这么设计？

如果不做 trim 救援，所有"模型刚好在 response_length 边缘写完答案"的样本都会被打成 unfinished → reward server 直接返回 0 → outcome=0。但这些样本的答案其实**已经写完了**，只是多了一些尾巴（比如多写了一条 reference、或者有空行）。

这种"误杀"会给 RL 训练带来噪声梯度。Trim 机制相当于一个温和的容错——只要答案格式完整 + 引用规范，就认为模型完成了任务，不因为多写了几个 token 而算失败。

但 trim 的判定**非常保守**（5 条格式硬条件 + 2 条 token 安全校验），避免误判把 unfinished 样本错误地标成成功。

**副作用**：trim 把"是否命中可裁剪答案格式"和"是否成功完成"绑在了一起，导致 `early_stop` / `natural` 这两个标签实际上是**trim 格式信号**而不是**停止时机信号**。这是命名 vs 真实语义脱节的根本原因。要彻底解决，需要把"trim 命中"和"生成停止原因"两个信号分开记录（比如增加 `template_trim_hit`，并把 rollout 的 `stop_reason/finish_reason` 持久化下来）。

---

## 6. 一次完整 trajectory 示例（附状态迁移）

假设一个搜索任务：

```
T=0  PENDING
     ├─ apply_chat_template(messages, tools) → prompt_ids
     └─ → GENERATING

T=1  GENERATING (turn 1)
     ├─ LLM 生成: "<think>...</think><tool_call>{search query='X'}</tool_call>"
     ├─ assistant_turns=1, response_mask 追加 1×N 个 1
     ├─ 解析 tool_calls = [search('X')]
     ├─ trim 检测: tool_calls 非空，跳过
     ├─ reward_history 追加 assistant message
     └─ → PROCESSING_TOOLS

T=2  PROCESSING_TOOLS
     ├─ budget 预检: search_count(0)+1=1 ≤ max_search_calls=80 ✓
     ├─ 执行 search('X') → tool_response
     ├─ tokenize tool_response → 追加到 prompt_ids，response_mask 追加 1×M 个 0
     ├─ search_count=1, total_tool_calls=1, user_turns=1
     ├─ reward_history 追加 tool message (含 tool_call_id)
     └─ → GENERATING

T=3  GENERATING (turn 2)
     ├─ LLM 生成: "<think>...</think><tool_call>{open(id=0)}</tool_call>"
     ├─ ...
     └─ → PROCESSING_TOOLS

... 多轮工具调用 ...

T=N  GENERATING (turn N)
     ├─ LLM 生成: "<think>I have enough info</think>
     │              ## Explanation with Citations
     │              ...
     │              ## Exact Answer
     │              [the answer]
     │              ## References
     │              [1] ..."
     ├─ 假设刚好达到 response_length=61440
     ├─ 基类 return TERMINATED, hit_limit=True, reason=response_limit
     ├─ 解码 assistant_text
     ├─ tool_calls 为空（这次没 tool_call）→ 进入 trim 分支
     ├─ _try_trim_completed_answer:
     │    ✓ 找到 ## Exact Answer
     │    ✓ 找到 ## References, 在 Exact Answer 之后
     │    ✓ References 后有 [1] 规范条目
     │    ✓ token round-trip 校验通过
     │    → 裁剪 response_ids 到 trimmed_len，response_mask 同步裁剪
     ├─ content_early_stopped=True
     ├─ hit_limit=False, termination_reason=None  ← 撤销
     └─ → TERMINATED (退出 while)

构造 unfinished_flags:
     ├─ task_unfinished=False (hit_limit/budget 都为 False)
     ├─ completion_finished_early_stop=True
     └─ completion_finished_natural=False

构造 AgentLoopOutput:
     ├─ extra_fields["content_early_stopped"]=True
     ├─ extra_fields["task_unfinished"]=False
     ├─ extra_fields["search_count"]=N_search, ... 等
     └─ 返回
```

---

## 7. `extra_fields` 输出字段速查

| 字段 | 类型 | 含义 |
|------|------|------|
| `messages` | list | CaRR-format reward_history，发给奖励服务器 |
| `task_unfinished` | bool | 是否未完成（核心二元判定） |
| `unfinished_limit` | bool | == hit_limit |
| `unfinished_budget` | bool | == hit_budget |
| `unfinished_empty_history` | bool | 空历史（异常） |
| `unfinished_no_final_assistant` | bool | 最后一条不是 assistant |
| `unfinished_fallback` | bool | empty_history OR no_final_assistant |
| `completion_finished_early_stop` | bool | 完成 AND content_early_stopped |
| `completion_finished_natural` | bool | 完成 AND NOT content_early_stopped |
| `content_early_stopped` | bool | trim 救援标志 |
| `tool_call_counts` | int | 实际执行的工具调用总数 |
| `search_count` | int | search 执行次数 |
| `open_count` | int | open 执行次数 |
| `find_count` | int | find 执行次数 |
| `parse_error_count` | int | tool_call JSON 解析失败次数（不参与终止） |
| `hit_limit` | bool | 撞到 response/turn 硬上限 |
| `hit_budget` | bool | 撞到 wall-time/tool budget |
| `termination_reason` | str/None | 终止原因 |
| `termination_response_limit` | float | 1.0 if reason=="response_limit" else 0.0 |
| `termination_*_limit/budget` | float | 同上，分别标识各种终止原因 |
| `rollout_elapsed_s` | float | wall-time 耗时（秒） |
| `response_length` | float | 当前 response_mask 长度 |
| `response_length_max` | float | 上限值 |
| `response_length_ratio` | float | response_length / max |

---

## 8. 设计决策与边界情况速记

1. **不再使用 INTERACTING**：CaRR 的环境反馈完全靠 tool_response，不需要单独的 interaction 角色
2. **基类 `extract_tool_calls` 在 limit 终止时不会被调用**——所以 CaRR 第 (1) 步要主动清空 stale `tool_calls`
3. **trim 是 post-hoc 救援**——不存在 "模型主动 early stop" 的硬路径
4. **budget 是预检不是后检**——`_tool_budget_termination_reason` 模拟新一批工具加上去会不会超 budget，超了就不执行
5. **`response_mask` 语义**：1 = 模型生成（参与 loss/RL），0 = 工具响应（不参与）
6. **session 必须关闭**：`finally` 块兜底，避免 tool server 端 session 泄漏
7. **wall-time 检查在循环开头**：优先级最高，无论当前在哪个状态
8. **parse_error_count 不参与终止**：仅作为统计，模型即使 parse error 也会继续生成

---

## 9. 与 async 版本（`CaRRAsyncPartialToolAgentLoop`）的关系

`CaRRAsyncPartialToolAgentLoop`（L610-1050）继承自 `CaRRToolAgentLoop`，重写 `run()` 以支持：
- **partial rollout**：rollout 可被 cancel，恢复时从 checkpoint 继续
- **跨 worker 状态持久化**：所有状态序列化到 `output.extra_fields["carr_async_state"]`
- **param version tracking**：记录该 trajectory 跨越了哪些参数版本（用于 importance sampling）

但 **early_stop / natural completion / 各种 termination 的判定逻辑完全一样**——async 版本只是把状态持久化和恢复加进去，核心状态机和 trim 机制都直接复用基类。

具体差异见 [`async_rl_explainer_20260321.md`](./async_rl_explainer_20260321.md) 和 [`agent_handoff_background_20260321.md`](./agent_handoff_background_20260321.md)。
