# 两个 Tool Server 优化方案详细解析

本文档对比分析两个优化方案的原理、实现细节和 trade-off。

---

## 0. 前置知识：当前 Tool 调用的完整路径

理解优化方案之前，先搞清楚一个工具调用从发起到返回经历了哪些层：

```
方案一保留的路径（仅在 Tool Server 加缓存）：
═══════════════════════════════════════════════════════════════

CaRRToolAgentLoop (carr_agent_loop.py)
  │  PROCESSING_TOOLS 状态
  │  调用 self._call_tool()
  ▼
CaRRBrowserTool.execute() (carr_browser_tool.py:51)
  │  从 agent_data 提取 request_id, search_forbidden_strs
  │  调用 session_manager.ensure_started()（如果是该 session 首次工具调用）
  ▼
CaRRSessionManager.call_server() (carr_session_manager.py:77)
  │  构建 JSON payload: {session_id, name, arguments, remote_env_info}
  │  aiohttp POST → http://localhost:7230
  │  等待响应（timeout 60s）
  ▼                                              ← 网络边界（localhost HTTP）
CaRR Tool Server (launch_server.py:65)
  │  解析请求
  │  查找 sandbox = session2sandbox[session_id]
  ├── browser.search:
  │     search_serper(query, num, forbidden_strs)  ← 外部 API（Serper.dev）
  │     sandbox["idx2url"] = idx2url               ← 存储搜索结果映射
  │     返回格式化文本
  ├── browser.open:
  │     url = sandbox["idx2url"][id]               ← 读取之前搜索的 URL 映射
  │     parse_url(url, forbidden_strs)             ← 外部 API（Jina Reader）
  │     sandbox["cur_web_content"] = content       ← 存储网页内容
  │     返回截断后内容（≤10000 chars）
  └── browser.find:
        content = sandbox["cur_web_content"]       ← 读取之前打开的网页内容
        find(pattern, content)                     ← 纯本地字符串匹配
        返回匹配片段
  ▼
响应 JSON: {"output": result_text}
  ▼
返回到 CaRRBrowserTool → AgentLoop → 拼接到 response


方案二去掉的路径（直接在 agent 侧调用底层函数）：
═══════════════════════════════════════════════════════════════

CaRRToolAgentLoop (carr_agent_loop.py)
  │  PROCESSING_TOOLS 状态
  │  调用 self._call_tool()
  ▼
CaRRBrowserTool.execute() (carr_browser_tool.py:51)
  │  从 agent_data 提取 request_id, search_forbidden_strs
  │  读取 agent_data.extra_fields 中的 idx2url / cur_web_content
  │  直接调用:
  ├── browser.search:
  │     result, idx2url = await search_serper(query, num, forbidden_strs)
  │     agent_data.extra_fields["idx2url"] = idx2url       ← 状态存在 agent_data
  │     查缓存 → 命中则跳过 API 调用
  ├── browser.open:
  │     url = agent_data.extra_fields["idx2url"][id]
  │     content = await parse_url(url, forbidden_strs)
  │     agent_data.extra_fields["cur_web_content"] = content
  │     查缓存 → 命中则跳过 API 调用
  └── browser.find:
        content = agent_data.extra_fields["cur_web_content"]
        results = find(pattern, content)                   ← 直接本地调用
  ▼
返回 ToolResponse → AgentLoop → 拼接到 response

    ↑ 注意：没有 HTTP 层，没有 session_manager，没有 Tool Server 进程
```

---

## 1. 方案一：Tool Server 全局缓存 + Single-Flight 去重

### 1.1 核心思路

**不改架构，只在现有 Tool Server 内部加缓存层。** 所有 HTTP 路径、session 管理、进程边界都不变。

### 1.2 它解决的问题

当 `rollout.n=8` 时，同一个 prompt 生成 8 条 trajectory。这 8 条 trajectory 大概率会：
- 搜索类似或相同的 query（比如都搜 "breakpoint graph genome"）
- 打开相同的 URL（搜索结果排序一样）

**当前没有缓存**，每条 trajectory 独立发起 HTTP 请求到 Serper/Jina，每次 5-15s。

```
没有缓存时（n=8 同一 prompt）：

trajectory 0: search("breakpoint graph") → Serper API  5s  ← 必须等
trajectory 1: search("breakpoint graph") → Serper API  5s  ← 完全重复！
trajectory 2: search("breakpoint graph") → Serper API  5s  ← 完全重复！
...
trajectory 7: search("breakpoint graph") → Serper API  5s  ← 完全重复！

有缓存 + single-flight 后：

trajectory 0: search("breakpoint graph") → Serper API  5s  ← miss，实际调用
trajectory 1: search("breakpoint graph") → 缓存命中      <1ms ← 瞬间返回
trajectory 2: search("breakpoint graph") → 缓存命中      <1ms
...
trajectory 7: search("breakpoint graph") → 缓存命中      <1ms

如果 trajectory 1 和 trajectory 0 几乎同时发起（还没写入缓存）：
trajectory 0: search("breakpoint graph") → Serper API（正在调用中...）
trajectory 1: search("breakpoint graph") → single-flight 等待 trajectory 0 的结果
                                           （不会发第二个 API 请求）
```

### 1.3 LRUCache 实现原理详解

```python
class LRUCache:
    def __init__(self, maxsize=10000):
        self._cache = OrderedDict()        # 有序字典，实现 LRU 淘汰
        self._inflight: dict[str, asyncio.Future] = {}  # 正在进行中的请求
        self._lock = asyncio.Lock()         # 异步锁，保护并发写入

    async def get_or_fetch(self, key_args, fetch_fn):
        key = md5(str(key_args))            # 将参数 hash 为 cache key
```

**三种路径**：

```
路径 A: 缓存命中 (Hot Path)
──────────────────────────
key 在 _cache 中 → 直接返回 → O(1)，无锁
最常见场景：n=8 中后 7 条 trajectory 的相同 query

路径 B: 缓存未命中，首个请求 (Cold Path)
──────────────────────────────────────
key 不在 _cache 中
→ 获取锁
→ 二次检查（double-check：锁竞争时可能已被其他协程填充）
→ 创建 asyncio.Future 放入 _inflight
→ 释放锁
→ 调用 fetch_fn()（实际的 Serper/Jina API 调用）
→ 设置 Future 结果
→ 获取锁 → 写入 _cache → 移除 _inflight → 释放锁
→ 返回结果

路径 C: 缓存未命中，但同一 key 正在请求中 (Single-Flight)
──────────────────────────────────────────────────────
key 不在 _cache 中
→ 获取锁
→ 发现 key 在 _inflight 中（有其他协程正在 fetch）
→ 取出那个 Future
→ 释放锁
→ await future（等待首个请求完成）
→ 返回结果（不会发第二个 API 请求）
```

### 1.4 Cache Key 设计

```python
# browser.search 的 cache key:
cache_key = (query, num, tuple(sorted(search_forbidden_strs)))
# 例: ("breakpoint graph genome", 10, ("What is E2...",))

# browser.open 的 cache key:
url = idx2url.get(id, id)  # 先解析 id → URL
cache_key = (url,)
# 例: ("https://en.wikipedia.org/wiki/Breakpoint_graph",)
```

**为什么这样设计**：
- `search_forbidden_strs` 影响搜索结果过滤，不同 forbidden_strs 可能产生不同结果
- `url` 而非 `id`：同一个 URL 可能通过不同的 search index 到达（trajectory A 搜到的第 3 个结果 = trajectory B 搜到的第 1 个结果）
- `find` 不需要缓存：纯本地操作，<1ms

### 1.5 为什么不需要 TTL

训练过程中搜索结果不会实时变化（不像生产搜索引擎需要新鲜度）。一个 epoch 内相同 query 的结果完全可以复用。`maxsize=5000` 的 LRU 会自然淘汰最久未使用的条目。

### 1.6 方案一的局限性

| 局限 | 说明 |
|------|------|
| **仍有 localhost HTTP 开销** | 每次工具调用仍需 JSON 序列化 → HTTP POST → JSON 反序列化 |
| **仍是单进程** | Quart 单进程 event loop，高并发下排队 |
| **缓存不跨进程** | 如果起多个 Tool Server 实例，缓存不共享 |
| **find 仍走 HTTP** | `browser.find` 是纯本地字符串匹配，但仍然经过 HTTP 往返 |
| **session 状态仍在 server** | idx2url / cur_web_content 仍在 Tool Server 的 sandbox 字典中 |

---

## 2. 方案二：去掉 Tool Server，Native Tool 直调 + 跨进程缓存

### 2.1 核心思路

**根本性重构：把 Tool Server 的职责拆解，消除中间 HTTP 层。**

- 将 `idx2url` / `cur_web_content` 这两个 session 状态从 Tool Server 的 `session2sandbox` 字典**搬到** `agent_data.extra_fields` 中
- 让 `CaRRBrowserTool.execute()` 直接 `import` 并调用 `web_search.py` 中的 `search_serper()` / `parse_url()` / `find()`
- 不再需要 Tool Server 进程、CaRRSessionManager、HTTP 连接池

### 2.2 为什么可以去掉 Tool Server

通过上面的路径分析可以看到，Tool Server 的 **唯一不可替代的职责** 是维护两个 session 变量：

| 变量 | 写入时机 | 读取时机 | 大小 |
|------|---------|---------|------|
| `idx2url` | `browser.search` 返回后 | `browser.open` 调用时 | dict, ~20 个 URL |
| `cur_web_content` | `browser.open` 返回后 | `browser.find` 调用时 | str, ≤10000 chars |

这两个变量的生命周期严格绑定到**单个 trajectory 的 agent loop 执行过程**：
- 它们在 `start_session` 时初始化为空
- 它们在 trajectory 的多轮工具调用中被填充和读取
- 它们在 `close_session` 时销毁

`agent_data` 正好有完全相同的生命周期——它在 `CaRRToolAgentLoop.run()` 开始时创建，在结束时销毁。所以把这两个变量存在 `agent_data.extra_fields` 中是天然匹配的。

而 `search_serper()` / `parse_url()` / `find()` 这三个函数本身是**完全无状态的**：
- 它们不读写任何全局变量
- 它们的输入/输出完全由参数决定
- 它们可以在任何 Python 进程中直接调用

### 2.3 状态迁移对比

```
当前（方案一保持不变的架构）：

  Agent Loop Worker (Ray worker 进程)          Tool Server (独立 Quart 进程)
  ┌──────────────────────┐                     ┌──────────────────────────┐
  │ CaRRBrowserTool      │  ── HTTP POST ──▶   │ session2sandbox          │
  │   (无状态代理)        │                     │   [req_001]              │
  │                      │  ◀── HTTP resp ──   │     idx2url: {0: url0}   │
  │                      │                     │     cur_web_content: "..." │
  └──────────────────────┘                     │                          │
                                               │ search_serper() ←─┐     │
                                               │ parse_url()    ←──┤     │
                                               │ find()         ←──┘     │
                                               └──────────────────────────┘

方案二改造后：

  Agent Loop Worker (Ray worker 进程)          Tool Server: 不再需要
  ┌──────────────────────────────────┐
  │ CaRRBrowserTool                  │
  │   agent_data.extra_fields:       │
  │     idx2url: {0: url0}           │         ┌─────────────────────┐
  │     cur_web_content: "..."       │         │ 跨进程共享缓存       │
  │                                  │         │ (diskcache / lmdb)   │
  │   直接调用:                       │         │                     │
  │     search_serper() ───缓存查询──▶│         │ key: query+num+hash │
  │     parse_url()     ───缓存查询──▶│         │ val: API 响应       │
  │     find()          (本地直调)     │         └─────────────────────┘
  └──────────────────────────────────┘
```

### 2.4 跨进程缓存的必要性

**为什么方案二需要跨进程缓存，而方案一不需要？**

方案一中 Tool Server 是单进程，所有 Ray worker 的请求都汇聚到这一个进程，所以 `LRUCache` 是 Python 进程内的 `dict`，天然对所有请求可见。

方案二去掉了 Tool Server，每个 Ray AgentLoopWorker 是独立的 Ray actor 进程。它们各自直接调用 `search_serper()`，进程间不共享内存。如果只用进程内 `dict`：

```
Worker 0: search("breakpoint graph") → Serper API 5s → 缓存到 worker 0 的内存
Worker 1: search("breakpoint graph") → Serper API 5s → 无法命中 worker 0 的缓存！
```

所以需要一个**跨进程可共享**的缓存后端。建议的方案：

| 方案 | 优点 | 缺点 |
|------|------|------|
| **diskcache** | 文件系统级别共享，纯 Python，零配置 | 磁盘 I/O（SSD 上 <1ms） |
| **lmdb** | 内存映射文件，极快读取 | 需额外安装，API 稍底层 |
| Redis | 功能最全，支持分布式 | 需启动额外进程，增加运维复杂度 |

**推荐 diskcache**：`pip install diskcache`，单机 8xH200 场景下 SSD 读写 <1ms，远快于 Serper API 的 5s。

### 2.5 Cache Key 设计（方案二）

```python
# search cache key:
key = f"search:{backend}:{normalize(query)}:{num}:{hash(sorted(forbidden_strs))}"

# open cache key:
key = f"open:{normalize(url)}:{hash(sorted(forbidden_strs))}"
```

**为什么 key 包含 `forbidden_hash`**：`search_forbidden_strs` 影响结果过滤（`contain_forbidden_str()` 函数会过滤掉包含 forbidden 文本的搜索结果）。不同 prompt 的 forbidden_strs 不同（通常是问题文本本身），所以缓存需要区分。

**Negative cache（负缓存）**：如果 API 调用失败（超时/错误），也缓存失败结果，避免对已知坏 URL 的重复请求。设一个短 TTL（如 60s）即可。

### 2.6 browser.find 的变化

方案二下 `browser.find` 完全不需要网络：

```python
# 当前（经过 HTTP）：
# CaRRBrowserTool.execute() → HTTP POST → Tool Server → find() → HTTP response

# 方案二（直接调用）：
from CaRR.tool_server.web_search import find

content = agent_data.extra_fields.get("cur_web_content", "")
results = find(pattern, content, max_results=20, context_length=200)
# 耗时: <1ms，无网络开销
```

这对 find 密集的 trajectory 有显著影响。一个典型 trajectory 可能有 5-10 次 find 调用，每次省去 HTTP 往返（~1-5ms × 10 = 10-50ms，虽然不大但积少成多）。

---

## 3. 两个方案的对比

### 3.1 改动范围

| 维度 | 方案一 | 方案二 |
|------|--------|--------|
| **改动文件数** | 1 个（launch_server.py） | 3-4 个（carr_browser_tool.py, carr_agent_loop.py, 可能新增 cache.py，删除 carr_session_manager.py 的使用） |
| **改动行数** | ~80 行（加 LRUCache 类 + 包裹调用） | ~200 行（重写 execute(), 状态迁移, 缓存层） |
| **架构变更** | 无（加层，不改结构） | 有（去掉 Tool Server 依赖，状态迁移） |
| **风险** | 低（缓存透明，出错 fallback 到原始调用） | 中（需要确保状态迁移正确，跨进程缓存需测试） |
| **回滚难度** | 极低（删除缓存代码即可） | 中（需恢复 HTTP 路径） |

### 3.2 性能提升

| 场景 | 方案一 | 方案二 |
|------|--------|--------|
| **n=8 相同 query 命中** | 缓存命中，但仍有 HTTP 往返 (~1-5ms) | 缓存命中，无 HTTP (~0.1ms) |
| **n=8 相同 URL open** | 缓存命中 + HTTP | 缓存命中，无 HTTP |
| **browser.find** | 仍走 HTTP (~1-5ms/次) | 直接本地调用 (<1ms/次) |
| **跨 prompt 相同 query** | 命中（单进程缓存） | 命中（跨进程缓存） |
| **Tool Server 排队** | 仍存在（单进程瓶颈） | 不存在（无 Tool Server） |
| **高并发 (128 trajectory)** | 单进程 event loop 排队 | 分散到多个 Ray worker |

### 3.3 架构影响

| 维度 | 方案一 | 方案二 |
|------|--------|--------|
| **进程数** | 仍需 Tool Server 进程 | 不需要 Tool Server 进程 |
| **运维复杂度** | 不变 | 减少（少一个服务） |
| **调试体验** | 可通过 Tool Server 日志观察所有工具调用 | 需在 agent loop worker 中加日志 |
| **多机扩展** | 多机需部署多个 Tool Server 或单点访问 | 每个节点自带 diskcache，天然分布式 |
| **与 CaRR 代码耦合** | 不改 CaRR 代码 | 需要 import CaRR 的 web_search 函数 |

### 3.4 实际延迟估算

假设一个 10 轮 trajectory，每轮 1 次 search + 1 次 open + 2 次 find：

| 操作 | 次数 | 当前延迟 | 方案一延迟 | 方案二延迟 |
|------|---:|---:|---:|---:|
| search (首次 miss) | 1 | 5s | 5s | 5s |
| search (缓存命中) | 9 | 45s | <1ms × 9 | <1ms × 9 |
| open (首次 miss) | 3 | 30s | 30s | 30s |
| open (缓存命中) | 7 | 70s | <1ms × 7 | <1ms × 7 |
| find (全部) | 20 | 20-100ms | 20-100ms | <1ms × 20 |
| HTTP 往返 | 40 | ~80ms | ~80ms | 0 |
| **合计** | - | **~150s** | **~35s** | **~35s** |

**结论**：两个方案在缓存命中场景下的延迟改善几乎相同，因为主要延迟来自外部 API（Serper/Jina），不是 localhost HTTP。方案二的优势主要体现在高并发下的排队消除和架构简化。

---

## 4. 方案二的独特优势：消除单点瓶颈

方案一保留的 **Tool Server 单进程** 在高并发下仍是瓶颈：

```
方案一, b16/n8 = 128 个 trajectory 并发：

所有 128 个 trajectory 的工具调用 → 汇聚到 1 个 Quart 进程
  ↓
Quart event loop 排队处理
  ↓
即使有缓存命中（<1ms），128 个请求的 HTTP 解析 + JSON 序列化仍需排队
  ↓
对于 miss 的请求（新 query/URL），仍然串行发出外部 API 调用


方案二, b16/n8 = 128 个 trajectory 并发：

128 个 trajectory 分散到 4-8 个 AgentLoopWorker
  ↓
每个 worker 内的 search/open 直接调用（无 HTTP 层）
  ↓
跨进程缓存在 SSD 上共享（diskcache 读取 <1ms）
  ↓
外部 API 调用分散到多个进程的 event loop，天然并行

```

### 4.1 并发公式

```
方案一：
  有效并发 = min(Quart 单进程能力, 外部 API 并发限制)
  ≈ min(~100 req/s localhost, Serper 无明确限制, Jina semaphore=128)

方案二：
  有效并发 = num_workers × 每 worker 内 asyncio 并发
  ≈ 4-8 × 128 = 512-1024 并发（受外部 API 限制）
```

---

## 5. 方案二的 Reward Server 扩展

方案二还提议扩展 reward server 为多实例：

```python
# 当前 (carr_reward.py:34):
REWARD_SERVER_URL = os.environ.get("CARR_REWARD_SERVER_URL", "http://localhost:8888")

# 改为:
REWARD_SERVER_URLS = os.environ.get("CARR_REWARD_SERVER_URLS", "http://localhost:8888")
urls = [u.strip() for u in REWARD_SERVER_URLS.split(",")]

# 调用时 round-robin 或 least-inflight:
url = select_server(urls)  # round-robin 或基于 pending 请求数选择
```

**这很简单因为 reward server 是无状态的**：
- `/evaluate` 端点的输入完全自包含（history, label, remote_env_info）
- 不依赖任何 server-side session 状态
- 每个实例独立运行，不需要共享状态

```bash
# 启动 4 个 reward server 实例
for port in 8881 8882 8883 8884; do
  (cd CaRR/deepsearch_rm_with_rubrics && python launch_server.py \
    --port $port --model_name deepseek-chat \
    --base_url https://api.deepseek.com --api_key "$DEEPSEEK_API_KEY") &
done
export CARR_REWARD_SERVER_URLS="http://localhost:8881,http://localhost:8882,http://localhost:8883,http://localhost:8884"
```

**为什么方案二建议先做 tool 再做 reward**：
1. Reward server 无状态，扩容几乎无风险
2. 但如果 tool 层仍是瓶颈（rollout 慢），扩 reward 没意义——reward 等的是 rollout 产出的 trajectory
3. 先解决 tool 瓶颈让 rollout 加速，再扩 reward 避免 reward 成为新瓶颈

---

## 6. 决策建议

### 如果优先考虑「快速上线、最低风险」：

**选方案一**。只改 `launch_server.py` 一个文件，加 ~80 行代码，不改客户端。缓存透明，出错 fallback 到原始调用。可以在 30 分钟内实现并测试。

### 如果优先考虑「正式训练的长期性能」：

**选方案二**。消除 Tool Server 单点瓶颈，状态迁移到 agent_data，跨进程缓存天然支持多 worker 并行。架构更干净，运维更简单。但需要 2-4 小时的开发和测试。

### 渐进式路径（推荐）：

1. **先做方案一**（30 分钟）→ 立即获得 60-80% 的 API 调用减少 + single-flight 去重
2. **再做方案二**（2-4 小时）→ 消除 HTTP 层和单点瓶颈
3. **最后扩 reward server**（30 分钟）→ 避免 reward 成为新瓶颈

方案一和方案二不冲突。方案一的缓存逻辑（LRUCache + single-flight）可以直接复用到方案二中，只是缓存后端从进程内 dict 换成 diskcache。
