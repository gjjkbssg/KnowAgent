"""
Agent ReAct 引擎

参考同类知识库系统的 internal/agent/engine.go，实现 Think → Analyze → Act → Observe 循环。

核心流程：
1. 构建系统 prompt（含工具说明）
2. 循环调用 LLM（带 function calling）
3. 如果 LLM 返回 tool_calls → 执行工具（支持并行） → 结果追加到上下文 → 继续循环
4. 如果 LLM 返回纯文本 → 作为最终回答
5. 达到 max_iterations 或卡死检测 → 强制结束
6. 上下文窗口管理：Consolidator（LLM 摘要）+ CompressContext（滑动窗口）
"""

import json
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from .agent_tools import DEFAULT_ALLOWED_TOOLS, ToolResult, ToolRegistry, get_tool_registry
from .context_manager import estimate_tokens, estimate_messages_tokens, manage_context_window, sort_tools_for_cache

logger = logging.getLogger(__name__)

MAX_REPEATED_TOOL_ROUNDS = 2  # 连续多少轮发出完全相同的工具调用即判定为卡死
MAX_CONSECUTIVE_FAILED_TOOL_ROUNDS = 3
MAX_CONTEXT_TOKENS = 120000  # 约 120K tokens 上限
MAX_TOOL_OUTPUT = 16 * 1024
PARALLEL_TOOL_WORKERS = 4


# ── 数据结构 ─────────────────────────────────────────────────────────
@dataclass
class ToolCallRecord:
    id: str
    name: str
    arguments: dict
    result: ToolResult | None = None

    def to_dict(self) -> dict:
        d = {"id": self.id, "name": self.name, "arguments": self.arguments}
        if self.result:
            d["result"] = self.result.to_dict()
        return d


@dataclass
class AgentStep:
    iteration: int
    thought: str = ""
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    timestamp: float = 0.0

    def to_dict(self) -> dict:
        d = {"iteration": self.iteration, "thought": self.thought}
        if self.tool_calls:
            d["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        d["timestamp"] = self.timestamp
        return d


MAX_AGENT_REFERENCES = 10


class ReferenceCollector:
    """
    收集一次 Agent 执行（含它派出的子 Agent）中工具检索到的 Chunk，作为答案的结构化引用。
    子 Agent 在各自线程里运行，所以需要加锁。
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._refs: dict[str, dict] = {}

    def add(self, data) -> None:
        if not isinstance(data, list):
            return
        with self._lock:
            for item in data:
                if isinstance(item, dict) and item.get("chunk_id") and item.get("content"):
                    self._refs.setdefault(item["chunk_id"], item)

    def for_answer(self, answer: str, limit: int = MAX_AGENT_REFERENCES) -> list[dict]:
        """按 Chunk 去重后的引用；答案里点名的文档排在前面，其余按检索先后。"""
        with self._lock:
            refs = list(self._refs.values())
        cited = [ref for ref in refs if ref.get("knowledge_title") and ref["knowledge_title"] in (answer or "")]
        cited_ids = {ref["chunk_id"] for ref in cited}
        return (cited + [ref for ref in refs if ref["chunk_id"] not in cited_ids])[:limit]


@dataclass
class AgentResult:
    content: str
    steps: list[AgentStep]
    total_iterations: int
    duration_ms: int
    stopped_reason: str = "completed"
    references: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "steps": [s.to_dict() for s in self.steps],
            "total_iterations": self.total_iterations,
            "duration_ms": self.duration_ms,
            "stopped_reason": self.stopped_reason,
        }


# ── 优雅降级：从工具结果综合答案 ────────────────────────────────────
# 参考同类知识库系统的 finalize.go：LLM 失败但有工具结果时综合生成

def _synthesize_from_tool_results(steps: list, query: str) -> str:
    """
    从工具结果中综合答案。
    当 LLM 调用失败但已有工具结果时使用。
    """
    # 收集所有工具结果
    tool_outputs = []
    for step in steps:
        for tc in step.tool_calls:
            if tc.result and tc.result.output:
                tool_outputs.append(f"[{tc.name}] {tc.result.output[:500]}")

    if not tool_outputs:
        return "抱歉，无法生成回答。请稍后重试。"

    # 构建简单的综合答案
    context = "\n\n".join(tool_outputs[:5])  # 最多取 5 个结果
    return f"根据检索到的信息：\n\n{context}\n\n以上是与您的问题相关的内容。"


# ── 系统 Prompt 模板 ─────────────────────────────────────────────────
# 设计原则：共享不可变前缀，兼容 DeepSeek V4 自动前缀缓存
# 静态前缀（所有 Agent 共享，可缓存）+ 动态上下文（每个会话不同）

SYSTEM_PROMPT_STATIC_PREFIX = """你是一个知识库问答助手，能够使用工具来帮助回答问题。

## 回答策略

### 第一步：搜索 Wiki 页面
Wiki 页面是预合成的结构化知识，包含文档之间的关系、对比和总结。
- 使用 wiki_search 搜索相关 Wiki 页面
- 使用 wiki_read_page 读取最相关的 Wiki 页面全文
- Wiki 页面通常能直接回答大部分问题

### 第二步：补充原始文档检索
如果 Wiki 内容不足以回答问题，再使用原始文档检索：
- 使用 knowledge_search 搜索相关文档片段
- 使用 grep_chunks 在已检索内容中查找特定信息
- 使用 get_document_info 获取文档元信息

### 第三步：综合分析与回答
- 综合 Wiki 页面和原始文档的信息
- 给出有组织、有深度的回答

## 回答质量要求
- **结构化**：使用标题、列表、表格组织信息
- **有深度**：不仅列出内容，还要分析关系、对比、演进
- **有引用**：引用具体文档标题和 Wiki 页面
- **有综合**：将多个来源的信息整合成连贯的分析

## 效率原则
- **并行调用**：多个独立的工具调用可以同时执行
- **正则合并**：grep_chunks 中用 `|` 合并多个关键词，不要分开调用
- **深度阅读**：搜索到结果后，必须读取完整内容再回答，不要仅凭摘要
- **见好就收**：如果已有足够信息，直接回答，不要再搜索"""

SYSTEM_PROMPT_WITH_TOOLS = SYSTEM_PROMPT_STATIC_PREFIX + """

## 可用工具
{tools_desc}

{custom_prompt}"""

SYSTEM_PROMPT_NO_TOOLS = """你是一个知识库问答助手。请根据提供的知识库上下文回答用户问题。
- 优先使用上下文中的信息回答
- 引用具体来源时注明文档标题
- 如果上下文中没有相关信息，如实说明

{custom_prompt}"""


# ── ReAct 引擎 ───────────────────────────────────────────────────────
class AgentEngine:
    def __init__(
        self,
        tenant,
        session_id: str,
        user_id: str = "",
        agent_config: dict | None = None,
    ):
        self.tenant = tenant
        self.session_id = session_id
        self.user_id = user_id
        self.config = agent_config or {}
        self.registry = get_tool_registry()

        self.max_iterations = self.config.get("max_rounds", 20)  # 默认 20 轮，参考同类知识库系统
        self.temperature = self.config.get("temperature", 0.7)
        self.custom_system_prompt = self.config.get("system_prompt", "")
        self.allowed_tools = self.config.get("allowed_tools", DEFAULT_ALLOWED_TOOLS)
        if not self.config.get("allow_actor_tool", True):
            self.allowed_tools = [tool for tool in self.allowed_tools if tool != "actor"]
        self._granted_tools = {tool["function"]["name"] for tool in self.registry.to_openai_tools(self.allowed_tools)}
        self.model_id = self.config.get("model_id", "")
        self.parallel_tools = self.config.get("parallel_tool_calls", True)
        self.actor_id = self.config.get("actor_id", "main")
        self.parent_message_id = self.config.get("parent_message_id", "")
        self.cancel_check = self.config.get("cancel_check")

    def _build_system_prompt(self) -> str:
        tools = self.registry.to_openai_tools(self.allowed_tools)
        if not tools:
            return SYSTEM_PROMPT_NO_TOOLS.format(custom_prompt=self.custom_system_prompt)

        tools_desc_lines = []
        for t in tools:
            fn = t["function"]
            tools_desc_lines.append(f"- **{fn['name']}**: {fn['description']}")
        tools_desc = "\n".join(tools_desc_lines)

        return SYSTEM_PROMPT_WITH_TOOLS.format(
            tools_desc=tools_desc,
            custom_prompt=self.custom_system_prompt,
        )

    def _build_context(self) -> dict:
        from .models import KnowledgeBase
        kb_ids = self.config.get("knowledge_base_ids", [])
        if not kb_ids:
            # 参考同类知识库系统：过滤系统内部知识库（is_temporary=True），避免 __chat_history__ 暴露给用户
            kb_ids = list(KnowledgeBase.objects.filter(tenant=self.tenant, deleted_at__isnull=True, is_temporary=False).values_list("id", flat=True))
        return {
            "tenant": self.tenant,
            "tenant_id": self.tenant.id,
            "kb_ids": kb_ids,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "actor_id": self.actor_id,
            "parent_message_id": self.parent_message_id,
            "allow_actor_tool": bool(self.config.get("allow_actor_tool", True)),
            "model_id": self.model_id,
            # 本次执行内的已见内容去重；不能跨会话或跨子 Agent 共享
            "seen_chunks": set(),
            "seen_wiki_slugs": set(),
            "references": self._references,
        }

    def _with_retry(self, call):
        from django.conf import settings

        from .retry_policy import is_retryable, retry_delay

        max_attempts = max(1, int(settings.AGENT_LLM_MAX_ATTEMPTS))
        for attempt in range(max_attempts):
            try:
                return call()
            except Exception as exc:
                if not is_retryable(exc) or attempt + 1 >= max_attempts:
                    raise
                delay = retry_delay(exc, attempt)
                logger.warning(
                    "Agent LLM call failed (attempt %s/%s, status=%s, error=%s), retrying in %.3fs",
                    attempt + 1,
                    max_attempts,
                    getattr(exc, "status_code", None),
                    type(exc).__name__,
                    delay,
                )
                remaining = delay
                while remaining > 0:
                    if callable(self.cancel_check) and self.cancel_check():
                        raise
                    sleep_for = min(0.5, remaining)
                    time.sleep(sleep_for)
                    remaining -= sleep_for
                if callable(self.cancel_check) and self.cancel_check():
                    raise

    def _call_llm_with_tools(self, messages: list[dict]) -> dict:
        """
        调用 LLM，支持重试。
        工具定义按字母排序，确保字节级稳定性（兼容 DeepSeek V4 自动前缀缓存）。
        """
        from .model_providers import chat_completion_raw
        tools = sort_tools_for_cache(self.registry.to_openai_tools(self.allowed_tools))
        return self._with_retry(
            lambda: chat_completion_raw(
                self.tenant,
                messages,
                model_id=self.model_id,
                tools=tools if tools else None,
                temperature=self.temperature,
            )
        )

    def _call_llm_simple(self, messages: list[dict]) -> str:
        from .model_providers import chat_completion
        return self._with_retry(lambda: chat_completion(self.tenant, messages, model_id=self.model_id))

    def _run_tool(self, tool_name: str, args: dict, context: dict) -> ToolResult:
        """执行工具前校验白名单：模型只能调用本 Agent 被授予的工具。"""
        if tool_name not in self._granted_tools:
            return ToolResult(
                output="",
                error=f"Tool not allowed for this agent: {tool_name}. Available tools: {sorted(self._granted_tools)}",
            )
        result = self.registry.execute_tool(tool_name, args, context)
        collector = context.get("references")
        if collector is not None:
            collector.add(result.data)
        return result

    def _execute_single_tool(self, tc: dict, context: dict) -> tuple[str, ToolCallRecord]:
        """执行单个工具调用。"""
        tc_id = tc.get("id", "")
        fn = tc.get("function", {})
        tool_name = fn.get("name", "")
        raw_args = fn.get("arguments", "{}")

        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except json.JSONDecodeError:
            args = {}

        tool_result = self._run_tool(tool_name, args, context)
        record = ToolCallRecord(id=tc_id, name=tool_name, arguments=args, result=tool_result)
        return tc_id, record

    @staticmethod
    def _tool_round_signature(tool_calls: list[dict]) -> tuple | None:
        """
        一轮工具调用的规范化签名（工具名 + 参数），用于检测原地打转。
        轮询后台子 Agent（actor status/wait）本来就需要重复调用，返回 None 表示不参与检测。
        """
        signature = []
        for tc in tool_calls:
            fn = tc.get("function", {})
            raw_args = fn.get("arguments", "{}")
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                normalized = json.dumps(args, sort_keys=True, ensure_ascii=False)
            except (json.JSONDecodeError, TypeError):
                args, normalized = {}, str(raw_args)
            if fn.get("name") == "actor" and isinstance(args, dict) and args.get("action") in {"status", "wait"}:
                return None
            signature.append((fn.get("name", ""), normalized))
        return tuple(sorted(signature))

    def _answer_without_tools(self, messages: list[dict], steps: list, reason: str = "stuck") -> str:
        """强制模型停止调用工具并基于已有信息作答。"""
        if reason == "tool_failures":
            nudge_content = "工具调用连续失败。请停止调用工具，基于已有信息直接给出最终回答；信息不足的部分请明确说明。"
        else:
            nudge_content = "你在重复调用相同的工具且没有新信息。请停止调用工具，基于已获得的信息直接给出最终回答；信息不足的部分请明确说明。"
        nudge = {"role": "user", "content": nudge_content}
        try:
            content = self._call_llm_simple([*messages, nudge])
        except Exception:
            logger.exception("[Agent] Forced final answer failed")
            content = ""
        return content.strip() or _synthesize_from_tool_results(steps, "")

    def execute(
        self,
        query: str,
        history: list[dict] | None = None,
        context_str: str = "",
        on_event: callable = None,
    ) -> AgentResult:
        # 子 Agent 复用主 Agent 的收集器，检索证据才能回到最终答案的引用里
        self._references = self.config.get("reference_collector") or ReferenceCollector()
        result = self._execute(query, history, context_str, on_event)
        result.references = self._references.for_answer(result.content)
        return result

    def _execute(
        self,
        query: str,
        history: list[dict] | None,
        context_str: str,
        on_event: callable,
    ) -> AgentResult:
        from .observability import trace_agent_execution, trace_llm_call, trace_tool_execution

        start_time = time.monotonic()
        steps: list[AgentStep] = []
        context = self._build_context()
        system_prompt = self._build_system_prompt()

        if context_str:
            user_content = f"{context_str}\n\n<user_question>\n{query}\n</user_question>"
        else:
            user_content = query

        messages = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_content})

        tools_available = bool(self._granted_tools)
        previous_round_signature = None
        repeated_rounds = 1
        consecutive_failed_tool_rounds = 0
        final_content = ""

        # 顶层 Agent 追踪
        with trace_agent_execution(
            session_id=self.session_id,
            user_id=self.user_id,
            query=query[:2000],
            agent_mode=self.config.get("agent_mode", ""),
        ) as agent_trace:
            agent_trace.metadata["tenant_id"] = self.tenant.id
            agent_trace.metadata["max_iterations"] = self.max_iterations
            agent_trace.metadata["allowed_tools"] = self.allowed_tools

            for iteration in range(1, self.max_iterations + 1):
                if callable(self.cancel_check) and self.cancel_check():
                    return AgentResult(
                        content=final_content or "生成已取消。",
                        steps=steps,
                        total_iterations=iteration - 1,
                        duration_ms=int((time.monotonic() - start_time) * 1000),
                        stopped_reason="cancelled",
                    )
                step = AgentStep(iteration=iteration, timestamp=time.time())

                # 上下文窗口管理（参考同类知识库系统的 manageContextWindow）
                # 1. 脱敏历史 KB 结果
                # 2. Consolidator（LLM 摘要压缩，token > 50% 时触发）
                # 3. 滑动窗口截断（token > 80% 时触发）
                messages = manage_context_window(
                    messages,
                    max_tokens=MAX_CONTEXT_TOKENS,
                    llm_caller=self._call_llm_simple if iteration > 1 else None,
                    enable_redact=True,
                )

                try:
                    # 每轮 LLM 调用追踪
                    with trace_llm_call(agent_trace, model=self.model_id or "default", messages=messages, tools=self.registry.to_openai_tools(self.allowed_tools) if tools_available else None) as llm_span:
                        if tools_available:
                            llm_response = self._call_llm_with_tools(messages)
                            content = llm_response.get("content", "") or ""
                            tool_calls = llm_response.get("tool_calls")
                            llm_span["content"] = content
                            llm_span["tool_calls"] = tool_calls
                        else:
                            content = self._call_llm_simple(messages)
                            tool_calls = None
                            llm_span["content"] = content
                except Exception as e:
                    logger.exception(f"Agent LLM call failed at iteration {iteration}")
                    # 优雅降级：如果有工具结果，尝试综合答案
                    # 参考同类知识库系统的 finalize.go：LLM 失败但有工具结果时综合生成
                    if steps and any(step.tool_calls for step in steps):
                        final_content = _synthesize_from_tool_results(steps, query)
                        logger.info(f"[Agent] Graceful degradation: synthesized answer from {len(steps)} steps")
                        return AgentResult(content=final_content, steps=steps, total_iterations=iteration - 1, duration_ms=int((time.monotonic() - start_time) * 1000), stopped_reason="degraded")
                    final_content = final_content or f"抱歉，处理过程中出现错误：{str(e)}"
                    return AgentResult(content=final_content, steps=steps, total_iterations=iteration - 1, duration_ms=int((time.monotonic() - start_time) * 1000), stopped_reason="error")

                step.thought = content
                if on_event and content:
                    on_event("thinking", {"iteration": iteration, "content": content})

                # ── 无工具调用 → 最终回答 ─────────────────────────────
                if not tool_calls:
                    # 空内容重试（参考同类知识库系统的 emptyContent 检测）
                    if not content.strip() and iteration < self.max_iterations:
                        logger.warning(f"[Agent] Empty content at iteration {iteration}, retrying with nudge")
                        messages.append({"role": "user", "content": "请提供你的完整回答。"})
                        continue

                    final_content = content
                    steps.append(step)
                    return AgentResult(content=final_content, steps=steps, total_iterations=iteration, duration_ms=int((time.monotonic() - start_time) * 1000), stopped_reason="completed")

                # ── 卡死检测：连续多轮发出完全相同的工具调用 ────────────
                round_signature = self._tool_round_signature(tool_calls)
                is_repeat = round_signature is not None and round_signature == previous_round_signature
                repeated_rounds = repeated_rounds + 1 if is_repeat else 1
                previous_round_signature = round_signature
                if repeated_rounds >= MAX_REPEATED_TOOL_ROUNDS:
                    logger.warning(f"[Agent] Identical tool calls repeated {repeated_rounds} rounds at iteration {iteration}, forcing final answer")
                    final_content = self._answer_without_tools(messages, steps, reason="stuck")
                    return AgentResult(content=final_content, steps=steps, total_iterations=iteration, duration_ms=int((time.monotonic() - start_time) * 1000), stopped_reason="stuck")

                # ── 有工具调用 → 执行工具（支持并行）─────────────────
                assistant_msg = {"role": "assistant", "content": content, "tool_calls": tool_calls}
                messages.append(assistant_msg)

                if self.parallel_tools and len(tool_calls) > 1:
                    # 并行执行
                    tool_results = {}
                    with ThreadPoolExecutor(max_workers=PARALLEL_TOOL_WORKERS) as executor:
                        futures = {executor.submit(self._execute_single_tool, tc, context): tc for tc in tool_calls}
                        for future in as_completed(futures):
                            try:
                                tc_id, record = future.result()
                                tool_results[tc_id] = record
                            except Exception as exc:
                                logger.exception("Parallel tool execution failed")

                    for tc in tool_calls:
                        tc_id = tc.get("id", "")
                        record = tool_results.get(tc_id)
                        if record:
                            step.tool_calls.append(record)
                            if on_event:
                                on_event("tool_call", {"iteration": iteration, "name": record.name, "arguments": record.arguments, "tool_call_id": tc_id})
                                on_event("tool_result", {"iteration": iteration, "name": record.name, "output": record.result.output[:500] if record.result else "", "error": record.result.error if record.result else "", "duration_ms": record.result.duration_ms if record.result else 0, "tool_call_id": tc_id})
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tc_id,
                                "name": record.name,
                                "content": (record.result.output[:MAX_TOOL_OUTPUT] if not record.result.error else f"Error: {record.result.error}") if record.result else "Error: No result",
                            })
                else:
                    # 顺序执行
                    for tc in tool_calls:
                        tc_id = tc.get("id", "")
                        fn = tc.get("function", {})
                        tool_name = fn.get("name", "")
                        raw_args = fn.get("arguments", "{}")

                        try:
                            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                        except json.JSONDecodeError:
                            args = {}

                        if on_event:
                            on_event("tool_call", {"iteration": iteration, "name": tool_name, "arguments": args, "tool_call_id": tc_id})

                        tool_result = self._run_tool(tool_name, args, context)
                        record = ToolCallRecord(id=tc_id, name=tool_name, arguments=args, result=tool_result)
                        step.tool_calls.append(record)

                        if on_event:
                            on_event("tool_result", {"iteration": iteration, "name": tool_name, "output": tool_result.output[:500], "error": tool_result.error, "duration_ms": tool_result.duration_ms, "tool_call_id": tc_id})

                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc_id,
                            "name": tool_name,
                            "content": tool_result.output[:MAX_TOOL_OUTPUT] if not tool_result.error else f"Error: {tool_result.error}",
                        })

                steps.append(step)
                if any(record.result is not None and not record.result.error for record in step.tool_calls):
                    consecutive_failed_tool_rounds = 0
                else:
                    consecutive_failed_tool_rounds += 1
                if consecutive_failed_tool_rounds >= MAX_CONSECUTIVE_FAILED_TOOL_ROUNDS:
                    logger.warning("[Agent] Tool calls failed for %s consecutive rounds, forcing final answer", consecutive_failed_tool_rounds)
                    final_content = self._answer_without_tools(messages, steps, reason="tool_failures")
                    return AgentResult(
                        content=final_content,
                        steps=steps,
                        total_iterations=iteration,
                        duration_ms=int((time.monotonic() - start_time) * 1000),
                        stopped_reason="tool_failures",
                    )

        # 达到最大迭代次数
        return AgentResult(content=final_content or "已达到最大推理轮数，基于当前信息给出回答。", steps=steps, total_iterations=self.max_iterations, duration_ms=int((time.monotonic() - start_time) * 1000), stopped_reason="max_iterations")
