import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import timedelta

from django.conf import settings
from django.db import OperationalError, close_old_connections, connection, transaction
from django.utils import timezone

from personal_knowledge_base.agent_actor import ActorRegistry
from personal_knowledge_base.agent_engine import AgentEngine
from personal_knowledge_base.agent_history import (
    build_agent_context_if_needed,
    build_agent_history_messages,
    build_normal_rag_messages,
    build_rag_history_messages,
    normalize_max_rounds,
)
from personal_knowledge_base.chat_history_kb import index_qa_to_kb_async
from personal_knowledge_base.chat_runtime import (
    run_database_background,
    save_session_state,
    schedule_chat_maintenance,
    schedule_title_generation,
    should_skip_expensive_prefetch,
)
from personal_knowledge_base.context_snapshot import (
    build_agent_history_with_snapshot,
    build_rag_history_with_snapshot,
    refresh_context_snapshot_async,
)
from personal_knowledge_base.memory import add_episode as memory_add_episode, is_memory_available, retrieve_memory
from personal_knowledge_base.model_providers import ModelConfigurationError, chat_completion, chat_completion_stream
from personal_knowledge_base.models import AgentToolApproval, KnowledgeBase, Message, Session, TaskRecord, User
from personal_knowledge_base.query_understand import INTENT_KB_SEARCH, understand_query
from personal_knowledge_base.serializers import message_dict
from personal_knowledge_base.stream_manager import stream_manager
from personal_knowledge_base.tasks import RESUME_APPROVAL_TASK, RESUME_FINISHED, RESUME_STARTED, submit_approval_resume
from services.context import ServiceContext
from services.errors import Conflict, InvalidInput, ServiceError, get_or_404

logger = logging.getLogger(__name__)


MULTI_AGENT_SYSTEM_PROMPT = """你是多 Agent 知识工作台的主 Agent。你可以直接回答简单问题，也可以通过 actor 工具把任务交给专业子 Agent。

可用子 Agent：
- doc_retriever：检索原始文档 chunk，适合需要精确证据、引用和关键词定位的问题。
- wiki_researcher：检索 Wiki 页面，适合需要结构化知识、全局脉络和概念页面的问题。
- graph_reasoner：查询知识图谱，适合实体关系、多跳关系和关联路径推理。
- answer_writer：整理多个子 Agent 的结果，适合把多路证据合成为最终回答草稿。

决策规则：
- 简单问候或无需知识库、文件、Wiki、图谱、互联网的问题，直接简短回答，不要调用工具。
- 涉及当前知识库、文档事实、引用来源、附件内容时，必须使用 actor.run(doc_retriever)，不要凭空回答。
- 涉及全局结构、概念网络、页面脉络时，使用 actor.run(wiki_researcher)。
- 涉及实体关系、上下游、影响链路、多跳关系时，使用 actor.run(graph_reasoner)。
- 多路结果复杂或需要形成最终稿时，再使用 answer_writer；不要为了简单问题滥用子 Agent。
- 子 Agent 结果足够后，综合回答，不要继续无意义调用工具。
"""

def apply_multi_agent_defaults(agent_config: dict, data: dict, kb_ids: list, parent_message_id: str = "") -> dict:
    config = dict(agent_config or {})
    config["agent_mode"] = "multi-agent"
    config["system_prompt"] = MULTI_AGENT_SYSTEM_PROMPT
    config["allowed_tools"] = ["actor", "thinking"]
    config["model_id"] = data.get("model_id", config.get("model_id", ""))
    config["knowledge_base_ids"] = kb_ids
    config["temperature"] = config.get("temperature", 0.7)
    config["max_rounds"] = normalize_max_rounds(config.get("max_rounds", 8), default=8, maximum=20)
    config["actor_id"] = "main"
    config["allow_actor_tool"] = True
    if parent_message_id:
        config["parent_message_id"] = parent_message_id
    return config

def _append_mcp_service_errors(context: str, errors: list[dict]) -> str:
    if not errors:
        return context or ""
    lines = [
        f"以下 MCP 服务不可用：{error.get('service_name') or 'MCP service'}（{str(error.get('error') or '')[:120]}）"
        for error in errors
    ]
    return "\n".join([part for part in [context or "", *lines] if part])

def _persist_assistant_content_with_retry(
    ctx: ServiceContext,
    assistant_id,
    content: str,
    max_attempts: int = 5,
) -> bool:
    """Persist streamed content without turning DB write failures into model fallbacks."""
    for attempt in range(max_attempts):
        try:
            close_old_connections()
            updated = Message.objects.filter(id=assistant_id, session__tenant=ctx.tenant).update(
                content=content,
                rendered_content=content,
                is_completed=True,
                updated_at=timezone.now(),
            )
            return bool(updated)
        except OperationalError as exc:
            if "locked" not in str(exc).lower() or attempt == max_attempts - 1:
                logger.exception("Persist streamed assistant content failed")
                return False
            time.sleep(0.05 * (attempt + 1))
    return False

def _build_document_header(refs: list[dict]) -> str:
    """构建文档头部 XML，列出所有涉及的知识条目及其描述。参考同类知识库系统的 buildDocumentHeader。"""
    seen = set()
    docs = []
    for r in refs:
        kid = r.get("knowledge_id", "")
        if kid in seen:
            continue
        seen.add(kid)
        title = r.get("knowledge_title", "")
        if not title:
            continue
        desc = r.get("knowledge_description", "").strip()
        if desc:
            docs.append(f"<document>\n<title>{title}</title>\n<description>{desc}</description>\n</document>")
        else:
            docs.append(f"<document>\n<title>{title}</title>\n</document>")
    if not docs:
        return ""
    return "<documents>\n" + "\n".join(docs) + "\n</documents>"

def _build_structured_context(refs: list[dict]) -> str:
    """构建结构化 XML context，每个 chunk 带编号。参考同类知识库系统的 into_chat_message。"""
    parts = []
    for i, r in enumerate(refs[:5], 1):
        content = r.get("content", "").strip()
        if content:
            parts.append(f'<context id="{i}">{content}</context>')
    return "\n".join(parts)

def _build_context_with_memory(refs: list[dict], memory_str: str, kb_names: str = "") -> str:
    """组装完整的上下文：知识库信息 + 文档头 + 结构化 context + 记忆。"""
    parts = []
    if kb_names:
        parts.append(f"<knowledge_base>\n{kb_names}\n</knowledge_base>")
    doc_header = _build_document_header(refs)
    if doc_header:
        parts.append(doc_header)
    structured = _build_structured_context(refs)
    if structured:
        parts.append(structured)
    if memory_str:
        parts.append(memory_str)
    return "\n\n".join(parts)

def _save_session_after_chat(ctx: ServiceContext, session, data, kb_ids, query):
    """保存 session 配置；标题生成放到后台，避免阻塞首答。"""
    try:
        save_session_state(session, data, kb_ids, query)
        schedule_title_generation(
            str(session.id),
            query,
            tenant_id=str(ctx.tenant.id) if ctx.tenant else None,
            model_id=data.get("model_id", ""),
        )
    finally:
        # 关闭数据库连接以释放锁
        from django.db import connection
        try:
            connection.close()
        except Exception:
            pass

APPROVAL_CONFIG_FIELDS = {
    "allowed_tools", "model_id", "temperature", "system_prompt", "max_rounds",
    "parallel_tool_calls", "agent_mode", "knowledge_base_ids", "actor_id",
    "parent_message_id", "allow_actor_tool",
}

def _persist_agent_result(ctx: ServiceContext, result, *, assistant_msg_id, stream, user_id, user,
                          user_msg_id, session_id, query, agent_config, mcp_service_ids,
                          enable_memory, enable_chat_history, enable_snapshot, resume_task=None):
    steps = [step.to_dict() for step in result.steps]
    refs = result.references
    awaiting_approval = getattr(result, "stopped_reason", "completed") == "awaiting_approval"
    if awaiting_approval:
        from django.conf import settings

        checkpoint = dict(getattr(result, "checkpoint", None) or {})
        checkpoint["chat"] = {
            "agent_config": {key: value for key, value in agent_config.items() if key in APPROVAL_CONFIG_FIELDS},
            "mcp_service_ids": list(mcp_service_ids or []),
            "user_id": user_id,
            "user_message_id": user_msg_id,
            "enable_memory": bool(enable_memory),
            "enable_chat_history": bool(enable_chat_history),
            "enable_snapshot": bool(enable_snapshot),
            "query": query,
            "session_id": session_id,
        }
        expires_at = timezone.now() + timedelta(seconds=settings.AGENT_APPROVAL_TTL_SECONDS)
    with transaction.atomic():
        messages = Message.objects.filter(id=assistant_msg_id)
        if resume_task is not None:
            if not resume_task.filter(result=RESUME_STARTED).update(result=RESUME_FINISHED, updated_at=timezone.now()):
                return
            messages = messages.filter(is_completed=False)
        if not messages.update(
            content=result.content,
            rendered_content=result.content,
            knowledge_references=refs,
            agent_steps=steps,
            agent_duration_ms=result.duration_ms,
            is_completed=not awaiting_approval,
            updated_at=timezone.now(),
        ):
            return
        if awaiting_approval:
            approval = AgentToolApproval.objects.create(
                tenant=ctx.tenant,
                session_id=session_id,
                message_id=assistant_msg_id,
                tool_calls=getattr(result, "pending_approval", []),
                checkpoint=checkpoint,
                expires_at=expires_at,
            )

    if awaiting_approval:
        if stream:
            stream.set_final_result(result.content, refs, steps, result.duration_ms)
            stream.append_event("approval_required", {
                "approval": {
                    "id": approval.id,
                    "status": approval.status,
                    "tool_calls": approval.tool_calls,
                    "expires_at": approval.expires_at.isoformat(),
                }
            })
        return approval

    if stream:
        stream.set_final_result(result.content, refs, steps, result.duration_ms)
        stream.append_event("complete", {"done": True, "content": result.content})
    schedule_chat_maintenance(
        tenant=ctx.tenant,
        user_message_id=user_msg_id,
        assistant_message_id=assistant_msg_id,
        session_id=session_id,
        query=query,
        answer=result.content,
        mode="agent",
        max_rounds=agent_config.get("max_rounds", 5),
        model_id=agent_config.get("model_id", ""),
        enable_memory=enable_memory and bool(user),
        user_id=user_id,
        enable_chat_history=enable_chat_history,
        enable_snapshot=enable_snapshot,
        indexer=index_qa_to_kb_async,
        snapshot_refresher=refresh_context_snapshot_async,
    )
    return None

def _run_agent_generation(
    ctx: ServiceContext,
    assistant_msg_id: str,
    session_id: str,
    user_msg_id: str,
    query: str,
    history_msgs: list,
    agent_context: str,
    agent_config: dict,
    refs: list,
    user_id: str,
    enable_memory: bool,
    user=None,
    enable_chat_history: bool = True,
    enable_snapshot: bool = True,
    mcp_service_ids: list | None = None,
):
    """
    在独立线程中运行 Agent 生成。
    事件通过 StreamManager 持久化，不依赖 SSE 连接。
    即使客户端断开，生成也会继续完成。
    """
    stream = stream_manager.get_stream(assistant_msg_id) or stream_manager.create_stream(assistant_msg_id, session_id)
    toolset = None

    try:
        from personal_knowledge_base.mcp_client import load_mcp_toolset

        toolset = load_mcp_toolset(ctx.tenant, mcp_service_ids)
        runtime_agent_config = dict(agent_config)
        runtime_agent_config["extra_tools"] = toolset.tools
        agent_context = _append_mcp_service_errors(agent_context, toolset.errors)
        engine = AgentEngine(
            tenant=ctx.tenant,
            session_id=session_id,
            user_id=user_id,
            agent_config=runtime_agent_config,
        )

        collected_content = []
        last_saved_content = {"text": ""}

        def on_event(event_type, event_data):
            collected_content.append((event_type, event_data))
            # 存入 StreamManager
            stream.append_event(event_type, event_data)
            # 定期保存中间内容到数据库
            if event_type == "thinking":
                content = event_data.get("content", "")
                if content and len(content) > len(last_saved_content["text"]):
                    last_saved_content["text"] = content
                    try:
                        Message.objects.filter(id=assistant_msg_id).update(
                            content=content,
                            rendered_content=content,
                            updated_at=timezone.now(),
                        )
                    except Exception:
                        pass

        # 执行 Agent
        result = engine.execute(query, history=history_msgs, context_str=agent_context, on_event=on_event)

        _persist_agent_result(
            ctx,
            result,
            assistant_msg_id=assistant_msg_id,
            stream=stream,
            user_id=user_id,
            user=user,
            user_msg_id=user_msg_id,
            session_id=session_id,
            query=query,
            agent_config=agent_config,
            mcp_service_ids=mcp_service_ids,
            enable_memory=enable_memory,
            enable_chat_history=enable_chat_history,
            enable_snapshot=enable_snapshot,
        )

        logger.info(f"[Agent] Generation completed for message {assistant_msg_id}")

    except Exception as e:
        logger.exception(f"[Agent] Generation failed for message {assistant_msg_id}")
        stream.append_event("error", {"content": str(e)})
        # 标记消息为完成（带错误）
        try:
            Message.objects.filter(id=assistant_msg_id).update(
                content=f"生成失败: {e}",
                rendered_content=f"生成失败: {e}",
                is_completed=True,
                updated_at=timezone.now(),
            )
        except Exception:
            pass
    finally:
        if toolset:
            toolset.close()
        # 关闭数据库连接以释放锁
        from django.db import connection
        try:
            connection.close()
        except Exception:
            pass

RESUME_INTERRUPTED = "审批后的生成已中断，无法确认工具是否已执行。为避免重复执行，本次操作不会重试。"

def _persist_resume_error(ctx: ServiceContext, approval_id: str, content: str, resume_task=None):
    with transaction.atomic():
        if resume_task is not None and not resume_task.exclude(result=RESUME_FINISHED).update(
            result=RESUME_FINISHED, updated_at=timezone.now(),
        ):
            return False
        return bool(Message.objects.filter(
            tool_approvals__id=approval_id,
            tool_approvals__tenant=ctx.tenant,
            tool_approvals__status__in=("approved", "rejected"),
            is_completed=False,
        ).update(content=content, rendered_content=content, is_completed=True, updated_at=timezone.now()))


def _stream_event_frame_count(event) -> int:
    event_type = event.event_type
    if event_type not in {"thinking", "tool_call", "tool_result", "complete", "error", "approval_required"} and not event_type.startswith("actor_"):
        return 0
    count = 1
    if event_type == "complete":
        count += 1
    if event_type in {"complete", "error", "approval_required"}:
        count += 1
    return count


def _resume_agent_generation(ctx: ServiceContext, approval_id: str, resume_task):
    # The worker may retry a database write. Never repeat the Agent invocation
    # once it could have reached a remote tool.
    if resume_task.filter(result=RESUME_STARTED).exists():
        _persist_resume_error(ctx, approval_id, RESUME_INTERRUPTED, resume_task)
        return
    eligible = AgentToolApproval.objects.filter(
        id=approval_id, tenant=ctx.tenant, status__in=("approved", "rejected"), message__is_completed=False,
    )
    approval = eligible.select_related("tenant").first()
    if not approval or not resume_task.filter(result__isnull=True).exists():
        return
    toolset = None
    stream = None
    checkpoint = approval.checkpoint or {}
    chat_state = checkpoint.get("chat", {})
    session_id = str(approval.session_id)
    message_id = str(approval.message_id)
    try:
        from personal_knowledge_base.mcp_client import load_mcp_toolset

        toolset = load_mcp_toolset(approval.tenant, chat_state.get("mcp_service_ids", []))
        agent_config = dict(chat_state.get("agent_config") or {})
        agent_config["extra_tools"] = toolset.tools
        engine = AgentEngine(
            tenant=approval.tenant,
            session_id=session_id,
            user_id=chat_state.get("user_id", ""),
            agent_config=agent_config,
        )
        if not resume_task.filter(result__isnull=True, id__in=eligible.values("id")).update(
            result=RESUME_STARTED, updated_at=timezone.now(),
        ):
            return
        previous_stream = stream_manager.get_stream(message_id)
        first_frame_id = None
        if previous_stream and previous_stream.first_event_frame_id is not None:
            first_frame_id = previous_stream.first_event_frame_id + sum(
                _stream_event_frame_count(event) for event in previous_stream.get_events()
            )
        stream = stream_manager.create_stream(message_id, session_id, first_event_frame_id=first_frame_id)
        result = engine.resume(checkpoint, approval.status, on_event=stream.append_event)
        user = User.objects.filter(id=chat_state.get("user_id", "")).first()
        resume_ctx = ServiceContext(tenant=approval.tenant, user=user, request_id=ctx.request_id)
        _persist_agent_result(
            resume_ctx,
            result,
            assistant_msg_id=message_id,
            stream=stream,
            user_id=chat_state.get("user_id", ""),
            user=user,
            user_msg_id=chat_state.get("user_message_id", ""),
            session_id=session_id,
            query=chat_state.get("query", ""),
            agent_config=agent_config,
            mcp_service_ids=chat_state.get("mcp_service_ids", []),
            enable_memory=chat_state.get("enable_memory", False),
            enable_chat_history=chat_state.get("enable_chat_history", True),
            enable_snapshot=chat_state.get("enable_snapshot", True),
            resume_task=resume_task,
        )
    except OperationalError:
        raise
    except Exception as exc:
        logger.exception("[Agent] Resume failed for approval %s", approval_id)
        if _persist_resume_error(ctx, approval_id, f"生成失败: {exc}", resume_task) and stream:
            stream.append_event("error", {"content": str(exc)})
    finally:
        if toolset:
            toolset.close()

def _quick_intent_detect(query: str) -> str | None:
    """
    快速意图检测：对简单查询用正则规则判断，跳过 LLM 调用。
    参考同类知识库系统的条件跳过设计。

    Returns:
        意图字符串（如果可以快速判断），None（如果需要 LLM 识别）
    """
    q = query.strip()
    # 短问候语
    if len(q) < 10 and any(w in q for w in ["你好", "hello", "hi", "嗨", "您好", "hey"]):
        return "chitchat"
    # 短查询且是问句形式，直接走搜索
    if len(q) < 20 and ("?" in q or "？" in q or q.startswith(("什么", "怎么", "如何", "为什么", "哪", "谁", "几"))):
        return INTENT_KB_SEARCH
    # 很短的查询，直接走搜索
    if len(q) < 8:
        return INTENT_KB_SEARCH
    # 其他情况需要 LLM 识别
    return None

def _safe_understand_query(ctx: ServiceContext, query: str) -> dict | None:
    """线程安全的查询理解包装"""
    try:
        return understand_query(ctx.tenant, query)
    except Exception:
        logger.exception("Query understanding failed in parallel task")
        return None

def _safe_retrieve_memory(ctx: ServiceContext, user_id: str, query: str) -> str:
    """线程安全的记忆检索包装，返回格式化的记忆上下文字符串"""
    try:
        from personal_knowledge_base.chat_history_kb import sanitize_internal_kb_mentions
        mem_ctx = retrieve_memory(ctx.tenant, user_id, query)
        if mem_ctx.related_episodes:
            return "\n\n<relevant_memory>\n" + "\n".join(
                f"- {sanitize_internal_kb_mentions(ctx.tenant, ep.summary)}" for ep in mem_ctx.related_episodes
            ) + "\n</relevant_memory>"
    except Exception:
        logger.exception("Memory retrieval failed in parallel task")
    return ""

def _safe_search_chat_history(ctx: ServiceContext, query: str) -> list[dict]:
    """线程安全的 ChatHistoryKB 检索包装。"""
    try:
        from personal_knowledge_base.chat_history_kb import search_chat_history
        return search_chat_history(ctx.tenant, query, limit=5)
    except Exception:
        logger.exception("ChatHistoryKB retrieval failed in parallel task")
    return []

def _build_kb_names(ctx: ServiceContext, kb_ids: list[str]) -> str:
    if not kb_ids:
        return ""
    kb_list = KnowledgeBase.objects.filter(
        id__in=kb_ids,
        tenant=ctx.tenant,
        deleted_at__isnull=True,
        is_temporary=False,
    ).values("name", "description")
    kb_lines = []
    for kb in kb_list:
        desc = (kb.get("description") or "").strip()
        kb_lines.append(f"- {kb['name']}" + (f"：{desc}" if desc else ""))
    return "当前知识库：\n" + "\n".join(kb_lines) if kb_lines else ""

def _build_agent_prefetch_context(ctx: ServiceContext, query: str, kb_ids: list[str], enable_memory: bool, data: dict) -> tuple[str, str]:
    user = ctx.user
    kb_names_str = _build_kb_names(ctx, kb_ids)
    if should_skip_expensive_prefetch(query, data):
        return kb_names_str, ""

    memory_context_str = ""
    chat_history_context_str = ""
    if connection.vendor == "sqlite":
        if enable_memory and user and is_memory_available():
            memory_context_str = _safe_retrieve_memory(ctx, str(user.id), query) or ""
        try:
            from personal_knowledge_base.chat_history_kb import format_chat_history_context, is_chat_history_enabled
            if user and is_chat_history_enabled(ctx.tenant):
                history_results = _safe_search_chat_history(ctx, query)
                if history_results:
                    chat_history_context_str = format_chat_history_context(history_results, tenant=ctx.tenant)
        except Exception:
            logger.exception("ChatHistoryKB config check failed")
    else:
        with ThreadPoolExecutor(max_workers=2) as stage_pool:
            future_memory = None
            if enable_memory and user and is_memory_available():
                future_memory = stage_pool.submit(_safe_retrieve_memory, ctx, str(user.id), query)

            future_chat_history = None
            try:
                from personal_knowledge_base.chat_history_kb import format_chat_history_context, is_chat_history_enabled
                chat_history_enabled = is_chat_history_enabled(ctx.tenant)
            except Exception:
                logger.exception("ChatHistoryKB config check failed")
                chat_history_enabled = False
                format_chat_history_context = None
            if user and chat_history_enabled:
                future_chat_history = stage_pool.submit(_safe_search_chat_history, ctx, query)

            if future_memory is not None:
                memory_context_str = future_memory.result() or ""

            if future_chat_history is not None:
                history_results = future_chat_history.result()
                if history_results and format_chat_history_context:
                    chat_history_context_str = format_chat_history_context(history_results, tenant=ctx.tenant)

    hidden_context = "\n\n".join(part for part in [memory_context_str, chat_history_context_str] if part)
    return kb_names_str, hidden_context

def _async_memory_store(ctx: ServiceContext, user_id: str, session_id: str, query: str, answer: str):
    """异步记忆存储，不阻塞主流程"""
    try:
        memory_add_episode(
            ctx.tenant, user_id, session_id,
            [{"role": "user", "content": query}, {"role": "assistant", "content": answer}],
        )
    except Exception:
        logger.exception("Async memory store failed")

def local_answer(query: str, refs: list[dict], agent=False):
    if not refs:
        return "很抱歉，我暂时无法在当前知识库中找到相关内容。"
    intro = "我根据知识库检索到了以下相关内容："
    bullets = "\n".join(f"- {r['knowledge_title']}: {r['content'][:180]}" for r in refs[:3])
    return f"{intro}\n{bullets}"


@dataclass(frozen=True)
class ChatTurn:
    message: Message
    payload: dict | None = None


def start_chat_turn(
    ctx: ServiceContext,
    session_id,
    data: dict,
    *,
    agent: bool,
    streaming: bool,
) -> ChatTurn:
    tenant = ctx.tenant
    user = ctx.user
    session = get_or_404(Session, id=session_id, tenant=tenant)
    query = data.get("query", "")
    request_id = ctx.request_id or str(uuid.uuid4())
    images = data.get("images") or []
    attachments = data.get("attachment_uploads") or data.get("attachments") or []
    mentioned_items = data.get("mentioned_items") or []
    user_msg = Message.objects.create(
        session=session,
        request_id=request_id,
        role="user",
        content=query,
        mentioned_items=mentioned_items,
        images=[{"url": img.get("data") or img.get("url", "")} if isinstance(img, dict) else img for img in images],
        attachments=[
            {
                "file_name": item.get("file_name") or item.get("name") or "attachment",
                "file_size": item.get("file_size") or item.get("size") or 0,
            }
            if isinstance(item, dict)
            else item
            for item in attachments
        ],
        is_completed=True,
        channel=data.get("channel", "web"),
    )
    kb_ids = data.get("knowledge_base_ids") or ([session.knowledge_base_id] if session.knowledge_base_id else [])

    enable_memory = data.get("enable_memory")
    if enable_memory is None:
        enable_memory = ((user.preferences if user else None) or {}).get("enable_memory", True)

    if agent:
        refs = []
        service_ids = [str(value) for value in (data.get("mcp_service_ids") or []) if value]
        skip_expensive_prefetch = should_skip_expensive_prefetch(query, data)
        agent_config = apply_multi_agent_defaults({}, data, kb_ids)
        history_msgs = build_agent_history_with_snapshot(
            session=session,
            current_user_message=user_msg,
            max_rounds=agent_config["max_rounds"],
            history_builder=build_agent_history_messages,
        )
        kb_names_str, hidden_context = _build_agent_prefetch_context(
            ctx,
            query,
            kb_ids,
            enable_memory,
            data,
        )
        agent_context = build_agent_context_if_needed(_build_context_with_memory, refs, hidden_context, kb_names_str)

        if streaming:
            assistant = Message.objects.create(
                session=session,
                request_id=request_id,
                role="assistant",
                content="",
                rendered_content="",
                knowledge_references=refs,
                is_completed=False,
                channel=data.get("channel", "web"),
            )
            agent_config = apply_multi_agent_defaults(agent_config, data, kb_ids, parent_message_id=assistant.id)
            _save_session_after_chat(ctx, session, data, kb_ids, query)
            generation_kwargs = {
                "assistant_msg_id": assistant.id,
                "session_id": str(session.id),
                "user_msg_id": str(user_msg.id),
                "query": query,
                "history_msgs": history_msgs,
                "agent_context": agent_context,
                "agent_config": agent_config,
                "refs": refs,
                "user_id": str(user.id) if user else "",
                "enable_memory": enable_memory and not skip_expensive_prefetch,
                "user": user,
                "enable_chat_history": not skip_expensive_prefetch,
                "enable_snapshot": not skip_expensive_prefetch,
                "mcp_service_ids": service_ids,
            }
            stream_manager.create_stream(assistant.id, str(session.id))
            run_database_background(lambda: _run_agent_generation(ctx, **generation_kwargs))
            return ChatTurn(assistant)

        assistant = Message.objects.create(
            session=session,
            request_id=request_id,
            role="assistant",
            content="",
            rendered_content="",
            knowledge_references=refs,
            is_completed=False,
            channel=data.get("channel", "web"),
        )
        agent_config = apply_multi_agent_defaults(agent_config, data, kb_ids, parent_message_id=assistant.id)
        _save_session_after_chat(ctx, session, data, kb_ids, query)
        from personal_knowledge_base.mcp_client import load_mcp_toolset

        toolset = load_mcp_toolset(tenant, service_ids)
        try:
            runtime_agent_config = dict(agent_config)
            runtime_agent_config["extra_tools"] = toolset.tools
            run_context = _append_mcp_service_errors(agent_context, toolset.errors)
            engine = AgentEngine(
                tenant=tenant,
                session_id=str(session.id),
                user_id=str(user.id) if user else "",
                agent_config=runtime_agent_config,
            )
            result = engine.execute(query, history=history_msgs, context_str=run_context)
        finally:
            toolset.close()
        _persist_agent_result(
            ctx,
            result,
            assistant_msg_id=assistant.id,
            stream=None,
            user_id=str(user.id) if user else "",
            user=user,
            user_msg_id=str(user_msg.id),
            session_id=str(session.id),
            query=query,
            agent_config=agent_config,
            mcp_service_ids=service_ids,
            enable_memory=enable_memory and not skip_expensive_prefetch,
            enable_chat_history=not skip_expensive_prefetch,
            enable_snapshot=not skip_expensive_prefetch,
        )
        assistant.refresh_from_db()
        return ChatTurn(
            assistant,
            {"message": message_dict(assistant), "answer": assistant.content, "references": result.references},
        )

    from personal_knowledge_base.rag_pipeline import run_rag_pipeline

    rag_ctx = run_rag_pipeline(
        tenant=tenant,
        query=query,
        kb_ids=kb_ids,
        session=session,
        user=user,
        enable_memory=enable_memory,
        model_id=data.get("model_id", ""),
    )
    intent = rag_ctx.intent
    search_query = rag_ctx.search_query
    refs = rag_ctx.refs
    system_prompt = rag_ctx.system_prompt
    user_prompt = rag_ctx.user_prompt

    if user_prompt != query:
        user_msg.rendered_content = user_prompt
        user_msg.save(update_fields=["rendered_content", "updated_at"])

    rag_max_rounds = normalize_max_rounds(session.max_rounds)
    history_msgs = build_rag_history_with_snapshot(
        session=session,
        current_user_message=user_msg,
        max_rounds=rag_max_rounds,
        history_builder=build_rag_history_messages,
    )
    llm_messages = build_normal_rag_messages(system_prompt, history_msgs, user_prompt)
    model_id = data.get("model_id", "")

    if streaming:
        assistant = Message.objects.create(
            session=session,
            request_id=request_id,
            role="assistant",
            content="",
            rendered_content="",
            knowledge_references=refs,
            agent_steps=[{"type": "knowledge_search", "query": search_query, "intent": intent, "count": len(refs)}],
            is_completed=False,
            channel=data.get("channel", "web"),
        )
        _save_session_after_chat(ctx, session, data, kb_ids, query)
        stream = stream_manager.create_stream(assistant.id, str(session.id))

        def run_normal_generation():
            collected = ""
            try:
                for token in chat_completion_stream(tenant, llm_messages, model_id):
                    collected += token
                    stream.append_event("thinking", {"content": collected})
            except Exception as exc:
                logger.warning(f"Normal stream generation failed: {exc}")
                try:
                    collected = chat_completion(tenant, llm_messages, model_id)
                except Exception:
                    collected = local_answer(query, refs, agent=False)

            stream.set_final_result(content=collected, refs=refs)
            stream.append_event("complete", {"done": True, "content": collected})

            if _persist_assistant_content_with_retry(ctx, assistant.id, collected):
                assistant.content = collected
                assistant.rendered_content = collected
                assistant.is_completed = True
            else:
                logger.error("Assistant message %s was generated but not persisted", assistant.id)

            schedule_chat_maintenance(
                tenant=tenant,
                user_message_id=str(user_msg.id),
                assistant_message_id=str(assistant.id),
                session_id=str(session.id),
                query=query,
                answer=collected,
                mode="rag",
                max_rounds=rag_max_rounds,
                model_id=model_id,
                enable_memory=enable_memory and bool(user),
                user_id=str(user.id) if user else "",
                indexer=index_qa_to_kb_async,
                snapshot_refresher=refresh_context_snapshot_async,
            )

        run_database_background(run_normal_generation)
        return ChatTurn(assistant)

    try:
        answer = chat_completion(tenant, llm_messages, model_id)
    except (ModelConfigurationError, Exception):
        answer = local_answer(query, refs, agent=False)

    assistant = Message.objects.create(
        session=session,
        request_id=request_id,
        role="assistant",
        content=answer,
        rendered_content=answer,
        knowledge_references=refs,
        agent_steps=[{"type": "knowledge_search", "query": search_query, "intent": intent, "count": len(refs)}],
        is_completed=True,
        channel=data.get("channel", "web"),
    )
    schedule_chat_maintenance(
        tenant=tenant,
        user_message_id=str(user_msg.id),
        assistant_message_id=str(assistant.id),
        session_id=str(session.id),
        query=query,
        answer=assistant.content,
        mode="rag",
        max_rounds=rag_max_rounds,
        model_id=model_id,
        enable_memory=enable_memory and bool(user),
        user_id=str(user.id) if user else "",
        indexer=index_qa_to_kb_async,
        snapshot_refresher=refresh_context_snapshot_async,
    )
    run_database_background(lambda: _save_session_after_chat(ctx, session, data, kb_ids, query))
    return ChatTurn(
        assistant,
        {"message": message_dict(assistant), "answer": assistant.content, "references": refs},
    )


def decide_tool_approval(ctx: ServiceContext, session_id, approval_id, data: dict):
    approval = get_or_404(
        AgentToolApproval,
        id=approval_id,
        tenant=ctx.tenant,
        session_id=session_id,
    )
    decision = data.get("decision")
    if decision not in {"approve", "reject"}:
        raise InvalidInput("decision must be approve or reject")

    expired_now = False
    claimed = False
    with transaction.atomic():
        now = timezone.now()
        claimed = AgentToolApproval.objects.filter(
            id=approval.id,
            status="pending",
            expires_at__gt=now,
        ).update(
            status="approved" if decision == "approve" else "rejected",
            decided_by=str(ctx.user.id) if ctx.user else "",
            decided_at=now,
            updated_at=now,
        )
        if not claimed:
            expired = AgentToolApproval.objects.filter(
                id=approval.id,
                status="pending",
                expires_at__lte=now,
            ).update(status="expired", decided_at=now, updated_at=now)
            if expired:
                expired_now = True
                approval.status = "expired"
                Message.objects.filter(id=approval.message_id).update(
                    content="等待确认超时，未执行该操作。",
                    rendered_content="等待确认超时，未执行该操作。",
                    is_completed=True,
                    updated_at=now,
                )
            else:
                approval.refresh_from_db()
        else:
            approval.status = "approved" if decision == "approve" else "rejected"
            approval.decided_by = str(ctx.user.id) if ctx.user else ""
            approval.decided_at = now
            TaskRecord.objects.create(id=approval.id, task_type=RESUME_APPROVAL_TASK)
            transaction.on_commit(lambda: submit_approval_resume(approval.id), robust=True)
    if expired_now:
        stream_manager.append_event(
            approval.message_id,
            "complete",
            {"done": True, "content": "等待确认超时，未执行该操作。"},
        )
        raise ServiceError("审批已过期，操作未执行。", status=410)
    if not claimed:
        raise Conflict(f"approval already {approval.status}", details={"status": approval.status})
    return {
        "approval": {
            "id": approval.id,
            "status": approval.status,
            "tool_calls": approval.tool_calls,
            "expires_at": approval.expires_at.isoformat(),
        },
        "message_id": approval.message_id,
    }
