from collections.abc import Mapping

from django.db.models import Q
from django.utils import timezone

from personal_knowledge_base.agent_actor import ActorRegistry
from personal_knowledge_base.context_snapshot import clear_context_snapshots
from personal_knowledge_base.http_helpers import bool_from_value, bounded_int, paginate
from personal_knowledge_base.memory import delete_session_memory
from personal_knowledge_base.model_providers import role_completion
from personal_knowledge_base.models import AgentToolApproval, KnowledgeBase, Message, Session
from personal_knowledge_base.serializers import message_dict, session_dict
from personal_knowledge_base.stream_manager import stream_manager
from services.context import ServiceContext
from services.errors import get_or_404


SESSION_CONFIG_FIELDS = {
    "agent_enabled",
    "agent_id",
    "model_id",
    "summary_model_id",
    "knowledge_base_ids",
    "web_search_enabled",
    "enable_memory",
    "mcp_service_ids",
}


def normalize_session_state(value=None):
    raw = value if isinstance(value, dict) else {}
    state = {
        "agent_enabled": bool_from_value(raw.get("agent_enabled"), False),
        "agent_id": str(raw.get("agent_id") or ""),
        "model_id": str(raw.get("model_id") or ""),
        "summary_model_id": str(raw.get("summary_model_id") or ""),
        "knowledge_base_ids": raw.get("knowledge_base_ids") if isinstance(raw.get("knowledge_base_ids"), list) else [],
        "web_search_enabled": bool_from_value(raw.get("web_search_enabled"), False),
        "enable_memory": bool_from_value(raw.get("enable_memory"), True),
        "mcp_service_ids": raw.get("mcp_service_ids") if isinstance(raw.get("mcp_service_ids"), list) else [],
    }
    return state


def session_state_from_payload(data, fallback=None):
    source = data.get("agent_config") if isinstance(data.get("agent_config"), dict) else data
    state = normalize_session_state(fallback)
    for field in SESSION_CONFIG_FIELDS:
        if field in source:
            state[field] = source[field]
    return normalize_session_state(state)


def get_tenant_session(ctx: ServiceContext, session_id):
    return get_or_404(Session, id=session_id, tenant=ctx.tenant)


def list_sessions(ctx: ServiceContext, params: Mapping):
    qs = Session.objects.filter(tenant=ctx.tenant, deleted_at__isnull=True).order_by("-is_pinned", "-updated_at")
    page, meta = paginate(qs, params)
    return {"items": [session_dict(session) for session in page], **meta}


def get_session(ctx: ServiceContext, session_id):
    return session_dict(get_tenant_session(ctx, session_id))


def create_session(ctx: ServiceContext, data: dict):
    state = session_state_from_payload(data)
    knowledge_base_id = data.get("knowledge_base_id") or (state["knowledge_base_ids"][0] if state["knowledge_base_ids"] else "")
    agent_id = data.get("agent_id") or state["agent_id"]
    session = Session.objects.create(
        tenant=ctx.tenant,
        title=data.get("title", "新的对话"),
        knowledge_base_id=knowledge_base_id,
        agent_id=agent_id,
        user_id=ctx.user.id if ctx.user else "",
        agent_config=state,
    )
    return session_dict(session)


def update_session(ctx: ServiceContext, session_id, data: dict):
    session = get_tenant_session(ctx, session_id)
    for field in ["title", "description", "knowledge_base_id", "agent_id"]:
        if field in data:
            setattr(session, field, data[field])
    if "agent_config" in data or any(field in data for field in SESSION_CONFIG_FIELDS):
        session.agent_config = session_state_from_payload(data, session.agent_config)
        if session.agent_config.get("agent_id"):
            session.agent_id = session.agent_config["agent_id"]
        if session.agent_config.get("knowledge_base_ids"):
            session.knowledge_base_id = session.agent_config["knowledge_base_ids"][0]
    session.save()
    return session_dict(session)


def delete_session(ctx: ServiceContext, session_id):
    session = get_tenant_session(ctx, session_id)
    session.deleted_at = timezone.now()
    session.save(update_fields=["deleted_at", "updated_at"])
    delete_session_memory(session.id)
    return {}


def batch_delete_sessions(ctx: ServiceContext, data: dict):
    if data.get("delete_all"):
        session_ids = list(Session.objects.filter(tenant=ctx.tenant).values_list("id", flat=True))
        Session.objects.filter(id__in=session_ids, tenant=ctx.tenant).update(deleted_at=timezone.now())
    else:
        session_ids = list(
            Session.objects.filter(id__in=data.get("ids", []), tenant=ctx.tenant).values_list("id", flat=True)
        )
        Session.objects.filter(id__in=session_ids, tenant=ctx.tenant).update(deleted_at=timezone.now())
    for session_id in session_ids:
        delete_session_memory(session_id)
    return {}


def pin_session(ctx: ServiceContext, session_id, pinned: bool):
    session = get_tenant_session(ctx, session_id)
    session.is_pinned = pinned
    session.pinned_at = timezone.now() if pinned else None
    session.save(update_fields=["is_pinned", "pinned_at", "updated_at"])
    return session_dict(session)


def clear_session_messages(ctx: ServiceContext, session_id):
    session = get_tenant_session(ctx, session_id)
    clear_context_snapshots(session)
    Message.objects.filter(session=session).delete()
    delete_session_memory(session.id)
    return {}


def generate_session_title(ctx: ServiceContext, session_id, data: dict):
    session = get_tenant_session(ctx, session_id)
    source = data.get("title") or data.get("query") or "新的对话"
    title = role_completion(
        "title",
        f"请为下面这次知识库对话生成一个 20 字以内的中文标题，只输出标题。\n\n{source}",
        source,
        40,
    )
    session.title = title[:80]
    session.save(update_fields=["title", "updated_at"])
    return session_dict(session)


def stop_session(ctx: ServiceContext, session_id, data: dict):
    session = get_tenant_session(ctx, session_id)
    message_id = data.get("message_id") or data.get("id")
    if message_id:
        matching_messages = Message.objects.filter(
            Q(id=message_id) | Q(request_id=message_id),
            session_id=session_id,
        )
        messages = list(matching_messages)
        message_by_id = {message.id: message for message in messages}
        matching_message_ids = list(message_by_id)
        target_message = next((message for message in messages if message.role == "assistant"), None)
        target_message = target_message or (messages[0] if messages else None)
        matching_messages.update(is_completed=True, updated_at=timezone.now())
        pending_approval_message_ids = list(
            AgentToolApproval.objects.filter(
                message_id__in=matching_message_ids,
                status="pending",
            ).values_list("message_id", flat=True).distinct()
        )
        if pending_approval_message_ids:
            cancelled = AgentToolApproval.objects.filter(
                message_id__in=matching_message_ids,
                status="pending",
            ).update(
                status="cancelled",
                decided_at=timezone.now(),
            )
            if cancelled:
                for approval_message_id in pending_approval_message_ids:
                    cancelled_message = message_by_id.get(approval_message_id)
                    stream_manager.append_event(
                        approval_message_id,
                        "complete",
                        {"done": True, "content": (cancelled_message.content if cancelled_message else "") or "生成已取消。"},
                    )
        parent_message_id = target_message.id if target_message else message_id
        for actor in session.agent_actors.filter(parent_message_id=parent_message_id, status__in=["pending", "running"]):
            ActorRegistry.cancel_actor(session, actor.actor_id)
    return {"session_id": session_id, "message_id": message_id, "stopped": True}


def load_messages(ctx: ServiceContext, session_id, params: Mapping):
    get_tenant_session(ctx, session_id)
    limit = bounded_int(params.get("limit"), 50, 1, 200)
    qs = Message.objects.filter(session_id=session_id, visible_to_user=True)
    before_time = params.get("before_time") or params.get("before")
    if before_time:
        try:
            parsed = timezone.datetime.fromisoformat(before_time.replace("Z", "+00:00"))
            if timezone.is_naive(parsed):
                parsed = timezone.make_aware(parsed)
            qs = qs.filter(created_at__lt=parsed)
        except Exception:
            pass
    qs = qs.order_by("-created_at")[:limit]
    items = [message_dict(message) for message in reversed(list(qs))]
    return {"items": items, "messages": items, "has_more": len(items) >= limit}


def search_messages(ctx: ServiceContext, data: dict):
    query = data.get("query") or data.get("q") or ""
    qs = Message.objects.filter(session__tenant=ctx.tenant, visible_to_user=True)
    if query:
        qs = qs.filter(content__icontains=query)
    return {"items": [message_dict(message) for message in qs.order_by("-created_at")[:50]]}


def chat_history_stats(ctx: ServiceContext):
    return {
        "total_sessions": Session.objects.filter(tenant=ctx.tenant).count(),
        "total_messages": Message.objects.filter(session__tenant=ctx.tenant).count(),
    }


def delete_message(ctx: ServiceContext, session_id, message_id):
    get_tenant_session(ctx, session_id)
    Message.objects.filter(id=message_id, session_id=session_id).delete()
    return {}
