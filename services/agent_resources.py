import secrets
from collections.abc import Mapping

from django.db.models import Q
from django.utils import timezone

from personal_knowledge_base.document_parsing import parser_capabilities
from personal_knowledge_base.http_helpers import paginate
from personal_knowledge_base.mcp_client import MCPClient, mcp_tool_requires_approval
from personal_knowledge_base.models import Chunk, GenericResource, Session, Tenant
from personal_knowledge_base.serializers import resource_dict, session_dict
from services.context import ServiceContext
from services.errors import get_or_404


def get_default_tenant():
    return Tenant.objects.first()


def get_resource(ctx: ServiceContext, resource_type: str, item_id) -> dict:
    item = get_or_404(GenericResource, id=item_id, resource_type=resource_type, tenant=ctx.tenant)
    return resource_dict(item)


def delete_resource(ctx: ServiceContext, resource_type: str, item_id) -> dict:
    item = get_or_404(GenericResource, id=item_id, resource_type=resource_type, tenant=ctx.tenant)
    item.deleted_at = timezone.now()
    item.save(update_fields=["deleted_at", "updated_at"])
    return {}


def update_resource(ctx: ServiceContext, resource_type: str, item_id, data: dict) -> dict:
    item = get_or_404(GenericResource, id=item_id, resource_type=resource_type, tenant=ctx.tenant)
    if resource_type == "mcp_services" and not data.get("api_key"):
        data.pop("api_key", None)
    item.name = data.get("name", item.name)
    item.status = data.get("status", item.status)
    item.data = {**(item.data or {}), **data}
    item.save()
    return resource_dict(item)


def list_resources(ctx: ServiceContext, resource_type: str, params: Mapping):
    if resource_type == "agents":
        seed_builtin_agents(ctx.tenant)
        qs = GenericResource.objects.filter(
            resource_type=resource_type,
            tenant=ctx.tenant,
            deleted_at__isnull=True,
            data__agent_mode="multi-agent",
        ).order_by("-updated_at")
    else:
        qs = GenericResource.objects.filter(
            resource_type=resource_type,
            tenant=ctx.tenant,
            deleted_at__isnull=True,
        ).order_by("-updated_at")
    keyword = params.get("keyword") or params.get("q") or params.get("query")
    if keyword:
        qs = qs.filter(Q(name__icontains=keyword) | Q(data__description__icontains=keyword))
    items, meta = paginate(qs, params)
    return [resource_dict(item) for item in items], meta


def create_resource(ctx: ServiceContext, resource_type: str, data: dict) -> dict:
    defaults = default_resource_payload(resource_type, data)
    item = GenericResource.objects.create(
        tenant=ctx.tenant,
        resource_type=resource_type,
        name=defaults.get("name", data.get("name", data.get("title", ""))),
        data=defaults,
        status=defaults.get("status", data.get("status", "active")),
    )
    return resource_dict(item)


def _mcp_service(ctx: ServiceContext, item_id):
    return get_or_404(
        GenericResource,
        id=item_id,
        resource_type="mcp_services",
        tenant=ctx.tenant,
        deleted_at__isnull=True,
    )


def _call_mcp_service(ctx: ServiceContext, item_id, *, tools: bool):
    service = _mcp_service(ctx, item_id)
    data = service.data or {}
    client = MCPClient(data.get("url") or "", api_key=data.get("api_key") or "", timeout=10)
    try:
        client.initialize()
        available_tools = client.list_tools()
        if not tools:
            return {
                "status": "ok",
                "available": True,
                "protocol_version": client.protocol_version,
                "tool_count": len(available_tools),
            }
        return {
            "items": [
                {
                    "name": tool["name"],
                    "description": tool["description"],
                    "input_schema": tool["inputSchema"],
                    "annotations": tool["annotations"],
                    "requires_approval": mcp_tool_requires_approval(
                        data.get("approval_mode", "always"),
                        tool["annotations"],
                    ),
                }
                for tool in available_tools
            ]
        }
    except Exception as exc:
        error = str(exc)
        if data.get("api_key"):
            error = error.replace(data["api_key"], "[redacted]")
        if not tools:
            return {"status": "error", "available": False, "error": error}
        return {"items": [], "error": error}
    finally:
        client.close()


def test_mcp_service(ctx: ServiceContext, item_id) -> dict:
    return _call_mcp_service(ctx, item_id, tools=False)


def list_mcp_service_tools(ctx: ServiceContext, item_id) -> dict:
    return _call_mcp_service(ctx, item_id, tools=True)


def static_types(resource_type: str, action: str) -> list[dict]:
    if resource_type == "vector_stores":
        return [{"type": "sqlite", "name": "SQLite sqlite-vec", "builtin": True}]
    if resource_type == "web_search_providers":
        return [{"provider": "duckduckgo"}, {"provider": "bing"}, {"provider": "google"}, {"provider": "searxng"}]
    if resource_type == "data_sources":
        return []
    if resource_type == "agents":
        if action == "placeholders":
            return [
                {"key": "query", "label": "用户问题", "fields": ["system_prompt", "context_template"]},
                {"key": "context", "label": "知识库上下文", "fields": ["system_prompt", "context_template"]},
                {"key": "history", "label": "历史对话", "fields": ["system_prompt"]},
                {"key": "current_date", "label": "当前日期", "fields": ["system_prompt"]},
            ]
        return [
            {"id": "multi-agent", "type": "multi-agent", "name": "智能助手", "agent_mode": "multi-agent", "description": "自动调度文档、Wiki、图谱子 Agent"}
        ]
    return []


def check_parser_engines(tenant) -> dict:
    capability = parser_capabilities(tenant)
    return {"status": "ok" if capability["available"] else "error", **capability}


def suggested_questions(tenant) -> dict:
    questions = []
    if tenant:
        chunks = Chunk.objects.filter(tenant=tenant, is_enabled=True).select_related("knowledge").order_by("-updated_at")[:6]
        for chunk in chunks:
            title = chunk.knowledge.title if chunk.knowledge_id else "知识库"
            questions.append({"question": f"{title} 的核心内容是什么？", "source": "knowledge"})
    if not questions:
        questions = [
            {"question": "这个知识库里有哪些重要内容？", "source": "builtin"},
            {"question": "请总结最近上传的资料。", "source": "builtin"},
            {"question": "帮我查找和当前问题相关的引用。", "source": "builtin"},
        ]
    return {"items": questions, "questions": questions}


def copy_agent(tenant, item_id) -> dict:
    source = get_or_404(GenericResource, id=item_id, resource_type="agents", tenant=tenant)
    clone_data = {**(source.data or {}), "name": f"{source.name} 副本", "copied_from": source.id}
    clone = GenericResource.objects.create(
        tenant=tenant,
        resource_type="agents",
        name=clone_data["name"],
        data=clone_data,
        status=source.status,
    )
    return resource_dict(clone)


def toggle_channel(tenant, resource_type: str, item_id) -> dict:
    item = get_or_404(GenericResource, id=item_id, resource_type=resource_type, tenant=tenant)
    enabled = not bool((item.data or {}).get("enabled", item.status == "active"))
    item.data = {**(item.data or {}), "enabled": enabled}
    item.status = "active" if enabled else "disabled"
    item.save(update_fields=["data", "status", "updated_at"])
    return resource_dict(item)


def rotate_embed_channel_token(tenant, item_id) -> dict:
    item = get_or_404(GenericResource, id=item_id, resource_type="embed_channels", tenant=tenant)
    token = secrets.token_urlsafe(24)
    item.data = {**(item.data or {}), "token": token}
    item.save(update_fields=["data", "updated_at"])
    return {"id": item.id, "token": token, "config": resource_dict(item)}


def create_embed_preview_session(tenant, item_id) -> dict:
    session = Session.objects.create(tenant=tenant, title="Agent 预览会话", agent_id=item_id or "")
    return {"session": session_dict(session)}


def action_stats() -> dict:
    return {"sessions": 0, "messages": 0, "last_active_at": None}


def action_items() -> dict:
    return {"items": []}


def action_status() -> dict:
    return {"status": "ok"}


def default_resource_payload(resource_type, data):
    payload = dict(data or {})
    if resource_type == "agents":
        payload["name"] = payload.get("name") or payload.get("title") or "智能助手"
        payload.setdefault("description", "自动调度文档、Wiki、图谱子 Agent。")
        payload["type"] = "multi-agent"
        payload["agent_type"] = "multi-agent"
        payload["agent_mode"] = "multi-agent"
        payload.setdefault("avatar", "")
        payload.setdefault("system_prompt", "你是多 Agent 知识工作台的主 Agent。简单问题可直接回答；复杂问题通过 actor 工具调度专业子 Agent。")
        payload.setdefault("opening_statement", "你好，我可以帮你检索知识库并整理答案。")
        payload.setdefault("suggested_questions", ["请总结这个知识库", "有哪些关键风险点？", "给我列出引用来源"])
        payload.setdefault("suggested_prompts", payload.get("suggested_questions") or [])
        payload.setdefault("kb_selection_mode", "selected" if payload.get("knowledge_base_ids") else "all")
        payload.setdefault("knowledge_base_ids", [])
        payload.setdefault("knowledge_bases", payload.get("knowledge_base_ids") or [])
        payload.setdefault("model_id", "")
        payload.setdefault("rerank_model_id", "")
        payload["allowed_tools"] = ["actor", "thinking"]
        payload["tools"] = ["actor", "thinking"]
        payload.setdefault("mcp_selection_mode", "selected" if payload.get("mcp_services") else "none")
        payload.setdefault("mcp_services", [])
        payload.setdefault("web_search_enabled", False)
        payload.setdefault("memory_enabled", True)
        payload.setdefault("rerank_enabled", True)
        payload.setdefault("temperature", 0.3)
        payload.setdefault("max_rounds", 8)
        payload.setdefault("status", "active")
    elif resource_type == "mcp_services":
        payload.setdefault("approval_mode", "always")
    elif resource_type == "embed_channels":
        payload.setdefault("name", "网页嵌入")
        payload.setdefault("enabled", True)
        payload.setdefault("token", secrets.token_urlsafe(24))
        payload.setdefault("allowed_origins", ["*"])
    elif resource_type == "im_channels":
        payload.setdefault("name", "IM 渠道")
        payload.setdefault("enabled", False)
        payload.setdefault("provider", "wechat")
    return payload


def seed_builtin_agents(tenant):
    presets = [
        {
            "id": f"builtin-multi-agent-assistant-{tenant.id}",
            "name": "智能助手",
            "description": "自动调度文档、Wiki、图谱子 Agent，适合默认问答入口。",
            "type": "multi-agent",
            "agent_mode": "multi-agent",
            "system_prompt": "你是多 Agent 知识工作台的主 Agent。简单问题可直接回答；需要文档证据时调用 doc_retriever；需要结构化知识时调用 wiki_researcher；需要实体关系推理时调用 graph_reasoner；需要综合多个结果时调用 answer_writer。",
            "allowed_tools": ["actor", "thinking"],
            "max_rounds": 8,
        },
    ]
    for preset in presets:
        if GenericResource.objects.filter(id=preset["id"]).exists():
            continue
        data = default_resource_payload("agents", preset)
        GenericResource.objects.create(
            id=preset["id"],
            tenant=tenant,
            resource_type="agents",
            name=data["name"],
            data=data,
            status="active",
        )
