from django.views.decorators.csrf import csrf_exempt

from personal_knowledge_base.http_helpers import auth_context, list_response, parse_body
from personal_knowledge_base.responses import fail, ok, service_errors
from services.agent_resources import (
    action_items,
    action_stats,
    action_status,
    check_parser_engines,
    copy_agent,
    create_embed_preview_session,
    create_resource,
    delete_resource,
    get_default_tenant,
    get_resource,
    list_mcp_service_tools,
    list_resources,
    rotate_embed_channel_token,
    static_types,
    suggested_questions,
    test_mcp_service,
    toggle_channel,
    update_resource,
)
from services.context import ServiceContext


def _context(request, user, tenant):
    return ServiceContext(
        tenant=tenant,
        user=user,
        request_id=request.headers.get("X-Request-ID", ""),
    )


# ---------------------------------------------------------------------------
# Core agent resource views
# ---------------------------------------------------------------------------

@csrf_exempt
@service_errors
def generic_collection(request, resource_type, item_id=None, extra=None, **kwargs):
    item_id = item_id or kwargs.get("log_id") or kwargs.get("share_id") or kwargs.get("inv_id")
    user, tenant = auth_context(request)
    if resource_type in {"system_settings"}:
        tenant = tenant or get_default_tenant()
    if not tenant:
        return fail("unauthorized", 401)
    ctx = _context(request, user, tenant)
    if item_id:
        if request.method == "GET":
            return ok(get_resource(ctx, resource_type, item_id))
        if request.method == "DELETE":
            return ok(delete_resource(ctx, resource_type, item_id))
        return ok(update_resource(ctx, resource_type, item_id, parse_body(request)))
    if request.method == "GET":
        items, meta = list_resources(ctx, resource_type, request.GET)
        return ok(list_response(items, meta))
    return ok(create_resource(ctx, resource_type, parse_body(request)), status=201)


@csrf_exempt
@service_errors
def generic_action(request, resource_type, action="", item_id=None, sub_id=None, **kwargs):
    item_id = item_id or kwargs.get("channel_id") or kwargs.get("pending_id")
    sub_id = sub_id or kwargs.get("tool_name") or kwargs.get("field")
    if resource_type == "mcp_services" and action in {"test", "tools"}:
        user, tenant = auth_context(request)
        if not tenant:
            return fail("unauthorized", 401)
        ctx = _context(request, user, tenant)
        if action == "test":
            return ok(test_mcp_service(ctx, item_id))
        return ok(list_mcp_service_tools(ctx, item_id))
    if action in {"types", "providers", "placeholders", "type-presets"}:
        return ok({"items": static_types(resource_type, action)})
    if action == "parser-engines/check":
        _, tenant = auth_context(request)
        return ok(check_parser_engines(tenant))
    if action in {"test", "validate-credentials", "validate", "storage-engine-check", "remote/check", "embedding/test", "rerank/check", "multimodal/test"}:
        return ok({"status": "ok", "available": True})
    if action == "suggested-questions":
        _, tenant = auth_context(request)
        return ok(suggested_questions(tenant))
    if resource_type == "agents" and action == "copy":
        _, tenant = auth_context(request)
        return ok(copy_agent(tenant, item_id), status=201)
    if resource_type in {"im_channels", "embed_channels"} and action == "toggle":
        _, tenant = auth_context(request)
        return ok(toggle_channel(tenant, resource_type, item_id))
    if resource_type == "embed_channels" and action == "rotate-token":
        _, tenant = auth_context(request)
        return ok(rotate_embed_channel_token(tenant, item_id))
    if resource_type == "embed_channels" and action == "preview-session":
        _, tenant = auth_context(request)
        return ok(create_embed_preview_session(tenant, item_id))
    if action == "stats":
        return ok(action_stats())
    if action in {"tools", "resources", "tool-approvals", "oauth/status", "stats", "logs", "members", "shares", "agent-shares", "join-requests", "shared-knowledge-bases", "shared-agents"}:
        return ok(action_items())
    if action in {"sync", "pause", "resume", "toggle", "rotate-token", "preview-session", "leave", "request-upgrade", "invite-code", "invite", "join", "join-request", "join-by-id", "promote", "revoke", "rebuild-links", "auto-fix"}:
        return ok(action_status())
    return ok(action_status())
