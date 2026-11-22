from django.http import FileResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt

from .authentication import tenant_scoped
from .http_helpers import auth_context, service_context
from .responses import ok, service_errors
from services.system import (
    create_embed_session,
    default_embed_response,
    embed_channel_config,
    embed_events,
    exchange_embed_token,
    get_embed_chunk,
    get_stored_file,
    health_status,
    im_callback as im_callback_service,
    initialization_config as initialization_config_service,
    parser_engines as parser_engines_service,
    stopped_embed_session,
    storage_status as storage_status_service,
    suggested_embed_questions,
    system_info as system_info_service,
    task_progress as task_progress_service,
    validate_embed_session,
)


def health(request):
    return JsonResponse(health_status())


@service_errors
def initialization_config(request, kb_id):
    return ok(initialization_config_service(kb_id))


@csrf_exempt
def initialization_update(request, kb_id):
    from knowledge.views import knowledge_bases

    return knowledge_bases(request, kb_id)


def system_info(request):
    return ok(system_info_service())


def parser_engines(request):
    _, tenant = auth_context(request)
    return ok(parser_engines_service(tenant))


def storage_status(request):
    return ok(storage_status_service())


@csrf_exempt
@tenant_scoped
@service_errors
def embed_public(request, channel_id, action=None, session_id=None, chunk_id=None):
    ctx = service_context(request)
    if session_id:
        validate_embed_session(ctx, session_id)
    if action == "exchange":
        return ok(exchange_embed_token(channel_id))
    if action == "config":
        return ok(embed_channel_config(ctx, channel_id))
    if action == "suggested-questions":
        return ok(suggested_embed_questions())
    if action == "chunks":
        return ok(get_embed_chunk(ctx, chunk_id))
    if action == "sessions":
        return ok(create_embed_session(ctx), status=201)
    if action in {"knowledge-chat", "agent-chat"}:
        from chat.views import chat_endpoint

        return chat_endpoint(request, session_id, agent=action == "agent-chat")
    if action == "messages":
        from chat.views import messages_load

        return messages_load(request, session_id)
    if action == "stop":
        return ok(stopped_embed_session())
    if action == "events":
        return ok(embed_events())
    return ok(default_embed_response())


@service_errors
def serve_file(request):
    _, tenant = auth_context(request)
    file_path = request.GET.get("file_path") or request.GET.get("path") or ""
    file_path = file_path.removeprefix("local://").lstrip("/")
    stored_file = get_stored_file(tenant, file_path)
    return FileResponse(stored_file.stream, content_type=stored_file.content_type)


def presigned_file(request):
    return serve_file(request)


@csrf_exempt
def im_callback(request, channel_id):
    return ok(im_callback_service(channel_id))


def task_progress(request, task_id):
    return ok(task_progress_service(task_id))
