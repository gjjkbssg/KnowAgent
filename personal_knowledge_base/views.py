import mimetypes
import secrets

from django.conf import settings
from django.core.files.storage import default_storage
from django.http import FileResponse, JsonResponse
from django.shortcuts import get_object_or_404
from django.views.decorators.csrf import csrf_exempt

from .authentication import tenant_scoped
from .graph_rag import graph_database_engine, graph_rag_enabled, neo4j_configured
from .http_helpers import auth_context
from .model_providers import bailian_status
from .models import Chunk, GenericResource, KnowledgeBase, Session, TaskRecord
from .responses import fail, ok
from .serializers import chunk_dict, kb_dict, session_dict
from .tasks import RESUME_APPROVAL_TASK, task_status


def health(request):
    return JsonResponse({"status": "ok"})


def initialization_config(request, kb_id):
    kb = get_object_or_404(KnowledgeBase, id=kb_id)
    return ok({"knowledge_base": kb_dict(kb), "config": kb_dict(kb, counts=False)})


@csrf_exempt
def initialization_update(request, kb_id):
    from knowledge.views import knowledge_bases

    return knowledge_bases(request, kb_id)


def system_info(request):
    return ok({
        "name": settings.APP_NAME,
        "edition": "django-lite",
        "version": "0.6.2-django",
        "storage": "FileSystemStorage",
        "cache": "LocMemCache",
        "vector": "sqlite-vec",
        "graph_database_engine": graph_database_engine(),
        "graph_rag_enabled": graph_rag_enabled(),
        "neo4j_configured": neo4j_configured(),
        "bailian": bailian_status(),
    })


def parser_engines(request):
    from .document_parsing import parser_capabilities

    _, tenant = auth_context(request)
    return ok({"items": [parser_capabilities(tenant)]})


def storage_status(request):
    return ok({"provider": "local", "status": "available"})


@csrf_exempt
@tenant_scoped
def embed_public(request, channel_id, action=None, session_id=None, chunk_id=None):
    tenant = request.auth_tenant
    channel = GenericResource.objects.filter(id=channel_id, resource_type="embed_channels", tenant=tenant).first()
    if session_id:
        get_object_or_404(Session, id=session_id, tenant=tenant)
    if action == "exchange":
        return ok({"token": f"embed-{channel_id}-{secrets.token_urlsafe(12)}", "session": {}})
    if action == "config":
        return ok(channel.data if channel else {"id": channel_id, "enabled": True})
    if action == "suggested-questions":
        return ok({"items": []})
    if action == "chunks":
        chunk = get_object_or_404(Chunk, id=chunk_id, tenant=tenant)
        return ok(chunk_dict(chunk))
    if action == "sessions":
        session = Session.objects.create(tenant=tenant, title="Embed chat")
        return ok(session_dict(session), status=201)
    if action in {"knowledge-chat", "agent-chat"}:
        from chat.views import chat_endpoint

        return chat_endpoint(request, session_id, agent=action == "agent-chat")
    if action == "messages":
        from chat.views import messages_load

        return messages_load(request, session_id)
    if action == "stop":
        return ok({"stopped": True})
    if action == "events":
        return ok({})
    return ok({})


def serve_file(request):
    _, tenant = auth_context(request)
    if not tenant:
        return fail("unauthorized", 401)
    file_path = request.GET.get("file_path") or request.GET.get("path") or ""
    file_path = file_path.removeprefix("local://").lstrip("/")
    if not file_path:
        return fail("file_path required", 400)
    # Uploads and extracted images are stored under "tenant-<id>/".
    if not file_path.startswith(f"tenant-{tenant.id}/") or not default_storage.exists(file_path):
        return fail("not found", 404)
    return FileResponse(default_storage.open(file_path, "rb"), content_type=mimetypes.guess_type(file_path)[0] or "application/octet-stream")


def presigned_file(request):
    return serve_file(request)


@csrf_exempt
def im_callback(request, channel_id):
    return ok({"channel_id": channel_id, "status": "received"})


def task_progress(request, task_id):
    if TaskRecord.objects.filter(id=task_id, task_type=RESUME_APPROVAL_TASK).exists():
        return ok({"status": "not_found", "progress": 0})
    return ok(task_status(task_id))
