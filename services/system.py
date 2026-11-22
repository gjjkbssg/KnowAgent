import mimetypes
import secrets
from dataclasses import dataclass

from django.conf import settings
from django.core.files.storage import default_storage

from personal_knowledge_base.document_parsing import parser_capabilities
from personal_knowledge_base.graph_rag import graph_database_engine, graph_rag_enabled, neo4j_configured
from personal_knowledge_base.model_providers import bailian_status
from personal_knowledge_base.models import Chunk, GenericResource, KnowledgeBase, Session
from personal_knowledge_base.serializers import chunk_dict, kb_dict, session_dict
from personal_knowledge_base.tasks import task_status
from services.context import ServiceContext
from services.errors import InvalidInput, NotFound, Unauthorized, get_or_404


@dataclass(frozen=True)
class StoredFile:
    stream: object
    content_type: str


def health_status() -> dict:
    return {"status": "ok"}


def initialization_config(kb_id) -> dict:
    kb = get_or_404(KnowledgeBase, id=kb_id)
    return {"knowledge_base": kb_dict(kb), "config": kb_dict(kb, counts=False)}


def system_info() -> dict:
    return {
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
    }


def parser_engines(tenant) -> dict:
    return {"items": [parser_capabilities(tenant)]}


def storage_status() -> dict:
    return {"provider": "local", "status": "available"}


def validate_embed_session(ctx: ServiceContext, session_id) -> None:
    get_or_404(Session, id=session_id, tenant=ctx.tenant)


def embed_channel_config(ctx: ServiceContext, channel_id) -> dict:
    channel = GenericResource.objects.filter(
        id=channel_id,
        resource_type="embed_channels",
        tenant=ctx.tenant,
    ).first()
    return channel.data if channel else {"id": channel_id, "enabled": True}


def exchange_embed_token(channel_id) -> dict:
    return {"token": f"embed-{channel_id}-{secrets.token_urlsafe(12)}", "session": {}}


def suggested_embed_questions() -> dict:
    return {"items": []}


def get_embed_chunk(ctx: ServiceContext, chunk_id) -> dict:
    chunk = get_or_404(Chunk, id=chunk_id, tenant=ctx.tenant)
    return chunk_dict(chunk)


def create_embed_session(ctx: ServiceContext) -> dict:
    session = Session.objects.create(tenant=ctx.tenant, title="Embed chat")
    return session_dict(session)


def stopped_embed_session() -> dict:
    return {"stopped": True}


def embed_events() -> dict:
    return {}


def default_embed_response() -> dict:
    return {}


def get_stored_file(tenant, file_path: str) -> StoredFile:
    if not tenant:
        raise Unauthorized("unauthorized")
    if not file_path:
        raise InvalidInput("file_path required")
    if not file_path.startswith(f"tenant-{tenant.id}/") or not default_storage.exists(file_path):
        raise NotFound("not found")
    return StoredFile(
        stream=default_storage.open(file_path, "rb"),
        content_type=mimetypes.guess_type(file_path)[0] or "application/octet-stream",
    )


def im_callback(channel_id) -> dict:
    return {"channel_id": channel_id, "status": "received"}


def task_progress(task_id):
    return task_status(task_id)
