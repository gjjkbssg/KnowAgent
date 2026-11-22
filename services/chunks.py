from collections.abc import Mapping

from django.db import transaction
from django.utils import timezone

from personal_knowledge_base.graph_rag import delete_knowledge_graph
from personal_knowledge_base.http_helpers import list_response, paginate
from personal_knowledge_base.model_providers import EmbeddingUnavailableError
from personal_knowledge_base.models import Chunk, Knowledge
from personal_knowledge_base.search import chunk_vector, delete_chunk_index, index_chunk
from personal_knowledge_base.serializers import chunk_dict

from .context import ServiceContext
from .errors import NotFound, ServiceError, get_or_404
from .knowledge import delete_knowledge_content


def chunks_collection(
    ctx: ServiceContext,
    *,
    method: str,
    knowledge_id=None,
    chunk_id=None,
    data=None,
    params: Mapping | None = None,
):
    params = params or {}
    data = data or {}
    if knowledge_id == "by-id" and chunk_id:
        knowledge_id = None
    if chunk_id and not knowledge_id:
        chunk = get_or_404(Chunk, id=chunk_id)
        if method == "GET":
            return chunk_dict(chunk)
        if method == "DELETE":
            delete_knowledge_graph(chunk.knowledge)
            delete_chunk_index(chunk.id, chunk.seq_id)
            chunk.delete()
            return {}
    if knowledge_id and chunk_id:
        chunk = get_or_404(Chunk, id=chunk_id, knowledge_id=knowledge_id)
        if method == "GET":
            return chunk_dict(chunk)
        if method == "DELETE":
            delete_knowledge_graph(chunk.knowledge)
            delete_chunk_index(chunk.id, chunk.seq_id)
            chunk.delete()
            return {}
        chunk.content = data.get("content", chunk.content)
        chunk.is_enabled = data.get("is_enabled", chunk.is_enabled)
        chunk.metadata = data.get("metadata", chunk.metadata)
        try:
            vector = chunk_vector(chunk)
        except EmbeddingUnavailableError as exc:
            raise ServiceError(
                f"Embedding 服务不可用，切片未保存：{exc}",
                status=503,
                code="embedding_unavailable",
            ) from exc
        with transaction.atomic():
            chunk.updated_at = timezone.now()
            saved = Chunk.objects.filter(id=chunk.id).update(
                content=chunk.content,
                is_enabled=chunk.is_enabled,
                metadata=chunk.metadata,
                updated_at=chunk.updated_at,
            )
            if not saved:
                raise NotFound("chunk not found")
            index_chunk(chunk, vector)
        return chunk_dict(chunk)
    if knowledge_id and method == "GET":
        chunks = Chunk.objects.filter(knowledge_id=knowledge_id).order_by("chunk_index")
        chunk_type = params.get("chunk_type")
        if chunk_type:
            chunks = chunks.filter(chunk_type=chunk_type)
        page, meta = paginate(chunks, params)
        return list_response([chunk_dict(chunk) for chunk in page], meta, ["chunks"])
    if knowledge_id and method == "DELETE":
        item = Knowledge.objects.filter(id=knowledge_id).first()
        if item:
            delete_knowledge_content(item)
        return {}
    raise NotFound("not found")
