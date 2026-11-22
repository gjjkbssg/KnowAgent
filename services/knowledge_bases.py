from django.db.models import Q
from django.utils import timezone

from personal_knowledge_base.graph_rag import delete_kb_graph, validate_extract_config
from personal_knowledge_base.http_helpers import list_response, paginate
from personal_knowledge_base.models import KnowledgeBase
from personal_knowledge_base.serializers import (
    DEFAULT_INDEXING_STRATEGY,
    kb_dict,
    normalize_indexing_strategy,
)
from personal_knowledge_base.wiki_ingest import cleanup_wiki_for_kb

from .context import ServiceContext
from .errors import InvalidInput, get_or_404


def default_chunk_config():
    return {"chunk_size": 512, "chunk_overlap": 50, "split_markers": ["\n\n", "\n", "。"], "keep_separator": True}


def normalize_kb_payload(data, existing=None, partial=False):
    kb_type = data.get("type", existing.type if existing else "document")
    if kb_type == "faq":
        raise InvalidInput("FAQ knowledge bases are no longer supported")
    if kb_type not in {"document", "wiki"}:
        raise InvalidInput("unsupported knowledge base type")

    config = data.get("config") or data
    existing_strategy = existing.indexing_strategy if existing else None
    raw_strategy = config.get("indexing_strategy", existing_strategy if partial else DEFAULT_INDEXING_STRATEGY)
    strategy = normalize_indexing_strategy(raw_strategy, kb_type)
    if not any(strategy.values()):
        raise InvalidInput("at least one indexing strategy must be enabled")

    wiki_config = config.get("wiki_config", existing.wiki_config if existing else None)
    if strategy["wiki_enabled"] and wiki_config is None:
        wiki_config = {}
    extract_config = config.get("extract_config", existing.extract_config if existing else None)
    if strategy["graph_enabled"]:
        from personal_knowledge_base.graph_rag import DEFAULT_EXTRACT_CONFIG

        graph_config = extract_config if isinstance(extract_config, dict) and extract_config else DEFAULT_EXTRACT_CONFIG
        extract_config = {**DEFAULT_EXTRACT_CONFIG, **graph_config, "enabled": True}
    elif isinstance(extract_config, dict):
        extract_config = {**extract_config, "enabled": False}
    error = validate_extract_config(extract_config)
    if error:
        raise InvalidInput(error)
    return {
        "type": "document",
        "indexing_strategy": strategy,
        "wiki_config": wiki_config,
        "extract_config": extract_config,
    }


def list_knowledge_bases(ctx: ServiceContext, params):
    qs = KnowledgeBase.objects.filter(
        tenant=ctx.tenant, deleted_at__isnull=True, is_temporary=False
    ).order_by("-is_pinned", "-updated_at")
    creator = params.get("creator")
    keyword = params.get("keyword") or params.get("q") or params.get("query")
    kb_type = params.get("type")
    if creator == "mine" and ctx.user:
        qs = qs.filter(creator_id=ctx.user.id)
    elif creator == "others" and ctx.user:
        qs = qs.exclude(creator_id=ctx.user.id)
    if keyword:
        qs = qs.filter(Q(name__icontains=keyword) | Q(description__icontains=keyword))
    if kb_type:
        if kb_type == "faq":
            qs = qs.none()
        elif kb_type == "wiki":
            qs = qs.filter(Q(type="wiki") | Q(indexing_strategy__wiki_enabled=True))
        else:
            qs = qs.filter(type=kb_type)
    page, meta = paginate(qs, params)
    items = [kb_dict(kb) for kb in page]
    for item in items:
        if ctx.user and item.get("creator_id") == ctx.user.id:
            item["creator_name"] = ctx.user.username
    return list_response(items, meta, ["knowledge_bases"])


def get_knowledge_base(ctx: ServiceContext, kb_id):
    return kb_dict(get_or_404(KnowledgeBase, id=kb_id, deleted_at__isnull=True))


def create_knowledge_base(ctx: ServiceContext, data):
    normalized = normalize_kb_payload(data)
    kb = KnowledgeBase.objects.create(
        tenant=ctx.tenant,
        name=data.get("name", "未命名知识库"),
        description=data.get("description", ""),
        type=normalized["type"],
        chunking_config=data.get("chunking_config") or default_chunk_config(),
        image_processing_config=data.get("image_processing_config") or {"enable_multimodal": False, "model_id": ""},
        embedding_model_id=data.get("embedding_model_id", ""),
        summary_model_id=data.get("summary_model_id", ""),
        storage_provider_config=data.get("storage_provider_config") or {"provider": "local"},
        vlm_config=data.get("vlm_config") or {},
        asr_config=data.get("asr_config"),
        extract_config=normalized["extract_config"],
        wiki_config=normalized["wiki_config"],
        indexing_strategy=normalized["indexing_strategy"],
        vector_store_id=data.get("vector_store_id") or "",
        creator_id=ctx.user.id if ctx.user else "",
    )
    return kb_dict(kb)


def update_knowledge_base(ctx: ServiceContext, kb_id, data):
    kb = get_or_404(KnowledgeBase, id=kb_id, deleted_at__isnull=True)
    config = data.get("config") or data
    normalized = normalize_kb_payload(data, kb, partial=True)
    for field in ["name", "description"]:
        if field in data:
            setattr(kb, field, data[field])
    for field in ["chunking_config", "image_processing_config"]:
        if field in config:
            setattr(kb, field, config[field])
    for field, value in normalized.items():
        setattr(kb, field, value)
    kb.save()
    return kb_dict(kb)


def delete_knowledge_base(ctx: ServiceContext, kb_id):
    kb = get_or_404(KnowledgeBase, id=kb_id, deleted_at__isnull=True)
    delete_kb_graph(kb)
    cleanup_wiki_for_kb(kb)
    kb.deleted_at = timezone.now()
    kb.save(update_fields=["deleted_at", "updated_at"])
    return {}


def set_kb_pin(ctx: ServiceContext, kb_id, pinned: bool | None):
    kb = get_or_404(KnowledgeBase, id=kb_id)
    kb.is_pinned = not kb.is_pinned if pinned is None else pinned
    kb.pinned_at = timezone.now() if kb.is_pinned else None
    kb.save(update_fields=["is_pinned", "pinned_at", "updated_at"])
    return kb_dict(kb)


def kb_copy(ctx: ServiceContext, data):
    src = get_or_404(KnowledgeBase, id=data.get("source_id"), tenant=ctx.tenant)
    clone = KnowledgeBase.objects.create(
        tenant=ctx.tenant,
        name=f"{src.name} copy",
        description=src.description,
        type=src.type,
        chunking_config=src.chunking_config,
        image_processing_config=src.image_processing_config,
        embedding_model_id=src.embedding_model_id,
        summary_model_id=src.summary_model_id,
        storage_provider_config=src.storage_provider_config,
        vlm_config=src.vlm_config,
        asr_config=src.asr_config,
        extract_config=src.extract_config,
        wiki_config=src.wiki_config,
        indexing_strategy=normalize_indexing_strategy(src.indexing_strategy, src.type),
        creator_id=ctx.user.id if ctx.user else "",
    )
    return {"knowledge_base": kb_dict(clone), "task_id": ""}


def kb_move_targets(ctx: ServiceContext, kb_id):
    source = get_or_404(KnowledgeBase, id=kb_id)
    qs = KnowledgeBase.objects.filter(tenant=ctx.tenant, type=source.type, deleted_at__isnull=True).exclude(id=kb_id)
    return {"items": [kb_dict(kb) for kb in qs]}
