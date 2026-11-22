import hashlib
import mimetypes
import time
import uuid
from collections.abc import Mapping
from pathlib import Path

from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.core.cache import cache
from django.db import IntegrityError, transaction
from django.db.models import Q
from django.utils import timezone

from personal_knowledge_base.document_processing import (
    detect_file_type,
    is_unsupported_media_file,
    process_knowledge,
)
from personal_knowledge_base.document_processing import process_graph as rebuild_knowledge_graph
from personal_knowledge_base.graph_rag import delete_knowledge_graph
from personal_knowledge_base.http_helpers import bounded_int, csv_values, list_response, paginate
from personal_knowledge_base.model_providers import EmbeddingUnavailableError, safe_json
from personal_knowledge_base.models import Chunk, Knowledge, KnowledgeBase, KnowledgeImage, TaskRecord
from personal_knowledge_base.multimodal import cleanup_knowledge_images
from personal_knowledge_base.search import delete_chunk_index, hybrid_search, update_chunk_index_scope
from personal_knowledge_base.serializers import knowledge_dict
from personal_knowledge_base.tasks import enqueue
from personal_knowledge_base.wiki_ingest import (
    cleanup_wiki_for_kb,
    cleanup_wiki_for_knowledge,
    enqueue_wiki_ingest,
    prepare_wiki_for_reparse,
)

from .context import ServiceContext
from .errors import InvalidInput, NotFound, ServiceError, get_or_404


KNOWLEDGE_TYPE_VALUES = {"file"}
PROCESSING_STATUSES = {"pending", "processing", "finalizing"}


def normalize_ids(data):
    ids = data.get("ids") or data.get("knowledge_ids") or data.get("knowledgeIds") or []
    if isinstance(ids, str):
        ids = csv_values(ids)
    seen = set()
    result = []
    for item in ids:
        item = str(item).strip()
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def extract_process_config(data, default=None):
    for key in ["process_config", "processConfig"]:
        if key in data:
            return safe_json(data.get(key))
    return default


def with_process_config(metadata, process_config, include_empty=False):
    metadata = dict(metadata or {})
    if process_config is None and not include_empty:
        return metadata
    config = process_config or {}
    metadata["process_config"] = config
    metadata["process_overrides"] = config
    return metadata


def delete_knowledge_content(item, cleanup_wiki=True):
    if cleanup_wiki:
        try:
            cleanup_wiki_for_knowledge(item)
        except Exception:
            pass
    delete_knowledge_graph(item)
    for chunk in Chunk.objects.filter(knowledge=item):
        delete_chunk_index(chunk.id, chunk.seq_id)
    Chunk.objects.filter(knowledge=item).delete()
    cleanup_knowledge_images(item)


def matching_process_tasks(knowledge_id):
    target = str(knowledge_id)
    return [
        (task_id, payload, progress)
        for task_id, payload, progress in TaskRecord.objects.filter(task_type="process_knowledge").values_list(
            "id", "payload", "progress"
        )
        if isinstance(payload, dict) and str(payload.get("knowledge_id") or "") == target
    ]


def retire_knowledge_for_delete(item):
    deleted_at = timezone.now()
    Knowledge.objects.filter(id=item.id).update(
        parse_status="cancelled",
        deleted_at=deleted_at,
        updated_at=deleted_at,
    )
    item.parse_status = "cancelled"
    item.deleted_at = deleted_at
    task_ids = [task_id for task_id, _, _ in matching_process_tasks(item.id)]
    TaskRecord.objects.filter(id__in=task_ids).delete()
    for task_id in task_ids:
        cache.delete(f"task:{task_id}")


def apply_knowledge_filters(qs, params: Mapping):
    keyword = params.get("keyword") or params.get("q") or params.get("query")
    if keyword:
        qs = qs.filter(
            Q(title__icontains=keyword)
            | Q(file_name__icontains=keyword)
            | Q(source__icontains=keyword)
            | Q(description__icontains=keyword)
            | Q(metadata__content__icontains=keyword)
        )
    tag_id = params.get("tag_id") or params.get("tagId")
    if tag_id:
        qs = qs.filter(tag_id=tag_id)
    statuses = csv_values(params.get("parse_status") or params.get("parseStatus"))
    if len(statuses) == 1:
        qs = qs.filter(parse_status=statuses[0])
    elif statuses:
        qs = qs.filter(parse_status__in=statuses)
    file_values = csv_values(params.get("file_types") or params.get("file_type") or params.get("fileType"))
    if file_values:
        file_query = Q()
        for value in file_values:
            file_query |= Q(type=value) if value in KNOWLEDGE_TYPE_VALUES else Q(file_type=value)
        qs = qs.filter(file_query)
    source = params.get("source") or params.get("channel")
    if source:
        qs = qs.filter(Q(type=source) if source in KNOWLEDGE_TYPE_VALUES else Q(channel=source) | Q(source=source))
    start_time = params.get("start_time") or params.get("startTime")
    end_time = params.get("end_time") or params.get("endTime")
    if start_time:
        qs = qs.filter(created_at__gte=start_time)
    if end_time:
        qs = qs.filter(created_at__lte=end_time)
    return qs


def knowledge_collection(ctx: ServiceContext, kb_id, *, method: str, params: Mapping):
    kb = get_or_404(KnowledgeBase, id=kb_id)
    if method == "GET":
        qs = Knowledge.objects.filter(knowledge_base=kb, deleted_at__isnull=True).order_by("-updated_at")
        qs = apply_knowledge_filters(qs, params)
        page, meta = paginate(qs, params)
        return list_response([knowledge_dict(item) for item in page], meta, ["knowledge"])
    if method == "DELETE":
        for item in Knowledge.objects.filter(knowledge_base=kb):
            delete_knowledge_content(item)
        Knowledge.objects.filter(knowledge_base=kb).update(deleted_at=timezone.now())
        cleanup_wiki_for_kb(kb)
        return {}
    raise ServiceError("method not allowed", status=405)


def knowledge_file(ctx: ServiceContext, kb_id, uploaded, *, form_data: Mapping):
    kb = get_or_404(KnowledgeBase, id=kb_id)
    if not uploaded:
        raise InvalidInput("file is required")
    file_type = detect_file_type(uploaded.name)
    if is_unsupported_media_file(uploaded.name):
        raise InvalidInput(
            "不支持音频或视频文件",
            code="unsupported_file_type",
            details={"file_name": uploaded.name, "file_type": file_type},
        )
    content = uploaded.read()
    file_hash = hashlib.sha256(content).hexdigest()
    existing = Knowledge.objects.filter(
        knowledge_base=kb,
        file_hash=file_hash,
        file_name=uploaded.name,
        deleted_at__isnull=True,
    ).order_by("-created_at").first()
    if existing:
        return {"data": {"knowledge": knowledge_dict(existing), "task_id": "", "deduplicated": True}}
    path = default_storage.save(f"tenant-{ctx.tenant.id}/{kb.id}/{uuid.uuid4()}-{uploaded.name}", ContentFile(content))
    item = None
    for attempt in range(3):
        try:
            with transaction.atomic():
                existing = Knowledge.objects.select_for_update().filter(
                    knowledge_base=kb,
                    file_hash=file_hash,
                    file_name=uploaded.name,
                    deleted_at__isnull=True,
                ).order_by("-created_at").first()
                if existing:
                    default_storage.delete(path)
                    return {"data": {"knowledge": knowledge_dict(existing), "task_id": "", "deduplicated": True}}
                item = Knowledge.objects.create(
                    tenant=ctx.tenant,
                    knowledge_base=kb,
                    type="file",
                    title=form_data.get("fileName") or uploaded.name,
                    source=uploaded.name,
                    parse_status="pending",
                    file_name=uploaded.name,
                    file_type=file_type,
                    file_size=len(content),
                    file_path=path,
                    file_hash=file_hash,
                    storage_size=len(content),
                    tag_id=form_data.get("tag_id", ""),
                    metadata=with_process_config({}, safe_json(form_data.get("process_config")), include_empty=True),
                )
            break
        except IntegrityError:
            default_storage.delete(path)
            existing = Knowledge.objects.filter(
                knowledge_base=kb,
                file_hash=file_hash,
                file_name=uploaded.name,
                deleted_at__isnull=True,
            ).order_by("-created_at").first()
            if existing:
                return {"data": {"knowledge": knowledge_dict(existing), "task_id": "", "deduplicated": True}}
            raise
        except Exception as exc:
            if "database is locked" in str(exc) and attempt < 2:
                time.sleep(2 * (attempt + 1))
                continue
            default_storage.delete(path)
            raise
    if item is None:
        default_storage.delete(path)
        raise ServiceError("upload failed after retries", status=500)
    task = enqueue("process_knowledge", lambda: (process_knowledge(item.id), {"knowledge_id": item.id})[1], {"knowledge_id": item.id})
    return {"data": {"knowledge": knowledge_dict(item), "task_id": task.id}, "status": 201}


def knowledge_detail(ctx: ServiceContext, knowledge_id, *, method: str, data=None):
    item = get_or_404(Knowledge, id=knowledge_id)
    if method == "GET":
        return knowledge_dict(item)
    if method == "DELETE":
        retire_knowledge_for_delete(item)
        delete_knowledge_content(item)
        return {"id": knowledge_id, "task_id": ""}
    data = data or {}
    lifecycle_status = data.get("lifecycle_status")
    if "lifecycle_status" in data and lifecycle_status not in {"current", "superseded"}:
        raise InvalidInput("invalid lifecycle_status")
    superseded_by = str(data.get("superseded_by") or "").strip()
    if "superseded_by" in data and lifecycle_status != "current" and superseded_by:
        if not Knowledge.objects.filter(id=superseded_by, tenant_id=item.tenant_id).exists():
            raise InvalidInput("superseded_by must reference knowledge in the same tenant")
    for field in ["title", "description", "enable_status", "tag_id"]:
        if field in data:
            setattr(item, field, data[field])
    if lifecycle_status == "current":
        metadata = dict(item.metadata or {})
        metadata.pop("lifecycle", None)
        item.metadata = metadata
    elif lifecycle_status == "superseded":
        metadata = dict(item.metadata or {})
        lifecycle = dict(metadata.get("lifecycle") or {})
        lifecycle["status"] = "superseded"
        lifecycle["superseded_by"] = superseded_by if "superseded_by" in data else lifecycle.get("superseded_by", "")
        metadata["lifecycle"] = lifecycle
        item.metadata = metadata
    elif "superseded_by" in data:
        metadata = dict(item.metadata or {})
        lifecycle = metadata.get("lifecycle")
        if isinstance(lifecycle, dict) and lifecycle.get("status") == "superseded":
            lifecycle = dict(lifecycle)
            lifecycle["superseded_by"] = superseded_by
            metadata["lifecycle"] = lifecycle
            item.metadata = metadata
    item.save()
    return knowledge_dict(item)


def knowledge_reparse(ctx: ServiceContext, knowledge_id, data=None):
    item = get_or_404(Knowledge, id=knowledge_id)
    prepare_wiki_for_reparse(item)
    delete_knowledge_content(item, cleanup_wiki=False)
    process_config = extract_process_config(data or {})
    update_fields = ["parse_status", "updated_at"]
    if process_config is not None:
        item.metadata = with_process_config(item.metadata, process_config)
        update_fields.append("metadata")
    item.parse_status = "pending"
    item.save(update_fields=update_fields)
    task = enqueue("process_knowledge", lambda: (process_knowledge(item.id), {"knowledge_id": item.id})[1], {"knowledge_id": item.id})
    return {"knowledge": knowledge_dict(item), "task_id": task.id}


def knowledge_cancel(ctx: ServiceContext, knowledge_id):
    item = get_or_404(Knowledge, id=knowledge_id)
    item.parse_status = "cancelled"
    item.save(update_fields=["parse_status", "updated_at"])
    for task_id, payload, progress in matching_process_tasks(item.id):
        if TaskRecord.objects.filter(id=task_id, status__in=["pending", "running"]).update(
            status="cancelled",
            payload={key: value for key, value in payload.items() if key != "_worker_token"},
            error_message="cancelled by user",
            updated_at=timezone.now(),
        ):
            cache.set(f"task:{task_id}", {"status": "cancelled", "progress": progress, "error_message": "cancelled by user"}, timeout=86400)
    return knowledge_dict(item)


def knowledge_batch_delete(ctx: ServiceContext, data, *, kb_id=None):
    source_kb_id = kb_id or data.get("kb_id") or data.get("knowledge_base_id") or data.get("source_kb_id")
    ids = normalize_ids(data)
    count = 0
    qs = Knowledge.objects.filter(id__in=ids, tenant=ctx.tenant, deleted_at__isnull=True)
    if source_kb_id:
        qs = qs.filter(knowledge_base_id=source_kb_id)
    for item in qs:
        retire_knowledge_for_delete(item)
        delete_knowledge_content(item)
        count += 1
    return {"deleted": count, "deleted_count": count, "ids": ids, "kb_id": source_kb_id, "task_id": ""}


def knowledge_move(ctx: ServiceContext, data, *, kb_id=None):
    ids = normalize_ids(data)
    source_kb_id = kb_id or data.get("source_kb_id") or data.get("kb_id")
    target_id = data.get("target_kb_id") or data.get("target_knowledge_base_id") or data.get("knowledge_base_id")
    target = get_or_404(KnowledgeBase, id=target_id, tenant=ctx.tenant)
    moved = 0
    qs = Knowledge.objects.filter(id__in=ids, tenant=ctx.tenant, deleted_at__isnull=True)
    if source_kb_id:
        qs = qs.filter(knowledge_base_id=source_kb_id)
    for item in qs:
        old_kb_id = item.knowledge_base_id
        cleanup_wiki_for_knowledge(item)
        delete_knowledge_graph(item)
        item.knowledge_base = target
        item.tenant = target.tenant
        item.save(update_fields=["knowledge_base", "tenant", "updated_at"])
        KnowledgeImage.objects.filter(knowledge=item).update(knowledge_base=target, tenant=target.tenant)
        chunks = []
        for chunk in Chunk.objects.filter(knowledge=item):
            chunk.knowledge_base = target
            chunk.tenant = target.tenant
            chunk.relation_chunks = None
            chunk.indirect_relation_chunks = None
            chunk.save(update_fields=["knowledge_base", "tenant", "updated_at"])
            update_chunk_index_scope(chunk)
            chunks.append(chunk)
        try:
            rebuild_knowledge_graph(item, chunks)
        except Exception:
            metadata = dict(item.metadata or {})
            warnings = list(metadata.get("processing_warnings") or [])
            warnings.append({"stage": "graph_move_rebuild", "message": f"graph rebuild skipped after move from {old_kb_id}"})
            metadata["processing_warnings"] = warnings
            item.metadata = metadata
            item.save(update_fields=["metadata", "updated_at"])
        try:
            enqueue_wiki_ingest(item)
        except Exception:
            metadata = dict(item.metadata or {})
            warnings = list(metadata.get("processing_warnings") or [])
            warnings.append({"stage": "wiki_move_rebuild", "message": f"wiki rebuild skipped after move from {old_kb_id}"})
            metadata["processing_warnings"] = warnings
            item.metadata = metadata
            item.save(update_fields=["metadata", "updated_at"])
        moved += 1
    return {
        "moved": moved,
        "knowledge_count": moved,
        "source_kb_id": source_kb_id,
        "target_kb_id": target.id,
        "target_knowledge_base_id": target.id,
        "task_id": "",
        "message": "Knowledge move task started",
    }


def knowledge_batch(ctx: ServiceContext, params: Mapping):
    ids = params.get("ids", "")
    items = Knowledge.objects.filter(
        id__in=[x for x in ids.split(",") if x], tenant=ctx.tenant, deleted_at__isnull=True
    )
    return {"items": [knowledge_dict(item) for item in items]}


def knowledge_search(ctx: ServiceContext, params: Mapping):
    qs = Knowledge.objects.filter(tenant=ctx.tenant, deleted_at__isnull=True)
    qs = apply_knowledge_filters(qs, params)
    page, meta = paginate(qs, params)
    return {"items": [knowledge_dict(item) for item in page], **meta}


def knowledge_search_post(ctx: ServiceContext, data):
    kb_ids = data.get("knowledge_base_ids") or data.get("kb_ids") or []
    query = data.get("query") or data.get("q") or ""
    top_k = bounded_int(data.get("top_k"), 10, 1, 100)
    results = hybrid_search(ctx.tenant.id, kb_ids, query, top_k)
    return {"items": results, "results": results}


def kb_hybrid_search(ctx: ServiceContext, kb_id, data):
    query = data.get("query") or data.get("q") or ""
    top_k = bounded_int(data.get("top_k") or data.get("limit"), 10, 1, 100)
    results = hybrid_search(ctx.tenant.id, [kb_id], query, top_k)
    return {"items": results, "results": results}


def knowledge_stats(ctx: ServiceContext, kb_id):
    kb = get_or_404(KnowledgeBase, id=kb_id)
    qs = Knowledge.objects.filter(knowledge_base=kb, deleted_at__isnull=True)
    total = qs.count()
    counts = {status: qs.filter(parse_status=status).count() for status in ["pending", "processing", "finalizing", "completed", "failed", "cancelled"]}
    processing = qs.filter(parse_status__in=PROCESSING_STATUSES).count()
    return {
        "knowledge_base_id": kb.id,
        "knowledge_count": total,
        "document_count": total,
        "total": total,
        "completed": counts["completed"],
        "processing": processing,
        "pending": counts["pending"],
        "failed": counts["failed"],
        "cancelled": counts["cancelled"],
        "chunk_count": Chunk.objects.filter(knowledge_base=kb, deleted_at__isnull=True).count(),
        "storage_size": sum(size or 0 for size in qs.values_list("storage_size", flat=True)),
        "status_counts": counts,
        "is_processing": processing > 0,
    }


def knowledge_spans(ctx: ServiceContext, knowledge_id):
    from personal_knowledge_base.span_tracker import SpanTracker

    item = get_or_404(Knowledge, id=knowledge_id)
    spans = SpanTracker(knowledge_id).get_spans()
    if not spans:
        spans = [{"name": "parse", "status": item.parse_status, "started_at": item.created_at.isoformat(), "finished_at": item.processed_at.isoformat() if item.processed_at else None}]
    return {"items": spans}


def knowledge_download(ctx: ServiceContext, knowledge_id):
    item = get_or_404(Knowledge, id=knowledge_id)
    filename = item.file_name or f"{item.title or item.id}.txt"
    if not item.file_path:
        return {"content": item.metadata.get("content", ""), "filename": filename, "as_attachment": True}
    return {"path": item.file_path, "filename": filename, "as_attachment": True}


def knowledge_preview(ctx: ServiceContext, knowledge_id):
    item = get_or_404(Knowledge, id=knowledge_id)
    filename = item.file_name or f"{item.title or item.id}.txt"
    file_type = detect_file_type(filename)
    inline_types = {"txt", "md", "markdown", "csv", "json", "log", "pdf", "jpg", "jpeg", "png", "gif", "bmp", "webp"}
    if item.file_path and file_type in inline_types:
        content_type = mimetypes.guess_type(filename)[0] or (
            "text/plain; charset=utf-8" if file_type in {"txt", "md", "markdown", "csv", "json", "log"} else "application/octet-stream"
        )
        return {"path": item.file_path, "filename": filename, "content_type": content_type}
    chunks = Chunk.objects.filter(knowledge=item, deleted_at__isnull=True).order_by("chunk_index")
    text = "\n\n".join(chunk.content for chunk in chunks)
    if not text:
        metadata = item.metadata or {}
        text = metadata.get("content") or metadata.get("summary") or item.error_message or "该文件暂无可预览文本。"
    return {"content": text, "filename": f"{Path(filename).stem or item.title}.txt", "content_type": "text/plain; charset=utf-8"}


def knowledge_image_content(ctx: ServiceContext, knowledge_id, image_id):
    image = get_or_404(
        KnowledgeImage,
        id=image_id,
        knowledge_id=knowledge_id,
        tenant=ctx.tenant,
        knowledge__tenant=ctx.tenant,
        knowledge__deleted_at__isnull=True,
    )
    if not image.storage_path or not default_storage.exists(image.storage_path):
        raise NotFound("image not found")
    return {"path": image.storage_path, "content_type": image.mime_type}
