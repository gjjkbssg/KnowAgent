import base64
import hashlib
import mimetypes
import time
from pathlib import Path

from django.conf import settings
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage

from .document_parsing.images import normalize_for_vlm
from .model_providers import ModelAccessDeniedError, vision_completion
from .models import Chunk, Knowledge, KnowledgeImage
from .ocr import paddle_ocr


OCR_PROMPT = """请对图片执行精确 OCR。按自然阅读顺序输出 Markdown；保留标题、段落、列表、表格、公式和关键布局。不要描述图片，也不要补充图片中不存在的内容。没有可识别文字时返回空字符串。"""
CAPTION_PROMPT = """请用中文描述图片中可用于知识库检索的客观信息。说明对象、结构、关系、图表趋势、关键标签和数值；不要猜测不可见信息，不要复述任务要求。"""
OCR_FALLBACK_MEAN_SCORE = 0.85


def _extension(mime_type: str) -> str:
    return mimetypes.guess_extension(mime_type) or ".bin"


def _image_info(image: KnowledgeImage) -> dict:
    return {
        "image_id": image.id,
        "source_type": image.source_type,
        "source_ref": image.source_ref,
        "mime_type": image.mime_type,
        "width": image.width,
        "height": image.height,
        "page_index": image.page_index,
        "block_index": image.block_index,
        "preview_url": f"/api/v1/knowledge/{image.knowledge_id}/images/{image.id}",
    }


def cleanup_knowledge_images(knowledge: Knowledge) -> None:
    owned_paths = set(
        KnowledgeImage.objects.filter(knowledge=knowledge, storage_owned=True).values_list("storage_path", flat=True)
    )
    KnowledgeImage.objects.filter(knowledge=knowledge).delete()
    for path in owned_paths:
        if path and not KnowledgeImage.objects.filter(storage_path=path, storage_owned=True).exists():
            default_storage.delete(path)


def _vlm_ocr(data_url: str, knowledge: Knowledge, model_id: str):
    try:
        text = vision_completion(
            knowledge.tenant,
            data_url,
            OCR_PROMPT,
            "image_ocr",
            model_id,
        ).strip()
        return text, [], ""
    except ModelAccessDeniedError as exc:
        circuit_error = f"OCR: {exc}"
        return "", [circuit_error], circuit_error
    except Exception as exc:
        return "", [f"OCR: {exc}"], ""


def _recognize_with_data_url(
    data: bytes,
    data_url: str,
    knowledge: Knowledge,
    engine: str,
    allow_vlm: bool = True,
    circuit_error: str = "",
):
    if engine not in {"vlm", "paddle", "paddle_vlm_fallback"}:
        raise ValueError(f"Unsupported OCR_ENGINE: {engine}")

    started = time.perf_counter()
    metadata = {
        "engine": engine,
        "mean_score": None,
        "low_score_ratio": None,
        "fallback_reason": "",
    }
    model_id = str((knowledge.knowledge_base.vlm_config or {}).get("model_id") or "")
    if engine == "vlm":
        if not allow_vlm:
            errors = [circuit_error] if circuit_error else []
            return "", metadata, errors, circuit_error, (time.perf_counter() - started) * 1000
        text, errors, circuit_error = _vlm_ocr(data_url, knowledge, model_id)
        return (
            text,
            metadata,
            errors,
            circuit_error,
            (time.perf_counter() - started) * 1000,
        )

    errors = []
    paddle_result = None
    fallback_reason = ""
    try:
        paddle_result = paddle_ocr(data)
    except Exception as exc:
        errors.append(f"Paddle OCR: {exc}")
        fallback_reason = f"paddle_error:{type(exc).__name__}"
    else:
        metadata["mean_score"] = paddle_result.mean_score
        metadata["low_score_ratio"] = paddle_result.low_score_ratio
        if engine == "paddle_vlm_fallback":
            if not paddle_result.lines:
                fallback_reason = "no_lines"
            elif paddle_result.mean_score < OCR_FALLBACK_MEAN_SCORE:
                fallback_reason = "mean_score_below_threshold"
            elif paddle_result.low_score_ratio > settings.OCR_FALLBACK_RATIO:
                fallback_reason = "low_score_ratio_above_threshold"

    if not fallback_reason or not allow_vlm:
        metadata["engine"] = "paddle"
        if circuit_error and circuit_error not in errors:
            errors.append(circuit_error)
        return (
            paddle_result.text if paddle_result else "",
            metadata,
            errors,
            circuit_error,
            (time.perf_counter() - started) * 1000,
        )

    metadata["engine"] = "vlm"
    metadata["fallback_reason"] = fallback_reason
    text, vlm_errors, circuit_error = _vlm_ocr(data_url, knowledge, model_id)
    errors.extend(vlm_errors)
    if paddle_result and paddle_result.lines and not text:
        metadata["engine"] = "paddle"
        metadata["fallback_failed"] = True
        if not vlm_errors:
            errors.append("VLM OCR returned no text")
        return (
            paddle_result.text,
            metadata,
            errors,
            circuit_error,
            (time.perf_counter() - started) * 1000,
        )
    return (
        text,
        metadata,
        errors,
        circuit_error,
        (time.perf_counter() - started) * 1000,
    )


def recognize_image(
    data: bytes,
    mime_type: str,
    knowledge: Knowledge,
    engine: str | None = None,
):
    normalized, normalized_mime, _, _ = normalize_for_vlm(data, mime_type)
    data_url = f"data:{normalized_mime};base64,{base64.b64encode(normalized).decode('ascii')}"
    return _recognize_with_data_url(
        normalized,
        data_url,
        knowledge,
        (engine or settings.OCR_ENGINE).strip().lower(),
    )


def analyze_image(
    data: bytes,
    mime_type: str,
    knowledge: Knowledge,
    *,
    allow_vlm: bool = True,
    circuit_error: str = "",
):
    normalized, normalized_mime, _, _ = normalize_for_vlm(data, mime_type)
    data_url = f"data:{normalized_mime};base64,{base64.b64encode(normalized).decode('ascii')}"
    model_id = str((knowledge.knowledge_base.vlm_config or {}).get("model_id") or "")
    ocr_text, ocr_metadata, errors, circuit_error, _ = _recognize_with_data_url(
        normalized,
        data_url,
        knowledge,
        settings.OCR_ENGINE,
        allow_vlm=allow_vlm,
        circuit_error=circuit_error,
    )
    if circuit_error or not allow_vlm:
        if circuit_error and circuit_error not in errors:
            errors.append(circuit_error)
        return ocr_text, "", errors, circuit_error, ocr_metadata
    try:
        caption = vision_completion(knowledge.tenant, data_url, CAPTION_PROMPT, "image_caption", model_id).strip()
    except ModelAccessDeniedError as exc:
        circuit_error = f"Caption: {exc}"
        return ocr_text, "", [*errors, circuit_error], circuit_error, ocr_metadata
    except Exception as exc:
        caption = ""
        errors.append(f"Caption: {exc}")
    return ocr_text, caption, errors, "", ocr_metadata


def _nearest_parent(text_chunks: list[Chunk], block_index: int) -> Chunk | None:
    candidates = [chunk for chunk in text_chunks if int((chunk.metadata or {}).get("block_index", -1)) <= block_index]
    return candidates[-1] if candidates else None


def process_document_images(knowledge: Knowledge, image_blocks, text_chunks: list[Chunk]) -> tuple[list[Chunk], list[dict]]:
    chunks = []
    warnings = []
    analysis_cache: dict[str, tuple[str, str, list[str], str, dict]] = {}
    circuit_error = ""
    next_index = max((chunk.chunk_index for chunk in text_chunks), default=-1) + 1

    for block in sorted(image_blocks, key=lambda item: item.block_index):
        content_hash = hashlib.sha256(block.data).hexdigest()
        if block.source_type == "standalone":
            storage_path = knowledge.file_path
            storage_owned = False
        else:
            storage_path = f"tenant-{knowledge.tenant_id}/{knowledge.knowledge_base_id}/{knowledge.id}/images/{content_hash}{_extension(block.mime_type)}"
            if not default_storage.exists(storage_path):
                storage_path = default_storage.save(storage_path, ContentFile(block.data))
            storage_owned = True

        image = KnowledgeImage.objects.create(
            tenant=knowledge.tenant,
            knowledge_base=knowledge.knowledge_base,
            knowledge=knowledge,
            content_hash=content_hash,
            storage_path=storage_path,
            storage_owned=storage_owned,
            mime_type=block.mime_type,
            width=block.width,
            height=block.height,
            source_type=block.source_type,
            source_ref=block.source_ref,
            page_index=block.page_index,
            block_index=block.block_index,
            metadata=block.metadata,
        )

        if circuit_error:
            if settings.OCR_ENGINE in {"paddle", "paddle_vlm_fallback"}:
                analysis = analyze_image(
                    block.data,
                    block.mime_type,
                    knowledge,
                    allow_vlm=False,
                    circuit_error=circuit_error,
                )
            else:
                analysis = "", "", [circuit_error], circuit_error, {}
        else:
            analysis = analysis_cache.get(content_hash)
            if analysis is None:
                analysis = analyze_image(block.data, block.mime_type, knowledge)
                analysis_cache[content_hash] = analysis
        ocr_text, caption, errors, analysis_circuit_error, ocr_metadata = analysis
        image.metadata = {
            **(image.metadata or {}),
            "ocr": ocr_metadata,
        }
        if analysis_circuit_error:
            circuit_error = analysis_circuit_error
        image.ocr_text = ocr_text
        image.caption = caption
        image.error_message = "; ".join(errors)
        image.status = "completed" if ocr_text and caption else "partial" if ocr_text or caption else "failed"
        image.save(
            update_fields=[
                "ocr_text",
                "caption",
                "error_message",
                "status",
                "metadata",
                "updated_at",
            ]
        )

        parent = _nearest_parent(text_chunks, block.block_index)
        image_chunk_metadata = {
            "title": knowledge.title,
            "block_index": block.block_index,
            "ocr_engine": ocr_metadata.get("engine"),
            "ocr_mean_score": ocr_metadata.get("mean_score"),
            "ocr_low_score_ratio": ocr_metadata.get("low_score_ratio"),
            "ocr_fallback_reason": ocr_metadata.get("fallback_reason", ""),
        }
        if parent is None:
            parent = Chunk.objects.create(
                tenant=knowledge.tenant,
                knowledge_base=knowledge.knowledge_base,
                knowledge=knowledge,
                content=f"[图片：{block.source_ref or knowledge.title}]",
                chunk_index=next_index,
                chunk_type="image_container",
                is_enabled=False,
                image_info=_image_info(image),
                metadata=image_chunk_metadata,
                content_hash=hashlib.sha256(f"container:{image.id}".encode()).hexdigest(),
            )
            chunks.append(parent)
            next_index += 1

        for chunk_type, content in (("image_ocr", ocr_text), ("image_caption", caption)):
            if not content:
                continue
            chunk = Chunk.objects.create(
                tenant=knowledge.tenant,
                knowledge_base=knowledge.knowledge_base,
                knowledge=knowledge,
                content=content,
                chunk_index=next_index,
                chunk_type=chunk_type,
                parent_chunk_id=parent.id,
                image_info=_image_info(image),
                metadata=image_chunk_metadata,
                content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
            )
            chunks.append(chunk)
            next_index += 1
        if errors:
            warnings.append({"stage": "multimodal", "image_id": image.id, "message": image.error_message})
    return chunks, warnings
