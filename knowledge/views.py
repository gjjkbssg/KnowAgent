from django.core.files.storage import default_storage
from django.http import FileResponse, HttpResponse
from django.utils.http import content_disposition_header
from django.views.decorators.csrf import csrf_exempt

from personal_knowledge_base.authentication import tenant_scoped
from personal_knowledge_base.http_helpers import bool_from_value, parse_body, service_context
from personal_knowledge_base.responses import fail, ok, service_errors
from services.chunks import (
    delete_chunk,
    delete_knowledge_chunks,
    get_chunk,
    list_chunks,
    update_chunk,
)
from services.knowledge import (
    knowledge_batch as knowledge_batch_service,
    knowledge_batch_delete as knowledge_batch_delete_service,
    knowledge_cancel as knowledge_cancel_service,
    delete_knowledge as delete_knowledge_service,
    delete_knowledge_collection as delete_knowledge_collection_service,
    knowledge_download as knowledge_download_service,
    get_knowledge as get_knowledge_service,
    kb_hybrid_search as kb_hybrid_search_service,
    list_knowledge as list_knowledge_service,
    knowledge_image_content as knowledge_image_content_service,
    knowledge_move as knowledge_move_service,
    knowledge_preview as knowledge_preview_service,
    knowledge_reparse as knowledge_reparse_service,
    knowledge_search as knowledge_search_service,
    knowledge_search_post as knowledge_search_post_service,
    knowledge_spans as knowledge_spans_service,
    knowledge_stats as knowledge_stats_service,
    update_knowledge as update_knowledge_service,
    upload_knowledge_file as upload_knowledge_file_service,
)
from services.knowledge_bases import (
    kb_copy as kb_copy_service,
    kb_move_targets as kb_move_targets_service,
    create_knowledge_base as create_knowledge_base_service,
    delete_knowledge_base as delete_knowledge_base_service,
    get_knowledge_base as get_knowledge_base_service,
    list_knowledge_bases as list_knowledge_bases_service,
    set_kb_pin as set_kb_pin_service,
    update_knowledge_base as update_knowledge_base_service,
)
from services.tags import (
    create_knowledge_tag as create_knowledge_tag_service,
    delete_knowledge_tag as delete_knowledge_tag_service,
    list_knowledge_tags as list_knowledge_tags_service,
    update_knowledge_tag as update_knowledge_tag_service,
)


@csrf_exempt
@tenant_scoped
@service_errors
def knowledge_bases(request, kb_id=None):
    ctx = service_context(request)
    if kb_id:
        if request.method == "GET":
            return ok(get_knowledge_base_service(ctx, kb_id))
        if request.method == "DELETE":
            return ok(delete_knowledge_base_service(ctx, kb_id))
        return ok(update_knowledge_base_service(ctx, kb_id, parse_body(request)))
    if request.method == "GET":
        return ok(list_knowledge_bases_service(ctx, request.GET))
    return ok(create_knowledge_base_service(ctx, parse_body(request)), status=201)


@csrf_exempt
@tenant_scoped
@service_errors
def kb_pin(request, kb_id):
    if request.method == "DELETE":
        pinned = False
    elif request.method in {"POST", "PUT"}:
        data = parse_body(request)
        value = data.get("is_pinned")
        pinned = None if value is None else bool_from_value(value)
    else:
        pinned = None
    return ok(set_kb_pin_service(service_context(request), kb_id, pinned))


@csrf_exempt
@tenant_scoped
@service_errors
def kb_copy(request):
    return ok(kb_copy_service(service_context(request), parse_body(request)))


@tenant_scoped
@service_errors
def kb_move_targets(request, kb_id):
    return ok(kb_move_targets_service(service_context(request), kb_id))


@csrf_exempt
@tenant_scoped
@service_errors
def knowledge_collection(request, kb_id):
    ctx = service_context(request)
    if request.method == "GET":
        return ok(list_knowledge_service(ctx, kb_id, request.GET))
    if request.method == "DELETE":
        return ok(delete_knowledge_collection_service(ctx, kb_id))
    return fail("method not allowed", 405)


@csrf_exempt
@tenant_scoped
@service_errors
def knowledge_file(request, kb_id):
    result = upload_knowledge_file_service(
        service_context(request),
        kb_id,
        request.FILES.get("file"),
        form_data=request.POST,
    )
    payload = {"knowledge": result.knowledge, "task_id": result.task_id}
    if result.deduplicated:
        payload["deduplicated"] = True
    return ok(payload, status=200 if result.deduplicated else 201)


@csrf_exempt
@tenant_scoped
@service_errors
def knowledge_detail(request, knowledge_id):
    ctx = service_context(request)
    if request.method == "GET":
        return ok(get_knowledge_service(ctx, knowledge_id))
    if request.method == "DELETE":
        return ok(delete_knowledge_service(ctx, knowledge_id))
    return ok(update_knowledge_service(ctx, knowledge_id, parse_body(request)))


@csrf_exempt
@tenant_scoped
@service_errors
def knowledge_reparse(request, knowledge_id):
    return ok(knowledge_reparse_service(service_context(request), knowledge_id, parse_body(request)))


@csrf_exempt
@tenant_scoped
@service_errors
def knowledge_cancel(request, knowledge_id):
    return ok(knowledge_cancel_service(service_context(request), knowledge_id))


@csrf_exempt
@tenant_scoped
@service_errors
def knowledge_batch_delete(request, kb_id=None):
    return ok(knowledge_batch_delete_service(service_context(request), parse_body(request), kb_id=kb_id))


@csrf_exempt
@tenant_scoped
@service_errors
def knowledge_move(request, kb_id=None):
    return ok(knowledge_move_service(service_context(request), parse_body(request), kb_id=kb_id))


@tenant_scoped
@service_errors
def knowledge_batch(request):
    return ok(knowledge_batch_service(service_context(request), request.GET))


@tenant_scoped
@service_errors
def knowledge_search(request):
    return ok(knowledge_search_service(service_context(request), request.GET))


@csrf_exempt
@tenant_scoped
@service_errors
def knowledge_search_post(request):
    return ok(knowledge_search_post_service(service_context(request), parse_body(request)))


@tenant_scoped
@service_errors
def knowledge_stats(request, kb_id):
    return ok(knowledge_stats_service(service_context(request), kb_id))


@tenant_scoped
@service_errors
def knowledge_spans(request, knowledge_id):
    return ok(knowledge_spans_service(service_context(request), knowledge_id))


@tenant_scoped
@service_errors
def knowledge_download(request, knowledge_id):
    result = knowledge_download_service(service_context(request), knowledge_id)
    if "path" in result:
        return FileResponse(default_storage.open(result["path"], "rb"), as_attachment=True, filename=result["filename"])
    response = HttpResponse(result["content"], content_type="text/plain; charset=utf-8")
    response["Content-Disposition"] = content_disposition_header(True, result["filename"])
    return response


@tenant_scoped
@service_errors
def knowledge_preview(request, knowledge_id):
    result = knowledge_preview_service(service_context(request), knowledge_id)
    if "path" in result:
        response = FileResponse(
            default_storage.open(result["path"], "rb"),
            as_attachment=False,
            filename=result["filename"],
            content_type=result["content_type"],
        )
    else:
        response = HttpResponse(result["content"], content_type=result["content_type"])
    response["Content-Disposition"] = content_disposition_header(False, result["filename"])
    return response


@tenant_scoped
@service_errors
def knowledge_image_content(request, knowledge_id, image_id):
    result = knowledge_image_content_service(service_context(request), knowledge_id, image_id)
    return FileResponse(default_storage.open(result["path"], "rb"), content_type=result["content_type"], as_attachment=False)


@csrf_exempt
@tenant_scoped
@service_errors
def chunks_collection(request, knowledge_id=None, chunk_id=None):
    ctx = service_context(request)
    if knowledge_id == "by-id" and chunk_id:
        knowledge_id = None
    if chunk_id:
        if request.method == "GET":
            return ok(get_chunk(ctx, chunk_id, knowledge_id))
        if request.method == "DELETE":
            return ok(delete_chunk(ctx, chunk_id, knowledge_id))
        if knowledge_id:
            return ok(update_chunk(ctx, knowledge_id, chunk_id, parse_body(request)))
    if knowledge_id:
        if request.method == "GET":
            return ok(list_chunks(ctx, knowledge_id, request.GET))
        if request.method == "DELETE":
            return ok(delete_knowledge_chunks(ctx, knowledge_id))
    return fail("not found", 404)


@csrf_exempt
@tenant_scoped
@service_errors
def knowledge_tags(request, kb_id, tag_id=None):
    ctx = service_context(request)
    if request.method == "GET":
        return ok(list_knowledge_tags_service(ctx, kb_id))
    if tag_id and request.method == "DELETE":
        return ok(delete_knowledge_tag_service(ctx, kb_id, tag_id))
    if tag_id:
        return ok(update_knowledge_tag_service(ctx, kb_id, tag_id, parse_body(request)))
    return ok(create_knowledge_tag_service(ctx, kb_id, parse_body(request)), status=201)


@csrf_exempt
@tenant_scoped
@service_errors
def kb_hybrid_search(request, kb_id):
    data = parse_body(request) if request.method == "POST" else request.GET
    return ok(kb_hybrid_search_service(service_context(request), kb_id, data))
