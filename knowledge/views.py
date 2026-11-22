from django.core.files.storage import default_storage
from django.http import FileResponse, HttpResponse
from django.utils.http import content_disposition_header
from django.views.decorators.csrf import csrf_exempt

from personal_knowledge_base.authentication import tenant_scoped
from personal_knowledge_base.http_helpers import auth_context, parse_body
from personal_knowledge_base.responses import ok, service_errors
from services.chunks import chunks_collection as chunks_service
from services.context import ServiceContext
from services.knowledge import (
    knowledge_batch as knowledge_batch_service,
    knowledge_batch_delete as knowledge_batch_delete_service,
    knowledge_cancel as knowledge_cancel_service,
    knowledge_collection as knowledge_collection_service,
    knowledge_detail as knowledge_detail_service,
    knowledge_download as knowledge_download_service,
    knowledge_file as knowledge_file_service,
    kb_hybrid_search as kb_hybrid_search_service,
    knowledge_image_content as knowledge_image_content_service,
    knowledge_move as knowledge_move_service,
    knowledge_preview as knowledge_preview_service,
    knowledge_reparse as knowledge_reparse_service,
    knowledge_search as knowledge_search_service,
    knowledge_search_post as knowledge_search_post_service,
    knowledge_spans as knowledge_spans_service,
    knowledge_stats as knowledge_stats_service,
)
from services.knowledge_bases import (
    kb_copy as kb_copy_service,
    kb_move_targets as kb_move_targets_service,
    kb_pin as kb_pin_service,
    knowledge_bases as knowledge_bases_service,
)
from services.tags import knowledge_tags as knowledge_tags_service


def _context(request):
    user, tenant = auth_context(request)
    return ServiceContext(
        tenant=tenant or request.auth_tenant,
        user=user or getattr(request, "auth_user", None),
        request_id=request.headers.get("X-Request-ID", ""),
    )


def _ok(result):
    if "data" in result:
        return ok(result["data"], status=result.get("status", 200))
    return ok(result)


@csrf_exempt
@tenant_scoped
@service_errors
def knowledge_bases(request, kb_id=None):
    params = request.GET
    data = parse_body(request) if request.method not in {"GET", "DELETE"} else {}
    return _ok(knowledge_bases_service(_context(request), method=request.method, kb_id=kb_id, data=data, params=params))


@csrf_exempt
@tenant_scoped
@service_errors
def kb_pin(request, kb_id):
    data = parse_body(request) if request.method in {"POST", "PUT"} else {}
    return ok(kb_pin_service(_context(request), kb_id, method=request.method, data=data))


@csrf_exempt
@tenant_scoped
@service_errors
def kb_copy(request):
    return ok(kb_copy_service(_context(request), parse_body(request)))


@tenant_scoped
@service_errors
def kb_move_targets(request, kb_id):
    return ok(kb_move_targets_service(_context(request), kb_id))


@csrf_exempt
@tenant_scoped
@service_errors
def knowledge_collection(request, kb_id):
    return ok(knowledge_collection_service(_context(request), kb_id, method=request.method, params=request.GET))


@csrf_exempt
@tenant_scoped
@service_errors
def knowledge_file(request, kb_id):
    result = knowledge_file_service(_context(request), kb_id, request.FILES.get("file"), form_data=request.POST)
    return _ok(result)


@csrf_exempt
@tenant_scoped
@service_errors
def knowledge_detail(request, knowledge_id):
    data = parse_body(request) if request.method not in {"GET", "DELETE"} else {}
    return ok(knowledge_detail_service(_context(request), knowledge_id, method=request.method, data=data))


@csrf_exempt
@tenant_scoped
@service_errors
def knowledge_reparse(request, knowledge_id):
    return ok(knowledge_reparse_service(_context(request), knowledge_id, parse_body(request)))


@csrf_exempt
@tenant_scoped
@service_errors
def knowledge_cancel(request, knowledge_id):
    return ok(knowledge_cancel_service(_context(request), knowledge_id))


@csrf_exempt
@tenant_scoped
@service_errors
def knowledge_batch_delete(request, kb_id=None):
    return ok(knowledge_batch_delete_service(_context(request), parse_body(request), kb_id=kb_id))


@csrf_exempt
@tenant_scoped
@service_errors
def knowledge_move(request, kb_id=None):
    return ok(knowledge_move_service(_context(request), parse_body(request), kb_id=kb_id))


@tenant_scoped
@service_errors
def knowledge_batch(request):
    return ok(knowledge_batch_service(_context(request), request.GET))


@tenant_scoped
@service_errors
def knowledge_search(request):
    return ok(knowledge_search_service(_context(request), request.GET))


@csrf_exempt
@tenant_scoped
@service_errors
def knowledge_search_post(request):
    return ok(knowledge_search_post_service(_context(request), parse_body(request)))


@tenant_scoped
@service_errors
def knowledge_stats(request, kb_id):
    return ok(knowledge_stats_service(_context(request), kb_id))


@tenant_scoped
@service_errors
def knowledge_spans(request, knowledge_id):
    return ok(knowledge_spans_service(_context(request), knowledge_id))


@tenant_scoped
@service_errors
def knowledge_download(request, knowledge_id):
    result = knowledge_download_service(_context(request), knowledge_id)
    if "path" in result:
        return FileResponse(default_storage.open(result["path"], "rb"), as_attachment=True, filename=result["filename"])
    response = HttpResponse(result["content"], content_type="text/plain; charset=utf-8")
    response["Content-Disposition"] = content_disposition_header(True, result["filename"])
    return response


@tenant_scoped
@service_errors
def knowledge_preview(request, knowledge_id):
    result = knowledge_preview_service(_context(request), knowledge_id)
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
    result = knowledge_image_content_service(_context(request), knowledge_id, image_id)
    return FileResponse(default_storage.open(result["path"], "rb"), content_type=result["content_type"], as_attachment=False)


@csrf_exempt
@tenant_scoped
@service_errors
def chunks_collection(request, knowledge_id=None, chunk_id=None):
    data = parse_body(request) if request.method not in {"GET", "DELETE"} else {}
    result = chunks_service(
        _context(request),
        method=request.method,
        knowledge_id=knowledge_id,
        chunk_id=chunk_id,
        data=data,
        params=request.GET,
    )
    return ok(result)


@csrf_exempt
@tenant_scoped
@service_errors
def knowledge_tags(request, kb_id, tag_id=None):
    data = parse_body(request) if request.method not in {"GET", "DELETE"} else {}
    result = knowledge_tags_service(_context(request), kb_id, method=request.method, tag_id=tag_id, data=data)
    return _ok(result)


@csrf_exempt
@tenant_scoped
@service_errors
def kb_hybrid_search(request, kb_id):
    data = parse_body(request) if request.method == "POST" else request.GET
    return ok(kb_hybrid_search_service(_context(request), kb_id, data))
