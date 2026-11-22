from django.views.decorators.csrf import csrf_exempt

from personal_knowledge_base.authentication import tenant_scoped
from personal_knowledge_base.http_helpers import auth_context, parse_body
from personal_knowledge_base.responses import ok, service_errors
from services.context import ServiceContext
from services.wiki import (
    wiki_folders as wiki_folders_service,
    wiki_index as wiki_index_service,
    wiki_issues as wiki_issues_service,
    wiki_lint as wiki_lint_service,
    wiki_log as wiki_log_service,
    wiki_pages as wiki_pages_service,
    wiki_search as wiki_search_service,
    wiki_stats as wiki_stats_service,
    wiki_graph_subgraph,
)


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
def wiki_pages(request, kb_id, slug=None):
    data = parse_body(request) if request.method not in {"GET", "DELETE"} else {}
    result = wiki_pages_service(_context(request), kb_id, method=request.method, slug=slug, data=data)
    return _ok(result)


@csrf_exempt
@tenant_scoped
@service_errors
def wiki_folders(request, kb_id, folder_id=None):
    data = parse_body(request) if request.method not in {"GET", "DELETE"} else {}
    result = wiki_folders_service(_context(request), kb_id, method=request.method, folder_id=folder_id, data=data)
    return _ok(result)


@tenant_scoped
@service_errors
def wiki_index(request, kb_id):
    return ok(wiki_index_service(_context(request), kb_id))


@tenant_scoped
@service_errors
def wiki_search(request, kb_id):
    return ok(wiki_search_service(_context(request), kb_id, request.GET))


@tenant_scoped
@service_errors
def wiki_graph(request, kb_id):
    return ok(wiki_graph_subgraph(_context(request), kb_id, request.GET))


@tenant_scoped
@service_errors
def wiki_stats(request, kb_id):
    return ok(wiki_stats_service(_context(request), kb_id))


@tenant_scoped
@service_errors
def wiki_log(request, kb_id):
    return ok(wiki_log_service(_context(request), kb_id))


@tenant_scoped
@service_errors
def wiki_lint(request, kb_id):
    return ok(wiki_lint_service(_context(request), kb_id))


@tenant_scoped
@service_errors
def wiki_issues(request, kb_id, issue_id=None):
    return ok(wiki_issues_service(_context(request), kb_id, issue_id))
