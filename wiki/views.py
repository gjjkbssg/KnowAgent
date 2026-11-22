from django.views.decorators.csrf import csrf_exempt

from personal_knowledge_base.authentication import tenant_scoped
from personal_knowledge_base.http_helpers import parse_body, service_context
from personal_knowledge_base.responses import fail, ok, service_errors
from services.wiki import (
    create_wiki_folder as create_wiki_folder_service,
    create_wiki_page as create_wiki_page_service,
    delete_wiki_folder as delete_wiki_folder_service,
    delete_wiki_page as delete_wiki_page_service,
    get_wiki_page as get_wiki_page_service,
    wiki_index as wiki_index_service,
    wiki_issues as wiki_issues_service,
    wiki_lint as wiki_lint_service,
    wiki_log as wiki_log_service,
    wiki_search as wiki_search_service,
    wiki_stats as wiki_stats_service,
    wiki_graph_subgraph,
    list_wiki_folders as list_wiki_folders_service,
    list_wiki_pages as list_wiki_pages_service,
    update_wiki_folder as update_wiki_folder_service,
    update_wiki_page as update_wiki_page_service,
)


@csrf_exempt
@tenant_scoped
@service_errors
def wiki_pages(request, kb_id, slug=None):
    ctx = service_context(request)
    if slug:
        if request.method == "GET":
            return ok(get_wiki_page_service(ctx, kb_id, slug))
        if request.method == "DELETE":
            return ok(delete_wiki_page_service(ctx, kb_id, slug))
        return ok(update_wiki_page_service(ctx, kb_id, slug, parse_body(request)))
    if request.method == "GET":
        return ok(list_wiki_pages_service(ctx, kb_id))
    return ok(create_wiki_page_service(ctx, kb_id, parse_body(request)), status=201)


@csrf_exempt
@tenant_scoped
@service_errors
def wiki_folders(request, kb_id, folder_id=None):
    ctx = service_context(request)
    if folder_id:
        if request.method == "DELETE":
            return ok(delete_wiki_folder_service(ctx, kb_id, folder_id))
        return ok(update_wiki_folder_service(ctx, kb_id, folder_id, parse_body(request)))
    if request.method == "GET":
        return ok(list_wiki_folders_service(ctx, kb_id))
    return ok(create_wiki_folder_service(ctx, kb_id, parse_body(request)), status=201)


@tenant_scoped
@service_errors
def wiki_index(request, kb_id):
    return ok(wiki_index_service(service_context(request), kb_id))


@tenant_scoped
@service_errors
def wiki_search(request, kb_id):
    return ok(wiki_search_service(service_context(request), kb_id, request.GET))


@tenant_scoped
@service_errors
def wiki_graph(request, kb_id):
    return ok(wiki_graph_subgraph(service_context(request), kb_id, request.GET))


@tenant_scoped
@service_errors
def wiki_stats(request, kb_id):
    return ok(wiki_stats_service(service_context(request), kb_id))


@tenant_scoped
@service_errors
def wiki_log(request, kb_id):
    return ok(wiki_log_service(service_context(request), kb_id))


@tenant_scoped
@service_errors
def wiki_lint(request, kb_id):
    return ok(wiki_lint_service(service_context(request), kb_id))


@tenant_scoped
@service_errors
def wiki_issues(request, kb_id, issue_id=None):
    return ok(wiki_issues_service(service_context(request), kb_id, issue_id))
