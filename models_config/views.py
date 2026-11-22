from django.views.decorators.csrf import csrf_exempt

from personal_knowledge_base.http_helpers import auth_context, parse_body
from personal_knowledge_base.responses import fail, ok, service_errors
from services.context import ServiceContext
from services.models_config import (
    create_model,
    delete_model,
    delete_model_credential,
    get_model,
    list_models,
    model_providers as model_providers_service,
    model_usage as model_usage_service,
    update_model as update_model_service,
    update_model_credentials,
)


def _context(request, tenant, user=None):
    return ServiceContext(
        tenant=tenant,
        user=user,
        request_id=request.headers.get("X-Request-ID", ""),
    )


@csrf_exempt
@service_errors
def models_collection(request, model_id=None):
    user, tenant = auth_context(request)
    if not tenant:
        return fail("unauthorized", 401)
    ctx = _context(request, tenant, user)
    if model_id:
        if request.method == "GET":
            return ok(get_model(ctx, model_id))
        if request.method == "DELETE":
            return ok(delete_model(ctx, model_id))
        return ok(update_model_service(ctx, model_id, parse_body(request)))
    if request.method == "GET":
        return ok(list_models(ctx, request.GET.get("type") or ""))
    return ok(create_model(ctx, parse_body(request)), status=201)


def model_providers(request):
    return ok(model_providers_service())


@service_errors
def model_usage(request):
    user, tenant = auth_context(request)
    if not tenant:
        return fail("unauthorized", 401)
    return ok(model_usage_service(_context(request, tenant, user), request.GET))


@csrf_exempt
@service_errors
def model_credentials(request, model_id, field=None):
    _, tenant = auth_context(request)
    if request.method == "DELETE":
        return ok(delete_model_credential(tenant, model_id, field))
    return ok(update_model_credentials(tenant, model_id, parse_body(request)))
