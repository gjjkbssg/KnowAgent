import json

from django.views.decorators.csrf import csrf_exempt

from personal_knowledge_base.http_helpers import auth_context, parse_body
from personal_knowledge_base.responses import fail, ok, service_errors
from services.context import ServiceContext
from services.accounts import (
    auto_setup,
    change_password,
    current_user,
    login,
    logout,
    refresh,
    register,
    switch_tenant as switch_tenant_service,
    update_preferences,
    validate_auth,
)
from services.tenants import (
    add_tenant_member,
    create_tenant,
    delete_tenant,
    delete_tenant_member,
    get_tenant_detail,
    get_tenant_kv,
    list_audit_logs,
    list_tenant_members,
    list_tenants,
    rotate_tenant_api_key,
    update_tenant,
    update_tenant_kv,
    update_tenant_member,
)


def _tenant_admin_context(request):
    user, tenant = auth_context(request)
    if user is None and tenant is None:
        return None
    return ServiceContext(
        tenant=tenant or (user.tenant if user else None),
        user=user,
        request_id=request.headers.get("X-Request-ID", ""),
    )


# ---------------------------------------------------------------------------
# Auth views
# ---------------------------------------------------------------------------

@csrf_exempt
@service_errors
def auth_register(request):
    return ok(register(parse_body(request)), status=201)


@csrf_exempt
@service_errors
def auth_auto_setup(request):
    data = auto_setup()
    response = ok(data, status=201 if data.get("require_password_change") else 200)
    if data.get("require_password_change"):
        response.content = json.dumps(
            {"success": True, "message": "success", "data": data}
        ).encode()
    return response


@csrf_exempt
@service_errors
def auth_login(request):
    return ok(login(parse_body(request)))


@csrf_exempt
@service_errors
def auth_refresh(request):
    return ok(refresh(parse_body(request)))


@csrf_exempt
@service_errors
def auth_logout(request):
    header = request.headers.get("Authorization", "")
    token = header.removeprefix("Bearer ").strip() if header.startswith("Bearer ") else ""
    return ok(logout(token))


@csrf_exempt
@service_errors
def auth_me(request):
    user, tenant = auth_context(request)
    return ok(current_user(user, tenant))


@csrf_exempt
@service_errors
def auth_validate(request):
    user, tenant = auth_context(request)
    return ok(validate_auth(user, tenant))


@csrf_exempt
@service_errors
def auth_preferences(request):
    user, _ = auth_context(request)
    return ok(update_preferences(user, parse_body(request)))


@csrf_exempt
@service_errors
def auth_change_password(request):
    user, _ = auth_context(request)
    return ok(change_password(user, parse_body(request)))


def auth_config(request):
    return ok({"registration_mode": "self_serve", "oidc_enabled": False})


def oidc_config(request):
    return ok({"enabled": False, "provider_display_name": ""})


def oidc_url(request):
    return ok({"authorization_url": "", "state": ""})


def oidc_callback(request):
    return ok({"success": False, "message": "OIDC is not configured"})


# ---------------------------------------------------------------------------
# Tenant views
# ---------------------------------------------------------------------------

@csrf_exempt
@service_errors
def switch_tenant(request):
    user, _ = auth_context(request)
    tenant_id = parse_body(request).get("tenant_id")
    return ok(switch_tenant_service(user, tenant_id))


@csrf_exempt
@service_errors
def tenants_collection(request):
    user, tenant = auth_context(request)
    if not user and not tenant:
        return fail("unauthorized", 401)
    if request.method == "GET":
        return ok(list_tenants(user, tenant))
    return ok(create_tenant(user, parse_body(request)), status=201)


@csrf_exempt
@service_errors
def tenant_detail(request, tenant_id):
    ctx = _tenant_admin_context(request)
    if ctx is None:
        return fail("unauthorized", 401)
    if request.method == "GET":
        return ok(get_tenant_detail(ctx, tenant_id))
    if request.method == "DELETE":
        return ok(delete_tenant(ctx, tenant_id))
    return ok(update_tenant(ctx, tenant_id, parse_body(request)))


@csrf_exempt
@service_errors
def tenant_members(request, tenant_id, user_id=None):
    ctx = _tenant_admin_context(request)
    if ctx is None:
        return fail("unauthorized", 401)
    if request.method == "GET":
        return ok(list_tenant_members(ctx, tenant_id))
    data = parse_body(request)
    if request.method == "POST":
        return ok(add_tenant_member(ctx, tenant_id, data))
    if request.method == "DELETE":
        return ok(delete_tenant_member(ctx, tenant_id, user_id))
    return ok(update_tenant_member(ctx, tenant_id, user_id, data))


@csrf_exempt
@service_errors
def tenant_api_key(request, tenant_id):
    ctx = _tenant_admin_context(request)
    if ctx is None:
        return fail("unauthorized", 401)
    return ok(rotate_tenant_api_key(ctx, tenant_id))


@csrf_exempt
@service_errors
def tenant_kv(request, key):
    _, tenant = auth_context(request)
    if not tenant:
        return fail("unauthorized", 401)
    if request.method == "GET":
        return ok(get_tenant_kv(tenant, key))
    return ok(update_tenant_kv(tenant, key, parse_body(request)))


# ---------------------------------------------------------------------------
# Audit views
# ---------------------------------------------------------------------------

@service_errors
def audit_logs(request, tenant_id=None):
    _, tenant = auth_context(request)
    return ok(list_audit_logs(tenant, tenant_id))
