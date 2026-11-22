import secrets

from django.db import transaction
from django.db.models import Q
from django.utils import timezone

from personal_knowledge_base.authentication import role_for
from personal_knowledge_base.models import AuditLog, Tenant, TenantMember, User
from personal_knowledge_base.serializers import membership_dict, tenant_dict, user_dict
from services.context import ServiceContext
from services.errors import (
    Conflict,
    Forbidden,
    InvalidInput,
    NotFound,
    Unauthorized,
    get_or_404,
)


TENANT_KV_FIELDS = {
    "agent-config": "agent_config",
    "agent_config": "agent_config",
    "context-config": "context_config",
    "context_config": "context_config",
    "conversation-config": "conversation_config",
    "conversation_config": "conversation_config",
    "web-search-config": "web_search_config",
    "web_search_config": "web_search_config",
    "parser-engine-config": "parser_engine_config",
    "parser_engine_config": "parser_engine_config",
    "storage-engine-config": "storage_engine_config",
    "storage_engine_config": "storage_engine_config",
    "chat-history-config": "chat_history_config",
    "chat_history_config": "chat_history_config",
    "retrieval-config": "retrieval_config",
    "retrieval_config": "retrieval_config",
    "prompt-templates": "credentials",
}

TENANT_ROLES = {"owner", "admin", "viewer"}
TENANT_MANAGEMENT_ROLES = {"owner", "admin"}


def _authorized_tenant(ctx: ServiceContext, tenant_id):
    user = ctx.user
    if user is None:
        if ctx.tenant is None:
            raise Unauthorized("unauthorized")
        if ctx.tenant.id != tenant_id:
            raise NotFound("not found")
        tenant = ctx.tenant
    elif user.is_system_admin:
        tenant = get_or_404(Tenant, id=tenant_id)
    else:
        membership = TenantMember.objects.filter(
            user=user,
            tenant_id=tenant_id,
            status="active",
        ).first()
        if membership is None:
            raise NotFound("not found")
        tenant = get_or_404(Tenant, id=tenant_id)
    return tenant, "admin" if user is None else role_for(user, tenant)


def _require_roles(role: str, allowed: set[str]) -> None:
    if role not in allowed:
        raise Forbidden("forbidden")


def _validate_tenant_role(role: str) -> None:
    if not isinstance(role, str) or role not in TENANT_ROLES:
        raise InvalidInput("invalid role")


def _tenant_dict_for_user(user, tenant) -> dict:
    include_api_key = (
        user is not None
        and (user.is_system_admin or role_for(user, tenant) in TENANT_MANAGEMENT_ROLES)
    )
    return tenant_dict(tenant, include_api_key=include_api_key)


def list_tenants(user, tenant) -> dict:
    if user and user.can_access_all_tenants:
        qs = Tenant.objects.filter(deleted_at__isnull=True)
    elif user:
        ids = TenantMember.objects.filter(user=user, status="active").values_list("tenant_id", flat=True)
        qs = Tenant.objects.filter(id__in=ids, deleted_at__isnull=True)
    else:
        qs = Tenant.objects.filter(id=tenant.id)
    items = [_tenant_dict_for_user(user, t) for t in qs]
    return {"items": items, "tenants": items}


def create_tenant(user, data: dict) -> dict:
    tenant = Tenant.objects.create(
        name=data.get("name", "新空间"),
        description=data.get("description", ""),
        api_key=secrets.token_urlsafe(24),
        business=data.get("business", "default"),
    )
    if user:
        TenantMember.objects.create(user=user, tenant=tenant, role="owner")
    return tenant_dict(tenant, include_api_key=True)


def get_tenant_detail(ctx: ServiceContext, tenant_id) -> dict:
    tenant, role = _authorized_tenant(ctx, tenant_id)
    include_api_key = ctx.user is not None and role in TENANT_MANAGEMENT_ROLES
    return tenant_dict(tenant, include_api_key=include_api_key)


def update_tenant(ctx: ServiceContext, tenant_id, data: dict) -> dict:
    target, role = _authorized_tenant(ctx, tenant_id)
    _require_roles(role, TENANT_MANAGEMENT_ROLES)
    for field in ["name", "description", "business", "status"]:
        if field in data:
            setattr(target, field, data[field])
    target.save()
    include_api_key = ctx.user is not None and role in TENANT_MANAGEMENT_ROLES
    return tenant_dict(target, include_api_key=include_api_key)


def delete_tenant(ctx: ServiceContext, tenant_id) -> dict:
    target, role = _authorized_tenant(ctx, tenant_id)
    _require_roles(role, {"owner"})
    target.deleted_at = timezone.now()
    target.save(update_fields=["deleted_at", "updated_at"])
    return {}


def list_tenant_members(ctx: ServiceContext, tenant_id) -> dict:
    tenant, _ = _authorized_tenant(ctx, tenant_id)
    members = TenantMember.objects.filter(tenant=tenant, status="active").select_related("user")
    return {"items": [{**membership_dict(m), "user": user_dict(m.user)} for m in members]}


def add_tenant_member(ctx: ServiceContext, tenant_id, data: dict) -> dict:
    tenant, role = _authorized_tenant(ctx, tenant_id)
    _require_roles(role, TENANT_MANAGEMENT_ROLES)
    member_role = data.get("role", "viewer")
    _validate_tenant_role(member_role)
    if member_role == "owner" and role != "owner":
        raise Forbidden("forbidden")
    user = User.objects.filter(Q(email=data.get("email")) | Q(id=data.get("user_id"))).first()
    if not user:
        raise NotFound("user not found")
    with transaction.atomic():
        tenant = _lock_tenant(tenant)
        existing_member = TenantMember.objects.filter(user=user, tenant=tenant).first()
        if (
            existing_member is not None
            and existing_member.role == "owner"
            and role != "owner"
        ):
            raise Forbidden("forbidden")
        if existing_member is not None and member_role != existing_member.role:
            _ensure_active_owner_remains(tenant, existing_member)
        member, _ = TenantMember.objects.update_or_create(
            user=user,
            tenant=tenant,
            defaults={"role": member_role, "status": "active"},
        )
    return membership_dict(member)


def _get_tenant_member(tenant, user_id):
    return get_or_404(TenantMember, tenant=tenant, user_id=user_id)


def _lock_tenant(tenant):
    return get_or_404(Tenant.objects.select_for_update(), id=tenant.id)


def _ensure_active_owner_remains(tenant, member):
    if member.status != "active" or member.role != "owner":
        return
    owner_count = TenantMember.objects.filter(
        tenant=tenant,
        status="active",
        role="owner",
    ).count()
    if owner_count <= 1:
        raise Conflict("tenant must keep at least one owner")


def delete_tenant_member(ctx: ServiceContext, tenant_id, user_id) -> dict:
    tenant, role = _authorized_tenant(ctx, tenant_id)
    _require_roles(role, TENANT_MANAGEMENT_ROLES)
    with transaction.atomic():
        tenant = _lock_tenant(tenant)
        member = _get_tenant_member(tenant, user_id)
        if member.role == "owner" and role != "owner":
            raise Forbidden("forbidden")
        _ensure_active_owner_remains(tenant, member)
        member.delete()
    return {}


def update_tenant_member(ctx: ServiceContext, tenant_id, user_id, data: dict) -> dict:
    tenant, role = _authorized_tenant(ctx, tenant_id)
    _require_roles(role, TENANT_MANAGEMENT_ROLES)
    with transaction.atomic():
        tenant = _lock_tenant(tenant)
        member = _get_tenant_member(tenant, user_id)
        if member.role == "owner" and role != "owner":
            raise Forbidden("forbidden")
        member_role = data.get("role", member.role)
        _validate_tenant_role(member_role)
        if member_role == "owner" and role != "owner":
            raise Forbidden("forbidden")
        if member_role != member.role:
            _ensure_active_owner_remains(tenant, member)
        member.role = member_role
        member.save(update_fields=["role", "updated_at"])
    return membership_dict(member)


def rotate_tenant_api_key(ctx: ServiceContext, tenant_id) -> dict:
    tenant, role = _authorized_tenant(ctx, tenant_id)
    _require_roles(role, TENANT_MANAGEMENT_ROLES)
    tenant.api_key = secrets.token_urlsafe(24)
    tenant.save(update_fields=["api_key", "updated_at"])
    return tenant_dict(tenant, include_api_key=True)


def tenant_kv_field(key: str) -> str:
    return TENANT_KV_FIELDS.get(
        key,
        f"{key.replace('-', '_')}_config" if not key.endswith("_config") else key.replace("-", "_"),
    )


def get_tenant_kv(tenant, key: str) -> dict:
    field = tenant_kv_field(key)
    value = getattr(tenant, field, None) if hasattr(tenant, field) else None
    return {"key": key, "field": field, "value": value or {}, "configured": bool(value)}


def update_tenant_kv(tenant, key: str, data: dict) -> dict:
    field = tenant_kv_field(key)
    if hasattr(tenant, field):
        value = data.get("value", data)
        setattr(tenant, field, value)
        tenant.save(update_fields=[field, "updated_at"])
    return {
        "key": key,
        "field": field,
        "value": getattr(tenant, field, None) if hasattr(tenant, field) else data,
    }


def list_audit_logs(tenant, tenant_id=None) -> dict:
    if not tenant:
        raise Unauthorized("unauthorized")
    if tenant_id and int(tenant_id) != tenant.id:
        raise Forbidden("forbidden")
    qs = AuditLog.objects.filter(tenant_id=tenant.id).order_by("-created_at")
    return {
        "items": [
            {
                "id": item.id,
                "action": item.action,
                "outcome": item.outcome,
                "created_at": item.created_at.isoformat(),
                "details": item.details,
            }
            for item in qs[:100]
        ]
    }
