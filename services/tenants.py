import secrets

from django.db.models import Q
from django.utils import timezone

from personal_knowledge_base.models import AuditLog, Tenant, TenantMember, User
from personal_knowledge_base.serializers import membership_dict, tenant_dict, user_dict
from services.errors import Forbidden, NotFound, Unauthorized, get_or_404


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


def list_tenants(user, tenant) -> dict:
    if user and user.can_access_all_tenants:
        qs = Tenant.objects.filter(deleted_at__isnull=True)
    elif user:
        ids = TenantMember.objects.filter(user=user, status="active").values_list("tenant_id", flat=True)
        qs = Tenant.objects.filter(id__in=ids, deleted_at__isnull=True)
    else:
        qs = Tenant.objects.filter(id=tenant.id)
    items = [tenant_dict(t) for t in qs]
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
    return tenant_dict(tenant)


def get_tenant(tenant_id):
    return get_or_404(Tenant, id=tenant_id)


def get_tenant_detail(tenant_id) -> dict:
    return tenant_dict(get_tenant(tenant_id))


def update_tenant(tenant_id, data: dict) -> dict:
    target = get_tenant(tenant_id)
    for field in ["name", "description", "business", "status"]:
        if field in data:
            setattr(target, field, data[field])
    target.save()
    return tenant_dict(target)


def delete_tenant(tenant_id) -> dict:
    target = get_tenant(tenant_id)
    target.deleted_at = timezone.now()
    target.save(update_fields=["deleted_at", "updated_at"])
    return {}


def list_tenant_members(tenant_id) -> dict:
    tenant = get_tenant(tenant_id)
    members = TenantMember.objects.filter(tenant=tenant, status="active").select_related("user")
    return {"items": [{**membership_dict(m), "user": user_dict(m.user)} for m in members]}


def add_tenant_member(tenant_id, data: dict) -> dict:
    tenant = get_tenant(tenant_id)
    user = User.objects.filter(Q(email=data.get("email")) | Q(id=data.get("user_id"))).first()
    if not user:
        raise NotFound("user not found")
    member, _ = TenantMember.objects.update_or_create(
        user=user,
        tenant=tenant,
        defaults={"role": data.get("role", "viewer"), "status": "active"},
    )
    return membership_dict(member)


def get_tenant_member(tenant_id, user_id):
    tenant = get_tenant(tenant_id)
    return get_or_404(TenantMember, tenant=tenant, user_id=user_id)


def delete_tenant_member(tenant_id, user_id) -> dict:
    member = get_tenant_member(tenant_id, user_id)
    member.delete()
    return {}


def update_tenant_member(tenant_id, user_id, data: dict) -> dict:
    member = get_tenant_member(tenant_id, user_id)
    member.role = data.get("role", member.role)
    member.save(update_fields=["role", "updated_at"])
    return membership_dict(member)


def rotate_tenant_api_key(tenant_id) -> dict:
    tenant = get_tenant(tenant_id)
    tenant.api_key = secrets.token_urlsafe(24)
    tenant.save(update_fields=["api_key", "updated_at"])
    return tenant_dict(tenant)


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
