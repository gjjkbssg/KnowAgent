import secrets

from django.conf import settings
from django.db.models import Q
from django.utils import timezone

from personal_knowledge_base.authentication import hash_password, issue_tokens, verify_password
from personal_knowledge_base.models import AuthToken, Tenant, TenantMember, User
from personal_knowledge_base.serializers import membership_dict, tenant_dict, user_dict
from services.errors import Conflict, Forbidden, InvalidInput, NotFound, Unauthorized


def register(data: dict) -> dict:
    username = data.get("username") or data.get("email", "").split("@")[0] or f"user-{secrets.token_hex(3)}"
    email = data.get("email") or f"{username}@local"
    password = data.get("password")
    if not password:
        raise InvalidInput("password is required")
    if User.objects.filter(Q(username=username) | Q(email=email)).exists():
        raise Conflict("user already exists", code="user_exists")
    tenant = Tenant.objects.create(name=f"{username} 的空间", api_key=secrets.token_urlsafe(24), business="default")
    user = User.objects.create(username=username, email=email, password_hash=hash_password(password), tenant=tenant)
    TenantMember.objects.create(user=user, tenant=tenant, role="owner")
    token, refresh = issue_tokens(user)
    return {"user": user_dict(user), "tenant": tenant_dict(tenant), "token": token, "refresh_token": refresh}


def auto_setup() -> dict:
    user = User.objects.order_by("created_at").first()
    if not user:
        random_password = secrets.token_urlsafe(12)
        data = register(
            {
                "username": "admin",
                "email": "admin@knowledge.local",
                "password": random_password,
            }
        )
        data["temp_password"] = random_password
        data["require_password_change"] = True
        return data
    if not settings.AUTO_LOGIN_ENABLED:
        raise Forbidden(
            "passwordless login is disabled; sign in with your password, or run `python manage.py reset_password` on the server if it is lost",
            code="auto_login_disabled",
        )
    token, refresh = issue_tokens(user)
    return {
        "user": user_dict(user),
        "tenant": tenant_dict(user.tenant),
        "token": token,
        "refresh_token": refresh,
    }


def login(data: dict) -> dict:
    login_value = data.get("email") or data.get("username") or ""
    user = User.objects.filter(Q(email=login_value) | Q(username=login_value), deleted_at__isnull=True).first()
    if not user or not verify_password(data.get("password", ""), user.password_hash):
        raise Unauthorized("invalid credentials", code="invalid_credentials")
    token, refresh = issue_tokens(user)
    tenant = user.tenant
    memberships = [membership_dict(m) for m in TenantMember.objects.filter(user=user, status="active")]
    return {
        "user": user_dict(user),
        "tenant": tenant_dict(tenant),
        "token": token,
        "refresh_token": refresh,
        "memberships": memberships,
    }


def refresh(data: dict) -> dict:
    token = data.get("refresh_token") or data.get("refreshToken") or data.get("token")
    auth = (
        AuthToken.objects.filter(
            token=token,
            token_type="refresh",
            is_revoked=False,
            expires_at__gt=timezone.now(),
        )
        .select_related("user")
        .first()
    )
    if not auth:
        raise Unauthorized("invalid refresh token", code="invalid_refresh")
    access, refresh_token = issue_tokens(auth.user)
    auth.is_revoked = True
    auth.save(update_fields=["is_revoked", "updated_at"])
    return {
        "token": access,
        "refreshToken": refresh_token,
        "refresh_token": refresh_token,
        "user": user_dict(auth.user),
        "tenant": tenant_dict(auth.user.tenant),
    }


def logout(token: str) -> dict:
    if token:
        AuthToken.objects.filter(token=token).update(is_revoked=True)
    return {}


def current_user(user, tenant) -> dict:
    if not user:
        raise Unauthorized("unauthorized", code="unauthorized")
    memberships = [membership_dict(m) for m in TenantMember.objects.filter(user=user, status="active")]
    return {
        "user": user_dict(user),
        "tenant": tenant_dict(tenant or user.tenant),
        "memberships": memberships,
    }


def validate_auth(user, tenant) -> dict:
    return {"valid": bool(user or tenant), "user": user_dict(user), "tenant": tenant_dict(tenant)}


def update_preferences(user, data: dict) -> dict:
    if not user:
        raise Unauthorized("unauthorized")
    prefs = user.preferences or {}
    prefs.update(data)
    user.preferences = prefs
    user.save(update_fields=["preferences", "updated_at"])
    return {"user": user_dict(user)}


def change_password(user, data: dict) -> dict:
    if not user:
        raise Unauthorized("unauthorized")
    if not verify_password(data.get("old_password", data.get("oldPassword", "")), user.password_hash):
        raise InvalidInput("old password mismatch")
    user.password_hash = hash_password(data.get("new_password", data.get("newPassword", "")))
    user.save(update_fields=["password_hash", "updated_at"])
    return {}


def switch_tenant(user, tenant_id) -> dict:
    if not user:
        raise Unauthorized("unauthorized")
    tenant = Tenant.objects.filter(id=tenant_id).first()
    if not tenant:
        raise NotFound("tenant not found")
    return {"tenant": tenant_dict(tenant), "user": user_dict(user)}
