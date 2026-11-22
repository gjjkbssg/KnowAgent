import uuid

from django.utils import timezone

from personal_knowledge_base.model_providers import bailian_status, env_models, provider_types
from personal_knowledge_base.model_types import canonical_model_type, frontend_model_group, is_removed_model_type, model_type_aliases
from personal_knowledge_base.model_usage import model_usage_summary
from personal_knowledge_base.models import ModelConfig
from personal_knowledge_base.serializers import model_dict
from services.context import ServiceContext
from services.errors import InvalidInput, get_or_404


def list_models(ctx: ServiceContext, model_type: str = "") -> dict:
    qs = ModelConfig.objects.filter(tenant=ctx.tenant, deleted_at__isnull=True)
    if model_type:
        qs = qs.filter(type__in=model_type_aliases(model_type))
    items = [model_dict(model) for model in qs]
    items = env_models(ctx.tenant, model_type or "") + items
    counts_by_type = {}
    for item in items:
        group = frontend_model_group(item.get("type") or item.get("raw_type"))
        counts_by_type[group] = counts_by_type.get(group, 0) + 1
    return {
        "items": items,
        "models": items,
        "total": len(items),
        "counts_by_type": counts_by_type,
        "bailian": bailian_status(),
    }


def get_model(ctx: ServiceContext, model_id) -> dict:
    return model_dict(get_or_404(ModelConfig, id=model_id, tenant=ctx.tenant))


def delete_model(ctx: ServiceContext, model_id) -> dict:
    model = get_or_404(ModelConfig, id=model_id, tenant=ctx.tenant)
    model.deleted_at = timezone.now()
    model.save(update_fields=["deleted_at", "updated_at"])
    return {}


def create_model(ctx: ServiceContext, data: dict) -> dict:
    _validate_model_type(data)
    model = ModelConfig(id=data.get("id") or f"{data.get('type', 'chat')}-{uuid.uuid4().hex[:8]}", tenant=ctx.tenant)
    _update_model(model, data)
    model.save()
    return model_dict(model)


def update_model(ctx: ServiceContext, model_id, data: dict) -> dict:
    model = get_or_404(ModelConfig, id=model_id, tenant=ctx.tenant)
    _validate_model_type(data)
    _update_model(model, data)
    return model_dict(model)


def _validate_model_type(data: dict) -> None:
    if is_removed_model_type(data.get("type")):
        raise InvalidInput("ASR model type is no longer supported", code="unsupported_model_type")


def _update_model(model: ModelConfig, data: dict) -> None:
    model.name = data.get("name", model.name or "model")
    model.display_name = data.get("display_name", data.get("displayName", model.display_name))
    model.type = canonical_model_type(data.get("type", model.type or "KnowledgeQA"))
    model.source = data.get("source", model.source or "openai")
    model.description = data.get("description", model.description)
    model.parameters = data.get("parameters", model.parameters or {})
    model.is_default = data.get("is_default", model.is_default)
    model.is_builtin = data.get("is_builtin", model.is_builtin)
    model.managed_by = data.get("managed_by", model.managed_by)
    model.status = data.get("status", model.status or "active")


def model_providers() -> dict:
    items = provider_types()
    return {"items": items, "providers": items}


def model_usage(ctx: ServiceContext, params) -> dict:
    return model_usage_summary(ctx.tenant, params)


def update_model_credentials(tenant, model_id, data: dict) -> dict:
    model = get_or_404(ModelConfig, id=model_id, tenant=tenant)
    params = model.parameters or {}
    params.update(data.get("credentials") or data)
    model.parameters = params
    model.save(update_fields=["parameters", "updated_at"])
    return model_dict(model)


def delete_model_credential(tenant, model_id, field) -> dict:
    model = get_or_404(ModelConfig, id=model_id, tenant=tenant)
    params = model.parameters or {}
    params.pop(field, None)
    model.parameters = params
    model.save(update_fields=["parameters", "updated_at"])
    return model_dict(model)
