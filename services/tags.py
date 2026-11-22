from personal_knowledge_base.models import KnowledgeBase, KnowledgeTag
from personal_knowledge_base.serializers import tag_dict

from .context import ServiceContext
from .errors import get_or_404


def list_knowledge_tags(ctx: ServiceContext, kb_id):
    kb = get_or_404(KnowledgeBase, id=kb_id)
    tags = KnowledgeTag.objects.filter(knowledge_base=kb).order_by("sort_order", "created_at")
    return {"items": [tag_dict(tag) for tag in tags]}


def create_knowledge_tag(ctx: ServiceContext, kb_id, data):
    kb = get_or_404(KnowledgeBase, id=kb_id)
    tag = KnowledgeTag.objects.create(
        tenant=ctx.tenant,
        knowledge_base=kb,
        name=data.get("name", "未命名"),
        color=data.get("color", ""),
        sort_order=data.get("sort_order", 0),
    )
    return tag_dict(tag)


def update_knowledge_tag(ctx: ServiceContext, kb_id, tag_id, data):
    kb = get_or_404(KnowledgeBase, id=kb_id)
    tag = get_or_404(KnowledgeTag, id=tag_id, knowledge_base=kb)
    tag.name = data.get("name", tag.name)
    tag.color = data.get("color", tag.color)
    tag.sort_order = data.get("sort_order", tag.sort_order)
    tag.save()
    return tag_dict(tag)


def delete_knowledge_tag(ctx: ServiceContext, kb_id, tag_id):
    kb = get_or_404(KnowledgeBase, id=kb_id)
    get_or_404(KnowledgeTag, id=tag_id, knowledge_base=kb).delete()
    return {}
