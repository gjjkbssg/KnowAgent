from personal_knowledge_base.models import KnowledgeBase, KnowledgeTag
from personal_knowledge_base.serializers import tag_dict

from .context import ServiceContext
from .errors import get_or_404


def knowledge_tags(ctx: ServiceContext, kb_id, *, method: str, tag_id=None, data=None):
    kb = get_or_404(KnowledgeBase, id=kb_id)
    data = data or {}
    if method == "GET":
        tags = KnowledgeTag.objects.filter(knowledge_base=kb).order_by("sort_order", "created_at")
        return {"items": [tag_dict(tag) for tag in tags]}
    if tag_id:
        tag = get_or_404(KnowledgeTag, id=tag_id, knowledge_base=kb)
        if method == "DELETE":
            tag.delete()
            return {}
        tag.name = data.get("name", tag.name)
        tag.color = data.get("color", tag.color)
        tag.sort_order = data.get("sort_order", tag.sort_order)
        tag.save()
        return tag_dict(tag)
    tag = KnowledgeTag.objects.create(
        tenant=ctx.tenant,
        knowledge_base=kb,
        name=data.get("name", "未命名"),
        color=data.get("color", ""),
        sort_order=data.get("sort_order", 0),
    )
    return {"data": tag_dict(tag), "status": 201}
