from django.conf import settings
from django.core.checks import Warning, register
from django.db import DatabaseError

from .search import vector_index_dim


@register()
def check_vector_index_dimension(app_configs, **kwargs):
    try:
        index_dim = vector_index_dim()
    except DatabaseError:
        return []
    if index_dim is None or index_dim == settings.APP_EMBEDDING_DIM:
        return []
    return [
        Warning(
            f"chunk_embeddings_vec uses {index_dim} dimensions, but APP_EMBEDDING_DIM is {settings.APP_EMBEDDING_DIM}.",
            hint="Run `python manage.py rebuild_vector_index` to rebuild the vector index.",
            id="personal_knowledge_base.W001",
        )
    ]
