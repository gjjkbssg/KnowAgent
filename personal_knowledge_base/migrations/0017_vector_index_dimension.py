import re

from django.conf import settings
from django.db import migrations


VECTOR_INDEX_DIM_RE = re.compile(r"\bembedding\s+float\s*\[\s*(\d+)\s*\]", re.I)


def resize_empty_vector_index(apps, schema_editor):
    connection = schema_editor.connection
    if connection.vendor != "sqlite":
        return
    with connection.cursor() as cursor:
        cursor.execute("SELECT sql FROM sqlite_master WHERE type = 'table' AND name = %s", ["chunk_embeddings_vec"])
        row = cursor.fetchone()
        if not row or not row[0]:
            return
        match = VECTOR_INDEX_DIM_RE.search(row[0])
        if not match:
            return
        current_dim = int(match.group(1))
        target_dim = int(settings.APP_EMBEDDING_DIM)
        if current_dim == target_dim:
            return
        cursor.execute("SELECT COUNT(*) FROM chunk_embeddings_vec")
        if cursor.fetchone()[0]:
            return
        cursor.execute("DROP TABLE chunk_embeddings_vec")
        cursor.execute(f"CREATE VIRTUAL TABLE chunk_embeddings_vec USING vec0(embedding float[{target_dim}])")


class Migration(migrations.Migration):
    dependencies = [
        ("personal_knowledge_base", "0016_renormalize_vectors"),
    ]

    operations = [
        migrations.RunPython(resize_empty_vector_index, reverse_code=migrations.RunPython.noop),
    ]
