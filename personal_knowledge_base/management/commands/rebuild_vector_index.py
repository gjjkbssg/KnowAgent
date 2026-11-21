from django.core.management.base import BaseCommand
from django.db import connection, transaction

from personal_knowledge_base.models import Chunk
from personal_knowledge_base.search import chunk_vector, ensure_search_tables, index_chunk


class Command(BaseCommand):
    help = "Rebuild sqlite-vec and FTS5 chunk indexes from stored chunks."

    def handle(self, *args, **options):
        chunks = Chunk.objects.filter(is_enabled=True).select_related("knowledge", "knowledge_base", "tenant")
        # 先算完所有向量再动现有索引：Embedding 服务中途失败时，旧的全文和向量索引原样保留
        vectors = [(chunk, chunk_vector(chunk)) for chunk in chunks.iterator()]
        with transaction.atomic():
            with connection.cursor() as cursor:
                cursor.execute("DROP TABLE IF EXISTS chunk_embeddings_vec")
                cursor.execute("DROP TABLE IF EXISTS chunks_fts")
            ensure_search_tables()
            for chunk, vector in vectors:
                index_chunk(chunk, vector)
        self.stdout.write(self.style.SUCCESS(f"Rebuilt search indexes for {len(vectors)} chunks."))
