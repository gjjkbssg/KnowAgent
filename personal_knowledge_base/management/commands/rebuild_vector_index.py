from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.db import connection, transaction

from personal_knowledge_base.models import Chunk
from personal_knowledge_base.search import chunk_vector, ensure_search_tables, index_chunk

MAX_PASSES = 5
ID_BATCH = 500


class ChunksChanged(Exception):
    pass


class Command(BaseCommand):
    help = "Rebuild sqlite-vec and FTS5 chunk indexes from stored chunks."

    def handle(self, *args, **options):
        enabled = Chunk.objects.filter(is_enabled=True)
        vectors: dict[str, tuple[Chunk, bytes]] = {}

        def embed(chunks):
            for chunk in chunks.select_related("knowledge", "knowledge_base", "tenant").iterator():
                vectors[chunk.id] = (chunk, chunk_vector(chunk, dim=settings.APP_EMBEDDING_DIM))

        # 向量全部在事务外算好再动现有索引：Embedding 服务中途失败时，旧的全文和向量索引原样保留
        embed(enabled)
        for _ in range(MAX_PASSES):
            try:
                with transaction.atomic():
                    # 第一条语句就是写操作，拿到写锁之后读到的 Chunk 不会再被别的连接改动
                    with connection.cursor() as cursor:
                        cursor.execute("DROP TABLE IF EXISTS chunk_embeddings_vec")
                        cursor.execute("DROP TABLE IF EXISTS chunks_fts")
                    ensure_search_tables()
                    current = dict(enabled.values_list("id", "updated_at"))
                    # 算向量期间新增或改过的 Chunk：回滚，补算后重来，否则它们会从索引里消失
                    stale = [
                        chunk_id
                        for chunk_id, updated_at in current.items()
                        if chunk_id not in vectors or vectors[chunk_id][0].updated_at != updated_at
                    ]
                    if stale:
                        raise ChunksChanged
                    for chunk_id in current:
                        index_chunk(*vectors[chunk_id])
                break
            except ChunksChanged:
                for start in range(0, len(stale), ID_BATCH):
                    embed(enabled.filter(id__in=stale[start : start + ID_BATCH]))
        else:
            raise CommandError("Chunks kept changing during the rebuild; pause document processing and run it again.")
        self.stdout.write(self.style.SUCCESS(f"Rebuilt search indexes for {len(current)} chunks."))
