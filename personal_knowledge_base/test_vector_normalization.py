import importlib
import math
import struct
from unittest.mock import patch

import sqlite_vec
from django.db import connection
from django.test import SimpleTestCase, TestCase

from personal_knowledge_base.models import Chunk, Knowledge, KnowledgeBase, Tenant
from personal_knowledge_base.search import _vector_search, index_chunk, pack_embedding

normalize_vector_table = importlib.import_module(
    "personal_knowledge_base.migrations.0016_renormalize_vectors"
).normalize_vector_table


class VectorNormalizationMigrationTests(TestCase):
    def test_migration_normalizes_only_non_unit_vectors_and_is_idempotent(self):
        unnormalized = [2.0] + [0.0] * 383
        second_unnormalized = [0.0, 3.0] + [0.0] * 382
        already_normalized = [0.6, 0.8] + [0.0] * 382
        zero_vector = [0.0] * 384
        untouched_blobs = {
            940002: sqlite_vec.serialize_float32(already_normalized),
            940003: sqlite_vec.serialize_float32(zero_vector),
        }
        with connection.cursor() as cursor:
            cursor.execute("SELECT vec_version()")
            cursor.fetchone()
            cursor.execute(
                "INSERT INTO chunk_embeddings_vec(rowid, embedding) VALUES (%s, %s)",
                [940000, sqlite_vec.serialize_float32(unnormalized)],
            )
            cursor.execute(
                "INSERT INTO chunk_embeddings_vec(rowid, embedding) VALUES (%s, %s)",
                [940001, sqlite_vec.serialize_float32(second_unnormalized)],
            )
            for rowid, embedding in untouched_blobs.items():
                cursor.execute(
                    "INSERT INTO chunk_embeddings_vec(rowid, embedding) VALUES (%s, %s)",
                    [rowid, embedding],
                )

        self.assertEqual(normalize_vector_table(connection), 2)

        with connection.cursor() as cursor:
            cursor.execute("SELECT rowid, embedding FROM chunk_embeddings_vec WHERE rowid BETWEEN %s AND %s", [940000, 940003])
            vectors = {rowid: bytes(embedding) for rowid, embedding in cursor.fetchall()}
        for rowid in (940000, 940001):
            normalized = struct.unpack("<384f", vectors[rowid])
            self.assertAlmostEqual(math.sqrt(sum(value * value for value in normalized)), 1.0, places=6)
        self.assertEqual(vectors[940002], untouched_blobs[940002])
        self.assertEqual(vectors[940003], untouched_blobs[940003])
        self.assertEqual(normalize_vector_table(connection), 0)

    def test_missing_vector_table_is_a_noop(self):
        with patch.object(connection.introspection, "table_names", return_value=[]):
            self.assertEqual(normalize_vector_table(connection), 0)


class VectorNormalizationRankingTests(TestCase):
    def setUp(self):
        self.tenant = Tenant.objects.create(name="vector normalization")
        self.kb = KnowledgeBase.objects.create(tenant=self.tenant, name="kb")
        self.query_vector = [1.0, 0.0] + [0.0] * 382

    def add_chunk_with_vector(self, title, vector):
        knowledge = Knowledge.objects.create(tenant=self.tenant, knowledge_base=self.kb, title=title, parse_status="completed")
        chunk = Chunk.objects.create(
            tenant=self.tenant,
            knowledge_base=self.kb,
            knowledge=knowledge,
            content=title,
            chunk_index=0,
        )
        index_chunk(chunk, vec=pack_embedding(vector))
        return chunk

    def test_migration_corrects_l2_ranking_for_truncated_prefixes(self):
        closer_direction = self.add_chunk_with_vector("closer direction", [0.2, 0.1] + [0.0] * 382)
        farther_direction = self.add_chunk_with_vector("farther direction", [0.8, 0.6] + [0.0] * 382)

        with patch("personal_knowledge_base.model_providers.embedding", return_value=[self.query_vector]):
            before = _vector_search(self.tenant.id, {self.kb.id}, "query", 2)
            self.assertGreater(before[farther_direction.id], before[closer_direction.id])

            self.assertEqual(normalize_vector_table(connection), 1)

            after = _vector_search(self.tenant.id, {self.kb.id}, "query", 2)

        self.assertGreater(after[closer_direction.id], after[farther_direction.id])
