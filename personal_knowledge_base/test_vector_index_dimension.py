import importlib
from io import StringIO
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import sqlite_vec
from django.conf import settings
from django.core.management import call_command
from django.core.checks import Warning
from django.db import DatabaseError, connection
from django.test import SimpleTestCase, TransactionTestCase, override_settings

from personal_knowledge_base.checks import check_vector_index_dimension
from personal_knowledge_base.models import Chunk, Knowledge, KnowledgeBase, Tenant
from personal_knowledge_base.model_providers import embedding
from personal_knowledge_base.search import (
    _rowid,
    _vector_search,
    effective_embedding_dim,
    index_chunk,
    stable_embedding,
    vector_index_dim,
)

resize_empty_vector_index = importlib.import_module(
    "personal_knowledge_base.migrations.0017_vector_index_dimension"
).resize_empty_vector_index


def _set_vector_index_dim(dim):
    with connection.cursor() as cursor:
        cursor.execute("DROP TABLE IF EXISTS chunk_embeddings_vec")
        cursor.execute(f"CREATE VIRTUAL TABLE chunk_embeddings_vec USING vec0(embedding float[{dim}])")
        cursor.execute("DELETE FROM chunks_fts")


class VectorIndexDimensionTests(SimpleTestCase):
    def test_vector_index_dim_parses_supported_dimensions(self):
        for dim in (384, 1024):
            with self.subTest(dim=dim):
                cursor = MagicMock()
                cursor.__enter__.return_value = cursor
                cursor.fetchone.return_value = (
                    f"CREATE VIRTUAL TABLE chunk_embeddings_vec USING vec0(embedding float[{dim}])",
                )
                database = SimpleNamespace(vendor="sqlite", cursor=MagicMock(return_value=cursor))
                with patch("personal_knowledge_base.search.connection", database):
                    self.assertEqual(vector_index_dim(), dim)

    def test_vector_index_dim_returns_none_when_table_is_absent(self):
        cursor = MagicMock()
        cursor.__enter__.return_value = cursor
        cursor.fetchone.return_value = None
        database = SimpleNamespace(vendor="sqlite", cursor=MagicMock(return_value=cursor))
        with patch("personal_knowledge_base.search.connection", database):
            self.assertIsNone(vector_index_dim())

    @override_settings(APP_EMBEDDING_DIM=1024)
    def test_check_warns_on_dimension_mismatch(self):
        with patch("personal_knowledge_base.checks.vector_index_dim", return_value=384):
            warnings = check_vector_index_dimension(None)

        self.assertEqual(len(warnings), 1)
        self.assertIsInstance(warnings[0], Warning)
        self.assertEqual(warnings[0].id, "personal_knowledge_base.W001")
        self.assertIn("384", warnings[0].msg)
        self.assertIn("1024", warnings[0].msg)
        self.assertIn("keep using the existing 384-d index", warnings[0].msg)
        self.assertIn("python manage.py rebuild_vector_index", warnings[0].hint)

    @override_settings(APP_EMBEDDING_DIM=1024)
    def test_check_is_silent_when_dimension_matches_or_table_is_absent(self):
        for dim in (1024, None):
            with self.subTest(dim=dim), patch("personal_knowledge_base.checks.vector_index_dim", return_value=dim):
                self.assertEqual(check_vector_index_dimension(None), [])

    @override_settings(APP_EMBEDDING_DIM=1024)
    def test_effective_dim_uses_existing_index_and_falls_back_when_absent_or_unavailable(self):
        with patch("personal_knowledge_base.search.vector_index_dim", return_value=384):
            self.assertEqual(effective_embedding_dim(), 384)
        with patch("personal_knowledge_base.search.vector_index_dim", return_value=None):
            self.assertEqual(effective_embedding_dim(), 1024)
        with patch("personal_knowledge_base.search.vector_index_dim", side_effect=DatabaseError):
            self.assertEqual(effective_embedding_dim(), 1024)

    def test_stable_embedding_defaults_to_the_effective_dimension(self):
        with patch("personal_knowledge_base.search.vector_index_dim", return_value=384):
            self.assertEqual(len(stable_embedding("stable hash")), 384)

    def test_check_ignores_database_errors(self):
        with patch("personal_knowledge_base.checks.vector_index_dim", side_effect=DatabaseError("database unavailable")):
            self.assertEqual(check_vector_index_dimension(None), [])

    def test_migration_is_a_noop_when_vector_table_is_absent(self):
        cursor = MagicMock()
        cursor.__enter__.return_value = cursor
        cursor.fetchone.return_value = None
        database = MagicMock(vendor="sqlite")
        database.cursor.return_value = cursor

        resize_empty_vector_index(None, SimpleNamespace(connection=database))

        cursor.execute.assert_called_once_with(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = %s",
            ["chunk_embeddings_vec"],
        )


class VectorIndexDimensionMigrationTests(TransactionTestCase):
    def test_empty_mismatched_table_is_resized_and_non_empty_table_is_preserved(self):
        configured_dim = settings.APP_EMBEDDING_DIM
        mismatched_dim = 384 if configured_dim != 384 else 1024
        rowid = 945001
        with connection.cursor() as cursor:
            cursor.execute("DELETE FROM chunk_embeddings_vec")

        try:
            with override_settings(APP_EMBEDDING_DIM=mismatched_dim):
                resize_empty_vector_index(None, SimpleNamespace(connection=connection))
                self.assertEqual(vector_index_dim(), mismatched_dim)

            embedding = sqlite_vec.serialize_float32([1.0] + [0.0] * (mismatched_dim - 1))
            with connection.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO chunk_embeddings_vec(rowid, embedding) VALUES (%s, %s)",
                    [rowid, embedding],
                )

            with override_settings(APP_EMBEDDING_DIM=configured_dim):
                resize_empty_vector_index(None, SimpleNamespace(connection=connection))
                self.assertEqual(vector_index_dim(), mismatched_dim)

            with connection.cursor() as cursor:
                cursor.execute("SELECT embedding FROM chunk_embeddings_vec WHERE rowid = %s", [rowid])
                self.assertEqual(bytes(cursor.fetchone()[0]), embedding)
        finally:
            with connection.cursor() as cursor:
                cursor.execute("DELETE FROM chunk_embeddings_vec WHERE rowid = %s", [rowid])
            with override_settings(APP_EMBEDDING_DIM=configured_dim):
                resize_empty_vector_index(None, SimpleNamespace(connection=connection))

        self.assertEqual(vector_index_dim(), configured_dim)


class RuntimeVectorIndexDimensionTests(TransactionTestCase):
    def create_chunk(self, name):
        tenant = Tenant.objects.create(name=name)
        knowledge_base = KnowledgeBase.objects.create(tenant=tenant, name="kb")
        knowledge = Knowledge.objects.create(
            tenant=tenant,
            knowledge_base=knowledge_base,
            title=name,
            parse_status="completed",
        )
        chunk = Chunk.objects.create(
            tenant=tenant,
            knowledge_base=knowledge_base,
            knowledge=knowledge,
            content=f"{name} vector content",
            chunk_index=0,
        )
        return tenant, knowledge_base, chunk

    def test_index_and_search_use_populated_existing_dimension_until_rebuild(self):
        restore_dim = vector_index_dim() or settings.APP_EMBEDDING_DIM
        tenant, knowledge_base, chunk = self.create_chunk("existing 384 index")
        chunk_rowid = chunk.seq_id if chunk.seq_id is not None else _rowid(chunk.id)
        sentinel_rowid = 987654321012345
        sentinel = sqlite_vec.serialize_float32([0.0, 1.0] + [0.0] * 382)

        try:
            _set_vector_index_dim(384)
            with connection.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO chunk_embeddings_vec(rowid, embedding) VALUES (%s, %s)",
                    [sentinel_rowid, sentinel],
                )

            raw_vector = [2.0] + [0.0] * 1023
            with override_settings(
                APP_EMBEDDING_DIM=1024,
                LLM_USE_ENV_EMBEDDING=True,
                LLM_EMBEDDING_API_KEY="stub",
                LLM_EMBEDDING_BASE_URL="https://embedding.example/v1",
                LLM_EMBEDDING_MODEL="test-embedding",
            ), patch(
                "personal_knowledge_base.model_providers.openai_compatible_embedding",
                return_value=[raw_vector],
            ):
                index_chunk(chunk)
                self.assertEqual(vector_index_dim(), 384)
                self.assertEqual(effective_embedding_dim(), 384)
                results = _vector_search(tenant.id, {knowledge_base.id}, "vector query", 3)

            self.assertIn(chunk.id, results)
            with connection.cursor() as cursor:
                cursor.execute("SELECT embedding FROM chunk_embeddings_vec WHERE rowid = %s", [chunk_rowid])
                stored = bytes(cursor.fetchone()[0])
                cursor.execute("SELECT embedding FROM chunk_embeddings_vec WHERE rowid = %s", [sentinel_rowid])
                self.assertEqual(bytes(cursor.fetchone()[0]), sentinel)
            self.assertEqual(len(stored), 384 * 4)
        finally:
            _set_vector_index_dim(restore_dim)

    def test_hash_embedding_uses_the_existing_index_dimension(self):
        restore_dim = vector_index_dim() or settings.APP_EMBEDDING_DIM
        tenant, _, _ = self.create_chunk("hash with existing index")

        try:
            _set_vector_index_dim(384)
            with override_settings(APP_EMBEDDING_DIM=1024, LLM_USE_ENV_EMBEDDING=False):
                self.assertEqual(len(embedding(tenant, ["local hash fallback"])[0]), 384)
        finally:
            _set_vector_index_dim(restore_dim)

    def test_rebuild_targets_configured_dimension_and_indexes_1024d_vectors(self):
        restore_dim = vector_index_dim() or settings.APP_EMBEDDING_DIM
        tenant, knowledge_base, chunk = self.create_chunk("rebuild to 1024")
        chunk_rowid = chunk.seq_id if chunk.seq_id is not None else _rowid(chunk.id)

        try:
            _set_vector_index_dim(384)
            raw_vector = [2.0] + [0.0] * 1023
            with override_settings(
                APP_EMBEDDING_DIM=1024,
                LLM_USE_ENV_EMBEDDING=True,
                LLM_EMBEDDING_API_KEY="stub",
                LLM_EMBEDDING_BASE_URL="https://embedding.example/v1",
                LLM_EMBEDDING_MODEL="test-embedding",
            ), patch(
                "personal_knowledge_base.model_providers.openai_compatible_embedding",
                return_value=[raw_vector],
            ):
                call_command("rebuild_vector_index", stdout=StringIO())

            self.assertEqual(vector_index_dim(), 1024)
            with connection.cursor() as cursor:
                cursor.execute("SELECT embedding FROM chunk_embeddings_vec WHERE rowid = %s", [chunk_rowid])
                self.assertEqual(len(bytes(cursor.fetchone()[0])), 1024 * 4)
        finally:
            _set_vector_index_dim(restore_dim)
