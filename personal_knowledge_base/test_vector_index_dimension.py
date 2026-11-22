import importlib
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import sqlite_vec
from django.conf import settings
from django.core.checks import Warning
from django.db import DatabaseError, connection
from django.test import SimpleTestCase, TransactionTestCase, override_settings

from personal_knowledge_base.checks import check_vector_index_dimension
from personal_knowledge_base.search import vector_index_dim

resize_empty_vector_index = importlib.import_module(
    "personal_knowledge_base.migrations.0017_vector_index_dimension"
).resize_empty_vector_index


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
        self.assertIn("python manage.py rebuild_vector_index", warnings[0].hint)

    @override_settings(APP_EMBEDDING_DIM=1024)
    def test_check_is_silent_when_dimension_matches_or_table_is_absent(self):
        for dim in (1024, None):
            with self.subTest(dim=dim), patch("personal_knowledge_base.checks.vector_index_dim", return_value=dim):
                self.assertEqual(check_vector_index_dimension(None), [])

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
