from unittest.mock import Mock, patch

import requests
from django.db import connection
from django.test import TestCase, override_settings

from personal_knowledge_base.model_providers import EmbeddingUnavailableError, embedding
from personal_knowledge_base.models import Chunk, Knowledge, KnowledgeBase, Tenant
from personal_knowledge_base.search import hybrid_search, index_chunk, stable_embedding

EMBEDDING_ENV = {
    "LLM_EMBEDDING_API_KEY": "embedding-key",
    "LLM_EMBEDDING_BASE_URL": "https://embedding.example/v1",
    "LLM_EMBEDDING_MODEL": "text-embedding-v4",
    "LLM_CHAT_API_KEY": "chat-key",
    "LLM_CHAT_BASE_URL": "https://chat.example/v1",
    "APP_EMBEDDING_DIM": 4,
}


class EnvEmbeddingTests(TestCase):
    def setUp(self):
        self.tenant = Tenant.objects.create(name="t")

    @override_settings(LLM_USE_ENV_EMBEDDING=True, **EMBEDDING_ENV)
    def test_env_embedding_calls_the_embedding_provider_not_the_chat_provider(self):
        with patch("personal_knowledge_base.model_providers.openai_compatible_embedding", return_value=[[0.1, 0.2, 0.3, 0.4]]) as call:
            vectors = embedding(self.tenant, ["hello"])

        call.assert_called_once_with("https://embedding.example/v1", "embedding-key", "text-embedding-v4", ["hello"])
        self.assertEqual(vectors, [[0.1, 0.2, 0.3, 0.4]])

    @override_settings(LLM_USE_ENV_EMBEDDING=False, **EMBEDDING_ENV)
    def test_without_the_switch_vectors_are_local_hashes(self):
        with patch("personal_knowledge_base.model_providers.openai_compatible_embedding") as call:
            vectors = embedding(self.tenant, ["hello"])

        call.assert_not_called()
        self.assertEqual(vectors, [stable_embedding("hello")])

    @override_settings(LLM_USE_ENV_EMBEDDING=True, **EMBEDDING_ENV)
    def test_provider_outage_raises_instead_of_returning_hash_vectors(self):
        outage = requests.ConnectionError("connection reset")
        with patch("personal_knowledge_base.model_providers.openai_compatible_embedding", side_effect=outage) as call, patch(
            "personal_knowledge_base.model_providers.time.sleep"
        ):
            with self.assertRaises(EmbeddingUnavailableError):
                embedding(self.tenant, ["hello"])

        self.assertEqual(call.call_count, 3)

    @override_settings(LLM_USE_ENV_EMBEDDING=True, **EMBEDDING_ENV)
    def test_transient_provider_error_is_retried(self):
        responses = [requests.Timeout("timed out"), [[0.1, 0.2, 0.3, 0.4]]]
        with patch("personal_knowledge_base.model_providers.openai_compatible_embedding", side_effect=responses), patch(
            "personal_knowledge_base.model_providers.time.sleep"
        ):
            vectors = embedding(self.tenant, ["hello"])

        self.assertEqual(vectors, [[0.1, 0.2, 0.3, 0.4]])

    @override_settings(LLM_USE_ENV_EMBEDDING=True, **EMBEDDING_ENV)
    def test_rejected_credentials_are_not_retried(self):
        rejected = requests.HTTPError("401 Unauthorized", response=Mock(status_code=401))
        with patch("personal_knowledge_base.model_providers.openai_compatible_embedding", side_effect=rejected) as call, patch(
            "personal_knowledge_base.model_providers.time.sleep"
        ):
            with self.assertRaises(EmbeddingUnavailableError):
                embedding(self.tenant, ["hello"])

        self.assertEqual(call.call_count, 1)


class EmbeddingOutageTests(TestCase):
    def setUp(self):
        self.tenant = Tenant.objects.create(name="outage")
        self.kb = KnowledgeBase.objects.create(tenant=self.tenant, name="kb")
        knowledge = Knowledge.objects.create(tenant=self.tenant, knowledge_base=self.kb, title="doc", parse_status="completed")
        self.chunk = Chunk.objects.create(
            tenant=self.tenant, knowledge_base=self.kb, knowledge=knowledge, content="kiwi orchard notes", chunk_index=0
        )

    def provider_down(self):
        return patch("personal_knowledge_base.model_providers.openai_compatible_embedding", side_effect=requests.ConnectionError("down"))

    def test_chunk_is_not_indexed_when_its_embedding_cannot_be_computed(self):
        with override_settings(LLM_USE_ENV_EMBEDDING=True, LLM_EMBEDDING_API_KEY="key"), self.provider_down(), patch(
            "personal_knowledge_base.model_providers.time.sleep"
        ):
            with self.assertRaises(EmbeddingUnavailableError):
                index_chunk(self.chunk)

        with connection.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM chunks_fts WHERE chunk_id = %s", [self.chunk.id])
            self.assertEqual(cursor.fetchone()[0], 0)

    def test_search_falls_back_to_full_text_when_the_query_cannot_be_embedded(self):
        index_chunk(self.chunk)

        with override_settings(LLM_USE_ENV_EMBEDDING=True, LLM_EMBEDDING_API_KEY="key"), self.provider_down(), patch(
            "personal_knowledge_base.model_providers.time.sleep"
        ):
            results = hybrid_search(self.tenant.id, [self.kb.id], "kiwi", 3)

        self.assertEqual([item["chunk_id"] for item in results], [self.chunk.id])
