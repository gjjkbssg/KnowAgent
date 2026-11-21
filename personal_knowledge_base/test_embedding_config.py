from unittest.mock import patch

from django.test import TestCase, override_settings

from personal_knowledge_base.model_providers import embedding
from personal_knowledge_base.models import Tenant
from personal_knowledge_base.search import stable_embedding

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
