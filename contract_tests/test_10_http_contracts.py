import time
import unittest
import uuid

from contract_tests.assertions import assert_snapshot, capture
from contract_tests.harness import admin, new_client


API = "/api/v1"
INDEXING = {
    "vector_enabled": True,
    "graph_enabled": False,
    "wiki_enabled": False,
}


class HttpApiContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.identity = admin()
        cls.client = new_client(cls.identity["token"])

    @classmethod
    def tearDownClass(cls):
        cls.client.close()

    def call(self, responses, label, method, path, expected_status=200, **kwargs):
        response = self.client.request(method, path, **kwargs)
        responses[label] = capture(self, label, response, expected_status)
        return response.json() if "application/json" in response.headers.get("content-type", "") else None

    def create_kb(self, responses, label, name):
        result = self.call(
            responses,
            label,
            "POST",
            f"{API}/knowledge-bases",
            201,
            json={"name": name, "description": "Contract fixture", "indexing_strategy": INDEXING},
        )
        return result["data"]["id"]

    def wait_for_knowledge(self, knowledge_id, expected_status="completed"):
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            response = self.client.get(f"{API}/knowledge/{knowledge_id}")
            if response.status_code == 200:
                knowledge = response.json().get("data", {})
                if knowledge.get("parse_status") == expected_status:
                    return knowledge
                if knowledge.get("parse_status") in {"failed", "error"}:
                    self.fail(f"Document processing failed: {knowledge}")
            time.sleep(0.25)
        self.fail(f"Knowledge {knowledge_id} did not reach {expected_status}")

    def upload(self, responses, label, kb_id, name, text, tag_id=""):
        data = {"tag_id": tag_id} if tag_id else {}
        result = self.call(
            responses,
            label,
            "POST",
            f"{API}/knowledge-bases/{kb_id}/knowledge/file",
            201,
            files={"file": (name, text.encode("utf-8"), "text/markdown")},
            data=data,
        )
        return result["data"]["knowledge"]["id"]

    def test_app_auth_boundaries_and_cross_tenant_resource_lookup(self):
        responses = {}
        with new_client() as unauthenticated:
            checks = [
                ("accounts", "GET", f"{API}/auth/me"),
                ("knowledge", "GET", f"{API}/knowledge-bases"),
                ("chat", "GET", f"{API}/sessions"),
                ("wiki", "GET", f"{API}/knowledge-bases/missing/wiki/pages"),
                ("agent", "GET", f"{API}/mcp-services"),
                ("models_config", "GET", f"{API}/models/usage"),
                ("personal_knowledge_base", "GET", f"{API}/rag-eval/questions"),
            ]
            for app, method, path in checks:
                response = unauthenticated.request(method, path)
                responses[f"unauthorized_{app}"] = capture(
                    self, f"unauthorized_{app}", response, 401
                )

        kb_id = self.create_kb(responses, "cross_tenant_fixture_kb", "Tenant A contract KB")
        registered = self.call(
            responses,
            "register_second_tenant",
            "POST",
            f"{API}/auth/register",
            201,
            json={
                "username": "contract-tenant-b",
                "email": "contract-tenant-b@example.test",
                "password": "Contract-password-tenant-b!",
            },
        )
        other = new_client(registered["data"]["token"])
        try:
            cross_tenant = other.get(f"{API}/knowledge-bases/{kb_id}")
            responses["cross_tenant_kb_detail"] = capture(
                self, "cross_tenant_kb_detail", cross_tenant, 404
            )
        finally:
            other.close()

        assert_snapshot(self, "auth_boundaries", responses)

    def test_knowledge_chunk_tag_wiki_move_copy_and_delete_routes(self):
        responses = {}
        source_kb = self.create_kb(responses, "create_source_kb", "Contract source KB")
        destination_kb = self.create_kb(
            responses, "create_destination_kb", "Contract destination KB"
        )

        self.call(responses, "list_kbs", "GET", f"{API}/knowledge-bases", 200)
        self.call(
            responses,
            "search_kbs",
            "GET",
            f"{API}/knowledge-bases",
            200,
            params={"keyword": "Contract source"},
        )
        self.call(
            responses,
            "get_source_kb",
            "GET",
            f"{API}/knowledge-bases/{source_kb}",
            200,
        )
        self.call(
            responses,
            "update_source_kb",
            "PUT",
            f"{API}/knowledge-bases/{source_kb}",
            200,
            json={"description": "Updated contract fixture"},
        )
        self.call(
            responses,
            "pin_source_kb",
            "PUT",
            f"{API}/knowledge-bases/{source_kb}/pin",
            200,
            json={"is_pinned": True},
        )
        self.call(
            responses,
            "move_targets",
            "GET",
            f"{API}/knowledge-bases/{source_kb}/move-targets",
            200,
        )
        copied = self.call(
            responses,
            "copy_kb",
            "POST",
            f"{API}/knowledge-bases/copy",
            200,
            json={"source_id": source_kb},
        )
        copied_kb = copied["data"]["knowledge_base"]["id"]

        tag = self.call(
            responses,
            "create_tag",
            "POST",
            f"{API}/knowledge-bases/{source_kb}/tags",
            201,
            json={"name": "contract-tag", "color": "#123456"},
        )
        tag_id = tag["data"]["id"]
        self.call(
            responses,
            "list_tags",
            "GET",
            f"{API}/knowledge-bases/{source_kb}/tags",
            200,
        )

        knowledge_id = self.upload(
            responses,
            "upload_markdown",
            source_kb,
            "contract-notes.md",
            "# Contract document\n\nA deterministic paragraph for black-box API contracts.",
            tag_id,
        )
        self.wait_for_knowledge(knowledge_id)
        self.call(
            responses,
            "list_knowledge",
            "GET",
            f"{API}/knowledge-bases/{source_kb}/knowledge",
            200,
            params={"keyword": "contract-notes"},
        )
        self.call(
            responses,
            "knowledge_stages",
            "GET",
            f"{API}/knowledge/{knowledge_id}/stages",
            200,
        )
        self.call(
            responses,
            "set_knowledge_tag",
            "PUT",
            f"{API}/knowledge/{knowledge_id}",
            200,
            json={"tag_id": tag_id, "title": "Contract document"},
        )

        self.call(
            responses,
            "reparse_knowledge",
            "POST",
            f"{API}/knowledge/{knowledge_id}/reparse",
            200,
            json={},
        )
        self.wait_for_knowledge(knowledge_id)
        self.call(
            responses,
            "cancel_knowledge",
            "POST",
            f"{API}/knowledge/{knowledge_id}/cancel-parse",
            200,
            json={},
        )

        chunks = self.call(
            responses,
            "list_chunks",
            "GET",
            f"{API}/chunks/{knowledge_id}",
            200,
        )
        self.assertTrue(chunks["data"]["items"], "The markdown fixture should produce a chunk")
        chunk_id = chunks["data"]["items"][0]["id"]
        self.call(
            responses,
            "update_chunk",
            "PUT",
            f"{API}/chunks/{knowledge_id}/{chunk_id}",
            200,
            json={"content": chunks["data"]["items"][0]["content"], "is_enabled": True},
        )

        missing_image = self.client.get(
            f"{API}/knowledge/{knowledge_id}/images/{uuid.UUID(int=0)}"
        )
        responses["knowledge_image_missing"] = capture(
            self, "knowledge_image_missing", missing_image, 404
        )

        self.call(
            responses,
            "move_knowledge",
            "POST",
            f"{API}/knowledge/move",
            200,
            json={
                "ids": [knowledge_id],
                "source_kb_id": source_kb,
                "target_knowledge_base_id": destination_kb,
            },
        )
        moved_chunks = self.call(
            responses,
            "list_moved_chunks",
            "GET",
            f"{API}/chunks/{knowledge_id}",
            200,
        )
        if moved_chunks["data"]["items"]:
            moved_chunk_id = moved_chunks["data"]["items"][0]["id"]
            self.call(
                responses,
                "delete_chunk",
                "DELETE",
                f"{API}/chunks/{knowledge_id}/{moved_chunk_id}",
                200,
            )
        self.call(
            responses,
            "delete_knowledge",
            "DELETE",
            f"{API}/knowledge/{knowledge_id}",
            200,
        )

        second_knowledge = self.upload(
            responses,
            "upload_for_batch_delete",
            destination_kb,
            "batch-delete.md",
            "# Batch delete\n\nA second deterministic document.",
        )
        self.wait_for_knowledge(second_knowledge)
        self.call(
            responses,
            "batch_delete_knowledge",
            "POST",
            f"{API}/knowledge/batch-delete",
            200,
            json={"ids": [second_knowledge], "kb_id": destination_kb},
        )

        self.call(
            responses,
            "create_wiki_page",
            "POST",
            f"{API}/knowledge-bases/{source_kb}/wiki/pages",
            201,
            json={
                "title": "Contract wiki page",
                "slug": "contract-page",
                "content": "Contract wiki content.",
                "summary": "Contract summary.",
            },
        )
        self.call(
            responses,
            "list_wiki_pages",
            "GET",
            f"{API}/knowledge-bases/{source_kb}/wiki/pages",
            200,
        )
        self.call(
            responses,
            "get_wiki_page",
            "GET",
            f"{API}/knowledge-bases/{source_kb}/wiki/pages/contract-page",
            200,
        )
        self.call(
            responses,
            "search_wiki",
            "GET",
            f"{API}/knowledge-bases/{source_kb}/wiki/search",
            200,
            params={"q": "Contract"},
        )
        self.call(
            responses,
            "wiki_graph",
            "GET",
            f"{API}/knowledge-bases/{source_kb}/wiki/graph",
            200,
        )

        self.call(
            responses,
            "delete_copied_kb",
            "DELETE",
            f"{API}/knowledge-bases/{copied_kb}",
            200,
        )
        self.call(
            responses,
            "delete_source_kb",
            "DELETE",
            f"{API}/knowledge-bases/{source_kb}",
            200,
        )
        self.call(
            responses,
            "delete_destination_kb",
            "DELETE",
            f"{API}/knowledge-bases/{destination_kb}",
            200,
        )

        assert_snapshot(self, "knowledge", responses)

    def test_sessions_and_messages_routes(self):
        responses = {}
        first = self.call(
            responses,
            "create_session",
            "POST",
            f"{API}/sessions",
            201,
            json={"title": "Contract session"},
        )
        session_id = first["data"]["id"]
        self.call(responses, "list_sessions", "GET", f"{API}/sessions", 200)
        self.call(
            responses,
            "get_session",
            "GET",
            f"{API}/sessions/{session_id}",
            200,
        )
        self.call(
            responses,
            "pin_session",
            "POST",
            f"{API}/sessions/{session_id}/pin",
            200,
        )
        self.call(
            responses,
            "unpin_session",
            "DELETE",
            f"{API}/sessions/{session_id}/pin",
            200,
        )
        self.call(
            responses,
            "stop_session",
            "POST",
            f"{API}/sessions/{session_id}/stop",
            200,
            json={"message_id": ""},
        )
        self.call(
            responses,
            "clear_session_messages",
            "DELETE",
            f"{API}/sessions/{session_id}/messages",
            200,
        )
        self.call(
            responses,
            "delete_session",
            "DELETE",
            f"{API}/sessions/{session_id}",
            200,
        )

        batch_ids = []
        for title in ("Batch session one", "Batch session two"):
            result = self.call(
                responses,
                f"create_{title.lower().replace(' ', '_')}",
                "POST",
                f"{API}/sessions",
                201,
                json={"title": title},
            )
            batch_ids.append(result["data"]["id"])
        self.call(
            responses,
            "delete_sessions",
            "DELETE",
            f"{API}/sessions/batch",
            200,
            json={"ids": batch_ids},
        )
        self.call(
            responses,
            "create_session_for_delete_all",
            "POST",
            f"{API}/sessions",
            201,
            json={"title": "Delete all fixture"},
        )
        self.call(
            responses,
            "delete_all_sessions",
            "DELETE",
            f"{API}/sessions/batch",
            200,
            json={"delete_all": True},
        )

        assert_snapshot(self, "sessions", responses)

    def test_models_system_tenant_kv_and_mcp_routes(self):
        responses = {}
        self.call(responses, "list_models", "GET", f"{API}/models", 200)
        self.call(
            responses,
            "model_usage",
            "GET",
            f"{API}/models/usage",
            200,
        )
        created_model = self.call(
            responses,
            "create_model",
            "POST",
            f"{API}/models",
            201,
            json={
                "id": "contract-local-model",
                "name": "Contract local model",
                "display_name": "Contract local model",
                "type": "KnowledgeQA",
                "source": "local",
            },
        )
        model_id = created_model["data"]["id"]
        self.call(
            responses,
            "update_model",
            "PUT",
            f"{API}/models/{model_id}",
            200,
            json={"display_name": "Updated contract model"},
        )
        self.call(
            responses,
            "update_model_credentials",
            "PUT",
            f"{API}/models/{model_id}/credentials",
            200,
            json={"credentials": {"api_key": "fake-contract-key"}},
        )
        self.call(
            responses,
            "delete_model_credential",
            "DELETE",
            f"{API}/models/{model_id}/credentials/api_key",
            200,
        )
        self.call(
            responses,
            "delete_model",
            "DELETE",
            f"{API}/models/{model_id}",
            200,
        )

        for label, method, path, kwargs in [
            ("system_info", "GET", f"{API}/system/info", {}),
            ("parser_engines", "GET", f"{API}/system/parser-engines", {}),
            ("storage_status", "GET", f"{API}/system/storage-engine-status", {}),
            ("vector_store_types", "GET", f"{API}/vector-stores/types", {}),
            ("web_search_provider_types", "GET", f"{API}/web-search-providers/types", {}),
            ("check_parser_engine", "POST", f"{API}/system/parser-engines/check", {"json": {}}),
            ("check_storage_engine", "POST", f"{API}/system/storage-engine-check", {"json": {}}),
        ]:
            self.call(responses, label, method, path, 200, **kwargs)

        self.call(
            responses,
            "get_tenant_kv",
            "GET",
            f"{API}/tenants/kv/context-config",
            200,
        )
        self.call(
            responses,
            "update_tenant_kv",
            "PUT",
            f"{API}/tenants/kv/context-config",
            200,
            json={"value": {"contract_fixture": True}},
        )

        service = self.call(
            responses,
            "create_mcp_service",
            "POST",
            f"{API}/mcp-services",
            201,
            json={"name": "Contract MCP", "url": "http://127.0.0.1:1"},
        )
        service_id = service["data"]["id"]
        self.call(responses, "list_mcp_services", "GET", f"{API}/mcp-services", 200)
        self.call(
            responses,
            "update_mcp_service",
            "PUT",
            f"{API}/mcp-services/{service_id}",
            200,
            json={"name": "Updated contract MCP"},
        )
        self.call(
            responses,
            "delete_mcp_service",
            "DELETE",
            f"{API}/mcp-services/{service_id}",
            200,
        )

        assert_snapshot(self, "models_system", responses)

    def test_rag_evaluation_question_and_history_routes(self):
        responses = {}
        questions = self.call(
            responses,
            "rag_eval_questions",
            "GET",
            f"{API}/rag-eval/questions",
            200,
        )
        self.assertTrue(questions["data"]["questions"])
        self.call(
            responses,
            "rag_eval_add_question",
            "POST",
            f"{API}/rag-eval/questions",
            200,
            json={
                "question": "What does the contract fixture test?",
                "ground_truth": "It tests the HTTP API contract.",
            },
        )
        self.call(
            responses,
            "rag_eval_history",
            "GET",
            f"{API}/rag-eval/history",
            200,
        )
        assert_snapshot(self, "rag_eval", responses)
