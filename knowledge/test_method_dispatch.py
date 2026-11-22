import json
from pathlib import Path
from unittest.mock import patch

from django.test import Client, TestCase

from personal_knowledge_base.models import Chunk, Knowledge, KnowledgeBase, KnowledgeTag, Tenant, WikiFolder


class MethodDispatchTests(TestCase):
    def setUp(self):
        self.client = Client()
        response = self.client.post("/api/v1/auth/auto-setup", content_type="application/json")
        self.assertIn(response.status_code, (200, 201))
        self.headers = {"HTTP_AUTHORIZATION": f"Bearer {response.json()['data']['token']}"}
        self.tenant = Tenant.objects.get(id=1)

    def _request(self, method, path, data=None):
        kwargs = dict(self.headers)
        if data is not None:
            kwargs.update(data=json.dumps(data), content_type="application/json")
        return getattr(self.client, method.lower())(path, **kwargs)

    def _knowledge_base(self, name="dispatch"):
        return KnowledgeBase.objects.create(tenant=self.tenant, name=name)

    def _knowledge(self, kb):
        return Knowledge.objects.create(
            tenant=self.tenant,
            knowledge_base=kb,
            type="file",
            title="before",
            source="dispatch.txt",
            file_name="dispatch.txt",
        )

    def test_knowledge_base_fallthrough_methods(self):
        for method in ("PUT", "PATCH", "DELETE"):
            with self.subTest(collection_method=method):
                name = f"created by {method}"
                response = self._request(method, "/api/v1/knowledge-bases", {"name": name})
                self.assertEqual(response.status_code, 201)
                self.assertEqual(response.json()["data"]["name"], name)

        kb = self._knowledge_base()
        for method in ("POST", "PATCH"):
            with self.subTest(detail_method=method):
                name = f"updated by {method}"
                response = self._request(method, f"/api/v1/knowledge-bases/{kb.id}", {"name": name})
                self.assertEqual(response.status_code, 200)
                kb.refresh_from_db()
                self.assertEqual(kb.name, name)

        response = self._request("PATCH", f"/api/v1/knowledge-bases/{kb.id}/pin", {"is_pinned": False})
        self.assertEqual(response.status_code, 200)
        kb.refresh_from_db()
        self.assertTrue(kb.is_pinned)

    def test_knowledge_file_fallthrough_reaches_upload_validation(self):
        kb = self._knowledge_base()
        response = self._request("PATCH", f"/api/v1/knowledge-bases/{kb.id}/knowledge/file")
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["message"], "file is required")

    def test_knowledge_detail_fallthrough_and_collection_405(self):
        kb = self._knowledge_base()
        knowledge = self._knowledge(kb)
        for method in ("POST", "PATCH"):
            with self.subTest(detail_method=method):
                title = f"updated by {method}"
                response = self._request(method, f"/api/v1/knowledge/{knowledge.id}", {"title": title})
                self.assertEqual(response.status_code, 200)
                knowledge.refresh_from_db()
                self.assertEqual(knowledge.title, title)

        response = self._request("PATCH", f"/api/v1/knowledge-bases/{kb.id}/knowledge")
        self.assertEqual(response.status_code, 405)
        self.assertEqual(
            response.json(),
            {
                "success": False,
                "error": {"code": "error", "message": "method not allowed", "details": {}},
                "message": "method not allowed",
            },
        )

    def test_chunk_fallthrough_methods(self):
        kb = self._knowledge_base()
        knowledge = self._knowledge(kb)
        chunk = Chunk.objects.create(
            tenant=self.tenant,
            knowledge_base=kb,
            knowledge=knowledge,
            content="before",
            chunk_index=0,
        )
        service_module = "services.chunks" if (Path(__file__).resolve().parents[1] / "services" / "chunks.py").exists() else "knowledge.views"
        with patch(f"{service_module}.chunk_vector", return_value=[]), patch(f"{service_module}.index_chunk"):
            for method in ("POST", "PATCH"):
                with self.subTest(detail_method=method):
                    content = f"updated by {method}"
                    response = self._request(
                        method,
                        f"/api/v1/chunks/{knowledge.id}/{chunk.id}",
                        {"content": content},
                    )
                    self.assertEqual(response.status_code, 200)
                    chunk.refresh_from_db()
                    self.assertEqual(chunk.content, content)

        by_id = self._request("PATCH", f"/api/v1/chunks/by-id/{chunk.id}", {"content": "ignored"})
        self.assertEqual(by_id.status_code, 404)
        self.assertEqual(by_id.json()["message"], "not found")
        collection = self._request("PATCH", f"/api/v1/chunks/{knowledge.id}")
        self.assertEqual(collection.status_code, 404)
        self.assertEqual(collection.json()["message"], "not found")

    def test_tag_fallthrough_methods(self):
        kb = self._knowledge_base()
        tag_ids = []
        for method in ("PUT", "PATCH", "DELETE"):
            with self.subTest(collection_method=method):
                response = self._request(method, f"/api/v1/knowledge-bases/{kb.id}/tags", {"name": method})
                self.assertEqual(response.status_code, 201)
                tag_ids.append(response.json()["data"]["id"])

        tag = KnowledgeTag.objects.get(id=tag_ids[0])
        for method in ("POST", "PATCH"):
            with self.subTest(detail_method=method):
                name = f"updated by {method}"
                response = self._request(
                    method,
                    f"/api/v1/knowledge-bases/{kb.id}/tags/{tag.id}",
                    {"name": name},
                )
                self.assertEqual(response.status_code, 200)
                tag.refresh_from_db()
                self.assertEqual(tag.name, name)

    def test_wiki_page_fallthrough_methods(self):
        kb = self._knowledge_base()
        page_slug = None
        for method in ("PUT", "PATCH", "DELETE"):
            with self.subTest(collection_method=method):
                slug = f"created-{method.lower()}"
                response = self._request(
                    method,
                    f"/api/v1/knowledge-bases/{kb.id}/wiki/pages",
                    {"slug": slug, "title": slug},
                )
                self.assertEqual(response.status_code, 201)
                if page_slug is None:
                    page_slug = response.json()["data"]["slug"]

        for method in ("POST", "PATCH"):
            with self.subTest(detail_method=method):
                title = f"updated by {method}"
                response = self._request(
                    method,
                    f"/api/v1/knowledge-bases/{kb.id}/wiki/pages/{page_slug}",
                    {"title": title},
                )
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["data"]["title"], title)

    def test_wiki_folder_fallthrough_methods(self):
        kb = self._knowledge_base()
        folder_ids = []
        for method in ("PUT", "PATCH", "DELETE"):
            with self.subTest(collection_method=method):
                response = self._request(
                    method,
                    f"/api/v1/knowledge-bases/{kb.id}/wiki/folders",
                    {"name": f"created by {method}"},
                )
                self.assertEqual(response.status_code, 201)
                folder_ids.append(response.json()["data"]["id"])

        folder = WikiFolder.objects.get(id=folder_ids[0])
        for method in ("GET", "POST", "PATCH"):
            with self.subTest(detail_method=method):
                data = {"name": f"updated by {method}"} if method != "GET" else None
                response = self._request(
                    method,
                    f"/api/v1/knowledge-bases/{kb.id}/wiki/folders/{folder.id}",
                    data,
                )
                self.assertEqual(response.status_code, 200)
                folder.refresh_from_db()
                if method != "GET":
                    self.assertEqual(folder.name, f"updated by {method}")
