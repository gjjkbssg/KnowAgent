import json
import tempfile

from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.test import Client, TestCase, override_settings

from .models import Chunk, Knowledge, KnowledgeBase, Message, Session, Tenant


class TenantIsolationTests(TestCase):
    def setUp(self):
        self.media_dir = tempfile.TemporaryDirectory()
        self.override = override_settings(MEDIA_ROOT=self.media_dir.name)
        self.override.enable()
        self.addCleanup(self.override.disable)
        self.addCleanup(self.media_dir.cleanup)
        self.client = Client()
        self.owner = self.register("owner")
        self.other = self.register("other")
        self.tenant = Tenant.objects.get(id=self.owner["tenant_id"])
        self.kb = KnowledgeBase.objects.create(tenant=self.tenant, name="私有知识库")
        self.knowledge = Knowledge.objects.create(
            tenant=self.tenant,
            knowledge_base=self.kb,
            type="file",
            title="私有文档",
            source="private.txt",
            file_name="private.txt",
            parse_status="completed",
        )
        self.chunk = Chunk.objects.create(
            tenant=self.tenant,
            knowledge_base=self.kb,
            knowledge=self.knowledge,
            content="只有所有者可见的内容",
            chunk_index=0,
        )
        self.session = Session.objects.create(tenant=self.tenant, title="私有会话")
        self.message = Message.objects.create(
            session=self.session, request_id="r1", role="user", content="私有消息", is_completed=True
        )
        self.file_path = default_storage.save(f"tenant-{self.tenant.id}/{self.kb.id}/private.txt", ContentFile(b"secret"))

    def register(self, name):
        response = self.client.post(
            "/api/v1/auth/register",
            data=json.dumps({"username": name, "email": f"{name}@example.com", "password": "pw-123456"}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 201)
        data = response.json()["data"]
        return {"headers": {"HTTP_AUTHORIZATION": f"Bearer {data['token']}"}, "tenant_id": data["tenant"]["id"]}

    def private_requests(self):
        kb, knowledge, chunk, session, message = self.kb.id, self.knowledge.id, self.chunk.id, self.session.id, self.message.id
        return [
            ("get", f"/api/v1/knowledge-bases/{kb}"),
            ("get", f"/api/v1/knowledge-bases/{kb}/knowledge"),
            ("get", f"/api/v1/knowledge-bases/{kb}/tags"),
            ("get", f"/api/v1/knowledge-bases/{kb}/knowledge/stats"),
            ("get", f"/api/v1/knowledge-bases/{kb}/wiki/pages"),
            ("get", f"/api/v1/knowledge/{knowledge}"),
            ("get", f"/api/v1/knowledge/{knowledge}/preview"),
            ("get", f"/api/v1/knowledge/{knowledge}/download"),
            ("get", f"/api/v1/chunks/{knowledge}"),
            ("get", f"/api/v1/chunks/by-id/{chunk}"),
            ("get", f"/api/v1/messages/{session}/load"),
            ("post", f"/api/v1/sessions/{session}/pin"),
            ("post", f"/api/v1/sessions/{session}/stop"),
            ("delete", f"/api/v1/sessions/{session}/messages"),
            ("delete", f"/api/v1/messages/{session}/{message}"),
            ("delete", f"/api/v1/chunks/by-id/{chunk}"),
            ("delete", f"/api/v1/knowledge/{knowledge}"),
            ("get", f"/files?file_path={self.file_path}"),
        ]

    def test_other_tenant_and_anonymous_cannot_reach_private_resources(self):
        for method, url in self.private_requests():
            with self.subTest(url=url, caller="anonymous"):
                self.assertEqual(getattr(self.client, method)(url).status_code, 401)
            with self.subTest(url=url, caller="other tenant"):
                self.assertEqual(getattr(self.client, method)(url, **self.other["headers"]).status_code, 404)
        self.assertTrue(Message.objects.filter(id=self.message.id).exists())
        self.assertTrue(Chunk.objects.filter(id=self.chunk.id).exists())
        self.assertFalse(Knowledge.objects.get(id=self.knowledge.id).deleted_at)

    def test_other_tenant_cannot_address_private_resources_by_body_id(self):
        headers = self.other["headers"]
        response = self.client.post(
            "/api/v1/knowledge/batch-delete",
            data=json.dumps({"ids": [self.knowledge.id]}),
            content_type="application/json",
            **headers,
        )
        self.assertEqual(response.json()["data"]["deleted"], 0)
        response = self.client.get(f"/api/v1/knowledge/batch?ids={self.knowledge.id}", **headers)
        self.assertEqual(response.json()["data"]["items"], [])
        response = self.client.post(
            "/api/v1/knowledge-bases/copy",
            data=json.dumps({"source_id": self.kb.id}),
            content_type="application/json",
            **headers,
        )
        self.assertEqual(response.status_code, 404)

    def test_owner_still_reaches_private_resources(self):
        headers = self.owner["headers"]
        for method, url in self.private_requests():
            if method != "get":
                continue
            with self.subTest(url=url):
                self.assertEqual(self.client.get(url, **headers).status_code, 200)
        self.assertEqual(self.client.post(f"/api/v1/sessions/{self.session.id}/pin", **headers).status_code, 200)
        self.assertEqual(self.client.delete(f"/api/v1/messages/{self.session.id}/{self.message.id}", **headers).status_code, 200)
        self.assertFalse(Message.objects.filter(id=self.message.id).exists())


class AutoSetupTests(TestCase):
    def auto_setup(self):
        return self.client.post("/api/v1/auth/auto-setup", content_type="application/json")

    @override_settings(AUTO_LOGIN_ENABLED=False)
    def test_bootstraps_first_admin_then_refuses_to_issue_tokens(self):
        first = self.auto_setup()
        self.assertEqual(first.status_code, 201)
        self.assertTrue(first.json()["data"]["temp_password"])

        second = self.auto_setup()
        self.assertEqual(second.status_code, 403)
        self.assertEqual(second.json()["error"]["code"], "auto_login_disabled")
        self.assertNotIn("data", second.json())

        login = self.client.post(
            "/api/v1/auth/login",
            data=json.dumps({"email": "admin@knowledge.local", "password": first.json()["data"]["temp_password"]}),
            content_type="application/json",
        )
        self.assertEqual(login.status_code, 200)

    @override_settings(AUTO_LOGIN_ENABLED=True)
    def test_local_development_keeps_passwordless_login(self):
        self.assertEqual(self.auto_setup().status_code, 201)
        again = self.auto_setup()
        self.assertEqual(again.status_code, 200)
        self.assertTrue(again.json()["data"]["token"])
