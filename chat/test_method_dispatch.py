import json

from django.test import Client, TestCase
from django.utils import timezone

from personal_knowledge_base.models import Session, Tenant


class ChatMethodDispatchTests(TestCase):
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

    def test_non_get_non_delete_session_collection_methods_create(self):
        for method in ("PUT", "PATCH"):
            with self.subTest(method=method):
                response = self._request(method, "/api/v1/sessions", {"title": f"created by {method}"})
                self.assertEqual(response.status_code, 201)
                self.assertTrue(Session.objects.filter(id=response.json()["data"]["id"], tenant=self.tenant).exists())

    def test_non_get_non_delete_session_detail_methods_update(self):
        session = Session.objects.create(tenant=self.tenant, title="before")
        for method in ("POST", "PATCH"):
            with self.subTest(method=method):
                title = f"updated by {method}"
                response = self._request(method, f"/api/v1/sessions/{session.id}", {"title": title})
                self.assertEqual(response.status_code, 200)
                session.refresh_from_db()
                self.assertEqual(session.title, title)

    def test_non_post_session_pin_method_unpins(self):
        session = Session.objects.create(
            tenant=self.tenant,
            title="pinned",
            is_pinned=True,
            pinned_at=timezone.now(),
        )
        response = self._request("PATCH", f"/api/v1/sessions/{session.id}/pin")
        self.assertEqual(response.status_code, 200)
        session.refresh_from_db()
        self.assertFalse(session.is_pinned)

    def test_non_post_continue_stream_ignores_body_message_id(self):
        session = Session.objects.create(tenant=self.tenant, title="stream")
        response = self._request(
            "PATCH",
            f"/api/v1/sessions/continue-stream/{session.id}",
            {"message_id": "body-only"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["message"], "message_id is required")

    def test_tool_approval_keeps_original_405_for_non_post(self):
        response = self._request("PATCH", "/api/v1/sessions/session-id/tool-approvals/approval-id")
        self.assertEqual(response.status_code, 405)
        self.assertEqual(response.json()["message"], "method not allowed")
