import json
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

from django.test import Client, TestCase
from django.utils import timezone

from personal_knowledge_base.models import AgentToolApproval, Message, Session, Tenant
from personal_knowledge_base.serializers import message_dict
from personal_knowledge_base.stream_manager import stream_manager


class ToolApprovalTests(TestCase):
    def setUp(self):
        self.client = Client()
        auth = self.client.post("/api/v1/auth/auto-setup", content_type="application/json")
        self.headers = {"HTTP_AUTHORIZATION": f"Bearer {auth.json()['data']['token']}"}
        self.tenant = Tenant.objects.first()
        self.session = Session.objects.create(tenant=self.tenant, title="approval")

    def create_pending(self, *, session=None, expires_at=None):
        session = session or self.session
        user_message = Message.objects.create(session=session, role="user", content="run this")
        message = Message.objects.create(session=session, role="assistant", content="Thinking", is_completed=False)
        return AgentToolApproval.objects.create(
            tenant=session.tenant,
            session=session,
            message=message,
            tool_calls=[{"id": "c1", "name": "mcp_server_write", "arguments": {"x": 1}, "service_name": "server", "remote_tool_name": "write"}],
            checkpoint={"query": "run this", "messages": [], "chat": {"query": "run this", "session_id": session.id}},
            expires_at=expires_at or timezone.now() + timedelta(hours=1),
        ), message, user_message

    def test_decision_claims_once_and_duplicate_returns_current_status(self):
        approval, message, _ = self.create_pending()
        with patch("chat.views.run_database_background", side_effect=lambda callback: callback()), patch(
            "chat.views._resume_agent_generation"
        ) as resume:
            response = self.client.post(
                f"/api/v1/sessions/{self.session.id}/tool-approvals/{approval.id}",
                data=json.dumps({"decision": "approve"}),
                content_type="application/json",
                **self.headers,
            )
            duplicate = self.client.post(
                f"/api/v1/sessions/{self.session.id}/tool-approvals/{approval.id}",
                data=json.dumps({"decision": "approve"}),
                content_type="application/json",
                **self.headers,
            )
        approval.refresh_from_db()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(approval.status, "approved")
        self.assertEqual(response.json()["data"]["approval"]["id"], approval.id)
        self.assertEqual(response.json()["data"]["message_id"], message.id)
        self.assertEqual(duplicate.status_code, 409)
        self.assertEqual(duplicate.json()["error"]["details"]["status"], "approved")
        resume.assert_called_once_with(approval.id)
        self.assertFalse(message.is_completed)

    def test_rejection_is_claimed_and_resumed_as_rejection(self):
        approval, _, _ = self.create_pending()
        with patch("chat.views.run_database_background", side_effect=lambda callback: callback()), patch(
            "chat.views._resume_agent_generation"
        ) as resume:
            response = self.client.post(
                f"/api/v1/sessions/{self.session.id}/tool-approvals/{approval.id}",
                data=json.dumps({"decision": "reject"}),
                content_type="application/json",
                **self.headers,
            )
        approval.refresh_from_db()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(approval.status, "rejected")
        resume.assert_called_once_with(approval.id)

    def test_invalid_decision_and_foreign_session_are_rejected(self):
        approval, _, _ = self.create_pending()
        invalid = self.client.post(
            f"/api/v1/sessions/{self.session.id}/tool-approvals/{approval.id}",
            data=json.dumps({"decision": "maybe"}),
            content_type="application/json",
            **self.headers,
        )
        other_session = Session.objects.create(tenant=self.tenant, title="other")
        foreign_path = self.client.post(
            f"/api/v1/sessions/{other_session.id}/tool-approvals/{approval.id}",
            data=json.dumps({"decision": "reject"}),
            content_type="application/json",
            **self.headers,
        )
        self.assertEqual(invalid.status_code, 400)
        self.assertEqual(foreign_path.status_code, 404)
        other_tenant = Tenant.objects.create(name="approval other tenant", api_key="approval-other-tenant")
        foreign_session = Session.objects.create(tenant=other_tenant, title="foreign")
        foreign_approval, _, _ = self.create_pending(session=foreign_session)
        foreign_tenant_path = self.client.post(
            f"/api/v1/sessions/{foreign_session.id}/tool-approvals/{foreign_approval.id}",
            data=json.dumps({"decision": "reject"}),
            content_type="application/json",
            **self.headers,
        )
        self.assertEqual(foreign_tenant_path.status_code, 404)

    def test_expired_approval_finalizes_message_and_returns_410(self):
        approval, message, _ = self.create_pending(expires_at=timezone.now() - timedelta(seconds=1))
        response = self.client.post(
            f"/api/v1/sessions/{self.session.id}/tool-approvals/{approval.id}",
            data=json.dumps({"decision": "approve"}),
            content_type="application/json",
            **self.headers,
        )
        approval.refresh_from_db()
        message.refresh_from_db()
        self.assertEqual(response.status_code, 410)
        self.assertEqual(approval.status, "expired")
        self.assertTrue(message.is_completed)
        self.assertEqual(message.content, "等待确认超时，未执行该操作。")

    def test_message_serialization_exposes_only_live_pending_approval(self):
        approval, message, _ = self.create_pending()
        self.assertEqual(message_dict(message)["pending_tool_approval"]["id"], approval.id)
        approval.expires_at = timezone.now() - timedelta(seconds=1)
        approval.save(update_fields=["expires_at"])
        self.assertNotIn("pending_tool_approval", message_dict(message))

    def test_continue_stream_replays_pending_approval_after_stream_loss(self):
        approval, message, _ = self.create_pending()
        stream_manager.remove_stream(message.id)
        response = self.client.get(
            f"/api/v1/sessions/continue-stream/{self.session.id}?message_id={message.id}",
            **self.headers,
        )
        body = b"".join(response.streaming_content).decode()
        self.assertIn("tool_approval_required", body)
        self.assertIn(approval.id, body)
        self.assertIn("event: done", body)

    def test_session_stop_cancels_pending_approval(self):
        approval, message, _ = self.create_pending()
        response = self.client.post(
            f"/api/v1/sessions/{self.session.id}/stop",
            data=json.dumps({"message_id": message.id}),
            content_type="application/json",
            **self.headers,
        )
        approval.refresh_from_db()
        message.refresh_from_db()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(approval.status, "cancelled")
        self.assertTrue(message.is_completed)

    def test_session_stop_by_request_id_cancels_assistant_approval(self):
        approval, message, user_message = self.create_pending()
        request_id = "shared-request-id"
        user_message.request_id = request_id
        user_message.save(update_fields=["request_id", "updated_at"])
        message.request_id = request_id
        message.save(update_fields=["request_id", "updated_at"])
        stream = stream_manager.create_stream(message.id, self.session.id)

        response = self.client.post(
            f"/api/v1/sessions/{self.session.id}/stop",
            data=json.dumps({"message_id": request_id}),
            content_type="application/json",
            **self.headers,
        )
        approval.refresh_from_db()
        message.refresh_from_db()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(approval.status, "cancelled")
        self.assertTrue(message.is_completed)
        self.assertTrue(stream.is_complete)

    def test_resume_reloads_checkpoint_without_stream_manager_state(self):
        from chat.views import _resume_agent_generation

        approval, message, _ = self.create_pending()
        approval.status = "approved"
        approval.checkpoint["chat"] = {
            "agent_config": {"allowed_tools": []},
            "mcp_service_ids": [],
            "user_id": "",
            "user_message_id": "",
            "enable_memory": False,
            "enable_chat_history": False,
            "enable_snapshot": False,
            "query": "run this",
            "session_id": str(self.session.id),
        }
        approval.save()
        stream_manager.remove_stream(message.id)
        toolset = SimpleNamespace(tools=[], close=lambda: None)
        result = SimpleNamespace(content="finished", references=[], steps=[], duration_ms=5)
        with (
            patch("personal_knowledge_base.mcp_client.load_mcp_toolset", return_value=toolset),
            patch("chat.views.AgentEngine") as engine,
            patch("chat.views.schedule_chat_maintenance"),
        ):
            engine.return_value.resume.return_value = result
            _resume_agent_generation(approval.id)

        engine.return_value.resume.assert_called_once()
        self.assertIsNotNone(stream_manager.get_stream(message.id))
        message.refresh_from_db()
        self.assertTrue(message.is_completed)
        self.assertEqual(message.content, "finished")

    def test_approval_stream_event_is_terminal(self):
        from chat.views import _persist_agent_result, stream_message_events

        message = Message.objects.create(session=self.session, role="assistant", content="", is_completed=False)
        stream = stream_manager.create_stream(message.id, self.session.id)
        result = SimpleNamespace(
            content="Thinking",
            references=[],
            steps=[],
            duration_ms=4,
            stopped_reason="awaiting_approval",
            pending_approval=[{"id": "c1", "name": "tool", "arguments": {}, "service_name": "srv", "remote_tool_name": "tool"}],
            checkpoint={"query": "q", "messages": [], "steps": [], "pending_round": {}},
        )
        with patch("chat.views.schedule_chat_maintenance"):
            _persist_agent_result(
                result,
                assistant_msg_id=message.id,
                stream=stream,
                tenant=self.tenant,
                user_id="",
                user=None,
                user_msg_id="",
                session_id=self.session.id,
                query="q",
                agent_config={},
                mcp_service_ids=[],
                enable_memory=False,
                enable_chat_history=False,
                enable_snapshot=False,
            )
        body = "".join(stream_message_events(message))
        self.assertIn("tool_approval_required", body)
        self.assertIn("event: done", body)
        self.assertTrue(stream_manager.is_complete(message.id))
