import json
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock, patch

from django.core.cache import cache
from django.db import IntegrityError, connections
from django.test import Client, TransactionTestCase
from django.utils import timezone

from personal_knowledge_base import tasks
from personal_knowledge_base.agent_engine import AgentEngine
from personal_knowledge_base.mcp_client import MCPTool
from personal_knowledge_base.models import AgentToolApproval, Message, Session, TaskRecord, Tenant
from personal_knowledge_base.serializers import message_dict
from personal_knowledge_base.stream_manager import stream_manager
from personal_knowledge_base.test_stream_resume import parse_sse
from services.context import ServiceContext


class ToolApprovalTests(TransactionTestCase):
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
        self.addCleanup(stream_manager.remove_stream, message.id)
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
        with patch("services.chat_generation.submit_approval_resume") as submit:
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
        submit.assert_called_once_with(approval.id)
        self.assertEqual(TaskRecord.objects.get(id=approval.id).status, "pending")
        self.assertFalse(message.is_completed)

    def test_rejection_queues_resume(self):
        approval, _, _ = self.create_pending()
        with patch("services.chat_generation.submit_approval_resume") as submit:
            response = self.client.post(
                f"/api/v1/sessions/{self.session.id}/tool-approvals/{approval.id}",
                data=json.dumps({"decision": "reject"}),
                content_type="application/json",
                **self.headers,
            )
        approval.refresh_from_db()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(approval.status, "rejected")
        submit.assert_called_once_with(approval.id)
        self.assertEqual(TaskRecord.objects.get(id=approval.id).task_type, tasks.RESUME_APPROVAL_TASK)

    def test_decision_rolls_back_if_task_insertion_fails(self):
        approval, _, _ = self.create_pending()
        with patch("services.chat_generation.TaskRecord.objects.create", side_effect=IntegrityError), self.assertRaises(IntegrityError):
            self.client.post(
                f"/api/v1/sessions/{self.session.id}/tool-approvals/{approval.id}",
                data=json.dumps({"decision": "approve"}),
                content_type="application/json",
                **self.headers,
            )
        approval.refresh_from_db()
        self.assertEqual(approval.status, "pending")
        self.assertFalse(TaskRecord.objects.filter(id=approval.id).exists())

    def test_public_progress_hides_approval_resumes_from_database_and_cache(self):
        approval, _, _ = self.create_pending()
        TaskRecord.objects.create(id=approval.id, task_type=tasks.RESUME_APPROVAL_TASK)
        url = f"/api/v1/knowledge/move/progress/{approval.id}"
        hidden = {"status": "not_found", "progress": 0}
        self.assertEqual(self.client.get(url).json()["data"], hidden)
        cache_key = f"task:{approval.id}"
        cache.set(cache_key, {"status": "running", "progress": 0.1})
        self.addCleanup(cache.delete, cache_key)
        self.assertEqual(self.client.get(url).json()["data"], hidden)

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
        self.assertFalse(TaskRecord.objects.filter(id=approval.id).exists())

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
            HTTP_LAST_EVENT_ID="9",
            **self.headers,
        )
        frames = parse_sse(response)
        self.assertEqual([(frame_id, event) for frame_id, event, _ in frames], [(10, "message"), (11, "done")])
        self.assertEqual(frames[0][2]["response_type"], "tool_approval_required")
        self.assertEqual(frames[0][2]["approval"]["id"], approval.id)

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

    def queue_approval_without_dispatch(self, decision="approve"):
        client = Mock()
        client.call_tool.return_value = ("written", False)
        tool = MCPTool("service", "Server", "write", "write", {}, {}, client, name="mcp_server_write")
        approval, message, _ = self.create_pending()
        engine = AgentEngine(
            self.tenant, self.session.id, agent_config={"allowed_tools": [], "extra_tools": [tool]},
        )
        with patch("personal_knowledge_base.model_providers.chat_completion_raw", return_value={
            "content": "Thinking",
            "tool_calls": [{"id": "c1", "type": "function", "function": {
                "name": tool.name(), "arguments": '{"x": 1}',
            }}],
        }):
            suspended = engine.execute("run this")
        self.assertEqual(suspended.stopped_reason, "awaiting_approval")
        client.call_tool.assert_not_called()
        approval.checkpoint = suspended.checkpoint
        approval.checkpoint["chat"] = {
            "agent_config": {"allowed_tools": []},
            "query": "run this",
        }
        approval.save(update_fields=["checkpoint"])
        with patch("services.chat_generation.submit_approval_resume"):
            response = self.client.post(
                f"/api/v1/sessions/{self.session.id}/tool-approvals/{approval.id}",
                data=json.dumps({"decision": decision}),
                content_type="application/json",
                **self.headers,
            )
        self.assertEqual(response.status_code, 200)
        return approval, message, client, SimpleNamespace(tools=[tool], close=lambda: None)

    def test_crash_after_decision_is_recovered_once_and_saved_message_can_be_streamed(self):
        approval, message, client, toolset = self.queue_approval_without_dispatch()
        self.assertEqual(TaskRecord.objects.get(id=approval.id).status, "pending")
        # Open the reconnect before recovery, then consume it without the worker's stream.
        response = self.client.get(
            f"/api/v1/sessions/continue-stream/{self.session.id}?message_id={message.id}",
            HTTP_LAST_EVENT_ID="9",
            **self.headers,
        )
        with (
            patch("personal_knowledge_base.mcp_client.load_mcp_toolset", return_value=toolset),
            patch("personal_knowledge_base.model_providers.chat_completion_raw", return_value={
                "content": "finished", "tool_calls": None,
            }),
            patch("services.chat_generation.schedule_chat_maintenance"),
        ):
            self.assertEqual(tasks.recover_incomplete_tasks()["recovered"], 1)
            self.assertEqual(tasks.recover_incomplete_tasks()["recovered"], 0)

        client.call_tool.assert_called_once()
        self.assertEqual(TaskRecord.objects.get(id=approval.id).status, "completed")
        message.refresh_from_db()
        self.assertTrue(message.is_completed)
        self.assertEqual(message.content, "finished")
        # The recovered stream also lacks the old round's IDs. Reconnect through saved state.
        replay = self.client.get(
            f"/api/v1/sessions/continue-stream/{self.session.id}?message_id={message.id}",
            HTTP_LAST_EVENT_ID="9",
            **self.headers,
        )
        recovered_frames = parse_sse(replay)
        self.assertEqual([frame_id for frame_id, _, _ in recovered_frames], [10, 11, 12])
        stream_manager.remove_stream(message.id)
        frames = parse_sse(response)
        self.assertEqual(frames, recovered_frames)
        self.assertEqual([frame_id for frame_id, _, _ in frames], [10, 11, 12])
        self.assertEqual(frames[0][2]["content"], "finished")
        self.assertEqual(frames[-1][1], "done")

    def test_recovered_rejection_completes_without_calling_the_tool(self):
        approval, message, client, toolset = self.queue_approval_without_dispatch("reject")
        with (
            patch("personal_knowledge_base.mcp_client.load_mcp_toolset", return_value=toolset),
            patch("personal_knowledge_base.model_providers.chat_completion_raw", return_value={
                "content": "rejected", "tool_calls": None,
            }),
            patch("services.chat_generation.schedule_chat_maintenance"),
        ):
            self.assertEqual(tasks.recover_incomplete_tasks()["recovered"], 1)
        client.call_tool.assert_not_called()
        self.assertEqual(TaskRecord.objects.get(id=approval.id).status, "completed")
        message.refresh_from_db()
        self.assertTrue(message.is_completed)
        self.assertEqual(message.content, "rejected")

    def test_reconnect_keeps_frame_ids_increasing_and_replayable_after_stream_replacement(self):
        approval, message, client, toolset = self.queue_approval_without_dispatch()
        old_stream = stream_manager.create_stream(message.id, self.session.id)
        old_stream.append_event("thinking", {"content": "waiting for approval"})
        old_stream.append_event("approval_required", {"approval": {"id": approval.id}})
        url = f"/api/v1/sessions/continue-stream/{self.session.id}?message_id={message.id}"
        # IDs 0–4 cover the header, thinking event, approval request and done frame.
        response = self.client.get(url, HTTP_LAST_EVENT_ID="4", **self.headers)
        with (
            patch("personal_knowledge_base.mcp_client.load_mcp_toolset", return_value=toolset),
            patch("personal_knowledge_base.model_providers.chat_completion_raw", return_value={
                "content": "finished", "tool_calls": None,
            }),
            patch("services.chat_generation.schedule_chat_maintenance"),
            patch("services.chat_generation.time.sleep", side_effect=lambda _: tasks.recover_incomplete_tasks()),
        ):
            frames = parse_sse(response)
        client.call_tool.assert_called_once()
        self.assertEqual([frame_id for frame_id, _, _ in frames], list(range(5, 5 + len(frames))))
        self.assertEqual(frames[0][2]["response_type"], "tool_call")
        self.assertEqual(frames[1][2]["response_type"], "tool_result")
        self.assertEqual(frames[-1][1], "done")
        self.assertEqual(parse_sse(self.client.get(url, HTTP_LAST_EVENT_ID="5", **self.headers)), frames[1:])

    def test_two_recovery_workers_claim_the_same_pending_resume_once(self):
        approval, message, client, toolset = self.queue_approval_without_dispatch()
        both_read_pending = threading.Barrier(2, timeout=5)
        first_worker_in_tool = threading.Event()
        second_worker_done = threading.Event()
        original_clean_payload = tasks._payload_without_worker_token

        def clean_payload(payload):
            both_read_pending.wait()
            if threading.current_thread().name == "contender_0":
                self.assertTrue(first_worker_in_tool.wait(5))
            return original_clean_payload(payload)

        def run_tool(*args, **kwargs):
            first_worker_in_tool.set()
            self.assertTrue(second_worker_done.wait(5))
            return "written", False

        def recover(*, contender=False):
            try:
                return tasks.recover_incomplete_tasks()
            finally:
                if contender:
                    second_worker_done.set()
                connections.close_all()

        client.call_tool.side_effect = run_tool
        with (
            patch.object(tasks, "_payload_without_worker_token", side_effect=clean_payload),
            patch("personal_knowledge_base.mcp_client.load_mcp_toolset", return_value=toolset),
            patch("personal_knowledge_base.model_providers.chat_completion_raw", return_value={
                "content": "finished", "tool_calls": None,
            }),
            patch("services.chat_generation.schedule_chat_maintenance"),
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="owner") as owner,
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="contender") as contender,
        ):
            first = owner.submit(recover)
            second = contender.submit(recover, contender=True)
            self.assertEqual(first.result(timeout=10)["recovered"], 1)
            self.assertEqual(second.result(timeout=10)["recovered"], 1)

        client.call_tool.assert_called_once()
        self.assertEqual(TaskRecord.objects.get(id=approval.id).status, "completed")
        message.refresh_from_db()
        self.assertTrue(message.is_completed)
        self.assertEqual(message.content, "finished")

    def test_stop_after_decision_prevents_queued_tool_execution(self):
        approval, message, client, _ = self.queue_approval_without_dispatch()
        response = self.client.post(
            f"/api/v1/sessions/{self.session.id}/stop",
            data=json.dumps({"message_id": message.id}),
            content_type="application/json",
            **self.headers,
        )
        self.assertEqual(response.status_code, 200)
        with patch("personal_knowledge_base.mcp_client.load_mcp_toolset") as load_tools:
            tasks.recover_incomplete_tasks()
        load_tools.assert_not_called()
        client.call_tool.assert_not_called()
        message.refresh_from_db()
        self.assertTrue(message.is_completed)
        self.assertEqual(message.content, "Thinking")
        self.assertEqual(TaskRecord.objects.get(id=approval.id).status, "completed")

    def test_expired_and_cancelled_approval_records_are_skipped_even_if_queued(self):
        for status in ("expired", "cancelled"):
            approval, _, _ = self.create_pending()
            AgentToolApproval.objects.filter(id=approval.id).update(status=status)
            TaskRecord.objects.create(id=approval.id, task_type=tasks.RESUME_APPROVAL_TASK)
        with patch("personal_knowledge_base.mcp_client.load_mcp_toolset") as load_tools:
            self.assertEqual(tasks.recover_incomplete_tasks()["recovered"], 2)
        load_tools.assert_not_called()

    def expire_resume_lease(self, approval_id, result=None):
        TaskRecord.objects.filter(id=approval_id).update(
            status="running",
            payload={tasks.WORKER_TOKEN_KEY: "crashed-owner"},
            result=result,
            updated_at=timezone.now() - timedelta(seconds=tasks.STALE_LEASE_SECONDS + 1),
        )

    def test_stale_lease_before_tool_execution_can_be_reclaimed(self):
        approval, message, client, toolset = self.queue_approval_without_dispatch()
        self.expire_resume_lease(approval.id)
        with (
            patch("personal_knowledge_base.mcp_client.load_mcp_toolset", return_value=toolset),
            patch("personal_knowledge_base.model_providers.chat_completion_raw", return_value={
                "content": "finished", "tool_calls": None,
            }),
            patch("services.chat_generation.schedule_chat_maintenance"),
        ):
            self.assertEqual(tasks.recover_incomplete_tasks()["stale_reset"], 1)
        client.call_tool.assert_called_once()
        message.refresh_from_db()
        self.assertTrue(message.is_completed)
        self.assertEqual(message.content, "finished")

    def test_stale_worker_that_may_have_called_the_tool_is_never_replayed(self):
        from services.chat_generation import RESUME_INTERRUPTED

        approval, message, client, _ = self.queue_approval_without_dispatch()
        self.expire_resume_lease(approval.id, tasks.RESUME_STARTED)
        with patch("personal_knowledge_base.mcp_client.load_mcp_toolset") as load_tools:
            self.assertEqual(tasks.recover_incomplete_tasks()["discarded"], 1)
            self.assertEqual(tasks.recover_incomplete_tasks()["recovered"], 0)
        load_tools.assert_not_called()
        client.call_tool.assert_not_called()
        self.assertEqual(TaskRecord.objects.get(id=approval.id).status, "failed")
        message.refresh_from_db()
        self.assertTrue(message.is_completed)
        self.assertEqual(message.content, RESUME_INTERRUPTED)

    def test_recovery_preserves_next_approval_after_result_was_committed(self):
        approval, message, client, toolset = self.queue_approval_without_dispatch()
        response = self.client.get(
            f"/api/v1/sessions/continue-stream/{self.session.id}?message_id={message.id}",
            HTTP_LAST_EVENT_ID="9",
            **self.headers,
        )
        with (
            patch("personal_knowledge_base.mcp_client.load_mcp_toolset", return_value=toolset),
            patch("personal_knowledge_base.model_providers.chat_completion_raw", return_value={
                "content": "Second approval",
                "tool_calls": [{"id": "c2", "type": "function", "function": {
                    "name": "mcp_server_write", "arguments": '{"x": 2}',
                }}],
            }),
        ):
            self.assertEqual(tasks.recover_incomplete_tasks()["recovered"], 1)
        self.expire_resume_lease(approval.id, tasks.RESUME_FINISHED)
        with patch("personal_knowledge_base.mcp_client.load_mcp_toolset") as load_tools:
            self.assertEqual(tasks.recover_incomplete_tasks()["recovered"], 0)
        load_tools.assert_not_called()
        client.call_tool.assert_called_once()
        self.assertEqual(TaskRecord.objects.get(id=approval.id).status, "completed")
        message.refresh_from_db()
        self.assertFalse(message.is_completed)
        self.assertEqual(message.content, "Second approval")
        pending = message.tool_approvals.get(status="pending")
        stream_manager.remove_stream(message.id)
        frames = parse_sse(response)
        self.assertEqual([(frame_id, event) for frame_id, event, _ in frames], [(10, "message"), (11, "done")])
        self.assertEqual(frames[0][2]["approval"]["id"], pending.id)

    def test_approval_stream_event_is_terminal(self):
        from chat.streaming import stream_message_events
        from services.chat_generation import _persist_agent_result

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
        with patch("services.chat_generation.schedule_chat_maintenance"):
            _persist_agent_result(
                ServiceContext(tenant=self.tenant, user=None),
                result,
                assistant_msg_id=message.id,
                stream=stream,
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
