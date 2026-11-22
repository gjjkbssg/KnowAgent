import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

from django.test import Client, TestCase

from personal_knowledge_base.models import GenericResource, Message, Session, Tenant


class MCPChatTests(TestCase):
    def setUp(self):
        self.client = Client()
        response = self.client.post("/api/v1/auth/auto-setup", content_type="application/json")
        self.assertIn(response.status_code, (200, 201))
        self.headers = {"HTTP_AUTHORIZATION": f"Bearer {response.json()['data']['token']}"}
        self.tenant = Tenant.objects.first()
        self.session = Session.objects.create(tenant=self.tenant, title="MCP chat")

    @staticmethod
    def result():
        return SimpleNamespace(content="Agent answer", references=[], steps=[], duration_ms=1)

    @staticmethod
    def approval_result():
        result = MCPChatTests.result()
        result.stopped_reason = "awaiting_approval"
        result.pending_approval = [{"id": "c1", "name": "tool", "arguments": {}, "service_name": "server", "remote_tool_name": "write"}]
        result.checkpoint = {"query": "question", "messages": [], "steps": [], "pending_round": {}}
        return result

    def post_agent_chat(self, data):
        return self.client.post(
            f"/api/v1/agent-chat/{self.session.id}",
            data=json.dumps(data),
            content_type="application/json",
            **self.headers,
        )

    def patch_chat(self):
        return (
            patch("chat.views.build_agent_history_with_snapshot", return_value=[]),
            patch("chat.views._build_agent_prefetch_context", return_value=("", "")),
            patch("chat.views.build_agent_context_if_needed", return_value="base context"),
            patch("chat.views.schedule_chat_maintenance"),
            patch("chat.views.AgentEngine"),
        )

    def test_cross_tenant_service_id_is_not_loaded(self):
        other_tenant = Tenant.objects.create(name="other chat tenant", api_key="other-chat-tenant-key")
        service = GenericResource.objects.create(
            tenant=other_tenant,
            resource_type="mcp_services",
            name="foreign",
            data={"url": "http://127.0.0.1:8000/mcp", "enabled": True},
        )
        history, prefetch, context, maintenance, engine = self.patch_chat()
        with (
            history,
            prefetch,
            context,
            maintenance,
            engine as engine_class,
            patch("personal_knowledge_base.mcp_client.MCPClient") as mcp_client,
        ):
            engine_class.return_value.execute.return_value = self.result()
            response = self.post_agent_chat({"query": "question", "mcp_service_ids": [service.id]})

        self.assertEqual(response.status_code, 200)
        mcp_client.assert_not_called()
        self.assertEqual(engine_class.call_args.kwargs["agent_config"]["extra_tools"], [])
        self.session.refresh_from_db()
        self.assertNotIn("extra_tools", self.session.agent_config)

    def test_non_streaming_failure_adds_context_and_closes_toolset(self):
        toolset = SimpleNamespace(tools=[], errors=[{"service_name": "offline", "error": "connection refused"}], close=Mock())
        history, prefetch, context, maintenance, engine = self.patch_chat()
        with history, prefetch, context, maintenance, engine as engine_class, patch(
            "personal_knowledge_base.mcp_client.load_mcp_toolset", return_value=toolset
        ) as load_toolset:
            engine_class.return_value.execute.return_value = self.result()
            response = self.post_agent_chat({"query": "question", "mcp_service_ids": [123]})

        self.assertEqual(response.status_code, 200)
        load_toolset.assert_called_once_with(self.tenant, ["123"])
        run_context = engine_class.return_value.execute.call_args.kwargs["context_str"]
        self.assertIn("以下 MCP 服务不可用：offline（connection refused）", run_context)
        toolset.close.assert_called_once()

    def test_non_streaming_approval_is_persisted_and_returned(self):
        toolset = SimpleNamespace(tools=[], errors=[], close=Mock())
        history, prefetch, context, maintenance, engine = self.patch_chat()
        with history, prefetch, context, maintenance, engine as engine_class, patch(
            "personal_knowledge_base.mcp_client.load_mcp_toolset", return_value=toolset
        ):
            engine_class.return_value.execute.return_value = self.approval_result()
            response = self.post_agent_chat({"query": "question", "mcp_service_ids": []})

        self.assertEqual(response.status_code, 200)
        message = response.json()["data"]["message"]
        self.assertFalse(message["is_completed"])
        self.assertEqual(message["pending_tool_approval"]["tool_calls"][0]["service_name"], "server")
        toolset.close.assert_called_once()

    def test_streaming_approval_emits_terminal_approval_event(self):
        toolset = SimpleNamespace(tools=[], errors=[], close=Mock())
        history, prefetch, context, maintenance, engine = self.patch_chat()
        with (
            history,
            prefetch,
            context,
            maintenance,
            engine as engine_class,
            patch("personal_knowledge_base.mcp_client.load_mcp_toolset", return_value=toolset),
            patch("chat.views.run_database_background", side_effect=lambda callback: callback()),
        ):
            engine_class.return_value.execute.return_value = self.approval_result()
            response = self.post_agent_chat({"query": "question", "stream": True, "mcp_service_ids": []})
        body = b"".join(response.streaming_content).decode()
        self.assertIn("tool_approval_required", body)
        self.assertIn("event: done", body)
        self.assertFalse(Message.objects.filter(session=self.session, role="assistant").get().is_completed)

    def test_streaming_failure_adds_context_completes_and_closes_toolset(self):
        toolset = SimpleNamespace(tools=[], errors=[{"service_name": "offline", "error": "timeout"}], close=Mock())
        history, prefetch, context, maintenance, engine = self.patch_chat()
        with (
            history,
            prefetch,
            context,
            maintenance,
            engine as engine_class,
            patch("personal_knowledge_base.mcp_client.load_mcp_toolset", return_value=toolset),
            patch("chat.views.run_database_background", side_effect=lambda callback: callback()),
        ):
            engine_class.return_value.execute.return_value = self.result()
            response = self.post_agent_chat({"query": "question", "stream": True, "mcp_service_ids": ["offline-id"]})

        self.assertEqual(response.status_code, 200)
        run_context = engine_class.return_value.execute.call_args.kwargs["context_str"]
        self.assertIn("以下 MCP 服务不可用：offline（timeout）", run_context)
        assistant = Message.objects.filter(session=self.session, role="assistant").get()
        self.assertTrue(assistant.is_completed)
        self.assertEqual(assistant.content, "Agent answer")
        self.session.refresh_from_db()
        self.assertNotIn("extra_tools", self.session.agent_config)
        toolset.close.assert_called_once()
