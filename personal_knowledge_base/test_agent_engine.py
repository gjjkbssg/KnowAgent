import json
import time
from unittest.mock import Mock, patch

import requests
from django.test import TestCase, override_settings

from . import model_providers
from .agent_engine import AgentEngine
from .agent_tools import KnowledgeSearchTool, ToolRegistry, ToolResult, get_tool_registry
from .model_providers import ModelHTTPError
from .models import Chunk, Knowledge, KnowledgeBase, Session, Tenant


def tool_call(call_id, name, arguments):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}


def search_hit(chunk_id, title):
    return {"chunk_id": chunk_id, "knowledge_title": title, "content": f"evidence from {title}", "score": 0.9}


class ScriptedModel:
    """Replays a fixed list of chat_completion_raw responses and records what was advertised."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.advertised = []

    def __call__(self, tenant, messages, model_id="", tools=None, temperature=None):
        self.advertised.append([tool["function"]["name"] for tool in tools or []])
        return self.responses.pop(0)


class AgentEngineTests(TestCase):
    def setUp(self):
        self.tenant = Tenant.objects.create(name="agent", api_key="agent-key")

    def run_engine(self, responses, allowed_tools, final_answer="forced answer", session_id="session"):
        model = ScriptedModel(responses)
        engine = AgentEngine(self.tenant, session_id, agent_config={"allowed_tools": allowed_tools, "knowledge_base_ids": ["kb"]})
        with patch.object(model_providers, "chat_completion_raw", model), patch.object(
            model_providers, "chat_completion", return_value=final_answer
        ) as final_call:
            result = engine.execute("question")
        return result, model, final_call

    def test_llm_tool_call_retries_with_retry_after(self):
        engine = AgentEngine(self.tenant, "session", agent_config={"allowed_tools": ["thinking"]})
        response = {"content": "ok", "tool_calls": None}
        with (
            override_settings(AGENT_LLM_MAX_ATTEMPTS=3),
            patch.object(model_providers, "chat_completion_raw", side_effect=[ModelHTTPError(429, 0.5, "limited"), response]) as call,
            patch("personal_knowledge_base.agent_engine.time.sleep") as sleep,
        ):
            result = engine._call_llm_with_tools([{"role": "user", "content": "question"}])

        self.assertEqual(result, response)
        self.assertEqual(call.call_count, 2)
        sleep.assert_called_once_with(0.5)

    def test_llm_tool_call_does_not_retry_non_retryable_status(self):
        engine = AgentEngine(self.tenant, "session", agent_config={"allowed_tools": ["thinking"]})
        with (
            override_settings(AGENT_LLM_MAX_ATTEMPTS=3),
            patch.object(model_providers, "chat_completion_raw", side_effect=ModelHTTPError(400, None, "bad request")) as call,
            patch("personal_knowledge_base.agent_engine.time.sleep") as sleep,
            self.assertRaises(ModelHTTPError),
        ):
            engine._call_llm_with_tools([{"role": "user", "content": "question"}])

        call.assert_called_once()
        sleep.assert_not_called()

    def test_llm_tool_call_stops_after_max_attempts(self):
        engine = AgentEngine(self.tenant, "session", agent_config={"allowed_tools": ["thinking"]})
        with (
            override_settings(AGENT_LLM_MAX_ATTEMPTS=3),
            patch.object(model_providers, "chat_completion_raw", side_effect=ModelHTTPError(503, None, "unavailable")) as call,
            patch("personal_knowledge_base.agent_engine.time.sleep"),
            patch("personal_knowledge_base.retry_policy.retry_delay", return_value=0),
            self.assertRaises(ModelHTTPError),
        ):
            engine._call_llm_with_tools([{"role": "user", "content": "question"}])

        self.assertEqual(call.call_count, 3)

    def test_llm_retry_is_cancelled_during_backoff(self):
        engine = AgentEngine(self.tenant, "session", agent_config={"allowed_tools": ["thinking"]})
        engine.cancel_check = Mock(side_effect=[False, True])
        with (
            override_settings(AGENT_LLM_MAX_ATTEMPTS=3),
            patch.object(model_providers, "chat_completion_raw", side_effect=ModelHTTPError(429, 1.0, "limited")) as call,
            patch("personal_knowledge_base.agent_engine.time.sleep") as sleep,
            self.assertRaises(ModelHTTPError),
        ):
            engine._call_llm_with_tools([{"role": "user", "content": "question"}])

        call.assert_called_once()
        sleep.assert_called_once_with(0.5)

    def test_simple_llm_call_uses_shared_retry_policy(self):
        engine = AgentEngine(self.tenant, "session", agent_config={"allowed_tools": ["thinking"]})
        with (
            override_settings(AGENT_LLM_MAX_ATTEMPTS=3),
            patch.object(model_providers, "chat_completion", side_effect=[requests.Timeout(), "ok"]) as call,
            patch("personal_knowledge_base.agent_engine.time.sleep"),
            patch("personal_knowledge_base.retry_policy.retry_delay", return_value=0),
        ):
            result = engine._call_llm_simple([{"role": "user", "content": "question"}])

        self.assertEqual(result, "ok")
        self.assertEqual(call.call_count, 2)

    def test_tool_outside_whitelist_is_rejected_at_execution(self):
        result, model, _ = self.run_engine(
            [
                {"content": "", "tool_calls": [tool_call("c1", "todo_write", {"todos": [{"content": "x", "status": "pending"}]})]},
                {"content": "done", "tool_calls": None},
            ],
            allowed_tools=["thinking"],
        )

        self.assertEqual(model.advertised[0], ["thinking"])
        record = result.steps[0].tool_calls[0]
        self.assertEqual(record.result.output, "")
        self.assertIn("Tool not allowed", record.result.error)
        self.assertEqual(result.content, "done")

    def test_whitelisted_tool_still_executes(self):
        result, _, _ = self.run_engine(
            [
                {"content": "", "tool_calls": [tool_call("c1", "thinking", {"thought": "plan"})]},
                {"content": "done", "tool_calls": None},
            ],
            allowed_tools=["thinking"],
        )

        self.assertEqual(result.steps[0].tool_calls[0].result.error, "")
        self.assertEqual(result.stopped_reason, "completed")

    def test_identical_tool_round_stops_the_loop_with_a_final_answer(self):
        same_call = {"content": "", "tool_calls": [tool_call("c1", "thinking", {"thought": "again"})]}
        result, model, final_call = self.run_engine([same_call, same_call, same_call], allowed_tools=["thinking"])

        self.assertEqual(result.stopped_reason, "stuck")
        self.assertEqual(result.content, "forced answer")
        self.assertEqual(result.total_iterations, 2)
        self.assertEqual(len(model.responses), 1)
        self.assertEqual(len(result.steps), 1)
        final_call.assert_called_once()

    def test_three_failed_sequential_tool_rounds_force_final_answer(self):
        responses = [
            {"content": "", "tool_calls": [tool_call(f"c{index}", "thinking", {"thought": str(index)})]}
            for index in range(3)
        ]
        with patch.object(AgentEngine, "_run_tool", return_value=ToolResult(output="", error="failed")):
            result, model, final_call = self.run_engine(responses, allowed_tools=["thinking"])

        self.assertEqual(result.stopped_reason, "tool_failures")
        self.assertEqual(result.total_iterations, 3)
        self.assertEqual(len(result.steps), 3)
        self.assertEqual(model.responses, [])
        final_call.assert_called_once()
        self.assertIn("工具调用连续失败", final_call.call_args.args[1][-1]["content"])

    def test_three_failed_parallel_tool_rounds_force_final_answer(self):
        responses = []
        for round_number in range(3):
            calls = [
                tool_call(f"c{round_number}-{index}", "thinking", {"thought": f"{round_number}-{index}"})
                for index in range(2)
            ]
            responses.append({"content": "", "tool_calls": calls})

        with patch.object(AgentEngine, "_run_tool", return_value=ToolResult(output="", error="failed")) as run_tool:
            result, _, _ = self.run_engine(responses, allowed_tools=["thinking"])

        self.assertEqual(result.stopped_reason, "tool_failures")
        self.assertEqual(result.total_iterations, 3)
        self.assertEqual(run_tool.call_count, 6)
        self.assertTrue(all(len(step.tool_calls) == 2 for step in result.steps))

    def test_successful_tool_round_resets_failed_round_count(self):
        responses = [
            {"content": "", "tool_calls": [tool_call(f"c{index}", "thinking", {"thought": str(index)})]}
            for index in range(4)
        ]
        responses.append({"content": "done", "tool_calls": None})
        results = [
            ToolResult(output="", error="failed"),
            ToolResult(output="ok"),
            ToolResult(output="", error="failed"),
            ToolResult(output="", error="failed"),
        ]
        with patch.object(AgentEngine, "_run_tool", side_effect=results) as run_tool:
            result, _, _ = self.run_engine(responses, allowed_tools=["thinking"])

        self.assertEqual(result.stopped_reason, "completed")
        self.assertEqual(result.total_iterations, 5)
        self.assertEqual(run_tool.call_count, 4)

    def test_different_arguments_are_not_treated_as_stuck(self):
        result, _, _ = self.run_engine(
            [
                {"content": "", "tool_calls": [tool_call("c1", "thinking", {"thought": "first"})]},
                {"content": "", "tool_calls": [tool_call("c2", "thinking", {"thought": "second"})]},
                {"content": "done", "tool_calls": None},
            ],
            allowed_tools=["thinking"],
        )

        self.assertEqual(result.stopped_reason, "completed")
        self.assertEqual(len(result.steps), 3)

    def test_polling_a_background_actor_is_not_treated_as_stuck(self):
        poll = [tool_call("c1", "actor", {"action": "wait", "actor_id": "doc_retriever-1"})]
        self.assertIsNone(AgentEngine._tool_round_signature(poll))

    def test_chunks_retrieved_by_tools_become_the_answer_references(self):
        hits = [search_hit("c-1", "安装手册"), search_hit("c-2", "发布说明")]
        search = [tool_call("c1", "knowledge_search", {"query": "怎么安装"})]
        with patch("personal_knowledge_base.search.hybrid_search", return_value=hits):
            result, _, _ = self.run_engine(
                [
                    {"content": "", "tool_calls": search},
                    {"content": "", "tool_calls": [tool_call("c2", "knowledge_search", {"query": "安装步骤"})]},
                    {"content": "根据《发布说明》，先执行迁移。", "tool_calls": None},
                ],
                allowed_tools=["knowledge_search"],
            )

        # Deduplicated across calls; the document named in the answer comes first.
        self.assertEqual([ref["chunk_id"] for ref in result.references], ["c-2", "c-1"])
        self.assertEqual(result.references[0]["content"], "evidence from 发布说明")

    def test_answer_without_retrieval_has_no_references(self):
        result, _, _ = self.run_engine([{"content": "你好", "tool_calls": None}], allowed_tools=["knowledge_search"])

        self.assertEqual(result.references, [])

    def test_evidence_found_by_a_subagent_reaches_the_main_answer(self):
        session = Session.objects.create(tenant=self.tenant, title="s")
        delegate = [tool_call("m1", "actor", {"action": "run", "subagent_type": "doc_retriever", "prompt": "查安装步骤"})]
        with patch("personal_knowledge_base.search.hybrid_search", return_value=[search_hit("c-9", "安装手册")]):
            result, _, _ = self.run_engine(
                [
                    {"content": "", "tool_calls": delegate},
                    {"content": "", "tool_calls": [tool_call("s1", "knowledge_search", {"query": "安装步骤"})]},
                    {"content": "子 Agent：见安装手册", "tool_calls": None},
                    {"content": "最终答案", "tool_calls": None},
                ],
                allowed_tools=["actor"],
                session_id=session.id,
            )

        self.assertEqual(result.content, "最终答案")
        self.assertEqual([ref["chunk_id"] for ref in result.references], ["c-9"])


PUBLIC_ADDRESS = [(2, 1, 6, "", ("93.184.216.34", 443))]


class ToolTenantScopeTests(TestCase):
    def setUp(self):
        self.tenant = Tenant.objects.create(name="mine", api_key="mine-key")
        self.kb = KnowledgeBase.objects.create(tenant=self.tenant, name="kb")
        self.other_kb = KnowledgeBase.objects.create(tenant=self.tenant, name="not selected")
        other_tenant = Tenant.objects.create(name="theirs", api_key="theirs-key")
        self.own = self.add_document(self.tenant, self.kb, "mine")
        self.unselected = self.add_document(self.tenant, self.other_kb, "unselected")
        self.foreign = self.add_document(other_tenant, KnowledgeBase.objects.create(tenant=other_tenant, name="kb"), "confidential")
        self.context = {"tenant_id": self.tenant.id, "kb_ids": [self.kb.id]}

    def add_document(self, tenant, kb, title):
        knowledge = Knowledge.objects.create(tenant=tenant, knowledge_base=kb, title=title, parse_status="completed")
        Chunk.objects.create(tenant=tenant, knowledge_base=kb, knowledge=knowledge, content=f"{title} body", chunk_index=0)
        return knowledge

    def test_document_tools_only_read_documents_in_scope(self):
        registry = get_tool_registry()
        for tool in ("get_document_info", "wiki_read_source_doc"):
            allowed = registry.execute_tool(tool, {"knowledge_id": self.own.id}, self.context)
            self.assertEqual(allowed.error, "", tool)
            self.assertIn("mine", allowed.output, tool)
            for document in (self.foreign, self.unselected):
                denied = registry.execute_tool(tool, {"knowledge_id": document.id}, self.context)
                self.assertIn("Document not found", denied.error, tool)
                self.assertEqual(denied.output, "", tool)


class FakeResponse:
    def __init__(self, text="", location=None):
        self.text = text
        self.headers = {"Location": location} if location else {}
        self.is_redirect = bool(location)

    def raise_for_status(self):
        return None


class RegistryTestTool:
    def __init__(self, name, execute, timeout_seconds=None, hard_timeout=False):
        self._name = name
        self._execute = execute
        self.timeout_seconds = timeout_seconds
        self.hard_timeout = hard_timeout

    def name(self):
        return self._name

    def execute(self, args, context):
        return self._execute(args, context)


class ToolIsolationTests(TestCase):
    def test_hard_timeout_returns_promptly_for_slow_tool(self):
        registry = ToolRegistry()

        def slow_tool(args, context):
            time.sleep(1)
            return ToolResult(output="late")

        registry.register(RegistryTestTool("slow", slow_tool, timeout_seconds=0.05, hard_timeout=True))
        started = time.monotonic()
        with self.assertLogs("personal_knowledge_base.agent_tools", level="WARNING"):
            result = registry.execute_tool("slow", {}, {})

        self.assertLess(time.monotonic() - started, 0.5)
        self.assertIn("timed out", result.error)

    def test_hard_timeout_returns_fast_tool_result(self):
        registry = ToolRegistry()
        registry.register(
            RegistryTestTool("fast", lambda args, context: ToolResult(output="done"), timeout_seconds=1, hard_timeout=True)
        )

        result = registry.execute_tool("fast", {}, {})

        self.assertEqual(result.output, "done")
        self.assertEqual(result.error, "")

    def test_inline_tool_gets_private_cooperative_deadline(self):
        registry = ToolRegistry()
        received = []
        registry.register(RegistryTestTool("inline", lambda args, context: received.append(context) or ToolResult("ok")))
        caller_context = {"value": "preserved"}

        with override_settings(AGENT_TOOL_TIMEOUT_SECONDS=2):
            result = registry.execute_tool("inline", {}, caller_context)

        self.assertEqual(result.output, "ok")
        self.assertIsNot(received[0], caller_context)
        self.assertGreater(received[0]["tool_deadline"], time.monotonic())
        self.assertEqual(caller_context, {"value": "preserved"})

    def test_network_tools_are_hard_timed_and_cap_request_timeout(self):
        registry = get_tool_registry()
        web_search = registry.get("web_search")
        web_fetch = registry.get("web_fetch")
        deadline = time.monotonic() + 0.2

        self.assertTrue(web_search.hard_timeout)
        self.assertTrue(web_fetch.hard_timeout)
        with patch("personal_knowledge_base.agent_tools.requests.get", return_value=FakeResponse(text="")) as search:
            web_search.execute({"query": "query"}, {"tool_deadline": deadline})
        with patch("personal_knowledge_base.agent_tools.fetch_public_url", return_value=FakeResponse(text="ok")) as fetch:
            web_fetch.execute({"url": "https://example.com"}, {"tool_deadline": deadline})

        self.assertLessEqual(search.call_args.kwargs["timeout"], 0.2)
        self.assertLessEqual(fetch.call_args.kwargs["timeout"], 0.2)

    def test_actor_timeout_covers_maximum_wait(self):
        actor_tool = get_tool_registry().get("actor")

        self.assertFalse(actor_tool.hard_timeout)
        self.assertEqual(actor_tool.timeout_seconds, 125.0)

    def test_web_fetch_parses_html(self):
        page = FakeResponse(text="<html><style>p{}</style><script>x()</script><p>Hello <b>KnowAgent</b></p></html>")

        with patch("socket.getaddrinfo", return_value=PUBLIC_ADDRESS), patch("requests.Session.get", return_value=page):
            result = get_tool_registry().execute_tool("web_fetch", {"url": "https://example.com"}, {})

        self.assertEqual(result.error, "")
        self.assertEqual(result.output, "Hello KnowAgent")

    def test_web_fetch_connects_to_the_address_it_validated(self):
        # DNS rebinding: the hostname must not be resolved a second time when connecting.
        with patch("socket.getaddrinfo", return_value=PUBLIC_ADDRESS), patch("requests.Session.get", return_value=FakeResponse(text="ok")) as get:
            get_tool_registry().execute_tool("web_fetch", {"url": "https://example.com:8443/docs?q=1"}, {})

        self.assertEqual(get.call_args.args[0], "https://93.184.216.34:8443/docs?q=1")
        self.assertEqual(get.call_args.kwargs["headers"]["Host"], "example.com:8443")
        self.assertFalse(get.call_args.kwargs["allow_redirects"])

    def test_web_fetch_sends_internationalized_hostnames_in_ascii_form(self):
        with patch("socket.getaddrinfo", return_value=PUBLIC_ADDRESS) as resolve, patch("requests.Session.get", return_value=FakeResponse(text="ok")) as get:
            result = get_tool_registry().execute_tool("web_fetch", {"url": "http://例子.中国/路径"}, {})

        self.assertEqual(result.error, "")
        self.assertEqual(resolve.call_args.args[0], "xn--fsqu00a.xn--fiqs8s")
        self.assertEqual(get.call_args.kwargs["headers"]["Host"], "xn--fsqu00a.xn--fiqs8s")

    def test_web_fetch_keeps_tls_hostname_when_going_through_a_proxy(self):
        from personal_knowledge_base.agent_tools import _PinnedHostAdapter

        adapter = _PinnedHostAdapter("example.com")
        for manager in (adapter.poolmanager, adapter.proxy_manager_for("http://proxy.internal:3128")):
            self.assertEqual(manager.connection_pool_kw["server_hostname"], "example.com")
            self.assertEqual(manager.connection_pool_kw["assert_hostname"], "example.com")

    def test_web_fetch_refuses_plain_http_through_a_proxy(self):
        # A forward proxy resolves the hostname itself, so the address validated here could not be enforced.
        with patch.dict("os.environ", {"HTTP_PROXY": "http://proxy.internal:3128", "NO_PROXY": ""}), patch("socket.getaddrinfo", return_value=PUBLIC_ADDRESS), patch("requests.Session.get") as get:
            result = get_tool_registry().execute_tool("web_fetch", {"url": "http://example.com/docs"}, {})

        self.assertIn("not allowed", result.error)
        get.assert_not_called()

    def test_web_fetch_refuses_internal_addresses(self):
        urls = ["http://127.0.0.1:8000/health", "http://169.254.169.254/latest/meta-data/", "http://10.0.0.5/", "file:///etc/passwd"]
        with patch("requests.Session.get") as get:
            for url in urls:
                result = get_tool_registry().execute_tool("web_fetch", {"url": url}, {})
                self.assertIn("not allowed", result.error, url)
        get.assert_not_called()

    def test_web_fetch_refuses_redirect_to_internal_address(self):
        def resolve(host, *args, **kwargs):
            return PUBLIC_ADDRESS if host == "example.com" else [(2, 1, 6, "", ("127.0.0.1", 80))]

        redirect = FakeResponse(location="http://localhost/admin")
        with patch("socket.getaddrinfo", side_effect=resolve), patch("requests.Session.get", return_value=redirect) as get:
            result = get_tool_registry().execute_tool("web_fetch", {"url": "https://example.com"}, {})

        self.assertIn("not allowed", result.error)
        self.assertEqual(get.call_count, 1)

    def test_seen_chunks_are_scoped_to_one_agent_execution(self):
        refs = [{"chunk_id": "chunk-1", "knowledge_title": "doc", "content": "evidence text", "score": 0.9}]
        first_agent, second_agent = {"tenant_id": 1, "kb_ids": ["kb"]}, {"tenant_id": 1, "kb_ids": ["kb"]}
        tool = KnowledgeSearchTool()
        with patch("personal_knowledge_base.search.hybrid_search", return_value=refs):
            first = tool.execute({"query": "q"}, first_agent)
            repeat = tool.execute({"query": "q"}, first_agent)
            other = tool.execute({"query": "q"}, second_agent)

        self.assertIn("evidence text", first.output)
        self.assertIn("content omitted", repeat.output)
        self.assertIn("evidence text", other.output)
