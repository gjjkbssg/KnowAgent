import json
import time
from unittest.mock import Mock, patch

import requests
from django.test import TestCase, override_settings

from . import model_providers
from .agent_engine import AgentEngine, AgentResult, AgentStep, ToolCallRecord, _synthesize_from_tool_results
from .agent_tools import KnowledgeSearchTool, Tool, ToolRegistry, ToolResult, get_tool_registry
from .model_providers import ModelHTTPError
from .mcp_client import MCPTool
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


class GatedTool(Tool):
    def __init__(self, calls, name="mcp_test_action"):
        self.calls = calls
        self.tool_name = name
        self.service_name = "test service"
        self.remote_tool_name = "action"

    def name(self):
        return self.tool_name

    def description(self):
        return "gated action"

    def parameters(self):
        return {"type": "object", "properties": {}}

    def requires_approval(self):
        return True

    def execute(self, args, context):
        self.calls.append(args)
        return ToolResult(output="executed")


class AgentEngineTests(TestCase):
    def setUp(self):
        self.tenant = Tenant.objects.create(name="agent", api_key="agent-key")

    def test_tool_result_fallback_lists_unique_retrieved_titles_without_raw_output(self):
        steps = [
            AgentStep(
                iteration=1,
                tool_calls=[
                    ToolCallRecord(
                        "call-1",
                        "knowledge_search",
                        {},
                        ToolResult(
                            output="secret raw tool output",
                            data=[
                                {"knowledge_title": "t1"},
                                {"knowledge_title": "t2"},
                                {"knowledge_title": "t1"},
                                {"knowledge_title": "t3"},
                                {"knowledge_title": "t4"},
                            ],
                        ),
                    ),
                    ToolCallRecord(
                        "call-2",
                        "knowledge_search",
                        {},
                        ToolResult(
                            output="another secret output",
                            data=[{"knowledge_title": "t5"}, {"knowledge_title": "t6"}],
                        ),
                    ),
                ],
            )
        ]

        answer = _synthesize_from_tool_results(steps, "question")

        self.assertEqual(
            answer,
            "抱歉，这次没能生成完整回答。已检索到的相关资料：《t1》《t2》《t3》《t4》《t5》。可以换个问法重试。",
        )
        self.assertNotIn("secret", answer)

    def test_tool_result_fallback_without_results_keeps_generic_apology(self):
        self.assertEqual(
            _synthesize_from_tool_results([], "question"),
            "抱歉，无法生成回答。请稍后重试。",
        )

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

    def test_max_iterations_force_final_answer_without_tools(self):
        model = ScriptedModel(
            [{"content": "", "tool_calls": [tool_call("c1", "thinking", {"thought": "plan"})]}]
        )
        engine = AgentEngine(
            self.tenant,
            "session",
            agent_config={"allowed_tools": ["thinking"], "knowledge_base_ids": ["kb"], "max_rounds": 1},
        )
        forced_answer = "根据已检索的信息，设备需要重新校准。"
        nudge = "已达到最大推理轮数。请停止调用工具，基于已获得的信息直接给出最终回答；信息不足的部分请明确说明。"

        with patch.object(model_providers, "chat_completion_raw", model), patch.object(
            model_providers, "chat_completion", return_value=forced_answer
        ) as final_call:
            result = engine.execute("设备如何处理？")

        self.assertEqual(result.content, forced_answer)
        self.assertEqual(result.stopped_reason, "max_iterations")
        final_call.assert_called_once()
        self.assertEqual(final_call.call_args.args[1][-1]["content"], nudge)

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

    def test_markup_only_answer_is_retried_until_visible_text_arrives(self):
        result, model, _ = self.run_engine(
            [
                {"content": "</th> </invoke>", "tool_calls": None},
                {"content": "真实回答", "tool_calls": None},
            ],
            allowed_tools=["knowledge_search"],
        )

        self.assertEqual(result.content, "真实回答")
        self.assertEqual(result.stopped_reason, "completed")
        self.assertEqual(result.total_iterations, 2)
        self.assertEqual(model.responses, [])

    def test_normal_answer_with_html_tag_is_returned_unchanged(self):
        answer = "第一行<br>第二行"
        result, _, _ = self.run_engine([{"content": answer, "tool_calls": None}], allowed_tools=["knowledge_search"])

        self.assertEqual(result.content, answer)

    def test_approval_suspends_before_tool_execution_and_resumes_in_new_engine(self):
        calls = []
        first_tool = GatedTool(calls)
        response = {"content": "Checking", "tool_calls": [tool_call("g1", first_tool.name(), {"x": 1})]}
        final = {"content": "done", "tool_calls": None}
        engine = AgentEngine(self.tenant, "session", agent_config={"allowed_tools": [], "extra_tools": [first_tool]})
        with patch.object(model_providers, "chat_completion_raw", side_effect=[response, final]):
            suspended = engine.execute("question")
            self.assertEqual(suspended.stopped_reason, "awaiting_approval")
            self.assertEqual(calls, [])
            json.dumps(suspended.checkpoint)
            second_tool = GatedTool(calls)
            resumed_engine = AgentEngine(self.tenant, "session", agent_config={"allowed_tools": [], "extra_tools": [second_tool]})
            resumed = resumed_engine.resume(suspended.checkpoint, "approved")

        self.assertEqual(calls, [{"x": 1}])
        self.assertEqual(resumed.content, "done")
        self.assertEqual(resumed.stopped_reason, "completed")

    def test_resume_does_not_execute_same_name_tool_from_another_service(self):
        client_a = Mock()
        client_b = Mock()
        tool_a = MCPTool("service-a", "Service A", "delete", "delete", {}, {}, client_a, name="mcp_shared_tool")
        tool_b = MCPTool("service-b", "Service B", "delete", "delete", {}, {}, client_b, name="mcp_shared_tool")
        response = {"content": "", "tool_calls": [tool_call("g1", tool_a.name(), {"id": "item-1"})]}

        with patch.object(
            model_providers,
            "chat_completion_raw",
            side_effect=[response, {"content": "done", "tool_calls": None}],
        ):
            suspended = AgentEngine(self.tenant, "session", agent_config={"allowed_tools": [], "extra_tools": [tool_a]}).execute("question")
            resumed = AgentEngine(self.tenant, "session", agent_config={"allowed_tools": [], "extra_tools": [tool_b]}).resume(
                suspended.checkpoint, "approved"
            )

        client_b.call_tool.assert_not_called()
        self.assertEqual(resumed.steps[0].tool_calls[0].result.error, "tool no longer available")
        self.assertEqual(suspended.pending_approval[0]["service_id"], "service-a")
        self.assertEqual(
            suspended.checkpoint["gated_tools"]["g1"],
            {"service_id": "service-a", "remote_tool_name": "delete"},
        )

    def test_last_round_approval_forces_answer_when_resumed_at_max_iterations(self):
        expected = "根据已处理的信息直接给出最终回答。"
        response = {
            "content": "我现在去删除",
            "tool_calls": [tool_call("g1", "mcp_test_action", {"x": 1})],
        }
        for decision in ("approved", "rejected"):
            with self.subTest(decision=decision):
                calls = []
                tool = GatedTool(calls)
                config = {"allowed_tools": [], "extra_tools": [tool], "max_rounds": 1}
                with (
                    patch.object(model_providers, "chat_completion_raw", return_value=response),
                    patch.object(model_providers, "chat_completion", return_value=expected) as final_call,
                ):
                    suspended = AgentEngine(self.tenant, "session", agent_config=config).execute("question")
                    resumed = AgentEngine(
                        self.tenant,
                        "session",
                        agent_config={"allowed_tools": [], "extra_tools": [GatedTool(calls)], "max_rounds": 1},
                    ).resume(suspended.checkpoint, decision)

                self.assertEqual(suspended.stopped_reason, "awaiting_approval")
                self.assertEqual(resumed.stopped_reason, "max_iterations")
                self.assertEqual(resumed.content, expected)
                final_call.assert_called_once()
                self.assertEqual(calls, [{"x": 1}] if decision == "approved" else [])

    def test_checkpoint_serialization_preserves_full_tool_result(self):
        output = "x" * 20000
        result = AgentResult(
            content="answer",
            steps=[AgentStep(1, tool_calls=[ToolCallRecord("c1", "tool", {"a": 1}, ToolResult(output, {"ok": True}, duration_ms=12))])],
            total_iterations=1,
            duration_ms=12,
        )
        restored = AgentResult.from_dict(result.to_checkpoint_dict())

        self.assertEqual(restored.steps[0].tool_calls[0].result.output, output)
        self.assertEqual(restored.steps[0].tool_calls[0].result.data, {"ok": True})
        self.assertEqual(restored.steps[0].tool_calls[0].result.duration_ms, 12)

    def test_rejected_call_is_recorded_without_execution(self):
        calls = []
        gated = GatedTool(calls)
        response = {"content": "", "tool_calls": [tool_call("g1", gated.name(), {})]}
        engine = AgentEngine(self.tenant, "session", agent_config={"allowed_tools": [], "extra_tools": [gated]})
        with patch.object(model_providers, "chat_completion_raw", side_effect=[response, {"content": "done", "tool_calls": None}]):
            suspended = engine.execute("question")
            result = AgentEngine(self.tenant, "session", agent_config={"allowed_tools": [], "extra_tools": [GatedTool(calls)]}).resume(
                suspended.checkpoint, "rejected"
            )

        self.assertEqual(calls, [])
        self.assertEqual(result.steps[0].tool_calls[0].result.error, "User rejected this tool call; it was not executed.")

    def test_mixed_gated_and_builtin_calls_are_all_suspended_then_run(self):
        executed = []
        gated = GatedTool(executed)
        response = {
            "content": "",
            "tool_calls": [
                tool_call("g1", gated.name(), {"x": 1}),
                tool_call("b1", "thinking", {"thought": "plan"}),
            ],
        }

        def run_tool(engine, name, args, context):
            executed.append(name)
            return ToolResult(output="ok")

        with patch.object(
            model_providers,
            "chat_completion_raw",
            side_effect=[response, {"content": "done", "tool_calls": None}],
        ), patch.object(AgentEngine, "_run_tool", autospec=True, side_effect=run_tool):
            first = AgentEngine(self.tenant, "session", agent_config={"allowed_tools": ["thinking"], "extra_tools": [gated]}).execute("question")
            self.assertEqual(executed, [])
            final = AgentEngine(
                self.tenant, "session",
                agent_config={"allowed_tools": ["thinking"], "extra_tools": [GatedTool(executed)]},
            ).resume(first.checkpoint, "approved")

        self.assertEqual(set(executed), {gated.name(), "thinking"})
        self.assertEqual(final.stopped_reason, "completed")

    def test_checkpoint_preserves_failed_round_counter_across_rejection(self):
        calls = []
        responses = [
            {"content": "", "tool_calls": [tool_call("b1", "thinking", {"thought": "fail"})]},
            {"content": "", "tool_calls": [tool_call("g1", "mcp_test_action", {"x": 1})]},
            {"content": "", "tool_calls": [tool_call("g2", "mcp_test_action", {"x": 2})]},
            {"content": "done", "tool_calls": None},
        ]

        def run_tool(engine, name, args, context):
            if name == "thinking":
                return ToolResult(output="", error="failed")
            calls.append(args)
            return ToolResult(output="ok")

        with patch.object(model_providers, "chat_completion_raw", side_effect=responses), patch.object(
            AgentEngine, "_run_tool", autospec=True, side_effect=run_tool
        ):
            first = AgentEngine(self.tenant, "session", agent_config={"allowed_tools": ["thinking"], "extra_tools": [GatedTool(calls)]}).execute("question")
            second = AgentEngine(self.tenant, "session", agent_config={"allowed_tools": ["thinking"], "extra_tools": [GatedTool(calls)]}).resume(first.checkpoint, "reject")

        self.assertEqual(first.checkpoint["consecutive_failed_tool_rounds"], 1)
        self.assertEqual(second.stopped_reason, "awaiting_approval")
        self.assertEqual(second.checkpoint["consecutive_failed_tool_rounds"], 1)
        self.assertEqual(calls, [])

    def test_approval_can_suspend_again_on_a_later_round(self):
        calls = []
        response1 = {"content": "", "tool_calls": [tool_call("g1", "mcp_test_action", {"x": 1})]}
        response2 = {"content": "", "tool_calls": [tool_call("g2", "mcp_test_action", {"x": 2})]}
        final = {"content": "done", "tool_calls": None}
        with patch.object(model_providers, "chat_completion_raw", side_effect=[response1, response2, final]):
            first_engine = AgentEngine(self.tenant, "session", agent_config={"allowed_tools": [], "extra_tools": [GatedTool(calls)]})
            first = first_engine.execute("question")
            second_engine = AgentEngine(self.tenant, "session", agent_config={"allowed_tools": [], "extra_tools": [GatedTool(calls)]})
            second = second_engine.resume(first.checkpoint, "approved")
            third_engine = AgentEngine(self.tenant, "session", agent_config={"allowed_tools": [], "extra_tools": [GatedTool(calls)]})
            third = third_engine.resume(second.checkpoint, "approved")

        self.assertEqual(second.stopped_reason, "awaiting_approval")
        self.assertEqual(third.content, "done")
        self.assertEqual(len(calls), 2)

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

    def test_knowledge_search_marks_superseded_results(self):
        refs = [{
            "chunk_id": "old-chunk",
            "knowledge_title": "Old guide",
            "lifecycle_status": "superseded",
            "content": "old evidence",
            "score": 0.9,
        }]

        with patch("personal_knowledge_base.search.hybrid_search", return_value=refs):
            result = KnowledgeSearchTool().execute({"query": "old"}, {"tenant_id": 1, "kb_ids": ["kb"]})

        self.assertIn("Old guide（已废止）", result.output)
        self.assertIn("标注“已废止”的资料只在用户问历史版本时作为依据", KnowledgeSearchTool().description())

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

    def test_execute_exposes_uncapped_retrieved_chunks_without_changing_references(self):
        tenant = Tenant.objects.create(name="Agent Engine retrieval test", api_key="agent-engine-retrieval-key")
        engine = AgentEngine(tenant, "session")
        refs = [
            {"chunk_id": f"chunk-{index}", "knowledge_title": f"doc-{index}", "content": f"evidence {index}"}
            for index in range(12)
        ]

        def execute(*args, **kwargs):
            engine._references.add(refs)
            return AgentResult("answer", [], 1, 0)

        with patch.object(engine, "_execute", side_effect=execute):
            result = engine.execute("question")

        self.assertEqual(len(result.references), 10)
        self.assertEqual(result.retrieved, refs)
        self.assertNotIn("retrieved", result.to_dict())
