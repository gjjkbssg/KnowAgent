import io
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from django.core.management import call_command
from django.test import TestCase

from .models import KnowledgeBase, ModelUsage, Session, Tenant


class AgentEvalCommandTests(TestCase):
    def setUp(self):
        self.tenant = Tenant.objects.create(name="agent-eval", api_key="agent-eval-test-key")
        self.kb = KnowledgeBase.objects.create(tenant=self.tenant, name="unit-kb")
        self.output_dir = Path(tempfile.mkdtemp())
        self.tasks_path = self.output_dir / "tasks.json"
        self.tasks_path.write_text(
            json.dumps(
                [
                    {
                        "id": "unit-1",
                        "category": "unit",
                        "question": "question",
                        "reference": "answer",
                        "gold_docs": ["gold-doc"],
                        "must": [["answer"]],
                    }
                ],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def test_command_runs_each_config_and_wires_metrics_and_scoring(self):
        refs = [
            {"chunk_id": "gold-1", "knowledge_title": "gold-doc", "content": "evidence one"},
            {"chunk_id": "other-1", "knowledge_title": "other-doc", "content": "evidence two"},
            {"chunk_id": "gold-2", "knowledge_title": "gold-doc second", "content": "evidence three"},
        ]
        engine_instances = []

        class StubAgentEngine:
            def __init__(self, tenant, session_id, agent_config):
                self.tenant = tenant
                self.session_id = session_id
                self.agent_config = agent_config
                engine_instances.append(self)

            def execute(self, question, history, context_str):
                self.execute_args = (question, history, context_str)
                self.session_was_real = Session.objects.filter(id=self.session_id).exists()
                ModelUsage.objects.create(tenant=self.tenant, total_tokens=17, request_count=2)
                content = "wrong" if self.agent_config.get("allow_actor_tool") is False else "answer"
                return SimpleNamespace(
                    content=content,
                    steps=[
                        SimpleNamespace(
                            tool_calls=[
                                SimpleNamespace(name="knowledge_search"),
                                SimpleNamespace(name="thinking"),
                                SimpleNamespace(name="knowledge_search"),
                            ]
                        )
                    ],
                    total_iterations=2,
                    stopped_reason="completed",
                    retrieved=refs,
                )

        def fake_chat_completion(tenant, messages, model_id=""):
            self.assertEqual([item["role"] for item in messages], ["system", "user"])
            ModelUsage.objects.create(tenant=tenant, total_tokens=11, request_count=1)
            return "answer"

        rag_context = SimpleNamespace(
            system_prompt="system prompt",
            user_prompt="user prompt",
            refs=refs,
        )
        with (
            patch(
                "personal_knowledge_base.management.commands.run_agent_eval.run_rag_pipeline",
                return_value=rag_context,
            ) as run_rag_pipeline,
            patch(
                "personal_knowledge_base.management.commands.run_agent_eval.chat_completion",
                side_effect=fake_chat_completion,
            ),
            patch(
                "personal_knowledge_base.management.commands.run_agent_eval.AgentEngine",
                StubAgentEngine,
            ),
        ):
            call_command(
                "run_agent_eval",
                tasks=str(self.tasks_path),
                kb_names="unit-kb",
                output_dir=str(self.output_dir),
                stdout=io.StringIO(),
            )

        run_rag_pipeline.assert_called_once_with(
            self.tenant,
            "question",
            [self.kb.id],
            enable_memory=False,
        )
        report = json.loads((self.output_dir / "agent_eval.json").read_text(encoding="utf-8"))
        self.assertEqual(report["task_count"], 1)
        self.assertEqual([row["config"] for row in report["results"]], ["rag", "agent", "multi_agent"])
        self.assertTrue((self.output_dir / "agent_eval.csv").is_file())
        self.assertTrue((self.output_dir / "agent_eval_summary.csv").is_file())

        by_config = {row["config"]: row for row in report["results"]}
        self.assertTrue(by_config["rag"]["correct"])
        self.assertFalse(by_config["agent"]["correct"])
        self.assertTrue(by_config["multi_agent"]["correct"])
        for row in report["results"]:
            self.assertEqual(row["retrieval_ids"], [0, -2, 0])
            self.assertAlmostEqual(row["retrieval_metrics"]["precision"], 1 / 3)
            self.assertIn("latency_ms", row)
            self.assertIn("stopped_reason", row)
            self.assertIsNone(row["error"])
        self.assertAlmostEqual(by_config["rag"]["generation_metrics"]["rouge1"], 1.0, delta=1e-8)
        self.assertEqual(by_config["agent"]["generation_metrics"]["rouge1"], 0.0)
        self.assertAlmostEqual(by_config["multi_agent"]["generation_metrics"]["rouge1"], 1.0, delta=1e-8)
        self.assertEqual(by_config["rag"]["total_tokens"], 11)
        self.assertEqual(by_config["rag"]["request_count"], 1)
        self.assertEqual(by_config["agent"]["total_tokens"], 17)
        self.assertEqual(by_config["agent"]["request_count"], 2)
        self.assertEqual(by_config["agent"]["tool_calls"], 3)
        self.assertEqual(
            by_config["agent"]["tool_name_histogram"],
            {"knowledge_search": 2, "thinking": 1},
        )
        self.assertEqual(by_config["rag"]["tool_calls"], 0)

        agent_instance = next(instance for instance in engine_instances if "allow_actor_tool" in instance.agent_config)
        self.assertNotIn("actor", agent_instance.agent_config["allowed_tools"])
        self.assertFalse(agent_instance.agent_config["allow_actor_tool"])
        self.assertEqual(agent_instance.agent_config["knowledge_base_ids"], [self.kb.id])
        self.assertEqual(agent_instance.agent_config["max_rounds"], 8)
        self.assertEqual(agent_instance.agent_config["temperature"], 0.7)
        self.assertEqual(agent_instance.execute_args, ("question", [], ""))

        multi_agent_instance = next(
            instance
            for instance in engine_instances
            if instance.agent_config.get("agent_mode") == "multi-agent"
        )
        self.assertTrue(multi_agent_instance.session_was_real)
        self.assertIn("actor", multi_agent_instance.agent_config["allowed_tools"])
        overall = {
            (row["config"], row["scope"]): row
            for row in report["summary"]
        }
        self.assertEqual(overall[("rag", "overall")]["accuracy"], 1.0)
        self.assertEqual(overall[("agent", "overall")]["accuracy"], 0.0)
        self.assertEqual(overall[("multi_agent", "overall")]["accuracy"], 1.0)
        self.assertIn("stopped_reason_counts", overall[("rag", "overall")])

    def test_case_errors_are_incorrect_and_do_not_stop_later_cases(self):
        self.tasks_path.write_text(
            json.dumps(
                [
                    {
                        "id": "failure",
                        "category": "unit",
                        "question": "fail",
                        "reference": "answer",
                        "gold_docs": ["gold-doc"],
                        "must": [["answer"]],
                    },
                    {
                        "id": "success",
                        "category": "unit",
                        "question": "succeed",
                        "reference": "answer",
                        "gold_docs": ["gold-doc"],
                        "must": [["answer"]],
                    },
                ],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        class StubAgentEngine:
            def __init__(self, tenant, session_id, agent_config):
                self.tenant = tenant

            def execute(self, question, history, context_str):
                if question == "fail":
                    raise RuntimeError("model unavailable")
                return SimpleNamespace(
                    content="answer",
                    steps=[],
                    total_iterations=1,
                    stopped_reason="completed",
                    retrieved=[],
                )

        with patch(
            "personal_knowledge_base.management.commands.run_agent_eval.AgentEngine",
            StubAgentEngine,
        ):
            call_command(
                "run_agent_eval",
                tasks=str(self.tasks_path),
                kb_names="unit-kb",
                configs="agent",
                output_dir=str(self.output_dir),
                stdout=io.StringIO(),
            )

        results = json.loads((self.output_dir / "agent_eval.json").read_text(encoding="utf-8"))["results"]
        self.assertEqual(len(results), 2)
        self.assertFalse(results[0]["correct"])
        self.assertEqual(results[0]["stopped_reason"], "error")
        self.assertIn("model unavailable", results[0]["error"])
        self.assertTrue(results[1]["correct"])
