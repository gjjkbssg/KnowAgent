from unittest.mock import patch

from django.test import TestCase

from personal_knowledge_base import model_providers
from personal_knowledge_base.models import KnowledgeBase, Message, Session, Tenant
from personal_knowledge_base.query_understand import understand_query
from personal_knowledge_base.rag_pipeline import SYSTEM_PROMPT_DEFAULT, _build_context_with_memory, run_rag_pipeline


class QueryRewriteHistoryTests(TestCase):
    def setUp(self):
        self.tenant = Tenant.objects.create(name="rewrite")
        self.kb = KnowledgeBase.objects.create(tenant=self.tenant, name="kb")
        self.session = Session.objects.create(tenant=self.tenant)

    def add_turn(self, request_id, question, answer, **assistant_fields):
        Message.objects.create(session=self.session, request_id=request_id, role="user", content=question, is_completed=True)
        Message.objects.create(
            session=self.session, request_id=request_id, role="assistant", content=answer, is_completed=True, **assistant_fields
        )

    def run_pipeline(self, query, rewrite):
        understanding = {"rewrite_query": rewrite, "intent": "kb_search", "image_description": ""}
        with patch("personal_knowledge_base.query_understand.understand_query", return_value=understanding) as understand, patch(
            "personal_knowledge_base.search.hybrid_search", return_value=[]
        ) as search:
            run_rag_pipeline(self.tenant, query, [self.kb.id], session=self.session)
        return understand, search

    def test_short_follow_up_is_rewritten_with_the_previous_turn(self):
        self.add_turn("r1", "XH-100 出现 E42 是什么意思？", "E42 表示 Wi-Fi 连接失败。")
        Message.objects.create(session=self.session, request_id="r2", role="user", content="它的排查顺序呢？", is_completed=True)

        understand, search = self.run_pipeline("它的排查顺序呢？", "XH-100 出现 E42 的排查顺序")

        history = understand.call_args.args[2]
        self.assertEqual(
            history,
            [
                {"role": "user", "content": "XH-100 出现 E42 是什么意思？"},
                {"role": "assistant", "content": "E42 表示 Wi-Fi 连接失败。"},
            ],
        )
        self.assertEqual(search.call_args.args[2], "XH-100 出现 E42 的排查顺序")

    def test_short_question_without_history_skips_the_rewrite_call(self):
        understand, search = self.run_pipeline("它的排查顺序呢？", "unused")

        understand.assert_not_called()
        self.assertEqual(search.call_args.args[2], "它的排查顺序呢？")

    def test_self_contained_short_question_skips_the_rewrite_call_even_with_history(self):
        self.add_turn("r1", "XH-100 出现 E42 是什么意思？", "E42 表示 Wi-Fi 连接失败。")

        understand, _ = self.run_pipeline("E17 怎么处理？", "unused")

        understand.assert_not_called()

    def test_subagent_messages_are_not_part_of_the_rewrite_history(self):
        self.add_turn("r1", "会议室有哪些设备？", "有 XH-100 网关和 E17 传感器。")
        self.add_turn("actor-doc_retriever-1", "内部检索指令", "内部检索结果", visible_to_user=False)

        understand, _ = self.run_pipeline("它们之间是什么关系呢？", "XH-100 和 E17 的关系")

        contents = [item["content"] for item in understand.call_args.args[2]]
        self.assertEqual(contents, ["会议室有哪些设备？", "有 XH-100 网关和 E17 传感器。"])

    def test_history_is_given_to_the_model_in_the_rewrite_prompt(self):
        history = [
            {"role": "user", "content": "XH-100 出现 E42 是什么意思？"},
            {"role": "assistant", "content": "E42 表示 Wi-Fi 连接失败。"},
        ]
        reply = '{"rewrite_query": "XH-100 出现 E42 的排查顺序", "intent": "kb_search", "image_description": ""}'
        with patch.object(model_providers, "role_completion", return_value=reply) as completion:
            result = understand_query(self.tenant, "它的排查顺序呢？", history)

        prompt = completion.call_args.args[1]
        self.assertIn("用户：XH-100 出现 E42 是什么意思？", prompt)
        self.assertIn("助手：E42 表示 Wi-Fi 连接失败。", prompt)
        self.assertEqual(result["rewrite_query"], "XH-100 出现 E42 的排查顺序")

    def test_rag_context_marks_superseded_titles_and_prompt_explains_policy(self):
        refs = [{"knowledge_title": "旧版说明", "lifecycle_status": "superseded", "content": "旧版内容"}]
        with patch("chat.views._build_document_header", return_value=""):
            context = _build_context_with_memory(refs, "", "")

        self.assertIn("旧版说明（已废止）", context)
        self.assertIn("标注“已废止”的资料只在用户问历史版本时作为依据", SYSTEM_PROMPT_DEFAULT)

    def test_prompt_has_no_history_section_for_a_first_question(self):
        reply = '{"rewrite_query": "q", "intent": "kb_search", "image_description": ""}'
        with patch.object(model_providers, "role_completion", return_value=reply) as completion:
            understand_query(self.tenant, "智能会议室的节能规则是如何根据空气质量自动调整的")

        self.assertNotIn("对话历史", completion.call_args.args[1])
