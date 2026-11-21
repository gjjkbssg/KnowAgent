import io
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from django.core.management import call_command
from django.core.management.base import CommandError
from django.test import TestCase, override_settings

from .management.commands.prepare_public_eval import PUBLIC_KB_NAME
from .models import Chunk, Knowledge, KnowledgeBase, Tenant
from .search import index_chunk

SAMPLE = {
    "passages": {
        "p1": "国家法定节假日一共 11 天。",
        "p2": "空气净化器滤芯建议每 6 个月更换一次。",
        "p3": "路由器恢复出厂设置需要长按复位键 10 秒。",
    },
    "questions": [{"id": "q1", "category": "DuRetrieval", "question": "法定节假日多少天", "expected_sources": ["p1"]}],
}


@override_settings(LLM_USE_ENV_EMBEDDING=False)
class PreparePublicEvalTests(TestCase):
    def setUp(self):
        self.tenant = Tenant.objects.create(name="eval")
        self.out = Path(tempfile.mkdtemp())
        self.questions = self.out / "questions.json"

    def prepare(self, *args):
        with patch("personal_knowledge_base.management.commands.prepare_public_eval.sample_dataset", return_value=SAMPLE):
            call_command("prepare_public_eval", *args, questions_out=str(self.questions), stdout=io.StringIO())

    def test_one_chunk_per_passage_keyed_by_passage_id(self):
        self.prepare()

        kb = KnowledgeBase.objects.get(tenant=self.tenant, name=PUBLIC_KB_NAME)
        self.assertEqual(Chunk.objects.filter(knowledge_base=kb).count(), 3)
        self.assertEqual(set(Knowledge.objects.filter(knowledge_base=kb).values_list("source", flat=True)), {"p1", "p2", "p3"})
        self.assertFalse(Chunk.objects.filter(knowledge_base=kb, seq_id__isnull=True).exists())
        self.assertEqual(json.loads(self.questions.read_text(encoding="utf-8")), SAMPLE["questions"])

    def test_eval_command_scores_against_passage_ids(self):
        self.prepare()

        call_command(
            "run_interview_eval",
            kb_name=PUBLIC_KB_NAME,
            questions=str(self.questions),
            output_dir=str(self.out),
            output_name="public",
            stdout=io.StringIO(),
        )

        report = json.loads((self.out / "public.json").read_text(encoding="utf-8"))
        self.assertEqual((report["question_count"], report["chunk_count"]), (1, 3))
        self.assertEqual(len(report["summary"]), 5)
        self.assertEqual(len(report["by_category"]), 5)
        # 三个段落全部会进 Top-10，按段落 ID 标注的正例一定被找到
        self.assertEqual(next(row for row in report["summary"] if row["variant"] == "full_pipeline")["hit_at_10"], 1.0)

    def test_noise_knowledge_bases_are_searched_together_with_the_target(self):
        self.prepare()
        noise = KnowledgeBase.objects.create(tenant=self.tenant, name="noise")
        item = Knowledge.objects.create(tenant=self.tenant, knowledge_base=noise, title="noise", source="n1", parse_status="completed")
        index_chunk(Chunk.objects.create(tenant=self.tenant, knowledge_base=noise, knowledge=item, content="法定节假日安排说明", chunk_index=0))

        call_command(
            "run_interview_eval",
            kb_name=PUBLIC_KB_NAME,
            noise_kb_name=["noise"],
            questions=str(self.questions),
            output_dir=str(self.out),
            output_name="noisy",
            stdout=io.StringIO(),
        )

        report = json.loads((self.out / "noisy.json").read_text(encoding="utf-8"))
        self.assertEqual((report["chunk_count"], report["noise_knowledge_bases"]), (4, ["noise"]))
        titles = {title for row in report["details"] for title in row["titles"]}
        self.assertIn("noise", titles)

    def test_refuses_to_overwrite_without_rebuild(self):
        self.prepare()
        with self.assertRaises(CommandError):
            self.prepare()

        self.prepare("--rebuild")
        self.assertEqual(Chunk.objects.filter(knowledge_base__name=PUBLIC_KB_NAME).count(), 3)
