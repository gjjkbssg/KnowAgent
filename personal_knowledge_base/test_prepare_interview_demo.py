import json
import tempfile
from pathlib import Path

from django.test import TestCase

from personal_knowledge_base.management.commands.prepare_interview_demo import apply_corpus_lifecycle
from personal_knowledge_base.models import Knowledge, KnowledgeBase, Tenant


class PrepareInterviewDemoLifecycleTests(TestCase):
    def test_lifecycle_file_marks_existing_documents_and_resolves_same_kb_target(self):
        tenant = Tenant.objects.create(name="lifecycle eval")
        kb = KnowledgeBase.objects.create(tenant=tenant, name="lifecycle eval")
        old = Knowledge.objects.create(
            tenant=tenant,
            knowledge_base=kb,
            type="file",
            title="Old rules",
            source="06_old.md",
            file_name="06_old.md",
            parse_status="completed",
            metadata={"process_config": {"graph_enabled": True}},
        )
        current = Knowledge.objects.create(
            tenant=tenant,
            knowledge_base=kb,
            type="file",
            title="New rules",
            source="07_new.md",
            file_name="07_new.md",
            parse_status="completed",
        )
        with tempfile.TemporaryDirectory() as directory:
            corpus_dir = Path(directory)
            (corpus_dir / "lifecycle.json").write_text(
                json.dumps(
                    {
                        "06_old.md": {
                            "status": "superseded",
                            "superseded_by_file": "07_new.md",
                        }
                    }
                ),
                encoding="utf-8",
            )

            apply_corpus_lifecycle(corpus_dir, kb)

        old.refresh_from_db()
        self.assertEqual(
            old.metadata,
            {
                "process_config": {"graph_enabled": True},
                "lifecycle": {"status": "superseded", "superseded_by": current.id},
            },
        )
