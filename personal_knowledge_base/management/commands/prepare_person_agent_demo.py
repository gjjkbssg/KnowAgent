import hashlib
from pathlib import Path

from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.core.management.base import BaseCommand, CommandError

from services.knowledge import delete_knowledge_content
from personal_knowledge_base.document_processing import detect_file_type, process_knowledge
from personal_knowledge_base.models import Chunk, Knowledge, KnowledgeBase, Tenant, User


DEMO_KB_NAME = "[A/B演示] 人物多跳检索"


class Command(BaseCommand):
    help = "Create the synthetic person multi-hop corpus used to compare one-shot RAG with Agent retrieval."

    def add_arguments(self, parser):
        parser.add_argument("--rebuild", action="store_true")
        parser.add_argument("--corpus-dir", default="interview_materials/person_case_corpus")

    def handle(self, *args, **options):
        tenant = Tenant.objects.order_by("created_at").first()
        if not tenant:
            raise CommandError("No tenant exists")
        user = User.objects.filter(tenant=tenant).order_by("created_at").first()
        corpus_dir = Path(options["corpus_dir"]).resolve()
        paths = sorted(corpus_dir.glob("*.md"))
        if not paths:
            raise CommandError(f"No Markdown files found in {corpus_dir}")

        kb = KnowledgeBase.objects.filter(tenant=tenant, name=DEMO_KB_NAME).first()
        if kb and options["rebuild"]:
            for item in Knowledge.objects.filter(knowledge_base=kb):
                delete_knowledge_content(item)
                if item.file_path and default_storage.exists(item.file_path):
                    default_storage.delete(item.file_path)
            kb.delete()
            kb = None

        kb = kb or KnowledgeBase.objects.create(
            tenant=tenant,
            name=DEMO_KB_NAME,
            description="虚构人物多跳检索 A/B 案例：先根据业绩识别人，再按姓名查后续项目与导师。",
            type="document",
            chunking_config={"chunk_size": 360, "chunk_overlap": 50},
            indexing_strategy={"vector_enabled": True, "keyword_enabled": True, "wiki_enabled": False, "graph_enabled": False},
            storage_provider_config={"provider": "local"},
            creator_id=user.id if user else "",
            is_pinned=True,
        )

        for path in paths:
            data = path.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            item = Knowledge.objects.filter(knowledge_base=kb, file_hash=digest, file_name=path.name).first()
            if item and item.parse_status == "completed":
                self.stdout.write(f"skip {path.name}")
                continue
            if item:
                delete_knowledge_content(item)
                item.delete()
            storage_path = default_storage.save(
                f"tenant-{tenant.id}/{kb.id}/person-case-{path.name}", ContentFile(data)
            )
            item = Knowledge.objects.create(
                tenant=tenant,
                knowledge_base=kb,
                type="file",
                title=path.stem,
                source=path.name,
                parse_status="pending",
                file_name=path.name,
                file_type=detect_file_type(path.name),
                file_size=len(data),
                file_path=storage_path,
                file_hash=digest,
                storage_size=len(data),
            )
            process_knowledge(item.id)
            item.refresh_from_db()
            if item.parse_status != "completed":
                raise CommandError(f"Processing failed for {path.name}: {item.error_message}")
            self.stdout.write(f"completed {path.name}")

        self.stdout.write(
            self.style.SUCCESS(
                f"person demo ready: kb_id={kb.id} documents={Knowledge.objects.filter(knowledge_base=kb).count()} "
                f"chunks={Chunk.objects.filter(knowledge_base=kb).count()}"
            )
        )
