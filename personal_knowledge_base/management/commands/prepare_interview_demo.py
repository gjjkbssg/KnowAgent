import hashlib
from pathlib import Path

from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.core.management.base import BaseCommand, CommandError

from knowledge.views import delete_knowledge_content
from personal_knowledge_base.document_processing import detect_file_type, process_knowledge
from personal_knowledge_base.graph_rag import DEFAULT_EXTRACT_CONFIG
from personal_knowledge_base.models import Chunk, Knowledge, KnowledgeBase, Tenant, User, WikiPage


DEMO_KB_NAME = "[面试演示] 智能会议室 Agent"


class Command(BaseCommand):
    help = "Create and synchronously process the Xiaomi interview demonstration knowledge base."

    def add_arguments(self, parser):
        parser.add_argument("--rebuild", action="store_true", help="Replace only the existing interview demo knowledge base.")
        parser.add_argument(
            "--corpus-dir",
            default="interview_materials/demo_corpus",
            help="Directory containing the synthetic demo corpus.",
        )

    def handle(self, *args, **options):
        tenant = Tenant.objects.order_by("created_at").first()
        if not tenant:
            raise CommandError("No tenant exists. Run auth auto-setup first.")
        user = User.objects.filter(tenant=tenant).order_by("created_at").first()
        corpus_dir = Path(options["corpus_dir"]).resolve()
        if not corpus_dir.is_dir():
            raise CommandError(f"Corpus directory does not exist: {corpus_dir}")

        existing = KnowledgeBase.objects.filter(tenant=tenant, name=DEMO_KB_NAME).first()
        if existing and options["rebuild"]:
            for item in Knowledge.objects.filter(knowledge_base=existing):
                delete_knowledge_content(item)
                if item.file_path and default_storage.exists(item.file_path):
                    default_storage.delete(item.file_path)
            existing.delete()
            existing = None

        kb = existing or KnowledgeBase.objects.create(
            tenant=tenant,
            name=DEMO_KB_NAME,
            description="小米训练营面试用合成智能会议室语料，展示混合检索、Rerank、Wiki、GraphRAG 和多模态解析。",
            type="document",
            chunking_config={"chunk_size": 420, "chunk_overlap": 80},
            image_processing_config={"enable_multimodal": True, "model_id": ""},
            embedding_model_id="",
            summary_model_id="",
            storage_provider_config={"provider": "local"},
            vlm_config={"enabled": True},
            extract_config={**DEFAULT_EXTRACT_CONFIG, "enabled": True},
            wiki_config={"enabled": True},
            indexing_strategy={
                "vector_enabled": True,
                "keyword_enabled": True,
                "wiki_enabled": True,
                "graph_enabled": True,
            },
            is_pinned=True,
            creator_id=user.id if user else "",
        )

        paths = sorted([*corpus_dir.glob("*.md"), *corpus_dir.glob("*.svg")])
        if not paths:
            raise CommandError(f"No demo files found in {corpus_dir}")

        for path in paths:
            data = path.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            item = Knowledge.objects.filter(
                knowledge_base=kb,
                file_hash=digest,
                file_name=path.name,
                deleted_at__isnull=True,
            ).first()
            if item and item.parse_status == "completed":
                self.stdout.write(f"skip {path.name}: already completed")
                continue
            if item:
                delete_knowledge_content(item)
                item.delete()

            storage_path = default_storage.save(
                f"tenant-{tenant.id}/{kb.id}/interview-{path.name}",
                ContentFile(data),
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
                metadata={"process_config": {"graph_enabled": True}},
            )
            self.stdout.write(f"processing {path.name} ...")
            process_knowledge(item.id)
            item.refresh_from_db()
            if item.parse_status != "completed":
                raise CommandError(f"Processing failed for {path.name}: {item.error_message}")
            graph = (item.metadata or {}).get("graph") or {}
            wiki = (item.metadata or {}).get("wiki") or {}
            self.stdout.write(
                self.style.SUCCESS(
                    f"completed {path.name}: chunks={Chunk.objects.filter(knowledge=item).count()} "
                    f"graph_nodes={graph.get('node_count', 0)} graph_relations={graph.get('relation_count', 0)} "
                    f"wiki_pages={wiki.get('pages', 0)}"
                )
            )

        self.stdout.write(
            self.style.SUCCESS(
                f"demo ready: kb_id={kb.id} documents={Knowledge.objects.filter(knowledge_base=kb).count()} "
                f"chunks={Chunk.objects.filter(knowledge_base=kb).count()} "
                f"wiki_pages={WikiPage.objects.filter(knowledge_base=kb).count()}"
            )
        )
