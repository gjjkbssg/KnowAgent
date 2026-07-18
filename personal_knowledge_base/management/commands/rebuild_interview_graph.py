from django.core.management.base import BaseCommand, CommandError

from personal_knowledge_base.document_processing import process_graph
from personal_knowledge_base.graph_rag import delete_kb_graph, graph_repository
from personal_knowledge_base.models import Chunk, Knowledge, KnowledgeBase

from .prepare_interview_demo import DEMO_KB_NAME


class Command(BaseCommand):
    help = "Rebuild Neo4j entities and relationships for the interview demo knowledge base."

    def handle(self, *args, **options):
        kb = KnowledgeBase.objects.filter(name=DEMO_KB_NAME).first()
        if not kb:
            raise CommandError("Interview demo knowledge base does not exist")
        if not graph_repository.available:
            raise CommandError("Neo4j is not enabled or unavailable")

        delete_kb_graph(kb)
        totals = {"documents": 0, "nodes": 0, "relations": 0}
        for knowledge in Knowledge.objects.filter(knowledge_base=kb, parse_status="completed").order_by("created_at"):
            chunks = list(Chunk.objects.filter(knowledge=knowledge, is_enabled=True).order_by("chunk_index"))
            graphs = process_graph(knowledge, chunks)
            node_count = sum(len(graph.get("node", [])) for graph in graphs)
            relation_count = sum(len(graph.get("relation", [])) for graph in graphs)
            metadata = dict(knowledge.metadata or {})
            metadata["graph"] = {
                "enabled": bool(graphs),
                "node_count": node_count,
                "relation_count": relation_count,
            }
            metadata.pop("processing_warnings", None)
            knowledge.metadata = metadata
            knowledge.save(update_fields=["metadata", "updated_at"])
            totals["documents"] += 1
            totals["nodes"] += node_count
            totals["relations"] += relation_count
            self.stdout.write(
                self.style.SUCCESS(
                    f"{knowledge.file_name}: nodes={node_count} relations={relation_count}"
                )
            )

        self.stdout.write(self.style.SUCCESS(f"graph rebuild complete: {totals}"))
