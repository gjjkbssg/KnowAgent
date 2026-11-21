import csv
import json
import time
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from personal_knowledge_base.model_providers import rerank
from personal_knowledge_base.models import Chunk, KnowledgeBase, Tenant
from personal_knowledge_base.rag_eval import run_rag_evaluation
from personal_knowledge_base.search import _fts_search, _ranked_ids, _vector_search, hybrid_search, rrf_fuse

from .prepare_interview_demo import DEMO_KB_NAME


def _chunk_rows(chunk_ids, score_map):
    chunks = {
        item.id: item
        for item in Chunk.objects.filter(id__in=chunk_ids).select_related("knowledge", "knowledge_base")
    }
    rows = []
    for chunk_id in chunk_ids:
        chunk = chunks.get(chunk_id)
        if not chunk:
            continue
        rows.append(
            {
                "chunk_id": chunk.id,
                "id": chunk.id,
                "content": chunk.content,
                "score": float(score_map.get(chunk.id, 0)),
                "knowledge_id": chunk.knowledge_id,
                "knowledge_base_id": chunk.knowledge_base_id,
                "knowledge_title": chunk.knowledge.title,
                "knowledge_base_name": chunk.knowledge_base.name,
                "metadata": chunk.metadata or {},
            }
        )
    return rows


def _rank_scores(score_map, limit=5):
    return [item[0] for item in sorted(score_map.items(), key=lambda item: item[1], reverse=True)[:limit]]


def _measure(fn):
    started = time.perf_counter()
    result = fn()
    return result, round((time.perf_counter() - started) * 1000, 2)


def _metrics(expected_titles, rows):
    titles = [str(row.get("knowledge_title") or "") for row in rows]
    expected = list(dict.fromkeys(expected_titles))
    matched = [title for title in expected if title in titles]
    first_rank = next((index for index, title in enumerate(titles, 1) if title in expected), None)
    return {
        "titles": titles,
        "hit_at_5": 1.0 if matched else 0.0,
        "recall_at_5": len(matched) / max(len(expected), 1),
        "reciprocal_rank": 1.0 / first_rank if first_rank else 0.0,
    }


class Command(BaseCommand):
    help = "Run reproducible retrieval and optional RAG generation evaluation for the interview demo."

    def add_arguments(self, parser):
        parser.add_argument("--questions", default="interview_materials/evaluation_questions.json")
        parser.add_argument("--output-dir", default="interview_materials/results")
        parser.add_argument("--with-generation", action="store_true")
        parser.add_argument("--generation-limit", type=int, default=5)

    def handle(self, *args, **options):
        question_path = Path(options["questions"]).resolve()
        questions = json.loads(question_path.read_text(encoding="utf-8"))
        tenant = Tenant.objects.order_by("created_at").first()
        if not tenant:
            raise CommandError("No tenant exists")
        kb = KnowledgeBase.objects.filter(tenant=tenant, name=DEMO_KB_NAME).first()
        if not kb:
            raise CommandError("Interview demo knowledge base does not exist")

        variants = ["fts_only", "vector_only", "hybrid_raw", "hybrid_rerank", "full_pipeline"]
        details = []
        for question in questions:
            query = question["question"]
            expected = question["expected_titles"]
            fts, fts_ms = _measure(lambda: _fts_search(tenant.id, {kb.id}, query, 5))
            vector, vector_ms = _measure(lambda: _vector_search(tenant.id, {kb.id}, query, 5))
            combined = rrf_fuse([_ranked_ids(fts), _ranked_ids(vector)])

            fts_rows = _chunk_rows(_rank_scores(fts), fts)
            vector_rows = _chunk_rows(_rank_scores(vector), vector)
            raw_rows = _chunk_rows(_rank_scores(combined), combined)
            reranked_rows, rerank_ms = _measure(lambda: rerank(query, _chunk_rows(_rank_scores(combined, 20), combined), 5, tenant))
            full_rows, full_ms = _measure(lambda: hybrid_search(tenant.id, [kb.id], query, 5))

            variant_rows = {
                "fts_only": (fts_rows, fts_ms),
                "vector_only": (vector_rows, vector_ms),
                "hybrid_raw": (raw_rows, round(fts_ms + vector_ms, 2)),
                "hybrid_rerank": (reranked_rows, round(fts_ms + vector_ms + rerank_ms, 2)),
                "full_pipeline": (full_rows, full_ms),
            }
            for variant, (rows, latency_ms) in variant_rows.items():
                metric = _metrics(expected, rows)
                details.append(
                    {
                        "question_id": question["id"],
                        "category": question["category"],
                        "question": query,
                        "variant": variant,
                        "latency_ms": latency_ms,
                        **metric,
                    }
                )
            self.stdout.write(f"evaluated {question['id']}")

        summary = []
        for variant in variants:
            rows = [row for row in details if row["variant"] == variant]
            summary.append(
                {
                    "variant": variant,
                    "questions": len(rows),
                    "hit_at_5": round(sum(row["hit_at_5"] for row in rows) / len(rows), 4),
                    "recall_at_5": round(sum(row["recall_at_5"] for row in rows) / len(rows), 4),
                    "mrr": round(sum(row["reciprocal_rank"] for row in rows) / len(rows), 4),
                    "avg_latency_ms": round(sum(row["latency_ms"] for row in rows) / len(rows), 2),
                }
            )

        output_dir = Path(options["output_dir"]).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "knowledge_base": {"id": kb.id, "name": kb.name},
            "question_count": len(questions),
            "metric_definition": {
                "hit_at_5": "At least one expected source document appears in the top five results.",
                "recall_at_5": "Fraction of expected source documents represented in the top five results.",
                "mrr": "Mean reciprocal rank of the first expected source document.",
            },
            "summary": summary,
            "details": details,
        }
        (output_dir / "retrieval_evaluation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        with (output_dir / "retrieval_evaluation.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=summary[0].keys())
            writer.writeheader()
            writer.writerows(summary)

        self.stdout.write(self.style.SUCCESS(json.dumps(summary, ensure_ascii=False, indent=2)))

        if options["with_generation"]:
            generation_questions = [
                {"question": item["question"], "ground_truth": item["ground_truth"]}
                for item in questions[: max(1, options["generation_limit"])]
            ]
            generation = run_rag_evaluation(tenant, generation_questions).to_dict()
            (output_dir / "generation_evaluation.json").write_text(
                json.dumps(generation, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            self.stdout.write(self.style.SUCCESS(json.dumps(generation, ensure_ascii=False, indent=2)))
