import csv
import json
import time
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from personal_knowledge_base.eval_metrics import generation_metrics, retrieval_metrics
from personal_knowledge_base.interview_eval_metrics import (
    DEFAULT_KS,
    gold_hits,
    gold_size,
    ranking_metrics,
    summarize,
)
from personal_knowledge_base.model_providers import rerank
from personal_knowledge_base.models import Chunk, KnowledgeBase, Tenant
from personal_knowledge_base.rag_eval import run_rag_evaluation
from personal_knowledge_base.search import (
    _fts_search,
    _ranked_ids,
    _vector_search,
    apply_rerank_threshold,
    apply_version_policy,
    hybrid_search,
    rrf_channel_weights,
    rrf_fuse,
)

from .prepare_interview_demo import DEMO_KB_NAME

TOP_K = max(DEFAULT_KS)
WEKNORA_RETRIEVAL_KEYS = ("precision", "recall", "ndcg3", "ndcg10", "mrr", "map")
WEKNORA_GENERATION_KEYS = ("bleu1", "bleu2", "bleu4", "rouge1", "rouge2", "rougel")


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
                "knowledge_source": chunk.knowledge.source,
                "knowledge_base_name": chunk.knowledge_base.name,
                "lifecycle_status": ((chunk.knowledge.metadata or {}).get("lifecycle") or {}).get("status", "current"),
                "metadata": chunk.metadata or {},
            }
        )
    return rows


def _rank_scores(score_map, limit=TOP_K):
    return [item[0] for item in sorted(score_map.items(), key=lambda item: item[1], reverse=True)[:limit]]


def _measure(fn):
    started = time.perf_counter()
    result = fn()
    return result, round((time.perf_counter() - started) * 1000, 2)


def _weknora_gold_items(question):
    items = [
        *(f"source:{source}" for source in question.get("expected_sources") or []),
        *(f"title:{title}" for title in question.get("expected_titles") or []),
        *(f"evidence:{evidence}" for evidence in question.get("evidence") or []),
    ]
    return list(dict.fromkeys(items))


def _weknora_retrieval_ids(question, rows):
    gold_items = _weknora_gold_items(question)
    ids = []
    for rank, row in enumerate(rows[:TOP_K]):
        hits = gold_hits(row, question)
        matched = next(
            (
                index
                for index, item in enumerate(gold_items)
                if item in hits
            ),
            None,
        )
        ids.append(matched if matched is not None else -(rank + 1))
    return ids


def _metrics(question, rows):
    for row in rows:
        # hybrid_search 的结果里没有来源 ID，从 Chunk 上补
        if "knowledge_source" not in row:
            chunk = Chunk.objects.filter(id=row.get("chunk_id") or row.get("id")).select_related("knowledge").first()
            row["knowledge_source"] = chunk.knowledge.source if chunk else ""
    hits_by_rank = [gold_hits(row, question) for row in rows[:TOP_K]]
    gold_items = _weknora_gold_items(question)
    weknora_metrics = retrieval_metrics(
        [list(range(len(gold_items)))],
        _weknora_retrieval_ids(question, rows),
    )
    return {
        "titles": [str(row.get("knowledge_title") or "") for row in rows],
        "result_count": len(rows),
        "scores": [round(row["score"], 4) for row in rows[:TOP_K]],
        "gold_flags": [bool(hits) for hits in hits_by_rank],
        **{f"weknora_{key}": value for key, value in weknora_metrics.items()},
        **ranking_metrics(hits_by_rank, gold_size(question)),
    }


def _summarize(rows):
    summary = summarize(rows)
    count = len(rows)
    for key in WEKNORA_RETRIEVAL_KEYS:
        summary[f"weknora_{key}"] = round(
            sum(row[f"weknora_{key}"] for row in rows) / count,
            4,
        ) if count else 0.0
    return summary


class Command(BaseCommand):
    help = "Run reproducible retrieval and optional RAG generation evaluation for the interview demo."

    def add_arguments(self, parser):
        parser.add_argument("--questions", default="interview_materials/evaluation_questions.json")
        parser.add_argument("--output-dir", default="interview_materials/results")
        parser.add_argument("--kb-name", default=DEMO_KB_NAME)
        parser.add_argument(
            "--noise-kb-name",
            action="append",
            default=[],
            help="Extra knowledge bases searched together with --kb-name, as distractors.",
        )
        parser.add_argument("--output-name", default="retrieval_evaluation")
        parser.add_argument("--with-generation", action="store_true")
        parser.add_argument("--generation-limit", type=int, default=5)

    def handle(self, *args, **options):
        question_path = Path(options["questions"]).resolve()
        questions = json.loads(question_path.read_text(encoding="utf-8"))
        tenant = Tenant.objects.order_by("created_at").first()
        if not tenant:
            raise CommandError("No tenant exists")
        kb = KnowledgeBase.objects.filter(tenant=tenant, name=options["kb_name"]).first()
        if not kb:
            raise CommandError(f"Knowledge base does not exist: {options['kb_name']}")
        kb_ids = [kb.id]
        for name in options["noise_kb_name"]:
            noise = KnowledgeBase.objects.filter(tenant=tenant, name=name).first()
            if not noise:
                raise CommandError(f"Knowledge base does not exist: {name}")
            kb_ids.append(noise.id)

        variants = [
            "fts_only",
            "vector_only",
            "hybrid_raw",
            "hybrid_rerank",
            "hybrid_rerank_threshold",
            "hybrid_rerank_threshold_version",
            "full_pipeline",
        ]
        details = []
        for question in questions:
            query = question["question"]
            fts, fts_ms = _measure(lambda: _fts_search(tenant.id, set(kb_ids), query, TOP_K))
            vector, vector_ms = _measure(lambda: _vector_search(tenant.id, set(kb_ids), query, TOP_K))
            weights = rrf_channel_weights()
            combined = rrf_fuse(
                [_ranked_ids(fts), _ranked_ids(vector)],
                weights=[weights["keyword"], weights["vector"]],
            )

            fts_rows = _chunk_rows(_rank_scores(fts), fts)
            vector_rows = _chunk_rows(_rank_scores(vector), vector)
            raw_rows = _chunk_rows(_rank_scores(combined), combined)
            reranked_rows, rerank_ms = _measure(lambda: rerank(query, _chunk_rows(_rank_scores(combined, 20), combined), TOP_K, tenant))
            threshold_rows = apply_rerank_threshold(
                reranked_rows,
                settings.RETRIEVAL_RERANK_THRESHOLD,
                settings.RETRIEVAL_RERANK_FALLBACK_MIN_SCORE,
            )
            rerank_latency = round(fts_ms + vector_ms + rerank_ms, 2)
            full_rows, full_ms = _measure(lambda: hybrid_search(tenant.id, kb_ids, query, TOP_K))

            variant_rows = {
                "fts_only": (fts_rows, fts_ms),
                "vector_only": (vector_rows, vector_ms),
                "hybrid_raw": (raw_rows, round(fts_ms + vector_ms, 2)),
                "hybrid_rerank": (reranked_rows, rerank_latency),
                "hybrid_rerank_threshold": (threshold_rows, rerank_latency),
                "hybrid_rerank_threshold_version": (apply_version_policy(threshold_rows, query), rerank_latency),
                "full_pipeline": (full_rows, full_ms),
            }
            for variant, (rows, latency_ms) in variant_rows.items():
                metric = _metrics(question, rows)
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

        summary = [
            {"variant": variant, **_summarize([row for row in details if row["variant"] == variant])}
            for variant in variants
        ]
        categories = list(dict.fromkeys(row["category"] for row in details))
        by_category = [
            {
                "category": category,
                "variant": variant,
                **_summarize([row for row in details if row["variant"] == variant and row["category"] == category]),
            }
            for category in categories
            for variant in variants
        ]

        output_dir = Path(options["output_dir"]).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "knowledge_base": {"id": kb.id, "name": kb.name},
            "question_count": len(questions),
            "chunk_count": Chunk.objects.filter(knowledge_base_id__in=kb_ids).count(),
            "noise_knowledge_bases": options["noise_kb_name"],
            "rrf_weights": rrf_channel_weights(),
            "rerank_threshold": {
                "threshold": settings.RETRIEVAL_RERANK_THRESHOLD,
                "fallback_min_score": settings.RETRIEVAL_RERANK_FALLBACK_MIN_SCORE,
            },
            "metric_definition": {
                "hit_at_k": "At least one gold item appears in the top k results.",
                "recall_at_k": "Fraction of distinct gold items found in the top k results.",
                "ndcg_at_k": "Binary-gain nDCG over distinct gold items in the top k results.",
                "mrr": "Mean reciprocal rank of the first result that matches a gold item (top 10).",
                "gold": "expected_sources (passage id), expected_titles (document title) or evidence (verbatim text).",
                "weknora_retrieval": "WeKnora metrics map ranked chunks to the first matching evidence item; unmatched chunks use unique negative ids.",
            },
            "summary": summary,
            "by_category": by_category,
            "details": details,
        }
        (output_dir / f"{options['output_name']}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        with (output_dir / f"{options['output_name']}.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=summary[0].keys())
            writer.writeheader()
            writer.writerows(summary)

        self.stdout.write(self.style.SUCCESS(json.dumps(summary, ensure_ascii=False, indent=2)))

        if options["with_generation"]:
            generation_questions = [
                {
                    "question": item["question"],
                    "ground_truth": item.get("ground_truth", ""),
                    "evidence": item.get("evidence") or [],
                }
                for item in questions[: max(1, options["generation_limit"])]
            ]
            generation_result = run_rag_evaluation(tenant, generation_questions)
            generation = generation_result.to_dict()
            generation_scores = []
            for index, (question, detail) in enumerate(zip(generation_questions, generation_result.details)):
                metrics = generation_metrics(detail.answer, "。".join(question["evidence"]))
                generation_scores.append(metrics)
                generation["details"][index]["weknora_generation_metrics"] = metrics
            generation["weknora_generation_summary"] = {
                key: round(sum(row[key] for row in generation_scores) / len(generation_scores), 4)
                if generation_scores
                else 0.0
                for key in WEKNORA_GENERATION_KEYS
            }
            (output_dir / "generation_evaluation.json").write_text(
                json.dumps(generation, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            self.stdout.write(self.style.SUCCESS(json.dumps(generation, ensure_ascii=False, indent=2)))
