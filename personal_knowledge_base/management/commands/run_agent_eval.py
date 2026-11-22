import csv
import json
import statistics
import time
import uuid
from collections import Counter
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError
from django.db.models import Sum

from personal_knowledge_base.agent_engine import AgentEngine
from personal_knowledge_base.agent_eval import doc_key, score_answer
from personal_knowledge_base.agent_history import build_normal_rag_messages
from personal_knowledge_base.agent_tools import DEFAULT_ALLOWED_TOOLS
from personal_knowledge_base.eval_metrics import generation_metrics, retrieval_metrics
from personal_knowledge_base.model_providers import chat_completion
from personal_knowledge_base.models import KnowledgeBase, ModelUsage, Session, Tenant
from personal_knowledge_base.rag_pipeline import run_rag_pipeline
from services.chat_generation import apply_multi_agent_defaults, local_answer


DEFAULT_KB_NAMES = "[评测] 智能楼宇领域集,[A/B演示] 人物多跳检索,[评测] DuRetrieval 子集"
DEFAULT_CONFIGS = ("rag", "agent", "multi_agent")
RETRIEVAL_METRIC_KEYS = ("precision", "recall", "ndcg3", "ndcg10", "mrr", "map")
GENERATION_METRIC_KEYS = ("bleu1", "bleu2", "bleu4", "rouge1", "rouge2", "rougel")
ROW_CSV_FIELDS = (
    "task_id",
    "category",
    "config",
    "question",
    "answer",
    "correct",
    "retrieval_ids",
    "retrieved_chunk_ids",
    "retrieval_metrics",
    "generation_metrics",
    "latency_ms",
    "total_tokens",
    "request_count",
    "tool_calls",
    "tool_name_histogram",
    "iterations",
    "stopped_reason",
    "error",
)


def _usage_totals(tenant):
    totals = ModelUsage.objects.filter(tenant=tenant, deleted_at__isnull=True).aggregate(
        total_tokens=Sum("total_tokens"),
        request_count=Sum("request_count"),
    )
    return {
        "total_tokens": totals["total_tokens"] or 0,
        "request_count": totals["request_count"] or 0,
    }


def _retrieval_ids(chunks, gold_docs):
    ids = []
    chunk_ids = []
    for index, chunk in enumerate(chunks or []):
        chunk_ids.append(str(chunk.get("chunk_id") or chunk.get("id") or ""))
        matched_doc = doc_key(str(chunk.get("knowledge_title") or ""), gold_docs)
        ids.append(gold_docs.index(matched_doc) if matched_doc is not None else -(index + 1))
    return ids, chunk_ids


def _run_rag(tenant, question, kb_ids):
    context = run_rag_pipeline(tenant, question, kb_ids, enable_memory=False)
    messages = build_normal_rag_messages(context.system_prompt, [], context.user_prompt)
    try:
        answer = chat_completion(tenant, messages)
    except Exception:
        answer = local_answer(question, context.refs, agent=False)
    return answer or "", context.refs or [], 0, "not_applicable", Counter()


def _run_agent(tenant, question, kb_ids, config_name):
    session = None
    if config_name == "agent":
        agent_config = {
            "allowed_tools": [name for name in DEFAULT_ALLOWED_TOOLS if name != "actor"],
            "allow_actor_tool": False,
            "knowledge_base_ids": kb_ids,
            "max_rounds": 8,
            "temperature": 0.7,
        }
        session_id = str(uuid.uuid4())
    else:
        session = Session.objects.create(
            tenant=tenant,
            knowledge_base_id=kb_ids[0] if kb_ids else "",
        )
        session_id = str(session.id)
        agent_config = apply_multi_agent_defaults({}, {}, kb_ids)

    result = AgentEngine(
        tenant=tenant,
        session_id=session_id,
        agent_config=agent_config,
    ).execute(question, history=[], context_str="")

    tool_names = Counter(
        call.name
        for step in result.steps
        for call in step.tool_calls
    )
    retrieved = getattr(result, "retrieved", None)
    if retrieved is None:
        retrieved = result.references
    return (
        result.content or "",
        retrieved or [],
        int(result.total_iterations or 0),
        result.stopped_reason or "completed",
        tool_names,
    )


def _summary_row(rows, config, category=None):
    count = len(rows)
    row = {
        "scope": "category" if category is not None else "overall",
        "config": config,
        "category": category or "",
        "questions": count,
        "accuracy": round(sum(bool(item["correct"]) for item in rows) / count, 4) if count else 0.0,
    }
    for key in RETRIEVAL_METRIC_KEYS:
        values = [
            item["retrieval_metrics"][key]
            for item in rows
            if item["retrieval_metrics"] is not None
        ]
        row[f"retrieval_{key}"] = round(sum(values) / len(values), 4) if values else None
    for key in GENERATION_METRIC_KEYS:
        values = [item["generation_metrics"][key] for item in rows]
        row[f"generation_{key}"] = round(sum(values) / len(values), 4) if values else 0.0

    latencies = [item["latency_ms"] for item in rows]
    row["mean_latency_ms"] = round(statistics.mean(latencies), 2) if latencies else 0.0
    row["median_latency_ms"] = round(statistics.median(latencies), 2) if latencies else 0.0
    row["mean_tokens"] = round(statistics.mean(item["total_tokens"] for item in rows), 2) if rows else 0.0
    row["mean_request_count"] = round(statistics.mean(item["request_count"] for item in rows), 2) if rows else 0.0
    row["mean_tool_calls"] = round(statistics.mean(item["tool_calls"] for item in rows), 2) if rows else 0.0
    row["stopped_reason_counts"] = dict(sorted(Counter(item["stopped_reason"] for item in rows).items()))
    return row


def _write_rows_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            values = dict(row)
            for key in ("retrieval_ids", "retrieved_chunk_ids", "retrieval_metrics", "generation_metrics", "tool_name_histogram"):
                values[key] = json.dumps(values[key], ensure_ascii=False)
            values["error"] = values["error"] or ""
            writer.writerow(values)


def _write_summary_csv(path, rows):
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            for row in rows:
                values = dict(row)
                values["stopped_reason_counts"] = json.dumps(values["stopped_reason_counts"], ensure_ascii=False)
                writer.writerow(values)


class Command(BaseCommand):
    help = "Evaluate RAG, single-agent, and multi-agent answers against the Agent evaluation set."

    def add_arguments(self, parser):
        parser.add_argument("--tasks", default="interview_materials/agent_eval_tasks.json")
        parser.add_argument("--kb-names", default=DEFAULT_KB_NAMES)
        parser.add_argument("--configs", default=",".join(DEFAULT_CONFIGS))
        parser.add_argument("--output-dir", default="interview_materials/results/agent_eval")
        parser.add_argument("--limit", type=int, default=0)

    def handle(self, *args, **options):
        task_path = Path(options["tasks"]).resolve()
        if not task_path.is_file():
            raise CommandError(f"Task file does not exist: {task_path}")
        tasks = json.loads(task_path.read_text(encoding="utf-8"))
        if not isinstance(tasks, list):
            raise CommandError("Task file must contain a JSON list")
        if options["limit"] > 0:
            tasks = tasks[: options["limit"]]

        kb_names = [name.strip() for name in options["kb_names"].split(",") if name.strip()]
        if not kb_names:
            raise CommandError("At least one knowledge base name is required")
        configs = list(dict.fromkeys(name.strip() for name in options["configs"].split(",") if name.strip()))
        invalid_configs = sorted(set(configs) - set(DEFAULT_CONFIGS))
        if not configs or invalid_configs:
            raise CommandError(f"Configs must be a comma-separated list of {', '.join(DEFAULT_CONFIGS)}")

        tenant = Tenant.objects.order_by("created_at").first()
        if not tenant:
            raise CommandError("No tenant exists")
        knowledge_bases = {
            item.name: item
            for item in KnowledgeBase.objects.filter(
                tenant=tenant,
                name__in=kb_names,
                deleted_at__isnull=True,
            )
        }
        missing = [name for name in kb_names if name not in knowledge_bases]
        if missing:
            raise CommandError(f"Knowledge bases do not exist: {', '.join(missing)}")
        kb_ids = [knowledge_bases[name].id for name in kb_names]

        results = []
        for task in tasks:
            for config in configs:
                started = time.perf_counter()
                usage_before = _usage_totals(tenant)
                answer = ""
                retrieved = []
                iterations = 0
                stopped_reason = "not_applicable" if config == "rag" else "completed"
                tool_names = Counter()
                error = None
                try:
                    if config == "rag":
                        answer, retrieved, iterations, stopped_reason, tool_names = _run_rag(
                            tenant,
                            task["question"],
                            kb_ids,
                        )
                    else:
                        answer, retrieved, iterations, stopped_reason, tool_names = _run_agent(
                            tenant,
                            task["question"],
                            kb_ids,
                            config,
                        )
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    stopped_reason = "error"

                usage_after = _usage_totals(tenant)
                mapped_ids, chunk_ids = _retrieval_ids(retrieved, task.get("gold_docs") or [])
                is_unanswerable = bool(task.get("unanswerable"))
                retrieval_result = None
                if not is_unanswerable:
                    retrieval_result = retrieval_metrics(
                        [list(range(len(task.get("gold_docs") or [])))],
                        mapped_ids,
                    )
                correct = False if error else score_answer(task, answer)
                results.append(
                    {
                        "task_id": task["id"],
                        "category": task.get("category", ""),
                        "config": config,
                        "question": task["question"],
                        "answer": answer,
                        "correct": bool(correct),
                        "retrieval_ids": mapped_ids,
                        "retrieved_chunk_ids": chunk_ids,
                        "retrieval_metrics": retrieval_result,
                        "generation_metrics": generation_metrics(answer, task.get("reference", "")),
                        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                        "total_tokens": max(usage_after["total_tokens"] - usage_before["total_tokens"], 0),
                        "request_count": max(usage_after["request_count"] - usage_before["request_count"], 0),
                        "tool_calls": sum(tool_names.values()),
                        "tool_name_histogram": dict(sorted(tool_names.items())),
                        "iterations": iterations,
                        "stopped_reason": stopped_reason,
                        "error": error,
                    }
                )
                self.stdout.write(f"evaluated {task['id']} ({config})")

        categories = list(dict.fromkeys(task.get("category", "") for task in tasks))
        summary = []
        for config in configs:
            config_rows = [item for item in results if item["config"] == config]
            summary.append(_summary_row(config_rows, config))
            for category in categories:
                category_rows = [
                    item
                    for item in config_rows
                    if item["category"] == category
                ]
                summary.append(_summary_row(category_rows, config, category=category))

        output_dir = Path(options["output_dir"]).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "knowledge_bases": [{"id": knowledge_bases[name].id, "name": name} for name in kb_names],
            "task_count": len(tasks),
            "configs": configs,
            "results": results,
            "summary": summary,
        }
        (output_dir / "agent_eval.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _write_rows_csv(output_dir / "agent_eval.csv", results)
        _write_summary_csv(output_dir / "agent_eval_summary.csv", summary)
        self.stdout.write(self.style.SUCCESS(json.dumps(summary, ensure_ascii=False, indent=2)))
