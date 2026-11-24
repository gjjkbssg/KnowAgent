import csv
import hashlib
import json
import re
import statistics
import time
from pathlib import Path

from django.conf import settings
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.core.management.base import BaseCommand, CommandError
from django.test.utils import override_settings

from personal_knowledge_base.document_parsing import parse_document
from personal_knowledge_base.document_processing import process_knowledge
from personal_knowledge_base.model_providers import chat_completion_raw
from personal_knowledge_base.models import Chunk, Knowledge, KnowledgeBase, ModelUsage, Tenant
from personal_knowledge_base.search import hybrid_search
from services.knowledge import delete_knowledge_content


SYSTEM_PROMPT = "你是招标文件问答助手。只根据给出的资料回答问题，答案简短；资料中没有答案时回答“资料中未找到”。"
FOOTER_RE = re.compile(r"—\s*\d+\s*—")
TOKEN_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")
PDF_EVAL_KB_MARKER = "created_by=run_pdf_parsing_eval"


def _compact(text):
    return "".join(str(text or "").split())


def _matches(content, expected):
    content = _compact(content)
    return all(_compact(value) in content for value in (expected or []))


def _mean(values):
    return round(statistics.mean(values), 3) if values else 0.0


def _summarize_questions(rows, with_answers):
    count = len(rows)
    answers = [row for row in rows if row["answer_correct"] is not None]
    tokens = {
        field: sum(row["answer_tokens"][field] for row in rows)
        for field in TOKEN_FIELDS
    }
    result = {
        "question_count": count,
        "hit_at_1": round(sum(row["hit@1"] for row in rows) / count, 4) if count else 0.0,
        "hit_at_5": round(sum(row["hit@5"] for row in rows) / count, 4) if count else 0.0,
        "mrr_at_10": round(sum(row["reciprocal_rank"] for row in rows) / count, 4) if count else 0.0,
        "mean_search_latency_ms": _mean([row["search_latency_ms"] for row in rows]),
    }
    if with_answers:
        result["answer_accuracy"] = (
            round(sum(bool(row["answer_correct"]) for row in answers) / len(answers), 4)
            if answers
            else 0.0
        )
        result["answer_count"] = len(answers)
        result["answer_tokens"] = tokens
    else:
        result["answer_accuracy"] = None
        result["answer_count"] = 0
        result["answer_tokens"] = tokens
    return result


def _tokens_after_call(tenant, count_before):
    usage = {field: 0 for field in TOKEN_FIELDS}
    usages = ModelUsage.objects.filter(tenant=tenant)
    if usages.count() <= count_before:
        return usage
    record = usages.order_by("-created_at").first()
    if record:
        return {
            "prompt_tokens": int(record.prompt_tokens or 0),
            "completion_tokens": int(record.completion_tokens or 0),
            "total_tokens": int(record.total_tokens or 0),
        }
    return usage


def _cleanup_knowledge_base(tenant, name):
    existing_bases = list(
        KnowledgeBase.objects.filter(tenant=tenant, name=name)
    )
    if not existing_bases:
        return
    if any(
        not (knowledge_base.description or "").startswith(PDF_EVAL_KB_MARKER)
        for knowledge_base in existing_bases
    ):
        raise CommandError(
            f"Refusing to delete unmarked knowledge base named {name!r}"
        )
    for knowledge_base in existing_bases:
        for knowledge in Knowledge.objects.filter(knowledge_base=knowledge_base):
            delete_knowledge_content(knowledge)
            if knowledge.file_path and default_storage.exists(knowledge.file_path):
                default_storage.delete(knowledge.file_path)
        knowledge_base.delete()


def _create_knowledge_base(tenant, mode, pdf_path, pdf_data):
    name = f"pdf-eval-{mode}"
    _cleanup_knowledge_base(tenant, name)
    knowledge_base = KnowledgeBase.objects.create(
        tenant=tenant,
        name=name,
        description=f"{PDF_EVAL_KB_MARKER}; PDF parsing evaluation ({mode})",
        chunking_config={"chunk_size": 512, "chunk_overlap": 50},
        indexing_strategy={
            "vector_enabled": True,
            "keyword_enabled": True,
            "graph_enabled": False,
            "wiki_enabled": False,
        },
        wiki_config={"enabled": False},
        extract_config={"enabled": False},
    )
    file_path = default_storage.save(
        f"tenant-{tenant.id}/{knowledge_base.id}/{pdf_path.name}",
        ContentFile(pdf_data),
    )
    digest = hashlib.sha256(pdf_data).hexdigest()
    knowledge = Knowledge.objects.create(
        tenant=tenant,
        knowledge_base=knowledge_base,
        type="file",
        title=pdf_path.stem,
        source=pdf_path.name,
        file_name=pdf_path.name,
        file_type="pdf",
        file_size=len(pdf_data),
        file_path=file_path,
        file_hash=digest,
        storage_size=len(pdf_data),
        metadata={"process_config": {"graph_enabled": False}},
    )
    return knowledge_base, knowledge


def _parse_stats(knowledge_base, parsed):
    chunks = list(
        Chunk.objects.filter(knowledge_base=knowledge_base)
        .order_by("chunk_index")
    )
    return {
        "chunk_count": len(chunks),
        "mean_chunk_chars": _mean([len(chunk.content or "") for chunk in chunks]),
        "footer_pattern_chunks": sum(
            bool(FOOTER_RE.search(chunk.content or "")) for chunk in chunks
        ),
        "table_chunks": sum(
            (chunk.metadata or {}).get("block_type") == "table" for chunk in chunks
        ),
        "heading_path_chunks": sum(
            bool((chunk.metadata or {}).get("heading_path")) for chunk in chunks
        ),
        "pdf_layout": parsed.metadata.get("pdf_layout") or {},
    }


def _answer(tenant, question, results):
    context = "\n".join(
        f"[{index}] {item.get('content', '')}"
        for index, item in enumerate(results[:5], start=1)
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"资料：\n{context}\n\n问题：{question}",
        },
    ]
    before_count = ModelUsage.objects.filter(tenant=tenant).count()
    try:
        response = chat_completion_raw(tenant, messages, temperature=0)
        answer = response.get("content") or ""
        error = None
    except Exception as exc:
        answer = ""
        error = type(exc).__name__
    return answer, _tokens_after_call(tenant, before_count), error


def _markdown_table(mode_reports, with_answers):
    lines = [
        "| Mode | Category | Questions | Hit@1 | Hit@5 | MRR@10 | Search ms | Answer accuracy | Answer tokens |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, report in mode_reports.items():
        rows = [("overall", report["retrieval"]["overall"], report["answers"]["overall"])]
        rows.extend(
            (
                category,
                metrics,
                report["answers"]["by_category"][category],
            )
            for category, metrics in report["retrieval"]["by_category"].items()
        )
        for category, retrieval, answers in rows:
            answer_accuracy = (
                f"{answers['answer_accuracy']:.4f}" if with_answers else "n/a"
            )
            tokens = answers["answer_tokens"]["total_tokens"] if with_answers else "n/a"
            lines.append(
                f"| {mode} | {category} | {retrieval['question_count']} "
                f"| {retrieval['hit_at_1']:.4f} | {retrieval['hit_at_5']:.4f} "
                f"| {retrieval['mrr_at_10']:.4f} "
                f"| {retrieval['mean_search_latency_ms']:.2f} "
                f"| {answer_accuracy} | {tokens} |"
            )
    return "\n".join(lines)


class Command(BaseCommand):
    help = "Compare legacy and layout-aware parsing on a real PDF retrieval task."

    def add_arguments(self, parser):
        parser.add_argument(
            "--pdf",
            default="interview_materials/pdf_eval/ahmu_tender_2026.pdf",
        )
        parser.add_argument(
            "--questions",
            default="interview_materials/pdf_eval/questions.json",
        )
        parser.add_argument("--modes", default="legacy,structured")
        parser.add_argument(
            "--output-dir",
            default="interview_materials/results/pdf_parsing",
        )
        parser.add_argument("--with-answers", action="store_true")

    def handle(self, *args, **options):
        modes = list(
            dict.fromkeys(
                mode.strip() for mode in options["modes"].split(",") if mode.strip()
            )
        )
        if not modes or any(mode not in {"legacy", "structured"} for mode in modes):
            raise CommandError("--modes must contain legacy and/or structured")

        def resolve_path(value):
            path = Path(value)
            return path if path.is_absolute() else Path(settings.BASE_DIR) / path

        pdf_path = resolve_path(options["pdf"])
        question_path = resolve_path(options["questions"])
        output_dir = resolve_path(options["output_dir"])
        if not pdf_path.is_file():
            raise CommandError(f"PDF file does not exist: {pdf_path}")
        if not question_path.is_file():
            raise CommandError(f"Questions file does not exist: {question_path}")
        try:
            questions = json.loads(question_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CommandError(f"Unable to read PDF evaluation questions: {exc}") from exc
        if not isinstance(questions, list) or not questions:
            raise CommandError("Questions file must contain a non-empty JSON list")

        tenant = Tenant.objects.order_by("created_at").first()
        if tenant is None:
            raise CommandError("No tenant exists for PDF parsing evaluation")
        pdf_data = pdf_path.read_bytes()
        with_answers = options["with_answers"]
        mode_reports = {}
        csv_rows = []

        for mode in modes:
            self.stdout.write(f"Processing {mode} mode...")
            with override_settings(PDF_LAYOUT_PARSING=mode == "structured"):
                parsed = parse_document(pdf_path.name, pdf_data)
                knowledge_base, knowledge = _create_knowledge_base(
                    tenant, mode, pdf_path, pdf_data
                )
                process_knowledge(knowledge.id)
                knowledge.refresh_from_db()
                if knowledge.parse_status != "completed":
                    raise CommandError(
                        f"{mode} processing failed: {knowledge.error_message}"
                    )

                per_question = []
                for question in questions:
                    started = time.perf_counter()
                    results = hybrid_search(
                        tenant.id,
                        [knowledge_base.id],
                        question["question"],
                        10,
                    )
                    search_latency_ms = (time.perf_counter() - started) * 1000
                    first_hit_rank = next(
                        (
                            rank
                            for rank, result in enumerate(results, start=1)
                            if _matches(
                                result.get("content", ""),
                                question.get("must_contain", []),
                            )
                        ),
                        None,
                    )
                    hit_at_1 = first_hit_rank == 1
                    hit_at_5 = first_hit_rank is not None and first_hit_rank <= 5
                    reciprocal_rank = (
                        1 / first_hit_rank if first_hit_rank is not None else 0.0
                    )
                    answer = ""
                    answer_correct = None
                    answer_tokens = {field: 0 for field in TOKEN_FIELDS}
                    answer_error = None
                    if with_answers:
                        answer, answer_tokens, answer_error = _answer(
                            tenant,
                            question["question"],
                            results,
                        )
                        answer_correct = _matches(
                            answer,
                            question.get("answer_contains", []),
                        )
                    row = {
                        "id": question["id"],
                        "category": question["category"],
                        "mode": mode,
                        "first_hit_rank": first_hit_rank,
                        "hit@1": hit_at_1,
                        "hit@5": hit_at_5,
                        "reciprocal_rank": reciprocal_rank,
                        "search_latency_ms": search_latency_ms,
                        "answer": answer,
                        "answer_correct": answer_correct,
                        "answer_tokens": answer_tokens,
                        "answer_error": answer_error,
                    }
                    per_question.append(row)
                    csv_rows.append(
                        {
                            "id": row["id"],
                            "category": row["category"],
                            "mode": mode,
                            "first_hit_rank": first_hit_rank or "",
                            "hit@1": hit_at_1,
                            "hit@5": hit_at_5,
                            "answer": answer,
                            "answer_correct": (
                                answer_correct if answer_correct is not None else ""
                            ),
                        }
                    )

                categories = list(dict.fromkeys(item["category"] for item in questions))
                retrieval_by_category = {
                    category: _summarize_questions(
                        [row for row in per_question if row["category"] == category],
                        False,
                    )
                    for category in categories
                }
                answers_by_category = {
                    category: _summarize_questions(
                        [row for row in per_question if row["category"] == category],
                        with_answers,
                    )
                    for category in categories
                }
                mode_reports[mode] = {
                    "knowledge_base": {
                        "id": knowledge_base.id,
                        "name": knowledge_base.name,
                    },
                    "parse_stats": _parse_stats(knowledge_base, parsed),
                    "retrieval": {
                        "overall": _summarize_questions(per_question, False),
                        "by_category": retrieval_by_category,
                    },
                    "answers": {
                        "overall": _summarize_questions(per_question, with_answers),
                        "by_category": answers_by_category,
                    },
                    "per_question": per_question,
                }

        output_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "pdf": pdf_path.name,
            "pdf_sha256": hashlib.sha256(pdf_data).hexdigest(),
            "question_count": len(questions),
            "with_answers": with_answers,
            "modes": mode_reports,
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        fields = [
            "id",
            "category",
            "mode",
            "first_hit_rank",
            "hit@1",
            "hit@5",
            "answer",
            "answer_correct",
        ]
        with (output_dir / "per_question.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fields,
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(csv_rows)
        markdown = _markdown_table(mode_reports, with_answers)
        self.stdout.write(markdown)
