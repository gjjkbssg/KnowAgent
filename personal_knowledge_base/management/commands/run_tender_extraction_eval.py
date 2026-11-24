import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from personal_knowledge_base.extraction.normalize import (
    normalize_amount,
    normalize_code,
    normalize_datetime,
    normalize_name,
)
from personal_knowledge_base.extraction.tender import (
    FIELDS,
    FieldValue,
    build_extraction_result,
    extract_llm,
    extract_rules,
    verify_evidence,
)
from personal_knowledge_base.models import Tenant


CSV_FIELDS = (
    "doc_id",
    "split",
    "method",
    "field",
    "gold",
    "pred",
    "correct",
    "in_review",
    "source",
    "evidence",
)


def _amount_value(value):
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return normalize_amount(str(value), default_unit="元")


def _canonical(field: str, value):
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if field == "project_code":
        return normalize_code(str(value))
    if field in {"project_name", "purchaser", "agency"}:
        return normalize_name(str(value))
    if field == "budget_yuan":
        return _amount_value(value)
    if field == "bid_deadline":
        return normalize_datetime(str(value))
    return value


def _is_correct(field: str, prediction, document: dict) -> bool:
    gold = document.get("fields", {}).get(field)
    if field == "project_code":
        accepted = [gold]
        accepted.extend((document.get("alternates") or {}).get("project_code") or [])
        normalized_accepted = {
            _canonical(field, value) for value in accepted if value is not None
        }
        return _canonical(field, prediction) in normalized_accepted
    if field == "budget_yuan":
        predicted_amount = _amount_value(prediction)
        gold_amount = _amount_value(gold)
        return (
            predicted_amount is not None
            and gold_amount is not None
            and abs(predicted_amount - gold_amount) < 0.01
        )
    return _canonical(field, prediction) == _canonical(field, gold)


def _prf(counts: dict[str, int]) -> dict[str, float | int]:
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def score_predictions(gold_documents: list[dict], predictions: list[dict]) -> dict:
    gold_by_id = {document["doc_id"]: document for document in gold_documents}
    field_counts = {
        field: {"tp": 0, "fp": 0, "fn": 0} for field in FIELDS
    }
    total_counts = {"tp": 0, "fp": 0, "fn": 0}
    reviewed = 0
    incorrect = 0
    reviewed_incorrect = 0
    exact_documents = 0
    evidence_dropped = 0

    for prediction in predictions:
        document = gold_by_id[prediction["doc_id"]]
        values = prediction.get("fields", {})
        needs_review = set(prediction.get("needs_review") or [])
        document_correct = True
        for field in FIELDS:
            predicted = values.get(field)
            gold = document.get("fields", {}).get(field)
            correct = _is_correct(field, predicted, document)
            counts = field_counts[field]
            if correct:
                counts["tp"] += 1
                total_counts["tp"] += 1
            else:
                document_correct = False
                incorrect += 1
                if predicted is not None:
                    counts["fp"] += 1
                    total_counts["fp"] += 1
                if gold is not None:
                    counts["fn"] += 1
                    total_counts["fn"] += 1
                if field in needs_review:
                    reviewed_incorrect += 1
            if field in needs_review:
                reviewed += 1
        if document_correct:
            exact_documents += 1
        evidence_dropped += len(prediction.get("evidence_dropped") or [])

    document_count = len(predictions)
    all_fields = document_count * len(FIELDS)
    return {
        "per_field": {field: _prf(counts) for field, counts in field_counts.items()},
        "micro": _prf(total_counts),
        "document_exact_match_rate": round(exact_documents / document_count, 4)
        if document_count
        else 0.0,
        "mean_latency_ms_per_doc": round(
            sum(float(item.get("latency_ms", 0) or 0) for item in predictions)
            / document_count,
            2,
        )
        if document_count
        else 0.0,
        "mean_tokens_per_doc": round(
            sum(int(item.get("tokens", 0) or 0) for item in predictions)
            / document_count,
            2,
        )
        if document_count
        else 0.0,
        "evidence_dropped_count": evidence_dropped,
        "review_rate": round(reviewed / all_fields, 4) if all_fields else 0.0,
        "review_catch_rate": round(reviewed_incorrect / incorrect, 4)
        if incorrect
        else None,
    }


def _field_rows(document: dict, prediction: dict, method: str) -> list[dict]:
    rows = []
    fields = prediction["fields"]
    needs_review = set(prediction["needs_review"])
    sources = prediction["sources"]
    for field in FIELDS:
        gold = document.get("fields", {}).get(field)
        predicted = fields.get(field)
        rows.append(
            {
                "doc_id": document["doc_id"],
                "split": document["split"],
                "method": method,
                "field": field,
                "gold": gold,
                "pred": predicted,
                "correct": _is_correct(field, predicted, document),
                "in_review": field in needs_review,
                "source": sources.get(field),
                "evidence": json.dumps(
                    prediction.get("evidence", {}).get(field),
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            }
        )
    return rows


def _new_null_llm_fields() -> dict[str, FieldValue]:
    return {field: FieldValue(None, None, "llm") for field in FIELDS}


def _render_metric(value):
    return "n/a" if value is None else str(value)


class Command(BaseCommand):
    help = "Evaluate tender notice field extraction methods against the labeled dataset."

    def add_arguments(self, parser):
        parser.add_argument("--methods", default="rules,llm,hybrid")
        parser.add_argument("--split", choices=("dev", "test", "all"), default="all")
        parser.add_argument("--no-evidence-check", action="store_true")
        parser.add_argument(
            "--output-dir",
            default="interview_materials/results/tender_extraction",
        )

    def handle(self, *args, **options):
        methods = [method.strip() for method in options["methods"].split(",") if method.strip()]
        if not methods or any(method not in {"rules", "llm", "hybrid"} for method in methods):
            raise CommandError("--methods must contain rules, llm, and/or hybrid")
        methods = list(dict.fromkeys(methods))
        split = options["split"]
        no_evidence_check = options["no_evidence_check"]
        output_dir = Path(options["output_dir"])
        if not output_dir.is_absolute():
            output_dir = Path(settings.BASE_DIR) / output_dir

        data_dir = Path(settings.BASE_DIR) / "interview_materials" / "tender_eval"
        try:
            dataset = json.loads((data_dir / "gold.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CommandError(f"Unable to read tender evaluation labels: {exc}") from exc
        documents = dataset["documents"]
        if split != "all":
            documents = [document for document in documents if document["split"] == split]
        if not documents:
            raise CommandError(f"No labeled documents found for split={split}")

        tenant = Tenant.objects.order_by("created_at").first()
        needs_llm = any(method in {"llm", "hybrid"} for method in methods)
        if needs_llm and tenant is None:
            raise CommandError("LLM evaluation requires at least one configured tenant")

        method_predictions = {method: [] for method in methods}
        errors = []
        docs_dir = data_dir / "docs"
        for document in documents:
            doc_id = document["doc_id"]
            path = docs_dir / f"{doc_id}.txt"
            try:
                text = path.read_text(encoding="utf-8")
            except OSError as exc:
                raise CommandError(f"Unable to read tender notice {doc_id}: {exc}") from exc

            rule_fields = {}
            rule_latency_ms = 0.0
            if "rules" in methods or "hybrid" in methods:
                started = time.perf_counter()
                rule_fields = extract_rules(text)
                rule_latency_ms = (time.perf_counter() - started) * 1000

            llm_fields = {}
            usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            llm_latency_ms = 0.0
            evidence_dropped = []
            if needs_llm:
                started = time.perf_counter()
                try:
                    llm_fields, usage = extract_llm(text, tenant)
                except Exception as exc:
                    llm_fields = _new_null_llm_fields()
                    errors.append(
                        {
                            "doc_id": doc_id,
                            "method": "llm",
                            "error": str(exc)[:500],
                        }
                    )
                llm_latency_ms = (time.perf_counter() - started) * 1000
                if not no_evidence_check:
                    llm_fields, evidence_dropped = verify_evidence(llm_fields, text)

            for method in methods:
                result = build_extraction_result(
                    method,
                    rule_fields=rule_fields,
                    llm_fields=llm_fields,
                    usage=usage if method != "rules" else {},
                    evidence_dropped=evidence_dropped if method != "rules" else [],
                )
                latency_ms = {
                    "rules": rule_latency_ms,
                    "llm": llm_latency_ms,
                    "hybrid": rule_latency_ms + llm_latency_ms,
                }[method]
                prediction = {
                    "doc_id": doc_id,
                    "split": document["split"],
                    "fields": result["fields"],
                    "evidence": result["evidence"],
                    "sources": result["sources"],
                    "needs_review": result["needs_review"],
                    "issues": result["issues"],
                    "evidence_dropped": result["evidence_dropped"],
                    "latency_ms": latency_ms,
                    "tokens": result["usage"].get("total_tokens", 0),
                }
                method_predictions[method].append(prediction)

        output_dir.mkdir(parents=True, exist_ok=True)
        summary_rows = []
        csv_rows = []
        for method, predictions in method_predictions.items():
            for document, prediction in zip(documents, predictions):
                csv_rows.extend(_field_rows(document, prediction, method))
            splits = sorted({document["split"] for document in documents})
            for current_split in splits:
                current_documents = [
                    document for document in documents if document["split"] == current_split
                ]
                current_ids = {document["doc_id"] for document in current_documents}
                current_predictions = [
                    prediction
                    for prediction in predictions
                    if prediction["doc_id"] in current_ids
                ]
                summary_rows.append(
                    {
                        "method": method,
                        "split": current_split,
                        "documents": len(current_documents),
                        **score_predictions(current_documents, current_predictions),
                    }
                )

        summary = {
            "metadata": {
                "dataset": "interview_materials/tender_eval/gold.json",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "methods": methods,
                "requested_split": split,
                "no_evidence_check": no_evidence_check,
                "document_count": len(documents),
                "tenant_id": str(tenant.pk) if tenant else None,
            },
            "results": summary_rows,
            "per_document": [
                {"method": method, **prediction}
                for method, predictions in method_predictions.items()
                for prediction in predictions
            ],
            "errors": errors,
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        with (output_dir / "per_doc.csv").open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS, lineterminator="\n")
            writer.writeheader()
            for row in csv_rows:
                writer.writerow(row)

        self.stdout.write(
            "| Method | Split | Micro P | Micro R | Micro F1 | Exact match | "
            "Latency ms/doc | Tokens/doc | Evidence dropped | Review rate | Review catch rate |"
        )
        self.stdout.write(
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
        )
        for row in summary_rows:
            micro = row["micro"]
            self.stdout.write(
                "| {method} | {split} | {precision} | {recall} | {f1} | {exact} | "
                "{latency} | {tokens} | {dropped} | {review} | {catch} |".format(
                    method=row["method"],
                    split=row["split"],
                    precision=micro["precision"],
                    recall=micro["recall"],
                    f1=micro["f1"],
                    exact=row["document_exact_match_rate"],
                    latency=row["mean_latency_ms_per_doc"],
                    tokens=row["mean_tokens_per_doc"],
                    dropped=row["evidence_dropped_count"],
                    review=row["review_rate"],
                    catch=_render_metric(row["review_catch_rate"]),
                )
            )
        self.stdout.write("| Method | Split | Field | Precision | Recall | F1 |")
        self.stdout.write("|---|---|---|---:|---:|---:|")
        for row in summary_rows:
            for field in FIELDS:
                metrics = row["per_field"][field]
                self.stdout.write(
                    f"| {row['method']} | {row['split']} | {field} | "
                    f"{metrics['precision']} | {metrics['recall']} | {metrics['f1']} |"
                )
        if errors:
            self.stderr.write(f"Model errors: {len(errors)}; details saved in summary.json")
        self.stdout.write(
            self.style.SUCCESS(
                f"Saved tender extraction evaluation to {output_dir}"
            )
        )
