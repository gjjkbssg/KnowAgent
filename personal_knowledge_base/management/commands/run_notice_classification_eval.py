import csv
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from personal_knowledge_base.extraction.classify import (
    LABELS,
    LABEL_NAMES,
    classify_llm,
    classify_rules,
    predict_tfidf,
    top_features,
    train_tfidf,
)
from personal_knowledge_base.models import Tenant


NONE_LABEL = "<none>"
PER_DOC_FIELDS = (
    "doc_id",
    "split",
    "method",
    "gold",
    "pred",
    "correct",
    "evidence",
)


def _load_json(path: Path, description: str):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CommandError(f"Unable to read {description}: {exc}") from exc


def _load_replay(path: Path) -> dict[str, dict]:
    records = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise CommandError(f"Unable to read LLM replay file: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CommandError(
                f"Invalid LLM replay JSON on line {line_number}: {exc}"
            ) from exc
        doc_id = record.get("doc_id") if isinstance(record, dict) else None
        if not doc_id or doc_id in records:
            raise CommandError(
                f"Invalid or duplicate doc_id in LLM replay on line {line_number}"
            )
        records[doc_id] = record
    return records


def _document_path(repo: Path, doc_id: str, label: str) -> Path:
    directory = (
        repo / "interview_materials" / "tender_eval" / "docs"
        if label == "gkzb"
        else repo / "interview_materials" / "notice_cls" / "docs"
    )
    return directory / f"{doc_id}.txt"


def _metrics(rows: list[dict]) -> dict:
    columns = [*LABELS, NONE_LABEL]
    matrix = {label: {column: 0 for column in columns} for label in LABELS}
    for row in rows:
        prediction = row["pred"] if row["pred"] in LABELS else NONE_LABEL
        matrix[row["gold"]][prediction] += 1

    per_class = {}
    for label in LABELS:
        tp = matrix[label][label]
        support = sum(matrix[label].values())
        predicted = sum(matrix[gold][label] for gold in LABELS)
        fp = predicted - tp
        fn = support - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / support if support else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }
    count = len(rows)
    latencies = [float(row.get("latency_ms") or 0) for row in rows]
    tokens = [int(row.get("tokens") or 0) for row in rows]
    evidence_rates = [
        bool(row.get("evidence_ok"))
        for row in rows
        if row.get("evidence_ok") is not None
    ]
    return {
        "documents": count,
        "accuracy": round(
            sum(row["correct"] for row in rows) / count if count else 0.0, 4
        ),
        "macro_f1": round(
            statistics.mean(item["f1"] for item in per_class.values()), 4
        ),
        "per_class": per_class,
        "confusion_matrix": {
            "labels": list(LABELS),
            "predicted_labels": columns,
            "counts": [[matrix[gold][pred] for pred in columns] for gold in LABELS],
        },
        "coverage": round(
            sum(row["pred"] in LABELS for row in rows) / count if count else 0.0,
            4,
        ),
        "mean_latency_ms_per_doc": round(
            sum(latencies) / count if count else 0.0, 2
        ),
        "mean_tokens_per_doc": round(sum(tokens) / count if count else 0.0, 2),
        "evidence_ok_rate": (
            round(sum(evidence_rates) / len(evidence_rates), 4)
            if evidence_rates
            else None
        ),
    }


def _write_confusion(path: Path, metrics: dict) -> None:
    predicted_labels = metrics["confusion_matrix"]["predicted_labels"]
    counts = metrics["confusion_matrix"]["counts"]
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(["gold", *predicted_labels])
        for label, row in zip(LABELS, counts):
            writer.writerow([label, *row])


def _cross_validate(texts: list[str], labels: list[str], analyzer: str, folds: int):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score, f1_score

    counts = {label: labels.count(label) for label in LABELS}
    if folds < 2 or min(counts.values()) < folds:
        raise CommandError(
            f"--cv={folds} requires at least {folds} documents per label; "
            f"counts are {counts}"
        )
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
    scores = []
    for train_indices, test_indices in splitter.split(texts, labels):
        model = train_tfidf(
            [texts[index] for index in train_indices],
            [labels[index] for index in train_indices],
            analyzer=analyzer,
        )
        predictions = predict_tfidf(model, [texts[index] for index in test_indices])
        gold = [labels[index] for index in test_indices]
        scores.append(
            {
                "accuracy": float(accuracy_score(gold, predictions)),
                "macro_f1": float(
                    f1_score(
                        gold,
                        predictions,
                        labels=list(LABELS),
                        average="macro",
                        zero_division=0,
                    )
                ),
            }
        )
    return {
        "label": f"{folds}-fold cross-validation on all 150 documents",
        "folds": scores,
        "accuracy_mean": round(statistics.mean(row["accuracy"] for row in scores), 4),
        "accuracy_std": round(statistics.pstdev(row["accuracy"] for row in scores), 4),
        "macro_f1_mean": round(statistics.mean(row["macro_f1"] for row in scores), 4),
        "macro_f1_std": round(statistics.pstdev(row["macro_f1"] for row in scores), 4),
    }


class Command(BaseCommand):
    help = "Evaluate notice classification methods against the labeled notice set."

    def add_arguments(self, parser):
        parser.add_argument(
            "--methods",
            default="rules,tfidf_char,tfidf_word,llm",
        )
        parser.add_argument("--split", choices=("test", "dev", "all"), default="test")
        parser.add_argument(
            "--output-dir",
            default="interview_materials/results/notice_cls",
        )
        parser.add_argument("--replay-llm")
        parser.add_argument("--cv", type=int, default=0)

    def handle(self, *args, **options):
        if options["cv"] < 0:
            raise CommandError("--cv must be zero or a positive fold count")
        methods = list(
            dict.fromkeys(
                method.strip()
                for method in options["methods"].split(",")
                if method.strip()
            )
        )
        supported = {"rules", "tfidf_char", "tfidf_word", "llm"}
        if not methods or any(method not in supported for method in methods):
            raise CommandError(
                "--methods must contain rules, tfidf_char, tfidf_word, and/or llm"
            )
        repo = Path(settings.BASE_DIR)
        labels_path = repo / "interview_materials" / "notice_cls" / "labels.json"
        documents = _load_json(labels_path, "notice classification labels")
        if not isinstance(documents, list) or not documents:
            raise CommandError("Notice classification labels must be a nonempty list")
        by_id = {row.get("doc_id"): row for row in documents}
        if len(by_id) != len(documents) or None in by_id:
            raise CommandError("Notice classification labels have missing/duplicate IDs")
        for row in documents:
            if row.get("label") not in LABELS or row.get("split") not in {"dev", "test"}:
                raise CommandError(f"Invalid label/split for {row.get('doc_id')}")

        split = options["split"]
        selected = [
            row for row in documents if split == "all" or row["split"] == split
        ]
        if not selected:
            raise CommandError(f"No labeled documents found for split={split}")
        dev = [row for row in documents if row["split"] == "dev"]
        needs_tfidf = any(method.startswith("tfidf_") for method in methods)
        needed_ids = {row["doc_id"] for row in selected}
        if needs_tfidf:
            needed_ids.update(row["doc_id"] for row in dev)
        if options["cv"] and needs_tfidf:
            needed_ids.update(row["doc_id"] for row in documents)
        text_by_id = {}
        for row in documents:
            if row["doc_id"] not in needed_ids:
                continue
            path = _document_path(repo, row["doc_id"], row["label"])
            try:
                text_by_id[row["doc_id"]] = path.read_text(encoding="utf-8")[:3000]
            except OSError as exc:
                raise CommandError(f"Unable to read notice {row['doc_id']}: {exc}") from exc

        output_dir = Path(options["output_dir"])
        if not output_dir.is_absolute():
            output_dir = repo / output_dir
        replay_path = options.get("replay_llm")
        if replay_path:
            replay_path = Path(replay_path)
            if not replay_path.is_absolute():
                replay_path = repo / replay_path
            replay_records = _load_replay(replay_path)
            missing = [row["doc_id"] for row in selected if row["doc_id"] not in replay_records]
            if missing:
                raise CommandError(
                    f"LLM replay is missing documents: {', '.join(missing[:5])}"
                )
        else:
            replay_records = None
        llm_tenant = None
        if "llm" in methods and replay_records is None:
            llm_tenant = Tenant.objects.order_by("created_at").first()
            if llm_tenant is None:
                raise CommandError("LLM evaluation requires at least one configured tenant")

        method_rows = {method: [] for method in methods}
        models = {}
        for method in methods:
            if method.startswith("tfidf_"):
                analyzer = method.removeprefix("tfidf_")
                models[method] = train_tfidf(
                    [text_by_id[row["doc_id"]] for row in dev],
                    [row["label"] for row in dev],
                    analyzer=analyzer,
                )

        raw_records = []
        for row in selected:
            doc_id = row["doc_id"]
            text = text_by_id[doc_id]
            for method in methods:
                prediction = None
                evidence = None
                latency_ms = 0.0
                tokens = 0
                evidence_ok = None
                if method == "rules":
                    started = time.perf_counter()
                    prediction, evidence = classify_rules(text)
                    latency_ms = (time.perf_counter() - started) * 1000
                elif method.startswith("tfidf_"):
                    started = time.perf_counter()
                    prediction = predict_tfidf(models[method], [text])[0]
                    latency_ms = (time.perf_counter() - started) * 1000
                elif method == "llm":
                    if replay_records is not None:
                        raw_record = replay_records[doc_id]
                        raw_records.append(raw_record)
                        if raw_record.get("error") or not isinstance(
                            raw_record.get("response"), dict
                        ):
                            latency_ms = float(raw_record.get("latency_ms") or 0)
                            tokens = int(
                                (raw_record.get("usage") or {}).get("total_tokens", 0)
                                or 0
                            )
                            evidence_ok = False
                        else:
                            prediction, evidence, usage = classify_llm(
                                text,
                                None,
                                response=raw_record["response"],
                                usage_override=raw_record.get("usage"),
                                latency_ms=raw_record.get("latency_ms"),
                            )
                            latency_ms = usage["latency_ms"]
                            tokens = usage["total_tokens"]
                            evidence_ok = usage["evidence_ok"]
                    else:
                        started = time.perf_counter()
                        error = None
                        response = None
                        usage = {}
                        try:
                            prediction, evidence, usage, response = classify_llm(
                                text,
                                llm_tenant,
                                include_response=True,
                            )
                        except Exception as exc:
                            error = str(exc)[:500]
                            latency_ms = (time.perf_counter() - started) * 1000
                            evidence_ok = False
                        else:
                            latency_ms = usage["latency_ms"]
                            tokens = usage["total_tokens"]
                            evidence_ok = usage["evidence_ok"]
                        raw_records.append(
                            {
                                "doc_id": doc_id,
                                "response": response,
                                "latency_ms": latency_ms,
                                "usage": usage,
                                **({"error": error} if error else {}),
                            }
                        )
                method_rows[method].append(
                    {
                        "doc_id": doc_id,
                        "split": row["split"],
                        "gold": row["label"],
                        "pred": prediction,
                        "evidence": evidence,
                        "correct": prediction == row["label"],
                        "latency_ms": latency_ms,
                        "tokens": tokens,
                        "evidence_ok": evidence_ok,
                    }
                )

        output_dir.mkdir(parents=True, exist_ok=True)
        if "llm" in methods and replay_records is None:
            with (output_dir / "llm_raw.jsonl").open(
                "w", encoding="utf-8", newline=""
            ) as raw_file:
                for record in raw_records:
                    raw_file.write(json.dumps(record, ensure_ascii=False) + "\n")

        summary = {
            "metadata": {
                "dataset": "interview_materials/notice_cls/labels.json",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "methods": methods,
                "requested_split": split,
                "dev_documents": len(dev),
                "evaluation_documents": len(selected),
            },
            "metrics": {},
            "cross_validation": {},
        }
        csv_rows = []
        feature_payload = {}
        for method, rows in method_rows.items():
            summary["metrics"][method] = _metrics(rows)
            _write_confusion(output_dir / f"confusion_{method}.csv", summary["metrics"][method])
            csv_rows.extend(
                {
                    "doc_id": row["doc_id"],
                    "split": row["split"],
                    "method": method,
                    "gold": row["gold"],
                    "pred": row["pred"] or "",
                    "correct": row["correct"],
                    "evidence": (row["evidence"] or "").strip(),
                }
                for row in rows
            )
            if method.startswith("tfidf_"):
                feature_payload[method] = top_features(models[method], k=10)
        if options["cv"]:
            all_texts = [text_by_id[row["doc_id"]] for row in documents]
            all_labels = [row["label"] for row in documents]
            for method in methods:
                if method.startswith("tfidf_"):
                    analyzer = method.removeprefix("tfidf_")
                    summary["cross_validation"][method] = _cross_validate(
                        all_texts,
                        all_labels,
                        analyzer,
                        options["cv"],
                    )

        with (output_dir / "per_doc.csv").open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(
                output, fieldnames=PER_DOC_FIELDS, lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(csv_rows)
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (output_dir / "top_features.json").write_text(
            json.dumps(feature_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        self._write_report(output_dir / "report.md", summary)
        self.stdout.write(f"Wrote notice classification evaluation to {output_dir}")

    @staticmethod
    def _write_report(path: Path, summary: dict) -> None:
        lines = [
            "# Notice classification evaluation",
            "",
            "Platform categories are the gold labels; they are not relabeled as semantic truth.",
            "",
            "| Method | Accuracy | Macro-F1 | Coverage | Mean ms/doc | Mean tokens/doc | Evidence OK |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for method, metrics in summary["metrics"].items():
            evidence_rate = metrics["evidence_ok_rate"]
            lines.append(
                f"| {method} | {metrics['accuracy']:.4f} | {metrics['macro_f1']:.4f} "
                f"| {metrics['coverage']:.4f} | {metrics['mean_latency_ms_per_doc']:.2f} "
                f"| {metrics['mean_tokens_per_doc']:.2f} "
                f"| {'n/a' if evidence_rate is None else f'{evidence_rate:.4f}'} |"
            )
        for method, metrics in summary["metrics"].items():
            lines.extend(["", f"## {method}", "", "Per-class precision / recall / F1:", ""])
            lines.append("| Label | Precision | Recall | F1 | Support |")
            lines.append("|---|---:|---:|---:|---:|")
            for label in LABELS:
                values = metrics["per_class"][label]
                lines.append(
                    f"| {label} ({LABEL_NAMES[label]}) | {values['precision']:.4f} "
                    f"| {values['recall']:.4f} | {values['f1']:.4f} | {values['support']} |"
                )
            lines.extend(
                [
                    "",
                    "Confusion matrix (rows=gold; columns=prediction, including no-fire):",
                    "",
                    "| Gold \\ Pred | " + " | ".join(metrics["confusion_matrix"]["predicted_labels"]) + " |",
                    "|---|" + "---:|" * len(metrics["confusion_matrix"]["predicted_labels"]),
                ]
            )
            for label, values in zip(
                LABELS, metrics["confusion_matrix"]["counts"]
            ):
                lines.append(f"| {label} | " + " | ".join(map(str, values)) + " |")
        if summary["cross_validation"]:
            lines.extend(["", "## Cross-validation (not test results)", ""])
            for method, metrics in summary["cross_validation"].items():
                lines.append(
                    f"- {method}, {metrics['label']}: accuracy "
                    f"{metrics['accuracy_mean']:.4f} ± {metrics['accuracy_std']:.4f}; "
                    f"macro-F1 {metrics['macro_f1_mean']:.4f} ± "
                    f"{metrics['macro_f1_std']:.4f}."
                )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
