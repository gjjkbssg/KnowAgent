import csv
import json
import re
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime
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


ORG_TRANSLATION = str.maketrans(
    {
        "(": "（",
        ")": "）",
        "[": "［",
        "]": "］",
        "{": "｛",
        "}": "｝",
        "<": "＜",
        ">": "＞",
        "【": "［",
        "】": "］",
        "“": '"',
        "”": '"',
        "„": '"',
        "＂": '"',
        "‘": "'",
        "’": "'",
        "‚": "'",
        "＇": "'",
    }
)

PROVINCES = (
    ("北京市", ("北京市", "北京")),
    ("天津市", ("天津市", "天津")),
    ("河北省", ("河北省", "河北")),
    ("山西省", ("山西省", "山西")),
    ("内蒙古自治区", ("内蒙古自治区", "内蒙古")),
    ("辽宁省", ("辽宁省", "辽宁")),
    ("吉林省", ("吉林省", "吉林")),
    ("黑龙江省", ("黑龙江省", "黑龙江")),
    ("上海市", ("上海市", "上海")),
    ("江苏省", ("江苏省", "江苏")),
    ("浙江省", ("浙江省", "浙江")),
    ("安徽省", ("安徽省", "安徽")),
    ("福建省", ("福建省", "福建")),
    ("江西省", ("江西省", "江西")),
    ("山东省", ("山东省", "山东")),
    ("河南省", ("河南省", "河南")),
    ("湖北省", ("湖北省", "湖北")),
    ("湖南省", ("湖南省", "湖南")),
    ("广东省", ("广东省", "广东")),
    ("广西壮族自治区", ("广西壮族自治区", "广西")),
    ("海南省", ("海南省", "海南")),
    ("重庆市", ("重庆市", "重庆")),
    ("四川省", ("四川省", "四川")),
    ("贵州省", ("贵州省", "贵州")),
    ("云南省", ("云南省", "云南")),
    ("西藏自治区", ("西藏自治区", "西藏")),
    ("陕西省", ("陕西省", "陕西")),
    ("甘肃省", ("甘肃省", "甘肃")),
    ("青海省", ("青海省", "青海")),
    ("宁夏回族自治区", ("宁夏回族自治区", "宁夏")),
    ("新疆维吾尔自治区", ("新疆维吾尔自治区", "新疆")),
    ("台湾省", ("台湾省", "台湾")),
    ("香港特别行政区", ("香港特别行政区", "香港")),
    ("澳门特别行政区", ("澳门特别行政区", "澳门")),
)


def _canonical_org(value: str) -> str:
    return normalize_name(value).translate(ORG_TRANSLATION)


def _province_match(value: str | None) -> str | None:
    if not value:
        return None
    matches = []
    for province, aliases in PROVINCES:
        for alias in aliases:
            position = value.find(alias)
            if position >= 0:
                matches.append((position, -len(alias), province))
    return min(matches)[2] if matches else None


def _org_province(name: str, address: str | None = None) -> str | None:
    return _province_match(name) or _province_match(address)


def _raw_org_name(field: str, value: str, evidence) -> str:
    label = (
        r"(?:采购人|招标人|名称|名\s*称)"
        if field == "purchaser"
        else r"(?:代理机构|采购代理机构|名称|名\s*称)"
    )
    quotes = evidence if isinstance(evidence, list) else [evidence]
    for quote in quotes:
        if not isinstance(quote, str):
            continue
        match = re.search(label + r"\s*[：:]\s*(?P<name>[^\r\n]+)", quote)
        if match:
            raw = match.group("name")
            raw = re.split(
                r"\s+(?:联系人|联系电话|联系方式|电话|联系地址|地址|邮编|电子邮箱)\s*[：:]",
                raw,
                maxsplit=1,
            )[0]
            if raw.strip():
                return raw.strip()
    return value.strip()


def _amount_value(value):
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return normalize_amount(str(value), default_unit="元")


def _canonical_field(field: str, value):
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


def _field_correct(field: str, prediction, document: dict) -> bool:
    gold = document.get("fields", {}).get(field)
    if field == "project_code":
        accepted = [gold, *(document.get("alternates") or {}).get("project_code", [])]
        return _canonical_field(field, prediction) in {
            _canonical_field(field, value)
            for value in accepted
            if value is not None
        }
    if field == "budget_yuan":
        predicted_amount = _amount_value(prediction)
        gold_amount = _amount_value(gold)
        return (
            predicted_amount is not None
            and gold_amount is not None
            and abs(predicted_amount - gold_amount) < 0.01
        )
    return _canonical_field(field, prediction) == _canonical_field(field, gold)


def _iso_datetime(value: str | None, doc_id: str) -> str:
    if not value:
        raise CommandError(f"Missing publish time for {doc_id}")
    normalized = normalize_datetime(str(value))
    if normalized and len(normalized) >= 10:
        return normalized
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).strftime(
            "%Y-%m-%d %H:%M"
        )
    except ValueError as exc:
        raise CommandError(f"Invalid publish time for {doc_id}: {value!r}") from exc


def _queries(sql_text: str) -> list[tuple[str, str]]:
    sections = re.findall(
        r"(?ms)^-- name:\s*([A-Za-z0-9_]+)\s*$\n(.*?)(?=^-- name:|\Z)",
        sql_text,
    )
    if not sections:
        raise CommandError("No '-- name: <id>' analysis queries found")
    return [(name, query.strip()) for name, query in sections if query.strip()]


def _markdown_table(columns: list[str], rows: list[tuple]) -> str:
    render = lambda value: "" if value is None else str(value).replace("|", "\\|").replace("\n", " ")
    lines = [
        "| " + " | ".join(render(value) for value in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend(
        "| " + " | ".join(render(value) for value in row) + " |"
        for row in rows
    )
    return "\n".join(lines)


class Command(BaseCommand):
    help = "Build the reproducible SQLite tender warehouse and exports."

    def add_arguments(self, parser):
        parser.add_argument("--output", default="/tmp/tender_warehouse.sqlite3")
        parser.add_argument(
            "--export-dir",
            default="interview_materials/results/tender_warehouse",
        )
        parser.add_argument(
            "--labels",
            default="interview_materials/notice_cls/labels.json",
        )
        parser.add_argument(
            "--gold",
            default="interview_materials/tender_eval/gold.json",
        )
        parser.add_argument(
            "--docs-dir",
            default="interview_materials/tender_eval/docs",
        )
        parser.add_argument(
            "--summaries",
            default="interview_materials/tender_warehouse/summaries.json",
        )
        parser.add_argument(
            "--llm-raw",
            default="interview_materials/results/tender_extraction/llm_raw.jsonl",
        )
        parser.add_argument(
            "--schema",
            default="interview_materials/tender_warehouse/schema.sql",
        )
        parser.add_argument(
            "--analysis-sql",
            default="interview_materials/tender_warehouse/analysis.sql",
        )
        parser.add_argument("--expected-notices", type=int, default=150)
        parser.add_argument("--expected-gkzb", type=int, default=60)

    def handle(self, *args, **options):
        repo = Path(settings.BASE_DIR)
        if sqlite3.sqlite_version_info < (3, 25, 0):
            raise CommandError(
                "Tender warehouse requires SQLite 3.25.0 or newer for window functions; "
                f"found {sqlite3.sqlite_version}"
            )

        def path_option(name):
            path = Path(options[name])
            return path if path.is_absolute() else repo / path

        try:
            labels = json.loads(self._read(path_option("labels"), "notice labels"))
            gold_data = json.loads(self._read(path_option("gold"), "tender gold"))
            summaries = json.loads(self._read(path_option("summaries"), "notice summaries"))
        except json.JSONDecodeError as exc:
            raise CommandError(f"Invalid warehouse input JSON: {exc}") from exc
        raw_records = self._read_jsonl(path_option("llm_raw"))
        schema_path = path_option("schema")
        analysis_path = path_option("analysis_sql")
        try:
            schema_sql = schema_path.read_text(encoding="utf-8")
            analysis_text = analysis_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise CommandError(f"Unable to read warehouse SQL: {exc}") from exc

        labels_by_id = {row["doc_id"]: row for row in labels}
        summaries_by_id = {row["doc_id"]: row for row in summaries}
        gold_documents = gold_data["documents"]
        gold_by_id = {row["doc_id"]: row for row in gold_documents}
        raw_by_id = {row["doc_id"]: row for row in raw_records}
        gkzb_documents = [
            row for row in gold_documents if labels_by_id.get(row["doc_id"], {}).get("label") == "gkzb"
        ]
        if (
            len(labels_by_id) != options["expected_notices"]
            or len(gkzb_documents) != options["expected_gkzb"]
        ):
            raise CommandError(
                "Expected "
                f"{options['expected_notices']} notice labels and "
                f"{options['expected_gkzb']} gkzb gold documents"
            )
        labeled_gkzb = {
            doc_id
            for doc_id, row in labels_by_id.items()
            if row.get("label") == "gkzb"
        }
        if set(gold_by_id) != labeled_gkzb:
            raise CommandError("gkzb notice labels and tender gold IDs do not match")
        for doc_id, label in labels_by_id.items():
            if (
                label.get("label") not in {"gkzb", "zbgg", "cjgg", "gzgg"}
                or label.get("split") not in {"dev", "test"}
                or not label.get("url")
                or not label.get("title")
            ):
                raise CommandError(f"Invalid label metadata for {doc_id}")
        if set(summaries_by_id) != set(labels_by_id):
            raise CommandError("Notice summary IDs do not match notice labels")
        if any(doc_id not in labels_by_id for doc_id in gold_by_id):
            raise CommandError("Tender gold contains IDs missing from notice labels")
        missing_raw = [row["doc_id"] for row in gkzb_documents if row["doc_id"] not in raw_by_id]
        if missing_raw:
            raise CommandError(f"Tender LLM replay is missing: {', '.join(missing_raw[:5])}")

        notice_rows = []
        for doc_id, label in labels_by_id.items():
            summary = summaries_by_id.get(doc_id)
            if not summary:
                raise CommandError(f"Missing summary for {doc_id}")
            summary_fields = summary.get("summary") or {}
            for key in (
                "公告时间",
                "采购单位",
                "采购单位地址",
                "代理机构名称",
                "行政区域",
            ):
                if not str(summary_fields.get(key) or "").strip():
                    raise CommandError(f"Missing required summary field {key} for {doc_id}")
            if not label.get("url") or not label.get("title"):
                raise CommandError(f"Missing URL/title for {doc_id}")
            notice_rows.append(
                (
                    doc_id,
                    label["label"],
                    label["url"],
                    label["title"],
                    _iso_datetime(label.get("publish_time"), doc_id),
                    label["split"],
                )
            )

        extraction_by_id = {}
        org_entries = defaultdict(list)
        docs_dir = path_option("docs_dir")
        for document in sorted(gkzb_documents, key=lambda row: row["doc_id"]):
            doc_id = document["doc_id"]
            if doc_id not in labels_by_id or doc_id not in summaries_by_id:
                raise CommandError(f"Missing notice metadata for tender {doc_id}")
            try:
                text = (docs_dir / f"{doc_id}.txt").read_text(encoding="utf-8")
            except OSError as exc:
                raise CommandError(f"Unable to read tender body {doc_id}: {exc}") from exc

            rule_fields = extract_rules(text)
            raw_record = raw_by_id[doc_id]
            dropped = []
            if raw_record.get("error") or not isinstance(raw_record.get("response"), dict):
                llm_fields = {
                    field: FieldValue(None, None, "llm")
                    for field in FIELDS
                }
            else:
                llm_fields, _ = extract_llm(
                    text,
                    None,
                    response=raw_record["response"],
                    usage_override=raw_record.get("usage"),
                )
                llm_fields, dropped = verify_evidence(llm_fields, text)
            result = build_extraction_result(
                "hybrid",
                rule_fields=rule_fields,
                llm_fields=llm_fields,
                usage=(raw_record.get("usage") or {}),
                evidence_dropped=dropped,
            )
            extraction_by_id[doc_id] = result
            summary_fields = summaries_by_id[doc_id].get("summary") or {}
            purchaser_address = summary_fields.get("采购单位地址")
            for field in ("purchaser", "agency"):
                name = result["fields"].get(field)
                if not isinstance(name, str) or not name.strip():
                    continue
                raw_name = _raw_org_name(
                    field,
                    name,
                    result["evidence"].get(field),
                )
                canonical = _canonical_org(raw_name)
                address = purchaser_address if field == "purchaser" else None
                org_entries[canonical].append(
                    {
                        "doc_id": doc_id,
                        "role": field,
                        "alias": raw_name,
                        "province": _org_province(raw_name, address),
                    }
                )

        output_path = path_option("output")
        export_dir = path_option("export_dir")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        export_dir.mkdir(parents=True, exist_ok=True)
        try:
            conn = sqlite3.connect(output_path)
            conn.execute("PRAGMA foreign_keys = ON")
            conn.executescript(schema_sql)
            org_ids = {}
            alias_to_org = {}
            for canonical in sorted(org_entries):
                entries = sorted(
                    org_entries[canonical],
                    key=lambda entry: (entry["doc_id"], entry["role"], entry["alias"]),
                )
                alias_counts = Counter(entry["alias"] for entry in entries)
                display_name = sorted(
                    alias_counts,
                    key=lambda alias: (-alias_counts[alias], alias),
                )[0]
                province = next(
                    (entry["province"] for entry in entries if entry["province"]),
                    None,
                )
                cursor = conn.execute(
                    "INSERT INTO organization(canonical_name, display_name, province) "
                    "VALUES (?, ?, ?)",
                    (canonical, display_name, province),
                )
                org_id = cursor.lastrowid
                org_ids[canonical] = org_id
                for alias in sorted(alias_counts):
                    previous = alias_to_org.get(alias)
                    if previous is not None and previous != org_id:
                        raise CommandError(f"Organization alias maps to multiple keys: {alias}")
                    alias_to_org[alias] = org_id
                    conn.execute(
                        "INSERT OR IGNORE INTO organization_alias(alias, org_id) VALUES (?, ?)",
                        (alias, org_id),
                    )

            conn.executemany(
                "INSERT INTO notice(doc_id, notice_type, url, title, publish_time, split) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                notice_rows,
            )
            for document in sorted(gkzb_documents, key=lambda row: row["doc_id"]):
                doc_id = document["doc_id"]
                result = extraction_by_id[doc_id]
                values = result["fields"]
                purchaser = values.get("purchaser")
                agency = values.get("agency")
                purchaser_id = org_ids.get(_canonical_org(purchaser)) if isinstance(purchaser, str) else None
                agency_id = org_ids.get(_canonical_org(agency)) if isinstance(agency, str) else None
                conn.execute(
                    "INSERT INTO tender_project(doc_id, project_name, project_code, purchaser_id, "
                    "agency_id, budget_yuan, bid_deadline, needs_review) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        doc_id,
                        values.get("project_name"),
                        values.get("project_code"),
                        purchaser_id,
                        agency_id,
                        _amount_value(values.get("budget_yuan")),
                        values.get("bid_deadline"),
                        int(bool(result["needs_review"])),
                    ),
                )
                for field in FIELDS:
                    value = values.get(field)
                    evidence = result["evidence"].get(field)
                    if isinstance(evidence, list):
                        evidence_text = json.dumps(evidence, ensure_ascii=False)
                    elif isinstance(evidence, str):
                        evidence_text = evidence
                    else:
                        evidence_text = None
                    conn.execute(
                        "INSERT INTO field_extraction(doc_id, field, value, evidence, source, "
                        "in_review, gold, correct) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            doc_id,
                            field,
                            None if value is None else str(value),
                            evidence_text,
                            result["sources"].get(field),
                            int(field in result["needs_review"]),
                            (
                                None
                                if document.get("fields", {}).get(field) is None
                                else json.dumps(
                                    document.get("fields", {}).get(field),
                                    ensure_ascii=False,
                                )
                            ),
                            int(_field_correct(field, value, document)),
                        ),
                    )
            conn.commit()
            query_results = []
            for name, query in _queries(analysis_text):
                try:
                    cursor = conn.execute(query)
                    query_results.append(
                        (name, query, [column[0] for column in cursor.description], cursor.fetchall())
                    )
                except sqlite3.Error as exc:
                    raise CommandError(f"Analysis query {name} failed: {exc}") from exc
            export_names = (
                "organization",
                "organization_alias",
                "notice",
                "tender_project",
                "field_extraction",
                "v_review_queue",
                "v_project_flat",
            )
            for table in export_names:
                cursor = conn.execute(f"SELECT * FROM {table}")
                columns = [column[0] for column in cursor.description]
                with (export_dir / f"{table}.csv").open(
                    "w", encoding="utf-8-sig", newline=""
                ) as output:
                    writer = csv.writer(output, lineterminator="\n")
                    writer.writerow(columns)
                    writer.writerows(cursor.fetchall())
            report_lines = [
                "# Tender warehouse report",
                "",
                "Publish time is used as an approximation of notice issue time for deadline analysis.",
                "",
            ]
            for name, query, columns, rows in query_results:
                report_lines.extend(
                    [
                        f"## {name}",
                        "",
                        "```sql",
                        query,
                        "```",
                        "",
                        _markdown_table(columns, rows),
                        "",
                    ]
                )
            (export_dir / "report.md").write_text(
                "\n".join(report_lines).rstrip() + "\n",
                encoding="utf-8",
            )
        except sqlite3.Error as exc:
            raise CommandError(f"Unable to build tender warehouse: {exc}") from exc
        finally:
            if "conn" in locals():
                conn.close()
        self.stdout.write(
            f"Built tender warehouse at {output_path}; exports and report: {export_dir}"
        )

    @staticmethod
    def _read(path: Path, description: str) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except OSError as exc:
            raise CommandError(f"Unable to read {description}: {exc}") from exc

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict]:
        records = []
        seen = set()
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise CommandError(f"Unable to read tender LLM replay: {exc}") from exc
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CommandError(
                    f"Invalid tender LLM JSON on line {line_number}: {exc}"
                ) from exc
            doc_id = record.get("doc_id") if isinstance(record, dict) else None
            if not doc_id or doc_id in seen:
                raise CommandError(
                    f"Invalid or duplicate tender LLM doc_id on line {line_number}"
                )
            seen.add(doc_id)
            records.append(record)
        return records
