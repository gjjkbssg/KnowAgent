import json
import sqlite3
import tempfile
from pathlib import Path

from django.core.management import call_command
from django.core.management.base import CommandError
from django.test import SimpleTestCase


class TenderWarehouseTests(SimpleTestCase):
    def test_two_document_warehouse_and_failed_rebuild_preserves_existing_db(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            docs_dir = root / "docs"
            docs_dir.mkdir()
            export_dir = root / "exports"
            db_path = root / "warehouse.sqlite3"
            labels = []
            summaries = []
            gold_documents = []
            raw_records = []
            bodies = (
                (
                    "gkzb_fixture_1",
                    "名称：ＡＢＣ科技有限公司 (合肥)",
                    "正确项目名",
                ),
                (
                    "gkzb_fixture_2",
                    "名称：ABC科技有限公司（合肥）",
                    "故意不一致的项目名",
                ),
            )
            for index, (doc_id, purchaser, gold_name) in enumerate(bodies, start=1):
                text = f"""项目名称：正确项目名
项目编号：ABC-{index}
预算金额（元）：100
提交投标文件截止时间：2026年12月1日09时00分
一、采购人信息
{purchaser}
二、采购代理机构信息
名称：示例代理机构
"""
                (docs_dir / f"{doc_id}.txt").write_text(text, encoding="utf-8")
                labels.append(
                    {
                        "doc_id": doc_id,
                        "label": "gkzb",
                        "split": "test",
                        "url": f"https://example.invalid/{index}",
                        "title": f"测试招标公告{index}",
                        "publish_time": "2026-11-24",
                    }
                )
                summaries.append(
                    {
                        "doc_id": doc_id,
                        "title": f"测试招标公告{index}",
                        "summary": {
                            "公告时间": "2026年11月24日",
                            "采购单位": "ＡＢＣ科技有限公司",
                            "采购单位地址": "安徽省合肥市",
                            "代理机构名称": "示例代理机构",
                            "行政区域": "安徽省",
                        },
                    }
                )
                gold_documents.append(
                    {
                        "doc_id": doc_id,
                        "split": "test",
                        "url": f"https://example.invalid/{index}",
                        "fields": {
                            "project_name": gold_name,
                            "project_code": f"ABC-{index}",
                            "purchaser": "ABC科技有限公司（合肥）",
                            "agency": "示例代理机构",
                            "budget_yuan": 100,
                            "bid_deadline": "2026-12-01 09:00",
                        },
                        "alternates": {},
                    }
                )
                raw_records.append({"doc_id": doc_id, "error": "fixture replay"})

            labels_path = root / "labels.json"
            labels_path.write_text(json.dumps(labels, ensure_ascii=False), encoding="utf-8")
            summaries_path = root / "summaries.json"
            summaries_path.write_text(
                json.dumps(summaries, ensure_ascii=False),
                encoding="utf-8",
            )
            gold_path = root / "gold.json"
            gold_path.write_text(
                json.dumps({"documents": gold_documents}, ensure_ascii=False),
                encoding="utf-8",
            )
            replay_path = root / "llm_raw.jsonl"
            replay_path.write_text(
                "".join(json.dumps(record) + "\n" for record in raw_records),
                encoding="utf-8",
            )

            call_command(
                "build_tender_warehouse",
                output=str(db_path),
                export_dir=str(export_dir),
                labels=str(labels_path),
                gold=str(gold_path),
                docs_dir=str(docs_dir),
                summaries=str(summaries_path),
                llm_raw=str(replay_path),
                expected_notices=2,
                expected_gkzb=2,
                verbosity=0,
            )

            with sqlite3.connect(db_path) as connection:
                aliases = connection.execute(
                    "SELECT COUNT(*) FROM organization_alias "
                    "WHERE org_id = (SELECT org_id FROM organization "
                    "WHERE canonical_name = ?)",
                    ("ABC科技有限公司（合肥）",),
                ).fetchone()[0]
                canonical_count = connection.execute(
                    "SELECT COUNT(*) FROM organization "
                    "WHERE canonical_name = ?",
                    ("ABC科技有限公司（合肥）",),
                ).fetchone()[0]
                review_count = connection.execute(
                    "SELECT COUNT(*) FROM v_review_queue WHERE doc_id = ?",
                    ("gkzb_fixture_2",),
                ).fetchone()[0]
                query_count = connection.execute(
                    "SELECT COUNT(*) FROM field_extraction"
                ).fetchone()[0]
            self.assertEqual(canonical_count, 1)
            self.assertEqual(aliases, 2)
            self.assertGreater(review_count, 0)
            self.assertEqual(query_count, 12)
            self.assertEqual((export_dir / "notice.csv").read_bytes()[:3], b"\xef\xbb\xbf")
            report = (export_dir / "report.md").read_text(encoding="utf-8")
            self.assertIn("## data_quality", report)
            self.assertIn("julianday", report)

            schema_path = root / "schema_with_failed_insert.sql"
            schema = (
                Path(__file__).resolve().parents[1]
                / "interview_materials"
                / "tender_warehouse"
                / "schema.sql"
            ).read_text(encoding="utf-8")
            schema_path.write_text(
                schema
                + """
CREATE TRIGGER fail_notice_insert
BEFORE INSERT ON notice
BEGIN
    SELECT RAISE(ABORT, 'forced insertion failure');
END;
""",
                encoding="utf-8",
            )
            with self.assertRaises(CommandError):
                call_command(
                    "build_tender_warehouse",
                    output=str(db_path),
                    export_dir=str(export_dir),
                    labels=str(labels_path),
                    gold=str(gold_path),
                    docs_dir=str(docs_dir),
                    summaries=str(summaries_path),
                    llm_raw=str(replay_path),
                    schema=str(schema_path),
                    expected_notices=2,
                    expected_gkzb=2,
                    verbosity=0,
                )

            with sqlite3.connect(db_path) as connection:
                existing_notices = connection.execute(
                    "SELECT COUNT(*) FROM notice"
                ).fetchone()[0]
            self.assertEqual(existing_notices, 2)
            self.assertEqual(list(root.glob(".warehouse.sqlite3.*.tmp")), [])
