import json
import tempfile
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from django.core.management import call_command
from django.test import SimpleTestCase, override_settings

from personal_knowledge_base.extraction.normalize import (
    chinese_amount_to_yuan,
    normalize_amount,
    normalize_code,
    normalize_company,
    normalize_datetime,
    normalize_name,
)
from personal_knowledge_base.extraction.tender import (
    FIELDS,
    FieldValue,
    SYSTEM_PROMPT,
    TOOL,
    _budget_value,
    build_extraction_result,
    extract,
    extract_llm,
    extract_rules,
    validate,
    verify_evidence,
)
from personal_knowledge_base.management.commands.run_tender_extraction_eval import (
    _field_rows,
    _is_correct,
    score_predictions,
)


class TenderNormalizationTests(SimpleTestCase):
    def test_jobprep_amount_examples_are_preserved(self):
        self.assertEqual(
            chinese_amount_to_yuan("壹佰贰拾万零伍佰元整"),
            1200500,
        )
        self.assertEqual(chinese_amount_to_yuan("叁亿肆仟万元"), 340000000)
        self.assertEqual(
            chinese_amount_to_yuan("伍万陆仟柒佰捌拾玖元伍角陆分"),
            56789.56,
        )
        self.assertEqual(normalize_amount("预算金额：1,234.5万元"), 12345000)
        self.assertEqual(normalize_amount("最高限价：人民币壹佰万元整"), 1000000)

    def test_amount_formats_and_rounding(self):
        self.assertEqual(normalize_amount("263.720000(万元)"), 2637200.0)
        self.assertEqual(normalize_amount("263.720000（万元）"), 2637200.0)
        self.assertEqual(
            normalize_amount("预算金额（元）：263.720000", default_unit="元"),
            263.72,
        )
        self.assertEqual(normalize_amount("￥160.000000万元"), 1600000.0)
        self.assertEqual(normalize_amount("378400.00000000006元"), 378400.0)

    def test_datetime_examples_and_new_formats(self):
        self.assertEqual(
            normalize_datetime("投标截止时间：2026年3月5日下午2时30分"),
            "2026-03-05 14:30",
        )
        self.assertEqual(
            normalize_datetime("开标时间 2026-03-05 09:30"),
            "2026-03-05 09:30",
        )
        self.assertEqual(
            normalize_datetime("2026年12月15日 09时30分00秒"),
            "2026-12-15 09:30",
        )
        self.assertEqual(
            normalize_datetime("2026-12-1509:00:00"),
            "2026-12-15 09:00",
        )
        self.assertEqual(normalize_datetime("10点00分"), "10:00")

    def test_code_and_name_normalization(self):
        self.assertEqual(
            normalize_company("安徽省招标集团股份有限公司 (合肥)"),
            "安徽省招标集团股份有限公司（合肥）",
        )
        self.assertEqual(normalize_company("ＡＢＣ科技有限公司"), "ABC科技有限公司")
        self.assertEqual(
            normalize_name("ＡＢＣ科技有限公司 (合肥)"),
            "ABC科技有限公司（合肥）",
        )
        self.assertEqual(normalize_code("ＡＢＣ　12 ３"), "ABC123")


class TenderRuleTests(SimpleTestCase):
    def test_rules_extract_labeled_fields_and_single_budget(self):
        text = """项目编号：ABC-2026
项目名称：测试项目
预算金额（元）：123456.78
四、提交投标文件截止时间、开标时间和地点
提交投标文件截止时间：2026年12月15日09时30分00秒
1.采购人信息
名 称：测试采购人
2.采购代理机构信息
名 称：测试代理公司
"""
        fields = extract_rules(text)
        self.assertEqual(fields["project_name"].value, "测试项目")
        self.assertEqual(fields["project_code"].value, "ABC-2026")
        self.assertEqual(fields["purchaser"].value, "测试采购人")
        self.assertEqual(fields["agency"].value, "测试代理公司")
        self.assertEqual(fields["budget_yuan"].value, 123456.78)
        self.assertEqual(fields["bid_deadline"].value, "2026-12-15 09:30")
        self.assertEqual(fields["budget_yuan"].source, "rule")

    def test_project_budget_precedes_package_budgets(self):
        text = """预算金额（元）：800000
标项名称：项目包一
预算金额（元）：300000
标项名称：项目包二
预算金额（元）：500000
"""
        fields = extract_rules(text)
        self.assertEqual(fields["budget_yuan"].value, 800000.0)

    def test_package_budgets_are_summed_when_no_project_total_exists(self):
        text = """包组编号：001
预算金额（元）：130000
包组编号：002
预算金额（元）：480000
包组编号：003
预算金额（元）：390000
"""
        fields = extract_rules(text)
        self.assertEqual(fields["budget_yuan"].value, 1000000.0)

    def test_rule_package_budget_sum_retains_all_evidence_quotes(self):
        text = """包组编号：001
预算金额（元）：130000
包组编号：002
预算金额（元）：480000
"""

        value = _budget_value(text)

        self.assertEqual(value.value, 610000.0)
        self.assertEqual(
            value.evidence,
            ["预算金额（元）：130000", "预算金额（元）：480000"],
        )

    def test_table_package_budget_sum_retains_each_row(self):
        text = """本包预算总金额（元）
包一 130000
包二 480000
"""

        value = _budget_value(text)

        self.assertEqual(value.value, 610000.0)
        self.assertEqual(value.evidence, ["包一 130000", "包二 480000"])

    def test_single_project_budget_keeps_string_evidence(self):
        value = _budget_value("预算金额（元）：800000")

        self.assertEqual(value.value, 800000.0)
        self.assertEqual(value.evidence, "预算金额（元）：800000")

    def test_hubei_deadline_uses_cutoff_not_file_start_time(self):
        text = """四、提交投标文件截止时间、开标时间和地点
1、开始时间：2026年11月25日00点00分（北京时间）
2、截止时间：2026年12月16日10点00分（北京时间）
"""
        self.assertEqual(
            extract_rules(text)["bid_deadline"].value,
            "2026-12-16 10:00",
        )

    def test_evidence_verification_drops_fabricated_string_value(self):
        fields = {
            "project_name": FieldValue("虚构项目", "公告中没有的证据", "llm"),
        }
        kept, dropped = verify_evidence(fields, "项目名称：真实项目")
        self.assertIsNone(kept["project_name"].value)
        self.assertEqual(dropped, ["project_name"])

    def test_evidence_verification_drops_truncated_project_code(self):
        quote = "项目编号：ABC1234"
        fields = {"project_code": FieldValue("ABC123", quote, "llm")}

        kept, dropped = verify_evidence(fields, quote)

        self.assertIsNone(kept["project_code"].value)
        self.assertEqual(dropped, ["project_code"])

    def test_evidence_verification_drops_deadline_not_in_quote(self):
        quote = "提交投标文件截止时间：2026年12月15日10时00分"
        fields = {
            "bid_deadline": FieldValue("2026-12-15 09:00", quote, "llm"),
        }

        kept, dropped = verify_evidence(fields, quote)

        self.assertIsNone(kept["bid_deadline"].value)
        self.assertEqual(dropped, ["bid_deadline"])

    def test_evidence_verification_accepts_hubei_cutoff_after_start_time(self):
        quote = (
            "1、开始时间：2026年11月25日00点00分（北京时间）"
            "2、截止时间：2026年12月16日10点00分（北京时间）"
        )
        fields = {
            "bid_deadline": FieldValue("2026-12-16 10:00", quote, "llm"),
        }

        kept, dropped = verify_evidence(fields, quote)

        self.assertEqual(kept["bid_deadline"].value, "2026-12-16 10:00")
        self.assertEqual(dropped, [])

    def test_evidence_verification_accepts_budget_sum_from_package_quotes(self):
        quotes = ["第1包预算金额：100万元", "第2包预算金额：250万元"]
        fields = {
            "budget_yuan": FieldValue(3500000.0, quotes, "llm"),
        }

        kept, dropped = verify_evidence(fields, "\n".join(quotes))

        self.assertEqual(kept["budget_yuan"].value, 3500000.0)
        self.assertEqual(dropped, [])

    def test_evidence_verification_requires_each_budget_quote_in_body(self):
        fields = {
            "budget_yuan": FieldValue(
                3500000.0,
                ["第1包预算金额：100万元", "第2包预算金额：250万元"],
                "llm",
            ),
        }

        kept, dropped = verify_evidence(fields, "第1包预算金额：100万元")

        self.assertIsNone(kept["budget_yuan"].value)
        self.assertEqual(dropped, ["budget_yuan"])

    def test_evidence_verification_drops_budget_not_in_quote(self):
        quote = "预算金额（元）：100"
        fields = {"budget_yuan": FieldValue(200.0, quote, "llm")}

        kept, dropped = verify_evidence(fields, quote)

        self.assertIsNone(kept["budget_yuan"].value)
        self.assertEqual(dropped, ["budget_yuan"])

    def test_evidence_verification_accepts_string_budget_evidence(self):
        quote = "预算金额（元）：100"
        fields = {"budget_yuan": FieldValue(100.0, quote, "llm")}

        kept, dropped = verify_evidence(fields, quote)

        self.assertEqual(kept["budget_yuan"].value, 100.0)
        self.assertEqual(dropped, [])

    def test_evidence_verification_accepts_bare_yuan_in_table_row(self):
        quote = (
            "湘潭大学机力院先进制造与智能化控制实验平台建设项目 "
            "1565195 其他仪器仪表 详见招标文件 1 1565195"
        )
        fields = {"budget_yuan": FieldValue(1565195.0, quote, "llm")}

        kept, dropped = verify_evidence(fields, quote)

        self.assertEqual(kept["budget_yuan"].value, 1565195.0)
        self.assertEqual(dropped, [])

    def test_evidence_verification_rejects_unrelated_bare_table_amounts(self):
        quote = "其他采购项目 886730.1 教学设备 详见招标文件 1 886730.1"
        fields = {"budget_yuan": FieldValue(1565195.0, quote, "llm")}

        kept, dropped = verify_evidence(fields, quote)

        self.assertIsNone(kept["budget_yuan"].value)
        self.assertEqual(dropped, ["budget_yuan"])

    @patch("personal_knowledge_base.extraction.tender.extract_llm")
    def test_hybrid_keeps_rule_on_conflict_and_requests_review(self, extract_llm_mock):
        text = "预算金额（元）：100\n项目预算金额（元）：200"
        extract_llm_mock.return_value = (
            {
                "budget_yuan": FieldValue(
                    200.0,
                    "项目预算金额（元）：200",
                    "llm",
                )
            },
            {"total_tokens": 10},
        )
        result = extract(text, None, "hybrid")
        self.assertEqual(result["fields"]["budget_yuan"], 100.0)
        self.assertIn("budget_yuan", result["needs_review"])
        self.assertEqual(result["sources"]["budget_yuan"], "rule")
        extract_llm_mock.assert_called_once()

    def test_validation_marks_invalid_budget_deadline_and_missing_names(self):
        issues = validate(
            {
                "budget_yuan": 0,
                "bid_deadline": "2036-01-01 09:00",
                "purchaser": "",
                "project_name": "",
            }
        )
        self.assertEqual(len(issues), 4)


class TenderLLMTests(SimpleTestCase):
    def test_budget_evidence_schema_and_prompt(self):
        properties = TOOL["function"]["parameters"]["properties"]
        self.assertEqual(
            properties["budget_yuan"]["properties"]["evidence"],
            {"type": ["array", "null"], "items": {"type": "string"}},
        )
        self.assertEqual(
            properties["project_name"]["properties"]["evidence"],
            {"type": ["string", "null"]},
        )
        self.assertIn(
            "1. 每个字段同时给出 value 和 evidence。除 budget_yuan 外，"
            "evidence 是从正文中原样复制的一段连续文字（不超过 80 字），且必须包含 value 本身。",
            SYSTEM_PROMPT,
        )
        self.assertIn(
            "6. budget_yuan：项目预算总金额，换算成元，只输出数字。evidence 是字符串数组："
            "有项目总预算时，给出写有总预算的 1 段原文；只有各包预算时，给出每个包预算的原文各 1 段，"
            "value 为各包之和。",
            SYSTEM_PROMPT,
        )

    @patch("personal_knowledge_base.extraction.tender.model_providers.chat_completion_raw")
    def test_llm_uses_function_call_and_normalizes_values(self, chat_completion_raw):
        values = {
            field: {"value": None, "evidence": None}
            for field in FIELDS
        }
        values.update(
            {
                "project_name": {
                    "value": "ＡＢＣ 项目",
                    "evidence": "项目名称：ＡＢＣ 项目",
                },
                "project_code": {"value": "ＡＢＣ １２３", "evidence": "编号：ＡＢＣ １２３"},
                "budget_yuan": {"value": "263.72万元", "evidence": "预算金额：263.72万元"},
                "bid_deadline": {
                    "value": "2026年12月15日 09时30分00秒",
                    "evidence": "截止时间：2026年12月15日 09时30分00秒",
                },
            }
        )
        chat_completion_raw.return_value = {
            "content": None,
            "tool_calls": [
                {
                    "function": {
                        "name": "submit_tender_fields",
                        "arguments": json.dumps(values, ensure_ascii=False),
                    }
                }
            ],
        }
        fields, usage = extract_llm("测试公告正文", None)
        chat_completion_raw.assert_called_once()
        call = chat_completion_raw.call_args
        self.assertEqual(call.kwargs["tools"], [TOOL])
        self.assertEqual(call.kwargs["temperature"], 0)
        self.assertEqual(call.args[1][0]["content"], SYSTEM_PROMPT)
        self.assertEqual(call.args[1][1]["content"], "公告正文：\n测试公告正文")
        self.assertEqual(fields["project_name"].value, "ABC项目")
        self.assertEqual(fields["project_code"].value, "ABC123")
        self.assertEqual(fields["budget_yuan"].value, 2637200.0)
        self.assertEqual(
            fields["budget_yuan"].evidence,
            ["预算金额：263.72万元"],
        )
        self.assertEqual(fields["bid_deadline"].value, "2026-12-15 09:30")
        self.assertEqual(fields["project_code"].source, "llm")
        self.assertEqual(usage["total_tokens"], 0)

    @patch("personal_knowledge_base.extraction.tender.model_providers.chat_completion_raw")
    def test_llm_preserves_package_budget_evidence_array(self, chat_completion_raw):
        quotes = ["第1包预算金额：100万元", "第2包预算金额：250万元"]
        values = {
            field: {"value": None, "evidence": None}
            for field in FIELDS
        }
        values["budget_yuan"] = {"value": 3500000, "evidence": quotes}
        chat_completion_raw.return_value = {
            "content": None,
            "tool_calls": [
                {
                    "function": {
                        "name": "submit_tender_fields",
                        "arguments": json.dumps(values, ensure_ascii=False),
                    }
                }
            ],
        }

        fields, _ = extract_llm("\n".join(quotes), None)
        verified, dropped = verify_evidence(fields, "\n".join(quotes))

        self.assertEqual(fields["budget_yuan"].evidence, quotes)
        self.assertEqual(verified["budget_yuan"].value, 3500000.0)
        self.assertEqual(dropped, [])

    def test_budget_evidence_is_serialized_in_json_and_csv_rows(self):
        quotes = ["第1包预算金额：100万元", "第2包预算金额：250万元"]
        result = build_extraction_result(
            "llm",
            llm_fields={"budget_yuan": FieldValue(3500000.0, quotes, "llm")},
        )
        serialized = json.dumps(result, ensure_ascii=False)
        self.assertEqual(
            json.loads(serialized)["evidence"]["budget_yuan"],
            quotes,
        )

        document = {
            "doc_id": "toy",
            "split": "dev",
            "fields": {"budget_yuan": 3500000.0},
        }
        prediction = {
            "fields": result["fields"],
            "evidence": result["evidence"],
            "needs_review": result["needs_review"],
            "sources": result["sources"],
        }
        budget_row = next(
            row
            for row in _field_rows(document, prediction, "llm")
            if row["field"] == "budget_yuan"
        )
        self.assertEqual(
            json.loads(budget_row["evidence"]),
            quotes,
        )

    @patch("personal_knowledge_base.extraction.tender.model_providers.chat_completion_raw")
    def test_llm_falls_back_to_json_content_without_tool_call(self, chat_completion_raw):
        response = {
            field: {"value": None, "evidence": None}
            for field in FIELDS
        }
        response["project_code"] = {
            "value": "CODE-1",
            "evidence": "项目编号：CODE-1",
        }
        chat_completion_raw.return_value = {
            "content": json.dumps(response, ensure_ascii=False),
            "tool_calls": [],
            "usage": {"prompt_tokens": 8, "completion_tokens": 3, "total_tokens": 11},
        }
        fields, usage = extract_llm("项目编号：CODE-1", None)
        self.assertEqual(fields["project_code"].value, "CODE-1")
        self.assertEqual(usage["total_tokens"], 11)


class TenderScoringTests(SimpleTestCase):
    def test_two_document_metrics_and_review_catch_rate(self):
        values = {
            "project_name": "Alpha project",
            "project_code": "A-1",
            "purchaser": "Buyer",
            "agency": "Agency",
            "budget_yuan": 100.0,
            "bid_deadline": "2026-12-01 09:00",
        }
        gold = [
            {
                "doc_id": "doc-1",
                "split": "dev",
                "fields": dict(values),
                "alternates": {},
            },
            {
                "doc_id": "doc-2",
                "split": "dev",
                "fields": dict(values),
                "alternates": {},
            },
        ]
        wrong = {
            "project_name": "Wrong project",
            "project_code": "WRONG",
            "purchaser": "Wrong buyer",
            "agency": "Wrong agency",
            "budget_yuan": 200.0,
            "bid_deadline": "2027-12-01 09:00",
        }
        predictions = [
            {
                "doc_id": "doc-1",
                "fields": dict(values),
                "needs_review": [],
                "latency_ms": 10,
                "tokens": 100,
                "evidence_dropped": ["agency"],
            },
            {
                "doc_id": "doc-2",
                "fields": wrong,
                "needs_review": list(FIELDS),
                "latency_ms": 30,
                "tokens": 300,
                "evidence_dropped": ["budget_yuan", "bid_deadline"],
            },
        ]

        scores = score_predictions(gold, predictions)
        self.assertEqual(scores["micro"], {
            "tp": 6,
            "fp": 6,
            "fn": 6,
            "precision": 0.5,
            "recall": 0.5,
            "f1": 0.5,
        })
        self.assertEqual(scores["per_field"]["project_code"]["f1"], 0.5)
        self.assertEqual(scores["document_exact_match_rate"], 0.5)
        self.assertEqual(scores["mean_latency_ms_per_doc"], 20)
        self.assertEqual(scores["mean_tokens_per_doc"], 200)
        self.assertEqual(scores["evidence_dropped_count"], 3)
        self.assertEqual(scores["review_rate"], 0.5)
        self.assertEqual(scores["review_catch_rate"], 1.0)

    def test_project_code_alternate_is_accepted(self):
        document = {
            "doc_id": "doc-1",
            "fields": {"project_code": "MAIN"},
            "alternates": {"project_code": ["ALT"]},
        }
        self.assertTrue(_is_correct("project_code", "ALT", document))


class TenderReplayTests(SimpleTestCase):
    def test_two_document_replay_copies_raw_response_latency_and_tokens(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            docs_dir = root / "interview_materials" / "tender_eval" / "docs"
            docs_dir.mkdir(parents=True)
            documents = []
            raw_records = []
            for index, split in ((1, "dev"), (2, "test")):
                name = f"项目{index}"
                code = f"CODE-{index}"
                purchaser = f"采购人{index}公司"
                agency = f"代理机构{index}公司"
                budget = 100000 * index
                deadline = f"2026-12-0{index} 09:00"
                date_text = f"2026年12月0{index}日09时00分"
                evidence = {
                    "project_name": f"项目名称：{name}",
                    "project_code": f"项目编号：{code}",
                    "purchaser": f"采购人：{purchaser}",
                    "agency": f"代理机构：{agency}",
                    "budget_yuan": f"预算金额（元）：{budget}",
                    "bid_deadline": f"提交投标文件截止时间：{date_text}",
                }
                values = {
                    "project_name": name,
                    "project_code": code,
                    "purchaser": purchaser,
                    "agency": agency,
                    "budget_yuan": budget,
                    "bid_deadline": deadline,
                }
                body = "\n".join(evidence.values())
                doc_id = f"doc-{index}"
                (docs_dir / f"{doc_id}.txt").write_text(body, encoding="utf-8")
                documents.append(
                    {
                        "doc_id": doc_id,
                        "split": split,
                        "fields": values,
                        "alternates": {},
                    }
                )
                tool_fields = {
                    field: {"value": values[field], "evidence": evidence[field]}
                    for field in FIELDS
                }
                response = {
                    "content": None,
                    "tool_calls": [
                        {
                            "function": {
                                "name": "submit_tender_fields",
                                "arguments": json.dumps(tool_fields, ensure_ascii=False),
                            }
                        }
                    ],
                    "finish_reason": "tool_calls",
                }
                total_tokens = index * 100
                raw_records.append(
                    {
                        "doc_id": doc_id,
                        "response": response,
                        "latency_ms": index * 12.5,
                        "rule_latency_ms": index * 1.25,
                        "usage": {
                            "prompt_tokens": total_tokens - 10,
                            "completion_tokens": 10,
                            "total_tokens": total_tokens,
                        },
                    }
                )

            dataset_path = root / "interview_materials" / "tender_eval" / "gold.json"
            dataset_path.write_text(
                json.dumps({"documents": documents}, ensure_ascii=False),
                encoding="utf-8",
            )
            replay_path = root / "llm_raw.jsonl"
            replay_path.write_text(
                "\n".join(
                    json.dumps(record, ensure_ascii=False) for record in raw_records
                )
                + "\n",
                encoding="utf-8",
            )
            output_dir = root / "results"
            tenant = type("TenantFixture", (), {"pk": "tenant-1"})()

            with override_settings(BASE_DIR=root):
                with (
                    patch(
                        "personal_knowledge_base.management.commands."
                        "run_tender_extraction_eval.Tenant.objects.order_by"
                    ) as order_by,
                    patch(
                        "personal_knowledge_base.extraction.tender."
                        "model_providers.chat_completion_raw"
                    ) as chat_completion_raw,
                ):
                    order_by.return_value.first.return_value = tenant
                    call_command(
                        "run_tender_extraction_eval",
                        methods="rules,llm,hybrid",
                        split="all",
                        output_dir=str(output_dir),
                        replay_llm=str(replay_path),
                        stdout=StringIO(),
                    )

            chat_completion_raw.assert_not_called()
            summary = json.loads(
                (output_dir / "summary.json").read_text(encoding="utf-8")
            )
            llm_predictions = {
                prediction["doc_id"]: prediction
                for prediction in summary["per_document"]
                if prediction["method"] == "llm"
            }
            hybrid_predictions = {
                prediction["doc_id"]: prediction
                for prediction in summary["per_document"]
                if prediction["method"] == "hybrid"
            }
            self.assertEqual(llm_predictions["doc-1"]["latency_ms"], 12.5)
            self.assertEqual(llm_predictions["doc-2"]["latency_ms"], 25.0)
            self.assertEqual(llm_predictions["doc-1"]["tokens"], 100)
            self.assertEqual(llm_predictions["doc-2"]["tokens"], 200)
            self.assertEqual(hybrid_predictions["doc-1"]["latency_ms"], 13.75)
            self.assertEqual(hybrid_predictions["doc-2"]["latency_ms"], 27.5)
            self.assertEqual(
                llm_predictions["doc-1"]["fields"]["project_code"],
                "CODE-1",
            )
            saved_records = [
                json.loads(line)
                for line in (output_dir / "llm_raw.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(saved_records, raw_records)
