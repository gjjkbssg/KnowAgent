import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from django.core.management import call_command
from django.test import SimpleTestCase

from personal_knowledge_base.extraction.classify import (
    LABELS,
    TOOL,
    classify_llm,
    classify_rules,
    predict_tfidf,
    top_features,
    train_tfidf,
)


class NoticeRuleTests(SimpleTestCase):
    def test_rules_classify_each_notice_type_with_evidence(self):
        examples = [
            (
                "gzgg",
                "原公告的采购项目编号：ABC-123",
            ),
            (
                "gkzb",
                "招标项目的潜在投标人应在政采云获取招标文件，并于12月前提交投标文件。",
            ),
            (
                "zbgg",
                "采购方式：公开招标\n四、中标结果公告",
            ),
            (
                "cjgg",
                "采购方式：竞争性磋商\n成交供应商：示例公司",
            ),
        ]
        for expected, text in examples:
            with self.subTest(label=expected):
                label, evidence = classify_rules(text)
                self.assertEqual(label, expected)
                self.assertTrue(evidence)
                self.assertIn(evidence, text)

    def test_rules_return_none_when_no_ordered_rule_fires(self):
        self.assertEqual(classify_rules("本文件介绍项目背景和采购单位信息。"), (None, None))


@unittest.skipUnless(importlib.util.find_spec("sklearn"), "scikit-learn is optional")
class NoticeTfidfTests(SimpleTestCase):
    def test_tiny_four_class_models_predict_and_expose_top_features(self):
        examples = {
            "gkzb": "公开招标公告 投标截止 获取招标文件",
            "zbgg": "中标结果 中标供应商 公开招标",
            "cjgg": "成交结果 竞争性磋商 询价",
            "gzgg": "更正公告 原公告 变更事项",
        }
        texts = [text for text in examples.values() for _ in range(2)]
        labels = [label for label in examples for _ in range(2)]
        for analyzer in ("char", "word"):
            with self.subTest(analyzer=analyzer):
                model = train_tfidf(texts, labels, analyzer=analyzer)
                self.assertEqual(predict_tfidf(model, texts), labels)
                features = top_features(model, k=3)
                self.assertEqual(set(features), set(LABELS))
                for label in LABELS:
                    self.assertTrue(features[label])
                    self.assertLessEqual(len(features[label]), 3)
                    self.assertTrue(all(isinstance(feature, str) for feature, _ in features[label]))


class NoticeLlmTests(SimpleTestCase):
    def test_tool_call_response_and_evidence_validation(self):
        response = {
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "submit_notice_type",
                        "arguments": json.dumps(
                            {"label": "gkzb", "evidence": "采购方式：公开招标"},
                            ensure_ascii=False,
                        ),
                    }
                }
            ],
        }
        text = "本公告正文：采购方式：公开招标。"
        with (
            patch(
                "personal_knowledge_base.extraction.classify._model_usage_totals",
                return_value={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            ),
            patch(
                "personal_knowledge_base.extraction.classify.model_providers.chat_completion_raw",
                return_value=response,
            ) as chat,
        ):
            label, evidence, usage = classify_llm(text, object())

        self.assertEqual((label, evidence), ("gkzb", "采购方式：公开招标"))
        self.assertTrue(usage["evidence_ok"])
        self.assertEqual(chat.call_args.kwargs["tools"], [TOOL])
        self.assertEqual(chat.call_args.kwargs["temperature"], 0)

    def test_content_json_fallback_and_bad_evidence(self):
        response = {
            "content": '```json\n{"label":"cjgg","evidence":"未出现在正文"}\n```',
            "tool_calls": None,
        }
        label, evidence, usage = classify_llm(
            "正文只有采购单位地址。",
            object(),
            response=response,
            usage_override={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )
        self.assertEqual((label, evidence), ("cjgg", "未出现在正文"))
        self.assertFalse(usage["evidence_ok"])
        self.assertEqual(usage["total_tokens"], 15)

    def test_invalid_label_is_returned_as_none(self):
        response = {
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "submit_notice_type",
                        "arguments": '{"label":"unknown","evidence":"采购单位"}',
                    }
                }
            ],
        }
        label, _, usage = classify_llm(
            "采购单位",
            object(),
            response=response,
            usage_override={},
        )
        self.assertIsNone(label)
        self.assertFalse(usage["evidence_ok"])


class NoticeEvalReplayTests(SimpleTestCase):
    def test_replay_does_not_rewrite_live_raw_archive(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_dir = root / "output"
            output_dir.mkdir()
            archive = output_dir / "llm_raw.jsonl"
            archive.write_bytes(b"previous live archive\n")
            body_path = root / "notice.txt"
            body_path.write_text("采购方式：公开招标", encoding="utf-8")
            replay_path = root / "replay.jsonl"
            replay = {
                "doc_id": "fixture-notice",
                "response": {
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "submit_notice_type",
                                "arguments": (
                                    '{"label":"gkzb","evidence":"采购方式：公开招标"}'
                                ),
                            }
                        }
                    ],
                },
                "latency_ms": 12.5,
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            }
            replay_path.write_text(json.dumps(replay) + "\n", encoding="utf-8")
            labels = [
                {
                    "doc_id": "fixture-notice",
                    "label": "gkzb",
                    "split": "test",
                    "url": "https://example.invalid/notice",
                    "title": "不参与分类的标题",
                    "publish_time": "2026-11-24",
                }
            ]
            with (
                patch(
                    "personal_knowledge_base.management.commands."
                    "run_notice_classification_eval._load_json",
                    return_value=labels,
                ),
                patch(
                    "personal_knowledge_base.management.commands."
                    "run_notice_classification_eval._document_path",
                    return_value=body_path,
                ),
                patch(
                    "personal_knowledge_base.extraction.classify."
                    "model_providers.chat_completion_raw",
                    side_effect=AssertionError("replay must not call the model"),
                ),
            ):
                call_command(
                    "run_notice_classification_eval",
                    methods="llm",
                    split="test",
                    output_dir=str(output_dir),
                    replay_llm=str(replay_path),
                    verbosity=0,
                )
            self.assertEqual(archive.read_bytes(), b"previous live archive\n")
