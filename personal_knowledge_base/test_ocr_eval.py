import csv
import io
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from django.core.management import call_command
from django.test import SimpleTestCase

from personal_knowledge_base.management.commands.run_ocr_eval import _normalize_for_cer


class OcrEvalTests(SimpleTestCase):
    def test_cer_normalizer_treats_reference_and_hypothesis_markers_symmetrically(self):
        reference = "CODE-12\n|#Table*`\n— 9 —\n"
        hypothesis = "CODE–12\nTable\n— 3 —\n"

        self.assertEqual(_normalize_for_cer(reference), "CODE12Table")
        self.assertEqual(_normalize_for_cer(reference), _normalize_for_cer(hypothesis))

    def test_rescore_reuses_raw_records_without_calling_engines_or_rewriting_raw_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            raw_path = directory / "ocr_raw.jsonl"
            page_record = {
                "record_type": "page",
                "engine": "paddle",
                "variant": "clean",
                "page": 1,
                "reference_text": "CODE-12\n— 9 —",
                "raw_hypothesis": "CODE–12\n— 3 —",
                "pages_sent_to_vlm": 0,
                "latency_ms": 10.0,
                "mean_score": 0.9,
                "low_score_ratio": 0.0,
                "fallback_reason": "",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "errors": "",
            }
            metadata = {
                "record_type": "metadata",
                "source_pdf": "fixture.pdf",
                "pages": [1],
                "variants": {"clean": {"dpi": 150}},
                "scanned_pdfs": {},
            }
            raw_path.write_text(
                "".join(
                    json.dumps(record, ensure_ascii=False) + "\n"
                    for record in (page_record, metadata)
                ),
                encoding="utf-8",
            )
            original_raw = raw_path.read_bytes()

            with (
                patch(
                    "personal_knowledge_base.management.commands.run_ocr_eval.recognize_image",
                    side_effect=AssertionError("rescore must not call OCR engines"),
                ) as recognize,
                patch(
                    "personal_knowledge_base.management.commands.run_ocr_eval.paddle_ocr",
                    side_effect=AssertionError("rescore must not call OCR engines"),
                ) as paddle,
            ):
                call_command(
                    "run_ocr_eval",
                    rescore=str(raw_path),
                    output_dir=temp_dir,
                    stdout=io.StringIO(),
                )

            self.assertEqual(raw_path.read_bytes(), original_raw)
            recognize.assert_not_called()
            paddle.assert_not_called()

            summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["metrics"]["paddle"]["clean"]["mean_cer"], 0.0)
            with (directory / "per_page.csv").open(encoding="utf-8", newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(float(row["cer"]), 0.0)
