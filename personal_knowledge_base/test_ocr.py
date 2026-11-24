import io
from types import SimpleNamespace
from unittest.mock import Mock, patch

from django.test import SimpleTestCase, TestCase, override_settings
from PIL import Image

from personal_knowledge_base.document_parsing.types import ImageBlock
from personal_knowledge_base.models import Chunk, Knowledge, KnowledgeBase, Tenant
from personal_knowledge_base.multimodal import analyze_image, process_document_images, recognize_image
from personal_knowledge_base.ocr import (
    OcrLine,
    OcrResult,
    character_error_rate,
    levenshtein_distance,
    paddle_ocr,
)


def _png_bytes():
    image = Image.new("RGB", (128, 128), "white")
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _result(text="paddle text", mean=0.92, low_ratio=0.0, lines=None):
    return OcrResult(
        text=text,
        lines=lines if lines is not None else [OcrLine(text, mean, (0, 0, 20, 10))],
        mean_score=mean,
        low_score_ratio=low_ratio,
        engine="paddle",
        latency_ms=2.0,
    )


class PaddleOcrTests(SimpleTestCase):
    def test_lines_are_sorted_and_grouped_into_rows(self):
        engine = Mock()
        engine.predict.return_value = [
            {
                "rec_texts": ["right", "next row", "left"],
                "rec_scores": [0.9, 0.7, 0.8],
                "rec_polys": [
                    [[60, 10], [90, 10], [90, 20], [60, 20]],
                    [[10, 40], [70, 40], [70, 50], [10, 50]],
                    [[10, 11], [40, 11], [40, 21], [10, 21]],
                ],
            }
        ]

        with patch("personal_knowledge_base.ocr._get_engine", return_value=engine):
            result = paddle_ocr(_png_bytes())

        self.assertEqual(result.text, "left right\nnext row")
        self.assertEqual([line.text for line in result.lines], ["left", "right", "next row"])
        self.assertAlmostEqual(result.mean_score, 0.8)
        self.assertAlmostEqual(result.low_score_ratio, 1 / 3)
        self.assertEqual(result.engine, "paddle")

    def test_cer_uses_whitespace_free_levenshtein(self):
        self.assertEqual(levenshtein_distance("kitten", "sitting"), 3)
        self.assertAlmostEqual(character_error_rate("a b\nc", "a c"), 1 / 3)
        self.assertEqual(character_error_rate("", ""), 0.0)
        self.assertEqual(character_error_rate("", "x"), 1.0)


class OcrRoutingTests(TestCase):
    def setUp(self):
        self.tenant = Tenant.objects.create(name="OCR tenant", api_key="ocr-tenant")
        self.knowledge_base = KnowledgeBase.objects.create(
            tenant=self.tenant,
            name="OCR KB",
            vlm_config={},
        )
        self.knowledge = Knowledge.objects.create(
            tenant=self.tenant,
            knowledge_base=self.knowledge_base,
            type="file",
            title="scan.png",
            file_name="scan.png",
            file_type="png",
        )
        self.image = _png_bytes()

    def test_vlm_and_paddle_routes(self):
        with (
            override_settings(OCR_ENGINE="vlm"),
            patch("personal_knowledge_base.multimodal.paddle_ocr") as paddle,
            patch("personal_knowledge_base.multimodal.vision_completion", return_value="vlm text") as vision,
        ):
            text, metadata, _, _, _ = recognize_image(
                self.image, "image/png", self.knowledge
            )
        self.assertEqual(text, "vlm text")
        self.assertEqual(metadata["engine"], "vlm")
        paddle.assert_not_called()
        vision.assert_called_once()

        with (
            override_settings(OCR_ENGINE="paddle"),
            patch("personal_knowledge_base.multimodal.paddle_ocr", return_value=_result()),
            patch("personal_knowledge_base.multimodal.vision_completion") as vision,
        ):
            text, metadata, _, _, _ = recognize_image(
                self.image, "image/png", self.knowledge
            )
        self.assertEqual(text, "paddle text")
        self.assertEqual(metadata["engine"], "paddle")
        vision.assert_not_called()

    def test_paddle_receives_normalized_raster_bytes(self):
        normalized = b"normalized-raster"
        with (
            override_settings(OCR_ENGINE="paddle"),
            patch(
                "personal_knowledge_base.multimodal.normalize_for_vlm",
                return_value=(normalized, "image/jpeg", 64, 64),
            ),
            patch("personal_knowledge_base.multimodal.paddle_ocr", return_value=_result()) as paddle,
            patch("personal_knowledge_base.multimodal.vision_completion", return_value="caption"),
        ):
            recognize_image(b"raw svg bytes", "image/svg+xml", self.knowledge)
            analyze_image(b"raw svg bytes", "image/svg+xml", self.knowledge)

        self.assertEqual(paddle.call_count, 2)
        self.assertEqual(paddle.call_args_list[0].args[0], normalized)
        self.assertEqual(paddle.call_args_list[1].args[0], normalized)

    def test_confidence_empty_and_exception_fallbacks(self):
        cases = [
            ("low mean", _result(mean=0.84), "mean_score_below_threshold"),
            ("high ratio", _result(mean=0.88, low_ratio=0.5), "low_score_ratio_above_threshold"),
            ("empty", _result(text="", mean=0.0, lines=[]), "no_lines"),
            ("exception", RuntimeError("engine failed"), "paddle_error:RuntimeError"),
        ]
        for name, paddle_result, reason in cases:
            with self.subTest(name=name):
                with (
                    override_settings(OCR_ENGINE="paddle_vlm_fallback", OCR_FALLBACK_RATIO=0.20),
                    patch(
                        "personal_knowledge_base.multimodal.paddle_ocr",
                        side_effect=paddle_result if isinstance(paddle_result, Exception) else None,
                        return_value=None if isinstance(paddle_result, Exception) else paddle_result,
                    ),
                    patch(
                        "personal_knowledge_base.multimodal.vision_completion",
                        return_value="vlm fallback",
                    ) as vision,
                ):
                    text, metadata, errors, _, _ = recognize_image(
                        self.image, "image/png", self.knowledge
                    )
                self.assertEqual(text, "vlm fallback")
                self.assertEqual(metadata["engine"], "vlm")
                self.assertEqual(metadata["fallback_reason"], reason)
                vision.assert_called_once()
                if name == "exception":
                    self.assertIn("Paddle OCR:", errors[0])

    def test_failed_vlm_fallback_preserves_paddle_text(self):
        cases = [
            ("exception", RuntimeError("VLM unavailable"), "VLM unavailable"),
            ("empty", "", "VLM OCR returned no text"),
        ]
        for name, vlm_result, expected_error in cases:
            with self.subTest(name=name):
                with (
                    override_settings(OCR_ENGINE="paddle_vlm_fallback"),
                    patch(
                        "personal_knowledge_base.multimodal.paddle_ocr",
                        return_value=_result("paddle remains", mean=0.84),
                    ),
                    patch(
                        "personal_knowledge_base.multimodal.vision_completion",
                        side_effect=vlm_result if isinstance(vlm_result, Exception) else None,
                        return_value=None if isinstance(vlm_result, Exception) else vlm_result,
                    ),
                ):
                    text, metadata, errors, _, _ = recognize_image(
                        self.image, "image/png", self.knowledge
                    )

                self.assertEqual(text, "paddle remains")
                self.assertEqual(metadata["engine"], "paddle")
                self.assertEqual(metadata["fallback_reason"], "mean_score_below_threshold")
                self.assertTrue(metadata["fallback_failed"])
                self.assertTrue(any(expected_error in error for error in errors))

    def test_paddle_exception_falls_back_even_in_paddle_mode(self):
        with (
            override_settings(OCR_ENGINE="paddle"),
            patch(
                "personal_knowledge_base.multimodal.paddle_ocr",
                side_effect=RuntimeError("paddle unavailable"),
            ),
            patch(
                "personal_knowledge_base.multimodal.vision_completion",
                return_value="vlm fallback",
            ) as vision,
        ):
            text, metadata, errors, _, _ = recognize_image(
                self.image, "image/png", self.knowledge
            )
        self.assertEqual(text, "vlm fallback")
        self.assertEqual(metadata["fallback_reason"], "paddle_error:RuntimeError")
        self.assertIn("Paddle OCR:", errors[0])
        vision.assert_called_once()

    def test_chunk_metadata_records_paddle_confidence(self):
        block = ImageBlock(
            self.image,
            "image/png",
            128,
            128,
            "standalone",
            "scan.png",
            0,
        )
        with (
            override_settings(OCR_ENGINE="paddle"),
            patch("personal_knowledge_base.multimodal.paddle_ocr", return_value=_result()),
            patch("personal_knowledge_base.multimodal.vision_completion", return_value="caption"),
        ):
            chunks, warnings = process_document_images(self.knowledge, [block], [])

        ocr_chunk = next(chunk for chunk in chunks if chunk.chunk_type == "image_ocr")
        self.assertEqual(ocr_chunk.metadata["ocr_engine"], "paddle")
        self.assertEqual(ocr_chunk.metadata["ocr_mean_score"], 0.92)
        self.assertEqual(ocr_chunk.metadata["ocr_low_score_ratio"], 0.0)
        self.assertEqual(ocr_chunk.metadata["ocr_fallback_reason"], "")
        self.assertEqual(warnings, [])
        self.assertEqual(Chunk.objects.filter(knowledge=self.knowledge).count(), 3)

    def test_local_ocr_continues_after_caption_opens_vlm_circuit(self):
        from personal_knowledge_base.model_providers import ModelAccessDeniedError

        blocks = [
            ImageBlock(self.image, "image/png", 128, 128, "standalone", f"scan-{index}.png", index)
            for index in range(2)
        ]
        with (
            override_settings(OCR_ENGINE="paddle"),
            patch(
                "personal_knowledge_base.multimodal.paddle_ocr",
                side_effect=[_result("first paddle"), _result("second paddle")],
            ) as paddle,
            patch(
                "personal_knowledge_base.multimodal.vision_completion",
                side_effect=ModelAccessDeniedError(403, "AllocationQuota.FreeTierOnly", "denied"),
            ) as vision,
        ):
            chunks, warnings = process_document_images(self.knowledge, blocks, [])

        self.assertEqual(paddle.call_count, 2)
        vision.assert_called_once()
        self.assertTrue(any(chunk.content == "second paddle" for chunk in chunks))
        self.assertTrue(any("denied" in warning["message"] for warning in warnings))
