import csv
import io
import json
import re
import shutil
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import fitz
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from PIL import Image, ImageDraw

from personal_knowledge_base.models import ModelUsage, Tenant
from personal_knowledge_base.multimodal import recognize_image
from personal_knowledge_base.ocr import character_error_rate, paddle_ocr


PAGES = (3, 4, 5, 7, 8, 34, 35, 36, 48, 51, 52, 60, 70, 82, 86)
FOOTER_RE = re.compile(r"(?m)^\s*[—–-]\s*\d+\s*[—–-]\s*$")
ENGINES = ("paddle", "vlm", "paddle_vlm_fallback")


def _resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path(settings.BASE_DIR) / path


def _reference_text(text: str) -> str:
    return FOOTER_RE.sub("", text or "")


def _clean_vlm_text(text: str) -> str:
    return re.sub(r"[|#*`-]", "", text or "")


def _render_page(page, dpi: int, quality: int, stamp: bool = False) -> bytes:
    pixmap = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), alpha=False)
    with Image.open(io.BytesIO(pixmap.tobytes("png"))) as opened:
        image = opened.convert("RGB")
    if stamp:
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        radius = min(image.size) * 0.08
        center_x, center_y = image.width / 2, image.height / 2
        draw.ellipse(
            (center_x - radius, center_y - radius, center_x + radius, center_y + radius),
            fill=(220, 0, 0, 85),
            outline=(180, 0, 0, 150),
            width=max(1, int(radius / 20)),
        )
        image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=quality)
    return output.getvalue()


def _usage_tokens(tenant, count_before: int) -> dict:
    queryset = ModelUsage.objects.filter(tenant=tenant)
    if queryset.count() <= count_before:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    record = queryset.order_by("-created_at").first()
    return {
        "prompt_tokens": int(record.prompt_tokens or 0),
        "completion_tokens": int(record.completion_tokens or 0),
        "total_tokens": int(record.total_tokens or 0),
    }


def _summarize(rows: list[dict]) -> dict:
    summary = {}
    for engine in ENGINES:
        summary[engine] = {}
        for variant in ("clean", "degraded"):
            group = [
                row
                for row in rows
                if row["engine"] == engine and row["variant"] == variant
            ]
            summary[engine][variant] = {
                "pages": len(group),
                "mean_cer": round(statistics.mean(row["cer"] for row in group), 6) if group else None,
                "median_cer": round(statistics.median(row["cer"] for row in group), 6) if group else None,
                "pages_sent_to_vlm": sum(row["pages_sent_to_vlm"] for row in group),
                "mean_latency_ms": round(statistics.mean(row["latency_ms"] for row in group), 2) if group else None,
                "vlm_tokens": sum(row["total_tokens"] for row in group),
            }
    return summary


class Command(BaseCommand):
    help = "Evaluate PaddleOCR, VLM OCR, and confidence-triggered VLM fallback."

    def add_arguments(self, parser):
        parser.add_argument(
            "--pdf",
            default="interview_materials/pdf_eval/ahmu_tender_2026.pdf",
        )
        parser.add_argument(
            "--p4",
            default="interview_materials/ocr_eval/p4.pdf",
        )
        parser.add_argument(
            "--p7",
            default="interview_materials/ocr_eval/p7.pdf",
        )
        parser.add_argument(
            "--output-dir",
            default="interview_materials/results/ocr",
        )

    def handle(self, *args, **options):
        pdf_path = _resolve_path(options["pdf"])
        output_dir = _resolve_path(options["output_dir"])
        scan_paths = [_resolve_path(options["p4"]), _resolve_path(options["p7"])]
        if not pdf_path.is_file():
            raise CommandError(f"OCR evaluation PDF does not exist: {pdf_path}")
        if any(not path.is_file() for path in scan_paths):
            raise CommandError("Both scanned PDF inputs must exist")

        tenant = Tenant.objects.order_by("created_at").first()
        if tenant is None:
            raise CommandError("No tenant exists in the OCR evaluation database")
        output_dir.mkdir(parents=True, exist_ok=True)
        pages = fitz.open(pdf_path)
        rows = []
        try:
            for page_number in PAGES:
                if page_number > len(pages):
                    raise CommandError(f"PDF has no page {page_number}")
                page = pages[page_number - 1]
                reference = _reference_text(page.get_text("text"))
                for variant, dpi, quality, stamp in (
                    ("clean", 150, 90, False),
                    ("degraded", 100, 35, True),
                ):
                    image_bytes = _render_page(page, dpi, quality, stamp)
                    for engine in ENGINES:
                        before_count = ModelUsage.objects.filter(tenant=tenant).count()
                        started = time.perf_counter()
                        text, metadata, errors, _, recognition_ms = recognize_image(
                            image_bytes,
                            "image/jpeg",
                            SimpleNamespace(
                                tenant=tenant,
                                knowledge_base=SimpleNamespace(vlm_config={}),
                            ),
                            engine=engine,
                        )
                        usage = _usage_tokens(tenant, before_count)
                        actual_engine = metadata["engine"]
                        hypothesis = _clean_vlm_text(text) if actual_engine == "vlm" else text
                        rows.append(
                            {
                                "engine": engine,
                                "variant": variant,
                                "page": page_number,
                                "cer": character_error_rate(reference, hypothesis),
                                "pages_sent_to_vlm": int(actual_engine == "vlm"),
                                "latency_ms": round(recognition_ms or (time.perf_counter() - started) * 1000, 2),
                                "mean_score": metadata["mean_score"],
                                "low_score_ratio": metadata["low_score_ratio"],
                                "fallback_reason": metadata["fallback_reason"],
                                "prompt_tokens": usage["prompt_tokens"],
                                "completion_tokens": usage["completion_tokens"],
                                "total_tokens": usage["total_tokens"],
                                "errors": "; ".join(errors),
                            }
                        )
        finally:
            pages.close()

        scan_results = {}
        copy_dir = Path(settings.BASE_DIR) / "interview_materials" / "ocr_eval"
        copy_dir.mkdir(parents=True, exist_ok=True)
        for scan_path in scan_paths:
            copy_path = copy_dir / scan_path.name
            if scan_path.resolve() != copy_path.resolve():
                shutil.copy2(scan_path, copy_path)
            document = fitz.open(scan_path)
            scan_lines = []
            try:
                for page_index, page in enumerate(document):
                    image_bytes = _render_page(page, 150, 90)
                    result = paddle_ocr(image_bytes)
                    scan_lines.extend(
                        {
                            "page": page_index + 1,
                            "text": line.text,
                            "score": line.score,
                        }
                        for line in result.lines
                    )
            finally:
                document.close()
            scan_results[scan_path.name] = {
                "line_count": len(scan_lines),
                "mean_score": statistics.mean(line["score"] for line in scan_lines) if scan_lines else 0.0,
                "lowest_five_lines": sorted(scan_lines, key=lambda line: line["score"])[:5],
            }

        summary = {
            "source_pdf": pdf_path.name,
            "pages": list(PAGES),
            "variants": {
                "clean": {"dpi": 150, "jpeg_quality": 90},
                "degraded": {"dpi": 100, "jpeg_quality": 35, "center_stamp": "semi-transparent red circle"},
            },
            "metrics": _summarize(rows),
            "scanned_pdfs": scan_results,
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        fieldnames = [
            "engine",
            "variant",
            "page",
            "cer",
            "pages_sent_to_vlm",
            "latency_ms",
            "mean_score",
            "low_score_ratio",
            "fallback_reason",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "errors",
        ]
        with (output_dir / "per_page.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        self.stdout.write(json.dumps(summary["metrics"], ensure_ascii=False, indent=2))
