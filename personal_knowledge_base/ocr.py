import io
import statistics
import threading
import time
from dataclasses import dataclass

from django.conf import settings
from PIL import Image, ImageOps


@dataclass(frozen=True)
class OcrLine:
    text: str
    score: float
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class OcrResult:
    text: str
    lines: list[OcrLine]
    mean_score: float
    low_score_ratio: float
    engine: str
    latency_ms: float


class OcrUnavailable(RuntimeError):
    pass


_engine = None
_engine_lock = threading.Lock()


def _get_engine():
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                try:
                    from paddleocr import PaddleOCR
                except ImportError as exc:
                    raise OcrUnavailable(
                        "PaddleOCR is unavailable; install requirements-ocr.txt"
                    ) from exc
                try:
                    _engine = PaddleOCR(
                        lang="ch",
                        use_doc_orientation_classify=False,
                        use_doc_unwarping=False,
                        use_textline_orientation=False,
                        enable_mkldnn=False,
                    )
                except Exception as exc:
                    raise OcrUnavailable(f"unable to initialize PaddleOCR: {exc}") from exc
    return _engine


def _box_bounds(box) -> tuple[float, float, float, float]:
    values = box.tolist() if hasattr(box, "tolist") else box
    if values is None:
        return 0.0, 0.0, 0.0, 0.0
    if len(values) == 4 and not isinstance(values[0], (list, tuple)):
        x0, y0, x1, y1 = (float(value) for value in values)
        return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)
    points = [
        (float(point[0]), float(point[1]))
        for point in values
        if point is not None and len(point) >= 2
    ]
    if not points:
        return 0.0, 0.0, 0.0, 0.0
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def _rows_text(lines: list[OcrLine]) -> str:
    rows = []
    for line in sorted(lines, key=lambda item: ((item.bbox[1] + item.bbox[3]) / 2, item.bbox[0])):
        center_y = (line.bbox[1] + line.bbox[3]) / 2
        height = max(1.0, line.bbox[3] - line.bbox[1])
        if rows:
            row = rows[-1]
            row_center = statistics.mean(
                (entry.bbox[1] + entry.bbox[3]) / 2 for entry in row
            )
            row_height = statistics.mean(
                max(1.0, entry.bbox[3] - entry.bbox[1]) for entry in row
            )
            if abs(center_y - row_center) <= max(height, row_height) / 2:
                row.append(line)
                continue
        rows.append([line])
    return "\n".join(
        " ".join(line.text.strip() for line in sorted(row, key=lambda item: item.bbox[0]) if line.text.strip())
        for row in rows
    )


def paddle_ocr(image_bytes: bytes) -> OcrResult:
    started = time.perf_counter()
    engine = _get_engine()
    try:
        import numpy as np
    except ImportError as exc:
        raise OcrUnavailable("NumPy is required by PaddleOCR") from exc
    with Image.open(io.BytesIO(image_bytes)) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        prediction = engine.predict(np.asarray(image))
    result = prediction[0] if prediction else {}
    texts = result.get("rec_texts")
    scores = result.get("rec_scores")
    texts = [] if texts is None else texts
    scores = [] if scores is None else scores
    boxes = result.get("rec_polys")
    if boxes is None:
        boxes = result.get("rec_boxes")
    boxes = [] if boxes is None else boxes
    lines = []
    for index, text in enumerate(texts):
        if not str(text).strip():
            continue
        score = float(scores[index]) if index < len(scores) else 0.0
        bbox = _box_bounds(boxes[index]) if index < len(boxes) else (0.0, 0.0, 0.0, 0.0)
        lines.append(OcrLine(str(text).strip(), score, bbox))
    lines.sort(
        key=lambda item: (
            round(
                ((item.bbox[1] + item.bbox[3]) / 2)
                / max(item.bbox[3] - item.bbox[1], 1.0)
            ),
            item.bbox[0],
        )
    )
    mean_score = statistics.mean(line.score for line in lines) if lines else 0.0
    low_confidence = float(getattr(settings, "OCR_LOW_CONFIDENCE", 0.8))
    low_score_ratio = (
        sum(line.score < low_confidence for line in lines) / len(lines)
        if lines
        else 0.0
    )
    return OcrResult(
        text=_rows_text(lines),
        lines=lines,
        mean_score=mean_score,
        low_score_ratio=low_score_ratio,
        engine="paddle",
        latency_ms=(time.perf_counter() - started) * 1000,
    )


def levenshtein_distance(left: str, right: str) -> int:
    try:
        from rapidfuzz.distance import Levenshtein
    except ImportError:
        if len(left) < len(right):
            left, right = right, left
        previous = list(range(len(right) + 1))
        for left_index, left_char in enumerate(left, start=1):
            current = [left_index]
            for right_index, right_char in enumerate(right, start=1):
                current.append(
                    min(
                        current[-1] + 1,
                        previous[right_index] + 1,
                        previous[right_index - 1] + (left_char != right_char),
                    )
                )
            previous = current
        return previous[-1]
    return Levenshtein.distance(left, right)


def character_error_rate(reference: str, hypothesis: str) -> float:
    reference = "".join(reference.split())
    hypothesis = "".join(hypothesis.split())
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return levenshtein_distance(reference, hypothesis) / len(reference)
