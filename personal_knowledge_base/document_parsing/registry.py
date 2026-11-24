import base64
import csv
import io
import json
import re
from threading import Lock
from pathlib import Path

from .images import ImageTooSmallError, InvalidImageError, guess_image_mime, inspect_image
from .remote_images import UnsafeRemoteImageError, download_remote_image
from .types import ImageBlock, ParsedDocument, ParseWarning, TextBlock


IMAGE_TYPES = {"jpg", "jpeg", "png", "gif", "bmp", "tif", "tiff", "webp", "svg"}
TEXT_TYPES = {"txt", "log", "py"}
MARKDOWN_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)", re.I)
DATA_IMAGE_RE = re.compile(r"^data:(image/[^;,]+);base64,(.+)$", re.I | re.S)
_PDF_TABLE_EXTRACTION_LOCK = Lock()


def _image_block(data: bytes, mime_type: str, source_type: str, source_ref: str, block_index: int, page_index=None, metadata=None):
    width, height, detected_mime = inspect_image(data, mime_type)
    return ImageBlock(data, detected_mime, width, height, source_type, source_ref, block_index, page_index, metadata or {})


def _warning(document: ParsedDocument, code: str, message: str, block_index=None, source_ref=""):
    document.warnings.append(ParseWarning(code, message, block_index, source_ref))


def parse_text(name: str, data: bytes) -> ParsedDocument:
    suffix = Path(name).suffix.lower().lstrip(".")
    text = data.decode("utf-8", errors="ignore")
    if suffix == "json":
        try:
            text = json.dumps(json.loads(text), ensure_ascii=False, indent=2)
        except Exception:
            pass
    elif suffix == "csv":
        try:
            text = "\n".join(" | ".join(row) for row in csv.reader(io.StringIO(text)))
        except Exception:
            pass
    elif suffix in {"html", "htm"}:
        text = re.sub(r"<(script|style).*?</\1>", "", text, flags=re.I | re.S)
        text = re.sub(r"<[^>]+>", " ", text)
    return ParsedDocument(text_blocks=[TextBlock(text, 0)] if text.strip() else [])


def parse_image(name: str, data: bytes) -> ParsedDocument:
    document = ParsedDocument()
    mime_type = "image/svg+xml" if name.lower().endswith(".svg") else guess_image_mime(name)
    try:
        document.images.append(_image_block(data, mime_type, "standalone", name, 0))
    except InvalidImageError as exc:
        _warning(document, "invalid_or_small_image", str(exc), 0, name)
    return document


def parse_markdown(name: str, data: bytes) -> ParsedDocument:
    document = ParsedDocument()
    text = data.decode("utf-8", errors="ignore")
    cursor = 0
    block_index = 0
    for match in MARKDOWN_IMAGE_RE.finditer(text):
        before = text[cursor:match.start()].strip()
        if before:
            document.text_blocks.append(TextBlock(before, block_index))
            block_index += 1
        alt, target = match.group(1), match.group(2).strip().strip("<>")
        try:
            data_match = DATA_IMAGE_RE.match(target)
            if data_match:
                image_data = base64.b64decode(data_match.group(2), validate=True)
                document.images.append(_image_block(image_data, data_match.group(1), "markdown_data", alt or "data-uri", block_index))
            elif target.startswith(("http://", "https://")):
                image_data, mime_type, resolved = download_remote_image(target)
                document.images.append(_image_block(image_data, mime_type, "markdown_remote", resolved, block_index))
            else:
                document.text_blocks.append(TextBlock(match.group(0), block_index))
                _warning(document, "relative_image_unavailable", "relative Markdown image is not part of the upload", block_index, target)
        except (ValueError, InvalidImageError, UnsafeRemoteImageError, OSError) as exc:
            document.text_blocks.append(TextBlock(match.group(0), block_index))
            _warning(document, "image_fetch_failed", str(exc), block_index, target)
        block_index += 1
        cursor = match.end()
    tail = text[cursor:].strip()
    if tail:
        document.text_blocks.append(TextBlock(tail, block_index))
    return document


def _parse_pdf_legacy(name: str, data: bytes) -> ParsedDocument:
    import fitz

    document = ParsedDocument()
    pdf = fitz.open(stream=data, filetype="pdf")
    block_index = 0
    try:
        for page_index, page in enumerate(pdf):
            page_text = page.get_text("text").strip()
            has_visual = bool(page.get_images(full=True) or page.get_drawings())
            if len(page_text) < 20 and has_visual:
                rendered = page.get_pixmap(dpi=150, alpha=False).tobytes("jpeg")
                document.images.append(_image_block(rendered, "image/jpeg", "scanned_pdf", f"page:{page_index + 1}", block_index, page_index))
                block_index += 1
                continue
            items = []
            for raw in page.get_text("dict").get("blocks", []):
                bbox = raw.get("bbox") or (0, 0, 0, 0)
                if raw.get("type") == 0:
                    value = "\n".join(
                        "".join(span.get("text", "") for span in line.get("spans", []))
                        for line in raw.get("lines", [])
                    ).strip()
                    if value:
                        items.append((bbox[1], bbox[0], "text", value, "", bbox))
                elif raw.get("type") == 1 and raw.get("image"):
                    items.append((bbox[1], bbox[0], "image", raw["image"], raw.get("ext", "png"), bbox))
            for _, _, kind, value, ext, bbox in sorted(items, key=lambda item: (item[0], item[1])):
                if kind == "text":
                    document.text_blocks.append(TextBlock(value, block_index, page_index, {"bbox": list(bbox)}))
                else:
                    try:
                        document.images.append(_image_block(value, guess_image_mime(f"image.{ext}"), "pdf_embedded", f"page:{page_index + 1}", block_index, page_index, {"bbox": list(bbox)}))
                    except ImageTooSmallError as exc:
                        _warning(document, "small_image_skipped", str(exc), block_index, f"page:{page_index + 1}")
                    except InvalidImageError as exc:
                        _warning(document, "invalid_image", str(exc), block_index, f"page:{page_index + 1}")
                block_index += 1
            for drawing_index, drawing in enumerate(page.get_drawings()):
                rect = drawing.get("rect")
                if not rect or rect.width < 64 or rect.height < 64:
                    continue
                try:
                    pixmap = page.get_pixmap(clip=rect, dpi=150, alpha=False)
                    rendered = pixmap.tobytes("png")
                    document.images.append(_image_block(rendered, "image/png", "pdf_vector", f"page:{page_index + 1}:drawing:{drawing_index}", block_index, page_index, {"bbox": [rect.x0, rect.y0, rect.x1, rect.y1]}))
                    block_index += 1
                except InvalidImageError:
                    continue
    finally:
        pdf.close()
    return document


def parse_pdf(name: str, data: bytes) -> ParsedDocument:
    from django.conf import settings

    if not settings.PDF_LAYOUT_PARSING:
        return _parse_pdf_legacy(name, data)
    return _parse_pdf_layout(name, data)


_PDF_L1_HEADING_RE = re.compile(r"^第[一二三四五六七八九十百]+[章篇部分]")
_PDF_L2_HEADING_RE = re.compile(r"^[一二三四五六七八九十]+、\S")
_PDF_L3_HEADING_RE = re.compile(r"^（[一二三四五六七八九十]+）\S")
_PDF_PAGE_NUMBER_PATTERNS = (
    re.compile(r"^[-—–]\s*\d+\s*[-—–]$"),
    re.compile(r"^第\s*\d+\s*页$"),
    re.compile(r"^\d+\s*/\s*\d+$"),
    re.compile(r"^\d{1,4}$"),
)
_PDF_AMOUNT_MARKER_RE = re.compile(
    r"(?:\d[\d,.]*\s*万?元|[￥¥]\s*\d|\d\s*%|合计|小计|总计)"
)
_CJK_CELL_CHAR_RE = re.compile(r"[\u3400-\u9fff]")


def _pdf_in_margin(bbox, page_height: float) -> bool:
    center_y = (float(bbox[1]) + float(bbox[3])) / 2
    return center_y <= page_height * 0.08 or center_y >= page_height * 0.92


def _pdf_heading_line(text: str) -> bool:
    value = text.strip()
    return bool(
        _PDF_L1_HEADING_RE.match(value)
        or (len(value) <= 40 and _PDF_L2_HEADING_RE.match(value))
        or (len(value) <= 40 and _PDF_L3_HEADING_RE.match(value))
    )


def _pdf_is_margin_text(item, page_height: float, repeated_header_keys) -> bool:
    if (
        item["kind"] != "text"
        or item.get("bbox") is None
        or not _pdf_in_margin(item["bbox"], page_height)
    ):
        return False
    value = item["text"].strip()
    if _pdf_heading_line(value) or _pdf_has_amount_marker(value):
        return False
    return (
        any(pattern.match(value) for pattern in _PDF_PAGE_NUMBER_PATTERNS)
        or _pdf_header_key(value) in repeated_header_keys
    )


def _pdf_header_key(text: str) -> str:
    return re.sub(r"\s+", "", re.sub(r"\d+", "#", text))


def _pdf_has_amount_marker(text: str) -> bool:
    return bool(_PDF_AMOUNT_MARKER_RE.search(text))


def _pdf_point_in_bbox(point, bbox) -> bool:
    return bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]


def _join_table_cell(left: str, right: str) -> str:
    left, right = (left or "").strip(), (right or "").strip()
    if not left:
        return right
    if not right:
        return left
    separator = (
        ""
        if _CJK_CELL_CHAR_RE.search(left[-1])
        and _CJK_CELL_CHAR_RE.search(right[0])
        else " "
    )
    return left + separator + right


def _normalize_table_cell(value) -> str:
    if value is None:
        return ""
    lines = [line.strip() for line in str(value).splitlines() if line.strip()]
    result = ""
    for line in lines:
        result = _join_table_cell(result, line)
    return result


def _normalize_table_rows(rows) -> list[list[str]]:
    normalized = [
        [_normalize_table_cell(cell) for cell in row]
        for row in (rows or [])
    ]
    column_count = max((len(row) for row in normalized), default=0)
    if not column_count:
        return []
    return [row + [""] * (column_count - len(row)) for row in normalized]


def _markdown_table(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    column_count = len(rows[0])

    def format_row(row):
        cells = [cell.replace("|", r"\|") for cell in row]
        return "| " + " | ".join(cells) + " |"

    lines = [format_row(rows[0]), format_row(["---"] * column_count)]
    lines.extend(format_row(row) for row in rows[1:])
    return "\n".join(lines)


def _table_header_like(row: list[str], previous_header: list[str]) -> bool:
    return row != previous_header and all(
        cell and len(cell) <= 12 and not re.search(r"\d", cell)
        for cell in row
    )


def _parse_pdf_layout(name: str, data: bytes) -> ParsedDocument:
    """Serialize table extraction because PyMuPDF table globals are shared across uploads."""
    import fitz

    document = ParsedDocument()
    pdf = fitz.open(stream=data, filetype="pdf")
    page_records = []
    header_pages = {}
    header_footer_removed = 0
    cross_page_merges = 0
    detected_tables = 0
    toc_pages = []
    try:
        for page_index, page in enumerate(pdf):
            page_text = page.get_text("text").strip()
            has_visual = bool(page.get_images(full=True) or page.get_drawings())
            if len(page_text) < 20 and has_visual:
                page_records.append(
                    {
                        "page": page,
                        "page_index": page_index,
                        "scanned_image": page.get_pixmap(dpi=150, alpha=False).tobytes("jpeg"),
                    }
                )
                continue

            text_items = []
            image_items = []
            for raw in page.get_text("dict").get("blocks", []):
                bbox = raw.get("bbox") or (0, 0, 0, 0)
                if raw.get("type") == 0:
                    for line in raw.get("lines", []):
                        value = "".join(
                            span.get("text", "") for span in line.get("spans", [])
                        ).strip()
                        if value:
                            text_items.append(
                                {
                                    "kind": "text",
                                    "text": value,
                                    "bbox": line.get("bbox") or bbox,
                                }
                            )
                elif raw.get("type") == 1 and raw.get("image"):
                    image_items.append(
                        {
                            "kind": "image",
                            "data": raw["image"],
                            "ext": raw.get("ext", "png"),
                            "bbox": bbox,
                        }
                    )

            table_items = []
            try:
                with _PDF_TABLE_EXTRACTION_LOCK:
                    found_tables = [
                        (list(table.bbox), table.extract())
                        for table in page.find_tables().tables
                    ]
            except Exception as exc:
                found_tables = []
                _warning(
                    document,
                    "pdf_table_extraction_failed",
                    str(exc),
                    source_ref=f"page:{page_index + 1}",
                )
            for bbox, extracted_rows in found_tables:
                rows = _normalize_table_rows(extracted_rows)
                if not rows:
                    continue
                table_items.append(
                    {
                        "kind": "table",
                        "bbox": bbox,
                        "y": bbox[1],
                        "x": bbox[0],
                        "text": _markdown_table(rows),
                        "rows": rows,
                        "metadata": {
                            "bbox": bbox,
                            "block_type": "table",
                            "header": rows[0],
                            "table_rows": rows,
                            "column_count": len(rows[0]),
                            "page_span": [page_index + 1, page_index + 1],
                        },
                    }
                )
                detected_tables += 1

            page_height = float(page.rect.height)
            page_candidate_keys = set()
            for item in text_items:
                if (
                    _pdf_in_margin(item["bbox"], page_height)
                    and not _pdf_has_amount_marker(item["text"])
                ):
                    key = _pdf_header_key(item["text"])
                    if key:
                        page_candidate_keys.add(key)
            for key in page_candidate_keys:
                header_pages.setdefault(key, set()).add(page_index)

            page_records.append(
                {
                    "page": page,
                    "page_index": page_index,
                    "text_items": text_items,
                    "image_items": image_items,
                    "table_items": table_items,
                }
            )

        minimum_pages = max(3, len(pdf) * 0.5)
        repeated_header_keys = {
            key
            for key, pages in header_pages.items()
            if len(pages) >= minimum_pages
        }

        for record in page_records:
            if "scanned_image" in record:
                record["items"] = [
                    {
                        "kind": "scanned_image",
                        "data": record["scanned_image"],
                        "bbox": None,
                    }
                ]
                continue
            page_index = record["page_index"]
            page_height = float(record["page"].rect.height)
            tables = record["table_items"]
            table_bboxes = [table["bbox"] for table in tables]
            retained_text = []
            for item in record["text_items"]:
                in_margin = _pdf_in_margin(item["bbox"], page_height)
                value = item["text"].strip()
                if (
                    in_margin
                    and not _pdf_has_amount_marker(value)
                    and _pdf_header_key(value) in repeated_header_keys
                ):
                    header_footer_removed += 1
                    continue
                bbox = item["bbox"]
                center = (
                    (float(bbox[0]) + float(bbox[2])) / 2,
                    (float(bbox[1]) + float(bbox[3])) / 2,
                )
                if any(_pdf_point_in_bbox(center, table_bbox) for table_bbox in table_bboxes):
                    continue
                retained_text.append(
                    {
                        "kind": "text",
                        "text": value,
                        "bbox": bbox,
                        "y": bbox[1],
                        "x": bbox[0],
                    }
                )

            headings = {
                item["text"].strip()
                for item in retained_text
                if _PDF_L1_HEADING_RE.match(item["text"].strip())
            }
            if len(headings) >= 4:
                toc_pages.append(page_index + 1)

            for item in record["image_items"]:
                bbox = item["bbox"]
                item.update({"y": bbox[1], "x": bbox[0]})
            record["items"] = sorted(
                [*retained_text, *tables, *record["image_items"]],
                key=lambda item: (item["y"], item["x"]),
            )

        for page_index in range(1, len(page_records)):
            previous_record = page_records[page_index - 1]
            previous_items = previous_record.get("items", [])
            current_record = page_records[page_index]
            current_items = current_record.get("items", [])
            if not current_items:
                continue
            previous_page_height = float(previous_record["page"].rect.height)
            current_page_height = float(current_record["page"].rect.height)
            previous_index = next(
                (
                    index
                    for index in range(len(previous_items) - 1, -1, -1)
                    if not _pdf_is_margin_text(
                        previous_items[index],
                        previous_page_height,
                        repeated_header_keys,
                    )
                ),
                None,
            )
            if previous_index is not None:
                previous = previous_items[previous_index]
                previous_bbox = previous.get("bbox")
            else:
                previous = previous_record.get("merged_table")
                previous_bbox = previous_record.get("merged_bbox")
            if previous is None:
                continue
            continuation_item_index = next(
                (
                    index
                    for index, item in enumerate(current_items)
                    if not _pdf_is_margin_text(
                        item,
                        current_page_height,
                        repeated_header_keys,
                    )
                ),
                None,
            )
            if continuation_item_index is None:
                continue
            continuation = current_items[continuation_item_index]
            if previous["kind"] != "table" or continuation["kind"] != "table":
                continue
            previous_metadata = previous["metadata"]
            continuation_metadata = continuation["metadata"]
            column_count = previous_metadata["column_count"]
            if column_count != continuation_metadata["column_count"]:
                continue
            if (
                previous_bbox is None
                or previous_bbox[3] < previous_page_height * 0.80
                or continuation["bbox"][1] > current_page_height * 0.20
            ):
                continue

            merged_rows = previous_metadata["table_rows"]
            continuation_rows = continuation_metadata["table_rows"]
            if continuation_rows and continuation_rows[0] == previous_metadata["header"]:
                continuation_rows = continuation_rows[1:]
            elif continuation_rows and _table_header_like(
                continuation_rows[0],
                previous_metadata["header"],
            ):
                continue
            for continuation_index, row in enumerate(continuation_rows):
                row = (row + [""] * column_count)[:column_count]
                if (
                    continuation_index == 0
                    and not row[0].strip()
                    and len(merged_rows) > 1
                ):
                    merged_rows[-1] = [
                        _join_table_cell(old, new)
                        for old, new in zip(merged_rows[-1], row)
                    ]
                else:
                    merged_rows.append(row)
            previous_metadata["table_rows"] = merged_rows
            previous_metadata["page_span"] = [
                previous_metadata["page_span"][0],
                page_index + 1,
            ]
            previous["text"] = _markdown_table(merged_rows)
            current_items.pop(continuation_item_index)
            current_record["merged_table"] = previous
            current_record["merged_bbox"] = continuation["bbox"]
            cross_page_merges += 1

        block_index = 0
        for record in page_records:
            page_index = record["page_index"]
            if "scanned_image" in record:
                document.images.append(
                    _image_block(
                        record["scanned_image"],
                        "image/jpeg",
                        "scanned_pdf",
                        f"page:{page_index + 1}",
                        block_index,
                        page_index,
                    )
                )
                block_index += 1
                continue

            for item in record.get("items", []):
                if item["kind"] == "text":
                    document.text_blocks.append(
                        TextBlock(
                            item["text"],
                            block_index,
                            page_index,
                            {"bbox": list(item["bbox"])},
                        )
                    )
                elif item["kind"] == "table":
                    document.text_blocks.append(
                        TextBlock(
                            item["text"],
                            block_index,
                            page_index,
                            item["metadata"],
                        )
                    )
                elif item["kind"] == "image":
                    try:
                        document.images.append(
                            _image_block(
                                item["data"],
                                guess_image_mime(f"image.{item['ext']}"),
                                "pdf_embedded",
                                f"page:{page_index + 1}",
                                block_index,
                                page_index,
                                {"bbox": list(item["bbox"])},
                            )
                        )
                    except ImageTooSmallError as exc:
                        _warning(
                            document,
                            "small_image_skipped",
                            str(exc),
                            block_index,
                            f"page:{page_index + 1}",
                        )
                    except InvalidImageError as exc:
                        _warning(
                            document,
                            "invalid_image",
                            str(exc),
                            block_index,
                            f"page:{page_index + 1}",
                        )
                block_index += 1

            for drawing_index, drawing in enumerate(record["page"].get_drawings()):
                rect = drawing.get("rect")
                if not rect or rect.width < 64 or rect.height < 64:
                    continue
                try:
                    pixmap = record["page"].get_pixmap(
                        clip=rect, dpi=150, alpha=False
                    )
                    rendered = pixmap.tobytes("png")
                    document.images.append(
                        _image_block(
                            rendered,
                            "image/png",
                            "pdf_vector",
                            f"page:{page_index + 1}:drawing:{drawing_index}",
                            block_index,
                            page_index,
                            {"bbox": [rect.x0, rect.y0, rect.x1, rect.y1]},
                        )
                    )
                    block_index += 1
                except InvalidImageError:
                    continue
        document.metadata["pdf_layout"] = {
            "tables": detected_tables,
            "cross_page_merges": cross_page_merges,
            "header_footer_removed": header_footer_removed,
            "toc_pages": toc_pages,
        }
    finally:
        pdf.close()
    return document


def parse_docx(name: str, data: bytes) -> ParsedDocument:
    import docx
    from docx.oxml.ns import qn

    source = docx.Document(io.BytesIO(data))
    document = ParsedDocument()
    block_index = 0

    def append_images(element):
        nonlocal block_index
        for blip in element.xpath(".//a:blip"):
            rel_id = blip.get(qn("r:embed"))
            part = source.part.related_parts.get(rel_id)
            if not part:
                continue
            try:
                document.images.append(_image_block(part.blob, part.content_type, "docx", rel_id, block_index))
            except InvalidImageError as exc:
                _warning(document, "invalid_or_small_image", str(exc), block_index, rel_id)
            block_index += 1

    for child in source.element.body.iterchildren():
        if child.tag == qn("w:p"):
            text = "".join(child.itertext()).strip()
            if text:
                document.text_blocks.append(TextBlock(text, block_index))
                block_index += 1
            append_images(child)
        elif child.tag == qn("w:tbl"):
            rows = []
            for row in child.xpath("./w:tr"):
                cells = ["".join(cell.itertext()).strip() for cell in row.xpath("./w:tc")]
                rows.append(" | ".join(cells))
            value = "\n".join(row for row in rows if row.strip())
            if value:
                document.text_blocks.append(TextBlock(value, block_index))
                block_index += 1
            append_images(child)
    return document


def parse_pptx(name: str, data: bytes) -> ParsedDocument:
    from pptx import Presentation
    from pptx.enum.shapes import MSO_SHAPE_TYPE

    source = Presentation(io.BytesIO(data))
    document = ParsedDocument()
    block_index = 0
    for page_index, slide in enumerate(source.slides):
        for shape_index, shape in enumerate(slide.shapes):
            text = getattr(shape, "text", "").strip() if getattr(shape, "has_text_frame", False) else ""
            if text:
                document.text_blocks.append(TextBlock(text, block_index, page_index, {"shape_index": shape_index}))
                block_index += 1
            if shape.shape_type != MSO_SHAPE_TYPE.PICTURE:
                continue
            extension = (shape.image.ext or "").lower()
            source_ref = f"slide:{page_index + 1}:shape:{shape_index}"
            if extension in {"wmf", "emf"}:
                _warning(document, "unsupported_vector_media", f".{extension} media is not supported", block_index, source_ref)
                block_index += 1
                continue
            try:
                document.images.append(_image_block(shape.image.blob, guess_image_mime(f"image.{extension}"), "pptx", source_ref, block_index, page_index, {"shape_index": shape_index}))
            except InvalidImageError as exc:
                _warning(document, "invalid_or_small_image", str(exc), block_index, source_ref)
            block_index += 1
    return document


def parse_document(name: str, data: bytes, engine: str = "builtin") -> ParsedDocument:
    if engine not in {"", "builtin", "plain"}:
        raise ValueError(f"unsupported parser engine: {engine}")
    suffix = Path(name or "").suffix.lower().lstrip(".")
    if suffix in IMAGE_TYPES:
        return parse_image(name, data)
    if suffix in {"md", "markdown"}:
        return parse_markdown(name, data)
    if suffix == "pdf":
        return parse_pdf(name, data)
    if suffix == "docx":
        return parse_docx(name, data)
    if suffix == "pptx":
        return parse_pptx(name, data)
    return parse_text(name, data)
