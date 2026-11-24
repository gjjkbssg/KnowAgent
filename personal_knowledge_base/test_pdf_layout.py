import fitz
from django.core.management.base import CommandError
from django.test import TestCase, override_settings

from personal_knowledge_base.document_parsing import parse_document
from personal_knowledge_base.document_parsing.registry import _parse_pdf_legacy
from personal_knowledge_base.document_parsing.types import ParsedDocument, TextBlock
from personal_knowledge_base.document_processing import create_text_chunks
from personal_knowledge_base.management.commands.run_pdf_parsing_eval import (
    _cleanup_knowledge_base,
)
from personal_knowledge_base.models import Chunk, Knowledge, KnowledgeBase, Tenant


def _text(page, value, x, y, size=10):
    page.insert_text((x, y), value, fontname="china-s", fontsize=size)


def _pdf_bytes(document):
    data = document.tobytes()
    document.close()
    return data


def _draw_table(page, rows, *, x=30, y=80, widths=(50, 90, 170), row_height=26):
    x_positions = [x]
    for width in widths:
        x_positions.append(x_positions[-1] + width)
    y_positions = [y + index * row_height for index in range(len(rows) + 1)]
    for y_position in y_positions:
        page.draw_line(
            (x_positions[0], y_position),
            (x_positions[-1], y_position),
            color=(0, 0, 0),
            width=0.8,
        )
    for x_position in x_positions:
        page.draw_line(
            (x_position, y_positions[0]),
            (x_position, y_positions[-1]),
            color=(0, 0, 0),
            width=0.8,
        )
    for row_index, row in enumerate(rows):
        for column_index, value in enumerate(row):
            rect = fitz.Rect(
                x_positions[column_index] + 3,
                y_positions[row_index] + 2,
                x_positions[column_index + 1] - 3,
                y_positions[row_index + 1] - 2,
            )
            page.insert_textbox(
                rect,
                value or "",
                fontname="china-s",
                fontsize=8,
            )


@override_settings(PDF_LAYOUT_PARSING=True)
class PdfLayoutParsingTests(TestCase):
    def setUp(self):
        self.tenant = Tenant.objects.create(
            name="PDF layout tenant",
            api_key="pdf-layout-tenant",
        )
        self.knowledge_base = KnowledgeBase.objects.create(
            tenant=self.tenant,
            name="PDF layout KB",
            chunking_config={"chunk_size": 512, "chunk_overlap": 50},
        )

    def create_knowledge(self, name="layout.pdf", chunk_size=None):
        if chunk_size is not None:
            self.knowledge_base.chunking_config = {
                "chunk_size": chunk_size,
                "chunk_overlap": 0,
            }
            self.knowledge_base.save(update_fields=["chunking_config"])
        return Knowledge.objects.create(
            tenant=self.tenant,
            knowledge_base=self.knowledge_base,
            type="file",
            title=name,
            source=name,
            file_name=name,
            file_type="pdf",
        )

    def test_repeated_headers_and_page_numbers_are_removed(self):
        pdf = fitz.open()
        for page_number in range(1, 5):
            page = pdf.new_page(width=400, height=500)
            _text(page, f"采购文件第{page_number}号", 40, 20)
            _text(page, f"— {page_number} —", 40, 490)
            _text(page, f"正文内容第{page_number}页包含采购公告信息。", 40, 130)

        parsed = parse_document("repeated.pdf", _pdf_bytes(pdf))
        content = "\n".join(block.text for block in parsed.text_blocks)

        self.assertNotIn("采购文件第", content)
        self.assertNotRegex(content, r"—\s*\d+\s*—")
        self.assertIn("正文内容第1页", content)
        self.assertEqual(
            parsed.metadata["pdf_layout"]["header_footer_removed"],
            8,
        )

    def test_twelve_page_footers_are_removed_and_table_continuation_merges(self):
        pdf = fitz.open()
        for page_number in range(1, 13):
            page = pdf.new_page(width=400, height=500)
            if page_number not in (9, 10):
                _text(page, f"正文内容第{page_number}页，包含足够长度的采购信息。", 40, 130)
            _text(page, f"— {page_number} —", 40, 490)
            if page_number == 9:
                _draw_table(
                    page,
                    [
                        ["条款号", "名称", "内容"],
                        ["31.1", "履约保证金", "不收取"],
                    ],
                    y=380,
                )
            elif page_number == 10:
                _draw_table(
                    page,
                    [
                        ["条款号", "名称", "内容"],
                        ["", "", "收取合同价的2.5%"],
                    ],
                )

        parsed = parse_document("twelve-page-continuation.pdf", _pdf_bytes(pdf))
        content = "\n".join(block.text for block in parsed.text_blocks)
        table = next(
            block
            for block in parsed.text_blocks
            if block.metadata.get("block_type") == "table"
        )

        self.assertNotRegex(content, r"—\s*\d+\s*—")
        self.assertEqual(
            parsed.metadata["pdf_layout"]["header_footer_removed"],
            12,
        )
        self.assertEqual(table.metadata["page_span"], [9, 10])
        self.assertEqual(
            table.metadata["table_rows"][1],
            ["31.1", "履约保证金", "不收取收取合同价的2.5%"],
        )
        self.assertEqual(parsed.metadata["pdf_layout"]["cross_page_merges"], 1)

    def test_two_page_footers_do_not_block_table_continuation_merge(self):
        pdf = fitz.open()
        first_page = pdf.new_page(width=400, height=500)
        _text(first_page, "— 1 —", 40, 490)
        _draw_table(
            first_page,
            [
                ["条款号", "名称", "内容"],
                ["31.1", "履约保证金", "不收取"],
            ],
            y=380,
        )
        second_page = pdf.new_page(width=400, height=500)
        _draw_table(
            second_page,
            [
                ["条款号", "名称", "内容"],
                ["", "", "收取合同价的2.5%"],
            ],
        )
        _text(second_page, "— 2 —", 40, 490)

        parsed = parse_document("two-page-continuation.pdf", _pdf_bytes(pdf))
        content = "\n".join(block.text for block in parsed.text_blocks)
        table = next(
            block
            for block in parsed.text_blocks
            if block.metadata.get("block_type") == "table"
        )

        self.assertIn("— 1 —", content)
        self.assertIn("— 2 —", content)
        self.assertEqual(table.metadata["page_span"], [1, 2])
        self.assertEqual(
            table.metadata["table_rows"][1],
            ["31.1", "履约保证金", "不收取收取合同价的2.5%"],
        )
        self.assertEqual(parsed.metadata["pdf_layout"]["cross_page_merges"], 1)
        self.assertEqual(parsed.metadata["pdf_layout"]["header_footer_removed"], 0)

    def test_single_page_bare_margin_amount_is_preserved(self):
        pdf = fitz.open()
        page = pdf.new_page(width=400, height=500)
        _text(page, "本页预算总额为128000元，项目内容清晰完整。", 40, 130)
        _text(page, "128000", 40, 490)

        parsed = parse_document("single-page-amount.pdf", _pdf_bytes(pdf))
        content = "\n".join(block.text for block in parsed.text_blocks)

        self.assertIn("128000", content)
        self.assertEqual(
            parsed.metadata["pdf_layout"]["header_footer_removed"],
            0,
        )

    def test_tables_become_markdown_blocks_without_duplicate_text(self):
        pdf = fitz.open()
        page = pdf.new_page(width=400, height=500)
        _draw_table(
            page,
            [
                ["序号", "名称", "说明"],
                ["1", "检测设备", "用于样品分析"],
            ],
        )

        parsed = parse_document("table.pdf", _pdf_bytes(pdf))
        table_blocks = [
            block
            for block in parsed.text_blocks
            if block.metadata.get("block_type") == "table"
        ]

        self.assertEqual(len(table_blocks), 1)
        table = table_blocks[0]
        self.assertIn("| 名称 |", table.text)
        self.assertIn("| 检测设备 |", table.text)
        self.assertEqual(table.text.count("检测设备"), 1)
        self.assertEqual(table.metadata["header"], ["序号", "名称", "说明"])
        self.assertEqual(table.metadata["page_span"], [1, 1])
        self.assertEqual(len(table.metadata["bbox"]), 4)
        self.assertEqual(parsed.metadata["pdf_layout"]["tables"], 1)

    def test_continuation_table_drops_header_and_joins_split_row(self):
        pdf = fitz.open()
        first_page = pdf.new_page(width=400, height=500)
        _draw_table(
            first_page,
            [
                ["条款号", "名称", "内容"],
                ["31.1", "履约保证金", "不收取"],
            ],
            y=380,
        )
        second_page = pdf.new_page(width=400, height=500)
        _draw_table(
            second_page,
            [
                ["条款号", "名称", "内容"],
                ["", "", "收取合同价的2.5%"],
            ],
            row_height=200,
        )
        third_page = pdf.new_page(width=400, height=500)
        _draw_table(
            third_page,
            [
                ["条款号", "名称", "内容"],
                ["", "", "合同签订时缴纳履约保证金并按合同约定执行"],
                ["32.1", "其他条款", "以合同为准"],
            ],
        )

        parsed = parse_document("continued.pdf", _pdf_bytes(pdf))
        table_blocks = [
            block
            for block in parsed.text_blocks
            if block.metadata.get("block_type") == "table"
        ]

        self.assertEqual(len(table_blocks), 1)
        table = table_blocks[0]
        self.assertEqual(table.metadata["page_span"], [1, 3])
        self.assertEqual(table.metadata["table_rows"][0], ["条款号", "名称", "内容"])
        self.assertEqual(len(table.metadata["table_rows"]), 3)
        self.assertEqual(table.metadata["table_rows"][1][0], "31.1")
        self.assertEqual(table.metadata["table_rows"][1][1], "履约保证金")
        self.assertIn(
            "不收取收取合同价的2.5% 合同签订时缴纳履约保证金并按合同约定执行",
            table.metadata["table_rows"][1][2],
        )
        self.assertEqual(
            table.metadata["table_rows"][2],
            ["32.1", "其他条款", "以合同为准"],
        )
        self.assertEqual(parsed.metadata["pdf_layout"]["cross_page_merges"], 2)
        self.assertEqual(parsed.metadata["pdf_layout"]["tables"], 3)

    def test_unrelated_same_width_tables_are_not_merged(self):
        pdf = fitz.open()
        first_page = pdf.new_page(width=400, height=500)
        _draw_table(
            first_page,
            [
                ["条款号", "名称", "内容"],
                [
                    "31.1",
                    "履约保证金",
                    "履约保函的提交方式和金额标准详见招标文件",
                ],
            ],
            y=380,
        )
        second_page = pdf.new_page(width=400, height=500)
        _draw_table(
            second_page,
            [
                ["采购序号", "设备名称", "主要规格"],
                ["1", "检测设备", "进口"],
                ["2", "激光设备", "国产"],
            ],
        )

        parsed = parse_document("unrelated-tables.pdf", _pdf_bytes(pdf))
        table_blocks = [
            block
            for block in parsed.text_blocks
            if block.metadata.get("block_type") == "table"
        ]

        self.assertEqual(len(table_blocks), 2)
        self.assertEqual(
            [block.metadata["page_span"] for block in table_blocks],
            [[1, 1], [2, 2]],
        )
        self.assertEqual(parsed.metadata["pdf_layout"]["cross_page_merges"], 0)

    def test_later_empty_first_cell_rows_remain_separate(self):
        pdf = fitz.open()
        first_page = pdf.new_page(width=400, height=500)
        _draw_table(
            first_page,
            [
                ["项目分组", "设备名称", "采购说明内容"],
                ["A", "设备", "主项说明"],
            ],
            y=380,
        )
        second_page = pdf.new_page(width=400, height=500)
        _draw_table(
            second_page,
            [
                ["项目分组", "设备名称", "采购说明内容"],
                ["", "", "续页说明"],
                ["", "", "同组下一个独立项目"],
                ["B", "另一设备", "另一组说明"],
            ],
        )

        parsed = parse_document("grouped-rows.pdf", _pdf_bytes(pdf))
        table = next(
            block
            for block in parsed.text_blocks
            if block.metadata.get("block_type") == "table"
        )

        self.assertEqual(
            table.metadata["table_rows"][1],
            ["A", "设备", "主项说明续页说明"],
        )
        self.assertEqual(
            table.metadata["table_rows"][2],
            ["", "", "同组下一个独立项目"],
        )
        self.assertEqual(
            table.metadata["table_rows"][3],
            ["B", "另一设备", "另一组说明"],
        )

    def test_toc_headings_are_ignored_and_content_gets_heading_paths(self):
        pdf = fitz.open()
        toc_page = pdf.new_page(width=400, height=600)
        for index, heading in enumerate(
            ["第一章 目录甲", "第二章 目录乙", "第三章 目录丙", "第四章 目录丁"]
        ):
            _text(toc_page, heading, 40, 100 + index * 30)
        content_page = pdf.new_page(width=400, height=600)
        _text(content_page, "第一章", 40, 100)
        _text(content_page, "招标公告", 40, 125)
        _text(content_page, "一、项目基本情况", 40, 160)
        _text(content_page, "3.1 合同标的与采购内容说明。", 40, 185)
        _text(content_page, "（一）采购方式", 40, 220)
        _text(content_page, "本项目采用公开招标方式。", 40, 245)

        parsed = parse_document("headings.pdf", _pdf_bytes(pdf))
        knowledge = self.create_knowledge()
        chunks = create_text_chunks(knowledge, parsed)
        toc_chunks = [chunk for chunk in chunks if chunk.metadata["page_index"] == 0]
        section_chunk = next(
            chunk
            for chunk in chunks
            if "一、项目基本情况" in chunk.content
        )
        numeric_heading_chunk = next(
            chunk for chunk in chunks if "3.1 合同标的" in chunk.content
        )
        subsection_chunk = next(
            chunk for chunk in chunks if "（一）采购方式" in chunk.content
        )

        self.assertEqual(parsed.metadata["pdf_layout"]["toc_pages"], [1])
        self.assertTrue(toc_chunks)
        self.assertTrue(all(chunk.metadata["heading_path"] == [] for chunk in toc_chunks))
        self.assertIn(
            "【第一章 招标公告 > 一、项目基本情况】\n",
            section_chunk.content,
        )
        self.assertEqual(
            section_chunk.metadata["heading_path"],
            ["第一章 招标公告", "一、项目基本情况"],
        )
        self.assertEqual(
            numeric_heading_chunk.metadata["heading_path"],
            ["第一章 招标公告", "一、项目基本情况"],
        )
        self.assertEqual(
            subsection_chunk.metadata["heading_path"],
            ["第一章 招标公告", "一、项目基本情况", "（一）采购方式"],
        )

    def test_peer_level_two_headings_replace_the_previous_heading(self):
        pdf = fitz.open()
        page = pdf.new_page(width=400, height=500)
        _text(page, "一、A", 40, 100)
        _text(page, "A部分的正文内容，足以单独形成一个分块。", 40, 125)
        _text(page, "二、B", 40, 160)
        _text(page, "B部分的正文内容，足以单独形成一个分块。", 40, 185)
        next_page = pdf.new_page(width=400, height=500)
        _text(next_page, "B部分跨页后的补充正文内容，也应保留当前标题路径。", 40, 120)

        parsed = parse_document("peer-headings.pdf", _pdf_bytes(pdf))
        chunks = create_text_chunks(self.create_knowledge(), parsed)
        first = next(chunk for chunk in chunks if "一、A" in chunk.content)
        second = next(chunk for chunk in chunks if "二、B" in chunk.content)
        continued = next(
            chunk for chunk in chunks if "B部分跨页后的补充正文" in chunk.content
        )

        self.assertEqual(first.metadata["heading_path"], ["一、A"])
        self.assertEqual(second.metadata["heading_path"], ["二、B"])
        self.assertEqual(continued.metadata["heading_path"], ["二、B"])

    def test_oversized_table_rows_keep_headers_and_key_cells_in_each_piece(self):
        knowledge = self.create_knowledge(chunk_size=256)
        header = ["序号", "名称", "说明"]
        rows = [header, ["1", "关键条款", "详细内容" * 100]]
        parsed = ParsedDocument(
            text_blocks=[
                TextBlock(
                    "| 序号 | 名称 | 说明 |\n| --- | --- | --- |\n| 1 | 关键条款 | 详细内容 |",
                    0,
                    page_index=0,
                    metadata={
                        "block_type": "table",
                        "bbox": [20, 20, 300, 100],
                        "header": header,
                        "table_rows": rows,
                        "page_span": [1, 1],
                    },
                )
            ],
            metadata={
                "pdf_layout": {
                    "tables": 1,
                    "cross_page_merges": 0,
                    "header_footer_removed": 0,
                    "toc_pages": [],
                }
            },
        )

        chunks = create_text_chunks(knowledge, parsed)

        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertEqual(chunk.metadata["block_type"], "table")
            self.assertEqual(chunk.metadata["heading_path"], [])
            self.assertNotIn("table_rows", chunk.metadata)
            self.assertEqual(chunk.metadata["header"], header)
            self.assertEqual(chunk.metadata["page_span"], [1, 1])
            self.assertEqual(chunk.metadata["column_count"], 3)
            self.assertEqual(chunk.metadata["row_range"], [0, 1])
            self.assertIn("| 序号 | 名称 | 说明 |", chunk.content)
            self.assertIn("| 1 | 关键条款 |", chunk.content)

    def test_long_key_cell_uses_a_preview_and_is_split_into_distinct_chunks(self):
        knowledge = self.create_knowledge(chunk_size=256)
        header = ["项目名称", "类型", "说明"]
        long_key = "第一列关键内容需要拆分并保留定位线索" * 30
        rows = [header, [long_key, "关键", "简短说明"]]
        parsed = ParsedDocument(
            text_blocks=[
                TextBlock(
                    "| 项目名称 | 类型 | 说明 |\n| --- | --- | --- |\n| 项目 | 关键 | 简短说明 |",
                    0,
                    page_index=0,
                    metadata={
                        "block_type": "table",
                        "bbox": [20, 20, 300, 100],
                        "header": header,
                        "table_rows": rows,
                        "page_span": [1, 1],
                    },
                )
            ],
            metadata={"pdf_layout": {"toc_pages": []}},
        )

        chunks = create_text_chunks(knowledge, parsed)
        preview = long_key[:40] + "…"
        row_lines = [chunk.content.splitlines()[-1] for chunk in chunks]

        self.assertGreater(len(chunks), 1)
        self.assertEqual(len(set(row_lines)), len(row_lines))
        for chunk in chunks:
            self.assertIn(preview, chunk.content)
            self.assertNotIn(long_key, chunk.content)
            self.assertLessEqual(len(chunk.content), 256)

    def test_disabling_layout_parser_matches_legacy_pdf_behavior(self):
        pdf = fitz.open()
        page = pdf.new_page(width=400, height=500)
        _text(page, "采购文件第1号", 40, 20)
        _text(page, "— 1 —", 40, 490)
        _text(page, "本页包含足够长度的招标正文内容供解析。", 40, 130)
        data = _pdf_bytes(pdf)

        with override_settings(PDF_LAYOUT_PARSING=False):
            parsed = parse_document("legacy.pdf", data)
        legacy = _parse_pdf_legacy("legacy.pdf", data)

        self.assertEqual(parsed, legacy)
        self.assertEqual(parsed.metadata, {})
        self.assertIn("采购文件第1号", "\n".join(block.text for block in parsed.text_blocks))
        self.assertEqual(Chunk.objects.count(), 0)

    def test_eval_cleanup_refuses_same_named_unmarked_knowledge_base(self):
        protected = KnowledgeBase.objects.create(
            tenant=self.tenant,
            name="pdf-eval-legacy",
            description="An unrelated knowledge base",
        )

        with self.assertRaises(CommandError):
            _cleanup_knowledge_base(self.tenant, "pdf-eval-legacy")

        self.assertTrue(
            KnowledgeBase.objects.filter(pk=protected.pk).exists()
        )
