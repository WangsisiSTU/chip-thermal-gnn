"""Render the unified technical Markdown document to a self-contained PDF.

The source of truth remains ``docs/technical_documentation.md``.  The renderer
uses ReportLab's built-in CJK font support, avoiding a system TeX or Pango
installation on Windows.
"""
from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import Image, Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "docs" / "technical_documentation.md"
DEFAULT_OUTPUT = ROOT / "docs" / "chip_thermal_gnn_technical_documentation.pdf"
FONT = "STSong-Light"
CODE_FONT = "Courier"


def make_styles() -> dict[str, ParagraphStyle]:
    pdfmetrics.registerFont(UnicodeCIDFont(FONT))
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ChineseTitle", parent=base["Title"], fontName=FONT, fontSize=22, leading=30,
            textColor=colors.HexColor("#102A43"), alignment=TA_CENTER, spaceAfter=12,
        ),
        "h2": ParagraphStyle(
            "ChineseH2", parent=base["Heading2"], fontName=FONT, fontSize=15, leading=21,
            textColor=colors.HexColor("#0B4F6C"), spaceBefore=15, spaceAfter=7,
        ),
        "h3": ParagraphStyle(
            "ChineseH3", parent=base["Heading3"], fontName=FONT, fontSize=12, leading=17,
            textColor=colors.HexColor("#155E75"), spaceBefore=11, spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "ChineseBody", parent=base["BodyText"], fontName=FONT, fontSize=9.6, leading=15.2,
            alignment=TA_JUSTIFY, spaceAfter=5,
        ),
        "quote": ParagraphStyle(
            "ChineseQuote", parent=base["BodyText"], fontName=FONT, fontSize=9.4, leading=14.5,
            leftIndent=10, rightIndent=8, textColor=colors.HexColor("#334155"),
            borderColor=colors.HexColor("#8ECAE6"), borderWidth=1.5, borderPadding=5,
        ),
        "table": ParagraphStyle(
            "ChineseTable", parent=base["BodyText"], fontName=FONT, fontSize=6.7, leading=9.0,
        ),
        "math": ParagraphStyle(
            "Math", parent=base["Code"], fontName=CODE_FONT, fontSize=7.8, leading=10.2,
            leftIndent=9, rightIndent=9, textColor=colors.HexColor("#1E293B"),
        ),
    }


def inline_markup(text: str) -> str:
    """Convert a conservative Markdown subset into ReportLab paragraph markup."""
    escaped = html.escape(text.strip())
    escaped = re.sub(
        r"!\[[^]]*\]\([^)]*\)", "", escaped,
    )
    escaped = re.sub(
        r"\[([^]]+)\]\(([^ )]+)\)",
        r'<link href="\2" color="#075985">\1</link>',
        escaped,
    )
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(r"`([^`]+)`", rf'<font name="{CODE_FONT}" size="7.5">\1</font>', escaped)
    escaped = re.sub(r"\\\((.*?)\\\)", rf'<font name="{CODE_FONT}" size="7.5">\1</font>', escaped)
    return escaped


def add_image(story: list, source_dir: Path, markdown_line: str) -> bool:
    match = re.search(r"!\[([^]]*)\]\(([^ )]+)\)", markdown_line)
    if not match:
        return False
    image_path = (source_dir / match.group(2)).resolve()
    if not image_path.exists():
        story.append(Paragraph(f"[图像未找到：{html.escape(match.group(2))}]", make_styles()["body"]))
        return True
    image = Image(str(image_path))
    max_width, max_height = 170 * mm, 210 * mm
    scale = min(max_width / image.imageWidth, max_height / image.imageHeight, 1.0)
    image.drawWidth = image.imageWidth * scale
    image.drawHeight = image.imageHeight * scale
    story.append(image)
    return True


def build_table(rows: list[str], styles: dict[str, ParagraphStyle]) -> Table:
    parsed = []
    for row in rows:
        cells = [cell.strip() for cell in row.strip().strip("|").split("|")]
        if all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in cells):
            continue
        parsed.append([Paragraph(inline_markup(cell), styles["table"]) for cell in cells])
    n_cols = max(len(row) for row in parsed)
    for row in parsed:
        row.extend([Paragraph("", styles["table"])] * (n_cols - len(row)))
    width = 178 * mm
    table = Table(parsed, colWidths=[width / n_cols] * n_cols, repeatRows=1, hAlign="CENTER")
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), FONT),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#B9C7D1")),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E7F4FA")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0C4A6E")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return table


def markdown_to_story(source: Path) -> list:
    styles = make_styles()
    story: list = []
    lines = source.read_text(encoding="utf-8").splitlines()
    paragraph: list[str] = []

    def flush_paragraph() -> None:
        if paragraph:
            story.append(Paragraph(inline_markup(" ".join(paragraph)), styles["body"]))
            paragraph.clear()

    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if not stripped:
            flush_paragraph()
            index += 1
            continue
        if stripped.startswith("```"):
            flush_paragraph()
            index += 1
            code = []
            while index < len(lines) and not lines[index].strip().startswith("```"):
                code.append(lines[index])
                index += 1
            story.append(Preformatted("\n".join(code), styles["math"], maxLineLength=118))
            story.append(Spacer(1, 4))
            index += 1
            continue
        if stripped == r"\[":
            flush_paragraph()
            index += 1
            equation = []
            while index < len(lines) and lines[index].strip() != r"\]":
                equation.append(lines[index])
                index += 1
            story.append(Preformatted("\n".join(equation), styles["math"], maxLineLength=112))
            story.append(Spacer(1, 4))
            index += 1
            continue
        if stripped.startswith("|") and "|" in stripped[1:]:
            flush_paragraph()
            table_rows = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_rows.append(lines[index])
                index += 1
            story.append(build_table(table_rows, styles))
            story.append(Spacer(1, 5))
            continue
        if add_image(story, source.parent, stripped):
            flush_paragraph()
            index += 1
            continue
        heading = re.match(r"^(#{1,3})\s+(.+)$", stripped)
        if heading:
            flush_paragraph()
            level, title = len(heading.group(1)), heading.group(2)
            style = styles["title"] if level == 1 else styles["h2"] if level == 2 else styles["h3"]
            story.append(Paragraph(inline_markup(title), style))
            index += 1
            continue
        if stripped.startswith("> "):
            flush_paragraph()
            story.append(Paragraph(inline_markup(stripped[2:]), styles["quote"]))
            index += 1
            continue
        if re.match(r"^[-*]\s+", stripped):
            flush_paragraph()
            story.append(Paragraph("• " + inline_markup(re.sub(r"^[-*]\s+", "", stripped)), styles["body"]))
            index += 1
            continue
        if re.match(r"^\d+\.\s+", stripped):
            flush_paragraph()
            story.append(Paragraph(inline_markup(stripped), styles["body"]))
            index += 1
            continue
        if stripped in {"---", "***"}:
            flush_paragraph()
            story.append(Spacer(1, 5))
            index += 1
            continue
        paragraph.append(stripped)
        index += 1
    flush_paragraph()
    return story


def add_footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setTitle("chip-thermal-gnn 统一技术说明")
    canvas.setAuthor("chip-thermal-gnn")
    canvas.setFont(FONT, 8)
    canvas.setFillColor(colors.HexColor("#667085"))
    canvas.drawCentredString(A4[0] / 2, 9 * mm, f"chip-thermal-gnn 统一技术说明 | 第 {doc.page} 页")
    canvas.restoreState()


def render(source: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(output), pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm,
        topMargin=18 * mm, bottomMargin=18 * mm, title="chip-thermal-gnn 统一技术说明",
    )
    document.build(markdown_to_story(source), onFirstPage=add_footer, onLaterPages=add_footer)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the unified technical document to PDF.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    render(args.source.resolve(), args.output.resolve())
    print(f"PDF written to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
