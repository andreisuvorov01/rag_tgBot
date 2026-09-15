"""Загрузчики PDF (мультestrатегия: таблицы + stream-реконструкция по отступам
+ OCR сканов), DOCX и изображений.

Практика Camelot/Docling/Unstructured: одна стратегия не покрывает все PDF,
поэтому на каждой странице пробуем несколько способов и берём лучший:
1) детектированные таблицы (pdfplumber, «lattice/stream» внутри);
2) если таблицы не дали фактов — реконструкция грида из текстовых строк
   по выравниванию (stream-стиль, 2+ пробела = граница колонки);
3) если текстового слоя нет — OCR (локальный Tesseract, затем VLM LLM API).
Каждый шаг пишется в diagnostics — всегда видно, почему что-то не извлеклось.
"""
from __future__ import annotations

import asyncio
import io
import logging
import re
from typing import Protocol

from ..config import settings
from .load_excel import ParsedChunk, ParsedDoc, grid_to_parsed, header_signature
from .normalize import UnitInfo, detect_unit, split_chunks

log = logging.getLogger(__name__)

try:  # OCR опционален: pytesseract + системный Tesseract должны быть установлены
    import pytesseract  # type: ignore

    OCR_AVAILABLE = True
except Exception:  # pragma: no cover
    OCR_AVAILABLE = False


class VlmOcrProto(Protocol):
    enabled: bool

    async def ocr_png(self, png_bytes: bytes) -> str: ...


def _tesseract_sync(pil_image) -> str:
    if not OCR_AVAILABLE:
        return ""
    try:
        return pytesseract.image_to_string(pil_image, lang="rus+eng") or ""
    except Exception as e:
        log.warning("Tesseract: %s", e)
        return ""


def _pil_to_png_bytes(pil_image) -> bytes:
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return buf.getvalue()


async def _ocr_page_image(pil_image, vlm_ocr: VlmOcrProto | None, section: str) -> tuple[str, str]:
    """Распознаёт изображение страницы: Tesseract -> VLM. Возвращает (текст, раздел)."""
    text = await asyncio.to_thread(_tesseract_sync, pil_image)
    if text.strip():
        return text, "OCR"
    if vlm_ocr is not None and vlm_ocr.enabled:
        vlm_text = await vlm_ocr.ocr_png(_pil_to_png_bytes(pil_image))
        if vlm_text.strip():
            return vlm_text, "OCR(VLM)"
    return "", section


def _scan_warning(page_no: int | None, vlm_ocr: VlmOcrProto | None) -> str:
    where = f"страница {page_no}" if page_no else "изображение"
    if not OCR_AVAILABLE:
        hint = (
            f"{where}: скан без текстового слоя. Локальный OCR не настроен — установите "
            f"Tesseract (rus+eng) и 'pip install pytesseract'"
        )
        if vlm_ocr is None or not vlm_ocr.enabled:
            hint += "; либо включите OCR_VLM_ENABLED=true для распознавания через vision-модель LLM API"
        return hint
    hint = f"{where}: скан, OCR не распознал текст"
    if vlm_ocr is None or not vlm_ocr.enabled:
        hint += "; можно включить OCR_VLM_ENABLED=true (распознавание через vision-модель LLM API)"
    return hint


def _text_lines_to_grid(text: str, min_columns: int = 2) -> list[list[str]] | None:
    """Stream-стратегия (Camelot stream): колонки распознаются по выравниванию —
    2+ пробела подряд считаются границей колонки."""
    rows: list[list[str]] = []
    for line in (text or "").splitlines():
        if not line.strip():
            continue
        cells = [c.strip() for c in re.split(r"\s{2,}", line.strip()) if c.strip()]
        if len(cells) >= min_columns:
            rows.append(cells)
    if len(rows) < 3:
        return None
    width = max(len(r) for r in rows)
    return [r + [None] * (width - len(r)) for r in rows]


_UNIT_LINE_RE = re.compile(
    r"(?:единица\s+измерения|ед\.\s*изм\.?|в\s+(?=тыс|млн|млрд))[:\s]*([^\n]{0,40})", re.I
)


def _document_unit(page_text: str) -> UnitInfo | None:
    """«Единица измерения: Тыс. руб.» / «в тыс. руб.» из текста страницы."""
    for m in _UNIT_LINE_RE.finditer(page_text or ""):
        info = detect_unit(m.group(0))
        if info.multiplier > 1 or info.currency:
            return info
    return None


async def load_pdf(path: str, vlm_ocr: VlmOcrProto | None = None) -> ParsedDoc:
    doc = ParsedDoc()
    try:
        import pdfplumber

        pdf = pdfplumber.open(path)
    except Exception as e:
        doc.warnings.append(f"не удалось открыть PDF: {e}")
        return doc
    with pdf:
        doc.sheets.append("pdf")
        # таблица, разорванная переносом страницы: та же шапка -> раздел продолжается
        prev_signature: tuple[str, ...] | None = None
        prev_section: str | None = None
        # единица документа: «Единица измерения: тыс. руб.» на титуле БФО действует
        # на все формы; у самих таблиц единицы нет
        doc_unit: UnitInfo | None = None
        total_pages = len(pdf.pages)
        max_pages = getattr(settings, "max_pdf_pages", 500)
        if total_pages > max_pages:
            doc.warnings.append(
                f"в PDF {total_pages} страниц — обработаны первые {max_pages} "
                f"(лимит MAX_PDF_PAGES); загрузите документ частями"
            )
        for i, page in enumerate(pdf.pages, start=1):
            if i > max_pages:
                break
            # одна битая страница не должна отменять уже разобранные
            try:
                page_text = page.extract_text() or ""
                tables = page.extract_tables() or []
            except Exception as e:
                doc.warnings.append(f"стр. {i}: не удалось прочитать ({e})")
                continue
            if doc_unit is None:
                doc_unit = _document_unit(page_text)

            # стратегия 1: детектированные таблицы
            table_facts = 0
            for t_idx, table in enumerate(tables, start=1):
                grid = [list(row) for row in table if row]
                if not grid:
                    continue
                signature = header_signature(grid)
                carried = prev_section if signature and signature == prev_signature else None
                facts, chunks, warns, sections = grid_to_parsed(
                    grid, f"стр. {i}, таблица {t_idx}", page=i, ledger_out=doc.ledger,
                    initial_section=carried, default_unit=doc_unit,
                )
                prev_signature, prev_section = signature, (sections[-1] if sections else None)
                doc.facts.extend(facts)
                doc.chunks.extend(chunks)
                doc.warnings.extend(warns)
                doc.sections.extend(sections)
                table_facts += len(facts)

            # стратегия 2: таблицы не дали фактов — реконструкция грида из текста
            if table_facts == 0 and len(page_text.strip()) >= 40:
                grid = _text_lines_to_grid(page_text)
                if grid:
                    facts, chunks, warns, sections = grid_to_parsed(
                        grid, f"стр. {i}", page=i, ledger_out=doc.ledger, default_unit=doc_unit,
                    )
                    if facts:
                        doc.facts.extend(facts)
                        doc.chunks.extend(chunks)
                        doc.warnings.extend(warns)
                        doc.sections.extend(sections)
                        doc.diagnostics.append(f"стр. {i}: таблицы не распознаны, факты получены stream-стратегией ({len(facts)})")

            if len(page_text.strip()) < 20:
                ocr_text, section = "", "OCR"
                try:
                    pil = page.to_image(resolution=200).original
                    ocr_text, section = await _ocr_page_image(pil, vlm_ocr, "OCR")
                except Exception as e:
                    log.warning("Рендер страницы %s: %s", i, e)
                if ocr_text.strip():
                    for ch in split_chunks(ocr_text):
                        doc.chunks.append(ParsedChunk(ch, page=i, section=section))
                    continue
                doc.warnings.append(_scan_warning(i, vlm_ocr))
                continue
            for ch in split_chunks(page_text):
                doc.chunks.append(ParsedChunk(ch, page=i, section=None))
        doc.diagnostics.append(f"страниц: {total_pages}")
    return doc


async def load_docx(path: str) -> ParsedDoc:
    doc = ParsedDoc()
    try:
        import docx as docx_lib

        document = docx_lib.Document(path)
    except Exception as e:
        doc.warnings.append(f"не удалось открыть DOCX: {e}")
        return doc
    doc.sheets.append("docx")
    full_text = "\n\n".join(p.text for p in document.paragraphs if p.text.strip())
    for ch in split_chunks(full_text):
        doc.chunks.append(ParsedChunk(ch))
    for t_idx, table in enumerate(document.tables, start=1):
        grid = [[cell.text for cell in row.cells] for row in table.rows]
        if grid:
            facts, chunks, warns, _sections = grid_to_parsed(grid, f"таблица {t_idx}", ledger_out=doc.ledger)
            doc.facts.extend(facts)
            doc.chunks.extend(chunks)
            doc.warnings.extend(warns)
    return doc


async def load_image(path: str, vlm_ocr: VlmOcrProto | None = None) -> ParsedDoc:
    doc = ParsedDoc()
    if not OCR_AVAILABLE and (vlm_ocr is None or not vlm_ocr.enabled):
        doc.warnings.append(_scan_warning(None, vlm_ocr))
        return doc
    try:
        from PIL import Image

        with Image.open(path) as img:
            img.load()
            text, section = await _ocr_page_image(img, vlm_ocr, "OCR")
    except Exception as e:
        doc.warnings.append(f"не удалось открыть изображение: {e}")
        return doc
    if text.strip():
        for ch in split_chunks(text):
            doc.chunks.append(ParsedChunk(ch, section=section))
    else:
        doc.warnings.append(_scan_warning(None, vlm_ocr))
    return doc
