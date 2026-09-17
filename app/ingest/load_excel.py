"""Загрузчики Excel/CSV + детальный парсер финансовых грид-таблиц.

Поддерживаемая точность:
- макет «показатель × период» и вертикальный (периоды в первой колонке);
- многоярусные шапки с объединёнными ячейками (xlsx): группы над периодами;
- варианты колонок план/факт/бюджет/прогноз/оценка -> сохраняются отдельно;
- разделы таблицы: строка-заголовок без чисел становится родительским
  показателем, «Итого/Всего» получает имя с префиксом раздела и связывается
  с ним, строки «в т.ч. …» становятся дочерними к предыдущему показателю;
- единицы измерения: из названия листа, шапки, суффикса строки
  («Выручка, тыс. руб.») и прямо из ячейки («12,7 млн»);
- числа: бухскобки-минус, пробельные разделители, «1.234,56», проценты;
- листы без периодов сохраняются как текстовые чанки для RAG.
"""
from __future__ import annotations

import io
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from .normalize import (
    _NUM_WITH_UNIT_STRIP_RE,
    Period,
    UnitInfo,
    _guess_kind,
    clean_metric_name,
    detect_unit,
    detect_variant,
    is_calendar_date,
    parse_number,
    parse_number_with_unit,
    parse_period,
    strip_unit_suffix,
)

log = logging.getLogger(__name__)

_SERVICE_ROW_RE = ("№", "код строки", "примечание", "примечания")
_TOTAL_PREFIXES = ("итого", "всего", "баланс")
# «Итого», «Всего», «Итого по разделу II» — безымянный итог раздела: получает имя раздела
_BARE_TOTAL_RE = re.compile(r"^(итого|всего)(\s+по\s+разделу\s+[ivx\d]+)?[:.]?$", re.I)
# «I. Внеоборотные активы», «2) Расходы» — нумерация раздела не часть имени
_SECTION_PREFIX_RE = re.compile(r"^(?:[ivx]+|\d+)[.)]\s*", re.I)
_VTH_BARE_RE = re.compile(r"^в\s*(т\.?\s*ч\.?|том\s+числе)\s*:?$", re.I)
_SHARE_NAMES = ("% к итогу", "уд. вес", "доля", "% от итога")
_VTH_RE = re.compile(r"^в\s*т\.?\s*ч\.?|^в\s*том\s+числе", re.I)


@dataclass
class ParsedFact:
    metric_name: str
    period: Period
    value: float
    unit: str | None
    currency: str | None
    sheet: str
    cell_ref: str
    variant: str = "fact"  # fact | plan | budget | forecast | estimate
    section: str | None = None
    parent_hint: str | None = None
    is_total: bool = False


@dataclass
class ParsedChunk:
    text: str
    page: int | None = None
    section: str | None = None


@dataclass
class LedgerTable:
    """Таблица-«выписка»: колонка дат + числовые колонки-суммы. Хранится
    ПОСТРОЧНО (каждая операция отдельно, с описанием) — агрегация по месяцам
    выполняется на уровне документа, а операции доступны для просмотра
    «операции за месяц» и категорий."""

    sheet: str
    header: str
    unit: str | None
    currency: str | None
    rows: list[tuple[Period, float, str]] = field(default_factory=list)  # (дата, сумма, описание)


@dataclass
class ParsedDoc:
    facts: list[ParsedFact] = field(default_factory=list)
    chunks: list[ParsedChunk] = field(default_factory=list)
    sections: list[str] = field(default_factory=list)
    ledger: list[LedgerTable] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)
    sheets: list[str] = field(default_factory=list)


def _is_service_name(name: str) -> bool:
    """«№», «№ п/п», «Код строки», «Примечание» — не показатели."""
    low = name.casefold()
    return low in ("№", "no", "n", "п/п", "№ п/п", "код") or any(tok in low for tok in _SERVICE_ROW_RE)


def _cell_letter(col: int) -> str:
    letters = ""
    col += 1
    while col:
        col, rem = divmod(col - 1, 26)
        letters = chr(65 + rem) + letters
    return letters


def _cell_ref(row: int, col: int) -> str:
    return f"{_cell_letter(col)}{row + 1}"


def _norm_text(v: object) -> str:
    if v is None:
        return ""
    return unicodedata.normalize("NFKC", str(v)).replace("\n", " ").strip()


def _is_texty(v: object) -> bool:
    """Текст ли это (а не число и не период).

    «12,7 млн» и «12,7%» — значения с единицей, а не названия показателей.
    Раньше parse_number отбрасывал их (в строке остаются буквы единицы), и
    такая ячейка считалась текстом: в макете «показатель | 12,7 млн»
    колонкой показателей выбиралась колонка со значением, строка не давала
    ни одного факта, а данные молча терялись.
    """
    if v is None or isinstance(v, (int, float, datetime, date)):
        return False
    s = _norm_text(v)
    if not s:
        return False
    if parse_period(s) is not None:
        return False
    info = detect_unit(s)
    if info.multiplier != 1.0 or info.unit or info.currency:
        stripped = _NUM_WITH_UNIT_STRIP_RE.sub(" ", s).strip()
        if stripped and parse_number(stripped) is not None:
            return False  # число с единицей — это значение
    return parse_number(s) is None


def _has_number(v: object) -> bool:
    """Есть ли в ячейке числовое значение (в т.ч. с единицей: «12,7 млн»)."""
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return True
    if v is None or isinstance(v, (datetime, date)):
        return False
    s = _norm_text(v)
    if not s:
        return False
    if parse_number(s) is not None:
        return True
    value, _ = parse_number_with_unit(s)
    return value is not None


def _xlsx_sheets(path: str) -> list[tuple[str, list[list[object]]]] | None:
    """Чтение xlsx через openpyxl с заполнением объединённых ячеек значением
    якоря — иначе многоярусные шапки и разделы теряют половину текста."""
    try:
        import openpyxl

        wb = openpyxl.load_workbook(path, data_only=True)
    except Exception as e:
        log.debug("openpyxl не открыл %s: %s", path, e)
        return None
    out = []
    for ws in wb.worksheets:
        # max_row/max_column у листа с форматированием доходят до 1 048 576 × 16 384 —
        # читаем построчно и обрезаем пустой хвост, а не выделяем грид по размерам листа
        n_cols = min(ws.max_column or 0, 512)
        grid: list[list[object]] = []
        empty_run = 0
        for row in ws.iter_rows(max_col=n_cols, values_only=True):
            cells = list(row)
            grid.append(cells)
            empty_run = 0 if any(v is not None for v in cells) else empty_run + 1
            if empty_run > 200 or len(grid) > 50_000:
                break  # ponytail: 200 пустых строк подряд = конец данных
        while grid and all(v is None for v in grid[-1]):
            grid.pop()
        for mr in ws.merged_cells.ranges:
            anchor = ws.cell(mr.min_row, mr.min_col).value
            for r in range(mr.min_row, min(mr.max_row, len(grid)) + 1):
                for c in range(mr.min_col, min(mr.max_col, n_cols) + 1):
                    if grid[r - 1][c - 1] is None:
                        grid[r - 1][c - 1] = anchor
        out.append((str(ws.title), grid))
    return out


def _cell_value(raw: object, base_multiplier: float) -> tuple[float | None, UnitInfo | None]:
    """(значение, единица из ячейки).

    Приоритет: явная единица в ячейке («%», «шт») и собственный множитель
    («млн», «тыс») важнее масштаба листа. Ячейка «12,7%» в листе «тыс. руб.»
    должна остаться 12,7%, а не превратиться в 12 700; «12,7 млн» не должно
    умножаться на множитель листа повторно. Множители ячейки и листа не
    перемножаются: они задают один и тот же масштаб разными способами.
    """
    if isinstance(raw, str):
        info = detect_unit(raw)
        has_explicit = bool(info.unit) or re.search(r"%|\bшт\b|количество|кол-во", raw, re.I)
        if has_explicit or info.multiplier != 1.0:
            value, unit_info = parse_number_with_unit(raw)
            if value is not None:
                if info.multiplier == 1.0 and not info.unit and info.currency:
                    # только валюта в ячейке — масштаб берём из листа
                    value *= base_multiplier
                return value, unit_info
        return parse_number(raw, base_multiplier), None
    value = parse_number(raw, base_multiplier)
    if value is not None:
        return value, None
    return None, None


def _facts_to_text_chunks(facts: list[ParsedFact], sheet_name: str, page: int | None, limit: int = 60) -> list[ParsedChunk]:
    """Таблица -> текстовые чанки «Показатель «X»: 2023 — 9,80 млн ₽; …».
    Практика RAGFlow (шаблонный чанкинг таблиц): сохранённые связи
    строка×колонка становятся находимыми ключевым поиском, а не только точным SQL."""
    from ..formatting import fmt_money

    if not facts or len(facts) > 400:
        return []
    by_metric: dict[str, list[ParsedFact]] = {}
    for f in facts:
        if _guess_kind(f.metric_name) == "identifier":
            continue  # «Показатель «инн»: … 7 721,55 млрд ₽» — не текст для поиска
        by_metric.setdefault(f.metric_name, []).append(f)
    chunks: list[ParsedChunk] = []
    for name, fs in list(by_metric.items())[:limit]:
        fs = sorted(fs, key=lambda f: (f.period.end, f.variant))
        unit_note = f", {fs[0].unit}" if fs[0].unit else ""
        parts = []
        for f in fs:
            val = fmt_money(f.value, f.currency) if f.unit != "%" else f"{f.value:g}%"
            suffix = "" if f.variant == "fact" else f" ({f.variant})"
            parts.append(f"{f.period.label}{suffix} — {val}")
        text = f"Показатель «{name}»{unit_note}: " + "; ".join(parts) + "."
        if len(text) <= 1500:
            chunks.append(ParsedChunk(text, page=page, section=f"{sheet_name}:{name[:40]}"))
    return chunks


def _try_ledger(
    grid: list[list[object]], n_rows: int, n_cols: int, sheet_name: str
) -> list[LedgerTable] | None:
    """Ledger-режим (практика парсинга выписок): таблица без шапки-периодов,
    но с колонкой дат и числовыми колонками. Каждая строка — операция;
    результат — помесячные суммы по каждой числовой колонке."""
    if n_rows < 4:
        return None

    # колонка дат: большинство ячеек — даты
    date_col, date_count = None, 0
    for c in range(n_cols):
        cnt = 0
        for r in range(1, n_rows):
            v = grid[r][c] if c < len(grid[r]) else None
            if v is None:
                continue
            if is_calendar_date(v):  # выписка — даты операций; месяцы в колонке — вертикальный макет
                cnt += 1
        if cnt > date_count:
            date_col, date_count = c, cnt
    if date_col is None or date_count < 3 or date_count < (n_rows - 1) * 0.25:
        return None

    numeric_cols: list[int] = []
    for c in range(n_cols):
        if c == date_col:
            continue
        nums = texts = 0
        for r in range(1, n_rows):
            v = grid[r][c] if c < len(grid[r]) else None
            if v is None or (isinstance(v, str) and not v.strip()):
                continue
            if parse_number(v) is not None and not _is_texty(v):
                nums += 1
            elif _is_texty(v):
                texts += 1
        if nums >= 2 and nums >= texts:
            numeric_cols.append(c)
    if not numeric_cols:
        return None

    sheet_unit = detect_unit(sheet_name)
    tables: list[LedgerTable] = []
    for c in numeric_cols:
        header = clean_metric_name(grid[0][c] if c < len(grid[0]) else None)
        header = (header or f"сумма (колонка {c + 1})").casefold()
        info = detect_unit(header)
        multiplier = info.multiplier or sheet_unit.multiplier
        rows: list[tuple[Period, float, str]] = []
        for r in range(1, n_rows):
            dcell = grid[r][date_col] if date_col < len(grid[r]) else None
            p = parse_period(dcell) if is_calendar_date(dcell) else None
            if p is None:
                continue
            vcell = grid[r][c] if c < len(grid[r]) else None
            value = parse_number(vcell, multiplier)
            if value is None and isinstance(vcell, str):
                value, _info = parse_number_with_unit(vcell)
            if value is None:
                continue
            # описание операции — самый длинный текст в строке (кроме даты и суммы)
            description = ""
            for cc in range(n_cols):
                if cc in (date_col, c):
                    continue
                cell = grid[r][cc] if cc < len(grid[r]) else None
                if _is_texty(cell):
                    t = _norm_text(cell)
                    if len(t) > len(description):
                        description = t[:200]
            rows.append((p, value, description))
        if not rows:
            continue
        tables.append(LedgerTable(
            sheet=sheet_name, header=header,
            unit=info.unit or sheet_unit.unit,
            currency=info.currency or sheet_unit.currency,
            rows=rows,
        ))
    return tables or None


def disambiguate_by_section(facts: list[ParsedFact]) -> None:
    """Одно имя в разных разделах — разные показатели: «Заемные средства»
    есть и в долгосрочных, и в краткосрочных обязательствах баланса.
    Такие строки получают суффикс раздела, иначе значения перезаписывают друг друга."""
    sections_by_name: dict[str, set[str]] = {}
    for f in facts:
        if f.section:
            sections_by_name.setdefault(f.metric_name, set()).add(f.section)
    clash = {n for n, s in sections_by_name.items() if len(s) > 1}
    for f in facts:
        if f.metric_name in clash and f.section and not f.metric_name.startswith(f.section):
            f.metric_name = f"{f.metric_name} ({f.section})"
        if f.parent_hint in clash and f.section:
            f.parent_hint = f"{f.parent_hint} ({f.section})"


def header_signature(grid: list[list[object]]) -> tuple[str, ...] | None:
    """Шапка таблицы (строка периодов) — чтобы понять, что таблица на следующей
    странице PDF продолжает предыдущую и раздел переносится через разрыв."""
    header_row, _ = _find_header(grid, len(grid), max((len(r) for r in grid), default=0))
    if header_row is None:
        return None
    return tuple(_norm_text(v).casefold() for v in grid[header_row])


def grid_to_parsed(
    grid: list[list[object]], sheet_name: str, page: int | None = None,
    ledger_out: list[LedgerTable] | None = None, initial_section: str | None = None,
    default_unit: UnitInfo | None = None, doc_name: str | None = None,
) -> tuple[list[ParsedFact], list[ParsedChunk], list[str], list[str]]:
    """-> (факты, чанки, предупреждения, разделы). При ledger_out — выписки
    (колонка дат + суммы) попадают туда для агрегации на уровне документа.
    initial_section — раздел, открытый на предыдущей странице той же таблицы;
    default_unit — единица документа (БФО: «Единица измерения: тыс. руб.» на
    титульной странице), если у самой таблицы единицы нет."""
    facts: list[ParsedFact] = []
    chunks: list[ParsedChunk] = []
    warnings: list[str] = []
    sections: list[str] = []
    n_rows = len(grid)
    n_cols = max((len(r) for r in grid), default=0)

    header_row, period_cols = _find_header(grid, n_rows, n_cols)

    if header_row is None:
        # ledger-режим первым: выписка «дата | описание | суммы» специфичнее
        # вертикального макета и должна перехватывать её
        ledger = _try_ledger(grid, n_rows, n_cols, sheet_name)
        if ledger and ledger_out is not None:
            ledger_out.extend(ledger)
            sheet_text = _grid_text(grid)
            if sheet_text:
                chunks.append(ParsedChunk(sheet_text, page=page, section=sheet_name))
            return facts, chunks, warnings, sections

        vert_facts = _try_vertical_layout(grid, n_rows, n_cols, sheet_name)
        if vert_facts:
            sheet_text = _grid_text(grid)
            if sheet_text:
                chunks.append(ParsedChunk(sheet_text, page=page, section=sheet_name))
            return vert_facts, chunks, warnings, sections

        cat = _try_categorical_layout(grid, n_rows, n_cols, sheet_name, doc_name)
        if cat:
            cat_facts, cat_warns, cat_section = cat
            warnings.extend(cat_warns)
            if cat_section:
                sections.append(cat_section)
            sheet_text = _grid_text(grid)
            if sheet_text:
                chunks.append(ParsedChunk(sheet_text, page=page, section=sheet_name))
            chunks.extend(_facts_to_text_chunks(cat_facts, sheet_name, page))
            return cat_facts, chunks, warnings, sections

        sheet_text = _grid_text(grid)
        if sheet_text:
            chunks.append(ParsedChunk(sheet_text, page=page, section=sheet_name))
        return facts, chunks, warnings, sections

    first_pcol = min(period_cols)
    metric_col = _find_metric_col(grid, header_row, first_pcol)

    # спецификация колонок: период + вариант (план/факт/…) из соседних ярусов шапки
    col_spec: dict[int, tuple[Period, str]] = {}
    for c, period in period_cols.items():
        variant = "fact"
        for qr in range(header_row - 2, header_row + 2):
            if qr == header_row or qr < 0 or qr >= n_rows:
                continue
            qtext = _norm_text(grid[qr][c] if c < len(grid[qr]) else None)
            if qtext and (detect_variant(qtext) or len(qtext) <= 15):
                v = detect_variant(f"{qtext} {_norm_text(grid[header_row][c])}")
                if v:
                    variant = v
                    break
        col_spec[c] = (period, variant)

    # единицы: название листа -> шапка над заголовком -> сама строка заголовка
    # (единица нередко стоит в первой строке рядом с периодами)
    sheet_unit = detect_unit(sheet_name)
    for r in range(max(0, header_row - 3), header_row + 1):
        for c in range(n_cols):
            if r == header_row and c in period_cols:
                continue
            v = grid[r][c]
            if v is None:
                continue
            info = detect_unit(str(v))
            if info.multiplier > 1 or info.currency:
                sheet_unit = info
                break
        if sheet_unit.multiplier > 1 or sheet_unit.currency:
            break
    if default_unit and not (sheet_unit.multiplier > 1 or sheet_unit.currency):
        sheet_unit = default_unit

    current_section: str | None = initial_section
    if initial_section:
        sections.append(initial_section)
    prev_metric_name: str | None = None

    prev_main_metric: str | None = None  # последний показатель из главной колонки (родитель «отступов»)

    for r in range(header_row + 1, n_rows):
        row = grid[r]
        name = clean_metric_name(row[metric_col]) if metric_col is not None and metric_col < len(row) else ""
        indented = False
        if not name:
            # ОФР/ОДДС: подстроки «в том числе» стоят в соседней колонке правее главной;
            # строки-разделы баланса («I. Внеоборотные активы») — в колонке левее
            right = [c for c in range((metric_col or 0) + 1, first_pcol) if _is_texty(row[c] if c < len(row) else None)]
            left = [c for c in range(0, metric_col or 0) if _is_texty(row[c] if c < len(row) else None)]
            if right:
                name, indented = clean_metric_name(row[right[0]]), True
            elif left:
                name = clean_metric_name(row[left[0]])
        if not name or parse_period(name) is not None:
            continue
        if _is_service_name(name):
            continue
        if _VTH_BARE_RE.match(name):
            continue  # «в том числе:» — не показатель и не раздел, дети идут следом с отступом
        row_unit = detect_unit(name)
        # Масштаб листа применяется, только если у строки нет своей единицы.
        # «Доля, %» — своя единица (multiplier=1.0), и раньше она наследовала
        # множитель листа: 12,5% в листе «тыс. руб.» становились 12500.
        if row_unit.unit or row_unit.currency or row_unit.multiplier != 1.0:
            multiplier = row_unit.multiplier
        else:
            multiplier = sheet_unit.multiplier
        currency = row_unit.currency or sheet_unit.currency
        unit = row_unit.unit or sheet_unit.unit
        name = strip_unit_suffix(name)  # «Выручка, тыс. руб.» -> «Выручка»

        # сначала собираем значения строки, чтобы отличить раздел от данных
        row_values: list[tuple[int, float, UnitInfo | None]] = []
        for c in col_spec:  # период и вариант здесь не нужны — берём их позже по c
            raw = grid[r][c] if c < len(grid[r]) else None
            value, cell_info = _cell_value(raw, multiplier)
            if value is None:
                continue
            row_values.append((c, value, cell_info))

        if not row_values:
            if any(_norm_text(grid[r][c] if c < len(grid[r]) else None) for c in col_spec):
                continue  # значения есть, но не числа («-», «н/д»): строка данных без фактов, не раздел
            # строка-раздел: заголовок группы без чисел
            current_section = _SECTION_PREFIX_RE.sub("", name.casefold()).strip()
            prev_main_metric = None
            if current_section not in sections:
                sections.append(current_section)
            text_vals = [
                _norm_text(grid[r][c]) for c in range(len(grid[r])) if _norm_text(grid[r][c])
            ]
            if len(text_vals) >= 3:
                chunks.append(
                    ParsedChunk(" | ".join(text_vals), page=page, section=f"{sheet_name}:{name[:60]}")
                )
            continue

        low = name.casefold()
        if low.isdigit():
            continue  # строка нумерации колонок («1 2 3 4 5 6»)
        is_total = low.startswith(_TOTAL_PREFIXES)
        metric_name = name.casefold()
        parent_hint: str | None = None

        m_vth = _VTH_RE.match(low)
        if m_vth and prev_metric_name:
            # «в т.ч. …» — дочерняя разбивка предыдущего показателя
            rest = clean_metric_name(low[m_vth.end():].strip(" :,.–-"))
            metric_name = f"{prev_metric_name} — {rest}" if rest else f"{prev_metric_name} (в т.ч.)"
            parent_hint = prev_metric_name
        elif indented and prev_main_metric:
            # подстрока с отступом (ОДДС: «Платежи — всего» -> «в связи с оплатой труда…»)
            parent_hint = prev_main_metric
        elif low.startswith("баланс"):
            current_section = None  # «Баланс (актив)» — итог всей формы, а не раздела
        elif is_total and current_section and _BARE_TOTAL_RE.match(low):
            metric_name = f"{current_section} итого"
        elif is_total and current_section and low.startswith(("итого", "всего")):
            metric_name = f"{current_section} {low}"
        elif low in _SHARE_NAMES and current_section:
            metric_name = f"{current_section} {low}"
            unit = unit or "%"

        for c, value, cell_info in row_values:
            period, variant = col_spec[c]
            f_unit = cell_info.unit if cell_info and cell_info.unit else unit
            f_cur = cell_info.currency if cell_info and cell_info.currency else currency
            facts.append(
                ParsedFact(
                    metric_name=metric_name,
                    period=period,
                    value=value,
                    unit=f_unit,
                    currency=f_cur,
                    sheet=sheet_name,
                    cell_ref=_cell_ref(r, c),
                    variant=variant,
                    section=current_section,
                    parent_hint=parent_hint,
                    is_total=is_total,
                )
            )
        prev_metric_name = metric_name
        if not indented:
            prev_main_metric = metric_name

    # раздел без единого факта («Настоящая выгрузка содержит информацию…» на титуле
    # БФО) — просто строка текста, в словарь показателей ей не место
    used = {f.section for f in facts if f.section}
    sections = [sec for sec in sections if sec in used]

    sheet_text = _grid_text(grid)
    if sheet_text:
        chunks.append(ParsedChunk(sheet_text, page=page, section=sheet_name))
    chunks.extend(_facts_to_text_chunks(facts, sheet_name, page))

    if not facts and not chunks:
        warnings.append(f"лист «{sheet_name}»: не найдено ни данных, ни текста")
    return facts, chunks, warnings, sections


def _period_from_header(v: object) -> Period | None:
    """Период из заголовка колонки, терпимый к суффиксам варианта: «2026 (план)»."""
    p = parse_period(v)
    if p is not None:
        return p
    s = _norm_text(v)
    if not s:
        return None
    stripped = re.sub(r"\([^)]*\)", " ", s).strip()
    if detect_variant(s) or detect_variant(stripped):
        return parse_period(stripped) or parse_period(stripped.split()[-1] if stripped.split() else "")
    return None


def _find_header(grid: list[list[object]], n_rows: int, n_cols: int) -> tuple[int | None, dict[int, Period]]:
    best_row, best_cols, best_count = None, {}, 0
    for r in range(min(25, n_rows)):
        cols: dict[int, Period] = {}
        for c in range(n_cols):
            v = grid[r][c] if c < len(grid[r]) else None
            if v is None:
                continue
            p = _period_from_header(v)
            if p is not None:
                cols[c] = p
        if len(cols) >= 2 and len(cols) > best_count:
            best_row, best_cols, best_count = r, cols, len(cols)
    if best_row is not None:
        return best_row, best_cols
    for r in range(min(25, n_rows)):
        cols: dict[int, Period] = {}
        for c in range(n_cols):
            v = grid[r][c] if c < len(grid[r]) else None
            p = _period_from_header(v) if v is not None else None
            if p is not None:
                cols[c] = p
        if len(cols) == 1:
            left = [c for c in range(next(iter(cols))) if _is_texty(grid[r][c] if c < len(grid[r]) else None)]
            if left:
                return r, cols
    return None, {}


def _find_metric_col(grid: list[list[object]], header_row: int, first_pcol: int) -> int | None:
    """Колонка названий — та, где текста больше всего (в ОФР/ОДДС подстроки
    «в том числе» лежат правее, и колонка отступов не должна победить)."""
    # считаем только строки с числами: строки-разделы («I. Внеоборотные активы»
    # в колонке «Пояснения») не должны сделать служебную колонку главной.
    # «Число» здесь — в том числе значение с единицей в ячейке («12,7 млн»),
    # иначе такие строки выпадают и колонка показателей определяется неверно.
    data_rows = [
        r for r in range(header_row + 1, len(grid))
        if any(_has_number(grid[r][c]) for c in range(first_pcol, len(grid[r])))
    ]
    total = max(1, len(data_rows))
    ratios = {}
    for c in range(first_pcol):
        texty = sum(1 for r in data_rows if _is_texty(grid[r][c] if c < len(grid[r]) else None))
        ratios[c] = texty / total
    # имена выровнены по левому краю, отступы «в том числе» уходят правее:
    # берём самую левую колонку с заметной долей текста
    good = [c for c, ratio in ratios.items() if ratio >= 0.2]
    if good:
        return good[0]
    best = max(ratios, key=ratios.get, default=None)
    return best if best is not None and ratios[best] > 0 else None


def _try_vertical_layout(
    grid: list[list[object]], n_rows: int, n_cols: int, sheet_name: str
) -> list[ParsedFact] | None:
    # колонка периодов — среди первых трёх (перед ней бывает «№»)
    period_rows: dict[int, Period] = {}
    pcol = 0
    for c in range(min(3, n_cols)):
        found: dict[int, Period] = {}
        for r in range(min(60, n_rows)):
            v = grid[r][c] if c < len(grid[r]) else None
            p = parse_period(v) if v is not None else None
            if p is not None:
                found[r] = p
        if len(found) > len(period_rows):
            period_rows, pcol = found, c
    if len(period_rows) < 3:
        return None
    # шапка с именами показателей — ближайшая текстовая строка над первым периодом
    # (выше может стоять заголовок отчёта)
    first = min(period_rows)
    header_row = next(
        (r for r in range(first - 1, -1, -1)
         if sum(_is_texty(grid[r][c] if c < len(grid[r]) else None) for c in range(n_cols)) >= 1),
        None,
    )
    if header_row is None:
        return None
    metric_cols = {
        c: strip_unit_suffix(clean_metric_name(grid[header_row][c])).casefold()
        for c in range(n_cols)
        if c != pcol and _is_texty(grid[header_row][c] if c < len(grid[header_row]) else None)
        and not _is_service_name(clean_metric_name(grid[header_row][c]))
    }
    if not metric_cols:
        return None
    facts: list[ParsedFact] = []
    sheet_unit = detect_unit(sheet_name)
    for r in range(header_row + 1):
        for v in grid[r]:
            if v is not None:
                info = detect_unit(str(v))
                if info.multiplier > 1 or info.currency:
                    sheet_unit = info
                    break
    for r, period in period_rows.items():
        for c, name in metric_cols.items():
            raw = grid[r][c] if c < len(grid[r]) else None
            value, cell_info = _cell_value(raw, sheet_unit.multiplier)
            if value is None:
                continue
            facts.append(
                ParsedFact(
                    metric_name=name,
                    period=period,
                    value=value,
                    unit=(cell_info.unit if cell_info and cell_info.unit else sheet_unit.unit),
                    currency=(cell_info.currency if cell_info and cell_info.currency else sheet_unit.currency),
                    sheet=sheet_name,
                    cell_ref=_cell_ref(r, c),
                )
            )
    return facts or None


_YEAR_IN_TEXT_RE = re.compile(r"(?<!\d)(20[0-4]\d)(?!\d)")
_PERIOD_WORDS_RE = re.compile(r"^(?:за|по итогам)\s+(?:\d{4}\s+)?(?:год[а-я]*|г\.)\s*", re.I)


def _period_in_text(text: str) -> tuple[Period | None, str]:
    """Период внутри произвольного текста: «за январь 2026 сделки», «Сделки_1 кв. 2026»,
    «сделки 2026-02.xlsx», «за 2025 год …». Пробуем окна из 1–3 слов (самые
    длинные первыми — «1 кв. 2026» раньше, чем «2026»). -> (период, найденный текст)."""
    words = re.sub(r"[_]+", " ", text or "").replace(".xlsx", "").replace(".csv", "").split()
    for size in (3, 2, 1):
        for i in range(len(words) - size + 1):
            chunk = " ".join(words[i:i + size])
            p = parse_period(chunk)
            if p is not None:
                return p, chunk
    return None, ""


def _try_categorical_layout(
    grid: list[list[object]], n_rows: int, n_cols: int, sheet_name: str, doc_name: str | None = None
) -> tuple[list[ParsedFact], list[str], str | None] | None:
    """Макет «категория | значение» без периодов: сводка сделок по источникам,
    расходы по статьям за год. Период берётся из заголовка над таблицей
    («за 2025 год …»), имени листа или файла («сделки 2025.xlsx»), иначе —
    текущий год с предупреждением.
    Заголовок становится разделом-родителем, строки — его детьми, итог
    (если строки «Итого» нет) досчитывается — так работают «из чего состоит»,
    доли и ранжирование."""
    # пара колонок «текст | число» с максимальным числом заполненных строк
    best: tuple[int, int, list[int]] | None = None
    for tc in range(n_cols):
        for vc in range(tc + 1, min(tc + 3, n_cols)):
            rows = [
                r for r in range(n_rows)
                if _is_texty(grid[r][tc] if tc < len(grid[r]) else None)
                and _has_number(grid[r][vc] if vc < len(grid[r]) else None)
            ]
            if len(rows) >= 3 and (best is None or len(rows) > len(best[2])):
                best = (tc, vc, rows)
    if best is None:
        return None
    tc, vc, rows = best

    # заголовок — ближайшая строка с текстом над первой строкой данных
    header = ""
    for r in range(rows[0] - 1, -1, -1):
        texts = [_norm_text(v) for v in grid[r] if _is_texty(v)]
        if texts:
            header = clean_metric_name(" ".join(texts))
            break

    warnings: list[str] = []
    # период — из заголовка над таблицей, имени листа или файла: месяц, квартал
    # или год («за январь 2026 …», «Сделки_февраль_2026.xlsx», «за 2025 год …»)
    period, found = None, ""
    for text in (header, sheet_name, doc_name or ""):
        period, found = _period_in_text(text)
        if period is not None:
            break
    if period is None:
        year = date.today().year
        period = Period("year", str(year), date(year, 1, 1), date(year, 12, 31))
        warnings.append(
            f"лист «{sheet_name}»: таблица без периодов — значения отнесены к {year} году. "
            "Если это месяц или другой год — нажмите кнопку с периодом ниже, либо укажите его "
            "в заголовке («за январь 2026 …») или в имени файла («Сделки январь 2026.xlsx»)"
        )

    unit = detect_unit(header)
    if not (unit.unit or unit.currency or unit.multiplier != 1.0):
        unit = detect_unit(sheet_name)
    section = strip_unit_suffix(header)
    if found and found in section:
        section = section.replace(found, " ")
    section = _PERIOD_WORDS_RE.sub("", _YEAR_IN_TEXT_RE.sub("", section))
    section = re.sub(r"^(?:за|по итогам)\s+", "", section.strip(" ,.:-"), flags=re.I).strip(" ,.:-").casefold()
    section = section or None

    facts: list[ParsedFact] = []
    total: ParsedFact | None = None
    for r in rows:
        name = strip_unit_suffix(clean_metric_name(grid[r][tc]))
        if not name or _is_service_name(name):
            continue
        value, cell_info = _cell_value(grid[r][vc], unit.multiplier)
        if value is None:
            continue
        low = name.casefold()
        is_total = low.startswith(_TOTAL_PREFIXES)
        fact = ParsedFact(
            metric_name=section if is_total and section else low,
            period=period, value=value,
            unit=cell_info.unit if cell_info and cell_info.unit else unit.unit,
            currency=cell_info.currency if cell_info and cell_info.currency else unit.currency,
            sheet=sheet_name, cell_ref=_cell_ref(r, vc),
            section=None if is_total else section, is_total=is_total,
        )
        if is_total:
            total = fact
        facts.append(fact)
    if not facts:
        return None
    if section and total is None:
        children = [f for f in facts if not f.is_total]
        facts.append(ParsedFact(
            metric_name=section, period=period, value=sum(f.value for f in children),
            unit=children[0].unit, currency=children[0].currency, sheet=sheet_name,
            cell_ref=f"{_cell_ref(rows[0], vc)}:{_cell_ref(rows[-1], vc)}", is_total=True,
        ))
    return facts, warnings, section


def _grid_text(grid: list[list[object]], max_chars: int = 4000) -> str:
    lines = []
    for row in grid:
        cells = [_norm_text(v) for v in row if v is not None and _norm_text(v)]
        if cells:
            lines.append(" | ".join(cells))
    text = "\n".join(lines)
    return text[:max_chars]


# ------------------------------------------------------------------ вход ---

def load_excel(path: str, doc_name: str | None = None) -> ParsedDoc:
    doc = ParsedDoc()
    sheets = _xlsx_sheets(path)
    if sheets is None:  # .xls или нестандартный файл — через pandas
        try:
            frames = pd.read_excel(path, sheet_name=None, header=None, dtype=object)
        except Exception as e:
            doc.warnings.append(f"не удалось открыть Excel: {e}")
            return doc
        sheets = [(str(name), df.where(pd.notna(df), None).values.tolist()) for name, df in frames.items()]
    for name, grid in sheets:
        doc.sheets.append(name)
        facts, chunks, warns, sections = grid_to_parsed(grid, name, ledger_out=doc.ledger, doc_name=doc_name)
        doc.facts.extend(facts)
        doc.chunks.extend(chunks)
        doc.sections.extend(sections)
        doc.warnings.extend(warns)
    return doc


def load_csv(path: str, doc_name: str | None = None) -> ParsedDoc:
    doc = ParsedDoc()
    raw = Path(path).read_bytes()
    text = None
    for enc in ("utf-8-sig", "cp1251"):  # выгрузки 1С/Excel в Windows — cp1251
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        text = raw.decode("utf-8", errors="replace")
    try:
        df = pd.read_csv(io.StringIO(text), header=None, dtype=object, sep=None, engine="python")
    except Exception as e:
        doc.warnings.append(f"не удалось открыть CSV: {e}")
        return doc
    doc.sheets.append("csv")
    grid = df.where(pd.notna(df), None).values.tolist()
    doc.facts, doc.chunks, doc.warnings, doc.sections = grid_to_parsed(
        grid, "csv", ledger_out=doc.ledger, doc_name=doc_name
    )
    return doc


def load_html(path: str) -> ParsedDoc:
    """HTML/HTM (веб-отчёты, выписки из интернет-банков): все таблицы файла.
    Кодировку определяем сами (utf-8 -> cp1251): pandas угадывает её неверно
    для русских файлов. read_html использует первую строку как заголовок
    колонок — возвращаем её в грид, иначе периоды теряются."""
    doc = ParsedDoc()
    raw = Path(path).read_bytes()
    text = None
    for enc in ("utf-8", "cp1251"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        text = raw.decode("utf-8", errors="replace")
    try:
        frames = pd.read_html(io.StringIO(text))
    except Exception as e:
        doc.warnings.append(f"не удалось открыть HTML: {e} (нужен пакет lxml)")
        return doc
    doc.sheets.append("html")
    for i, df in enumerate(frames, start=1):
        named_header = not all(isinstance(c, int) for c in df.columns)
        values = df.where(pd.notna(df), None).astype(object).values.tolist()
        grid = ([list(map(str, df.columns))] + values) if named_header else values
        facts, chunks, warns, sections = grid_to_parsed(grid, f"таблица {i}", ledger_out=doc.ledger)
        doc.facts.extend(facts)
        doc.chunks.extend(chunks)
        doc.warnings.extend(warns)
        doc.sections.extend(sections)
    doc.diagnostics.append(f"HTML: таблиц {len(frames)}, фактов {len(doc.facts)}")
    return doc


def load_ods(path: str) -> ParsedDoc:
    """OpenDocument-таблицы (LibreOffice / 1С) через pandas (движок odf)."""
    doc = ParsedDoc()
    try:
        frames = pd.read_excel(path, sheet_name=None, header=None, dtype=object, engine="odf")
    except Exception as e:
        doc.warnings.append(f"не удалось открыть ODS: {e} (нужен пакет odfpy)")
        return doc
    for name, df in frames.items():
        doc.sheets.append(str(name))
        grid = df.where(pd.notna(df), None).values.tolist()
        facts, chunks, warns, sections = grid_to_parsed(grid, str(name), ledger_out=doc.ledger)
        doc.facts.extend(facts)
        doc.chunks.extend(chunks)
        doc.warnings.extend(warns)
        doc.sections.extend(sections)
    return doc
