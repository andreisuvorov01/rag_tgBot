"""Ручной ввод расходов/доходов сообщениями: «расход: 1500 кофе».

Каждая запись дублируется в три хранилища (всё локально, ничего не уходит в сеть):
1. ledger_operations — построчные операции (просмотр «операции за месяц», категории);
2. facts по показателю «личные расходы» / «личные доходы» — помесячная сумма,
   поэтому работают вопросы, сравнения и прогноз — как для загруженных отчётов;
3. Excel-журнал {DATA_DIR}/расходы.xlsx — обычный файл рядом с проектом.

Формат сообщения:
  расход: 1500 кофе              -> 1500 ₽, категория по описанию, сегодня
  расход: 2 500,50 руб — такси   -> сумма с копейками
  расход: 300 обед (03.01)       -> с датой день.месяц[.год]
  доход: 50000 зарплата          -> доходы учитываются отдельно
"""
from __future__ import annotations

import asyncio
import calendar
import hashlib
import logging
import re
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path

from openpyxl import Workbook
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from .categories import categorize
from .config import Settings
from .formatting import fmt_money
from .ingest.normalize import _guess_kind, slugify_code
from .money import to_decimal
from .storage import (
    Document,
    Fact,
    LedgerOperation,
    Metric,
    audit,
    find_metric_by_name,
)

log = logging.getLogger(__name__)

# Служебный документ-контейнер: операции журнала ссылаются на него же, как
# операции выписки — на документ-выписку. uploaded_by=0 (системный): /clear
# пользователя его не удаляет — журнал принадлежит организации целиком.
JOURNAL_DOC_NAME = "Журнал расходов (сообщения)"

# «расход: 1500 кофе», «расходы - 2 500,50 руб — такси», «доход 50000 зарплата»
_ENTRY_RE = re.compile(
    r"^\s*(расходы|расход|доходы|доход)\s*[:\-–—]?\s*"
    r"(\d+(?:[ \u00a0\u202f]\d{3})*(?:[.,]\d+)?)\s*"
    r"(?:руб\.?|rub|₽)?\s*[,;—–-]?\s*(.*)$",
    re.IGNORECASE,
)
# дата в конце описания: «такси 01.03», «обед (03.01.2026)», «кофе 3.1.26»
_DATE_TAIL_RE = re.compile(r"[\(\[]?(\d{1,2})\.(\d{1,2})(?:\.(\d{2,4}))?[\)\]]?\s*$")

MONTHS_NOM = (
    "", "январь", "февраль", "март", "апрель", "май", "июнь",
    "июль", "август", "сентябрь", "октябрь", "ноябрь", "декабрь",
)

JOURNAL_HEADERS = ["Дата", "Тип", "Сумма", "Категория", "Описание"]


@dataclass
class ParsedEntry:
    kind: str  # 'expense' | 'income'
    amount: Decimal
    description: str
    when: date


def _parse_when(desc: str, today: date) -> tuple[date, str]:
    """Достаёт дату из конца описания. Нет даты — сегодня."""
    m = _DATE_TAIL_RE.search(desc or "")
    if not m:
        return today, desc.strip()
    d, mth, y = int(m.group(1)), int(m.group(2)), m.group(3)
    year = today.year
    if y:
        year = 2000 + int(y) if len(y) == 2 else int(y)
    try:
        when = date(year, mth, d)
    except ValueError:
        return today, desc.strip()  # «32.13» — не дата, оставляем как текст
    return when, desc[: m.start()].strip(" ,;-—–()[]")


def parse_entry_message(text: str, today: date | None = None) -> ParsedEntry | None:
    """«расход: 1500 кофе» -> ParsedEntry. None — сообщение не про расход/доход."""
    m = _ENTRY_RE.match((text or "").strip())
    if not m:
        return None
    kind = "income" if m.group(1).lower().startswith("доход") else "expense"
    raw = m.group(2).replace(" ", "").replace("\u00a0", "").replace("\u202f", "").replace(",", ".")
    amount = to_decimal(raw)
    if amount is None:
        return None
    amount = abs(amount)
    if not (0 < amount < Decimal(10) ** 15):
        return None
    when, description = _parse_when(m.group(3) or "", today or date.today())
    return ParsedEntry(kind=kind, amount=amount, description=description, when=when)


# ---------------------------------------------------------------------------
# База данных
# ---------------------------------------------------------------------------

async def get_journal_document(session: AsyncSession, org_id: int) -> Document | None:
    return (
        await session.scalars(
            select(Document).where(
                Document.org_id == org_id, Document.original_name == JOURNAL_DOC_NAME
            )
        )
    ).first()


async def _ensure_journal_document(session: AsyncSession, settings: Settings, org_id: int) -> Document:
    doc = await get_journal_document(session, org_id)
    if doc is not None:
        return doc
    doc = Document(
        org_id=org_id,
        original_name=JOURNAL_DOC_NAME,
        stored_path=str(settings.expense_journal_path),
        file_hash=hashlib.sha256(f"journal:{org_id}".encode()).hexdigest(),
        doc_type="ledger",
        uploaded_by=0,  # системный: не удаляется командой /clear пользователя
        status="processed",
        meta={},
    )
    session.add(doc)
    await session.flush()
    await audit(session, 0, "expense_journal_created", f"org={org_id}")
    return doc


def _metric_name(settings: Settings, kind: str) -> str:
    name = settings.income_metric_name if kind == "income" else settings.expense_metric_name
    return name.strip().casefold()


async def _ensure_metric(session: AsyncSession, org_id: int, name: str, emb=None) -> Metric:
    m = await find_metric_by_name(session, org_id, name)
    if m is not None:
        return m
    m = Metric(
        org_id=org_id,
        code=slugify_code(name),
        name=name,
        kind=_guess_kind(name),
    )
    if emb is not None:
        try:
            m.embedding = await emb.embed_query(name)
        except Exception as e:  # вектор — желателен, но не критичен
            log.warning("Эмбеддинг показателя «%s» не построен: %s", name, e)
    session.add(m)
    await session.flush()
    return m


async def _refresh_month_fact(
    session: AsyncSession, org_id: int, doc: Document, metric: Metric, period_label: str,
    period_start: date, period_end: date,
) -> float:
    """Пересчитывает факт месяца по всем операциям журнала этого показателя."""
    ops = (
        await session.scalars(
            select(LedgerOperation).where(
                LedgerOperation.org_id == org_id,
                LedgerOperation.document_id == doc.id,
                LedgerOperation.header == metric.name,
                LedgerOperation.period_label == period_label,
            )
        )
    ).all()
    total = sum(o.value for o in ops)
    await session.execute(
        delete(Fact).where(
            Fact.document_id == doc.id,
            Fact.metric_id == metric.id,
            Fact.period_start == period_start,
        )
    )
    if ops:
        session.add(
            Fact(
                org_id=org_id,
                metric_id=metric.id,
                document_id=doc.id,
                period_type="month",
                period_label=period_label,
                period_start=period_start,
                period_end=period_end,
                value=total,
                unit="руб",
                currency="RUB",
                variant="fact",
                sheet="журнал",
                cell_ref="",
            )
        )
    return total


async def add_entry(
    session: AsyncSession,
    settings: Settings,
    *,
    org_id: int,
    user_id: int,
    entry: ParsedEntry,
    emb=None,
) -> dict:
    """Пишет операцию + факт месяца. Excel обновляет вызывающий после commit."""
    name = _metric_name(settings, entry.kind)
    doc = await _ensure_journal_document(session, settings, org_id)
    metric = await _ensure_metric(session, org_id, name, emb=emb)

    y, mth = entry.when.year, entry.when.month
    last_day = calendar.monthrange(y, mth)[1]
    label = f"{mth:02d}.{y}"
    category = categorize(entry.description)

    session.add(
        LedgerOperation(
            org_id=org_id,
            document_id=doc.id,
            header=metric.name,
            period_label=label,
            period_start=date(y, mth, 1),
            period_end=date(y, mth, last_day),
            date_actual=entry.when,
            description=entry.description,
            value=entry.amount,
            currency="RUB",
            category=category,
        )
    )
    await session.flush()
    total = await _refresh_month_fact(
        session, org_id, doc, metric, label, date(y, mth, 1), date(y, mth, last_day)
    )
    ops = (
        await session.scalars(
            select(LedgerOperation).where(
                LedgerOperation.org_id == org_id,
                LedgerOperation.header == metric.name,
                LedgerOperation.period_label == label,
            )
        )
    ).all()
    await audit(session, user_id, f"{entry.kind}_entry", f"{entry.amount:g} {entry.description} [{category}]")

    word = "Расход" if entry.kind == "expense" else "Доход"
    desc = f" ({entry.description})" if entry.description else ""
    date_note = "" if entry.when == date.today() else f", дата {entry.when:%d.%m.%Y}"
    text = (
        f"✅ {word} записан: <b>{fmt_money(entry.amount)}</b> — {category}{desc}{date_note}\n"
        f"Итого «{metric.name}» за {MONTHS_NOM[mth]} {y}: "
        f"<b>{fmt_money(total)}</b> ({len(ops)} зап.)\n"
        f"Журнал: <code>{settings.expense_journal_path.name}</code> · "
        f"/ledger — выгрузить · /undo — отменить"
    )
    return {"text": text, "metric_id": metric.id, "month_total": total}


async def delete_last_entry(session: AsyncSession, org_id: int) -> dict | None:
    """Удаляет последнюю запись журнала и пересчитывает факт её месяца."""
    doc = await get_journal_document(session, org_id)
    if doc is None:
        return None
    last = (
        await session.scalars(
            select(LedgerOperation)
            .where(LedgerOperation.document_id == doc.id)
            .order_by(LedgerOperation.id.desc())
        )
    ).first()
    if last is None:
        return None
    metric = await find_metric_by_name(session, org_id, last.header)
    info = {
        "amount": last.value,
        "description": last.description,
        "category": last.category,
        "date": last.date_actual,
    }
    await session.delete(last)
    await session.flush()
    if metric is not None:
        await _refresh_month_fact(
            session, org_id, doc, metric, last.period_label, last.period_start, last.period_end
        )
    return info


# ---------------------------------------------------------------------------
# Excel-журнал (зеркало базы; /ledger всегда актуализирует файл из БД)
# ---------------------------------------------------------------------------

def _write_journal(path: Path, rows: list[tuple]) -> None:
    wb = Workbook()
    ws = wb.active
    ws.title = "Журнал"
    ws.append(JOURNAL_HEADERS)
    for r in rows:
        # openpyxl не принимает Decimal — в ячейку пишем float, точность
        # хранения в БД при этом не теряется
        ws.append([float(v) if isinstance(v, Decimal) else v for v in r])
    for col, width in zip("ABCDE", (12, 10, 14, 26, 40), strict=True):
        ws.column_dimensions[col].width = width
    for row in ws.iter_rows(min_row=2, min_col=3, max_col=3):
        row[0].number_format = "#,##0.00"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        wb.save(path)
    except PermissionError:
        raise
    wb.close()


async def sync_journal_file(session: AsyncSession, settings: Settings, org_id: int) -> int:
    """Перезаписывает Excel-журнал актуальным содержимым БД. -> число строк."""
    doc = await get_journal_document(session, org_id)
    rows: list[tuple] = []
    if doc is not None:
        ops = (
            await session.scalars(
                select(LedgerOperation)
                .where(LedgerOperation.document_id == doc.id)
                .order_by(LedgerOperation.date_actual, LedgerOperation.id)
            )
        ).all()
        income_header = settings.income_metric_name.strip().casefold()
        rows = [
            (
                o.date_actual.strftime("%d.%m.%Y"),
                "доход" if o.header == income_header else "расход",
                o.value,
                o.category,
                o.description,
            )
            for o in ops
        ]
    if rows:
        await asyncio.to_thread(_write_journal, settings.expense_journal_path, rows)
    return len(rows)


async def month_report(session: AsyncSession, settings: Settings, org_id: int, when: date | None = None) -> str:
    """Сводка за текущий месяц для ответа /ledger: суммы по категориям."""
    when = when or date.today()
    label = f"{when.month:02d}.{when.year}"
    doc = await get_journal_document(session, org_id)
    if doc is None:
        return ""
    ops = (
        await session.scalars(
            select(LedgerOperation).where(
                LedgerOperation.document_id == doc.id,
                LedgerOperation.period_label == label,
            )
        )
    ).all()
    if not ops:
        return ""
    income_header = settings.income_metric_name.strip().casefold()
    expenses = [o for o in ops if o.header != income_header]
    incomes = [o for o in ops if o.header == income_header]
    by_cat: dict[str, Decimal] = {}
    for o in expenses:
        amount = o.value if isinstance(o.value, Decimal) else (to_decimal(o.value) or Decimal(0))
        by_cat[o.category] = by_cat.get(o.category, Decimal(0)) + amount
    lines = [f"<b>Журнал за {MONTHS_NOM[when.month]} {when.year}</b>"]
    if by_cat:
        lines.append("Расходы: <b>" + fmt_money(sum(by_cat.values())) + f"</b> ({len(expenses)} зап.)")
        for cat, v in sorted(by_cat.items(), key=lambda kv: -kv[1]):
            lines.append(f"  • {cat}: {fmt_money(v)}")
    if incomes:
        total_in = sum(
            (o.value if isinstance(o.value, Decimal) else (to_decimal(o.value) or Decimal(0)))
            for o in incomes
        )
        lines.append("Доходы: <b>" + fmt_money(total_in) + f"</b> ({len(incomes)} зап.)")
    return "\n".join(lines)
