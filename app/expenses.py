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
    add_synonym,
    audit,
    find_metric_by_name,
)

log = logging.getLogger(__name__)

# Служебный документ-контейнер: операции журнала ссылаются на него же, как
# операции выписки — на документ-выписку. uploaded_by=0 (системный): /clear
# пользователя его не удаляет — журнал принадлежит организации целиком.
JOURNAL_DOC_NAME = "Журнал расходов (сообщения)"

# «расход: 1500 кофе», «расходы - 2 500,50 руб — такси», «доход 50000 зарплата».
# Захватываем хвост целиком (описание ИЛИ сумму), потому что порядок бывает
# любой: «расход: кофе 1500» — тоже естественная формулировка.
_ENTRY_RE = re.compile(
    r"^\s*(?:(?:я|мы|сегодня|вчера|только что|опять|снова)\s+)*"
    r"(расходы|расход|доходы|доход|"
    # живая речь владельца: «потратил 3000 на подарок», «взял 5000 с карты компании»
    r"потратил[аи]?|купил[аи]?|заплатил[аи]?|оплатил[аи]?|взял[аи]?|снял[аи]?|перев[её]л[аи]?|"
    r"отдал[аи]?|ушло|получил[аи]?|пришло|заработал[аи]?)"
    r"\s*[:\-–—]?\s*(.*)$",
    re.IGNORECASE | re.DOTALL,
)
_INCOME_WORDS = ("доход", "получил", "пришло", "заработал")
# «1500 кофе» без префикса: сумма и короткое описание, без вопроса
_BARE_ENTRY_RE = re.compile(
    # «1500 кофе» и «грузоперевозка 60к»: сумма с одной стороны, описание без цифр —
    # с другой («2 гис 2025» — вопрос про показатель, а не 2 ₽ за «гис 2025»)
    r"^\s*(?:(\d[\d \u00a0\u202f.,]*)\s*(?:к|k|тыс\w*|млн\w*)?\s*(?:руб\w*|rub|₽|р\.?)?\s+[^?\d]{1,60}"
    r"|[^?\d]{1,60}?\s+(\d[\d \u00a0\u202f.,]*)\s*(?:к|k|тыс\w*|млн\w*)?\s*(?:руб\w*|rub|₽|р\.?)?)\s*$",
    re.IGNORECASE,
)
_QUESTION_WORDS_RE = re.compile(
    r"сколько|какой|какая|какие|каков|что|почему|прогноз|сравни|покажи|динамик|состав|доля", re.IGNORECASE
)
# синонимы показателей журнала — чтобы «сколько я потратил в августе» находило «личные расходы»
# без «мои расходы»/«из компании»: после отбрасывания стоп-слов они превращаются в
# «расходы»/«компании» и перехватывали бы вопросы о расходах и выручке компании
EXPENSE_SYNONYMS = ("личные траты", "траты", "потратил", "взял из компании")
INCOME_SYNONYMS = ("мои доходы", "личный доход")
# Число: «1500», «2 500,50», «1.500,00» (европейский формат из 1С), «1,234.56».
# Общая часть для обоих шаблонов ниже, чтобы форматы не расходились.
_NUMBER = (
    r"\d{1,3}(?:[ \u00a0\u202f]\d{3})+(?:[.,]\d+)?"
    r"|\d{1,3}(?:[.]\d{3})+(?:,\d+)?"
    r"|\d+(?:[.,]\d+)?"
)
# сумма в начале («1500 кофе», «2 500,50 руб — такси»); после числа допускаем
# запятую, скобку или пробел — но не букву, иначе «кофе 1500» даст сумму из ничего
# «60к», «60k», «12 тыс», «1,5 млн» — множитель после числа
_MULT = r"(?:\s*(к|k|тыс\w*|млн\w*))?"
_AMOUNT_RE = re.compile(
    rf"^({_NUMBER}){_MULT}\s*(?:руб\w*|rub|₽|р\.?)?(?:[(\[,;]|\s|$)",
    re.IGNORECASE,
)
# сумма в конце описания: «кофе 1500», «такси — 700,50 руб», «грузоперевозка 60к»
_AMOUNT_TAIL_RE = re.compile(
    rf"(?:^|[\s(—-])({_NUMBER}){_MULT}\s*(?:руб\w*|rub|₽|р\.?)?[\s)\]]*$",
    re.IGNORECASE,
)
_MULTIPLIERS = {"к": 1000, "k": 1000, "тыс": 1000, "млн": 1_000_000}
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
    """«расход: 1500 кофе» и «расход: кофе 1500» -> ParsedEntry.

    Порядок «сумма ↔ описание» в живых сообщениях произвольный, поэтому сумму
    ищем сначала в начале хвоста (и только если она там есть и не относится к
    дате), затем в конце. Описание из одного числа («расход: 1500») даёт запись
    без описания — это тоже валидный ввод.
    """
    text = (text or "").strip()
    m = _ENTRY_RE.match(text)
    if m:
        kind = "income" if m.group(1).lower().startswith(_INCOME_WORDS) else "expense"
        rest = (m.group(2) or "").strip()
    else:
        _when, body = _parse_when(text, today or date.today())  # «такси 700 (05.09)» — дата не часть описания
        bare = _BARE_ENTRY_RE.match(body)
        if bare is None or _QUESTION_WORDS_RE.search(text):
            return None
        head = (bare.group(1) or bare.group(2) or "").strip()
        if re.fullmatch(r"(19|20)\d\d", head):  # «2024 выручка» — это вопрос про год, а не 2024 ₽
            return None
        kind, rest = "expense", text
    if not rest:
        return None

    def _amount(raw: str, mult: str | None = None) -> Decimal | None:
        # разделитель тысяч — пробел, апостроф, неразрывный пробел; «1.500,00»
        # приходит из 1С и LibreOffice, поэтому поддерживаем оба порядка
        cleaned = raw.replace(" ", "").replace("\u00a0", "").replace("\u202f", "")
        cleaned = cleaned.replace("'", "").replace("\u2019", "")
        if "," in cleaned and "." in cleaned:
            cleaned = cleaned.replace(".", "").replace(",", ".")
        elif re.match(r"^\d{1,3}(?:\.\d{3})+$", cleaned):
            cleaned = cleaned.replace(".", "")          # 1.500 = 1500
        else:
            cleaned = cleaned.replace(",", ".")
        value = to_decimal(cleaned)
        if value is None:
            return None
        value = abs(value)
        if mult:
            value *= _MULTIPLIERS.get(mult.casefold()[:3].rstrip("."), 1)
        return value if 0 < value < Decimal(10) ** 15 else None

    # Дату отделяем ДО поиска суммы: «кофе 1500 (03.01.2026)» иначе не находился
    # хвост суммы — мешала скобка, а не число.
    when, rest = _parse_when(rest, today or date.today())

    amount: Decimal | None = None
    description = rest

    head = _AMOUNT_RE.match(rest)
    if head is not None:
        amount = _amount(head.group(1), head.group(2))
        if amount is not None:
            description = rest[head.end():].strip(" ,;—–-()[]")

    if amount is None:
        tail = _AMOUNT_TAIL_RE.search(rest)
        if tail is not None:
            # «расход: 1500» — описание из одного числа: это сумма, а не описание
            amount = _amount(tail.group(1), tail.group(2))
            if amount is not None:
                description = rest[: tail.start()].strip(" ,;—–-()[]")

    if amount is None:
        return None
    # «потратил 3000 на подарок» -> «подарок», «купил кофе за 1500» -> «кофе»;
    # «взял 5000 с карты» — предлог посередине остаётся в описании
    description = re.sub(r"^(?:на|за)\s+", "", description, flags=re.IGNORECASE)
    description = re.sub(r"\s+(?:на|за|в|по|с|со|у)$", "", description, flags=re.IGNORECASE)
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


async def _ensure_metric(
    session: AsyncSession, org_id: int, name: str, emb=None, synonyms: tuple[str, ...] = ()
) -> Metric:
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
    for syn in synonyms:
        await add_synonym(session, m.id, syn)
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
    metric = await _ensure_metric(
        session, org_id, name, emb=emb,
        synonyms=INCOME_SYNONYMS if entry.kind == "income" else EXPENSE_SYNONYMS,
    )

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


async def ledger_rows(session: AsyncSession, settings: Settings, org_id: int) -> list[dict]:
    """Операции журнала для отчёта: дата, вид, сумма, категория, описание.

    Отдаём в том же порядке, что и Excel-журнал. Суммы — Decimal: отчёт по
    расходам должен сходиться копейка в копейку, а не «примерно».
    """
    doc = await get_journal_document(session, org_id)
    if doc is None:
        return []
    ops = (
        await session.scalars(
            select(LedgerOperation)
            .where(LedgerOperation.document_id == doc.id)
            .order_by(LedgerOperation.date_actual, LedgerOperation.id)
        )
    ).all()
    income_header = settings.income_metric_name.strip().casefold()
    return [
        {
            "when": o.date_actual,
            "kind": "income" if o.header == income_header else "expense",
            "amount": o.value,
            "category": o.category or "Прочее",
            "description": o.description or "",
        }
        for o in ops
    ]


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
