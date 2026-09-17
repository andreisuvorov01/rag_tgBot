"""Тесты ручного ввода расходов: парсинг сообщений, запись в БД, Excel-журнал."""
from datetime import date

import openpyxl

from app.config import settings
from app.embeddings import EmbeddingService
from app.expenses import (
    JOURNAL_DOC_NAME,
    add_entry,
    delete_last_entry,
    month_report,
    parse_entry_message,
    sync_journal_file,
)
from app.storage import (
    Fact,
    LedgerOperation,
    make_engine,
    make_sessionmaker,
    series_for_metric,
)


async def _setup(tmp_path):
    settings.database_url = f"sqlite+aiosqlite:///{tmp_path}/t.db"
    settings.data_dir = tmp_path
    engine = await make_engine(settings)
    sessions = make_sessionmaker(engine)
    emb = EmbeddingService(settings)
    return engine, sessions, emb


# ---------------------------------------------------------------- парсер
def test_parse_simple_expense():
    e = parse_entry_message("расход: 1500 кофе", today=date(2026, 3, 15))
    assert e is not None
    assert e.kind == "expense"
    assert e.amount == 1500
    assert e.description == "кофе"
    assert e.when == date(2026, 3, 15)


def test_parse_variants():
    # без двоеточия, с тире
    e = parse_entry_message("расход - 2 500,50 руб — такси", today=date(2026, 1, 10))
    assert e is not None and e.amount == 2500.50
    assert e.description == "такси"
    # доход
    e2 = parse_entry_message("доход: 50000 зарплата")
    assert e2 is not None and e2.kind == "income" and e2.description == "зарплата"
    # расходы (множественное)
    e3 = parse_entry_message("Расходы: 300 обед")
    assert e3 is not None and e3.kind == "expense"
    # не расход — None
    assert parse_entry_message("какая выручка за 2024?") is None
    assert parse_entry_message("расходы выросли на 15%") is None
    assert parse_entry_message("прогноз расходов на 2026: 120000") is None


def test_parse_with_date():
    e = parse_entry_message("расход: 300 обед (03.01)", today=date(2026, 3, 15))
    assert e is not None
    assert e.when == date(2026, 1, 3)
    assert e.description == "обед"
    # полный год
    e2 = parse_entry_message("расход: 100 проезд 05.02.2025")
    assert e2 is not None and e2.when == date(2025, 2, 5)
    # «32.13» — не дата, остаётся описанием, запись — сегодня
    e3 = parse_entry_message("расход: 100 доставка 32.13", today=date(2026, 6, 1))
    assert e3 is not None and e3.when == date(2026, 6, 1)


def test_parse_categories():
    assert parse_entry_message("расход: 1500 кофе").description
    from app.categories import categorize

    assert categorize("кофе с коллегами") == "Кафе и рестораны"
    assert categorize("такси в аэропорт") == "Транспорт"
    assert categorize("продукты пятёрочка") == "Продукты"
    assert categorize("оплата налога УСН") == "Налоги и взносы"
    assert categorize("зарплата за март") == "Зарплата"
    assert categorize("что-то непонятное") == "Прочее"


# ---------------------------------------------------------------- запись
async def test_add_entry_and_excel(tmp_path):
    engine, sessions, emb = await _setup(tmp_path)
    async with sessions() as s:
        r1 = await add_entry(s, settings, org_id=1, user_id=42,
                             entry=parse_entry_message("расход: 1500 кофе", date(2026, 3, 15)))
        r2 = await add_entry(s, settings, org_id=1, user_id=42,
                             entry=parse_entry_message("расход: 2 500,50 такси", date(2026, 3, 16)))
        r3 = await add_entry(s, settings, org_id=1, user_id=42,
                             entry=parse_entry_message("доход: 50000 зарплата", date(2026, 3, 5)))
        n = await sync_journal_file(s, settings, 1)
        await s.commit()

    assert r1["month_total"] == 1500
    assert r2["month_total"] == 4000.50  # накопилось за месяц
    assert r3["month_total"] == 50000    # доходы — отдельный показатель
    assert n == 3

    # Excel-файл создан и заполнен (строки отсортированы по дате)
    path = settings.expense_journal_path
    assert path.exists()
    wb = openpyxl.load_workbook(path)
    ws = wb.active
    rows = list(ws.values)
    assert rows[0] == ("Дата", "Тип", "Сумма", "Категория", "Описание")
    assert len(rows) == 4
    by_desc = {r[4]: r for r in rows[1:]}
    assert by_desc["кофе"][1] == "расход" and by_desc["кофе"][2] == 1500
    assert by_desc["такси"][2] == 2500.50 and by_desc["такси"][3] == "Транспорт"
    assert by_desc["зарплата"][1] == "доход" and by_desc["зарплата"][2] == 50000

    # факт месяца по показателю: вопросы/прогноз работают
    async with sessions() as s:
        rows_series = await series_for_metric(s, 1, r1["metric_id"])
    assert len(rows_series) == 1
    assert rows_series[0]["period_label"] == "03.2026"
    assert abs(rows_series[0]["value"] - 4000.50) < 1e-6

    # сводка месяца
    async with sessions() as s:
        text = await month_report(s, settings, 1, when=date(2026, 3, 1))
    assert "Журнал за март 2026" in text
    assert "Кафе и рестораны" in text and "Транспорт" in text

    await engine.dispose()


async def test_undo_last_entry(tmp_path):
    engine, sessions, emb = await _setup(tmp_path)
    async with sessions() as s:
        await add_entry(s, settings, org_id=1, user_id=1,
                        entry=parse_entry_message("расход: 1000 книги", date(2026, 2, 10)))
        await add_entry(s, settings, org_id=1, user_id=1,
                        entry=parse_entry_message("расход: 200 метро", date(2026, 2, 11)))
        info = await delete_last_entry(s, 1)
        await sync_journal_file(s, settings, 1)
        await s.commit()

    assert info is not None and info["amount"] == 200
    async with sessions() as s:
        ops = (await s.scalars(
            __import__("sqlalchemy").select(LedgerOperation)
        )).all()
        facts = (await s.scalars(
            __import__("sqlalchemy").select(Fact)
        )).all()
    assert len(ops) == 1 and ops[0].value == 1000
    assert len(facts) == 1 and facts[0].value == 1000  # факт пересчитан

    # undo на пустом журнале
    async with sessions() as s:
        await delete_last_entry(s, 1)  # удаляет единственную
        empty = await delete_last_entry(s, 1)
        await s.commit()
    assert empty is None
    await engine.dispose()


async def test_journal_survives_clear(tmp_path):
    """Журнал привязан к системному документу (uploaded_by=0): /clear пользователя
    его не удаляет — Excel-журнал и факты сохраняются."""
    from sqlalchemy import select

    from app.storage import delete_user_documents

    engine, sessions, emb = await _setup(tmp_path)
    async with sessions() as s:
        await add_entry(s, settings, org_id=1, user_id=42,
                        entry=parse_entry_message("расход: 700 суши", date(2026, 4, 1)))
        await s.commit()
    async with sessions() as s:
        await delete_user_documents(s, 1, 42)  # /clear пользователя 42
        await s.commit()
    async with sessions() as s:
        ops = (await s.scalars(select(LedgerOperation))).all()
        docs = (await s.scalars(select(__import__("app.storage", fromlist=["Document"]).Document))).all()
    assert len(ops) == 1
    assert len(docs) == 1 and docs[0].original_name == JOURNAL_DOC_NAME
    await engine.dispose()


# ------------------------------------------------------------ свободная форма и связь с отчётами

def test_parse_entry_free_form():
    from datetime import date

    from app.expenses import parse_entry_message

    e = parse_entry_message("потратил 3000 на подарок", today=date(2026, 9, 17))
    assert e and e.kind == "expense" and e.amount == 3000 and e.description == "подарок"
    e = parse_entry_message("1500 кофе")
    assert e and e.amount == 1500 and e.description == "кофе"
    e = parse_entry_message("взял 5000 с карты компании")
    assert e and e.amount == 5000 and e.description == "с карты компании"
    assert parse_entry_message("2024 выручка") is None          # год, а не сумма
    assert parse_entry_message("сколько 1500 кофе?") is None    # вопрос
    assert parse_entry_message("потратил в августе?") is None   # без суммы — вопрос


async def test_journal_with_company_report(tmp_path):
    """Личные расходы рядом с отчётом компании: месяцы, итог, состав по
    категориям, свёртка месяцев в год для сравнения с годовым показателем."""
    from datetime import date
    from pathlib import Path

    from app.expenses import add_entry, parse_entry_message
    from app.ingest.pipeline import process_document
    from app.llm import make_llm
    from app.qa import AnswerPipeline

    settings.database_url = f"sqlite+aiosqlite:///{tmp_path / 'j.db'}"
    settings.expense_journal_file = str(tmp_path / "расходы.xlsx")
    settings.llm_provider, settings.embeddings_provider, settings.send_charts = "mock", "hash", False
    settings.classify_with_llm = False
    engine = await make_engine(settings)
    sessions = make_sessionmaker(engine)
    emb = EmbeddingService(settings)
    report = Path(__file__).resolve().parents[1] / "Финансовый_отчёт_ООО_Вектор_2023-2025.xlsx"
    async with sessions() as s:
        await process_document(s, emb, settings, org_id=1, user_id=1,
                               original_name=report.name, content=report.read_bytes())
        for msg in ("расход: 40000 ноутбук (10.11.2025)", "расход: 8000 ужин (03.12.2025)",
                    "потратил 700 на такси (05.12.2025)"):
            await add_entry(s, settings, org_id=1, user_id=1, entry=parse_entry_message(msg, today=date(2025, 12, 31)), emb=emb)
        await s.commit()
    pipe = AnswerPipeline(sessions, emb, make_llm(settings), settings)

    out = await pipe.answer(1, 1, "сколько я потратил в декабре?")
    assert "12.2025" in out.text and "8,70 тыс." in out.text and "11.2025" not in out.text

    out = await pipe.answer(1, 1, "сколько всего личных расходов за 2025?")
    assert "Итого за 2 мес.: <b>48,70 тыс. ₽</b>" in out.text

    out = await pipe.answer(1, 1, "из чего состоят личные расходы за 2025?")
    assert "Кафе и рестораны" in out.text and "Транспорт" in out.text and "Техника" in out.text

    out = await pipe.answer(1, 1, "какая доля личных расходов в расходах компании?")
    assert "Доля «личные расходы» в «итого расходы» за 2025 (2 мес.)" in out.text
    assert "Доля: <b>0,0%</b>" in out.text  # 48,7 тыс. против сотен миллионов

    out = await pipe.answer(1, 1, "на сколько выросли личные расходы с ноября по декабрь?")
    assert "11.2025" in out.text and "12.2025" in out.text and "-78,2%" in out.text

    out = await pipe.answer(1, 1, "из чего состоят личные расходы?")
    assert "за 12.2025" in out.text  # последний месяц по дате
    await engine.dispose()
