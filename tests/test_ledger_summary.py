"""Тесты ledger-UX (операции за месяц с категориями), сводного отчёта
по документу и кросс-энкодер-реранкера (со стабом модели)."""
import asyncio

import openpyxl

from app.categories import categorize
from app.config import settings
from app.embeddings import EmbeddingService
from app.ingest.pipeline import process_document
from app.llm import make_llm
from app.qa import AnswerPipeline
from app.rerank import CrossEncoderReranker
from app.storage import (
    ledger_months,
    ledger_ops_for_month,
    make_engine,
    make_sessionmaker,
    series_for_metric,
)


async def _setup(tmp_path):
    settings.database_url = f"sqlite+aiosqlite:///{tmp_path}/t.db"
    settings.data_dir = tmp_path
    engine = await make_engine(settings)
    return engine, make_sessionmaker(engine), EmbeddingService(settings)


def _statement(path):
    """Выписка: колонка дат + описания + суммы (ledger-формат)."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Дата", "Назначение", "Сумма"])
    ws.append(["15.01.2026", "Оплата поставщику ООО Ромашка", "500 000"])
    ws.append(["20.01.2026", "Налог ФНС по декларации", "120 000"])
    ws.append(["02.02.2026", "Аренда офиса за февраль", "90 000"])
    ws.append(["25.02.2026", "Комиссия банка за РКО", "1 500"])
    ws.append(["28.02.2026", "Перевод на депозит", "300 000"])
    wb.save(path)


async def test_ledger_operations_stored_and_categorized(tmp_path):
    engine, sessions, emb = await _setup(tmp_path)
    llm = make_llm(settings)
    src = tmp_path / "statement.xlsx"
    _statement(src)

    async with sessions() as s:
        report = await process_document(s, emb, settings, org_id=1, user_id=1,
                                        original_name="statement.xlsx", content=src.read_bytes())
        await s.commit()
    assert report.status == "processed"
    # транзакции агрегированы в помесячные факты (2 месяца)
    assert report.facts == 2
    assert any("ledger" in d for d in report.diagnostics)

    async with sessions() as s:
        from app.storage import find_metric_by_name

        m = await find_metric_by_name(s, 1, "сумма")
        months = await ledger_months(s, 1, m.name)
        ops = await ledger_ops_for_month(s, 1, m.name, "01.2026")

    assert [(x["label"], round(x["total"], 2)) for x in months] == [("01.2026", 620000.0), ("02.2026", 391500.0)]
    assert len(ops) == 2
    assert ops[0].category == "Поставщики и подрядчики"
    assert {o.category for o in ops if o.description.startswith("Налог")} == {"Налоги и взносы"}

    # ряд для прогноза построен из помесячных агрегатов
    async with sessions() as s:
        rows = await series_for_metric(s, 1, m.id, user_id=1)
    assert len(rows) == 2 and rows[0]["value"] == 620000.0
    await llm.close()
    await engine.dispose()


async def test_ledger_ops_view_and_categories(tmp_path):
    engine, sessions, emb = await _setup(tmp_path)
    llm = make_llm(settings)
    pipeline = AnswerPipeline(sessions, emb, llm, settings, org_names=[])
    src = tmp_path / "statement.xlsx"
    _statement(src)
    async with sessions() as s:
        await process_document(s, emb, settings, org_id=1, user_id=1,
                               original_name="statement.xlsx", content=src.read_bytes())
        await s.commit()
        from app.storage import find_metric_by_name

        m = await find_metric_by_name(s, 1, "сумма")

    outcome = await pipeline.answer(1, 1, "операции за январь", metric_override="сумма",
                                    intent_override="factual")
    assert outcome.ops_months, "кнопка «Операции за месяц» должна появиться"
    assert outcome.ops_months[0][0] == "01.2026"
    assert outcome.table_metric_id == m.id

    # категории суммируются по описаниям
    async with sessions() as s:
        ops = await ledger_ops_for_month(s, 1, "сумма", "02.2026")
    cats = {o.category: o.value for o in ops}
    assert cats["Аренда"] == 90000.0 and cats["Банк"] == 1500.0
    await llm.close()
    await engine.dispose()


def test_categorize_rules():
    assert categorize("Оплата налога ФНС по декларации") == "Налоги и взносы"
    assert categorize("Зарплата сотрудников за март") == "Зарплата"
    assert categorize("Комиссия банка за РКО") == "Банк"
    assert categorize("Что-то непонятное") == "Прочее"


# ------------------------------------------------------------ сводный отчёт

async def test_document_summary(tmp_path):
    engine, sessions, emb = await _setup(tmp_path)
    llm = make_llm(settings)
    pipeline = AnswerPipeline(sessions, emb, llm, settings, org_names=[])

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Показатель", "2024", "2025"])
    ws.append(["Выручка", "100", "150"])
    ws.append(["Запасы", "80", "60"])
    src = tmp_path / "rep.xlsx"
    wb.save(src)
    async with sessions() as s:
        r1 = await process_document(s, emb, settings, org_id=1, user_id=1,
                                    original_name="rep.xlsx", content=src.read_bytes())
        await s.commit()

    async with sessions() as s:
        text = await __import__("app.summarize", fromlist=["document_summary"]).document_summary(
            pipeline, s, 1, r1.document_id, 1)

    assert "Обзор документа" in text
    assert "выручка" in text and "запасы" in text
    assert "2024" in text and "2025" in text
    await llm.close()
    await engine.dispose()


# ------------------------------------------------------------ кросс-энкодер

def test_crossencoder_with_stubbed_model(monkeypatch):
    """Сортировка по оценкам стаба — без скачивания настоящей модели."""
    r = CrossEncoderReranker("stub-model")
    monkeypatch.setattr(r, "_load", lambda: type("M", (), {
        "predict": staticmethod(lambda pairs: [0.2, 0.95]),
    })())
    chunks = [
        {"id": 1, "body": "слабый фрагмент"},
        {"id": 2, "body": "сильный фрагмент"},
    ]
    out = asyncio.run(r.rerank("запрос", chunks, k=2))
    assert [c["id"] for c in out] == [2, 1]
    assert out[0]["rerank_score"] == 0.95
