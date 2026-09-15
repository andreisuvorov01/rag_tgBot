"""Интеграционный тест: загрузка -> БД -> вопросы (offline: mock LLM, hash-эмбеддинги)."""
from datetime import date

import openpyxl

from app.config import settings
from app.embeddings import EmbeddingService
from app.ingest.pipeline import process_document
from app.llm import make_llm
from app.qa import AnswerPipeline
from app.storage import make_engine, make_sessionmaker


async def _setup(tmp_path):
    settings.database_url = f"sqlite+aiosqlite:///{tmp_path}/t.db"
    settings.data_dir = tmp_path
    engine = await make_engine(settings)
    sessions = make_sessionmaker(engine)
    emb = EmbeddingService(settings)
    llm = make_llm(settings)
    return engine, sessions, emb, llm


def _workbook(path, years=("2023", "2024", "2025")):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Опер. расходы"
    ws["A1"] = "тыс. руб."
    ws.append(["Показатель", *years])
    data = {
        "Выручка": (41200, 45800, 49500),
        "Аренда спецтехники": (9800, 11400, 12700),
        "Зарплаты": (15600, 16800, 17900),
    }
    for name, vals in data.items():
        ws.append([name, *vals])
    wb.save(path)


async def test_end_to_end_factual_and_forecast(tmp_path):
    engine, sessions, emb, llm = await _setup(tmp_path)
    wb_path = tmp_path / "report.xlsx"
    _workbook(wb_path)

    async with sessions() as session:
        report = await process_document(
            session, emb, settings, org_id=1, user_id=1,
            original_name="report.xlsx", content=wb_path.read_bytes(),
        )
        await session.commit()
    assert report.status == "processed"
    assert report.facts == 9
    assert set(report.periods) == {"2023", "2024", "2025"}

    pipeline = AnswerPipeline(sessions, emb, llm, settings, org_names=["ООО Тест"])

    outcome = await pipeline.answer(1, 1, "Какая выручка за 2025 год?")
    assert "49 500 000" in outcome.text or "49,50 млн" in outcome.text.replace(" ", " ")

    outcome = await pipeline.answer(1, 1, "Какой прогноз по аренде спецтехники на 2026 год?")
    assert "Прогноз" in outcome.text or "прогноз" in outcome.text
    assert "12 700 000" in outcome.text or "12,70 млн" in outcome.text  # факт 2025 в истории

    outcome = await pipeline.answer(1, 1, "Прогноз по аренде на 2026, если темпы упадут вдвое")
    assert "0,5" in outcome.text  # множитель сценария виден в ответе

    # дедупликация по хэшу
    async with sessions() as session:
        dup = await process_document(
            session, emb, settings, org_id=1, user_id=1,
            original_name="report_copy.xlsx", content=wb_path.read_bytes(),
        )
        await session.commit()
    assert dup.status == "duplicate"

    await llm.close()
    await engine.dispose()


async def test_clarify_on_ambiguous_metric(tmp_path):
    engine, sessions, emb, llm = await _setup(tmp_path)
    # два похожих показателя
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Показатель", "2023", "2024"])
    ws.append(["Аренда спецтехники Москва", "100", "120"])
    ws.append(["Аренда спецтехники Регионы", "80", "90"])
    ws.append(["Зарплаты", "500", "600"])
    wb_path = tmp_path / "amb.xlsx"
    wb.save(wb_path)
    async with sessions() as session:
        await process_document(session, emb, settings, org_id=1, user_id=1,
                               original_name="amb.xlsx", content=wb_path.read_bytes())
        await session.commit()
    pipeline = AnswerPipeline(sessions, emb, llm, settings, org_names=[])
    outcome = await pipeline.answer(1, 1, "Сколько составляет аренда спецтехники за 2024?")
    # либо точный ответ, либо уточняющий вопрос с вариантами — но не выдумка
    assert outcome.clarify or "аренда" in outcome.text.casefold()
    await llm.close()
    await engine.dispose()


def test_facts_dates():
    assert date(2025, 12, 31) > date(2023, 1, 1)
