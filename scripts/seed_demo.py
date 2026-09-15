"""Демо-сценарий без Telegram и без внешних API (LLM_PROVIDER=mock, эмбеддинги hash).

Генерирует отчёты за 2023–2025 (с разнобоем названий показателей, как в жизни),
загружает их через настоящий конвейер и задаёт типовые вопросы.

Запуск:  python -m scripts.seed_demo
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openpyxl import Workbook

from app.config import settings
from app.embeddings import EmbeddingService
from app.ingest.pipeline import process_document
from app.llm import make_llm
from app.qa import AnswerPipeline
from app.storage import make_engine, make_sessionmaker

DATA_DIR = Path("data/demo")


def make_report(path: Path, year: int, rent_name: str) -> None:
    """Годовой отчёт: раздел «Операционные расходы» с детьми и «Итого»,
    у 2026 — ещё и колонка «2026 (план)». Значения в тыс. руб.
    Название позиции аренды различается между годами (как в реальных отчётах)."""
    data = {
        2024: {"Выручка": (37800, 41200), "Аренда склада": (2800, 3100),
               "Аренда СММ": (8700, 9800), "Зарплаты": (14100, 15600),
               "Итого": (25600, 28500)},
        2025: {"Выручка": (41200, 45800), "Аренда склада": (3100, 3300),
               "Аренда спецтехники": (9800, 11400), "Зарплаты": (15600, 16800),
               "Итого": (28500, 31500)},
        2026: {"Выручка": (45800, 49500), "Аренда склада": (3300, 3500),
               "Аренда спецтехники": (11400, 12700), "Зарплаты": (16800, 17900),
               "Итого": (31500, 34100)},
    }
    plan_2026 = {"Выручка": 48000, "Аренда склада": 3400,
                 "Аренда спецтехники": 12500, "Зарплаты": 17500, "Итого": 33400}
    with_plan = year == 2026
    wb = Workbook()
    ws = wb.active
    ws.title = "Опер. расходы"
    ws["A1"] = f"Отчет о финансовых результатах за {year} год"
    ws["A2"] = "тыс. руб."
    ws.append(["Показатель", f"{year - 1}", f"{year}"] + ([f"{year} (план)"] if with_plan else []))
    ws.append(["Выручка", *data[year]["Выручка"]] + ([plan_2026["Выручка"]] if with_plan else []))
    ws.append(["Операционные расходы"])  # раздел — группа без чисел
    for name in ("Аренда склада", rent_name if year == 2024 else "Аренда спецтехники", "Зарплаты"):
        key = name if name in data[year] else next(k for k in data[year] if k.startswith("Аренда СММ") or k == "Аренда спецтехники")
        row = [name, *data[year][key]]
        if with_plan:
            row.append(plan_2026[key] if key in plan_2026 else plan_2026.get(name, ""))
        ws.append(row)
    ws.append(["Итого", *data[year]["Итого"]] + ([plan_2026["Итого"]] if with_plan else []))
    wb.save(path)


def make_breakdown(path: Path) -> None:
    """Разбивка по направлениям: вертикальный макет (периоды в первой колонке).
    Сумма по регионам сходится с годовыми отчётами (2023: 8700, 2024: 9800, 2025: 11400)."""
    wb = Workbook()
    ws = wb.active
    ws.title = "Аренда по регионам, тыс. руб."
    ws.append(["Период", "Аренда спецтехники Москва", "Аренда спецтехники Регионы"])
    for year in (2023, 2024, 2025):
        ms = {2023: 5100, 2024: 5600, 2025: 6900}[year]
        rg = {2023: 3600, 2024: 4200, 2025: 4500}[year]
        ws.append([f"{year}", ms, rg])
    wb.save(path)


def make_notes(path: Path) -> None:
    import docx

    d = docx.Document()
    d.add_heading("Пояснительная записка к отчету за 2025 год", level=1)
    d.add_paragraph(
        "Рост расходов на аренду спецтехники в 2025 году связан с увеличением парка "
        "техники на 15% и переходом на новых двух поставщиков по региональным проектам. "
        "Доля аренды спецтехники в операционных расходах выросла с 6,1% до 7,4%."
    )
    d.add_paragraph(
        "В 2026 году планируется продолжение программы расширения регионального парка, "
        "ожидается заключение долгосрочных контрактов на аренду с фиксацией ставок. "
        "Риски: рост ставок аренды на рынке спецтехники и логистические ограничения."
    )
    d.save(path)


async def main() -> None:
    # демо-сценарий всегда offline и герметичен
    settings.embeddings_provider = "hash"
    settings.llm_provider = "mock"

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    files = [
        ("Отчет_2024.xlsx", make_report(DATA_DIR / "Отчет_2024.xlsx", 2024, "Аренда СММ")),
        ("Отчет_2025.xlsx", make_report(DATA_DIR / "Отчет_2025.xlsx", 2025, "Аренда спецтехники")),
        ("Отчет_2026.xlsx", make_report(DATA_DIR / "Отчет_2026.xlsx", 2026, "Аренда спецтехники")),
        ("Разбивка_по_регионам.xlsx", make_breakdown(DATA_DIR / "Разбивка_по_регионам.xlsx")),
        ("Пояснительная_записка_2025.docx", make_notes(DATA_DIR / "Пояснительная_записка_2025.docx")),
    ]

    settings.database_url = "sqlite+aiosqlite:///./data/demo/demo.db"
    settings.data_dir = DATA_DIR  # загрузки демо — в data/demo/uploads, не рядом с боевыми
    engine = await make_engine(settings)
    sessions = make_sessionmaker(engine)
    emb = EmbeddingService(settings)
    llm = make_llm(settings)  # mock (offline)
    pipeline = AnswerPipeline(sessions, emb, llm, settings, org_names=["ООО Демо"])

    for name, _ in files:
        path = DATA_DIR / name
        async with sessions() as session:
            report = await process_document(
                session, emb, settings,
                org_id=1, user_id=42, original_name=name,
                content=path.read_bytes(),
            )
            await session.commit()
        print(f"\n=== ЗАГРУЗКА {name} ===")
        print(report.summary())

    questions = [
        "Какая выручка за 2026 год?",
        "Какой прогноз по аренде спецтехники на 2027 год?",
        "Прогноз по аренде спецтехники на 2027, если темпы упадут вдвое",
        "Из чего состоит итого операционных расходов за 2026 год?",
        "Насколько факт 2026 отличается от плана по выручке?",
        "Какие позиции выросли сильнее всего?",
        "Почему выросла аренда спецтехники?",
    ]
    for q in questions:
        outcome = await pipeline.answer(1, 42, q)
        print(f"\n=== ВОПРОС: {q}")
        print(outcome.text)
        if outcome.clarify:
            print("[Уточнение:] " + " | ".join(outcome.clarify))

    await llm.close()
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
