"""Golden-набор вопросов для регрессионной оценки системы (практика FinanceBench).

Прогоняет фиксированные вопросы по демо-данным и проверяет, что в ответах
присутствуют ожидаемые значения. Работает полностью offline (mock LLM + hash
эмбеддинги) и пригоден как smoke-тест после изменений конвейера.

Запуск:  python -m scripts.eval_demo   (код возврата 1 при провале)
"""
from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402
from app.embeddings import EmbeddingService  # noqa: E402
from app.ingest.pipeline import process_document  # noqa: E402
from app.llm import make_llm  # noqa: E402
from app.qa import AnswerPipeline  # noqa: E402
from app.storage import make_engine, make_sessionmaker  # noqa: E402
from scripts.seed_demo import (  # noqa: E402
    DATA_DIR,
    make_breakdown,
    make_notes,
    make_report,
)

# (вопрос, [подстроки, обязательные к появлению в ответе])
GOLDEN: list[tuple[str, list[str]]] = [
    ("Какая выручка за 2026 год?", ["49,50 млн"]),
    ("Какой прогноз по аренде спецтехники на 2027 год?", ["13,9", "интервал"]),
    ("Прогноз по аренде спецтехники на 2027, если темпы упадут вдвое", ["0,50", "13,3"]),
    ("Из чего состоит итого операционных расходов за 2026 год?", ["зарплаты", "аренда склада", "аренда спецтехники"]),
    ("Насколько факт 2026 отличается от плана по выручке?", ["+3,1%", "48,00 млн"]),
    ("Какие позиции выросли сильнее всего?", ["москва"]),
    ("Почему выросла аренда спецтехники?", ["парка техники"]),
]


async def main() -> int:
    # golden-прогон всегда offline и герметичен
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
    # свежая БД и папка загрузок на каждый прогон: иначе каждый запуск
    # добавлял копии демо-файлов в data/uploads
    workdir = Path(tempfile.mkdtemp(prefix="rag-eval-"))
    settings.database_url = f"sqlite+aiosqlite:///{workdir.as_posix()}/eval.db"
    settings.data_dir = workdir
    engine = await make_engine(settings)
    sessions = make_sessionmaker(engine)
    emb = EmbeddingService(settings)
    llm = make_llm(settings)
    pipeline = AnswerPipeline(sessions, emb, llm, settings, org_names=["ООО Демо"])

    for name, _ in files:
        path = DATA_DIR / name
        async with sessions() as session:
            await process_document(session, emb, settings, org_id=1, user_id=42,
                                   original_name=name, content=path.read_bytes())
            await session.commit()

    failed = 0
    for question, expected in GOLDEN:
        outcome = await pipeline.answer(1, 42, question)
        missing = [e for e in expected if e not in outcome.text]
        status = "PASS" if not missing else "FAIL"
        if missing:
            failed += 1
        print(f"[{status}] {question}")
        if missing:
            print(f"    нет подстрок: {missing}")
            print(f"    ответ: {outcome.text[:400]}")

    print(f"\nИтог: {len(GOLDEN) - failed}/{len(GOLDEN)} вопросов прошли golden-проверку")
    await llm.close()
    await engine.dispose()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
