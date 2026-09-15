"""End-to-end прогон на реальном файле отчёта.

Загружает файл в чистую БД и задаёт набор вопросов по всем намерениям,
печатая ответ, маршрут графа и предупреждения загрузки.

Запуск:  python -m scripts.e2e_report <путь.xlsx> [--provider mock|hash]
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402
from app.embeddings import EmbeddingService  # noqa: E402
from app.ingest.pipeline import process_document  # noqa: E402
from app.llm import make_llm  # noqa: E402
from app.qa import AnswerPipeline  # noqa: E402
from app.storage import make_engine, make_sessionmaker  # noqa: E402

QUESTIONS = [
    "какая выручка за 2024 год?",
    "какая выручка за 2025 год?",
    "на сколько выросла выручка с 2023 по 2025?",
    "из чего состоит итого расходы?",
    "какие позиции выросли сильнее всего?",
    "что говорится о рисках?",
    "прогноз по выручке на 2026",
    "прогноз по выручке на 2026, если темпы упадут вдвое",
]


async def main() -> int:
    if len(sys.argv) < 2:
        print("Использование: python -m scripts.e2e_report <файл.xlsx> [--provider mock|hash]")
        return 1
    path = Path(sys.argv[1])
    if "--provider" in sys.argv:
        settings.embeddings_provider = sys.argv[sys.argv.index("--provider") + 1]
    # изолированная БД, чтобы не трогать рабочие данные
    db = Path("rag-tmp/e2e.db")
    db.parent.mkdir(parents=True, exist_ok=True)
    db.unlink(missing_ok=True)
    settings.database_url = f"sqlite+aiosqlite:///{db}"
    settings.send_charts = False

    engine = await make_engine(settings)
    sessions = make_sessionmaker(engine)
    emb = EmbeddingService(settings)
    llm = make_llm(settings)
    print(f"LLM: {settings.llm_provider} ({settings.llm_model}) | "
          f"эмбеддинги: {emb.provider} ({emb.model})")

    pipeline = AnswerPipeline(sessions, emb, llm, settings, org_names=["ООО Вектор"])

    print(f"\n=== загрузка {path.name} ===")
    async with sessions() as s:
        report = await process_document(
            s, emb, settings, org_id=1, user_id=1,
            original_name=path.name, content=path.read_bytes(),
        )
        await s.commit()
    print(f"фактов: {report.facts} | чанков: {report.chunks} | статус: {report.status}")
    if report.warnings:
        print(f"предупреждений: {len(report.warnings)}")
        for w in report.warnings[:8]:
            print(f"  ⚠️ {w}")
    if report.diagnostics:
        for d in report.diagnostics[:6]:
            print(f"  · {d}")

    failures = 0
    print("\n=== вопросы ===")
    for q in QUESTIONS:
        outcome = await pipeline.answer(1, 1, q)
        text = outcome.text.strip()
        print(f"\n▶ {q}")
        print("  " + text.replace("\n", "\n  ")[:900])
        if not text or text.startswith("🤷") and "данных не найдено" in text:
            failures += 1
            print("  !! пустой или безданных ответ")
        # ни одно выдуманное Число не должно проходить: ответ либо шаблон,
        # либо проверен верификатором — печатаем только для контроля
        if "Произошла ошибка" in text or "Внутренняя ошибка" in text:
            failures += 1

    print(f"\n=== итог: вопросов {len(QUESTIONS)}, проблемных {failures} ===")
    await llm.close()
    await engine.dispose()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
