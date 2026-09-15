"""Замер этапов обработки вопроса: где именно уходит время.

Запуск:  python -m scripts.profile_latency [--llm] [файл.xlsx]
Без --llm считается только поиск и аналитика (без внешней модели).
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.analytics import forecast, prepare_series  # noqa: E402
from app.config import settings  # noqa: E402
from app.embeddings import EmbeddingService  # noqa: E402
from app.ingest.pipeline import process_document  # noqa: E402
from app.llm import make_llm  # noqa: E402
from app.qa import AnswerPipeline  # noqa: E402
from app.rag import hybrid_search  # noqa: E402
from app.storage import make_engine, make_sessionmaker, series_for_metric  # noqa: E402


class Timer:
    def __init__(self) -> None:
        self.marks: list[tuple[str, float]] = []

    def mark(self, name: str, started: float) -> None:
        self.marks.append((name, time.perf_counter() - started))

    def report(self, title: str) -> None:
        total = sum(d for _, d in self.marks)
        print(f"\n--- {title}: {total:.2f} с ---")
        for name, dur in self.marks:
            print(f"  {dur:7.2f} с  {name}")


async def main() -> int:
    use_llm = "--llm" in sys.argv
    path = next((Path(a) for a in sys.argv[1:] if a.endswith(".xlsx")),
                Path("Финансовый_отчёт_ООО_Вектор_2023-2025.xlsx"))
    db = Path("rag-tmp/profile.db")
    db.unlink(missing_ok=True)
    settings.database_url = f"sqlite+aiosqlite:///{db}"
    settings.send_charts = False
    if not use_llm:
        settings.llm_provider = "mock"

    engine = await make_engine(settings)
    sessions = make_sessionmaker(engine)
    emb = EmbeddingService(settings)
    llm = make_llm(settings)
    pipeline = AnswerPipeline(sessions, emb, llm, settings, org_names=["ООО Вектор"])
    print(f"LLM={settings.llm_provider} reranker={settings.reranker} embeddings={emb.provider}")

    t = Timer()
    s0 = time.perf_counter()
    async with sessions() as s:
        await process_document(s, emb, settings, org_id=1, user_id=1,
                               original_name=path.name, content=path.read_bytes())
        await s.commit()
    print(f"загрузка+индексация: {time.perf_counter() - s0:.2f} с")

    # 1. эмбеддинг запроса
    s0 = time.perf_counter()
    for _ in range(3):
        await emb.embed_query("какая выручка за 2024 год")
    t.mark("3x embed_query (cold+warm)", s0)

    # 2. гибридный поиск
    async with sessions() as s:
        s0 = time.perf_counter()
        await hybrid_search(s, emb, 1, "что говорится о рисках", k=6, user_id=1)
        t.mark("hybrid_search (1-й, строит индекс)", s0)
        s0 = time.perf_counter()
        await hybrid_search(s, emb, 1, "что говорится о рисках", k=6, user_id=1)
        t.mark("hybrid_search (повтор, из кэша)", s0)

    # 3. ряд и прогноз (без LLM)
    async with sessions() as s:
        s0 = time.perf_counter()
        rows = await series_for_metric(s, 1, 1, user_id=1)
        t.mark(f"series_for_metric ({len(rows)} точек)", s0)
        s0 = time.perf_counter()
        forecast(prepare_series(rows), target_year=2026)
        t.mark("forecast (ансамбль+бэктест)", s0)

    # прогрев модели: первый вызов включает её в память и занимает десятки
    # секунд — без прогрева замеры конвейера несопоставимы
    if use_llm:
        s0 = time.perf_counter()
        await llm.chat([{"role": "user", "content": "привет"}], max_tokens=5)
        print(f"прогрев модели: {time.perf_counter() - s0:.2f} с (в зачёт не идёт)")

    # 4. полный ответ через конвейер
    for label, question in (
        ("вопрос «какая выручка за 2024»", "какая выручка за 2024 год?"),
        ("кнопка «Прогноз»", None),
    ):
        s0 = time.perf_counter()
        if question is not None:
            await pipeline.answer(1, 1, question)
        else:
            await pipeline.answer(1, 1, "прогноз выручка", metric_override="выручка",
                                  intent_override="forecast")
        t.mark(f"pipeline.answer: {label}", s0)

    # 5. чистые вызовы LLM
    if use_llm:
        from app.prompts import CLASSIFIER_SYSTEM, COMPOSER_SYSTEM
        s0 = time.perf_counter()
        await llm.chat([{"role": "system", "content": CLASSIFIER_SYSTEM},
                        {"role": "user", "content": "какая выручка за 2024 год?"}],
                       json_mode=True, temperature=0.0, max_tokens=300)
        t.mark("LLM: классификатор (max_tokens=300)", s0)
        s0 = time.perf_counter()
        await llm.chat([{"role": "system", "content": COMPOSER_SYSTEM},
                        {"role": "user", "content": "ВОПРОС: тест\nДАННЫЕ: {}"}],
                       max_tokens=450)
        t.mark("LLM: композитор (max_tokens=450)", s0)

    t.report("Замеры")
    await llm.close()
    await engine.dispose()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
