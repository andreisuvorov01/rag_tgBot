"""Сквозной прогон системы на РЕАЛЬНОМ LLM API (не mock).

Отличается от `scripts/eval_demo.py` (тот offline и герметичен): здесь тот же
golden-набор вопросов идёт через живой API, поэтому проверяется не только
конвейер, но и связка «провайдер → клиент → верификатор → ответ», а также
фактический расход токенов и денег.

Запуск:
    python -m scripts.e2e_live                 # демо-данные, реальный API
    python -m scripts.e2e_live --questions 3   # только первые N вопросов
    python -m scripts.e2e_live --report <файл> # свой документ вместо демо

Требует ключ в .env (LLM_API_KEY). Расход минимальный: демо-отчёты маленькие,
вопросы — из golden-набора. Итог печатается с ценой по LLM_PRICES.
"""
from __future__ import annotations

import argparse
import asyncio
import re
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    _reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(_reconfigure):
        _reconfigure(encoding="utf-8")

from app.config import settings  # noqa: E402
from app.embeddings import EmbeddingService  # noqa: E402
from app.ingest.pipeline import process_document  # noqa: E402
from app.llm import make_llm  # noqa: E402
from app.qa import AnswerPipeline  # noqa: E402
from app.storage import make_engine, make_sessionmaker  # noqa: E402
from scripts.seed_demo import DATA_DIR, make_breakdown, make_notes, make_report  # noqa: E402


def _normalize(text: str) -> str:
    """Текст для сравнения: убираем разметку, пробелы-разделители тысяч и
    приводим десятичный разделитель к точке.

    Нужно потому, что шаблон печатает «49,50 млн ₽», а модель — «49 500 000 ₽»:
    это одно и то же число, и проверять надо данные, а не формат изложения.
    """
    out = text.replace("<b>", "").replace("</b>", "").replace("<i>", "").replace("</i>", "")
    out = out.replace("<pre>", "").replace("</pre>", "").replace("\u00a0", " ")
    out = out.replace(",", ".")
    # «49 500 000» -> «49500000» (пробел как разделитель тысяч)
    out = re.sub(r"(?<=\d)\s+(?=\d{3}\b)", "", out)
    out = re.sub(r"(?<=\d)\s+(?=\d)", "", out)
    return out


def _check(answer: str, expected: list[str]) -> list[str]:
    """Непроверенные альтернативы: элемент вида "a|b" считается найденным,
    если найдена любая из частей. Регистр не важен: модель пишет «Интервал»,
    шаблон — «интервал»."""
    norm = _normalize(answer).casefold()
    missing = []
    for item in expected:
        if not any(_normalize(alt).casefold() in norm for alt in item.split("|")):
            missing.append(item)
    return missing


def _looks_truncated(text: str) -> bool:
    """Ответ оборван лимитом токенов.

    Признаки: незакрытая разметка или обрыв на запятой/союзе. По последнему
    символу судить нельзя — модель может закончить перечислением, и это норма.
    """
    stripped = text.strip()
    if not stripped:
        return True
    for tag in ("b", "i", "pre", "code"):
        if stripped.count(f"<{tag}>") != stripped.count(f"</{tag}>"):
            return True
    return bool(re.search(r"[,;:\-—]\s*(</\w+>)?$", stripped))


# «49 500 000 ₽». Это одно и то же число, поэтому альтернативы перечислены через
# «|», а числовое сравнение нормализует пробелы и разделители (_check).
LIVE_GOLDEN: list[tuple[str, list[str]]] = [
    ("Какая выручка за 2026 год?", ["49,50 млн|49 500 000"]),
    ("Какой прогноз по аренде спецтехники на 2027 год?", ["13,9|13 944|13 943", "интервал|диапазон"]),
    ("Прогноз по аренде спецтехники на 2027, если темпы упадут вдвое", ["0,50|0.5", "13,3|13 3"]),
    ("Из чего состоит итого операционных расходов за 2026 год?",
     ["зарплаты", "аренда склада", "аренда спецтехники"]),
    ("Насколько факт 2026 отличается от плана по выручке?", ["+3,1|3,13|3.13", "48,00 млн|48 000 000"]),
    ("Какие позиции выросли сильнее всего?", ["москва|спецтехник"]),
    ("Почему выросла аренда спецтехники?", ["парка техники|парк техники|техник"]),
]


def _prepare_demo_files() -> list[tuple[str, Path]]:
    """Демо-отчёты генерируются локально (как в seed_demo/eval_demo)."""
    # генераторы пишут файл на диск и ничего не возвращают (см. seed_demo)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    make_report(DATA_DIR / "Отчет_2024.xlsx", 2024, "Аренда СММ")
    make_report(DATA_DIR / "Отчет_2025.xlsx", 2025, "Аренда спецтехники")
    make_report(DATA_DIR / "Отчет_2026.xlsx", 2026, "Аренда спецтехники")
    make_breakdown(DATA_DIR / "Разбивка_по_регионам.xlsx")
    make_notes(DATA_DIR / "Пояснительная_записка_2025.docx")
    names = [
        "Отчет_2024.xlsx",
        "Отчет_2025.xlsx",
        "Отчет_2026.xlsx",
        "Разбивка_по_регионам.xlsx",
        "Пояснительная_записка_2025.docx",
    ]
    return [(name, DATA_DIR / name) for name in names]


async def main() -> int:
    ap = argparse.ArgumentParser(description="Сквозной прогон на реальном LLM API")
    ap.add_argument("--questions", type=int, default=0, help="сколько вопросов задать (0 = все)")
    ap.add_argument("--report", help="свой файл отчёта вместо демо-данных")
    ap.add_argument("--org", default="ООО Демо", help="название организации для маскирования")
    args = ap.parse_args()

    if settings.llm_provider == "mock":
        print("LLM_PROVIDER=mock — это offline-режим, живой API не проверяется.\n"
              "Впишите ключ и LLM_PROVIDER=openai_compatible в .env.")
        return 2
    if not (settings.llm_api_key or "").strip():
        print("LLM_API_KEY не задан в .env — проверять нечего.")
        return 2

    workdir = Path(tempfile.mkdtemp(prefix="rag-live-"))
    settings.database_url = f"sqlite+aiosqlite:///{workdir.as_posix()}/live.db"
    settings.data_dir = workdir
    settings.llm_usage_log = str(workdir / "usage.jsonl")

    if args.report:
        src = Path(args.report)
        if not src.exists():
            print(f"Файл не найден: {src}")
            return 2
        files = [(src.name, src)]
    else:
        files = _prepare_demo_files()

    engine = await make_engine(settings)
    sessions = make_sessionmaker(engine)
    emb = EmbeddingService(settings)
    llm = make_llm(settings)
    pipeline = AnswerPipeline(sessions, emb, llm, settings, org_names=[args.org])

    print(f"=== Сквозной прогон: {settings.llm_model} @ {settings.llm_api_base} ===")
    print(f"рассуждения: {settings.llm_thinking} | ответ: compose_mode={settings.compose_mode}, "
          f"max_tokens={settings.composer_max_tokens}")
    print(f"эмбеддинги: {emb.provider} ({settings.effective_embeddings_model}) | БД: {workdir}\n")

    # --- загрузка документов (парсинг локальный, LLM не участвует) ---
    t0 = time.perf_counter()
    for name, path in files:
        async with sessions() as session:
            report = await process_document(
                session, emb, settings, org_id=1, user_id=42,
                original_name=name, content=path.read_bytes(),
            )
            await session.commit()
        facts = getattr(report, "facts_count", None)
        print(f"[ЗАГРУЗКА] {name}: фактов {facts if facts is not None else '—'}")
    print(f"загрузка заняла {time.perf_counter() - t0:.1f} с (0 токенов)\n")

    # --- вопросы ---
    questions = LIVE_GOLDEN[: args.questions] if args.questions else LIVE_GOLDEN
    failed = 0
    truncated = 0
    first_explain: tuple[str, str] | None = None
    print("=== Вопросы ===")
    for question, expected in questions:
        started = time.perf_counter()
        try:
            outcome = await pipeline.answer(1, 42, question)
        except Exception as e:
            print(f"[ОШИБКА] {question}\n    {type(e).__name__}: {e}")
            failed += 1
            continue
        elapsed = time.perf_counter() - started
        missing = _check(outcome.text, expected)
        cut = _looks_truncated(outcome.text)
        if cut:
            truncated += 1
        status = "OK  " if not missing and not cut else "FAIL"
        if missing or cut:
            failed += 1
        print(f"[{status}] {question}  ({elapsed:.1f} с, {len(outcome.text)} симв.)")
        if outcome.balance_notice:
            print("    ⚠ предупреждение о балансе API")
        if cut:
            print(f"    ⚠ ответ оборван лимитом токенов (COMPOSER_MAX_TOKENS="
                  f"{settings.composer_max_tokens}); хвост: {outcome.text[-60:]!r}")
        if missing:
            print(f"    нет значений: {missing}")
            print(f"    ответ: {outcome.text[:600]}")
        if outcome.payload_type == "explain" and first_explain is None:
            first_explain = (question, outcome.text)

    # --- расход ---
    from app.usage import format_snapshot, usage_snapshot

    async with sessions() as s:
        snap = await usage_snapshot(s, settings)
    print("\n=== Расход ===")
    print(format_snapshot(snap).replace("<b>", "").replace("</b>", "").replace("<i>", "").replace("</i>", ""))
    answered = len(questions) - failed
    questions_with_model = snap["tasks"].get("compose", {}).get("calls", 0) or len(questions)
    if snap.get("cost") is not None and questions:
        print(f"\nЦена вопроса (по факту, на {len(questions)} вопроса): "
              f"{snap['cost'] / max(1, len(questions)):.5f}")
        print(f"Цена одного ответа модели: {snap['cost'] / max(1, questions_with_model):.5f} "
              f"(тариф из LLM_PRICES)")
    print(f"\nИтог: {answered}/{len(questions)} вопросов прошли проверку "
          f"(данные найдены и ответ не оборван)")
    if truncated:
        print(f"Оборванных ответов: {truncated} — поднимите COMPOSER_MAX_TOKENS")

    # --- повторный вопрос: кэш должен стоить 0 обращений к API ---
    # Проверяем на вопросе типа explain: ответы, собранные кодом (факт, прогноз),
    # кэшировать нечего — они и так отдаются шаблоном без вызова модели.
    probe = first_explain[0] if first_explain else (questions[0][0] if questions else None)
    if probe:
        before = sum(m["calls"] for m in snap["models"])
        repeated = await pipeline.answer(1, 42, probe)
        async with sessions() as s:
            snap2 = await usage_snapshot(s, settings)
        after = sum(m["calls"] for m in snap2["models"])
        same = first_explain[1] == repeated.text if first_explain else None
        verdict = "кэш работает" if after - before == 0 else "КЭШ НЕ СРАБОТАЛ"
        print(f"Повторный вопрос ({probe[:40]}…): обращений к API +{after - before} — {verdict}"
              + (f", текст совпал: {same}" if same is not None else ""))

    await llm.close()
    await engine.dispose()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
