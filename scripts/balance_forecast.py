"""На сколько хватит баланса API: оценка числа вопросов по цене провайдера.

Цены задаются в юанях за 1 млн токенов (тарифы DeepSeek — по умолчанию).
Две оценки:

1. **по факту** — если в БД есть зафиксированный расход (`/usage`,
   `app_meta`), берём реальные токены на вопрос и считаем остаток по ним.
   Это точная оценка для вашей манеры спрашивать;
2. **по модели расходов** — если истории нет, считаем по структуре промптов
   системы (см. таблицу в docs/deploy.md).

Запуск:
    python -m scripts.balance_forecast                 # DeepSeek flash, 33 ¥
    python -m scripts.balance_forecast --budget 50
    python -m scripts.balance_forecast --model pro
    python -m scripts.balance_forecast --price-input 2 --price-output 8
    python -m scripts.balance_forecast --cache-hit 0.5   # доля кэшированного входа
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    _reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(_reconfigure):
        _reconfigure(encoding="utf-8")

from app.config import settings  # noqa: E402

# Тарифы DeepSeek (api-docs.deepseek.com, раздел «Модель & 价格»), ¥ за 1 млн токенов.
# «Пик» — будни 9:00–12:00 и 14:00–18:00 по Пекину, остальное время «свободно»
# (свободный тариф = половина пикового). Кэш-хит — повторяющийся префикс промпта.
DEEPSEEK_PRICES = {
    "flash": {"in_off": 1.0, "in_peak": 2.0, "out_off": 4.0, "out_peak": 8.0,
              "cache_off": 0.02, "cache_peak": 0.04},
    "pro": {"in_off": 4.5, "in_peak": 9.0, "out_off": 13.5, "out_peak": 27.0,
            "cache_off": 0.15, "cache_peak": 0.30},
}

# Структура расхода на один вопрос (токены входа/выхода), оценка по промптам:
#   контекст «что говорится о…»: 5 фрагментов × 200 символов
#     (COMPOSER_CONTEXT_ITEMS/_CHARS) + подписи источников ≈ 420 токенов;
#   системная инструкция композитора (~1200 символов) + вопрос + JSON ≈ 600;
#   итого вход ~1,6 тыс., выход ограничен COMPOSER_MAX_TOKENS (200–300).
#   ранжирование: RERANK_MAX_ITEMS=8 × RERANK_SNIPPET_CHARS=200 ≈ 700 токенов
#     плюс инструкция и запрос ≈ 400.
# classify — отдельный вызов только при CLASSIFY_WITH_LLM=true.
PROFILES = {
    "факт/сравнение/состав/прогноз (шаблон, модель не нужна)": {"calls": 0, "in": 0, "out": 0},
    "вопрос по тексту документов («что говорится о…»)": {"calls": 1, "in": 1600, "out": 250},
    "ранжирование / SQL-выборка": {"calls": 1, "in": 1100, "out": 250},
}


def _prompt_tokens_classify() -> int:
    """Токены промпта классификатора (системная инструкция + вопрос)."""
    from app.prompts import CLASSIFIER_SYSTEM

    return len(CLASSIFIER_SYSTEM) // 3 + 20


def cost(in_tokens: float, out_tokens: float, price: dict, *, peak: bool,
         cache_hit: float = 0.0) -> float:
    """Стоимость в юанях. cache_hit — доля входа, взятого из кэша провайдера."""
    hit = max(0.0, min(1.0, cache_hit))
    in_rate = price["in_peak"] if peak else price["in_off"]
    out_rate = price["out_peak"] if peak else price["out_off"]
    cache_rate = price["cache_peak"] if peak else price["cache_off"]
    missed = in_tokens * (1.0 - hit)
    cached = in_tokens * hit
    return (missed * in_rate + cached * cache_rate) / 1e6 + out_tokens * out_rate / 1e6


def fmt_questions(count: float) -> str:
    if count >= 1000:
        return f"{count / 1000:.1f} тыс."
    return f"{count:.0f}"


async def actual_tokens_per_question() -> dict | None:
    """Реальный расход на вопрос из БД (если API уже вызывался)."""
    from app.storage import make_engine, make_sessionmaker
    from app.usage import usage_snapshot

    engine = await make_engine(settings)
    sessions = make_sessionmaker(engine)
    try:
        async with sessions() as s:
            snap = await usage_snapshot(s, settings)
    finally:
        await engine.dispose()
    total_calls = sum(m["calls"] for m in snap["models"])
    if not total_calls:
        return None
    compose_calls = snap["tasks"].get("compose", {}).get("calls", 0) or total_calls
    return {
        "in": snap["prompt_tokens"] / compose_calls,
        "out": snap["completion_tokens"] / compose_calls,
        "cached": snap["cached_tokens"] / compose_calls,
        "calls": total_calls,
        "questions": compose_calls,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="На сколько вопросов хватит баланса API")
    ap.add_argument("--budget", type=float, default=33.0, help="баланс в юанях (по умолчанию 33)")
    ap.add_argument("--model", choices=sorted(DEEPSEEK_PRICES), default="flash",
                    help="тариф DeepSeek: flash (по умолчанию) или pro")
    ap.add_argument("--price-input", type=float, help="своя цена входа, ¥ за 1 млн (свободный тариф)")
    ap.add_argument("--price-output", type=float, help="своя цена выхода, ¥ за 1 млн")
    ap.add_argument("--peak", action="store_true", help="считать по пиковому тарифу")
    ap.add_argument("--cache-hit", type=float, default=0.0,
                    help="доля входа из кэша провайдера (0..1), по умолчанию 0")
    args = ap.parse_args()

    price = dict(DEEPSEEK_PRICES[args.model])
    if args.price_input:
        price["in_off"] = price["in_peak"] = args.price_input
    if args.price_output:
        price["out_off"] = price["out_peak"] = args.price_output

    print(f"=== На сколько хватит {args.budget:g} ¥ ({args.model}, "
          f"{'пиковый' if args.peak else 'свободный'} тариф) ===\n")
    print(f"Тариф: вход {price['in_peak'] if args.peak else price['in_off']:g} ¥/млн, "
          f"выход {price['out_peak'] if args.peak else price['out_off']:g} ¥/млн, "
          f"кэш-хит {price['cache_peak'] if args.peak else price['cache_off']:g} ¥/млн")

    actual = asyncio.run(actual_tokens_per_question())
    if actual:
        per_q = cost(actual["in"], actual["out"], price, peak=args.peak,
                     cache_hit=args.cache_hit)
        print(f"\n[по факту] зафиксировано вызовов: {actual['calls']}, "
              f"вопросов: {actual['questions']}")
        print(f"           на вопрос: вход {actual['in']:,.0f} ток., "
              f"выход {actual['out']:,.0f} ток."
              + (f", из них кэш {actual['cached']:,.0f}" if actual.get("cached") else ""))
        print(f"           цена вопроса: {per_q:.5f} ¥  →  "
              f"хватит на {fmt_questions(args.budget / per_q)} вопросов")
    else:
        print("\n[по факту] расход в БД ещё не зафиксирован "
              "(внешний API не вызывался) — ниже оценка по структуре промптов")

    print("\n[по модели расходов]")
    print(f"{'тип вопроса':52} {'вызовов':>8} {'¥/вопрос':>10} {'хватит на':>12}")
    classify_in = _prompt_tokens_classify()
    # классификация вызывается, только если включена (по умолчанию — правила)
    for label, prof in PROFILES.items():
        in_tok = prof["in"] + (classify_in if settings.classify_with_llm and prof["calls"] else 0)
        per_q = cost(in_tok, prof["out"], price, peak=args.peak, cache_hit=args.cache_hit)
        have = "∞ (токены не тратятся)" if per_q == 0 else fmt_questions(args.budget / per_q)
        print(f"{label:52} {prof['calls']:>8} {per_q:>10.5f} {have:>12}")

    # Смешанный профиль: большинство вопросов — факты и сравнения (бесплатно),
    # модель нужна в каждом четвёртом (вывод по документам/ранжирование).
    ctx = PROFILES["вопрос по тексту документов («что говорится о…»)"]
    mixed_per_q = cost(ctx["in"], ctx["out"], price, peak=args.peak,
                       cache_hit=args.cache_hit) / 4
    print(f"\n[смешанный поток] если модель нужна в 1 вопросе из 4: "
          f"{mixed_per_q:.5f} ¥/вопрос → хватит на {fmt_questions(args.budget / mixed_per_q)} вопросов")
    print("Проверка на своих вопросах: посмотрите /usage до и после серии из 10–20 "
          "вопросов —\nскрипт выше возьмёт фактические токены из БД и пересчитает остаток сам.")

    print("\nЧто уменьшает расход: LLM_MODEL_SMALL на служебные шаги, "
          "COMPOSE_MODE=auto,\nRERANKER=crossencoder, CLASSIFY_WITH_LLM=false, "
          "ANSWER_CACHE_SIZE, COMPOSER_CONTEXT_CHARS.")
    print("Реальный расход: python -m scripts.llm_usage")
    return 0


if __name__ == "__main__":
    sys.exit(main())
