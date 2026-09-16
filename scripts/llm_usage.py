"""Отчёт о расходе токенов LLM API — единственной платной части системы.

Запуск:
    python -m scripts.llm_usage                 # сводка из БД (то же, что /usage)
    python -m scripts.llm_usage --balance       # только события «баланс кончился»
    python -m scripts.llm_usage --journal       # последние 20 записей журнала
    python -m scripts.llm_usage --journal -n 50
    python -m scripts.llm_usage --json          # машинночитаемый вид
    python -m scripts.llm_usage --price "deepseek-flash=1/4"

Цены можно задать и в .env: LLM_PRICES="модель=вход/выход,..." — за 1 млн токенов.
Без цен отчёт показывает токены, но не деньги.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    # иначе эмодзи отчёта падают с UnicodeEncodeError на Windows (cp1251)
    _reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(_reconfigure):
        _reconfigure(encoding="utf-8")

from app.config import settings  # noqa: E402
from app.usage import (  # noqa: E402
    balance_events,
    format_balance_state,
    format_budget_left,
    format_snapshot,
    read_journal,
    usage_snapshot,
)


def _strip_tags(text: str) -> str:
    for tag in ("<b>", "</b>", "<i>", "</i>", "<pre>", "</pre>"):
        text = text.replace(tag, "")
    return text


async def summary(as_json: bool, price_override: str | None, events_only: bool = False) -> int:
    if price_override:
        settings.llm_prices = price_override
    from app.storage import make_engine, make_sessionmaker

    engine = await make_engine(settings)
    sessions = make_sessionmaker(engine)
    try:
        async with sessions() as s:
            snap = await usage_snapshot(s, settings)
            events = await balance_events(s)
    finally:
        await engine.dispose()
    if as_json:
        snap["balance_events"] = events
        print(json.dumps(snap, ensure_ascii=False, indent=2))
        return 0
    if events_only:
        if not events:
            print("Событий «баланс API исчерпан» не зафиксировано.")
            return 0
        print("События «баланс API исчерпан»:")
        print(f"  всего: {events.get('count', 0)}")
        print(f"  первый: {str(events.get('first_at') or '')[:19].replace('T', ' ')}")
        print(f"  последний: {str(events.get('last_at') or '')[:19].replace('T', ' ')}")
        print(f"  модель: {events.get('last_model') or '—'}")
        print(f"  ответ провайдера: {events.get('last_reason') or '—'}")
        print("\nОтветы в эти моменты собирались шаблоном из тех же данных — "
              "цифры верные, формулировок модели нет.")
        return 0
    print(format_snapshot(snap))
    calls = sum(m["calls"] for m in snap["models"])
    if calls:
        per_q = snap["tasks"].get("compose", {}).get("calls", 0) or 0
        if per_q:
            print(f"\nВопросов с обращением к модели: {per_q} — "
                  f"≈ {snap['prompt_tokens'] / per_q:,.0f} токенов входа на вопрос")
        print(f"Журнал всех записей: {settings.llm_usage_path}")
    for block in (format_budget_left(snap, settings), format_balance_state(events)):
        if block:
            # разметка адресована Telegram — в консоли убираем теги
            print("\n" + _strip_tags(block))
    return 0


def journal(limit: int) -> int:
    rows = read_journal(settings.llm_usage_path, limit=limit)
    if not rows:
        print(f"Журнал пуст: {settings.llm_usage_path}")
        return 0
    print(f"{'время':20} {'задача':10} {'модель':22} {'вход':>8} {'выход':>8}")
    for r in rows:
        task = str(r.get("event") or r.get("task") or "")
        print(
            f"{str(r.get('ts', ''))[:19]:20} {task:10} "
            f"{str(r.get('model', ''))[:22]:22} {int(r.get('prompt_tokens') or 0):>8,} "
            f"{int(r.get('completion_tokens') or 0):>8,}"
            + (f"   ← {str(r.get('reason'))[:60]}" if r.get("event") else "")
        )
    total_in = sum(int(r.get("prompt_tokens") or 0) for r in rows)
    total_out = sum(int(r.get("completion_tokens") or 0) for r in rows)
    print(f"\nЗа {len(rows)} записей: вход {total_in:,}, выход {total_out:,}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Расход токенов LLM API")
    ap.add_argument("--journal", action="store_true", help="показать последние записи журнала")
    ap.add_argument("-n", "--limit", type=int, default=20, help="сколько записей журнала показать")
    ap.add_argument("--json", action="store_true", help="вывести сводку в JSON")
    ap.add_argument("--balance", action="store_true",
                    help="только события «баланс API исчерпан» (когда и сколько раз)")
    ap.add_argument("--price", help='цены: "модель=вход/выход,..." за 1 млн токенов')
    args = ap.parse_args()
    if args.journal:
        return journal(args.limit)
    return asyncio.run(summary(args.json, args.price, events_only=args.balance))


if __name__ == "__main__":
    sys.exit(main())
