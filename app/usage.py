"""Учёт расхода токенов LLM API.

Внешний API — единственная платная часть системы, поэтому расход нужно видеть
до того, как придёт счёт. Здесь три уровня:

1. журнал `{DATA_DIR}/llm_usage.jsonl` — каждая запись к API (задача, модель,
   токены). Пишется построчно, файл можно читать чем угодно и он не растёт в
   памяти;
2. агрегат в таблице `app_meta` — `llm_usage:<модель>` = JSON с итогами по
   задачам. Переживает перезапуск и доступен боту без чтения журнала;
3. отчёт `scripts/llm_usage.py` и команда `/usage` в Telegram — сводка с
   оценкой в деньгах по ценам из `LLM_PRICES`.
"""
from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from .config import Settings
from .storage import get_meta, set_meta

log = logging.getLogger(__name__)

META_PREFIX = "llm_usage:"
# События «баланс API исчерпан»: чтобы после пополнения счёта было видно,
# сколько запросов упало и когда это началось.
BALANCE_META_KEY = "llm_balance_events"


def log_call(settings: Settings, record: dict[str, Any]) -> None:
    """Дописать одну запись расхода в журнал. Ошибки не пробрасываются:
    учёт не должен влиять на ответ пользователю."""
    try:
        path = settings.llm_usage_path
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps({"ts": datetime.now(UTC).isoformat(timespec="seconds"), **record},
                          ensure_ascii=False)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as e:
        log.warning("Не удалось записать расход токенов в журнал: %s", e)


async def add_usage(session: AsyncSession, record: dict[str, Any]) -> None:
    """Агрегировать расход в app_meta (по модели), чтобы /usage не читал журнал."""
    model = str(record.get("model") or "unknown")
    key = f"{META_PREFIX}{model}"
    try:
        raw = await get_meta(session, key)
        data: dict[str, Any] = json.loads(raw) if raw else {}
    except Exception:
        data = {}
    tasks = data.setdefault("tasks", {})
    task = str(record.get("task") or "other")
    t = tasks.setdefault(task, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                                "cached_tokens": 0, "reasoning_tokens": 0})
    t["calls"] += 1
    t["prompt_tokens"] += int(record.get("prompt_tokens") or 0)
    t["completion_tokens"] += int(record.get("completion_tokens") or 0)
    t["cached_tokens"] += int(record.get("cached_tokens") or 0)
    t["reasoning_tokens"] = int(t.get("reasoning_tokens", 0)) + int(
        record.get("reasoning_tokens") or 0)
    data["calls"] = int(data.get("calls", 0)) + 1
    data["prompt_tokens"] = int(data.get("prompt_tokens", 0)) + int(record.get("prompt_tokens") or 0)
    data["completion_tokens"] = int(data.get("completion_tokens", 0)) + int(
        record.get("completion_tokens") or 0)
    data["cached_tokens"] = int(data.get("cached_tokens", 0)) + int(record.get("cached_tokens") or 0)
    data["reasoning_tokens"] = int(data.get("reasoning_tokens", 0)) + int(
        record.get("reasoning_tokens") or 0)
    data["updated_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    await set_meta(session, key, json.dumps(data, ensure_ascii=False))


async def usage_snapshot(session: AsyncSession, settings: Settings) -> dict[str, Any]:
    """Сводка расхода: по моделям и задачам, с оценкой стоимости."""
    from sqlalchemy import select

    from .storage import AppMeta

    rows = (await session.execute(
        select(AppMeta.key, AppMeta.value).where(AppMeta.key.like(f"{META_PREFIX}%"))
    )).all()
    prices = settings.llm_price_map
    models: list[dict[str, Any]] = []
    total_in = total_out = total_cached = total_reasoning = 0
    cost_total = 0.0
    task_totals: dict[str, dict[str, int]] = {}
    for key, value in rows:
        try:
            data = json.loads(value)
        except json.JSONDecodeError:
            continue
        model = str(key)[len(META_PREFIX):]
        tin = int(data.get("prompt_tokens", 0))
        tout = int(data.get("completion_tokens", 0))
        tcached = int(data.get("cached_tokens", 0))
        treason = int(data.get("reasoning_tokens", 0))
        total_in += tin
        total_out += tout
        total_cached += tcached
        total_reasoning += treason
        price = prices.get(model)
        # кэшированный вход у провайдеров дешевле (обычно 10-25% цены) — оценка
        # по полной цене даёт верхнюю границу, а не занижает расход
        cost = None
        if price:
            cost = tin / 1e6 * price[0] + tout / 1e6 * price[1]
            cost_total += cost
        for task, t in (data.get("tasks") or {}).items():
            agg = task_totals.setdefault(task, {"calls": 0, "prompt_tokens": 0,
                                                "completion_tokens": 0, "reasoning_tokens": 0})
            agg["calls"] += int(t.get("calls", 0))
            agg["prompt_tokens"] += int(t.get("prompt_tokens", 0))
            agg["completion_tokens"] += int(t.get("completion_tokens", 0))
            agg["reasoning_tokens"] = int(agg.get("reasoning_tokens", 0)) + int(
                t.get("reasoning_tokens", 0))
        models.append({
            "model": model,
            "calls": int(data.get("calls", 0)),
            "prompt_tokens": tin,
            "completion_tokens": tout,
            "cached_tokens": tcached,
            "reasoning_tokens": treason,
            "cost": cost,
        })
    return {
        "models": sorted(models, key=lambda m: -(m["prompt_tokens"] + m["completion_tokens"])),
        "tasks": task_totals,
        "prompt_tokens": total_in,
        "completion_tokens": total_out,
        "cached_tokens": total_cached,
        "reasoning_tokens": total_reasoning,
        "cost": cost_total if prices else None,
        "priced_models": sorted(prices),
    }


def format_snapshot(snap: dict[str, Any], *, per_question: int | None = None) -> str:
    """Текстовый отчёт о расходе (для /usage и scripts/llm_usage.py)."""
    calls = sum(m["calls"] for m in snap["models"])
    lines = ["📊 <b>Расход токенов LLM</b>", ""]
    if not calls:
        lines.append("Пока ни одного вызова внешнего API не зафиксировано.")
        lines.append("<i>Офлайн-режим и шаблонные ответы токенов не тратят.</i>")
        return "\n".join(lines)
    lines.append(f"Всего вызовов: <b>{calls}</b>")
    lines.append(f"Вход: <b>{snap['prompt_tokens']:,}</b> · выход: <b>{snap['completion_tokens']:,}</b>")
    if snap.get("cached_tokens"):
        lines.append(f"Из входа взято из кэша провайдера: {snap['cached_tokens']:,}")
    if snap.get("reasoning_tokens"):
        lines.append(
            f"Из выхода — «мысли» модели: {snap['reasoning_tokens']:,} "
            f"<i>(оплачиваются, но в ответе их нет: выключите LLM_THINKING)</i>"
        )
    if snap.get("cost") is not None:
        lines.append(f"Оценка стоимости: <b>${snap['cost']:.4f}</b>")
        if per_question and calls:
            lines.append(f"≈ ${snap['cost'] / max(1, per_question):.5f} на вопрос")
    lines.append("")
    lines.append("<b>По моделям:</b>")
    for m in snap["models"]:
        cost = f" · ${m['cost']:.4f}" if m["cost"] is not None else ""
        lines.append(
            f"• {m['model']}: {m['calls']} выз., "
            f"{m['prompt_tokens']:,}→{m['completion_tokens']:,}{cost}"
        )
    if snap["tasks"]:
        lines.append("")
        lines.append("<b>По задачам:</b>")
        names = {"classify": "классификация", "rerank": "реранкинг", "sql": "SQL",
                 "compose": "ответ", "ocr": "OCR", "other": "прочее"}
        for task, t in sorted(snap["tasks"].items(), key=lambda kv: -kv[1]["calls"]):
            lines.append(
                f"• {names.get(task, task)}: {t['calls']} выз., "
                f"{t['prompt_tokens']:,}→{t['completion_tokens']:,}"
            )
    unpriced = [m["model"] for m in snap["models"] if m["cost"] is None]
    if unpriced:
        lines.append("")
        lines.append(
            f"<i>Без цены: {', '.join(unpriced)}. Задайте LLM_PRICES "
            f"(\"модель=вход/выход\"), чтобы видеть деньги.</i>"
        )
    return "\n".join(lines)


def read_journal(path: Path, *, limit: int = 0) -> list[dict[str, Any]]:
    """Записи журнала (для scripts/llm_usage.py --journal)."""
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out[-limit:] if limit else out


# --------------------------------------------------------------- баланс API

def log_balance_event(settings: Settings, record: dict[str, Any]) -> None:
    """Записать событие «баланс исчерпан» в тот же журнал, что и расход."""
    log_call(settings, {"event": "balance_exhausted", **record})


async def add_balance_event(session: AsyncSession, record: dict[str, Any]) -> None:
    """Счётчик событий 402 в app_meta: без него после пополнения счёта нельзя
    понять, сколько запросов упало и когда это началось."""
    try:
        raw = await get_meta(session, BALANCE_META_KEY)
        data: dict[str, Any] = json.loads(raw) if raw else {}
    except Exception:
        data = {}
    data["count"] = int(data.get("count", 0)) + 1
    data["last_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    data["last_reason"] = str(record.get("reason") or "")[:200]
    data["last_model"] = str(record.get("model") or "")
    data.setdefault("first_at", data["last_at"])
    await set_meta(session, BALANCE_META_KEY, json.dumps(data, ensure_ascii=False))


async def balance_events(session: AsyncSession) -> dict[str, Any]:
    """Сводка по событиям «баланс исчерпан» (пусто -> {})."""
    try:
        raw = await get_meta(session, BALANCE_META_KEY)
    except Exception:
        return {}
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def format_balance_state(events: dict[str, Any]) -> str:
    """Строка для /usage про исчерпания баланса."""
    if not events:
        return ""
    when = str(events.get("last_at") or "")[:19].replace("T", " ")
    return (
        f"\n⚠️ <b>Баланс API исчерпывался:</b> {events.get('count', 0)} раз, "
        f"последний — {when}\n"
        f"<i>Ответы в этот момент собирались шаблоном (данные те же). "
        f"Пополните счёт у провайдера.</i>"
    )


def format_budget_left(snap: dict[str, Any], settings: Settings) -> str:
    """Остаток баланса и грубая оценка «на сколько ещё хватит».

    Работает только когда заданы и LLM_BALANCE, и LLM_PRICES (в одних деньгах).
    Оценка числа вопросов — по средней цене вопроса, которая уже зафиксирована
    в учёте; без истории вопросы не оцениваются.
    """
    budget = float(settings.llm_balance or 0)
    spent = snap.get("cost")
    if budget <= 0 or spent is None:
        return ""
    left = budget - spent
    lines = [f"\n💰 <b>Баланс:</b> {budget:g} − израсходовано {spent:.4f} "
             f"= осталось <b>{left:.4f}</b>"]
    questions = snap["tasks"].get("compose", {}).get("calls", 0)
    if questions and spent > 0:
        per_q = spent / questions
        left_q = max(0.0, left) / per_q
        lines.append(f"<i>Средняя цена вопроса {per_q:.5f}; остатка хватит "
                     f"примерно на {left_q:,.0f} вопросов с моделью.</i>")
    if left <= 0:
        lines.append("<i>Баланс исчерпан — пополните счёт, иначе ответы будут "
                     "шаблонными.</i>")
    return "\n".join(lines)
