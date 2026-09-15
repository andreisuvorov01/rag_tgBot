"""Форматирование ответов для Telegram (HTML) и шаблонный рендер ответов из JSON-данных.

render_answer() — базовый офлайн-рендер, который используется:
1) как fallback, если внешний LLM API недоступен;
2) как MockLLM в offline-демо и тестах.
"""
from __future__ import annotations

import html
from decimal import Decimal
from typing import Any

from .money import to_decimal

TELEGRAM_LIMIT = 4000

# Заголовки групп показателей (единые для бота и шаблонов)
KIND_TITLES = {
    "revenue": "📈 Доходы",
    "expense": "📉 Расходы",
    "asset": "🏦 Активы",
    "liability": "📋 Обязательства",
    "identifier": "🔢 Реквизиты",
    "other": "• Прочее",
}


def escape(text: str | None) -> str:
    return html.escape(str(text or ""), quote=False)


def fmt_number(value, digits: int = 2) -> str:
    """Число в русской записи. Принимает Decimal (деньги) и float (расчёты):
    Decimal форматируется сам, иначе `Decimal / float` бросает TypeError."""
    if isinstance(value, Decimal):
        s = f"{value:,.{digits}f}"
    else:
        s = f"{float(value):,.{digits}f}"
    return s.replace(",", " ").replace(".", ",")


def fmt_money(value, currency: str | None = "RUB", digits: int = 2) -> str:
    if value is None:
        return "—"
    cur = {"RUB": "₽", "USD": "$", "EUR": "€"}.get((currency or "RUB").upper(), currency or "")
    amount = value if isinstance(value, Decimal) else to_decimal(value)
    if amount is None:
        return "—"
    abs_v = abs(amount)
    if abs_v >= Decimal(10) ** 9:
        return f"{fmt_number(amount / Decimal(10) ** 9, digits)} млрд {cur}".strip()
    if abs_v >= Decimal(10) ** 6:
        return f"{fmt_number(amount / Decimal(10) ** 6, digits)} млн {cur}".strip()
    if abs_v >= Decimal(1000):
        return f"{fmt_number(amount / Decimal(1000), digits)} тыс. {cur}".strip()
    return f"{fmt_number(amount, digits)} {cur}".strip()


def fmt_pct(value, digits: int = 1) -> str:
    if value is None:
        return "—"
    v = value if isinstance(value, Decimal) else to_decimal(value)
    if v is None:
        return "—"
    return f"{v:+,.{digits}f}%".replace(",", " ").replace(".", ",")


def render_table(rows: list[list[str]]) -> str:
    """Моноширинная таблица для <pre>."""
    if not rows:
        return ""
    widths = [max(len(str(r[i])) for r in rows) for i in range(len(rows[0]))]
    lines = []
    for i, row in enumerate(rows):
        line = "  ".join(str(cell).ljust(widths[j]) for j, cell in enumerate(row))
        lines.append(line)
        if i == 0:
            lines.append("  ".join("-" * w for w in widths))
    return "\n".join(lines)


def chunk_message(text: str, limit: int = TELEGRAM_LIMIT) -> list[str]:
    """Разбить текст на части не длиннее limit.

    Абзацы — предпочтительная граница, но один длинный абзац (LLM охотно
    выдаёт «простыню» без пустых строк) тоже обязан быть разрезан: раньше
    такой текст уходил одним сообщением и Telegram отклонял его целиком,
    так что пользователь не получал ответа вовсе.
    """
    if len(text) <= limit:
        return [text]
    parts: list[str] = []
    current = ""
    for para in text.split("\n\n"):
        # длинный абзац режем по строкам, затем жёстко по символам
        pieces = para.split("\n") if len(para) > limit else [para]
        for piece in pieces:
            while len(piece) > limit:
                parts.append(piece[:limit])
                piece = piece[limit:]
            if len(current) + len(piece) + 2 > limit and current:
                parts.append(current.strip())
                current = ""
            current += piece + "\n\n"
    if current.strip():
        parts.append(current.strip())
    return [p for p in parts if p]


# ---------------------------------------------------------------------------
# Шаблонный рендер ответов из payload (структура данных описана в qa.py)
# ---------------------------------------------------------------------------

def _render_history_table(payload: dict[str, Any]) -> str:
    metric = payload.get("metric", {})
    cur = metric.get("currency")
    unit = metric.get("unit")
    rows = [["Период", "Значение", "Источник"]]
    for r in payload.get("history", []):
        val = fmt_money(r.get("value"), cur) if unit != "%" else f"{fmt_number(r.get('value', 0))} %"
        rows.append([r.get("label", ""), val, r.get("source", "")])
    return f"<pre>{escape(render_table(rows))}</pre>"


def _sources_block(payload: dict[str, Any]) -> str:
    lines = []
    for ctx in payload.get("context", []) or []:
        snippet = ctx.get("text", "")
        if len(snippet) > 350:
            snippet = snippet[:347] + "…"
        lines.append(f"• «{escape(snippet)}» — <i>{escape(ctx.get('source', ''))}</i>")
    return "\n".join(lines)


def render_answer(payload: dict[str, Any]) -> str:
    ptype = payload.get("type", "")
    metric = payload.get("metric", {}) or {}
    name = escape(metric.get("name") or "показатель")
    cur = metric.get("currency")

    if ptype == "factual":
        rows = payload.get("history", [])
        body = _render_history_table(payload)
        return (
            f"📊 <b>Данные по показателю «{name}»</b>\n\n{body}"
            + (f"\n⚠️ Значение(я) требуют подтверждения: {escape(', '.join(payload.get('notes', [])))}" if payload.get("notes") else "")
        )

    if ptype == "compare":
        comp = payload.get("computed", {})
        body = _render_history_table(payload)
        if payload.get("plan_mode"):
            return (
                f"🧮 <b>План/факт по показателю «{name}» за {escape(comp.get('period', ''))}:</b>\n\n{body}\n\n"
                f"Отклонение факта от плана: <b>{fmt_pct(comp.get('deviation_pct'))}</b>"
                f" ({fmt_money(comp.get('abs_change'), cur)})."
            )
        return (
            f"🧮 <b>Сравнение по показателю «{name}»</b>\n\n{body}\n\n"
            f"Изменение: <b>{fmt_pct(comp.get('change_pct'))}</b>"
            f" ({fmt_money(comp.get('abs_change'), cur)})."
        )

    if ptype == "breakdown":
        rows = [["Позиция", "Значение", "Доля"]]
        for it in payload.get("items", []):
            rows.append([
                it.get("name", ""),
                fmt_money(it.get("value"), cur) if metric.get("unit") != "%" else f"{fmt_number(it.get('value', 0))} %",
                fmt_pct(it.get("share_pct")),
            ])
        return (
            f"🧮 <b>Состав показателя «{name}» за {escape(payload.get('period_label', ''))}</b>\n\n"
            f"<pre>{escape(render_table(rows))}</pre>\n"
            f"Итого: <b>{fmt_money(payload.get('total'), cur)}</b>. "
            f"<i>Значения из документов; источник каждой позиции — по запросу «покажи источники».</i>"
        )

    if ptype == "rank":
        lines = []
        for i, r in enumerate(payload.get("ranking", []), 1):
            lines.append(f"{i}. {escape(r['name'])}: {fmt_pct(r.get('growth_pct'))} ({escape(r.get('label_from', ''))} → {escape(r.get('label_to', ''))})")
        return "🧮 <b>Ранжирование по темпу роста</b>\n\n" + "\n".join(lines)

    if ptype == "explain":
        context = _sources_block(payload)
        return (
            f"📚 <b>Что говорят документы</b>\n\n{context or 'В загруженных документах релевантных фрагментов не найдено.'}"
        )

    if ptype == "forecast":
        fc = payload.get("forecast", {})
        comp = payload.get("computed", {})
        scenario = fc.get("scenario")
        scenario_note = ""
        if scenario:
            scenario_note = (
                f"\n\n<i>Сценарий «что если» (множитель динамики ×{fmt_number(scenario['growth_multiplier'])}):"
                f" базовая оценка {fmt_money(fc.get('base_without_scenario'), cur)} скорректирована до"
                f" {fmt_money(fc.get('base'), cur)}.</i>"
            )
        ctx = _sources_block(payload)
        parts = [
            f"🔮 <b>Прогноз по показателю «{name}» на {escape(fc.get('target', ''))} год:"
            f" {fmt_money(fc.get('base'), cur)}</b> "
            f"(интервал {fmt_money(fc.get('low'), cur)} — {fmt_money(fc.get('high'), cur)},"
            f" уверенность: {escape(fc.get('confidence', ''))}).",
            "\n📊 <b>Данные из документов:</b>",
            _render_history_table(payload),
            "\n🧮 <b>Расчёты:</b>",
            f"• средний темп роста: <b>{fmt_pct(comp.get('avg_growth_pct'))}</b> в год;",
            f"• среднегодовой темп (CAGR): <b>{fmt_pct(comp.get('cagr_pct'))}</b>;",
            f"• линейный тренд: {fmt_money(comp.get('trend_value'), cur)} на {escape(fc.get('target', ''))} год;",
            f"• проверка на истории: средняя ошибка ≈ {fmt_number(fc.get('backtest_error_pct', 0), 1)}%;",
            "• вклад методов: "
            + "; ".join(
                f"{escape(k)} — {fmt_money(v, cur)}" for k, v in (fc.get("methods") or {}).items()
            )
            + ".",
        ]
        if payload.get("context"):
            parts.append("\n⚠️ <b>Контекст из документов:</b>\n" + ctx)
        if fc.get("notes"):
            parts.append("\n⚠️ <b>Ограничения:</b> " + "; ".join(escape(n) for n in fc["notes"]) + ".")
        parts.append(
            "\n<i>Прогноз — оценка ИИ-системы, не финансовая гарантия. Каждую цифру можно проверить: источники указаны в таблице.</i>"
        )
        return "\n".join(parts) + scenario_note

    if ptype == "doc_summary":
        kind_titles = {
            "revenue": "📈 Доходы", "expense": "📉 Расходы", "asset": "🏦 Активы",
            "liability": "📋 Обязательства", "identifier": "🔢 Реквизиты", "other": "• Прочее",
        }
        lines = [
            f"📖 <b>Обзор документа «{escape(payload.get('document', ''))}»</b>",
            f"Загружен: {escape(payload.get('uploaded', ''))} · "
            f"Показателей: {payload.get('facts', 0)} · Фрагментов: {payload.get('chunks', 0)}",
        ]
        if payload.get("periods"):
            lines.append(f"Периоды: {escape(', '.join(payload['periods'][:8]))}")
        for group in payload.get("metrics_by_kind", []):
            title = kind_titles.get(group["kind"], "• Прочее")
            lines.append(f"\n{title}:")
            for item in group["items"]:
                val = fmt_money(item.get("value"), None) if group["kind"] != "identifier" \
                    else fmt_number(item.get("value", 0), 0)
                lines.append(f"  • {escape(item['name'])} — {val} ({escape(item['label'])})")
        if payload.get("operations"):
            lines.append(f"\n💳 Операций в выписке: {payload['operations']} — спросите «операции за месяц».")
        if payload.get("warnings"):
            lines.append("⚠️ " + "; ".join(escape(w) for w in payload["warnings"][:4]))
        if payload.get("excerpt"):
            excerpt = escape(payload["excerpt"][:250])
            lines.append(f"\n<i>Из документа: {excerpt}…</i>")
        lines.append("\n<i>Задавайте вопросы по показателям или запросите прогноз.</i>")
        return "\n".join(lines)

    if ptype == "sql_result":
        return (
            f"📊 <b>Результат по запросу</b> «{escape(payload.get('query', ''))}»\n\n"
            f"<pre>{escape(payload.get('table', ''))}</pre>\n"
            f"<i>Строк всего: {payload.get('rows_total', '?')}. Данные из SQL-базы фактов.</i>"
        )

    if ptype == "nodata":
        return (
            f"🤷 По запросу «{escape(payload.get('query', ''))}» данных не найдено.\n"
            f"Загрузите соответствующие отчёты или переформулируйте вопрос. "
            f"Доступные показатели: /metrics"
        )

    return escape(str(payload.get("text", "Не удалось сформировать ответ.")))
