"""Ежедневная сводка для подписчиков: состояние базы, ключевые показатели,
топ роста и прогноз по главной метрике. Собирается программно — числа
заземлены, LLM не участвует."""
from __future__ import annotations

import logging
from datetime import date

from sqlalchemy.ext.asyncio import AsyncSession

from .analytics import forecast, prepare_series
from .storage import org_documents, org_metrics, series_for_metric

log = logging.getLogger(__name__)


async def build_digest(session: AsyncSession, org_id: int, user_id: int) -> str:
    docs = await org_documents(session, org_id, user_id)
    metrics = await org_metrics(session, org_id)
    real = [m for m in metrics if m.kind != "identifier"]

    today = date.today()
    lines = [f"📅 <b>Ежедневная сводка за {today:%d.%m.%Y}</b>", ""]
    lines.append(f"Документов: {len(docs)}, показателей: {len(real)}")
    active_docs = sum(1 for d in docs if d.status == "processed" and d.superseded_by_id is None)
    lines.append(f"Актуальных версий документов: {active_docs}")

    # последние значения по ключевым показателям
    key_metrics = [m for m in real if m.kind in ("revenue", "expense")][:5]
    with_rows: list[tuple[object, list[dict]]] = []
    for m in key_metrics:
        rows = await series_for_metric(session, org_id, m.id, user_id=user_id)
        if rows:
            with_rows.append((m, rows))
    if with_rows:
        lines.append("")
        lines.append("<b>Ключевые показатели (последнее значение):</b>")
        for m, rows in with_rows:
            last = rows[-1]
            lines.append(f"• {m.name}: {last['value']:,.4g}".replace(",", " "))

    # топ роста за доступную историю
    growth = []
    for m, rows in with_rows:
        if len(rows) >= 2 and rows[0]["value"]:
            growth.append((m.name, (rows[-1]["value"] - rows[0]["value"]) / abs(rows[0]["value"]) * 100))
    if growth:
        growth.sort(key=lambda t: t[1], reverse=True)
        lines.append("")
        lines.append("<b>Топ роста за историю:</b>")
        for name, pct in growth[:3]:
            lines.append(f"• {name}: {pct:+.1f}%")

    # прогноз по первой метрике с достаточной историей
    for m, rows in with_rows:
        points = prepare_series(rows)
        if len(points) >= 3:
            fc = forecast(points)
            if not fc.get("error"):
                lines.append("")
                lines.append(
                    f"🔮 Прогноз по «{m.name}» на {fc['target']}: "
                    f"<b>{fc['base']:,.4g}</b>".replace(",", " ")
                    + f" (интервал {fc['low']:,.4g} — {fc['high']:,.4g})".replace(",", " ")
                )
            break

    lines.append("")
    lines.append("<i>Отправить: /unsubscribe. Подробности — в диалоге с ботом.</i>")
    return "\n".join(lines)
