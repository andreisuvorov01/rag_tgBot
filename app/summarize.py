"""Сводный отчёт по документу: обзор показателей, периодов и содержимого
одним сообщением. Числа собираются программно; текст — через LLM-композитор
(offline — шаблон)."""
from __future__ import annotations

import logging

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from .storage import Chunk, Document, Fact, LedgerOperation, Metric

log = logging.getLogger(__name__)


async def document_summary(
    pipeline, session: AsyncSession, org_id: int, document_id: int, user_id: int
) -> str:
    doc = await session.get(Document, document_id)
    # id приходит из callback-данных: чужая организация и чужие личные документы недоступны
    if doc is None or doc.org_id != org_id or (doc.is_private and doc.uploaded_by != user_id):
        return "Документ не найден"
    meta = doc.meta or {}

    rows = (await session.execute(
        select(Fact, Metric)
        .join(Metric, Metric.id == Fact.metric_id)
        .where(Fact.document_id == document_id, Fact.needs_review.is_(False))
    )).all()
    by_metric: dict[str, dict] = {}
    for fact, metric in rows:
        e = by_metric.setdefault(metric.name, {
            "name": metric.name, "kind": metric.kind,
            "label": fact.period_label, "value": fact.value, "end": fact.period_end,
        })
        if fact.period_end > e["end"]:
            e.update(label=fact.period_label, value=fact.value, end=fact.period_end)
    metrics_by_kind: dict[str, list] = {}
    for e in by_metric.values():
        metrics_by_kind.setdefault(e["kind"], []).append(e)
    for lst in metrics_by_kind.values():
        lst.sort(key=lambda e: -abs(float(e["value"])))

    ops_count = (await session.execute(
        select(func.count()).select_from(LedgerOperation).where(LedgerOperation.document_id == document_id)
    )).scalar_one()

    excerpt = ""
    first_chunk = (await session.scalars(
        select(Chunk).where(Chunk.document_id == document_id).order_by(Chunk.id).limit(1)
    )).first()
    if first_chunk:
        excerpt = first_chunk.body[:300]

    payload = {
        "type": "doc_summary",
        "document": doc.original_name,
        "uploaded": doc.uploaded_at.strftime("%d.%m.%Y") if doc.uploaded_at else "",
        "facts": meta.get("facts", len(by_metric)),
        "chunks": meta.get("chunks", 0),
        "periods": meta.get("periods") or [],
        "metrics_by_kind": [
            {"kind": kind, "items": items[:10]} for kind, items in metrics_by_kind.items()
        ],
        "operations": int(ops_count or 0),
        "warnings": meta.get("warnings") or [],
        "excerpt": excerpt,
        "doc_id": document_id,
    }
    return await pipeline._compose("расскажи об этом отчёте подробно", payload)
