"""Опциональный Qdrant-бэкенд для векторного поиска (практика: pgvector
остаётся по умолчанию; Qdrant включается при росте объёмов).

Включение: в .env —  VECTOR_BACKEND=qdrant  +  QDRANT_URL=...
В Qdrant зеркалятся ТОЛЬКО неприватные чанки (личные документы ищутся
по прежней SQL-ветке — ACL не может быть обойдён).

Клиент синхронный, поэтому все вызовы уходят в отдельный поток: иначе
сетевой запрос к Qdrant (timeout 10 с) блокировал бы event loop бота для
всех пользователей — search_chunks вызывается на каждый вопрос.
"""
from __future__ import annotations

import asyncio
import logging

from .config import settings

log = logging.getLogger(__name__)

COLLECTION = "rag_chunks"
_client = None


def enabled() -> bool:
    if settings.vector_backend != "qdrant" or not settings.qdrant_url:
        return False
    try:
        import qdrant_client  # noqa: F401

        return True
    except ImportError:
        log.warning("VECTOR_BACKEND=qdrant, но пакет qdrant-client не установлен — используется SQL-ветка")
        return False


def get_client():
    global _client
    if _client is None:
        from qdrant_client import QdrantClient

        _client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None, timeout=10)
    return _client


def _ensure_collection(client, dim: int) -> None:
    from qdrant_client import models

    if not client.collection_exists(COLLECTION):
        client.create_collection(
            COLLECTION,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
        )


async def upsert_chunks(items: list[dict]) -> bool:
    """items: [{id, org_id, embedding, body, page, section, document_id}].
    Лучшие усилия: сбой зеркала не ломает загрузку."""
    if not enabled() or not items:
        return False

    def _do() -> bool:
        from qdrant_client import models

        client = get_client()
        _ensure_collection(client, len(items[0]["embedding"]))
        points = [
            models.PointStruct(
                id=it["id"],
                vector=it["embedding"],
                payload={
                    "org_id": it["org_id"],
                    "document_id": it["document_id"],
                    "body": it["body"],
                    "page": it["page"],
                    "section": it["section"],
                },
            )
            for it in items
        ]
        client.upsert(COLLECTION, points=points, wait=False)
        return True

    try:
        return await asyncio.to_thread(_do)
    except Exception as e:
        log.warning("Qdrant upsert не удался: %s", e)
        return False


async def delete_document(document_id: int) -> bool:
    """Удаляет точки документа из зеркала (перечитывание файла). Лучшие усилия."""
    if not enabled():
        return False

    def _do() -> bool:
        from qdrant_client import models

        get_client().delete(
            COLLECTION,
            points_selector=models.Filter(
                must=[models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id))]
            ),
            # wait=True: иначе удаление может приземлиться уже ПОСЛЕ повторной
            # вставки точек того же документа (reparse) и стереть свежий индекс
            wait=True,
        )
        return True

    try:
        return await asyncio.to_thread(_do)
    except Exception as e:
        log.warning("Qdrant delete не удался: %s", e)
        return False


async def search_chunks(query_vec: list[float], org_id: int, k: int) -> list[dict] | None:
    """Поиск только по неприватным чанкам (в Qdrant их и храним).
    None — бэкенд недоступен, вызывающий уходит в SQL-ветку."""
    if not enabled():
        return None

    def _do() -> list[dict]:
        from qdrant_client import models

        client = get_client()
        hits = client.query_points(
            COLLECTION,
            query=query_vec,
            limit=k,
            query_filter=models.Filter(
                must=[models.FieldCondition(key="org_id", match=models.MatchValue(value=org_id))]
            ),
            with_payload=True,
        ).points
        return [
            {
                "id": h.id,
                "document_id": h.payload.get("document_id"),
                "body": h.payload.get("body", ""),
                "page": h.payload.get("page"),
                "section": h.payload.get("section"),
                "score": float(h.score),
            }
            for h in hits
        ]

    try:
        return await asyncio.to_thread(_do)
    except Exception as e:
        log.warning("Qdrant search не удался: %s", e)
        return None
