"""Переиндексация эмбеддингов после смены провайдера/модели.

Эмбеддинги несовместимы между моделями (другая размерность и семантика),
поэтому при смене EMBEDDINGS_PROVIDER нужно пересчитать все векторы.

Запуск:  python -m scripts.reindex
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select  # noqa: E402

from app.config import settings  # noqa: E402
from app.embeddings import EmbeddingService, embeddings_fingerprint  # noqa: E402
from app.storage import Chunk, Metric, get_meta, make_engine, make_sessionmaker, set_meta  # noqa: E402

FINGERPRINT_KEY = "embeddings_fingerprint"


def embeddings_fingerprint_from_settings() -> str:
    return embeddings_fingerprint(settings)


async def reindex_all(engine, emb: EmbeddingService) -> tuple[int, int]:
    """Пересчитывает векторы всех чанков и показателей. -> (чанки, метрики)."""
    sessions = make_sessionmaker(engine)
    n_chunks = n_metrics = 0
    async with sessions() as s:
        chunks = (await s.scalars(select(Chunk).order_by(Chunk.id))).all()
        for i in range(0, len(chunks), 32):
            batch = chunks[i : i + 32]
            vecs = await emb.embed_passages([c.body for c in batch])
            for chunk, vec in zip(batch, vecs, strict=True):
                chunk.embedding = vec
            n_chunks += len(batch)
            print(f"  чанки: {n_chunks}/{len(chunks)}")
        metrics = (await s.scalars(select(Metric).order_by(Metric.id))).all()
        if metrics:
            vecs = await emb.embed_texts([m.name for m in metrics], is_query=True)
            for metric, vec in zip(metrics, vecs, strict=True):
                metric.embedding = vec
            n_metrics = len(metrics)
        await set_meta(s, FINGERPRINT_KEY, embeddings_fingerprint_from_settings())
        await s.commit()
    return n_chunks, n_metrics


async def main() -> None:
    engine = await make_engine(settings)
    emb = EmbeddingService(settings)
    fp = embeddings_fingerprint_from_settings()
    print(f"Провайдер: {emb.provider}, модель: {emb.model}, отпечаток: {fp}")
    if emb.provider == "hash" and settings.embeddings_provider != "hash":
        print(
            "⚠️ Похоже, в EMBEDDINGS_PROVIDER указано имя модели или пакет "
            "sentence-transformers недоступен в этом интерпретаторе.\n"
            "   Правильно: EMBEDDINGS_PROVIDER=sentence_transformers, "
            "модель — в EMBEDDINGS_LOCAL_MODEL.\n"
            "   Запускайте: venv\\Scripts\\python.exe -m scripts.reindex"
        )
        await engine.dispose()
        return 2
    sessions = make_sessionmaker(engine)
    async with sessions() as s:
        stored = await get_meta(s, FINGERPRINT_KEY)
    if stored == fp:
        print("Отпечаток совпадает — переиндексация не требуется.")
        await engine.dispose()
        return
    chunks, metrics = await reindex_all(engine, emb)
    print(f"Готово: переиндексировано {chunks} чанков и {metrics} показателей.")
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
