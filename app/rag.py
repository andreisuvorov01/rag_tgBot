"""Гибридный поиск по текстовым фрагментам: плотные (эмбеддинги) + разреженные
(BM25) результаты, слитые рекипрокным ранговым фьюжном (RRF).

Практика заимствована у enterprise-RAG платформ (Onyx/Danswer — Vespa BM25 +
векторы + RRF) и Anthropic Contextual Retrieval: комбинация dense+sparse
снижает долю промахов поиска примерно вдвое по сравнению с dense-only.
"""
from __future__ import annotations

import asyncio
import logging
import math
from collections import Counter

from sqlalchemy import select as sa_select
from sqlalchemy.ext.asyncio import AsyncSession

from .embeddings import EmbeddingService, _tokens
from .storage import Chunk, Document, vector_search, visible_documents_condition

log = logging.getLogger(__name__)

K1 = 1.5
B = 0.75
RRF_K = 60


def bm25_search(corpus: list[dict], query: str, k: int = 12) -> list[dict]:
    """Классический Okapi BM25 по корпусу чанков организации.
    Токены — с префиксным стеммингом (устойчив к русской морфологии)."""
    return bm25_search_indexed(build_index(corpus), query, k)


def build_index(corpus: list[dict]) -> dict:
    """Инвертированный индекс корпуса: токен -> [(позиция, частота)].

    Считается один раз на версию корпуса (см. _index_cache): раньше корпус
    ре-токенизировался на КАЖДЫЙ вариант запроса, то есть O(корпус) работы
    в event loop на один вопрос.
    """
    docs_tokens = [_tokens(d["body"]) for d in corpus]
    df: Counter = Counter()
    postings: dict[str, list[tuple[int, int]]] = {}
    lengths: list[int] = []
    for i, toks in enumerate(docs_tokens):
        lengths.append(len(toks) or 1)
        tf = Counter(toks)
        for term, freq in tf.items():
            postings.setdefault(term, []).append((i, freq))
        df.update(tf.keys())
    total = sum(lengths)
    return {
        "corpus": corpus,
        "postings": postings,
        "df": df,
        "lengths": lengths,
        "n": len(corpus),
        "avgdl": (total / len(lengths)) if lengths else 1.0,
    }


def bm25_search_indexed(index: dict, query: str, k: int = 12) -> list[dict]:
    """BM25 по готовому инвертированному индексу: перебираются только
    документы, содержащие слова запроса, а не весь корпус."""
    if not index or not index["n"]:
        return []
    corpus = index["corpus"]
    postings = index["postings"]
    df = index["df"]
    lengths = index["lengths"]
    n = index["n"]
    avgdl = index["avgdl"] or 1.0

    scores: dict[int, float] = {}
    for term in _tokens(query):
        plist = postings.get(term)
        if not plist:
            continue
        idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
        for i, tf in plist:
            dl = lengths[i]
            scores[i] = scores.get(i, 0.0) + idf * tf * (K1 + 1) / (tf + K1 * (1 - B + B * dl / avgdl))
    if not scores:
        return []
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return [{**corpus[i], "score": s} for i, s in ranked]


def rrf_fuse(result_lists: list[list[dict]], top: int = 8) -> list[dict]:
    """Reciprocal Rank Fusion: score(d) = Σ 1/(RRF_K + rank_i(d))."""
    scores: dict[int, float] = {}
    meta: dict[int, dict] = {}
    for results in result_lists:
        for rank, r in enumerate(results):
            rid = r["id"]
            scores[rid] = scores.get(rid, 0.0) + 1.0 / (RRF_K + rank + 1)
            meta.setdefault(rid, r)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top]
    fused = []
    for rid, s in ranked:
        d = dict(meta[rid])
        d["rrf_score"] = s
        fused.append(d)
    return fused


def focus_snippet(body: str, query: str, max_chars: int = 600, radius: int = 1) -> str:
    """Строки фрагмента, в которых есть слова вопроса (± соседние): «кто автор»
    показывает «ФИО руководителя … Лесных Ю. А.», а не начало страницы.
    Без пересечений — начало фрагмента.

    Склейка идёт с пометкой «…», когда между сохранёнными строками есть
    пропуск: иначе строки из разных мест таблицы выглядели бы одной цитатой.
    """
    lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
    qtoks = set(_tokens(query))
    if not qtoks or len(lines) <= 2:
        return body[:max_chars]
    hits = [i for i, ln in enumerate(lines) if qtoks & set(_tokens(ln))]
    if not hits:
        return body[:max_chars]
    keep = sorted({j for i in hits for j in range(max(0, i - radius), min(len(lines), i + radius + 1))})
    out: list[str] = []
    total, prev = 0, None
    for j in keep:
        if total + len(lines[j]) > max_chars:
            break
        if prev is not None and j > prev + 1:
            out.append("…")  # пропуск строк — разрыв внутри цитаты
        out.append(lines[j])
        total += len(lines[j]) + 1
        prev = j
    return "\n".join(out) or body[:max_chars]


# ---------------------------------------------------------------------------
# Кэш корпуса и BM25-индекса
# ---------------------------------------------------------------------------
# Ключ — (org_id, видимая выборка), значение — (версия индекса, корпус, индекс).
# Версия берётся из storage.index_version() и растёт при любой записи чанков,
# поэтому кэш не может «залипнуть» на устаревших данных, а корпус не
# перечитывается и не ре-токенизируется на каждый вопрос.
_index_cache: dict[tuple, tuple] = {}
_INDEX_CACHE_MAX = 8


def invalidate_index_cache() -> None:
    _index_cache.clear()


async def _corpus_for(session: AsyncSession, org_id: int, user_id: int | None) -> list[dict]:
    """Chunk-корпус для BM25: только актуальные и видимые пользователю
    документы, без дублей по (документ, раздел)."""
    from .storage import index_version

    key = (org_id, user_id)
    version = index_version()
    cached = _index_cache.get(key)
    if cached and cached[0] == version:
        return cached[1]

    rows = (
        await session.scalars(
            sa_select(Chunk)
            .join(Document, Document.id == Chunk.document_id)
            .where(
                Chunk.org_id == org_id,
                Document.superseded_by_id.is_(None),
                visible_documents_condition(user_id),
            )
        )
    ).all()
    corpus: list[dict] = []
    seen: set[tuple[int, str | None]] = set()
    for c in rows:
        dedup_key = (c.document_id, c.section)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        corpus.append({"id": c.id, "document_id": c.document_id, "body": c.body,
                       "page": c.page, "section": c.section})
    if len(_index_cache) >= _INDEX_CACHE_MAX:
        _index_cache.clear()
    _index_cache[key] = (version, corpus, None)
    return corpus


async def _index_for(org_id: int, user_id: int | None, corpus: list[dict]) -> dict:
    """BM25-индекс для корпуса; пересобирается только при смене версии."""
    key = (org_id, user_id)
    cached = _index_cache.get(key)
    if cached and cached[2] is not None and cached[1] is corpus:
        return cached[2]
    index = await asyncio.to_thread(build_index, corpus)
    _index_cache[key] = (cached[0] if cached else 0, corpus, index)
    return index


async def hybrid_search(
    session: AsyncSession, emb: EmbeddingService, org_id: int, query: str, k: int = 6,
    user_id: int | None = None,
) -> list[dict]:
    """Гибридный мультивариантный поиск (multi-query retrieval): запрос
    расширяется аббревиатурами/синонимами, каждый вариант ищется плотно и
    разреженно, все списки сливаются RRF. Приватные документы других
    пользователей исключены (permission-aware)."""
    from .query_expansion import expand_query

    variants = expand_query(query)
    dense: list[list[dict]] = []
    sparse: list[list[dict]] = []
    for variant in variants:
        dense.append(await vector_search(
            session, org_id=org_id, query_vec=await emb.embed_query(variant),
            kind="chunk", k=max(k * 3, 12), user_id=user_id,
        ))

    corpus = await _corpus_for(session, org_id, user_id)
    index = await _index_for(org_id, user_id, corpus)

    def _sparse() -> list[list[dict]]:
        # BM25 по индексу — в потоке: на большом корпусе это заметная
        # CPU-работа, а event loop обслуживает других пользователей
        return [bm25_search_indexed(index, variant, k=max(k * 3, 12)) for variant in variants]

    sparse: list[list[dict]] = await asyncio.to_thread(_sparse)

    fused: list[dict] = rrf_fuse(dense + sparse, top=k)
    dense_scores: dict[int, float] = {}
    for ranked in dense:
        for item in ranked:
            dense_scores[item["id"]] = max(dense_scores.get(item["id"], 0.0), item["score"])
    for row in fused:
        row["semantic_score"] = dense_scores.get(row["id"])
    return fused
