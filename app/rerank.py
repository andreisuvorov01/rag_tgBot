"""Reranking — перез ранжирование результатов гибридного поиска (практика
Onyx и Anthropic: reranker как финальный шаг после RRF даёт наибольший
дополнительный прирост точности).

Варианты:
- llm — внешний LLM оценивает релевантность каждого фрагмента запросу (0..2);
  данные фрагментов и так уходят в промпт композитора, отдельной утечки нет;
- crossencoder — локальный кросс-энкодер (sentence-transformers), ничего не покидает сервер;
- none — без перез ранжирования (порядок RRF).
"""
from __future__ import annotations

import asyncio
import json
import logging

from .config import Settings
from .llm import BaseLLM, MockLLM, parse_json_block

log = logging.getLogger(__name__)

RERANK_PROMPT = """[TASK=rerank]
Ты оцениваешь релевантность фрагментов финансовых документов запросу пользователя.
Для каждого фрагмента верни оценку релевантности: 2 — прямо отвечает на запрос,
1 — частично полезен (та же тема), 0 — нерелевантен.
Отвечай ТОЛЬКО JSON: {"scores": [{"id": <id>, "score": <0|1|2>}, ...]} — по всем фрагментам.
"""


class BaseReranker:
    async def rerank(self, query: str, chunks: list[dict], k: int) -> list[dict]:
        return chunks[:k]


class LLMReranker(BaseReranker):
    """Реранкинг моделью. Самая «токеноголодная» ступень конвейера: в промпт
    уходят фрагменты документов. Поэтому объём ограничен настройками
    (RERANK_MAX_ITEMS, RERANK_SNIPPET_CHARS), а вызов помечен задачей
    rerank — он обслуживается дешёвой моделью (LLM_MODEL_SMALL)."""

    def __init__(self, llm: BaseLLM, settings: Settings | None = None):
        self.llm = llm
        self.s = settings
        self.max_items = settings.rerank_max_items if settings else 8
        self.snippet_chars = settings.rerank_snippet_chars if settings else 200
        self.max_tokens = settings.rerank_max_tokens if settings else 600

    async def rerank(self, query: str, chunks: list[dict], k: int) -> list[dict]:
        if len(chunks) <= 1 or isinstance(self.llm, MockLLM):
            return chunks[:k]
        items = [
            {"id": c["id"], "text": (c.get("body") or "")[: self.snippet_chars]}
            for c in chunks[: self.max_items]
        ]
        # JSON-режим просим только у основной модели: у дешёвых моделей
        # response_format поддерживается не всегда, а лишний отказ — это
        # потраченные впустую токены (промпт уже требует «ТОЛЬКО JSON»)
        main_model = self._uses_main_model()
        try:
            raw = await self.llm.chat(
                [
                    {"role": "system", "content": RERANK_PROMPT},
                    {"role": "user", "content": json.dumps({"query": query, "fragments": items}, ensure_ascii=False)},
                ],
                json_mode=main_model,
                temperature=0.0,
                max_tokens=self.max_tokens,
                task="rerank",
            )
            parsed = parse_json_block(raw) or {}
            by_id: dict[int, float] = {}
            for s in parsed.get("scores", []):
                try:
                    by_id[int(s["id"])] = max(0.0, min(2.0, float(s["score"])))
                except (KeyError, TypeError, ValueError):
                    continue
            if by_id:
                chunks = sorted(chunks, key=lambda c: by_id.get(c["id"], 0.0), reverse=True)
                for c in chunks:
                    c["rerank_score"] = by_id.get(c["id"])
        except Exception as e:
            # сбой реранкера не должен ломать ответ — остаётся порядок RRF
            log.warning("LLM-reranker недоступен (%s) — порядок RRF сохранён", e)
        return chunks[:k]

    def _uses_main_model(self) -> bool:
        """Пойдёт ли вызов на основную (дорогую) модель."""
        if self.s is None:
            return True
        small = (self.s.llm_model_small or "").strip()
        if not small or "rerank" not in self.s.small_model_tasks:
            return True
        return small == self.s.llm_model


class CrossEncoderReranker(BaseReranker):
    """Локальный кросс-энкодер: пары (запрос, документ) скорятся нейросетью.
    Полностью локально; требует sentence-transformers и скачивания модели."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            log.info("Загрузка кросс-энкодера %s ...", self.model_name)
            self._model = CrossEncoder(self.model_name, max_length=512)
        return self._model

    def _score_sync(self, query: str, bodies: list[str]) -> list[float]:
        model = self._load()
        pairs = [(query, b) for b in bodies]
        scores = model.predict(pairs)
        return [float(s) for s in scores]

    async def rerank(self, query: str, chunks: list[dict], k: int) -> list[dict]:
        if len(chunks) <= 1:
            return chunks[:k]
        try:
            scores = await asyncio.to_thread(self._score_sync, query, [c["body"] for c in chunks])
        except Exception as e:
            log.warning("Кросс-энкодер недоступен (%s) — порядок RRF сохранён", e)
            return chunks[:k]
        for c, s in zip(chunks, scores, strict=True):
            c["rerank_score"] = s
        chunks = sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)
        return chunks[:k]


class HeuristicReranker(BaseReranker):
    """Офлайн-реранкер без API: комбинирует признаки — перекрытие токенов
    запроса и фрагмента, точная фраза, семантическая близость. Работает
    мгновенно и детерминированно; включается автоматически, когда LLM API
    недоступен (провайдер mock)."""

    def __init__(self, settings: Settings):
        self.s = settings

    async def rerank(self, query: str, chunks: list[dict], k: int) -> list[dict]:
        from .embeddings import _tokens

        qtoks = set(_tokens(query))
        qphrase = query.casefold().strip()
        scored = []
        for c in chunks:
            ctoks = set(_tokens(c["body"]))
            overlap = len(qtoks & ctoks) / max(len(qtoks), 1)
            phrase = 1.0 if qphrase and qphrase in c["body"].casefold() else 0.0
            sem = min((c.get("semantic_score") or 0.0) * 2.0, 1.0)
            # карточка показателя полезна, только если вопрос про этот показатель
            card = 0.15 if overlap > 0 and (c.get("section") or "").startswith("карточка:") else 0.0
            score = 0.5 * overlap + 0.2 * sem + 0.2 * phrase + card
            if len(c["body"]) < 100:
                score *= 0.7  # обрывок шапки таблицы редко содержит ответ
            c = dict(c)
            c["rerank_score"] = score
            scored.append((score, c))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [c for _, c in scored[:k]]


def make_reranker(settings: Settings, llm: BaseLLM) -> BaseReranker:
    if settings.reranker == "crossencoder":
        return CrossEncoderReranker(settings.rerank_model)
    if settings.reranker == "llm":
        return LLMReranker(llm, settings)
    if settings.reranker == "heuristic":
        return HeuristicReranker(settings)
    if settings.reranker == "auto":
        # живой поиск без внешнего API — эвристики; с API — модель оценивает
        return HeuristicReranker(settings) if settings.llm_provider == "mock" else LLMReranker(llm, settings)
    return BaseReranker()
