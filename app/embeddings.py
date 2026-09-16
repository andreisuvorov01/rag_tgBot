"""Сервис эмбеддингов. Выполняется ЛОКАЛЬНО (вариант 2): документы и запросы
не покидают сервер на этапе векторизации.

Провайдеры:
- sentence_transformers — локальная модель (multilingual-e5-large / bge-m3) на CPU;
- api — OpenAI-совместимый endpoint (например, локальный vLLM с bge-m3);
- hash — детерминированное feature-hashing для dev/тестов без нейросети.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from typing import Any, Literal

import httpx
import numpy as np

from .config import Settings

log = logging.getLogger(__name__)

Tokenizer = Literal["word"]

_WORD_RE = re.compile(r"[\w]+", re.UNICODE)


# служебные слова вопроса: без них «что говорится об аудите» ищет «аудит», а не «об»
STOPWORDS = frozenset(["и", "в", "во", "на", "по", "за", "от", "до", "из", "у", "о", "об", "обо", "с", "со", "к", "ко", "для", "при", "про", "без", "над", "под", "между", "через", "а", "но", "или", "же", "ли", "бы", "не", "ни", "да", "нет", "то", "это", "эта", "этот", "эти", "тот", "та", "те", "там", "тут", "здесь", "что", "чем", "чему", "кто", "кого", "кому", "как", "какой", "какая", "какое", "какие", "каков", "какова", "где", "когда", "куда", "откуда", "сколько", "почему", "зачем", "который", "которая", "которые", "чей", "чья", "чьи", "был", "была", "было", "были", "быть", "есть", "будет", "будут", "является", "являются", "мне", "мой", "моя", "мои", "наш", "наша", "наше", "наши", "ваш", "ваша", "ваши", "его", "её", "их", "ему", "ей", "им", "них", "покажи", "скажи", "расскажи", "объясни", "говорится", "сказано", "указано", "написано", "документ", "документе", "документа", "отчёт", "отчет", "отчёте", "отчете", "файл", "файле", "компания", "компании", "организация", "организации"])


def _tokens(text: str) -> list[str]:
    # префиксный «стемминг» (5 символов) сглаживает русскую морфологию:
    # «аренде»/«аренда» дают один и тот же признак
    return [
        t.casefold()[:5] for t in _WORD_RE.findall(text)
        if len(t) > 1 and t.casefold() not in STOPWORDS
    ]


class EmbeddingService:
    def __init__(self, settings: Settings):
        self.s = settings
        self.provider: str = settings.effective_embeddings_provider
        self.model: str = settings.effective_embeddings_model
        self._st_model = None
        self._st_lock = asyncio.Lock()
        # Маскирование для внешнего embeddings-API: AnonymizingLLM закрывает
        # только чат-вызовы, а векторизация внешним endpoint'ом раньше уходила
        # наружу сырым текстом. Заполняется в main.build_pipeline().
        self.anonymize_org_names: list[str] = []

    # ------------------------------------------------------------------
    def _embed_hash(self, text: str) -> list[float]:
        dim = self.s.embeddings_dim
        vec = np.zeros(dim, dtype=np.float32)
        counts: dict[str, int] = {}
        for tok in _tokens(text):
            counts[tok] = counts.get(tok, 0) + 1
        for tok, tf in counts.items():
            h = hashlib.md5(tok.encode("utf-8")).digest()
            idx = int.from_bytes(h[:4], "little") % dim
            sign = 1.0 if h[4] % 2 == 0 else -1.0
            vec[idx] += sign * (1.0 + np.log(tf))
        norm = float(np.linalg.norm(vec))
        return (vec / norm).tolist() if norm > 0 else vec.tolist()

    async def _get_st_model(self):
        """Ленивая загрузка модели под блокировкой: прогрев и первый вопрос
        не запускают две загрузки параллельно."""
        async with self._st_lock:
            if self._st_model is None:
                loaded: Any = await asyncio.to_thread(self._load_st_sync)
                self._st_model = loaded
        return self._st_model

    def _load_st_sync(self) -> Any:
        """GPU (CUDA) в fp16, если доступен: bge-m3 на GTX 1050 Ti считает
        в ~10 раз быстрее CPU и занимает ~1,2 ГБ видеопамяти; иначе CPU.

        Возвращает SentenceTransformer; аннотация Any, потому что
        sentence-transformers — опциональная зависимость и импортируется
        внутри метода.
        """
        import torch
        from sentence_transformers import SentenceTransformer

        device = self.s.embeddings_device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda"):
            try:
                model = SentenceTransformer(self.model, device=device, model_kwargs={"torch_dtype": torch.float16})
                model.encode(["прогрев"], show_progress_bar=False)  # OOM/несовместимость ловим сразу
                log.info("Embedding-модель %s загружена на %s (fp16)", self.model, torch.cuda.get_device_name(0))
                return model
            except Exception as e:
                log.warning("GPU недоступен для эмбеддингов (%s) — работаю на CPU", e)
        log.info("Загрузка локальной embedding-модели %s на CPU ...", self.model)
        return SentenceTransformer(self.model, device="cpu")

    def _encode_sync(self, model, texts: list[str], is_query: bool) -> list[list[float]]:
        model_name = (self.model or "").lower()
        if "e5" in model_name:
            prefix = "query: " if is_query else "passage: "
            texts = [prefix + t for t in texts]
        emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [e.tolist() for e in np.asarray(emb)]

    async def embed_texts(self, texts: list[str], *, is_query: bool = False) -> list[list[float]]:
        if not texts:
            return []
        if self.provider == "hash":
            return [self._embed_hash(t) for t in texts]
        if self.provider == "sentence_transformers":
            model = await self._get_st_model()
            return await asyncio.to_thread(self._encode_sync, model, texts, is_query)
        if self.provider == "api":
            base = self.s.embeddings_api_base.rstrip("/")
            # локальный endpoint (vLLM/Ollama) — без системного прокси
            from urllib.parse import urlparse

            host = (urlparse(base).hostname or "").lower()
            local = host in ("localhost", "127.0.0.1", "::1", "0.0.0.0")
            if not local and self.s.anonymize_prompts:
                # внешний эмбеддинг-endpoint: маскируем ПД так же, как в чат-вызовах,
                # иначе сырые тексты документов уходят наружу без защиты
                from .security import anonymize_text

                texts = [anonymize_text(t, self.anonymize_org_names) for t in texts]
            async with httpx.AsyncClient(timeout=120, trust_env=not local) as client:
                resp = await client.post(
                    f"{base}/embeddings",
                    headers={"Authorization": f"Bearer {self.s.embeddings_api_key}"} if self.s.embeddings_api_key else {},
                    json={"model": self.s.embeddings_api_model, "input": texts},
                )
                resp.raise_for_status()
                data = resp.json()["data"]
                return [d["embedding"] for d in data]
        raise ValueError(f"Неизвестный embeddings_provider: {self.provider}")

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed_texts([text], is_query=True))[0]

    async def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return await self.embed_texts(texts, is_query=False)


def cosine(a: list[float], b: list[float]) -> float:
    va, vb = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb)) or 1.0
    return float(np.dot(va, vb) / denom)


def embeddings_fingerprint(settings: Settings) -> str:
    """Отпечаток конфигурации эмбеддингов: векторы разных провайдеров/моделей
    несовместимы — при смене требуется переиндексация (scripts/reindex.py).
    Учитывает фактический (effective) провайдер."""
    provider = settings.effective_embeddings_provider
    model = {
        "hash": "",
        "sentence_transformers": settings.effective_embeddings_model,
        "api": settings.embeddings_api_model,
    }.get(provider, "")
    return f"{provider}:{model}:{settings.embeddings_dim}"
