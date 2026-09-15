"""Клиенты LLM (генерация через внешний API — единственный компонент Варианта 2,
работающий снаружи). Архитектура допускает смену провайдера без переделки системы.

- openai_compatible: OpenAI / OpenRouter / DeepSeek / YandexGPT (v1) / локальный vLLM/Ollama
- gigachat: Sber GigaChat (OAuth + REST)
- mock: детерминированный offline-режим для тестов и демо (без сети)
"""
from __future__ import annotations

import base64
import json
import logging
import re
import uuid
from typing import Any
from urllib.parse import urlparse

import httpx

from .config import Settings
from .formatting import render_answer

log = logging.getLogger(__name__)

Message = dict[str, str]  # {"role": ..., "content": ...}


def _is_local_url(url: str) -> bool:
    """localhost/127.0.0.1/::1 — запросы туда не должны идти через системный
    прокси (Windows: httpx с trust_env читает прокси из реестра, и локальный
    Ollama/LM Studio через него недостижим)."""
    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return False
    return host in ("localhost", "127.0.0.1", "::1", "0.0.0.0")


class LLMError(RuntimeError):
    pass


def parse_json_block(text: str) -> dict[str, Any] | None:
    """Достаёт JSON из ответа модели (в т.ч. из ```json ...```)."""
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = m.group(1) if m else None
    if candidate is None:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            candidate = text[start : end + 1]
    if candidate is None:
        return None
    try:
        obj = json.loads(candidate)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


class BaseLLM:
    async def chat(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        json_mode: bool = False,
        max_tokens: int | None = None,
    ) -> str:
        raise NotImplementedError

    async def close(self) -> None:  # noqa: B027
        pass


class OpenAICompatibleLLM(BaseLLM):
    def __init__(self, settings: Settings):
        self.s = settings
        # локальный endpoint (Ollama/vLLM/LM Studio) — без системного прокси
        self._client = httpx.AsyncClient(
            timeout=settings.llm_timeout_s, trust_env=not _is_local_url(settings.llm_api_base)
        )

    async def chat(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        json_mode: bool = False,
        max_tokens: int | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.s.llm_model,
            "messages": messages,
            "temperature": self.s.llm_temperature if temperature is None else temperature,
            "max_tokens": max_tokens or self.s.llm_max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        try:
            resp = await self._client.post(
                f"{self.s.llm_api_base.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {self.s.llm_api_key}"},
                json=payload,
            )
        except httpx.HTTPError as e:
            raise LLMError(f"LLM API недоступен: {e}") from e
        if resp.status_code >= 400 and json_mode:
            # часть провайдеров не поддерживает response_format — повтор без него
            payload.pop("response_format", None)
            return await self.chat(messages, temperature=temperature, json_mode=False, max_tokens=max_tokens)
        if resp.status_code >= 400:
            raise LLMError(f"LLM API {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError) as e:
            raise LLMError(f"Неожиданный ответ LLM API: {str(data)[:300]}") from e

    async def close(self) -> None:
        await self._client.aclose()


class GigaChatLLM(BaseLLM):
    def __init__(self, settings: Settings):
        self.s = settings
        self._client = httpx.AsyncClient(
            timeout=settings.llm_timeout_s, verify=not settings.gigachat_insecure
        )
        self._token: str | None = None

    async def _ensure_token(self) -> str:
        if self._token:
            return self._token
        auth = base64.b64encode(
            f"{self.s.gigachat_client_id}:{self.s.gigachat_client_secret}".encode()
        ).decode()
        resp = await self._client.post(
            self.s.gigachat_oauth_url,
            headers={
                "Authorization": f"Basic {auth}",
                "RqUID": str(uuid.uuid4()),
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            json={"scope": self.s.gigachat_scope},
        )
        if resp.status_code >= 400:
            raise LLMError(f"GigaChat OAuth {resp.status_code}: {resp.text[:300]}")
        self._token = resp.json()["access_token"]
        return self._token

    async def chat(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        json_mode: bool = False,
        max_tokens: int | None = None,
    ) -> str:
        token = await self._ensure_token()
        payload: dict[str, Any] = {
            "model": self.s.llm_model or "GigaChat",
            "messages": messages,
            "temperature": self.s.llm_temperature if temperature is None else temperature,
            "max_tokens": max_tokens or self.s.llm_max_tokens,
        }
        resp = await self._client.post(
            f"{self.s.gigachat_api_base.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {token}"},
            json=payload,
        )
        if resp.status_code == 401:  # токен истёк
            self._token = None
            return await self.chat(messages, temperature=temperature, json_mode=json_mode, max_tokens=max_tokens)
        if resp.status_code >= 400:
            raise LLMError(f"GigaChat {resp.status_code}: {resp.text[:300]}")
        try:
            return resp.json()["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError) as e:
            raise LLMError(f"Неожиданный ответ GigaChat: {resp.text[:300]}") from e

    async def close(self) -> None:
        await self._client.aclose()


class AnthropicLLM(BaseLLM):
    """Нативный протокол Anthropic Messages API с prompt caching:
    статический системный промпт и блок ДАННЫХ помечаются cache_control —
    кэш переиспользуется при повторах верификатора и одинаковых запросах.
    (OpenAI/DeepSeek кэшируют автоматически — им префикс-кэширование не нужно.)"""

    def __init__(self, settings: Settings, transport: httpx.BaseTransport | None = None):
        self.s = settings
        self._client = httpx.AsyncClient(timeout=settings.llm_timeout_s, transport=transport)

    @staticmethod
    def _build_payload(settings: Settings, messages: list[Message], *, temperature: float, max_tokens: int) -> dict:
        system_texts = [m["content"] for m in messages if m["role"] == "system"]
        conv = [{"role": m["role"], "content": m["content"]} for m in messages if m["role"] != "system"]
        system_block: dict = {"type": "text", "text": "\n\n".join(system_texts)}
        if settings.prompt_cache:
            system_block["cache_control"] = {"type": "ephemeral"}
            if conv:
                last = conv[-1]
                last["content"] = [{"type": "text", "text": last["content"], "cache_control": {"type": "ephemeral"}}]
        return {
            "model": settings.llm_model or "claude-sonnet-4-5",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": [system_block],
            "messages": conv,
        }

    async def chat(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        json_mode: bool = False,
        max_tokens: int | None = None,
    ) -> str:
        if json_mode:
            messages = list(messages)
            messages[-1] = {**messages[-1], "content": messages[-1]["content"] + "\n\nОтвечай ТОЛЬКО валидным JSON-объектом без пояснений."}
        payload = self._build_payload(
            self.s, messages,
            temperature=self.s.llm_temperature if temperature is None else temperature,
            max_tokens=max_tokens or self.s.llm_max_tokens,
        )
        try:
            resp = await self._client.post(
                f"{self.s.anthropic_api_base.rstrip('/')}/v1/messages",
                headers={
                    "x-api-key": self.s.anthropic_api_key,
                    "anthropic-version": self.s.anthropic_version,
                },
                json=payload,
            )
        except httpx.HTTPError as e:
            raise LLMError(f"Anthropic API недоступен: {e}") from e
        if resp.status_code >= 400:
            raise LLMError(f"Anthropic API {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")

    async def close(self) -> None:
        await self._client.aclose()


class AnonymizingLLM(BaseLLM):
    """Обёртка, маскирующая персональные данные во ВСЕХ исходящих промптах.

    Раньше anonymize_text вызывался только в композере, а классификатор,
    Text-to-SQL и LLM-реранкер (он отправляет сырые фрагменты документов)
    уходили наружу без маскирования. Единая точка входа убирает этот класс
    ошибок: добавить новый вызов LLM мимо маскирования больше нельзя.
    """

    def __init__(self, inner: BaseLLM, org_names: list[str] | None = None):
        self._inner = inner
        self._org_names = org_names or []

    async def chat(self, messages, **kwargs) -> str:
        from .security import anonymize_text

        masked = []
        for m in messages:
            content = m.get("content")
            if isinstance(content, str):
                content = anonymize_text(content, self._org_names)
            masked.append({**m, "content": content})
        return await self._inner.chat(masked, **kwargs)

    async def close(self) -> None:
        await self._inner.close()


class MockLLM(BaseLLM):
    """Offline-режим: маршрутизация по ключевым словам, ответы собираются
    шаблонизатором из тех же JSON-данных, что уходят в промпт настоящей LLM."""

    async def chat(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        json_mode: bool = False,
        max_tokens: int | None = None,
    ) -> str:
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        user = "\n".join(m["content"] for m in messages if m["role"] == "user")
        if "[TASK=classify]" in system:
            return json.dumps(_mock_classify(user), ensure_ascii=False)
        if "[TASK=sql]" in system:
            return "-- mock: SQL не сгенерирован"
        if "[TASK=compose]" in system:
            payload = parse_json_block(user) or {}
            return render_answer(payload)
        return "Готово."


def _mock_classify(query: str) -> dict[str, Any]:
    q = query.casefold()
    q = re.sub(r"[\u0301]", "", q)
    scenario = _mock_scenario(query)
    years = [int(y) for y in re.findall(r"20\d\d", query)]
    metric_query = re.sub(r"\b20\d\d\b", "", query)
    _stop = (
        r"прогноз\w*|предскаж\w*|ожидает\w*|перспектив\w*|почему|из-за чего|сравни|"
        r"по сравнению|каки\w+\s+позици\w+|сколько|какая|какой|какие|каков|во сколько|"
        r"\bбыл\w?\b|\bбыли\b|\bесть\b|\bу\b|компани\w*|организаци\w*|"
        r"на сколько|покажи|что\s+говор\w+|объясни|расскажи|год\w*|факт\w*|"
        r"план\w*|отлич\w*|"
        r"\bпо\b|\bна\b|\bза\b|\bесли\b|\bтемп\w*|\bупад\w*|\bвдвое\b|\bв\b|\bраз\b"
    )
    metric_query = re.sub(_stop, " ", metric_query, flags=re.IGNORECASE)
    metric_query = re.sub(r"[?.!,;:]+", " ", metric_query)
    metric_query = re.sub(r"\s+", " ", metric_query).strip() or query

    if re.search(r"прогноз|предскаж|ожидает|перспектив", q):
        intent = "forecast"
    elif re.search(
        r"почему|из-за чего|что говорит|объясни|расскажи|риск|упомина|в чем причина|"
        # реквизиты, люди, текст документа — ответ ищется во фрагментах, а не в SQL
        r"\bкто\b|\bгде\b|\bкогда\b|как называ|автор|подписа|руководител|директор|аудит|"
        r"\bинн\b|\bкпп\b|\bогрн\b|\bокпо\b|\bбик\b|адрес|местонахожд|наименован|"
        r"единиц\w* измерен|дата|номер|о чём|о чем|про что|содерж",
        q,
    ):
        intent = "explain"
    elif re.search(r"из чего состоит|структур|разбив|детализ|расшифров|состав показателя", q):
        intent = "breakdown"
    elif re.search(r"по сравнению|сравни|изменил|во сколько|на ?сколько|отлич\w*|рост|снижени", q):
        intent = "compare"
    elif re.search(r"каки\w+\s+позици|сильнее всего|больше всего|ранжир|топ-?\d", q):
        intent = "rank"
    elif re.search(r"\d|сколько|какая|какой|каков|покажи", q):
        intent = "factual"
    else:
        intent = "smalltalk"
    return {
        "intent": intent,
        "metric_query": metric_query,
        "years": years,
        "target_year": max(years) if years and intent == "forecast" else None,
        "scenario": scenario,
    }


def _mock_scenario(query: str) -> dict[str, Any] | None:
    q = query.casefold()
    decline = bool(re.search(r"упад|сниз|замедл|сократ|меньше", q))
    boost = bool(re.search(r"вырас|выраст|ускор|увелич|больше", q))
    if not (decline or boost):
        return None
    word_factor = {"вдвое": 2.0, "в два раза": 2.0, "втрое": 3.0, "в три раза": 3.0, "вчетверо": 4.0}
    factor = None
    for phrase, f in word_factor.items():
        if phrase in q:
            factor = f
            break
    if factor is None:
        m = re.search(r"в\s+(\d+([.,]\d+)?)\s*раз", q)
        if m:
            factor = float(m.group(1).replace(",", "."))
    if factor is not None:
        return {"growth_multiplier": 1.0 / factor if decline else factor}
    m = re.search(r"на\s+(\d+([.,]\d+)?)\s*%", q)
    if m:
        pct = float(m.group(1).replace(",", ".")) / 100
        return {"growth_multiplier": 1.0 - pct if decline else 1.0 + pct}
    return {"growth_multiplier": 0.5 if decline else 1.5}


def make_llm(settings: Settings) -> BaseLLM:
    provider = settings.llm_provider
    if provider == "openai_compatible":
        return OpenAICompatibleLLM(settings)
    if provider == "gigachat":
        return GigaChatLLM(settings)
    if provider == "anthropic":
        return AnthropicLLM(settings)
    if provider == "mock":
        return MockLLM()
    raise ValueError(f"Неизвестный llm_provider: {provider}")
