"""Клиенты LLM (генерация через внешний API — единственный компонент Варианта 2,
работающий снаружи). Архитектура допускает смену провайдера без переделки системы.

- openai_compatible: OpenAI / OpenRouter / DeepSeek / YandexGPT (v1) / локальный vLLM/Ollama
- gigachat: Sber GigaChat (OAuth + REST)
- mock: детерминированный offline-режим для тестов и демо (без сети)
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import urlparse

import httpx

from .config import Settings
from .formatting import render_answer

log = logging.getLogger(__name__)

Message = dict[str, str]  # {"role": ..., "content": ...}

# Назначение вызова. По нему выбирается модель (дешёвая для служебных шагов,
# основная только для формулировки ответа) и ведётся учёт расхода токенов.
Task = Literal["classify", "rerank", "sql", "compose", "ocr", "other"]

# задачи, которые не формулируют ответ и потому обслуживаются дешёвой моделью
SMALL_TASKS: frozenset[str] = frozenset({"classify", "rerank", "sql"})

_JSON_SUFFIX = "\n\nОтвечай ТОЛЬКО валидным JSON-объектом, без пояснений и markdown."


@dataclass
class ChatResult:
    """Ответ модели вместе с расходом токенов (usage от провайдера)."""

    text: str
    task: str = "other"
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    # «Мысли» модели (DeepSeek thinking, OpenAI reasoning): входят в
    # completion_tokens, то есть оплачиваются, но в ответе пользователю их нет
    reasoning_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


# Колбэк учёта: вызывается после каждого фактического запроса к API.
UsageHook = Callable[[ChatResult], None]


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


class LLMBalanceError(LLMError):
    """Баланс ключа API исчерпан (HTTP 402), либо запрос отвергнут по квоте.

    Отдельный тип нужен, чтобы отличить «кончились деньги» от сетевого сбоя:
    повторять запрос бессмысленно, а пользователю надо сказать прямо, что
    нужно пополнить счёт, а не отделываться общим «LLM недоступен».
    """


# Предупреждение пользователю в Telegram (HTML, как остальные ответы бота).
BALANCE_NOTICE = (
    "⚠️ <b>На ключе LLM API закончился баланс.</b>\n"
    "Ответ собран шаблоном из ваших данных — цифры верные, но без формулировок "
    "модели. Пополните счёт у провайдера.\n"
    "<i>Накопленный расход и даты сбоев: /usage</i>"
)


# Признаки исчерпанного баланса/квоты в теле ответа. DeepSeek отдаёт
# 402 Insufficient Balance; OpenAI — insufficient_quota; прочие — свои
# формулировки (в т.ч. китайские, у DeepSeek бывает локализованный текст).
_BALANCE_MARKERS = (
    "insufficient balance",
    "insufficient_quota",
    "insufficient quota",
    "exceeded your current quota",
    "not enough balance",
    "no credit",
    "out of credits",
    "credit balance is too low",
    "billing",
    "quota exceeded",
    "余额不足",
    "欠费",
    "配额",
)


def balance_exhausted(status_code: int, body: str) -> bool:
    """Исчерпан ли баланс/квота по ответу провайдера."""
    if status_code == 402:
        return True
    # 400 добавлен ради локализованных ответов (DeepSeek отдаёт «余额不足» с кодом
    # 400), но только при явном маркере денег/квоты: обычная ошибка формата
    # запроса таких слов не содержит. 401/403/429 — квоты и лимиты шлюзов.
    if status_code not in (400, 401, 403, 429):
        return False
    low = (body or "").casefold()
    return any(m in low for m in _BALANCE_MARKERS)


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


def _missing_api_key(settings: Settings) -> bool:
    """Ключ не задан, а endpoint внешний (не локальный)?

    У локальных движков (Ollama/vLLM/LM Studio) ключ не нужен, поэтому пустой
    LLM_API_KEY — ошибка только для внешнего API. Проверяем это заранее: иначе
    httpx падает с невнятным «Illegal header value b'Bearer '», и клиент трижды
    повторяет заведомо нерабочий запрос.
    """
    if settings.llm_provider != "openai_compatible":
        return False
    if (settings.llm_api_key or "").strip():
        return False
    return not _is_local_url(settings.llm_api_base)


def _no_key_hint(settings: Settings) -> str:
    base = settings.llm_api_base or ""
    hint = "локальный движок ключ не требует — поставьте LLM_API_KEY=ollama или EMPTY"
    if "deepseek" in base.lower():
        hint = "ключ DeepSeek: https://platform.deepseek.com/api_keys"
    elif "openai.com" in base.lower():
        hint = "ключ OpenAI: https://platform.openai.com/api-keys"
    elif "openrouter" in base.lower():
        hint = "ключ OpenRouter: https://openrouter.ai/keys"
    return (
        f"LLM_API_KEY не задан, а LLM_API_BASE указывает на внешний API ({base}). "
        f"Впишите ключ в .env ({hint})."
    )


class BaseLLM:
    async def chat(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        json_mode: bool = False,
        max_tokens: int | None = None,
        task: str = "other",
    ) -> str:
        raise NotImplementedError

    async def chat_result(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        json_mode: bool = False,
        max_tokens: int | None = None,
        task: str = "other",
    ) -> ChatResult:
        """Как chat(), но возвращает и расход токенов. Реализации, которые
        usage не разбирают (mock), возвращают нули — учёт просто пропускается."""
        text = await self.chat(
            messages, temperature=temperature, json_mode=json_mode,
            max_tokens=max_tokens, task=task,
        )
        return ChatResult(text=text, task=task)

    async def close(self) -> None:  # noqa: B027
        pass


class OpenAICompatibleLLM(BaseLLM):
    def __init__(self, settings: Settings):
        self.s = settings
        # локальный endpoint (Ollama/vLLM/LM Studio) — без системного прокси
        self._client = httpx.AsyncClient(
            timeout=settings.llm_timeout_s, trust_env=not _is_local_url(settings.llm_api_base)
        )
        self.usage_hook: UsageHook | None = None
        # карта «задача -> модель»: служебные шаги уходят на дешёвую модель
        self._task_models = _task_model_map(settings)
        # Баланс: время последнего 402 и признак «уже сообщили». Пока пауза не
        # истекла, в API не ходим — иначе каждый вопрос пользователя даёт
        # пачку бессмысленных 402 и ошибки в журнале провайдера.
        self._balance_exhausted_at: float | None = None
        self._balance_notified = False
        # Последняя ошибка баланса: по ней пайплайн (у которого нет ссылки на
        # клиента напрямую) узнаёт, что ответ деградировал из-за денег. Нужна
        # потому, что классификатор и Text-to-SQL глотают ошибку и продолжают
        # работу — поймать исключение в одном месте невозможно.
        self._balance_error: LLMBalanceError | None = None

    def model_for(self, task: str) -> str:
        return self._task_models.get(task, self.s.llm_model)

    def take_balance_error(self) -> LLMBalanceError | None:
        """Отдать накопленную ошибку баланса и забыть её (для учёта события)."""
        error, self._balance_error = self._balance_error, None
        return error

    def balance_ok(self) -> bool:
        """Можно ли идти в API: False, если недавно получили 402."""
        if self._balance_exhausted_at is None:
            return True
        pause = max(0.0, float(self.s.llm_balance_retry_s or 0))
        # pause=0 — паузы нет, каждый вопрос снова пробует API (счёт могли пополнить).
        # Проверять `pause and ...` нельзя: 0 ложен, и состояние не сбрасывалось бы.
        if pause == 0 or _now() - self._balance_exhausted_at >= pause:
            log.info("Пауза после исчерпания баланса истекла — пробую API снова")
            self._balance_exhausted_at = None
            return True
        return False

    def take_balance_notice(self) -> bool:
        """True — пользователю ещё не сообщали об исчерпанном балансе."""
        if self._balance_exhausted_at is None or self._balance_notified:
            return False
        self._balance_notified = True
        return True

    async def chat(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        json_mode: bool = False,
        max_tokens: int | None = None,
        task: str = "other",
    ) -> str:
        return (
            await self.chat_result(
                messages, temperature=temperature, json_mode=json_mode,
                max_tokens=max_tokens, task=task,
            )
        ).text

    async def chat_result(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        json_mode: bool = False,
        max_tokens: int | None = None,
        task: str = "other",
    ) -> ChatResult:
        model = self.model_for(task)
        conv: list[Message] = list(messages)
        # не все провайдеры умеют response_format. Если для задачи назначена
        # дешёвая модель, JSON просим словами: лишний 400 и повтор — это
        # потерянные токены и время, а разбор JSON у нас и так есть.
        if json_mode and model != self.s.llm_model:
            json_mode = False
            conv = _ask_json_in_text(conv)
        payload: dict[str, Any] = {
            "model": model,
            "messages": conv,
            "temperature": self.s.llm_temperature if temperature is None else temperature,
            "max_tokens": max_tokens or self.s.llm_max_tokens,
        }
        # Режим рассуждений: по умолчанию (DeepSeek) он включён на высоком усилии —
        # это скрытые токены, которые оплачиваются и съедают max_tokens.
        payload.update(_thinking_payload(self.s, task))
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        resp = await self._post_chat(payload)
        data = resp.json()
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError) as e:
            raise LLMError(f"Неожиданный ответ LLM API: {str(data)[:300]}") from e
        usage = _usage_from(data.get("usage"))
        # Пустой content при непустых «мыслях» означает, что рассуждения съели
        # весь лимит токенов (типично для моделей с thinking по умолчанию).
        # Ответа нет — это не «модель промолчала», а потерянные деньги.
        if not text.strip() and usage.get("reasoning_tokens"):
            if "thinking" in payload:
                retry: dict[str, Any] = dict(payload)
                retry.pop("thinking", None)
                retry.pop("reasoning_effort", None)
                resp = await self._post_chat(retry)
                data = resp.json()
                try:
                    text = data["choices"][0]["message"]["content"] or ""
                except (KeyError, IndexError) as e:
                    raise LLMError(f"Неожиданный ответ LLM API: {str(data)[:300]}") from e
                usage = _usage_from(data.get("usage"))
            if not text.strip():
                raise LLMError(
                    "Модель израсходовала весь лимит токенов на рассуждения и вернула "
                    f"пустой ответ ({usage.get('reasoning_tokens')} ток. мыслей). "
                    "Выключите режим рассуждений (LLM_THINKING=disabled) или "
                    "увеличьте лимит токенов для этого шага."
                )
        result = ChatResult(text=text, task=task, model=model, **usage)
        if self.usage_hook is not None:
            try:
                self.usage_hook(result)
            except Exception as e:  # учёт не должен ломать ответ
                log.warning("Учёт расхода токенов не удался: %s", e)
        return result

    async def _post_chat(self, payload: dict[str, Any]) -> httpx.Response:
        """POST /chat/completions с самокоррекцией и повторами.

        Совместимость: часть провайдеров/моделей не принимает response_format,
        max_tokens (новые модели OpenAI ждут max_completion_tokens) или
        нестандартную temperature. Повторы: 429/5xx/сеть — иначе один
        рейт-лимит отбрасывал весь вопрос на шаблонный ответ.
        """
        url = f"{self.s.llm_api_base.rstrip('/')}/chat/completions"
        if _missing_api_key(self.s):
            # без ключа запрос заведомо не пройдёт: не тратим повторы и не
            # показываем пользователю «Illegal header value»
            raise LLMError(_no_key_hint(self.s))
        headers = {"Authorization": f"Bearer {self.s.llm_api_key}"}
        body = dict(payload)
        attempts = max(1, self.s.llm_retries + 1)
        for attempt in range(attempts):
            if attempt:
                await asyncio.sleep(_backoff(self.s.llm_retry_base_s, attempt))
            try:
                resp = await self._client.post(url, headers=headers, json=body)
            except httpx.HTTPError as e:
                if attempt + 1 < attempts:
                    log.warning("LLM API недоступен (%s) — повтор %d/%d", e, attempt + 1, attempts - 1)
                    continue
                raise LLMError(f"LLM API недоступен: {e}") from e

            if resp.status_code < 400:
                self._balance_exhausted_at = None
                self._balance_notified = False
                return resp

            text = resp.text[:300]
            # Баланс/квота: повторять бессмысленно — деньги не появятся от
            # ожидания, а каждый повтор пишет ошибку в журнал провайдера
            if balance_exhausted(resp.status_code, resp.text):
                self._balance_exhausted_at = _now()
                log.error(
                    "LLM API: исчерпан баланс (HTTP %d) — до пополнения счёта "
                    "ответы собираются шаблоном без вызовов API. Ответ провайдера: %s",
                    resp.status_code, text[:200],
                )
                error = LLMBalanceError(
                    f"Баланс LLM API исчерпан (HTTP {resp.status_code}): {text[:200]}"
                )
                self._balance_error = error
                raise error
            if resp.status_code == 400:
                if body.pop("response_format", None) is not None:
                    log.info("Провайдер не принял response_format — повтор без него")
                    continue
                if body.get("max_tokens") is not None and "max_tokens" in text.lower():
                    # новые модели OpenAI: max_tokens -> max_completion_tokens
                    body["max_completion_tokens"] = body.pop("max_tokens")
                    log.info("Модель требует max_completion_tokens — повтор с ним")
                    continue
                if (
                    body.get("max_completion_tokens") is not None
                    and "max_completion_tokens" in text.lower()
                ):
                    body["max_tokens"] = body.pop("max_completion_tokens")
                    log.info("Модель требует max_tokens — повтор с ним")
                    continue
                if "temperature" in text.lower() and "temperature" in body:
                    body.pop("temperature", None)
                    log.info("Модель не принимает temperature — повтор без него")
                    continue
            if (resp.status_code == 429 or resp.status_code >= 500) and attempt + 1 < attempts:
                log.warning(
                    "LLM API %d — повтор %d/%d через %.1f с",
                    resp.status_code, attempt + 1, attempts - 1,
                    _retry_after(resp) or _backoff(self.s.llm_retry_base_s, attempt + 1),
                )
                wait = _retry_after(resp)
                if wait:
                    await asyncio.sleep(min(wait, self.s.llm_retry_max_s))
                continue
            raise LLMError(f"LLM API {resp.status_code}: {text}")
        raise LLMError("LLM API: исчерпаны попытки запроса")

    async def close(self) -> None:
        await self._client.aclose()


def _task_model_map(settings: Settings) -> dict[str, str]:
    """Задача -> модель. Пустая карта означает «одна модель на всё» (как было)."""
    small = (settings.llm_model_small or "").strip()
    if not small:
        return {}
    tasks = {t.strip() for t in (settings.llm_model_small_tasks or "").split(",") if t.strip()}
    return {t: small for t in (tasks or SMALL_TASKS) if t != "compose"}


def _ask_json_in_text(messages: list[Message]) -> list[Message]:
    """Просьба вернуть JSON словами — для моделей без response_format."""
    if not messages:
        return messages
    out = list(messages)
    last = out[-1]
    if "JSON" not in (last.get("content") or ""):
        out[-1] = {**last, "content": (last.get("content") or "") + _JSON_SUFFIX}
    return out


def _usage_from(raw: Any) -> dict[str, int]:
    """usage из ответа провайдера (у разных провайдеров имена чуть разные)."""
    if not isinstance(raw, dict):
        return {}
    details = raw.get("prompt_tokens_details")
    cached = 0
    if isinstance(details, dict):
        cached = int(details.get("cached_tokens") or 0)
    out_details = raw.get("completion_tokens_details")
    reasoning = 0
    if isinstance(out_details, dict):
        reasoning = int(out_details.get("reasoning_tokens") or 0)
    return {
        "prompt_tokens": int(raw.get("prompt_tokens") or 0),
        "completion_tokens": int(raw.get("completion_tokens") or 0),
        "cached_tokens": cached or int(raw.get("prompt_cache_hit_tokens") or 0),
        "reasoning_tokens": reasoning,
    }


def _thinking_payload(settings: Settings, task: str = "other") -> dict[str, Any]:
    """Параметры режима рассуждений для OpenAI-совместимого API.

    DeepSeek: {"thinking": {"type": "disabled"}} + reasoning_effort. Прочие
    провайдеры этих полей не знают, поэтому по умолчанию не отправляем ничего.

    Если задан LLM_THINKING_TASKS, рассуждения включаются только для этих задач:
    составление ответа от них выигрывает, а классификация/реранкинг/SQL — нет
    (там нужно переформатировать данные в JSON, и «мысли» дороже ответа).
    """
    mode = (settings.llm_thinking or "").strip().lower()
    tasks = settings.thinking_tasks
    if tasks and task not in tasks:
        # для задач не из списка режим выключен явно, а не «по умолчанию»
        return {"thinking": {"type": "disabled"}}
    if mode in ("", "default", "auto"):
        return {}
    if mode in ("disabled", "off", "none"):
        return {"thinking": {"type": "disabled"}}
    # low/high/max — включаем рассуждения с заданным усилием
    return {"thinking": {"type": "enabled"}, "reasoning_effort": mode}


def _retry_after(resp: httpx.Response) -> float | None:
    raw = resp.headers.get("retry-after")
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        return None


def _backoff(base: float, attempt: int) -> float:
    return min(base * (2 ** max(0, attempt - 1)), 30.0)


def _now() -> float:
    """Монотонные секунды процесса — для отсчёта паузы после 402."""
    import time

    return time.monotonic()


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
        task: str = "other",
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
            return await self.chat(messages, temperature=temperature, json_mode=json_mode,
                                   max_tokens=max_tokens, task=task)
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

    def __init__(self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None):
        self.s = settings
        self._client = httpx.AsyncClient(timeout=settings.llm_timeout_s, transport=transport)

    @staticmethod
    def _build_payload(settings: Settings, messages: list[Message], *, temperature: float, max_tokens: int) -> dict:
        system_texts = [m["content"] for m in messages if m["role"] == "system"]
        # content может быть как строкой, так и списком блоков (prompt caching) —
        # поэтому dict[str, Any], а не выведенный dict[str, str]
        conv: list[dict[str, Any]] = [
            {"role": m["role"], "content": m["content"]} for m in messages if m["role"] != "system"
        ]
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
        task: str = "other",
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

    async def chat_result(self, messages, **kwargs) -> ChatResult:
        from .security import anonymize_text

        masked = []
        for m in messages:
            content = m.get("content")
            if isinstance(content, str):
                content = anonymize_text(content, self._org_names)
            masked.append({**m, "content": content})
        return await self._inner.chat_result(masked, **kwargs)

    def model_for(self, task: str) -> str:
        """Модель, которой уйдёт вызов задачи (для учёта и логов)."""
        inner = getattr(self._inner, "model_for", None)
        if callable(inner):
            return str(inner(task))
        s = getattr(self._inner, "s", None)
        return str(getattr(s, "llm_model", "") or "")

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
        task: str = "other",
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
