"""Экономия токенов и устойчивость внешнего LLM API.

Проверяем то, за что платит пользователь: расход считается, служебные шаги
уходят на дешёвую модель, повторный вопрос не оплачивается дважды, а рейт-лимит
не отбрасывает ответ на шаблон.
"""
from __future__ import annotations

import asyncio
import json
from collections import OrderedDict
from typing import Any

import httpx
import pytest

from app.config import Settings
from app.config import settings as global_settings
from app.llm import ChatResult, MockLLM, OpenAICompatibleLLM, parse_json_block
from app.qa import AnswerPipeline
from app.rerank import LLMReranker, make_reranker
from app.usage import format_snapshot, log_call, read_journal, usage_snapshot


def _settings(**kw: Any) -> Settings:
    base: dict[str, Any] = {
        "llm_provider": "openai_compatible",
        "llm_api_base": "https://api.example.com/v1",
        # ключ по умолчанию задан: тестам нужен путь до HTTP-слоя (MockTransport),
        # а не отказ «LLM_API_KEY не задан». Проверка пустого ключа — отдельный
        # тест, который передаёт llm_api_key="".
        "llm_api_key": "k",
        "llm_model": "big-model",
    }
    base.update(kw)
    return _test_settings(**base)


def _test_settings(**kw: Any) -> Settings:
    """Настройки, не зависящие от .env машины.

    `_env_file=None` отключает чтение файла (иначе в тесты попадут реальные
    ключи, тарифы, LLM_BALANCE и локальная модель), но в аннотациях
    pydantic-settings этого параметра нет — отсюда точечное подавление.
    """
    return Settings(_env_file=None, **kw)  # type: ignore[call-arg]


def _transport(handler):
    return httpx.MockTransport(handler)


# --------------------------------------------------------------- выбор модели

def test_small_model_serves_service_tasks_only():
    s = _settings(llm_model_small="small-model")
    llm = OpenAICompatibleLLM(s)
    assert llm.model_for("classify") == "small-model"
    assert llm.model_for("rerank") == "small-model"
    assert llm.model_for("sql") == "small-model"
    # ответ пользователю формулирует только основная модель
    assert llm.model_for("compose") == "big-model"
    assert llm.model_for("other") == "big-model"


def test_small_model_tasks_are_configurable():
    s = _settings(llm_model_small="small-model", llm_model_small_tasks="rerank")
    llm = OpenAICompatibleLLM(s)
    assert llm.model_for("rerank") == "small-model"
    assert llm.model_for("classify") == "big-model"


def test_compose_never_goes_to_small_model():
    """Даже если compose указать в списке — ответ остаётся на основной модели."""
    s = _settings(llm_model_small="small-model", llm_model_small_tasks="compose,classify")
    llm = OpenAICompatibleLLM(s)
    assert llm.model_for("compose") == "big-model"


def test_single_model_when_small_not_set():
    llm = OpenAICompatibleLLM(_settings())
    assert {llm.model_for(t) for t in ("classify", "rerank", "sql", "compose")} == {"big-model"}


# ------------------------------------------------------------------ расход

def test_usage_counts_tokens_and_cached_prefix():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "привет"}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20,
                      "prompt_tokens_details": {"cached_tokens": 60}},
        })

    s = _settings()
    llm = OpenAICompatibleLLM(s)
    llm._client = httpx.AsyncClient(transport=_transport(handler))
    seen: list[ChatResult] = []
    llm.usage_hook = seen.append
    result = asyncio.run(llm.chat_result([{"role": "user", "content": "hi"}], task="compose"))
    assert result.text == "привет"
    assert result.prompt_tokens == 100
    assert result.completion_tokens == 20
    assert result.cached_tokens == 60
    assert result.total_tokens == 120
    assert result.task == "compose"
    assert len(seen) == 1 and seen[0].model == "big-model"


def test_usage_hook_failure_does_not_break_answer():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "ок"}}]})

    llm = OpenAICompatibleLLM(_settings())
    llm._client = httpx.AsyncClient(transport=_transport(handler))

    def boom(_):
        raise RuntimeError("учёт сломался")

    llm.usage_hook = boom
    assert asyncio.run(llm.chat([{"role": "user", "content": "hi"}])) == "ок"


# -------------------------------------------------- встроенный подсчёт в /usage

def test_usage_aggregate_in_meta(tmp_path):
    """Агрегат пишется в app_meta и переживает процесс: /usage читает его."""
    s = _settings(data_dir=tmp_path, llm_prices="small-model=0.10/0.40")
    s.database_url = f"sqlite+aiosqlite:///{tmp_path}/u.db"

    async def run():
        from app.storage import make_engine, make_sessionmaker

        engine = await make_engine(s)
        sessions = make_sessionmaker(engine)
        async with sessions() as session:
            from app.usage import add_usage

            for _ in range(3):
                await add_usage(session, {"task": "classify", "model": "small-model",
                                          "prompt_tokens": 100, "completion_tokens": 10})
            await add_usage(session, {"task": "compose", "model": "big-model",
                                      "prompt_tokens": 1000, "completion_tokens": 200})
            await session.commit()
            snap = await usage_snapshot(session, s)
        await engine.dispose()
        return snap

    snap = asyncio.run(run())
    assert snap["prompt_tokens"] == 1300
    assert snap["completion_tokens"] == 230
    assert snap["tasks"]["classify"]["calls"] == 3
    # цена задана только для small-model — big остаётся без оценки, но total считается
    assert snap["cost"] is not None
    assert abs(snap["cost"] - (300 / 1e6 * 0.10 + 30 / 1e6 * 0.40)) < 1e-9

    text = format_snapshot(snap)
    assert "small-model" in text and "классификация" in text
    assert "big-model" in text  # в списке «без цены»


def test_journal_records_calls(tmp_path):
    s = _settings(data_dir=tmp_path)
    log_call(s, {"task": "compose", "model": "m", "prompt_tokens": 5, "completion_tokens": 7})
    rows = read_journal(s.llm_usage_path)
    assert len(rows) == 1
    assert rows[0]["task"] == "compose"
    assert rows[0]["prompt_tokens"] == 5
    assert "ts" in rows[0]


# ---------------------------------------------------------------- устойчивость

def test_retry_on_rate_limit_then_success():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(429, headers={"retry-after": "0"}, text="slow down")
        return httpx.Response(200, json={"choices": [{"message": {"content": "готово"}}]})

    s = _settings(llm_retry_base_s=0.001)
    llm = OpenAICompatibleLLM(s)
    llm._client = httpx.AsyncClient(transport=_transport(handler))
    assert asyncio.run(llm.chat([{"role": "user", "content": "hi"}])) == "готово"
    assert len(calls) == 2


def test_retry_on_server_error_gives_up_with_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="unavailable")

    s = _settings(llm_retries=1, llm_retry_base_s=0.001)
    llm = OpenAICompatibleLLM(s)
    llm._client = httpx.AsyncClient(transport=_transport(handler))
    with pytest.raises(Exception) as err:
        asyncio.run(llm.chat([{"role": "user", "content": "hi"}]))
    assert "503" in str(err.value)


def test_max_tokens_falls_back_to_max_completion_tokens():
    """Новые модели OpenAI требуют max_completion_tokens — иначе весь ответ
    уходил в шаблон из-за 400."""
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        if "max_completion_tokens" not in body:
            return httpx.Response(400, json={"error": {"message": "Unsupported parameter: 'max_tokens'"}})
        return httpx.Response(200, json={"choices": [{"message": {"content": "ок"}}]})

    llm = OpenAICompatibleLLM(_settings())
    llm._client = httpx.AsyncClient(transport=_transport(handler))
    assert asyncio.run(llm.chat([{"role": "user", "content": "hi"}])) == "ок"
    assert "max_tokens" in seen[0] and "max_completion_tokens" not in seen[0]
    assert "max_completion_tokens" in seen[1] and "max_tokens" not in seen[1]


def test_temperature_rejected_then_retried_without_it():
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        if "temperature" in body:
            return httpx.Response(400, json={"error": {"message": "Unsupported value: 'temperature'"}})
        return httpx.Response(200, json={"choices": [{"message": {"content": "ок"}}]})

    llm = OpenAICompatibleLLM(_settings())
    llm._client = httpx.AsyncClient(transport=_transport(handler))
    assert asyncio.run(llm.chat([{"role": "user", "content": "hi"}])) == "ок"
    assert "temperature" in seen[0]
    assert "temperature" not in seen[1]


def test_json_mode_falls_back_when_unsupported():
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        if "response_format" in body:
            return httpx.Response(400, json={"error": {"message": "response_format is not supported"}})
        return httpx.Response(200, json={
            "choices": [{"message": {"content": '{"intent": "factual"}'}}]})

    llm = OpenAICompatibleLLM(_settings())
    llm._client = httpx.AsyncClient(transport=_transport(handler))
    out = asyncio.run(llm.chat([{"role": "user", "content": "выручка 2024"}], json_mode=True,
                               task="classify"))
    assert parse_json_block(out) == {"intent": "factual"}
    assert "response_format" in seen[0]
    assert all("response_format" not in b for b in seen[1:])


def test_small_model_json_asked_in_words_not_response_format():
    """У дешёвой модели JSON просим текстом: 400 на response_format — это
    потерянные токены промпта."""
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

    s = _settings(llm_model_small="small-model")
    llm = OpenAICompatibleLLM(s)
    llm._client = httpx.AsyncClient(transport=_transport(handler))
    asyncio.run(llm.chat([{"role": "user", "content": "вопрос"}], json_mode=True, task="classify"))
    assert "response_format" not in seen[0]
    assert "JSON" in seen[0]["messages"][-1]["content"]
    assert seen[0]["model"] == "small-model"


# ------------------------------------------------------------------- реранкер

def test_reranker_respects_token_budget_settings():
    s = _settings(rerank_max_items=3, rerank_snippet_chars=50)
    r = LLMReranker(MockLLM(), s)
    assert r.max_items == 3
    assert r.snippet_chars == 50
    assert s.rerank_max_items == 3


def test_auto_reranker_uses_llm_when_api_configured():
    s = _settings(reranker="auto")
    assert isinstance(make_reranker(s, OpenAICompatibleLLM(s)), LLMReranker)


# ---------------------------------------------------------------- кэш ответов

def test_repeated_question_costs_nothing(tmp_path):
    """Повторный вопрос отдаётся из кэша: ноль вызовов LLM и ноль токенов.

    Проверяем именно факт записи в кэш: если условие записи снова станет
    проверкой «на ложность», пустой OrderedDict сделает кэш мёртвым, и этот
    тест обязан упасть (счётчик обращений к API вырастет вдвое)."""
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        # ответ без цифр: у верификатора в этом payload нет допустимых чисел,
        # и любая цифра в тексте модели отправила бы ответ в шаблон.
        # Различимость вызовов даёт сам текст: если кэш не сработает, второй
        # ответ будет другим — это и проверит assert second.text == first.text
        calls.append(1)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": f"Ответ модели, вызов {len(calls)}."}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        })

    async def run():
        from app.embeddings import EmbeddingService
        from app.storage import make_engine, make_sessionmaker

        s = _test_settings()
        s.database_url = f"sqlite+aiosqlite:///{tmp_path}/c.db"
        s.data_dir = tmp_path
        s.llm_provider = "openai_compatible"
        s.llm_api_base = "https://api.example.com/v1"
        s.llm_model = "big-model"
        s.llm_api_key = "k"          # ключ нужен, чтобы дойти до MockTransport
        s.answer_cache_size = 8
        engine = await make_engine(s)
        sessions = make_sessionmaker(engine)
        emb = EmbeddingService(s)
        llm = OpenAICompatibleLLM(s)
        llm._client = httpx.AsyncClient(transport=_transport(handler))
        pipeline = AnswerPipeline(sessions, emb, llm, s, org_names=[])
        first = await pipeline.answer(1, 1, "что говорится о рисках?")
        cached_len = len(pipeline._answer_cache or {})
        calls_after_first = len(calls)
        second = await pipeline.answer(1, 1, "что говорится о рисках?")
        await engine.dispose()
        return first, second, cached_len, calls_after_first, len(calls)

    first, second, cached_len, calls_after_first, calls_total = asyncio.run(run())
    # ответ записан в кэш — иначе второй вопрос снова пойдёт к модели
    assert cached_len == 1
    assert calls_after_first >= 1
    assert calls_total == calls_after_first, "повторный вопрос снова обратился к API"
    assert second.text == first.text


def test_cache_key_is_scoped_by_user(tmp_path):
    """Приватные документы: ответ одного сотрудника не должен достаться коллеге.

    Конвейер разграничен по user_id (витрина facts_visible, 🔒 документы),
    поэтому ключ кэша обязан включать пользователя."""
    pipeline = AnswerPipeline.__new__(AnswerPipeline)
    pipeline.s = _test_settings()
    pipeline._answer_cache = None

    k_user1 = pipeline._cache_key(1, 1, "что говорится о рисках?", None, None)
    k_user2 = pipeline._cache_key(1, 2, "что говорится о рисках?", None, None)
    k_shared = pipeline._cache_key(1, None, "что говорится о рисках?", None, None)
    k_org2 = pipeline._cache_key(2, 1, "что говорится о рисках?", None, None)

    assert k_user1 != k_user2, "ответ на приватный документ утечёт другому пользователю"
    assert k_user1 != k_org2
    # общий контур (сводки без пользователя) стабилен
    assert k_shared == pipeline._cache_key(1, None, "что говорится о рисках?", None, None)
    assert k_shared not in (k_user1, k_user2)


def test_cache_starts_empty_but_writable():
    """Пустой OrderedDict — валидный кэш: проверка «на ложность» его отключила бы."""
    from app.qa import QAOutcome

    s = _test_settings()
    s.answer_cache_size = 4
    pipeline = AnswerPipeline.__new__(AnswerPipeline)
    pipeline.s = s
    pipeline._answer_cache = OrderedDict()
    assert not pipeline._answer_cache          # пустой контейнер ложен...
    assert pipeline._cache_get("нет") is None  # ...но чтение работает
    pipeline._cache_put("k", QAOutcome(text="ответ"))
    assert list(pipeline._answer_cache) == ["k"]
    cached = pipeline._cache_get("k")
    assert cached is not None and cached.text == "ответ"


def test_cache_skips_outcomes_with_attachments():
    """У фактических ответов есть график/кнопки — их кэшировать нельзя."""
    from app.qa import QAOutcome

    s = _test_settings()
    s.answer_cache_size = 4
    pipeline = AnswerPipeline.__new__(AnswerPipeline)
    pipeline.s = s
    pipeline._answer_cache = OrderedDict()

    pipeline._cache_put("chart", QAOutcome(text="прогноз", chart_png=b"png"))
    pipeline._cache_put("table", QAOutcome(text="факт", table_metric_id=7))
    pipeline._cache_put("clarify", QAOutcome(text="уточните", clarify=["выручка"]))
    assert pipeline._answer_cache == {}
    # LRU: при переполнении вытесняется самый старый
    for i in range(5):
        pipeline._cache_put(f"q{i}", QAOutcome(text=f"ответ {i}"))
    assert len(pipeline._answer_cache) == 4
    assert "q0" not in pipeline._answer_cache and "q4" in pipeline._answer_cache


def test_cache_can_be_disabled():
    from app.qa import QAOutcome

    s = _test_settings()
    s.answer_cache_size = 0
    pipeline = AnswerPipeline.__new__(AnswerPipeline)
    pipeline.s = s
    pipeline._answer_cache = None
    assert pipeline._cache_get("x") is None
    pipeline._cache_put("x", QAOutcome(text="ответ"))
    assert pipeline._answer_cache is None


# ------------------------------------------------- локальные эмбеддинги

def test_remote_embeddings_refused_by_default():
    """Векторизация через платный внешний API запрещена по умолчанию: это и
    расход на токены, и вынос текстов документов наружу."""
    from app.main import remote_embeddings_reason

    s = _settings(embeddings_provider="api",
                  embeddings_api_base="https://api.openai.com/v1",
                  require_local_embeddings=True)
    assert remote_embeddings_reason(s) == "https://api.openai.com/v1"


def test_remote_embeddings_allowed_with_explicit_override():
    from app.main import remote_embeddings_reason

    s = _settings(embeddings_provider="api",
                  embeddings_api_base="https://api.openai.com/v1",
                  require_local_embeddings=False)
    assert remote_embeddings_reason(s) is None


def test_local_embeddings_and_local_endpoint_pass():
    from app.main import remote_embeddings_reason

    local_model = _settings(embeddings_provider="sentence_transformers")
    assert remote_embeddings_reason(local_model) is None
    # свой vLLM/Ollama на том же хосте — данные не покидают контур
    local_api = _settings(embeddings_provider="api",
                          embeddings_api_base="http://localhost:8000/v1")
    assert remote_embeddings_reason(local_api) is None


def test_refusal_message_explains_the_way_out():
    pytest = __import__("pytest")

    from app.main import _refuse_remote_embeddings

    s = _settings(embeddings_provider="api",
                  embeddings_api_base="https://api.example.com/v1")
    with pytest.raises(SystemExit) as err:
        _refuse_remote_embeddings(s)
    text = str(err.value)
    assert "REQUIRE_LOCAL_EMBEDDINGS=false" in text
    assert "sentence_transformers" in text


# ------------------------------------------------- бюджет промпта композитора

def test_composer_context_budget_is_configurable():
    """Промпт ответа — главный расход: 5 фрагментов × 200 символов по умолчанию,
    и обе границы меняются настройками."""
    from app.qa import _compact_for_llm

    payload = {
        "type": "explain",
        "context": [{"text": "ф" * 500, "source": "файл · лист · ячейка · очень длинный путь источника"}
                    for _ in range(10)],
    }
    default = _settings()
    compact = _compact_for_llm(payload, default)
    assert len(compact["context"]) == default.composer_context_items == 5
    assert len(compact["context"][0]["text"]) == default.composer_context_chars == 200

    tight = _settings(composer_context_items=2, composer_context_chars=50)
    small = _compact_for_llm(payload, tight)
    assert len(small["context"]) == 2
    assert len(small["context"][0]["text"]) == 50
    # источник тоже обрезается, но не длиннее 40 символов
    assert len(small["context"][0]["source"]) == 40


def test_compaction_keeps_all_numbers():
    """Экономия не должна трогать числа: по ним работает верификатор."""
    from app.qa import _compact_for_llm

    payload = {
        "type": "compare",
        "metric": {"name": "выручка", "unit": "руб"},
        "computed": {"change_pct": 8.1, "abs_change": 3700000.0},
        "history": [{"label": "2024", "value": 45800000.0, "source": "очень длинный путь · лист · C4"},
                    {"label": "2025", "value": 49500000.0}],
    }
    compact = _compact_for_llm(payload, _settings())
    assert compact["computed"] == {"change_pct": 8.1, "abs_change": 3700000.0}
    assert [r["value"] for r in compact["history"]] == [45800000.0, 49500000.0]
    # путь источника в промпт не тащим
    assert all("source" not in r for r in compact["history"])


# ------------------------------------------------- баланс ключа API (402)

def test_balance_detection_patterns():
    """DeepSeek: 402 Insufficient Balance. Прочие провайдеры — свои маркеры."""
    from app.llm import balance_exhausted

    assert balance_exhausted(402, '{"error":{"message":"Insufficient Balance"}}')
    assert balance_exhausted(429, '{"error":{"code":"insufficient_quota"}}')
    assert balance_exhausted(403, "Your credit balance is too low")
    assert balance_exhausted(400, "余额不足")          # локализованный текст DeepSeek
    # обычные ошибки деньгами не считаются
    assert not balance_exhausted(401, '{"error":{"message":"Authentication Fails"}}')
    assert not balance_exhausted(400, "Unsupported parameter: max_tokens")
    assert not balance_exhausted(429, "Rate limit reached")
    assert not balance_exhausted(500, "Server Error")


def test_402_raises_balance_error_without_retries():
    """На 402 повторять нечего: деньги не появятся от ожидания."""
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(402, json={"error": {"message": "Insufficient Balance"}})

    from app.llm import LLMBalanceError

    s = _settings(llm_retries=3, llm_retry_base_s=0.001)
    llm = OpenAICompatibleLLM(s)
    llm._client = httpx.AsyncClient(transport=_transport(handler))
    with pytest.raises(LLMBalanceError) as err:
        asyncio.run(llm.chat([{"role": "user", "content": "hi"}]))
    assert "исчерпан" in str(err.value)
    assert len(calls) == 1, "402 не должен повторяться"


def test_balance_pause_stops_further_calls():
    """После 402 пауза: в API не ходим, пока не истечёт LLM_BALANCE_RETRY_S."""
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(402, json={"error": {"message": "Insufficient Balance"}})

    from app.llm import LLMBalanceError

    s = _settings(llm_balance_retry_s=3600)
    llm = OpenAICompatibleLLM(s)
    llm._client = httpx.AsyncClient(transport=_transport(handler))
    assert llm.balance_ok() is True
    with pytest.raises(LLMBalanceError):
        asyncio.run(llm.chat([{"role": "user", "content": "hi"}]))
    assert llm.balance_ok() is False, "без паузы каждый вопрос давал бы пачку 402"
    # предупреждение пользователю показываем ровно один раз на эпизод
    assert llm.take_balance_notice() is True
    assert llm.take_balance_notice() is False
    assert len(calls) == 1


def test_balance_pause_expires_and_retries():
    """Пауза истекла — клиент снова пробует API (например, счёт пополнен)."""
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(402, json={"error": {"message": "Insufficient Balance"}})
        return httpx.Response(200, json={"choices": [{"message": {"content": "ок"}}]})

    from app.llm import LLMBalanceError

    s = _settings(llm_balance_retry_s=0)   # 0 — паузы нет, пробуем каждый раз
    llm = OpenAICompatibleLLM(s)
    llm._client = httpx.AsyncClient(transport=_transport(handler))
    with pytest.raises(LLMBalanceError):
        asyncio.run(llm.chat([{"role": "user", "content": "hi"}]))
    assert llm.balance_ok() is True
    assert asyncio.run(llm.chat([{"role": "user", "content": "hi"}])) == "ок"


def test_successful_call_clears_balance_state():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "ок"}}]})

    llm = OpenAICompatibleLLM(_settings())
    llm._client = httpx.AsyncClient(transport=_transport(handler))
    llm._balance_exhausted_at = 123.0
    asyncio.run(llm.chat([{"role": "user", "content": "hi"}]))
    assert llm.balance_ok() is True


def test_balance_event_is_recorded_and_reported(tmp_path):
    """Событие 402 попадает в app_meta и выводится в сводке — по нему видно,
    сколько запросов упало, пока не было денег."""
    from app.usage import add_balance_event, balance_events, format_balance_state

    s = _settings(data_dir=tmp_path)
    s.database_url = f"sqlite+aiosqlite:///{tmp_path}/b.db"

    async def run():
        from app.storage import make_engine, make_sessionmaker

        engine = await make_engine(s)
        sessions = make_sessionmaker(engine)
        async with sessions() as session:
            await add_balance_event(session, {"reason": "HTTP 402: Insufficient Balance",
                                              "model": "deepseek-flash"})
            await add_balance_event(session, {"reason": "HTTP 402: Insufficient Balance",
                                              "model": "deepseek-flash"})
            await session.commit()
            events = await balance_events(session)
        await engine.dispose()
        return events

    events = asyncio.run(run())
    assert events["count"] == 2
    assert events["first_at"] and events["last_at"]
    assert "Insufficient Balance" in events["last_reason"]
    text = format_balance_state(events)
    assert "2 раз" in text and "шаблоном" in text
    assert format_balance_state({}) == ""


def test_balance_event_goes_to_journal(tmp_path):
    from app.usage import log_balance_event, read_journal

    s = _settings(data_dir=tmp_path)
    log_balance_event(s, {"reason": "HTTP 402", "model": "deepseek-flash"})
    rows = read_journal(s.llm_usage_path)
    assert rows and rows[0]["event"] == "balance_exhausted"
    assert rows[0]["reason"] == "HTTP 402"


def test_exhausted_balance_skips_api_and_warns(tmp_path):
    """Полный путь конвейера: 402 → шаблонный ответ + предупреждение,
    и следующий вопрос уже не идёт в API (ноль обращений)."""
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(402, json={"error": {"message": "Insufficient Balance"}})

    async def run():
        from app.embeddings import EmbeddingService
        from app.storage import make_engine, make_sessionmaker

        s = _test_settings()
        s.database_url = f"sqlite+aiosqlite:///{tmp_path}/bal.db"
        s.data_dir = tmp_path
        s.llm_provider = "openai_compatible"
        s.llm_api_base = "https://api.deepseek.com/v1"
        s.llm_model = "deepseek-flash"
        s.llm_api_key = "k"            # ключ есть: проверяем именно 402, а не его отсутствие
        s.llm_balance_retry_s = 3600
        s.answer_cache_size = 0
        s.compose_mode = "llm"          # пользователь ждёт текст модели
        # классификация моделью: именно этот шаг глотает ошибку баланса и
        # продолжает работу — проверяем, что событие всё равно фиксируется
        s.classify_with_llm = True
        engine = await make_engine(s)
        sessions = make_sessionmaker(engine)
        emb = EmbeddingService(s)
        llm = OpenAICompatibleLLM(s)
        llm._client = httpx.AsyncClient(transport=_transport(handler))
        pipeline = AnswerPipeline(sessions, emb, llm, s, org_names=[])
        first = await pipeline.answer(1, 1, "почему выросли расходы?")
        after_first = len(calls)
        second = await pipeline.answer(1, 1, "что говорится о рисках?")
        from app.usage import balance_events

        async with sessions() as s2:
            events = await balance_events(s2)
        await engine.dispose()
        return first, second, after_first, len(calls), events

    first, second, after_first, total, events = asyncio.run(run())
    assert first.balance_notice, "пользователь должен узнать, что баланс кончился"
    assert first.text, "ответ всё равно отдаётся — шаблоном из тех же данных"
    assert total == after_first, "после 402 в API больше не ходим"
    assert after_first >= 1
    # событие записано, хотя классификатор ошибку проглотил
    assert events.get("count", 0) >= 1
    assert "исчерпан" in str(events.get("last_reason", ""))


def test_budget_left_line(tmp_path):
    """Строка «осталось / на сколько ещё хватит» в /usage."""
    from app.usage import format_budget_left

    snap = {
        "models": [{"model": "deepseek-flash", "calls": 10, "prompt_tokens": 16000,
                    "completion_tokens": 2500, "cached_tokens": 0, "cost": 0.026}],
        "tasks": {"compose": {"calls": 10, "prompt_tokens": 16000, "completion_tokens": 2500}},
        "prompt_tokens": 16000, "completion_tokens": 2500, "cached_tokens": 0,
        "cost": 0.026, "priced_models": ["deepseek-flash"],
    }
    s = _settings(llm_balance=33.0, llm_prices="deepseek-flash=1/4")
    text = format_budget_left(snap, s)
    assert "33" in text and "осталось" in text
    # средняя цена вопроса 0.0026 → остатка (~33) хватает на ~12 тыс. вопросов
    assert "12,6" in text or "12.6" in text

    # без заданного баланса или без цен строки нет
    assert format_budget_left(snap, _settings(llm_prices="deepseek-flash=1/4")) == ""
    assert format_budget_left({**snap, "cost": None}, s) == ""

    # исчерпанный баланс — прямое предупреждение
    broke = {**snap, "cost": 40.0}
    assert "исчерпан" in format_budget_left(broke, s)


def test_missing_api_key_fails_fast_without_network():
    """Пустой ключ на внешнем endpoint: понятная ошибка и ни одного запроса.

    Иначе httpx падал с «Illegal header value b'Bearer '», клиент трижды
    повторял заведомо нерабочий вызов, а пользователь видел мусор.
    """
    from app.llm import LLMError, _missing_api_key, _no_key_hint

    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ок"}}]})

    s = _settings(llm_api_key="")
    assert _missing_api_key(s) is True
    hint = _no_key_hint(s)
    assert "LLM_API_KEY не задан" in hint and "api.example.com" in hint

    llm = OpenAICompatibleLLM(s)
    llm._client = httpx.AsyncClient(transport=_transport(handler))
    with pytest.raises(LLMError) as err:
        asyncio.run(llm.chat([{"role": "user", "content": "hi"}]))
    assert "LLM_API_KEY" in str(err.value)
    assert calls == [], "без ключа запрос уходить не должен"


def test_local_engine_needs_no_key():
    """Ollama/vLLM ключ не проверяют: пустой LLM_API_KEY там — норма."""
    from app.llm import _missing_api_key

    s = _settings(llm_api_key="", llm_api_base="http://localhost:11434/v1")
    assert _missing_api_key(s) is False


def test_deepseek_hint_points_to_key_page():
    from app.llm import _no_key_hint

    s = _settings(llm_api_key="", llm_api_base="https://api.deepseek.com/v1")
    assert "platform.deepseek.com" in _no_key_hint(s)


# ------------------------------------------- производные числа в прогнозе

def _forecast_payload() -> dict:
    return {
        "type": "forecast",
        "metric": {"name": "аренда спецтехники", "unit": "руб", "currency": "RUB"},
        "history": [{"label": "2026", "value": 12_700_000.0}],
        "forecast": {
            "target": 2027,
            "base": 13_944_033.42,
            "low": 13_165_927.38,
            "high": 14_722_139.47,
            "methods": {"naive": 12_700_000.0, "mean_growth": 14_406_723.0},
            "backtest_error_pct": 4.36,
            "confidence": "средняя",
        },
    }


def test_derived_percentage_in_forecast_is_allowed():
    """«Прогноз на 9,8% выше факта» — модель посчитала сама; это не галлюцинация."""
    from app.agents import derived_numbers, verify_answer

    payload = _forecast_payload()
    template = "прогноз 13,94 млн ₽"
    # 13 944 033 / 12 700 000 − 1 = 9,795% — модель пишет округлённо
    assert verify_answer("Прогноз на 9,8% выше факта 2026.", template, payload)[0]
    assert verify_answer("Рост составит около 10%.", template, payload)[0]
    # разность прогноза и факта: 1 244 033 ₽
    assert verify_answer("Это на 1 244 033 ₽ больше факта.", template, payload)[0]
    assert derived_numbers(payload), "производные должны вычисляться"
    # ширина интервала: 14 722 139 − 13 165 927 = 1 556 212
    assert verify_answer("Интервал шириной 1,56 млн ₽.", template, payload)[0]


def test_derived_numbers_do_not_open_a_hole_for_invention():
    """Главное: выдуманное число всё равно отклоняется."""
    from app.agents import verify_answer

    payload = _forecast_payload()
    template = "прогноз 13,94 млн ₽"
    for invented in ("Прогноз 15,7 млн ₽", "Рост составит 43%", "Это на 987 654 ₽ больше",
                     "Уверенность 87%", "Разница 5 000 000 ₽"):
        ok, foreign = verify_answer(invented, template, payload)
        assert not ok, f"выдуманное число прошло проверку: {invented}"
        assert foreign


def test_derived_allowed_only_for_calculating_types():
    """Производные разрешены там, где модель обязана считать: прогноз, состав,
    сравнение, ранжирование. В простом факте — нет: считать там нечего."""
    from app.agents import derived_numbers, verify_answer

    payload = _forecast_payload()
    for ptype in ("forecast", "compare", "breakdown", "rank"):
        payload["type"] = ptype
        assert derived_numbers(payload), f"{ptype}: производные должны быть разрешены"

    payload["type"] = "factual"
    assert derived_numbers(payload) == []
    ok, foreign = verify_answer("Прогноз на 9,8% выше факта.", "13,94 млн ₽", payload)
    assert not ok and 9.8 in foreign


def test_derived_can_be_switched_off():
    from app.agents import verify_answer

    payload = _forecast_payload()
    ok, foreign = verify_answer("Прогноз на 9,8% выше факта.", "13,94 млн ₽", payload,
                                allow_derived=False)
    assert not ok and 9.8 in foreign


def test_derived_in_plan_fact_comparison():
    """План/факт: «отклонение +3,13%» и «на 1,5 млн ₽» — тоже производные."""
    from app.agents import verify_answer

    payload = {
        "type": "compare",
        "metric": {"name": "выручка", "unit": "руб", "currency": "RUB"},
        "history": [{"label": "2026 факт", "value": 49_500_000.0}],
        "computed": {"plan_value": 48_000_000.0, "abs_change": 1_500_000.0,
                     "deviation_pct": 3.125},
    }
    template = "план 48,00 млн ₽"
    assert verify_answer("Факт выше плана на 3,13%.", template, payload)[0]
    assert verify_answer("Отклонение 1 500 000 ₽.", template, payload)[0]
    ok, foreign = verify_answer("Факт выше плана на 7,5%.", template, payload)
    assert not ok, "процент, которого нет в расчёте, должен отклоняться"


def test_derived_share_of_total_is_allowed():
    """«Статья — 90% от итога»: модель считает долю сама (в payload её нет)."""
    from app.agents import verify_answer

    payload = {
        "type": "breakdown",
        "metric": {"name": "операционные расходы", "unit": "руб", "currency": "RUB"},
        "total": 30_600_000.0,
        "items": [
            {"name": "зарплаты", "value": 27_540_000.0},
            {"name": "аренда склада", "value": 1_800_000.0},
            {"name": "аренда спецтехники", "value": 1_260_000.0},
        ],
    }
    template = "итого 30,60 млн ₽"
    # 27,54 / 30,6 = 90%
    assert verify_answer("Зарплаты занимают 90% расходов.", template, payload)[0]
    assert verify_answer("Это 5,9% от итога.", template, payload)[0]   # 1,8/30,6
    ok, foreign = verify_answer("Зарплаты занимают 62% расходов.", template, payload)
    assert not ok, "доля, не соответствующая данным, должна отклоняться"


def test_derived_sum_of_items_is_allowed():
    """«Зарплаты и аренда спецтехники вместе — 90% расходов»: модель сложила
    две статьи и поделила на итог. Именно этот ответ отклонялся как выдуманный
    (в логе: посторонние [90.0] и [89.7])."""
    from app.agents import verify_answer

    payload = {
        "type": "breakdown",
        "metric": {"name": "операционные расходы", "unit": "руб", "currency": "RUB"},
        "total": 34_100_000.0,
        "items": [
            {"name": "зарплаты", "value": 17_900_000.0, "share_pct": 52.5},
            {"name": "аренда спецтехники", "value": 12_700_000.0, "share_pct": 37.2},
            {"name": "аренда склада", "value": 3_500_000.0, "share_pct": 10.3},
        ],
    }
    template = "итого 34,10 млн ₽"
    assert verify_answer("Вместе это ~90% всех затрат.", template, payload)[0]
    assert verify_answer("Зарплаты и аренда спецтехники: вместе 30 600 000 ₽, "
                         "или около 89,7%.", template, payload)[0]
    # а вот выдуманная сумма по-прежнему не проходит
    ok, foreign = verify_answer("Вместе это 25 000 000 ₽.", template, payload)
    assert not ok and 25_000_000.0 in foreign


def test_percent_band_uses_the_data_range():
    """Проценты в пределах разброса данных допускаются (модель округляет и
    усредняет), выдуманный процент вне диапазона — нет."""
    from app.agents import verify_answer

    payload = {
        "type": "forecast",
        "metric": {"name": "аренда", "unit": "руб", "currency": "RUB"},
        "history": [{"label": "2026", "value": 12_700_000.0}],
        "forecast": {"base": 13_944_033.0, "low": 13_165_927.0, "high": 14_722_139.0,
                     "backtest_error_pct": 4.36},
    }
    template = "13,94 млн ₽"
    # проценты данных здесь 4,36 и ~9,8 — «4,5%» внутри диапазона, «43%» — нет
    assert verify_answer("Ошибка бэктеста около 4,5%.", template, payload)[0]
    ok, foreign = verify_answer("Ошибка бэктеста 43%.", template, payload)
    assert not ok and 43.0 in foreign
    # абсолютные суммы поблажкой не покрываются
    ok, foreign = verify_answer("Прогноз 20 000 000 ₽.", template, payload)
    assert not ok and 20_000_000.0 in foreign


def test_derived_from_percentages():
    """Разность двух процентов из данных: «на 4,4 п.п. выше»."""
    from app.agents import verify_answer

    payload = {
        "type": "compare",
        "metric": {"name": "выручка", "unit": "%"},
        "computed": {"change_pct": 8.1, "avg_growth_pct": 12.5},
    }
    assert verify_answer("Разрыв 4,4 п.п.", "8,1%", payload)[0]


# ------------------------------------------------------- формат отчёта /usage

def test_format_snapshot_empty():
    text = format_snapshot({"models": [], "tasks": {}, "prompt_tokens": 0,
                            "completion_tokens": 0, "cached_tokens": 0, "cost": None,
                            "priced_models": []})
    assert "не зафиксировано" in text


def test_global_settings_expose_token_economy_defaults():
    assert global_settings.llm_retries >= 1
    assert global_settings.classifier_max_tokens <= 200
    assert global_settings.rerank_max_items <= 10
    assert global_settings.small_model_tasks >= {"classify", "rerank", "sql"}
    assert global_settings.llm_usage_path.name == "llm_usage.jsonl"
    assert global_settings.llm_price_map == {} or isinstance(global_settings.llm_price_map, dict)


def test_price_map_parsing():
    s = _settings(llm_prices="a=0.1/0.4, b=2,мусор")
    assert s.llm_price_map == {"a": (0.1, 0.4), "b": (2.0, 2.0)}
