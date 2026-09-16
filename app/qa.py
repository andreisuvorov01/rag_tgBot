"""QA-конвейер: вопрос -> классификация -> резолв показателя -> маршрут
(SQL / векторный поиск / аналитика) -> сборка данных -> LLM-композитор.

LLM используется на трёх шагах: классификация запроса, генерация SQL для
гибких вопросов (под защитой whitelist'а операций) и составление ответа.
Все числа в ответ приходят из ДАННЫХ (JSON), считанных программно.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from .agents import VerificationAgent
from .analytics import forecast as forecast_fn
from .analytics import prepare_series
from .config import Settings
from .embeddings import EmbeddingService
from .formatting import fmt_number, render_answer, render_table
from .graph import AgentGraph
from .llm import BALANCE_NOTICE, BaseLLM, LLMBalanceError, LLMError, parse_json_block
from .prompts import CLASSIFIER_SYSTEM, COMPOSER_SYSTEM, SQL_SYSTEM
from .rag import focus_snippet, hybrid_search
from .rerank import make_reranker
from .storage import (
    Metric,
    audit,
    find_metric_by_name,
    metric_by_id,
    metric_children,
    org_metrics,
    run_readonly,
    series_for_metric,
    vector_search,
)

log = logging.getLogger(__name__)

_FORBIDDEN_SQL = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|grant|revoke|copy|vacuum|"
    # серверные функции чтения файлов/сна/расширений — read-only транзакция их не запрещает
    r"pragma|attach|detach|load_extension|readfile|writefile|pg_sleep|pg_read_\w+|pg_ls_\w+|"
    r"pg_stat_\w+|lo_import|lo_export|dblink|current_setting|set_config)\b",
    re.I,
)
# таблицы, к которым Text-to-SQL вправе обращаться; users/audit_log/documents и
# прочее — нет (сгенерированный SQL приходит из недоверенного текста вопроса)
_SQL_ALLOWED_TABLES = {"metrics", "facts_visible"}
_SQL_TABLE_REF = re.compile(r"\b(?:from|join)\s+([a-zA-Z_][\w.]*)", re.I)


def sql_is_safe(sql: str) -> bool:
    """Один SELECT, без DML/DDL и опасных функций, только разрешённые таблицы."""
    if not sql.lower().startswith("select") or ";" in sql or _FORBIDDEN_SQL.search(sql):
        return False
    # EXTRACT(YEAR FROM x) / SUBSTRING(x FROM 1) — это не ссылки на таблицы
    scan = re.sub(r"\b(extract|substring|trim|position|overlay)\s*\([^()]*\)", "", sql, flags=re.I)
    refs = {t.lower() for t in _SQL_TABLE_REF.findall(scan)}
    return "facts_visible" in refs and refs <= _SQL_ALLOWED_TABLES
# бюджет контекста для композитора (практика enterprise-RAG: не заливать промпт целиком)
MAX_CONTEXT_ITEMS = 5
MAX_HISTORY_ROWS = 24
# уточнения, опирающиеся на память диалога: «а за 2023?», «по нему?», «а теперь аренда?»
FOLLOWUP_RE = re.compile(
    r"по нему|по ней|по этой|этот показатель|эта позиция|тот же|а теперь|а за|а по|"
    r"^\s*а\b|и за\s|а в\s",
    re.I,
)
# явная условная формулировка сценария «что если» — без неё множитель не берём
_SCENARIO_RE = re.compile(r"\bесли\b|\bпри\s+условии\b|\bчто\s+если\b", re.I)

# Ответы этих типов код собирает полностью: числа посчитаны, таблица сверстана,
# источники подписаны. Пересказ моделью не добавляет данных, но стоит десятки
# секунд на CPU, поэтому в режиме compose_mode=auto они отдаются шаблоном.
# «explain» и «rank» здесь НЕТ намеренно: там модель формулирует вывод по
# найденным фрагментам, и это её настоящая работа.
TEMPLATE_FIRST_TYPES = frozenset({
    "factual", "compare", "breakdown", "forecast", "sql_result", "doc_summary",
})


def visible_facts_subquery(dialect: str) -> str:
    """Подзапрос 'видимых фактов' для Text-to-SQL: только своя организация,
    приватные документы других пользователей исключены на уровне SQL; параметры
    :org и :uid подставляются детерминированно после генерации — LLM не может
    их обойти."""
    if dialect == "postgresql":
        flag = "COALESCE(d.is_private, FALSE) = FALSE"
    else:
        flag = "COALESCE(d.is_private, 0) = 0"
    return (
        "(SELECT f.* FROM facts f JOIN documents d ON d.id = f.document_id "
        f"WHERE f.org_id = :org AND d.superseded_by_id IS NULL AND ({flag} OR d.uploaded_by = :uid))"
    )


def _safe_int_years(raw) -> list[int]:
    """Годы из ответа LLM: отфильтровать null/мусор (маленькие модели
    возвращают [2026, null] или строки)."""
    out: list[int] = []
    for y in raw or []:
        try:
            y = int(y)
        except (TypeError, ValueError):
            continue
        if 1990 < y < 2100:
            out.append(y)
    return out


def _safe_target_year(raw, years: list[int]) -> int | None:
    try:
        y = int(raw)
    except (TypeError, ValueError):
        y = None
    return y if y is not None and 1990 < y < 2100 else (max(years) if years else None)


@dataclass
class QAOutcome:
    text: str
    clarify: list[str] = field(default_factory=list)
    chart_png: bytes | None = None       # график прогноза (PNG для Telegram)
    table_metric_id: int | None = None   # «⬇️ Excel» и быстрые действия по показателю
    ops_months: list[tuple[str, float]] = field(default_factory=list)  # месяцы выписки
    # предупреждение «на ключе API закончился баланс» (показывается один раз
    # на эпизод, чтобы пользователь понимал, почему ответ стал шаблонным)
    balance_notice: str = ""
    # тип собранных данных (factual/explain/rank/...): по нему видно, потерял ли
    # ответ смысл без модели
    payload_type: str = ""


def _compact_for_llm(payload: dict, settings: Settings | None = None) -> dict:
    """Сжатая копия payload для промпта композитора.

    Полный JSON содержит всё, что нужно проверке и шаблону, но для генерации
    лишён смысла и раздувает промпт: длинные пути источников («файл · лист ·
    ячейка») не нужны модели, чтобы сформулировать вывод, а полный
    template_hint дублирует данные таблицей. Числа, единицы, метки периодов и
    вычисленные значения сохраняются полностью — по ним верификатор и сверяет.

    Бюджет контекста настраивается (COMPOSER_CONTEXT_ITEMS / _CHARS): это самая
    дорогая часть промпта, и на внешнем API её размер — прямые деньги.
    """
    ctx_items = settings.composer_context_items if settings else MAX_CONTEXT_ITEMS
    ctx_chars = settings.composer_context_chars if settings else 350
    ctx_source_chars = min(40, ctx_chars)

    def _rows(rows: list | None) -> list:
        out = []
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            item = {k: row[k] for k in ("label", "value") if k in row}
            for extra in ("name", "growth_pct", "share_pct", "label_from", "label_to"):
                if extra in row:
                    item[extra] = row[extra]
            out.append(item)
        return out

    compact: dict = {"type": payload.get("type")}
    for key in ("metric", "computed", "forecast", "total", "period_label", "query", "notes"):
        if key in payload:
            compact[key] = payload[key]
    if "history" in payload:
        compact["history"] = _rows(payload.get("history"))
    if "items" in payload:
        compact["items"] = _rows(payload.get("items"))
    if "ranking" in payload:
        compact["ranking"] = _rows(payload.get("ranking"))
    # контекст оставляем — он про смысл («что говорится о рисках»), но режем
    # и число фрагментов, и длину каждого: для формулировки вывода хватает начала
    if payload.get("context"):
        compact["context"] = [
            {"text": (c.get("text") or "")[:ctx_chars],
             "source": (c.get("source") or "")[:ctx_source_chars]}
            for c in payload["context"][:ctx_items]
            if isinstance(c, dict)
        ]
    return {k: v for k, v in compact.items() if v is not None}


class AnswerPipeline:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        emb: EmbeddingService,
        llm: BaseLLM,
        settings: Settings,
        org_names: list[str] | None = None,
    ):
        self.sessions = session_factory
        self.emb = emb
        self.llm = llm
        self.s = settings
        self.org_names = org_names or []
        # Единая точка маскирования: оборачиваем клиент, чтобы ИНН/счета/
        # названия организаций не уходили наружу ни из классификатора, ни из
        # Text-to-SQL, ни из LLM-реранкера — раньше маскировался только
        # промпт композитора.
        if settings.anonymize_prompts and settings.llm_provider != "mock":
            from .llm import AnonymizingLLM

            self.llm = AnonymizingLLM(llm, self.org_names)
        self.reranker = make_reranker(settings, self.llm)
        self.verifier = VerificationAgent(settings)
        # память диалога: последний показатель пользователя для уточнений
        self.dialog_memory: dict[int, dict] = {}
        # Учёт расхода токенов: единственная платная часть — внешний API. Хук
        # вешается на фактический клиент (в т.ч. под маскирующей обёрткой), а
        # агрегат пишется в app_meta, чтобы /usage не читал журнал целиком.
        self._pending_usage: list[dict] = []
        if settings.answer_cache_size:
            from collections import OrderedDict

            self._answer_cache: OrderedDict[str, QAOutcome] | None = OrderedDict()
        else:
            self._answer_cache = None
        inner = getattr(self.llm, "_inner", self.llm)
        if hasattr(inner, "usage_hook"):
            inner.usage_hook = self._on_llm_usage

    def _on_llm_usage(self, result) -> None:
        """Записать расход вызова в журнал и запомнить для агрегата в БД."""
        record = {
            "task": getattr(result, "task", "other"),
            "model": getattr(result, "model", ""),
            "prompt_tokens": getattr(result, "prompt_tokens", 0),
            "completion_tokens": getattr(result, "completion_tokens", 0),
            "cached_tokens": getattr(result, "cached_tokens", 0),
            "reasoning_tokens": getattr(result, "reasoning_tokens", 0),
        }
        from .usage import log_call

        log_call(self.s, record)
        self._pending_usage.append(record)
        log.debug("LLM %s/%s: %s+%s токенов", record["task"], record["model"],
                  record["prompt_tokens"], record["completion_tokens"])

    async def flush_usage(self, session: AsyncSession) -> None:
        """Перенести накопленный расход в агрегат. Вызывается перед commit."""
        if not self._pending_usage:
            return
        from .usage import add_usage

        pending, self._pending_usage = self._pending_usage, []
        for record in pending:
            try:
                await add_usage(session, record)
            except Exception as e:
                log.warning("Не удалось агрегировать расход токенов: %s", e)

    def _cache_key(self, org_id: int, user_id: int | None, query: str,
                   metric_override: str | None, intent_override: str | None) -> str:
        """Ключ кэша. user_id обязателен: доступ к документам разграничен по
        пользователю (витрина facts_visible и 🔒 приватные документы), поэтому
        ответ, собранный по личному документу одного сотрудника, не должен
        достаться коллеге из той же организации. None — общий контур (сводки),
        где выдача одинаковa для всех."""
        import hashlib

        raw = (f"{org_id}|{user_id if user_id is not None else '*'}|"
               f"{intent_override or ''}|{metric_override or ''}|{query.strip().casefold()}")
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _cache_get(self, key: str) -> QAOutcome | None:
        # кэш выключен (None) — но НЕ пустой: пустой OrderedDict ложен, и
        # проверка на «ложность» делала бы первый же _cache_put пустышкой
        if self._answer_cache is None:
            return None
        item = self._answer_cache.get(key)
        if item is not None:
            self._answer_cache.move_to_end(key)
        return item

    def _cache_put(self, key: str, outcome: QAOutcome) -> None:
        """Кэшируем только «текстовые» ответы модели (причины, ранжирование,
        выжимка из документов): они целиком помещаются в текст и повтор
        оплачивается дважды. Ответы, собранные кодом (факт, сравнение, прогноз),
        и так отдаются за доли секунды без вызова модели — кэшировать нечего.
        Так мы не теряем ни график, ни кнопки Excel у фактических ответов."""
        if self._answer_cache is None or not outcome.text:
            return
        if outcome.chart_png or outcome.table_metric_id is not None or outcome.clarify:
            return
        self._answer_cache[key] = outcome
        self._answer_cache.move_to_end(key)
        while len(self._answer_cache) > self.s.answer_cache_size:
            self._answer_cache.popitem(last=False)

    # ------------------------------------------------------------------
    async def answer(
        self, org_id: int, user_id: int, query: str, *, metric_override: str | None = None,
        intent_override: str | None = None,
    ) -> QAOutcome:
        cache_key = self._cache_key(org_id, user_id, query, metric_override, intent_override)
        cached = self._cache_get(cache_key)
        if cached is not None:
            log.info("Ответ взят из кэша — вызовов LLM и токенов: 0")
            return cached
        self._pending_usage.clear()  # не приписывать расход прошлого вопроса
        # Баланс ключа исчерпан (402): пока пауза не истекла, в API не ходим.
        # Иначе на каждый вопрос уходила бы пачка 402 в журнал провайдера, а
        # пользователь не понимал бы, почему ответ стал шаблонным.
        balance_down = not self._balance_ok()
        async with self.sessions() as session:
            try:
                if balance_down:
                    outcome = await self._answer(
                        session, org_id, user_id, query, metric_override,
                        intent_override=intent_override, llm_offline=True,
                    )
                else:
                    outcome = await self._answer(session, org_id, user_id, query, metric_override,
                                                 intent_override=intent_override)
                await self._note_balance_from_client(session)
                self._cache_put(cache_key, outcome)
                await self.flush_usage(session)
                await session.commit()
                return self._with_balance_notice(outcome)
            except LLMError as e:
                # Сюда доходят ошибки, которые не перехватили внутренние шаги
                # (композитор кончившийся баланс не глотает). «Кончились деньги»
                # отличается от сетевого сбоя: свой лог и своё событие в учёте.
                is_balance = isinstance(e, LLMBalanceError)
                await self._note_balance_from_client(session)
                if is_balance:
                    log.error("Баланс LLM API исчерпан — ответ шаблоном: %s", e)
                else:
                    log.warning("LLM недоступен: %s — отвечаю шаблоном", e)
                outcome = await self._answer(
                    session, org_id, user_id, query, metric_override,
                    intent_override=intent_override, llm_offline=True,
                )
                await self.flush_usage(session)
                await session.commit()
                return self._with_balance_notice(outcome)

    async def _note_balance_from_client(self, session: AsyncSession) -> None:
        """Записать событие, если клиент упёрся в 402 на этом вопросе.

        Событие берётся у самого клиента, а не из пойманного исключения: шаги
        вроде классификатора и Text-to-SQL глотают ошибку и продолжают работу,
        поэтому ловить исключение в одном месте недостаточно — иначе ответ
        молча деградирует до шаблона без предупреждения и без записи в учёт.
        """
        take = getattr(self._inner_llm(), "take_balance_error", None)
        error = take() if callable(take) else None
        if error is None:
            return
        await self._note_balance(session, reason=str(error))

    def _answer_needs_model(self, outcome: QAOutcome) -> bool:
        """True — ответ этого типа без модели теряет смысл (вывод по документам,
        ранжирование, SQL-выборка). Тогда о кончившемся балансе надо сказать."""
        payload_type = getattr(outcome, "payload_type", "")
        mode = (self.s.compose_mode or "auto").lower()
        if mode == "template":
            return False          # модель выключена самим пользователем
        if mode == "llm":
            return True           # он ожидает текст модели на каждый вопрос
        return payload_type not in TEMPLATE_FIRST_TYPES

    async def _note_balance(self, session: AsyncSession, *, reason: str) -> None:
        """Зафиксировать событие «баланс исчерпан»: журнал + счётчик в app_meta."""
        from .usage import add_balance_event, log_balance_event

        record = {"reason": reason, "model": self.s.llm_model}
        log_balance_event(self.s, record)
        try:
            await add_balance_event(session, record)
        except Exception as e:
            log.warning("Не удалось записать событие по балансу: %s", e)

    def _inner_llm(self):
        """Фактический клиент под маскирующей обёрткой (AnonymizingLLM)."""
        return getattr(self.llm, "_inner", self.llm)

    def _balance_ok(self) -> bool:
        check = getattr(self._inner_llm(), "balance_ok", None)
        return bool(check()) if callable(check) else True

    def _with_balance_notice(self, outcome: QAOutcome) -> QAOutcome:
        """Добавить предупреждение о закончившемся балансе — один раз на эпизод.

        Предупреждаем только там, где ответ без модели реально теряет смысл
        (вывод по документам, ранжирование, SQL): факт или сравнение и так
        отвечаются шаблоном, и «пополните баланс» к ним отношения не имеет.
        """
        if not self._answer_needs_model(outcome):
            return outcome
        take = getattr(self._inner_llm(), "take_balance_notice", None)
        if callable(take) and take():
            outcome.balance_notice = BALANCE_NOTICE
        return outcome

    async def _answer(
        self,
        session: AsyncSession,
        org_id: int,
        user_id: int,
        query: str,
        metric_override: str | None = None,
        *,
        llm_offline: bool = False,
        intent_override: str | None = None,
    ) -> QAOutcome:
        """QA-конвейер как граф агентов: классификация → резолв → ветвление
        по намерению → исполнитель → композитор. state['trace'] — маршрут."""
        graph = self._build_graph(session, org_id, user_id, query, metric_override,
                                  llm_offline, intent_override)
        state = await graph.run({"query": query, "org_id": org_id, "user_id": user_id})
        await audit(session, user_id, "question", f"[{state.get('intent', '?')}] {query}")
        log.debug("QA trace: %s", state.get("trace"))
        payload = state.get("payload") or {}
        outcome = QAOutcome(
            text=state.get("text", ""),
            clarify=state.get("clarify", []),
            chart_png=state.get("chart"),
            payload_type=str(payload.get("type") or ""),
        )
        metric = state.get("metric")
        if metric is not None and payload.get("type") in ("factual", "forecast", "breakdown", "compare"):
            outcome.table_metric_id = metric.id
            from .storage import ledger_months

            months = await ledger_months(session, org_id, metric.name.casefold())
            if months:
                outcome.ops_months = [(m["label"], round(m["total"], 2)) for m in months[-6:]]
        return outcome

    def _build_graph(
        self, session: AsyncSession, org_id: int, user_id: int, query: str,
        metric_override: str | None, llm_offline: bool,
        intent_override: str | None = None,
    ) -> AgentGraph:
        g = AgentGraph("qa")

        async def classify(st: dict) -> dict:
            # Кнопка «Прогноз/Динамика/Состав» уже сообщает и намерение, и
            # показатель — вызов модели здесь не нужен. На CPU-модели это
            # ~19 секунд ожидания ни за что (замер scripts/profile_latency).
            if intent_override and metric_override:
                st["intent"] = intent_override
                st["metric_query"] = metric_override
                st["years"] = []
                st["target_year"] = None
                st["wants_plan"] = bool(re.search(r"план|бюджет", query, re.I))
                st["scenario"] = None
                st["skipped_classify"] = True
                return st
            cls = await self._classify(query, llm_offline=llm_offline)
            st["intent"] = intent_override or cls.get("intent") or "factual"
            mq = cls.get("metric_query")
            st["metric_query"] = metric_override or (str(mq).strip() if mq else None) or query
            st["years"] = _safe_int_years(cls.get("years"))
            st["target_year"] = _safe_target_year(cls.get("target_year"), st["years"])
            st["wants_plan"] = bool(re.search(r"план|бюджет", query, re.I))
            # сценарий «что если» приходит от классификатора; если модель его
            # не вернула — извлекаем регуляркой, но ТОЛЬКО при явной условной
            # формулировке («если темпы упадут вдвое»), иначе обычный вопрос
            # «вырастет ли выручка?» получал бы подставной множитель ×1.5
            st["scenario"] = cls.get("scenario") or None
            return st

        async def resolve(st: dict) -> dict:
            metric, ambiguous = None, None
            if st["metric_query"]:
                metric, ambiguous = await self._resolve_metric(session, org_id, st["metric_query"])
            if metric is None and not ambiguous:
                # память диалога: «а за 2023?», «по нему?» — берём последний показатель
                mem = self.dialog_memory.get(user_id)
                # «2023?», «а в 2024» — уточнение к прошлому показателю; «сотрудники» — новый вопрос
                short = not re.search(r"[а-яёa-z]{3,}", st["metric_query"] or "")
                if mem and (FOLLOWUP_RE.search(query) or short):
                    metric = await metric_by_id(session, int(mem["metric_id"]))
                    if metric:
                        st["from_memory"] = True
            st["metric"], st["ambiguous"] = metric, ambiguous
            if metric is not None:
                self.dialog_memory[user_id] = {"metric_id": metric.id, "name": metric.name}
            return st

        async def explain(st: dict) -> dict:
            st["payload"] = await self._explain(session, org_id, user_id, query)
            return st

        async def forecast(st: dict) -> dict:
            st["payload"] = await self._forecast(
                session, org_id, user_id, query, st["metric"], st["target_year"],
                scenario=st.get("scenario"),
            )
            # график прогноза картинкой
            if self.s.send_charts and st["payload"].get("type") == "forecast":
                from .charts import render_forecast_chart

                st["chart"] = await asyncio.to_thread(
                    render_forecast_chart,
                    st["payload"].get("history", []),
                    st["payload"].get("forecast", {}),
                    (st["payload"].get("metric") or {}).get("currency"),
                )
            return st

        async def compare(st: dict) -> dict:
            st["payload"] = await self._compare(
                session, org_id, user_id, st["metric"], st["years"], plan_mode=st["wants_plan"]
            )
            return st

        async def rank(st: dict) -> dict:
            st["payload"] = await self._rank(session, org_id, user_id, st["metric"])
            return st

        async def breakdown(st: dict) -> dict:
            st["payload"] = await self._breakdown(session, org_id, user_id, st["metric"], st["years"])
            return st

        async def factual(st: dict) -> dict:
            st["payload"] = await self._factual(session, org_id, user_id, st["metric"], st["years"])
            st["try_sql"] = st["payload"].get("type") == "nodata" and st["metric"] is None
            return st

        async def text2sql(st: dict) -> dict:
            sql_payload = await self._text_to_sql(session, org_id, user_id, query)
            if sql_payload:
                st["payload"] = sql_payload
            else:
                # показатель не распознан и SQL не помог — ищем ответ в тексте документов
                st["payload"] = await self._explain(session, org_id, user_id, query)
            return st

        async def compose(st: dict) -> dict:
            st["text"] = await self._compose(query, st.get("payload", {}), llm_offline=llm_offline)
            return st

        async def help_node(st: dict) -> dict:
            st["text"] = self._help_text()
            return st

        async def clarify_node(st: dict) -> dict:
            st["text"] = "Уточните, пожалуйста, какой показатель вы имеете в виду:"
            st["clarify"] = st.get("ambiguous") or []
            return st

        def route_intent(st: dict) -> str:
            return st["intent"] if st["intent"] in ("smalltalk", "explain") else "resolve"

        def route_resolve(st: dict) -> str:
            if st.get("ambiguous") and st["intent"] != "rank":
                return "clarify"  # ранжирование работает по всем показателям — уточнять нечего
            return st["intent"] if st["intent"] in ("forecast", "compare", "rank", "breakdown") else "factual"

        def route_factual(st: dict) -> str:
            return "text2sql" if st.get("try_sql") else "compose"

        return (
            g.node("classify", classify)
            .node("resolve", resolve)
            .node("explain", explain)
            .node("forecast", forecast)
            .node("compare", compare)
            .node("rank", rank)
            .node("breakdown", breakdown)
            .node("factual", factual)
            .node("text2sql", text2sql)
            .node("compose", compose)
            .node("help", help_node)
            .node("clarify", clarify_node)
            .set_entry("classify")
            .branch("classify", route_intent, {"smalltalk": "help", "explain": "explain", "__default__": "resolve"})
            .branch("resolve", route_resolve, {
                "clarify": "clarify", "forecast": "forecast", "compare": "compare",
                "rank": "rank", "breakdown": "breakdown", "__default__": "factual",
            })
            .branch("factual", route_factual, {"text2sql": "text2sql", "__default__": "compose"})
            .edge("explain", "compose")
            .edge("forecast", "compose")
            .edge("compare", "compose")
            .edge("rank", "compose")
            .edge("breakdown", "compose")
            .edge("text2sql", "compose")
        )

    # ------------------------------------------------------------ классификация
    # Детерминированные паттерны («из чего состоит», «на сколько выросла»,
    # «сравни», «топ-5», «прогноз») надёжнее маленькой локальной модели:
    # её вердикт по таким формулировкам флуктуирует. Явное попадание правила
    # перекрывает интент LLM; тонкие случаи (explain/factual) остаются модели.
    _STRONG_INTENTS = frozenset({"breakdown", "compare", "rank", "forecast"})

    async def _classify(self, query: str, *, llm_offline: bool = False) -> dict:
        from .llm import _mock_classify

        if llm_offline or not self.s.classify_with_llm:
            # Правила дешевле и на этой модели точнее: классификатор тратит
            # ~18 секунд на CPU ради ~40 токенов JSON, а его вердикт всё равно
            # перекрывается правилами для compare/rank/breakdown/forecast.
            return _mock_classify(query)
        try:
            raw = await self.llm.chat(
                [
                    {"role": "system", "content": CLASSIFIER_SYSTEM},
                    {"role": "user", "content": query},
                ],
                json_mode=True,
                temperature=0.0,
                # JSON по схеме укладывается в 120 токенов; прежние 300 модель
                # тратила на пояснения вокруг ответа, а это прямая оплата
                max_tokens=self.s.classifier_max_tokens,
                task="classify",
            )
            parsed = parse_json_block(raw)
            if parsed and "intent" in parsed:
                mock_intent = _mock_classify(query).get("intent")
                if mock_intent in self._STRONG_INTENTS and parsed.get("intent") != mock_intent:
                    log.info(
                        "Интент скорректирован правилом: %s -> %s (%s)",
                        parsed.get("intent"), mock_intent, query[:60],
                    )
                    parsed["intent"] = mock_intent
                return parsed
        except LLMError:
            raise
        except Exception as e:
            log.warning("Классификация не удалась (%s), использую правила", e)
        return _mock_classify(query)

    # ------------------------------------------------------------ резолв метрики
    async def _resolve_metric(self, session: AsyncSession, org_id: int, metric_query: str):
        exact = await find_metric_by_name(session, org_id, metric_query.strip().casefold())
        if exact:
            return exact, None
        from .embeddings import _tokens
        from .query_expansion import expand_query

        # лексический резолв до векторов: все слова имени показателя (или его
        # синонима) есть в вопросе — с учётом падежей через префиксы токенов:
        # «прогноз по аренде спецтехники» -> «аренда спецтехники»,
        # «итого операционных расходов» -> «операционные расходы итого».
        # Векторы одинаково «видят» «аренда спецтехники москва» и переспрашивают.
        q_toks = set(_tokens(metric_query))
        if q_toks:
            from sqlalchemy import select as sa_select

            from .storage import MetricSynonym

            metrics = [m for m in await org_metrics(session, org_id) if m.kind != "identifier"]
            syn_rows = (await session.execute(
                sa_select(MetricSynonym.metric_id, MetricSynonym.text)
                .join(Metric, Metric.id == MetricSynonym.metric_id).where(Metric.org_id == org_id)
            )).all()
            names: list[tuple[str, int]] = [(m.name, m.id) for m in metrics] + [(t, mid) for mid, t in syn_rows]
            best_m, best_len = None, 0
            for name, mid in names:
                nt = set(_tokens(name))
                if nt and nt <= q_toks and len(nt) > best_len:
                    best_m, best_len = mid, len(nt)
            if best_m is not None:
                return await metric_by_id(session, best_m), None

        # мультивариантный резолв: исходная формулировка + обогащённые варианты
        # («ндс» -> «ндс налог на добавленную стоимость»)
        best: dict[int, dict] = {}
        for variant in expand_query(metric_query):
            qvec = await self.emb.embed_query(variant)
            for cand in await vector_search(session, org_id=org_id, query_vec=qvec, kind="metric", k=4):
                cid = int(cand["id"])
                if cid not in best or cand["score"] > best[cid]["score"]:
                    best[cid] = cand
        candidates = sorted(best.values(), key=lambda c: c["score"], reverse=True)[:4]
        if self.emb.provider == "hash":
            # hash-векторы «похожи» и при коллизиях: без общего слова совпадения нет
            qt = set(_tokens(metric_query))
            candidates = [c for c in candidates if qt & set(_tokens(c["name"]))]
        if not candidates:
            return None, None
        top, second = candidates[0], (candidates[1] if len(candidates) > 1 else None)
        thr = self.s.metric_match_threshold
        if top["score"] >= thr and (second is None or top["score"] - second["score"] > 0.05):
            m = await metric_by_id(session, int(top["id"]))
            if m:
                return m, None
        if top["score"] >= 0.35:
            return None, [c["name"] for c in candidates[:3]]
        return None, None

    # ------------------------------------------------------------ исполнители
    def _metric_block(self, metric, rows: list[dict]) -> dict:
        unit = (rows[-1]["unit"] if rows else None) or (metric.unit if metric else None)
        currency = (rows[-1]["currency"] if rows else None) or (metric.currency if metric else None)
        return {"name": metric.name if metric else "—", "unit": unit, "currency": currency}

    def _rows_in_years(self, rows: list[dict], years: list[int]) -> list[dict]:
        if not years:
            return rows
        picked = [r for r in rows if any(y in range(r["period_start"].year, r["period_end"].year + 1) for y in years)]
        return picked or rows

    def _history(self, rows: list[dict]) -> list[dict]:
        return [
            {
                "label": r["period_label"],
                "value": r["value"],
                "source": f"{r['document_name']} · {r['sheet']} · {r['cell_ref']}",
            }
            for r in rows
        ]

    async def _factual(self, session, org_id, user_id, metric, years) -> dict:
        if metric is None:
            return {"type": "nodata", "query": "показатель не распознан"}
        rows = await series_for_metric(session, org_id, metric.id, user_id=user_id)
        rows = self._rows_in_years(rows, years)
        if not rows:
            return {"type": "nodata", "query": metric.name}
        return {"type": "factual", "metric": self._metric_block(metric, rows), "history": self._history(rows)}

    async def _compare(self, session, org_id, user_id, metric, years, plan_mode: bool = False) -> dict:
        if metric is None:
            return {"type": "nodata", "query": "показатель не распознан"}
        if plan_mode:
            # variance-анализ план/факт (практика Datarails FP&A Genius / Copilot for Finance)
            fact_rows = self._rows_in_years(
                await series_for_metric(session, org_id, metric.id, variant="fact", user_id=user_id), years
            )
            plan_rows = await series_for_metric(session, org_id, metric.id, variant="plan", user_id=user_id)
            plan_by_end = {r["period_end"]: r for r in plan_rows}
            pairs = [(f, plan_by_end[f["period_end"]]) for f in fact_rows if f["period_end"] in plan_by_end]
            if not pairs:
                return {"type": "nodata", "query": f"{metric.name}: нет плановых данных — загрузите отчёт с колонкой «план»"}
            p_last, f_last = pairs[-1][1], pairs[-1][0]
            dev = (f_last["value"] - p_last["value"]) / p_last["value"] * 100 if p_last["value"] else None
            history = [
                {**self._history([p_last])[0], "label": f"{p_last['period_label']} (план)"},
                {**self._history([f_last])[0], "label": f"{f_last['period_label']} (факт)"},
            ]
            return {
                "type": "compare",
                "plan_mode": True,
                "metric": self._metric_block(metric, fact_rows),
                "history": history,
                "computed": {
                    "deviation_pct": dev,
                    "abs_change": f_last["value"] - p_last["value"],
                    "period": f_last["period_label"],
                },
            }
        rows = await series_for_metric(session, org_id, metric.id, user_id=user_id)
        if len(rows) < 2:
            return {"type": "nodata", "query": metric.name}
        if len(years) >= 2:
            # «с 2023 по 2025» — LLM нередко возвращает и промежуточные годы
            # [2023, 2024, 2025]; границы сравнения — первый и последний
            a = [r for r in rows if years[0] in range(r["period_start"].year, r["period_end"].year + 1)]
            b = [r for r in rows if years[-1] in range(r["period_start"].year, r["period_end"].year + 1)]
            rows_pair = ([a[-1]] if a else []) + ([b[-1]] if b else [])
        else:
            rows_pair = [rows[0], rows[-1]]
        if len(rows_pair) < 2:
            return {"type": "nodata", "query": metric.name}
        first, last = rows_pair[0], rows_pair[-1]
        change = last["value"] - first["value"]
        pct = change / first["value"] * 100 if first["value"] else None
        return {
            "type": "compare",
            "metric": self._metric_block(metric, rows),
            "history": self._history(rows_pair),
            "computed": {"change_pct": pct, "abs_change": change},
        }

    async def _rank(self, session, org_id, user_id, metric) -> dict:
        candidates = []
        if metric:
            candidates = await metric_children(session, metric.id)
        if not candidates:
            all_m = await org_metrics(session, org_id)
            candidates = [m for m in all_m if m.kind in ("revenue", "expense") and m.id != (metric.id if metric else None)]
        ranking = []
        for m in candidates[:40]:
            rows = await series_for_metric(session, org_id, m.id, user_id=user_id)
            if len(rows) < 2:
                continue
            first, last = rows[0], rows[-1]
            if first["period_type"] != last["period_type"] or first["value"] == 0:
                continue
            ranking.append(
                {
                    "name": m.name,
                    "growth_pct": (last["value"] - first["value"]) / abs(first["value"]) * 100,
                    "label_from": first["period_label"],
                    "label_to": last["period_label"],
                }
            )
        if not ranking:
            return {"type": "nodata", "query": "нет показателей с историей минимум за два периода"}
        ranking.sort(key=lambda r: r["growth_pct"], reverse=True)
        return {"type": "rank", "ranking": ranking[:10]}

    async def _breakdown(self, session, org_id, user_id, metric, years) -> dict:
        """Drill-down: состав показателя по дочерним позициям (практика
        FP&A-систем — детализация итога до позиций с долями)."""
        if metric is None:
            return {"type": "nodata", "query": "показатель не распознан"}
        kids = await metric_children(session, metric.id)
        if not kids and metric.parent_id:
            # итог обычно «сидит» в разделе рядом с детьми — берём одних родителей
            siblings = await metric_children(session, metric.parent_id)
            kids = [m for m in siblings if m.id != metric.id]
        if not kids:
            return {"type": "nodata", "query": f"у показателя «{metric.name}» нет детализации по позициям"}
        series_map: dict[str, dict] = {}
        for ch in kids:
            rows = await series_for_metric(session, org_id, ch.id, user_id=user_id)
            if rows:
                series_map[ch.name] = {r["period_end"]: r for r in rows}
        if not series_map:
            return {"type": "nodata", "query": metric.name}
        ends = {e for m in series_map.values() for e in m}
        if years:
            ends = {e for e in ends if any(y in (e.year,) for y in years)}
        if not ends:
            return {"type": "nodata", "query": metric.name}
        # последний период, в котором есть хотя бы 2 позиции
        from collections import Counter

        counts = Counter(e for m in series_map.values() for e in m)
        target = max((e for e in ends if counts[e] >= min(2, len(series_map))), default=max(ends))
        items = []
        for name, m in series_map.items():
            if target not in m:
                continue
            r = m[target]
            items.append({"name": name, "value": r["value"], "source": f"{r['document_name']} · {r['sheet']} · {r['cell_ref']}"})
        parent_rows = await series_for_metric(session, org_id, metric.id, user_id=user_id)
        total = next((r["value"] for r in parent_rows if r["period_end"] == target), None)
        if total is None:
            total = sum(i["value"] for i in items)
        for i in items:
            i["share_pct"] = i["value"] / total * 100 if total else None
        items.sort(key=lambda i: i["value"], reverse=True)
        return {
            "type": "breakdown",
            "metric": self._metric_block(metric, parent_rows),
            "period_label": series_map[items[0]["name"]][target]["period_label"],
            "total": total,
            "items": items[:12],
        }

    async def _explain(self, session, org_id, user_id, query: str) -> dict:
        found = await hybrid_search(session, self.emb, org_id, query, k=max(self.s.rag_top_k * 2, 10), user_id=user_id)
        found = await self.reranker.rerank(query, found, k=self.s.rag_top_k)
        context = []
        if found:
            from sqlalchemy import select

            from .storage import Document

            doc_ids = {f["document_id"] for f in found}
            docs = {
                d.id: d.original_name
                for d in (await session.scalars(select(Document).where(Document.id.in_(doc_ids)))).all()
            }
            from .embeddings import _tokens
            from .query_expansion import expand_query

            focus_query = " ".join(expand_query(query))
            qtoks = set(_tokens(focus_query))
            # фрагмент без единого слова вопроса — не ответ; для настоящей модели
            # допускаем и чисто семантическое совпадение
            found = [
                f for f in found
                if qtoks & set(_tokens(f["body"]))
                or (self.emb.provider != "hash" and (f.get("semantic_score") or 0) >= 0.5)
            ]
            for f in found:
                src = f"стр. {f['page']}" if f.get("page") else ""
                context.append({"text": focus_snippet(f["body"], focus_query), "source": f"{docs.get(f['document_id'], '?')} {src}".strip()})
        return {"type": "explain", "context": context}

    async def _forecast(self, session, org_id, user_id, query, metric, target_year,
                        scenario: dict | None = None) -> dict:
        if metric is None:
            return {"type": "nodata", "query": "показатель не распознан"}
        rows = await series_for_metric(session, org_id, metric.id, user_id=user_id)
        points = prepare_series(rows)
        if len(points) < 2:
            return {"type": "nodata", "query": metric.name}
        if scenario is None:
            # резервный разбор «что если» регуляркой — только когда в вопросе
            # есть явное условие; см. classify()
            from .llm import _mock_scenario

            if _SCENARIO_RE.search(query):
                scenario = _mock_scenario(query)
        mult = (scenario or {}).get("growth_multiplier")
        fc = forecast_fn(points, target_year=target_year, growth_multiplier=mult)
        if fc.get("error"):
            return {"type": "nodata", "query": metric.name, "notes": [fc["error"]]}
        # Производные для ответа считаем сами и отдаём модели готовыми. Без этого
        # она считала их сама («прогноз на 9,8% выше факта», «интервал шириной
        # 1,56 млн»), верификатор отклонял числа как отсутствующие в ДАННЫХ, и
        # верный ответ уходил в шаблон. Считать их кодом — ещё и точнее.
        computed = dict(fc.get("computed") or {})
        fact = float(points[-1].value or 0)
        base = float(fc.get("base") or 0)
        low = float(fc.get("low") or 0)
        high = float(fc.get("high") or 0)
        if fact:
            computed["forecast_vs_fact_pct"] = round((base - fact) / fact * 100, 2)
            computed["forecast_vs_fact_abs"] = round(base - fact, 2)
        if low and high:
            computed["interval_width_abs"] = round(high - low, 2)
            computed["interval_width_pct"] = round((high - low) / base * 100, 2) if base else None
        base_wo = fc.get("base_without_scenario")
        if mult is not None and base_wo:
            computed["scenario_effect_pct"] = round((base - float(base_wo)) / float(base_wo) * 100, 2)
            computed["scenario_effect_abs"] = round(base - float(base_wo), 2)
        # Текстовый контекст для прогноза не ищем: числа прогноза считает код,
        # а поиск добавлял 9 эмбеддингов и заметную задержку, принося в ответ
        # карточку самого же показателя («⚠️ Контекст: «Карточка п…»).
        payload = {
            "type": "forecast",
            "metric": self._metric_block(metric, rows),
            "history": self._history(points_to_rows(points)),
            "forecast": fc,
            "computed": {k: v for k, v in computed.items() if v is not None},
            "context": [],
        }
        return payload

    # ------------------------------------------------------------ Text-to-SQL
    async def _text_to_sql(self, session: AsyncSession, org_id: int, user_id: int, query: str) -> dict | None:
        try:
            raw = await self.llm.chat(
                [
                    {"role": "system", "content": SQL_SYSTEM},
                    {"role": "user", "content": f"ВОПРОС: {query}"},
                ],
                temperature=0.0,
                max_tokens=self.s.sql_max_tokens,
                task="sql",
            )
        except LLMError:
            return None
        sql = re.sub(r"```(?:sql)?|```", "", raw).strip().rstrip(";").strip()
        if not sql_is_safe(sql):
            log.info("Text-to-SQL отклонён guard'ом: %s", sql[:200])
            return None
        sql += " LIMIT 100" if "limit" not in sql.lower() else ""

        # организация и приватность подставляются детерминированно после генерации:
        # facts_visible -> подзапрос с :org/:uid; сгенерированный SQL не может их обойти
        sql_exec = re.sub(r"\bfacts_visible\b", visible_facts_subquery(session.bind.dialect.name), sql, flags=re.I)

        # сгенерированный SQL выполняется на выделенном соединении под жёстким
        # read-only: запись невозможна даже при дыре в текстовом фильтре, а
        # режим снимается до возврата соединения в пул (иначе следующие записи
        # падали бы с «attempt to write a readonly database»)
        async with self.sessions() as ro:
            try:
                result = await run_readonly(ro, text(sql_exec), {"uid": user_id, "org": org_id})
                rows = result.fetchmany(100)
                cols = list(result.keys())
            except Exception as e:
                log.info("Text-to-SQL не выполнен (read-only): %s", e)
                return None
            finally:
                await ro.rollback()
        if not rows:
            return {"type": "nodata", "query": query}
        data = [[str(c) for c in cols]] + [
            [fmt_number(float(v)) if isinstance(v, (int, float)) else str(v) for v in row]
            for row in rows[:10]
        ]
        return {
            "type": "sql_result",
            "query": query,
            "table": render_table(data),
            "rows_total": len(rows),
        }

    # ------------------------------------------------------------ композитор
    async def _compose(self, query: str, payload: dict, *, llm_offline: bool = False) -> str:
        payload = dict(payload)
        # бюджет контекста: не заливаем весь индекс в промпт
        ctx_items = self.s.composer_context_items or MAX_CONTEXT_ITEMS
        hist_rows = self.s.composer_history_rows or MAX_HISTORY_ROWS
        if payload.get("context") and len(payload["context"]) > ctx_items:
            payload["context"] = payload["context"][:ctx_items]
        if payload.get("history") and len(payload["history"]) > hist_rows:
            payload["history"] = payload["history"][:hist_rows]
        template = render_answer(payload)
        # В промпт уходит сжатая копия: полный template_hint дублировал данные
        # и раздувал промпт втрое. Замер (scripts/profile_llm_prompt) показал,
        # что на CPU доминирует именно размер промпта: 386 токенов — 30 с,
        # 33 токена — 1.5 с. Все числа при сжатии сохраняются.
        blob = json.dumps(_compact_for_llm(payload, self.s), ensure_ascii=False, default=str)
        # маскирование выполняет AnonymizingLLM на уровне клиента — здесь его
        # больше нет, иначе текст маскировался бы дважды
        from .llm import MockLLM

        if llm_offline or type(self.llm) is MockLLM:
            return template  # без внешней модели ответ — шаблон из тех же данных, без масок [ИНН]

        # Режим auto: там, где код уже посчитал и сверстал ответ (таблица, доли,
        # прогноз, отклонение от плана), модель не добавляет ни одной цифры —
        # только пересказ, за который пользователь платит десятками секунд
        # ожидания на CPU. Замер: вопрос с моделью 13 с, кнопка «Прогноз» 37 с
        # (два вызова из-за повтора после верификатора).
        mode = (self.s.compose_mode or "auto").lower()
        if mode == "template" or (mode == "auto" and payload.get("type") in TEMPLATE_FIRST_TYPES):
            return template

        base_msg = f"ВОПРОС: {query}\n\nДАННЫЕ (JSON):\n```json\n{blob}\n```"
        # Контур генерация -> критика -> повтор: верифицирующий агент проверяет,
        # что все числа ответа присутствуют в данных; иначе повтор с замечанием,
        # после исчерпания попыток — шаблонный ответ (grounded by construction).
        # max_tokens уменьшен вдвое против прежних 900: локальные модели 1–2B
        # любят «лить воду», а верификатор всё равно отклонит лишние числа.
        foreign: list[float] = []
        misattributed: list[str] = []
        for attempt in range(self.s.answer_retries + 1):
            user_msg = base_msg + (
                self.verifier.correction_note(foreign, misattributed) if attempt > 0 else ""
            )
            try:
                answer = await self.llm.chat(
                    [
                        {"role": "system", "content": COMPOSER_SYSTEM},
                        {"role": "user", "content": user_msg},
                    ],
                    # ограничение прямо влияет и на задержку, и на оплату:
                    # COMPOSER_MAX_TOKENS — это верхняя граница стоимости ответа
                    max_tokens=self.s.composer_max_tokens,
                    task="compose",
                )
            except LLMBalanceError:
                # Композитор НЕ перехватывает кончившийся баланс: событие нужно
                # записать и показать пользователю предупреждение. Иначе ответ
                # молча деградирует до шаблона, и никто не понимает почему.
                raise
            except LLMError:
                log.warning("LLM недоступен — использую шаблонный ответ")
                return template
            answer = answer.strip()
            if not answer:
                continue
            # payload передаётся в проверку: модель обязана брать числа из
            # ДАННЫХ, а шаблон их переформатирует (3 500 000 -> «3,50 млн ₽»)
            review = await self.verifier.review(answer, template, payload)
            if review.ok:
                if attempt:
                    log.info("Ответ принят с попытки %d", attempt + 1)
                return answer
            log.info(
                "Верификатор отклонил попытку %d/%d: посторонние=%s, периоды=%s",
                attempt + 1, self.s.answer_retries + 1,
                review.foreign_numbers[:5], review.misattributed[:2],
            )
            foreign = review.foreign_numbers
            misattributed = review.misattributed
        log.warning("Верификатор отклонил ответ после %d попыток — использую шаблон",
                    self.s.answer_retries + 1)
        return template

    def _help_text(self) -> str:
        return (
            "Я финансовый ассистент. Умею:\n"
            "• отвечать на вопросы по загруженным отчётам — «какая выручка за 2024?»;\n"
            "• сравнивать периоды — «на сколько выросла аренда с 2023 по 2025?»;\n"
            "• ранжировать — «какие позиции выросли сильнее всего?»;\n"
            "• искать по тексту документов — «что говорится о рисках?»;\n"
            "• прогнозировать — «прогноз по позиции X на 2026», «а если темпы упадут вдвое?»;\n"
            "• вести журнал расходов — «расход: 1500 кофе», «доход: 50000 зарплата»;\n"
            "  /ledger — выгрузить Excel, /undo — отменить последнюю запись.\n\n"
            "Просто пришлите файл отчёта (xlsx, csv, pdf, docx) и задавайте вопросы."
        )


def points_to_rows(points) -> list[dict]:
    return [
        {
            "period_type": p.ptype,
            "period_label": p.label,
            "period_start": p.start,
            "period_end": p.end,
            "value": p.value,
            "unit": None,
            "currency": None,
            "sheet": "",
            "cell_ref": "",
            "document_name": p.source.split(" · ")[0] if p.source else "",
        }
        for p in points
    ]
