"""Конфигурация приложения: все параметры берутся из переменных окружения / .env."""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- Telegram ---
    bot_token: str = ""
    # Список разрешённых Telegram ID через запятую. Пусто -> вход только по коду регистрации.
    allowed_user_ids: str = ""
    reg_code: str = "changeme"
    # Прокси для long polling (если api.telegram.org недоступен напрямую),
    # например: http://127.0.0.1:10809 или socks5://127.0.0.1:1080
    telegram_proxy: str = ""
    # Ходить на api.telegram.org только по IPv4 (помогает при сломанном IPv6,
    # когда домен резолвится, но соединение висит)
    telegram_force_ipv4: bool = True
    # Таймаут пробы связи с Telegram при старте, секунды
    telegram_probe_timeout: int = 45

    # --- Векторный бэкенд ---
    # auto (по умолчанию: pgvector на PostgreSQL / перебор на SQLite) | qdrant
    # При qdrant: зеркалятся неприватные чанки, личные ищутся через SQL-ветку.
    vector_backend: str = "auto"
    qdrant_url: str = ""       # например http://localhost:6333
    qdrant_api_key: str = ""

    # --- Хранилище и файлы ---
    data_dir: Path = Path("data")
    # Продакшен: postgresql+asyncpg://rag:rag@localhost:5432/rag
    # Разработка: sqlite+aiosqlite:///./data/app.db (векторный поиск тогда brute-force по косинусу)
    database_url: str = "sqlite+aiosqlite:///./data/app.db"
    max_file_mb: int = 20
    allowed_extensions: str = "xlsx,xls,csv,pdf,docx,html,htm,ods,png,jpg,jpeg,tiff"
    # Лимиты распаковки архивных форматов (xlsx/docx/ods — это zip): 20 МБ
    # сжатого файла легко разворачиваются в десятки ГБ и вешают процесс.
    max_uncompressed_mb: int = 300
    max_archive_entries: int = 5000
    # Потолок страниц PDF: каждая страница — рендер и, при включённом OCR,
    # отдельный вызов vision-модели
    max_pdf_pages: int = 500

    # --- LLM: внешний API генерации (вариант 2 — всё локально, кроме генерации) ---
    # openai_compatible | gigachat | anthropic | mock (offline-режим для тестов и демо)
    llm_provider: str = "mock"
    # OpenAI-совместимый endpoint: OpenAI, OpenRouter, DeepSeek, YandexGPT (v1), локальный vLLM/Ollama
    llm_api_base: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 1500
    llm_timeout_s: float = 90.0
    # Режим рассуждений провайдера (DeepSeek: thinking включён ПО УМОЛЧАНИЮ).
    # "disabled" — выключить; "low"/"high" — включить с усилием.
    # Для наших задач рассуждения не нужны, а стоят дорого: они идут в
    # completion_tokens (оплачиваются) и съедают max_tokens, из-за чего при
    # жёстком лимите ответ приходит ПУСТЫМ. Включённый режим к тому же
    # игнорирует temperature.
    llm_thinking: str = "disabled"

    # --- Экономия токенов (внешний API платный, служебные шаги — самые частые) ---
    # Дешёвая модель для шагов, которые не формулируют ответ: классификация,
    # реранкинг, генерация SQL. Композитор всегда работает на LLM_MODEL.
    # Пусто — одна модель на всё (поведение прежних версий).
    llm_model_small: str = ""
    # Список задач для дешёвой модели через запятую. compose исключается всегда.
    llm_model_small_tasks: str = "classify,rerank,sql"
    # Повторы при 429/5xx и сетевых сбоях: один рейт-лимит больше не отбрасывает
    # вопрос на шаблонный ответ
    llm_retries: int = 3
    llm_retry_base_s: float = 1.0     # экспоненциально: 1, 2, 4 с
    llm_retry_max_s: float = 20.0     # потолок ожидания по Retry-After
    # Журнал расхода токенов (JSONL). Пусто -> {DATA_DIR}/llm_usage.jsonl
    llm_usage_log: str = ""
    # Пауза после ответа «баланс исчерпан» (HTTP 402): пока не истекла, к API не
    # обращаемся вовсе — ответы собираются шаблоном, пользователю выдаётся
    # предупреждение. Проверка баланса иначе стоила бы одной ошибки на вопрос.
    # 0 — не делать паузу (пробовать каждый раз).
    llm_balance_retry_s: int = 300
    # Цена за 1 млн токенов для оценки расхода: "модель=вход/выход,..."
    # например  "deepseek-flash=1/4"  (¥ за 1 млн, свободный тариф DeepSeek)
    llm_prices: str = ""
    # Баланс ключа, в тех же деньгах, что и LLM_PRICES (например 33 для 33 ¥).
    # Нужен только для строки «осталось / на сколько ещё хватит» в /usage.
    llm_balance: float = 0.0
    # Сколько чистых (не замаскированных) ответов помнить, чтобы не платить за
    # повторный вопрос дважды. 0 — кэш выключен.
    answer_cache_size: int = 64
    # Лимит токенов на служебные шаги: JSON-ответы короткие, больше не нужно
    classifier_max_tokens: int = 120
    sql_max_tokens: int = 400
    rerank_max_tokens: int = 600
    # Сколько кандидатов и символов фрагмента уходит в LLM-реранкер
    rerank_max_items: int = 8
    rerank_snippet_chars: int = 200
    # Бюджет промпта композитора: сколько фрагментов документов и сколько
    # символов каждого попадает в промпт ответа. Именно этот блок раздувает
    # входной промпт сильнее всего: 5 фрагментов × 200 символов ≈ 450 токенов
    # входа на каждый вопрос «что говорится в документах». Уменьшение —
    # прямая экономия; слишком сильное режет качество вывода по документам.
    composer_context_items: int = 5
    composer_context_chars: int = 200
    # Сколько строк истории уходит в промпт (ряды длинных помесячных рядов)
    composer_history_rows: int = 24

    # --- GigaChat (собственный протокол: OAuth + REST) ---
    gigachat_api_base: str = "https://gigachat.devices.sberbank.ru/api/v1"
    gigachat_oauth_url: str = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    gigachat_client_id: str = ""
    gigachat_client_secret: str = ""
    gigachat_scope: str = "GIGACHAT_API_PERS"
    gigachat_insecure: bool = False  # True отключает проверку TLS — только для отладки

    # --- Anthropic (нативный протокол, с prompt caching) ---
    anthropic_api_base: str = "https://api.anthropic.com"
    anthropic_api_key: str = ""
    anthropic_version: str = "2023-06-01"
    # Помечать статический системный промпт и блок данных cache_control (кэш
    # переиспользуется при повторах верификатора и повторных вопросах)
    prompt_cache: bool = True

    # --- Эмбеддинги: выполняются ЛОКАЛЬНО ---
    # sentence_transformers (CPU, e5/bge) | api (OpenAI-совместимый endpoint, напр. локальный vLLM) | hash (dev/test)
    embeddings_provider: str = "hash"
    embeddings_dim: int = 768
    embeddings_local_model: str = "intfloat/multilingual-e5-large"
    # auto — CUDA-видеокарта, если есть (fp16), иначе CPU; можно принудительно cpu | cuda
    embeddings_device: str = "auto"
    embeddings_api_base: str = ""
    embeddings_api_key: str = ""
    embeddings_api_model: str = ""
    # Запрещать внешний embeddings-endpoint. Векторизация — самая объёмная
    # операция по тексту: через платный API это заметные деньги за каждый
    # загруженный документ, а локальная модель считает то же бесплатно.
    # Осознанное исключение (например, нет ОЗУ под модель) — ALLOW_REMOTE_EMBEDDINGS=true.
    require_local_embeddings: bool = True

    # --- Ручной ввод расходов/доходов («расход: 1500 кофе») ---
    # Запись дублируется в Excel-журнал и в показатели «личные расходы/доходы»
    expense_metric_name: str = "личные расходы"
    income_metric_name: str = "личные доходы"
    # Путь к Excel-журналу; пусто -> {DATA_DIR}/расходы.xlsx
    expense_journal_file: str = ""

    # --- Безопасность ---
    anonymize_prompts: bool = True  # маскировать имена организаций/ИНН/реквизиты перед отправкой в LLM API
    # Названия организаций для маскирования через запятую. Пусто — берутся
    # названия организаций из БД, чтобы реальные имена не уходили в промпт.
    anonymize_org_names: str = ""

    # --- Графики и регламентные дайджесты ---
    send_charts: bool = True   # график прогноза картинкой в ответе
    digest_hour: int = 9       # час (локальный) ежедневной сводки для подписчиков

    # --- OCR сканов ---
    # Локальный Tesseract ставится отдельно (пакет tesseract-ocr rus+eng + pip install pytesseract).
    # Опциональный fallback — распознавание через vision-модель LLM API.
    # ВНИМАНИЕ: при этом изображения страниц уходят во внешний API!
    ocr_vlm_enabled: bool = False
    ocr_vlm_model: str = ""  # пусто -> используется LLM_MODEL

    # --- RAG / прогноз ---
    chunk_words: int = 300
    rag_top_k: int = 6
    metric_match_threshold: float = 0.55

    # --- Reranking (перез ранжирование после гибридного поиска) ---
    # auto (по умолчанию: с API — LLM-оценка, без API — офлайн-эвристики)
    # | llm | crossencoder (локальный кросс-энкодер, pip install sentence-transformers) | none
    reranker: str = "auto"
    rerank_model: str = "BAAI/bge-reranker-base"

    # --- Агент-верификатор ответов (grounding) ---
    # Проверяет, что все числа в ответе LLM присутствуют в данных; иначе —
    # повтор генерации с замечанием, затем шаблонный ответ.
    verify_answers: bool = True
    # Разрешать производные числа (разности и проценты роста) в прогнозе и
    # план/факте: там модель обязана считать «на сколько отличается» сама, и без
    # этого допуска корректные ответы отклонялись. Считаются точно из данных,
    # поэтому выдуманное значение по-прежнему не проходит.
    verify_allow_derived: bool = True
    # 2 попытки: при 0 первый же отказ верификатора (частая ложная тревога на
    # слабых моделях) сразу отдавал шаблон, и работа LLM пропадала
    answer_retries: int = 2
    # Спрашивать ли LLM о намерении и годе. Определение намерения уже
    # выполняется правилами (они же перекрывают вердикт модели для compare/
    # rank/breakdown/forecast), а на CPU классификатор стоит ~18 секунд за
    # ~40 токенов JSON. False — отвечаем заметно быстрее.
    classify_with_llm: bool = False
    # Сколько токенов максимум разрешено композитору. На CPU генерация идёт
    # последовательно (~10 токенов/с), поэтому это прямое время ожидания.
    composer_max_tokens: int = 200
    # Кого пускать к формулировке ответа:
    #   auto (по умолчанию) — для ответов, которые код уже посчитал и сверстал
    #     (факт, сравнение, состав, прогноз, план/факт), отдаётся шаблон, а
    #     модель вызывается только там, где она реально добавляет смысл —
    #     «что говорится в документах», причины, ранжирование;
    #   llm — всегда пересказывать моделью (медленнее на CPU);
    #   template — никогда (быстрее всего, полностью офлайн).
    compose_mode: str = "auto"

    # --- Строгость старта ---
    # True — не запускаться с дефолтным REG_CODE при пустом whitelist: иначе
    # зарегистрироваться сможет любой, кто знает код из README.
    strict_startup: bool = True

    @property
    def default_reg_codes(self) -> set[str]:
        return {"", "changeme", "change-me-please", "rag-2026"}

    @property
    def effective_embeddings_provider(self) -> str:
        """Фактически доступный провайдер с самокоррекцией частой ошибки:
        если в EMBEDDINGS_PROVIDER указано имя модели (BAAI/bge-m3,
        intfloat/multilingual-e5-large), трактуем его как модель для
        sentence_transformers. Если пакет не установлен (запуск не через venv) —
        откат на hash с понятным предупреждением вместо падения."""
        p = (self.embeddings_provider or "").strip()
        if "/" in p and p not in ("api",):
            # в поле провайдера — имя модели: используем sentence_transformers
            return "sentence_transformers" if self._st_available() else "hash"
        if p == "sentence_transformers":
            return "sentence_transformers" if self._st_available() else "hash"
        return p

    @property
    def effective_embeddings_model(self) -> str:
        """Модель эмбеддингов с учётом самокоррекции провайдера."""
        p = (self.embeddings_provider or "").strip()
        if self.effective_embeddings_provider == "sentence_transformers":
            return p if "/" in p else self.embeddings_local_model
        return self.embeddings_local_model

    def _st_available(self) -> bool:
        try:
            import importlib.util

            return importlib.util.find_spec("sentence_transformers") is not None
        except Exception:
            return False

    @property
    def allowed_ids(self) -> set[int]:
        raw = (self.allowed_user_ids or "").replace(" ", "")
        return {int(x) for x in raw.split(",") if x}

    @property
    def allowed_exts(self) -> set[str]:
        return {x.strip().lower() for x in self.allowed_extensions.split(",") if x.strip()}

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def expense_journal_path(self) -> Path:
        """Excel-журнал ручных расходов/доходов (хранится локально)."""
        p = Path(self.expense_journal_file) if self.expense_journal_file else None
        if p is not None:
            return p if p.is_absolute() else self.data_dir / p
        return self.data_dir / "расходы.xlsx"

    @property
    def llm_usage_path(self) -> Path:
        """Журнал расхода токенов (JSONL) — рядом с данными."""
        p = Path(self.llm_usage_log) if self.llm_usage_log else None
        if p is not None:
            return p if p.is_absolute() else self.data_dir / p
        return self.data_dir / "llm_usage.jsonl"

    @property
    def llm_price_map(self) -> dict[str, tuple[float, float]]:
        """Цены моделей из строки 'модель=вход/выход,...' (за 1 млн токенов)."""
        out: dict[str, tuple[float, float]] = {}
        for item in (self.llm_prices or "").split(","):
            item = item.strip()
            if not item or "=" not in item:
                continue
            name, _, rates = item.partition("=")
            price_in, _, price_out = rates.partition("/")
            try:
                out[name.strip()] = (float(price_in), float(price_out or price_in))
            except ValueError:
                continue
        return out

    @property
    def small_model_tasks(self) -> set[str]:
        return {t.strip() for t in (self.llm_model_small_tasks or "").split(",") if t.strip()}


settings = Settings()
