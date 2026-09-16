"""Диагностика готовности к локальному запуску.

Проверяет зависимости, конфигурацию, БД, эмбеддинги, LLM и доступность
Telegram API; печатает отчёт с конкретными подсказками.

Запуск:  python scripts/check_env.py
         python scripts/check_env.py --llm   # + замер стоимости шагов LLM
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    _reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(_reconfigure):
        _reconfigure(encoding="utf-8")

OK, WARN, FAIL = "  [OK]  ", " [ВНИМ] ", " [СТОП] "
problems: list[str] = []
# заполняется в main(): флаги диагностики доступны вложенным проверкам
args: argparse.Namespace = argparse.Namespace(llm=False)


def line(status: str, text: str) -> None:
    print(f"{status}{text}")
    if status == FAIL:
        problems.append(text)


def masked(value: str, keep: int = 4) -> str:
    if not value:
        return "— не задано"
    if len(value) <= keep:
        return "*" * len(value)
    return value[:keep] + "…" + "*" * 6


def main() -> int:
    global args
    ap = argparse.ArgumentParser(description="Диагностика готовности к запуску")
    ap.add_argument("--llm", action="store_true",
                    help="пробные вызовы по шагам LLM (тратит токены) для проверки JSON и объёма промптов")
    args = ap.parse_args()

    # Кэш моделей — рядом с проектом, как и при запуске бота (app/main.py:
    # setup_huggingface_env). Иначе диагностика качает ~0,5 ГБ модели в
    # ~/.cache/huggingface, а бот потом ищет их в другом месте и качает снова.
    import os

    hf_home = os.environ.get("HF_HOME") or str((Path.cwd() / "hf-cache").resolve())
    os.environ.setdefault("HF_HOME", hf_home)

    print("=== Диагностика локального запуска ===\n")
    print(f"  каталог: {Path.cwd()}")
    print(f"  кэш моделей (HF_HOME): {os.environ['HF_HOME']}\n")

    # --- Python и зависимости ---
    import platform

    line(OK, f"Python {sys.version.split()[0]} ({platform.system()})")
    missing = []
    for mod in ("aiogram", "sqlalchemy", "pandas", "openpyxl", "httpx", "pgvector", "pydantic_settings"):
        try:
            m = __import__(mod)
            line(OK, f"{mod} {getattr(m, '__version__', '')}".rstrip())
        except Exception as e:
            missing.append(mod)
            line(FAIL, f"{mod}: {e}")
    try:
        import statsmodels  # noqa: F401

        line(OK, "statsmodels (метод Хольта доступен)")
    except Exception:
        line(WARN, "statsmodels не установлен — прогноз без метода Хольта (не критично)")
    try:
        import pytesseract  # noqa: F401

        line(OK, "pytesseract — OCR сканов доступен")
    except Exception:
        line(WARN, "OCR не настроен (сканы будут давать предупреждение; см. README)")

    # --- Диск ---
    print()
    import shutil

    seen_drives = set()
    for drive in ("C:\\", str(Path.cwd().anchor) or "C:\\"):
        if drive in seen_drives:
            continue
        seen_drives.add(drive)
        try:
            free_gb = shutil.disk_usage(drive).free / 2**30
            if free_gb < 5:
                line(WARN, f"На диске {drive} свободно всего {free_gb:.1f} ГБ —"
                           f" установка пакетов/моделей может падать (см. setup_venv.bat)")
            else:
                line(OK, f"Диск {drive}: свободно {free_gb:.0f} ГБ")
        except Exception:
            pass

    # --- Конфигурация ---
    print()
    from app.config import settings
    from app.llm import make_llm  # noqa: E402  (после settings)

    line(OK, f"Файл .env: {'найден' if Path('.env').exists() else 'ОТСУТСТВУЕТ (работают значения по умолчанию)'}")

    if settings.bot_token:
        if ":" in settings.bot_token:
            line(OK, f"BOT_TOKEN: {masked(settings.bot_token)}")
        else:
            line(FAIL, "BOT_TOKEN задан, но не похож на токен (ожидается '123456:ABC-DEF...')")
    else:
        line(WARN, "BOT_TOKEN не задан — бот не стартует. Возьмите токен в @BotFather -> /newbot и впишите в .env")

    if settings.allowed_ids:
        line(OK, f"Доступ по whitelist: {len(settings.allowed_ids)} ID")
    else:
        line(OK, f"Доступ по коду регистрации: REG_CODE={settings.reg_code!r}")

    if settings.telegram_proxy:
        line(OK, f"Прокси Telegram: {settings.telegram_proxy}")

    # --- БД ---
    print()

    async def db_check() -> None:
        from app.storage import make_engine

        engine = await make_engine(settings)
        print()  # отделяем диагностику схемы от строки «БД: ...»
        is_pg = engine.dialect.name == "postgresql"

        # Что именно за база: при развёртывании чаще всего путают базу и
        # схему — таблицы создаются не в том месте, куда потом идёт бот.
        async with engine.connect() as conn:
            from sqlalchemy import text

            try:
                if is_pg:
                    db, user, schema, server = (
                        await conn.execute(text(
                            "SELECT current_database(), current_user, "
                            "current_schema(), current_setting('server_version')"
                        ))
                    ).one()
                    line(OK, f"БД: {engine.dialect.name} — соединение работает "
                             f"({user}@{db}, схема {schema}, PostgreSQL {server.split()[0]})")
                else:
                    path = (await conn.execute(text("PRAGMA database_list"))).all()
                    schema = "main"
                    line(OK, f"БД: {engine.dialect.name} — соединение работает "
                             f"({path[0][2] if path else '—'})")
            except Exception as e:
                line(FAIL, f"БД недоступна: {e} (проверьте DATABASE_URL в .env)")
                await engine.dispose()
                return

            if is_pg:
                # pgvector: без расширения таблицу с колонкой vector создать
                # нельзя, и ошибка выглядит как «таблица не существует»
                try:
                    version = (
                        await conn.execute(text(
                            "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
                        ))
                    ).scalar_one_or_none()
                    if version:
                        line(OK, f"Расширение pgvector: {version}")
                    else:
                        line(FAIL, "Расширение pgvector не установлено в базе "
                                   f"«{db}» — таблицы с векторами создать нельзя.\n"
                                   f"        Выполните: sudo -u postgres psql -d {db} "
                                   f"-c 'CREATE EXTENSION IF NOT EXISTS vector;'")
                except Exception as e:
                    line(WARN, f"Не удалось проверить расширение vector: {e}")

            # какие таблицы реально есть (и в какой схеме)
            try:
                if is_pg:
                    rows = (await conn.execute(text(
                        "SELECT table_schema, table_name FROM information_schema.tables "
                        "WHERE table_schema NOT IN ('pg_catalog', 'information_schema') "
                        "ORDER BY table_schema, table_name"
                    ))).all()
                    present = {(r[0], r[1]) for r in rows}
                else:
                    rows = (await conn.execute(text(
                        "SELECT name FROM sqlite_master WHERE type = 'table'"
                    ))).all()
                    present = {(schema, r[0]) for r in rows}
                ours = {t for t in ("documents", "facts", "metrics", "chunks", "users",
                                    "organizations", "ledger_operations", "app_meta")
                        if (schema, t) in present}
                if ours:
                    line(OK, f"Таблицы в схеме {schema}: {len(ours)} из 8 ключевых — "
                             f"({', '.join(sorted(ours))})")
                elif present:
                    other = ", ".join(f"{s}.{t}" for s, t in sorted(present)[:5])
                    line(FAIL, f"Таблиц приложения в схеме {schema} нет, но в базе есть "
                               f"другие: {other}\n"
                               f"        Похоже, таблицы созданы в другой схеме/базе — "
                               f"проверьте DATABASE_URL и search_path.")
                else:
                    line(FAIL, f"В базе «{db if is_pg else schema}» нет ни одной таблицы "
                               f"приложения.\n"
                               f"        make_engine() создаёт их при старте: запустите "
                               f"`python -m app.main`"
                               + (" (или проверьте, что CREATE EXTENSION vector выполнен "
                                  "в этой же базе)." if is_pg else "."))
            except Exception as e:
                line(WARN, f"Не удалось прочитать список таблиц: {e}")

        # Каждая проверка — в СВОЕЙ транзакции: иначе первая же ошибка аварийно
        # завершает транзакцию, и все следующие запросы отвечают
        # InFailedSQLTransaction вместо настоящей причины.
        for table in ("documents", "facts", "metrics", "chunks"):
            try:
                async with engine.connect() as conn:
                    from sqlalchemy import text

                    n = (await conn.execute(text(f"SELECT COUNT(*) FROM {table}"))).scalar_one()
                line(OK, f"  {table}: {n} записей")
            except Exception as e:
                line(FAIL, f"  таблица {table}: {type(e).__name__}: {str(e).splitlines()[0][:160]}")
        await engine.dispose()

    try:
        asyncio.run(db_check())
    except Exception as e:
        line(FAIL, f"БД недоступна: {e} (проверьте DATABASE_URL в .env)")

    # --- Эмбеддинги (локально) ---
    print()

    async def emb_check() -> None:
        from app.embeddings import EmbeddingService

        emb = EmbeddingService(settings)
        vec = await emb.embed_query("проверка эмбеддингов")
        line(OK, f"Эмбеддинги [{settings.embeddings_provider}]: вектор dim={len(vec)}")
        if settings.embeddings_provider == "hash":
            line(WARN, "Провайдер hash —dev-режим; для смыслового поиска поставьте "
                       "sentence-transformers (EMBEDDINGS_PROVIDER=sentence_transformers)")
        if settings.embeddings_provider == "sentence_transformers":
            if settings.effective_embeddings_provider != "sentence_transformers":
                line(FAIL, "EMBEDDINGS_PROVIDER=sentence_transformers, но пакет не установлен — "
                           "запустите start.bat (venv) или setup_venv.bat")
            else:
                import os

                hf = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
                if hf.exists() and any(hf.iterdir()):
                    line(OK, "Модель эмбеддингов уже скачана (hf-cache)")
                else:
                    line(WARN, "При первом запуске скачается модель эмбеддингов (~0.5–2 ГБ)")

    try:
        asyncio.run(emb_check())
    except Exception as e:
        line(FAIL, f"Эмбеддинги не работают: {e}")

    # --- LLM (внешний API) ---
    print()

    async def llm_check(deep: bool = False) -> None:
        llm = make_llm(settings)
        if settings.llm_provider == "mock":
            line(OK, "LLM_PROVIDER=mock — offline-режим: ответы собираются шаблонами, вся аналитика работает")
            line(WARN, "Для живого языка ответов включите реальный API (см. .env: openai_compatible / gigachat / anthropic)")
            return
        # ключ проверяем до сетевой пробы: иначе получим невнятное
        # «Illegal header value b'Bearer '» и три бессмысленных повтора
        from app.llm import _missing_api_key, _no_key_hint

        if _missing_api_key(settings):
            line(FAIL, _no_key_hint(settings))
            return
        try:
            text = await llm.chat(
                [{"role": "system", "content": "ping"}, {"role": "user", "content": "ping"}],
                max_tokens=10, temperature=0.0, task="other",
            )
            line(OK, f"LLM API [{settings.llm_provider}/{settings.llm_model}]: отвечает ({text[:20]!r}...)")
        except Exception as e:
            line(FAIL, f"LLM API недоступен: {e} — проверьте ключи/базовый URL в .env "
                       f"(или временно поставьте LLM_PROVIDER=mock)")
            return
        try:
            _report_token_economy(llm)
            if deep:
                await _deep_llm_check(llm)
        finally:
            await llm.close()

    def _report_token_economy(llm) -> None:
        """Что настроено для экономии токенов: какая модель на каком шаге."""
        small = (settings.llm_model_small or "").strip()
        if small:
            tasks = ", ".join(sorted(settings.small_model_tasks)) or "—"
            line(OK, f"Дешёвая модель для служебных шагов: {small} ({tasks})")
        else:
            line(WARN, "LLM_MODEL_SMALL не задан: классификация, реранкинг и SQL идут "
                       "на основную (дорогую) модель. Дешёвая модель на эти шаги экономит "
                       "больше всего — они вызываются чаще композитора")
        if settings.llm_provider == "openai_compatible":
            host = (settings.llm_api_base or "").lower()
            if "localhost" in host or "127.0.0.1" in host:
                line(WARN, "LLM_API_BASE указывает на локальную модель: токены не тратятся, "
                           "но служебные шаги дороги по времени (см. CLASSIFY_WITH_LLM)")
        if settings.embeddings_provider == "api" and not settings.anonymize_prompts:
            line(WARN, "Эмбеддинги через API при ANONYMIZE_PROMPTS=false: тексты документов "
                       "уходят наружу без маскирования")
        if not settings.answer_cache_size:
            line(WARN, "ANSWER_CACHE_SIZE=0: повторный вопрос снова оплачивается полностью")
        if not settings.llm_price_map:
            line(WARN, "LLM_PRICES не задан — /usage покажет токены без денег. "
                       'Пример: LLM_PRICES="deepseek-chat=0.27/1.10"')
        else:
            line(OK, f"Цены заданы для: {', '.join(sorted(settings.llm_price_map))}")

    async def _deep_llm_check(llm) -> None:
        """Реальные вызовы по трём шагам: проверить JSON-режим и измерить объём
        промптов в токенах (оценка по символам — без лишних зависимостей).
        Стоит несколько сотен токенов: запускать осознанно (--llm)."""
        from app.prompts import CLASSIFIER_SYSTEM, COMPOSER_SYSTEM, SQL_SYSTEM

        print()
        print("  Пробные вызовы по шагам (тратят токены):")
        cases = [
            ("classify", CLASSIFIER_SYSTEM, "какая выручка за 2024 год?", True, 120),
            ("sql", SQL_SYSTEM, "ВОПРОС: сколько всего фактов по выручке?", False, 400),
            ("rerank", "Оцени релевантность фрагментов. Отвечай только JSON.",
             '{"query": "риски", "fragments": [{"id": 1, "text": "Риск снижения спроса"}]}', True, 200),
            ("compose", COMPOSER_SYSTEM,
             'ВОПРОС: какая выручка за 2024?\n\nДАННЫЕ (JSON):\n{"type":"factual","value":468500000}',
             False, 150),
        ]
        for task, system, user, json_mode, max_tokens in cases:
            model = llm.model_for(task) if hasattr(llm, "model_for") else settings.llm_model
            prompt_chars = len(system) + len(user)
            try:
                out = await llm.chat(
                    [{"role": "system", "content": system}, {"role": "user", "content": user}],
                    json_mode=json_mode, temperature=0.0, max_tokens=max_tokens, task=task,
                )
            except Exception as e:
                line(FAIL, f"  шаг {task} ({model}): {e}")
                continue
            note = ""
            if json_mode:
                from app.llm import parse_json_block

                note = " JSON распознан" if parse_json_block(out) else " ⚠ JSON не распознан (сработает запасной разбор)"
            line(OK, f"  шаг {task:9} модель {model:22} промпт ~{prompt_chars // 4:>5} ток., "
                     f"ответ ~{len(out) // 4:>4} ток.{note}")
        line(OK, "Подробный расход (реальные цифры от провайдера): python -m scripts.llm_usage")

    try:
        asyncio.run(llm_check(args.llm))
    except Exception as e:
        line(FAIL, f"Проверка LLM упала: {e}")

    # --- Telegram API ---
    print()

    def tg_check() -> None:
        import socket

        import httpx

        # DNS: какие адреса видны
        try:
            infos = socket.getaddrinfo("api.telegram.org", 443)
            v4 = sorted({i[4][0] for i in infos if i[0] == socket.AF_INET})
            v6 = sorted({i[4][0] for i in infos if i[0] == socket.AF_INET6})
            line(OK, f"DNS api.telegram.org: IPv4 {v4[:2] or '—'} | IPv6 {v6[:1] or '—'}")
        except Exception as e:
            line(FAIL, f"DNS не резолвит api.telegram.org: {e} — проверьте DNS (8.8.8.8 / 1.1.1.1)")
            return

        timeout = 8
        try:
            r = httpx.get("https://api.telegram.org", timeout=timeout)
            line(OK, f"api.telegram.org доступен (HTTP {r.status_code})")
            return
        except Exception as first_err:
            print(f"        обычное подключение не прошло ({type(first_err).__name__})")

        # проба 2: только IPv4 (частый случай: IPv6-маршрут сломан)
        try:
            transport = httpx.HTTPTransport(local_address="0.0.0.0")
            with httpx.Client(timeout=timeout, transport=transport) as client:
                r = client.get("https://api.telegram.org")
            line(WARN, "Работает только по IPv4 — в .env уже включено TELEGRAM_FORCE_IPV4=true, бот пойдёт по IPv4")
            return
        except Exception as ipv4_err:
            print(f"        IPv4-подключение тоже не прошло ({type(ipv4_err).__name__})")

        # проба 3: системный прокси Windows (приложение Telegram может работать через него)
        proxy = _detect_windows_proxy()
        if proxy:
            try:
                r = httpx.get("https://api.telegram.org", timeout=timeout, proxy=proxy)
                line(WARN, f"Через системный прокси доступно! Добавьте в .env:\n"
                           f"        TELEGRAM_PROXY={proxy}")
                return
            except Exception:
                print(f"        системный прокси {proxy} не помогает")

        line(FAIL,
             "api.telegram.org недоступен напрямую. Приложение Telegram использует другие серверы,"
             " поэтому мессенджер работает, а бот — нет. Решения:\n"
             "        1) включите ваш VPN/прокси-клиент и укажите в .env его адрес:\n"
             "           TELEGRAM_PROXY=socks5://127.0.0.1:1080  или  http://127.0.0.1:10809\n"
             "        2) проверьте, пускает ли другой DNS/сеть (мобильный интернет)")

    def _detect_windows_proxy() -> str | None:
        try:
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows\CurrentVersion\Internet Settings",
            )
            enabled, _ = winreg.QueryValueEx(key, "ProxyEnable")
            if not enabled:
                return None
            raw, _ = winreg.QueryValueEx(key, "ProxyServer")
            winreg.CloseKey(key)
            if "=" in raw:  # формат "http=...;https=...;socks=..."
                parts = dict(p.split("=", 1) for p in raw.split(";") if "=" in p)
                raw = parts.get("https") or parts.get("http") or parts.get("socks") or ""
            if not raw:
                return None
            if raw.startswith(("http", "socks")):
                return raw
            return f"http://{raw}"
        except Exception:
            return None

    try:
        tg_check()
    except Exception as e:
        line(FAIL, f"Проверка сети: {e}")

    # --- Итог ---
    print()
    if any("BOT_TOKEN" in p for p in problems):
        print(">>> СЕРВИС ЕЩЁ НЕ ГОТОВ: впишите BOT_TOKEN в .env (остальное проверено).")
        print(">>> После этого запустите: start.bat  или  python -m app.main")
    elif problems:
        print(">>> Есть блокирующие проблемы (см. [СТОП] выше).")
        return 1
    else:
        print(">>> ВСЁ ГОТОВО К ЗАПУСКУ: start.bat  или  python -m app.main")
    return 0


if __name__ == "__main__":
    sys.exit(main())
