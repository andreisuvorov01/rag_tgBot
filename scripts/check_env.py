"""Диагностика готовности к локальному запуску.

Проверяет зависимости, конфигурацию, БД, эмбеддинги, LLM и доступность
Telegram API; печатает отчёт с конкретными подсказками.

Запуск:  python scripts/check_env.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")

OK, WARN, FAIL = "  [OK]  ", " [ВНИМ] ", " [СТОП] "
problems: list[str] = []


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
    print("=== Диагностика локального запуска ===\n")

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
        from app.storage import make_engine, make_sessionmaker

        engine = await make_engine(settings)
        line(OK, f"БД: {engine.dialect.name} — таблицы созданы, соединение работает")
        sessions = make_sessionmaker(engine)
        async with sessions() as s:
            from sqlalchemy import text

            for table in ("documents", "facts", "metrics", "chunks"):
                try:
                    n = (await s.execute(text(f"SELECT COUNT(*) FROM {table}"))).scalar_one()
                    line(OK, f"  {table}: {n} записей")
                except Exception as e:
                    line(FAIL, f"  таблица {table}: {e}")
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

    async def llm_check() -> None:
        llm = make_llm(settings)
        if settings.llm_provider == "mock":
            line(OK, "LLM_PROVIDER=mock — offline-режим: ответы собираются шаблонами, вся аналитика работает")
            line(WARN, "Для живого языка ответов включите реальный API (см. .env: openai_compatible / gigachat / anthropic)")
            return
        try:
            text = await llm.chat(
                [{"role": "system", "content": "ping"}, {"role": "user", "content": "ping"}],
                max_tokens=10, temperature=0.0,
            )
            line(OK, f"LLM API [{settings.llm_provider}/{settings.llm_model}]: отвечает ({text[:20]!r}...)")
        except Exception as e:
            line(FAIL, f"LLM API недоступен: {e} — проверьте ключи/базовый URL в .env "
                       f"(или временно поставьте LLM_PROVIDER=mock)")
        finally:
            await llm.close()

    try:
        asyncio.run(llm_check())
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
            r = httpx.get("https://api.telegram.org", timeout=timeout, transport=transport)
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
