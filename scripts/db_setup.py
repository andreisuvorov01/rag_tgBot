"""Почему таблицы не создаются: пошаговая проверка схемы БД.

Запуск на сервере (из каталога проекта):
    venv/bin/python scripts/db_setup.py

Скрипт делает ровно то же, что `make_engine()`, но с отчётом на каждом шаге:
какая база и схема видны соединению, есть ли расширение vector, что говорит
`create_all`, какие таблицы появились сразу после него и куда пишет сама модель.
Если таблиц нет — печатает текст ошибки PostgreSQL, а не «relation does not
exist» через три экрана стека.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    _reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(_reconfigure):
        _reconfigure(encoding="utf-8")

from sqlalchemy import text  # noqa: E402

from app.config import settings  # noqa: E402
from app.storage import Base, make_engine  # noqa: E402


def _safe_url(url: str) -> str:
    """Скрыть пароль в выводе."""
    if "@" not in url or "://" not in url:
        return url
    scheme, rest = url.split("://", 1)
    creds, host = rest.rsplit("@", 1)
    user = creds.split(":", 1)[0]
    return f"{scheme}://{user}:***@{host}"


async def main() -> int:
    print("=== Проверка схемы БД ===\n")
    print(f"DATABASE_URL: {_safe_url(settings.database_url)}")
    print(f"моделей в метаданных: {len(Base.metadata.tables)}")
    print(f"  {', '.join(sorted(Base.metadata.tables))}\n")

    try:
        engine = await make_engine(settings)
    except Exception as e:
        # самая частая причина: нет прав на создание таблиц (CREATE на схему
        # public) или расширения vector — тогда приложение падало бы так же
        print(f"\n[ОШИБКА] make_engine() не смог подготовить схему: "
              f"{type(e).__name__}: {str(e).splitlines()[0][:300]}")
        if "readonly" in str(e).lower() or "permission" in str(e).lower():
            print("  Похоже на права: проверьте владельца каталога data/ и права "
                  "пользователя БД (CREATE на схему public).")
        if "extension" in str(e).lower() or "vector" in str(e).lower():
            print("  Для pgvector нужен CREATE EXTENSION vector в этой базе: "
                  "sudo -u postgres psql -d <база> -c 'CREATE EXTENSION IF NOT EXISTS vector;'")
        return 1

    problems: list[str] = []
    is_pg = engine.dialect.name == "postgresql"

    async with engine.connect() as conn:
        db = user = schema = path = "?"
        try:
            if is_pg:
                db, user, schema, path = (await conn.execute(text(
                    "SELECT current_database(), current_user, current_schema(), "
                    "current_setting('search_path')"
                ))).one()
                print(f"соединение: {user}@{db}, текущая схема {schema}, search_path={path}")
            else:
                files = (await conn.execute(text("PRAGMA database_list"))).all()
                db = files[0][2] if files else "?"
                print(f"соединение: {engine.dialect.name}, файл {db} (схем нет)")
        except Exception as e:
            print(f"[ОШИБКА] не удалось прочитать параметры соединения: {e}")
            problems.append("соединение")

        if is_pg:
            ext = (await conn.execute(text(
                "SELECT extname, extversion, n.nspname FROM pg_extension e "
                "JOIN pg_namespace n ON n.oid = e.extnamespace WHERE extname = 'vector'"
            ))).all()
            if ext:
                print(f"расширение vector: {ext[0][1]} в схеме {ext[0][2]}")
            else:
                print("[ОШИБКА] расширения vector нет в этой базе")
                problems.append("pgvector")

        # что создала make_engine(): перечисляем все схемы, а не только текущую
        rows = (await conn.execute(text(
            "SELECT table_schema, table_name FROM information_schema.tables "
            "WHERE table_schema NOT IN ('pg_catalog', 'information_schema') "
            "ORDER BY table_schema, table_name"
        ))).all() if is_pg else (
            await conn.execute(text("SELECT 'main', name FROM sqlite_master WHERE type='table'"))
        ).all()
        print(f"\nпосле make_engine() таблиц в базе: {len(rows)}")
        for sch, name in rows[:15]:
            print(f"  {sch}.{name}")

        ours = {t for t in Base.metadata.tables if any(r[1] == t for r in rows)}
        print(f"\nиз моделей приложения найдено: {len(ours)} из {len(Base.metadata.tables)}")

        if ours and is_pg and schema != "?" and not any(r[0] == schema for r in rows):
            print(f"[ОШИБКА] таблицы есть, но НЕ в текущей схеме {schema}: "
                  f"SQLAlchemy создаёт их в первой схеме search_path ({path}). "
                  f"Добавьте схему в search_path в DATABASE_URL: "
                  f"?options=-csearch_path=<схема>,public")
            problems.append("search_path")
        elif not ours:
            print("[ОШИБКА] ни одной таблицы приложения не создано.")
            # повторяем создание вручную, чтобы увидеть настоящий текст ошибки
            try:
                async with engine.begin() as conn2:
                    await conn2.run_sync(Base.metadata.create_all)
                print("  повторный create_all прошёл — таблицы должны появиться; "
                      "запустите скрипт ещё раз")
            except Exception as e:
                print(f"  create_all упал: {type(e).__name__}: {str(e).splitlines()[0][:300]}")
                problems.append("create_all")

    # проверка «глазами ORM»: видит ли модель ту же таблицу
    if not problems:
        from sqlalchemy import select

        from app.storage import AppMeta, make_sessionmaker

        sessions = make_sessionmaker(engine)
        try:
            async with sessions() as s:
                await s.execute(select(AppMeta).limit(1))
            print("\nORM видит app_meta: да")
        except Exception as e:
            print(f"\n[ОШИБКА] ORM не видит app_meta: {type(e).__name__}: "
                  f"{str(e).splitlines()[0][:200]}")
            problems.append("orm")

    await engine.dispose()
    print("\nИтог:", "всё в порядке" if not problems else f"проблемы: {', '.join(problems)}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
