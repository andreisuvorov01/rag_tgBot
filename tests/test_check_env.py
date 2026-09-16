"""Диагностика окружения (`scripts/check_env.py`) на реальных базах.

Проверяем то, на чём спотыкается развёртывание: чистая база (таблиц ещё нет)
и главное — что одна упавшая проверка не превращает остальные в
«InFailedSQLTransaction» (именно так выглядел отчёт на сервере, и настоящая
причина тонула в каскаде ошибок).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CHECK_ENV = ROOT / "scripts" / "check_env.py"


def _load_check_env():
    """Импортируем scripts/check_env.py как модуль (это скрипт, не пакет)."""
    spec = importlib.util.spec_from_file_location("check_env_mod", CHECK_ENV)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_env_mod"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def clean_env(tmp_path, monkeypatch):
    """Свежая SQLite-база в отдельном каталоге: ни таблиц, ни .env машины.

    Патчим сам объект settings, а не переменные окружения: `settings`
    создаётся при импорте и уже прочитал .env, поэтому DATABASE_URL из
    окружения на него не влияет (иначе тест ходил бы в рабочую базу).
    """
    from app.config import settings

    monkeypatch.setattr(settings, "database_url",
                        f"sqlite+aiosqlite:///{(tmp_path / 'clean.db').as_posix()}")
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(settings, "llm_provider", "mock")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-cache"))
    # main() разбирает argv — под pytest там пути тестов, поэтому гасим их
    monkeypatch.setattr(sys, "argv", ["check_env.py"])
    return tmp_path


def _run(module, capsys) -> str:
    module.main()
    return capsys.readouterr().out


def test_fresh_database_is_reported_ok(clean_env, capsys):
    """Пустая база — не ошибка: таблицы создаёт сам диагностический прогон,
    и отчёт должен это показать, а не пугать «таблица не существует».
    """
    out = _run(_load_check_env(), capsys)

    assert "соединение работает" in out
    assert "8 из 8 ключевых" in out, out
    # каскада ошибок транзакции быть не должно
    assert "InFailedSQLTransaction" not in out
    assert "does not exist" not in out


def test_table_counts_are_shown(clean_env, capsys):
    out = _run(_load_check_env(), capsys)
    for table in ("documents", "facts", "metrics", "chunks"):
        assert f"{table}: 0 записей" in out, out


def test_paths_and_cache_are_printed(clean_env, capsys):
    """Видно, из какого каталога запущено и куда качаются модели.

    Раньше HF_HOME брался из окружения, и диагностика качала ~0,5 ГБ модели в
    ~/.cache/huggingface, а бот потом искал их рядом с проектом и качал снова.
    """
    out = _run(_load_check_env(), capsys)
    assert "каталог:" in out
    assert "кэш моделей (HF_HOME):" in out
    assert str(clean_env / "hf-cache") in out


def test_each_table_is_checked_its_own_transaction(clean_env, capsys, monkeypatch):
    """Каждая таблица проверяется отдельной транзакцией и упоминается один раз.

    Регресс: все проверки шли в одной сессии, и первая же ошибка аварийно
    завершала транзакцию — три оставшиеся строки врали про
    InFailedSQLTransaction вместо настоящей причины.
    """
    module = _load_check_env()
    mentions: list[str] = []
    real_line = module.line

    def spy(status: str, text: str) -> None:
        mentions.append(text)
        real_line(status, text)

    monkeypatch.setattr(module, "line", spy)
    out = _run(module, capsys)

    for table in ("documents", "facts", "metrics", "chunks"):
        hits = [m for m in mentions if f"{table}: " in m]
        assert len(hits) == 1, f"{table}: {hits}"
    assert "InFailedSQLTransaction" not in out


def test_source_explains_missing_pgvector_and_who_creates_tables():
    """Тексты подсказок: точная команда для pgvector и кто создаёт таблицы.

    Поднять настоящий PostgreSQL в тестах нельзя, поэтому проверяем, что
    подсказки вообще есть и идут в правильном порядке: сначала причина
    (расширение), потом следствие (список таблиц).
    """
    source = CHECK_ENV.read_text(encoding="utf-8")
    assert "CREATE EXTENSION IF NOT EXISTS vector" in source
    assert "psql -d {db}" in source
    assert "make_engine() создаёт их при старте" in source
    assert source.index("Расширение pgvector:") < source.index("Таблицы в схеме")
