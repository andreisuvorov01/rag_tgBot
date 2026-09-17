"""Удаление документов из памяти бота прямо из чата.

Проверяем то, что пользователь видит и чем рискует: кнопка появляется только у
своих файлов, удаление требует подтверждения, чужие и системные документы не
удаляются, а вместе с файлом уходят его данные (факты, фрагменты, показатели).
"""
from __future__ import annotations

import openpyxl
from sqlalchemy import select

from app.bot import BotApp
from app.config import Settings
from app.embeddings import EmbeddingService
from app.expenses import JOURNAL_DOC_NAME, add_entry, parse_entry_message
from app.ingest.pipeline import process_document
from app.llm import make_llm
from app.qa import AnswerPipeline
from app.storage import (
    Chunk,
    Document,
    Fact,
    Metric,
    make_engine,
    make_sessionmaker,
    register_user,
)


def _settings(tmp_path) -> Settings:
    # не зависим от .env машины
    return Settings(  # type: ignore[call-arg]
        _env_file=None,
        database_url=f"sqlite+aiosqlite:///{tmp_path}/del.db",
        data_dir=tmp_path,
        embeddings_provider="hash",
        llm_provider="mock",
    )


def _report_file(tmp_path, name: str, metric: str, value: str) -> bytes:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Показатель", "2024"])
    ws.append([metric, value])
    path = tmp_path / name
    wb.save(path)
    return path.read_bytes()


async def _setup(tmp_path):
    s = _settings(tmp_path)
    engine = await make_engine(s)
    sessions = make_sessionmaker(engine)
    emb = EmbeddingService(s)
    llm = make_llm(s)
    app = BotApp(s, AnswerPipeline(sessions, emb, llm, s, org_names=[]), sessions)
    async with sessions() as session:
        await register_user(session, 1, "Первый", "ООО Тест")
        await session.commit()
    return s, engine, sessions, emb, llm, app


async def _upload(sessions, emb, s, *, name: str, user_id: int, metric: str, value: str,
                  tmp_path) -> int:
    async with sessions() as session:
        report = await process_document(
            session, emb, s, org_id=1, user_id=user_id,
            original_name=name, content=_report_file(tmp_path, name, metric, value),
        )
        await session.commit()
        doc_id = report.document_id
        assert doc_id is not None
        return doc_id


async def test_delete_removes_data_and_file(tmp_path):
    """Удаление убирает и данные, и оригинал файла с диска."""
    s, engine, sessions, emb, llm, app = await _setup(tmp_path)
    doc_id = await _upload(sessions, emb, s, name="Отчет.xlsx", user_id=1,
                           metric="Выручка", value="100", tmp_path=tmp_path)

    async with sessions() as session:
        doc = await session.get(Document, doc_id)
        stored = doc.stored_path
        facts_before = len((await session.scalars(
            select(Fact).where(Fact.document_id == doc_id))).all())
    assert facts_before > 0
    from pathlib import Path
    assert Path(stored).exists()

    info = await app._delete_document(org_id=1, user_id=1, doc_id=doc_id)
    assert info is not None and info["name"] == "Отчет.xlsx"
    assert info["facts"] == facts_before

    async with sessions() as session:
        assert await session.get(Document, doc_id) is None
        assert not (await session.scalars(
            select(Fact).where(Fact.document_id == doc_id))).all()
        assert not (await session.scalars(
            select(Chunk).where(Chunk.document_id == doc_id))).all()
    assert not Path(stored).exists(), "оригинал файла остался на диске"

    await llm.close()
    await engine.dispose()


async def test_delete_is_not_allowed_for_foreign_document(tmp_path):
    """Коллега по организации не может удалить чужой файл."""
    s, engine, sessions, emb, llm, app = await _setup(tmp_path)
    async with sessions() as session:
        await register_user(session, 2, "Второй", "ООО Тест")   # та же организация
        await session.commit()
    doc_id = await _upload(sessions, emb, s, name="Чужой.xlsx", user_id=1,
                           metric="Выручка", value="100", tmp_path=tmp_path)

    assert await app._delete_document(org_id=1, user_id=2, doc_id=doc_id) is None
    async with sessions() as session:
        assert await session.get(Document, doc_id) is not None, "чужой документ удалён"

    await llm.close()
    await engine.dispose()


async def test_delete_is_not_allowed_from_other_org(tmp_path):
    """Документ другой организации недоступен даже по id."""
    s, engine, sessions, emb, llm, app = await _setup(tmp_path)
    doc_id = await _upload(sessions, emb, s, name="Свой.xlsx", user_id=1,
                           metric="Выручка", value="100", tmp_path=tmp_path)
    assert await app._delete_document(org_id=99, user_id=1, doc_id=doc_id) is None
    async with sessions() as session:
        assert await session.get(Document, doc_id) is not None
    await llm.close()
    await engine.dispose()


async def test_journal_document_is_protected(tmp_path):
    """Журнал расходов — не файл, а контейнер операций: удалять его нельзя.

    Иначе кнопка «Удалить» на журнале стирала бы历史 расходов «расход: 1500 кофе».
    """
    s, engine, sessions, emb, llm, app = await _setup(tmp_path)
    entry = parse_entry_message("расход: 1500 кофе")
    assert entry is not None
    async with sessions() as session:
        await add_entry(session, s, org_id=1, user_id=1, entry=entry)
        await session.commit()
        journal = (await session.scalars(
            select(Document).where(Document.original_name == JOURNAL_DOC_NAME)
        )).first()
    assert journal is not None
    journal_id = journal.id

    # uploaded_by=0 у журнала, поэтому «своим» он не считается и удаления нет
    assert await app._delete_document(org_id=1, user_id=1, doc_id=journal_id) is None
    async with sessions() as session:
        assert await session.get(Document, journal_id) is not None
        # операции журнала на месте — расход не потерян
        metrics = (await session.scalars(
            select(Metric).where(Metric.org_id == 1))).all()
        assert any("расход" in m.name for m in metrics)

    await llm.close()
    await engine.dispose()


async def test_orphan_metrics_are_removed(tmp_path):
    """После удаления документа показатели-призраки уходят из словаря."""
    s, engine, sessions, emb, llm, app = await _setup(tmp_path)
    doc_id = await _upload(sessions, emb, s, name="Разовый.xlsx", user_id=1,
                           metric="Уникальный показатель", value="100", tmp_path=tmp_path)
    async with sessions() as session:
        names = {m.name for m in (await session.scalars(
            select(Metric).where(Metric.org_id == 1))).all()}
    assert "уникальный показатель" in names

    await app._delete_document(org_id=1, user_id=1, doc_id=doc_id)

    async with sessions() as session:
        left = {m.name for m in (await session.scalars(
            select(Metric).where(Metric.org_id == 1))).all()}
    assert "уникальный показатель" not in left, "показатель без фактов остался в словаре"

    await llm.close()
    await engine.dispose()


async def test_delete_button_only_for_own_documents(tmp_path):
    """Кнопка «Удалить» есть только на своих файлах; перечитать — и на общих."""
    s, engine, sessions, emb, llm, app = await _setup(tmp_path)
    await _upload(sessions, emb, s, name="Мой.xlsx", user_id=1,
                  metric="Выручка", value="100", tmp_path=tmp_path)
    await _upload(sessions, emb, s, name="Коллеги.xlsx", user_id=2,
                  metric="Аренда", value="50", tmp_path=tmp_path)
    async with sessions() as session:
        docs = (await session.scalars(select(Document))).all()

    mine = [d for d in docs if d.original_name == "Мой.xlsx"]
    theirs = [d for d in docs if d.original_name == "Коллеги.xlsx"]

    def labels(rows):
        return [b.text for row in rows for b in row]

    my_buttons = labels(app._document_action_rows(mine, viewer_id=1))
    assert any("Удалить" in t for t in my_buttons)
    assert any("Перечитать" in t for t in my_buttons)

    their_buttons = labels(app._document_action_rows(theirs, viewer_id=1))
    assert not any("Удалить" in t for t in their_buttons), "кнопка удаления на чужом файле"
    assert not any("Перечитать" in t for t in their_buttons)

    await llm.close()
    await engine.dispose()


async def test_confirm_keyboard_carries_document_id(tmp_path):
    """Подтверждение удаления адресует именно тот документ, который выбран."""
    s, engine, sessions, emb, llm, app = await _setup(tmp_path)
    kb = app._doc_del_keyboard(42)
    data = [b.callback_data for row in kb.inline_keyboard for b in row]
    assert "docdel:yes:42" in data
    assert "docdel:no" in data
    await llm.close()
    await engine.dispose()


async def test_clear_still_removes_everything(tmp_path):
    """Прежний /clear продолжает работать: удаляет все свои документы."""
    from app.storage import delete_user_documents

    s, engine, sessions, emb, llm, app = await _setup(tmp_path)
    await _upload(sessions, emb, s, name="Первый.xlsx", user_id=1,
                  metric="Выручка", value="100", tmp_path=tmp_path)
    await _upload(sessions, emb, s, name="Второй.xlsx", user_id=1,
                  metric="Аренда", value="50", tmp_path=tmp_path)

    async with sessions() as session:
        paths = await delete_user_documents(session, 1, 1)
        await session.commit()
    assert len(paths) == 2
    async with sessions() as session:
        left = (await session.scalars(
            select(Document).where(Document.uploaded_by == 1))).all()
    assert not left
    await llm.close()
    await engine.dispose()


async def test_wipe_org_data(tmp_path):
    """/wipe: документы всех пользователей (включая системный журнал трат),
    факты, операции, словарь показателей с синонимами — всё пусто."""
    from datetime import date
    from pathlib import Path

    from sqlalchemy import func, select

    from app.config import settings
    from app.embeddings import EmbeddingService
    from app.expenses import add_entry, parse_entry_message
    from app.ingest.pipeline import process_document
    from app.storage import (
        Document,
        Fact,
        LedgerOperation,
        Metric,
        MetricSynonym,
        make_engine,
        make_sessionmaker,
        wipe_org_data,
    )

    settings.database_url = f"sqlite+aiosqlite:///{tmp_path / 'wipe.db'}"
    settings.expense_journal_file = str(tmp_path / "расходы.xlsx")
    settings.embeddings_provider = "hash"
    engine = await make_engine(settings)
    sessions = make_sessionmaker(engine)
    emb = EmbeddingService(settings)
    report = Path(__file__).resolve().parents[1] / "Финансовый_отчёт_ООО_Вектор_2023-2025.xlsx"
    async with sessions() as s:
        await process_document(s, emb, settings, org_id=1, user_id=1,
                               original_name=report.name, content=report.read_bytes())
        await add_entry(s, settings, org_id=1, user_id=1,
                        entry=parse_entry_message("расход: 1500 кофе", today=date(2026, 9, 17)), emb=emb)
        await s.commit()
        assert (await s.scalar(select(func.count()).select_from(Metric))) > 0
        paths = await wipe_org_data(s, 1, 1)
        await s.commit()
        assert len(paths) == 2  # отчёт + журнал (системный документ, uploaded_by=0)
        for model in (Document, Fact, LedgerOperation, Metric, MetricSynonym):
            assert (await s.scalar(select(func.count()).select_from(model))) == 0, model.__name__
    await engine.dispose()
