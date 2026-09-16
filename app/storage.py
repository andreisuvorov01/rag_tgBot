"""Хранилище: SQLAlchemy-модели, инициализация БД, векторный поиск, репозитории.

PostgreSQL (+pgvector) — продакшен; SQLite — режим разработки/тестов
(эмбеддинги хранятся как JSON, поиск — перебор по косинусной близости).
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
from datetime import UTC, date, datetime
from decimal import Decimal

import numpy as np
from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    or_,
    select,
    text,
)
from sqlalchemy.dialects.sqlite import JSON as SQLiteJSON
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, aliased, mapped_column, relationship

from .config import Settings
from .money import Money, as_float, money

log = logging.getLogger(__name__)

VectorCol = Vector().with_variant(SQLiteJSON, "sqlite")

# Счётчик изменений текстового индекса (чанки/документы). Нужен, чтобы кэш
# BM25 в rag.py знал, когда перечитать корпус: без него либо поиск работает по
# устаревшему индексу, либо корпус перечитывается и ре-токенизируется на
# каждый вопрос (O(корпус) работы в event loop).
_index_version = 0


def index_version() -> int:
    return _index_version


def bump_index_version() -> None:
    global _index_version
    _index_version += 1


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class Organization(Base):
    __tablename__ = "organizations"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(Text, unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class User(Base):
    __tablename__ = "users"
    telegram_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    org_id: Mapped[int] = mapped_column(ForeignKey("organizations.id"))
    display_name: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (UniqueConstraint("org_id", "file_hash", name="uq_doc_hash"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    org_id: Mapped[int] = mapped_column(ForeignKey("organizations.id"), index=True)
    original_name: Mapped[str] = mapped_column(Text)
    stored_path: Mapped[str] = mapped_column(Text)
    file_hash: Mapped[str] = mapped_column(String(64))
    doc_type: Mapped[str] = mapped_column(String(16))
    status: Mapped[str] = mapped_column(String(16), default="processing")  # processed|failed|duplicate
    # версионирование: если документ переиздан, здесь id актуальной версии;
    # факты старой версии сохраняются (история), но в ряды не попадают
    superseded_by_id: Mapped[int | None] = mapped_column(ForeignKey("documents.id"), nullable=True)
    # ACL: личный документ виден только загрузившему (практика permission-aware
    # retrieval); False/None — общий доступ в рамках организации
    is_private: Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    uploaded_by: Mapped[int] = mapped_column(BigInteger, default=0)
    uploaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    meta: Mapped[dict] = mapped_column(JSON, default=dict)

    facts: Mapped[list[Fact]] = relationship(back_populates="document")


class Metric(Base):
    __tablename__ = "metrics"
    __table_args__ = (UniqueConstraint("org_id", "code", name="uq_metric_code"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    org_id: Mapped[int] = mapped_column(ForeignKey("organizations.id"), index=True)
    code: Mapped[str] = mapped_column(String(128))
    name: Mapped[str] = mapped_column(Text)
    unit: Mapped[str | None] = mapped_column(String(32), nullable=True)  # 'руб'|'%':'шт'|None
    currency: Mapped[str | None] = mapped_column(String(8), nullable=True)
    kind: Mapped[str] = mapped_column(String(16), default="other")  # revenue|expense|asset|liability|other
    parent_id: Mapped[int | None] = mapped_column(ForeignKey("metrics.id"), nullable=True)
    embedding: Mapped[list | None] = mapped_column(VectorCol, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class MetricSynonym(Base):
    __tablename__ = "metric_synonyms"
    __table_args__ = (UniqueConstraint("metric_id", "text", name="uq_syn_text"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    metric_id: Mapped[int] = mapped_column(ForeignKey("metrics.id"), index=True)
    text: Mapped[str] = mapped_column(Text)


class Fact(Base):
    __tablename__ = "facts"
    __table_args__ = (Index("ix_facts_lookup", "org_id", "metric_id", "period_end"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    org_id: Mapped[int] = mapped_column(ForeignKey("organizations.id"), index=True)
    metric_id: Mapped[int] = mapped_column(ForeignKey("metrics.id"), index=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"), index=True)
    period_type: Mapped[str] = mapped_column(String(8))  # year|quarter|month|day
    period_label: Mapped[str] = mapped_column(String(32))
    period_start: Mapped[date] = mapped_column(Date)
    period_end: Mapped[date] = mapped_column(Date)
    value: Mapped[Decimal] = mapped_column(Money)
    unit: Mapped[str | None] = mapped_column(String(32), nullable=True)
    currency: Mapped[str | None] = mapped_column(String(8), nullable=True)
    # вариант значения: fact | plan | budget | forecast | estimate (план/факт из шапки)
    variant: Mapped[str] = mapped_column(String(16), default="fact")
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    needs_review: Mapped[bool] = mapped_column(Boolean, default=False)
    sheet: Mapped[str] = mapped_column(String(128), default="")
    cell_ref: Mapped[str] = mapped_column(String(64), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    document: Mapped[Document] = relationship(back_populates="facts")


class Chunk(Base):
    __tablename__ = "chunks"
    id: Mapped[int] = mapped_column(primary_key=True)
    org_id: Mapped[int] = mapped_column(ForeignKey("organizations.id"), index=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"), index=True)
    body: Mapped[str] = mapped_column(Text)
    page: Mapped[int | None] = mapped_column(Integer, nullable=True)
    section: Mapped[str | None] = mapped_column(String(256), nullable=True)
    embedding: Mapped[list | None] = mapped_column(VectorCol, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class AuditLog(Base):
    __tablename__ = "audit_log"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(BigInteger, index=True)
    action: Mapped[str] = mapped_column(String(64))
    detail: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class AppMeta(Base):
    """Служебные пары ключ-значение (например, отпечаток провайдера эмбеддингов)."""
    __tablename__ = "app_meta"
    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[str] = mapped_column(Text, default="")


class DigestSubscription(Base):
    """Подписка на ежедневную сводку: одному Telegram-чату — одна запись."""
    __tablename__ = "digest_subscriptions"
    telegram_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    org_id: Mapped[int] = mapped_column(ForeignKey("organizations.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class LedgerOperation(Base):
    """Операция из выписки/реестра (ledger-режим): дата, описание, сумма.
    Хранится построчно — для просмотра «операции за месяц» и категорий."""
    __tablename__ = "ledger_operations"
    id: Mapped[int] = mapped_column(primary_key=True)
    org_id: Mapped[int] = mapped_column(ForeignKey("organizations.id"), index=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"), index=True)
    header: Mapped[str] = mapped_column(String(128), index=True)  # имя числовой колонки
    period_label: Mapped[str] = mapped_column(String(32), index=True)
    period_start: Mapped[date] = mapped_column(Date)
    period_end: Mapped[date] = mapped_column(Date)
    date_actual: Mapped[date] = mapped_column(Date)
    description: Mapped[str] = mapped_column(Text, default="")
    value: Mapped[Decimal] = mapped_column(Money)
    currency: Mapped[str | None] = mapped_column(String(8), nullable=True)
    category: Mapped[str] = mapped_column(String(64), default="Прочее")


async def _migrate_money_to_text(conn, table: str) -> None:
    """Привести числовую колонку value к точному TEXT-хранению (SQLite).

    SQLAlchemy кладёт Numeric в SQLite как REAL и возвращает float, поэтому
    точность денег обеспечивается только текстовым хранением.

    Промежуточная таблица строится из DDL существующей таблицы, а не из
    модели: у модели свои имена индексов (`ix_facts_lookup` и подобные),
    и попытка создать их для staging падает с «index ... already exists».
    Здесь индексы создаются уже после подмены таблицы, когда старые имена
    освободились вместе со старой таблицей.
    """
    info = (await conn.execute(text(f"PRAGMA table_info({table})"))).all()
    value_col = next((row for row in info if row[1] == "value"), None)
    if value_col is None:
        return
    declared = str(value_col[2] or "").upper()
    if any(t in declared for t in ("CHAR", "TEXT", "CLOB")):
        return  # уже мигрировано
    log.info("Миграция %s.value: %s -> TEXT (точные деньги)", table, declared or "?")

    ddl_cols = []
    for _cid, name, coltype, notnull, dflt, pk in info:
        col = f'"{name}"'
        if name == "value":
            col += " TEXT"
        else:
            col += f" {coltype}" if coltype else ""
            if pk:
                col += " PRIMARY KEY"
            if notnull:
                col += " NOT NULL"
            if dflt is not None:
                col += f" DEFAULT {dflt}"
        ddl_cols.append(col)

    staging = f"{table}__money_new"
    await conn.execute(text(f"DROP TABLE IF EXISTS {staging}"))
    await conn.execute(text(f'CREATE TABLE "{staging}" ({", ".join(ddl_cols)})'))
    await conn.execute(text(f'INSERT INTO "{staging}" SELECT * FROM "{table}"'))

    moved = (await conn.execute(text(f'SELECT count(*) FROM "{staging}"'))).scalar() or 0
    original = (await conn.execute(text(f'SELECT count(*) FROM "{table}"'))).scalar() or 0
    if moved != original:
        await conn.execute(text(f'DROP TABLE "{staging}"'))
        raise RuntimeError(
            f"перенесено {moved} из {original} строк — таблица {table} не тронута"
        )

    # значения переписываем как точные десятичные строки
    rows = (await conn.execute(text(f'SELECT id, value FROM "{staging}"'))).all()
    for row_id, raw in rows:
        d = money(raw)
        await conn.execute(
            text(f'UPDATE "{staging}" SET value = :v WHERE id = :i'),
            {"v": None if d is None else format(d, "f"), "i": row_id},
        )

    index_sql = [
        r[0] for r in (
            await conn.execute(text(
                "SELECT sql FROM sqlite_master WHERE type='index' AND tbl_name=:t "
                "AND sql IS NOT NULL"
            ), {"t": table})
        ).all()
    ]
    await conn.execute(text(f'DROP TABLE "{table}"'))
    await conn.execute(text(f'ALTER TABLE "{staging}" RENAME TO "{table}"'))
    for sql in index_sql:
        await conn.execute(text(sql))
    log.info("Миграция %s.value завершена: перенесено %d значений", table, moved)


async def migrate_money_columns(conn) -> None:
    """Разовые миграции схемы под точные деньги (идемпотентно).

    Ошибку не глотаем молча: при неудаче таблица остаётся нетронутой, но в
    логе должно быть видно, что точность денег не включена.
    """
    for table in ("facts", "ledger_operations"):
        try:
            await _migrate_money_to_text(conn, table)
        except Exception as e:
            log.error("Миграция %s.value не выполнена (%s) — данные не изменены", table, e)


async def make_engine(settings: Settings) -> AsyncEngine:
    # SQLite: таймаут ожидания блокировки 30 с (иначе мгновенный
    # "database is locked" при двух одновременных писателях)
    connect_args = {"timeout": 30} if settings.database_url.startswith("sqlite") else {}
    engine = create_async_engine(settings.database_url, echo=False, connect_args=connect_args)
    if engine.dialect.name == "postgresql":
        try:
            async with engine.begin() as conn:
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        except Exception as e:  # extension может быть уже установлен администратором
            log.warning("pgvector extension: %s", e)
    async with engine.begin() as conn:
        if engine.dialect.name == "sqlite":
            # WAL: читатели не блокируют писателя и наоборот (два процесса
            # — бот и reindex — больше не конфликтуют мгновенной ошибкой)
            await conn.execute(text("PRAGMA journal_mode=WAL"))
            await conn.execute(text("PRAGMA synchronous=NORMAL"))
        await conn.run_sync(Base.metadata.create_all)
        # лёгкие миграции для БД, созданных до появления новых колонок
        for ddl in (
            "ALTER TABLE documents ADD COLUMN superseded_by_id INTEGER",
            "ALTER TABLE facts ADD COLUMN variant VARCHAR(16) DEFAULT 'fact'",
            "ALTER TABLE documents ADD COLUMN is_private BOOLEAN",
        ):
            with contextlib.suppress(Exception):
                await conn.execute(text(ddl))  # колонка уже существует
        if engine.dialect.name == "sqlite":
            # точные деньги: value из REAL в TEXT (только SQLite — в PostgreSQL
            # точность даёт NUMERIC)
            await migrate_money_columns(conn)
    return engine


def make_sessionmaker(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


# ---------------------------------------------------------------------------
# Векторный поиск
# ---------------------------------------------------------------------------

def _to_pg_literal(vec: list[float]) -> str:
    return "[" + ",".join(f"{x:.7g}" for x in vec) + "]"


async def vector_search(
    session: AsyncSession, *, org_id: int, query_vec: list[float], kind: str, k: int = 5,
    user_id: int | None = None,
) -> list[dict]:
    """kind='chunk' -> релевантные фрагменты (с учётом приватности документов);
    kind='metric' -> показатели словаря."""
    dialect = session.bind.dialect.name
    qv = _to_pg_literal(query_vec)
    if dialect == "postgresql":
        if kind == "chunk":
            sql = text(
                "SELECT c.id, c.document_id, c.body, c.page, c.section, "
                "1 - (c.embedding <=> CAST(:qv AS vector)) AS score "
                "FROM chunks c JOIN documents d ON d.id = c.document_id "
                "WHERE c.org_id = :org AND c.embedding IS NOT NULL "
                "AND d.superseded_by_id IS NULL "
                "AND (COALESCE(d.is_private, FALSE) = FALSE OR d.uploaded_by = :uid) "
                "ORDER BY c.embedding <=> CAST(:qv AS vector) LIMIT :k"
            )
            params = {"qv": qv, "org": org_id, "k": k, "uid": user_id or -1}
        else:
            sql = text(
                "SELECT id, code, name, unit, currency, 1 - (embedding <=> CAST(:qv AS vector)) AS score "
                "FROM metrics WHERE org_id = :org AND embedding IS NOT NULL "
                "AND kind <> 'identifier' "
                "ORDER BY embedding <=> CAST(:qv AS vector) LIMIT :k"
            )
            params = {"qv": qv, "org": org_id, "k": k}
        rows = (await session.execute(sql, params)).mappings().all()
        results = [dict(r) for r in rows]
        if kind == "chunk":
            # Qdrant-зеркало неприватных чанков: объединяем с SQL-веткой
            # (SQL покрывает личные документы владельца)
            from . import qdrant_store

            mirror = await qdrant_store.search_chunks(query_vec, org_id, k)
            if mirror:
                # зеркало может отставать (документ стал личным/переизданным/
                # удалён после загрузки) — видимость перепроверяется по SQL.
                # Берём «белый список» валидных документов, а не только
                # приватные: иначе возвращался бы чанк удалённого или
                # переизданного документа и цитировался как актуальный.
                candidate_docs = {m["document_id"] for m in mirror}
                ok_docs = set(
                    (await session.scalars(
                        select(Document.id).where(
                            Document.id.in_(candidate_docs),
                            Document.org_id == org_id,
                            Document.superseded_by_id.is_(None),
                            visible_documents_condition(user_id),
                        )
                    )).all()
                )
                merged = {r["id"]: r for r in results}
                for m in mirror:
                    if m["id"] not in merged and m["document_id"] in ok_docs:
                        merged[m["id"]] = m
                results = sorted(merged.values(), key=lambda r: r["score"], reverse=True)[:k]
        return results

    # SQLite/dev: загружаем и считаем косинус в Python
    q = np.asarray(query_vec, dtype=np.float32)
    if kind == "chunk":
        docs = (
            await session.scalars(
                select(Document).where(
                    Document.org_id == org_id,
                    Document.superseded_by_id.is_(None),
                )
            )
        ).all()
        visible = {
            d.id for d in docs
            if (not d.is_private) or (user_id is not None and d.uploaded_by == user_id)
        }
        stmt = select(Chunk).where(
            Chunk.org_id == org_id, Chunk.embedding.is_not(None), Chunk.document_id.in_(visible or {0})
        )
    else:
        stmt = select(Metric).where(
            Metric.org_id == org_id,
            Metric.embedding.is_not(None),
            Metric.kind != "identifier",
        )
    out: list[dict] = []
    for obj in (await session.scalars(stmt)).all():
        emb = obj.embedding
        if isinstance(emb, str):
            emb = json.loads(emb)
        v = np.asarray(emb, dtype=np.float32)
        denom = float(np.linalg.norm(q) * np.linalg.norm(v)) or 1.0
        score = float(np.dot(q, v) / denom)
        if kind == "chunk":
            out.append(
                {"id": obj.id, "document_id": obj.document_id, "body": obj.body,
                 "page": obj.page, "section": obj.section, "score": score}
            )
        else:
            out.append(
                {"id": obj.id, "code": obj.code, "name": obj.name, "unit": obj.unit,
                 "currency": obj.currency, "score": score}
            )
    out.sort(key=lambda d: d["score"], reverse=True)
    return out[:k]


# ---------------------------------------------------------------------------
# Репозитории
# ---------------------------------------------------------------------------

async def audit(session: AsyncSession, user_id: int, action: str, detail: str | None = None) -> None:
    session.add(AuditLog(user_id=user_id, action=action, detail=detail))


async def get_or_create_org(session: AsyncSession, name: str = "Основная организация") -> Organization:
    org = (await session.scalars(select(Organization).where(Organization.name == name))).first()
    if org is None:
        org = Organization(name=name)
        session.add(org)
        await session.flush()
    return org


async def get_user(session: AsyncSession, telegram_id: int) -> User | None:
    return (await session.scalars(select(User).where(User.telegram_id == telegram_id))).first()


async def register_user(session: AsyncSession, telegram_id: int, display_name: str, org_name: str) -> User:
    org = await get_or_create_org(session, org_name)
    user = User(telegram_id=telegram_id, org_id=org.id, display_name=display_name)
    session.add(user)
    await session.flush()
    return user


async def find_metric_by_name(session: AsyncSession, org_id: int, name: str) -> Metric | None:
    norm = name.strip().casefold()
    m = (
        await session.scalars(select(Metric).where(Metric.org_id == org_id, Metric.name == norm))
    ).first()
    if m:
        return m
    syn = (
        await session.scalars(
            select(MetricSynonym)
            .join(Metric, Metric.id == MetricSynonym.metric_id)
            .where(MetricSynonym.text == norm, Metric.org_id == org_id)
        )
    ).first()
    if syn:
        return (await session.scalars(select(Metric).where(Metric.id == syn.metric_id))).first()
    return None


async def metric_by_id(session: AsyncSession, metric_id: int) -> Metric | None:
    return (await session.scalars(select(Metric).where(Metric.id == metric_id))).first()


async def metric_children(session: AsyncSession, metric_id: int) -> list[Metric]:
    return list(
        (await session.scalars(select(Metric).where(Metric.parent_id == metric_id))).all()
    )


async def add_synonym(session: AsyncSession, metric_id: int, text_norm: str) -> None:
    session.add(MetricSynonym(metric_id=metric_id, text=text_norm))


def visible_documents_condition(user_id: int | None):
    """SQLAlchemy-условие видимости документов: личные — только загрузившему.
    NULL в is_private трактуется как общий доступ (совместимость со старыми БД)."""
    shared = or_(Document.is_private.is_(False), Document.is_private.is_(None))
    if user_id is None:
        return shared
    return or_(shared, Document.uploaded_by == user_id)


async def series_for_metric(
    session: AsyncSession, org_id: int, metric_id: int, *, period_type: str | None = None,
    variant: str = "fact", user_id: int | None = None,
) -> list[dict]:
    """Хронологический ряд показателя по актуальным (не переизданным) документам
    с учётом приватности (user_id). variant='fact' — фактические значения
    (по умолчанию); 'plan'/'budget'/… — плановые варианты из шапок таблиц.
    При нескольких источниках за период приоритет у последнего загруженного."""
    if variant == "fact":
        variant_cond = or_(Fact.variant == "fact", Fact.variant.is_(None))
    else:
        variant_cond = Fact.variant == variant
    stmt = (
        select(Fact, Document)
        .join(Document, Document.id == Fact.document_id)
        .where(
            Fact.org_id == org_id,
            Fact.metric_id == metric_id,
            Fact.needs_review.is_(False),
            Document.superseded_by_id.is_(None),
            variant_cond,
            visible_documents_condition(user_id),
        )
        .order_by(Fact.period_end, Fact.document_id)
    )
    if period_type:
        stmt = stmt.where(Fact.period_type == period_type)
    rows = (await session.execute(stmt)).all()
    best: dict[date, dict] = {}
    for fact, doc in rows:
        best[fact.period_end] = {
            "period_type": fact.period_type,
            "period_label": fact.period_label,
            "period_start": fact.period_start,
            "period_end": fact.period_end,
            "value": as_float(fact.value),
            "unit": fact.unit,
            "currency": fact.currency,
            "sheet": fact.sheet,
            "cell_ref": fact.cell_ref,
            "document_id": doc.id,
            "document_name": doc.original_name,
        }
    return [best[k] for k in sorted(best)]


async def org_metrics(session: AsyncSession, org_id: int) -> list[Metric]:
    return list(
        (await session.scalars(select(Metric).where(Metric.org_id == org_id).order_by(Metric.name))).all()
    )


async def org_documents(session: AsyncSession, org_id: int, user_id: int | None = None) -> list[Document]:
    stmt = (
        select(Document)
        .where(Document.org_id == org_id, visible_documents_condition(user_id))
        .order_by(Document.uploaded_at.desc())
        .limit(50)
    )
    return list((await session.scalars(stmt)).all())


async def set_document_private(
    session: AsyncSession, document_id: int, user_id: int, private: bool | None = None
) -> Document | None:
    """Переключает личный доступ. Право есть только у загрузившего документ.
    private=None — инвертировать текущее состояние (личный <-> общий)."""
    doc = await session.get(Document, document_id)
    if doc is None or doc.uploaded_by != user_id:
        return None
    doc.is_private = (not doc.is_private) if private is None else private
    await audit(session, user_id, "document_private" if doc.is_private else "document_shared",
                f"{doc.original_name}: {'личный' if doc.is_private else 'общий'} доступ")
    return doc


# ---------------------------------------------------------------------------
# Версионирование документов (переиздания)
# ---------------------------------------------------------------------------

async def supersede_candidates(
    session: AsyncSession, org_id: int, new_doc: Document, limit: int = 3
) -> list[Document]:
    """Обработанные документы той же организации с пересекающимися периодами —
    кандидаты на замену новым документом (переиздание)."""
    stmt = (
        select(Document)
        .where(
            Document.org_id == org_id,
            Document.id != new_doc.id,
            Document.status == "processed",
            Document.superseded_by_id.is_(None),
            visible_documents_condition(new_doc.uploaded_by),
        )
        .order_by(Document.uploaded_at.desc())
        .limit(20)
    )
    new_periods = set((new_doc.meta or {}).get("periods") or [])
    if not new_periods:
        return []
    out = []
    for d in (await session.scalars(stmt)).all():
        if new_periods & set((d.meta or {}).get("periods") or []):
            out.append(d)
            if len(out) >= limit:
                break
    return out


async def supersedes(
    session: AsyncSession, old_doc_id: int, new_doc_id: int, *, org_id: int
) -> Document | None:
    """Помечает старый документ переизданным новым. Факты старой версии
    сохраняются, но исключаются из аналитических рядов. Оба документа должны
    принадлежать организации вызывающего (id приходят из callback-данных)."""
    old = await session.get(Document, old_doc_id)
    new = await session.get(Document, new_doc_id)
    if not old or not new or old.org_id != org_id or new.org_id != org_id or old.id == new.id:
        return None
    old.superseded_by_id = new.id
    return old


# ---------------------------------------------------------------------------
# Read-only режим для выполнения LLM-сгенерированного SQL
# ---------------------------------------------------------------------------

async def run_readonly(session: AsyncSession, stmt, params: dict | None = None):
    """Выполнить запрос гарантированно в режиме «только чтение».

    Раньше query_only выставлялся на сессии из общего пула, а снимался в
    finally. Если снятие не проходило (ошибка/отмена запроса), соединение
    возвращалось в пул с включённым query_only, и последующая обычная запись
    падала с «attempt to write a readonly database». Здесь режим выставляется
    на выделенном соединении и снимается до его возврата в пул.
    """
    dialect = session.bind.dialect.name
    conn = await session.connection()

    def _run(sync_conn):
        if dialect != "postgresql":
            sync_conn.exec_driver_sql("PRAGMA query_only=ON")
        else:
            sync_conn.exec_driver_sql("SET TRANSACTION READ ONLY")
        try:
            return sync_conn.execute(stmt, params or {})
        finally:
            if dialect != "postgresql":
                with contextlib.suppress(Exception):
                    sync_conn.exec_driver_sql("PRAGMA query_only=OFF")

    return await conn.run_sync(_run)


async def enforce_readonly(session: AsyncSession) -> None:
    """Включить режим «только чтение» на сессии.

    Для LLM-сгенерированного SQL используйте `run_readonly`: он снимает режим
    до возврата соединения в пул. Эта пара оставлена для явного управления
    сессией (тесты) и требует симметричного `release_readonly`.
    """
    dialect = session.bind.dialect.name
    if dialect == "postgresql":
        await session.execute(text("SET TRANSACTION READ ONLY"))
    else:
        await session.execute(text("PRAGMA query_only=ON"))


async def release_readonly(session: AsyncSession) -> None:
    """Снимает query_only на SQLite-соединении (уровень соединения в пуле)."""
    if session.bind.dialect.name != "postgresql":
        with contextlib.suppress(Exception):
            await session.execute(text("PRAGMA query_only=OFF"))


async def confirm_facts(session: AsyncSession, document_id: int, *, org_id: int) -> int:
    from sqlalchemy import update

    res = await session.execute(
        update(Fact)
        .where(Fact.document_id == document_id, Fact.org_id == org_id)
        .values(needs_review=False, confidence=1.0)
    )
    return res.rowcount or 0


async def get_meta(session: AsyncSession, key: str) -> str | None:
    row = await session.get(AppMeta, key)
    return row.value if row else None


async def set_meta(session: AsyncSession, key: str, value: str) -> None:
    row = await session.get(AppMeta, key)
    if row:
        row.value = value
    else:
        session.add(AppMeta(key=key, value=value))


async def list_digest_subs(session: AsyncSession) -> list[tuple[int, int]]:
    res = await session.execute(select(DigestSubscription.telegram_id, DigestSubscription.org_id))
    return [(int(t), int(o)) for t, o in res.all()]


async def subscribe_digest(session: AsyncSession, telegram_id: int, org_id: int) -> bool:
    """True — подписка создана, False — уже была."""
    existing = await session.get(DigestSubscription, telegram_id)
    if existing:
        return False
    session.add(DigestSubscription(telegram_id=telegram_id, org_id=org_id))
    await audit(session, telegram_id, "digest_subscribed")
    return True


async def delete_document(
    session: AsyncSession, org_id: int, document_id: int,
) -> dict | None:
    """Удалить ОДИН документ со всеми его данными.

    Нужно, чтобы убрать файл из памяти прямо из чата: иначе единственный способ
    — /clear, который стирает сразу всё, включая историю чата.

    Возвращает {'name', 'path', 'facts', 'chunks'} — путь к оригиналу удаляет
    вызывающий после commit (при откате БД файл должен уцелеть). None, если
    документа нет или он из другой организации.

    Документ-журнал расходов (uploaded_by=0) удалять нельзя: это контейнер
    операций «расход: 1500 кофе», а не загруженный файл.
    """
    from sqlalchemy import delete, func, update

    doc = await session.get(Document, document_id)
    if doc is None or doc.org_id != org_id:
        return None
    if doc.uploaded_by == 0:
        return None
    facts = (
        await session.execute(select(func.count()).select_from(Fact).where(Fact.document_id == doc.id))
    ).scalar_one()
    chunks = (
        await session.execute(select(func.count()).select_from(Chunk).where(Chunk.document_id == doc.id))
    ).scalar_one()
    # переиздания ссылаются на этот документ — ссылку снимаем, иначе запись
    # останется «переизданной» без актуальной версии
    await session.execute(
        update(Document).where(Document.superseded_by_id == doc.id).values(superseded_by_id=None)
    )
    for model in (Fact, Chunk, LedgerOperation):
        await session.execute(delete(model).where(model.document_id == doc.id))
    name, path = doc.original_name, doc.stored_path
    await session.delete(doc)
    bump_index_version()   # чанки удалены — BM25-кэш недействителен
    from . import qdrant_store

    await qdrant_store.delete_document(doc.id)
    await _drop_orphan_metrics(session, org_id)
    await audit(session, 0, "document_deleted", f"{name} (id={document_id})")
    return {"name": name, "path": path, "facts": facts, "chunks": chunks}


async def _drop_orphan_metrics(session: AsyncSession, org_id: int) -> int:
    """Убрать показатели, на которые больше ничего не ссылается.

    Два прохода: сначала листья, затем разделы, опустевшие после удаления
    листьев. Без этого словарь показателей копил бы «призраков» удалённых
    документов и они попадали бы в подсказки при уточнении.
    """
    from sqlalchemy import delete, exists

    child = aliased(Metric)
    removed = 0
    while True:
        has_fact = exists().where(Fact.metric_id == Metric.id)
        has_child = exists().where(child.parent_id == Metric.id)
        orphan = (await session.scalars(
            select(Metric.id).where(Metric.org_id == org_id, ~has_fact, ~has_child)
        )).all()
        if not orphan:
            return removed
        await session.execute(delete(MetricSynonym).where(MetricSynonym.metric_id.in_(orphan)))
        await session.execute(delete(Metric).where(Metric.id.in_(orphan)))
        removed += len(orphan)


async def delete_user_documents(session: AsyncSession, org_id: int, user_id: int) -> list[str]:
    """Удаляет все документы, загруженные пользователем: факты, чанки, операции
    выписок, записи документов; показатели словаря, оставшиеся без фактов и
    дочерних, тоже убираются. Возвращает пути оригиналов — файлы удаляет
    вызывающий после commit (иначе при откате данные потеряются раньше файлов)."""
    from sqlalchemy import delete, update

    docs = (await session.scalars(
        select(Document).where(Document.org_id == org_id, Document.uploaded_by == user_id)
    )).all()
    if not docs:
        return []
    ids = [d.id for d in docs]
    paths = [d.stored_path for d in docs]
    await session.execute(update(Document).where(Document.superseded_by_id.in_(ids)).values(superseded_by_id=None))
    for model in (Fact, Chunk, LedgerOperation):
        await session.execute(delete(model).where(model.document_id.in_(ids)))
    await session.execute(delete(Document).where(Document.id.in_(ids)))
    bump_index_version()  # чанки удалены — BM25-кэш недействителен
    from . import qdrant_store

    for doc_id in ids:
        await qdrant_store.delete_document(doc_id)
    # словарь: показатели без фактов и без детей больше ни на что не указывают
    await _drop_orphan_metrics(session, org_id)
    await audit(session, user_id, "documents_cleared", f"{len(ids)} документов")
    return paths


async def cleanup_orphan_uploads(session: AsyncSession, settings: Settings) -> tuple[int, int]:
    """Удалить из каталога загрузок файлы, на которые нет ссылок в БД.

    Оригиналы копятся вечно: демо-прогоны, тесты и неудачные загрузки
    оставляют файлы, а запись документа откатывается. Запускается при старте.
    Возвращает (удалено файлов, освобождено байт).
    """
    upload_dir = settings.upload_dir
    if not upload_dir.is_dir():
        return 0, 0
    referenced = {
        os.path.basename(p)
        for p in (await session.scalars(select(Document.stored_path))).all()
        if p
    }
    removed = freed = 0
    for entry in upload_dir.iterdir():
        if not entry.is_file() or entry.name in referenced:
            continue
        try:
            size = entry.stat().st_size
            entry.unlink()
        except OSError as e:
            log.warning("Не удалось удалить сироту %s: %s", entry, e)
            continue
        removed += 1
        freed += size
    if removed:
        log.info("Очистка загрузок: удалено %d файлов-сирот (%.1f МБ)", removed, freed / 1e6)
    return removed, freed


async def unsubscribe_digest(session: AsyncSession, telegram_id: int) -> bool:
    row = await session.get(DigestSubscription, telegram_id)
    if row is None:
        return False
    await session.delete(row)
    await audit(session, telegram_id, "digest_unsubscribed")
    return True


async def ledger_months(session: AsyncSession, org_id: int, header: str) -> list[dict]:
    """Месяцы с операциями выписки: [{label, start, total, count}] по возрастанию.

    Сумма считается в Decimal (точные деньги), наружу отдаётся float —
    на этой границе работают форматирование и отчёты.
    """
    ops = (await session.scalars(
        select(LedgerOperation).where(
            LedgerOperation.org_id == org_id,
            LedgerOperation.header == header.casefold(),
        )
    )).all()
    agg: dict[str, dict] = {}
    totals: dict[str, Decimal] = {}
    for o in ops:
        a = agg.setdefault(o.period_label, {"label": o.period_label, "start": o.period_start,
                                            "total": 0.0, "count": 0})
        totals[o.period_label] = totals.get(o.period_label, Decimal(0)) + (o.value or Decimal(0))
        a["count"] += 1
    for label, total in totals.items():
        agg[label]["total"] = float(total)
    return sorted(agg.values(), key=lambda x: x["start"])


async def ledger_ops_for_month(
    session: AsyncSession, org_id: int, header: str, label: str
) -> list[LedgerOperation]:
    """Операции выписки за конкретный месяц (для просмотра и категорий)."""
    return list((await session.scalars(
        select(LedgerOperation)
        .where(
            LedgerOperation.org_id == org_id,
            LedgerOperation.header == header.casefold(),
            LedgerOperation.period_label == label,
        )
        .order_by(LedgerOperation.date_actual, LedgerOperation.id)
    )).all())
