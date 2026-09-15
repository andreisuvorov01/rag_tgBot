"""Конвейер обработки документов: файл -> данные + векторный индекс.

Шаги: регистрация и дедупликация -> парсинг формата -> нормализация ->
сопоставление показателей со словарём (точно -> синоним -> семантический
поиск -> создание нового) -> сохранение фактов и чанков с эмбеддингами ->
аудит. Спорные значения помечаются needs_review для подтверждения человеком.
"""
from __future__ import annotations

import asyncio
import calendar
import hashlib
import logging
import re
import uuid
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import date as _date
from decimal import Decimal
from html import escape as html_escape
from pathlib import Path
from statistics import median

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import Settings
from ..embeddings import EmbeddingService
from ..formatting import fmt_money
from ..money import to_decimal
from ..ocr import VlmOcr
from ..storage import (
    Chunk,
    Document,
    Fact,
    Metric,
    add_synonym,
    audit,
    bump_index_version,
    find_metric_by_name,
    supersede_candidates,
)
from .load_excel import ParsedFact, disambiguate_by_section, load_csv, load_excel, load_html, load_ods
from .load_pdf_docx import load_docx, load_image, load_pdf
from .normalize import Period, _guess_kind, slugify_code

log = logging.getLogger(__name__)

# progress(этап, доля 0..1) — бот рисует прогресс-бар; по умолчанию — тишина
ProgressFn = Callable[[str, float], Awaitable[None]]


async def _no_progress(stage: str, fraction: float) -> None:
    pass


EMBED_BATCH = 8  # чанков за один вызов модели: между пачками обновляется прогресс

@dataclass
class IngestReport:
    document_id: int | None = None
    status: str = "processed"  # processed | duplicate | failed
    message: str = ""
    facts: int = 0
    chunks: int = 0
    periods: list[str] = field(default_factory=list)
    new_metrics: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)
    # (id, имя) документов, которые новый документ может заменить как переиздание
    supersede_candidates: list[tuple[int, str]] = field(default_factory=list)

    def summary(self) -> str:
        from html import escape

        lines = [f"<b>{self.message}</b>"]
        if self.periods:
            shown = self.periods[:8]
            lines.append("Периоды: " + ", ".join(shown) + (f" …(+{len(self.periods) - 8})" if len(self.periods) > 8 else ""))
        lines.append(f"Показателей: {self.facts}, текстовых фрагментов: {self.chunks}")
        if self.new_metrics:
            names = [escape(n[:40] + ("…" if len(n) > 40 else "")) for n in self.new_metrics[:5]]
            more = f" …и ещё {len(self.new_metrics) - 5}" if len(self.new_metrics) > 5 else ""
            lines.append(f"Новых показателей в словаре: {len(self.new_metrics)} ({', '.join(names)}{more})")
        if self.supersede_candidates:
            lines.append(
                "Если это переиздание — замените один из прежних документов кнопкой ниже: "
                + ", ".join(f"«{escape(name)}»" for _, name in self.supersede_candidates)
            )
        if self.warnings:
            lines.append("⚠️ Замечания:")
            lines.extend(f"• {escape(w[:160])}" for w in self.warnings[:6])
            if len(self.warnings) > 6:
                lines.append(f"• …и ещё {len(self.warnings) - 6}")
        return "\n".join(lines)


_BARE_TOTAL_NAME = re.compile(r"(^|\s)(итого|всего)(\s|$)")


def _validate_totals(facts) -> list[str]:
    """Сверка арифметики: сумма строк-детей против «Итого/Всего» того же листа
    и периода. Дети берутся из раздела итога и того же типа показателя
    (расходы не суммируются с выручкой). Расхождение > 2% — предупреждение.

    Вариант входит в ключ группировки: план и факт одного периода — разные
    числа, и их суммирование давало ложные предупреждения на каждой
    план/факт-таблице.
    """
    by_period: dict[tuple[str, object, str], list] = defaultdict(list)
    for f in facts:
        by_period[(f.sheet, f.period.end, f.variant)].append(f)
    warns: list[str] = []
    for (sheet, _pend, _variant), fs in by_period.items():
        totals = [f for f in fs if f.is_total]
        if not totals:
            continue
        children = [f for f in fs if not f.is_total]
        if len(children) < 2:
            continue
        for t in totals:
            if not _BARE_TOTAL_NAME.search(t.metric_name):
                continue  # «Баланс (актив)» суммирует разделы, а не строки — не сверяем
            group = [c for c in children if c.section == t.section and c.parent_hint is None]
            if t.section is None or not group:
                # раздела нет — отбираем строки того же типа (расходы не суммируются с выручкой)
                group = [c for c in children if c.parent_hint is None]
                t_kind = _guess_kind(t.metric_name)
                if t_kind != "other":
                    kind_group = [c for c in group if _guess_kind(c.metric_name) == t_kind]
                    if len(kind_group) >= 2:
                        group = kind_group
            if len(group) < 2:
                continue
            ssum = sum(c.value for c in group)
            if t.value and abs(ssum - t.value) / abs(t.value) > 0.02:
                warns.append(
                    f"«{t.metric_name}» {t.period.label} ({sheet}): сумма строк "
                    f"{fmt_money(ssum, t.currency)} не сходится с итогом {fmt_money(t.value, t.currency)}"
                )
    return warns[:6]


async def _link_hierarchy(session: AsyncSession, metric_map: dict[str, int | None], parsed) -> None:
    """Проставляет parent_id по разделам листа и подсказкам «в т.ч.»."""
    hints: dict[str, tuple[str | None, str | None]] = {}
    for f in parsed.facts:
        section = (f.section or "").casefold() or None
        parent_hint = (f.parent_hint or "").casefold() or None
        hints.setdefault(f.metric_name, (section, parent_hint))
    for name, (section, parent_hint) in hints.items():
        child_id = metric_map.get(name)
        if not child_id:
            continue
        parent_id = metric_map.get(section) if section else None
        if parent_id is None and parent_hint:
            parent_id = metric_map.get(parent_hint)
        if parent_id and parent_id != child_id:
            child = await session.get(Metric, child_id)
            if child and child.parent_id is None:
                child.parent_id = parent_id


def _clean_name(raw: str) -> str:
    s = re.sub(r"[.,;:]+$", "", raw.strip()).strip()
    s = re.sub(r"\s+", " ", s)
    return s.casefold()


async def _resolve_metrics(
    session: AsyncSession, emb: EmbeddingService, settings: Settings, org_id: int, names: list[str]
) -> tuple[dict[str, int | None], list[str], dict[str, float]]:
    """Возвращает: имя -> metric_id (или None при неоднозначности),
    список созданных имён, словарь 'имя -> needs_review'."""
    result: dict[str, int | None] = {}
    created: list[str] = []
    review: dict[str, float] = {}
    existing = list(
        (await session.scalars(select(Metric).where(Metric.org_id == org_id))).all()
    )
    existing_vecs: list[tuple[int, np.ndarray | None]] = []
    for m in existing:
        vec = m.embedding
        if isinstance(vec, str):
            vec = np.array(json_loads(vec), dtype=np.float32)
        elif vec is not None:
            vec = np.asarray(vec, dtype=np.float32)
        existing_vecs.append((m.id, vec))

    doc_names = {_clean_name(n) for n in names}
    # векторы имён считаем одной пачкой (N вызовов модели -> 1): точные
    # совпадения со словарём в неё не попадают
    pending: list[str] = []
    for raw_name in sorted(set(names)):
        exact = await find_metric_by_name(session, org_id, _clean_name(raw_name))
        if exact:
            result[raw_name] = exact.id
        else:
            pending.append(raw_name)
    name_vecs = dict(zip(
        pending, await emb.embed_texts([_clean_name(r) for r in pending], is_query=True),
        strict=False,  # провайдер может вернуть меньше векторов — лишние имена просто не получат вектор
    )) if pending else {}

    for raw_name in pending:
        name = _clean_name(raw_name)
        # показатель мог появиться на предыдущей итерации (два написания одного имени)
        exact = await find_metric_by_name(session, org_id, name)
        if exact:
            result[raw_name] = exact.id
            continue
        vec = np.asarray(name_vecs[raw_name], dtype=np.float32)

        # более длинное имя, содержащее существующий показатель, — это
        # дочерняя разбивка (например, «аренда спецтехники москва»),
        # а не синоним родителя
        parent_id = next(
            (
                m.id
                for m in existing
                if len(name) > len(m.name) and m.name in name
            ),
            None,
        )
        if parent_id is None:
            # «активы» и «обязательства» не сливаются, как бы ни были похожи имена
            # («отложенные налоговые активы» ≠ «отложенные налоговые обязательства»)
            kind = _guess_kind(name)
            kinds = {m.id: m.kind for m in existing}
            names_by_id = {m.id: m.name for m in existing}
            best_id, best_score, second = None, -1.0, -1.0
            for mid, evec in existing_vecs:
                if evec is None or not evec.size:
                    continue
                if kind != "other" and kinds.get(mid, "other") not in ("other", kind):
                    continue
                if names_by_id.get(mid) in doc_names:
                    # две разные строки одного документа — разные показатели
                    # («чистая прибыль» и «валовая прибыль»), сливать нельзя
                    continue
                denom = float(np.linalg.norm(vec) * np.linalg.norm(evec)) or 1.0
                score = float(np.dot(vec, evec) / denom)
                if score > best_score:
                    second, best_score, best_id = best_score, score, mid
                elif score > second:
                    second = score
            if best_score >= settings.metric_match_threshold and best_score - second > 0.08:
                result[raw_name] = best_id
                # актуальное имя — из последнего отчёта: «Аренда СММ» (2024) ->
                # «Аренда спецтехники» (2025); старое остаётся синонимом
                merged = next(m for m in existing if m.id == best_id)
                if merged.name != name:
                    await add_synonym(session, best_id, merged.name)
                    merged.name = name
                    merged.embedding = vec.tolist()
                    existing_vecs = [(mid, vec if mid == best_id else ev) for mid, ev in existing_vecs]
                continue
            if best_score >= settings.metric_match_threshold:
                # близко два кандидата — создаём с флагом проверки
                review[raw_name] = best_score

        code = slugify_code(name)
        base_code, i = code, 2
        while (await session.scalars(
            select(Metric).where(Metric.org_id == org_id, Metric.code == code)
        )).first():
            code = f"{base_code}_{i}"
            i += 1
        metric = Metric(
            org_id=org_id,
            code=code,
            name=name,
            kind=_guess_kind(name),
            parent_id=parent_id,
            embedding=vec.tolist(),
        )
        session.add(metric)
        await session.flush()
        existing.append(metric)
        existing_vecs.append((metric.id, vec))
        result[raw_name] = metric.id
        created.append(name)
    return result, created, review


def json_loads(s: str):  # локальный импорт чтобы не тащить json в сигнатуры
    import json

    return json.loads(s)


async def fixup_metric_kinds(session: AsyncSession, org_id: int) -> int:
    """Пересчитывает тип показателей, созданных до появления новых правил
    (например, «инн» из выписки должен стать identifier, а не other)."""
    from ..storage import org_metrics

    changed = 0
    for m in await org_metrics(session, org_id):
        if m.kind != "other":
            continue
        new_kind = _guess_kind(m.name)
        if new_kind != "other":
            m.kind = new_kind
            changed += 1
    return changed


def check_archive_bomb(path: Path, settings: Settings) -> str | None:
    """Проверка zip-архива (xlsx/docx/ods) ДО распаковки.

    openpyxl/python-docx/odfpy материализуют книгу целиком, а лимит строк
    проверяется уже после — 20 МБ сжатого файла может развернуться в десятки
    гигабайт и уронить бот. Считаем суммарный распакованный размер и число
    элементов по метаданным архива, не распаковывая его.

    Возвращает текст ошибки либо None, если файл безопасен.
    """
    import zipfile

    try:
        with zipfile.ZipFile(path) as zf:
            infos = zf.infolist()
            if len(infos) > settings.max_archive_entries:
                return (f"в архиве {len(infos)} элементов — больше допустимых "
                        f"{settings.max_archive_entries}")
            total = sum(i.file_size for i in infos)
            budget = settings.max_uncompressed_mb * 1024 * 1024
            if total > budget:
                return (f"распакованный размер {total / 1024 / 1024:.0f} МБ больше "
                        f"допустимых {settings.max_uncompressed_mb} МБ")
    except zipfile.BadZipFile:
        return None  # не zip — парсер сам решит (например .xls)
    except OSError as e:
        return f"не удалось прочитать архив: {e}"
    return None


async def process_document(
    session: AsyncSession,
    emb: EmbeddingService,
    settings: Settings,
    *,
    org_id: int,
    user_id: int,
    original_name: str,
    content: bytes,
    progress: ProgressFn = _no_progress,
) -> IngestReport:
    report = IngestReport()
    ext = Path(original_name).suffix.lower().lstrip(".")
    file_hash = hashlib.sha256(content).hexdigest()

    dup = (
        await session.scalars(
            select(Document).where(Document.org_id == org_id, Document.file_hash == file_hash)
        )
    ).first()
    if dup:
        # файл уже обработан ранее — показываем реальные данные существующей версии,
        # а не нули нового вызова
        meta = dup.meta or {}
        report.status = "duplicate"
        report.document_id = dup.id
        report.facts = meta.get("facts", 0)
        report.chunks = meta.get("chunks", 0)
        report.periods = meta.get("periods", [])
        report.new_metrics = []
        report.warnings = meta.get("warnings", [])
        report.message = (
            f"Этот файл уже загружен и обработан («{html_escape(dup.original_name)}»). "
            f"Его данные уже в базе — можно задавать вопросы. Список: /documents"
        )
        return report

    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    safe_name = f"{uuid.uuid4().hex[:8]}_{re.sub(r'[^\w.\-]', '_', original_name)}"
    stored = settings.upload_dir / safe_name
    stored.write_bytes(content)

    if ext in ("xlsx", "docx", "ods"):
        bomb = check_archive_bomb(stored, settings)
        if bomb:
            stored.unlink(missing_ok=True)
            report.status = "failed"
            report.message = f"Файл отклонён: {bomb}."
            return report

    doc = Document(
        org_id=org_id,
        original_name=original_name,
        stored_path=str(stored),
        file_hash=file_hash,
        doc_type=ext,
        uploaded_by=user_id,
        status="processing",
    )
    session.add(doc)
    await session.flush()
    try:
        report = await _extract_and_store(session, emb, settings, doc, user_id, progress)
    except Exception:
        # запись документа откатится вместе с сессией — файл-сирота не нужен
        stored.unlink(missing_ok=True)
        raise
    report.supersede_candidates = [
        (d.id, d.original_name) for d in await supersede_candidates(session, org_id, doc)
    ]
    await audit(session, user_id, "document_processed", f"{original_name}: {report.facts} фактов, {report.chunks} чанков")
    report.message = f"Документ «{html_escape(original_name)}» обработан"
    return report


async def _store_operations(session: AsyncSession, org_id: int, doc: Document, ledger_tables: list) -> int:
    """Сохраняет операции выписок построчно (для просмотра «операции за месяц»
    и категорий). Прежние операции документа заменяются. Кап 5000 строк."""
    from sqlalchemy import delete

    from ..categories import categorize
    from ..storage import LedgerOperation

    await session.execute(delete(LedgerOperation).where(LedgerOperation.document_id == doc.id))
    count = 0
    for t in ledger_tables:
        for period, value, description in t.rows:
            if count >= 5000:
                break
            # просмотр группируется по месяцам: метка и границы — месячные
            y, mth = period.end.year, period.end.month
            last = calendar.monthrange(y, mth)[1]
            session.add(LedgerOperation(
                org_id=org_id,
                document_id=doc.id,
                header=t.header.casefold(),
                period_label=f"{mth:02d}.{y}",
                period_start=_date(y, mth, 1),
                period_end=_date(y, mth, last),
                date_actual=period.start,
                description=description,
                value=value,
                currency=t.currency,
                category=categorize(description),
            ))
            count += 1
    return count


async def _build_metric_cards(
    session: AsyncSession, emb: EmbeddingService, org_id: int, doc: Document,
    metric_map: dict[str, int | None],
) -> int:
    """Карточка показателя: имя + тип + раздел + вся история значений —
    единый векторный документ. Запрос «дебиторка» попадает в карточку
    «дебиторская задолженность» со всеми периодами сразу. Старая карточка
    метрики заменяется (актуальность после каждой загрузки).

    Значения собираются ТОЛЬКО из общих документов (user_id=None): карточка
    привязывается к текущему документу, который может быть общим, поэтому
    включение в неё значений личного документа владельца утекало бы их
    коллегам по организации.
    """
    from ..formatting import fmt_money
    from ..storage import Chunk, Metric, series_for_metric

    cards: list[tuple[str, str]] = []
    seen: set[int] = set()
    for mid in metric_map.values():
        if mid is None or mid in seen:
            continue
        seen.add(mid)
        metric = await session.get(Metric, mid)
        if metric is None or metric.kind == "identifier":
            continue
        rows = await series_for_metric(session, org_id, mid, user_id=None)
        if not rows:
            continue
        parent = await session.get(Metric, metric.parent_id) if metric.parent_id else None
        unit_note = f", {rows[0]['unit']}" if rows[0]["unit"] else ""
        vals = "; ".join(
            f"{r['period_label']} — {fmt_money(r['value'], r['currency'])}" for r in rows[:24]
        )
        card = (
            f"Карточка показателя «{metric.name}»{unit_note} (тип: {metric.kind}"
            + (f"; раздел «{parent.name}»" if parent else "")
            + f"). Значения по периодам: {vals}."
        )
        section = f"карточка:{metric.name}"
        from sqlalchemy import delete

        await session.execute(
            delete(Chunk).where(Chunk.org_id == org_id, Chunk.section == section)
        )
        cards.append((card, section))

    if not cards:
        return 0
    vectors = await emb.embed_passages([text for text, _ in cards])
    for (text, section), vec in zip(cards, vectors, strict=False):
        session.add(Chunk(
            org_id=org_id, document_id=doc.id, body=text,
            page=None, section=section, embedding=vec,
        ))
    return len(cards)


def _ledger_to_facts(ledger_tables: list) -> list[ParsedFact]:
    """Помесячная агрегация выписок на уровне документа: транзакции (построчно,
    с описанием) с разных страниц/таблиц одного типа складываются в факты
    «метрика × месяц»."""
    from collections import defaultdict

    per_metric: dict[str, dict[tuple[int, int], Decimal]] = defaultdict(lambda: defaultdict(lambda: Decimal(0)))
    units: dict[str, tuple[str | None, str | None]] = {}
    sheets: dict[str, set[str]] = defaultdict(set)
    for t in ledger_tables:
        key = t.header.casefold()
        for period, value, _description in t.rows:
            k = (period.end.year, period.end.month)
            per_metric[key][k] = per_metric[key][k] + (value if isinstance(value, Decimal) else (to_decimal(value) or Decimal(0)))
        units[key] = (t.unit, t.currency)
        sheets[key].add(t.sheet)

    facts: list[ParsedFact] = []
    for header, months in per_metric.items():
        unit, currency = units.get(header, (None, None))
        for (y, mth), value in sorted(months.items()):
            last = calendar.monthrange(y, mth)[1]
            facts.append(ParsedFact(
                metric_name=header,
                period=Period("month", f"{mth:02d}.{y}", _date(y, mth, 1), _date(y, mth, last)),
                value=value,
                unit=unit,
                currency=currency,
                sheet=", ".join(sorted(sheets[header]))[:120],
                cell_ref="",
            ))
    return facts


async def _extract_and_store(
    session: AsyncSession, emb: EmbeddingService, settings: Settings, doc: Document, user_id: int,
    progress: ProgressFn = _no_progress,
) -> IngestReport:
    """Парсинг оригинального файла документа и запись результатов: факты,
    чанки, словарь показателей. Вызывается при загрузке нового файла и при
    команде «перечитать файл»."""
    report = IngestReport(document_id=doc.id)
    original_name = doc.original_name
    org_id = doc.org_id
    stored = Path(doc.stored_path)
    ext = doc.doc_type
    vlm = VlmOcr(settings)
    await progress("чтение файла", 0.05)
    # синхронные парсеры уходят в поток: разбор большого xlsx/html не
    # останавливает event loop (бот продолжает отвечать другим пользователям)
    if ext in ("xlsx", "xls"):
        parsed = await asyncio.to_thread(load_excel, str(stored))
    elif ext == "csv":
        parsed = await asyncio.to_thread(load_csv, str(stored))
    elif ext == "pdf":
        parsed = await load_pdf(str(stored), vlm_ocr=vlm)
    elif ext == "docx":
        parsed = await load_docx(str(stored))
    elif ext in ("html", "htm"):
        parsed = await asyncio.to_thread(load_html, str(stored))
    elif ext == "ods":
        parsed = await asyncio.to_thread(load_ods, str(stored))
    elif ext in ("png", "jpg", "jpeg", "tiff"):
        parsed = await load_image(str(stored), vlm_ocr=vlm)
    else:
        raise ValueError(f"формат .{ext} не поддерживается")
    await progress("сопоставление показателей", 0.25)
    report.warnings.extend(parsed.warnings)
    disambiguate_by_section(parsed.facts)
    report.warnings.extend(_validate_totals(parsed.facts))
    report.diagnostics = list(parsed.diagnostics)

    # --- ledger-режим: выписки «дата | операция | суммы» -> помесячные факты ---
    if parsed.ledger:
        ledger_facts = _ledger_to_facts(parsed.ledger)
        if ledger_facts:
            parsed.facts.extend(ledger_facts)
            report.diagnostics.append(
                f"ledger: таблиц-выписок {len(parsed.ledger)} -> помесячных фактов {len(ledger_facts)}"
            )
        ops_saved = await _store_operations(session, org_id, doc, parsed.ledger)
        report.diagnostics.append(f"операций сохранено для просмотра: {ops_saved}")

    # --- сопоставление со словарём показателей (включая разделы-группы) ---
    metric_map, created, review = {}, [], {}
    if parsed.facts or parsed.sections:
        metric_map, created, review = await _resolve_metrics(
            session, emb, settings, org_id,
            [f.metric_name for f in parsed.facts] + list(parsed.sections),
        )
    report.new_metrics = created
    await _link_hierarchy(session, metric_map, parsed)

    # --- защита от скачков масштаба: новое значение, отличающееся на порядки
    # от истории показателя, помечается для проверки человеком ---
    existing_vals: dict[int, list[Decimal]] = {}
    metric_ids = {metric_map[f.metric_name] for f in parsed.facts if metric_map.get(f.metric_name)}
    if metric_ids:
        hist = await session.execute(
            select(Fact.metric_id, Fact.value).where(
                Fact.metric_id.in_(metric_ids), Fact.needs_review.is_(False)
            )
        )
        for mid, val in hist.all():
            d = val if isinstance(val, Decimal) else to_decimal(val)
            if d is not None:
                existing_vals.setdefault(mid, []).append(d)

    def _scale_suspect(metric_id: int | None, value) -> bool:
        vals = existing_vals.get(metric_id or 0)
        if not vals or not value:
            return False
        med = median(vals)
        if not med:
            return False
        # Decimal, иначе деление Decimal на float бросает TypeError
        v = value if isinstance(value, Decimal) else to_decimal(value)
        m = med if isinstance(med, Decimal) else to_decimal(med)
        if v is None or m is None or not m:
            return False
        ratio = v / m
        return ratio > 50 or ratio < Decimal("0.02")

    # --- сохранение фактов ---
    periods: set[str] = set()
    for f in parsed.facts:
        metric_id = metric_map.get(f.metric_name)
        if metric_id is None:
            report.warnings.append(f"показатель «{f.metric_name}» пропущен: неоднозначное соответствие")
            continue
        needs_review = f.metric_name in review or _scale_suspect(metric_id, f.value)
        if needs_review:
            report.warnings.append(f"«{f.metric_name}» {f.period.label}: значение {f.value:.4g} сильно отличается от истории — требуется подтверждение")
        session.add(
            Fact(
                org_id=org_id,
                metric_id=metric_id,
                document_id=doc.id,
                period_type=f.period.ptype,
                period_label=f.period.label,
                period_start=f.period.start,
                period_end=f.period.end,
                value=f.value,
                unit=f.unit,
                currency=f.currency,
                variant=f.variant,
                sheet=f.sheet,
                cell_ref=f.cell_ref,
                needs_review=needs_review,
                confidence=0.5 if needs_review else 1.0,
            )
        )
        if _guess_kind(f.metric_name) != "identifier":
            periods.add(f.period.label)  # дата реквизитов («ИНН на 13.09.2026») — не период отчёта
    report.facts = len(parsed.facts)
    report.periods = sorted(periods)

    if report.facts == 0:
        if parsed.chunks:
            report.warnings.insert(
                0, "таблиц с колонками периодов не найдено — текст проиндексирован "
                   "(по нему отвечают вопросы «что говорится о…»), числовых рядов нет"
            )
        else:
            report.warnings.insert(
                0, "данные не извлечены: если это скан — настройте OCR (см. README); "
                   "если PDF/Excel без таблиц — система не нашла показателей"
            )

    # --- эмбеддинги и сохранение текстовых чанков ---
    if parsed.chunks:
        # Contextual Retrieval (Anthropic): перед векторизацией каждый чанк
        # аннотируется контекстом документа/раздела — снижает промахи поиска.
        # В базе храним чистый текст, вектор строится по аннотированному.
        annotated = []
        for chunk in parsed.chunks:
            prefix = f"Документ: {original_name}"
            if chunk.section:
                prefix += f", фрагмент: {chunk.section}"
            annotated.append(f"{prefix}. {chunk.text}")
        vectors: list[list[float]] = []
        for i in range(0, len(annotated), EMBED_BATCH):
            await progress(f"векторизация фрагментов {min(i + EMBED_BATCH, len(annotated))}/{len(annotated)}",
                           0.4 + 0.4 * i / len(annotated))
            vectors.extend(await emb.embed_passages(annotated[i:i + EMBED_BATCH]))
        chunk_rows = []
        for chunk, vec in zip(parsed.chunks, vectors, strict=False):
            # strict=False осознанно: при коротком ответе провайдера чанк без
            # вектора не индексируется, а не роняет всю загрузку
            row = Chunk(
                org_id=org_id,
                document_id=doc.id,
                body=chunk.text,
                page=chunk.page,
                section=chunk.section,
                embedding=vec,
            )
            session.add(row)
            chunk_rows.append(row)
        # зеркало неприватных чанков в Qdrant (если включён) — лучшие усилия
        if not doc.is_private:
            await session.flush()
            from app import qdrant_store

            await qdrant_store.upsert_chunks([
                {"id": r.id, "org_id": org_id, "document_id": doc.id,
                 "embedding": r.embedding, "body": r.body,
                 "page": r.page, "section": r.section}
                for r in chunk_rows
            ])
    report.chunks = len(parsed.chunks)
    # чанки изменились — BM25-кэш в rag.py должен перечитать корпус
    bump_index_version()

    # --- карточки показателей: один векторный документ на метрику со всей
    # историей и иерархией — запрос по любому названию находит полный контекст ---
    await progress("карточки показателей", 0.85)
    n_cards = await _build_metric_cards(session, emb, org_id, doc, metric_map)
    await progress("сохранение", 0.97)
    report.diagnostics.append(f"карточек показателей обновлено: {n_cards}")

    doc.status = "processed"
    doc.meta = {
        "sheets": parsed.sheets,
        "facts": report.facts,
        "chunks": report.chunks,
        "periods": report.periods,
        "warnings": report.warnings,
        "new_metrics": created,
        "diagnostics": report.diagnostics,
    }
    report.message = f"Документ «{html_escape(original_name)}» обработан"
    return report


async def reparse_document(
    session: AsyncSession, emb: EmbeddingService, settings: Settings, *,
    document_id: int, user_id: int, org_id: int | None = None,
    progress: ProgressFn = _no_progress,
) -> IngestReport:
    """«Перечитать файл»: заново парсит оригинал существующего документа
    обновлённым парсером, заменяя прежние факты и чанки. org_id — организация
    вызывающего: чужой документ (id из callback-данных) перечитать нельзя."""
    doc = await session.get(Document, document_id)
    if doc is None or (org_id is not None and doc.org_id != org_id):
        return IngestReport(status="failed", message="Документ не найден")
    if doc.is_private and doc.uploaded_by != user_id:
        return IngestReport(status="failed", message="Документ личный — перечитать может только владелец")
    if not Path(doc.stored_path).exists():
        return IngestReport(status="failed", message="Оригинальный файл не найден на диске")

    from sqlalchemy import delete

    from app import qdrant_store

    await session.execute(delete(Chunk).where(Chunk.document_id == doc.id))
    await session.execute(delete(Fact).where(Fact.document_id == doc.id))
    await qdrant_store.delete_document(doc.id)
    bump_index_version()

    report = await _extract_and_store(session, emb, settings, doc, user_id, progress)
    report.message = f"Файл «{html_escape(doc.original_name)}» перечитан заново"
    await audit(session, user_id, "document_reparsed", f"{doc.original_name}: {report.facts} фактов, {report.chunks} чанков")
    return report
