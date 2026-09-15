"""Тесты исправлений: версионирование переизданий, read-only Text-to-SQL,
изоляция словаря по организациям, поведение OCR-загрузчика без Tesseract."""
import asyncio

import openpyxl
import pytest
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from app.config import settings
from app.embeddings import EmbeddingService
from app.ingest import load_pdf_docx
from app.ingest.pipeline import process_document
from app.llm import make_llm
from app.storage import (
    enforce_readonly,
    find_metric_by_name,
    make_engine,
    make_sessionmaker,
    register_user,
    release_readonly,
    series_for_metric,
    supersedes,
)


def _wb(path, rent_2025: float):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Показатель", "2024", "2025"])
    ws.append(["Выручка", "41200", "45800"])
    ws.append(["Аренда спецтехники", "9800", str(rent_2025)])
    wb.save(path)


async def _setup(tmp_path):
    settings.database_url = f"sqlite+aiosqlite:///{tmp_path}/t.db"
    settings.data_dir = tmp_path
    engine = await make_engine(settings)
    return engine, make_sessionmaker(engine), EmbeddingService(settings)


async def test_supersede_flow(tmp_path):
    engine, sessions, emb = await _setup(tmp_path)
    llm = make_llm(settings)

    p1 = tmp_path / "v1.xlsx"
    _wb(p1, 11400)
    p2 = tmp_path / "v2.xlsx"
    _wb(p2, 12000)  # исправленные данные за тот же период

    async with sessions() as s:
        r1 = await process_document(s, emb, settings, org_id=1, user_id=1,
                                    original_name="report.xlsx", content=p1.read_bytes())
        await s.commit()
        r2 = await process_document(s, emb, settings, org_id=1, user_id=1,
                                    original_name="report_fixed.xlsx", content=p2.read_bytes())
        await s.commit()

    assert r1.status == "processed" and r2.status == "processed"
    # новый документ с пересекающимися периодами видит первый как кандидата
    assert (r1.document_id, "report.xlsx") in r2.supersede_candidates

    async with sessions() as s:
        m = await find_metric_by_name(s, 1, "аренда спецтехники")
        rows_before = await series_for_metric(s, 1, m.id)
        assert {r["period_label"] for r in rows_before} == {"2024", "2025"}

        old = await supersedes(s, r1.document_id, r2.document_id, org_id=1)
        await s.commit()
    assert old is not None and old.superseded_by_id == r2.document_id

    async with sessions() as s:
        m = await find_metric_by_name(s, 1, "аренда спецтехники")
        rows_after = await series_for_metric(s, 1, m.id)
    values = {r["period_label"]: r["value"] for r in rows_after}
    # данные переизданного документа исключены, остались значения новой версии
    # (в тестовой книге единицы не указаны — множитель 1)
    assert values["2025"] == 12_000.0
    assert values["2024"] == 9_800.0
    await llm.close()
    await engine.dispose()


async def test_text_to_sql_readonly_enforced(tmp_path):
    engine, sessions, emb = await _setup(tmp_path)
    llm = make_llm(settings)
    async with sessions() as s:
        await enforce_readonly(s)
        # именно ошибка БД, а не любое исключение: так тест поймает и регресс,
        # при котором запись проходит
        with pytest.raises(OperationalError):
            await s.execute(text("INSERT INTO organizations(name) VALUES ('взлом')"))
        await release_readonly(s)
        await s.rollback()
        # после снятия ограничения запись работает (сырой SQL: все колонки явно)
        await s.execute(
            text("INSERT INTO organizations(id, name, created_at) VALUES (99, 'ok', '2026-01-01 00:00:00')")
        )
        await s.commit()
    await llm.close()
    await engine.dispose()


async def test_synonyms_org_scoped(tmp_path):
    engine, sessions, emb = await _setup(tmp_path)
    llm = make_llm(settings)
    async with sessions() as s:
        org1 = await register_user(s, 1, "u1", "ООО Один")
        org2 = await register_user(s, 2, "u2", "ООО Два")
        from app.ingest.pipeline import _resolve_metrics

        await _resolve_metrics(s, emb, settings, org1.org_id, ["выручка"])
        # в организации 1 создаём синоним вручную
        from app.storage import MetricSynonym

        m = (await s.execute(text("SELECT id FROM metrics LIMIT 1"))).scalar_one()
        s.add(MetricSynonym(metric_id=m, text="оборот"))
        await s.commit()
        # организация 2 не должна видеть чужой словарь
        assert await find_metric_by_name(s, org2.org_id, "оборот") is None
        assert await find_metric_by_name(s, org1.org_id, "оборот") is not None
    await llm.close()
    await engine.dispose()


@pytest.mark.skipif(load_pdf_docx.OCR_AVAILABLE, reason="pytesseract установлен — проверяется только fallback")
def test_image_without_ocr_warns_with_hint(tmp_path):
    from PIL import Image

    from app.ingest.load_pdf_docx import load_image

    p = tmp_path / "scan.png"
    Image.new("RGB", (20, 20), color="white").save(p)

    async def run():
        class DisabledVlm:
            enabled = False

            async def ocr_png(self, png_bytes: bytes) -> str:
                return ""

        return await load_image(str(p), vlm_ocr=DisabledVlm())

    doc = asyncio.run(run())
    assert doc.chunks == []
    assert any("Tesseract" in w or "OCR_VLM_ENABLED" in w for w in doc.warnings)
