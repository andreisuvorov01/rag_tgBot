"""Тесты лучших практик, заимствованных из аналогов:
гибридный поиск BM25+RRF (Onyx), сезонный наив и multi-fold бэктест (Nixtla),
перекрытие чанков и контекстные аннотации (Anthropic), план/факт и drill-down
(Datarails/Copilot for Finance)."""
import calendar
from datetime import date

import openpyxl

from app.analytics import forecast
from app.config import settings
from app.embeddings import EmbeddingService
from app.ingest.normalize import split_chunks
from app.ingest.pipeline import process_document
from app.llm import make_llm
from app.qa import AnswerPipeline
from app.rag import bm25_search, hybrid_search, rrf_fuse
from app.storage import make_engine, make_sessionmaker

# ------------------------------------------------------------ гибридный поиск

def test_bm25_ranks_keyword_match_first():
    corpus = [
        {"id": 1, "body": "Выручка компании стабильно росла три года подряд."},
        {"id": 2, "body": "Основные риски связаны с зависимостью от единственного поставщика комплектующих."},
        {"id": 3, "body": "Зарплаты сотрудников индексируются ежегодно."},
    ]
    hits = bm25_search(corpus, "риски поставок", k=3)
    assert hits[0]["id"] == 2
    hits = bm25_search(corpus, "зарплаты сотрудников", k=3)
    assert hits[0]["id"] == 3


def test_rrf_fusion_prefers_consensus():
    a = [{"id": 1, "body": "a"}, {"id": 2, "body": "b"}, {"id": 3, "body": "c"}]
    b = [{"id": 2, "body": "b"}, {"id": 1, "body": "a"}, {"id": 4, "body": "d"}]
    fused = rrf_fuse([a, b], top=4)
    # элементы 1 и 2 присутствуют в обоих списках — выше одиночников 3 и 4
    top2 = {fused[0]["id"], fused[1]["id"]}
    assert top2 == {1, 2}
    assert fused[0]["rrf_score"] > fused[2]["rrf_score"]


async def test_hybrid_search_end_to_end(tmp_path):
    settings.database_url = f"sqlite+aiosqlite:///{tmp_path}/t.db"
    settings.data_dir = tmp_path
    engine = await make_engine(settings)
    sessions = make_sessionmaker(engine)
    emb = EmbeddingService(settings)

    import io

    import docx as docx_lib

    d = docx_lib.Document()
    d.add_paragraph("Риски: зависимость от единственного поставщика логистических услуг.")
    buf = io.BytesIO()
    d.save(buf)
    async with sessions() as s:
        await process_document(s, emb, settings, org_id=1, user_id=1,
                               original_name="notes.docx", content=buf.getvalue())
        await s.commit()
        found = await hybrid_search(s, emb, 1, "риски поставщиков", k=3)
    assert found and "поставщик" in found[0]["body"]
    assert all("rrf_score" in f for f in found)
    await engine.dispose()


# ------------------------------------------------------------ чанки

def test_split_chunks_overlap():
    text = "\n\n".join(" ".join(f"слово{i}" for i in range(100)) for _ in range(3))
    chunks = split_chunks(text, max_words=120)
    assert len(chunks) >= 2
    # перекрытие: конец предыдущего чанка встречается в начале следующего
    tail_words = chunks[0].split()[-8:]
    assert " ".join(tail_words) in chunks[1]


# ------------------------------------------------------------ прогноз

def test_forecast_monthly_includes_seasonal_naive():
    points = []
    for i in range(24):
        y, m = 2020 + i // 12, i % 12 + 1
        last = calendar.monthrange(y, m)[1]
        points.append(type("P", (), {
            "label": f"{m:02d}.{y}", "start": date(y, m, 1), "end": date(y, m, last),
            "value": 100.0 + i, "source": "t", "ptype": "month",
        })())
    fc = forecast(points, target_year=2027)
    assert "seasonal_naive" in fc["methods"]
    # бэктест теперь средний по нескольким складам
    assert fc["backtest_error_pct"] >= 0


# ------------------------------------------------------------ план/факт и breakdown

def _wb_plan(path):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Показатель", "2024", "2024 (план)"])
    ws.append(["Выручка", "41 200", "40 000"])
    wb.save(path)


def _wb_breakdown(path):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Показатель", "2025"])
    ws.append(["Расходы"])  # раздел
    ws.append(["Зарплаты", "500"])
    ws.append(["Аренда", "300"])
    ws.append(["Итого", "800"])
    wb.save(path)


async def _pipeline(tmp_path):
    settings.database_url = f"sqlite+aiosqlite:///{tmp_path}/t.db"
    settings.data_dir = tmp_path
    engine = await make_engine(settings)
    sessions = make_sessionmaker(engine)
    emb = EmbeddingService(settings)
    llm = make_llm(settings)
    return engine, sessions, emb, llm, AnswerPipeline(sessions, emb, llm, settings, org_names=[])


async def test_plan_fact_variance(tmp_path):
    engine, sessions, emb, llm, pipeline = await _pipeline(tmp_path)
    p = tmp_path / "pf.xlsx"
    _wb_plan(p)
    async with sessions() as s:
        await process_document(s, emb, settings, org_id=1, user_id=1,
                               original_name="pf.xlsx", content=p.read_bytes())
        await s.commit()
    outcome = await pipeline.answer(1, 1, "Насколько факт 2024 отличается от плана по выручке?")
    assert "План/факт" in outcome.text
    assert "+3,0%" in outcome.text  # (41200-40000)/40000
    assert "40,00 тыс" in outcome.text or "40 000" in outcome.text
    await llm.close()
    await engine.dispose()


async def test_breakdown_composition(tmp_path):
    engine, sessions, emb, llm, pipeline = await _pipeline(tmp_path)
    p = tmp_path / "bd.xlsx"
    _wb_breakdown(p)
    async with sessions() as s:
        await process_document(s, emb, settings, org_id=1, user_id=1,
                               original_name="bd.xlsx", content=p.read_bytes())
        await s.commit()
    outcome = await pipeline.answer(1, 1, "Из чего состоит итого расходов за 2025 год?")
    assert "Состав" in outcome.text
    assert "зарплаты" in outcome.text and "аренда" in outcome.text
    assert "62,5" in outcome.text  # доля зарплат 500/800
    await llm.close()
    await engine.dispose()
