"""Тесты третьей волны практик: графовая оркестрация агентов, ACL документов
(permission-aware retrieval, практика Onyx), prompt caching (Anthropic)."""
import json

import httpx
import openpyxl
import pytest

from app.config import settings
from app.embeddings import EmbeddingService
from app.graph import AgentGraph, GraphError
from app.ingest.pipeline import process_document
from app.llm import AnthropicLLM, make_llm
from app.qa import AnswerPipeline, visible_facts_subquery
from app.rag import hybrid_search
from app.storage import (
    make_engine,
    make_sessionmaker,
    series_for_metric,
    set_document_private,
    vector_search,
)

# ------------------------------------------------------------ графовый движок

async def test_graph_linear_and_branch():
    g = AgentGraph("t")
    async def a(st):
        st["path"] = ["a"]
        return st
    async def b(st):
        st["path"].append("b")
        return st
    async def c(st):
        st["path"].append("c")
        return st
    g.node("a", a).node("b", b).node("c", c).set_entry("a")
    g.branch("a", lambda st: "x", {"x": "b", "__default__": "c"}).edge("b", "c")
    state = await g.run({})
    assert state["path"] == ["a", "b", "c"] and state["trace"] == ["a", "b", "c"]


async def test_graph_step_limit_guard():
    g = AgentGraph("loop", max_steps=5)
    async def loop(st): return st
    g.node("a", loop).edge("a", "a").set_entry("a")
    with pytest.raises(GraphError):
        await g.run({})


# ------------------------------------------------------------ ACL документов

async def _setup(tmp_path, db="t.db"):
    settings.database_url = f"sqlite+aiosqlite:///{tmp_path}/{db}"
    settings.data_dir = tmp_path
    engine = await make_engine(settings)
    return engine, make_sessionmaker(engine), EmbeddingService(settings)


def _wb(path, value: str, year: str = "2024"):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Показатель", year])
    ws.append(["Выручка", value])
    wb.save(path)


async def test_private_document_hidden_from_others(tmp_path):
    engine, sessions, emb = await _setup(tmp_path)
    llm = make_llm(settings)
    pipeline = AnswerPipeline(sessions, emb, llm, settings, org_names=[])

    p1 = tmp_path / "shared.xlsx"
    _wb(p1, "100", year="2024")
    p2 = tmp_path / "secret.xlsx"
    _wb(p2, "999", year="2025")  # разные периоды: перекрытие не мешает проверке ACL

    async with sessions() as s:
        await process_document(s, emb, settings, org_id=1, user_id=1,
                               original_name="shared.xlsx", content=p1.read_bytes())
        r2 = await process_document(s, emb, settings, org_id=1, user_id=2,
                                    original_name="secret.xlsx", content=p2.read_bytes())
        await s.commit()
        # user2 делает свой документ личным
        doc = await set_document_private(s, r2.document_id, user_id=2, private=True)
        await s.commit()
    assert doc is not None and doc.is_private
    # не владелец не может менять доступ
    async with sessions() as s:
        assert await set_document_private(s, r2.document_id, user_id=3, private=False) is None

    async with sessions() as s:
        from app.storage import find_metric_by_name

        m = await find_metric_by_name(s, 1, "выручка")
        owner_rows = await series_for_metric(s, 1, m.id, user_id=2)
        other_rows = await series_for_metric(s, 1, m.id, user_id=3)
    assert {r["value"] for r in owner_rows} == {100.0, 999.0}
    assert {r["value"] for r in other_rows} == {100.0}

    # RAG: чужой личный фрагмент не находится
    async with sessions() as s:
        found_other = await hybrid_search(s, emb, 1, "выручка", k=5, user_id=3)
        found_owner = await hybrid_search(s, emb, 1, "выручка", k=5, user_id=2)
    bodies_other = " ".join(f["body"] for f in found_other)
    bodies_owner = " ".join(f["body"] for f in found_owner)
    assert "999" not in bodies_other and "999" in bodies_owner

    # Ответ владельцу содержит его данные, коллеге — нет
    out_owner = await pipeline.answer(1, 2, "Какая выручка?", metric_override="выручка")
    out_other = await pipeline.answer(1, 3, "Какая выручка?", metric_override="выручка")
    assert "999" in out_owner.text and "999" not in out_other.text

    await llm.close()
    await engine.dispose()


async def test_text_to_sql_uses_visible_facts(tmp_path):
    engine, sessions, emb = await _setup(tmp_path, db="t2.db")

    class SQLLLM:
        async def chat(self, messages, **kw):
            system = messages[0]["content"]
            if "[TASK=sql]" in system:
                return "SELECT m.name, SUM(f.value) AS total FROM facts_visible f JOIN metrics m ON m.id = f.metric_id WHERE f.needs_review = false GROUP BY m.name"
            return ""

        async def close(self):  # noqa: B027
            pass

    llm = SQLLLM()
    pipeline = AnswerPipeline(sessions, emb, llm, settings, org_names=[])

    p1 = tmp_path / "s1.xlsx"
    _wb(p1, "100")
    p2 = tmp_path / "s2.xlsx"
    _wb(p2, "999")
    async with sessions() as s:
        await process_document(s, emb, settings, org_id=1, user_id=1,
                               original_name="s1.xlsx", content=p1.read_bytes())
        r2 = await process_document(s, emb, settings, org_id=1, user_id=2,
                                    original_name="s2.xlsx", content=p2.read_bytes())
        await s.commit()
        await set_document_private(s, r2.document_id, user_id=2, private=True)
        await s.commit()

    # прямой вызов Text-to-SQL:
    async with sessions() as s:
        payload = await pipeline._text_to_sql(s, 1, 1, "сумма выручки по показателям")
    assert payload is not None and payload["type"] == "sql_result"
    assert "999" not in payload["table"] and "100" in payload["table"]

    # запрос без facts_visible отклоняется
    class RawFactsLLM(SQLLLM):
        async def chat(self, messages, **kw):
            if "[TASK=sql]" in messages[0]["content"]:
                return "SELECT * FROM facts WHERE value > 0"
            return ""
    pipeline.llm = RawFactsLLM()
    async with sessions() as s:
        assert await pipeline._text_to_sql(s, 1, 1, "взлом") is None

    await engine.dispose()


def test_visible_facts_subquery_has_uid_param():
    for dialect in ("postgresql", "sqlite"):
        q = visible_facts_subquery(dialect)
        assert ":uid" in q and "documents" in q and "facts" in q


# ------------------------------------------------------------ prompt caching

def test_anthropic_payload_cache_control():
    s = settings.model_copy(update={
        "llm_provider": "anthropic", "llm_model": "claude-sonnet-4-5", "prompt_cache": True,
    })
    messages = [
        {"role": "system", "content": "СИСТЕМНЫЙ ПРОМПТ"},
        {"role": "user", "content": "ВОПРОС + ДАННЫЕ"},
    ]
    payload = AnthropicLLM._build_payload(s, messages, temperature=0.2, max_tokens=900)
    assert payload["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert payload["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert payload["messages"][0]["content"][0]["text"] == "ВОПРОС + ДАННЫЕ"


async def test_anthropic_client_sends_cache_headers():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"content": [{"type": "text", "text": "ответ"}]})

    s = settings.model_copy(update={"llm_provider": "anthropic", "llm_api_key": "key"})
    llm = AnthropicLLM(s, transport=httpx.MockTransport(handler))
    text = await llm.chat([{"role": "system", "content": "S"}, {"role": "user", "content": "U"}])
    assert text == "ответ"
    assert captured["system"][0]["cache_control"]["type"] == "ephemeral"
    assert captured["max_tokens"] == settings.llm_max_tokens
    await llm.close()


def test_effective_provider_falls_back_without_package(monkeypatch):
    import importlib.util

    s = settings.model_copy(update={"embeddings_provider": "sentence_transformers"})
    orig = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec",
        lambda name: None if name == "sentence_transformers" else orig(name),
    )
    assert s.effective_embeddings_provider == "hash"


# ------------------------------------------------------------ идентификаторы

async def test_identifier_metrics_excluded_from_resolution(tmp_path):
    """«ИНН» из выписки — реквизит, а не показатель: не подбирается в ответах."""
    engine, sessions, emb = await _setup(tmp_path)
    llm = make_llm(settings)

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Реквизиты", "13.09.2026"])
    ws.append(["ИНН", "7949000000"])
    ws.append(["БИК", "044525225"])
    ws.append(["Выручка", "100"])
    p = tmp_path / "req.xlsx"
    wb.save(p)
    async with sessions() as s:
        await process_document(s, emb, settings, org_id=1, user_id=1,
                               original_name="req.xlsx", content=p.read_bytes())
        await s.commit()

        from app.storage import find_metric_by_name

        inn = await find_metric_by_name(s, 1, "инн")
        assert inn is not None and inn.kind == "identifier"
        bic = await find_metric_by_name(s, 1, "бик")
        assert bic is not None and bic.kind == "identifier"

        # семантический поиск показателей идентификаторы не возвращает
        found = await vector_search(s, org_id=1, query_vec=await emb.embed_query("инн"),
                                    kind="metric", k=5)
        assert all(f["name"] != "инн" for f in found)
    await llm.close()
    await engine.dispose()


# ------------------------------------------------------------ перечитать файл

async def test_reparse_document(tmp_path):
    engine, sessions, emb = await _setup(tmp_path)
    llm = make_llm(settings)

    p = tmp_path / "r.xlsx"
    _wb(p, "100")
    async with sessions() as s:
        r1 = await process_document(s, emb, settings, org_id=1, user_id=1,
                                    original_name="r.xlsx", content=p.read_bytes())
        await s.commit()

    from app.ingest.pipeline import reparse_document
    from app.storage import find_metric_by_name

    async with sessions() as s:
        report = await reparse_document(s, emb, settings,
                                        document_id=r1.document_id, user_id=1)
        await s.commit()
    assert report.status == "processed" and report.facts == 1

    # факт не задвоился после перечитывания
    async with sessions() as s:
        m = await find_metric_by_name(s, 1, "выручка")
        rows = await series_for_metric(s, 1, m.id, user_id=1)
    assert len(rows) == 1 and rows[0]["value"] == 100.0
    await llm.close()
    await engine.dispose()


def test_make_llm_anthropic():
    s = settings.model_copy(update={"llm_provider": "anthropic"})
    assert isinstance(make_llm(s), AnthropicLLM)


# ------------------------------------------------------------ переиндексация

async def test_reindex_after_provider_change(tmp_path):
    from app.main import check_embeddings_fingerprint
    from app.storage import get_meta

    engine, sessions, emb = await _setup(tmp_path)
    llm = make_llm(settings)

    p = tmp_path / "r.xlsx"
    _wb(p, "100")
    async with sessions() as s:
        await process_document(s, emb, settings, org_id=1, user_id=1,
                               original_name="r.xlsx", content=p.read_bytes())
        await s.commit()

    # «смена модели»: другая размерность делает старые векторы невалидными
    old_dim = settings.embeddings_dim
    try:
        settings.embeddings_dim = 512
        emb2 = EmbeddingService(settings)

        from scripts.reindex import FINGERPRINT_KEY, reindex_all

        n_chunks, n_metrics = await reindex_all(engine, emb2)
        assert n_chunks > 0
        async with sessions() as s:
            from sqlalchemy import select

            from app.storage import Chunk

            chunks = (await s.scalars(select(Chunk))).all()
            assert all(len(c.embedding) == 512 for c in chunks)
            stored = await get_meta(s, FINGERPRINT_KEY)
        assert "512" in stored

        # после переиндексации предупреждение не выдаётся (fingerprint совпадает)
        import contextlib
        from io import StringIO

        buf = StringIO()
        with contextlib.redirect_stdout(buf):
            await check_embeddings_fingerprint(engine, sessions)
        assert "ВНИМАНИЕ" not in buf.getvalue()
    finally:
        settings.embeddings_dim = old_dim

    await llm.close()
    await engine.dispose()
