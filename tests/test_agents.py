"""Тесты агентов: верификатор чисел (генерация→критика→повтор) и reranker."""
import openpyxl

from app.agents import verify_answer
from app.config import settings
from app.embeddings import EmbeddingService
from app.ingest.pipeline import process_document
from app.llm import MockLLM
from app.qa import AnswerPipeline
from app.rerank import BaseReranker, LLMReranker, make_reranker
from app.storage import make_engine, make_sessionmaker

# ------------------------------------------------------------ верификатор

def test_verify_answer_accepts_grounded_numbers():
    template = "Прогноз на 2026: 13,97 млн ₽ (интервал 13,12 — 14,82), рост +13,9% за 3 точек."
    assert verify_answer("Прогноз: 13,97 млн ₽, рост +13,9%.", template)[0]
    assert verify_answer("На 2026 год ожидается около 14 млн ₽.", template)[0]  # 14 в допуске 0.5%
    assert verify_answer("Отчёт за 2024 год.", template)[0]  # год


def test_verify_answer_rejects_invented_numbers():
    template = "Прогноз на 2026: 13,97 млн ₽."
    ok, foreign = verify_answer("Прогноз 15,73 млн ₽ — точная цифра.", template)
    # foreign — величина в базовых единицах (13,97 млн -> 1.397e7), поэтому
    # 15,73 млн приводится к 1.573e7: так замечание к повтору называет то же
    # число, в котором модель считает
    assert not ok and foreign == [15730000.0]
    ok, foreign = verify_answer("Выручка составила 41 200 555 руб.", template)
    assert not ok


def test_verify_answer_accepts_raw_payload_values():
    """Регресс: модель берёт числа из блока ДАННЫЕ, где они не отформатированы,
    а шаблон печатает их как «41,20 тыс. ₽». Обе формы должны приниматься."""
    payload = {
        "type": "factual",
        "metric": {"name": "выручка", "unit": "руб", "currency": "RUB"},
        "history": [{"label": "2024", "value": 41200.0, "source": "o.xlsx · Sheet1 · B2"}],
    }
    from app.formatting import render_answer

    template = render_answer(payload)
    assert verify_answer("Выручка за 2024 составила 41 200 руб.", template, payload)[0]
    assert verify_answer("Выручка за 2024 составила 41,20 тыс. руб.", template, payload)[0]
    # выдуманное всё равно отбивается
    assert not verify_answer("Выручка за 2024 составила 41 500 руб.", template, payload)[0]


# ------------------------------------------------------------ reranker

def test_llm_reranker_with_mock_keeps_rrf_order():
    r = LLMReranker(MockLLM())
    chunks = [{"id": 1, "body": "a"}, {"id": 2, "body": "b"}, {"id": 3, "body": "c"}]
    import asyncio

    result = asyncio.run(r.rerank("запрос", chunks, k=2))
    assert [c["id"] for c in result] == [1, 2]


def test_llm_reranker_failure_falls_back_to_rrf_order():
    class BrokenLLM:
        async def chat(self, *a, **kw):
            raise RuntimeError("api down")

    import asyncio

    r = LLMReranker(BrokenLLM())
    chunks = [{"id": 1, "body": "a"}, {"id": 2, "body": "b"}]
    result = asyncio.run(r.rerank("запрос", chunks, k=2))
    assert [c["id"] for c in result] == [1, 2]


def test_make_reranker_none():
    settings.reranker = "none"
    assert isinstance(make_reranker(settings, MockLLM()), BaseReranker)
    settings.reranker = "llm"


# ------------------------------------------------------------ контур критики

class LyingComposerLLM(MockLLM):
    """Классифицирует и маршрутизирует как mock, но в композиторе придумывает числа."""

    async def chat(self, messages, *, temperature=None, json_mode=False, max_tokens=None):
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        if "[TASK=compose]" in system and "ОТКЛОНЕНА" not in messages[-1]["content"]:
            return "<b>Выручка за 2024 год: 41 200 555 руб.</b>"
        if "[TASK=compose]" in system:
            return "<b>Выручка за 2024 год: 41 200 555 руб.</b>"  # «врёт» и на повторе
        return await super().chat(messages, temperature=temperature, json_mode=json_mode, max_tokens=max_tokens)


async def test_verification_agent_falls_back_to_template(tmp_path):
    settings.database_url = f"sqlite+aiosqlite:///{tmp_path}/t.db"
    settings.data_dir = tmp_path
    engine = await make_engine(settings)
    sessions = make_sessionmaker(engine)
    emb = EmbeddingService(settings)
    llm = LyingComposerLLM()
    pipeline = AnswerPipeline(sessions, emb, llm, settings, org_names=[])

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Показатель", "2024"])
    ws.append(["Выручка", "41200"])
    p = tmp_path / "r.xlsx"
    wb.save(p)
    async with sessions() as s:
        await process_document(s, emb, settings, org_id=1, user_id=1,
                               original_name="r.xlsx", content=p.read_bytes())
        await s.commit()

    outcome = await pipeline.answer(1, 1, "Какая выручка за 2024 год?")
    # выдуманное число верификатор отбил, отдан шаблонный ответ с данными
    assert "555" not in outcome.text
    assert "41,20 тыс" in outcome.text

    await llm.close() if hasattr(llm, "close") else None
    await engine.dispose()


async def test_verification_disabled_passes_llm_text(tmp_path):
    settings.database_url = f"sqlite+aiosqlite:///{tmp_path}/t2.db"
    settings.data_dir = tmp_path
    settings.verify_answers = False
    engine = await make_engine(settings)
    sessions = make_sessionmaker(engine)
    emb = EmbeddingService(settings)
    llm = LyingComposerLLM()
    pipeline = AnswerPipeline(sessions, emb, llm, settings, org_names=[])

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Показатель", "2024"])
    ws.append(["Выручка", "41200"])
    p = tmp_path / "r2.xlsx"
    wb.save(p)
    async with sessions() as s:
        await process_document(s, emb, settings, org_id=1, user_id=1,
                               original_name="r2.xlsx", content=p.read_bytes())
        await s.commit()

    outcome = await pipeline.answer(1, 1, "Какая выручка за 2024 год?")
    assert "41 200 555" in outcome.text  # проверка выключена — LLM-текст проходит как есть

    settings.verify_answers = True
    await engine.dispose()
