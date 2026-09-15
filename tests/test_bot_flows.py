"""Тесты потоков интерфейса через фейковый Telegram-харнесс (без сети):
сообщения-двойники записывают отправленное, BotApp исполняет реальные методы.

Именно слой UI пропускал прод-баги (InlineKeyboardMarkup, NameError:settings) —
этот файл закрывает дыру."""
from types import SimpleNamespace

import openpyxl

from app.bot import BotApp, main_menu_inline, menu_keyboard
from app.config import settings
from app.embeddings import EmbeddingService
from app.ingest.pipeline import process_document
from app.llm import make_llm
from app.qa import AnswerPipeline, QAOutcome
from app.storage import make_engine, make_sessionmaker, register_user


class FakeMessage:
    """Двойник aiogram Message: записывает всё отправленное."""

    def __init__(self, user_id: int = 1, chat_id: int = 100, text: str = ""):
        self.from_user = SimpleNamespace(id=user_id, full_name="Тест Юзер")
        self.chat = SimpleNamespace(id=chat_id)
        self.text = text
        self.sent: list[dict] = []
        self.answers_photo: list = []

    async def answer(self, text=None, reply_markup=None, parse_mode=None, caption=None, **kw):
        self.sent.append({"text": text, "reply_markup": reply_markup,
                          "parse_mode": parse_mode, "caption": caption})
        return FakeMessage(user_id=self.from_user.id, chat_id=self.chat.id)

    async def edit_text(self, text=None, reply_markup=None, **kw):
        self.sent.append({"text": text, "reply_markup": reply_markup, "edited": True})
        return self


class FakeCallback:
    def __init__(self, message: FakeMessage, data: str, user_id: int = 1):
        self.message = message
        self.data = data
        self.from_user = SimpleNamespace(id=user_id, full_name="Тест")
        self.answered: list = []

    async def answer(self, text=None, show_alert=False):
        self.answered.append(text)


def _buttons(kb) -> list[tuple[str, str | None]]:
    if kb is None:
        return []
    return [(b.text, b.callback_data) for row in kb.inline_keyboard for b in row] \
        if hasattr(kb, "inline_keyboard") else \
        [(b.text, None) for row in kb.keyboard for b in row]


async def _setup(tmp_path, seed_file: bool = True):
    settings.database_url = f"sqlite+aiosqlite:///{tmp_path}/t.db"
    settings.data_dir = tmp_path
    settings.embeddings_provider = "hash"
    settings.llm_provider = "mock"
    engine = await make_engine(settings)
    sessions = make_sessionmaker(engine)
    emb = EmbeddingService(settings)
    llm = make_llm(settings)
    pipeline = AnswerPipeline(sessions, emb, llm, settings, org_names=[])
    app = BotApp(settings, pipeline, sessions)
    async with sessions() as s:
        await register_user(s, 1, "Тест", "ООО Тест")
        await s.commit()
    if seed_file:
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.append(["Показатель", "2023", "2024"])
        ws.append(["Выручка", "100", "120"])
        ws.append(["Аренда", "50", "60"])
        ws.append(["Зарплаты", "40", "45"])
        src = tmp_path / "r.xlsx"
        wb.save(src)
        async with sessions() as s:
            await process_document(s, emb, settings, org_id=1, user_id=1,
                                   original_name="r.xlsx", content=src.read_bytes())
            await s.commit()
    return engine, sessions, emb, llm, pipeline, app


def test_menu_keyboards_content():
    bottom = _buttons(menu_keyboard())
    assert ("📊 Показатели", None) in bottom and ("📄 Документы", None) in bottom
    inline = _buttons(main_menu_inline())
    assert ("menu:metrics", ) and [c for _, c in inline] == ["menu:metrics", "menu:documents", "menu:help"]


async def test_metrics_flow_grouped(tmp_path):
    engine, sessions, emb, llm, pipeline, app = await _setup(tmp_path)
    msg = FakeMessage(user_id=1)
    await app._send_metrics(msg)
    text = "\n".join(x["text"] for x in msg.sent if x["text"])
    assert "Словарь показателей" in text
    assert "выручка" in text and "аренда" in text
    assert "📈 Доходы" in text  # группировка по типу
    # нижнее меню прилагается к ответу
    assert any(_buttons(x["reply_markup"]) and ("📊 Показатели", None) in _buttons(x["reply_markup"])
               for x in msg.sent if x["reply_markup"] is not None)
    await llm.close()
    await engine.dispose()


async def test_documents_flow_with_periods_and_reparse(tmp_path):
    engine, sessions, emb, llm, pipeline, app = await _setup(tmp_path)
    msg = FakeMessage(user_id=1)
    await app._send_documents(msg)
    text = "\n".join(x["text"] for x in msg.sent if x["text"])
    assert "r.xlsx" in text and "периоды: 2023, 2024" in text
    kb = next(x["reply_markup"] for x in msg.sent if x["reply_markup"] is not None)
    buttons = _buttons(kb)
    assert any("Перечитать" in t for t, _ in buttons)
    assert any(c and c.startswith("reparse:") for _, c in buttons)
    await llm.close()
    await engine.dispose()


async def test_answer_flow_with_quick_actions_and_chart(tmp_path):
    engine, sessions, emb, llm, pipeline, app = await _setup(tmp_path)
    outcome = await pipeline.answer(1, 1, "Какая выручка за 2024 год?",
                                    intent_override="factual")
    assert outcome.table_metric_id is not None

    msg = FakeMessage(user_id=1)
    await app._send_answer(msg, outcome, "Какая выручка за 2024 год?")
    text = "\n".join(x["text"] for x in msg.sent if x["text"])
    assert "120" in text
    # кнопки быстрых действий + Excel с правильными callback
    kb = next(x["reply_markup"] for x in msg.sent if x["reply_markup"] is not None)
    callbacks = [c for _, c in _buttons(kb)]
    assert f"act:forecast:{outcome.table_metric_id}" in callbacks
    assert f"act:compare:{outcome.table_metric_id}" in callbacks
    assert f"act:breakdown:{outcome.table_metric_id}" in callbacks
    assert f"xlsx:{outcome.table_metric_id}" in callbacks
    await llm.close()
    await engine.dispose()


async def test_intent_override_forecast_with_chart(tmp_path):
    """Кнопка «🔮 Прогноз»: intent_override форсирует прогноз, график строится."""
    engine, sessions, emb, llm, pipeline, app = await _setup(tmp_path)
    outcome = await pipeline.answer(1, 1, "прогноз выручка",
                                    metric_override="выручка", intent_override="forecast")
    assert "Прогноз" in outcome.text or "прогноз" in outcome.text
    if settings.send_charts:
        assert outcome.chart_png and outcome.chart_png[:8] == b"\x89PNG\r\n\x1a\n"
    await llm.close()
    await engine.dispose()


async def test_quick_action_uses_callback_user_not_bot(tmp_path):
    """Кнопки «Прогноз/Динамика/Состав»: пользователь берётся из нажатия.

    Регресс: `_answer_with_feedback` получал только сообщение, а у сообщения
    бота `from_user` — сам бот. Его id в базе не зарегистрирован, поэтому
    нажатие любой кнопки отвечало «пользователь не зарегистрирован».
    """
    engine, sessions, emb, llm, pipeline, app = await _setup(tmp_path)
    outcome = await pipeline.answer(1, 1, "Какая выручка за 2024 год?",
                                    intent_override="factual")
    mid = outcome.table_metric_id
    assert mid is not None

    # сообщение отправлено ботом (999), кнопку нажал человек (1)
    msg = FakeMessage(user_id=999)
    call = FakeCallback(msg, data=f"act:forecast:{mid}", user_id=1)
    await app._answer_with_feedback(
        call.message, query="прогноз выручка",
        metric_override="выручка", intent_override="forecast", event=call,
    )
    texts = "\n".join(x["text"] or "" for x in msg.sent)
    assert "пользователь не зарегистрирован" not in texts
    assert any("Прогноз" in (x["text"] or "") for x in msg.sent)
    # идёт видимая обратная связь, а не тишина
    assert any("Считаю" in (x["text"] or "") for x in msg.sent)
    await llm.close()
    await engine.dispose()


async def test_followup_uses_dialog_memory(tmp_path):
    """«а за 2023?» после вопроса о выручке — берёт показатель из памяти диалога."""
    engine, sessions, emb, llm, pipeline, app = await _setup(tmp_path)
    await pipeline.answer(1, 1, "Какая выручка за 2024 год?")
    outcome = await pipeline.answer(1, 1, "а за 2023?")
    text = outcome.text
    assert "100" in text.replace(" ", "") or "100" in text
    await llm.close()
    await engine.dispose()


async def test_send_answer_survives_bad_entities(tmp_path):
    """Битая HTML-сущность в ответе не теряет сообщение — уходит простым текстом."""
    engine, sessions, emb, llm, pipeline, app = await _setup(tmp_path)
    msg = FakeMessage(user_id=1)
    bad = QAOutcome(text="Ответ с <непарным тегом и 1<2")
    await app._send_answer(msg, bad, "тест")
    assert any("непарным" in (x["text"] or "") for x in msg.sent)
    assert any(x.get("parse_mode") is None for x in msg.sent)  # fallback сработал
    await llm.close()
    await engine.dispose()
