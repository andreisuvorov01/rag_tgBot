"""Регрессия UI бота: клавиатуры собираются валидными для aiogram
(поле называется inline_keyboard — инлайн-ошибка из прод-запуска 13.09.2026)."""
from types import SimpleNamespace

from aiogram.types import InlineKeyboardMarkup

from app.bot import BotApp, main_menu_inline, menu_keyboard
from app.config import settings


def _app() -> BotApp:
    return BotApp(settings, pipeline=None, session_factory=None)


def test_postprocess_keyboard_valid_markup():
    app = _app()
    report = SimpleNamespace(
        status="processed", document_id=5,
        warnings=["что-то подозрительное"],
        supersede_candidates=[(3, "old.xlsx")],
    )
    kb = app._postprocess_keyboard(report)
    assert isinstance(kb, InlineKeyboardMarkup)
    texts = [b.text for row in kb.inline_keyboard for b in row]
    callbacks = [b.callback_data for row in kb.inline_keyboard for b in row]
    assert any("Перечитать файл" in t for t in texts)
    assert any("личным" in t for t in texts)
    assert any("Принять" in t for t in texts)
    assert any("Переиздание" in t for t in texts)
    assert "reparse:5" in callbacks and "private:5" in callbacks
    assert "confirm:5" in callbacks and "supersede:5:3" in callbacks
    assert "summary:5" in callbacks  # обзор документа


def test_postprocess_keyboard_minimal():
    app = _app()
    report = SimpleNamespace(status="processed", document_id=6, warnings=[], supersede_candidates=[])
    kb = app._postprocess_keyboard(report)
    assert kb is not None
    callbacks = [b.callback_data for row in kb.inline_keyboard for b in row]
    assert callbacks == ["summary:6", "reparse:6", "private:6"]  # обзор + перечитать + личный


def test_clarify_keyboard_valid_markup():
    """Клавиатура уточнения показателя собирается с правильным полем."""
    from aiogram.types import InlineKeyboardButton

    candidates = ["выручка", "аренда спецтехники"]
    buttons = [
        [InlineKeyboardButton(text=name[:60], callback_data=f"metric_pick:{i}")]
        for i, name in enumerate(candidates)
    ]
    kb = InlineKeyboardMarkup(inline_keyboard=buttons)
    assert len(kb.inline_keyboard) == 2
    assert kb.inline_keyboard[1][0].callback_data == "metric_pick:1"


def test_main_menu_keyboards():
    """Нижнее меню и инлайн-главное меню валидны и содержат основные действия."""
    rk = menu_keyboard()
    reply_texts = [b.text for row in rk.keyboard for b in row]
    assert "📊 Показатели" in reply_texts and "📄 Документы" in reply_texts and "❓ Помощь" in reply_texts

    im = main_menu_inline()
    assert isinstance(im, InlineKeyboardMarkup)
    callbacks = [b.callback_data for row in im.inline_keyboard for b in row]
    assert callbacks == ["menu:metrics", "menu:documents", "menu:help"]
