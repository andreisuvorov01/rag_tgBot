"""Telegram-интерфейс: приём файлов, статусы обработки, вопросы, уточнения.

Доступ: по whitelist ID или коду регистрации (REG_CODE). Оригиналы документов
остаются на нашем сервере; наружу (в LLM API) уходят только промпты с данными.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
from datetime import date
from pathlib import Path

from aiogram import BaseMiddleware, Bot, Dispatcher, F, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramAPIError, TelegramBadRequest, TelegramRetryAfter
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import (
    BufferedInputFile,
    CallbackQuery,
    ErrorEvent,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    Message,
    ReplyKeyboardMarkup,
)

from .config import Settings, settings
from .formatting import KIND_TITLES, TELEGRAM_LIMIT, chunk_message, fmt_money
from .formatting import escape as escape_html
from .qa import AnswerPipeline, QAOutcome
from .security import authenticate, register_with_code
from .storage import (
    Document,
    Organization,
    confirm_facts,
    delete_user_documents,
    ledger_months,
    ledger_ops_for_month,
    metric_by_id,
    org_documents,
    org_metrics,
    series_for_metric,
    set_document_private,
    subscribe_digest,
    supersedes,
    unsubscribe_digest,
)

# Нижнее меню: текст кнопок перехватывается в on_question и исполняет команду
MENU_TEXTS = ("📊 Показатели", "📄 Документы", "❓ Помощь")
ACCESS_DENIED = "Доступ закрыт. Отправьте /start и введите код регистрации."


def menu_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="📊 Показатели"), KeyboardButton(text="📄 Документы")],
            [KeyboardButton(text="❓ Помощь")],
        ],
        resize_keyboard=True,
        input_field_placeholder="Задайте вопрос или пришлите файл отчёта",
    )


def main_menu_inline() -> InlineKeyboardMarkup:
    """Инлайн-главное меню (/menu): кнопки исполняют действия через callback."""
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="📊 Показатели", callback_data="menu:metrics"),
            InlineKeyboardButton(text="📄 Документы", callback_data="menu:documents"),
        ],
        [InlineKeyboardButton(text="❓ Помощь", callback_data="menu:help")],
    ])

log = logging.getLogger(__name__)


class RegState(StatesGroup):
    awaiting_code = State()


async def ack(call: CallbackQuery, text: str | None = None, show_alert: bool = False) -> None:
    """answerCallbackQuery должен уйти в течение ~15 с после нажатия — иначе
    Telegram отвечает «query is too old». Долгие действия подтверждают нажатие
    сразу, а поздний/повторный ответ не должен превращать успех в ошибку."""
    try:
        await call.answer(text, show_alert=show_alert)
    except TelegramBadRequest as e:
        if "query is too old" in str(e) or "query ID is invalid" in str(e):
            log.debug("callback answer после таймаута: %s", e)
            return
        raise


class CallbackAuth(BaseMiddleware):
    """Все инлайн-кнопки требуют зарегистрированного пользователя: кнопка
    в старом сообщении не должна работать после отзыва доступа."""

    def __init__(self, app: BotApp):
        self.app = app

    async def __call__(self, handler, event: CallbackQuery, data: dict):
        if await self.app._user_by_id(event.from_user.id) is None:
            await event.answer(ACCESS_DENIED, show_alert=True)
            return None
        return await handler(event, data)


class BotApp:
    def __init__(self, settings: Settings, pipeline: AnswerPipeline, session_factory):
        self.s = settings
        self.pipeline = pipeline
        self.sessions = session_factory
        self.router = Router()
        self.router.callback_query.outer_middleware(CallbackAuth(self))
        self._register_handlers()
        # Уточнения и исходный вопрос хранятся РАЗДЕЛЬНО: раньше ключ вида
        # «<chat_id>:q» лежал в том же словаре, что и списки вариантов, из-за
        # чего тип был недостоверен, а чтение через .get(..., "") давало список
        # вместо строки — запрос превращался в "[]" и уточнение отвечало не на
        # то, что спросили.
        self._pending_clarify: dict[int, list[str]] = {}
        self._pending_query: dict[int, str] = {}

    # ------------------------------------------------------------------
    def _register_handlers(self) -> None:
        r = self.router

        @r.message(Command("start"))
        async def cmd_start(message: Message, state: FSMContext):
            user = await self._user(message)
            if user:
                # нижнее меню + инлайн-главное меню выдаются и уже зарегистрированным
                await message.answer(self._welcome(registered=True), reply_markup=menu_keyboard())
                await message.answer("Главное меню:", reply_markup=main_menu_inline())
                return
            await state.set_state(RegState.awaiting_code)
            await message.answer(
                "Здравствуйте! Это ассистент по финансовым отчётам.\n"
                "Для доступа отправьте код регистрации (запросите у администратора)."
            )

        @r.message(Command("menu"))
        async def cmd_menu(message: Message):
            if not await self._auth_guard(message):
                return
            await message.answer("Главное меню:", reply_markup=main_menu_inline())

        @r.callback_query(F.data.startswith("menu:"))
        async def cb_menu(call: CallbackQuery):
            action = call.data.split(":")[1]
            # call.message.from_user — это бот; настоящий пользователь в call.from_user
            if action == "metrics":
                await self._send_metrics(call.message, user_id=call.from_user.id)
            elif action == "documents":
                await self._send_documents(call.message, user_id=call.from_user.id)
            else:
                await call.message.answer(self._help_text(), reply_markup=menu_keyboard())
            await ack(call)

        @r.message(RegState.awaiting_code, F.text)
        async def reg_code(message: Message, state: FSMContext):
            user = await self._run_in_session(
                lambda s: register_with_code(s, self.s, message.from_user.id,
                                             message.from_user.full_name, message.text.strip())
            )
            if user is None:
                await message.answer("Код не принят. Попробуйте ещё раз или /start.")
                return
            await state.clear()
            await message.answer(self._welcome(registered=True), reply_markup=menu_keyboard())

        @r.message(Command("help"))
        async def cmd_help(message: Message):
            if not await self._auth_guard(message):
                return
            outcome = await self.pipeline.answer(await self._org_id(message), message.from_user.id, "помощь")
            await message.answer(outcome.text, reply_markup=menu_keyboard())

        @r.message(Command("metrics"))
        async def cmd_metrics(message: Message):
            await self._send_metrics(message)

        @r.message(Command("documents"))
        async def cmd_documents(message: Message):
            await self._send_documents(message)

        @r.message(F.document)
        async def on_document(message: Message, bot: Bot):
            if not await self._auth_guard(message):
                return
            doc = message.document
            ext = Path(doc.file_name or "file").suffix.lower().lstrip(".")
            if ext not in self.s.allowed_exts:
                await message.answer(f"Формат .{ext} не поддерживается. Разрешены: {', '.join(sorted(self.s.allowed_exts))}.")
                return
            if (doc.file_size or 0) > self.s.max_file_mb * 1024 * 1024:
                await message.answer(f"Файл больше {self.s.max_file_mb} МБ. Разделите его или пришлите архивом поменьше.")
                return
            status = await message.answer(f"⏳ Принял «{escape_html(doc.file_name or 'file')}», начинаю обработку…")
            bar = ProgressBar(status, f"📄 {escape_html(doc.file_name or 'file')}")
            try:
                await bar("загрузка из Telegram", 0.02)
                file = await bot.get_file(doc.file_id)
                buf = await bot.download_file(file.file_path)
                content = buf.read() if hasattr(buf, "read") else Path(str(buf)).read_bytes()
                org_id = await self._org_id(message)
                from .ingest.pipeline import process_document

                async with self.sessions() as session:
                    report = await process_document(
                        session, self.pipeline.emb, self.s,
                        org_id=org_id,
                        user_id=message.from_user.id,
                        original_name=doc.file_name or "file",
                        content=content,
                        progress=bar,
                    )
                    await session.commit()
                kb = self._postprocess_keyboard(report)
                summary = report.summary()
                try:
                    await status.edit_text(summary, reply_markup=kb)
                except Exception:
                    # клавиатура не критична: сводка должна дойти даже при её ошибке
                    await status.edit_text(summary)
            except Exception as e:
                log.exception("Ошибка обработки файла")
                await status.edit_text(f"Не удалось обработать файл: {escape_html(str(e))}")

        @r.callback_query(F.data.startswith("supersede:"))
        async def cb_supersede(call: CallbackQuery):
            _, new_id, old_id = call.data.split(":")
            org_id = await self._org_id(call)
            async with self.sessions() as session:
                old = await supersedes(session, int(old_id), int(new_id), org_id=org_id)
                if old:
                    await session.commit()
            await call.message.edit_reply_markup(reply_markup=None)
            if old:
                await call.message.answer(
                    f"🔄 Документ «{escape_html(old.original_name)}» помечен как переизданный: "
                    f"его данные исключены из анализа, актуальна новая версия. "
                    f"История значений сохранена в базе."
                )
            await ack(call)

        @r.callback_query(F.data.startswith("private:"))
        async def cb_private(call: CallbackQuery):
            doc_id = int(call.data.split(":")[1])
            async with self.sessions() as session:
                doc = await set_document_private(session, doc_id, call.from_user.id, private=None)
                if doc:
                    await session.commit()
            await call.message.edit_reply_markup(reply_markup=None)
            if doc:
                state = "теперь виден только вам" if doc.is_private else "снова в общем доступе"
                await call.message.answer(
                    f"📄 Документ «{escape_html(doc.original_name)}» {state}."
                )
                await ack(call)
            else:
                await ack(call, "Поменять доступ может только тот, кто загрузил документ", show_alert=True)

        @r.message(Command("reset"))
        async def cmd_reset(message: Message):
            if not await self._auth_guard(message):
                return
            self.pipeline.dialog_memory.pop(message.from_user.id, None)
            await message.answer(
                "🧹 Память диалога очищена — следующий вопрос обрабатывается с нуля.",
                reply_markup=menu_keyboard(),
            )

        @r.message(Command("usage"))
        async def cmd_usage(message: Message):
            """Расход токенов внешнего LLM API: единственная платная часть."""
            if not await self._auth_guard(message):
                return
            from .usage import (
                balance_events,
                format_balance_state,
                format_budget_left,
                format_snapshot,
                usage_snapshot,
            )

            async with self.sessions() as s:
                snap = await usage_snapshot(s, self.s)
                events = await balance_events(s)
            text = format_snapshot(snap, per_question=snap["tasks"].get("compose", {}).get("calls"))
            text += format_budget_left(snap, self.s)
            text += format_balance_state(events)
            if not self.pipeline._balance_ok():
                text += ("\n\n🚫 <b>Сейчас API не опрашивается:</b> баланс исчерпан, "
                         "ответы собираются шаблоном. После пополнения счёта связь "
                         "восстановится автоматически.")
            await message.answer(text, reply_markup=menu_keyboard())

        @r.message(Command("report"))
        async def cmd_report(message: Message):
            """Отчёт по личным расходам из бюджета компании: колонка «Личные расходы».

            Файл собирается из базы при каждом запросе — оригиналы загруженных
            отчётов компании остаются нетронутыми.
            """
            if not await self._auth_guard(message):
                return
            from aiogram.types import BufferedInputFile

            from .expenses import ledger_rows
            from .export import expenses_report_filename, expenses_report_xlsx

            org_id = await self._org_id(message)
            async with self.sessions() as s:
                rows = await ledger_rows(s, self.s, org_id)
                org = await s.get(Organization, org_id)
            if not rows:
                await message.answer(
                    "В журнале пока нет записей, поэтому отчёт будет пустым.\n"
                    "Внесите расход сообщением — «расход: 1500 кофе» — и повторите /report.",
                    reply_markup=menu_keyboard(),
                )
                return
            today = date.today()
            data = await asyncio.to_thread(
                expenses_report_xlsx, rows, org_name=(org.name if org else ""), today=today
            )
            expenses = [r for r in rows if r["kind"] == "expense"]
            total = sum(r["amount"] for r in expenses)
            await message.answer_document(
                BufferedInputFile(data, filename=expenses_report_filename(today)),
                caption=(
                    f"📊 Личные расходы из бюджета компании\n"
                    f"Записей: {len(rows)} (расходов {len(expenses)})\n"
                    f"Сумма расходов: <b>{fmt_money(total)}</b>\n"
                    f"<i>Лист «Отчёт» — месяцы и колонка «Личные расходы»; "
                    f"лист «По категориям» — структура по месяцам.</i>"
                ),
            )

        @r.message(Command("ledger"))
        async def cmd_ledger(message: Message):
            """Журнал расходов: сводка за месяц + выгрузка Excel-файла."""
            if not await self._auth_guard(message):
                return
            from .expenses import month_report, sync_journal_file

            org_id = await self._org_id(message)
            async with self.sessions() as s:
                report_text = await month_report(s, self.s, org_id)
                n_rows = await sync_journal_file(s, self.s, org_id)
                await s.commit()
            if n_rows == 0:
                await message.answer(
                    "Журнал пуст. Записывайте расходы сообщениями: «расход: 1500 кофе», "
                    "«доход: 50000 зарплата» — всё попадёт в Excel и станет показателем.",
                    reply_markup=menu_keyboard(),
                )
                return
            try:
                from aiogram.types import FSInputFile

                caption = report_text or f"Журнал расходов: {n_rows} записей"
                await message.answer_document(
                    FSInputFile(self.s.expense_journal_path, filename=self.s.expense_journal_path.name),
                    caption=f"⬇️ {caption}\n<i>(файл лежит локально: {self.s.expense_journal_path})</i>",
                )
            except FileNotFoundError:
                await message.answer("Файл журнала ещё не создан — добавьте первую запись.")
            except Exception as e:
                log.exception("Excel-журнал")
                await message.answer(f"Сводка получена, но выгрузить файл не удалось: {escape_html(str(e))}\n{report_text}")

        @r.message(Command("undo"))
        async def cmd_undo(message: Message):
            """Отменяет последнюю запись журнала расходов."""
            if not await self._auth_guard(message):
                return
            from .expenses import delete_last_entry, sync_journal_file

            org_id = await self._org_id(message)
            async with self.sessions() as s:
                info = await delete_last_entry(s, org_id)
                if info is not None:
                    await sync_journal_file(s, self.s, org_id)
                await s.commit()
            if info is None:
                await message.answer("Журнал пуст — нечего отменять.")
                return
            desc = f" ({info['description']})" if info["description"] else ""
            await message.answer(
                f"↩️ Отменено: {fmt_money(info['amount'])} — {escape_html(info['category'])}"
                f"{desc} от {info['date']:%d.%m.%Y}"
            )

        @r.message(Command("clear"))
        async def cmd_clear(message: Message):
            if not await self._auth_guard(message):
                return
            org_id = await self._org_id(message)
            async with self.sessions() as s:
                mine = [d for d in await org_documents(s, org_id, message.from_user.id)
                        if d.uploaded_by == message.from_user.id]
            await message.answer(
                "🗑 Стереть историю чата и удалить ваши файлы?\n"
                f"Будут удалены сообщения диалога, {len(mine)} загруженных вами документов "
                "с их данными (факты, фрагменты, операции) и оригиналы с диска. Это необратимо.",
                reply_markup=InlineKeyboardMarkup(inline_keyboard=[[
                    InlineKeyboardButton(text="🗑 Да, удалить всё", callback_data="clear:yes"),
                    InlineKeyboardButton(text="Отмена", callback_data="clear:no"),
                ]]),
            )

        @r.callback_query(F.data.startswith("clear:"))
        async def cb_clear(call: CallbackQuery, bot: Bot):
            if call.data != "clear:yes":
                await call.message.edit_text("Отменено — ничего не удалено.")
                await ack(call)
                return
            user_id = call.from_user.id
            org_id = await self._org_id(call)
            async with self.sessions() as s:
                paths = await delete_user_documents(s, org_id, user_id)
                await s.commit()
            # файлы — только после commit: при откате БД оригиналы должны уцелеть
            for p in paths:
                Path(p).unlink(missing_ok=True)
            self.pipeline.dialog_memory.pop(user_id, None)
            self._pending_clarify.pop(call.message.chat.id, None)
            self._pending_query.pop(call.message.chat.id, None)
            await ack(call)
            await clear_chat_history(bot, call.message.chat.id, call.message.message_id)
            await call.message.answer(
                f"🗑 История чата стёрта, удалено документов: {len(paths)}, память диалога очищена.\n"
                "Telegram позволяет боту удалять только сообщения за последние 48 часов — "
                "более старые останутся.",
                reply_markup=menu_keyboard(),
            )

        @r.message(Command("subscribe"))
        async def cmd_subscribe(message: Message):
            if not await self._auth_guard(message):
                return
            org_id = await self._org_id(message)
            async with self.sessions() as s:
                created = await subscribe_digest(s, message.from_user.id, org_id)
                await s.commit()
            if created:
                await message.answer(
                    f"📅 Подписка на ежедневную сводку включена — буду присылать в {self.s.digest_hour}:00.",
                    reply_markup=menu_keyboard(),
                )
            else:
                await message.answer("Подписка уже активна. Отписаться: /unsubscribe", reply_markup=menu_keyboard())

        @r.message(Command("unsubscribe"))
        async def cmd_unsubscribe(message: Message):
            if not await self._auth_guard(message):
                return
            async with self.sessions() as s:
                removed = await unsubscribe_digest(s, message.from_user.id)
                await s.commit()
            await message.answer(
                "📅 Ежедневная сводка отключена." if removed else "Подписки и не было.",
                reply_markup=menu_keyboard(),
            )

        @r.callback_query(F.data.startswith("act:"))
        async def cb_act(call: CallbackQuery):
            # быстрые действия «Прогноз/Динамика/Состав» по показателю из ответа
            _, action, metric_id = call.data.split(":")
            async with self.sessions() as s:
                m = await metric_by_id(s, int(metric_id))
            if m is None:
                await ack(call, "Показатель не найден", show_alert=True)
                return
            await ack(call)  # долгий расчёт: сразу подтверждаем нажатие
            labels = {"forecast": "прогноз", "compare": "динамика", "breakdown": "состав"}
            await self._answer_with_feedback(
                call.message,
                query=f"{labels.get(action, action)} {m.name}",
                metric_override=m.name,
                intent_override=action,
                event=call,  # пользователь — в call.from_user, не в call.message
            )

        @r.callback_query(F.data.startswith("xlsx:"))
        async def cb_xlsx(call: CallbackQuery):
            metric_id = int(call.data.split(":")[1])
            try:
                from .export import metric_rows_to_xlsx

                org_id = await self._org_id(call)
                async with self.sessions() as session:
                    m = await metric_by_id(session, metric_id)
                    rows = await series_for_metric(session, org_id, metric_id, user_id=call.from_user.id)
                if m is None or m.org_id != org_id or not rows:
                    await ack(call, "Данных нет", show_alert=True)
                    return
                data = metric_rows_to_xlsx(m.name, rows)
                await call.message.answer_document(
                    BufferedInputFile(data, filename=f"{m.code}.xlsx"),
                    caption=f"⬇️ {escape_html(m.name)}: история значений с источниками",
                )
                await ack(call)
            except Exception as e:
                log.exception("Excel export")
                await ack(call, f"Ошибка выгрузки: {e}", show_alert=True)

        @r.callback_query(F.data.startswith("reparse:"))
        async def cb_reparse(call: CallbackQuery):
            doc_id = int(call.data.split(":")[1])
            user_id = call.from_user.id
            org_id = await self._org_id(call)
            # перечитывание удаляет и заново извлекает факты документа, поэтому
            # доступно только загрузившему его пользователю: иначе коллега по
            # организации мог бы переписать чужой общий документ
            async with self.sessions() as s:
                doc = await s.get(Document, doc_id)
                owner_ok = bool(
                    doc and doc.org_id == org_id
                    and (doc.uploaded_by == user_id or doc.uploaded_by == 0)
                )
            if not owner_ok:
                await ack(call, "Перечитать может только автор документа", show_alert=True)
                return
            await ack(call)  # разбор файла дольше 15 с — подтверждаем нажатие сразу
            note = await call.message.answer("⏳ Перечитываю файл обновлённым парсером…")
            try:
                from .ingest.pipeline import reparse_document

                async with self.sessions() as session:
                    report = await reparse_document(
                        session, self.pipeline.emb, self.s,
                        document_id=doc_id, user_id=user_id,
                        org_id=org_id,
                        progress=ProgressBar(note, "🔄 Перечитываю файл"),
                    )
                    await session.commit()
                try:
                    await note.edit_text(report.summary(), reply_markup=self._postprocess_keyboard(report))
                except Exception:
                    await note.edit_text(report.summary())
            except Exception as e:
                log.exception("Ошибка перечитывания")
                await note.edit_text(f"Ошибка перечитывания: {escape_html(str(e))}")

        @r.callback_query(F.data.startswith("opsm:"))
        async def cb_ops_months(call: CallbackQuery):
            # выбор месяца с операциями
            metric_id = int(call.data.split(":")[1])
            org_id = await self._org_id(call)
            async with self.sessions() as s:
                m = await metric_by_id(s, metric_id)
                if m is not None and m.org_id != org_id:
                    m = None
                months = await ledger_months(s, org_id, m.name.casefold()) if m else []
            if not m or not months:
                await ack(call, "Операций нет", show_alert=True)
                return
            rows = [
                [InlineKeyboardButton(
                    text=f"{mn['label']} — {fmt_money(mn['total'], None)} ({mn['count']} оп.)",
                    callback_data=f"ops:{metric_id}:{mn['label']}",
                )]
                for mn in months[-8:][::-1]
            ]
            await call.message.answer(
                f"📊 Операции по «{escape_html(m.name)}» — выберите месяц:",
                reply_markup=InlineKeyboardMarkup(inline_keyboard=rows),
            )
            await ack(call)

        @r.callback_query(F.data.startswith("ops:"))
        async def cb_ops_view(call: CallbackQuery):
            _, metric_id, label = call.data.split(":", 2)
            org_id = await self._org_id(call)
            async with self.sessions() as s:
                m = await metric_by_id(s, int(metric_id))
                if m is not None and m.org_id != org_id:
                    m = None
                ops = await ledger_ops_for_month(s, org_id, m.name.casefold(), label) if m else []
            if not m or not ops:
                await ack(call, "Операций за этот месяц нет", show_alert=True)
                return
            total = sum(o.value for o in ops)
            cats: dict[str, float] = {}
            for o in ops:
                cats[o.category] = cats.get(o.category, 0.0) + o.value
            lines = [f"💳 <b>Операции за {escape_html(label)}</b> — {escape_html(m.name)}",
                     f"Всего: <b>{fmt_money(total)}</b> по {len(ops)} операциям", ""]
            for cat, v in sorted(cats.items(), key=lambda kv: -abs(kv[1])):
                lines.append(f"• {escape_html(cat)}: {fmt_money(v)}")
            lines.append("")
            for o in ops[:25]:
                desc = (o.description or "—")[:40]
                lines.append(f"{o.date_actual:%d.%m} {escape_html(desc)} — {fmt_money(o.value)} [{o.category}]")
            if len(ops) > 25:
                lines.append(f"…и ещё {len(ops) - 25} операций (полный список — в Excel по показателю)")
            for part in chunk_message("\n".join(lines)):
                await call.message.answer(part)
            await ack(call)

        @r.callback_query(F.data.startswith("summary:"))
        async def cb_summary(call: CallbackQuery):
            doc_id = int(call.data.split(":")[1])
            await ack(call)
            note = await call.message.answer("⏳ Готовлю обзор документа…")
            try:
                from .summarize import document_summary

                async with self.sessions() as s:
                    text = await document_summary(self.pipeline, s, (await self._org_id(call)),
                                                  doc_id, call.from_user.id)
                for part in chunk_message(text):
                    await call.message.answer(part)
                await note.delete()
            except Exception as e:
                log.exception("Ошибка обзора документа")
                await note.edit_text(f"Ошибка обзора: {escape_html(str(e))}")

        @r.callback_query(F.data.startswith("confirm:"))
        async def cb_confirm(call: CallbackQuery):
            doc_id = int(call.data.split(":")[1])
            async with self.sessions() as session:
                n = await confirm_facts(session, doc_id, org_id=await self._org_id(call))
                await session.commit()
            await call.message.edit_reply_markup(reply_markup=None)
            await ack(call, f"Подтверждено показателей: {n}")

        @r.callback_query(F.data.startswith("metric_pick:"))
        async def cb_metric_pick(call: CallbackQuery):
            idx = int(call.data.split(":")[1])
            candidates = self._pending_clarify.get(call.message.chat.id, [])
            query = self._pending_query.get(call.message.chat.id, "")
            await ack(call)
            if 0 <= idx < len(candidates):
                await call.message.edit_reply_markup(reply_markup=None)
                outcome = await self.pipeline.answer(
                    await self._org_id(call), call.from_user.id, query,
                    metric_override=candidates[idx],
                )
                await self._send_answer(call.message, outcome, query)

        @r.message(F.text & ~F.text.startswith("/"))
        async def on_question(message: Message, bot: Bot):
            if not await self._auth_guard(message):
                return
            text = message.text.strip()
            if text in MENU_TEXTS:  # кнопки нижнего меню
                if text == "📊 Показатели":
                    await cmd_metrics(message)
                elif text == "📄 Документы":
                    await cmd_documents(message)
                else:
                    await cmd_help(message)
                return
            # «расход: 1500 кофе» / «доход: 50000 зарплата» — мгновенная запись
            # в Excel-журнал и показатель, без LLM
            from .expenses import add_entry, parse_entry_message, sync_journal_file

            entry = parse_entry_message(text)
            if entry is not None:
                org_id = await self._org_id(message)
                async with self.sessions() as s:
                    result = await add_entry(
                        s, self.s, org_id=org_id, user_id=message.from_user.id,
                        entry=entry, emb=self.pipeline.emb,
                    )
                    await sync_journal_file(s, self.s, org_id)
                    await s.commit()
                await message.answer(result["text"], reply_markup=menu_keyboard())
                return
            await bot.send_chat_action(message.chat.id, "typing")
            await self._answer_with_feedback(message, query=message.text.strip())

    # ------------------------------------------------------------------
    async def _send_metrics(self, message: Message, user_id: int | None = None) -> None:
        user = await self._user_by_id(user_id or message.from_user.id)
        if user is None:
            await message.answer(ACCESS_DENIED)
            return
        org_id = user.org_id
        async with self.sessions() as session:
            metrics = await org_metrics(session, org_id)
        if not metrics:
            await message.answer("Словарь показателей пуст — загрузите отчёты.", reply_markup=menu_keyboard())
            return
        shown = [m for m in metrics if m.kind != "identifier"] or metrics
        groups: dict[str, list] = {}
        for m in shown:
            groups.setdefault(m.kind, []).append(m)
        order = ["revenue", "expense", "asset", "liability", "other"]
        lines = []
        for kind in order:
            if kind not in groups:
                continue
            lines.append(f"\n{KIND_TITLES.get(kind, '• Прочее')}")
            lines.extend(f"  • <code>{escape_html(m.name)}</code>" for m in groups[kind][:12])
            if len(groups[kind]) > 12:
                lines.append(f"  …и ещё {len(groups[kind]) - 12}")
        body = (
            f"<b>Словарь показателей</b> ({len(shown)} из {len(metrics)}):\n" + "\n".join(lines)
            + "\n\n💡 Задайте вопрос по любому показателю."
        )
        # длинный словарь может превысить лимит Telegram: без разбиения
        # кнопка «Показатели» молча ничего не отправляла
        for i, part in enumerate(chunk_message(body)):
            await message.answer(part, reply_markup=menu_keyboard() if i == 0 else None)

    async def _send_documents(self, message: Message, user_id: int | None = None) -> None:
        user = await self._user_by_id(user_id or message.from_user.id)
        if user is None:
            await message.answer(ACCESS_DENIED)
            return
        async with self.sessions() as session:
            docs = await org_documents(session, user.org_id, user.telegram_id)
        if not docs:
            await message.answer("Документы ещё не загружены — просто пришлите файл.", reply_markup=menu_keyboard())
            return
        lines = [
            f"• <code>{escape_html(d.original_name)}</code> — {d.status}"
            + (" (переиздан, из анализа исключён)" if d.superseded_by_id else "")
            + (" 🔒 личный" if d.is_private else "")
            + f", {d.meta.get('facts', 0)} показателей, {d.meta.get('chunks', 0)} фрагментов"
            + (f"; периоды: {', '.join((d.meta.get('periods') or [])[:6])}" if d.meta.get("periods") else "")
            for d in docs[:20]
        ]
        rows = []
        for d in docs[:3]:
            if d.status != "processed":
                continue
            buttons = [
                InlineKeyboardButton(text=f"📖 Обзор «{d.original_name[:30]}»", callback_data=f"summary:{d.id}"),
            ]
            # «Перечитать» удаляет и заново извлекает факты — только для своих
            # документов, иначе кнопка предлагала бы перезапись чужого отчёта
            if d.uploaded_by in (user.telegram_id, 0):
                buttons.append(InlineKeyboardButton(text="🔄 Перечитать", callback_data=f"reparse:{d.id}"))
            rows.append(buttons)
        kb = InlineKeyboardMarkup(inline_keyboard=rows) if rows else None
        body = (
            "<b>Загруженные документы:</b>\n" + "\n".join(lines)
            + ("\n\n💡 «Перечитать» — заново разобрать файл обновлённым парсером." if rows else "")
        )
        for i, part in enumerate(chunk_message(body)):
            await message.answer(part, reply_markup=kb if i == 0 else None)

    def _postprocess_keyboard(self, report) -> InlineKeyboardMarkup | None:
        rows = []
        if report.status == "processed":
            rows.append([InlineKeyboardButton(text="📖 Рассказать об отчёте",
                                              callback_data=f"summary:{report.document_id}")])
            rows.append([InlineKeyboardButton(text="🔄 Перечитать файл",
                                              callback_data=f"reparse:{report.document_id}")])
            rows.append([InlineKeyboardButton(text="🔒 Сделать личным (виден только мне)",
                                              callback_data=f"private:{report.document_id}")])
        if report.status == "processed" and report.warnings:
            rows.append([InlineKeyboardButton(text="✅ Принять как есть", callback_data=f"confirm:{report.document_id}")])
        for old_id, old_name in report.supersede_candidates:
            rows.append([
                InlineKeyboardButton(
                    text=f"🔄 Переиздание вместо «{old_name[:32]}»",
                    callback_data=f"supersede:{report.document_id}:{old_id}",
                )
            ])
        return InlineKeyboardMarkup(inline_keyboard=rows) if rows else None

    async def _send_answer(self, message: Message, outcome: QAOutcome, query: str = "") -> None:
        # «Закончился баланс на ключе API» — показываем ПЕРЕД ответом, иначе
        # пользователь не поймёт, почему текст стал шаблонным
        if outcome.balance_notice:
            with contextlib.suppress(Exception):
                await message.answer(outcome.balance_notice)
        for part in chunk_message(outcome.text):
            try:
                await message.answer(part)
            except TelegramBadRequest as e:
                if "can't parse entities" in str(e).casefold():
                    # LLM вставил битую HTML-сущность — отправляем как простой текст
                    await message.answer(part, parse_mode=None)
                elif "message is too long" in str(e).casefold():
                    # страховка: часть всё же превысила лимит — режем её жёстко
                    for sub in chunk_message(part, limit=TELEGRAM_LIMIT // 2):
                        await message.answer(sub, parse_mode=None)
                else:
                    raise
        # график прогноза картинкой
        if outcome.chart_png and self.s.send_charts:
            from aiogram.types import BufferedInputFile

            await message.answer_photo(
                BufferedInputFile(outcome.chart_png, filename="forecast.png"),
                caption="📊 История и прогноз",
            )
        # быстрые действия по показателю: прогноз, динамика, состав, Excel
        if outcome.table_metric_id:
            mid = outcome.table_metric_id
            rows = [[
                InlineKeyboardButton(text="🔮 Прогноз", callback_data=f"act:forecast:{mid}"),
                InlineKeyboardButton(text="📈 Динамика", callback_data=f"act:compare:{mid}"),
                InlineKeyboardButton(text="🧮 Состав", callback_data=f"act:breakdown:{mid}"),
            ]]
            rows.append([InlineKeyboardButton(text="⬇️ Выгрузить в Excel", callback_data=f"xlsx:{mid}")])
            await message.answer("Дальше по этому показателю:", reply_markup=InlineKeyboardMarkup(inline_keyboard=rows))
        # выписка: просмотр операций по месяцам
        if outcome.ops_months and outcome.table_metric_id:
            mid = outcome.table_metric_id
            await message.answer(
                "💳 По этому показателю есть выписка — можно посмотреть операции:",
                reply_markup=InlineKeyboardMarkup(inline_keyboard=[
                    [InlineKeyboardButton(text="📊 Операции за месяц", callback_data=f"opsm:{mid}")]
                ]),
            )
        if outcome.clarify:
            self._pending_clarify[message.chat.id] = outcome.clarify
            self._pending_query[message.chat.id] = query
            buttons = [
                [InlineKeyboardButton(text=name[:60], callback_data=f"metric_pick:{i}")]
                for i, name in enumerate(outcome.clarify)
            ]
            await message.answer("Уточните показатель:", reply_markup=InlineKeyboardMarkup(inline_keyboard=buttons))

    async def _answer_with_feedback(
        self,
        message: Message,
        *,
        query: str,
        metric_override: str | None = None,
        intent_override: str | None = None,
        event: Message | CallbackQuery | None = None,
    ) -> None:
        """Задать вопрос конвейеру, показав пользователю, что работа идёт.

        Общий путь для текстового вопроса и кнопок «Прогноз/Динамика/Состав».
        Раньше кнопки не показывали ничего до самого ответа: модель на CPU
        думает десятки секунд, и выглядело так, будто ничего не происходит.
        Через 8 секунд добавляем прошедшее время, чтобы ожидание было видимым.

        `event` — источник нажатия/сообщения для определения пользователя.
        Для кнопки это обязательно: у сообщения бота `from_user` — сам бот,
        поэтому `_org_id(message)` не находил пользователя и падал с
        «пользователь не зарегистрирован».
        """
        source = event or message
        started = asyncio.get_running_loop().time()
        thinking = await message.answer("⏳ Считаю — это может занять до минуты…")
        ticker = asyncio.create_task(self._elapsed_ticker(thinking, started))
        try:
            outcome = await self.pipeline.answer(
                await self._org_id(source), source.from_user.id, query,
                metric_override=metric_override, intent_override=intent_override,
            )
            await self._send_answer(message, outcome, query=query)
        except Exception as e:
            log.exception("Ошибка ответа")
            with contextlib.suppress(Exception):
                await thinking.edit_text(f"Произошла ошибка: {escape_html(str(e))}")
            return
        finally:
            ticker.cancel()
        with contextlib.suppress(Exception):
            await thinking.delete()

    @staticmethod
    async def _elapsed_ticker(thinking: Message, started: float) -> None:
        """Раз в 5 секунд показывать, сколько уже идёт обработка."""
        try:
            while True:
                await asyncio.sleep(5)
                elapsed = int(asyncio.get_running_loop().time() - started)
                if elapsed < 8:
                    continue
                with contextlib.suppress(Exception):
                    await thinking.edit_text(f"⏳ Считаю… {elapsed} с. Обычно 10–60 секунд на CPU.")
        except asyncio.CancelledError:
            pass

    async def _auth_guard(self, message: Message) -> bool:
        user = await self._user(message)
        if user is None:
            await message.answer(ACCESS_DENIED)
            return False
        return True

    def _help_text(self) -> str:
        """Текст помощи для инлайн-кнопки «❓ Помощь».

        Раньше здесь вызывался метод AnswerPipeline, которого у BotApp нет:
        нажатие кнопки падало с AttributeError, и пользователь не получал
        ничего.
        """
        return (
            "Я финансовый ассистент. Умею:\n"
            "• отвечать на вопросы по загруженным отчётам — «какая выручка за 2024?»;\n"
            "• сравнивать периоды — «на сколько выросла аренда с 2023 по 2025?»;\n"
            "• ранжировать — «какие позиции выросли сильнее всего?»;\n"
            "• искать по тексту документов — «что говорится о рисках?»;\n"
            "• прогнозировать — «прогноз по позиции X на 2026»;\n"
            "• вести журнал расходов — «расход: 1500 кофе», «доход: 50000 зарплата».\n\n"
            "Просто пришлите файл отчёта (xlsx, csv, pdf, docx) и задавайте вопросы."
        )

    async def _org_id(self, event: Message | CallbackQuery) -> int:
        """event.from_user — автор сообщения или нажатия кнопки (у CallbackQuery
        call.message.from_user — это сам бот, поэтому передавать нужно call)."""
        user = await self._user(event)
        if user is None:
            raise PermissionError("пользователь не зарегистрирован")
        return user.org_id

    async def _user(self, event: Message | CallbackQuery):
        return await self._user_by_id(event.from_user.id)

    async def _user_by_id(self, user_id: int):
        allowed = self.s.allowed_ids
        if allowed and user_id not in allowed:
            return None
        return await self._run_in_session(lambda s: authenticate(s, self.s, user_id))

    async def _run_in_session(self, fn):
        async with self.sessions() as session:
            return await fn(session)

    def _welcome(self, registered: bool) -> str:
        return (
            "✅ Доступ открыт!\n\n"
            "Пришлите файл финансового отчёта (xlsx, csv, pdf, docx, html, ods) — я извлеку данные.\n"
            "Затем спрашивайте: «какая выручка за 2024?», «прогноз по аренде на 2026?», "
            "«из чего состоит итог расходов?», «факт 2026 vs план?», "
            "«какие позиции выросли сильнее всего?»\n\n"
            "💸 Расходы на лету: «расход: 1500 кофе», «доход: 50000 зарплата» — "
            "запись падает в Excel-журнал и в показатель «личные расходы» "
            "(с датой: «расход: 300 обед (03.01)»).\n\n"
            "Команды: /menu — главное меню, /metrics — показатели, /documents — документы, /help — примеры,\n"
            "/ledger — журнал расходов в Excel, /undo — отменить последнюю запись,\n"
            "/report — отчёт «Личные расходы из бюджета компании»,\n"
            "/usage — расход токенов LLM API,\n"
            "/reset — забыть контекст диалога, /clear — стереть историю чата."
        )


class ProgressBar:
    """Прогресс обработки файла в одном редактируемом сообщении:
    «▰▰▰▱▱▱▱▱▱▱ 30% · векторизация фрагментов 8/40». Правки не чаще раза
    в секунду (лимиты Telegram на editMessageText), последний этап — всегда.
    Ошибки правки (в т.ч. flood-control) не должны прерывать разбор файла:
    прогресс — вспомогательный элемент, а не часть конвейера."""

    def __init__(self, message: Message, title: str, min_interval: float = 1.0):
        self.message = message
        self.title = title
        self.min_interval = min_interval
        self._last = 0.0

    async def __call__(self, stage: str, fraction: float) -> None:
        now = asyncio.get_running_loop().time()
        if now - self._last < self.min_interval and fraction < 0.97:
            return
        self._last = now
        filled = round(max(0.0, min(fraction, 1.0)) * 10)
        text = f"{self.title}\n{'▰' * filled}{'▱' * (10 - filled)} {int(fraction * 100)}% · {escape_html(stage)}"
        try:
            await self.message.edit_text(text)
        except TelegramBadRequest:
            pass  # «message is not modified» и подобное — прогресс не критичен
        except TelegramRetryAfter as e:
            # flood-control: ждём указанное время и не считаем это сбоем разбора
            self._last = asyncio.get_running_loop().time() + float(e.retry_after)
            log.info("Flood-control на editMessageText: пауза %s с", e.retry_after)
        except TelegramAPIError as e:
            log.debug("Прогресс не обновлён: %s", e)


async def clear_chat_history(bot: Bot, chat_id: int, last_message_id: int, max_batches: int = 20) -> None:
    """Удаляет сообщения диалога (и бота, и пользователя) от последнего к
    ранним. В личном чате message_id идут подряд, поэтому идём пачками по 100
    вниз (несуществующие id Telegram пропускает молча). Сообщения старше
    48 часов API удалять не даёт: если пачка отклонена и поштучно не удалилось
    ничего — граница достигнута, обход завершается."""
    top = last_message_id
    for _ in range(max_batches):
        ids = list(range(top, max(top - 100, 0), -1))
        if not ids:
            return
        try:
            await bot.delete_messages(chat_id, ids)
        except TelegramBadRequest:
            ok = 0
            for mid in ids:
                try:
                    ok += bool(await bot.delete_message(chat_id, mid))
                except TelegramBadRequest:
                    continue
            if ok == 0:
                return
        top -= 100


class TelegramSession(AiohttpSession):
    """AiohttpSession с опциональным IPv4-форсированием для ПРЯМОГО соединения
    (при сломанном IPv6-маршруте соединение висит). Когда работает прокси,
    DNS/resolution делает прокси-коннектор — family не трогаем."""

    def __init__(self, *, proxy: str | None = None, **kwargs):
        force = bool(settings.telegram_force_ipv4) and proxy is None
        super().__init__(proxy=proxy, **kwargs)
        if force:
            import socket

            self._connector_init.setdefault("family", socket.AF_INET)


async def _print_connection_diagnostics(settings: Settings) -> None:
    """Подробный разбор: какой путь к api.telegram.org работает и что вписать в .env."""
    import httpx

    from .nettools import detect_windows_proxy, env_proxy_vars

    print("\nДиагностика подключения к api.telegram.org:")
    env = env_proxy_vars()
    if env:
        for k, v in env.items():
            print(f"  переменная {k}={v} — будет использована ботом (trust_env)")
    else:
        print("  переменные окружения прокси (HTTP(S)_PROXY) не заданы")

    reg = detect_windows_proxy()
    print(f"  системный прокси Windows: {reg or 'не задан'}")

    async def probe(client: httpx.AsyncClient, label: str) -> bool:
        try:
            r = await client.get("https://api.telegram.org", timeout=8)
            print(f"  [работает] {label} (HTTP {r.status_code})")
            return True
        except Exception as e:
            print(f"  [нет]      {label} ({type(e).__name__})")
            return False

    async with httpx.AsyncClient(trust_env=False) as c:
        direct_ok = await probe(c, "напрямую, без прокси")
    if env:
        async with httpx.AsyncClient(trust_env=True) as c:
            await probe(c, "через переменные окружения прокси")
    if reg:
        async with httpx.AsyncClient(proxy=reg) as c:
            ok = await probe(c, f"через системный прокси {reg}")
            if ok:
                print(f"\n  >>> РЕШЕНИЕ: добавьте в .env строку  TELEGRAM_PROXY={reg}")

    if direct_ok:
        print(
            "  Прямое подключение работает, но бот не смог соединиться. Возможные причины:\n"
            "  - антивирус/файрвол блокирует сетевой доступ конкретно python.exe\n"
            "    (добавьте venv\\Scripts\\python.exe в исключения);\n"
            "  - короткий таймаут на медленной сети — увеличьте в .env: TELEGRAM_PROBE_TIMEOUT=60\n"
        )


async def run_bot(settings: Settings, pipeline: AnswerPipeline, session_factory) -> None:
    if not settings.bot_token:
        print(
            "\nНе задан BOT_TOKEN.\n"
            "1. В Telegram откройте @BotFather -> /newbot -> получите токен.\n"
            "2. Впишите его в файл .env (BOT_TOKEN=...).\n"
            "3. Запустите снова: start.bat или python -m app.main\n"
        )
        return
    try:
        probe_timeout = getattr(settings, "telegram_probe_timeout", 45)
        # HTML по умолчанию: <b>/<i>/<pre> в ответах рендерятся, а не печатаются текстом
        default = DefaultBotProperties(parse_mode=ParseMode.HTML)
        dp = Dispatcher(storage=MemoryStorage())
        app = BotApp(settings, pipeline, session_factory)
        dp.include_router(app.router)

        @dp.errors()
        async def on_error(event: ErrorEvent):
            """Глобальный обработчик: без него любое необработанное исключение в
            хендлере оставляло пользователя без ответа (и без единого признака,
            что что-то сломалось, — только трейсбек в логе)."""
            log.exception("Необработанная ошибка при обработке апдейта: %s", event.exception)
            update = event.update
            target = update.message or (update.callback_query.message if update.callback_query else None)
            if target is not None:
                with contextlib.suppress(Exception):
                    await target.answer("⚠️ Внутренняя ошибка. Попробуйте ещё раз или /start.")
            return True

        # Перебираем варианты подключения и берём первый живой.
        # Удачный способ запоминается в БД и на следующем старте пробуется первым.
        from .nettools import detect_windows_proxy
        from .storage import get_meta, set_meta

        last_ok = None
        try:
            async with session_factory() as s:
                last_ok = await get_meta(s, "telegram_last_conn")
        except Exception:
            pass

        candidates: list[tuple[str, str | None]] = []
        if settings.telegram_proxy:
            candidates.append((f"прокси из .env ({settings.telegram_proxy})", settings.telegram_proxy))
        reg = detect_windows_proxy()
        if reg and reg != settings.telegram_proxy:
            candidates.append((f"системный прокси ({reg})", reg))
        candidates.append(("напрямую", None))
        if last_ok:
            candidates.sort(key=lambda c: c[0] != last_ok)

        bot = None
        for label, proxy in candidates:
            bot_try = Bot(token=settings.bot_token, session=TelegramSession(proxy=proxy), default=default)
            try:
                await asyncio.wait_for(bot_try.delete_webhook(drop_pending_updates=True), timeout=probe_timeout)
                bot = bot_try
                print(f"Подключение к Telegram: {label}")
                log.info("Подключение к Telegram: %s", label)
                try:
                    async with session_factory() as s:
                        await set_meta(s, "telegram_last_conn", label)
                        await s.commit()
                except Exception:
                    pass
                break
            except Exception as e:
                print(f"  вариант «{label}» не работает: {type(e).__name__}")
                await bot_try.session.close()

        if bot is None:
            print("\nНи один вариант подключения не сработал. Проверьте интернет/прокси-клиент\n"
                  "(он должен быть запущен) и запустите снова.\n")
            return
        log.info("Бот запущен (long polling). Останов: Ctrl+C")
        digest_task = asyncio.create_task(digest_loop(bot, session_factory, settings))
        try:
            await dp.start_polling(bot)
        finally:
            digest_task.cancel()
    except Exception as e:
        name = type(e).__name__
        if "TokenValidation" in name or ("token" in str(e).casefold() and "invalid" in str(e).casefold()):
            print("\nТокен бота невалиден. Проверьте BOT_TOKEN в .env (токен из @BotFather целиком).\n")
        elif "ClientConnector" in name or "network" in str(e).casefold() or "unreachable" in str(e).casefold():
            print(
                "\nНет доступа к api.telegram.org.\n"
                "Если сеть блокирует Telegram — укажите прокси в .env:\n"
                "  TELEGRAM_PROXY=http://127.0.0.1:10809   (или socks5://...)\n"
                f"Детали: {name}: {e}\n"
            )
        else:
            raise


async def digest_loop(bot: Bot, session_factory, settings: Settings) -> None:
    """Ежедневная сводка подписчикам в заданный час (DIGEST_HOUR)."""
    from datetime import datetime, timedelta

    from .digest import build_digest
    from .storage import list_digest_subs

    while True:
        try:
            now = datetime.now()
            target = now.replace(hour=settings.digest_hour, minute=0, second=0, microsecond=0)
            if target <= now:
                target += timedelta(days=1)
            await asyncio.sleep(max((target - now).total_seconds(), 1))
            async with session_factory() as s:
                subs = await list_digest_subs(s)
            for tg_id, org_id in subs:
                try:
                    async with session_factory() as s:
                        text = await build_digest(s, org_id, tg_id)
                    await bot.send_message(tg_id, text)
                except Exception as e:
                    log.warning("Дайджест для %s не отправлен: %s", tg_id, e)
        except asyncio.CancelledError:
            return
        except Exception as e:
            log.warning("Digest loop: %s", e)
            await asyncio.sleep(60)

