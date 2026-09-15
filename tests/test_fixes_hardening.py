"""Регрессионные тесты исправлений: единицы в ячейках/строках, обрезка длинных
ответов, маскирование промптов и свежесть сводки."""
import asyncio
from datetime import date
from decimal import Decimal

from app.agents import claim_misattributions, verify_answer
from app.config import settings
from app.formatting import TELEGRAM_LIMIT, chunk_message, render_answer
from app.ingest.load_excel import _cell_value, _is_texty, grid_to_parsed
from app.ingest.normalize import parse_number
from app.ingest.pipeline import _validate_totals
from app.llm import AnonymizingLLM, BaseLLM, MockLLM

# --------------------------------------------------- единицы в Excel

def test_percent_row_does_not_inherit_sheet_scale():
    """«Доля, %» в листе «тыс. руб.» остаётся процентом, а не ×1000."""
    facts, _, _, _ = grid_to_parsed(
        [["Показатель", "2024"], ["Выручка", 100], ["Доля, %", 12.5]],
        "Выручка, тыс. руб.",
    )
    by_name = {f.metric_name: f for f in facts}
    assert by_name["доля"].value == 12.5
    assert by_name["доля"].unit == "%"
    # обычная строка масштаб листа по-прежнему получает
    assert by_name["выручка"].value == 100_000


def test_percent_cell_keeps_percent_unit():
    """Ячейка «12,7%» — проценты, а не 12 700 руб."""
    facts, _, _, _ = grid_to_parsed(
        [["Показатель", "2024"], ["Рентабельность", "12,7%"]],
        "Выручка, тыс. руб.",
    )
    assert facts and facts[0].value == Decimal("12.7") and facts[0].unit == "%"


def test_cell_with_unit_is_a_value_not_a_metric_name():
    """«12,7 млн» — значение с единицей, поэтому колонка показателей не
    определяется по ней, и данные не теряются."""
    assert _is_texty("Выручка") is True
    assert _is_texty("12,7 млн") is False
    assert _is_texty("12,7%") is False
    facts, _, _, _ = grid_to_parsed(
        [["Показатель", "2024"], ["Выручка", "12,7 млн"]], "Лист1"
    )
    assert facts and facts[0].metric_name == "выручка"
    assert facts[0].value == 12_700_000


def test_cell_value_precedence():
    """Множители ячейки и листа не перемножаются; своя единица важнее листа."""
    assert _cell_value("12,7 млн", 1000)[0] == Decimal("12700000")   # млн, не ×1000 сверху
    assert _cell_value("41200", 1000)[0] == Decimal("41200000")      # голое число — масштаб листа
    assert _cell_value("12,7%", 1000)[0] == Decimal("12.7")
    assert _cell_value("12,7 руб.", 1000)[0] == Decimal("12.7")      # своя валюта без множителя


# --------------------------------------------------- сверка итогов

class _F:
    def __init__(self, name, value, variant="fact", section=None, is_total=False, parent_hint=None):

        self.metric_name = name
        self.value = value
        self.variant = variant
        self.section = section
        self.is_total = is_total
        self.parent_hint = parent_hint
        self.sheet = "S"
        self.currency = "RUB"

        class _P:
            end = date(2024, 12, 31)
            label = "2024"

        self.period = _P()


def test_totals_reconciliation_does_not_mix_plan_and_fact():
    """План и факт одного периода — разные числа, их нельзя суммировать вместе."""
    facts = [
        _F("итого расходов", 200, variant="plan", is_total=True),
        _F("аренда", 100, variant="plan"),
        _F("аренда", 110, variant="fact"),
        _F("итого расходов", 220, variant="fact", is_total=True),
    ]
    assert _validate_totals(facts) == []


# --------------------------------------------------- длинные сообщения

def test_chunk_message_splits_single_long_paragraph():
    """Один длинный абзац без пустых строк тоже обязан быть разрезан."""
    text = "а" * (TELEGRAM_LIMIT * 2 + 100)
    parts = chunk_message(text)
    assert len(parts) >= 3
    assert all(len(p) <= TELEGRAM_LIMIT for p in parts)
    assert "".join(parts).replace("\n\n", "") == text.replace("\n\n", "")


def test_chunk_message_keeps_short_text_intact():
    assert chunk_message("короткий ответ") == ["короткий ответ"]


# --------------------------------------------------- маскирование промптов

class _Recorder(BaseLLM):
    def __init__(self):
        self.seen = None

    async def chat(self, messages, **kwargs):
        self.seen = messages
        return "ok"


def test_anonymizing_llm_masks_every_call_site():
    """ИНН/счета/названия организаций маскируются на уровне клиента, поэтому
    классификатор, Text-to-SQL и реранкер тоже защищены."""
    inner = _Recorder()
    llm = AnonymizingLLM(inner, ["ООО Вектор"])
    asyncio.run(llm.chat([{"role": "user", "content": "ИНН 7721546864 в ООО Вектор"}]))
    content = inner.seen[0]["content"]
    assert "7721546864" not in content
    assert "ООО Вектор" not in content


def test_anonymizing_llm_passes_through_mock_shape():
    """Обёртка не ломает формат сообщений (роль/содержимое)."""
    llm = AnonymizingLLM(MockLLM(), [])
    out = asyncio.run(llm.chat([{"role": "user", "content": "привет"}]))
    assert isinstance(out, str)


# --------------------------------------------------- свежесть RAG-корпуса

def test_render_answer_escapes_metric_names():
    """Шаблон экранирует имена, пришедшие из документов."""
    body = render_answer({
        "type": "factual",
        "metric": {"name": "<b>x</b>", "unit": "руб", "currency": "RUB"},
        "history": [{"label": "2024", "value": 1.0, "source": "a.xlsx"}],
    })
    assert "<b>x</b>" not in body
    assert "&lt;b&gt;" in body


def test_verifier_rejects_invented_small_number_with_unit():
    """Выдуманное «25 млн руб.» не проходит, хотя 25 — маленькое число."""
    template = "Выручка за 2024: 985,00 тыс. ₽."
    payload = {
        "type": "factual",
        "metric": {"name": "выручка", "unit": "руб", "currency": "RUB"},
        "history": [{"label": "2024", "value": 985000.0, "source": "a.xlsx"}],
    }
    ok, foreign = verify_answer("Выручка за 2024: 25 млн руб.", template, payload)
    assert not ok and foreign == [25_000_000.0]
    # а структурный счётчик без единицы допускается
    assert verify_answer("Данные по 3 источникам.", template, payload)[0]


# --------------------------------------------------- привязка числа к периоду

_HIST = {
    "type": "factual",
    "metric": {"name": "выручка", "unit": "руб", "currency": "RUB"},
    "history": [
        {"label": "2023", "value": 412_000_000.0, "source": "s"},
        {"label": "2024", "value": 468_500_000.0, "source": "s"},
        {"label": "2025", "value": 536_200_000.0, "source": "s"},
    ],
}


def test_attribution_detects_swapped_periods():
    """Число настоящее, но относится к другому году — это ошибка ответа."""
    bad = claim_misattributions("Выручка за 2025 год составила 412,00 млн ₽.", _HIST)
    assert bad and "2023" in bad[0]


def test_attribution_ignores_valid_answers():
    """Ложное обвинение хуже пропуска: верные ответы не трогаем."""
    valid = [
        "Выручка в 2024 году составила 468,50 млн ₽, что больше, чем в 2023 (412,00 млн ₽).",
        "📊 Данные: 2023 — 412,00 млн ₽; 2024 — 468,50 млн ₽; 2025 — 536,20 млн ₽.",
        "Выручка выросла с 412,00 млн ₽ до 536,20 млн ₽.",
        "Данные из o.xlsx · Отчёт · B4",
        "В 2024 году 12 месяцев.",
    ]
    for text in valid:
        assert claim_misattributions(text, _HIST) == [], text


def test_attribution_needs_history():
    """Без истории показателя проверка привязки не выполняется."""
    assert claim_misattributions("За 2024 — 999,00 млн ₽.", {"type": "nodata"}) == []
    assert claim_misattributions("За 2024 — 999,00 млн ₽.", None) == []


# --------------------------------------------------- точные деньги

def test_money_column_round_trips_exactly():
    """Decimal переживает запись и чтение без потери знаков (SQLite: TEXT)."""
    import asyncio as _asyncio
    from decimal import Decimal as _D

    from sqlalchemy import select as _select

    from app.storage import Fact, make_engine, make_sessionmaker

    async def run(tmp: str):
        settings.database_url = f"sqlite+aiosqlite:///{tmp}/money.db"
        settings.data_dir = __import__("pathlib").Path(tmp)
        engine = await make_engine(settings)
        S = make_sessionmaker(engine)
        values = [_D("12699999.999999998"), _D("0.000001"), _D("-3.30"), _D("8300000")]
        async with S() as s:
            for _i, v in enumerate(values, start=1):
                s.add(Fact(org_id=1, metric_id=1, document_id=1, period_type="year",
                           period_label="2024", period_start=date(2024, 1, 1),
                           period_end=date(2024, 12, 31), value=v))
            await s.commit()
        async with S() as s:
            got = (await s.execute(_select(Fact.value).order_by(Fact.id))).scalars().all()
        await engine.dispose()
        return got

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        got = _asyncio.run(run(tmp))
    assert all(isinstance(v, _D) for v in got), got
    assert got == [_D("12699999.999999998"), _D("0.000001"), _D("-3.30"), _D("8300000")]


def test_sum_of_decimals_is_exact():
    """Сумма десятичных не накапливает ошибку, в отличие от float."""
    from decimal import Decimal as _D

    tenth = parse_number("0,1")
    assert sum([tenth] * 10) == _D("1")
    # последовательное накопление во float даёт 0.9999999999999999
    total = 0.0
    for _ in range(10):
        total += 0.1
    assert total != 1.0


# --------------------------------------------------- очистка каталога загрузок

def test_cleanup_orphan_uploads(tmp_path):
    """Файлы без ссылки в БД удаляются, связанные — остаются."""
    import asyncio as _asyncio

    from app.storage import Document, cleanup_orphan_uploads, make_engine, make_sessionmaker

    async def run():
        settings.database_url = f"sqlite+aiosqlite:///{tmp_path}/orph.db"
        settings.data_dir = tmp_path
        engine = await make_engine(settings)
        S = make_sessionmaker(engine)
        settings.upload_dir.mkdir(parents=True, exist_ok=True)
        keep = settings.upload_dir / "keep.xlsx"
        orphan = settings.upload_dir / "orphan.xlsx"
        keep.write_bytes(b"x")
        orphan.write_bytes(b"y")
        async with S() as s:
            s.add(Document(org_id=1, original_name="keep.xlsx", stored_path=str(keep),
                           file_hash="h", doc_type="xlsx", status="processed"))
            await s.commit()
            removed, freed = await cleanup_orphan_uploads(s, settings)
            await s.commit()
        await engine.dispose()
        return removed, freed, keep.exists(), orphan.exists()

    removed, freed, keep_exists, orphan_exists = _asyncio.run(run())
    assert removed == 1 and freed == 1
    assert keep_exists and not orphan_exists
