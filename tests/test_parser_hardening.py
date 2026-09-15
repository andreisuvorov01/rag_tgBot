"""Усиление парсера: форматы периодов/чисел из реальных отчётов, вертикальный
макет с заголовком над таблицей, строки-прочерки, CSV в cp1251, огромный лист."""
from decimal import Decimal

import openpyxl

from app.ingest.load_excel import load_csv, load_excel
from app.ingest.normalize import parse_number, parse_period, strip_unit_suffix


def test_period_formats_from_real_reports():
    expect = {
        "3 кв. 2024 г.": ("quarter", "2024-09-30"),
        "I квартал 2024 года": ("quarter", "2024-03-31"),
        "Q1'24": ("quarter", "2024-03-31"),
        "янв 2024": ("month", "2024-01-31"),
        "Сент. 2025": ("month", "2025-09-30"),
        "2024-01": ("month", "2024-01-31"),
        "на 31.12.2024": ("year", "2024-12-31"),
        "12 мес. 2024": ("year", "2024-12-31"),
        "9 месяцев 2024": ("month", "2024-09-30"),
        "9М 2024": ("month", "2024-09-30"),
        "1H2025": ("halfyear", "2025-06-30"),
        "1 п/г 2025": ("halfyear", "2025-06-30"),
        "2025 г.2": ("year", "2025-12-31"),
    }
    for raw, (ptype, end) in expect.items():
        p = parse_period(raw)
        assert p is not None and (p.ptype, str(p.end)) == (ptype, end), raw
    # названия строк, похожие на периоды, периодами не считаются
    for raw in ("Маркетинг 2024", "Долг", "Итого", "Дебиторка"):
        assert parse_period(raw) is None, raw


def test_number_formats_english_and_unicode_minus():
    assert parse_number("1,234.56") == Decimal("1234.56")
    assert parse_number("1,234,567") == Decimal("1234567")
    assert parse_number("−1 200") == Decimal("-1200")  # юникод-минус
    assert parse_number("1,5") == Decimal("1.5") and parse_number("1,500") == Decimal("1.5")  # русская десятичная запятая не ломается


def test_strip_unit_suffix():
    assert strip_unit_suffix("Выручка, тыс. руб.") == "Выручка"
    assert strip_unit_suffix("Аренда (млн руб.)") == "Аренда"
    assert strip_unit_suffix("Доля, %") == "Доля"
    assert strip_unit_suffix("тыс. руб.") == "тыс. руб."  # нечего оставлять — не трогаем


def test_vertical_layout_with_title_row_and_dash_rows(tmp_path):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Динамика"
    ws.append(["Отчёт о продажах по месяцам, тыс. руб."])   # заголовок отчёта над шапкой
    ws.append(["№", "Месяц", "Выручка", "Расходы"])
    for i, (m, rev, exp) in enumerate([("янв 2024", "1 200", "800"), ("фев 2024", "1 350", "-"),
                                       ("мар 2024", "1 500", "900"), ("апр 2024", "1 600", "950")], 1):
        ws.append([i, m, rev, exp])
    p = tmp_path / "vert.xlsx"
    wb.save(p)
    doc = load_excel(str(p))
    got = {(f.metric_name, f.period.label): f.value for f in doc.facts}
    assert got[("выручка", "01.2024")] == 1_200_000.0   # единица из заголовка отчёта
    assert got[("выручка", "03.2024")] == 1_500_000.0
    assert ("расходы", "02.2024") not in got and got[("расходы", "04.2024")] == 950_000.0


def test_dash_row_is_not_a_section(tmp_path):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Показатель", "2023", "2024"])
    ws.append(["Выручка", "100", "120"])
    ws.append(["Дивиденды", "-", "-"])      # данных нет, но это не раздел
    ws.append(["Аренда", "50", "60"])
    p = tmp_path / "dash.xlsx"
    wb.save(p)
    doc = load_excel(str(p))
    assert doc.sections == []
    assert all(f.section is None for f in doc.facts)


def test_csv_cp1251(tmp_path):
    p = tmp_path / "report.csv"
    p.write_bytes("Показатель;2023;2024\nВыручка;100;120\nАренда;50;60\n".encode("cp1251"))
    doc = load_csv(str(p))
    names = {f.metric_name for f in doc.facts}
    assert names == {"выручка", "аренда"}


def test_huge_formatted_sheet_does_not_allocate_full_grid(tmp_path):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Показатель", "2023", "2024"])
    ws.append(["Выручка", "100", "120"])
    ws.cell(row=200_000, column=1).value = None  # «раздутые» размеры листа без данных
    ws.cell(row=200_000, column=1).number_format = "0.00"
    p = tmp_path / "huge.xlsx"
    wb.save(p)
    doc = load_excel(str(p))
    assert {f.metric_name for f in doc.facts} == {"выручка"}


def test_bfo_balance_sheet_grid():
    """Бухгалтерский баланс из ГИР БО (pdfplumber-грид): разделы в колонке
    «Пояснения», «Итого по разделу I», сноски «капитал5», единица с титула,
    одно имя в двух разделах, «в том числе» с отступом в соседней колонке."""
    from app.ingest.load_excel import disambiguate_by_section, grid_to_parsed
    from app.ingest.normalize import detect_unit
    from app.ingest.pipeline import _validate_totals

    grid = [
        ["Пояснения1", "Наименование показателя", None, "Код\nстроки", "На 31 декабря\n2025 г.2", "На 31 декабря\n2024 г.3"],
        ["1", "2", None, "3", "4", "5"],
        ["Актив", None, None, None, None, None],
        ["I. Внеоборотные активы", None, None, None, None, None],
        ["4.2", "Основные средства", None, "1150", "100", "90"],
        ["", "Финансовые вложения", None, "1170", "50", "40"],
        ["", "Итого по разделу I", None, "1100", "150", "130"],
        ["IV. Долгосрочные обязательства", None, None, None, None, None],
        ["4.9", "Заемные средства", None, "1410", "10", "10"],
        ["", "Итого по разделу IV", None, "1400", "10", "10"],
        ["V. Краткосрочные обязательства", None, None, None, None, None],
        ["4.9", "Заемные средства", None, "1510", "20", "25"],
        ["4.8", "Уставный капитал5", None, "1310", "5", "5"],
        ["", "", "в том числе:", "", "", ""],
        ["", "", "оплаченный капитал", "1311", "5", "5"],
        ["", "Итого по разделу V", None, "1500", "25", "30"],
        ["", "БАЛАНС (пассив)", None, "1700", "185", "170"],
    ]
    facts, _chunks, _warns, sections = grid_to_parsed(
        grid, "стр. 2", default_unit=detect_unit("Единица измерения Тыс. руб."),
    )
    disambiguate_by_section(facts)
    by = {(f.metric_name, f.period.label): f for f in facts}
    assert sections[0] == "внеоборотные активы" and "актив" not in sections  # «Актив» без своих строк — не раздел
    assert by[("основные средства", "2025")].value == 100_000 and by[("основные средства", "2025")].currency == "RUB"
    assert by[("внеоборотные активы итого", "2025")].is_total
    assert ("заемные средства (долгосрочные обязательства)", "2025") in by
    assert by[("заемные средства (краткосрочные обязательства)", "2025")].value == 20_000
    assert ("уставный капитал", "2025") in by                      # сноска «5» отброшена
    assert by[("оплаченный капитал", "2025")].parent_hint == "уставный капитал"
    assert by[("баланс (пассив)", "2025")].section is None
    assert ("в том числе", "2025") not in by
    assert _validate_totals(facts) == []                            # итоги разделов сходятся


async def test_distinct_rows_of_one_document_never_merge(tmp_path):
    """«Чистая прибыль» и «валовая прибыль» из одной таблицы похожи по
    векторам, но это разные строки — синонимами не становятся."""
    from app.config import settings
    from app.embeddings import EmbeddingService
    from app.ingest.pipeline import process_document
    from app.storage import find_metric_by_name, make_engine, make_sessionmaker, register_user

    settings.database_url = f"sqlite+aiosqlite:///{tmp_path}/t.db"
    settings.data_dir = tmp_path
    engine = await make_engine(settings)
    sessions = make_sessionmaker(engine)
    emb = EmbeddingService(settings)
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Показатель", "2024", "2025"])
    ws.append(["Валовая прибыль (убыток)", "100", "110"])
    ws.append(["Чистая прибыль (убыток)", "10", "12"])
    ws.append(["Отложенные налоговые активы", "5", "6"])
    ws.append(["Отложенные налоговые обязательства", "7", "8"])
    p = tmp_path / "pl.xlsx"
    wb.save(p)
    async with sessions() as s:
        await register_user(s, 1, "A", "ООО")
        await process_document(s, emb, settings, org_id=1, user_id=1, original_name="pl.xlsx", content=p.read_bytes())
        await s.commit()
        for name in ("валовая прибыль (убыток)", "чистая прибыль (убыток)",
                     "отложенные налоговые активы", "отложенные налоговые обязательства"):
            assert (await find_metric_by_name(s, 1, name)) is not None, name
        assert (await find_metric_by_name(s, 1, "чистая прибыль (убыток)")).id != \
               (await find_metric_by_name(s, 1, "валовая прибыль (убыток)")).id
    await engine.dispose()
