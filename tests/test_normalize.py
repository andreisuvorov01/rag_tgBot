"""Тесты нормализации: числа, единицы, периоды."""
from datetime import date
from decimal import Decimal

from app.ingest.normalize import detect_unit, parse_number, parse_period, slugify_code, split_chunks


def test_parse_number_basic():
    # результат — Decimal (точные деньги): 12,7 * 1e6 во float давало
    # 12699999.999999998, и эта «копейка» уезжала в базу
    assert parse_number("12 700 000,00") == Decimal("12700000.00")
    assert parse_number("12 700 000.00") == Decimal("12700000.00")
    assert parse_number("12,7") == Decimal("12.7")
    assert parse_number(42) == Decimal(42)
    assert parse_number("(1 200)") == Decimal(-1200)
    assert parse_number("1 200-") == Decimal(-1200)
    assert parse_number("—") is None
    assert parse_number("-") is None
    assert parse_number("н/д") is None
    assert parse_number("x") is None
    assert parse_number("") is None
    assert parse_number(None) is None
    assert parse_number("12 700", 1000.0) == Decimal("12700000.0")


def test_parse_number_is_exact_where_float_is_not():
    """Причина перехода на Decimal: масштабирование без потери знака."""
    assert parse_number("12,7", 1e6) == Decimal("12700000")
    assert Decimal("0.1") + Decimal("0.2") == Decimal("0.3")
    # проверенные расхождения float-пути, ради которых всё и делалось:
    assert 8.3 * 1e6 == 8300000.000000001        # не 8300000
    assert parse_number("8,3", 1e6) == Decimal("8300000")
    total = 0.0
    for _ in range(10):
        total += 0.1
    assert total != 1.0                           # накапливающаяся ошибка сумм
    assert sum([Decimal("0.1")] * 10) == Decimal("1.0")


def test_parse_number_unicode_spaces():
    assert parse_number("12\u00a0700") == Decimal("12700")
    assert parse_number("12\u202f700,5") == Decimal("12700.5")


def test_detect_unit():
    u = detect_unit("Операционные расходы, тыс. руб.")
    assert u.multiplier == 1e3 and u.currency == "RUB"
    u = detect_unit("Выручка, млн ₽")
    assert u.multiplier == 1e6 and u.currency == "RUB"
    u = detect_unit("Рентабельность, %")
    assert u.unit == "%"
    u = detect_unit("Показатель")
    assert u.multiplier == 1.0 and u.unit is None


def test_parse_period_year():
    for text in ("2024", "2024 г.", "FY2024", "2024 год"):
        p = parse_period(text)
        assert p is not None and p.ptype == "year", text
        assert p.start == date(2024, 1, 1) and p.end == date(2024, 12, 31)


def test_parse_period_quarter():
    p = parse_period("1 кв. 2024")
    assert p and p.ptype == "quarter" and p.start == date(2024, 1, 1) and p.end == date(2024, 3, 31)
    p = parse_period("Q3 2024")
    assert p and p.ptype == "quarter" and p.end == date(2024, 9, 30)
    p = parse_period("II квартал 2025")
    assert p and p.ptype == "quarter" and p.start == date(2025, 4, 1)


def test_parse_period_month():
    p = parse_period("январь 2024")
    assert p and p.ptype == "month" and p.start == date(2024, 1, 1) and p.end == date(2024, 1, 31)
    p = parse_period("03.2025")
    assert p and p.ptype == "month" and p.end == date(2025, 3, 31)


def test_parse_period_range_and_date():
    p = parse_period("01.01.2023-31.12.2023")
    assert p and p.ptype == "year" and p.label == "2023"
    p = parse_period("15.03.2024")
    assert p and p.ptype == "day"


def test_parse_period_rejects_data():
    assert parse_period("Выручка") is None
    assert parse_period("12 700 000") is None
    assert parse_period("") is None


def test_slugify():
    assert slugify_code("Аренда спецтехники") == "arenda_spetstehniki"
    assert slugify_code("Выручка!") .startswith("vyruchka")


def test_split_chunks():
    text = "\n\n".join(" ".join(["слово"] * 100) for _ in range(5))
    chunks = split_chunks(text, max_words=150)
    assert len(chunks) >= 2
    assert all(len(c.split()) <= 150 for c in chunks)
