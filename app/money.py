"""Денежные величины: точный тип хранения и Decimal-арифметика.

Почему не float
---------------
`Fact.value` хранился как Float, поэтому сумма строк считалась двоичными
дробями: `12.7 * 1e6 == 12699999.999999998`, и это значение попадало в базу.
Для финансовой отчётности нужна десятичная арифметика с известным числом
знаков, иначе сверка итогов и агрегаты расходятся на копейки.

Как храним
----------
SQLite не имеет настоящего NUMERIC: даже `Numeric` SQLAlchemy кладёт REAL и
возвращает float. Поэтому для SQLite значение хранится как TEXT с
фиксированным масштабом, а для PostgreSQL — как NUMERIC(28, 6) (там точность
обеспечивает сама СУБД, и SQL-агрегаты вида SUM(f.value) остаются числовыми).

Масштаб 6 знаков выбран так, чтобы одним типом покрыть и деньги, и проценты:
копейки с запасом, а `8,3%` не превращается в 8,3 с потерей хвоста.
"""
from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

from sqlalchemy import Numeric, Text
from sqlalchemy.types import TypeDecorator

# знаков после запятой в хранилище
SCALE = 6
_QUANT = Decimal(1).scaleb(-SCALE)


def to_decimal(value: object) -> Decimal | None:
    """Привести значение к Decimal без потери точности, где это возможно.

    Для float берём repr: `Decimal(0.1)` даёт 0.1000000000000000055511151231,
    а `Decimal("0.1")` — ровно 0.1.
    """
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    if isinstance(value, bool):
        return None
    try:
        if isinstance(value, float):
            if value != value or value in (float("inf"), float("-inf")):
                return None
            return Decimal(repr(value))
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None


def quantize(value: Decimal) -> Decimal:
    """Округлить до масштаба хранилища (половина вверх, как в бухгалтерии)."""
    return value.quantize(_QUANT, rounding=ROUND_HALF_UP)


def scale_multiplier(value: object) -> Decimal:
    """Множитель единиц (1e3/1e6/1e9) как Decimal.

    `Decimal * float` в Python бросает TypeError, поэтому все множители
    приводятся к Decimal до умножения.
    """
    d = to_decimal(value)
    return Decimal(1) if d is None else d


def money(value: object) -> Decimal | None:
    """Нормализовать значение для записи в БД."""
    d = to_decimal(value)
    return None if d is None else d


def as_float(value: object) -> float:
    """Значение для аналитики: numpy/statsmodels работают с float.

    Огрубление происходит только на границе расчётов, в хранилище значение
    остаётся точным.
    """
    d = to_decimal(value)
    return 0.0 if d is None else float(d)


class Money(TypeDecorator):
    """Точное десятичное число: NUMERIC на PostgreSQL, TEXT на SQLite."""

    impl = Numeric(28, SCALE)
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(Numeric(28, SCALE))
        return dialect.type_descriptor(Text())

    def process_bind_param(self, value, dialect):
        d = money(value)
        if d is None:
            return None
        return d if dialect.name == "postgresql" else format(d, "f")
    def process_result_value(self, value, dialect):
        if value is None:
            return None
        return to_decimal(value)
