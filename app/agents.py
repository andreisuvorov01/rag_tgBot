"""Агент-верификатор (агент-критик) — контур генерация → критика → повтор.

Практика grounding (FinanceBench): LLM в финансовом QA регулярно «дорисовывает»
числа. Верификатор извлекает числа из ответа и проверяет, что каждое
присутствует в ДАННЫХ, которые модель видела, либо в шаблонном ответе,
собранном программно из тех же данных.

Почему сверка идёт и с payload, и с шаблоном
--------------------------------------------
Шаблон рендерится через fmt_money и «переписывает» значения: 3 500 000
превращается в «3,50 млн ₽», а в промпт уходит сырой JSON с 3500000.0.
Сверка только по шаблону поэтому отвергала корректные ответы (регресс:
модель честно брала 3500000 из блока ДАННЫЕ и получала отказ). Теперь
допустимым считается число, которое есть либо среди сырых значений payload,
либо среди отрендеренных — с учётом единиц (млн/тыс → множитель) и валюты.

Число в ответе принимается, если совпало по величине с допустимым и единицы
совместимы. Выдуманное целое 1..31 без единицы («3 точки») допускается как
структурный счётчик, но «25 млн руб.» — уже нет: у него есть единица.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

# Разделитель тысяч — ТОЛЬКО пробел/апостроф. Запятая и точка здесь
# десятичные: иначе «Периоды 2023 2024» склеивались в 2023202.
_NUM_RE = re.compile(
    r"\d{1,3}(?:[ \u00a0\u202f'\u2019]\d{3})+(?:[.,]\d+)?"
    r"|\d+(?:[.,]\d+)?"
)
_UNIT_RE = re.compile(
    r"\s*(млрд|млн|тыс|thousand|руб|рублей|рубля|₽|р\.|rub|usd|долл\w*|\$|eur|евро|€|%)",
    re.I,
)
_UNIT_FACTOR = {
    "млрд": 1e9,
    "млн": 1e6,
    "тыс": 1e3,
    "thousand": 1e3,
}
_UNIT_KIND = {
    "руб": "₽", "рублей": "₽", "рубля": "₽", "₽": "₽", "р.": "₽", "rub": "₽",
    "usd": "$", "долл": "$", "долларов": "$", "долл.": "$", "$": "$",
    "eur": "€", "евро": "€", "€": "€",
    "%": "%",
}
# структурные числа: количество точек, пунктов, источников — единицы нет
_SMALL_COUNTER_MAX = 31


@dataclass(frozen=True)
class _Num:
    """Число с нормализованной величиной и типом единицы."""
    raw: float          # как записано в тексте
    value: float        # величина в базовых единицах (млн -> ×1e6)
    unit: str           # '₽' | '$' | '€' | '%' | '' (нет единицы)


def _to_float(digits: str) -> float | None:
    s = digits.replace(" ", "").replace("\u00a0", "").replace("\u202f", "")
    s = s.replace("'", "").replace("\u2019", "")
    # запятая — десятичный разделитель (русская локаль), точка — тоже
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".")
    else:
        s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def _extract(text: str) -> list[_Num]:
    """Все числа текста с единицами, приведённые к базовым единицам."""
    out: list[_Num] = []
    for m in _NUM_RE.finditer(text or ""):
        val = _to_float(m.group(0))
        if val is None:
            continue
        # единицы могут идти цепочкой: «13,97 млн ₽» = множитель + валюта
        pos, unit = m.end(), ""
        while True:
            um = _UNIT_RE.match(text, pos)
            if not um:
                break
            token = um.group(1).casefold()
            factor = _UNIT_FACTOR.get(token, 1.0)
            if factor != 1.0:
                val = val * factor
            else:
                kind = _UNIT_KIND.get(token, "")
                if kind:
                    unit = kind
                elif unit == "":
                    # неизвестная единица — значимая (не структурная)
                    unit = token
            pos = um.end()
        out.append(_Num(raw=val, value=val, unit=unit))
    return out


def _extract_values(text: str) -> list[float]:
    """Обратная совместимость: только величины."""
    return [n.value for n in _extract(text)]


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= max(abs(b) * 0.005, 0.011)


# Деноминации, в которых модель может записать то же самое значение. В отчётах
# суммы почти всегда даны в тысячах или миллионах, и LLM естественно пишет
# «468,5 млн» там, где в ДАННЫХ лежит 468500000.0. Сверка обязана это
# принимать, иначе корректный ответ отклоняется и сгорают все попытки.
_SCALE_FACTORS = (1.0, 1e3, 1e6, 1e9)


def _same_amount(a: float, b: float) -> bool:
    """Совпадают ли величины с точностью до деноминации (млн/тыс/млрд)."""
    if _close(a, b):
        return True
    return any(_close(a * f, b) for f in _SCALE_FACTORS[1:])


def _payload_numbers(payload: dict[str, Any] | None) -> list[_Num]:
    """Допустимые числа из payload: значения с единицами/валютами.

    Идём по известным числовым полям, чтобы не подбирать мусор из текста
    (ссылки на ячейки 'B12', имена листов и т.п. числами не являются).
    """
    if not payload:
        return []
    out: list[_Num] = []

    def add(v: Any, unit: str) -> None:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return
        out.append(_Num(raw=float(v), value=float(v), unit=unit))

    metric = payload.get("metric") or {}
    cur = (metric.get("currency") or "RUB")
    unit_field = metric.get("unit")
    unit = "%" if unit_field == "%" else _UNIT_KIND.get(str(cur).casefold(), "")

    for row in payload.get("history") or []:
        add((row or {}).get("value"), unit)
    parent = payload.get("parent") or {}
    add(parent.get("value"), unit)
    add(parent.get("share_pct"), "%")

    # сравнение / план-факт
    comp = payload.get("computed") or {}
    add(comp.get("abs_change"), unit)
    add(comp.get("deviation_pct"), "%")
    add(comp.get("change_pct"), "%")
    add(comp.get("ratio_pct"), "%")
    add(comp.get("total"), unit)
    add(comp.get("months"), "")
    add(comp.get("avg_growth_pct"), "%")
    add(comp.get("cagr_pct"), "%")
    add(comp.get("trend_value"), unit)
    # производные прогноза, посчитанные кодом (см. AnswerPipeline._forecast)
    add(comp.get("forecast_vs_fact_pct"), "%")
    add(comp.get("forecast_vs_fact_abs"), unit)
    add(comp.get("interval_width_abs"), unit)
    add(comp.get("interval_width_pct"), "%")
    add(comp.get("scenario_effect_pct"), "%")
    add(comp.get("scenario_effect_abs"), unit)

    # сводка показателей для общего ответа (fallback)
    for d in payload.get("digest") or []:
        add((d or {}).get("value"), "%" if (d or {}).get("unit") == "%" else unit)
        add((d or {}).get("periods"), "")

    # состав показателя
    for item in payload.get("items") or []:
        add((item or {}).get("value"), unit)
        add((item or {}).get("share_pct"), "%")
    add(payload.get("total"), unit)

    # ранжирование
    for r in payload.get("ranking") or []:
        add((r or {}).get("growth_pct"), "%")

    # прогноз
    fc = payload.get("forecast") or {}
    add(fc.get("base"), unit)
    add(fc.get("low"), unit)
    add(fc.get("high"), unit)
    add(fc.get("base_without_scenario"), unit)
    add(fc.get("reference"), unit)
    add(fc.get("fact_to_date"), unit)
    add(fc.get("rest_forecast"), unit)
    add(fc.get("backtest_error_pct"), "%")
    for v in (fc.get("methods") or {}).values():
        add(v, unit)
    sc = fc.get("scenario") or {}
    add(sc.get("growth_multiplier"), "")

    # обзор документа
    add(payload.get("facts"), "")
    add(payload.get("chunks"), "")
    add(payload.get("operations"), "")
    for group in payload.get("metrics_by_kind") or []:
        for item in (group or {}).get("items") or []:
            add((item or {}).get("value"), unit)
    return out


def _allowed(answer_num: _Num, allowed: list[_Num]) -> bool:
    """Число ответа допустимо, если совпало по величине с учётом единиц.

    Величина сверяется также с точностью до деноминации: «468,5» и
    468500000.0 — одно и то же значение, записанное в млн и в рублях.
    """
    if answer_num.unit and answer_num.unit not in ("₽", "$", "€", "%"):
        # единица не распознана как валюта/процент — принимаем по величине
        return any(_same_amount(answer_num.value, a.value) for a in allowed)
    for a in allowed:
        if not _same_amount(answer_num.value, a.value):
            continue
        if a.unit == answer_num.unit or not a.unit or not answer_num.unit:
            return True
    return False


# Типы ответов, где модель обязана считать производные сама: «на сколько
# изменилось», «во сколько раз», «какую долю составляет», «на сколько прогноз
# отличается от факта». Только здесь производные и разрешены.
_DERIVED_TYPES = frozenset({"forecast", "compare", "breakdown", "sql_result", "rank"})
_DERIVED_LIMIT = 4000   # защита от комбинаторного взрыва на больших payload


def _derived_pool(payload: dict[str, Any] | None) -> list[_Num]:
    """Все числа, из которых модель вправе считать производные.

    Для прогноза это история + прогнозные значения; для состава — статьи и
    итог; для план/факта — план, факт и уже посчитанные отклонения. Именно
    отношения и разности этих чисел модель и приводит в тексте («доля 90%»,
    «на 1,6 млн больше», «на 4,4 п.п. выше»).
    """
    if not payload:
        return []
    ptype = str(payload.get("type") or "")
    if ptype not in _DERIVED_TYPES:
        return []
    pool = [n for n in _payload_numbers(payload) if n.unit in ("₽", "$", "€", "%")]
    fc = payload.get("forecast") or {}

    def add(v: Any, unit: str) -> None:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return
        pool.append(_Num(raw=float(v), value=float(v), unit=unit))

    # веса прогнозного ансамбля: модель считает по ним вклад методов
    for w in (fc.get("weights") or {}).values():
        add(w, "%")
    return pool


def derived_numbers(payload: dict[str, Any] | None) -> list[_Num]:
    """Производные числа, которые модель вправе посчитать сама.

    В прогнозе, составе и план/факте мало процитировать значения: нужно сказать,
    на сколько изменилось, во сколько раз, какую долю составляет статья, на
    сколько процентов прогноз отличается от факта. Этих чисел в payload нет —
    модель считает их сама, и без такого допуска верификатор отклонял корректные
    ответы (в логе: «посторонние числа: [10.0]», «[30600000.0, 89.7]»), после
    чего ответ уходил в шаблон.

    Считаем производные ТОЧНО из чисел payload, поэтому выдуманное значение
    по-прежнему не проходит: попасть в список можно лишь как разность или
    отношение двух чисел из ДАННЫХ. Абсолютные суммы при этом проверяются
    строго — производные дают только проценты и разности.
    """
    pool = _derived_pool(payload)
    if len(pool) < 2:
        return []
    out: list[_Num] = []

    def _pct_series(value: float) -> list[_Num]:
        """Процент в том виде, как его пишет модель: «10,2 %», «10 %», «90 %»."""
        return [_Num(raw=float(r), value=float(r), unit="%")
                for r in (round(value, 6), round(value, 1), round(value))]

    money = [n for n in pool if n.unit != "%"]
    # Суммы статей: «зарплаты и аренда спецтехники вместе — 30,6 млн, это 90%».
    # Модель складывает 2–3 крупнейшие статьи и делит на итог — без сумм в
    # производных такие корректные ответы отклонялись.
    for i, a in enumerate(money):
        for b in money[i + 1:]:
            if a.unit != b.unit:
                continue
            out.append(_Num(raw=a.value + b.value, value=a.value + b.value, unit=a.unit))
    sums = list(out)

    for i, a in enumerate(pool):
        for b in pool[i + 1:]:
            if a.unit != b.unit:
                continue
            diff = abs(a.value - b.value)
            if diff > 0:
                out.append(_Num(raw=diff, value=diff, unit=a.unit))
            hi, lo = max(a.value, b.value), min(a.value, b.value)
            if lo > 0:
                ratio = hi / lo
                out.extend(_pct_series((ratio - 1.0) * 100.0))
                out.extend(_pct_series(100.0 / ratio))
                # доля одного в другом: 27,54 млн из 30,6 млн = 90%
                out.extend(_pct_series(lo / hi * 100.0))
            if len(out) >= _DERIVED_LIMIT:
                return out

    # доли сумм: сумма двух статей против итога и против третьей статьи
    for s in sums:
        for other in money:
            if other.value > 0 and s.value > 0 and other.value != s.value:
                lo, hi = min(s.value, other.value), max(s.value, other.value)
                out.extend(_pct_series(lo / hi * 100.0))
        if len(out) >= _DERIVED_LIMIT:
            break
    return out


def _percent_in_data_range(answer_num: _Num, allowed: list[_Num]) -> bool:
    """Процент в пределах разброса процентов из данных (плюс-минус 2 п.п.).

    Последняя страховка от ложного отклонения: модель считает проценты от
    округлённых чисел («~90%» там, где точно 89,7%) или усредняет их, и точного
    совпадения может не быть. Диапазон берём из самих данных, поэтому выдуманный
    процент (например 43% при данных 4–13%) по-прежнему не проходит. Абсолютные
    суммы этой поблажкой не покрываются — только проценты.
    """
    if answer_num.unit != "%":
        return False
    pcts = [a.value for a in allowed if a.unit == "%"]
    if not pcts:
        return False
    return min(pcts) - 2.0 <= answer_num.value <= max(pcts) + 2.0


def verify_answer(
    answer: str, template: str, payload: dict[str, Any] | None = None,
    *, allow_derived: bool = True,
) -> tuple[bool, list[float]]:
    """-> (пройден, список посторонних чисел).

    Допустимыми считаются числа из payload (то, что видела модель) и из
    шаблонного ответа. Годы и структурные счётчики без единицы пропускаются.
    Для прогноза, состава, сравнения и ранжирования дополнительно разрешены
    производные числа (разности, суммы и проценты роста), посчитанные из данных —
    см. derived_numbers.
    """
    allowed = _payload_numbers(payload) + _extract(template)
    if allow_derived:
        allowed = allowed + derived_numbers(payload)
    foreign: list[float] = []
    for n in _extract(answer):
        if n.unit == "" and n.raw.is_integer() and 1900 <= n.raw <= 2100:
            continue  # год
        if n.unit == "" and 0 < n.raw <= _SMALL_COUNTER_MAX and n.raw.is_integer():
            continue  # счётчик без единицы: «3 точки», «2 источника»
        if _allowed(n, allowed):
            continue
        if allow_derived and str((payload or {}).get("type") or "") in _DERIVED_TYPES \
                and _percent_in_data_range(n, _payload_numbers(payload)):
            continue
        foreign.append(n.raw)
    return (not foreign), foreign


@dataclass
class VerificationResult:
    ok: bool
    foreign_numbers: list[float]
    misattributed: list[str] = field(default_factory=list)


# пара «период → число» в тексте: «2024: 536,20 млн ₽», «за 2023 — 412 млн»
_CLAIM_RE = re.compile(
    r"(?P<period>(?:19|20)\d{2}|[1-4]\s*кв\.?\s*(?:20\d{2})?|\d{1,2}\.\d{4})"
    r"[^\d\n]{0,24}?"
    r"(?P<num>\d[\d \u00a0\u202f']*(?:[.,]\d+)?)"
    r"\s*(?P<unit>млрд|млн|тыс|thousand|руб\w*|₽|rub|usd|долл\w*|\$|eur|евро|€|%)?",
    re.I,
)
_CLAIM_NOISE = ("стр", "лист", "sheet", "таблица", "табл", "раздел", "пункт", "рис")


def claim_misattributions(answer: str, payload: dict[str, Any] | None) -> list[str]:
    """Найти числа, приписанные «не своему» периоду.

    Величина может быть настоящей, но относиться к другому году: «выручка за
    2024 — 412 млн», тогда как 412 млн — это 2023. Верификатор величин такое
    пропускает, потому что число действительно есть в данных.

    Проверяем только однозначные пары «период → число» и только когда
    значение опознано среди данных payload. Неоднозначное (число встречается
    в нескольких периодах, ссылка на ячейку, номер строки) не трогаем:
    ложное обвинение хуже пропуска, оно выбрасывает верный ответ.
    """
    if not payload:
        return []
    # период -> множество величин, которые к нему относятся
    by_period: dict[str, set[float]] = {}
    for row in payload.get("history") or []:
        label = str((row or {}).get("label") or "").strip()
        value = (row or {}).get("value")
        if not label or not isinstance(value, (int, float)):
            continue
        year = re.search(r"(?:19|20)\d{2}", label)
        key = year.group(0) if year else label
        by_period.setdefault(key, set()).add(float(value))
    if not by_period:
        return []
    all_values = {v for vals in by_period.values() for v in vals}
    known_periods = set(by_period)

    problems: list[str] = []
    for m in _CLAIM_RE.finditer(answer or ""):
        # «стр. 12» / «лист 3» — это не пары «период: значение»
        prefix = (answer[max(0, m.start() - 12):m.start()] or "").casefold()
        if any(w in prefix for w in _CLAIM_NOISE):
            continue
        period_raw = m.group("period").strip()
        year = re.search(r"(?:19|20)\d{2}", period_raw)
        period = year.group(0) if year else period_raw
        if period not in known_periods:
            continue
        num = _to_float(m.group("num"))
        if num is None:
            continue
        unit = (m.group("unit") or "").casefold()
        factor = _UNIT_FACTOR.get(unit, 1.0)
        value = num * factor
        # значение должно быть опознано в данных — иначе это дело проверки величин
        if not any(_close(value, v) for v in all_values):
            continue
        here = by_period.get(period, set())
        if any(_close(value, v) for v in here):
            continue
        # куда это число относится на самом деле
        actual = sorted(p for p, vals in by_period.items() if any(_close(value, v) for v in vals))
        problems.append(
            f"{m.group(0).strip()!r}: значение относится к {', '.join(actual)}, а не к {period}"
        )
    return problems


class VerificationAgent:
    """Агент-критик: прогоняет ответ через проверку заземления чисел."""

    def __init__(self, settings):
        self.enabled = settings.verify_answers
        self.retries = max(0, settings.answer_retries)
        # производные числа (разности, проценты роста) разрешены для прогноза и
        # план/факта: там модель обязана считать их сама
        self.allow_derived = getattr(settings, "verify_allow_derived", True)

    async def review(
        self, answer: str, template: str, payload: dict[str, Any] | None = None
    ) -> VerificationResult:
        if not self.enabled:
            return VerificationResult(True, [])
        ok, foreign = verify_answer(answer, template, payload, allow_derived=self.allow_derived)
        if not ok:
            log.warning("Верификатор: посторонние числа в ответе LLM: %s", foreign)
        # отдельно — числа, приписанные не тому периоду
        bad_pairs = claim_misattributions(answer, payload)
        if bad_pairs:
            log.warning("Верификатор: неверная привязка к периоду: %s", bad_pairs)
        return VerificationResult(ok and not bad_pairs, foreign, bad_pairs)

    def correction_note(self, foreign: list[float] | None = None,
                        misattributed: list[str] | None = None) -> str:
        """Замечание для повтора: перечисляем конкретные числа, иначе модель
        не понимает, что именно убирать, и повторяется."""
        listed = ""
        if foreign:
            shown = ", ".join(f"{n:g}" for n in foreign[:6])
            listed += f" Посторонние числа: {shown}."
        if misattributed:
            listed += " " + "; ".join(misattributed[:4]) + "."
        return (
            "\n\nПРЕДЫДУЩАЯ ПОПЫТКА ОТКЛОНЕНА ПРОВЕРКОЙ: в ответе были числа, "
            "отсутствующие в блоке ДАННЫЕ либо отнесённые не к тому периоду." + listed +
            " Перепиши ответ, используя ТОЛЬКО числа из ДАННЫХ (можно писать их "
            "как в ДАННЫХ, так и в сокращённом виде: 3 500 000 = 3,50 млн) и "
            "указывая каждый раз тот период, к которому число относится; "
            "если данных недостаточно — прямо скажи об этом."
        )
