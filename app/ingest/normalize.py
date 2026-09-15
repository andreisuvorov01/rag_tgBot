"""Нормализация: числа (бухгалтерские скобки, разделители, единицы),
периоды (годы, кварталы, месяцы, диапазоны), разбиение текста на чанки."""
from __future__ import annotations

import calendar
import re
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal

from ..money import quantize, scale_multiplier, to_decimal

# ---------------------------------------------------------------- числа ---

_NONE_TOKENS = {"", "-", "—", "–", "x", "х", "н/д", "н.д.", "na", "n/a", "null", "none", "."}
_UNIT_PATTERNS = [
    (re.compile(r"млрд", re.I), 1e9),
    (re.compile(r"млн", re.I), 1e6),
    (re.compile(r"\bтыс|\bтыс\.|thousand", re.I), 1e3),
]
_CURRENCY_PATTERNS = [
    (re.compile(r"₽|руб", re.I), "RUB"),
    (re.compile(r"\$|usd|долл", re.I), "USD"),
    (re.compile(r"€|eur|евро", re.I), "EUR"),
]
# Единый словарь слов-единиц. Список выводится из тех же токенов, что понимает
# detect_unit: раньше «снятие единицы» в parse_number_with_unit знало меньше
# слов, чем detect_unit, и «12,7 евро» / «12,7 тыс. штук» отбрасывались молча.
_UNIT_WORDS = tuple(dict.fromkeys([
    "млрд", "млн", "тыс", "thousand",
    "руб", "рублей", "рубля", "rub",
    "usd", "долл", "долларов", "долл.",
    "eur", "евро", "евро.",
    "шт", "штук", "штука", "штуки",
]))
_UNIT_WORD_RE = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in _UNIT_WORDS) + r")\b\.?",
    re.I,
)
_NUM_WITH_UNIT_STRIP_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(w) for w in _UNIT_WORDS) + r")\b\.?"
    r"|[%₽$€]",
    re.I,
)


@dataclass
class UnitInfo:
    multiplier: float = 1.0
    unit: str | None = None  # 'руб' | '%' | 'шт' | ...
    currency: str | None = None


def detect_unit(text: str) -> UnitInfo:
    info = UnitInfo()
    low = (text or "").casefold()
    for pat, mult in _UNIT_PATTERNS:
        if pat.search(low):
            info.multiplier = mult
            break
    for pat, cur in _CURRENCY_PATTERNS:
        if pat.search(low):
            info.currency = cur
            info.unit = "руб" if cur == "RUB" else cur.lower()
            break
    # «процент» — единица только как оборот «в процентах» / хвост «, процентов»;
    # «Проценты к получению» — это название показателя, не единица
    if re.search(r"%|в процентах|[,(]\s*процент\w*\s*\)?\s*$", low):
        info.unit = "%"
        info.multiplier = 1.0
    if re.search(r"\bшт\b|количество|кол-во", low):
        info.unit = info.unit or "шт"
    return info


def parse_number(raw: object, unit_multiplier: float = 1.0) -> Decimal | None:
    """Строка/значение -> Decimal. Бухгалтерские скобки = минус, пробелы и
    неразрывные пробелы = разделители тысяч, запятая = десятичный разделитель.
    Понимает европейский формат «1.234,56» (точка — разделитель тысяч).

    Возвращается Decimal, а не float: 12,7 * 1e6 во float даёт
    12699999.999999998, и такая «копейка» уезжала в базу и в сверку итогов.
    Значение округляется до масштаба хранилища (6 знаков) — этого хватает и
    для копеек, и для долей процента.
    """
    if raw is None or isinstance(raw, (datetime, date)):
        return None
    if isinstance(raw, bool):
        return None
    mult = scale_multiplier(unit_multiplier)
    if isinstance(raw, (int, float)):
        d = to_decimal(raw)
        if d is None:  # NaN/inf
            return None
        return quantize(d * mult)
    s = unicodedata.normalize("NFKC", str(raw)).strip()
    if s.casefold() in _NONE_TOKENS:
        return None
    s = s.replace("\u2212", "-")  # юникод-минус из Excel/1С
    negative = False
    if re.fullmatch(r"\(.*\)", s):
        negative = True
        s = s[1:-1]
    if s.endswith("-") or s.startswith("-"):
        negative = True
        s = s.replace("-", "").strip()
    if s.endswith("%"):
        s = s[:-1].strip()
    s = re.sub(r"[\s\u00a0\u202f']", "", s)
    # европейский формат тысяч: 1.234,56 / 1.234 (точки между группами цифр)
    if re.fullmatch(r"\d{1,3}(\.\d{3})+(,\d+)?", s):
        s = s.replace(".", "").replace(",", ".")
    # английский: 1,234.56 / 1,234,567 — только с точкой-десятичной или двумя
    # группами: «1,500» в русском отчёте — это 1,5, а не полторы тысячи
    elif re.fullmatch(r"\d{1,3}(,\d{3})+\.\d+|\d{1,3}(,\d{3}){2,}", s):
        s = s.replace(",", "")
    else:
        s = s.replace(",", ".")
    if not re.fullmatch(r"[+-]?\d+(\.\d+)?", s):
        return None
    d = to_decimal(s)
    if d is None:
        return None
    value = d * mult
    if negative:
        value = -value
    # «(0)» -> 0, а не -0
    return quantize(value) if value != 0 else Decimal(0)


def parse_number_with_unit(raw: object) -> tuple[Decimal | None, UnitInfo | None]:
    """Число с единицей прямо в ячейке: «12,7 млн», «9 800 тыс. руб.».
    Возвращает (значение, единицы) либо (None, None), если это не число с единицей."""
    if raw is None or isinstance(raw, (float, int, datetime, date)):
        return (parse_number(raw), None) if isinstance(raw, (float, int)) and not isinstance(raw, bool) else (None, None)
    s = unicodedata.normalize("NFKC", str(raw)).strip()
    if not s or s.casefold() in _NONE_TOKENS:
        return None, None
    info = detect_unit(s)
    t = _NUM_WITH_UNIT_STRIP_RE.sub(" ", s)
    # разделители тысяч убираем только между цифрами, десятичный разделитель сохраняем
    t = re.sub(r"(?<=\d)[\s\u00a0\u202f'](?=\d)", "", t).strip()
    value = parse_number(t, info.multiplier)
    if value is None:
        return None, None
    return value, (info if (info.multiplier > 1 or info.unit) else None)


_VARIANTS = {
    "план": "plan",
    "планов": "plan",
    "бюджет": "budget",
    "прогноз": "forecast",
    "оценк": "estimate",
    "факт": "fact",
}


def detect_variant(text: str) -> str | None:
    """Вариант значения из заголовка колонки: «2024 (план)» -> 'plan'."""
    low = (text or "").casefold()
    for stem, variant in _VARIANTS.items():
        if stem in low:
            return variant
    return None


# --------------------------------------------------------------- периоды ---

_ROMAN = {"i": 1, "ii": 2, "iii": 3, "iv": 4}
# полные названия и стандартные сокращения; у сокращений нет хвоста [а-я]* —
# иначе «Маркетинг 2024» в названии строки читался бы как март 2024
_MONTH_STEM = (
    r"(январ[а-я]*|янв\.?|феврал[а-я]*|фев\.?|март[а-я]*|мар\.?|апрел[а-я]*|апр\.?|ма[йя]|"
    r"июн[а-я]*|июл[а-я]*|август[а-я]*|авг\.?|сентябр[а-я]*|сент?\.?|октябр[а-я]*|окт\.?|"
    r"ноябр[а-я]*|ноя\.?|декабр[а-я]*|дек\.?)"
)
_MONTH_BY_PREFIX = {"янв": 1, "фев": 2, "мар": 3, "апр": 4, "май": 5, "мая": 5, "июн": 6,
                    "июл": 7, "авг": 8, "сен": 9, "окт": 10, "ноя": 11, "дек": 12}


def _month_num(stem: str) -> int:
    return _MONTH_BY_PREFIX.get(stem[:3], 0)


@dataclass
class Period:
    ptype: str  # year | halfyear | quarter | month | day
    label: str
    start: date
    end: date

    def sort_key(self) -> date:
        return self.end


def period_from_date(d: date) -> Period:
    last = calendar.monthrange(d.year, d.month)[1]
    if d.day == last:
        if d.month == 12:
            return Period("year", str(d.year), date(d.year, 1, 1), date(d.year, 12, 31))
        if d.month in (3, 6, 9):  # конец квартала
            q = d.month // 3
            return Period("quarter", f"{q} кв. {d.year}", date(d.year, 3 * q - 2, 1), d)
        return Period("month", f"{d.month:02d}.{d.year}", date(d.year, d.month, 1), d)
    return Period("day", d.strftime("%d.%m.%Y"), d, d)


_YEAR = r"(20[0-4]\d)"


def _month_period(y: int, m: int) -> Period:
    return Period("month", f"{m:02d}.{y}", date(y, m, 1), date(y, m, calendar.monthrange(y, m)[1]))


def _quarter_period(y: int, q: int) -> Period:
    return Period("quarter", f"{q} кв. {y}", date(y, 3 * q - 2, 1),
                  date(y, 3 * q, calendar.monthrange(y, 3 * q)[1]))


def _ytd_period(y: int, n: int) -> Period:
    """Нарастающим итогом с начала года: «9 мес. 2024», «1 полугодие 2025»."""
    if n == 12:
        return Period("year", str(y), date(y, 1, 1), date(y, 12, 31))
    if n == 6:
        return Period("halfyear", f"1 полугодие {y}", date(y, 1, 1), date(y, 6, 30))
    return Period("month", f"{n} мес. {y}", date(y, 1, 1), date(y, n, calendar.monthrange(y, n)[1]))


def parse_period(raw: object) -> Period | None:
    """Заголовок колонки/строки -> Period. Понимает: 2024, '2024 г.2' (со сноской),
    'FY2024', '1 кв. 2024 г.', 'Q3 2024', "Q1'24", 'I квартал 2025 года', '9М 2024',
    'январь 2024', 'янв. 24', 'сент. 2025', '01.2024', '2024-01',
    '01.01.2023-31.12.2023', 'На 31 декабря 2025 г.' и 'на 31.12.2024' (БФО),
    'За январь–март 2025 г.', '9 месяцев 2024', '12 мес. 2024', '1 полугодие 2025',
    '1H2025'; и объекты date/datetime."""
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return period_from_date(raw.date())
    if isinstance(raw, date):
        return period_from_date(raw)
    s = unicodedata.normalize("NFKC", str(raw)).strip().casefold()
    if not s or len(s) > 40:
        return None
    s = re.sub(r"\s+", " ", s).replace("’", "'")
    # хвост «г.», «года», «year» и сноска-цифра после него: «2025 г.2», «3 кв. 2024 г.»
    s = re.sub(r"(?<=[\d\s.])\s*(?:г\.?|год[а-я]*|year)\s*\d?\s*$", "", s).strip()
    # «на 31.12.2024», «на 31 декабря 2025» — дата баланса
    s = re.sub(r"^на\s+", "", s)

    # диапазон дат
    m = re.fullmatch(r"(\d{2}[./]\d{2}[./]\d{4})\s*[-–—т]+\s*(\d{2}[./]\d{2}[./]\d{4})", s)
    if m:
        d1, d2 = _parse_date(m.group(1)), _parse_date(m.group(2))
        if d1 and d2:
            if d1.month == 1 and d1.day == 1 and d2.month == 12 and d2.day == 31:
                return Period("year", str(d1.year), d1, d2)
            return Period("month", d1.strftime("%m.%Y"), d1, d2) if d1.month == d2.month else Period("quarter", f"{d1.year}", d1, d2)
        return None

    # «31 декабря 2025», «1 марта 2024»
    m = re.fullmatch(rf"(\d{{1,2}})\s*{_MONTH_STEM}\s*{_YEAR}", s)
    if m:
        try:
            return period_from_date(date(int(m.group(3)), _month_num(m.group(2)), int(m.group(1))))
        except ValueError:
            return None

    # «За январь–март 2025», «январь-декабрь 2024»
    m = re.fullmatch(rf"(?:за\s*)?{_MONTH_STEM}\s*[-–—]\s*{_MONTH_STEM}\s*{_YEAR}", s)
    if m:
        m1, m2, y = _month_num(m.group(1)), _month_num(m.group(2)), int(m.group(3))
        if m1 and m2 and m1 <= m2:
            return Period("month", f"{m1:02d}–{m2:02d}.{y}", date(y, m1, 1),
                          date(y, m2, calendar.monthrange(y, m2)[1]))
        return None

    # «за 3 месяца 2025», «12 мес. 2024», «9М 2024», «9m2024»
    m = re.fullmatch(rf"(?:за\s*)?(\d{{1,2}})\s*(?:мес[а-я]*\.?|м|m)\s*{_YEAR}", s)
    if m:
        n, y = int(m.group(1)), int(m.group(2))
        return _ytd_period(y, n) if 1 <= n <= 12 else None

    # «1 полугодие 2025», «1 п/г 2025», «1H2025», «H1 2025»
    m = re.fullmatch(rf"(?:за\s*)?([12])\s*(?:полугоди[а-я]*|п/г|h)\s*{_YEAR}", s) \
        or re.fullmatch(rf"h([12])\s*'?\s*{_YEAR}", s)
    if m:
        h, y = int(m.group(1)), int(m.group(2))
        return _ytd_period(y, 6) if h == 1 else Period("halfyear", f"2 полугодие {y}", date(y, 1, 1), date(y, 12, 31))

    # год: 2024, FY2024, «за 2024»
    m = re.fullmatch(rf"(?:fy|за)?\s*{_YEAR}", s)
    if m:
        y = int(m.group(1))
        return Period("year", str(y), date(y, 1, 1), date(y, 12, 31))

    # квартал: '1 кв 2024', 'Q1 2024', "Q1'24", 'I квартал 2024', '2024 Q3', 'кв.3 2025', '1кв2025'
    quarter_patterns = [
        rf"([1-4])\s*[-.\s]*(?:кв|квартал|q)[а-я]*\.?[-.\s]*{_YEAR}",
        rf"{_YEAR}\s*[-.\s]*(?:кв|квартал|q)[а-я]*\.?[-.\s]*([1-4])",
        rf"(?:кв|q)[а-я]*\.?\s*([1-4])\s*[-.\s']*(?:{_YEAR}|(\d\d))",
        rf"(i{{1,3}}|iv)\s*[-.\s]*(?:кв|квартал)[а-я]*\.?[-.\s]*{_YEAR}",
        rf"{_YEAR}\s*[-.\s]*([1-4])\s*кв",
    ]
    for pat in quarter_patterns:
        m = re.fullmatch(pat, s)
        if not m:
            continue
        groups = [g for g in m.groups() if g]
        a, b = groups[0], groups[1]
        if a.isdigit() and len(a) == 4:      # '2024 Q3'
            y, q = int(a), int(b)
        elif a.isdigit():                    # '1 кв. 2024', "Q1'24"
            q, y = int(a), int(b) if len(b) == 4 else 2000 + int(b)
        else:                                # римские
            q, y = _ROMAN.get(a, 0), int(b)
        return _quarter_period(y, q) if 1 <= q <= 4 else None

    # месяц: 'январь 2024', 'янв. 24', 'сент 2025', '01.2024', '1/2024', '2024-01'
    m = re.fullmatch(rf"{_MONTH_STEM}\s*(20[0-4]\d|\d\d)", s)
    if m:
        y = int(m.group(2))
        return _month_period(y + 2000 if y < 100 else y, _month_num(m.group(1)))
    m = re.fullmatch(rf"(0?[1-9]|1[0-2])\s*[./]\s*{_YEAR}", s) or re.fullmatch(rf"{_YEAR}-(0?[1-9]|1[0-2])", s)
    if m:
        a, b = m.group(1), m.group(2)
        num, y = (int(b), int(a)) if len(a) == 4 else (int(a), int(b))
        return _month_period(y, num)

    # одиночная дата
    d = _parse_date(s)
    return period_from_date(d) if d else None


def is_calendar_date(v: object) -> bool:
    """Ячейка — конкретная дата (объект date или «15.01.2026»), а не метка
    периода: по этому выписка (даты операций) отличается от таблицы с месяцами."""
    if isinstance(v, (datetime, date)):
        return True
    if not isinstance(v, str):
        return False
    return _parse_date(unicodedata.normalize("NFKC", v).strip()) is not None


def _parse_date(s: str) -> date | None:
    for fmt in ("%d.%m.%Y", "%d/%m/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(s.strip(), fmt).date()
        except ValueError:
            continue
    return None


# порядок важен: балансовые слова («активы», «обязательства») сильнее
# доходно-расходных — «отложенные налоговые активы» это актив, а не расход по налогу
_KIND_KEYWORDS = {
    "asset": ("актив", "денежн", "дебитор", "запас", "основные средств", "имущество", "вложени"),
    "liability": ("кредиторск", "обязательств", "пассив", "займ", "заемн", "долг", "кредит"),
    "expense": ("расход", "затрат", "себестоим", "аренда", "зарплат", "оплат",
                "налог", "амортиз", "закуп", "коммунал", "процент уплач", "логистик", "платеж"),
    "revenue": ("выручк", "доход", "продаж", "поступлен", "оборот"),  # после expense: «себестоимость продаж» — расход
    # реквизиты: числа-идентификаторы, а не финансовые показатели — исключаются
    # из подбора ответов, чтобы «ИНН: 7949…» не отвечал на вопрос о выручке
    "identifier": ("инн", "кпп", "бик", "огрн", "окпо", "р/с", "расчётн", "расчетн",
                   "счёт №", "счет №", "номер счёт", "лицев счёт", "договор №", "номер договор"),
}


def _guess_kind(name: str) -> str:
    low = name.casefold()
    for kind, stems in _KIND_KEYWORDS.items():
        if any(stem in low for stem in stems):
            return kind
    return "other"


# ------------------------------------------------------------------ текст ---

_UNIT_SUFFIX_RE = re.compile(
    r"[\s,;]*\(?\s*(?:в\s+)?(?:тыс|млн|млрд)\.?\s*(?:руб|₽|долл|usd|eur|евро|шт)?\.?\s*\)?\s*$"
    r"|[\s,;]*\(?\s*(?:руб|₽|usd|eur|евро|шт|%)\.?\s*\)?\s*$",
    re.I,
)


def clean_metric_name(raw: object) -> str:
    if raw is None:
        return ""
    s = unicodedata.normalize("NFKC", str(raw)).replace("\n", " ").strip()
    s = re.sub(r"\s*[–—]\s*", " - ", s)  # «Платежи – всего» и «Платежи - всего» — одно имя
    s = re.sub(r"\s+", " ", s).strip(" :;")
    # сноска БФО: «Уставный капитал5», «Чистая прибыль (убыток)4»
    s = re.sub(r"(?<=[а-яёa-z)])\d{1,2}$", "", s)
    return s


def strip_unit_suffix(name: str) -> str:
    """«Выручка, тыс. руб.» -> «Выручка»: единица уходит в detect_unit,
    а в словаре остаётся чистое имя показателя (иначе «выручка» не находится)."""
    stripped = _UNIT_SUFFIX_RE.sub("", name).strip(" ,;:")
    return stripped or name


def split_chunks(text: str, max_words: int = 300, overlap_words: int | None = None) -> list[str]:
    """Абзацеориентированное разбиение на смысловые фрагменты с перекрытием:
    последние ~1/6 слов чанка переносятся в начало следующего, чтобы мысль,
    разрезанная границей, осталась доступной хотя бы в одном фрагменте."""
    overlap_words = max_words // 6 if overlap_words is None else max(0, overlap_words)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n|\r\n\s*\r\n", text) if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    count = 0
    for para in paragraphs:
        words = para.split()
        if count + len(words) > max_words and current:
            chunks.append("\n\n".join(current))
            if overlap_words:
                tail = " ".join(" ".join(current).split()[-overlap_words:])
                current, count = [tail], len(tail.split())
            else:
                current, count = [], 0
        current.append(para)
        count += len(words)
    if current:
        chunks.append("\n\n".join(current))
    # перекрытие бессмысленно, если чанк один
    return chunks


_TRANSLIT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e", "ж": "zh",
    "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o",
    "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f", "х": "h", "ц": "ts",
    "ч": "ch", "ш": "sh", "щ": "sch", "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
}


def slugify_code(name: str, prefix: str = "m") -> str:
    s = name.casefold()
    s = "".join(_TRANSLIT.get(ch, ch) for ch in s)
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    s = re.sub(r"_+", "_", s)[:60]
    return s or f"{prefix}_{abs(hash(name)) % 100000}"
