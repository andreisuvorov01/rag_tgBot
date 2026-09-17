"""Аналитический движок: подготовка рядов, статистика, прогнозный ансамбль.

Принцип системы: прогноз считает код, а не LLM. Методы выбираются по объёму
данных, взвешиваются по ошибке обратной проверки (backtest), выдаётся
доверительный интервал и честная оценка уверенности.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date

import numpy as np

log = logging.getLogger(__name__)

Z80 = 1.28  # квантиль нормального распределения для 80% интервала
_FALLBACK_REL_ERR = 0.15


@dataclass
class SeriesPoint:
    label: str
    start: date
    end: date
    value: float
    source: str
    ptype: str


def prepare_series(rows: list[dict]) -> list[SeriesPoint]:
    """rows — результат storage.series_for_metric. Оставляем доминирующую
    периодичность (в отчётах часто смешаны годовые и квартальные данные)."""
    if not rows:
        return []
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["period_type"]] = counts.get(r["period_type"], 0) + 1
    dominant = max(counts, key=counts.get)
    points = [
        SeriesPoint(
            label=r["period_label"],
            start=r["period_start"],
            end=r["period_end"],
            value=float(r["value"]),
            source=f"{r['document_name']} · {r['sheet']} · {r['cell_ref']}",
            ptype=r["period_type"],
        )
        for r in rows
        if r["period_type"] == dominant
    ]
    points.sort(key=lambda p: p.end)
    return points


def _growth_rates(values: np.ndarray) -> np.ndarray:
    prev = values[:-1]
    cur = values[1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        rates = np.where(prev != 0, cur / np.where(prev == 0, np.nan, prev) - 1.0, np.nan)
    return rates[~np.isnan(rates)]


def _predict_path(name: str, values: np.ndarray, steps: int, season: int = 1) -> np.ndarray | None:
    """Прогноз по шагам: массив длины `steps`. season — длина сезонного цикла."""
    n = len(values)
    last = float(values[-1])
    if name == "naive":
        return np.full(steps, last)
    if name == "seasonal_naive":
        # значение того же месяца/квартала прошлого цикла — сильный бейзлайн
        if season < 2 or n < season:
            return None
        return np.array([float(values[-season + ((i - 1) % season)]) for i in range(1, steps + 1)])
    if name == "mean_growth":
        rates = _growth_rates(values)
        if rates.size == 0:
            return None
        g = float(np.prod(1.0 + rates) ** (1.0 / rates.size) - 1.0)
        return np.array([last * (1.0 + g) ** i for i in range(1, steps + 1)])
    if name == "linear_trend":
        if n < 2:
            return None
        x = np.arange(n, dtype=float)
        slope, intercept = np.polyfit(x, values, 1)
        xs = np.arange(n, n + steps, dtype=float)
        return slope * xs + intercept
    if name == "holt":
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing

            model = ExponentialSmoothing(values, trend="add", damped_trend=True).fit()
            return np.asarray(model.forecast(steps), dtype=float)
        except Exception as e:  # statsmodels может отсутствовать или не сойтись
            log.debug("Holt skipped: %s", e)
            return None
    return None


def _predict_method(name: str, values: np.ndarray, steps: int, season: int = 1, window: int | None = None) -> float | None:
    """Сумма прогноза за последние `window` шагов из `steps` (окно целевого года).

    «Прогноз на 2027» по месяцам, кончающимся сентябрём 2026, — это 15 шагов
    вперёд, из которых считаются только 12 месяцев 2027 года. Раньше в ответ
    уходила сумма всех 15 (а для годовых рядов — сумма 2026 и 2027 вместе)."""
    path = _predict_path(name, values, steps, season)
    if path is None:
        return None
    window = steps if window is None else min(max(window, 1), steps)
    return float(np.sum(path[steps - window:]))


def _window_start(last_end: date, ptype: str) -> date:
    """Дата начала первого прогнозного периода (шаг вперёд от последней точки)."""
    if ptype == "year":
        return date(last_end.year + 1, 1, 1)
    if ptype == "quarter":
        m = last_end.month + 3
        y = last_end.year + (m - 1) // 12
        return date(y, (m - 1) % 12 + 1, 1)
    m = last_end.month + 1
    y = last_end.year + (m - 1) // 12
    return date(y, (m - 1) % 12 + 1, 1)


def _forecast_window(ptype: str, last_end: date, target_year: int | None, steps: int) -> int:
    """Сколько последних шагов горизонта относятся к целевому году: полный год
    (12 месяцев / 4 квартала / 1 год) либо остаток текущего года, если данные
    кончаются внутри него."""
    if target_year is None:
        return 1
    per_year = {"year": 1, "quarter": 4}.get(ptype, 12)
    return min(per_year, steps) if target_year > last_end.year else steps


def _steps_for_target(ptype: str, last_end: date, target_year: int | None) -> int:
    """Число периодов от конца данных до конца target_year.

    Для месячного ряда, заканчивающегося мартом 2024, прогноз «на 2026» —
    это 9 месяцев 2024, 12 месяцев 2025 и 12 месяцев 2026 = 33 шага.
    Прежний расчёт по разнице лет давал 24 и молча терял часть окна.
    """
    if target_year is None:
        return 1
    start = _window_start(last_end, ptype)
    per_year = {"year": 1, "quarter": 4}.get(ptype, 12)
    if ptype == "year":
        return max(target_year - start.year + 1, 1)
    months = (target_year - start.year) * 12 + (12 - start.month) + 1
    return max(round(months / (12 / per_year)), 1)


def metric_stats(points: list[SeriesPoint]) -> dict:
    values = np.array([p.value for p in points], dtype=float)
    first, last = float(values[0]), float(values[-1])
    rates = _growth_rates(values)
    gaps = [(points[i + 1].end - points[i].end).days for i in range(len(points) - 1)]
    period_days = float(np.median(gaps)) if gaps else 365.0
    span_years = max(period_days * (len(points) - 1) / 365.25, 1e-9)
    cagr = (last / first) ** (1.0 / span_years) - 1.0 if first > 0 and last > 0 else None
    return {
        "first": first,
        "last": last,
        "first_label": points[0].label,
        "last_label": points[-1].label,
        "n_points": len(points),
        "avg_growth_pct": float(np.mean(rates)) * 100 if rates.size else None,
        "cagr_pct": cagr * 100 if cagr is not None else None,
    }


def forecast(
    points: list[SeriesPoint],
    *,
    target_year: int | None = None,
    growth_multiplier: float | None = None,
) -> dict:
    """Возвращает блок forecast + computed для payload композитора."""
    if len(points) < 2:
        return {"target": None, "error": "недостаточно данных: нужно минимум 2 периода"}
    ptype = points[0].ptype
    values = np.array([p.value for p in points], dtype=float)
    n = len(values)
    last_end = points[-1].end
    steps = _steps_for_target(ptype, last_end, target_year)
    window = _forecast_window(ptype, last_end, target_year, steps)
    target_label = str(target_year) if target_year else _next_label(points[-1])
    season = {"month": 12, "quarter": 4}.get(ptype, 1)
    # сопоставимый факт: для годового окна — сумма последних `window` точек
    # (последние 12 месяцев), для одной точки — она сама
    reference = float(np.sum(values[-window:])) if window > 1 else float(values[-1])

    method_names = ["naive", "mean_growth", "linear_trend"]
    if n >= 8:
        method_names.append("holt")
    if season > 1 and n >= season * 2:
        # сезонный наивный — обязательный бейзлайн для регулярных рядов
        method_names.append("seasonal_naive")
    preds, errors = {}, {}
    for name in method_names:
        pred = _predict_method(name, values, steps, season, window)
        if pred is None or not math.isfinite(pred):
            continue
        preds[name] = pred
        if n >= 3:
            # многоскладочный walk-forward бэктест (практика Nixtla/sktime):
            # до 3 складов. Горизонт склада — окно основного прогноза (год
            # или его остаток) — иначе «ошибка» относится к другому окну и веса
            # ансамбля подбираются не под ту задачу, которую решаем.
            fold_errs = []
            for f in range(window, window + 3):
                train = values[: n - f]
                if len(train) < 2:
                    continue  # тренд/темпы на одной точке не строятся
                actual = float(np.sum(values[n - f: n - f + window]))
                bt = _predict_method(name, train, window, season)
                if bt is not None and math.isfinite(bt):
                    fold_errs.append(abs(bt - actual) / max(abs(actual), 1e-9))
            if fold_errs:
                errors[name] = sum(fold_errs) / len(fold_errs)
        if name == "naive" and steps > 1:
            # «плоский» наивный на длинном горизонте — слабое допущение
            errors.setdefault(name, errors.get(name, 0.5))

    if not preds:
        return {"target": target_label, "error": "не удалось построить ни одной модели прогноза"}

    if errors:
        # Не у каждого метода есть ошибка бэктеста: короткий ряд (n < 3) или
        # несошедшиеся фолды. Таким методам — пессимистичный вес (худшая из
        # известных ошибок), а не KeyError. Регрессия: 2 точки выручки
        # + горизонт 2+ года -> errors = {naive: 0.5}, preds = 3 метода.
        default_err = max(max(errors.values()), _FALLBACK_REL_ERR)
        weights = {m: 1.0 / (errors.get(m, default_err) + 0.05) for m in preds}
    else:
        weights = {m: 1.0 for m in preds}
    wsum = sum(weights.values())
    weights = {m: w / wsum for m, w in weights.items()}
    base = sum(weights[m] * preds[m] for m in preds)

    rel_err: float
    if errors:
        covered = sum(weights[m] for m in preds if m in errors) or 1.0
        rel_err = float(sum(weights[m] * errors[m] for m in preds if m in errors) / covered)
    else:
        rel_err = _FALLBACK_REL_ERR

    base_wo = base
    if growth_multiplier is not None:
        # «что если» масштабирует ПРИРОСТ относительно сопоставимого факта
        # (последние 12 месяцев для годового окна), а не подмешивает прогноз
        # к последнему значению. Прежняя формула last + (base - last) * mult
        # при mult=0.5 меняла результат лишь на десятую часть.
        base = reference + (base - reference) * growth_multiplier

    # Полуширина считается от |base|: при отрицательной базе (крутой спад)
    # прежняя формула давала вывернутый интервал «0,00 — -88,76 млрд».
    half = abs(base) * rel_err * Z80
    if ptype == "year" or steps == 1:
        half *= math.sqrt(max(steps, 1))
    low, high = base - half, base + half
    if base > 0:
        low = max(low, 0.0)  # положительный ряд не уходит ниже нуля

    if n >= 12 and rel_err <= 0.07:
        confidence = "высокая"
    elif n >= 4 and rel_err <= 0.2:
        confidence = "средняя"
    else:
        confidence = "низкая"

    notes = []
    if n < 4:
        notes.append(f"короткий ряд ({n} точек) — интервал широкий")
    if rel_err > 0.2:
        notes.append(f"высокая ошибка на исторической проверке ({rel_err*100:.0f}%)")
    if any(v <= 0 for v in values):
        notes.append("ряд содержит нулевые/отрицательные значения — темповые модели ограничены")
    if base < 0 < float(values[-1]):
        notes.append("ансамбль ушёл в минус (резкий спад при коротком ряде) — отнеситесь к прогнозу осторожно")

    # данные кончаются внутри целевого года: ответ — весь год, факт + прогноз остатка
    fact_to_date = 0.0
    if target_year is not None and ptype != "year" and target_year == last_end.year:
        fact_to_date = float(np.sum([p.value for p in points if p.start.year == target_year]))
        months_fact = sum(1 for p in points if p.start.year == target_year)
        notes.append(
            f"в {target_year} году уже есть факт за {months_fact} {'мес.' if ptype == 'month' else 'кв.'}; "
            f"прогноз — на остаток года, итог за год = факт + прогноз"
        )

    stats = metric_stats(points)
    if "linear_trend" in preds:
        stats["trend_value"] = float(preds["linear_trend"]) + fact_to_date
    methods_fmt = {m: float(preds[m]) + fact_to_date for m in preds}
    return {
        "target": target_label,
        "base": float(base) + fact_to_date,
        "low": float(low) + fact_to_date,
        "high": float(high) + fact_to_date,
        "reference": reference,
        "reference_label": (
            f"последние {window} мес." if ptype == "month" and window > 1
            else f"последние {window} кв." if ptype == "quarter" and window > 1
            else points[-1].label
        ),
        "fact_to_date": fact_to_date or None,
        "rest_forecast": float(base) if fact_to_date else None,
        "methods": methods_fmt,
        "weights": {m: round(w, 2) for m, w in weights.items()},
        "backtest_error_pct": rel_err * 100,
        "confidence": confidence,
        "scenario": {"growth_multiplier": growth_multiplier} if growth_multiplier else None,
        "base_without_scenario": float(base_wo) + fact_to_date if growth_multiplier else None,
        "notes": notes,
        "computed": stats,
    }


def _next_label(last_point: SeriesPoint) -> str:
    nxt = _window_start(last_point.end, last_point.ptype)
    if last_point.ptype == "year":
        return str(nxt.year)
    if last_point.ptype == "quarter":
        return f"{(nxt.month - 1) // 3 + 1} кв. {nxt.year}"
    if last_point.ptype == "month":
        return f"{nxt.month:02d}.{nxt.year}"
    return f"след. период после {last_point.label}"
