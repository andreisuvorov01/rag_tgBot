"""Тесты прогнозного движка: тренды, ансамбль, бэктестинг, сценарии."""
from datetime import date

from app.analytics import forecast, metric_stats, prepare_series


def _points(values: list[float], ptype: str = "year") -> list:
    import calendar as _cal

    points = []
    for i, v in enumerate(values):
        if ptype == "year":
            y, m = 2020 + i, 12
        else:
            y, m = 2020 + i // 12, i % 12 + 1
        last_day = _cal.monthrange(y, m)[1]
        label = str(y) if ptype == "year" else f"{m:02d}.{y}"
        points.append(
            type("P", (), {
                "label": label, "start": date(y, m, 1), "end": date(y, m, last_day),
                "value": v, "source": "test.xlsx · лист · A1", "ptype": ptype,
            })()
        )
    return points


def test_prepare_series_dominant_period():
    rows = [
        {"period_type": "year", "period_label": "2023", "period_start": date(2023, 1, 1),
         "period_end": date(2023, 12, 31), "value": 9.8, "unit": "руб", "currency": "RUB",
         "sheet": "", "cell_ref": "", "document_name": "a.xlsx"},
        {"period_type": "month", "period_label": "01.2024", "period_start": date(2024, 1, 1),
         "period_end": date(2024, 1, 31), "value": 1.0, "unit": "руб", "currency": "RUB",
         "sheet": "", "cell_ref": "", "document_name": "b.xlsx"},
        {"period_type": "year", "period_label": "2024", "period_start": date(2024, 1, 1),
         "period_end": date(2024, 12, 31), "value": 11.4, "unit": "руб", "currency": "RUB",
         "sheet": "", "cell_ref": "", "document_name": "b.xlsx"},
    ]
    points = prepare_series(rows)
    assert len(points) == 2 and points[0].value == 9.8 and points[-1].value == 11.4


def test_stats():
    points = _points([9.8, 11.4, 12.7])
    s = metric_stats(points)
    assert s["n_points"] == 3
    assert abs(s["avg_growth_pct"] - 13.85) < 0.5
    assert abs(s["cagr_pct"] - 13.84) < 0.5


def test_forecast_linear():
    fc = forecast(_points([10.0, 12.0, 14.0]))
    assert fc.get("error") is None
    # линейный тренд: следующий шаг = 16; ансамбль должен быть близок
    assert 14.5 < fc["base"] < 17.5
    assert fc["low"] < fc["base"] < fc["high"]
    assert fc["confidence"] in ("низкая", "средняя", "высокая")
    assert "linear_trend" in fc["methods"]


def test_forecast_scenario_multiplier():
    base = forecast(_points([9.8, 11.4, 12.7]))
    slow = forecast(_points([9.8, 11.4, 12.7]), growth_multiplier=0.5)
    # замедление вдвое должно опустить базу, но оставить выше последнего факта
    assert slow["base"] < base["base"]
    assert slow["base"] > 12.7
    assert slow["scenario"] == {"growth_multiplier": 0.5}


def test_forecast_monthly_year_target():
    points = _points([100.0 + i for i in range(24)], ptype="month")
    fc = forecast(points, target_year=2027)
    assert fc.get("error") is None
    # ряд 100..123 (2020-01..2021-12); 2027 — 72 шага вперёд, но в ответ идёт
    # только окно 2027 года (12 последних шагов): линейный тренд 184..195 ≈ 2274,
    # наивный 123 × 12 = 1476. Раньше сюда уходила сумма всех 72 месяцев (≈ 11 490).
    assert 1400 < fc["base"] < 2400
    assert fc["reference"] == sum(range(112, 124))  # сопоставимый факт — последние 12 месяцев


def test_forecast_year_target_is_one_year_not_cumulative():
    """«Прогноз выручки на 2027» по годам 2023–2025 — значение 2027 года,
    а не сумма 2026 + 2027 (прежнее поведение удваивало ответ)."""
    points = _points([400.0, 470.0, 536.0])  # 2020–2022
    one = forecast(points, target_year=2023)["base"]
    two = forecast(points, target_year=2024)["base"]
    assert 536 < one < 650 and one < two < 800


def test_forecast_current_year_adds_fact_to_date():
    """Данные по сентябрь 2026, «прогноз на 2026» — весь год: факт 9 мес. + прогноз 3 мес."""
    points = _points([100.0] * 9, ptype="month")  # 01.2020–09.2020
    fc = forecast(points, target_year=2020)
    assert fc["fact_to_date"] == 900.0 and fc["rest_forecast"] > 0
    assert abs(fc["base"] - (900.0 + fc["rest_forecast"])) < 1e-6
    assert 1100 < fc["base"] < 1300


def test_forecast_insufficient_data():
    fc = forecast(_points([1.0]))
    assert fc.get("error")


def test_forecast_two_points_multiyear_horizon():
    """Регрессия KeyError 'mean_growth': ряд из 2 точек (выручка 2024-2025),
    горизонт 2+ года. naive получает ошибку 0.5 через setdefault, у остальных
    методов бэктеста нет — веса должны считаться без KeyError."""
    points = _points([657.482, 135.400])  # млрд, как в реальной базе
    fc = forecast(points, target_year=2027)  # 6 шагов вперёд
    assert fc.get("error") is None
    assert set(fc["methods"]) == {"naive", "mean_growth", "linear_trend"}
    # веса распределены по всем методам, сумма = 1 (в выводе округлены до 2 знаков)
    assert abs(sum(fc["weights"].values()) - 1.0) < 0.02
    # интервал не вывернут и база внутри него
    assert fc["low"] <= fc["base"] <= fc["high"]
    assert fc["confidence"] == "низкая"  # 2 точки — честная низкая уверенность
    # ансамбль уходит в минус на положительном ряду — должно быть предупреждение
    assert any("минус" in n for n in fc["notes"])


def test_forecast_negative_base_interval_order():
    """Отрицательная база (резкий спад): low <= high без зануления нижней
    границы — прежний max(..., 0) давал интервал «0,00 — -88,76»."""
    fc = forecast(_points([657.482, 135.400]), target_year=2027)
    assert fc["low"] <= fc["high"]
    assert fc["low"] < 0  # не зажат нулём
    # положительная база: низ не уходит ниже нуля
    fc2 = forecast(_points([10.0, 12.0, 14.0]), target_year=2023)
    assert fc2["low"] >= 0
