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
    # ряд 100..123 (2020-01..2021-12); 2027 = 72 шага вперёд,
    # линейный тренд даёт сумму 124..195 ≈ 11490, наивные методы ниже
    assert 9000 < fc["base"] < 12500


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
