"""Графики для ответов бота: история + прогноз с доверительным интервалом.
matplotlib рендерит в Agg (без GUI) в PNG-байты; при недоступности библиотеки
возвращается None — бот просто не отправляет картинку."""
from __future__ import annotations

import io
import logging
from typing import Any

log = logging.getLogger(__name__)


def render_forecast_chart(history: list[dict[str, Any]], forecast: dict[str, Any],
                          currency: str | None = None) -> bytes | None:
    """history: [{'label','value'}]; forecast: {target, base, low, high}."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = [str(h["label"]) for h in history]
        values = [float(h["value"]) for h in history]
        if not labels:
            return None

        # крупные значения показываем в миллиардах/миллионах
        scale, suffix = 1.0, ""
        peak = max(values + [float(forecast.get("base") or 0)])
        if peak >= 1e9:
            scale, suffix = 1e9, " млрд"
        elif peak >= 1e6:
            scale, suffix = 1e6, " млн"
        elif peak >= 1e3:
            scale, suffix = 1e3, " тыс"

        fig, ax = plt.subplots(figsize=(7.0, 3.2), dpi=140)
        ax.plot(labels, [v / scale for v in values], marker="o", linewidth=2, label="История", color="#1f77b4")

        base = float(forecast.get("base") or 0)
        low, high = float(forecast.get("low") or base), float(forecast.get("high") or base)
        target = str(forecast.get("target") or "прогноз")
        ax.plot([labels[-1], target], [values[-1] / scale, base / scale],
                marker="s", linestyle="--", linewidth=2, color="#ff7f0e", label="Прогноз")
        ax.fill_between([labels[-1], target],
                        [values[-1] / scale, low / scale],
                        [values[-1] / scale, high / scale],
                        alpha=0.2, color="#ff7f0e", label="Интервал 80%")
        ax.annotate(f"{base / scale:,.2f}{suffix}".replace(",", " ").replace(".", ","),
                    xy=(len(labels), base / scale), xytext=(-6, 8), ha="right",
                    textcoords="offset points", color="#b35a00", fontsize=9)

        cur = {"RUB": "₽", "USD": "$", "EUR": "€"}.get((currency or "").upper(), "")
        ax.set_title(f"Прогноз на {target}", fontsize=11)
        ax.set_ylabel(f"значение{',' if suffix else ''} {suffix.strip() if suffix else ''} {cur}".strip(), fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")
        ax.tick_params(axis="x", labelsize=8)
        if len(labels) > 8:
            # много точек — показываем каждую k-ю подпись, последнюю и прогноз всегда
            k = -(-len(labels) // 8)
            ticks = [i for i in range(len(labels)) if i % k == 0 or i == len(labels) - 1] + [len(labels)]
            ax.set_xticks(ticks)
            ax.set_xticklabels([labels[i] if i < len(labels) else target for i in ticks], rotation=30, ha="right")
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        return buf.getvalue()
    except Exception as e:
        log.warning("График не построен: %s", e)
        return None
