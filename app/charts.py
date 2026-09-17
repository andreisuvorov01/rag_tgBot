"""Графики для ответов бота: прогноз, состав, сравнение, рейтинг.

matplotlib рендерит в Agg (без GUI) в PNG-байты; при недоступности библиотеки
возвращается None — бот просто не отправляет картинку. Все графики в одном
стиле: белый фон, без рамки сверху/справа, подписи значений у точек и
столбцов, суммы в тыс./млн/млрд.
"""
from __future__ import annotations

import io
import logging
from typing import Any

log = logging.getLogger(__name__)

BLUE, ORANGE, GREEN, RED, GREY = "#2563eb", "#f97316", "#16a34a", "#dc2626", "#94a3b8"
PALETTE = ["#2563eb", "#f97316", "#16a34a", "#8b5cf6", "#ec4899", "#14b8a6", "#eab308",
           "#64748b", "#ef4444", "#0ea5e9", "#a3e635", "#f472b6"]
_CUR = {"RUB": "₽", "USD": "$", "EUR": "€"}


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#cbd5e1", "axes.labelcolor": "#334155",
        "xtick.color": "#475569", "ytick.color": "#475569",
        "grid.color": "#e2e8f0", "grid.linewidth": 0.8,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })
    return plt


def _scale(values: list[float]) -> tuple[float, str]:
    peak = max((abs(v) for v in values), default=0.0)
    if peak >= 1e9:
        return 1e9, "млрд"
    if peak >= 1e6:
        return 1e6, "млн"
    if peak >= 1e3:
        return 1e3, "тыс."
    return 1.0, ""


def _fmt(v: float, scale: float, suffix: str, unit: str) -> str:
    x = v / scale
    digits = 0 if scale == 1 and unit == "шт" else (1 if abs(x) >= 100 else 2)
    s = f"{x:,.{digits}f}".replace(",", " ").replace(".", ",")
    return " ".join(p for p in (s, suffix, unit) if p)


def _unit(unit: str | None, currency: str | None) -> str:
    if currency:
        return _CUR.get(currency.upper(), currency)
    return unit or ""


def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return buf.getvalue()


def _guard(fn):
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # график — приятное дополнение, не повод ронять ответ
            log.warning("График не построен (%s): %s", fn.__name__, e)
            return None
    return wrapper


# ------------------------------------------------------------------ прогноз ---

@_guard
def render_forecast_chart(history: list[dict[str, Any]], forecast: dict[str, Any],
                          currency: str | None = None, unit: str | None = None) -> bytes | None:
    """history: [{'label','value'}] одной периодичности с горизонтом прогноза;
    forecast: {target, base, low, high}. История и прогноз — одна линия
    (сплошная -> пунктир), интервал — заливка у точки прогноза."""
    plt = _plt()
    labels = [str(h["label"]) for h in history]
    values = [float(h["value"]) for h in history]
    if not labels:
        return None
    base = float(forecast.get("base") or 0)
    low, high = float(forecast.get("low") or base), float(forecast.get("high") or base)
    target = str(forecast.get("target") or "прогноз")
    scale, suffix = _scale(values + [base, high])
    u = _unit(unit, currency)
    xs = list(range(len(labels) + 1))

    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    ax.plot(xs[:-1], [v / scale for v in values], color=BLUE, linewidth=2.2, marker="o",
            markersize=5, label="Факт", zorder=3)
    ax.plot(xs[-2:], [values[-1] / scale, base / scale], color=ORANGE, linewidth=2.2,
            linestyle=(0, (4, 2)), marker="s", markersize=6, label="Прогноз", zorder=3)
    ax.fill_between(xs[-2:], [values[-1] / scale, low / scale], [values[-1] / scale, high / scale],
                    color=ORANGE, alpha=0.18, linewidth=0, label="Интервал 80 %")
    ax.vlines(xs[-1], low / scale, high / scale, color=ORANGE, alpha=0.6, linewidth=1.2)
    # подписи точек: при длинном ряде — только максимум, минимум и последняя
    shown = set(range(len(values))) if len(values) <= 8 else {
        values.index(max(values)), values.index(min(values)), len(values) - 1,
    }
    for x, v in zip(xs[:-1], values, strict=True):
        if x in shown:
            ax.annotate(_fmt(v, scale, "", ""), (x, v / scale), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=8, color="#334155")
    ax.annotate(_fmt(base, scale, suffix, u), (xs[-1], high / scale), textcoords="offset points",
                xytext=(0, 8), ha="center", va="bottom", fontsize=9, fontweight="bold", color="#c2410c")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels + [target], rotation=30 if len(xs) > 8 else 0, ha="right" if len(xs) > 8 else "center")
    ax.set_xlim(-0.4, len(xs) - 0.3)
    top = max(values + [high, base]) / scale
    ax.set_ylim(0, top * 1.22)  # от нуля (масштаб не преувеличивает колебания), запас под подпись
    ax.set_ylabel(f"{suffix} {u}".strip())
    ax.grid(True, axis="y")
    ax.set_title(f"Прогноз на {target}")
    ax.legend(frameon=False, loc="lower left")
    return _png(fig)


# ------------------------------------------------------------------- состав ---

@_guard
def render_breakdown_chart(title: str, items: list[dict[str, Any]], total: float | None = None,
                           currency: str | None = None, unit: str | None = None) -> bytes | None:
    """Горизонтальные столбцы по убыванию с суммой и долей — состав раздела,
    категории журнала, месяцы по убыванию."""
    plt = _plt()
    items = [i for i in items if i.get("value") is not None][:12]
    if not items:
        return None
    names = [str(i.get("name", ""))[:28] for i in items][::-1]
    values = [float(i["value"]) for i in items][::-1]
    shares = [i.get("share_pct") for i in items][::-1]
    scale, suffix = _scale(values)
    u = _unit(unit, currency)

    fig, ax = plt.subplots(figsize=(7.4, 0.42 * len(items) + 1.4))
    bars = ax.barh(names, [v / scale for v in values], color=BLUE, height=0.62, zorder=3)
    bars[-1].set_color(ORANGE)  # лидер
    xmax = max(v / scale for v in values)
    for bar, v, sh in zip(bars, values, shares, strict=True):
        label = _fmt(v, scale, "", "")
        if sh is not None:
            label += f"  ·  {float(sh):.1f} %".replace(".", ",")
        ax.text(bar.get_width() + xmax * 0.015, bar.get_y() + bar.get_height() / 2, label,
                va="center", fontsize=8, color="#334155")
    ax.set_xlim(0, xmax * 1.35)
    ax.set_xlabel(f"{suffix} {u}".strip())
    ax.grid(True, axis="x")
    ax.tick_params(axis="y", length=0)
    sub = f"итого {_fmt(float(total), *_scale([float(total)]), u)}" if total else ""
    ax.set_title(f"{title}\n{sub}" if sub else title)
    return _png(fig)


# ---------------------------------------------------------------- сравнение ---

@_guard
def render_compare_chart(title: str, history: list[dict[str, Any]], currency: str | None = None,
                         unit: str | None = None, change_pct: float | None = None) -> bytes | None:
    """Два-шесть столбцов (периоды или два показателя) с подписями и стрелкой изменения."""
    plt = _plt()
    labels = [str(h.get("label", "")) for h in history][:8]
    values = [float(h.get("value") or 0) for h in history][:8]
    if len(values) < 2:
        return None
    scale, suffix = _scale(values)
    u = _unit(unit, currency)
    colors = [GREY] * (len(values) - 1) + [GREEN if values[-1] >= values[0] else RED]
    if len(values) == 2:
        colors = [BLUE, ORANGE]

    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    bars = ax.bar(labels, [v / scale for v in values], color=colors, width=0.55, zorder=3)
    for bar, v in zip(bars, values, strict=True):
        ax.annotate(_fmt(v, scale, "", ""), (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    textcoords="offset points", xytext=(0, 5), ha="center", fontsize=9, color="#334155")
    if change_pct is not None:
        sign = "+" if change_pct >= 0 else "−"
        ax.annotate(f"{sign}{abs(change_pct):.1f} %".replace(".", ","),
                    (len(values) - 1, values[-1] / scale), textcoords="offset points", xytext=(0, 22),
                    ha="center", fontsize=11, fontweight="bold", color=GREEN if change_pct >= 0 else RED)
    ax.set_ylabel(f"{suffix} {u}".strip())
    ax.grid(True, axis="y")
    ax.set_title(title)
    ax.margins(y=0.3)
    return _png(fig)


# ------------------------------------------------------------------ рейтинг ---

@_guard
def render_rank_chart(title: str, ranking: list[dict[str, Any]]) -> bytes | None:
    """Темп роста по позициям: зелёные столбцы вправо, красные влево."""
    plt = _plt()
    rows = [r for r in ranking if r.get("growth_pct") is not None][:12]
    if not rows:
        return None
    names = [str(r.get("name", ""))[:28] for r in rows][::-1]
    pct = [float(r["growth_pct"]) for r in rows][::-1]

    first, last = rows[0].get("label_from"), rows[0].get("label_to")
    if first and last and all(r.get("label_from") == first and r.get("label_to") == last for r in rows):
        title = f"{title}: {first} → {last}"
    fig, ax = plt.subplots(figsize=(7.4, 0.42 * len(rows) + 1.4))
    bars = ax.barh(names, pct, color=[GREEN if p >= 0 else RED for p in pct], height=0.62, zorder=3)
    span = max(abs(p) for p in pct) or 1.0
    for bar, p in zip(bars, pct, strict=True):
        x = bar.get_width()
        ax.text(x + (span * 0.02 if p >= 0 else -span * 0.02), bar.get_y() + bar.get_height() / 2,
                f"{'+' if p >= 0 else '−'}{abs(p):.1f} %".replace(".", ","),
                va="center", ha="left" if p >= 0 else "right", fontsize=8, color="#334155")
    ax.axvline(0, color="#94a3b8", linewidth=1)
    lo, hi = min(pct + [0.0]), max(pct + [0.0])
    ax.set_xlim(lo - span * 0.35 if lo < 0 else 0, hi + span * 0.35)
    ax.set_xlabel("изменение, %")
    ax.grid(True, axis="x")
    ax.tick_params(axis="y", length=0)
    ax.set_title(title)
    return _png(fig)
