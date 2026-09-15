"""Экспорт данных показателя в Excel (xlsx в памяти, отправляется документом)."""
from __future__ import annotations

import io
from typing import Any

from openpyxl import Workbook


def metric_rows_to_xlsx(title: str, rows: list[dict[str, Any]]) -> bytes:
    """rows — результат storage.series_for_metric: period_label/value/unit/
    currency/document_name/sheet/cell_ref."""
    wb = Workbook()
    ws = wb.active
    ws.title = (title or "Показатель")[:31]
    ws.append(["Период", "Значение", "Единица", "Валюта", "Документ", "Лист", "Ячейка"])
    for r in rows:
        ws.append([
            r.get("period_label"),
            r.get("value"),
            r.get("unit"),
            r.get("currency"),
            r.get("document_name"),
            r.get("sheet"),
            r.get("cell_ref"),
        ])
    ws.column_dimensions["A"].width = 14
    ws.column_dimensions["B"].width = 16
    ws.column_dimensions["E"].width = 40
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()
