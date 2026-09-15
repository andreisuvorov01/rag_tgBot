"""Тесты мультestrатегии парсинга (практики Camelot/Docling/Unstructured):
stream-реконструкция текста, HTML-загрузчик, компактные кварталы, диагностика."""
from app.ingest.load_excel import grid_to_parsed, load_html
from app.ingest.load_pdf_docx import _text_lines_to_grid
from app.ingest.normalize import parse_period


def test_compact_quarter_headers():
    for text, label in (("1кв2025", "1 кв. 2025"), ("2025Q3", "3 кв. 2025")):
        p = parse_period(text)
        assert p is not None and p.ptype == "quarter" and p.label == label, text


def test_bfo_style_headers():
    p = parse_period("На 31 декабря 2025 г.2")
    assert p.ptype == "year" and p.label == "2025"
    p = parse_period("За январь–март 2025 г.")
    assert p.ptype == "month" and p.start.strftime("%d.%m.%Y") == "01.01.2025"
    p = parse_period("за 3 месяца 2025")
    assert p.ptype == "month" and p.end.strftime("%d.%m.%Y") == "31.03.2025"
    p = parse_period("1 полугодие 2025")
    assert p.ptype == "halfyear" and p.end.strftime("%d.%m.%Y") == "30.06.2025"


def test_text_lines_to_grid_whitespace_columns():
    text = (
        "Наименование            2024        2025\n"
        "Выручка                 41 200      45 800\n"
        "Аренда                   9 800      11 400\n"
    )
    grid = _text_lines_to_grid(text)
    assert grid is not None and len(grid) == 3 and len(grid[0]) == 3
    facts, _chunks, warns, _sections = grid_to_parsed(grid, "стр. 1")
    by = {(f.metric_name, f.period.label): f.value for f in facts}
    assert by[("выручка", "2025")] == 45800.0
    assert by[("аренда", "2024")] == 9800.0
    assert warns == []


def test_stream_fallback_edge_cases():
    assert _text_lines_to_grid("") is None
    assert _text_lines_to_grid("одна строка без колонок") is None
    # одна колонка — не таблица
    assert _text_lines_to_grid("строка 1\nстрока 2\nстрока 3") is None


def test_load_html(tmp_path):
    p = tmp_path / "report.html"
    p.write_text(
        """<html><body>
        <table>
          <tr><th>Показатель</th><th>2024</th><th>2025</th></tr>
          <tr><td>Выручка</td><td>41 200</td><td>45 800</td></tr>
          <tr><td>Аренда</td><td>9 800</td><td>11 400</td></tr>
        </table>
        </body></html>""",
        encoding="utf-8",
    )
    doc = load_html(str(p))
    facts = {(f.metric_name, f.period.label): f.value for f in doc.facts}
    assert facts[("выручка", "2025")] == 45800.0
    assert facts[("аренда", "2024")] == 9800.0
    assert doc.diagnostics  # диагностика извлечения заполнена
