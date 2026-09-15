"""Отладка разбора любого файла: показывает, что извлечёт конвейер и почему.

Запуск:  python -m scripts.parse_file <путь к файлу> [--full]
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402
from app.ingest.load_excel import load_csv, load_excel, load_html, load_ods  # noqa: E402
from app.ingest.load_pdf_docx import load_docx, load_image, load_pdf  # noqa: E402
from app.ocr import VlmOcr  # noqa: E402


def main() -> int:
    if len(sys.argv) < 2:
        print("Использование: python -m scripts.parse_file <файл> [--full]")
        return 1
    path = Path(sys.argv[1])
    full = "--full" in sys.argv
    ext = path.suffix.lower().lstrip(".")
    vlm = VlmOcr(settings)

    if ext in ("xlsx", "xls"):
        doc = load_excel(str(path))
    elif ext == "csv":
        doc = load_csv(str(path))
    elif ext == "pdf":
        doc = asyncio.run(load_pdf(str(path), vlm_ocr=vlm))
    elif ext == "docx":
        doc = asyncio.run(load_docx(str(path)))
    elif ext in ("html", "htm"):
        doc = load_html(str(path))
    elif ext == "ods":
        doc = load_ods(str(path))
    elif ext in ("png", "jpg", "jpeg", "tiff"):
        doc = asyncio.run(load_image(str(path), vlm_ocr=vlm))
    else:
        print(f"Формат .{ext} не поддерживается")
        return 1

    print(f"\n=== {path.name} ===")
    print(f"Листы/разделы: {doc.sheets}")
    print(f"Фактов: {len(doc.facts)} | чанков: {len(doc.chunks)} | разделов: {len(doc.sections)}")
    periods = sorted({f.period.label for f in doc.facts})
    if periods:
        print(f"Периоды: {', '.join(periods)}")
    metrics = sorted({f.metric_name for f in doc.facts})
    if metrics:
        print(f"Показатели ({len(metrics)}): {', '.join(metrics[:25])}")
    for d in doc.diagnostics:
        print(f"· {d}")
    if doc.warnings:
        print("⚠️ Замечания:")
        for w in doc.warnings:
            print(f"  - {w}")
    if full:
        for f in doc.facts[:50]:
            print(f"  {f.metric_name} | {f.period.label} | {f.value} | {f.sheet} {f.cell_ref}")
        for c in doc.chunks[:10]:
            print(f"  [чанк {len(c.text)} симв.] {c.text[:120]}...")
    print("\nПодсказка: --full покажет все факты и фрагменты.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
