"""Отчёт «Личные расходы из бюджета компании» в Excel — без Telegram.

Собирает файл из журнала расходов (записи вида «расход: 1500 кофе») тем же
кодом, что и команда /report в боте. Оригиналы загруженных отчётов компании
не изменяются: файл каждый раз строится из базы.

Запуск:
    python -m scripts.expenses_report                  # в data/личные_расходы_ГГГГ-ММ.xlsx
    python -m scripts.expenses_report --out C:\\отчёт.xlsx
    python -m scripts.expenses_report --list           # только показать записи журнала
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    _reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(_reconfigure):
        _reconfigure(encoding="utf-8")

from app.config import settings  # noqa: E402
from app.export import expenses_report_filename, expenses_report_xlsx  # noqa: E402
from app.formatting import fmt_money  # noqa: E402


async def main() -> int:
    ap = argparse.ArgumentParser(description="Отчёт по личным расходам в Excel")
    ap.add_argument("--out", help="путь к файлу (по умолчанию — рядом с данными)")
    ap.add_argument("--org", type=int, default=1, help="id организации (по умолчанию 1)")
    ap.add_argument("--list", action="store_true", help="показать записи журнала и не создавать файл")
    args = ap.parse_args()

    from app.expenses import ledger_rows
    from app.storage import Organization, make_engine, make_sessionmaker

    engine = await make_engine(settings)
    sessions = make_sessionmaker(engine)
    try:
        async with sessions() as s:
            rows = await ledger_rows(s, settings, args.org)
            org = await s.get(Organization, args.org)
            org_name = org.name if org else ""
    finally:
        await engine.dispose()

    if not rows:
        print("Журнал пуст: записей вида «расход: 1500 кофе» в базе нет.")
        print("Проверьте, что смотрите ту же БД (DATABASE_URL в .env) и организацию (--org).")
        return 1

    expenses = [r for r in rows if r["kind"] == "expense"]
    incomes = [r for r in rows if r["kind"] == "income"]
    total_exp = sum(r["amount"] for r in expenses)
    total_inc = sum(r["amount"] for r in incomes)
    print(f"Записей в журнале: {len(rows)} (расходов {len(expenses)}, доходов {len(incomes)})")
    print(f"Расходы: {fmt_money(total_exp)} | доходы: {fmt_money(total_inc)} | "
          f"сальдо: {fmt_money(total_exp - total_inc)}")

    if args.list:
        for r in rows:
            kind = "доход" if r["kind"] == "income" else "расход"
            print(f"  {r['when']:%d.%m.%Y}  {kind:7} {fmt_money(r['amount']):>14}  "
                  f"{r['category']:20} {r['description']}")
        return 0

    today = date.today()
    data = await asyncio.to_thread(expenses_report_xlsx, rows,
                                   org_name=org_name, today=today)
    out = Path(args.out) if args.out else settings.data_dir / expenses_report_filename(today)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        out.write_bytes(data)
    except PermissionError:
        print(f"Не удалось записать {out}: файл открыт в Excel. Закройте его и повторите.")
        return 1
    months = sorted({(r["when"].year, r["when"].month) for r in rows})
    print(f"\nФайл отчёта: {out}")
    print(f"  лист «Отчёт»: {len(months)} мес. + колонка «Личные расходы» и структура по категориям")
    print("  лист «По категориям»: категории × месяцы")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
