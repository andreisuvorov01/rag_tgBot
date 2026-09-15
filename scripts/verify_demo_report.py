"""РџСЂРѕРІРµСЂРєР° РґРµРјРѕ-РѕС‚С‡С‘С‚Р° РЅР° РїРѕР»РЅРѕС†РµРЅРЅРѕРј РєРѕРЅРІРµР№РµСЂРµ: РїР°СЂСЃРёРЅРі -> Р‘Р” -> РІРѕРїСЂРѕСЃС‹.

Р Р°Р±РѕС‚Р°РµС‚ РЅР° РІСЂРµРјРµРЅРЅРѕР№ Р‘Р” (СЂРµР°Р»СЊРЅР°СЏ Р±Р°Р·Р° РЅРµ С‚СЂРѕРіР°РµС‚СЃСЏ), СЃ РЅР°СЃС‚РѕСЏС‰РёРјРё
СЌРјР±РµРґРґРёРЅРіР°РјРё (e5-small) Рё Р»РѕРєР°Р»СЊРЅРѕР№ LLM (Ollama, РґРѕР»Р¶РµРЅ Р±С‹С‚СЊ Р·Р°РїСѓС‰РµРЅ).

Р—Р°РїСѓСЃРє: venv\\Scripts\\python.exe scripts\\verify_demo_report.py
"""
from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEMO = Path(__file__).resolve().parents[1] / "Р¤РёРЅР°РЅСЃРѕРІС‹Р№_РѕС‚С‡С‘С‚_РћРћРћ_Р’РµРєС‚РѕСЂ_2023-2025.xlsx"

os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[1] / "hf-cache"))

from app.config import settings  # noqa: E402

QUESTIONS = [
    ("РєР°РєР°СЏ РІС‹СЂСѓС‡РєР° Р·Р° 2025 РіРѕРґ?", ["536"]),
    ("РЅР° СЃРєРѕР»СЊРєРѕ РІС‹СЂРѕСЃР»Р° С‡РёСЃС‚Р°СЏ РїСЂРёР±С‹Р»СЊ СЃ 2023 РїРѕ 2025 РіРѕРґ?", ["59"]),
    ("РїСЂРѕРіРЅРѕР· РїРѕ РІС‹СЂСѓС‡РєРµ РЅР° 2026 РіРѕРґ", ["2026"]),
    ("РёР· С‡РµРіРѕ СЃРѕСЃС‚РѕРёС‚ РёС‚РѕРіРѕ СЂР°СЃС…РѕРґРѕРІ?", ["РјР°С‚РµСЂРёР°Р»"]),
    ("СЃСЂР°РІРЅРё С„Р°РєС‚ Рё РїР»Р°РЅ 2025 РїРѕ РІС‹СЂСѓС‡РєРµ", ["РїР»Р°РЅ"]),
    ("С‡С‚Рѕ РіРѕРІРѕСЂРёС‚СЃСЏ Рѕ СЂРёСЃРєР°С…?", ["СЂРёСЃРє"]),
    ("РєС‚Рѕ РіРµРЅРµСЂР°Р»СЊРЅС‹Р№ РґРёСЂРµРєС‚РѕСЂ РєРѕРјРїР°РЅРёРё?", ["РЎРѕРєРѕР»РѕРІ"]),
    ("РєР°РєРёРµ РїРѕСЃС‚СѓРїР»РµРЅРёСЏ Р·Р° РѕРєС‚СЏР±СЂСЊ 2025 РіРѕРґР°?", ["95,6", "95.6"]),
]


async def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="rag_demo_verify_"))
    settings.database_url = f"sqlite+aiosqlite:///{tmp}/demo.db"
    settings.data_dir = tmp

    from app.embeddings import EmbeddingService
    from app.ingest.pipeline import process_document
    from app.llm import make_llm
    from app.qa import AnswerPipeline
    from app.storage import (
        find_metric_by_name,
        ledger_months,
        make_engine,
        make_sessionmaker,
        metric_children,
        series_for_metric,
    )

    engine = await make_engine(settings)
    sessions = make_sessionmaker(engine)
    emb = EmbeddingService(settings)
    llm = make_llm(settings)
    print(f"Р­РјР±РµРґРґРёРЅРіРё: {emb.provider} / {emb.model}; LLM: {settings.llm_model}\n")

    # --- 1. Р·Р°РіСЂСѓР·РєР° С„Р°Р№Р»Р° ---
    async with sessions() as s:
        report = await process_document(
            s, emb, settings, org_id=1, user_id=1,
            original_name=DEMO.name, content=DEMO.read_bytes(),
        )
        await s.commit()
    print(report.summary())
    assert report.status == "processed", report.status
    assert not report.warnings, f"РїСЂРµРґСѓРїСЂРµР¶РґРµРЅРёСЏ РїР°СЂСЃРµСЂР°: {report.warnings}"

    # --- 2. СЃС‚СЂСѓРєС‚СѓСЂР° Р‘Р” ---
    async with sessions() as s:
        m_rev = await find_metric_by_name(s, 1, "РІС‹СЂСѓС‡РєР°")
        fact_rows = await series_for_metric(s, 1, m_rev.id)
        plan_rows = await series_for_metric(s, 1, m_rev.id, variant="plan")
        m_exp = await find_metric_by_name(s, 1, "РёС‚РѕРіРѕ СЂР°СЃС…РѕРґС‹")
        exp_kids = await metric_children(s, m_exp.id)
        m_act = await find_metric_by_name(s, 1, "РёС‚РѕРіРѕ Р°РєС‚РёРІС‹")
        act_kids = await metric_children(s, m_act.id)
        months_out = await ledger_months(s, 1, "СЃРїРёСЃР°РЅРёСЏ")
        months_in = await ledger_months(s, 1, "РїРѕСЃС‚СѓРїР»РµРЅРёСЏ")

    assert report.facts == 107, f"С„Р°РєС‚РѕРІ {report.facts}, РѕР¶РёРґР°Р»РѕСЃСЊ 107"
    assert [round(r["value"] / 1e6, 1) for r in fact_rows] == [412.0, 468.5, 536.2], \
        [r["value"] for r in fact_rows]
    assert {r["period_label"]: round(r["value"] / 1e6) for r in plan_rows} == {"2025": 520, "2026": 610}
    assert len(exp_kids) == 5, [k.name for k in exp_kids]
    assert len(act_kids) == 4, [k.name for k in act_kids]
    assert len(months_out) == 3 and len(months_in) == 3
    assert [round(m["total"] / 1e6, 2) for m in months_out] == [38.99, 23.43, 16.92], \
        [m["total"] for m in months_out]
    print(f"\n[OK] С„Р°РєС‚РѕРІ: {report.facts}; РІС‹СЂСѓС‡РєР° 412в†’468,5в†’536,2 РјР»РЅ; РїР»Р°РЅ 520/610 РјР»РЅ")
    print(f"[OK] РґРµС‚Р°Р»РёР·Р°С†РёСЏ СЂР°СЃС…РѕРґРѕРІ: {len(exp_kids)} СЌР»РµРјРµРЅС‚РѕРІ; Р°РєС‚РёРІРѕРІ: {len(act_kids)} СЃС‚Р°С‚РµР№")
    print("[OK] РІС‹РїРёСЃРєР°: СЃРїРёСЃР°РЅРёСЏ РїРѕ РјРµСЃСЏС†Р°Рј "
          + ", ".join(f"{m['label']} вЂ” {m['total']/1e6:.2f} РјР»РЅ" for m in months_out))

    # --- 3. РІРѕРїСЂРѕСЃС‹ С‡РµСЂРµР· QA-РєРѕРЅРІРµР№РµСЂ (СЂРµР°Р»СЊРЅР°СЏ LLM) ---
    pipeline = AnswerPipeline(sessions, emb, llm, settings, org_names=["РћРћРћ В«Р’РµРєС‚РѕСЂВ»"])
    failed = []
    for q, expects in QUESTIONS:
        outcome = await pipeline.answer(1, 1, q)
        text = outcome.text
        short = " ".join(text.split())[:220]
        ok = any(e.lower() in text.lower() for e in expects)
        print(f"\n[{'OK ' if ok else 'FAIL'}] Р’: {q}")
        print(f"        Рћ: {short}")
        if not ok:
            failed.append(q)

    await llm.close()
    await engine.dispose()
    shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n=== РС‚РѕРі: {len(QUESTIONS) - len(failed)}/{len(QUESTIONS)} РІРѕРїСЂРѕСЃРѕРІ вЂ” OK ===")
    if failed:
        print("РџСЂРѕР±Р»РµРјРЅС‹Рµ РІРѕРїСЂРѕСЃС‹:", "; ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        sys.exit(asyncio.run(main()))
