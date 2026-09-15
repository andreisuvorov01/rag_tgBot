"""End-to-end smoke-С‚РµСЃС‚ РЅР° РљРћРџРР СЂРµР°Р»СЊРЅРѕР№ Р‘Р”: РЅР°СЃС‚РѕСЏС‰РёРµ СЌРјР±РµРґРґРёРЅРіРё (e5-small)
+ РЅР°СЃС‚РѕСЏС‰РёР№ Р»РѕРєР°Р»СЊРЅС‹Р№ LLM (Ollama Qwen2.5-1.5B) + Excel-Р¶СѓСЂРЅР°Р» СЂР°СЃС…РѕРґРѕРІ.

РќРёС‡РµРіРѕ РЅРµ РїРёС€РµС‚ РІ СЂР°Р±РѕС‡СѓСЋ Р±Р°Р·Сѓ: СЂР°Р±РѕС‚Р°РµС‚ СЃ РєРѕРїРёРµР№ РІ temp-РєР°С‚Р°Р»РѕРіРµ.
Р—Р°РїСѓСЃРє: venv\\Scripts\\python.exe scripts\\smoke_e2e.py
"""
import asyncio
import contextlib
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402


async def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="rag_smoke_"))
    # РєРѕРїРёСЏ СЂРµР°Р»СЊРЅРѕР№ Р‘Р” + РѕС‚РґРµР»СЊРЅС‹Р№ data-dir: РїРѕР»СЊР·РѕРІР°С‚РµР»СЊСЃРєРёРµ РґР°РЅРЅС‹Рµ РЅРµ С‚СЂРѕРіР°РµРј
    shutil.copy(Path("data/app.db"), tmp / "app.db")
    settings.database_url = f"sqlite+aiosqlite:///{tmp}/app.db"
    settings.data_dir = tmp
    settings.upload_dir.mkdir(parents=True, exist_ok=True)

    from app.embeddings import EmbeddingService
    from app.expenses import add_entry, month_report, parse_entry_message, sync_journal_file
    from app.llm import make_llm
    from app.qa import AnswerPipeline
    from app.storage import make_engine, make_sessionmaker

    engine = await make_engine(settings)
    sessions = make_sessionmaker(engine)
    emb = EmbeddingService(settings)
    llm = make_llm(settings)
    pipeline = AnswerPipeline(sessions, emb, llm, settings, org_names=["РћСЃРЅРѕРІРЅР°СЏ РѕСЂРіР°РЅРёР·Р°С†РёСЏ"])

    print(f"LLM: {settings.llm_provider} / {settings.llm_model}")
    print(f"Р­РјР±РµРґРґРёРЅРіРё: {emb.provider} / {emb.model}")
    ok = 0
    failed = 0

    # 1) С„Р°РєС‚РёС‡РµСЃРєРёР№ РІРѕРїСЂРѕСЃ РїРѕ Р·Р°РіСЂСѓР¶РµРЅРЅРѕРјСѓ РѕС‚С‡С‘С‚Сѓ (СЂРµР°Р»СЊРЅС‹Рµ РґР°РЅРЅС‹Рµ РІ Р‘Р”)
    for q in [
        "РєР°РєР°СЏ РІС‹СЂСѓС‡РєР° Р·Р° 2024 РіРѕРґ?",
        "РїСЂРѕРіРЅРѕР· РїРѕ РІС‹СЂСѓС‡РєРµ РЅР° 2026 РіРѕРґ",
        "С‡С‚Рѕ РіРѕРІРѕСЂРёС‚СЃСЏ Рѕ СЂРёСЃРєР°С…?",
    ]:
        outcome = await pipeline.answer(1, 1, q)
        text = outcome.text.replace("\n", " ")[:160]
        status = "OK " if outcome.text and len(outcome.text) > 30 else "FAIL"
        ok, failed = (ok + (status == "OK "), failed + (status == "FAIL"))
        print(f"\n[{status}] Q: {q}\n      A: {text}")

    # 2) СЂСѓС‡РЅРѕР№ СЂР°СЃС…РѕРґ: СЃРѕРѕР±С‰РµРЅРёРµ -> Р‘Р” -> С„Р°РєС‚ -> Excel
    async with sessions() as s:
        r = await add_entry(
            s, settings, org_id=1, user_id=1,
            entry=parse_entry_message("СЂР°СЃС…РѕРґ: 1500 РєРѕС„Рµ СЃ РєРѕР»Р»РµРіР°РјРё"),
            emb=emb,
        )
        await sync_journal_file(s, settings, 1)
        await s.commit()
    xlsx = settings.expense_journal_path
    print(f"\n[OK ] СЂР°СЃС…РѕРґ Р·Р°РїРёСЃР°РЅ, РёС‚РѕРіРѕ РјРµСЃСЏС†: {r['month_total']}")
    print(f"      Excel: {xlsx} (exists={xlsx.exists()})")
    ok += 1

    # 3) РІРѕРїСЂРѕСЃ РїРѕ СЂР°СЃС…РѕРґР°Рј С‡РµСЂРµР· С‚РѕС‚ Р¶Рµ QA-РєРѕРЅРІРµР№РµСЂ
    outcome = await pipeline.answer(1, 1, "СЃРєРѕР»СЊРєРѕ Р»РёС‡РЅС‹С… СЂР°СЃС…РѕРґРѕРІ РІ СЌС‚РѕРј РјРµСЃСЏС†Рµ?")
    print("\n[OK ] Q: СЃРєРѕР»СЊРєРѕ Р»РёС‡РЅС‹С… СЂР°СЃС…РѕРґРѕРІ РІ СЌС‚РѕРј РјРµСЃСЏС†Рµ?")
    print(f"      A: {outcome.text.replace(chr(10), ' ')[:200]}")
    ok += 1

    async with sessions() as s:
        report = await month_report(s, settings, 1)
    print(f"\n      РЎРІРѕРґРєР°: {report.replace(chr(10), ' | ')[:200]}")

    await llm.close()
    await engine.dispose()
    print(f"\n=== РС‚РѕРі: {ok} OK, {failed} FAIL ===")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        sys.exit(asyncio.run(main()))
