"""Замер: что доминирует в задержке — размер промпта (prefill) или генерация.

Запуск:  python -m scripts.profile_llm_prompt
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402
from app.formatting import render_answer  # noqa: E402
from app.llm import make_llm  # noqa: E402
from app.prompts import CLASSIFIER_SYSTEM, COMPOSER_SYSTEM  # noqa: E402

FENCE = "```json"


def composer_user(blob: str, question: str) -> str:
    return f"ВОПРОС: {question}\n\nДАННЫЕ (JSON):\n{FENCE}\n{blob}\n```"


async def timed(llm, label: str, msgs, **kw) -> str:
    started = time.perf_counter()
    out = await llm.chat(msgs, **kw)
    dur = time.perf_counter() - started
    print(f"  {label:44} {dur:6.1f} с   ответ ~{len(out) // 3} токенов")
    return out


async def main() -> int:
    llm = make_llm(settings)
    await timed(llm, "прогрев", [{"role": "user", "content": "ок"}], max_tokens=3)

    payload = {
        "type": "factual",
        "metric": {"name": "выручка", "unit": "руб", "currency": "RUB"},
        "history": [
            {"label": str(y), "value": v,
             "source": "Финансовый_отчёт_ООО_Вектор_2023-2025.xlsx · Отчёт о финансовых результатах · C4"}
            for y, v in ((2023, 412000000.0), (2024, 468500000.0), (2025, 536200000.0))
        ],
    }
    payload["template_hint"] = render_answer(payload)
    blob = json.dumps(payload, ensure_ascii=False, default=str)
    question = "какая выручка за 2024 год?"
    big = composer_user(blob, question)
    small = composer_user('{"type":"factual","value":468500000}', question)
    print(f"\nразмер промпта композитора: полный {len(big)} симв. (~{len(big) // 3} токенов), "
          f"урезанный {len(small)} симв. (~{len(small) // 3} токенов)\n")

    print("классификатор:")
    await timed(llm, "max_tokens=120 (как сейчас)", [
        {"role": "system", "content": CLASSIFIER_SYSTEM},
        {"role": "user", "content": question},
    ], json_mode=True, temperature=0.0, max_tokens=120)

    print("композитор:")
    await timed(llm, "полный промпт, max_tokens=260", [
        {"role": "system", "content": COMPOSER_SYSTEM},
        {"role": "user", "content": big},
    ], max_tokens=260)
    await timed(llm, "урезанный промпт, max_tokens=260", [
        {"role": "system", "content": COMPOSER_SYSTEM},
        {"role": "user", "content": small},
    ], max_tokens=260)
    await timed(llm, "полный промпт, max_tokens=80", [
        {"role": "system", "content": COMPOSER_SYSTEM},
        {"role": "user", "content": big},
    ], max_tokens=80)

    await llm.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
