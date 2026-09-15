"""Проба классификатора Qwen на демо-вопросах."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx  # noqa: E402

from app.prompts import CLASSIFIER_SYSTEM  # noqa: E402

QUESTIONS = [
    "какая выручка за 2025 год?",
    "на сколько выросла чистая прибыль с 2023 по 2025 год?",
    "прогноз по выручке на 2026 год",
    "из чего состоит итого расходов?",
    "сравни факт и план 2025 по выручке",
    "что говорится о рисках?",
    "кто генеральный директор компании?",
    "какие поступления были в октябре 2025?",
    "сколько поступлений в октябре 2025?",
    "какие поступления за октябрь 2025 года?",
]


async def main():
    async with httpx.AsyncClient(timeout=120, trust_env=False) as c:
        for q in QUESTIONS:
            r = await c.post(
                "http://localhost:11434/v1/chat/completions",
                json={
                    "model": "qwen2.5:1.5b-instruct-q4_K_M",
                    "messages": [
                        {"role": "system", "content": CLASSIFIER_SYSTEM},
                        {"role": "user", "content": q},
                    ],
                    "temperature": 0.0,
                    "max_tokens": 300,
                    "response_format": {"type": "json_object"},
                },
            )
            body = r.json()["choices"][0]["message"]["content"].replace("\n", " ")
            print(f"{q}\n    -> {body[:260]}\n")


asyncio.run(main())
