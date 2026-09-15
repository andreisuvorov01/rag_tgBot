"""Быстрая проверка локального Ollama через OpenAI-совместимый API (как делает app/llm.py)."""
import asyncio
import time

import httpx

BASE = "http://localhost:11434/v1"
MODEL = "qwen2.5:1.5b-instruct-q4_K_M"


async def main():
    # trust_env=False: системный прокси Windows не должен перехватывать localhost
    async with httpx.AsyncClient(timeout=180, trust_env=False) as c:
        # 1) классификация (json_mode) — как qa._classify
        t0 = time.perf_counter()
        r = await c.post(
            f"{BASE}/chat/completions",
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": "Ты классификатор запросов. Отвечай ТОЛЬКО JSON."},
                    {"role": "user", "content": "какая выручка за 2024 год?"},
                ],
                "temperature": 0.0,
                "max_tokens": 100,
                "response_format": {"type": "json_object"},
            },
        )
        dt = time.perf_counter() - t0
        print(f"[classify {dt:.1f}s] HTTP {r.status_code}")
        if r.status_code == 200:
            print("  ->", r.json()["choices"][0]["message"]["content"][:200])

        # 2) композитор ответа — как qa._compose
        t0 = time.perf_counter()
        r = await c.post(
            f"{BASE}/chat/completions",
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": "Ты финансовый ассистент. Отвечай кратко по-русски, используй только числа из данных."},
                    {"role": "user", "content": "ВОПРОС: на сколько выросла выручка?\n\nДАННЫЕ (JSON): {\"metric\": \"Выручка\", \"history\": [{\"2024\": 45800000}, {\"2025\": 49500000}], \"computed\": {\"change_pct\": 8.1, \"abs_change\": 3700000}}"},
                ],
                "temperature": 0.2,
                "max_tokens": 300,
            },
        )
        dt = time.perf_counter() - t0
        print(f"[compose  {dt:.1f}s] HTTP {r.status_code}")
        if r.status_code == 200:
            print("  ->", r.json()["choices"][0]["message"]["content"][:400])
        else:
            print("  body:", r.text[:300])


if __name__ == "__main__":
    asyncio.run(main())
