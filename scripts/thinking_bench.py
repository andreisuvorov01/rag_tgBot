"""Замер: что даёт режим рассуждений модели и сколько он стоит.

Прогоняет один и тот же golden-набор вопросов в нескольких режимах
(`LLM_THINKING=disabled|low|high`) на реальном API и собирает метрики:

  качество — верные числа в ответе, оборванные ответы, отклонения верификатором
             (верификатор ловит выдуманные числа — их доля и есть «галлюцинации»);
  цена    — токены входа/выхода, «мысли» отдельно, деньги по LLM_PRICES;
  скорость — время ответа.

Запуск:
    python -m scripts.thinking_bench                 # disabled / low / high
    python -m scripts.thinking_bench --modes disabled high
    python -m scripts.thinking_bench --questions 4

Расход: демо-данные маленькие, но каждый режим — это полный прогон вопросов.
По умолчанию это ~50–80 вызовов API, то есть несколько копеек.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    _reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(_reconfigure):
        _reconfigure(encoding="utf-8")

from app.config import settings  # noqa: E402
from app.embeddings import EmbeddingService  # noqa: E402
from app.ingest.pipeline import process_document  # noqa: E402
from app.llm import make_llm  # noqa: E402
from app.qa import AnswerPipeline  # noqa: E402
from app.storage import make_engine, make_sessionmaker  # noqa: E402
from scripts.e2e_live import (  # noqa: E402
    LIVE_GOLDEN,
    _check,
    _looks_truncated,
    _prepare_demo_files,
)


class RejectionCounter(logging.Handler):
    """Считает отклонения верификатора — по ним видно долю выдуманных чисел."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.foreign: list[float] = []
        self.attempts = 0

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        if "посторонние числа" not in msg:
            return
        self.attempts += 1
        for token in msg.split("[", 1)[-1].split("]", 1)[0].split(","):
            with contextlib.suppress(ValueError):
                self.foreign.append(float(token.strip()))


async def run_mode(mode: str, questions: list[tuple[str, list[str]]], workdir: Path,
                   repeat: int = 1) -> dict:
    """Несколько прогонов golden-набора в заданном режиме; метрики усредняются.

    Один прогон из 7 вопросов шумит: ответы формулируются по-разному, и разница
    «6/7 против 7/7» может быть случайной. Поэтому по умолчанию делаем повторы
    и отдаём средние — иначе вывод «рассуждения лучше/хуже» не защищён.
    """
    runs = [await _run_once(mode, questions, workdir, i) for i in range(repeat)]
    keys = [k for k in runs[0] if k != "mode"]
    out: dict = {"mode": mode, "repeat": repeat}
    for key in keys:
        values = [r[key] for r in runs if isinstance(r[key], (int, float))]
        out[key] = (sum(values) / len(values)) if values else runs[0][key]
    return out


async def _run_once(mode: str, questions: list[tuple[str, list[str]]], workdir: Path,
                    index: int) -> dict:
    """Один прогон golden-набора в заданном режиме рассуждений."""
    from app.usage import usage_snapshot

    settings.llm_thinking = mode
    settings.database_url = (
        f"sqlite+aiosqlite:///{(workdir / f'{mode}-{index}.db').as_posix()}"
    )
    settings.data_dir = workdir
    settings.llm_usage_log = str(workdir / f"usage-{mode}-{index}.jsonl")

    counter = RejectionCounter()
    qa_logger = logging.getLogger("app.qa")
    verifier_logger = logging.getLogger("app.agents")
    qa_logger.addHandler(counter)
    verifier_logger.addHandler(counter)
    qa_logger.setLevel(logging.WARNING)
    verifier_logger.setLevel(logging.WARNING)

    engine = await make_engine(settings)
    sessions = make_sessionmaker(engine)
    emb = EmbeddingService(settings)
    llm = make_llm(settings)
    pipeline = AnswerPipeline(sessions, emb, llm, settings, org_names=["ООО Демо"])

    try:
        for name, path in _prepare_demo_files():
            async with sessions() as session:
                await process_document(session, emb, settings, org_id=1, user_id=42,
                                       original_name=name, content=path.read_bytes())
                await session.commit()

        ok = truncated = 0
        expected_found = rejected = 0
        latencies: list[float] = []
        for question, expected in questions:
            started = time.perf_counter()
            outcome = await pipeline.answer(1, 42, question)
            latencies.append(time.perf_counter() - started)
            missing = _check(outcome.text, expected)
            cut = _looks_truncated(outcome.text)
            if not missing:
                expected_found += 1        # содержание на месте (формат любой)
            if cut:
                truncated += 1
            # ответ ушёл в шаблон: верификатор не пропустил формулировку модели
            if outcome.text.lstrip().startswith("📚") or "Ограничения" in outcome.text[:80]:
                rejected += 1
            if not missing and not cut:
                ok += 1

        async with sessions() as s:
            snap = await usage_snapshot(s, settings)
    finally:
        qa_logger.removeHandler(counter)
        verifier_logger.removeHandler(counter)
        await llm.close()
        await engine.dispose()

    calls = sum(m["calls"] for m in snap["models"])
    return {
        "mode": mode,
        "questions": len(questions),
        "ok": ok,
        "expected_found": expected_found,
        "rejected_by_verifier": rejected,
        "truncated": truncated,
        "rejections": counter.attempts,
        "foreign": len(counter.foreign),
        "calls": calls,
        "prompt_tokens": snap["prompt_tokens"],
        "completion_tokens": snap["completion_tokens"],
        "reasoning_tokens": snap.get("reasoning_tokens", 0),
        "cached_tokens": snap["cached_tokens"],
        "cost": snap.get("cost"),
        "avg_latency": sum(latencies) / max(1, len(latencies)),
        "calls_per_question": calls / max(1, len(questions)),
    }


def _line(label: str, results: list[dict], fmt: Callable[[dict], Any], *,
          best: str | None = None) -> str:
    cells = []
    for r in results:
        value = fmt(r)
        marker = " ←" if best and r["mode"] == best else ""
        cells.append(f"{value}{marker}")
    return f"{label:34}" + "".join(f"{c:>16}" for c in cells)


async def main() -> int:
    ap = argparse.ArgumentParser(description="Замер режима рассуждений: качество и цена")
    ap.add_argument("--modes", nargs="+", default=["disabled", "low", "high"],
                    help="режимы LLM_THINKING для сравнения")
    ap.add_argument("--questions", type=int, default=0, help="сколько вопросов (0 = весь набор)")
    ap.add_argument("--for-compose", action="store_true",
                    help="включать рассуждения только для составления ответа "
                         "(LLM_THINKING_TASKS=compose) — сравнение «где они реально нужны»")
    ap.add_argument("--repeat", type=int, default=3,
                    help="прогонов на режим (усредняем: один прогон шумит)")
    args = ap.parse_args()

    if settings.llm_provider == "mock" or not (settings.llm_api_key or "").strip():
        print("Нужен живой API: LLM_PROVIDER=openai_compatible и LLM_API_KEY в .env")
        return 2
    if args.for_compose:
        settings.llm_thinking_tasks = "compose"

    questions = LIVE_GOLDEN[: args.questions] if args.questions else LIVE_GOLDEN
    workdir = Path(tempfile.mkdtemp(prefix="rag-thinking-"))
    print("=== Режим рассуждений: сравнение ===")
    print(f"модель {settings.llm_model} | вопросов {len(questions)} | "
          f"тариф {settings.llm_prices} | режимы: {', '.join(args.modes)}")
    if args.for_compose:
        print("рассуждения включаются только для задачи compose "
              "(классификация, реранкинг и SQL — без них)")
    print()

    results: list[dict] = []
    for mode in args.modes:
        print(f"… режим {mode}: {args.repeat} прогон(а) по {len(questions)} вопросов", flush=True)
        results.append(await run_mode(mode, questions, workdir, repeat=args.repeat))

    print("\n" + "=" * 96)
    header = " " * 34 + "".join(f"{m:>16}" for m in args.modes)
    print(header)
    print("-" * len(header))
    print(_line("ожидаемые числа в ответе", results,
                lambda r: f"{r['expected_found']:.1f}/{r['questions']}",
                best=max(results, key=lambda r: r["expected_found"])["mode"]))
    print(_line("ответы ушли в шаблон", results, lambda r: f"{r['rejected_by_verifier']:.1f}",
                best=min(results, key=lambda r: r["rejected_by_verifier"])["mode"]))
    print(_line("верных и не оборванных", results,
                lambda r: f"{r['ok']:.1f}/{r['questions']}",
                best=max(results, key=lambda r: r["ok"])["mode"]))
    print(_line("оборванных ответов", results, lambda r: f"{r['truncated']:.1f}",
                best=min(results, key=lambda r: r["truncated"])["mode"]))
    print(_line("отклонений верификатором", results, lambda r: f"{r['rejections']:.1f}",
                best=min(results, key=lambda r: r["rejections"])["mode"]))
    print(_line("  из них выдуманных чисел", results, lambda r: f"{r['foreign']:.1f}"))
    print(_line("вызовов API", results, lambda r: f"{r['calls']:.1f}"))
    print(_line("  на вопрос", results, lambda r: f"{r['calls_per_question']:.1f}"))
    print(_line("токенов входа", results, lambda r: f"{r['prompt_tokens']:,.0f}"))
    print(_line("токенов выхода", results, lambda r: f"{r['completion_tokens']:,.0f}"))
    print(_line("  из них «мысли»", results, lambda r: f"{r['reasoning_tokens']:,.0f}"))
    print(_line("токенов из кэша", results, lambda r: f"{r['cached_tokens']:,.0f}"))
    if all(r["cost"] is not None for r in results):
        print(_line("стоимость прогона", results, lambda r: f"{r['cost']:.4f}"))
        print(_line("  на вопрос", results,
                    lambda r: f"{r['cost'] / max(1, r['questions']):.5f}",
                    best=min(results, key=lambda r: r["cost"])["mode"]))
    print(_line("среднее время ответа, с", results, lambda r: f"{r['avg_latency']:.1f}",
                best=min(results, key=lambda r: r["avg_latency"])["mode"]))

    disabled = next((r for r in results if r["mode"] == "disabled"), None)
    print("\nВывод:")
    for r in results:
        if r["mode"] == "disabled" or disabled is None:
            continue
        cost_ratio = (r["cost"] / disabled["cost"]) if (r["cost"] and disabled["cost"]) else None
        note = f"дороже в {cost_ratio:.2f}×" if cost_ratio else "цена не считается"
        print(f"  {r['mode']}: числа на месте {r['expected_found']:.1f}/{r['questions']} против "
              f"{disabled['expected_found']:.1f}/{disabled['questions']} без рассуждений; {note}; "
              f"в шаблон ушло {r['rejected_by_verifier']:.1f} против "
              f"{disabled['rejected_by_verifier']:.1f}; "
              f"время {r['avg_latency']:.1f} с против {disabled['avg_latency']:.1f} с")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
