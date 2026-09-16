"""Точка входа: инициализация БД и сервисов, запуск Telegram-бота.

Вариант развертывания 2 (гибрид): всё локально — PostgreSQL/SQLite, документы,
эмбеддинги, аналитика; наружу — только генерация через LLM API.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

from .bot import run_bot
from .config import settings
from .embeddings import EmbeddingService
from .llm import make_llm
from .qa import AnswerPipeline
from .storage import make_engine, make_sessionmaker


def setup_huggingface_env() -> None:
    """Кэш моделей — рядом с проектом (на D:), чтобы не зависеть от места на C:.
    HF_HUB_OFFLINE не выставляем: offline-режим ломает загрузку кэшированной
    модели sentence-transformers (нет локального pytorch_model.bin)."""
    hf_home = os.environ.get("HF_HOME") or str((Path.cwd() / "hf-cache").resolve())
    os.environ.setdefault("HF_HOME", hf_home)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )


async def build_pipeline():
    setup_logging()
    setup_huggingface_env()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    engine = await make_engine(settings)
    session_factory = make_sessionmaker(engine)
    emb = EmbeddingService(settings)
    llm = make_llm(settings)
    if settings.embeddings_provider == "sentence_transformers" and emb.provider == "hash":
        print(
            "\n⚠️  EMBEDDINGS_PROVIDER=sentence_transformers, но пакет не установлен.\n"
            "    Запустите start.bat (использует venv) или setup_venv.bat —\n"
            "    сейчас работает dev-режим hash.\n"
        )
    _refuse_remote_embeddings(settings)
    # имена организаций — для маскирования и в промптах, и в запросах к внешнему
    # embeddings-API (AnonymizingLLM закрывает только чат-вызовы)
    org_names = await _anonymize_names(engine, session_factory)
    emb.anonymize_org_names = org_names
    _warn_token_economy(settings)
    if settings.strict_startup and not settings.allowed_ids and settings.reg_code in settings.default_reg_codes:
        raise SystemExit(
            "\n❌ Отказ запуска: ALLOWED_USER_IDS пуст, а REG_CODE — значение по умолчанию.\n"
            "   Зарегистрироваться сможет любой, кто знает код из README.\n"
            "   Задайте в .env свой REG_CODE или заполните ALLOWED_USER_IDS.\n"
            "   Осознанное исключение (только для локальной отладки): STRICT_STARTUP=false\n"
        )
    if not settings.allowed_ids and settings.reg_code in settings.default_reg_codes:
        print(
            "\n⚠️  ALLOWED_USER_IDS пуст, а REG_CODE — значение по умолчанию: зарегистрироваться\n"
            "    сможет любой, кто знает код из README. Задайте свой REG_CODE в .env.\n"
        )
    # предупреждаем о выносе данных наружу: маскирование выключено, а endpoint не локальный
    if settings.llm_provider != "mock" and not settings.anonymize_prompts:
        from urllib.parse import urlparse

        host = (urlparse(settings.llm_api_base).hostname or "").lower()
        if host not in ("localhost", "127.0.0.1", "::1", "0.0.0.0", ""):
            print(
                f"\n⚠️  ANONYMIZE_PROMPTS=false, а LLM_API_BASE указывает на внешний хост ({host}).\n"
                "    Промпты (фрагменты документов, названия показателей) уйдут без маскирования.\n"
            )
    pipeline = AnswerPipeline(session_factory, emb, llm, settings, org_names=org_names)
    await check_embeddings_fingerprint(engine, session_factory)
    async with session_factory() as s:
        from sqlalchemy import select

        from .ingest.pipeline import fixup_metric_kinds
        from .storage import Organization

        for (oid,) in (await s.execute(select(Organization.id))).all():
            await fixup_metric_kinds(s, org_id=oid)
        # оригиналы неудачных загрузок и демо-прогонов иначе копятся вечно
        from .storage import cleanup_orphan_uploads

        await cleanup_orphan_uploads(s, settings)
        await s.commit()
    return engine, session_factory, emb, llm, pipeline


def _warn_token_economy(settings) -> None:
    """Подсказки по расходу: внешний API платный, а служебные шаги вызываются
    чаще композитора. Ничего не блокируем — только показываем, что настроено."""
    notes: list[str] = []
    external_llm = settings.llm_provider == "openai_compatible" and not _is_local(settings.llm_api_base)
    if external_llm:
        if not (settings.llm_model_small or "").strip():
            notes.append(
                "LLM_MODEL_SMALL не задан: классификация, реранкинг и SQL идут на "
                "основную модель. Дешёвая модель на эти шаги — самая крупная экономия."
            )
        if not settings.llm_price_map:
            notes.append('LLM_PRICES не задан: /usage покажет токены без денег '
                         '(например LLM_PRICES="deepseek-chat=0.27/1.10").')
    if (settings.embeddings_provider == "api"
            and not _is_local(settings.embeddings_api_base or "http://localhost")
            and not settings.anonymize_prompts):
        notes.append(
            "EMBEDDINGS_PROVIDER=api указывает на внешний хост, а ANONYMIZE_PROMPTS=false: "
            "тексты документов уходят наружу без маскирования. Локальные эмбеддинги "
            "(sentence_transformers) не тратят токены и ничего не отправляют."
        )
    if notes:
        print()
        for n in notes:
            print(f"⚠️  {n}")


def _is_local(url: str) -> bool:
    from urllib.parse import urlparse

    host = (urlparse(url or "").hostname or "").lower()
    return host in ("localhost", "127.0.0.1", "::1", "0.0.0.0", "")


def remote_embeddings_reason(settings) -> str | None:
    """Почему запуск с этой конфигурацией эмбеддингов недопустим (или None).

    Векторизация — самая объёмная операция по тексту: каждый загруженный
    документ превращается в десятки фрагментов. Через платный внешний API это
    прямые деньги и вынос текстов документов наружу, а локальная модель делает
    то же бесплатно. Поэтому внешний embeddings-endpoint запрещён по умолчанию,
    а не просто не рекомендуется.
    """
    if not settings.require_local_embeddings:
        return None
    if settings.effective_embeddings_provider != "api":
        return None
    base = settings.embeddings_api_base or ""
    if _is_local(base):
        return None
    return base or "хост не задан"


def _refuse_remote_embeddings(settings) -> None:
    reason = remote_embeddings_reason(settings)
    if reason is None:
        return
    raise SystemExit(
        "\n❌ Отказ запуска: EMBEDDINGS_PROVIDER=api указывает на внешний хост "
        f"({reason}).\n"
        "   Векторизация документов через платный API — это расход на токены и вынос\n"
        "   текстов документов наружу. Локальные эмбеддинги бесплатны:\n"
        "     EMBEDDINGS_PROVIDER=sentence_transformers\n"
        "     EMBEDDINGS_LOCAL_MODEL=intfloat/multilingual-e5-small   (dim 384, ~0,5 ГБ ОЗУ)\n"
        "   Если у сервера нет ОЗУ под модель — осознанное исключение:\n"
        "     REQUIRE_LOCAL_EMBEDDINGS=false  (и ANONYMIZE_PROMPTS=true обязателен)\n"
    )


async def _anonymize_names(engine, session_factory) -> list[str]:
    """Имена организаций для маскирования в промптах.

    Реальные названия лежат в таблице organizations, поэтому брать их из БД
    правильнее, чем подставлять константу: прежний захардкоженный список
    («Основная организация») не маскировал ни одного настоящего названия.
    """
    from sqlalchemy import select

    from .storage import Organization

    names: list[str] = [n.strip() for n in settings.anonymize_org_names.split(",") if n.strip()]
    try:
        async with session_factory() as s:
            names.extend(str(n) for n in (await s.scalars(select(Organization.name))).all() if n)
    except Exception as e:
        log = logging.getLogger(__name__)
        log.warning("Не удалось прочитать названия организаций для маскирования: %s", e)
    # длинные первыми: иначе короткое имя «ООО» съест часть длинного
    return sorted({n for n in names if n}, key=len, reverse=True)


async def warmup_embeddings(emb: EmbeddingService) -> None:
    """Прогрев модели эмбеддингов в фоне сразу после старта, чтобы первый
    вопрос пользователя не ждал загрузку модели."""
    try:
        await emb.embed_query("прогрев модели")
        logging.getLogger(__name__).info("Прогрев эмбеддингов завершён")
    except Exception as e:
        logging.getLogger(__name__).warning("Прогрев эмбеддингов не удался: %s", e)


async def check_embeddings_fingerprint(engine, session_factory) -> None:
    """Предупреждает, если векторная база проиндексирована другим провайдером —
    векторы несовместимы, нужна переиндексация (python -m scripts.reindex)."""
    from sqlalchemy import func, select

    from .embeddings import embeddings_fingerprint
    from .storage import Chunk, Metric, get_meta, set_meta

    current = embeddings_fingerprint(settings)
    async with session_factory() as s:
        stored = await get_meta(s, "embeddings_fingerprint")
        n_emb = (
            await s.execute(
                select(func.count()).select_from(Chunk).where(Chunk.embedding.is_not(None))
            )
        ).scalar_one()
        n_metrics = (
            await s.execute(
                select(func.count()).select_from(Metric).where(Metric.embedding.is_not(None))
            )
        ).scalar_one()
        if stored == current:
            return
        if stored is None and n_emb == 0 and n_metrics == 0:
            await set_meta(s, "embeddings_fingerprint", current)
            await s.commit()
            return
        if stored is None:
            # база старого образца без отпечатка — считаем её приведённой к текущему
            await set_meta(s, "embeddings_fingerprint", current)
            await s.commit()
            return
        print(
            f"\n⚠️  ВНИМАНИЕ: эмбеддинги в базе от другой конфигурации ({stored}), "
            f"сейчас {current}.\n   Векторы несовместимы — семантический поиск будет некорректен.\n"
            f"   Переиндексируйте:  python -m scripts.reindex\n"
        )


async def main() -> None:
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
        # reconfigure есть только у TextIOWrapper; у подменённого stdout его
        # может не быть — тогда печатаем как есть (getattr + callable)
        for stream in (sys.stdout, sys.stderr):
            reconfigure = getattr(stream, "reconfigure", None)
            if callable(reconfigure):
                reconfigure(encoding="utf-8")
    engine, session_factory, emb, llm, pipeline = await build_pipeline()
    try:
        print(
            "\n=== Финансовый ассистент (RAG + LLM) ===\n"
            f"БД:           {settings.database_url.split('://')[0]}\n"
            f"LLM:          {settings.llm_provider} ({settings.llm_model or '—'})"
            + ("  [offline-шаблоны]" if settings.llm_provider == "mock" else "")
            + f"\nЭмбеддинги:   {settings.embeddings_provider}\n"
            f"Реранкер:     {settings.reranker}\n"
            f"Верификатор:  {'вкл' if settings.verify_answers else 'выкл'}\n"
        )
        # прогрев модели в фоне: первый вопрос пользователя не ждёт её загрузку
        warmup_task = asyncio.create_task(warmup_embeddings(emb))
        await run_bot(settings, pipeline, session_factory)
        warmup_task.cancel()
    finally:
        await llm.close()
        await engine.dispose()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹ Бот остановлен. До связи!")
