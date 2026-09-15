"""Безопасность: доступ к боту по whitelist/коду регистрации и
обезличивание текста перед отправкой во внешний LLM API (Вариант 2)."""
from __future__ import annotations

import re

from sqlalchemy.ext.asyncio import AsyncSession

from .config import Settings
from .storage import User, audit, get_user, register_user

# число с дробной частью (135400273000.0 в JSON данных) — сумма, а не ИНН/счёт
INN_RE = re.compile(r"(?<![\d.])(?:\d{10}|\d{12})(?!\d)(?![.,]\d)")
ACCOUNT_RE = re.compile(r"(?<![\d.])\d{20}(?!\d)(?![.,]\d)")
PHONE_RE = re.compile(r"\+7[\d\s()\-]{10,16}")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PASSPORT_RE = re.compile(r"(?<!\d)\d{4}\s\d{6}(?!\d)")


async def authenticate(session: AsyncSession, settings: Settings, telegram_id: int) -> User | None:
    """Пользователь допущен, если он в whitelist ИЛИ уже зарегистрирован в БД."""
    user = await get_user(session, telegram_id)
    if user:
        return user
    return None


async def register_with_code(
    session: AsyncSession, settings: Settings, telegram_id: int, display_name: str, code: str
) -> User | None:
    if not code or code.strip() != settings.reg_code:
        return None
    allowed = settings.allowed_ids
    if allowed and telegram_id not in allowed:
        return None
    user = await register_user(session, telegram_id, display_name, "Основная организация")
    await audit(session, telegram_id, "user_registered", display_name)
    await session.commit()
    return user


def anonymize_text(text: str, org_names: list[str]) -> str:
    """Маскирование идентификаторов перед отправкой промпта наружу.
    Числовые значения показателей не трогаем — без них ответ невозможен."""
    masked = PASSPORT_RE.sub("[ПАСПОРТ]", text)
    masked = ACCOUNT_RE.sub("[СЧЁТ]", masked)
    masked = INN_RE.sub("[ИНН]", masked)
    masked = PHONE_RE.sub("[ТЕЛЕФОН]", masked)
    masked = EMAIL_RE.sub("[EMAIL]", masked)
    for i, name in enumerate(sorted(org_names, key=len, reverse=True)):
        if not name:
            continue
        masked = re.sub(re.escape(name), f"[ORG{i}]", masked, flags=re.IGNORECASE)
    return masked
