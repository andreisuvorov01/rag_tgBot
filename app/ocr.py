"""OCR сканов.

Приоритет — локальный Tesseract (данные не покидают сервер). Если он не
установлен и включён OCR_VLM_ENABLED=true, страница уходит в vision-модель
внешнего LLM API (OpenAI-совместимый endpoint). Это единственный случай,
когда изображение документа покидает контур, — поэтому режим выключен по умолчанию.
"""
from __future__ import annotations

import base64
import logging

import httpx

from .config import Settings

log = logging.getLogger(__name__)

VLM_PROMPT = (
    "Это страница финансового документа. Извлеки весь текст с изображения "
    "максимально точно: числа, названия показателей, заголовки. Таблицы передавай "
    "строками вида «ячейка | ячейка | ячейка», сохраняя заголовки. Без комментариев."
)


class VlmOcr:
    def __init__(self, settings: Settings):
        self.s = settings
        self.enabled = bool(
            settings.ocr_vlm_enabled
            and settings.llm_provider == "openai_compatible"
            and settings.llm_api_base
            and settings.llm_api_key
        )
        self.model = settings.ocr_vlm_model or settings.llm_model

    async def ocr_png(self, png_bytes: bytes) -> str:
        if not self.enabled:
            return ""
        b64 = base64.b64encode(png_bytes).decode()
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": VLM_PROMPT},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    ],
                }
            ],
            "max_tokens": 2000,
            "temperature": 0.0,
        }
        try:
            async with httpx.AsyncClient(timeout=180) as client:
                resp = await client.post(
                    f"{self.s.llm_api_base.rstrip('/')}/chat/completions",
                    headers={"Authorization": f"Bearer {self.s.llm_api_key}"},
                    json=payload,
                )
            if resp.status_code >= 400:
                log.warning("VLM OCR %s: %s", resp.status_code, resp.text[:200])
                return ""
            return resp.json()["choices"][0]["message"]["content"] or ""
        except Exception as e:
            log.warning("VLM OCR недоступен: %s", e)
            return ""
