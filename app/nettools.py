"""Сетевые утилиты: обнаружение прокси (переменные окружения и системный
прокси Windows) для диагностики и настройки подключения к Telegram."""
from __future__ import annotations

import os

_PROXY_ENV_KEYS = (
    "HTTP_PROXY", "http_proxy",
    "HTTPS_PROXY", "https_proxy",
    "ALL_PROXY", "all_proxy",
)


def env_proxy_vars() -> dict[str, str]:
    """Прокси из переменных окружения (httpx их использует по умолчанию,
    aiohttp — только при trust_env=True)."""
    return {k: v for k, v in ((k, os.environ.get(k)) for k in _PROXY_ENV_KEYS) if v}


def parse_proxy_server(raw: str) -> str | None:
    """Значение ProxyServer из реестра Windows -> URL прокси.
    Форматы: '127.0.0.1:8080' или 'http=...;https=...;socks=...'."""
    raw = (raw or "").strip()
    if not raw:
        return None
    if "=" in raw:
        parts = dict(p.split("=", 1) for p in raw.split(";") if "=" in p)
        raw = parts.get("https") or parts.get("http") or parts.get("socks") or ""
        raw = raw.strip()
        if not raw:
            return None
    if raw.startswith(("http://", "https://", "socks4://", "socks5://")):
        return raw
    if raw.startswith("socks"):  # socks=127.0.0.1:1080 без схемы после парсинга
        return f"socks5://{raw.split('://', 1)[-1]}"
    return f"http://{raw}"


def detect_windows_proxy() -> str | None:
    """Системный прокси Windows (которым пользуется браузер/Telegram-приложение)."""
    try:
        import winreg

        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Internet Settings",
        )
        enabled, _ = winreg.QueryValueEx(key, "ProxyEnable")
        raw, _ = winreg.QueryValueEx(key, "ProxyServer")
        winreg.CloseKey(key)
        if not enabled:
            return None
        return parse_proxy_server(raw)
    except Exception:
        return None
