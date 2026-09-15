from app.security import anonymize_text


def test_anonymize_identifiers():
    text = "Организация ООО Ромашка, ИНН 7701234567, р/с 40702810400000012345, тел. +7 495 123-45-67"
    out = anonymize_text(text, ["ООО Ромашка"])
    assert "7701234567" not in out
    assert "40702810400000012345" not in out
    assert "[ИНН]" in out and "[СЧЁТ]" in out and "[ТЕЛЕФОН]" in out
    assert "[ORG0]" in out and "ООО Ромашка" not in out


def test_anonymize_keeps_financial_values():
    text = "Выручка 45 800 000 руб. за 2025 год"
    out = anonymize_text(text, [])
    assert "45 800 000" in out


def test_anonymize_does_not_mask_large_amounts_in_json():
    """Сумма в миллиардах (12 цифр) в JSON-данных композитора — не ИНН."""
    blob = '{"value": 135400273000.0, "inn": "7721546864", "small": 1354002730.5, "v": 36550478000.0}'
    out = anonymize_text(blob, [])
    assert "135400273000.0" in out and "1354002730.5" in out and "36550478000.0" in out
    assert "7721546864" not in out
