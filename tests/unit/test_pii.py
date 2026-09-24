import pytest
from app.observability.pii import redact_pii

@pytest.mark.parametrize("input_text,expected", [
    # Должно замаскировать
    ("Позвони мне +7 (916) 123-45-67", "Позвони мне [PHONE]"),
    ("Мой email: ivan@example.com", "Мой email: [EMAIL]"),
    ("Карта 4276 5500 1234 5678", "Карта [CARD]"),
    
    # НЕ должно замаскировать (ложные срабатывания)
    ("Версия 1.12.34", "Версия 1.12.34"),  # не телефон
    ("IP 192.168.1.1", "IP 192.168.1.1"),   # не телефон
    ("Дата 12.05.2026", "Дата 12.05.2026"), # не телефон
    ("ID бага 102", "ID бага 102"),         # не карта
])
def test_redact_pii(input_text, expected):
    assert redact_pii(input_text) == expected