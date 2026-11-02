import json

from src.diagnostics import log_debug_turn


def test_debug_log_records_redacted_turn_and_retrieved_chunks(tmp_path, monkeypatch):
    log_path = tmp_path / "chat.jsonl"
    monkeypatch.setenv("DEBUG_LOGGING", "true")
    monkeypatch.setenv("DEBUG_LOG_FILE", str(log_path))
    lead = {"name": "Test Person", "phone_or_email": "test.person@example.com"}
    retrieved = [{
        "source": "04_pricing_rate_card.pdf",
        "score": 0.83,
        "text": "Premium interiors are INR 1950 per square foot.",
    }]

    recorded = log_debug_turn(
        "My name is Test Person, email test.person@example.com",
        "We will follow up at test.person@example.com",
        "session-private-id",
        lead,
        retrieved,
    )

    assert recorded is True
    event = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])
    assert event["user_message"] == "My name is [REDACTED], email [REDACTED]"
    assert "test.person@example.com" not in event["assistant_message"]
    assert event["retrieved_chunks"] == [{
        "source": "04_pricing_rate_card.pdf",
        "score": 0.83,
        "snippet": "Premium interiors are INR 1950 per square foot.",
    }]
    assert event["conversation_id"] != "session-private-id"


def test_debug_log_is_disabled_by_default(tmp_path, monkeypatch):
    log_path = tmp_path / "chat.jsonl"
    monkeypatch.delenv("DEBUG_LOGGING", raising=False)
    monkeypatch.setenv("DEBUG_LOG_FILE", str(log_path))

    assert log_debug_turn("hello", "hi", "session-1", {}, []) is False
    assert not log_path.exists()
