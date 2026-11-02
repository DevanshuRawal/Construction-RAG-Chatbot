"""Opt-in, PII-redacted JSONL diagnostics for chat turns and retrieved evidence."""
from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from src.assistant import ConstructionAssistant

logger = logging.getLogger(__name__)


def log_debug_turn(user_message, assistant_message, session_id, lead, retrieved_chunks) -> bool:
    """Append one redacted turn with source/snippet trace when DEBUG_LOGGING is enabled."""
    if os.getenv("DEBUG_LOGGING", "false").strip().lower() not in {"1", "true", "yes", "on"}:
        return False

    try:
        redact = lambda value: ConstructionAssistant._redact_for_model(str(value), lead)
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "conversation_id": hashlib.sha256(str(session_id).encode("utf-8")).hexdigest()[:16],
            "user_message": redact(user_message),
            "assistant_message": redact(assistant_message),
            "retrieved_chunks": [
                {
                    "source": chunk.get("source", "unknown"),
                    "score": round(float(chunk.get("score", 0.0)), 4),
                    "snippet": redact(chunk.get("text", ""))[:300],
                }
                for chunk in retrieved_chunks
            ],
        }
        path = Path(os.getenv("DEBUG_LOG_FILE", "debug_logs/conversations.jsonl"))
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        return True
    except (OSError, TypeError, ValueError):
        logger.exception("Could not write redacted chat diagnostics")
        return False
