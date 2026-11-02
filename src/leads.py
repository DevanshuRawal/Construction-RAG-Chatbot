"""Privacy-conscious append-only lead capture."""
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken


FIELDS = ("name", "phone_or_email", "city", "project_type", "area_sqft", "budget", "start_window", "contact_consent")


def _cipher() -> Fernet:
    key = os.getenv("LEADS_ENCRYPTION_KEY")
    if not key:
        raise RuntimeError("LEADS_ENCRYPTION_KEY is required; refusing to store lead data in plaintext")
    try:
        return Fernet(key.encode("ascii"))
    except (UnicodeError, ValueError) as exc:
        raise RuntimeError("LEADS_ENCRYPTION_KEY is invalid; refusing to store lead data") from exc


def prune_expired_leads(path: str = "leads.enc.jsonl", retention_days: int = 180, now: datetime | None = None) -> int:
    if retention_days <= 0:
        raise ValueError("Lead retention must be a positive number of days")
    target = Path(path)
    if not target.exists():
        return 0
    cipher = _cipher()
    cutoff = (now or datetime.now(timezone.utc)) - timedelta(days=retention_days)
    retained, removed = [], 0
    try:
        for line in target.read_text(encoding="utf-8").splitlines():
            envelope = json.loads(line)
            if envelope.get("version") != 1 or not isinstance(envelope.get("payload"), str):
                raise ValueError("unsupported lead envelope")
            lead = json.loads(cipher.decrypt(envelope["payload"].encode("ascii")).decode("utf-8"))
            created_at = datetime.fromisoformat(lead["created_at"])
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)
            if created_at < cutoff:
                removed += 1
            else:
                retained.append(line)
    except (ValueError, KeyError, TypeError, UnicodeError, InvalidToken) as exc:
        raise RuntimeError("Encrypted lead storage is invalid or its key does not match; no records were changed") from exc
    if removed:
        temporary = target.with_name(target.name + ".tmp")
        temporary.write_text("\n".join(retained) + ("\n" if retained else ""), encoding="utf-8")
        os.replace(temporary, target)
    return removed


def save_lead(details: dict, path: str = "leads.enc.jsonl") -> None:
    cipher = _cipher()
    lead = {key: details.get(key) for key in FIELDS}
    lead["created_at"] = datetime.now(timezone.utc).isoformat()
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    prune_expired_leads(str(target), int(os.getenv("LEADS_RETENTION_DAYS", "180")))
    encrypted = cipher.encrypt(json.dumps(lead, ensure_ascii=False).encode("utf-8")).decode("ascii")
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"version": 1, "payload": encrypted}) + "\n")

