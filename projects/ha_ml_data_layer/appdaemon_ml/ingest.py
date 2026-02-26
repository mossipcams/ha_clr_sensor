"""Raw event ingestion functions."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from typing import Any


def _to_utc_iso(ts: datetime | None) -> str:
    if ts is None:
        ts = datetime.now(UTC)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return ts.astimezone(UTC).replace(microsecond=0).isoformat()


def _dedupe_key(
    event_type: str,
    entity_id: str | None,
    state: str | None,
    occurred_at_utc: str,
) -> str:
    payload = f"{event_type}|{entity_id or ''}|{state or ''}|{occurred_at_utc}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def record_raw_event(
    conn: sqlite3.Connection,
    *,
    event_type: str,
    entity_id: str | None = None,
    state: str | None = None,
    attributes: dict[str, Any] | None = None,
    occurred_at: datetime | None = None,
) -> int | None:
    occurred_at_utc = _to_utc_iso(occurred_at)
    dedupe_key = _dedupe_key(event_type, entity_id, state, occurred_at_utc)
    cur = conn.execute(
        """
        INSERT OR IGNORE INTO raw_events(
            event_type,
            entity_id,
            state,
            attributes_json,
            occurred_at_utc,
            dedupe_key,
            created_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            event_type,
            entity_id,
            state,
            json.dumps(attributes or {}, sort_keys=True),
            occurred_at_utc,
            dedupe_key,
            datetime.now(UTC).replace(microsecond=0).isoformat(),
        ),
    )
    conn.commit()
    if cur.rowcount == 0:
        return None
    return int(cur.lastrowid)
