from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from appdaemon_ml.db import connect, ensure_schema
from appdaemon_ml.ingest import record_raw_event


def test_record_raw_event_inserts_and_dedupes(tmp_path: Path) -> None:
    db_path = tmp_path / "ha_ml_data_layer.db"
    ensure_schema(db_path)
    conn = connect(db_path)
    try:
        occurred = datetime(2026, 2, 25, 6, 0, tzinfo=UTC)
        row_id_1 = record_raw_event(
            conn,
            event_type="state_changed",
            entity_id="sensor.bedroom",
            state="on",
            occurred_at=occurred,
        )
        row_id_2 = record_raw_event(
            conn,
            event_type="state_changed",
            entity_id="sensor.bedroom",
            state="on",
            occurred_at=occurred,
        )
        assert row_id_1 is not None
        assert row_id_2 is None

        rows = conn.execute(
            "SELECT occurred_at_utc FROM raw_events ORDER BY id ASC"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][0].endswith("+00:00")
    finally:
        conn.close()
