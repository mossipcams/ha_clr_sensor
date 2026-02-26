from __future__ import annotations

from pathlib import Path

from appdaemon_ml.db import connect, ensure_schema
from appdaemon_ml.labels import capture_label_from_helpers


def test_capture_label_handles_cross_midnight(tmp_path: Path) -> None:
    db_path = tmp_path / "ha_ml_data_layer.db"
    ensure_schema(db_path)
    conn = connect(db_path)
    try:
        label_id = capture_label_from_helpers(
            conn,
            sleep_start="23:30:00",
            sleep_end="06:45:00",
            local_date="2026-02-25",
            timezone_name="UTC",
        )
        assert label_id > 0
        row = conn.execute(
            "SELECT label_start_utc, label_end_utc, local_date, timezone FROM labels"
        ).fetchone()
        assert row["label_start_utc"].startswith("2026-02-25T23:30:00")
        assert row["label_end_utc"].startswith("2026-02-26T06:45:00")
        assert row["local_date"] == "2026-02-25"
        assert row["timezone"] == "UTC"
    finally:
        conn.close()
