from __future__ import annotations

from pathlib import Path

from appdaemon_ml.app import get_diagnostics
from appdaemon_ml.db import connect, ensure_schema


def test_diagnostics_report_readiness_and_degraded_state(tmp_path: Path) -> None:
    db_path = tmp_path / "ha_ml_data_layer.db"
    ensure_schema(db_path)
    conn = connect(db_path)
    try:
        conn.execute(
            """
            INSERT INTO raw_events(event_type, entity_id, state, attributes_json, occurred_at_utc, dedupe_key, created_at_utc)
            VALUES ('state_changed','sensor.a','on','{}','2026-02-25T01:00:00+00:00','a','2026-02-25T01:00:00+00:00')
            """
        )
        conn.execute(
            """
            INSERT INTO lightgbm_training_runs(started_at_utc, finished_at_utc, status, row_count, day_count, notes)
            VALUES ('2026-02-25T01:00:00+00:00','2026-02-25T01:05:00+00:00','failed',0,0,'boom')
            """
        )
        conn.commit()

        diag = get_diagnostics(conn)
        assert diag["raw_event_count"] == 1
        assert diag["degraded"] is True
        assert "lightgbm_last_status" in diag
    finally:
        conn.close()
