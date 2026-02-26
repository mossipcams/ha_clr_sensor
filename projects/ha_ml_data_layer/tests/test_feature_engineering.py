from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from appdaemon_ml.db import connect, ensure_schema
from appdaemon_ml.features import compute_window_features
from appdaemon_ml.ingest import record_raw_event


def test_compute_window_features_is_deterministic(tmp_path: Path) -> None:
    db_path = tmp_path / "ha_ml_data_layer.db"
    ensure_schema(db_path)
    conn = connect(db_path)
    try:
        t1 = datetime(2026, 2, 25, 0, 5, tzinfo=UTC)
        t2 = datetime(2026, 2, 25, 0, 10, tzinfo=UTC)
        t3 = datetime(2026, 2, 25, 0, 20, tzinfo=UTC)
        record_raw_event(conn, event_type="state_changed", entity_id="a", state="on", occurred_at=t1)
        record_raw_event(conn, event_type="state_changed", entity_id="a", state="off", occurred_at=t2)
        record_raw_event(conn, event_type="state_changed", entity_id="b", state="on", occurred_at=t3)

        count = compute_window_features(
            conn,
            window_start=t1,
            window_end=datetime(2026, 2, 25, 1, 0, tzinfo=UTC),
            feature_set_version="v1",
        )
        assert count == 2

        rows = conn.execute(
            "SELECT feature_name, feature_value FROM features ORDER BY feature_name ASC"
        ).fetchall()
        assert rows[0]["feature_name"] == "event_count"
        assert rows[0]["feature_value"] == 3
        assert rows[1]["feature_name"] == "on_ratio"
        assert rows[1]["feature_value"] == 2 / 3
    finally:
        conn.close()
