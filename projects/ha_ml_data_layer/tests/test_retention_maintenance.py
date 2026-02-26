from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from appdaemon_ml.db import connect, ensure_schema, run_retention_maintenance


def test_retention_trims_raw_and_features_preserving_labels(tmp_path: Path) -> None:
    db_path = tmp_path / "ha_ml_data_layer.db"
    ensure_schema(db_path)
    conn = connect(db_path)
    try:
        now = datetime(2026, 2, 25, 12, 0, tzinfo=UTC)
        old = (now - timedelta(days=10)).isoformat()
        recent = (now - timedelta(hours=2)).isoformat()

        conn.execute(
            """
            INSERT INTO raw_events(event_type, entity_id, state, attributes_json, occurred_at_utc, dedupe_key, created_at_utc)
            VALUES ('state_changed','sensor.a','on','{}', ?, 'old_raw', ?),
                   ('state_changed','sensor.a','off','{}', ?, 'new_raw', ?)
            """,
            (old, old, recent, recent),
        )
        conn.execute(
            """
            INSERT INTO features(window_start_utc, window_end_utc, feature_set_version, feature_name, feature_value, computed_at_utc)
            VALUES (?, ?, 'v1','event_count',1, ?),
                   (?, ?, 'v1','event_count',2, ?)
            """,
            (old, old, old, recent, recent, recent),
        )
        conn.execute(
            """
            INSERT INTO labels(label_start_utc, label_end_utc, local_date, timezone, source, created_at_utc)
            VALUES ('2026-02-24T23:00:00+00:00','2026-02-25T06:00:00+00:00','2026-02-24','UTC','sleep_window', ?)
            """,
            (recent,),
        )
        conn.commit()

        run_retention_maintenance(conn, now=now, raw_days=7, feature_days=7)

        raw_count = conn.execute("SELECT COUNT(*) FROM raw_events").fetchone()[0]
        feat_count = conn.execute("SELECT COUNT(*) FROM features").fetchone()[0]
        lbl_count = conn.execute("SELECT COUNT(*) FROM labels").fetchone()[0]
        assert raw_count == 1
        assert feat_count == 1
        assert lbl_count == 1
    finally:
        conn.close()
