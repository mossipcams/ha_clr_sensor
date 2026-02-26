from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from appdaemon_ml.contracts import get_valid_feature_label_pairs
from appdaemon_ml.db import connect, ensure_schema
from appdaemon_ml.labels import capture_label_from_helpers


def test_pairing_rules_exclude_leakage_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "ha_ml_data_layer.db"
    ensure_schema(db_path)
    conn = connect(db_path)
    try:
        capture_label_from_helpers(
            conn,
            sleep_start="23:00:00",
            sleep_end="06:00:00",
            local_date="2026-02-25",
            timezone_name="UTC",
        )
        conn.execute(
            """
            INSERT INTO features(
                window_start_utc, window_end_utc, feature_set_version, feature_name, feature_value, computed_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "2026-02-25T05:00:00+00:00",
                "2026-02-26T05:59:00+00:00",
                "v1",
                "event_count",
                3.0,
                datetime(2026, 2, 26, 6, 0, tzinfo=UTC).isoformat(),
            ),
        )
        conn.execute(
            """
            INSERT INTO features(
                window_start_utc, window_end_utc, feature_set_version, feature_name, feature_value, computed_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "2026-02-26T06:00:00+00:00",
                "2026-02-26T06:01:00+00:00",
                "v1",
                "event_count",
                99.0,
                datetime(2026, 2, 26, 6, 2, tzinfo=UTC).isoformat(),
            ),
        )
        conn.commit()

        pairs = get_valid_feature_label_pairs(conn)
        assert len(pairs) == 1
        assert pairs[0]["feature_value"] == 3.0
    finally:
        conn.close()
