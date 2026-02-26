from __future__ import annotations

from pathlib import Path

from appdaemon_ml.db import connect, ensure_schema


def test_bocpd_contract_views_are_queryable(tmp_path: Path) -> None:
    db_path = tmp_path / "ha_ml_data_layer.db"
    ensure_schema(db_path)
    conn = connect(db_path)
    try:
        conn.execute(
            """
            INSERT INTO features(window_start_utc, window_end_utc, feature_set_version, feature_name, feature_value, computed_at_utc)
            VALUES ('2026-02-25T00:00:00+00:00','2026-02-25T01:00:00+00:00','v1','event_count',3,'2026-02-25T01:01:00+00:00')
            """
        )
        conn.execute(
            """
            INSERT INTO bocpd_training_runs(started_at_utc, finished_at_utc, status, point_count, notes)
            VALUES ('2026-02-25T01:00:00+00:00','2026-02-25T01:01:00+00:00','completed',1,'ok')
            """
        )
        conn.execute(
            """
            INSERT INTO bocpd_model_state(run_id, created_at_utc, hazard_rate, state_json)
            VALUES (1,'2026-02-25T01:01:00+00:00',0.1,'{\"count\":1}')
            """
        )
        conn.commit()

        stream = conn.execute(
            "SELECT feature_name, feature_value FROM vw_bocpd_feature_stream"
        ).fetchone()
        assert stream["feature_name"] == "event_count"
        assert stream["feature_value"] == 3

        state = conn.execute(
            "SELECT hazard_rate, state_json FROM vw_bocpd_latest_state"
        ).fetchone()
        assert state["hazard_rate"] == 0.1
    finally:
        conn.close()
