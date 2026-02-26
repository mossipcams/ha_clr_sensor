from __future__ import annotations

from pathlib import Path

from appdaemon_ml.db import connect, ensure_schema


def test_clr_contract_views_have_expected_shape(tmp_path: Path) -> None:
    db_path = tmp_path / "ha_ml_data_layer.db"
    ensure_schema(db_path)
    conn = connect(db_path)
    try:
        conn.execute(
            """
            INSERT INTO labels(label_start_utc, label_end_utc, local_date, timezone, source, created_at_utc)
            VALUES ('2026-02-25T23:00:00+00:00','2026-02-26T06:00:00+00:00','2026-02-25','UTC','sleep_window','2026-02-26T06:01:00+00:00')
            """
        )
        conn.execute(
            """
            INSERT INTO features(window_start_utc, window_end_utc, feature_set_version, feature_name, feature_value, computed_at_utc)
            VALUES ('2026-02-25T22:00:00+00:00','2026-02-26T05:30:00+00:00','v1','event_count',7,'2026-02-26T05:31:00+00:00')
            """
        )
        conn.execute(
            """
            INSERT INTO lightgbm_training_runs(started_at_utc, finished_at_utc, status, row_count, day_count, notes)
            VALUES ('2026-02-26T06:00:00+00:00','2026-02-26T06:01:00+00:00','completed',1,1,'ok')
            """
        )
        conn.execute(
            """
            INSERT INTO lightgbm_model_artifacts(run_id, created_at_utc, model_type, feature_set_version, artifact_json)
            VALUES (1,'2026-02-26T06:01:00+00:00','lightgbm_like','v1','{\"model\":{\"intercept\":0,\"weights\":[]},\"feature_names\":[]}')
            """
        )
        conn.commit()

        clr_row = conn.execute(
            "SELECT feature_name, feature_value, target FROM vw_lightgbm_training_dataset"
        ).fetchone()
        assert clr_row["feature_name"] == "event_count"
        assert clr_row["target"] == 1

        latest = conn.execute(
            "SELECT model_type, feature_set_version FROM vw_lightgbm_latest_model_artifact"
        ).fetchone()
        assert latest["model_type"] == "lightgbm_like"
        assert latest["feature_set_version"] == "v1"
    finally:
        conn.close()
