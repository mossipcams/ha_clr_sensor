from __future__ import annotations

import json
from pathlib import Path

from appdaemon_ml.lightgbm_train import run_lightgbm_training_job
from appdaemon_ml.db import connect, ensure_schema


def test_lightgbm_training_job_persists_run_and_artifact(tmp_path: Path) -> None:
    db_path = tmp_path / "ha_ml_data_layer.db"
    ensure_schema(db_path)
    conn = connect(db_path)
    try:
        conn.execute(
            """
            INSERT INTO labels(label_start_utc, label_end_utc, local_date, timezone, source, created_at_utc)
            VALUES
            ('2026-02-25T23:00:00+00:00', '2026-02-26T06:00:00+00:00', '2026-02-25', 'UTC', 'sleep_window', '2026-02-26T06:01:00+00:00'),
            ('2026-02-26T23:00:00+00:00', '2026-02-27T06:00:00+00:00', '2026-02-26', 'UTC', 'manual_awake', '2026-02-27T06:01:00+00:00')
            """
        )
        conn.execute(
            """
            INSERT INTO features(window_start_utc, window_end_utc, feature_set_version, feature_name, feature_value, computed_at_utc)
            VALUES
            ('2026-02-25T22:00:00+00:00', '2026-02-26T05:30:00+00:00', 'v1', 'event_count', 10, '2026-02-26T05:31:00+00:00'),
            ('2026-02-26T22:00:00+00:00', '2026-02-27T05:30:00+00:00', 'v1', 'event_count', 1, '2026-02-27T05:31:00+00:00')
            """
        )
        conn.commit()

        run_id = run_lightgbm_training_job(conn, min_labeled_rows=2, min_labeled_days=2)
        assert run_id is not None

        run = conn.execute(
            "SELECT status, row_count, day_count FROM lightgbm_training_runs WHERE id = ?",
            (run_id,),
        ).fetchone()
        assert run["status"] == "completed"
        assert run["row_count"] >= 2
        assert run["day_count"] == 2

        artifact = conn.execute(
            "SELECT model_type, feature_set_version, artifact_json FROM lightgbm_model_artifacts WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        assert artifact["model_type"] == "lightgbm_like"
        payload = json.loads(artifact["artifact_json"])
        assert "model" in payload
        assert "weights" in payload["model"]
        assert "feature_names" in payload
    finally:
        conn.close()
