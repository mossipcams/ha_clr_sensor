from __future__ import annotations

import json
from pathlib import Path

from appdaemon_ml.bocpd_train import run_bocpd_state_job
from appdaemon_ml.db import connect, ensure_schema


def test_bocpd_training_state_job_persists_state_and_run(tmp_path: Path) -> None:
    db_path = tmp_path / "ha_ml_data_layer.db"
    ensure_schema(db_path)
    conn = connect(db_path)
    try:
        conn.execute(
            """
            INSERT INTO features(window_start_utc, window_end_utc, feature_set_version, feature_name, feature_value, computed_at_utc)
            VALUES
            ('2026-02-25T00:00:00+00:00', '2026-02-25T01:00:00+00:00', 'v1', 'event_count', 1, '2026-02-25T01:01:00+00:00'),
            ('2026-02-25T01:00:00+00:00', '2026-02-25T02:00:00+00:00', 'v1', 'event_count', 2, '2026-02-25T02:01:00+00:00'),
            ('2026-02-25T02:00:00+00:00', '2026-02-25T03:00:00+00:00', 'v1', 'event_count', 3, '2026-02-25T03:01:00+00:00')
            """
        )
        conn.commit()

        run_id = run_bocpd_state_job(conn, hazard_rate=0.2)
        assert run_id is not None

        run = conn.execute(
            "SELECT status, point_count FROM bocpd_training_runs WHERE id = ?",
            (run_id,),
        ).fetchone()
        assert run["status"] == "completed"
        assert run["point_count"] == 3

        state = conn.execute(
            "SELECT hazard_rate, state_json FROM bocpd_model_state WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        assert state["hazard_rate"] == 0.2
        payload = json.loads(state["state_json"])
        assert payload["count"] == 3
    finally:
        conn.close()
