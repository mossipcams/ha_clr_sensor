from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from appdaemon_ml.app import AppDaemonMLDataLayer
from appdaemon_ml.db import connect


def test_e2e_data_layer_flow(tmp_path: Path) -> None:
    db_path = tmp_path / "ha_ml_data_layer.db"
    app = AppDaemonMLDataLayer(db_path=db_path, timezone_name="UTC")
    app.initialize()

    app.handle_event(
        event_type="state_changed",
        entity_id="sensor.bedroom",
        state="on",
        occurred_at=datetime(2026, 2, 25, 0, 5, tzinfo=UTC),
    )
    app.handle_event(
        event_type="state_changed",
        entity_id="sensor.bedroom",
        state="off",
        occurred_at=datetime(2026, 2, 25, 0, 20, tzinfo=UTC),
    )

    app.run_nightly_pipeline(
        local_date="2026-02-25",
        sleep_start="23:00:00",
        sleep_end="06:00:00",
        window_start=datetime(2026, 2, 25, 0, 0, tzinfo=UTC),
        window_end=datetime(2026, 2, 26, 0, 0, tzinfo=UTC),
    )

    conn = connect(db_path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM features").fetchone()[0] > 0
        assert conn.execute("SELECT COUNT(*) FROM labels").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM lightgbm_training_runs").fetchone()[0] >= 1
        assert conn.execute("SELECT COUNT(*) FROM bocpd_training_runs").fetchone()[0] >= 1
    finally:
        conn.close()
