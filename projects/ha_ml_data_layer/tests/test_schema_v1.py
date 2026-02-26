from __future__ import annotations

import sqlite3
from pathlib import Path

from appdaemon_ml.db import ensure_schema


def _table_names(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    return {row[0] for row in rows}


def _view_names(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='view'").fetchall()
    return {row[0] for row in rows}


def test_ensure_schema_creates_v1_tables_views_and_metadata(tmp_path: Path) -> None:
    db_path = tmp_path / "ha_ml_data_layer.db"
    ensure_schema(db_path, target_version=1)
    conn = sqlite3.connect(db_path)
    try:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] >= 5000

        expected_tables = {
            "raw_events",
            "features",
            "labels",
            "lightgbm_training_runs",
            "lightgbm_model_artifacts",
            "bocpd_training_runs",
            "bocpd_model_state",
            "metadata",
        }
        assert expected_tables.issubset(_table_names(conn))

        expected_views = {
            "vw_lightgbm_training_dataset",
            "vw_lightgbm_latest_model_artifact",
            "vw_bocpd_feature_stream",
            "vw_bocpd_latest_state",
            "vw_latest_feature_snapshot",
        }
        assert expected_views.issubset(_view_names(conn))

        keys = {
            row[0]: row[1]
            for row in conn.execute("SELECT key, value FROM metadata").fetchall()
        }
        assert keys["schema_version"] == "1"
        assert keys["feature_set_version"] == "v1"
        assert keys["contract_version"] == "1"
    finally:
        conn.close()
