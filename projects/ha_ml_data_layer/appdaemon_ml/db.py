"""SQLite schema/bootstrap helpers for the AppDaemon ML data layer."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def connect(db_path: Path | str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def _create_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS raw_events (
            id INTEGER PRIMARY KEY,
            event_type TEXT NOT NULL,
            entity_id TEXT,
            state TEXT,
            attributes_json TEXT,
            occurred_at_utc TEXT NOT NULL,
            dedupe_key TEXT NOT NULL UNIQUE,
            created_at_utc TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_raw_events_occurred_at
            ON raw_events (occurred_at_utc);

        CREATE TABLE IF NOT EXISTS features (
            id INTEGER PRIMARY KEY,
            window_start_utc TEXT NOT NULL,
            window_end_utc TEXT NOT NULL,
            feature_set_version TEXT NOT NULL,
            feature_name TEXT NOT NULL,
            feature_value REAL NOT NULL,
            computed_at_utc TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_features_window_end
            ON features (window_end_utc);
        CREATE INDEX IF NOT EXISTS idx_features_name_window
            ON features (feature_name, window_end_utc);

        CREATE TABLE IF NOT EXISTS labels (
            id INTEGER PRIMARY KEY,
            label_start_utc TEXT NOT NULL,
            label_end_utc TEXT NOT NULL,
            local_date TEXT NOT NULL,
            timezone TEXT NOT NULL,
            source TEXT NOT NULL,
            created_at_utc TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_labels_end ON labels (label_end_utc);

        CREATE TABLE IF NOT EXISTS lightgbm_training_runs (
            id INTEGER PRIMARY KEY,
            started_at_utc TEXT NOT NULL,
            finished_at_utc TEXT,
            status TEXT NOT NULL,
            row_count INTEGER NOT NULL DEFAULT 0,
            day_count INTEGER NOT NULL DEFAULT 0,
            notes TEXT
        );

        CREATE TABLE IF NOT EXISTS lightgbm_model_artifacts (
            id INTEGER PRIMARY KEY,
            run_id INTEGER NOT NULL,
            created_at_utc TEXT NOT NULL,
            model_type TEXT NOT NULL,
            feature_set_version TEXT NOT NULL,
            artifact_json TEXT NOT NULL,
            FOREIGN KEY (run_id) REFERENCES lightgbm_training_runs(id)
        );
        CREATE INDEX IF NOT EXISTS idx_lightgbm_model_artifacts_created
            ON lightgbm_model_artifacts (created_at_utc);

        CREATE TABLE IF NOT EXISTS bocpd_training_runs (
            id INTEGER PRIMARY KEY,
            started_at_utc TEXT NOT NULL,
            finished_at_utc TEXT,
            status TEXT NOT NULL,
            point_count INTEGER NOT NULL DEFAULT 0,
            notes TEXT
        );

        CREATE TABLE IF NOT EXISTS bocpd_model_state (
            id INTEGER PRIMARY KEY,
            run_id INTEGER NOT NULL,
            created_at_utc TEXT NOT NULL,
            hazard_rate REAL NOT NULL,
            state_json TEXT NOT NULL,
            FOREIGN KEY (run_id) REFERENCES bocpd_training_runs(id)
        );
        CREATE INDEX IF NOT EXISTS idx_bocpd_state_created
            ON bocpd_model_state (created_at_utc);

        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at_utc TEXT NOT NULL
        );
        """
    )


def _create_views(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE VIEW IF NOT EXISTS vw_lightgbm_training_dataset AS
        SELECT
            l.id AS label_id,
            f.id AS feature_id,
            f.feature_name,
            f.feature_value,
            f.window_end_utc,
            l.label_end_utc,
            l.local_date,
            CASE
                WHEN l.source = 'sleep_window' THEN 1
                ELSE 0
            END AS target
        FROM features f
        JOIN labels l
          ON f.window_end_utc <= l.label_end_utc;

        CREATE VIEW IF NOT EXISTS vw_lightgbm_latest_model_artifact AS
        SELECT *
        FROM lightgbm_model_artifacts
        ORDER BY created_at_utc DESC, id DESC
        LIMIT 1;

        CREATE VIEW IF NOT EXISTS vw_bocpd_feature_stream AS
        SELECT
            id,
            window_end_utc AS ts_utc,
            feature_name,
            feature_value,
            feature_set_version
        FROM features
        ORDER BY window_end_utc ASC, id ASC;

        CREATE VIEW IF NOT EXISTS vw_bocpd_latest_state AS
        SELECT *
        FROM bocpd_model_state
        ORDER BY created_at_utc DESC, id DESC
        LIMIT 1;

        CREATE VIEW IF NOT EXISTS vw_latest_feature_snapshot AS
        SELECT f.*
        FROM features f
        JOIN (
          SELECT feature_name, MAX(window_end_utc) AS max_end
          FROM features
          GROUP BY feature_name
        ) latest
          ON latest.feature_name = f.feature_name
         AND latest.max_end = f.window_end_utc;
        """
    )


def _set_metadata(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        """
        INSERT INTO metadata(key, value, updated_at_utc)
        VALUES (?, ?, ?)
        ON CONFLICT(key) DO UPDATE SET
            value = excluded.value,
            updated_at_utc = excluded.updated_at_utc
        """,
        (key, value, _utc_now()),
    )


def ensure_schema(db_path: Path | str, target_version: int = 1) -> None:
    if target_version != 1:
        raise ValueError("Only schema version 1 is supported.")

    conn = connect(db_path)
    try:
        _create_tables(conn)
        _create_views(conn)
        _set_metadata(conn, "schema_version", str(target_version))
        _set_metadata(conn, "feature_set_version", "v1")
        _set_metadata(conn, "contract_version", "1")
        conn.commit()
    finally:
        conn.close()


def run_retention_maintenance(
    conn: sqlite3.Connection,
    *,
    now: datetime | None = None,
    raw_days: int = 30,
    feature_days: int = 90,
) -> None:
    if now is None:
        now = datetime.now(UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    raw_cutoff = (now - timedelta(days=raw_days)).replace(microsecond=0).isoformat()
    feature_cutoff = (now - timedelta(days=feature_days)).replace(microsecond=0).isoformat()

    conn.execute("DELETE FROM raw_events WHERE occurred_at_utc < ?", (raw_cutoff,))
    conn.execute("DELETE FROM features WHERE window_end_utc < ?", (feature_cutoff,))
    _set_metadata(conn, "last_retention_at", _utc_now())
    conn.commit()
