"""App lifecycle orchestration and diagnostics."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from appdaemon_ml.bocpd_train import run_bocpd_state_job
from appdaemon_ml.lightgbm_train import run_lightgbm_training_job
from appdaemon_ml.db import connect, ensure_schema
from appdaemon_ml.features import compute_window_features
from appdaemon_ml.ingest import record_raw_event
from appdaemon_ml.labels import capture_label_from_helpers


def get_diagnostics(conn: sqlite3.Connection) -> dict[str, object]:
    raw_event_count = conn.execute("SELECT COUNT(*) FROM raw_events").fetchone()[0]
    feature_count = conn.execute("SELECT COUNT(*) FROM features").fetchone()[0]
    label_count = conn.execute("SELECT COUNT(*) FROM labels").fetchone()[0]
    lightgbm_status = conn.execute(
        """
        SELECT status
        FROM lightgbm_training_runs
        ORDER BY id DESC
        LIMIT 1
        """
    ).fetchone()
    bocpd_status = conn.execute(
        """
        SELECT status
        FROM bocpd_training_runs
        ORDER BY id DESC
        LIMIT 1
        """
    ).fetchone()
    degraded = (lightgbm_status and lightgbm_status[0] == "failed") or (
        bocpd_status and bocpd_status[0] == "failed"
    )

    return {
        "timestamp_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "raw_event_count": raw_event_count,
        "feature_count": feature_count,
        "label_count": label_count,
        "lightgbm_last_status": lightgbm_status[0] if lightgbm_status else None,
        "bocpd_last_status": bocpd_status[0] if bocpd_status else None,
        "degraded": bool(degraded),
    }


@dataclass(slots=True)
class AppDaemonMLDataLayer:
    db_path: Path
    timezone_name: str = "UTC"

    def initialize(self) -> None:
        ensure_schema(self.db_path)

    def handle_event(
        self,
        *,
        event_type: str,
        entity_id: str | None = None,
        state: str | None = None,
        occurred_at: datetime | None = None,
    ) -> int | None:
        conn = connect(self.db_path)
        try:
            row_id = record_raw_event(
                conn,
                event_type=event_type,
                entity_id=entity_id,
                state=state,
                occurred_at=occurred_at,
            )
            conn.execute(
                """
                INSERT INTO metadata(key, value, updated_at_utc)
                VALUES ('last_ingest_at', ?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at_utc = excluded.updated_at_utc
                """,
                (
                    datetime.now(UTC).replace(microsecond=0).isoformat(),
                    datetime.now(UTC).replace(microsecond=0).isoformat(),
                ),
            )
            conn.commit()
            return row_id
        finally:
            conn.close()

    def run_nightly_pipeline(
        self,
        *,
        local_date: str,
        sleep_start: str,
        sleep_end: str,
        window_start: datetime,
        window_end: datetime,
    ) -> None:
        conn = connect(self.db_path)
        try:
            compute_window_features(
                conn,
                window_start=window_start,
                window_end=window_end,
                feature_set_version="v1",
            )
            capture_label_from_helpers(
                conn,
                sleep_start=sleep_start,
                sleep_end=sleep_end,
                local_date=local_date,
                timezone_name=self.timezone_name,
            )
            run_lightgbm_training_job(conn, min_labeled_rows=1, min_labeled_days=1)
            run_bocpd_state_job(conn)
        finally:
            conn.close()
