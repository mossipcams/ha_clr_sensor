"""BOCPD-like training/state update routines."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def run_bocpd_state_job(
    conn: sqlite3.Connection,
    *,
    hazard_rate: float = 0.1,
) -> int | None:
    started_at = _utc_now()
    run_cur = conn.execute(
        """
        INSERT INTO bocpd_training_runs(started_at_utc, status, point_count, notes)
        VALUES (?, ?, 0, ?)
        """,
        (started_at, "started", ""),
    )
    run_id = int(run_cur.lastrowid)

    rows = conn.execute(
        """
        SELECT feature_value
        FROM vw_bocpd_feature_stream
        WHERE feature_name = 'event_count'
        ORDER BY ts_utc ASC, id ASC
        """
    ).fetchall()
    values = [float(row["feature_value"]) for row in rows]
    point_count = len(values)
    if point_count == 0:
        conn.execute(
            """
            UPDATE bocpd_training_runs
            SET finished_at_utc = ?, status = ?, point_count = ?, notes = ?
            WHERE id = ?
            """,
            (_utc_now(), "skipped", 0, "no points", run_id),
        )
        conn.commit()
        return None

    mean_value = sum(values) / point_count
    state = {"count": point_count, "mean_event_count": mean_value}
    conn.execute(
        """
        INSERT INTO bocpd_model_state(run_id, created_at_utc, hazard_rate, state_json)
        VALUES (?, ?, ?, ?)
        """,
        (run_id, _utc_now(), float(hazard_rate), json.dumps(state, sort_keys=True)),
    )
    conn.execute(
        """
        UPDATE bocpd_training_runs
        SET finished_at_utc = ?, status = ?, point_count = ?, notes = ?
        WHERE id = ?
        """,
        (_utc_now(), "completed", point_count, "ok", run_id),
    )
    conn.commit()
    return run_id
