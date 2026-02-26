"""Feature engineering pipeline."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime


def _iso_utc(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).replace(microsecond=0).isoformat()


def compute_window_features(
    conn: sqlite3.Connection,
    *,
    window_start: datetime,
    window_end: datetime,
    feature_set_version: str = "v1",
) -> int:
    start_utc = _iso_utc(window_start)
    end_utc = _iso_utc(window_end)
    rows = conn.execute(
        """
        SELECT state
        FROM raw_events
        WHERE occurred_at_utc >= ?
          AND occurred_at_utc < ?
        ORDER BY occurred_at_utc ASC
        """,
        (start_utc, end_utc),
    ).fetchall()

    event_count = len(rows)
    on_count = sum(1 for row in rows if row["state"] == "on")
    on_ratio = (on_count / event_count) if event_count else 0.0
    computed_at = datetime.now(UTC).replace(microsecond=0).isoformat()

    data = [
        ("event_count", float(event_count)),
        ("on_ratio", float(on_ratio)),
    ]
    conn.executemany(
        """
        INSERT INTO features(
            window_start_utc,
            window_end_utc,
            feature_set_version,
            feature_name,
            feature_value,
            computed_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            (start_utc, end_utc, feature_set_version, name, value, computed_at)
            for name, value in data
        ],
    )
    conn.commit()
    return len(data)
