"""Label capture from Home Assistant helper values."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo


def _parse_time_hms(value: str) -> tuple[int, int, int]:
    parts = value.split(":")
    if len(parts) != 3:
        raise ValueError("Time must be HH:MM:SS")
    return int(parts[0]), int(parts[1]), int(parts[2])


def capture_label_from_helpers(
    conn: sqlite3.Connection,
    *,
    sleep_start: str,
    sleep_end: str,
    local_date: str,
    timezone_name: str,
) -> int:
    tz = ZoneInfo(timezone_name)
    start_h, start_m, start_s = _parse_time_hms(sleep_start)
    end_h, end_m, end_s = _parse_time_hms(sleep_end)
    local_day = datetime.fromisoformat(local_date)

    start_local = local_day.replace(
        hour=start_h,
        minute=start_m,
        second=start_s,
        tzinfo=tz,
    )
    end_local = local_day.replace(
        hour=end_h,
        minute=end_m,
        second=end_s,
        tzinfo=tz,
    )
    if end_local <= start_local:
        end_local = end_local + timedelta(days=1)

    now_utc = datetime.now(UTC).replace(microsecond=0).isoformat()
    cur = conn.execute(
        """
        INSERT INTO labels(
            label_start_utc,
            label_end_utc,
            local_date,
            timezone,
            source,
            created_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            start_local.astimezone(UTC).replace(microsecond=0).isoformat(),
            end_local.astimezone(UTC).replace(microsecond=0).isoformat(),
            local_date,
            timezone_name,
            "sleep_window",
            now_utc,
        ),
    )
    conn.commit()
    return int(cur.lastrowid)
