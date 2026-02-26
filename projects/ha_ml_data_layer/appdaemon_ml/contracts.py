"""Contract and pairing helpers for pull-only consumers."""

from __future__ import annotations

import sqlite3


def get_valid_feature_label_pairs(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    rows = conn.execute(
        """
        SELECT
            f.id AS feature_id,
            l.id AS label_id,
            f.feature_name,
            f.feature_value,
            f.window_end_utc,
            l.label_end_utc
        FROM features f
        JOIN labels l
          ON f.window_end_utc <= l.label_end_utc
        ORDER BY f.window_end_utc ASC, f.id ASC
        """
    ).fetchall()
    return list(rows)


def validate_pull_contracts(conn: sqlite3.Connection) -> dict[str, object]:
    contract_version = conn.execute(
        "SELECT value FROM metadata WHERE key = 'contract_version'"
    ).fetchone()
    views = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'view' ORDER BY name ASC"
    ).fetchall()
    view_names = [row["name"] for row in views]
    return {
        "contract_version": contract_version["value"] if contract_version else None,
        "views": view_names,
    }
