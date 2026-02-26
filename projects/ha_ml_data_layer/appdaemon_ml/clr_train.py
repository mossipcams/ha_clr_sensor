"""Backward-compatible wrapper for LightGBM training jobs."""

from __future__ import annotations

import sqlite3
from appdaemon_ml.lightgbm_train import run_lightgbm_training_job


def run_clr_training_job(
    conn: sqlite3.Connection,
    *,
    min_labeled_rows: int = 20,
    min_labeled_days: int = 5,
) -> int | None:
    return run_lightgbm_training_job(
        conn,
        min_labeled_rows=min_labeled_rows,
        min_labeled_days=min_labeled_days,
    )
