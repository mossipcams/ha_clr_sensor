"""SQLite LightGBM artifact loader for ML-data-layer integration."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
import re

from .const import DEFAULT_ML_ARTIFACT_VIEW


@dataclass(slots=True)
class LightGBMModelArtifact:
    """Parsed LightGBM model artifact payload."""

    model_payload: dict[str, object]
    feature_names: list[str]
    model_type: str
    feature_set_version: str
    created_at_utc: str | None


def _load_latest_artifact_row(
    *,
    db_path: str,
    artifact_view: str,
) -> sqlite3.Row:
    """Read the latest artifact row from a configured contract view."""
    if not db_path:
        raise ValueError("ml_db_path is required")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", artifact_view):
        raise ValueError("Invalid artifact view name")

    db_file = Path(db_path)
    if not db_file.exists():
        raise FileNotFoundError(db_path)

    conn = sqlite3.connect(db_file)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            f"SELECT created_at_utc, model_type, feature_set_version, artifact_json FROM {artifact_view} LIMIT 1"
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        raise ValueError("No LightGBM artifact row available")
    return row


def load_latest_lightgbm_model_artifact(
    db_path: str,
    artifact_view: str = DEFAULT_ML_ARTIFACT_VIEW,
) -> LightGBMModelArtifact:
    """Load and parse latest LightGBM model artifact from SQLite contract view."""
    row = _load_latest_artifact_row(db_path=db_path, artifact_view=artifact_view)

    payload = json.loads(row["artifact_json"])
    model_payload = dict(payload.get("model", {}))
    feature_names = [str(name) for name in payload.get("feature_names", [])]

    return LightGBMModelArtifact(
        model_payload=model_payload,
        feature_names=feature_names,
        model_type=str(row["model_type"]),
        feature_set_version=str(row["feature_set_version"]),
        created_at_utc=row["created_at_utc"],
    )
