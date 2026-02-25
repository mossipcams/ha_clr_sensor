"""SQLite CLR artifact loader for ML-data-layer integration."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
import re

from .const import DEFAULT_ML_ARTIFACT_VIEW


@dataclass(slots=True)
class ClrModelArtifact:
    """Parsed CLR model artifact payload."""

    intercept: float
    coefficients: dict[str, float]
    feature_names: list[str]
    model_type: str
    feature_set_version: str
    created_at_utc: str | None


def load_latest_clr_model_artifact(
    db_path: str,
    artifact_view: str = DEFAULT_ML_ARTIFACT_VIEW,
) -> ClrModelArtifact:
    """Load and parse latest CLR model artifact from SQLite contract view."""
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
        raise ValueError("No CLR artifact row available")

    payload = json.loads(row["artifact_json"])
    model = dict(payload.get("model", {}))
    feature_names = [str(name) for name in payload.get("feature_names", [])]
    coefficients = [float(value) for value in model.get("coefficients", [])]
    intercept = float(model.get("intercept", 0.0))
    if len(feature_names) != len(coefficients):
        raise ValueError("feature_names and coefficients length mismatch")

    return ClrModelArtifact(
        intercept=intercept,
        coefficients={name: coef for name, coef in zip(feature_names, coefficients)},
        feature_names=feature_names,
        model_type=str(row["model_type"]),
        feature_set_version=str(row["feature_set_version"]),
        created_at_utc=row["created_at_utc"],
    )
