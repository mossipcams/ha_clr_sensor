from __future__ import annotations

import json
import sqlite3
import sys
import types
from pathlib import Path

homeassistant = types.ModuleType("homeassistant")
config_entries = types.ModuleType("homeassistant.config_entries")
core = types.ModuleType("homeassistant.core")
config_entries.ConfigEntry = object
core.HomeAssistant = object
sys.modules.setdefault("homeassistant", homeassistant)
sys.modules.setdefault("homeassistant.config_entries", config_entries)
sys.modules.setdefault("homeassistant.core", core)

from custom_components.calibrated_logistic_regression.ml_artifact import (
    load_latest_lightgbm_model_artifact,
)


def test_load_latest_lightgbm_model_artifact_parses_payload(tmp_path: Path) -> None:
    db_path = tmp_path / "ha_ml_data_layer.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE lightgbm_model_artifacts (
                id INTEGER PRIMARY KEY,
                created_at_utc TEXT NOT NULL,
                model_type TEXT NOT NULL,
                feature_set_version TEXT NOT NULL,
                artifact_json TEXT NOT NULL
            );
            CREATE VIEW vw_lightgbm_latest_model_artifact AS
            SELECT *
            FROM lightgbm_model_artifacts
            ORDER BY created_at_utc DESC, id DESC
            LIMIT 1;
            """
        )
        artifact_json = json.dumps(
            {
                "model": {
                    "type": "lightgbm_like",
                    "intercept": -1.25,
                    "weights": [0.5, 0.25],
                },
                "feature_names": ["event_count", "on_ratio"],
            }
        )
        conn.execute(
            """
            INSERT INTO lightgbm_model_artifacts(created_at_utc, model_type, feature_set_version, artifact_json)
            VALUES ('2026-02-25T00:00:00+00:00', 'lightgbm_like', 'v2', ?)
            """,
            (artifact_json,),
        )
        conn.commit()
    finally:
        conn.close()

    artifact = load_latest_lightgbm_model_artifact(str(db_path))
    assert artifact.model_type == "lightgbm_like"
    assert artifact.feature_set_version == "v2"
    assert artifact.feature_names == ["event_count", "on_ratio"]
    assert artifact.model_payload["type"] == "lightgbm_like"
    assert artifact.model_payload["intercept"] == -1.25
