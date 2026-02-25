from __future__ import annotations

import sqlite3
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

homeassistant = types.ModuleType("homeassistant")
config_entries = types.ModuleType("homeassistant.config_entries")
core = types.ModuleType("homeassistant.core")
config_entries.ConfigEntry = object
core.HomeAssistant = object
sys.modules["homeassistant"] = homeassistant
sys.modules["homeassistant.config_entries"] = config_entries
sys.modules["homeassistant.core"] = core

from custom_components.calibrated_logistic_regression.feature_provider import (
    FeatureVectorResult,
    HassStateFeatureProvider,
    SqliteSnapshotFeatureProvider,
)


class _State:
    def __init__(self, state: str) -> None:
        self.state = state


def test_hass_feature_provider_maps_numeric_and_categorical_values() -> None:
    hass = MagicMock()
    hass.states.get.side_effect = lambda entity_id: {
        "sensor.temp": _State("21.5"),
        "binary_sensor.window": _State("on"),
    }.get(entity_id)

    provider = HassStateFeatureProvider(
        hass=hass,
        required_features=["sensor.temp", "binary_sensor.window"],
        feature_types={"sensor.temp": "numeric", "binary_sensor.window": "categorical"},
        state_mappings={"binary_sensor.window": {"on": 1.0, "off": 0.0}},
    )

    vector = provider.load()

    assert isinstance(vector, FeatureVectorResult)
    assert vector.feature_values == {"sensor.temp": 21.5, "binary_sensor.window": 1.0}
    assert vector.missing_features == []
    assert vector.mapped_state_values == {"binary_sensor.window": "on"}


def test_hass_feature_provider_marks_missing_when_state_unmapped() -> None:
    hass = MagicMock()
    hass.states.get.return_value = _State("unknown_status")

    provider = HassStateFeatureProvider(
        hass=hass,
        required_features=["sensor.status"],
        feature_types={"sensor.status": "categorical"},
        state_mappings={},
    )

    vector = provider.load()

    assert vector.feature_values == {}
    assert vector.missing_features == ["sensor.status"]


def test_sqlite_snapshot_feature_provider_reads_latest_feature_values(tmp_path: Path) -> None:
    db_path = tmp_path / "ha_ml_data_layer.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE features (
                id INTEGER PRIMARY KEY,
                window_end_utc TEXT NOT NULL,
                feature_name TEXT NOT NULL,
                feature_value REAL NOT NULL
            );
            CREATE VIEW vw_latest_feature_snapshot AS
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
        conn.execute(
            "INSERT INTO features(window_end_utc, feature_name, feature_value) VALUES ('2026-02-25T00:00:00+00:00', 'event_count', 5.0)"
        )
        conn.execute(
            "INSERT INTO features(window_end_utc, feature_name, feature_value) VALUES ('2026-02-26T00:00:00+00:00', 'event_count', 7.0)"
        )
        conn.commit()
    finally:
        conn.close()

    provider = SqliteSnapshotFeatureProvider(
        db_path=str(db_path),
        snapshot_view="vw_latest_feature_snapshot",
        required_features=["event_count", "on_ratio"],
    )

    vector = provider.load()

    assert vector.feature_values == {"event_count": 7.0}
    assert vector.missing_features == ["on_ratio"]
    assert vector.mapped_state_values == {}
