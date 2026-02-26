from __future__ import annotations

import sys
import types
from datetime import datetime
from unittest.mock import MagicMock

# Minimal Home Assistant stubs required to import sensor module.
homeassistant = types.ModuleType("homeassistant")
components = types.ModuleType("homeassistant.components")
sensor_component = types.ModuleType("homeassistant.components.sensor")
config_entries = types.ModuleType("homeassistant.config_entries")
core = types.ModuleType("homeassistant.core")
entity_platform = types.ModuleType("homeassistant.helpers.entity_platform")
event_helpers = types.ModuleType("homeassistant.helpers.event")
restore_state = types.ModuleType("homeassistant.helpers.restore_state")


class SensorEntity:
    async def async_added_to_hass(self) -> None:
        return None

    def async_on_remove(self, remove_callback):
        return None

    def async_write_ha_state(self) -> None:
        return None


class SensorStateClass:
    MEASUREMENT = "measurement"


class RestoreEntity:
    async def async_get_last_state(self):
        return None


sensor_component.SensorEntity = SensorEntity
sensor_component.SensorStateClass = SensorStateClass
config_entries.ConfigEntry = object
core.Event = object
core.HomeAssistant = object
core.callback = lambda fn: fn
entity_platform.AddEntitiesCallback = object
event_helpers.async_track_state_change_event = lambda hass, entities, cb: lambda: None
restore_state.RestoreEntity = RestoreEntity

sys.modules.setdefault("homeassistant", homeassistant)
sys.modules.setdefault("homeassistant.components", components)
sys.modules.setdefault("homeassistant.components.sensor", sensor_component)
sys.modules.setdefault("homeassistant.config_entries", config_entries)
sys.modules.setdefault("homeassistant.core", core)
sys.modules.setdefault("homeassistant.helpers.entity_platform", entity_platform)
sys.modules.setdefault("homeassistant.helpers.event", event_helpers)
sys.modules.setdefault("homeassistant.helpers.restore_state", restore_state)

from custom_components.calibrated_logistic_regression.sensor import CalibratedLogisticRegressionSensor


class _ModelProvider:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def load(self):
        from custom_components.calibrated_logistic_regression.lightgbm_inference import LightGBMModelSpec
        from custom_components.calibrated_logistic_regression.model_provider import ModelProviderResult

        return ModelProviderResult(
            model=LightGBMModelSpec(
                feature_names=["event_count"],
                model_payload={"intercept": -1.0, "weights": [1.0]},
            ),
            source="ml_data_layer",
            artifact_error=None,
            artifact_meta={"model_type": "lightgbm_like"},
        )


class _FeatureProvider:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def load(self):
        from custom_components.calibrated_logistic_regression.feature_provider import FeatureVectorResult

        return FeatureVectorResult(
            feature_values={"event_count": 3.0},
            missing_features=[],
            mapped_state_values={},
        )


def test_sensor_can_use_ml_snapshot_feature_source(monkeypatch) -> None:
    monkeypatch.setattr(
        "custom_components.calibrated_logistic_regression.sensor.SqliteLightGBMModelProvider",
        _ModelProvider,
    )
    monkeypatch.setattr(
        "custom_components.calibrated_logistic_regression.sensor.SqliteSnapshotFeatureProvider",
        _FeatureProvider,
    )

    hass = MagicMock()
    entry = MagicMock()
    entry.entry_id = "entry-ml"
    entry.title = "ML MindML"
    entry.data = {
        "name": "ML MindML",
        "required_features": ["event_count"],
        "threshold": 50.0,
        "ml_db_path": "/tmp/ha_ml_data_layer.db",
        "ml_artifact_view": "vw_clr_latest_model_artifact",
        "ml_feature_source": "ml_snapshot",
        "ml_feature_view": "vw_latest_feature_snapshot",
    }
    entry.options = {}

    sensor = CalibratedLogisticRegressionSensor(hass, entry)
    sensor._recompute_state(datetime.now())

    attrs = sensor.extra_state_attributes
    assert sensor.available is True
    assert attrs["model_source"] == "ml_data_layer"
    assert attrs["feature_source"] == "ml_snapshot"
    assert attrs["model_runtime"] == "lightgbm"
    assert attrs["missing_features"] == []
    assert attrs["feature_values"] == {"event_count": 3.0}
