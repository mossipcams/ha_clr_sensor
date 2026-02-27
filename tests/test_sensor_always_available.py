"""Sensors must stay available so diagnostic attributes remain visible in HA.

When ``available`` returns False, Home Assistant strips all attributes from
the state machine.  This creates a catch-22: the very diagnostics that
explain *why* the sensor cannot compute a value (``unavailable_reason``,
``model_artifact_error``, ``missing_features``, etc.) become invisible.

The correct HA pattern for "I cannot produce a value right now" is
``native_value = None`` (displayed as *unknown*) with the entity staying
available so attributes remain accessible.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

from homeassistant.core import State

from custom_components.mindml.lightgbm_inference import LightGBMModelSpec
from custom_components.mindml.model_provider import ModelProviderResult
from custom_components.mindml.sensor import CalibratedLogisticRegressionSensor


def _build_entry() -> MagicMock:
    entry = MagicMock()
    entry.entry_id = "entry-1"
    entry.title = "Kitchen MindML"
    entry.data = {
        "name": "Kitchen MindML",
        "goal": "risk",
        "required_features": ["sensor.a", "sensor.b"],
        "feature_types": {"sensor.a": "numeric", "sensor.b": "numeric"},
        "threshold": 50.0,
        "state_mappings": {},
        "ml_db_path": "/tmp/ha_ml_data_layer.db",
        "ml_artifact_view": "vw_clr_latest_model_artifact",
        "ml_feature_source": "hass_state",
    }
    entry.options = {}
    return entry


def test_sensor_available_when_features_missing(monkeypatch) -> None:
    """Sensor must stay available when features are missing so attrs are visible."""
    hass = MagicMock()
    hass.states.get.side_effect = lambda entity_id: {
        "sensor.a": State("sensor.a", "4"),
    }.get(entity_id)  # sensor.b missing

    class _Provider:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def load(self):
            return ModelProviderResult(
                model=LightGBMModelSpec(
                    feature_names=["sensor.a", "sensor.b"],
                    model_payload={"intercept": 0.0, "weights": [1.0, 1.0]},
                ),
                source="ml_data_layer",
                artifact_error=None,
                artifact_meta={},
            )

    monkeypatch.setattr(
        "custom_components.mindml.sensor.SqliteLightGBMModelProvider",
        _Provider,
    )

    sensor = CalibratedLogisticRegressionSensor(hass, _build_entry())
    sensor._recompute_state(datetime.now())

    assert sensor.available is True
    assert sensor.native_value is None
    attrs = sensor.extra_state_attributes
    assert attrs["unavailable_reason"] == "missing_or_unmapped_features"
    assert "sensor.b" in attrs["missing_features"]
