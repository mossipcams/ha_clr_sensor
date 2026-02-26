from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

from homeassistant.core import State

from custom_components.calibrated_logistic_regression.sensor import CalibratedLogisticRegressionSensor


def test_config_entry_to_sensor_probability_smoke_path(monkeypatch) -> None:
    hass = MagicMock()
    hass.states.get.side_effect = lambda entity_id: {
        "sensor.temperature": State("sensor.temperature", "21.5"),
        "sensor.humidity": State("sensor.humidity", "55"),
    }.get(entity_id)

    class _Provider:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def load(self):
            from custom_components.calibrated_logistic_regression.lightgbm_inference import LightGBMModelSpec
            from custom_components.calibrated_logistic_regression.model_provider import ModelProviderResult

            return ModelProviderResult(
                model=LightGBMModelSpec(
                    feature_names=["sensor.temperature", "sensor.humidity"],
                    model_payload={"intercept": -8.0, "weights": [0.2, 0.1]},
                ),
                source="ml_data_layer",
                artifact_error=None,
                artifact_meta={},
            )

    monkeypatch.setattr(
        "custom_components.calibrated_logistic_regression.sensor.SqliteLightGBMModelProvider",
        _Provider,
    )

    entry = MagicMock()
    entry.entry_id = "entry-smoke"
    entry.title = "Living Room Risk"
    entry.data = {
        "name": "Living Room Risk",
        "required_features": ["sensor.temperature", "sensor.humidity"],
        "threshold": 50.0,
        "ml_db_path": "/tmp/ha_ml_data_layer.db",
    }
    entry.options = {}

    sensor = CalibratedLogisticRegressionSensor(hass, entry)
    sensor._recompute_state(datetime.now())

    assert sensor.available is True
    assert sensor.native_value is not None
    assert 0.0 <= sensor.native_value <= 100.0
    attrs = sensor.extra_state_attributes
    assert attrs["missing_features"] == []
    assert attrs["feature_values"]["sensor.temperature"] == 21.5
