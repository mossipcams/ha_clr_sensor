from __future__ import annotations

import asyncio
from datetime import datetime
from unittest.mock import MagicMock

from homeassistant.core import State

from custom_components.calibrated_logistic_regression.const import DOMAIN
from custom_components.calibrated_logistic_regression.sensor import (
    CalibratedLogisticRegressionSensor,
    async_setup_entry,
)


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


def test_async_setup_entry_adds_one_sensor() -> None:
    hass = MagicMock()
    hass.data = {DOMAIN: {}}
    entry = _build_entry()
    added = []

    asyncio.run(async_setup_entry(hass, entry, lambda entities: added.extend(entities)))

    assert len(added) == 1
    assert isinstance(added[0], CalibratedLogisticRegressionSensor)


def test_sensor_unavailable_reason_when_required_feature_missing(monkeypatch) -> None:
    hass = MagicMock()
    hass.states.get.side_effect = lambda entity_id: {
        "sensor.a": State("sensor.a", "4"),
    }.get(entity_id)

    class _Provider:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def load(self):
            from custom_components.calibrated_logistic_regression.lightgbm_inference import LightGBMModelSpec
            from custom_components.calibrated_logistic_regression.model_provider import ModelProviderResult

            return ModelProviderResult(
                model=LightGBMModelSpec(feature_names=["sensor.a", "sensor.b"], model_payload={"intercept": 0.0, "weights": [1.0, 1.0]}),
                source="ml_data_layer",
                artifact_error=None,
                artifact_meta={},
            )

    monkeypatch.setattr(
        "custom_components.calibrated_logistic_regression.sensor.SqliteLightGBMModelProvider",
        _Provider,
    )

    sensor = CalibratedLogisticRegressionSensor(hass, _build_entry())
    sensor._recompute_state(datetime.now())

    assert sensor.available is False
    attrs = sensor.extra_state_attributes
    assert attrs["missing_features"] == ["sensor.b"]
    assert attrs["unavailable_reason"] == "missing_or_unmapped_features"


def test_sensor_updates_probability_attributes(monkeypatch) -> None:
    hass = MagicMock()
    hass.states.get.side_effect = lambda entity_id: {
        "sensor.a": State("sensor.a", "2"),
        "sensor.b": State("sensor.b", "1"),
    }.get(entity_id)

    class _Provider:
        last_kwargs: dict[str, object] = {}

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            _Provider.last_kwargs = dict(kwargs)

        def load(self):
            from custom_components.calibrated_logistic_regression.lightgbm_inference import LightGBMModelSpec
            from custom_components.calibrated_logistic_regression.model_provider import ModelProviderResult

            return ModelProviderResult(
                model=LightGBMModelSpec(feature_names=["sensor.a", "sensor.b"], model_payload={"intercept": -1.0, "weights": [1.0, 0.0]}),
                source="ml_data_layer",
                artifact_error=None,
                artifact_meta={},
            )

    monkeypatch.setattr(
        "custom_components.calibrated_logistic_regression.sensor.SqliteLightGBMModelProvider",
        _Provider,
    )

    sensor = CalibratedLogisticRegressionSensor(hass, _build_entry())
    sensor._recompute_state(datetime.now())

    assert sensor.available is True
    assert sensor.native_value is not None
    attrs = sensor.extra_state_attributes
    assert attrs["model_runtime"] == "lightgbm"
    assert attrs["missing_features"] == []
    assert attrs["feature_values"]["sensor.a"] == 2.0
    assert attrs["decision"] in {"positive", "negative"}


def test_sensor_resolves_default_ml_db_path_when_missing(monkeypatch) -> None:
    hass = MagicMock()
    hass.states.get.return_value = State("sensor.a", "2")
    hass.config.path.return_value = "/config/ha_ml_data_layer.db"
    entry = _build_entry()
    entry.data.pop("ml_db_path")

    class _Provider:
        last_kwargs: dict[str, object] = {}

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            _Provider.last_kwargs = dict(kwargs)

        def load(self):
            from custom_components.calibrated_logistic_regression.lightgbm_inference import LightGBMModelSpec
            from custom_components.calibrated_logistic_regression.model_provider import ModelProviderResult

            return ModelProviderResult(
                model=LightGBMModelSpec(feature_names=["sensor.a"], model_payload={"intercept": 0.0, "weights": [0.0]}),
                source="manual",
                artifact_error="missing artifact",
                artifact_meta={},
            )

    monkeypatch.setattr(
        "custom_components.calibrated_logistic_regression.sensor.SqliteLightGBMModelProvider",
        _Provider,
    )

    sensor = CalibratedLogisticRegressionSensor(hass, entry)

    assert _Provider.last_kwargs["db_path"] == "/homeassistant/appdaemon/ha_ml_data_layer.db"
    assert sensor.extra_state_attributes["model_runtime"] == "lightgbm"
