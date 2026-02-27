from __future__ import annotations

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

from homeassistant.core import State

from custom_components.mindml.const import DOMAIN
from custom_components.mindml.sensor import (
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
            from custom_components.mindml.lightgbm_inference import LightGBMModelSpec
            from custom_components.mindml.model_provider import ModelProviderResult

            return ModelProviderResult(
                model=LightGBMModelSpec(feature_names=["sensor.a", "sensor.b"], model_payload={"intercept": 0.0, "weights": [1.0, 1.0]}),
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
            from custom_components.mindml.lightgbm_inference import LightGBMModelSpec
            from custom_components.mindml.model_provider import ModelProviderResult

            return ModelProviderResult(
                model=LightGBMModelSpec(feature_names=["sensor.a", "sensor.b"], model_payload={"intercept": -1.0, "weights": [1.0, 0.0]}),
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
            from custom_components.mindml.lightgbm_inference import LightGBMModelSpec
            from custom_components.mindml.model_provider import ModelProviderResult

            return ModelProviderResult(
                model=LightGBMModelSpec(feature_names=["sensor.a"], model_payload={"intercept": 0.0, "weights": [0.0]}),
                source="manual",
                artifact_error="missing artifact",
                artifact_meta={},
            )

    monkeypatch.setattr(
        "custom_components.mindml.sensor.SqliteLightGBMModelProvider",
        _Provider,
    )

    sensor = CalibratedLogisticRegressionSensor(hass, entry)

    assert _Provider.last_kwargs["db_path"] == "/config/appdaemon/ha_ml_data_layer.db"
    assert sensor.extra_state_attributes["model_runtime"] == "lightgbm"


def test_sensor_hass_state_keeps_configured_features_when_model_has_abstract_names(
    monkeypatch,
) -> None:
    hass = MagicMock()
    entry = _build_entry()

    class _Provider:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def load(self):
            from custom_components.mindml.lightgbm_inference import LightGBMModelSpec
            from custom_components.mindml.model_provider import ModelProviderResult

            return ModelProviderResult(
                model=LightGBMModelSpec(
                    feature_names=["event_count", "on_ratio"],
                    model_payload={"intercept": 0.0, "weights": [0.0, 0.0]},
                ),
                source="ml_data_layer",
                artifact_error=None,
                artifact_meta={},
            )

    monkeypatch.setattr(
        "custom_components.mindml.sensor.SqliteLightGBMModelProvider",
        _Provider,
    )

    sensor = CalibratedLogisticRegressionSensor(hass, entry)
    assert sensor._required_features == ["sensor.a", "sensor.b"]


def test_sensor_restores_state_from_last_known(monkeypatch) -> None:
    hass = MagicMock()
    entry = _build_entry()

    class _Provider:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def load(self):
            from custom_components.mindml.lightgbm_inference import LightGBMModelSpec
            from custom_components.mindml.model_provider import ModelProviderResult

            return ModelProviderResult(
                model=LightGBMModelSpec(
                    feature_names=["sensor.a", "sensor.b"],
                    model_payload={"intercept": 0.0, "weights": [0.0, 0.0]},
                ),
                source="ml_data_layer",
                artifact_error=None,
                artifact_meta={},
            )

    class _LastState:
        state = "72.5"
        attributes = {
            "raw_probability": 0.725,
            "linear_score": 1.23,
            "feature_values": {"sensor.a": 1.0},
            "feature_contributions": {"sensor.a": 0.5},
            "missing_features": ["sensor.b"],
            "last_computed_at": "2026-02-26T00:00:00+00:00",
            "is_above_threshold": True,
            "decision": "positive",
        }

    monkeypatch.setattr(
        "custom_components.mindml.sensor.SqliteLightGBMModelProvider",
        _Provider,
    )

    sensor = CalibratedLogisticRegressionSensor(hass, entry)
    sensor.async_get_last_state = AsyncMock(return_value=_LastState())
    sensor._recompute_state = lambda now: None

    asyncio.run(sensor.async_added_to_hass())

    assert sensor.native_value == 72.5
    attrs = sensor.extra_state_attributes
    assert attrs["raw_probability"] == 0.725
    assert attrs["linear_score"] == 1.23
    assert attrs["feature_values"] == {"sensor.a": 1.0}
    assert attrs["feature_contributions"] == {"sensor.a": 0.5}
    assert attrs["missing_features"] == ["sensor.b"]
    assert attrs["last_computed_at"] == "2026-02-26T00:00:00+00:00"
    assert attrs["is_above_threshold"] is True
    assert attrs["decision"] == "positive"


def test_sensor_handles_no_previous_state(monkeypatch) -> None:
    hass = MagicMock()
    entry = _build_entry()

    class _Provider:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def load(self):
            from custom_components.mindml.lightgbm_inference import LightGBMModelSpec
            from custom_components.mindml.model_provider import ModelProviderResult

            return ModelProviderResult(
                model=LightGBMModelSpec(
                    feature_names=["sensor.a", "sensor.b"],
                    model_payload={"intercept": 0.0, "weights": [0.0, 0.0]},
                ),
                source="ml_data_layer",
                artifact_error=None,
                artifact_meta={},
            )

    monkeypatch.setattr(
        "custom_components.mindml.sensor.SqliteLightGBMModelProvider",
        _Provider,
    )

    sensor = CalibratedLogisticRegressionSensor(hass, entry)
    sensor.async_get_last_state = AsyncMock(return_value=None)
    sensor._recompute_state = lambda now: None

    asyncio.run(sensor.async_added_to_hass())

    assert sensor.native_value is None
    assert sensor.available is False


def test_sensor_surfaces_model_artifact_error_reason(monkeypatch) -> None:
    hass = MagicMock()
    hass.states.get.side_effect = lambda entity_id: {
        "sensor.a": State("sensor.a", "2"),
        "sensor.b": State("sensor.b", "1"),
    }.get(entity_id)

    class _Provider:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def load(self):
            from custom_components.mindml.lightgbm_inference import LightGBMModelSpec
            from custom_components.mindml.model_provider import ModelProviderResult

            return ModelProviderResult(
                model=LightGBMModelSpec(feature_names=["sensor.a", "sensor.b"], model_payload={}),
                source="manual",
                artifact_error="db_not_found: /config/appdaemon/ha_ml_data_layer.db",
                artifact_meta={},
            )

    monkeypatch.setattr(
        "custom_components.mindml.sensor.SqliteLightGBMModelProvider",
        _Provider,
    )

    sensor = CalibratedLogisticRegressionSensor(hass, _build_entry())
    sensor._recompute_state(datetime.now())

    attrs = sensor.extra_state_attributes
    assert sensor.available is False
    assert attrs["model_artifact_error"] is not None
    assert attrs["unavailable_reason"] == "model_artifact_error"
