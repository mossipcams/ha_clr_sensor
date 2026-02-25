from __future__ import annotations

import asyncio
from datetime import datetime
from unittest.mock import MagicMock

from homeassistant.core import State

from custom_components.calibrated_logistic_regression.const import DOMAIN
from custom_components.calibrated_logistic_regression.ml_artifact import ClrModelArtifact
from custom_components.calibrated_logistic_regression.sensor import (
    CalibratedLogisticRegressionSensor,
    async_setup_entry,
)


def _build_entry() -> MagicMock:
    entry = MagicMock()
    entry.entry_id = "entry-1"
    entry.title = "Kitchen CLR"
    entry.data = {
        "name": "Kitchen CLR",
        "goal": "risk",
        "intercept": -0.2,
        "coefficients": {"sensor.a": 0.4, "sensor.b": -0.1},
        "feature_types": {"sensor.a": "numeric", "sensor.b": "numeric"},
        "calibration_slope": 1.2,
        "calibration_intercept": -0.05,
        "threshold": 50.0,
        "required_features": ["sensor.a", "sensor.b"],
        "state_mappings": {},
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


def test_sensor_unavailable_reason_when_required_feature_missing() -> None:
    hass = MagicMock()
    hass.states.get.side_effect = lambda entity_id: {
        "sensor.a": State("sensor.a", "4"),
    }.get(entity_id)

    sensor = CalibratedLogisticRegressionSensor(hass, _build_entry())
    sensor._recompute_state(datetime.now())

    assert sensor.available is False
    attrs = sensor.extra_state_attributes
    assert attrs["missing_features"] == ["sensor.b"]
    assert attrs["unavailable_reason"] == "missing_or_unmapped_features"
    assert attrs["decision_threshold"] == 50.0
    assert attrs["is_above_threshold"] is None
    assert attrs["decision"] is None


def test_sensor_updates_probability_and_explainability_attributes() -> None:
    hass = MagicMock()
    hass.states.get.side_effect = lambda entity_id: {
        "sensor.a": State("sensor.a", "2"),
        "sensor.b": State("sensor.b", "1"),
    }.get(entity_id)

    sensor = CalibratedLogisticRegressionSensor(hass, _build_entry())
    sensor._recompute_state(datetime.now())

    assert sensor.available is True
    assert sensor.native_value is not None
    attrs = sensor.extra_state_attributes
    assert "raw_probability" in attrs
    assert "linear_score" in attrs
    assert attrs["missing_features"] == []
    assert attrs["feature_values"]["sensor.a"] == 2.0
    assert attrs["feature_contributions"]["sensor.a"] == 0.8
    assert attrs["mapped_state_values"] == {}
    assert attrs["unavailable_reason"] is None
    assert attrs["last_computed_at"] is not None
    assert attrs["decision_threshold"] == 50.0
    assert attrs["is_above_threshold"] is True
    assert attrs["decision"] == "positive"


def test_sensor_uses_state_mapping_for_non_numeric_state() -> None:
    hass = MagicMock()
    hass.states.get.side_effect = lambda entity_id: {
        "binary_sensor.window": State("binary_sensor.window", "on"),
    }.get(entity_id)

    entry = MagicMock()
    entry.entry_id = "entry-map"
    entry.title = "Window Risk"
    entry.data = {
        "name": "Window Risk",
        "goal": "risk",
        "intercept": 0.0,
        "coefficients": {"binary_sensor.window": 1.0},
        "feature_types": {"binary_sensor.window": "categorical"},
        "calibration_slope": 1.0,
        "calibration_intercept": 0.0,
        "required_features": ["binary_sensor.window"],
        "state_mappings": {"binary_sensor.window": {"on": 1, "off": 0}},
    }
    entry.options = {}

    sensor = CalibratedLogisticRegressionSensor(hass, entry)
    sensor._recompute_state(datetime.now())

    assert sensor.available is True
    assert sensor.extra_state_attributes["feature_values"]["binary_sensor.window"] == 1.0
    assert sensor.extra_state_attributes["mapped_state_values"]["binary_sensor.window"] == "on"


def test_sensor_auto_maps_known_categorical_state_when_mapping_missing() -> None:
    hass = MagicMock()
    hass.states.get.side_effect = lambda entity_id: {
        "binary_sensor.window": State("binary_sensor.window", "off"),
    }.get(entity_id)

    entry = MagicMock()
    entry.entry_id = "entry-auto-map"
    entry.title = "Window Risk"
    entry.data = {
        "name": "Window Risk",
        "goal": "risk",
        "intercept": 0.0,
        "coefficients": {"binary_sensor.window": 1.0},
        "feature_types": {"binary_sensor.window": "categorical"},
        "calibration_slope": 1.0,
        "calibration_intercept": 0.0,
        "required_features": ["binary_sensor.window"],
        "state_mappings": {},
    }
    entry.options = {}

    sensor = CalibratedLogisticRegressionSensor(hass, entry)
    sensor._recompute_state(datetime.now())

    assert sensor.available is True
    assert sensor.extra_state_attributes["feature_values"]["binary_sensor.window"] == 0.0


def test_sensor_marks_negative_when_probability_below_threshold() -> None:
    hass = MagicMock()
    hass.states.get.side_effect = lambda entity_id: {
        "sensor.a": State("sensor.a", "1"),
    }.get(entity_id)

    entry = MagicMock()
    entry.entry_id = "entry-threshold"
    entry.title = "Threshold Test"
    entry.data = {
        "name": "Threshold Test",
        "goal": "risk",
        "intercept": 0.0,
        "coefficients": {"sensor.a": 1.0},
        "feature_types": {"sensor.a": "numeric"},
        "calibration_slope": 1.0,
        "calibration_intercept": 0.0,
        "threshold": 90.0,
        "required_features": ["sensor.a"],
        "state_mappings": {},
    }
    entry.options = {}

    sensor = CalibratedLogisticRegressionSensor(hass, entry)
    sensor._recompute_state(datetime.now())

    assert sensor.available is True
    attrs = sensor.extra_state_attributes
    assert attrs["decision_threshold"] == 90.0
    assert attrs["is_above_threshold"] is False
    assert attrs["decision"] == "negative"


def test_sensor_can_load_model_from_ml_data_layer(monkeypatch) -> None:
    hass = MagicMock()
    hass.states.get.side_effect = lambda entity_id: {
        "sensor.a": State("sensor.a", "2"),
        "sensor.b": State("sensor.b", "1"),
    }.get(entity_id)

    entry = _build_entry()
    entry.data["ml_db_path"] = "/tmp/ha_ml_data_layer.db"
    entry.data["ml_artifact_view"] = "vw_clr_latest_model_artifact"

    monkeypatch.setattr(
        "custom_components.calibrated_logistic_regression.sensor.load_latest_clr_model_artifact",
        lambda db_path, artifact_view: ClrModelArtifact(
            intercept=-1.0,
            coefficients={"sensor.a": 1.0, "sensor.b": 0.0},
            feature_names=["sensor.a", "sensor.b"],
            model_type="sklearn_logistic_regression",
            feature_set_version="v1",
            created_at_utc="2026-02-25T00:00:00+00:00",
        ),
    )

    sensor = CalibratedLogisticRegressionSensor(hass, entry)
    sensor._recompute_state(datetime.now())

    attrs = sensor.extra_state_attributes
    assert attrs["model_source"] == "ml_data_layer"
    assert attrs["model_artifact_error"] is None
    assert sensor.available is True
