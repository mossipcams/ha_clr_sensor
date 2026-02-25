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
    entry.title = "Kitchen CLR"
    entry.data = {
        "name": "Kitchen CLR",
        "goal": "risk",
        "intercept": -0.2,
        "coefficients": {"sensor.a": 0.4, "sensor.b": -0.1},
        "feature_types": {"sensor.a": "numeric", "sensor.b": "numeric"},
        "calibration_slope": 1.2,
        "calibration_intercept": -0.05,
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
