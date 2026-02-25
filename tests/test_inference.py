from __future__ import annotations

import math
import sys
import types

homeassistant = types.ModuleType("homeassistant")
config_entries = types.ModuleType("homeassistant.config_entries")
core = types.ModuleType("homeassistant.core")
config_entries.ConfigEntry = object
core.HomeAssistant = object
sys.modules["homeassistant"] = homeassistant
sys.modules["homeassistant.config_entries"] = config_entries
sys.modules["homeassistant.core"] = core

from custom_components.calibrated_logistic_regression.inference import (
    CalibrationSpec,
    InferenceResult,
    ModelSpec,
    run_inference,
)


def test_run_inference_returns_unavailable_when_features_missing() -> None:
    result = run_inference(
        feature_values={"sensor.a": 2.0},
        missing_features=["sensor.b"],
        model=ModelSpec(intercept=0.0, coefficients={"sensor.a": 1.0, "sensor.b": 1.0}),
        calibration=CalibrationSpec(slope=1.0, intercept=0.0),
        threshold=50.0,
    )

    assert isinstance(result, InferenceResult)
    assert result.available is False
    assert result.native_value is None
    assert result.raw_probability is None
    assert result.linear_score is None
    assert result.feature_contributions == {}
    assert result.unavailable_reason == "missing_or_unmapped_features"
    assert result.decision is None


def test_run_inference_returns_calibrated_probability_and_decision() -> None:
    result = run_inference(
        feature_values={"sensor.a": 2.0, "sensor.b": 1.0},
        missing_features=[],
        model=ModelSpec(intercept=-1.0, coefficients={"sensor.a": 1.0, "sensor.b": 0.0}),
        calibration=CalibrationSpec(slope=1.0, intercept=0.0),
        threshold=80.0,
    )

    expected_raw = 1.0 / (1.0 + math.exp(-1.0))

    assert result.available is True
    assert result.raw_probability == expected_raw
    assert result.linear_score == 1.0
    assert result.feature_contributions == {"sensor.a": 2.0, "sensor.b": 0.0}
    assert result.native_value is not None
    assert result.native_value == expected_raw * 100.0
    assert result.is_above_threshold is False
    assert result.decision == "negative"
    assert result.unavailable_reason is None
