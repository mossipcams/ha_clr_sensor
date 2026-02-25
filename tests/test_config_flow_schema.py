from __future__ import annotations

from custom_components.calibrated_logistic_regression.config_flow import (
    _build_feature_types_schema,
    _build_features_schema,
    _build_mappings_schema,
    _build_model_schema,
    _build_preview_schema,
    _build_user_schema,
)


def test_user_schema_contains_name_and_goal() -> None:
    schema = _build_user_schema()
    keys = [str(k.schema) for k in schema.schema]
    assert "name" in keys
    assert "goal" in keys


def test_features_schema_contains_required_features() -> None:
    schema = _build_features_schema("sensor.a")
    keys = [str(k.schema) for k in schema.schema]
    assert "required_features" in keys


def test_feature_types_schema_contains_feature_types() -> None:
    schema = _build_feature_types_schema('{"sensor.a": "numeric"}')
    keys = [str(k.schema) for k in schema.schema]
    assert "feature_types" in keys


def test_mappings_schema_contains_state_mappings() -> None:
    schema = _build_mappings_schema("{}")
    keys = [str(k.schema) for k in schema.schema]
    assert "state_mappings" in keys


def test_model_schema_contains_coefficients_and_calibration() -> None:
    schema = _build_model_schema('{"sensor.a": 1.0}')
    keys = [str(k.schema) for k in schema.schema]
    assert "coefficients" in keys
    assert "intercept" in keys
    assert "calibration_slope" in keys
    assert "calibration_intercept" in keys


def test_preview_schema_has_confirmation_toggle() -> None:
    schema = _build_preview_schema()
    keys = [str(k.schema) for k in schema.schema]
    assert "confirm" in keys
