from __future__ import annotations

from custom_components.calibrated_logistic_regression.config_flow import (
    _build_features_schema,
    _build_preview_schema,
    _build_states_schema,
    _build_user_schema,
)


def test_user_schema_contains_name_goal_and_ml_settings() -> None:
    schema = _build_user_schema()
    keys = [str(k.schema) for k in schema.schema]
    assert "name" in keys
    assert "goal" in keys
    assert "ml_db_path" in keys
    assert "ml_artifact_view" in keys
    assert "ml_feature_source" in keys
    assert "ml_feature_view" in keys


def test_features_schema_contains_required_features() -> None:
    schema = _build_features_schema(
        ["sensor.a"],
        {"sensor.a": "22"},
        50.0,
    )
    keys = [str(k.schema) for k in schema.schema]
    assert "required_features" in keys
    assert "sensor.a" in keys
    assert "threshold" in keys
    assert "state_mappings" not in keys


def test_states_schema_contains_feature_fields_and_threshold() -> None:
    schema = _build_states_schema(["sensor.a"], {"sensor.a": "22"}, 50.0)
    keys = [str(k.schema) for k in schema.schema]
    assert "sensor.a" in keys
    assert "threshold" in keys


def test_preview_schema_has_confirmation_toggle() -> None:
    schema = _build_preview_schema()
    keys = [str(k.schema) for k in schema.schema]
    assert "confirm" in keys
