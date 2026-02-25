from __future__ import annotations

from custom_components.calibrated_logistic_regression.feature_mapping import (
    parse_feature_types,
    parse_state_mappings,
    validate_categorical_mappings,
)


def test_parse_feature_types_requires_selected_features() -> None:
    parsed = parse_feature_types(
        '{"sensor.a": "numeric", "binary_sensor.window": "categorical"}',
        ["sensor.a", "binary_sensor.window"],
    )
    assert parsed == {"sensor.a": "numeric", "binary_sensor.window": "categorical"}


def test_parse_feature_types_rejects_unknown_values() -> None:
    assert parse_feature_types('{"sensor.a": "weird"}', ["sensor.a"]) is None


def test_parse_state_mappings_coerces_to_float() -> None:
    parsed = parse_state_mappings('{"binary_sensor.window": {"on": 1, "off": 0}}')
    assert parsed == {"binary_sensor.window": {"on": 1.0, "off": 0.0}}


def test_validate_categorical_mappings_reports_missing() -> None:
    missing = validate_categorical_mappings(
        feature_types={"sensor.a": "numeric", "binary_sensor.window": "categorical"},
        state_mappings={},
    )
    assert missing == ["binary_sensor.window"]
