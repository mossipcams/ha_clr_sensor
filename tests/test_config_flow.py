from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from custom_components.calibrated_logistic_regression.config_flow import (
    CalibratedLogisticRegressionConfigFlow,
    ClrOptionsFlow,
)


def _new_flow() -> CalibratedLogisticRegressionConfigFlow:
    flow = CalibratedLogisticRegressionConfigFlow()
    flow.hass = MagicMock()
    flow._async_current_entries = MagicMock(return_value=[])
    return flow


def test_wizard_happy_path_creates_entry() -> None:
    flow = _new_flow()

    user_result = asyncio.run(flow.async_step_user({"name": "Kitchen CLR", "goal": "risk"}))
    assert user_result["type"] == "form"
    assert user_result["step_id"] == "features"

    features_result = asyncio.run(
        flow.async_step_features({"required_features": "sensor.a, binary_sensor.window"})
    )
    assert features_result["type"] == "form"
    assert features_result["step_id"] == "feature_types"

    feature_types_result = asyncio.run(
        flow.async_step_feature_types(
            {"feature_types": '{"sensor.a": "numeric", "binary_sensor.window": "categorical"}'}
        )
    )
    assert feature_types_result["type"] == "form"
    assert feature_types_result["step_id"] == "mappings"

    mappings_result = asyncio.run(
        flow.async_step_mappings(
            {"state_mappings": '{"binary_sensor.window": {"on": 1, "off": 0}}'}
        )
    )
    assert mappings_result["type"] == "form"
    assert mappings_result["step_id"] == "model"

    model_result = asyncio.run(
        flow.async_step_model(
            {
                "intercept": -0.2,
                "coefficients": '{"sensor.a": 0.5, "binary_sensor.window": 0.8}',
                "calibration_slope": 1.0,
                "calibration_intercept": 0.0,
            }
        )
    )
    assert model_result["type"] == "form"
    assert model_result["step_id"] == "preview"

    preview_result = asyncio.run(flow.async_step_preview({"confirm": True}))
    assert preview_result["type"] == "create_entry"
    assert preview_result["title"] == "Kitchen CLR"
    assert preview_result["data"]["goal"] == "risk"
    assert preview_result["data"]["feature_types"]["binary_sensor.window"] == "categorical"


def test_user_step_aborts_duplicate_name() -> None:
    flow = CalibratedLogisticRegressionConfigFlow()
    flow.hass = MagicMock()
    existing = MagicMock()
    existing.data = {"name": "Kitchen CLR"}
    flow._async_current_entries = MagicMock(return_value=[existing])

    result = asyncio.run(flow.async_step_user({"name": "Kitchen CLR", "goal": "risk"}))

    assert result["type"] == "abort"
    assert result["reason"] == "already_configured"


def test_feature_types_step_rejects_invalid_type() -> None:
    flow = _new_flow()
    asyncio.run(flow.async_step_user({"name": "Kitchen CLR", "goal": "risk"}))
    asyncio.run(flow.async_step_features({"required_features": "sensor.a"}))

    result = asyncio.run(flow.async_step_feature_types({"feature_types": '{"sensor.a": "bad"}'}))

    assert result["type"] == "form"
    assert result["errors"]["feature_types"] == "invalid_feature_types"


def test_mappings_step_requires_categorical_mappings() -> None:
    flow = _new_flow()
    asyncio.run(flow.async_step_user({"name": "Kitchen CLR", "goal": "risk"}))
    asyncio.run(flow.async_step_features({"required_features": "binary_sensor.window"}))
    asyncio.run(
        flow.async_step_feature_types(
            {"feature_types": '{"binary_sensor.window": "categorical"}'}
        )
    )

    result = asyncio.run(flow.async_step_mappings({"state_mappings": "{}"}))

    assert result["type"] == "form"
    assert result["errors"]["state_mappings"] == "missing_categorical_mappings"


def test_model_step_rejects_coefficient_mismatch() -> None:
    flow = _new_flow()
    asyncio.run(flow.async_step_user({"name": "Kitchen CLR", "goal": "risk"}))
    asyncio.run(flow.async_step_features({"required_features": "sensor.a,sensor.b"}))
    asyncio.run(
        flow.async_step_feature_types(
            {"feature_types": '{"sensor.a": "numeric", "sensor.b": "numeric"}'}
        )
    )
    asyncio.run(flow.async_step_mappings({"state_mappings": "{}"}))

    result = asyncio.run(
        flow.async_step_model(
            {
                "intercept": 0.0,
                "coefficients": '{"sensor.a": 1.0}',
                "calibration_slope": 1.0,
                "calibration_intercept": 0.0,
            }
        )
    )

    assert result["type"] == "form"
    assert result["errors"]["coefficients"] == "coefficient_mismatch"


def test_options_flow_shows_management_menu() -> None:
    entry = MagicMock()
    entry.options = {}
    entry.data = {}

    flow = ClrOptionsFlow(entry)
    result = asyncio.run(flow.async_step_init())

    assert result["type"] == "menu"
