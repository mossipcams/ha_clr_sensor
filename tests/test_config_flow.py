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
    flow.hass.states.get.side_effect = lambda entity_id: {
        "sensor.a": MagicMock(state="22.5"),
        "binary_sensor.window": MagicMock(state="on"),
        "sensor.b": MagicMock(state="5"),
    }.get(entity_id)
    return flow


def test_wizard_happy_path_creates_entry_from_entities_states_and_threshold() -> None:
    flow = _new_flow()

    user_result = asyncio.run(flow.async_step_user({"name": "Kitchen CLR", "goal": "risk"}))
    assert user_result["type"] == "form"
    assert user_result["step_id"] == "features"

    features_result = asyncio.run(
        flow.async_step_features(
            {"required_features": ["sensor.a", "binary_sensor.window"]}
        )
    )
    assert features_result["type"] == "form"
    assert features_result["step_id"] == "states"

    states_result = asyncio.run(
        flow.async_step_states(
            {
                "sensor.a": "22.5",
                "binary_sensor.window": "on",
                "threshold": 65.0,
            }
        )
    )
    assert states_result["type"] == "form"
    assert states_result["step_id"] == "preview"

    preview_result = asyncio.run(flow.async_step_preview({"confirm": True}))
    assert preview_result["type"] == "create_entry"
    assert preview_result["title"] == "Kitchen CLR"
    assert preview_result["data"]["goal"] == "risk"
    assert preview_result["data"]["feature_types"]["binary_sensor.window"] == "categorical"
    assert preview_result["data"]["state_mappings"]["binary_sensor.window"] == {
        "on": 1.0,
        "off": 0.0,
    }
    assert preview_result["data"]["coefficients"] == {
        "sensor.a": 1.0,
        "binary_sensor.window": 1.0,
    }
    assert preview_result["data"]["threshold"] == 65.0
    assert preview_result["data"]["ml_db_path"] == ""
    assert preview_result["data"]["ml_artifact_view"] == "vw_clr_latest_model_artifact"


def test_user_step_aborts_duplicate_name() -> None:
    flow = CalibratedLogisticRegressionConfigFlow()
    flow.hass = MagicMock()
    existing = MagicMock()
    existing.data = {"name": "Kitchen CLR"}
    flow._async_current_entries = MagicMock(return_value=[existing])

    result = asyncio.run(flow.async_step_user({"name": "Kitchen CLR", "goal": "risk"}))

    assert result["type"] == "abort"
    assert result["reason"] == "already_configured"


def test_features_step_moves_to_states_without_auto_deciding_from_hass_state() -> None:
    flow = _new_flow()
    asyncio.run(flow.async_step_user({"name": "Kitchen CLR", "goal": "risk"}))

    result = asyncio.run(flow.async_step_features({"required_features": ["sensor.a"]}))
    assert result["type"] == "form"
    assert result["step_id"] == "states"


def test_states_step_requires_all_feature_states_from_user_input() -> None:
    flow = _new_flow()
    asyncio.run(flow.async_step_user({"name": "Kitchen CLR", "goal": "risk"}))
    asyncio.run(flow.async_step_features({"required_features": ["sensor.a", "sensor.b"]}))

    result = asyncio.run(
        flow.async_step_states(
            {
                "sensor.a": "22.5",
                "threshold": 50.0,
            }
        )
    )

    assert result["type"] == "form"
    assert result["errors"]["sensor.b"] == "required"


def test_states_step_infers_types_and_fallback_mapping_from_user_provided_states() -> None:
    flow = _new_flow()
    asyncio.run(flow.async_step_user({"name": "Kitchen CLR", "goal": "risk"}))
    asyncio.run(flow.async_step_features({"required_features": ["sensor.status_text"]}))

    result = asyncio.run(
        flow.async_step_states(
            {
                "sensor.status_text": "mystery",
                "threshold": 50.0,
            }
        )
    )

    assert result["type"] == "form"
    assert result["step_id"] == "preview"
    assert flow._draft["feature_types"]["sensor.status_text"] == "categorical"
    assert flow._draft["state_mappings"]["sensor.status_text"] == {"mystery": 1.0}


def test_options_flow_shows_management_menu() -> None:
    entry = MagicMock()
    entry.options = {}
    entry.data = {}

    flow = ClrOptionsFlow(entry)
    result = asyncio.run(flow.async_step_init())

    assert result["type"] == "menu"
    assert "threshold" in result["menu_options"]
    assert "model" in result["menu_options"]


def test_options_flow_threshold_updates_value() -> None:
    entry = MagicMock()
    entry.options = {}
    entry.data = {"threshold": 50.0}

    flow = ClrOptionsFlow(entry)
    result = asyncio.run(flow.async_step_threshold({"threshold": 72.5}))

    assert result["type"] == "create_entry"
    assert result["data"]["threshold"] == 72.5
