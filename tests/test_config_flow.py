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

    user_result = asyncio.run(
        flow.async_step_user(
            {
                "name": "Kitchen MindML",
                "goal": "risk",
                "ml_db_path": "/tmp/ha_ml_data_layer.db",
            }
        )
    )
    assert user_result["type"] == "form"
    assert user_result["step_id"] == "features"

    features_result = asyncio.run(
        flow.async_step_features({"required_features": ["sensor.a", "binary_sensor.window"]})
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
    assert preview_result["title"] == "Kitchen MindML"
    assert preview_result["data"]["goal"] == "risk"
    assert preview_result["data"]["feature_types"]["binary_sensor.window"] == "categorical"
    assert preview_result["data"]["state_mappings"]["binary_sensor.window"] == {"on": 1.0, "off": 0.0}
    assert preview_result["data"]["threshold"] == 65.0
    assert preview_result["data"]["ml_db_path"] == "/tmp/ha_ml_data_layer.db"


def test_user_step_aborts_duplicate_name() -> None:
    flow = CalibratedLogisticRegressionConfigFlow()
    flow.hass = MagicMock()
    existing = MagicMock()
    existing.data = {"name": "Kitchen MindML"}
    flow._async_current_entries = MagicMock(return_value=[existing])

    result = asyncio.run(
        flow.async_step_user(
            {
                "name": "Kitchen MindML",
                "goal": "risk",
                "ml_db_path": "/tmp/ha_ml_data_layer.db",
            }
        )
    )

    assert result["type"] == "abort"
    assert result["reason"] == "already_configured"


def test_options_flow_shows_management_menu() -> None:
    entry = MagicMock()
    entry.options = {}
    entry.data = {}

    flow = ClrOptionsFlow(entry)
    result = asyncio.run(flow.async_step_init())

    assert result["type"] == "menu"
    assert "model" in result["menu_options"]
    assert "feature_source" in result["menu_options"]
    assert "decision" in result["menu_options"]
    assert "mappings" not in result["menu_options"]


def test_options_flow_decision_updates_threshold() -> None:
    entry = MagicMock()
    entry.options = {}
    entry.data = {"threshold": 50.0}

    flow = ClrOptionsFlow(entry)
    result = asyncio.run(flow.async_step_decision({"threshold": 72.5}))

    assert result["type"] == "create_entry"
    assert result["data"]["threshold"] == 72.5


def test_options_flow_features_updates_required_features() -> None:
    entry = MagicMock()
    entry.options = {}
    entry.data = {"required_features": ["sensor.a"], "state_mappings": {}}

    flow = ClrOptionsFlow(entry)
    result = asyncio.run(
        flow.async_step_features(
            {
                "required_features": ["sensor.a", "binary_sensor.window"],
            }
        )
    )

    assert result["type"] == "create_entry"
    assert result["data"]["required_features"] == ["sensor.a", "binary_sensor.window"]


def test_user_step_allows_blank_ml_db_path_and_continues() -> None:
    flow = _new_flow()

    user_result = asyncio.run(
        flow.async_step_user(
            {
                "name": "Kitchen MindML",
                "goal": "risk",
                "ml_db_path": "",
            }
        )
    )

    assert user_result["type"] == "form"
    assert user_result["step_id"] == "features"


def test_user_step_blank_ml_db_path_uses_appdaemon_default() -> None:
    flow = _new_flow()

    asyncio.run(
        flow.async_step_user(
            {
                "name": "Kitchen MindML",
                "goal": "risk",
                "ml_db_path": "",
            }
        )
    )

    assert flow._draft["ml_db_path"] == "/homeassistant/appdaemon/ha_ml_data_layer.db"
