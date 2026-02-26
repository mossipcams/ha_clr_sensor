from __future__ import annotations

import asyncio
import logging
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


def test_wizard_happy_path_creates_entry_from_looped_feature_state_pairs() -> None:
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

    first_pair = asyncio.run(
        flow.async_step_features(
            {
                "feature": "sensor.a",
                "state": "22.5",
                "threshold": 65.0,
            }
        )
    )
    assert first_pair["type"] == "form"
    assert first_pair["step_id"] == "feature_more"
    feature_more_keys = [str(k.schema) for k in first_pair["data_schema"].schema]
    assert "next_action" in feature_more_keys

    add_step = asyncio.run(flow.async_step_feature_more({"next_action": "add_feature"}))
    assert add_step["type"] == "form"
    assert add_step["step_id"] == "features"

    second_pair = asyncio.run(
        flow.async_step_features(
            {
                "feature": "binary_sensor.window",
                "state": "on",
                "threshold": 65.0,
            }
        )
    )
    assert second_pair["type"] == "form"
    assert second_pair["step_id"] == "feature_more"

    created = asyncio.run(flow.async_step_feature_more({"next_action": "finish_features"}))
    assert created["type"] == "create_entry"
    assert created["title"] == "Kitchen MindML"
    assert created["data"]["required_features"] == ["sensor.a", "binary_sensor.window"]
    assert created["data"]["feature_states"] == {
        "sensor.a": "22.5",
        "binary_sensor.window": "on",
    }
    assert created["data"]["threshold"] == 65.0


def test_wizard_features_step_requires_feature_and_state() -> None:
    flow = _new_flow()
    asyncio.run(
        flow.async_step_user(
            {
                "name": "Kitchen MindML",
                "goal": "risk",
                "ml_db_path": "/tmp/ha_ml_data_layer.db",
            }
        )
    )

    result = asyncio.run(
        flow.async_step_features(
            {
                "feature": "sensor.a",
                "state": "",
                "threshold": 65.0,
            }
        )
    )
    assert result["type"] == "form"
    assert result["step_id"] == "features"
    assert result["errors"]["state"] == "required"


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
    assert "features" in result["menu_options"]


def test_options_flow_decision_updates_threshold() -> None:
    entry = MagicMock()
    entry.options = {}
    entry.data = {"threshold": 50.0}

    flow = ClrOptionsFlow(entry)
    result = asyncio.run(flow.async_step_decision({"threshold": 72.5}))

    assert result["type"] == "create_entry"
    assert result["data"]["threshold"] == 72.5


def test_options_flow_features_updates_configuration_via_loop() -> None:
    entry = MagicMock()
    entry.options = {}
    entry.data = {
        "required_features": ["sensor.a"],
        "feature_states": {"sensor.a": "22.5"},
        "state_mappings": {},
        "threshold": 50.0,
    }

    flow = ClrOptionsFlow(entry)
    first_pair = asyncio.run(
        flow.async_step_features(
            {
                "feature": "sensor.a",
                "state": "23.0",
                "threshold": 65.0,
            }
        )
    )
    assert first_pair["type"] == "form"
    assert first_pair["step_id"] == "feature_more"

    asyncio.run(flow.async_step_feature_more({"next_action": "add_feature"}))
    asyncio.run(
        flow.async_step_features(
            {
                "feature": "binary_sensor.window",
                "state": "off",
                "threshold": 65.0,
            }
        )
    )
    updated = asyncio.run(flow.async_step_feature_more({"next_action": "finish_features"}))

    assert updated["type"] == "create_entry"
    assert updated["data"]["required_features"] == ["sensor.a", "binary_sensor.window"]
    assert updated["data"]["feature_states"] == {
        "sensor.a": "23.0",
        "binary_sensor.window": "off",
    }
    assert updated["data"]["threshold"] == 65.0


def test_options_flow_model_preserves_existing_feature_config() -> None:
    entry = MagicMock()
    entry.options = {
        "required_features": ["sensor.a"],
        "feature_states": {"sensor.a": "22.5"},
        "feature_types": {"sensor.a": "numeric"},
        "state_mappings": {},
        "threshold": 50.0,
    }
    entry.data = {}

    flow = ClrOptionsFlow(entry)
    updated = asyncio.run(
        flow.async_step_model(
            {
                "ml_db_path": "/tmp/ha_ml_data_layer.db",
                "ml_artifact_view": "vw_lightgbm_latest_model_artifact",
            }
        )
    )

    assert updated["type"] == "create_entry"
    assert updated["data"]["required_features"] == ["sensor.a"]
    assert updated["data"]["feature_states"] == {"sensor.a": "22.5"}
    assert updated["data"]["ml_artifact_view"] == "vw_lightgbm_latest_model_artifact"


def test_options_flow_decision_preserves_existing_feature_config() -> None:
    entry = MagicMock()
    entry.options = {
        "required_features": ["sensor.a"],
        "feature_states": {"sensor.a": "22.5"},
        "feature_types": {"sensor.a": "numeric"},
        "state_mappings": {},
        "threshold": 50.0,
    }
    entry.data = {}

    flow = ClrOptionsFlow(entry)
    updated = asyncio.run(flow.async_step_decision({"threshold": 72.5}))

    assert updated["type"] == "create_entry"
    assert updated["data"]["threshold"] == 72.5
    assert updated["data"]["required_features"] == ["sensor.a"]
    assert updated["data"]["feature_states"] == {"sensor.a": "22.5"}


def test_options_flow_features_preserve_existing_ml_settings() -> None:
    entry = MagicMock()
    entry.options = {
        "required_features": ["sensor.a"],
        "feature_states": {"sensor.a": "22.5"},
        "state_mappings": {},
        "feature_types": {"sensor.a": "numeric"},
        "threshold": 50.0,
        "ml_db_path": "/tmp/ha_ml_data_layer.db",
        "ml_artifact_view": "vw_lightgbm_latest_model_artifact",
        "ml_feature_source": "ml_snapshot",
        "ml_feature_view": "vw_latest_feature_snapshot",
    }
    entry.data = {}

    flow = ClrOptionsFlow(entry)
    asyncio.run(
        flow.async_step_features(
            {
                "feature": "sensor.a",
                "state": "23.0",
                "threshold": 65.0,
            }
        )
    )
    updated = asyncio.run(flow.async_step_feature_more({"next_action": "finish_features"}))

    assert updated["type"] == "create_entry"
    assert updated["data"]["required_features"] == ["sensor.a"]
    assert updated["data"]["ml_db_path"] == "/tmp/ha_ml_data_layer.db"
    assert updated["data"]["ml_artifact_view"] == "vw_lightgbm_latest_model_artifact"
    assert updated["data"]["ml_feature_source"] == "ml_snapshot"
    assert updated["data"]["ml_feature_view"] == "vw_latest_feature_snapshot"


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


def test_wizard_features_step_accepts_list_payload_and_creates_entry() -> None:
    flow = _new_flow()

    asyncio.run(
        flow.async_step_user(
            {
                "name": "Bedroom MindML",
                "goal": "risk",
                "ml_db_path": "/tmp/ha_ml_data_layer.db",
            }
        )
    )

    first_pair = asyncio.run(
        flow.async_step_features(
            {
                "feature": ["sensor.a", "binary_sensor.window"],
                "state": "on",
                "threshold": 65.0,
            }
        )
    )
    assert first_pair["type"] == "form"
    assert first_pair["step_id"] == "feature_more"

    created = asyncio.run(flow.async_step_feature_more({"next_action": "finish_features"}))
    assert created["type"] == "create_entry"
    assert created["data"]["required_features"] == ["sensor.a", "binary_sensor.window"]
    assert created["data"]["feature_states"] == {
        "sensor.a": "on",
        "binary_sensor.window": "on",
    }


def test_options_flow_features_accepts_list_payload_and_preserves_ml_settings() -> None:
    entry = MagicMock()
    entry.options = {
        "required_features": ["sensor.a"],
        "feature_states": {"sensor.a": "22.5"},
        "state_mappings": {},
        "feature_types": {"sensor.a": "numeric"},
        "threshold": 50.0,
        "ml_db_path": "/tmp/ha_ml_data_layer.db",
        "ml_artifact_view": "vw_lightgbm_latest_model_artifact",
        "ml_feature_source": "ml_snapshot",
        "ml_feature_view": "vw_latest_feature_snapshot",
    }
    entry.data = {}

    flow = ClrOptionsFlow(entry)
    asyncio.run(
        flow.async_step_features(
            {
                "feature": ["sensor.a", "binary_sensor.window"],
                "state": "off",
                "threshold": 70.0,
            }
        )
    )
    updated = asyncio.run(flow.async_step_feature_more({"next_action": "finish_features"}))

    assert updated["type"] == "create_entry"
    assert updated["data"]["required_features"] == ["sensor.a", "binary_sensor.window"]
    assert updated["data"]["feature_states"] == {
        "sensor.a": "off",
        "binary_sensor.window": "off",
    }
    assert updated["data"]["ml_db_path"] == "/tmp/ha_ml_data_layer.db"
    assert updated["data"]["ml_artifact_view"] == "vw_lightgbm_latest_model_artifact"
    assert updated["data"]["ml_feature_source"] == "ml_snapshot"
    assert updated["data"]["ml_feature_view"] == "vw_latest_feature_snapshot"


def test_wizard_list_payload_updates_existing_feature_without_duplicates() -> None:
    flow = _new_flow()
    asyncio.run(
        flow.async_step_user(
            {
                "name": "Living Room MindML",
                "goal": "risk",
                "ml_db_path": "/tmp/ha_ml_data_layer.db",
            }
        )
    )

    asyncio.run(
        flow.async_step_features(
            {
                "feature": "sensor.a",
                "state": "22.5",
                "threshold": 60.0,
            }
        )
    )
    asyncio.run(flow.async_step_feature_more({"next_action": "add_feature"}))
    asyncio.run(
        flow.async_step_features(
            {
                "feature": ["sensor.a", "binary_sensor.window"],
                "state": "off",
                "threshold": 60.0,
            }
        )
    )
    created = asyncio.run(flow.async_step_feature_more({"next_action": "finish_features"}))

    assert created["type"] == "create_entry"
    assert created["data"]["required_features"] == ["sensor.a", "binary_sensor.window"]
    assert created["data"]["feature_states"] == {
        "sensor.a": "off",
        "binary_sensor.window": "off",
    }


def test_wizard_finish_features_persists_entry_without_preview_step() -> None:
    flow = _new_flow()
    asyncio.run(
        flow.async_step_user(
            {
                "name": "Hallway MindML",
                "goal": "risk",
                "ml_db_path": "/tmp/ha_ml_data_layer.db",
            }
        )
    )
    asyncio.run(
        flow.async_step_features(
            {
                "feature": "sensor.a",
                "state": "on",
                "threshold": 55.0,
            }
        )
    )
    result = asyncio.run(flow.async_step_feature_more({"next_action": "finish_features"}))

    assert result["type"] == "create_entry"
    assert result["title"] == "Hallway MindML"
    assert result["data"]["required_features"] == ["sensor.a"]


def test_wizard_logs_feature_normalization_and_finish_summary(caplog) -> None:
    flow = _new_flow()
    asyncio.run(
        flow.async_step_user(
            {
                "name": "Office MindML",
                "goal": "risk",
                "ml_db_path": "/tmp/ha_ml_data_layer.db",
            }
        )
    )
    caplog.set_level(
        logging.DEBUG,
        logger="custom_components.calibrated_logistic_regression.config_flow",
    )

    asyncio.run(
        flow.async_step_features(
            {
                "feature": ["sensor.a", "binary_sensor.window"],
                "state": "on",
                "threshold": 65.0,
            }
        )
    )
    asyncio.run(flow.async_step_feature_more({"next_action": "finish_features"}))

    messages = [record.getMessage() for record in caplog.records]
    assert any("normalized_features" in message for message in messages)
    assert any("finish_features" in message for message in messages)
