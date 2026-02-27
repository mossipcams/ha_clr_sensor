"""Tests for rolling_window_hours in config flow."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from custom_components.mindml.config_flow import (
    CalibratedLogisticRegressionConfigFlow,
    ClrOptionsFlow,
)


def _new_flow() -> CalibratedLogisticRegressionConfigFlow:
    flow = CalibratedLogisticRegressionConfigFlow()
    flow.hass = MagicMock()
    flow._async_current_entries = MagicMock(return_value=[])
    return flow


def test_rolling_window_hours_persisted_in_entry() -> None:
    flow = _new_flow()

    asyncio.run(
        flow.async_step_user(
            {
                "name": "Kitchen MindML",
                "goal": "risk",
                "ml_db_path": "/tmp/ha_ml_data_layer.db",
                "rolling_window_hours": 4.0,
            }
        )
    )

    created = asyncio.run(
        flow.async_step_features(
            {
                "feature": "binary_sensor.motion",
                "state": "on",
                "threshold": 50.0,
            }
        )
    )
    assert created["type"] == "create_entry"
    assert created["data"]["rolling_window_hours"] == 4.0


def test_options_feature_source_persists_rolling_window_hours() -> None:
    config_entry = MagicMock()
    config_entry.entry_id = "entry-1"
    config_entry.data = {
        "name": "Kitchen MindML",
        "goal": "risk",
        "required_features": ["binary_sensor.motion"],
        "feature_states": {"binary_sensor.motion": "on"},
        "feature_types": {"binary_sensor.motion": "numeric"},
        "state_mappings": {},
        "threshold": 50.0,
        "ml_db_path": "/tmp/ha_ml_data_layer.db",
        "ml_artifact_view": "vw_clr_latest_model_artifact",
        "ml_feature_source": "hass_state",
        "ml_feature_view": "vw_latest_feature_snapshot",
        "bed_presence_entity": "",
        "rolling_window_hours": 7.0,
    }
    config_entry.options = {}

    options_flow = ClrOptionsFlow(config_entry)
    options_flow.hass = MagicMock()

    result = asyncio.run(
        options_flow.async_step_feature_source(
            {
                "ml_feature_source": "hass_state",
                "ml_feature_view": "vw_latest_feature_snapshot",
                "rolling_window_hours": 3.5,
            }
        )
    )
    assert result["type"] == "create_entry"
    assert result["data"]["rolling_window_hours"] == 3.5
