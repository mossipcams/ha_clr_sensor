"""Integration tests: rolling window tracker wired into sensor."""

from __future__ import annotations

from unittest.mock import MagicMock

from custom_components.mindml.sensor import CalibratedLogisticRegressionSensor


def _build_entry() -> MagicMock:
    entry = MagicMock()
    entry.entry_id = "entry-1"
    entry.title = "Kitchen MindML"
    entry.data = {
        "name": "Kitchen MindML",
        "goal": "risk",
        "required_features": ["binary_sensor.motion"],
        "feature_types": {"binary_sensor.motion": "numeric"},
        "feature_states": {"binary_sensor.motion": "on"},
        "threshold": 50.0,
        "state_mappings": {},
        "ml_db_path": "/tmp/ha_ml_data_layer.db",
        "ml_artifact_view": "vw_clr_latest_model_artifact",
        "ml_feature_source": "hass_state",
        "bed_presence_entity": "",
    }
    entry.options = {}
    return entry


def test_tracker_created_in_hass_state_mode(monkeypatch) -> None:
    hass = MagicMock()

    class _Provider:
        def __init__(self, **kwargs):
            pass

        def load(self):
            from custom_components.mindml.lightgbm_inference import LightGBMModelSpec
            from custom_components.mindml.model_provider import ModelProviderResult

            return ModelProviderResult(
                model=LightGBMModelSpec(
                    feature_names=["event_count", "on_ratio"],
                    model_payload={"intercept": 0.0, "weights": [0.0, 0.0]},
                ),
                source="ml_data_layer",
                artifact_error=None,
                artifact_meta={},
            )

    monkeypatch.setattr(
        "custom_components.mindml.sensor.SqliteLightGBMModelProvider",
        _Provider,
    )

    sensor = CalibratedLogisticRegressionSensor(hass, _build_entry())
    assert sensor._rolling_window_tracker is not None


def test_events_flow_from_callback_to_tracker(monkeypatch) -> None:
    hass = MagicMock()
    hass.states.get.return_value = None
    captured_callback = {}

    def _track_state(hass_arg, entities, cb):
        captured_callback["cb"] = cb
        return lambda: None

    monkeypatch.setattr(
        "custom_components.mindml.sensor.async_track_state_change_event",
        _track_state,
    )

    class _Provider:
        def __init__(self, **kwargs):
            pass

        def load(self):
            from custom_components.mindml.lightgbm_inference import LightGBMModelSpec
            from custom_components.mindml.model_provider import ModelProviderResult

            return ModelProviderResult(
                model=LightGBMModelSpec(
                    feature_names=["event_count", "on_ratio"],
                    model_payload={"intercept": 0.0, "weights": [0.0, 0.0]},
                ),
                source="ml_data_layer",
                artifact_error=None,
                artifact_meta={},
            )

    monkeypatch.setattr(
        "custom_components.mindml.sensor.SqliteLightGBMModelProvider",
        _Provider,
    )

    import asyncio
    from unittest.mock import AsyncMock

    sensor = CalibratedLogisticRegressionSensor(hass, _build_entry())
    sensor.async_get_last_state = AsyncMock(return_value=None)
    asyncio.run(sensor.async_added_to_hass())

    assert "cb" in captured_callback

    # Simulate a state_changed event with entity_id and new state
    event = MagicMock()
    event.data = {
        "entity_id": "binary_sensor.motion",
        "new_state": MagicMock(state="on"),
    }
    captured_callback["cb"](event)

    assert sensor._rolling_window_tracker.event_count == 1


def test_computed_features_appear_in_feature_values(monkeypatch) -> None:
    from datetime import datetime
    from homeassistant.core import State

    hass = MagicMock()
    hass.states.get.side_effect = lambda eid: State(eid, "1.0")

    class _Provider:
        def __init__(self, **kwargs):
            pass

        def load(self):
            from custom_components.mindml.lightgbm_inference import LightGBMModelSpec
            from custom_components.mindml.model_provider import ModelProviderResult

            return ModelProviderResult(
                model=LightGBMModelSpec(
                    feature_names=["event_count", "on_ratio"],
                    model_payload={"intercept": 0.0, "weights": [1.0, 1.0]},
                ),
                source="ml_data_layer",
                artifact_error=None,
                artifact_meta={},
            )

    monkeypatch.setattr(
        "custom_components.mindml.sensor.SqliteLightGBMModelProvider",
        _Provider,
    )

    sensor = CalibratedLogisticRegressionSensor(hass, _build_entry())
    # Manually add events to the tracker
    # feature_states filter only allows "on" events for binary_sensor.motion
    sensor._rolling_window_tracker.record_event("binary_sensor.motion", "on")
    sensor._rolling_window_tracker.record_event("binary_sensor.motion", "on")
    sensor._rolling_window_tracker.record_event("binary_sensor.motion", "off")  # filtered out

    sensor._recompute_state(datetime.now())

    attrs = sensor.extra_state_attributes
    assert attrs["feature_values"]["event_count"] == 2.0
    assert attrs["feature_values"]["on_ratio"] == 1.0


def test_watched_entities_includes_feature_states_entities(monkeypatch) -> None:
    hass = MagicMock()
    hass.states.get.return_value = None
    captured_entities = {}

    def _track_state(hass_arg, entities, cb):
        captured_entities["entities"] = list(entities)
        return lambda: None

    monkeypatch.setattr(
        "custom_components.mindml.sensor.async_track_state_change_event",
        _track_state,
    )

    class _Provider:
        def __init__(self, **kwargs):
            pass

        def load(self):
            from custom_components.mindml.lightgbm_inference import LightGBMModelSpec
            from custom_components.mindml.model_provider import ModelProviderResult

            return ModelProviderResult(
                model=LightGBMModelSpec(
                    feature_names=["event_count", "on_ratio"],
                    model_payload={"intercept": 0.0, "weights": [0.0, 0.0]},
                ),
                source="ml_data_layer",
                artifact_error=None,
                artifact_meta={},
            )

    monkeypatch.setattr(
        "custom_components.mindml.sensor.SqliteLightGBMModelProvider",
        _Provider,
    )

    import asyncio
    from unittest.mock import AsyncMock

    entry = _build_entry()
    # Add extra entity in feature_states that is NOT in required_features
    entry.data["feature_states"]["binary_sensor.door"] = "open"

    sensor = CalibratedLogisticRegressionSensor(hass, entry)
    sensor.async_get_last_state = AsyncMock(return_value=None)
    asyncio.run(sensor.async_added_to_hass())

    entities = captured_entities["entities"]
    assert "binary_sensor.motion" in entities
    assert "binary_sensor.door" in entities


def test_diagnostics_attributes_present(monkeypatch) -> None:
    hass = MagicMock()

    class _Provider:
        def __init__(self, **kwargs):
            pass

        def load(self):
            from custom_components.mindml.lightgbm_inference import LightGBMModelSpec
            from custom_components.mindml.model_provider import ModelProviderResult

            return ModelProviderResult(
                model=LightGBMModelSpec(
                    feature_names=["event_count", "on_ratio"],
                    model_payload={"intercept": 0.0, "weights": [0.0, 0.0]},
                ),
                source="ml_data_layer",
                artifact_error=None,
                artifact_meta={},
            )

    monkeypatch.setattr(
        "custom_components.mindml.sensor.SqliteLightGBMModelProvider",
        _Provider,
    )

    sensor = CalibratedLogisticRegressionSensor(hass, _build_entry())
    attrs = sensor.extra_state_attributes
    assert attrs["rolling_window_hours"] == 7.0
    assert attrs["rolling_window_event_count"] == 0
