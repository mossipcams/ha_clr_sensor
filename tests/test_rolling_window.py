"""Unit tests for RollingWindowTracker."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from custom_components.mindml.rolling_window import RollingWindowTracker


def test_empty_tracker_returns_zero_event_count() -> None:
    tracker = RollingWindowTracker(window_hours=7.0)
    result = tracker.compute_features(["event_count", "on_ratio"])
    assert result["event_count"] == 0.0


def test_empty_tracker_returns_zero_on_ratio() -> None:
    tracker = RollingWindowTracker(window_hours=7.0)
    result = tracker.compute_features(["event_count", "on_ratio"])
    assert result["on_ratio"] == 0.0


def test_empty_tracker_event_count_property_is_zero() -> None:
    tracker = RollingWindowTracker(window_hours=7.0)
    assert tracker.event_count == 0


def test_single_event_increments_count() -> None:
    tracker = RollingWindowTracker(window_hours=7.0)
    tracker.record_event("binary_sensor.motion", "on")
    result = tracker.compute_features(["event_count"])
    assert result["event_count"] == 1.0


def test_on_ratio_all_on() -> None:
    tracker = RollingWindowTracker(window_hours=7.0)
    tracker.record_event("binary_sensor.motion", "on")
    tracker.record_event("binary_sensor.motion", "on")
    result = tracker.compute_features(["on_ratio"])
    assert result["on_ratio"] == 1.0


def test_on_ratio_mixed_events() -> None:
    tracker = RollingWindowTracker(window_hours=7.0)
    tracker.record_event("binary_sensor.motion", "on")
    tracker.record_event("binary_sensor.motion", "off")
    tracker.record_event("binary_sensor.motion", "on")
    tracker.record_event("binary_sensor.motion", "off")
    result = tracker.compute_features(["on_ratio"])
    assert result["on_ratio"] == 0.5


def test_old_events_pruned() -> None:
    tracker = RollingWindowTracker(window_hours=1.0)
    old_time = datetime.now(UTC) - timedelta(hours=2)
    tracker._events.append((old_time, "binary_sensor.motion", "on"))
    tracker.record_event("binary_sensor.motion", "on")
    result = tracker.compute_features(["event_count"])
    assert result["event_count"] == 1.0


def test_ingestion_filter_matching_event_recorded() -> None:
    tracker = RollingWindowTracker(
        window_hours=7.0,
        feature_states={"binary_sensor.motion": "on"},
    )
    tracker.record_event("binary_sensor.motion", "on")
    assert tracker.event_count == 1


def test_ingestion_filter_non_matching_state_rejected() -> None:
    tracker = RollingWindowTracker(
        window_hours=7.0,
        feature_states={"binary_sensor.motion": "on"},
    )
    tracker.record_event("binary_sensor.motion", "off")
    assert tracker.event_count == 0


def test_ingestion_filter_non_matching_entity_rejected() -> None:
    tracker = RollingWindowTracker(
        window_hours=7.0,
        feature_states={"binary_sensor.motion": "on"},
    )
    tracker.record_event("binary_sensor.door", "on")
    assert tracker.event_count == 0


def test_empty_feature_states_records_all_events() -> None:
    tracker = RollingWindowTracker(window_hours=7.0, feature_states={})
    tracker.record_event("binary_sensor.motion", "on")
    tracker.record_event("binary_sensor.door", "open")
    tracker.record_event("sensor.temperature", "22.5")
    assert tracker.event_count == 3


def test_always_returns_both_features() -> None:
    tracker = RollingWindowTracker(window_hours=7.0)
    tracker.record_event("binary_sensor.motion", "on")
    result = tracker.compute_features(["event_count"])
    assert "event_count" in result
    assert "on_ratio" in result


def test_returns_both_even_with_unrelated_required_features() -> None:
    tracker = RollingWindowTracker(window_hours=7.0)
    tracker.record_event("binary_sensor.motion", "on")
    result = tracker.compute_features(["sensor.a", "sensor.b"])
    assert result == {"event_count": 1.0, "on_ratio": 1.0}
