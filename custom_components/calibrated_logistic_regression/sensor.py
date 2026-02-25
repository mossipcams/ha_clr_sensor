"""Sensor platform for Calibrated Logistic Regression."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from homeassistant.components.sensor import SensorEntity, SensorStateClass
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import Event, HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.event import async_track_state_change_event

from .const import (
    CONF_CALIBRATION_INTERCEPT,
    CONF_CALIBRATION_SLOPE,
    CONF_COEFFICIENTS,
    CONF_FEATURE_TYPES,
    CONF_INTERCEPT,
    CONF_NAME,
    CONF_REQUIRED_FEATURES,
    CONF_STATE_MAPPINGS,
)
from .feature_mapping import FEATURE_TYPE_CATEGORICAL
from .model import calibrated_probability, logistic_probability, parse_float


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up sensor entities for a config entry."""
    async_add_entities([CalibratedLogisticRegressionSensor(hass, entry)])


class CalibratedLogisticRegressionSensor(SensorEntity):
    """Probability sensor backed by calibrated logistic regression."""

    _attr_has_entity_name = True
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_icon = "mdi:chart-bell-curve-cumulative"

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Initialize the sensor."""
        self.hass = hass

        config = dict(entry.data)
        config.update(entry.options)

        self._name = str(config.get(CONF_NAME, entry.title or "CLR Probability"))
        self._intercept = float(config.get(CONF_INTERCEPT, 0.0))
        self._coefficients: dict[str, float] = {
            entity_id: float(weight)
            for entity_id, weight in dict(config.get(CONF_COEFFICIENTS, {})).items()
        }
        default_required = list(self._coefficients.keys())
        self._required_features: list[str] = list(
            config.get(CONF_REQUIRED_FEATURES, default_required)
        )
        self._feature_types: dict[str, str] = {
            feature_id: str(feature_type).strip().casefold()
            for feature_id, feature_type in dict(
                config.get(CONF_FEATURE_TYPES, {})
            ).items()
        }
        self._state_mappings: dict[str, dict[str, float]] = {}
        raw_mappings = dict(config.get(CONF_STATE_MAPPINGS, {}))
        for entity_id, states in raw_mappings.items():
            if not isinstance(states, dict):
                continue
            self._state_mappings[entity_id] = {
                str(state_name).casefold(): float(encoded)
                for state_name, encoded in states.items()
            }

        self._calibration_slope = float(config.get(CONF_CALIBRATION_SLOPE, 1.0))
        self._calibration_intercept = float(config.get(CONF_CALIBRATION_INTERCEPT, 0.0))

        self._attr_name = self._name
        self._attr_unique_id = f"{entry.entry_id}_calibrated_probability"
        self._attr_native_unit_of_measurement = "%"
        self._attr_suggested_display_precision = 2

        self._available = False
        self._native_value: float | None = None
        self._raw_probability: float | None = None
        self._linear_score: float | None = None
        self._missing_features: list[str] = []
        self._feature_values: dict[str, float] = {}
        self._feature_contributions: dict[str, float] = {}
        self._mapped_state_values: dict[str, str] = {}
        self._unavailable_reason: str | None = None
        self._last_computed_at: str | None = None

    async def async_added_to_hass(self) -> None:
        """Subscribe to source entity updates."""
        await super().async_added_to_hass()
        watched_entities = list(dict.fromkeys(self._required_features))

        @callback
        def _handle_state_change(event: Event) -> None:
            self._recompute_state(datetime.now(UTC))
            self.async_write_ha_state()

        self.async_on_remove(
            async_track_state_change_event(
                self.hass,
                watched_entities,
                _handle_state_change,
            )
        )
        self._recompute_state(datetime.now(UTC))

    @property
    def available(self) -> bool:
        """Return availability based on feature readiness."""
        return self._available

    @property
    def native_value(self) -> float | None:
        """Return calibrated probability in percent."""
        return self._native_value

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Expose model diagnostics for transparency."""
        return {
            "raw_probability": self._raw_probability,
            "linear_score": self._linear_score,
            "feature_values": dict(self._feature_values),
            "feature_contributions": dict(self._feature_contributions),
            "mapped_state_values": dict(self._mapped_state_values),
            "missing_features": list(self._missing_features),
            "required_features": list(self._required_features),
            "state_mappings": {
                entity_id: dict(states) for entity_id, states in self._state_mappings.items()
            },
            "unavailable_reason": self._unavailable_reason,
            "last_computed_at": self._last_computed_at,
        }

    def _encoded_feature_value(self, entity_id: str, raw_state: str) -> tuple[float | None, str | None]:
        """Return numeric value for a source feature, with optional categorical mapping."""
        parsed = parse_float(raw_state)
        if parsed is not None:
            return parsed, None

        feature_type = self._feature_types.get(entity_id, "numeric")
        if feature_type != FEATURE_TYPE_CATEGORICAL:
            return None, None

        normalized_state = raw_state.casefold()
        entity_mapping = self._state_mappings.get(entity_id, {})
        encoded = entity_mapping.get(normalized_state)
        if encoded is None:
            return None, None
        return encoded, raw_state

    def _recompute_state(self, now: datetime) -> None:
        """Recompute probability from current source states."""
        feature_values: dict[str, float] = {}
        missing: list[str] = []
        mapped_state_values: dict[str, str] = {}

        for entity_id in self._required_features:
            state = self.hass.states.get(entity_id)
            if state is None:
                missing.append(entity_id)
                continue

            encoded, mapped_from = self._encoded_feature_value(entity_id, state.state)
            if encoded is None:
                missing.append(entity_id)
                continue
            feature_values[entity_id] = encoded
            if mapped_from is not None:
                mapped_state_values[entity_id] = mapped_from

        self._feature_values = feature_values
        self._mapped_state_values = mapped_state_values
        self._missing_features = missing
        self._last_computed_at = now.astimezone(UTC).isoformat()

        if missing:
            self._available = False
            self._native_value = None
            self._raw_probability = None
            self._linear_score = None
            self._feature_contributions = {}
            self._unavailable_reason = "missing_or_unmapped_features"
            return

        raw_probability, linear_score = logistic_probability(
            features=feature_values,
            coefficients=self._coefficients,
            intercept=self._intercept,
        )
        calibrated = calibrated_probability(
            base_probability=raw_probability,
            calibration_slope=self._calibration_slope,
            calibration_intercept=self._calibration_intercept,
        )

        self._available = True
        self._raw_probability = raw_probability
        self._linear_score = linear_score
        self._native_value = calibrated * 100.0
        self._feature_contributions = {
            feature_id: self._coefficients.get(feature_id, 0.0) * value
            for feature_id, value in feature_values.items()
        }
        self._unavailable_reason = None
