"""Sensor platform for LightGBM probability scoring."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from homeassistant.components.sensor import SensorEntity, SensorStateClass
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import Event, HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.event import async_track_state_change_event
from homeassistant.helpers.restore_state import RestoreEntity

from .const import (
    CONF_FEATURE_TYPES,
    CONF_ML_ARTIFACT_VIEW,
    CONF_ML_DB_PATH,
    CONF_ML_FEATURE_SOURCE,
    CONF_ML_FEATURE_VIEW,
    CONF_NAME,
    CONF_REQUIRED_FEATURES,
    CONF_STATE_MAPPINGS,
    CONF_THRESHOLD,
    DEFAULT_ML_ARTIFACT_VIEW,
    DEFAULT_ML_FEATURE_SOURCE,
    DEFAULT_ML_FEATURE_VIEW,
    DEFAULT_THRESHOLD,
    DOMAIN,
)
from .feature_provider import RealtimeHistoryFeatureProvider, SqliteSnapshotFeatureProvider
from .lightgbm_inference import LightGBMModelSpec, run_lightgbm_inference
from .model_provider import ModelProviderResult, SqliteLightGBMModelProvider
from .paths import resolve_ml_db_path


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up sensor entities for a config entry."""
    async_add_entities([CalibratedLogisticRegressionSensor(hass, entry)])


class CalibratedLogisticRegressionSensor(SensorEntity, RestoreEntity):
    """Probability sensor backed by LightGBM model artifacts."""

    _attr_has_entity_name = True
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_icon = "mdi:chart-bell-curve-cumulative"

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Initialize the sensor."""
        self.hass = hass
        self._entry_id = entry.entry_id

        config = dict(entry.data)
        config.update(entry.options)

        self._name = str(config.get(CONF_NAME, entry.title or "MindML Probability"))
        self._ml_db_path = resolve_ml_db_path(self.hass, config.get(CONF_ML_DB_PATH, ""))
        self._ml_artifact_view = str(
            config.get(CONF_ML_ARTIFACT_VIEW, DEFAULT_ML_ARTIFACT_VIEW)
        ).strip() or DEFAULT_ML_ARTIFACT_VIEW
        self._ml_feature_source = str(
            config.get(CONF_ML_FEATURE_SOURCE, DEFAULT_ML_FEATURE_SOURCE)
        ).strip() or DEFAULT_ML_FEATURE_SOURCE
        self._ml_feature_view = str(
            config.get(CONF_ML_FEATURE_VIEW, DEFAULT_ML_FEATURE_VIEW)
        ).strip() or DEFAULT_ML_FEATURE_VIEW

        requested_features = list(config.get(CONF_REQUIRED_FEATURES, []))
        model_provider = SqliteLightGBMModelProvider(
            db_path=self._ml_db_path,
            artifact_view=self._ml_artifact_view,
            fallback_feature_names=requested_features,
        )
        model_result: ModelProviderResult = model_provider.load()

        self._model: LightGBMModelSpec = model_result.model
        self._model_source = model_result.source
        self._model_artifact_error = model_result.artifact_error
        self._model_artifact_meta: dict[str, Any] = dict(model_result.artifact_meta)

        self._required_features: list[str] = list(config.get(CONF_REQUIRED_FEATURES, []))
        if self._model.feature_names and self._ml_feature_source == "ml_snapshot":
            self._required_features = list(self._model.feature_names)

        self._feature_types: dict[str, str] = {
            feature_id: str(feature_type).strip().casefold()
            for feature_id, feature_type in dict(config.get(CONF_FEATURE_TYPES, {})).items()
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

        self._threshold = float(config.get(CONF_THRESHOLD, DEFAULT_THRESHOLD))

        if self._ml_feature_source == "ml_snapshot" and self._ml_db_path:
            self._feature_provider = SqliteSnapshotFeatureProvider(
                db_path=self._ml_db_path,
                snapshot_view=self._ml_feature_view,
                required_features=self._required_features,
            )
        else:
            self._ml_feature_source = "hass_state"
            self._feature_provider = RealtimeHistoryFeatureProvider(
                hass=self.hass,
                required_features=self._required_features,
                feature_types=self._feature_types,
                state_mappings=self._state_mappings,
                # Computed features (event_count, on_ratio) are not available via HA state.
                # Users needing data-layer features should use ml_snapshot mode.
                history_feature_loader=lambda required: {},
            )

        self._attr_name = self._name
        self._attr_unique_id = f"{entry.entry_id}_mindml_probability"
        self._attr_native_unit_of_measurement = "%"
        self._attr_suggested_display_precision = 2
        self._attr_should_poll = self._ml_feature_source == "ml_snapshot"

        self._available = False
        self._native_value: float | None = None
        self._raw_probability: float | None = None
        self._linear_score: float | None = None
        self._missing_features: list[str] = []
        self._feature_values: dict[str, float] = {}
        self._feature_contributions: dict[str, float] = {}
        self._mapped_state_values: dict[str, str] = {}
        self._feature_provider_error: str | None = None
        self._unavailable_reason: str | None = None
        self._last_computed_at: str | None = None
        self._is_above_threshold: bool | None = None
        self._decision: str | None = None

    async def async_added_to_hass(self) -> None:
        """Subscribe to source entity updates."""
        await super().async_added_to_hass()
        last_state = await self.async_get_last_state()
        if last_state is not None and last_state.state not in (None, "unknown", "unavailable"):
            try:
                self._native_value = float(last_state.state)
                self._available = True
            except (TypeError, ValueError):
                pass
            attrs = last_state.attributes or {}
            if attrs.get("raw_probability") is not None:
                self._raw_probability = attrs["raw_probability"]
            if attrs.get("linear_score") is not None:
                self._linear_score = attrs["linear_score"]
            self._feature_values = dict(attrs.get("feature_values") or {})
            self._feature_contributions = dict(attrs.get("feature_contributions") or {})
            self._missing_features = list(attrs.get("missing_features") or [])
            self._last_computed_at = attrs.get("last_computed_at")
            self._is_above_threshold = attrs.get("is_above_threshold")
            self._decision = attrs.get("decision")

        if self._ml_feature_source == "hass_state":
            watched_entities = list(dict.fromkeys(self._required_features))

            @callback
            def _handle_state_change(event: Event) -> None:
                del event
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

    async def async_update(self) -> None:
        """Refresh state when polling is enabled."""
        self._recompute_state(datetime.now(UTC))

    @property
    def available(self) -> bool:
        return self._available

    @property
    def native_value(self) -> float | None:
        return self._native_value

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
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
            "feature_provider_error": self._feature_provider_error,
            "last_computed_at": self._last_computed_at,
            "decision_threshold": self._threshold,
            "is_above_threshold": self._is_above_threshold,
            "decision": self._decision,
            "model_source": self._model_source,
            "model_runtime": "lightgbm",
            "model_artifact_error": self._model_artifact_error,
            "model_artifact_meta": dict(self._model_artifact_meta),
            "feature_source": self._ml_feature_source,
            "feature_view": self._ml_feature_view,
        }

    def _recompute_state(self, now: datetime) -> None:
        try:
            feature_vector = self._feature_provider.load()
            self._feature_provider_error = None
        except Exception as exc:  # pragma: no cover
            self._feature_values = {}
            self._mapped_state_values = {}
            self._missing_features = list(self._required_features)
            self._last_computed_at = now.astimezone(UTC).isoformat()
            self._feature_provider_error = str(exc)
            self._available = False
            self._native_value = None
            self._raw_probability = None
            self._linear_score = None
            self._feature_contributions = {}
            self._unavailable_reason = "feature_source_error"
            self._is_above_threshold = None
            self._decision = None
            self._store_runtime_diagnostics()
            return

        self._feature_values = dict(feature_vector.feature_values)
        self._mapped_state_values = dict(feature_vector.mapped_state_values)
        self._missing_features = list(feature_vector.missing_features)
        self._last_computed_at = now.astimezone(UTC).isoformat()

        result = run_lightgbm_inference(
            feature_values=self._feature_values,
            missing_features=self._missing_features,
            model=self._model,
            threshold=self._threshold,
        )
        self._available = result.available
        self._native_value = result.native_value
        self._raw_probability = result.raw_probability
        self._linear_score = result.linear_score
        self._feature_contributions = dict(result.feature_contributions)
        self._unavailable_reason = result.unavailable_reason
        self._is_above_threshold = result.is_above_threshold
        self._decision = result.decision
        self._store_runtime_diagnostics()

    def _store_runtime_diagnostics(self) -> None:
        """Persist lightweight runtime status for diagnostics endpoint."""
        if not isinstance(getattr(self.hass, "data", None), dict):
            self.hass.data = {}
        domain_data = self.hass.data.setdefault(DOMAIN, {})
        entry_data = domain_data.setdefault(self._entry_id, {})
        entry_data["runtime"] = {
            "available": self._available,
            "feature_source": self._ml_feature_source,
            "missing_features": list(self._missing_features),
            "unavailable_reason": self._unavailable_reason,
            "feature_provider_error": self._feature_provider_error,
            "last_computed_at": self._last_computed_at,
            "model_source": self._model_source,
        }
