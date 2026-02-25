"""Config flow for Calibrated Logistic Regression."""

from __future__ import annotations

import json
from typing import Any

import voluptuous as vol
from homeassistant import config_entries
from homeassistant.data_entry_flow import FlowResult
from homeassistant.helpers import selector

from .const import (
    CONF_CALIBRATION_INTERCEPT,
    CONF_CALIBRATION_SLOPE,
    CONF_COEFFICIENTS,
    CONF_FEATURE_TYPES,
    CONF_FEATURE_STATES,
    CONF_GOAL,
    CONF_INTERCEPT,
    CONF_ML_ARTIFACT_VIEW,
    CONF_ML_DB_PATH,
    CONF_ML_FEATURE_SOURCE,
    CONF_ML_FEATURE_VIEW,
    CONF_NAME,
    CONF_REQUIRED_FEATURES,
    CONF_STATE_MAPPINGS,
    CONF_THRESHOLD,
    DEFAULT_CALIBRATION_INTERCEPT,
    DEFAULT_CALIBRATION_SLOPE,
    DEFAULT_GOAL,
    DEFAULT_ML_ARTIFACT_VIEW,
    DEFAULT_ML_FEATURE_SOURCE,
    DEFAULT_ML_FEATURE_VIEW,
    DEFAULT_THRESHOLD,
    DOMAIN,
)
from .feature_mapping import (
    FEATURE_TYPE_CATEGORICAL,
    FEATURE_TYPE_NUMERIC,
    infer_feature_types_from_states,
    infer_state_mappings_from_states,
    parse_required_features,
    parse_state_mappings,
)


def _build_user_schema() -> vol.Schema:
    return vol.Schema(
        {
            vol.Required(CONF_NAME): str,
            vol.Required(CONF_GOAL, default=DEFAULT_GOAL): selector.SelectSelector(
                selector.SelectSelectorConfig(
                    options=[
                        selector.SelectOptionDict(value="risk", label="Risk"),
                        selector.SelectOptionDict(
                            value="event_probability",
                            label="Event Probability",
                        ),
                        selector.SelectOptionDict(
                            value="success_probability",
                            label="Success Probability",
                        ),
                    ],
                    mode=selector.SelectSelectorMode.DROPDOWN,
                )
            ),
            vol.Optional(CONF_ML_DB_PATH, default=""): str,
            vol.Optional(CONF_ML_ARTIFACT_VIEW, default=DEFAULT_ML_ARTIFACT_VIEW): str,
            vol.Optional(
                CONF_ML_FEATURE_SOURCE,
                default=DEFAULT_ML_FEATURE_SOURCE,
            ): selector.SelectSelector(
                selector.SelectSelectorConfig(
                    options=[
                        selector.SelectOptionDict(
                            value="hass_state",
                            label="Home Assistant States",
                        ),
                        selector.SelectOptionDict(
                            value="ml_snapshot",
                            label="ML Snapshot View",
                        ),
                    ],
                    mode=selector.SelectSelectorMode.DROPDOWN,
                )
            ),
            vol.Optional(CONF_ML_FEATURE_VIEW, default=DEFAULT_ML_FEATURE_VIEW): str,
        }
    )


def _build_features_schema(default_features: list[str]) -> vol.Schema:
    return vol.Schema(
        {
            vol.Required(CONF_REQUIRED_FEATURES, default=default_features): selector.EntitySelector(
                selector.EntitySelectorConfig(multiple=True)
            ),
        }
    )


def _build_mappings_schema(default_mappings: str) -> vol.Schema:
    return vol.Schema(
        {
            vol.Optional(CONF_STATE_MAPPINGS, default=default_mappings): str,
        }
    )


def _build_states_schema(
    required_features: list[str],
    default_states: dict[str, str],
    default_threshold: float,
) -> vol.Schema:
    schema: dict[Any, Any] = {
        vol.Optional(CONF_THRESHOLD, default=default_threshold): vol.Coerce(float),
    }
    for feature in required_features:
        schema[vol.Required(feature, default=default_states.get(feature, ""))] = str
    return vol.Schema(schema)


def _build_preview_schema() -> vol.Schema:
    return vol.Schema(
        {
            vol.Required("confirm", default=True): bool,
        }
    )


class CalibratedLogisticRegressionConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for the integration."""

    VERSION = 1

    def __init__(self) -> None:
        self._draft: dict[str, Any] = {}

    async def async_step_user(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Collect integration name and goal."""
        errors: dict[str, str] = {}
        if user_input is not None:
            name = str(user_input[CONF_NAME]).strip()
            goal = str(user_input[CONF_GOAL]).strip()
            if not name:
                errors[CONF_NAME] = "required"
            if not goal:
                errors[CONF_GOAL] = "required"

            if not errors:
                for entry in self._async_current_entries():
                    if str(entry.data.get(CONF_NAME, "")).strip().casefold() == name.casefold():
                        return self.async_abort(reason="already_configured")
                self._draft[CONF_NAME] = name
                self._draft[CONF_GOAL] = goal
                ml_db_path = str(user_input.get(CONF_ML_DB_PATH, "")).strip()
                ml_artifact_view = str(
                    user_input.get(CONF_ML_ARTIFACT_VIEW, DEFAULT_ML_ARTIFACT_VIEW)
                ).strip()
                ml_feature_source = str(
                    user_input.get(CONF_ML_FEATURE_SOURCE, DEFAULT_ML_FEATURE_SOURCE)
                ).strip() or DEFAULT_ML_FEATURE_SOURCE
                ml_feature_view = str(
                    user_input.get(CONF_ML_FEATURE_VIEW, DEFAULT_ML_FEATURE_VIEW)
                ).strip() or DEFAULT_ML_FEATURE_VIEW
                if ml_db_path:
                    self._draft[CONF_ML_DB_PATH] = ml_db_path
                    self._draft[CONF_ML_ARTIFACT_VIEW] = (
                        ml_artifact_view or DEFAULT_ML_ARTIFACT_VIEW
                    )
                self._draft[CONF_ML_FEATURE_SOURCE] = ml_feature_source
                self._draft[CONF_ML_FEATURE_VIEW] = ml_feature_view
                return await self.async_step_features()

        return self.async_show_form(
            step_id="user",
            data_schema=_build_user_schema(),
            errors=errors,
            description_placeholders={
                "goal_options": "risk, event_probability, success_probability",
            },
        )

    async def async_step_features(
        self,
        user_input: dict[str, Any] | None = None,
    ) -> FlowResult:
        """Collect feature entity IDs."""
        errors: dict[str, str] = {}
        if user_input is not None:
            required_features = parse_required_features(user_input[CONF_REQUIRED_FEATURES])
            if not required_features:
                errors[CONF_REQUIRED_FEATURES] = "required"
            if not errors:
                self._draft[CONF_REQUIRED_FEATURES] = required_features
                return await self.async_step_states()

        default_features = list(self._draft.get(CONF_REQUIRED_FEATURES, []))
        return self.async_show_form(
            step_id="features",
            data_schema=_build_features_schema(default_features),
            errors=errors,
            description_placeholders={
                "features_help": "Pick entities like sensor.temperature or binary_sensor.door.",
            },
        )

    async def async_step_states(
        self,
        user_input: dict[str, Any] | None = None,
    ) -> FlowResult:
        """Collect user-provided states and optional probability threshold."""
        errors: dict[str, str] = {}
        required_features = list(self._draft.get(CONF_REQUIRED_FEATURES, []))

        if user_input is not None:
            feature_states: dict[str, str] = {}
            for feature in required_features:
                raw_value = user_input.get(feature)
                if raw_value is None or str(raw_value).strip() == "":
                    errors[feature] = "required"
                    continue
                feature_states[feature] = str(raw_value)
            if not errors:
                self._draft[CONF_FEATURE_STATES] = {
                    feature: feature_states[feature]
                    for feature in required_features
                }
                feature_types = infer_feature_types_from_states(self._draft[CONF_FEATURE_STATES])
                self._draft[CONF_FEATURE_TYPES] = {
                    feature: feature_types.get(feature, FEATURE_TYPE_NUMERIC)
                    for feature in required_features
                }

                inferred_state_mappings = infer_state_mappings_from_states(self._draft[CONF_FEATURE_STATES])
                final_state_mappings: dict[str, dict[str, float]] = {}
                for feature in required_features:
                    if self._draft[CONF_FEATURE_TYPES][feature] != FEATURE_TYPE_CATEGORICAL:
                        continue
                    if feature in inferred_state_mappings:
                        final_state_mappings[feature] = inferred_state_mappings[feature]
                        continue
                    final_state_mappings[feature] = {
                        self._draft[CONF_FEATURE_STATES][feature].casefold(): 1.0
                    }
                self._draft[CONF_STATE_MAPPINGS] = final_state_mappings

                self._draft[CONF_THRESHOLD] = float(
                    user_input.get(CONF_THRESHOLD, DEFAULT_THRESHOLD)
                )
                self._draft[CONF_INTERCEPT] = 0.0
                self._draft[CONF_COEFFICIENTS] = {
                    feature: 1.0 for feature in required_features
                }
                self._draft[CONF_CALIBRATION_SLOPE] = DEFAULT_CALIBRATION_SLOPE
                self._draft[CONF_CALIBRATION_INTERCEPT] = DEFAULT_CALIBRATION_INTERCEPT
                return await self.async_step_preview()

        default_states = dict(self._draft.get(CONF_FEATURE_STATES, {}))
        default_threshold = float(self._draft.get(CONF_THRESHOLD, DEFAULT_THRESHOLD))
        return self.async_show_form(
            step_id="states",
            data_schema=_build_states_schema(required_features, default_states, default_threshold),
            errors=errors,
            description_placeholders={
                "states_help": ", ".join(required_features),
            },
        )

    async def async_step_preview(
        self,
        user_input: dict[str, Any] | None = None,
    ) -> FlowResult:
        """Show confirmation before creating entry."""
        if user_input is not None and bool(user_input.get("confirm")):
            return self.async_create_entry(
                title=str(self._draft[CONF_NAME]),
                data={
                    CONF_NAME: self._draft[CONF_NAME],
                    CONF_GOAL: self._draft[CONF_GOAL],
                    CONF_REQUIRED_FEATURES: self._draft[CONF_REQUIRED_FEATURES],
                    CONF_FEATURE_TYPES: self._draft[CONF_FEATURE_TYPES],
                    CONF_FEATURE_STATES: self._draft[CONF_FEATURE_STATES],
                    CONF_STATE_MAPPINGS: self._draft[CONF_STATE_MAPPINGS],
                    CONF_THRESHOLD: self._draft[CONF_THRESHOLD],
                    CONF_INTERCEPT: self._draft[CONF_INTERCEPT],
                    CONF_COEFFICIENTS: self._draft[CONF_COEFFICIENTS],
                    CONF_CALIBRATION_SLOPE: self._draft[CONF_CALIBRATION_SLOPE],
                    CONF_CALIBRATION_INTERCEPT: self._draft[CONF_CALIBRATION_INTERCEPT],
                    CONF_ML_DB_PATH: self._draft.get(CONF_ML_DB_PATH, ""),
                    CONF_ML_ARTIFACT_VIEW: self._draft.get(
                        CONF_ML_ARTIFACT_VIEW, DEFAULT_ML_ARTIFACT_VIEW
                    ),
                    CONF_ML_FEATURE_SOURCE: self._draft.get(
                        CONF_ML_FEATURE_SOURCE, DEFAULT_ML_FEATURE_SOURCE
                    ),
                    CONF_ML_FEATURE_VIEW: self._draft.get(
                        CONF_ML_FEATURE_VIEW, DEFAULT_ML_FEATURE_VIEW
                    ),
                },
            )

        return self.async_show_form(
            step_id="preview",
            data_schema=_build_preview_schema(),
            description_placeholders={
                "name": str(self._draft.get(CONF_NAME, "")),
                "goal": str(self._draft.get(CONF_GOAL, "")),
                "required_features": ", ".join(
                    self._draft.get(CONF_REQUIRED_FEATURES, [])
                ),
            },
        )

    @staticmethod
    def async_get_options_flow(config_entry):
        """Get options flow."""
        return ClrOptionsFlow(config_entry)


class ClrOptionsFlow(config_entries.OptionsFlow):
    """Handle options flow for post-setup configuration changes."""

    def __init__(self, config_entry) -> None:
        self._config_entry = config_entry

    async def async_step_init(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Show management sections."""
        return self.async_show_menu(
            step_id="init",
            menu_options=["features", "mappings", "threshold", "calibration", "model", "diagnostics"],
        )

    async def async_step_model(
        self,
        user_input: dict[str, Any] | None = None,
    ) -> FlowResult:
        """Manage external ML artifact source settings."""
        if user_input is not None:
            return self.async_create_entry(
                title="",
                data={
                    CONF_ML_DB_PATH: str(user_input.get(CONF_ML_DB_PATH, "")).strip(),
                    CONF_ML_ARTIFACT_VIEW: str(
                        user_input.get(CONF_ML_ARTIFACT_VIEW, DEFAULT_ML_ARTIFACT_VIEW)
                    ).strip()
                    or DEFAULT_ML_ARTIFACT_VIEW,
                    CONF_ML_FEATURE_SOURCE: str(
                        user_input.get(CONF_ML_FEATURE_SOURCE, DEFAULT_ML_FEATURE_SOURCE)
                    ).strip()
                    or DEFAULT_ML_FEATURE_SOURCE,
                    CONF_ML_FEATURE_VIEW: str(
                        user_input.get(CONF_ML_FEATURE_VIEW, DEFAULT_ML_FEATURE_VIEW)
                    ).strip()
                    or DEFAULT_ML_FEATURE_VIEW,
                },
            )

        default_db_path = self._config_entry.options.get(
            CONF_ML_DB_PATH,
            self._config_entry.data.get(CONF_ML_DB_PATH, ""),
        )
        default_view = self._config_entry.options.get(
            CONF_ML_ARTIFACT_VIEW,
            self._config_entry.data.get(CONF_ML_ARTIFACT_VIEW, DEFAULT_ML_ARTIFACT_VIEW),
        )
        default_feature_source = self._config_entry.options.get(
            CONF_ML_FEATURE_SOURCE,
            self._config_entry.data.get(CONF_ML_FEATURE_SOURCE, DEFAULT_ML_FEATURE_SOURCE),
        )
        default_feature_view = self._config_entry.options.get(
            CONF_ML_FEATURE_VIEW,
            self._config_entry.data.get(CONF_ML_FEATURE_VIEW, DEFAULT_ML_FEATURE_VIEW),
        )
        return self.async_show_form(
            step_id="model",
            data_schema=vol.Schema(
                {
                    vol.Optional(CONF_ML_DB_PATH, default=default_db_path): str,
                    vol.Required(CONF_ML_ARTIFACT_VIEW, default=default_view): str,
                    vol.Required(
                        CONF_ML_FEATURE_SOURCE, default=default_feature_source
                    ): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=[
                                selector.SelectOptionDict(
                                    value="hass_state",
                                    label="Home Assistant States",
                                ),
                                selector.SelectOptionDict(
                                    value="ml_snapshot",
                                    label="ML Snapshot View",
                                ),
                            ],
                            mode=selector.SelectSelectorMode.DROPDOWN,
                        )
                    ),
                    vol.Required(CONF_ML_FEATURE_VIEW, default=default_feature_view): str,
                }
            ),
        )

    async def async_step_features(
        self,
        user_input: dict[str, Any] | None = None,
    ) -> FlowResult:
        """Manage selected required features."""
        errors: dict[str, str] = {}
        if user_input is not None:
            required_features = parse_required_features(user_input[CONF_REQUIRED_FEATURES])
            if not required_features:
                errors[CONF_REQUIRED_FEATURES] = "required"
            else:
                return self.async_create_entry(
                    title="",
                    data={CONF_REQUIRED_FEATURES: required_features},
                )

        defaults = self._config_entry.options.get(
            CONF_REQUIRED_FEATURES,
            self._config_entry.data.get(CONF_REQUIRED_FEATURES, []),
        )
        return self.async_show_form(
            step_id="features",
            data_schema=_build_features_schema(defaults),
            errors=errors,
            description_placeholders={
                "features_help": "Pick entities to include as model features.",
            },
        )

    async def async_step_mappings(
        self,
        user_input: dict[str, Any] | None = None,
    ) -> FlowResult:
        """Manage categorical mappings."""
        errors: dict[str, str] = {}
        if user_input is not None:
            state_mappings = parse_state_mappings(str(user_input.get(CONF_STATE_MAPPINGS, "{}")))
            if state_mappings is None:
                errors[CONF_STATE_MAPPINGS] = "invalid_state_mappings"
            else:
                return self.async_create_entry(
                    title="",
                    data={CONF_STATE_MAPPINGS: state_mappings},
                )

        defaults = self._config_entry.options.get(
            CONF_STATE_MAPPINGS,
            self._config_entry.data.get(CONF_STATE_MAPPINGS, {}),
        )
        return self.async_show_form(
            step_id="mappings",
            data_schema=_build_mappings_schema(json.dumps(defaults)),
            errors=errors,
        )

    async def async_step_calibration(
        self,
        user_input: dict[str, Any] | None = None,
    ) -> FlowResult:
        """Manage calibration settings."""
        if user_input is not None:
            return self.async_create_entry(
                title="",
                data={
                    CONF_CALIBRATION_SLOPE: float(user_input[CONF_CALIBRATION_SLOPE]),
                    CONF_CALIBRATION_INTERCEPT: float(
                        user_input[CONF_CALIBRATION_INTERCEPT]
                    ),
                },
            )

        defaults_slope = self._config_entry.options.get(
            CONF_CALIBRATION_SLOPE,
            self._config_entry.data.get(
                CONF_CALIBRATION_SLOPE,
                DEFAULT_CALIBRATION_SLOPE,
            ),
        )
        defaults_intercept = self._config_entry.options.get(
            CONF_CALIBRATION_INTERCEPT,
            self._config_entry.data.get(
                CONF_CALIBRATION_INTERCEPT,
                DEFAULT_CALIBRATION_INTERCEPT,
            ),
        )
        return self.async_show_form(
            step_id="calibration",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_CALIBRATION_SLOPE, default=defaults_slope): vol.Coerce(float),
                    vol.Required(
                        CONF_CALIBRATION_INTERCEPT,
                        default=defaults_intercept,
                    ): vol.Coerce(float),
                }
            ),
        )

    async def async_step_threshold(
        self,
        user_input: dict[str, Any] | None = None,
    ) -> FlowResult:
        """Manage decision threshold."""
        if user_input is not None:
            return self.async_create_entry(
                title="",
                data={CONF_THRESHOLD: float(user_input[CONF_THRESHOLD])},
            )

        default_threshold = self._config_entry.options.get(
            CONF_THRESHOLD,
            self._config_entry.data.get(CONF_THRESHOLD, DEFAULT_THRESHOLD),
        )
        return self.async_show_form(
            step_id="threshold",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_THRESHOLD, default=default_threshold): vol.Coerce(float),
                }
            ),
        )

    async def async_step_diagnostics(
        self,
        user_input: dict[str, Any] | None = None,
    ) -> FlowResult:
        """Read-only diagnostics hint step."""
        del user_input
        return self.async_show_form(
            step_id="diagnostics",
            data_schema=vol.Schema({}),
            description_placeholders={
                "configured_features": ", ".join(
                    self._config_entry.options.get(
                        CONF_REQUIRED_FEATURES,
                        self._config_entry.data.get(CONF_REQUIRED_FEATURES, []),
                    )
                )
            },
        )
