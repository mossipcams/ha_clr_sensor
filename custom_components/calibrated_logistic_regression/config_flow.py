"""Config flow for the LightGBM probability sensor."""

from __future__ import annotations

from typing import Any

import voluptuous as vol
from homeassistant import config_entries
from homeassistant.data_entry_flow import FlowResult
from homeassistant.helpers import selector

from .const import (
    CONF_FEATURE_TYPES,
    CONF_FEATURE_STATES,
    CONF_GOAL,
    CONF_ML_ARTIFACT_VIEW,
    CONF_ML_DB_PATH,
    CONF_ML_FEATURE_SOURCE,
    CONF_ML_FEATURE_VIEW,
    CONF_NAME,
    CONF_REQUIRED_FEATURES,
    CONF_STATE_MAPPINGS,
    CONF_THRESHOLD,
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
)
from .paths import resolve_ml_db_path


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


def _build_features_schema(
    default_features: list[str],
    default_states: dict[str, str] | None = None,
    default_threshold: float = DEFAULT_THRESHOLD,
) -> vol.Schema:
    states = default_states or {}
    schema: dict[Any, Any] = {
        vol.Required(CONF_REQUIRED_FEATURES, default=default_features): selector.EntitySelector(
            selector.EntitySelectorConfig(multiple=True)
        ),
        vol.Optional(CONF_THRESHOLD, default=default_threshold): vol.Coerce(float),
    }
    for feature in default_features:
        schema[vol.Optional(feature, default=states.get(feature, ""))] = str
    return vol.Schema(schema)


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
    return vol.Schema({vol.Required("confirm", default=True): bool})


class CalibratedLogisticRegressionConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for the integration."""

    VERSION = 1

    def __init__(self) -> None:
        self._draft: dict[str, Any] = {}

    async def async_step_user(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        errors: dict[str, str] = {}
        if user_input is not None:
            name = str(user_input[CONF_NAME]).strip()
            goal = str(user_input[CONF_GOAL]).strip()
            ml_db_path = str(user_input.get(CONF_ML_DB_PATH, "")).strip()
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
                self._draft[CONF_ML_DB_PATH] = resolve_ml_db_path(self.hass, ml_db_path)
                self._draft[CONF_ML_ARTIFACT_VIEW] = str(
                    user_input.get(CONF_ML_ARTIFACT_VIEW, DEFAULT_ML_ARTIFACT_VIEW)
                ).strip() or DEFAULT_ML_ARTIFACT_VIEW
                self._draft[CONF_ML_FEATURE_SOURCE] = str(
                    user_input.get(CONF_ML_FEATURE_SOURCE, DEFAULT_ML_FEATURE_SOURCE)
                ).strip() or DEFAULT_ML_FEATURE_SOURCE
                self._draft[CONF_ML_FEATURE_VIEW] = str(
                    user_input.get(CONF_ML_FEATURE_VIEW, DEFAULT_ML_FEATURE_VIEW)
                ).strip() or DEFAULT_ML_FEATURE_VIEW
                return await self.async_step_features()

        return self.async_show_form(
            step_id="user",
            data_schema=_build_user_schema(),
            errors=errors,
            description_placeholders={
                "goal_options": "risk, event_probability, success_probability",
            },
        )

    async def async_step_features(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        errors: dict[str, str] = {}
        default_features = list(self._draft.get(CONF_REQUIRED_FEATURES, []))
        default_states = dict(self._draft.get(CONF_FEATURE_STATES, {}))
        default_threshold = float(self._draft.get(CONF_THRESHOLD, DEFAULT_THRESHOLD))

        if user_input is not None:
            required_features = parse_required_features(user_input[CONF_REQUIRED_FEATURES])
            default_features = list(required_features)
            default_threshold = float(user_input.get(CONF_THRESHOLD, default_threshold))
            if not required_features:
                errors[CONF_REQUIRED_FEATURES] = "required"
            feature_states: dict[str, str] = {}
            for feature in required_features:
                raw_value = user_input.get(feature)
                if raw_value is None or str(raw_value).strip() == "":
                    errors[feature] = "required"
                    feature_states[feature] = ""
                    continue
                feature_states[feature] = str(raw_value)
            default_states = feature_states

            if not errors:
                self._draft[CONF_REQUIRED_FEATURES] = required_features
                self._draft[CONF_FEATURE_STATES] = {feature: feature_states[feature] for feature in required_features}
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
                    else:
                        final_state_mappings[feature] = {
                            self._draft[CONF_FEATURE_STATES][feature].casefold(): 1.0
                        }
                self._draft[CONF_STATE_MAPPINGS] = final_state_mappings
                self._draft[CONF_THRESHOLD] = default_threshold
                return await self.async_step_preview()

        return self.async_show_form(
            step_id="features",
            data_schema=_build_features_schema(default_features, default_states, default_threshold),
            errors=errors,
            description_placeholders={
                "features_help": "Pick entities like sensor.temperature or binary_sensor.door.",
            },
        )

    async def async_step_states(self, user_input: dict[str, Any] | None = None) -> FlowResult:
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
                self._draft[CONF_FEATURE_STATES] = {feature: feature_states[feature] for feature in required_features}
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
                    else:
                        final_state_mappings[feature] = {
                            self._draft[CONF_FEATURE_STATES][feature].casefold(): 1.0
                        }
                self._draft[CONF_STATE_MAPPINGS] = final_state_mappings
                self._draft[CONF_THRESHOLD] = float(user_input.get(CONF_THRESHOLD, DEFAULT_THRESHOLD))
                return await self.async_step_preview()

        default_states = dict(self._draft.get(CONF_FEATURE_STATES, {}))
        default_threshold = float(self._draft.get(CONF_THRESHOLD, DEFAULT_THRESHOLD))
        return self.async_show_form(
            step_id="states",
            data_schema=_build_states_schema(required_features, default_states, default_threshold),
            errors=errors,
            description_placeholders={"states_help": ", ".join(required_features)},
        )

    async def async_step_preview(self, user_input: dict[str, Any] | None = None) -> FlowResult:
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
                    CONF_ML_DB_PATH: self._draft[CONF_ML_DB_PATH],
                    CONF_ML_ARTIFACT_VIEW: self._draft[CONF_ML_ARTIFACT_VIEW],
                    CONF_ML_FEATURE_SOURCE: self._draft[CONF_ML_FEATURE_SOURCE],
                    CONF_ML_FEATURE_VIEW: self._draft[CONF_ML_FEATURE_VIEW],
                },
            )

        return self.async_show_form(
            step_id="preview",
            data_schema=_build_preview_schema(),
            description_placeholders={
                "name": str(self._draft.get(CONF_NAME, "")),
                "goal": str(self._draft.get(CONF_GOAL, "")),
                "required_features": ", ".join(self._draft.get(CONF_REQUIRED_FEATURES, [])),
            },
        )

    @staticmethod
    def async_get_options_flow(config_entry):
        return ClrOptionsFlow(config_entry)


class ClrOptionsFlow(config_entries.OptionsFlow):
    """Handle options flow for post-setup configuration changes."""

    def __init__(self, config_entry) -> None:
        self._config_entry = config_entry
        self._draft: dict[str, Any] = {}

    async def async_step_init(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        del user_input
        return self.async_show_menu(
            step_id="init",
            menu_options=[
                "model",
                "feature_source",
                "decision",
                "features",
                "diagnostics",
            ],
        )

    async def async_step_model(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        if user_input is not None:
            return self.async_create_entry(
                title="",
                data={
                    CONF_ML_DB_PATH: resolve_ml_db_path(
                        self.hass,
                        str(user_input.get(CONF_ML_DB_PATH, "")).strip(),
                    ),
                    CONF_ML_ARTIFACT_VIEW: str(
                        user_input.get(CONF_ML_ARTIFACT_VIEW, DEFAULT_ML_ARTIFACT_VIEW)
                    ).strip()
                    or DEFAULT_ML_ARTIFACT_VIEW,
                },
            )

        default_db_path = self._config_entry.options.get(
            CONF_ML_DB_PATH,
            resolve_ml_db_path(self.hass, self._config_entry.data.get(CONF_ML_DB_PATH, "")),
        )
        default_view = self._config_entry.options.get(
            CONF_ML_ARTIFACT_VIEW,
            self._config_entry.data.get(CONF_ML_ARTIFACT_VIEW, DEFAULT_ML_ARTIFACT_VIEW),
        )
        return self.async_show_form(
            step_id="model",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_ML_DB_PATH, default=default_db_path): str,
                    vol.Required(CONF_ML_ARTIFACT_VIEW, default=default_view): str,
                }
            ),
        )

    async def async_step_feature_source(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        if user_input is not None:
            return self.async_create_entry(
                title="",
                data={
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

        default_feature_source = self._config_entry.options.get(
            CONF_ML_FEATURE_SOURCE,
            self._config_entry.data.get(CONF_ML_FEATURE_SOURCE, DEFAULT_ML_FEATURE_SOURCE),
        )
        default_feature_view = self._config_entry.options.get(
            CONF_ML_FEATURE_VIEW,
            self._config_entry.data.get(CONF_ML_FEATURE_VIEW, DEFAULT_ML_FEATURE_VIEW),
        )
        return self.async_show_form(
            step_id="feature_source",
            data_schema=vol.Schema(
                {
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

    async def async_step_decision(self, user_input: dict[str, Any] | None = None) -> FlowResult:
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
            step_id="decision",
            data_schema=vol.Schema({vol.Required(CONF_THRESHOLD, default=default_threshold): vol.Coerce(float)}),
        )

    async def async_step_features(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        errors: dict[str, str] = {}
        default_features = self._config_entry.options.get(
            CONF_REQUIRED_FEATURES,
            self._config_entry.data.get(CONF_REQUIRED_FEATURES, []),
        )
        existing_states = self._config_entry.options.get(
            CONF_FEATURE_STATES,
            self._config_entry.data.get(CONF_FEATURE_STATES, {}),
        )
        default_states = {feature: str(existing_states.get(feature, "")) for feature in default_features}
        default_threshold = float(
            self._config_entry.options.get(
                CONF_THRESHOLD,
                self._config_entry.data.get(CONF_THRESHOLD, DEFAULT_THRESHOLD),
            )
        )

        if user_input is not None:
            required_features = parse_required_features(user_input[CONF_REQUIRED_FEATURES])
            default_features = list(required_features)
            default_threshold = float(user_input.get(CONF_THRESHOLD, default_threshold))
            if not required_features:
                errors[CONF_REQUIRED_FEATURES] = "required"
            feature_states: dict[str, str] = {}
            for feature in required_features:
                raw_value = user_input.get(feature)
                if raw_value is None or str(raw_value).strip() == "":
                    errors[feature] = "required"
                    feature_states[feature] = ""
                    continue
                feature_states[feature] = str(raw_value)
            default_states = feature_states

            if not errors:
                feature_types = infer_feature_types_from_states(feature_states)
                normalized_feature_types = {
                    feature: feature_types.get(feature, FEATURE_TYPE_NUMERIC)
                    for feature in required_features
                }

                inferred_state_mappings = infer_state_mappings_from_states(feature_states)
                state_mappings: dict[str, dict[str, float]] = {}
                for feature in required_features:
                    if normalized_feature_types[feature] != FEATURE_TYPE_CATEGORICAL:
                        continue
                    if feature in inferred_state_mappings:
                        state_mappings[feature] = inferred_state_mappings[feature]
                    else:
                        state_mappings[feature] = {feature_states[feature].casefold(): 1.0}

                return self.async_create_entry(
                    title="",
                    data={
                        CONF_REQUIRED_FEATURES: required_features,
                        CONF_FEATURE_STATES: feature_states,
                        CONF_FEATURE_TYPES: normalized_feature_types,
                        CONF_STATE_MAPPINGS: state_mappings,
                        CONF_THRESHOLD: default_threshold,
                    },
                )

        return self.async_show_form(
            step_id="features",
            data_schema=_build_features_schema(default_features, default_states, default_threshold),
            errors=errors,
            description_placeholders={
                "features_help": "Pick entities to include as model features."
            },
        )

    async def async_step_states(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        errors: dict[str, str] = {}
        required_features = list(
            self._draft.get(
                CONF_REQUIRED_FEATURES,
                self._config_entry.options.get(
                    CONF_REQUIRED_FEATURES,
                    self._config_entry.data.get(CONF_REQUIRED_FEATURES, []),
                ),
            )
        )

        if user_input is not None:
            feature_states: dict[str, str] = {}
            for feature in required_features:
                raw_value = user_input.get(feature)
                if raw_value is None or str(raw_value).strip() == "":
                    errors[feature] = "required"
                    continue
                feature_states[feature] = str(raw_value)

            if not errors:
                feature_types = infer_feature_types_from_states(feature_states)
                normalized_feature_types = {
                    feature: feature_types.get(feature, FEATURE_TYPE_NUMERIC)
                    for feature in required_features
                }

                inferred_state_mappings = infer_state_mappings_from_states(feature_states)
                state_mappings: dict[str, dict[str, float]] = {}
                for feature in required_features:
                    if normalized_feature_types[feature] != FEATURE_TYPE_CATEGORICAL:
                        continue
                    if feature in inferred_state_mappings:
                        state_mappings[feature] = inferred_state_mappings[feature]
                    else:
                        state_mappings[feature] = {feature_states[feature].casefold(): 1.0}

                return self.async_create_entry(
                    title="",
                    data={
                        CONF_REQUIRED_FEATURES: required_features,
                        CONF_FEATURE_STATES: feature_states,
                        CONF_FEATURE_TYPES: normalized_feature_types,
                        CONF_STATE_MAPPINGS: state_mappings,
                        CONF_THRESHOLD: float(user_input.get(CONF_THRESHOLD, DEFAULT_THRESHOLD)),
                    },
                )

        existing_states = self._config_entry.options.get(
            CONF_FEATURE_STATES,
            self._config_entry.data.get(CONF_FEATURE_STATES, {}),
        )
        default_states = {
            feature: str(existing_states.get(feature, ""))
            for feature in required_features
        }
        default_threshold = float(
            self._config_entry.options.get(
                CONF_THRESHOLD,
                self._config_entry.data.get(CONF_THRESHOLD, DEFAULT_THRESHOLD),
            )
        )
        return self.async_show_form(
            step_id="states",
            data_schema=_build_states_schema(required_features, default_states, default_threshold),
            errors=errors,
            description_placeholders={"states_help": ", ".join(required_features)},
        )

    async def async_step_diagnostics(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        del user_input
        runtime = self.hass.data.get(DOMAIN, {}).get(self._config_entry.entry_id, {}).get("runtime", {})
        return self.async_show_form(
            step_id="diagnostics",
            data_schema=vol.Schema({}),
            description_placeholders={
                "configured_features": ", ".join(
                    self._config_entry.options.get(
                        CONF_REQUIRED_FEATURES,
                        self._config_entry.data.get(CONF_REQUIRED_FEATURES, []),
                    )
                ),
                "missing_features": ", ".join(runtime.get("missing_features", [])) or "none",
                "last_computed_at": str(runtime.get("last_computed_at", "n/a")),
            },
        )
