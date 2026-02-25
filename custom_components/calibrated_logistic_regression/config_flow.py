"""Config flow for Calibrated Logistic Regression."""

from __future__ import annotations

import json
from typing import Any

import voluptuous as vol
from homeassistant import config_entries
from homeassistant.data_entry_flow import FlowResult

from .const import (
    CONF_CALIBRATION_INTERCEPT,
    CONF_CALIBRATION_SLOPE,
    CONF_COEFFICIENTS,
    CONF_INTERCEPT,
    CONF_NAME,
    CONF_REQUIRED_FEATURES,
    CONF_STATE_MAPPINGS,
    DEFAULT_CALIBRATION_INTERCEPT,
    DEFAULT_CALIBRATION_SLOPE,
    DOMAIN,
)


def _parse_coefficients(raw: str) -> dict[str, float] | None:
    """Parse coefficients JSON and coerce numeric values."""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None

    coefficients: dict[str, float] = {}
    for key, value in parsed.items():
        if not isinstance(key, str) or not key:
            return None
        try:
            coefficients[key] = float(value)
        except (TypeError, ValueError):
            return None
    return coefficients


def _parse_required_features(raw: str) -> list[str]:
    """Parse comma-separated list of required feature entity IDs."""
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_state_mappings(raw: str) -> dict[str, dict[str, float]] | None:
    """Parse nested JSON mapping for non-numeric state encoding."""
    if not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None

    mappings: dict[str, dict[str, float]] = {}
    for entity_id, states in parsed.items():
        if not isinstance(entity_id, str) or not entity_id:
            return None
        if not isinstance(states, dict):
            return None

        per_state: dict[str, float] = {}
        for state_name, encoded_value in states.items():
            if not isinstance(state_name, str) or not state_name:
                return None
            try:
                per_state[state_name] = float(encoded_value)
            except (TypeError, ValueError):
                return None
        mappings[entity_id] = per_state

    return mappings


class CalibratedLogisticRegressionConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for the integration."""

    VERSION = 1

    async def async_step_user(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Handle the user step."""
        errors: dict[str, str] = {}

        if user_input is not None:
            coefficients = _parse_coefficients(str(user_input[CONF_COEFFICIENTS]))
            if coefficients is None:
                errors[CONF_COEFFICIENTS] = "invalid_coefficients"

            required_features = _parse_required_features(
                str(user_input[CONF_REQUIRED_FEATURES])
            )
            if not required_features:
                errors[CONF_REQUIRED_FEATURES] = "required"

            state_mappings = _parse_state_mappings(
                str(user_input.get(CONF_STATE_MAPPINGS, "{}"))
            )
            if state_mappings is None:
                errors[CONF_STATE_MAPPINGS] = "invalid_state_mappings"

            name = str(user_input[CONF_NAME]).strip()
            if not name:
                errors[CONF_NAME] = "required"

            if not errors:
                for entry in self._async_current_entries():
                    if (
                        str(entry.data.get(CONF_NAME, "")).strip().casefold()
                        == name.casefold()
                    ):
                        return self.async_abort(reason="already_configured")

                return self.async_create_entry(
                    title=name,
                    data={
                        CONF_NAME: name,
                        CONF_INTERCEPT: float(user_input[CONF_INTERCEPT]),
                        CONF_COEFFICIENTS: coefficients,
                        CONF_REQUIRED_FEATURES: required_features,
                        CONF_STATE_MAPPINGS: state_mappings,
                        CONF_CALIBRATION_SLOPE: float(user_input[CONF_CALIBRATION_SLOPE]),
                        CONF_CALIBRATION_INTERCEPT: float(
                            user_input[CONF_CALIBRATION_INTERCEPT]
                        ),
                    },
                )

        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_NAME): str,
                    vol.Required(CONF_INTERCEPT, default=0.0): vol.Coerce(float),
                    vol.Required(CONF_COEFFICIENTS, default='{"sensor.example": 1.0}'): str,
                    vol.Required(CONF_REQUIRED_FEATURES, default="sensor.example"): str,
                    vol.Optional(CONF_STATE_MAPPINGS, default="{}"): str,
                    vol.Required(
                        CONF_CALIBRATION_SLOPE,
                        default=DEFAULT_CALIBRATION_SLOPE,
                    ): vol.Coerce(float),
                    vol.Required(
                        CONF_CALIBRATION_INTERCEPT,
                        default=DEFAULT_CALIBRATION_INTERCEPT,
                    ): vol.Coerce(float),
                }
            ),
            errors=errors,
        )

    @staticmethod
    def async_get_options_flow(config_entry):
        """Get options flow."""
        return ClrOptionsFlow(config_entry)


class ClrOptionsFlow(config_entries.OptionsFlow):
    """Handle options flow for calibration updates."""

    def __init__(self, config_entry) -> None:
        self._config_entry = config_entry

    async def async_step_init(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Manage options."""
        errors: dict[str, str] = {}
        if user_input is not None:
            state_mappings = _parse_state_mappings(
                str(user_input.get(CONF_STATE_MAPPINGS, "{}"))
            )
            if state_mappings is None:
                errors[CONF_STATE_MAPPINGS] = "invalid_state_mappings"
            else:
                return self.async_create_entry(
                    title="",
                    data={
                        CONF_CALIBRATION_SLOPE: float(user_input[CONF_CALIBRATION_SLOPE]),
                        CONF_CALIBRATION_INTERCEPT: float(
                            user_input[CONF_CALIBRATION_INTERCEPT]
                        ),
                        CONF_STATE_MAPPINGS: state_mappings,
                    },
                )

        default_mappings = self._config_entry.options.get(
            CONF_STATE_MAPPINGS,
            self._config_entry.data.get(CONF_STATE_MAPPINGS, {}),
        )
        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema(
                {
                    vol.Required(
                        CONF_CALIBRATION_SLOPE,
                        default=self._config_entry.options.get(
                            CONF_CALIBRATION_SLOPE,
                            DEFAULT_CALIBRATION_SLOPE,
                        ),
                    ): vol.Coerce(float),
                    vol.Required(
                        CONF_CALIBRATION_INTERCEPT,
                        default=self._config_entry.options.get(
                            CONF_CALIBRATION_INTERCEPT,
                            DEFAULT_CALIBRATION_INTERCEPT,
                        ),
                    ): vol.Coerce(float),
                    vol.Optional(
                        CONF_STATE_MAPPINGS,
                        default=json.dumps(default_mappings),
                    ): str,
                }
            ),
            errors=errors,
        )
