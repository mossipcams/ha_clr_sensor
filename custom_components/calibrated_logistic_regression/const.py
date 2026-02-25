"""Constants for the Calibrated Logistic Regression integration."""

from __future__ import annotations

DOMAIN = "calibrated_logistic_regression"
PLATFORMS: list[str] = ["sensor"]

CONF_NAME = "name"
CONF_GOAL = "goal"
CONF_INTERCEPT = "intercept"
CONF_COEFFICIENTS = "coefficients"
CONF_REQUIRED_FEATURES = "required_features"
CONF_FEATURE_TYPES = "feature_types"
CONF_STATE_MAPPINGS = "state_mappings"
CONF_FEATURE_STATES = "feature_states"
CONF_THRESHOLD = "threshold"
CONF_CALIBRATION_SLOPE = "calibration_slope"
CONF_CALIBRATION_INTERCEPT = "calibration_intercept"
CONF_ML_DB_PATH = "ml_db_path"
CONF_ML_ARTIFACT_VIEW = "ml_artifact_view"
CONF_ML_FEATURE_SOURCE = "ml_feature_source"
CONF_ML_FEATURE_VIEW = "ml_feature_view"
DEFAULT_ML_ARTIFACT_VIEW = "vw_clr_latest_model_artifact"
DEFAULT_ML_FEATURE_SOURCE = "hass_state"
DEFAULT_ML_FEATURE_VIEW = "vw_latest_feature_snapshot"

DEFAULT_GOAL = "risk"
DEFAULT_THRESHOLD = 50.0
DEFAULT_CALIBRATION_SLOPE = 1.0
DEFAULT_CALIBRATION_INTERCEPT = 0.0
