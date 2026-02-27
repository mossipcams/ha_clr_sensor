"""Constants for the LightGBM probability sensor integration."""

from __future__ import annotations

DOMAIN = "mindml"
PLATFORMS: list[str] = ["sensor"]

CONF_NAME = "name"
CONF_GOAL = "goal"
CONF_REQUIRED_FEATURES = "required_features"
CONF_FEATURE_TYPES = "feature_types"
CONF_STATE_MAPPINGS = "state_mappings"
CONF_FEATURE_STATES = "feature_states"
CONF_THRESHOLD = "threshold"
CONF_ML_DB_PATH = "ml_db_path"
CONF_ML_ARTIFACT_VIEW = "ml_artifact_view"
CONF_ML_FEATURE_SOURCE = "ml_feature_source"
CONF_ML_FEATURE_VIEW = "ml_feature_view"
CONF_BED_PRESENCE_ENTITY = "bed_presence_entity"
CONF_ROLLING_WINDOW_HOURS = "rolling_window_hours"

DEFAULT_ML_ARTIFACT_VIEW = "vw_lightgbm_latest_model_artifact"
DEFAULT_ML_FEATURE_SOURCE = "hass_state"
DEFAULT_ML_FEATURE_VIEW = "vw_latest_feature_snapshot"
DEFAULT_ML_DB_FILENAME = "ha_ml_data_layer.db"
DEFAULT_ML_DB_PATH = "/config/appdaemon/ha_ml_data_layer.db"

DEFAULT_GOAL = "risk"
DEFAULT_THRESHOLD = 50.0
DEFAULT_ROLLING_WINDOW_HOURS = 7.0
