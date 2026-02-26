from __future__ import annotations

import sys
import types

# Minimal HA stubs for importing config_flow helper functions.
vol = types.ModuleType("voluptuous")
homeassistant = types.ModuleType("homeassistant")
config_entries = types.ModuleType("homeassistant.config_entries")
core = types.ModuleType("homeassistant.core")
data_entry_flow = types.ModuleType("homeassistant.data_entry_flow")
helpers = types.ModuleType("homeassistant.helpers")
selector = types.ModuleType("homeassistant.helpers.selector")


class _ConfigFlow:
    @classmethod
    def __init_subclass__(cls, **kwargs):
        return super().__init_subclass__()


class _OptionsFlow:
    pass


class _FlowResult(dict):
    pass


class _Marker:
    def __init__(self, key, default=None):
        self.schema = key
        self.default = default


def Required(key, default=None):
    return _Marker(key, default=default)


def Optional(key, default=None):
    return _Marker(key, default=default)


class Schema:
    def __init__(self, schema):
        self.schema = schema


def Coerce(_coerce_type):
    return _coerce_type


class SelectSelectorMode:
    DROPDOWN = "dropdown"


class SelectOptionDict:
    def __init__(self, value, label):
        self.value = value
        self.label = label


class SelectSelectorConfig:
    def __init__(self, options, mode):
        self.options = options
        self.mode = mode


class SelectSelector:
    def __init__(self, config):
        self.config = config


class EntitySelectorConfig:
    def __init__(self, multiple=False):
        self.multiple = multiple


class EntitySelector:
    def __init__(self, config):
        self.config = config


config_entries.ConfigFlow = _ConfigFlow
config_entries.OptionsFlow = _OptionsFlow
config_entries.ConfigEntry = object
data_entry_flow.FlowResult = _FlowResult
core.HomeAssistant = object
vol.Required = Required
vol.Optional = Optional
vol.Schema = Schema
vol.Coerce = Coerce
selector.SelectSelectorMode = SelectSelectorMode
selector.SelectOptionDict = SelectOptionDict
selector.SelectSelectorConfig = SelectSelectorConfig
selector.SelectSelector = SelectSelector
selector.EntitySelectorConfig = EntitySelectorConfig
selector.EntitySelector = EntitySelector
helpers.selector = selector

sys.modules.setdefault("homeassistant", homeassistant)
sys.modules.setdefault("voluptuous", vol)
sys.modules.setdefault("homeassistant.config_entries", config_entries)
sys.modules.setdefault("homeassistant.core", core)
sys.modules.setdefault("homeassistant.data_entry_flow", data_entry_flow)
sys.modules.setdefault("homeassistant.helpers", helpers)
sys.modules.setdefault("homeassistant.helpers.selector", selector)

from custom_components.calibrated_logistic_regression.config_flow import _build_user_schema
from custom_components.calibrated_logistic_regression.const import (
    DEFAULT_ML_ARTIFACT_VIEW,
)


def test_user_schema_contains_ml_feature_source_and_view() -> None:
    schema = _build_user_schema()
    keys = [str(k.schema) for k in schema.schema]
    assert "ml_feature_source" in keys
    assert "ml_feature_view" in keys


def test_default_ml_artifact_view_is_lightgbm_contract() -> None:
    assert DEFAULT_ML_ARTIFACT_VIEW == "vw_lightgbm_latest_model_artifact"
