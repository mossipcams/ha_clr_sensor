from __future__ import annotations

import sys
import types

homeassistant = types.ModuleType("homeassistant")
config_entries = types.ModuleType("homeassistant.config_entries")
core = types.ModuleType("homeassistant.core")
config_entries.ConfigEntry = object
core.HomeAssistant = object
sys.modules.setdefault("homeassistant", homeassistant)
sys.modules.setdefault("homeassistant.config_entries", config_entries)
sys.modules.setdefault("homeassistant.core", core)

from custom_components.mindml.ml_artifact import LightGBMModelArtifact
from custom_components.mindml.model_provider import (
    SqliteLightGBMModelProvider,
)


def test_sqlite_lightgbm_model_provider_returns_ml_artifact_model() -> None:
    provider = SqliteLightGBMModelProvider(
        db_path="/tmp/ha_ml_data_layer.db",
        artifact_view="vw_clr_latest_model_artifact",
        fallback_feature_names=["event_count"],
        artifact_loader=lambda db_path, artifact_view: LightGBMModelArtifact(
            model_payload={"booster_model_str": "serialized-booster"},
            feature_names=["event_count"],
            model_type="lightgbm_binary_classifier",
            feature_set_version="v1",
            created_at_utc="2026-02-25T12:00:00+00:00",
        ),
    )

    result = provider.load()

    assert result.source == "ml_data_layer"
    assert result.model.feature_names == ["event_count"]
    assert result.model.model_payload == {"booster_model_str": "serialized-booster"}
    assert result.artifact_error is None
    assert result.artifact_meta["model_type"] == "lightgbm_binary_classifier"


def test_sqlite_lightgbm_model_provider_falls_back_when_loader_fails() -> None:
    provider = SqliteLightGBMModelProvider(
        db_path="/tmp/ha_ml_data_layer.db",
        artifact_view="vw_clr_latest_model_artifact",
        fallback_feature_names=["event_count"],
        artifact_loader=lambda db_path, artifact_view: (_ for _ in ()).throw(ValueError("bad artifact")),
    )

    result = provider.load()

    assert result.source == "manual"
    assert result.model.feature_names == ["event_count"]
    assert result.model.model_payload == {}
    assert result.artifact_error == "bad artifact"
