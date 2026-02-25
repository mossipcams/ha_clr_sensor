from __future__ import annotations

import sys
import types

homeassistant = types.ModuleType("homeassistant")
config_entries = types.ModuleType("homeassistant.config_entries")
core = types.ModuleType("homeassistant.core")
config_entries.ConfigEntry = object
core.HomeAssistant = object
sys.modules["homeassistant"] = homeassistant
sys.modules["homeassistant.config_entries"] = config_entries
sys.modules["homeassistant.core"] = core

from custom_components.calibrated_logistic_regression.ml_artifact import ClrModelArtifact
from custom_components.calibrated_logistic_regression.model_provider import (
    ManualModelProvider,
    SqliteArtifactModelProvider,
)


def test_manual_model_provider_returns_manual_model() -> None:
    provider = ManualModelProvider(intercept=0.5, coefficients={"sensor.a": 1.25})

    result = provider.load()

    assert result.source == "manual"
    assert result.model.intercept == 0.5
    assert result.model.coefficients == {"sensor.a": 1.25}
    assert result.artifact_error is None
    assert result.artifact_meta == {}


def test_sqlite_model_provider_returns_ml_artifact_model() -> None:
    provider = SqliteArtifactModelProvider(
        db_path="/tmp/ha_ml_data_layer.db",
        artifact_view="vw_clr_latest_model_artifact",
        fallback_intercept=0.0,
        fallback_coefficients={"sensor.a": 0.0},
        artifact_loader=lambda db_path, artifact_view: ClrModelArtifact(
            intercept=-1.0,
            coefficients={"sensor.a": 2.0},
            feature_names=["sensor.a"],
            model_type="sklearn_logistic_regression",
            feature_set_version="v1",
            created_at_utc="2026-02-25T12:00:00+00:00",
        ),
    )

    result = provider.load()

    assert result.source == "ml_data_layer"
    assert result.model.intercept == -1.0
    assert result.model.coefficients == {"sensor.a": 2.0}
    assert result.artifact_error is None
    assert result.artifact_meta["model_type"] == "sklearn_logistic_regression"
    assert result.artifact_meta["artifact_view"] == "vw_clr_latest_model_artifact"


def test_sqlite_model_provider_falls_back_to_manual_when_loader_fails() -> None:
    provider = SqliteArtifactModelProvider(
        db_path="/tmp/ha_ml_data_layer.db",
        artifact_view="vw_clr_latest_model_artifact",
        fallback_intercept=0.75,
        fallback_coefficients={"sensor.a": 0.1},
        artifact_loader=lambda db_path, artifact_view: (_ for _ in ()).throw(
            ValueError("bad artifact")
        ),
    )

    result = provider.load()

    assert result.source == "manual"
    assert result.model.intercept == 0.75
    assert result.model.coefficients == {"sensor.a": 0.1}
    assert result.artifact_error == "bad artifact"
    assert result.artifact_meta == {}
