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

from custom_components.mindml.lightgbm_inference import (
    LightGBMModelSpec,
    run_lightgbm_inference,
)


def test_lightgbm_inference_returns_unavailable_when_missing_features() -> None:
    result = run_lightgbm_inference(
        feature_values={"event_count": 4.0},
        missing_features=["on_ratio"],
        model=LightGBMModelSpec(
            feature_names=["event_count", "on_ratio"],
            model_payload={"intercept": -1.0, "weights": [0.4, 0.3]},
        ),
        threshold=70.0,
    )

    assert result.available is False
    assert result.native_value is None
    assert result.unavailable_reason == "missing_or_unmapped_features"


def test_lightgbm_inference_uses_booster_model_str_for_probability(monkeypatch) -> None:
    class _Booster:
        def __init__(self, *, model_str: str) -> None:
            assert model_str == "serialized-booster"

        def predict(self, rows, raw_score: bool = False, pred_contrib: bool = False):
            assert rows == [[4.0, 0.5]]
            if pred_contrib:
                return [[0.8, 0.1, -0.7]]
            if raw_score:
                return [0.2]
            return [0.549833997312478]

    monkeypatch.setitem(sys.modules, "lightgbm", types.SimpleNamespace(Booster=_Booster))

    result = run_lightgbm_inference(
        feature_values={"event_count": 4.0, "on_ratio": 0.5},
        missing_features=[],
        model=LightGBMModelSpec(
            feature_names=["event_count", "on_ratio"],
            model_payload={"booster_model_str": "serialized-booster"},
        ),
        threshold=50.0,
    )

    assert result.available is True
    assert result.linear_score == 0.2
    assert result.raw_probability == 0.549833997312478
    assert result.native_value == 54.98339973124779
    assert result.feature_contributions == {"event_count": 0.8, "on_ratio": 0.1}
    assert result.is_above_threshold is True
    assert result.decision == "positive"


def test_lightgbm_inference_returns_unavailable_when_booster_runtime_fails(monkeypatch) -> None:
    class _Booster:
        def __init__(self, *, model_str: str) -> None:
            raise ValueError(model_str)

    monkeypatch.setitem(sys.modules, "lightgbm", types.SimpleNamespace(Booster=_Booster))

    result = run_lightgbm_inference(
        feature_values={"event_count": 4.0, "on_ratio": 0.5},
        missing_features=[],
        model=LightGBMModelSpec(
            feature_names=["event_count", "on_ratio"],
            model_payload={"booster_model_str": "bad-booster"},
        ),
        threshold=50.0,
    )

    assert result.available is False
    assert result.native_value is None
    assert result.unavailable_reason == "lightgbm_inference_error"


def test_lightgbm_inference_returns_unavailable_when_model_payload_missing() -> None:
    result = run_lightgbm_inference(
        feature_values={"event_count": 4.0, "on_ratio": 0.5},
        missing_features=[],
        model=LightGBMModelSpec(
            feature_names=["event_count", "on_ratio"],
            model_payload={},
        ),
        threshold=50.0,
    )

    assert result.available is False
    assert result.native_value is None
    assert result.unavailable_reason == "model_payload_missing"
