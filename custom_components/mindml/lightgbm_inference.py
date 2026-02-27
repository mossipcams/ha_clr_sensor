"""LightGBM-compatible inference strategy for runtime scoring."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any

from .model import safe_sigmoid


@dataclass(slots=True)
class InferenceResult:
    """Normalized inference output consumed by the sensor entity."""

    available: bool
    native_value: float | None
    raw_probability: float | None
    linear_score: float | None
    feature_contributions: dict[str, float]
    unavailable_reason: str | None
    is_above_threshold: bool | None
    decision: str | None


@dataclass(slots=True)
class LightGBMModelSpec:
    """Model payload and ordered feature names used for inference."""

    feature_names: list[str]
    model_payload: dict[str, Any]


def run_lightgbm_inference(
    *,
    feature_values: dict[str, float],
    missing_features: list[str],
    model: LightGBMModelSpec,
    threshold: float,
) -> InferenceResult:
    """Compute a probability using a LightGBM-like payload contract."""
    if missing_features:
        return InferenceResult(
            available=False,
            native_value=None,
            raw_probability=None,
            linear_score=None,
            feature_contributions={},
            unavailable_reason="missing_or_unmapped_features",
            is_above_threshold=None,
            decision=None,
        )

    ordered_row = [[float(feature_values.get(name, 0.0)) for name in model.feature_names]]
    booster_model_str = model.model_payload.get("booster_model_str")
    if isinstance(booster_model_str, str) and booster_model_str.strip():
        try:
            lightgbm = import_module("lightgbm")
            booster = lightgbm.Booster(model_str=booster_model_str)
            raw_probability = float(booster.predict(ordered_row)[0])
            linear_score = float(booster.predict(ordered_row, raw_score=True)[0])
        except Exception:
            return InferenceResult(
                available=False,
                native_value=None,
                raw_probability=None,
                linear_score=None,
                feature_contributions={},
                unavailable_reason="lightgbm_inference_error",
                is_above_threshold=None,
                decision=None,
            )

        feature_contributions: dict[str, float] = {}
        try:
            contributions = booster.predict(ordered_row, pred_contrib=True)[0]
            for index, feature_name in enumerate(model.feature_names):
                if index < len(contributions):
                    feature_contributions[feature_name] = float(contributions[index])
        except Exception:
            feature_contributions = {}

        native_value = raw_probability * 100.0
        is_above_threshold = native_value >= threshold
        return InferenceResult(
            available=True,
            native_value=native_value,
            raw_probability=raw_probability,
            linear_score=linear_score,
            feature_contributions=feature_contributions,
            unavailable_reason=None,
            is_above_threshold=is_above_threshold,
            decision="positive" if is_above_threshold else "negative",
        )

    has_legacy_linear_payload = "weights" in model.model_payload or "intercept" in model.model_payload
    if not has_legacy_linear_payload:
        return InferenceResult(
            available=False,
            native_value=None,
            raw_probability=None,
            linear_score=None,
            feature_contributions={},
            unavailable_reason="model_payload_missing",
            is_above_threshold=None,
            decision=None,
        )

    intercept = float(model.model_payload.get("intercept", 0.0))
    raw_weights = list(model.model_payload.get("weights", []))
    linear_score = intercept
    feature_contributions: dict[str, float] = {}

    for index, feature_name in enumerate(model.feature_names):
        value = float(feature_values.get(feature_name, 0.0))
        weight = float(raw_weights[index]) if index < len(raw_weights) else 0.0
        contribution = weight * value
        feature_contributions[feature_name] = contribution
        linear_score += contribution

    raw_probability = safe_sigmoid(linear_score)
    native_value = raw_probability * 100.0
    is_above_threshold = native_value >= threshold

    return InferenceResult(
        available=True,
        native_value=native_value,
        raw_probability=raw_probability,
        linear_score=linear_score,
        feature_contributions=feature_contributions,
        unavailable_reason=None,
        is_above_threshold=is_above_threshold,
        decision="positive" if is_above_threshold else "negative",
    )
