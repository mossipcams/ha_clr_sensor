"""Inference primitives for CLR prediction and decisioning."""

from __future__ import annotations

from dataclasses import dataclass

from .model import calibrated_probability, logistic_probability


@dataclass(slots=True)
class ModelSpec:
    """Model coefficients and intercept for CLR scoring."""

    intercept: float
    coefficients: dict[str, float]


@dataclass(slots=True)
class CalibrationSpec:
    """Calibration parameters for post-logit correction."""

    slope: float
    intercept: float


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


def run_inference(
    *,
    feature_values: dict[str, float],
    missing_features: list[str],
    model: ModelSpec,
    calibration: CalibrationSpec,
    threshold: float,
) -> InferenceResult:
    """Compute CLR inference result from a prepared feature vector."""
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

    raw_probability, linear_score = logistic_probability(
        features=feature_values,
        coefficients=model.coefficients,
        intercept=model.intercept,
    )
    calibrated = calibrated_probability(
        base_probability=raw_probability,
        calibration_slope=calibration.slope,
        calibration_intercept=calibration.intercept,
    )
    native_value = calibrated * 100.0
    is_above_threshold = native_value >= threshold

    return InferenceResult(
        available=True,
        native_value=native_value,
        raw_probability=raw_probability,
        linear_score=linear_score,
        feature_contributions={
            feature_id: model.coefficients.get(feature_id, 0.0) * value
            for feature_id, value in feature_values.items()
        },
        unavailable_reason=None,
        is_above_threshold=is_above_threshold,
        decision="positive" if is_above_threshold else "negative",
    )
