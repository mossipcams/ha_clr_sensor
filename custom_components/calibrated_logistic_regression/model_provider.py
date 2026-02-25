"""Model provider abstractions for manual and ML artifact-backed inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .inference import ModelSpec
from .ml_artifact import ClrModelArtifact, load_latest_clr_model_artifact


@dataclass(slots=True)
class ModelProviderResult:
    """Loaded model details plus source diagnostics."""

    model: ModelSpec
    source: str
    artifact_error: str | None
    artifact_meta: dict[str, object]


class ManualModelProvider:
    """Provides model coefficients directly from config."""

    def __init__(self, *, intercept: float, coefficients: dict[str, float]) -> None:
        self._intercept = intercept
        self._coefficients = dict(coefficients)

    def load(self) -> ModelProviderResult:
        return ModelProviderResult(
            model=ModelSpec(intercept=self._intercept, coefficients=dict(self._coefficients)),
            source="manual",
            artifact_error=None,
            artifact_meta={},
        )


class SqliteArtifactModelProvider:
    """Loads model from ML layer artifact view with manual fallback."""

    def __init__(
        self,
        *,
        db_path: str,
        artifact_view: str,
        fallback_intercept: float,
        fallback_coefficients: dict[str, float],
        artifact_loader: Callable[[str, str], ClrModelArtifact] = load_latest_clr_model_artifact,
    ) -> None:
        self._db_path = db_path
        self._artifact_view = artifact_view
        self._fallback_intercept = fallback_intercept
        self._fallback_coefficients = dict(fallback_coefficients)
        self._artifact_loader = artifact_loader

    def load(self) -> ModelProviderResult:
        fallback_model = ModelSpec(
            intercept=self._fallback_intercept,
            coefficients=dict(self._fallback_coefficients),
        )
        try:
            artifact = self._artifact_loader(self._db_path, self._artifact_view)
        except Exception as exc:  # pragma: no cover - runtime fallback guard
            return ModelProviderResult(
                model=fallback_model,
                source="manual",
                artifact_error=str(exc),
                artifact_meta={},
            )

        return ModelProviderResult(
            model=ModelSpec(intercept=artifact.intercept, coefficients=dict(artifact.coefficients)),
            source="ml_data_layer",
            artifact_error=None,
            artifact_meta={
                "model_type": artifact.model_type,
                "feature_set_version": artifact.feature_set_version,
                "created_at_utc": artifact.created_at_utc,
                "artifact_view": self._artifact_view,
                "db_path": self._db_path,
            },
        )
