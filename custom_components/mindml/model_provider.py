"""Model provider abstractions for ML artifact-backed LightGBM inference."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .lightgbm_inference import LightGBMModelSpec
from .ml_artifact import LightGBMModelArtifact, load_latest_lightgbm_model_artifact


@dataclass(slots=True)
class ModelProviderResult:
    """Loaded model details plus source diagnostics."""

    model: LightGBMModelSpec
    source: str
    artifact_error: str | None
    artifact_meta: dict[str, object]


class SqliteLightGBMModelProvider:
    """Loads LightGBM model payload from ML artifact view."""

    def __init__(
        self,
        *,
        db_path: str,
        artifact_view: str,
        fallback_feature_names: list[str],
        artifact_loader: Callable[[str, str], LightGBMModelArtifact] = load_latest_lightgbm_model_artifact,
    ) -> None:
        self._db_path = db_path
        self._artifact_view = artifact_view
        self._fallback_feature_names = list(fallback_feature_names)
        self._artifact_loader = artifact_loader

    def _validate_contract_version(self) -> str | None:
        db_file = Path(self._db_path)
        if not db_file.exists():
            return None
        conn = sqlite3.connect(db_file)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute(
                "SELECT value FROM metadata WHERE key = 'contract_version'"
            ).fetchone()
        except sqlite3.Error as exc:
            return f"contract_version check failed: {exc}"
        finally:
            conn.close()
        if row is None:
            return "contract_version check failed: metadata key missing"
        contract_version = str(row["value"])
        if contract_version != "2":
            return (
                "contract_version mismatch: expected 2 "
                f"but found {contract_version}"
            )
        return None

    def load(self) -> ModelProviderResult:
        try:
            contract_error = self._validate_contract_version()
            if contract_error is not None:
                raise ValueError(contract_error)
            artifact = self._artifact_loader(self._db_path, self._artifact_view)
            model = LightGBMModelSpec(
                feature_names=list(artifact.feature_names),
                model_payload=dict(artifact.model_payload),
            )
            artifact_meta = {
                "model_type": artifact.model_type,
                "feature_set_version": artifact.feature_set_version,
                "created_at_utc": artifact.created_at_utc,
                "artifact_view": self._artifact_view,
                "db_path": self._db_path,
            }
            return ModelProviderResult(
                model=model,
                source="ml_data_layer",
                artifact_error=None,
                artifact_meta=artifact_meta,
            )
        except Exception as exc:  # pragma: no cover - runtime fallback guard
            fallback = LightGBMModelSpec(
                feature_names=list(self._fallback_feature_names),
                model_payload={},
            )
            return ModelProviderResult(
                model=fallback,
                source="manual",
                artifact_error=str(exc),
                artifact_meta={},
            )
