"""Feature vector providers for CLR inference."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

from .feature_mapping import FEATURE_TYPE_CATEGORICAL, infer_state_mappings_from_states
from .model import parse_float


@dataclass(slots=True)
class FeatureVectorResult:
    """Prepared feature vector and mapping diagnostics."""

    feature_values: dict[str, float]
    missing_features: list[str]
    mapped_state_values: dict[str, str]


class HassStateFeatureProvider:
    """Build a feature vector directly from Home Assistant entity states."""

    def __init__(
        self,
        *,
        hass: Any,
        required_features: list[str],
        feature_types: dict[str, str],
        state_mappings: dict[str, dict[str, float]],
    ) -> None:
        self._hass = hass
        self._required_features = list(required_features)
        self._feature_types = dict(feature_types)
        self._state_mappings = {
            entity_id: {str(name).casefold(): float(value) for name, value in mapping.items()}
            for entity_id, mapping in state_mappings.items()
        }

    def _encoded_feature_value(self, entity_id: str, raw_state: str) -> tuple[float | None, str | None]:
        parsed = parse_float(raw_state)
        if parsed is not None:
            return parsed, None

        feature_type = self._feature_types.get(entity_id, "numeric")
        if feature_type != FEATURE_TYPE_CATEGORICAL:
            return None, None

        normalized_state = raw_state.casefold()
        encoded = self._state_mappings.get(entity_id, {}).get(normalized_state)
        if encoded is not None:
            return encoded, raw_state

        inferred = infer_state_mappings_from_states({entity_id: raw_state})
        inferred_encoded = inferred.get(entity_id, {}).get(normalized_state)
        if inferred_encoded is None:
            return None, None
        return inferred_encoded, raw_state

    def load(self) -> FeatureVectorResult:
        feature_values: dict[str, float] = {}
        missing: list[str] = []
        mapped_state_values: dict[str, str] = {}

        for entity_id in self._required_features:
            state = self._hass.states.get(entity_id)
            if state is None:
                missing.append(entity_id)
                continue

            encoded, mapped_from = self._encoded_feature_value(entity_id, state.state)
            if encoded is None:
                missing.append(entity_id)
                continue
            feature_values[entity_id] = encoded
            if mapped_from is not None:
                mapped_state_values[entity_id] = mapped_from

        return FeatureVectorResult(
            feature_values=feature_values,
            missing_features=missing,
            mapped_state_values=mapped_state_values,
        )


class SqliteSnapshotFeatureProvider:
    """Build feature vector from ML data-layer latest feature snapshot view."""

    def __init__(
        self,
        *,
        db_path: str,
        snapshot_view: str,
        required_features: list[str],
    ) -> None:
        self._db_path = db_path
        self._snapshot_view = snapshot_view
        self._required_features = list(required_features)

    def load(self) -> FeatureVectorResult:
        if not self._db_path:
            raise ValueError("ml_db_path is required")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self._snapshot_view):
            raise ValueError("Invalid snapshot view name")

        db_file = Path(self._db_path)
        if not db_file.exists():
            raise FileNotFoundError(self._db_path)

        conn = sqlite3.connect(db_file)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                f"SELECT feature_name, feature_value FROM {self._snapshot_view}"
            ).fetchall()
        finally:
            conn.close()

        values_by_name: dict[str, float] = {}
        for row in rows:
            feature_name = str(row["feature_name"])
            parsed = parse_float(row["feature_value"])
            if parsed is not None:
                values_by_name[feature_name] = parsed

        feature_values: dict[str, float] = {}
        missing: list[str] = []
        for feature in self._required_features:
            value = values_by_name.get(feature)
            if value is None:
                missing.append(feature)
                continue
            feature_values[feature] = value

        return FeatureVectorResult(
            feature_values=feature_values,
            missing_features=missing,
            mapped_state_values={},
        )
