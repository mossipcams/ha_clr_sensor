"""Parsing and validation helpers for feature typing and categorical mappings."""

from __future__ import annotations

import json
from typing import Final

FEATURE_TYPE_NUMERIC: Final = "numeric"
FEATURE_TYPE_CATEGORICAL: Final = "categorical"
_VALID_FEATURE_TYPES: Final[frozenset[str]] = frozenset(
    {FEATURE_TYPE_NUMERIC, FEATURE_TYPE_CATEGORICAL}
)


def parse_required_features(raw: str) -> list[str]:
    """Parse comma-separated list of required feature entity IDs."""
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_coefficients(raw: str) -> dict[str, float] | None:
    """Parse coefficients JSON and coerce numeric values."""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None

    coefficients: dict[str, float] = {}
    for key, value in parsed.items():
        if not isinstance(key, str) or not key:
            return None
        try:
            coefficients[key] = float(value)
        except (TypeError, ValueError):
            return None
    return coefficients


def parse_feature_types(
    raw: str,
    required_features: list[str],
) -> dict[str, str] | None:
    """Parse/validate feature types for selected features."""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None

    selected = set(required_features)
    parsed_keys = set(parsed.keys())
    if parsed_keys != selected:
        return None

    feature_types: dict[str, str] = {}
    for feature, feature_type in parsed.items():
        if not isinstance(feature, str) or feature not in selected:
            return None
        if not isinstance(feature_type, str):
            return None

        normalized = feature_type.strip().casefold()
        if normalized not in _VALID_FEATURE_TYPES:
            return None
        feature_types[feature] = normalized

    return feature_types


def parse_state_mappings(raw: str) -> dict[str, dict[str, float]] | None:
    """Parse nested JSON mapping for non-numeric state encoding."""
    if not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None

    mappings: dict[str, dict[str, float]] = {}
    for entity_id, states in parsed.items():
        if not isinstance(entity_id, str) or not entity_id:
            return None
        if not isinstance(states, dict):
            return None

        per_state: dict[str, float] = {}
        for state_name, encoded_value in states.items():
            if not isinstance(state_name, str) or not state_name:
                return None
            try:
                per_state[state_name] = float(encoded_value)
            except (TypeError, ValueError):
                return None
        mappings[entity_id] = per_state

    return mappings


def validate_categorical_mappings(
    *,
    feature_types: dict[str, str],
    state_mappings: dict[str, dict[str, float]],
) -> list[str]:
    """Return sorted list of categorical features missing mapping tables."""
    missing = [
        feature
        for feature, feature_type in feature_types.items()
        if feature_type == FEATURE_TYPE_CATEGORICAL and not state_mappings.get(feature)
    ]
    missing.sort()
    return missing
