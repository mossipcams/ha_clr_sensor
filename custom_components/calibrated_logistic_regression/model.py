"""Shared numeric helpers for runtime feature/model scoring."""

from __future__ import annotations

import math


def parse_float(value: object) -> float | None:
    """Parse finite float values from user/entity input."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(parsed):
        return None
    return parsed


def safe_sigmoid(x: float) -> float:
    """Numerically stable sigmoid implementation."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)
