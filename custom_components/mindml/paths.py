"""Path helpers for integration runtime defaults."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .const import DEFAULT_ML_DB_FILENAME, DEFAULT_ML_DB_PATH


def resolve_ml_db_path(hass: Any, configured_path: object) -> str:
    """Return user-configured ML DB path or a Home Assistant-relative default."""
    raw_path = str(configured_path or "").strip()
    if raw_path:
        return raw_path

    candidates: list[Path] = [Path(DEFAULT_ML_DB_PATH)]
    config_base = None
    try:
        config_base = hass.config.path() if hass is not None else None
    except Exception:
        config_base = None
    if isinstance(config_base, str) and config_base:
        candidates.append(Path(config_base) / "appdaemon" / DEFAULT_ML_DB_FILENAME)

    addon_configs_root = Path("/addon_configs")
    if addon_configs_root.exists():
        candidates.extend(
            addon_configs_root.glob(f"*_appdaemon/appdaemon/{DEFAULT_ML_DB_FILENAME}")
        )

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return DEFAULT_ML_DB_PATH
