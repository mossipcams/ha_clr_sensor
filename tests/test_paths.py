from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from custom_components.mindml.paths import resolve_ml_db_path


def test_resolve_ml_db_path_discovers_config_appdaemon_db_when_blank(monkeypatch) -> None:
    hass = MagicMock()
    hass.config.path.return_value = "/config"

    def _fake_exists(self: Path) -> bool:
        return str(self) == "/config/appdaemon/ha_ml_data_layer.db"

    monkeypatch.setattr(Path, "exists", _fake_exists)

    resolved = resolve_ml_db_path(hass, "")

    assert resolved == "/config/appdaemon/ha_ml_data_layer.db"


def test_resolve_ml_db_path_discovers_homeassistant_appdaemon_db_when_blank(
    monkeypatch,
) -> None:
    hass = MagicMock()
    hass.config.path.return_value = "/config"

    def _fake_exists(self: Path) -> bool:
        return str(self) == "/config/appdaemon/ha_ml_data_layer.db"

    monkeypatch.setattr(Path, "exists", _fake_exists)

    resolved = resolve_ml_db_path(hass, "")

    assert resolved == "/config/appdaemon/ha_ml_data_layer.db"
